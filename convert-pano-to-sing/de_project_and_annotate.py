import os
import json
import numpy as np
import cv2
from pathlib import Path
# --- FIX: Import trigonometric functions from numpy ---
from numpy import pi, sin, cos, tan, atan, atan2

# --- CONFIGURATION ---
INPUT_DIR = Path('data/images')
OUTPUT_DIR = Path('data_transformed')
COCO_INPUT_FILE = '_annotations_instance.coco.json'
FOV_DEGREES = 90  # Target Field-of-View for the new 'single view' image
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

# Define the three angles for the three 'new' images to be generated from each panorama
# 0: Front view | pi/2 (90 deg): Left view | -pi/2 (-90 deg): Right view
PANO_ANGLES = [0, pi/2, -pi/2] 

# --- SETUP AND UTILITIES ---

def setup_directories():
    """Create output structure and ensure the input file exists."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / 'images').mkdir(exist_ok=True)
    
    if not os.path.exists(COCO_INPUT_FILE):
        raise FileNotFoundError(f"Input COCO JSON file not found at: {COCO_INPUT_FILE}")

def project_annotations(annotation, pano_width, pano_height, target_w, target_h, target_angle_rad):
    """
    Transforms the polygon coordinates from the panorama domain to the rectilinear domain.
    Keeps partially visible polygons by clipping/ignoring out-of-frame points.
    """
    new_annotations = []

    # Calculate projection parameters
    f = target_w / (2 * tan(np.deg2rad(FOV_DEGREES) / 2))  # Focal length

    for ann_index in range(len(annotation['segmentation'])):
        segmentation = annotation['segmentation'][ann_index]
        points = np.array(segmentation).reshape(-1, 2)

        new_segmentation = []

        for x, y in points:
            # 1. Panorama pixel → spherical coords
            theta = (x / pano_width - 0.5) * (2 * pi)
            phi = (y / pano_height - 0.5) * pi

            # Adjust theta by camera view angle
            theta_adj = theta - target_angle_rad

            # 2. Spherical → Cartesian (unit sphere)
            x_sphere = cos(phi) * sin(theta_adj)
            y_sphere = sin(phi)
            z_sphere = cos(phi) * cos(theta_adj)

            if z_sphere <= 0:
                # Point is behind camera → skip it
                continue

            # 3. Perspective projection
            x_proj = f * x_sphere / z_sphere
            y_proj = f * y_sphere / z_sphere

            x_new = int(x_proj + target_w / 2)
            y_new = int(y_proj + target_h / 2)

            # Clip to image bounds
            x_new = np.clip(x_new, 0, target_w - 1)
            y_new = np.clip(y_new, 0, target_h - 1)

            new_segmentation.extend([x_new, y_new])

        # Keep polygon if it has ≥ 3 points (6 coords)
        if len(new_segmentation) >= 6:
            original_ann = next(a for a in coco_data['annotations'] if a['id'] == annotation['id'][ann_index])

            new_annotations.append({
                'id': original_ann['id'],
                'image_id': original_ann['image_id'],
                'category_id': original_ann['category_id'],
                'segmentation': [new_segmentation],
                'bbox': original_ann['bbox'],   # will be updated later
                'area': original_ann['area']    # will be updated later
            })

    return new_annotations

def numpy_encoder(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, np.integer):
        return int(obj) # Convert numpy integer to standard Python int
    elif isinstance(obj, np.floating):
        return float(obj) # Convert numpy float to standard Python float
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert numpy array to standard Python list
    # Optionally, you can raise a TypeError for other unsupported types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def update_bbox_and_area(ann):
    """Calculate min-max bbox and area for the new polygon."""
    points = np.array(ann['segmentation'][0]).reshape(-1, 2)
    x_coords, y_coords = points[:, 0], points[:, 1]
    
    # Clip coordinates to bounds of the target image
    x_coords = np.clip(x_coords, 0, TARGET_WIDTH - 1)
    y_coords = np.clip(y_coords, 0, TARGET_HEIGHT - 1)

    xmin = int(np.min(x_coords))
    ymin = int(np.min(y_coords))
    xmax = int(np.max(x_coords))
    ymax = int(np.max(y_coords))
    
    # New Bbox is [x, y, width, height]
    ann['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
    
    # For simplicity in this demo, area is approximated by bbox area
    ann['area'] = ann['bbox'][2] * ann['bbox'][3]
    return ann

def deproject_image(pano_img, target_angle_rad, output_filename):
    """De-projects the panoramic image using OpenCV's remap."""
    
    map_x = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), np.float32)
    map_y = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), np.float32)
    
    f = TARGET_WIDTH / (2 * tan(np.deg2rad(FOV_DEGREES) / 2))

    for y_new in range(TARGET_HEIGHT):
        for x_new in range(TARGET_WIDTH):
            # Coordinates in the projected image plane (centered at 0, 0)
            x_proj = x_new - TARGET_WIDTH / 2
            y_proj = y_new - TARGET_HEIGHT / 2
            
            # Inverse rectilinear projection (to spherical coordinates)
            # FIX: Used atan2 for accurate angle and robust math
            theta_adj = atan2(x_proj, f) 
            phi = atan(y_proj / np.sqrt(x_proj**2 + f**2))
            
            # Map back to panorama coordinates
            theta = theta_adj + target_angle_rad
            
            # Normalize spherical coordinates to panorama pixel space
            x_pano = (theta / pi) / 2 + 0.5
            y_pano = phi / (pi / 2) / 2 + 0.5
            
            map_x[y_new, x_new] = x_pano * pano_img.shape[1]
            map_y[y_new, x_new] = y_pano * pano_img.shape[0]

    # Warp the image
    rect_img = cv2.remap(pano_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    cv2.imwrite(str(OUTPUT_DIR / 'images' / output_filename), rect_img)
    return rect_img.shape[1], rect_img.shape[0]

# --- MAIN EXECUTION ---

def main():
    setup_directories()
    
    global coco_data # Access global coco_data for annotation mapping
    with open(COCO_INPUT_FILE, 'r') as f:
        coco_data = json.load(f)

    # 1. Extract COCO metadata (categories, info, licenses)
    new_coco = {
        "info": coco_data['info'],
        "licenses": coco_data['licenses'],
        "categories": coco_data['categories'],
        "images": [],
        "annotations": []
    }
    
    # Create unique IDs for new images and annotations
    new_image_id_counter = max(img['id'] for img in coco_data['images']) + 1 if coco_data['images'] else 1
    new_annotation_id_counter = max(ann['id'] for ann in coco_data['annotations']) + 1 if coco_data['annotations'] else 1
    
    # Map original annotations by image ID
    anns_map = {img['id']: [] for img in coco_data['images']}
    for ann in coco_data['annotations']:
        anns_map[ann['image_id']].append(ann)

    print(f"Processing {len(coco_data['images'])} panoramic images...")

    # 2. Iterate through all panoramic images
    for original_img in coco_data['images']:
        original_image_id = original_img['id']
        pano_filename = original_img['file_name']
        pano_path = INPUT_DIR / pano_filename
        
        if not pano_path.exists():
            print(f"Skipping {pano_filename}: Image file not found.")
            continue
            
        pano_img = cv2.imread(str(pano_path))
        pano_height, pano_width, _ = pano_img.shape
        
        original_annotations_data = {
            'segmentation': [a['segmentation'][0] for a in anns_map.get(original_image_id, [])],
            'category_id': [a['category_id'] for a in anns_map.get(original_image_id, [])],
            'id': [a['id'] for a in anns_map.get(original_image_id, [])],
            'image_id': [a['image_id'] for a in anns_map.get(original_image_id, [])]
        }
        
        # 3. Generate three new rectilinear views from each panorama
        for i, angle_rad in enumerate(PANO_ANGLES):
            angle_name = {0: 'FRONT', pi/2: 'LEFT', -pi/2: 'RIGHT'}[angle_rad]
            
            # Create a unique filename for the new image
            # Ensures uniqueness across runs if original images share names
            new_filename = f"{original_image_id}_{Path(pano_filename).stem}_{angle_name}.jpg" 

            # A. Re-project the image (saves to disk)
            deproject_image(pano_img, angle_rad, new_filename)

            # B. Add new image record to COCO
            new_coco['images'].append({
                'id': new_image_id_counter,
                'file_name': new_filename,
                'width': TARGET_WIDTH,
                'height': TARGET_HEIGHT,
                'original_pano_id': original_image_id
            })

            # C. Re-project and update annotations
            reprojected_anns = project_annotations(
                original_annotations_data,
                pano_width, pano_height, TARGET_WIDTH, TARGET_HEIGHT, angle_rad
            )

            # D. Finalize new annotations
            for ann in reprojected_anns:
                ann = update_bbox_and_area(ann)
                ann['id'] = new_annotation_id_counter
                ann['image_id'] = new_image_id_counter
                new_coco['annotations'].append(ann)
                new_annotation_id_counter += 1

            new_image_id_counter += 1
            print(f"  -> Generated {new_filename} with {len(reprojected_anns)} annotations.")

    # 4. Save the new COCO JSON file
    output_json_path = OUTPUT_DIR / 'annotations_transformed.coco.json'
    with open(output_json_path, 'w') as f:
        json.dump(new_coco, f, indent=4, default=numpy_encoder)

    print("\n--- Script Complete ---")
    print(f"Generated {len(new_coco['images'])} new images.")
    print(f"New COCO JSON saved to: {output_json_path}")
    print("\nNext step: Run the verification script or upload the 'data_transformed' folder to Roboflow.")


if __name__ == "__main__":
    # Wrap the main execution in a try-except to catch file errors gracefully
    try:
        main()
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("Ensure your file paths are correct in the script's CONFIGURATION block.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
