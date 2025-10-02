import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_DIR = Path('data_transformed')
IMAGE_DIR = OUTPUT_DIR / 'images'
COCO_INPUT_FILE = OUTPUT_DIR / 'annotations_transformed.coco.json'

# --- UTILITIES ---

def get_annotations_by_image_id(coco_data):
    """Organizes annotations and images for easy lookup."""
    annotations_map = {}
    images_map = {img['id']: img for img in coco_data['images']}
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_map:
            annotations_map[image_id] = []
        annotations_map[image_id].append(ann)
        
    return images_map, annotations_map, {cat['id']: cat['name'] for cat in coco_data['categories']}

def visualize_random_images(images_map, annotations_map, category_map, num_samples=10):
    """Selects and visualizes images in separate OpenCV windows with masks overlaid."""

    # Select images that actually have annotations
    annotated_image_ids = list(annotations_map.keys())
    if not annotated_image_ids:
        print("No images with valid annotations found to visualize.")
        return

    # Randomly sample image IDs
    sample_ids = np.random.choice(annotated_image_ids, min(num_samples, len(annotated_image_ids)), replace=False)

    for image_id in sample_ids:
        img_info = images_map[image_id]
        img_filename = img_info['file_name']
        img_path = IMAGE_DIR / img_filename

        # Load the transformed image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image {img_filename}")
            continue

        # Overlay the masks
        for ann in annotations_map[image_id]:
            category_id = ann['category_id']
            class_name = category_map.get(category_id, f"Class_{category_id}")

            # Fixed light blue color in BGR
            color = (255, 200, 100)  # (Blue, Green, Red)

            for segment in ann['segmentation']:
                points = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
                overlay = img.copy()
                cv2.fillPoly(overlay, [points], color)
                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Draw label near bbox
                x, y, w, h = ann['bbox']
                cv2.putText(img, class_name, (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


        # Show image in its own window
        window_name = f"Image: {img_filename}"
        cv2.imshow(window_name, img)

        # Wait for a keypress, close this window on any key
        print(f"Showing {img_filename}. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    print("Finished displaying all sampled images.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not COCO_INPUT_FILE.exists():
        print(f"Error: COCO JSON file not found at {COCO_INPUT_FILE}")
        print("Please run 'de_project_and_annotate.py' first.")
    else:
        print(f"Loading data from {COCO_INPUT_FILE}...")
        try:
            with open(COCO_INPUT_FILE, 'r') as f:
                coco_data = json.load(f)
            
            images_map, annotations_map, category_map = get_annotations_by_image_id(coco_data)
            
            # Change '5' to the number of images you want to randomly sample and display
            visualize_random_images(images_map, annotations_map, category_map, num_samples=10)
            
        except json.JSONDecodeError:
            print("Error: Failed to decode the COCO JSON file. Check for formatting errors.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")