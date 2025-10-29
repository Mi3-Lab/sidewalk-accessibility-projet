import os
import json
import numpy as np
import cv2
from pathlib import Path
from numpy import pi, sin, cos, tan, atan2

# --- CONFIGURATION ---
INPUT_IMG_DIR = Path('train/images')
INPUT_LABEL_DIR = Path('train/labels')
OUTPUT_DIR = Path('data_transformed')
FOV_DEGREES = 90
TARGET_WIDTH = 640
TARGET_HEIGHT = 640
PANO_ANGLES = [0, pi/2, -pi/2]  # Front, Left, Right views
ANGLE_NAMES = {0: 'FRONT', pi/2: 'LEFT', -pi/2: 'RIGHT'}
CLASS_NAMES = ["curbramp", "sidewalk"]  # <-- EDIT YOUR CLASSES HERE

# --- SETUP ---
def setup_directories():
    for view in ANGLE_NAMES.values():
        (OUTPUT_DIR / 'images' / view).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / view).mkdir(parents=True, exist_ok=True)
    if not INPUT_IMG_DIR.exists() or not INPUT_LABEL_DIR.exists():
        raise FileNotFoundError("Input images or labels directory not found.")

def polygon_area(coords):
    pts = np.array(coords).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# --- I/O HELPERS ---
def load_yolo_annotations(label_path, img_w, img_h):
    anns = []
    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            xs = np.array(coords[0::2]) * img_w
            ys = np.array(coords[1::2]) * img_h
            anns.append({
                'instance_id': i,
                'class': cls,
                'points': np.stack((xs, ys), axis=-1)
            })
    return anns

def save_yolo_annotations(anns, save_path):
    with open(save_path, 'w') as f:
        for ann in anns:
            pts = np.array(ann['points'], dtype=np.float32)
            pts[:, 0] /= TARGET_WIDTH
            pts[:, 1] /= TARGET_HEIGHT
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            f.write(f"{ann['class']} {coords}\n")

# --- PROJECTION ---
def project_annotations(anns, pano_w, pano_h, target_angle_rad):
    new_anns = []
    MARGIN = 10
    MAX_CLIPPED_POINTS = 5
    f = TARGET_WIDTH / (2 * tan(np.deg2rad(FOV_DEGREES) / 2))

    for ann in anns:
        new_pts = []
        clipped_count = 0
        
        for (x, y) in ann['points']:
            theta = (x / pano_w - 0.5) * (2 * pi)
            phi = (y / pano_h - 0.5) * pi
            theta_adj = theta - target_angle_rad
            x_s = cos(phi) * sin(theta_adj)
            y_s = sin(phi)
            z_s = cos(phi) * cos(theta_adj)
            if z_s <= 0:
                clipped_count += 1
                continue
            x_proj = f * x_s / z_s
            y_proj = f * y_s / z_s
            x_new = int(x_proj + TARGET_WIDTH / 2)
            y_new = int(y_proj + TARGET_HEIGHT / 2)
            if not (MARGIN <= x_new < TARGET_WIDTH - MARGIN and MARGIN <= y_new < TARGET_HEIGHT - MARGIN):
                clipped_count += 1
                x_new = np.clip(x_new, 0, TARGET_WIDTH - 1)
                y_new = np.clip(y_new, 0, TARGET_HEIGHT - 1)
            new_pts.append(np.array([x_new, y_new]))

        if len(new_pts) >= 3 and clipped_count <= MAX_CLIPPED_POINTS:
            new_anns.append({
                'instance_id': ann['instance_id'],
                'class': ann['class'],
                'points': np.array(new_pts)
            })
    return new_anns

def deproject_image(pano_img, target_angle_rad, out_path):
    map_x = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
    map_y = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
    f = TARGET_WIDTH / (2 * tan(np.deg2rad(FOV_DEGREES) / 2))
    for y_new in range(TARGET_HEIGHT):
        for x_new in range(TARGET_WIDTH):
            x_proj = x_new - TARGET_WIDTH / 2
            y_proj = y_new - TARGET_HEIGHT / 2
            theta_adj = atan2(x_proj, f)
            phi = np.arctan(y_proj / np.sqrt(x_proj**2 + f**2))
            theta = theta_adj + target_angle_rad
            x_pano = (theta / pi) / 2 + 0.5
            y_pano = phi / (pi / 2) / 2 + 0.5
            map_x[y_new, x_new] = x_pano * pano_img.shape[1]
            map_y[y_new, x_new] = y_pano * pano_img.shape[0]
    rect_img = cv2.remap(pano_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    cv2.imwrite(str(out_path), rect_img)

# --- YAML GENERATOR ---
def write_data_yaml():
    yaml_path = OUTPUT_DIR / "data.yaml"
    content = f"""train: {OUTPUT_DIR / 'images'}
val: {OUTPUT_DIR / 'images'}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"✅ data.yaml created at {yaml_path}")

# --- MAIN ---
def main():
    setup_directories()
    write_data_yaml() 

    pano_files = sorted([f for f in INPUT_IMG_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    print(f"Processing {len(pano_files)} panoramas...")

    instance_map = {}

    for pano_path in pano_files:
        base = pano_path.stem
        label_path = INPUT_LABEL_DIR / f"{base}.txt"
        if not label_path.exists():
            print(f"Skipping {base}: no label file found.")
            continue

        pano_img = cv2.imread(str(pano_path))
        pano_h, pano_w, _ = pano_img.shape
        anns = load_yolo_annotations(label_path, pano_w, pano_h)
        instance_map[base] = {"original_ids": [a["instance_id"] for a in anns], "views": {}}

        for angle_rad in PANO_ANGLES:
            angle_name = ANGLE_NAMES[angle_rad]
            view_img_dir = OUTPUT_DIR / 'images' / angle_name
            view_lbl_dir = OUTPUT_DIR / 'labels' / angle_name

            new_name = f"{base}_{angle_name}"
            img_out = view_img_dir / f"{new_name}.jpg"
            lbl_out = view_lbl_dir / f"{new_name}.txt"

            deproject_image(pano_img, angle_rad, img_out)
            reprojected = project_annotations(anns, pano_w, pano_h, angle_rad)
            save_yolo_annotations(reprojected, lbl_out)

            instance_map[base]["views"][angle_name] = [a["instance_id"] for a in reprojected]
            print(f"  -> {new_name}: {len(reprojected)} masks")

    with open(OUTPUT_DIR / "instance_map.json", "w") as f:
        json.dump(instance_map, f, indent=4)
    print("Saved instance map for debugging.")

    print("\n--- ✅ Complete ---")
    print(f"Images in: {OUTPUT_DIR / 'images/*'}")
    print(f"Labels in: {OUTPUT_DIR / 'labels/*'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
