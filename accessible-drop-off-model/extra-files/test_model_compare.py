import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
from scipy import ndimage
import pandas as pd
import sys
import os
import glob

# ---------- PATHS ----------
sys.path.append('extra-files')
from training.sidewalk_model_train_compare import AccessibilityModel

# ---------- CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "accessibility_checkpoint_compare.pth"  
mobility_classes = ["Cane", "Walker", "MobilityScooter", "ManualWheelchair", "MotorizedWheelchair"]

# ---------- LOAD ACCESSIBILITY MODEL ----------
def load_model(model_path):
    """Load the Q-based accessibility model"""
    model = AccessibilityModel(embed_dim=256, num_heads=8, num_layers=3)
    model.to(device)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model weights from checkpoint.")
            if "epoch" in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
            if "val_loss" in checkpoint:
                print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights (plain state_dict).")
    else:
        raise ValueError("Unrecognized checkpoint format.")
    
    model.eval()
    return model


# ---------- SIDEWALK DETECTION ----------
mask_model = YOLO("bestv12.pt")

def extract_sidewalk_crop(scene_image, crop_size=(224, 224), conf=0.5):
    """Extract sidewalk regions using YOLO sidewalk mask"""
    results = mask_model.predict(scene_image, classes=[1], conf=conf, verbose=False)
    mask_arr = np.zeros((scene_image.height, scene_image.width), dtype=bool)

    if results and results[0].masks is not None:
        masks_data = results[0].masks.data
        composite_mask = torch.any(masks_data, dim=0).cpu().numpy().astype("uint8")
        mask_pil = Image.fromarray(composite_mask * 255).convert("L")
        if mask_pil.size != scene_image.size:
            mask_pil = mask_pil.resize(scene_image.size, resample=Image.NEAREST)
        mask_arr = np.array(mask_pil) > 0

    # Default: use center crop if no sidewalks found
    if mask_arr.sum() == 0:
        w, h = scene_image.size
        crop_w, crop_h = min(w, h), min(w, h)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        sidewalk_crop = scene_image.crop((left, top, left + crop_w, top + crop_h))
    else:
        # Find and crop the largest sidewalk (TODO: multiple sidewalks)
        labeled, ncomponents = ndimage.label(mask_arr)
        if ncomponents > 0:
            sizes = ndimage.sum(mask_arr, labeled, range(1, ncomponents + 1))
            largest_component = (labeled == (np.argmax(sizes) + 1))
            ys, xs = np.where(largest_component)
            miny, maxy = ys.min(), ys.max()
            minx, maxx = xs.min(), xs.max()
            pad = 16
            minx = max(0, minx - pad)
            miny = max(0, miny - pad)
            maxx = min(scene_image.width - 1, maxx + pad)
            maxy = min(scene_image.height - 1, maxy + pad)
            sidewalk_crop = scene_image.crop((minx, miny, maxx + 1, maxy + 1))
        else:
            sidewalk_crop = scene_image.copy()

    return sidewalk_crop.resize(crop_size, resample=Image.BILINEAR)


# ---------- TEST FUNCTIONS ----------
def test_single_image():
    """Run model on a single image"""
    model = load_model(model_path)
    if model is None:
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_image_path = "extra_images/Crosswalk/320.jpg"
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    scene_img = Image.open(test_image_path).convert("RGB")
    sidewalk_img = extract_sidewalk_crop(scene_img)

    scene_tensor = transform(scene_img).unsqueeze(0).to(device)
    sidewalk_tensor = transform(sidewalk_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(scene_tensor, sidewalk_tensor)
        predictions = output.cpu().numpy().flatten()

    print("\nAccessibility Scores (0 = inaccessible, 1 = fully accessible):")
    for i, mobility_class in enumerate(mobility_classes):
        score = predictions[i]
        level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        print(f"  {mobility_class:20s}: {score:.3f} ({level})")

    print(f"  {'Overall Average':20s}: {np.mean(predictions):.3f}")


def test_multiple_images():
    """Test on multiple sample images"""
    model = load_model(model_path)
    if model is None:
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_images = [
        "Project_Sidewalk_Data/sidewalk-images/gsv-amsterdam-26619-4-5.png",
        "Project_Sidewalk_Data/sidewalk-images/gsv-seattle-100492-3-0.png",
        "Project_Sidewalk_Data/sidewalk-images/gsv-chicago-631-4-3.png",
        "Project_Sidewalk_Data/sidewalk-images/gsv-st_louis-2022-4-4.png",
        "Project_Sidewalk_Data/sidewalk-images/gsv-oradell-456-4-2.png"
    ]

    print("\n" + "="*80)
    print("TESTING MODEL (Q-BASED TRAINING) ON MULTIPLE IMAGES")
    print("="*80)

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_path}")
            continue

        print(f"\n📷 {os.path.basename(img_path)}")
        scene_img = Image.open(img_path).convert("RGB")
        sidewalk_img = extract_sidewalk_crop(scene_img)

        scene_tensor = transform(scene_img).unsqueeze(0).to(device)
        sidewalk_tensor = transform(sidewalk_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(scene_tensor, sidewalk_tensor)
            predictions = output.cpu().numpy().flatten()

        print("Accessibility Scores:")
        for i, mobility_class in enumerate(mobility_classes):
            score = predictions[i]
            level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            print(f"  {mobility_class:20s}: {score:.3f} ({level})")

        print(f"  {'Overall Average':20s}: {np.mean(predictions):.3f}")


def compare_with_ground_truth():
    """Compare predictions with ground truth Q-based scores"""
    model = load_model(model_path)
    if model is None:
        return

    try:
        gt_df = pd.read_csv("Project_Sidewalk_Data/aggregated_q.csv")
    except FileNotFoundError:
        print("Ground truth file not found. Please run training to generate aggregated_q.csv")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n" + "="*80)
    print("COMPARING MODEL PREDICTIONS WITH Q-BASED GROUND TRUTH")
    print("="*80)

    for idx in range(min(15, len(gt_df))):
        row = gt_df.iloc[idx]
        image_id = str(int(row["ImageID"]))

        pattern = f"Project_Sidewalk_Data/sidewalk-images/*{image_id}*.png"
        matches = glob.glob(pattern)
        if not matches:
            print(f"Image not found for ID {image_id}")
            continue

        img_path = matches[0]
        print(f"\n📸 Image ID: {image_id} ({os.path.basename(img_path)})")

        scene_img = Image.open(img_path).convert("RGB")
        sidewalk_img = extract_sidewalk_crop(scene_img)

        scene_tensor = transform(scene_img).unsqueeze(0).to(device)
        sidewalk_tensor = transform(sidewalk_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(scene_tensor, sidewalk_tensor)
            predictions = output.cpu().numpy().flatten()

        print("Comparison (Predicted vs Ground Truth):")
        total_error = 0
        for i, mobility_class in enumerate(mobility_classes):
            pred = predictions[i]
            gt = row.get(mobility_class, 0.0)
            error = abs(pred - gt)
            total_error += error
            print(f"  {mobility_class:20s}: {pred:.3f} vs {gt:.3f} (error: {error:.3f})")

        print(f"  {'Average Error':20s}: {total_error / len(mobility_classes):.3f}")


if __name__ == "__main__":

    test_multiple_images()
    compare_with_ground_truth()

    # test_single_image()
