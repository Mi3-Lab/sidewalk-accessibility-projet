import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
from scipy import ndimage
import sys
import os

# TODO: CURRENTLY ONLT USES LARGEST SIDEWALK - DETECT ALL SIDEWALKS

sys.path.append('extra-files')
from sidewalk_model_train_fixed import AccessibilityModel

# ---------- CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "accessibility_checkpoint_fixed.pth"  
mobility_classes = ["Cane", "Walker", "MobilityScooter", "ManualWheelchair", "MotorizedWheelchair"]

# ---------- LOAD FIXED MODEL ----------
def load_model(model_path):
    """Load the fixed accessibility model"""
    model = AccessibilityModel(embed_dim=256, num_heads=8, num_layers=3)
    model.to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded fixed model weights from checkpoint.")
                if "epoch" in checkpoint:
                    print(f"Checkpoint epoch: {checkpoint['epoch']}")
                if "val_loss" in checkpoint:
                    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights from plain state_dict file.")
        else:
            raise ValueError("Unrecognized checkpoint format.")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        return None
    
    model.eval()
    return model

# Load CV Model for Sidewalk Extraction
mask_model = YOLO("bestv12.pt")

def extract_sidewalk_crop(scene_image, crop_size=(224, 224), conf=0.5):
    """Extract sidewalk region using YOLO model - same as training"""
    results = mask_model.predict(scene_image, classes=[1], conf=conf, verbose=False)
    mask_arr = np.zeros((scene_image.height, scene_image.width), dtype=bool)

    if results and results[0].masks is not None:
        masks_data = results[0].masks.data
        composite_mask = torch.any(masks_data, dim=0).cpu().numpy().astype("uint8")
        mask_pil = Image.fromarray(composite_mask * 255).convert("L")
        if mask_pil.size != scene_image.size:
            mask_pil = mask_pil.resize(scene_image.size, resample=Image.NEAREST)
        mask_arr = np.array(mask_pil) > 0

    if mask_arr.sum() == 0:
        # No sidewalk detected, use center crop
        w, h = scene_image.size
        crop_w, crop_h = min(w, h), min(w, h)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        sidewalk_crop = scene_image.crop((left, top, left + crop_w, top + crop_h))
    else:
        # Find largest connected component ONLY USES LARGEST SIDEWALK
        # TODO : DETECT ALL SIDEWALKS
        labeled, ncomponents = ndimage.label(mask_arr)
        if ncomponents > 0:
            sizes = ndimage.sum(mask_arr, labeled, range(1, ncomponents + 1))
            largest_component = (labeled == (np.argmax(sizes) + 1))
            ys, xs = np.where(largest_component)
            miny, maxy = ys.min(), ys.max()
            minx, maxx = xs.min(), xs.max()
            
            # Add padding
            pad = 16
            minx = max(0, minx - pad)
            miny = max(0, miny - pad)
            maxx = min(scene_image.width - 1, maxx + pad)
            maxy = min(scene_image.height - 1, maxy + pad)
            
            sidewalk_crop = scene_image.crop((minx, miny, maxx + 1, maxy + 1))
        else:
            sidewalk_crop = scene_image.copy()

    return sidewalk_crop.resize(crop_size, resample=Image.BILINEAR)

def test_single_image():
    """Test model on a single image"""
    model = load_model(model_path)
    if model is None:
        return
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_image_path = "extra_images/Crosswalk/320.jpg"

    scene_img = Image.open(test_image_path).convert("RGB")
    sidewalk_img = extract_sidewalk_crop(scene_img)
    
    # Transform
    scene_tensor = transform(scene_img).unsqueeze(0).to(device)
    sidewalk_tensor = transform(sidewalk_img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(scene_tensor, sidewalk_tensor)
        predictions = output.cpu().numpy().flatten()
    
    # Display results
    print("Accessibility Scores (0.0 = inaccessible, 1.0 = fully accessible):")
    for i, mobility_class in enumerate(mobility_classes):
        score = predictions[i]
        accessibility_level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        print(f"  {mobility_class:20s}: {score:.3f} ({accessibility_level})")
    
    # Overall accessibility
    overall_score = np.mean(predictions)
    print(f"  {'Overall Average':20s}: {overall_score:.3f}")



def test_multiple_images():
    """Test the model on multiple images to see the range of predictions"""
    
    # Load the fixed model
    model = load_model(model_path)
    if model is None:
        return
    
    # Transform 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test images
    test_images = [
        "Project_Sidewalk_Data/sidewalk-images/gsv-amsterdam-26619-4-5.png",
        "Project_Sidewalk_Data/sidewalk-images/gsv-seattle-100492-3-0.png", 
        "Project_Sidewalk_Data/sidewalk-images/gsv-chicago-631-4-3.png",
        "Project_Sidewalk_Data/sidewalk-images/gsv-st_louis-2022-4-4.png",
        "Project_Sidewalk_Data/sidewalk-images/gsv-oradell-456-4-2.png"
    ]
    
    print("\n" + "="*80)
    print("TESTING FIXED MODEL ON MULTIPLE IMAGES")
    print("="*80)
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue
            
        print(f"\nTesting: {os.path.basename(img_path)}")
        
        # Load and process image
        scene_img = Image.open(img_path).convert("RGB")
        sidewalk_img = extract_sidewalk_crop(scene_img)
        
        # Transform
        scene_tensor = transform(scene_img).unsqueeze(0).to(device)
        sidewalk_tensor = transform(sidewalk_img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(scene_tensor, sidewalk_tensor)
            predictions = output.cpu().numpy().flatten()
        
        # Display results
        print("Accessibility Scores (0.0 = inaccessible, 1.0 = fully accessible):")
        for i, mobility_class in enumerate(mobility_classes):
            score = predictions[i]
            accessibility_level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            print(f"  {mobility_class:20s}: {score:.3f} ({accessibility_level})")
        
        # Overall accessibility
        overall_score = np.mean(predictions)
        print(f"  {'Overall Average':20s}: {overall_score:.3f}")

def compare_with_ground_truth():
    """Compare predictions with ground truth from aggregated data"""
    
    # Load the fixed model
    model = load_model(model_path)
    if model is None:
        return
    
    # Load ground truth
    import pandas as pd
    try:
        gt_df = pd.read_csv("Project_Sidewalk_Data/aggregated_fixed.csv")
    except FileNotFoundError:
        print("Ground truth file not found. Please run the training script first to generate aggregated_fixed.csv")
        return
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*80)
    print("COMPARING PREDICTIONS WITH GROUND TRUTH")
    print("="*80)
    
    # Test on first few samples
    for idx in range(min(5, len(gt_df))):
        row = gt_df.iloc[idx]
        image_id = str(int(row["ImageID"]))
        
        # Find corresponding image
        img_path = None
        for path in ["Project_Sidewalk_Data/sidewalk-images/gsv-amsterdam-{}-4-5.png".format(image_id),
                     "Project_Sidewalk_Data/sidewalk-images/gsv-seattle-{}-3-0.png".format(image_id),
                     "Project_Sidewalk_Data/sidewalk-images/gsv-chicago-{}-4-3.png".format(image_id),
                     "Project_Sidewalk_Data/sidewalk-images/gsv-st_louis-{}-4-4.png".format(image_id),
                     "Project_Sidewalk_Data/sidewalk-images/gsv-oradell-{}-4-2.png".format(image_id)]:
            if os.path.exists(path):
                img_path = path
                break
        
        # Try a more general search
        if img_path is None:
            import glob
            pattern = f"Project_Sidewalk_Data/sidewalk-images/*{image_id}*.png"
            matches = glob.glob(pattern)
            if matches:
                img_path = matches[0]
        
        if img_path is None:
            print(f"❌ Image not found for ID: {image_id}")
            continue
        
        print(f"\n📸 Image ID: {image_id} ({os.path.basename(img_path)})")
        
        # Load and process image
        scene_img = Image.open(img_path).convert("RGB")
        sidewalk_img = extract_sidewalk_crop(scene_img)
        
        # Transform
        scene_tensor = transform(scene_img).unsqueeze(0).to(device)
        sidewalk_tensor = transform(sidewalk_img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(scene_tensor, sidewalk_tensor)
            predictions = output.cpu().numpy().flatten()
        
        # Compare with ground truth
        print("Comparison (Predicted vs Ground Truth):")
        total_error = 0
        for i, mobility_class in enumerate(mobility_classes):
            pred_score = predictions[i]
            gt_score = row[mobility_class]
            error = abs(pred_score - gt_score)
            total_error += error
            print(f"  {mobility_class:20s}: {pred_score:.3f} vs {gt_score:.3f} (error: {error:.3f})")
        
        avg_error = total_error / len(mobility_classes)
        print(f"  {'Average Error':20s}: {avg_error:.3f}")

if __name__ == "__main__":
    print("FIXED ACCESSIBILITY MODEL TESTING")
    print("This version uses the corrected model architecture and data processing.")
    
    # # Test multiple images
    # test_multiple_images()
    
    # Compare with ground truth if available
    # compare_with_ground_truth()

     # Test single image
    test_single_image()