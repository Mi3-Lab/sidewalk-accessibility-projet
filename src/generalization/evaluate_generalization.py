#!/usr/bin/env python3
"""
Generalization Test: Evaluate trained model on new cities (out-of-distribution).

Task 5.7: Test if model trained on GSV data generalizes to new Project Sidewalk cities.

Metrics:
  - Agreement with Project Sidewalk structural labels (CurbRamp = accessible, NoCurbRamp = inaccessible)
  - Per-aid agreement rate
  - Confusion matrix vs PS labels

Usage:
    python src/generalization/evaluate_generalization.py \
        --encoder dinov2-large \
        --checkpoint results/models/dinov2-large/best.pt \
        --test_images data/generalization/test_images.csv \
        --output_dir results/generalization/dinov2-large \
        --seed 42

Expected input CSV (test_images.csv):
    image_id,image_path,ps_label,city,source
    001,data/generalization/images/dc_001.jpg,CurbRamp,Washington DC,PSW
    002,data/generalization/images/col_001.jpg,NoCurbRamp,Columbus OH,PSW
    ...

PS_label: CurbRamp (accessible) or NoCurbRamp (inaccessible)
source: PSW (Project Sidewalk) or external source

Output:
    results/generalization/dinov2-large/
    ├── predictions.csv          # All predictions with agreement
    ├── agreement_summary.json   # Overall accuracy by encoder/aid
    └── confusion_matrix_per_aid.json # Breakdown per mobility aid
"""

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

# Optional W&B tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Constants (match train.py exactly)
RANDOM_STATE = 42
CLASS3 = ["no", "unsure", "yes"]
AIDS = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]

# Map PS labels to binary accessibility
# CurbRamp = accessible = "yes"
# NoCurbRamp = inaccessible = "no"
PS_LABEL_MAP = {
    "CurbRamp": "yes",
    "NoCurbRamp": "no",
}

# Inverse mapping for validation
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS3)}


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set reproducibility seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_encoder_and_get_features(image_path: str, encoder_key: str, device: str) -> np.ndarray:
    """
    Load image and extract features using the specified encoder.
    
    Supported encoders:
      - clip-vit-b32, clip-vit-b16, clip-vit-l14
      - dinov2-base, dinov2-large
      - siglip2-base, siglip2-so400m
      - vit-b16-sup
    
    Returns: (1, feature_dim) numpy array
    """
    ENCODERS = {
        "clip-vit-b32":   ("openai/clip-vit-base-patch32",     "clip"),
        "clip-vit-b16":   ("openai/clip-vit-base-patch16",     "clip"),
        "clip-vit-l14":   ("openai/clip-vit-large-patch14",    "clip"),
        "dinov2-base":    ("facebook/dinov2-base",              "dinov2"),
        "dinov2-large":   ("facebook/dinov2-large",             "dinov2"),
        "siglip2-base":   ("google/siglip2-base-patch16-224",   "siglip"),
        "siglip2-so400m": ("google/siglip2-so400m-patch14-384", "siglip"),
        "vit-b16-sup":    ("vit_base_patch16_224.augreg2_in21k_ft_in1k", "timm"),
    }
    
    if encoder_key not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder_key}. Available: {list(ENCODERS.keys())}")
    
    model_id, encoder_type = ENCODERS[encoder_key]
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Load encoder and extract features
    if encoder_type == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model = model.to(device).eval()
        
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            feat = outputs / outputs.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy()
    
    elif encoder_type == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model = model.to(device).eval()
        
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feat = outputs.last_hidden_state[:, 0, :]  # CLS token
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy()
    
    elif encoder_type == "siglip":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model = model.to(device).eval()
        
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            feat = outputs / outputs.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy()
    
    elif encoder_type == "timm":
        import timm
        import torchvision.transforms as transforms
        
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model = model.to(device).eval()
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img_t = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_t)
            feat = outputs / outputs.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy()
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return feat


def predict_single_image(
    image_path: str,
    encoder_key: str,
    models_checkpoint_dir: Path,
    device: str,
) -> dict:
    """
    Make predictions for a single image across all mobility aids.
    
    Returns dict:
        {
            aid1: {prob_no, prob_unsure, prob_yes, pred_class},
            aid2: {...},
            ...
        }
    """
    feat = load_encoder_and_get_features(image_path, encoder_key, device)
    
    results = {}
    for aid in AIDS:
        aid_key = aid.lower().replace(" ", "_")
        models_checkpoint_dir = Path(models_checkpoint_dir)
        
        # Load scaler and probe for this aid
        scaler_path = models_checkpoint_dir / f"{aid_key}" / "scaler.joblib"
        probe_path = models_checkpoint_dir / f"{aid_key}" / "probe.pth"
        
        if not scaler_path.exists() or not probe_path.exists():
            print(f"  ⚠️  Checkpoint not found for {aid} (expected at {models_checkpoint_dir}/{aid_key}/)")
            continue
        
        scaler = joblib.load(scaler_path)
        
        # Load probe model
        probe = nn.Linear(feat.shape[1], 3)
        probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
        probe = probe.to(device).eval()
        
        # Normalize features and predict
        feat_scaled = scaler.transform(feat)
        feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = probe(feat_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_class = CLASS3[np.argmax(probs)]
        
        results[aid] = {
            "prob_no": float(probs[0]),
            "prob_unsure": float(probs[1]),
            "prob_yes": float(probs[2]),
            "pred_class": pred_class,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generalization on out-of-distribution cities."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2-large",
        help="Encoder key (e.g., dinov2-large, clip-vit-b32)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Directory containing trained probes (e.g., results/models/dinov2-large/)",
    )
    parser.add_argument(
        "--test_images",
        type=str,
        required=True,
        help="CSV with image_path, ps_label, city, source columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )
    # W&B tracking arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable W&B experiment tracking",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sidewalk-generalization",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="W&B run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="W&B entity (username or team name)",
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize W&B if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_config = {
            "encoder": args.encoder,
            "seed": args.seed,
            "device": device,
        }
        wandb_run_name = args.wandb_run_name or f"gen_{args.encoder}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            tags=["generalization", "transfer", args.encoder],
        )
        print(f"✅ W&B tracking enabled: {args.wandb_project}/{wandb_run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  W&B requested but not installed. Install with: pip install wandb")
        args.use_wandb = False
    
    # Load test data
    print(f"\n[Generalization] Loading test images from {args.test_images}...")
    test_df = pd.read_csv(args.test_images)
    print(f"  Found {len(test_df)} images")
    
    # Predict on all test images
    print(f"\n[Generalization] Predicting on {len(test_df)} images using {args.encoder}...")
    predictions = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = row["image_path"]
        ps_label = row.get("ps_label", "unknown")
        city = row.get("city", "unknown")
        
        if not Path(image_path).exists():
            print(f"  ⚠️  Image not found: {image_path}")
            continue
        
        preds_by_aid = predict_single_image(
            image_path,
            args.encoder,
            args.checkpoint,
            device,
        )
        
        # Convert binary PS labels to YES/NO (assuming binary accessibility)
        ps_class = PS_LABEL_MAP.get(ps_label, "unknown")
        
        # Average predictions across all aids
        all_probs_yes = [p["prob_yes"] for p in preds_by_aid.values()]
        if all_probs_yes:
            avg_prob_yes = np.mean(all_probs_yes)
            pred_binary = "yes" if avg_prob_yes >= 0.5 else "no"
        else:
            pred_binary = "unknown"
            avg_prob_yes = 0.0
        
        # Agreement check
        agrees = (pred_binary == ps_class) if ps_class != "unknown" else None
        
        # Store row
        pred_row = {
            "image_id": row.get("image_id", idx),
            "image_path": image_path,
            "city": city,
            "ps_label": ps_label,
            "ps_class": ps_class,
            "pred_binary": pred_binary,
            "avg_prob_yes": avg_prob_yes,
            "agrees": agrees,
        }
        
        # Add per-aid predictions
        for aid, preds in preds_by_aid.items():
            pred_row[f"{aid}_pred"] = preds["pred_class"]
            pred_row[f"{aid}_prob_yes"] = preds["prob_yes"]
        
        predictions.append(pred_row)
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Compute summary stats
    print(f"\n[Generalization] Computing summary statistics...")
    
    # Binary agreement (yes/no)
    valid_mask = pred_df["ps_class"] != "unknown"
    pred_df_valid = pred_df[valid_mask]
    
    if len(pred_df_valid) > 0:
        overall_accuracy = accuracy_score(pred_df_valid["ps_class"], pred_df_valid["pred_binary"])
        overall_balanced_acc = balanced_accuracy_score(pred_df_valid["ps_class"], pred_df_valid["pred_binary"])
        conf_matrix = confusion_matrix(pred_df_valid["ps_class"], pred_df_valid["pred_binary"], labels=["no", "yes"])
    else:
        overall_accuracy = 0.0
        overall_balanced_acc = 0.0
        conf_matrix = np.array([[0, 0], [0, 0]])
    
    # Per-city accuracy
    city_accuracy = {}
    for city in pred_df_valid["city"].unique():
        city_mask = (pred_df["city"] == city) & valid_mask
        if city_mask.sum() > 0:
            city_preds = pred_df[city_mask]
            city_acc = accuracy_score(city_preds["ps_class"], city_preds["pred_binary"])
            city_accuracy[city] = {
                "n_samples": int(city_mask.sum()),
                "accuracy": float(city_acc),
            }
    
    # Summary JSON
    summary = {
        "encoder": args.encoder,
        "n_test_samples": len(pred_df),
        "n_valid_labels": len(pred_df_valid),
        "overall_accuracy": float(overall_accuracy),
        "overall_balanced_accuracy": float(overall_balanced_acc),
        "confusion_matrix": {
            "labels": ["no", "yes"],
            "matrix": conf_matrix.tolist(),
        },
        "per_city_accuracy": city_accuracy,
    }
    
    # Log to W&B if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "encoder": args.encoder,
            "n_test_samples": len(pred_df),
            "n_valid_labels": len(pred_df_valid),
            "overall_accuracy": overall_accuracy,
            "overall_balanced_accuracy": overall_balanced_acc,
            "confusion_matrix": conf_matrix,
        })
        
        # Log per-city accuracies
        for city, stats in city_accuracy.items():
            wandb.log({f"city_{city}_accuracy": stats["accuracy"]})
        
        print(f"  📊 Metrics logged to W&B")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save predictions CSV
    pred_csv = Path(args.output_dir) / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"  ✅ Predictions saved to {pred_csv}")
    
    # Save summary JSON
    summary_json = Path(args.output_dir) / "agreement_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary saved to {summary_json}")
    
    # Print results
    print("\n" + "="*70)
    print(f"GENERALIZATION TEST RESULTS — {args.encoder}")
    print("="*70)
    print(f"Test samples:           {len(pred_df)}")
    print(f"Valid PS labels:        {len(pred_df_valid)}")
    print(f"Overall Accuracy:       {overall_accuracy:.3f}")
    print(f"Balanced Accuracy:      {overall_balanced_acc:.3f}")
    print(f"\nConfusion Matrix (no, yes):")
    print(f"  {conf_matrix}")
    print(f"\nPer-City Accuracy:")
    for city, stats in city_accuracy.items():
        print(f"  {city:20s}: {stats['accuracy']:.3f} (n={stats['n_samples']})")
    print("="*70 + "\n")    
    # Close W&B if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("✅ W&B run finished")

if __name__ == "__main__":
    main()
