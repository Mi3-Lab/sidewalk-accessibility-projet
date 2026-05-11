#!/usr/bin/env python3
"""
Train final model checkpoints for generalization testing.

Follows the exact same pipeline as paper/manuscript.tex:
- Load tallies_firebase.json (vote distributions from 829 users)
- Extract features with frozen encoder
- Train per-aid linear probe with Soft-KL loss
- Save checkpoints for generalization evaluation

Usage:
    python src/models/train_final_model.py \
        --encoder dinov2-large \
        --output_dir results/models/dinov2-large \
        --loss_type soft_kl

Output:
    results/models/dinov2-large/
    ├── walking_cane/
    │   ├── probe.pth
    │   └── scaler.joblib
    ├── walker/
    │   ├── probe.pth
    │   └── scaler.joblib
    └── ... (5 aids total)
"""

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

# Constants (match paper exactly)
RANDOM_STATE = 42
CLASS3 = ["no", "unsure", "yes"]
AIDS = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]

ENCODERS = {
    "clip-vit-b32": ("openai/clip-vit-base-patch32", "clip"),
    "clip-vit-b16": ("openai/clip-vit-base-patch16", "clip"),
    "clip-vit-l14": ("openai/clip-vit-large-patch14", "clip"),
    "dinov2-base": ("facebook/dinov2-base", "dino"),
    "dinov2-large": ("facebook/dinov2-large", "dino"),
    "siglip2-base": ("google/siglip2-base-patch16-224", "siglip"),
    "siglip2-so400m": ("google/siglip2-so400m-patch14-384", "siglip"),
    "vit-b16-sup": ("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", "timm"),
}


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_encoder(encoder_key: str, device: str):
    """Load frozen vision encoder (match paper Section 3.1)."""
    if encoder_key not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder_key}")
    
    model_id, encoder_type = ENCODERS[encoder_key]
    print(f"  Loading {encoder_key} ({model_id})...")
    
    if encoder_type == "clip":
        model = AutoModel.from_pretrained(model_id)
        model = model.vision_model
    elif encoder_type == "dino":
        model_name = model_id.split("/")[1]
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    elif encoder_type == "siglip":
        model = AutoModel.from_pretrained(model_id)
        model = model.vision_model
    elif encoder_type == "timm":
        import timm
        model = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)
    
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def extract_features(image_path: str, encoder, processor, device: str) -> np.ndarray:
    """Extract L2-normalized features (match paper Section 3.1)."""
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = encoder(**inputs)
            
            if hasattr(outputs, 'pooler_output'):
                feat = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                feat = outputs.last_hidden_state[:, 0, :]
            else:
                feat = outputs[0][:, 0, :]
        
        # L2 normalize
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()
    except Exception as e:
        return None


def train_probe_for_aid(X_train, y_train_dist, X_val, y_val_dist, 
                        loss_type="soft_kl", device="cuda"):
    """Train linear probe for one mobility aid (match paper Section 3.2-3.3)."""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Per-aid linear head
    feat_dim = X_train.shape[1]
    probe = nn.Linear(feat_dim, 3)
    probe = probe.to(device)
    
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_f1 = 0
    best_weights = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(50):
        probe.train()
        X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_dist_tensor = torch.tensor(y_train_dist, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        logits = probe(X_tensor)
        
        if loss_type == "soft_kl":
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, y_dist_tensor, reduction="batchmean")
        else:
            y_hard = torch.argmax(y_dist_tensor, dim=1)
            loss = F.cross_entropy(logits, y_hard)
        
        loss.backward()
        optimizer.step()
        
        probe.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            logits_val = probe(X_val_tensor)
            preds_val = torch.argmax(logits_val, dim=1).cpu().numpy()
            
            y_val_hard = np.argmax(y_val_dist, axis=1)
            f1 = f1_score(y_val_hard, preds_val, average="macro", zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_weights is not None:
        probe.load_state_dict(best_weights)
    
    return probe, scaler


def main():
    parser = argparse.ArgumentParser(
        description="Train final model checkpoints (paper pipeline)."
    )
    parser.add_argument(
        "--tallies_json",
        type=str,
        default="data/processed/tallies_firebase.json",
        help="Vote distributions from 829 users",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/images/sidewalk-images",
        help="Sidewalk crop images directory",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2-large",
        help="Frozen encoder to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/models/dinov2-large",
        help="Where to save checkpoints",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["soft_kl", "hard_ce"],
        default="soft_kl",
        help="Loss function (paper: soft_kl is primary)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*70)
    print(f"TRAIN FINAL MODEL — {args.encoder.upper()}")
    print("="*70)
    print(f"Loss:   {args.loss_type}")
    print(f"Device: {device}")
    print("")
    
    # Load vote distributions
    print(f"📂 Loading distributions from {args.tallies_json}...")
    with open(args.tallies_json, "r") as f:
        tallies = json.load(f)
    print(f"   Found {len(tallies)} image entries")
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    # Load encoder and processor
    print(f"\n🔄 Loading encoder...")
    encoder = load_encoder(args.encoder, device)
    model_id = ENCODERS[args.encoder][0]
    processor = AutoImageProcessor.from_pretrained(model_id)
    
    # Process each mobility aid
    print(f"\n📊 Training per-aid probes...")
    for aid in AIDS:
        aid_key = aid.lower().replace(" ", "_")
        print(f"\n  {aid}:")
        
        X_list = []
        y_dist_list = []
        
        for image_id, image_data in tallies.items():
            if aid not in image_data:
                continue
            
            img_path = images_dir / image_data["image_filename"]
            if not img_path.exists():
                continue
            
            feat = extract_features(str(img_path), encoder, processor, device)
            if feat is None:
                continue
            
            dist_dict = image_data[aid]
            y_dist = np.array([
                dist_dict.get("p_no", 0.0),
                dist_dict.get("p_unsure", 0.0),
                dist_dict.get("p_yes", 0.0)
            ])
            
            X_list.append(feat[0])
            y_dist_list.append(y_dist)
        
        if len(X_list) == 0:
            print(f"    ⚠️  No samples found")
            continue
        
        X = np.array(X_list)
        y_dist = np.array(y_dist_list)
        
        print(f"    Samples: {len(X)}")
        
        n = len(X)
        split_idx = int(0.8 * n)
        idx = np.random.permutation(n)
        
        X_train = X[idx[:split_idx]]
        y_train_dist = y_dist[idx[:split_idx]]
        
        X_val = X[idx[split_idx:]]
        y_val_dist = y_dist[idx[split_idx:]]
        
        print(f"    Training...")
        probe, scaler = train_probe_for_aid(
            X_train, y_train_dist,
            X_val, y_val_dist,
            loss_type=args.loss_type,
            device=device
        )
        
        aid_dir = output_dir / aid_key
        aid_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(probe.state_dict(), aid_dir / "probe.pth")
        joblib.dump(scaler, aid_dir / "scaler.joblib")
        
        print(f"    ✅ Saved: {aid_dir}/")
    
    print(f"\n" + "="*70)
    print(f"✅ All checkpoints saved to {output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
