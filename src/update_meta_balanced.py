#!/usr/bin/env python3
"""
Update existing meta.json files with balanced_accuracy by re-evaluating on validation data.
"""

import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

CLASS3 = ['no', 'unsure', 'yes']
AIDS = ["Walking cane", "Walker", "Mobility scooter", "Manual wheelchair", "Motorized wheelchair"]

def stratified_split_min1(df, test_size=0.2, random_state=42):
    """Split ensuring min 1 sample per class in test."""
    classes = df['label'].unique()
    train_indices = []
    test_indices = []
    for cls in classes:
        cls_df = df[df['label'] == cls]
        if len(cls_df) == 1:
            # If only 1 sample, put in train
            train_indices.extend(cls_df.index)
        else:
            cls_train, cls_test = train_test_split(cls_df.index, test_size=test_size, random_state=random_state, stratify=None)
            train_indices.extend(cls_train)
            test_indices.extend(cls_test)
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    return train_df, test_df

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, processor, device

def load_vision_model(model_name="resnet50"):
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "resnet50":
        model = resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == "convnext_base":
        model = timm.create_model('convnext_base', pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Unsupported model")
    model = model.to(device)
    model.eval()
    return model, transform, device

def extract_features_batch(images, model, processor, device, model_type="clip", batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        if model_type == "clip":
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                feat = outputs / outputs.norm(dim=-1, keepdim=True)
                feat = feat.cpu().numpy()
        else:
            batch_tensors = []
            for img in batch_images:
                tensor = processor(img).unsqueeze(0).to(device)
                batch_tensors.append(tensor)
            batch_tensor = torch.cat(batch_tensors, dim=0)
            with torch.no_grad():
                outputs = model(batch_tensor)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                if outputs.dim() == 4:
                    feat = outputs.view(outputs.size(0), -1).cpu().numpy()
                elif outputs.dim() == 3:
                    feat = outputs[:, 0, :].cpu().numpy()
                elif outputs.dim() == 2:
                    feat = outputs.cpu().numpy()
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")
        features.append(feat)
    return np.vstack(features)

def update_meta(models_dir, tallies_json, images_dir):
    tallies = pd.read_json(tallies_json)
    tallies['path'] = tallies['ImageID'].apply(lambda x: Path(images_dir) / f"{x}.jpg")
    tallies = tallies[tallies['path'].apply(lambda p: p.exists())]

    for subdir in Path(models_dir).iterdir():
        if subdir.is_dir():
            meta_path = subdir / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if 'balanced_accuracy' in meta:
                    continue  # Already updated

                aid = meta['aid']
                model_name = meta['model_name']
                model_type = meta['backbone'].split('::')[0]

                df_aid = tallies[tallies['MobilityAid'] == aid]
                if df_aid.empty:
                    continue

                # Map labels
                label_map = {'no': 0, 'unsure': 1, 'yes': 2}
                df_aid = df_aid.copy()
                df_aid['label'] = df_aid['Accessible'].map(label_map)

                # Split
                train_df, val_df = stratified_split_min1(df_aid, test_size=0.2, random_state=42)

                # Load model
                if model_type == "clip":
                    model, processor, device = load_clip_model(model_name)
                else:
                    model, processor, device = load_vision_model(model_name)

                # Extract features for val
                val_images = [Image.open(p).convert("RGB") for p in val_df['path']]
                X_val = extract_features_batch(val_images, model, processor, device, model_type)
                y_val = val_df['label'].values

                # Load scaler and probe
                scaler = joblib.load(subdir / "scaler.joblib")
                probe = torch.nn.Linear(X_val.shape[1], 3)
                probe.load_state_dict(torch.load(subdir / "probe.pth"))
                probe.eval()
                probe = probe.to(device)

                # Predict
                X_val_s = scaler.transform(X_val)
                X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = probe(X_val_t)
                    y_pred = outputs.argmax(dim=1).cpu().numpy()

                # Compute balanced accuracy
                balanced_acc = balanced_accuracy_score(y_val, y_pred)

                # Update meta
                meta['balanced_accuracy'] = balanced_acc
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)
                print(f"Updated {meta_path} with balanced_accuracy: {balanced_acc:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--tallies_json", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    args = parser.parse_args()
    update_meta(args.models_dir, args.tallies_json, args.images_dir)