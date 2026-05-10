import argparse
import pandas as pd
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import joblib
import json
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models import resnet50
import timm

CLASS3 = ['no', 'unsure', 'yes']
AIDS = ["Walking cane", "Walker", "Mobility scooter", "Manual wheelchair", "Motorized wheelchair"]

def stratified_split_min1(df, label_col='argmax_label', test_size=0.2, random_state=42, max_tries=20):
    # Use train_test_split with stratify for simplicity
    from sklearn.model_selection import train_test_split
    tr_df, va_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col])
    return tr_df, va_df

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, processor, device

def load_vision_model(model_name="resnet50"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "resnet50":
        model = resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
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
    elif model_name == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = model.to(device)
        model.eval()
        return model, processor, device
    else:
        raise ValueError("Unsupported model")
    model = model.to(device)
    model.eval()
    return model, transform, device

def extract_features(model, processor, device, image_paths):
    feats = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting features"):
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            feat = outputs.pooler_output / outputs.pooler_output.norm(dim=-1, keepdim=True)  # L2 normalize
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0)

def extract_vision_features(model, transform, device, image_paths):
    feats = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting vision features"):
            image = Image.open(path).convert("RGB")
            inputs = transform(image).unsqueeze(0).to(device)
            outputs = model(inputs)
            # Handle output shape for different model types
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            # ResNet: (1, 2048, 1, 1) -> (1, 2048)
            if outputs.dim() == 4:
                feat = outputs.squeeze().flatten(0).cpu().numpy().reshape(1, -1)
            # ViT: (1, seq_len, hidden_dim) or (1, hidden_dim)
            elif outputs.dim() == 3:
                # Use class token (first token)
                feat = outputs[:, 0, :].cpu().numpy().reshape(1, -1)
            elif outputs.dim() == 2:
                feat = outputs.cpu().numpy().reshape(1, -1)
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
            feats.append(feat)
    return np.concatenate(feats, axis=0)

def find_image_path(image_id, images_dir):
    from pathlib import Path
    images_dir = Path(images_dir)
    for ext in ['.png', '.jpg', '.jpeg']:
        candidates = list(images_dir.glob(f"*{image_id}*{ext}"))
        if candidates:
            return candidates[0]  # Take first match
    return None

PROBE_LR     = 1e-3
PROBE_EPOCHS = 50
PROBE_WD     = 1e-4
TRAIN_SPLIT  = 0.8   # 80 % train / 20 % val


def train_aid_model(model, processor, device, df_aid, output_dir, aid, model_name, model_type="clip"):
    # Split at panorama level to prevent leakage between rectilinear crops of the same panorama
    unique_panos = df_aid["ImageID"].unique()
    n_val = max(1, int(len(unique_panos) * (1 - TRAIN_SPLIT)))
    rng = np.random.default_rng(42)
    val_panos = set(rng.choice(unique_panos, size=n_val, replace=False))
    tr_df = df_aid[~df_aid["ImageID"].isin(val_panos)]
    va_df = df_aid[df_aid["ImageID"].isin(val_panos)]

    def _feats(df_):
        paths = df_["path"].tolist()
        if model_type == "clip":
            return extract_features(model, processor, device, paths)
        elif model_type in ["resnet50", "vit_base_patch16_224", "convnext_base", "dinov2"]:
            return extract_vision_features(model, processor, device, paths)
        raise ValueError(f"Unknown model_type: {model_type}")

    X_train = _feats(tr_df)
    X_val   = _feats(va_df)

    label_map = {'p_no': 0, 'p_unsure': 1, 'p_yes': 2}
    y_train = tr_df['argmax_label'].map(label_map).values
    y_val   = va_df['argmax_label'].map(label_map).values
    w_train = tr_df['sample_weight'].values

    scaler    = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    import torch.nn as nn
    import torch.optim as optim
    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = nn.Linear(X_train_s.shape[1], 3).to(probe_device)
    optimizer = optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    criterion = nn.CrossEntropyLoss(reduction="none")

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(probe_device)
    y_train_t = torch.tensor(y_train,   dtype=torch.long).to(probe_device)
    w_train_t = torch.tensor(w_train,   dtype=torch.float32).to(probe_device)

    probe.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        losses = criterion(probe(X_train_t), y_train_t)
        (losses * w_train_t).mean().backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(probe_device)
        y_pred  = probe(X_val_t).argmax(dim=1).cpu().numpy()

    acc          = accuracy_score(y_val, y_pred)
    macro_f1     = f1_score(y_val, y_pred, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)

    # Class distribution for the paper table
    label_rmap = {0: 'no', 1: 'unsure', 2: 'yes'}
    class_dist = {
        label_rmap[i]: int((df_aid['argmax_label'].map(label_map) == i).sum())
        for i in range(3)
    }

    subdir = Path(output_dir) / f"{aid.lower().replace(' ', '_')}_{model_type}"
    subdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, subdir / "scaler.joblib")
    torch.save(probe.state_dict(), subdir / "probe.pth")

    meta = {
        "aid":             aid,
        "class_names":     CLASS3,
        "class_distribution": class_dist,
        "n_panoramas_total": int(df_aid["ImageID"].nunique()),
        "n_train_panos":   int(len(unique_panos) - n_val),
        "n_val_panos":     int(n_val),
        "train_val_split": f"{int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}",
        "accuracy":        acc,
        "macro_f1":        macro_f1,
        "balanced_accuracy": balanced_acc,
        "model_name":      model_name,
        "backbone":        f"{model_type}::{model_name}",
        "hyperparameters": {
            "lr":          PROBE_LR,
            "weight_decay": PROBE_WD,
            "epochs":      PROBE_EPOCHS,
            "random_state": 42,
            "scaler":      "StandardScaler(with_mean=False)",
            "optimizer":   "Adam",
        },
        "classes_seen": [0, 1, 2],
    }
    with open(subdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return macro_f1

def main():
    parser = argparse.ArgumentParser(description="Train CLIP-based models.")
    parser.add_argument("--tallies_json", type=str, required=True, help="Path to tallies JSON")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for models")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="Model name")
    parser.add_argument("--model_type", type=str, default="clip", choices=["clip", "resnet50", "vit_base_patch16_224", "convnext_base", "dinov2"], help="Model type")
    args = parser.parse_args()

    tallies = pd.read_json(args.tallies_json)
    # Add path column
    tallies['path'] = tallies['ImageID'].apply(lambda x: find_image_path(x, args.images_dir))
    tallies = tallies[tallies['path'].notna()]  # Filter out missing images

    if args.model_type == "clip":
        model, processor, device = load_clip_model(args.model_name)
    else:
        model, processor, device = load_vision_model(args.model_name)
    
    for aid in AIDS:
        df_aid = tallies[tallies['MobilityAid'] == aid]
        if df_aid.empty:
            continue
        acc = train_aid_model(model, processor, device, df_aid, args.output_dir, aid, args.model_name, args.model_type)
        print(f"{aid}: Val Macro F1 {acc:.3f}")

if __name__ == "__main__":
    main()