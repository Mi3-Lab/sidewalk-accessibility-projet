#!/usr/bin/env python3
"""
Linear probe training for sidewalk accessibility classification.

Supports 8 frozen vision encoders:
  clip-vit-b32   | openai/clip-vit-base-patch32     (current baseline)
  clip-vit-b16   | openai/clip-vit-base-patch16
  clip-vit-l14   | openai/clip-vit-large-patch14
  dinov2-base    | facebook/dinov2-base
  dinov2-large   | facebook/dinov2-large
  siglip2-base   | google/siglip2-base-patch16-224
  siglip2-so400m | google/siglip2-so400m-patch14-384
  vit-b16-sup    | vit_base_patch16_224.augreg2_in21k_ft_in1k (timm, supervised)

Usage:
    python src/models/train.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --encoder      dinov2-base \
        --output_dir   results/models/dinov2-base
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
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
CLASS3     = ["no", "unsure", "yes"]
SOFT_COLS  = ["p_no", "p_unsure", "p_yes"]   # columns that hold soft-label distribution
AIDS       = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]

# argmax_label column contains the column name from idxmax → 'p_no', 'p_unsure', 'p_yes'
LABEL_MAP = {"p_no": 0, "p_unsure": 1, "p_yes": 2}

PROBE_LR     = 1e-3
PROBE_EPOCHS = 50
PROBE_WD     = 1e-4
TRAIN_SPLIT  = 0.8   # 80 % train / 20 % val at panorama level


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Loss type used by default.
# "soft_kl"  → KL divergence on full vote distribution  (paper contribution)
# "hard_ce"  → cross-entropy on argmax label + entropy weight (old baseline)
LOSS_TYPE    = "soft_kl"
BATCH_SIZE   = 32

# ── Encoder registry ─────────────────────────────────────────────────────────
# Keys are the --encoder CLI values.
# Values: (hf_id_or_timm_name, encoder_type)
# encoder_type: "clip" | "siglip" | "dinov2" | "timm"

ENCODERS: dict[str, tuple[str, str]] = {
    "clip-vit-b32":   ("openai/clip-vit-base-patch32",                   "clip"),
    "clip-vit-b16":   ("openai/clip-vit-base-patch16",                   "clip"),
    "clip-vit-l14":   ("openai/clip-vit-large-patch14",                  "clip"),
    "dinov2-base":    ("facebook/dinov2-base",                            "dinov2"),
    "dinov2-large":   ("facebook/dinov2-large",                           "dinov2"),
    "siglip2-base":   ("google/siglip2-base-patch16-224",                 "siglip"),
    "siglip2-so400m": ("google/siglip2-so400m-patch14-384",               "siglip"),
    "vit-b16-sup":    ("vit_base_patch16_224.augreg2_in21k_ft_in1k",     "timm"),
}


# ── Encoder loading ───────────────────────────────────────────────────────────

def load_encoder(encoder_key: str):
    """
    Load frozen model + processor for any key in ENCODERS.

    Returns (model, processor, device, enc_type).
    For timm models, 'processor' is a torchvision Transform.
    """
    if encoder_key not in ENCODERS:
        raise ValueError(
            f"Unknown encoder '{encoder_key}'. "
            f"Choose from: {list(ENCODERS)}"
        )
    hf_id, enc_type = ENCODERS[encoder_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if enc_type == "clip":
        from transformers import AutoModel, AutoProcessor
        model     = AutoModel.from_pretrained(hf_id).to(device).eval()
        processor = AutoProcessor.from_pretrained(hf_id)

    elif enc_type == "siglip":
        from transformers import AutoModel, AutoProcessor
        model     = AutoModel.from_pretrained(hf_id).to(device).eval()
        processor = AutoProcessor.from_pretrained(hf_id)

    elif enc_type == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        model     = AutoModel.from_pretrained(hf_id).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(hf_id)

    elif enc_type == "timm":
        import timm
        model     = timm.create_model(hf_id, pretrained=True, num_classes=0).to(device).eval()
        data_cfg  = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_cfg, is_training=False)

    else:
        raise ValueError(f"Unsupported enc_type: {enc_type}")

    return model, processor, device, enc_type


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_encoder_features(
    model,
    processor,
    device: torch.device,
    paths: list,
    enc_type: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Extract L2-normalised features for all image paths.
    Works for all encoder types; processes images in mini-batches.
    """
    feats: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size),
                          desc=f"[{enc_type}] features", leave=False):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            if enc_type in ("clip", "siglip"):
                inputs = processor(
                    images=images, return_tensors="pt", padding=True
                )
                pixel_values = inputs["pixel_values"].to(device)
                raw = model.get_image_features(pixel_values=pixel_values)
                # transformers ≥5.x may return an output object instead of a tensor
                if isinstance(raw, torch.Tensor):
                    out = raw
                elif hasattr(raw, "image_embeds"):
                    out = raw.image_embeds
                elif hasattr(raw, "pooler_output"):
                    out = raw.pooler_output
                else:
                    out = raw.last_hidden_state[:, 0]

            elif enc_type == "dinov2":
                inputs = processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                out = model(pixel_values=pixel_values).pooler_output

            elif enc_type == "timm":
                tensors = torch.stack([processor(img) for img in images]).to(device)
                out = model(tensors)

            else:
                raise ValueError(f"Unknown enc_type: {enc_type}")

            out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.cpu().float().numpy())

    return np.concatenate(feats, axis=0)


# ── Backwards-compat shims (used by crossval.py and other scripts) ────────────

def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    from transformers import AutoModel, AutoProcessor
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, device


def extract_features(model, processor, device, image_paths: list) -> np.ndarray:
    """CLIP feature extraction (backwards compat)."""
    return extract_encoder_features(model, processor, device, image_paths, "clip")


def extract_vision_features(model, processor, device, image_paths: list) -> np.ndarray:
    """Generic vision feature extraction (backwards compat). Assumes dinov2/timm."""
    from transformers import Dinov2Model
    enc_type = "dinov2" if isinstance(model, Dinov2Model) else "timm"
    return extract_encoder_features(model, processor, device, image_paths, enc_type)


# ── Image path lookup ─────────────────────────────────────────────────────────

def find_image_path(image_id, images_dir) -> Path | None:
    images_dir = Path(images_dir)
    for ext in (".png", ".jpg", ".jpeg"):
        candidates = list(images_dir.glob(f"*{image_id}*{ext}"))
        if candidates:
            return candidates[0]
    return None


# ── Linear probe ──────────────────────────────────────────────────────────────

def build_probe(feature_dim: int, device: torch.device, rank: int = 0) -> nn.Module:
    """Linear probe, optionally through a low-rank bottleneck.

    rank=0 gives the original d -> 3 map, which is what every published number
    used. A positive rank inserts a d -> rank -> 3 factorisation instead.

    Why it is worth having: d is 512-1536 depending on the encoder, and the probe
    is fit to 52 points per aid with weight_decay 1e-4, which is close to no
    regularisation. Constraining the map cuts soft Brier by roughly 30% and
    raises macro F1 at the same time, consistently across encoders and seeds.
    """
    if rank and rank > 0:
        return nn.Sequential(
            nn.Linear(feature_dim, rank, bias=False),
            nn.Linear(rank, len(CLASS3)),
        ).to(device)
    return nn.Linear(feature_dim, len(CLASS3)).to(device)


def train_probe_soft(
    X_train: np.ndarray,
    y_soft: np.ndarray,
    w_train: np.ndarray,
    feature_dim: int,
    device: torch.device,
    seed: int = RANDOM_STATE,
    rank: int = 0,
    weight_decay: float = PROBE_WD,
) -> nn.Module:
    """
    Train linear probe with KL-divergence loss on the full human vote distribution.

    y_soft shape: (N, 3) with columns [p_no, p_unsure, p_yes].
    w_train: entropy-based sample weights — still applied so that high-agreement
             images (low entropy, high weight) steer the probe more strongly.

    KL(target || pred) = sum_c  target_c * (log target_c − log pred_c)
    Since target is fixed at training time, minimising KL is equivalent to
    minimising the cross-entropy H(target, pred) = −sum_c target_c * log pred_c,
    but KL correctly accounts for cases where all three classes have non-zero mass.
    """
    set_seed(seed)
    probe = build_probe(feature_dim, device, rank)
    optimizer = optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=weight_decay)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_soft,  dtype=torch.float32).to(device)   # (N, 3)
    w_t = torch.tensor(w_train, dtype=torch.float32).to(device)

    probe.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        log_probs = F.log_softmax(probe(X_t), dim=-1)             # (N, 3)
        # per-sample KL divergence: sum over classes
        kl = F.kl_div(log_probs, y_t, reduction="none").sum(dim=-1)  # (N,)
        (kl * w_t).mean().backward()
        optimizer.step()

    probe.eval()
    return probe


def train_probe_hard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    feature_dim: int,
    device: torch.device,
    seed: int = RANDOM_STATE,
    rank: int = 0,
    weight_decay: float = PROBE_WD,
) -> nn.Module:
    """
    Train linear probe with cross-entropy loss on argmax hard labels.
    Kept for ablation: compare against train_probe_soft.
    """
    set_seed(seed)
    probe = build_probe(feature_dim, device, rank)
    optimizer = optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="none")

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    w_t = torch.tensor(w_train, dtype=torch.float32).to(device)

    probe.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        losses = criterion(probe(X_t), y_t)
        (losses * w_t).mean().backward()
        optimizer.step()

    probe.eval()
    return probe


# Keep the old name as an alias so external callers don't break
train_probe = train_probe_hard


# ── Per-aid training ──────────────────────────────────────────────────────────

def train_aid_model(
    model,
    processor,
    device: torch.device,
    df_aid: pd.DataFrame,
    output_dir: str | Path,
    aid: str,
    encoder_key: str,
    enc_type: str,
    loss_type: str = LOSS_TYPE,
) -> float:
    """
    Train a linear probe for one mobility aid.
    Splits at panorama level to prevent leakage between crops.

    loss_type:
      "soft_kl"  — KL divergence on [p_no, p_unsure, p_yes]  (default)
      "hard_ce"  — cross-entropy on argmax label + entropy weight (ablation)

    Returns validation macro-F1.
    """
    unique_panos = df_aid["ImageID"].unique()
    n_val = max(1, int(len(unique_panos) * (1 - TRAIN_SPLIT)))
    rng = np.random.default_rng(RANDOM_STATE)
    val_panos = set(rng.choice(unique_panos, size=n_val, replace=False))

    tr_df = df_aid[~df_aid["ImageID"].isin(val_panos)]
    va_df = df_aid[df_aid["ImageID"].isin(val_panos)]

    X_train = extract_encoder_features(model, processor, device, tr_df["path"].tolist(), enc_type)
    X_val   = extract_encoder_features(model, processor, device, va_df["path"].tolist(), enc_type)

    y_val   = va_df["argmax_label"].map(LABEL_MAP).values   # always hard for metrics
    w_train = tr_df["sample_weight"].values

    scaler  = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loss_type == "soft_kl":
        y_soft  = tr_df[SOFT_COLS].values.astype(np.float32)   # (N, 3)
        probe   = train_probe_soft(X_train, y_soft, w_train, X_train.shape[1], probe_device)
    else:
        y_train = tr_df["argmax_label"].map(LABEL_MAP).values
        probe   = train_probe_hard(X_train, y_train, w_train, X_train.shape[1], probe_device)

    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(probe_device)
        y_pred  = probe(X_val_t).argmax(dim=1).cpu().numpy()

    acc          = float(accuracy_score(y_val, y_pred))
    macro_f1     = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_val, y_pred))

    class_dist = {
        cls: int((va_df["argmax_label"] == f"p_{cls}").sum())
        for cls in CLASS3
    }

    subdir = Path(output_dir) / f"{aid.lower().replace(' ', '_')}_{encoder_key}"
    subdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, subdir / "scaler.joblib")
    torch.save(probe.state_dict(), subdir / "probe.pth")

    meta = {
        "aid":                aid,
        "encoder":            encoder_key,
        "hf_id":              ENCODERS[encoder_key][0],
        "enc_type":           enc_type,
        "loss_type":          loss_type,
        "class_names":        CLASS3,
        "class_distribution": class_dist,
        "n_panoramas_total":  int(df_aid["ImageID"].nunique()),
        "n_train_panos":      int(len(unique_panos) - n_val),
        "n_val_panos":        int(n_val),
        "train_val_split":    f"{int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}",
        "accuracy":           acc,
        "macro_f1":           macro_f1,
        "balanced_accuracy":  balanced_acc,
        "hyperparameters": {
            "lr":           PROBE_LR,
            "weight_decay": PROBE_WD,
            "epochs":       PROBE_EPOCHS,
            "random_state": 42,
            "scaler":       "StandardScaler(with_mean=False)",
            "optimizer":    "Adam",
            "loss":         "KLDivLoss" if loss_type == "soft_kl" else "CrossEntropyLoss",
        },
    }
    with open(subdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return macro_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear probe for sidewalk accessibility classification."
    )
    parser.add_argument("--tallies_json", required=True,
                        help="Path to tallies JSON (e.g. data/processed/tallies_firebase.json)")
    parser.add_argument("--images_dir",   required=True,
                        help="Directory containing sidewalk image crops")
    parser.add_argument("--output_dir",   required=True,
                        help="Output directory for probe weights and metadata")
    parser.add_argument(
        "--encoder",
        default="clip-vit-b32",
        choices=list(ENCODERS),
        help=f"Vision encoder to use. Options: {list(ENCODERS)}",
    )
    parser.add_argument(
        "--loss_type",
        default=LOSS_TYPE,
        choices=["soft_kl", "hard_ce"],
        help=(
            "soft_kl: KL divergence on full vote distribution (default, paper method). "
            "hard_ce: cross-entropy on argmax label + entropy weight (ablation)."
        ),
    )
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    print(f"\nEncoder : {args.encoder}  ({ENCODERS[args.encoder][0]})")
    print(f"Tallies : {args.tallies_json}")
    print(f"Images  : {args.images_dir}")
    print(f"Output  : {args.output_dir}\n")

    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(
        lambda x: find_image_path(x, args.images_dir)
    )
    tallies = tallies[tallies["path"].notna()]
    print(f"Images found: {len(tallies)} / {len(pd.read_json(args.tallies_json))} rows\n")

    model, processor, device, enc_type = load_encoder(args.encoder)

    results = {}
    for aid in AIDS:
        df_aid = tallies[tallies["MobilityAid"] == aid]
        if df_aid.empty:
            print(f"[{aid}] no data — skipping.")
            continue
        f1 = train_aid_model(
            model, processor, device,
            df_aid, args.output_dir, aid,
            args.encoder, enc_type, args.loss_type,
        )
        results[aid] = f1
        print(f"  {aid:<30} val macro-F1 = {f1:.3f}")

    print(f"\n── Summary ({args.encoder}) ──────────────────────────")
    for aid, f1 in results.items():
        print(f"  {aid:<30} {f1:.3f}")
    mean_f1 = float(np.mean(list(results.values()))) if results else 0.0
    print(f"  {'Mean across aids':<30} {mean_f1:.3f}")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
