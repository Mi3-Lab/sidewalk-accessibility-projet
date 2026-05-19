#!/usr/bin/env python3
"""
Generalization Test: Evaluate trained model on new cities (out-of-distribution).

Loads trained per-aid probes (from train_final_model.py), extracts features
for all test images in a single batch pass (encoder loaded ONCE), then
compares predictions to Project Sidewalk ground-truth labels if available.

Supports two modes:
  - Labelled mode:   test CSV has ps_label (CurbRamp/NoCurbRamp) → reports accuracy
  - Inference mode:  test CSV has no valid labels → saves predictions only

Usage:
    python src/generalization/evaluate_generalization.py \
        --encoder    dinov2-large \
        --checkpoint results/models/dinov2-large \
        --test_images data/generalization/test_images.csv \
        --output_dir  results/generalization/dinov2-large \
        --wandb_project sidewalk-generalization

Expected CSV columns (ps_label optional):
    image_id, image_path, city, ps_label
    ps_label values: CurbRamp (accessible), NoCurbRamp (inaccessible), or blank/unknown
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "models"))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

from train import (
    AIDS,
    CLASS3,
    ENCODERS,
    LABEL_MAP,
    RANDOM_STATE,
    extract_encoder_features,
    load_encoder,
    set_seed,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

PS_LABEL_MAP = {"CurbRamp": "yes", "NoCurbRamp": "no"}


def load_probe(checkpoint_dir: Path, aid: str, feature_dim: int, device: torch.device) -> tuple:
    """Load scaler + probe for one aid. Returns (probe, scaler) or (None, None)."""
    aid_key = aid.lower().replace(" ", "_")
    scaler_path = checkpoint_dir / aid_key / "scaler.joblib"
    probe_path  = checkpoint_dir / aid_key / "probe.pth"

    if not scaler_path.exists() or not probe_path.exists():
        return None, None

    scaler = joblib.load(scaler_path)
    probe  = nn.Linear(feature_dim, len(CLASS3))
    probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
    probe  = probe.to(device).eval()
    return probe, scaler


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained probes on out-of-distribution cities."
    )
    parser.add_argument("--encoder",       default="dinov2-large", choices=list(ENCODERS))
    parser.add_argument("--checkpoint",    required=True,
                        help="Directory with per-aid probe checkpoints (train_final_model.py output).")
    parser.add_argument("--test_images",   required=True,
                        help="CSV with columns: image_id, image_path, city[, ps_label]")
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--seed",          type=int, default=RANDOM_STATE)
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name. Omit to disable tracking.")
    args = parser.parse_args()

    set_seed(args.seed)
    checkpoint_dir = Path(args.checkpoint)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test CSV ─────────────────────────────────────────────────────────
    test_df = pd.read_csv(args.test_images)
    print(f"Test images: {len(test_df)} rows")

    # Validate required columns
    for col in ("image_id", "image_path"):
        if col not in test_df.columns:
            print(f"ERROR: CSV missing required column '{col}'.")
            sys.exit(1)

    # Keep only existing images
    test_df = test_df[test_df["image_path"].apply(lambda p: Path(p).exists())]
    if test_df.empty:
        print("ERROR: No valid image files found. Check image_path column.")
        sys.exit(1)
    print(f"Found: {len(test_df)} images on disk")

    # ── Load encoder once ─────────────────────────────────────────────────────
    print(f"\nLoading encoder: {args.encoder}")
    model, processor, device, enc_type = load_encoder(args.encoder)

    # ── Extract features (single batch pass) ─────────────────────────────────
    print("Extracting features …")
    paths = test_df["image_path"].tolist()
    X = extract_encoder_features(model, processor, device, paths, enc_type)
    feature_dim = X.shape[1]
    print(f"Features: {X.shape}")

    # ── Per-aid predictions ───────────────────────────────────────────────────
    results_by_aid: dict[str, np.ndarray] = {}
    probs_by_aid:   dict[str, np.ndarray] = {}

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for aid in AIDS:
        probe, scaler = load_probe(checkpoint_dir, aid, feature_dim, probe_device)
        if probe is None:
            print(f"  [{aid}] checkpoint missing — skipping.")
            continue

        X_scaled = scaler.transform(X)
        X_t      = torch.tensor(X_scaled, dtype=torch.float32).to(probe_device)
        with torch.no_grad():
            logits = probe(X_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        results_by_aid[aid] = probs.argmax(axis=1)
        probs_by_aid[aid]   = probs
        print(f"  [{aid}] done")

    if not results_by_aid:
        print("ERROR: No checkpoints found. Run train_final_model.py first.")
        sys.exit(1)

    # ── Build predictions DataFrame ───────────────────────────────────────────
    pred_df = test_df[["image_id", "image_path"]].copy()
    if "city"     in test_df.columns: pred_df["city"]     = test_df["city"].values
    if "ps_label" in test_df.columns: pred_df["ps_label"] = test_df["ps_label"].values

    # Average p_yes across aids → binary prediction
    p_yes_all = np.stack([probs_by_aid[a][:, 2] for a in results_by_aid], axis=1)
    avg_p_yes = p_yes_all.mean(axis=1)
    pred_df["avg_p_yes"]    = avg_p_yes
    pred_df["pred_binary"]  = np.where(avg_p_yes >= 0.5, "yes", "no")

    # Per-aid columns
    for aid, preds in results_by_aid.items():
        aid_key = aid.lower().replace(" ", "_")
        inv_map = {v: k for k, v in LABEL_MAP.items()}
        pred_df[f"{aid_key}_pred"]   = [inv_map[p] for p in preds]
        pred_df[f"{aid_key}_p_yes"]  = probs_by_aid[aid][:, 2]

    # ── Agreement metrics (only if labelled) ─────────────────────────────────
    has_labels = bool(
        "ps_label" in pred_df.columns
        and pred_df["ps_label"].isin(PS_LABEL_MAP).any()
    )

    summary: dict = {
        "encoder":        args.encoder,
        "n_test_samples": len(pred_df),
        "n_valid_labels": 0,
        "labelled_mode":  has_labels,
    }

    if has_labels:
        pred_df["ps_class"] = pred_df["ps_label"].map(PS_LABEL_MAP)
        valid = pred_df[pred_df["ps_class"].notna()].copy()
        summary["n_valid_labels"] = len(valid)

        acc      = float(accuracy_score(valid["ps_class"], valid["pred_binary"]))
        bal_acc  = float(balanced_accuracy_score(valid["ps_class"], valid["pred_binary"]))
        cm       = confusion_matrix(valid["ps_class"], valid["pred_binary"], labels=["no", "yes"])

        summary.update({
            "overall_accuracy":          acc,
            "overall_balanced_accuracy": bal_acc,
            "confusion_matrix":          {"labels": ["no", "yes"], "matrix": cm.tolist()},
        })

        per_city = {}
        if "city" in valid.columns:
            for city, grp in valid.groupby("city"):
                per_city[city] = {
                    "n": len(grp),
                    "accuracy":         float(accuracy_score(grp["ps_class"], grp["pred_binary"])),
                    "balanced_accuracy": float(balanced_accuracy_score(grp["ps_class"], grp["pred_binary"])),
                }
        summary["per_city"] = per_city

        print(f"\nOverall accuracy:          {acc:.3f}")
        print(f"Overall balanced accuracy: {bal_acc:.3f}")
        print(f"Confusion matrix:\n{cm}")
    else:
        print("\nNo valid PS labels found → inference-only mode (no accuracy metrics).")
        print(f"avg_p_yes statistics: mean={avg_p_yes.mean():.3f}, std={avg_p_yes.std():.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    pred_csv = output_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nPredictions → {pred_csv}")

    summary_json = output_dir / "agreement_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary     → {summary_json}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    if args.wandb_project and WANDB_AVAILABLE:
        run = wandb.init(
            project=args.wandb_project,
            name=f"gen_{args.encoder}",
            tags=["generalization", args.encoder],
            config={"encoder": args.encoder, "seed": args.seed, "n_images": len(pred_df)},
        )
        metrics = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        run.log(metrics)
        run.summary.update(metrics)
        # Table of predictions
        cols = ["image_id", "city", "ps_label", "pred_binary", "avg_p_yes"] if has_labels \
               else ["image_id", "city", "pred_binary", "avg_p_yes"]
        cols = [c for c in cols if c in pred_df.columns]
        tbl  = wandb.Table(dataframe=pred_df[cols])
        run.log({"predictions": tbl})
        run.finish()
        print(f"W&B → {run.url}")


if __name__ == "__main__":
    main()
