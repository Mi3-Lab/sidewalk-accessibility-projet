#!/usr/bin/env python3
"""
Train final model checkpoints (all data, no held-out set) for deployment/generalization.

Uses the exact same pipeline as crossval.py but trains on the full dataset so
the probe has seen every available example. These checkpoints are used by:
  - src/generalization/evaluate_generalization.py
  - src/routing/demo.py

Usage:
    python src/models/train_final_model.py \
        --encoder    dinov2-large \
        --output_dir results/models/dinov2-large \
        --loss_type  soft_kl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import torch

from train import (
    AIDS,
    ENCODERS,
    LOSS_TYPE,
    PROBE_EPOCHS,
    PROBE_LR,
    PROBE_WD,
    RANDOM_STATE,
    SOFT_COLS,
    LABEL_MAP,
    extract_encoder_features,
    find_image_path,
    load_encoder,
    set_seed,
    train_probe_soft,
    train_probe_hard,
)

import pandas as pd
from sklearn.preprocessing import StandardScaler


def train_final_probe(
    model,
    processor,
    device: torch.device,
    df_aid: pd.DataFrame,
    aid: str,
    enc_type: str,
    output_dir: Path,
    loss_type: str = LOSS_TYPE,
) -> None:
    """Train on full dataset (no val split) and save probe + scaler."""
    df_aid = df_aid.copy()
    df_aid["label_int"] = df_aid["argmax_label"].map(LABEL_MAP)

    X = extract_encoder_features(model, processor, device, df_aid["path"].tolist(), enc_type)
    w = df_aid["sample_weight"].values

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loss_type == "soft_kl":
        y_soft = df_aid[SOFT_COLS].values.astype(np.float32)
        probe  = train_probe_soft(X_scaled, y_soft, w, X_scaled.shape[1], probe_device)
    else:
        y_hard = df_aid["label_int"].values
        probe  = train_probe_hard(X_scaled, y_hard, w, X_scaled.shape[1], probe_device)

    aid_key = aid.lower().replace(" ", "_")
    aid_dir = output_dir / aid_key
    aid_dir.mkdir(parents=True, exist_ok=True)

    torch.save(probe.state_dict(), aid_dir / "probe.pth")
    joblib.dump(scaler, aid_dir / "scaler.joblib")

    print(f"  [{aid}] {len(df_aid)} samples → {aid_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train final per-aid probes on full dataset."
    )
    parser.add_argument("--tallies_json", default="data/processed/tallies_firebase.json")
    parser.add_argument("--images_dir",   default="data/images/sidewalk-images")
    parser.add_argument("--encoder",      default="dinov2-large", choices=list(ENCODERS))
    parser.add_argument("--output_dir",   default="results/models/dinov2-large")
    parser.add_argument(
        "--loss_type", default=LOSS_TYPE, choices=["soft_kl", "hard_ce"],
        help="soft_kl: paper method (outputs p_yes for routing). hard_ce: argmax F1 baseline.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEncoder  : {args.encoder}  ({ENCODERS[args.encoder][0]})")
    print(f"Loss     : {args.loss_type}")
    print(f"Output   : {output_dir}")
    print(f"Seed     : {args.seed}\n")

    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(
        lambda x: find_image_path(x, args.images_dir)
    )
    tallies = tallies[tallies["path"].notna()]
    print(f"Images found: {len(tallies)} rows\n")

    model, processor, device, enc_type = load_encoder(args.encoder)

    for aid in AIDS:
        df_aid = tallies[tallies["MobilityAid"] == aid]
        if df_aid.empty:
            print(f"  [{aid}] no data — skipping.")
            continue
        train_final_probe(model, processor, device, df_aid, aid, enc_type, output_dir, args.loss_type)

    print(f"\nAll checkpoints saved → {output_dir}/")


if __name__ == "__main__":
    main()
