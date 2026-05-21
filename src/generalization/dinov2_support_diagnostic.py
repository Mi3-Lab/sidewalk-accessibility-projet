#!/usr/bin/env python3
"""DINOv2 feature-support diagnostic for routing/generalization images.

This compares the 52-view audit support to external GSV images in the same
feature space used by the DINOv2-large probe. It is a support/transfer sanity
check, not a calibration claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

sys.path.insert(0, str(Path(__file__).parents[1] / "models"))
from train import extract_encoder_features, load_encoder, set_seed


def summarize(vals: np.ndarray) -> dict:
    return {
        "n": int(len(vals)),
        "mean": round(float(np.mean(vals)), 4),
        "p25": round(float(np.percentile(vals, 25)), 4),
        "p50": round(float(np.percentile(vals, 50)), 4),
        "p75": round(float(np.percentile(vals, 75)), 4),
        "p90": round(float(np.percentile(vals, 90)), 4),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="dinov2-large")
    parser.add_argument("--audit_dir", type=Path, default=Path("data/images/sidewalk-images"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/generalization/test_images.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/generalization/dinov2_support"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_paths = sorted(args.audit_dir.glob("*.png")) + sorted(args.audit_dir.glob("*.jpg"))
    test_df = pd.read_csv(args.test_csv)
    test_df = test_df[test_df["image_path"].apply(lambda p: Path(p).exists())].copy()

    paths = [str(p) for p in audit_paths] + test_df["image_path"].tolist()
    records = (
        [{"split": "audit", "city": "audit_10city", "image_path": str(p)} for p in audit_paths]
        + [
            {"split": "external", "city": row["city"], "image_path": row["image_path"]}
            for _, row in test_df.iterrows()
        ]
    )

    print(f"Audit images: {len(audit_paths)}")
    print(f"External images: {len(test_df)}")
    print(f"Loading encoder: {args.encoder}")
    model, processor, device, enc_type = load_encoder(args.encoder)
    print(f"Device: {device}")

    feats = extract_encoder_features(
        model, processor, device, paths, enc_type, batch_size=args.batch_size
    )
    records_df = pd.DataFrame(records)
    audit_mask = records_df["split"].values == "audit"
    X_audit = feats[audit_mask]
    X_ext = feats[~audit_mask]
    ext_df = records_df[~audit_mask].reset_index(drop=True)

    # Features are already L2-normalised by extract_encoder_features.
    audit_d = pairwise_distances(X_audit, X_audit, metric="cosine")
    np.fill_diagonal(audit_d, np.inf)
    audit_nn = audit_d.min(axis=1)

    ext_d = pairwise_distances(X_ext, X_audit, metric="cosine")
    ext_nn = ext_d.min(axis=1)

    threshold = float(np.percentile(audit_nn, 90))
    rows = [
        {
            "group": "audit_leave_one_out",
            **summarize(audit_nn),
            "pct_above_audit_p90": 10.0,
        }
    ]
    for city, idx in ext_df.groupby("city").groups.items():
        vals = ext_nn[list(idx)]
        row = {"group": city, **summarize(vals)}
        row["pct_above_audit_p90"] = round(100.0 * float(np.mean(vals > threshold)), 1)
        rows.append(row)

    per_image = ext_df.copy()
    per_image["nearest_audit_cosine"] = ext_nn
    per_image["above_audit_p90"] = ext_nn > threshold

    write_csv(args.output_dir / "dinov2_support_summary.csv", rows)
    per_image.to_csv(args.output_dir / "dinov2_support_per_image.csv", index=False)
    np.savez_compressed(args.output_dir / "dinov2_support_features.npz", features=feats)

    summary = {
        "encoder": args.encoder,
        "distance": "cosine on L2-normalised DINOv2 features",
        "audit_images": int(audit_mask.sum()),
        "external_images": int((~audit_mask).sum()),
        "audit_p90_threshold": round(threshold, 4),
        "rows": rows,
    }
    (args.output_dir / "dinov2_support_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
