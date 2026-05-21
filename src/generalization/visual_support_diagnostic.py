#!/usr/bin/env python3
"""Low-level visual support diagnostic for routing extrapolation.

The main model uses DINOv2 features, but this script deliberately avoids torch
so it can run on a login node. It compares audit-survey images against external
GSV images using simple perceptual descriptors (HSV histograms, HOG, edge
density, and intensity moments). The output is a proxy support check, not a
calibration claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def image_descriptor(path: Path, size: int = 128) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(size, size),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
    ).compute(gray).flatten()
    hog = hog / (np.linalg.norm(hog) + 1e-8)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([edges.mean() / 255.0])
    moments = np.array([gray.mean() / 255.0, gray.std() / 255.0])

    return np.concatenate([hist, hog, edge_density, moments]).astype(np.float32)


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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_dir", type=Path, default=Path("data/images/sidewalk-images"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/generalization/test_images.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/generalization/visual_support"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_paths = sorted(args.audit_dir.glob("*.png")) + sorted(args.audit_dir.glob("*.jpg"))
    test_df = pd.read_csv(args.test_csv)
    test_df = test_df[test_df["image_path"].apply(lambda p: Path(p).exists())].copy()

    records = []
    descriptors = []
    for path in audit_paths:
        records.append({"split": "audit", "city": "audit_10city", "image_path": str(path)})
        descriptors.append(image_descriptor(path))
    for _, row in test_df.iterrows():
        records.append({"split": "external", "city": row["city"], "image_path": row["image_path"]})
        descriptors.append(image_descriptor(Path(row["image_path"])))

    X = np.vstack(descriptors)
    X = StandardScaler().fit_transform(X)
    records_df = pd.DataFrame(records)

    audit_mask = records_df["split"].values == "audit"
    X_audit = X[audit_mask]
    X_ext = X[~audit_mask]
    ext_df = records_df[~audit_mask].reset_index(drop=True)

    # Leave-one-out nearest-neighbor distances within the audit set establish a
    # rough in-support reference scale.
    audit_d = pairwise_distances(X_audit, X_audit, metric="cosine")
    np.fill_diagonal(audit_d, np.inf)
    audit_nn = audit_d.min(axis=1)

    ext_d = pairwise_distances(X_ext, X_audit, metric="cosine")
    ext_nn = ext_d.min(axis=1)

    threshold = float(np.percentile(audit_nn, 90))
    rows = []
    rows.append({"group": "audit_leave_one_out", **summarize(audit_nn), "pct_above_audit_p90": 10.0})
    for city, idx in ext_df.groupby("city").groups.items():
        vals = ext_nn[list(idx)]
        row = {"group": city, **summarize(vals)}
        row["pct_above_audit_p90"] = round(100.0 * float(np.mean(vals > threshold)), 1)
        rows.append(row)

    per_image = ext_df.copy()
    per_image["nearest_audit_cosine"] = ext_nn
    per_image["above_audit_p90"] = ext_nn > threshold

    write_csv(args.output_dir / "visual_support_summary.csv", rows)
    per_image.to_csv(args.output_dir / "visual_support_per_image.csv", index=False)

    summary = {
        "descriptor": "HSV histogram + HOG + Canny edge density + intensity moments",
        "distance": "cosine after StandardScaler",
        "audit_images": int(audit_mask.sum()),
        "external_images": int((~audit_mask).sum()),
        "audit_p90_threshold": round(threshold, 4),
        "rows": rows,
    }
    (args.output_dir / "visual_support_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
