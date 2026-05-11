#!/usr/bin/env python3
"""
Aggregate all CV results from results/cv/<encoder>/cv_results.json
and produce a summary table + CSV.

Usage:
    python src/models/summarize_cv.py --results_dir results/cv
    python src/models/summarize_cv.py --results_dir results/cv --output results/cv_summary.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AIDS = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]

ENCODER_ORDER = [
    "clip-vit-b32",
    "clip-vit-b16",
    "clip-vit-l14",
    "vit-b16-sup",
    "dinov2-base",
    "dinov2-large",
    "siglip2-base",
    "siglip2-so400m",
]


def load_results(results_dir: Path) -> list[dict]:
    records = []
    for json_path in sorted(results_dir.glob("*/cv_results.json")):
        encoder = json_path.parent.name
        with open(json_path) as f:
            data = json.load(f)
        for aid, v in data.get("aids", {}).items():
            records.append({
                "encoder":          encoder,
                "aid":              aid,
                "macro_f1_mean":    v["macro_f1_mean"],
                "macro_f1_std":     v["macro_f1_std"],
                "bal_acc_mean":     v["balanced_acc_mean"],
                "bal_acc_std":      v["balanced_acc_std"],
                "brier_soft_mean":  v.get("brier_soft_mean", float("nan")),
                "brier_hard_mean":  v.get("brier_hard_mean", float("nan")),
                "ece_mean":         v.get("ece_mean",         float("nan")),
                "n_folds":          v["n_folds"],
            })
    return records


def print_table(df: pd.DataFrame, metric: str = "macro_f1_mean") -> None:
    std_col = metric.replace("mean", "std")
    pivot = df.pivot(index="encoder", columns="aid", values=metric)

    # Keep only aids present in data, in preferred order
    aids_present = [a for a in AIDS if a in pivot.columns]
    pivot = pivot[aids_present]

    # Reorder encoders
    enc_order = [e for e in ENCODER_ORDER if e in pivot.index]
    remaining = [e for e in pivot.index if e not in enc_order]
    pivot = pivot.loc[enc_order + remaining]

    # Add mean across aids
    pivot["MEAN"] = pivot.mean(axis=1)

    col_w = 14
    header = f"{'Encoder':<22}" + "".join(f"{a[:col_w]:>{col_w}}" for a in aids_present) + f"{'MEAN':>{col_w}}"
    print(header)
    print("-" * len(header))

    for enc in pivot.index:
        row_vals = []
        for aid in aids_present:
            mean_val = pivot.loc[enc, aid]
            std_row  = df[(df["encoder"] == enc) & (df["aid"] == aid)][std_col]
            std_val  = std_row.values[0] if len(std_row) else float("nan")
            row_vals.append(f"{mean_val:.3f}±{std_val:.3f}")
        mean_all = pivot.loc[enc, "MEAN"]
        row = f"{enc:<22}" + "".join(f"{v:>{col_w}}" for v in row_vals) + f"{mean_all:>{col_w}.3f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/cv",
                        help="Directory containing <encoder>/cv_results.json files")
    parser.add_argument("--output", default=None,
                        help="Path to save summary CSV (optional)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_results(results_dir)

    if not records:
        print(f"No cv_results.json found under {results_dir}/")
        print("Run ./run_cv.sh first.")
        return

    df = pd.DataFrame(records)
    encoders_found = sorted(df["encoder"].unique())
    aids_found = sorted(df["aid"].unique())

    print(f"\nResults found: {len(encoders_found)} encoders, {len(aids_found)} aids")
    print(f"Encoders: {encoders_found}\n")

    print("── Macro-F1 (mean ± std across 5 folds) ─────────────────────────────────")
    print_table(df, "macro_f1_mean")

    print("\n── Balanced Accuracy (mean ± std across 5 folds) ────────────────────────")
    print_table(df, "bal_acc_mean")

    # Per-encoder overall mean + calibration
    print("\n── Overall summary (mean across all aids) ───────────────────────────────")
    overall = (
        df.groupby("encoder")[["macro_f1_mean", "brier_soft_mean", "brier_hard_mean", "ece_mean"]]
        .mean()
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index()
    )
    print(f"  {'encoder':<22} {'macro_f1':>9} {'brier_soft':>11} {'brier_hard':>11} {'ece':>7}")
    print("  " + "-" * 62)
    for _, row in overall.iterrows():
        print(
            f"  {row['encoder']:<22} {row['macro_f1_mean']:>9.3f}"
            f" {row['brier_soft_mean']:>11.4f}"
            f" {row['brier_hard_mean']:>11.4f}"
            f" {row['ece_mean']:>7.4f}"
        )

    # Save CSV
    out_path = Path(args.output) if args.output else results_dir / "cv_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSummary CSV saved → {out_path}")


if __name__ == "__main__":
    main()
