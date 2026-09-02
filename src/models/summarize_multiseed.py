#!/usr/bin/env python3
"""
Aggregate repeated-seed CV results into a stability table.

Reads results/cv_multiseed/<loss>/<encoder>/cv_results_multiseed.json and
reports, per encoder, the mean±std of each metric *across independent fold
partitions*. This is the number the CoRL reviewer asked for: a single seed=42
split over 52 panoramas cannot show whether a result is stable, so we repeat
the whole CV over N seeds and show the spread.

Usage:
    python src/models/summarize_multiseed.py --results_dir results/cv_multiseed
    python src/models/summarize_multiseed.py --results_dir results/cv_multiseed --latex
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENCODER_ORDER = [
    "clip-vit-b32",
    "clip-vit-b16",
    "clip-vit-l14",
    "dinov2-base",
    "dinov2-large",
    "siglip2-base",
    "siglip2-so400m",
    "vit-b16-sup",
]

ENCODER_LABEL = {
    "clip-vit-b32":   "CLIP ViT-B/32",
    "clip-vit-b16":   "CLIP ViT-B/16",
    "clip-vit-l14":   "CLIP ViT-L/14",
    "dinov2-base":    "DINOv2-base",
    "dinov2-large":   "DINOv2-large",
    "siglip2-base":   "SigLIP2-base",
    "siglip2-so400m": "SigLIP2-SO400M",
    "vit-b16-sup":    "ViT-B/16-sup",
}

METRICS = ["macro_f1", "brier_soft", "ece", "entropy_corr"]


def load(results_dir: Path) -> pd.DataFrame:
    rows = []
    for loss_dir in sorted(results_dir.glob("*")):
        if not loss_dir.is_dir():
            continue
        for path in sorted(loss_dir.glob("*/cv_results_multiseed.json")):
            with open(path) as f:
                data = json.load(f)
            ov = data["overall"]
            row = {
                "loss":    loss_dir.name,
                "encoder": path.parent.name,
                "n_seeds": data["n_seeds"],
                "seeds":   f"{data['seeds'][0]}-{data['seeds'][-1]}",
            }
            for m in METRICS:
                row[f"{m}_mean"] = ov[f"{m}_mean_over_seeds"]
                row[f"{m}_std"]  = ov[f"{m}_std_over_seeds"]
                vals = ov[f"{m}_per_seed"]
                row[f"{m}_min"]  = float(np.min(vals))
                row[f"{m}_max"]  = float(np.max(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    for loss in sorted(df["loss"].unique()):
        sub = df[df["loss"] == loss].copy()
        sub["order"] = sub["encoder"].apply(
            lambda e: ENCODER_ORDER.index(e) if e in ENCODER_ORDER else 99
        )
        sub = sub.sort_values("order")

        n_seeds = sub["n_seeds"].iloc[0] if len(sub) else 0
        print(f"\n{'═' * 82}")
        print(f" {loss}  —  mean ± std across {n_seeds} independent fold partitions")
        print(f"{'═' * 82}")
        print(f"{'Encoder':<18}{'macro F1':>18}{'Brier soft':>18}{'ECE':>18}{'H-corr':>18}")
        print("-" * 100)
        for _, r in sub.iterrows():
            print(
                f"{ENCODER_LABEL.get(r['encoder'], r['encoder']):<18}"
                f"{r['macro_f1_mean']:>11.3f} ±{r['macro_f1_std']:.3f}"
                f"{r['brier_soft_mean']:>11.4f} ±{r['brier_soft_std']:.4f}"
                f"{r['ece_mean']:>11.4f} ±{r['ece_std']:.4f}"
                f"{r['entropy_corr_mean']:>11.3f} ±{r['entropy_corr_std']:.3f}"
            )
        if len(sub):
            print("-" * 82)
            print(
                f"{'mean':<18}"
                f"{sub['macro_f1_mean'].mean():>11.3f} ±{sub['macro_f1_std'].mean():.3f}"
                f"{sub['brier_soft_mean'].mean():>11.4f} ±{sub['brier_soft_std'].mean():.4f}"
                f"{sub['ece_mean'].mean():>11.4f} ±{sub['ece_std'].mean():.4f}"
                f"{sub['entropy_corr_mean'].mean():>11.3f} ±{sub['entropy_corr_std'].mean():.3f}"
            )

    # Soft-KL vs Hard-CE, the comparison the paper turns on
    if {"soft_kl", "hard_ce"}.issubset(set(df["loss"])):
        soft = df[df["loss"] == "soft_kl"].set_index("encoder")
        hard = df[df["loss"] == "hard_ce"].set_index("encoder")
        common = [e for e in ENCODER_ORDER if e in soft.index and e in hard.index]

        print(f"\n{'═' * 82}")
        print(" Soft-KL vs Hard-CE, with across-seed spread")
        print(f"{'═' * 82}")
        print(f"{'Encoder':<18}{'Brier soft-KL':>18}{'Brier hard-CE':>18}{'ratio':>10}"
              f"{'F1 soft':>9}{'F1 hard':>9}")
        print("-" * 82)
        for e in common:
            s, h = soft.loc[e], hard.loc[e]
            ratio = h["brier_soft_mean"] / s["brier_soft_mean"] if s["brier_soft_mean"] else float("nan")
            print(
                f"{ENCODER_LABEL.get(e, e):<18}"
                f"{s['brier_soft_mean']:>11.4f} ±{s['brier_soft_std']:.4f}"
                f"{h['brier_soft_mean']:>11.4f} ±{h['brier_soft_std']:.4f}"
                f"{ratio:>9.2f}x"
                f"{s['macro_f1_mean']:>9.3f}{h['macro_f1_mean']:>9.3f}"
            )
        if common:
            s_mean = soft.loc[common, "brier_soft_mean"].mean()
            h_mean = hard.loc[common, "brier_soft_mean"].mean()
            print("-" * 82)
            print(f"{'mean':<18}{s_mean:>18.4f}{h_mean:>18.4f}"
                  f"{h_mean / s_mean:>9.2f}x"
                  f"{soft.loc[common, 'macro_f1_mean'].mean():>9.3f}"
                  f"{hard.loc[common, 'macro_f1_mean'].mean():>9.3f}")


def latex_table(df: pd.DataFrame) -> str:
    """Booktabs table of Soft-KL vs Hard-CE with across-seed std."""
    soft = df[df["loss"] == "soft_kl"].set_index("encoder")
    hard = df[df["loss"] == "hard_ce"].set_index("encoder")
    common = [e for e in ENCODER_ORDER if e in soft.index and e in hard.index]

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Encoder & $F_1$ Soft-KL & $F_1$ Hard-CE & $\brier$ Soft-KL & $\brier$ Hard-CE \\",
        r"\midrule",
    ]
    for e in common:
        s, h = soft.loc[e], hard.loc[e]
        lines.append(
            f"{ENCODER_LABEL.get(e, e)} & "
            f"${s['macro_f1_mean']:.3f}\\pm{s['macro_f1_std']:.3f}$ & "
            f"${h['macro_f1_mean']:.3f}\\pm{h['macro_f1_std']:.3f}$ & "
            f"${s['brier_soft_mean']:.3f}\\pm{s['brier_soft_std']:.3f}$ & "
            f"${h['brier_soft_mean']:.3f}\\pm{h['brier_soft_std']:.3f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results/cv_multiseed")
    p.add_argument("--output", default="", help="Optional CSV path.")
    p.add_argument("--latex", action="store_true", help="Also print a LaTeX table.")
    args = p.parse_args()

    df = load(Path(args.results_dir))
    if df.empty:
        print(f"No cv_results_multiseed.json found under {args.results_dir}")
        print("(the multi-seed job may still be queued or running)")
        return

    print_table(df)

    if args.latex:
        print("\n" + latex_table(df))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
