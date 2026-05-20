#!/usr/bin/env python3
"""
Generate figure: Soft-KL vs Hard-CE routing comparison (cross-evaluation).

Shows for each aid:
  - Standard (distance-only) route: grey bar
  - Soft-KL optimal route scored by Soft-KL (calibrated ground truth): blue bar
  - Hard-CE optimal route scored by Soft-KL (calibrated view of HCE path): red bar
  - Hard-CE self-reported score (inflated): red hatch bar

Usage:
    python src/routing/plot_hardce_comparison.py \
        --input  results/routing/hard_ce_comparison/cross_eval.json \
        --output results/routing/hard_ce_comparison/routing_hardce_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

AIDS_SHORT = {
    "Walking cane":       "Cane",
    "Walker":             "Walker",
    "Mobility scooter":   "Scooter",
    "Manual wheelchair":  "Man. WC",
    "Motorized wheelchair": "Mot. WC",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/routing/hard_ce_comparison/cross_eval.json")
    parser.add_argument("--output", default="results/routing/hard_ce_comparison/routing_hardce_comparison.png")
    args = parser.parse_args()

    data = json.load(open(args.input))

    aids   = [r["aid"] for r in data]
    labels = [AIDS_SHORT[a] for a in aids]

    p_std = np.array([r["p_yes_std_by_skl"] for r in data])
    p_skl = np.array([r["p_yes_skl_by_skl"] for r in data])
    p_hce_cal  = np.array([r["p_yes_hce_by_skl"] for r in data])   # HCE route, calibrated
    p_hce_self = np.array([r["p_yes_hce_by_hce"] for r in data])   # HCE self-reported

    x = np.arange(len(aids))
    width = 0.20

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_std  = ax.bar(x - 1.5*width, p_std,      width, label="Shortest path",             color="#aaaaaa", zorder=3)
    bars_skl  = ax.bar(x - 0.5*width, p_skl,      width, label="Soft-KL route (calibrated)",color="#2980b9", zorder=3)
    bars_hcec = ax.bar(x + 0.5*width, p_hce_cal,  width, label="Hard-CE route (calibrated)", color="#e74c3c", zorder=3)
    bars_hces = ax.bar(x + 1.5*width, p_hce_self, width, label="Hard-CE route (self-report, inflated)",
                       color="#e74c3c", alpha=0.35, hatch="///", zorder=3)

    # Annotate calibrated gaps
    for i, r in enumerate(data):
        gap = r["calibrated_gap"]
        y_top = p_skl[i] + 0.003
        ax.annotate(f"Δ={gap:+.3f}", xy=(x[i] - 0.5*width, y_top), ha="center", va="bottom",
                    fontsize=6.5, color="#2980b9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean $\\hat{p}_\\mathrm{yes}$ along route", fontsize=10)
    ax.set_ylim(0.38, 0.85)
    ax.set_title("Routing quality: Soft-KL vs Hard-CE edge scores\n"
                 "(evaluated by calibrated Soft-KL metric — ground truth)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved → {out}")
    print(f"Saved → {str(out).replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()
