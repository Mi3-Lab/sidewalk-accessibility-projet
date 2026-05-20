#!/usr/bin/env python3
"""
Generate figure: Soft-KL vs Hard-CE routing comparison (cross-evaluation).

Two-panel layout:
  Left  — calibrated routing quality: std / Soft-KL route / Hard-CE route,
           all evaluated by the Soft-KL (calibrated) metric.
  Right — Hard-CE overestimation: how much Hard-CE inflates its own route
           quality vs what the calibrated metric says.

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
import numpy as np

AIDS_SHORT = {
    "Walking cane":         "Cane",
    "Walker":               "Walker",
    "Mobility scooter":     "Scooter",
    "Manual wheelchair":    "Man. WC",
    "Motorized wheelchair": "Mot. WC",
}

CLR_STD  = "#aaaaaa"
CLR_SKL  = "#2980b9"
CLR_HCE  = "#e74c3c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/routing/hard_ce_comparison/cross_eval.json")
    parser.add_argument("--output", default="results/routing/hard_ce_comparison/routing_hardce_comparison.png")
    args = parser.parse_args()

    data   = json.load(open(args.input))
    labels = [AIDS_SHORT[r["aid"]] for r in data]
    x      = np.arange(len(data))
    w      = 0.25

    p_std      = np.array([r["p_yes_std_by_skl"] for r in data])
    p_skl      = np.array([r["p_yes_skl_by_skl"] for r in data])
    p_hce_cal  = np.array([r["p_yes_hce_by_skl"] for r in data])
    p_hce_self = np.array([r["p_yes_hce_by_hce"] for r in data])
    overest    = p_hce_self - p_hce_cal          # inflation = self-report minus reality
    gap        = np.array([r["calibrated_gap"]   for r in data])  # SKL − HCE (calibrated)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [3, 2]})

    # ── Left panel: calibrated routing quality ─────────────────────────────
    ax1.bar(x - w, p_std,     w, label="Shortest path",              color=CLR_STD, zorder=3)
    ax1.bar(x,     p_skl,     w, label="Soft-KL route",              color=CLR_SKL, zorder=3)
    ax1.bar(x + w, p_hce_cal, w, label="Hard-CE route",              color=CLR_HCE, zorder=3)

    # Annotate the gap (Soft-KL advantage over Hard-CE)
    for i in range(len(data)):
        y_top = max(p_skl[i], p_hce_cal[i]) + 0.004
        ax1.annotate(f"Δ={gap[i]:+.3f}",
                     xy=(x[i] + 0.5*w, y_top), ha="right", va="bottom",
                     fontsize=7, color=CLR_SKL, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Mean $\\hat{p}_\\mathrm{yes}$ along route\n(Soft-KL calibrated metric)", fontsize=9)
    ax1.set_ylim(0.38, 0.78)
    ax1.set_title("(a) Routing quality — Soft-KL beats Hard-CE\non every aid (calibrated evaluation)", fontsize=9)
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Right panel: Hard-CE overestimation ───────────────────────────────
    ax2.bar(x, overest, 0.45, color=CLR_HCE, zorder=3, alpha=0.85)
    for i in range(len(data)):
        ax2.text(x[i], overest[i] + 0.002, f"+{overest[i]:.3f}",
                 ha="center", va="bottom", fontsize=8, color=CLR_HCE, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Self-report minus calibrated $\\hat{p}_\\mathrm{yes}$", fontsize=9)
    ax2.set_ylim(0, 0.14)
    ax2.set_title("(b) Hard-CE overestimates own\nroute quality (miscalibration)", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved → {out}")
    print(f"Saved → {str(out).replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
