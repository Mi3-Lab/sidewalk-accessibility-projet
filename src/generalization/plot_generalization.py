#!/usr/bin/env python3
"""
Bar chart: balanced accuracy per city for generalization evaluation.

Usage:
    python src/generalization/plot_generalization.py \
        --summary results/generalization/dinov2-large/agreement_summary.json \
        --output  results/generalization/dinov2-large/generalization_figure
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CITY_META = {
    "pittsburgh": {"label": "Pittsburgh\n(USA)",  "continent": "North America", "color": "#4393c3"},
    "detroit":    {"label": "Detroit\n(USA)",      "continent": "North America", "color": "#2166ac"},
    "zurich":     {"label": "Zurich\n(Europe)",    "continent": "Europe",        "color": "#d6604d"},
    "taipei":     {"label": "Taipei\n(Asia)",      "continent": "Asia",          "color": "#4dac26"},
}

CONTINENT_COLORS = {
    "North America": "#4393c3",
    "Europe":        "#d6604d",
    "Asia":          "#4dac26",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/generalization/dinov2-large/agreement_summary.json")
    parser.add_argument("--output",  default="results/generalization/dinov2-large/generalization_figure")
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    per_city = data["per_city"]
    overall_bal_acc = data["overall_balanced_accuracy"]

    # Sort: North America first, then Europe, then Asia
    order = ["pittsburgh", "detroit", "zurich", "taipei"]
    cities    = [c for c in order if c in per_city]
    bal_accs  = [per_city[c]["balanced_accuracy"] for c in cities]
    ns        = [per_city[c]["n"] for c in cities]
    labels    = [CITY_META[c]["label"] for c in cities]
    colors    = [CITY_META[c]["color"] for c in cities]
    continents = [CITY_META[c]["continent"] for c in cities]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(cities))
    bars = ax.bar(x, bal_accs, color=colors, edgecolor="white", linewidth=0.8, width=0.55, zorder=3)

    # Chance line
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Chance (0.50)", zorder=4)

    # Overall mean line
    ax.axhline(overall_bal_acc, color="gray", linestyle=":", linewidth=1.2,
               label=f"Overall mean ({overall_bal_acc:.3f})", zorder=4)

    # Annotate bars with value + n
    for bar, acc, n in zip(bars, bal_accs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{acc:.3f}\n(n={n})",
                ha="center", va="bottom", fontsize=9.5)

    # Legend for continents
    legend_handles = [
        mpatches.Patch(color=CONTINENT_COLORS[c], label=c)
        for c in ["North America", "Europe", "Asia"]
    ]
    legend_handles += [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Chance (0.50)"),
        plt.Line2D([0], [0], color="gray",  linestyle=":",  linewidth=1.2,
                   label=f"Overall ({overall_bal_acc:.3f})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title("Generalization: DINOv2-large across unseen cities\n"
                 "(trained on Pittsburgh passability votes, tested on CurbRamp/NoCurbRamp labels)", fontsize=10)
    ax.set_ylim(0.40, 0.72)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generalization figure → {out}.png")
    print(f"\nPer-city balanced accuracy:")
    for c, ba, n in zip(cities, bal_accs, ns):
        print(f"  {c:<12}  {ba:.3f}  (n={n})")
    print(f"  Overall:     {overall_bal_acc:.3f}  (n={data['n_test_samples']})")


if __name__ == "__main__":
    main()
