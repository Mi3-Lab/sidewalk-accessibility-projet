#!/usr/bin/env python3
"""
Generate all paper figures using seaborn.

Outputs (results/figures/):
  fig1_encoder_f1_heatmap.pdf      — encoder × aid macro-F1 heatmap (soft_kl)
  fig2_brier_comparison.pdf        — brier_soft: soft_kl vs hard_ce per encoder
  fig3_latency_vs_f1.pdf           — latency vs F1 scatter
  fig4_zeroshot_vs_probe.pdf       — zero-shot VLMs vs trained probes
  fig5_entropy_histogram.pdf       — vote distribution entropy per (image×aid)

Usage:
    python src/models/plot_results.py \
        --tallies_json data/processed/tallies_firebase.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy as scipy_entropy

# ── Global style ───────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", context="paper", font="serif")
PALETTE_BLUE  = "#2563EB"
PALETTE_RED   = "#DC2626"
PALETTE_GREEN = "#16A34A"

ENCODER_LABELS = {
    "clip-vit-b32":   "CLIP B/32",
    "clip-vit-b16":   "CLIP B/16",
    "clip-vit-l14":   "CLIP L/14",
    "dinov2-base":    "DINOv2-B",
    "dinov2-large":   "DINOv2-L",
    "siglip2-base":   "SigLIP2-B",
    "siglip2-so400m": "SigLIP2-SO",
    "vit-b16-sup":    "ViT-B/16-sup",
}

AIDS_SHORT = {
    "Walking cane":        "Walk. cane",
    "Walker":              "Walker",
    "Mobility scooter":    "Mob. scooter",
    "Manual wheelchair":   "Manual WC",
    "Motorized wheelchair":"Motor. WC",
}

ENCODER_ORDER = list(ENCODER_LABELS.keys())


def load_cv(cv_dir: Path, loss_type: str) -> dict:
    data = {}
    for enc in ENCODER_ORDER:
        p = cv_dir / loss_type / enc / "cv_results.json"
        if p.exists():
            data[enc] = json.load(open(p))
    return data


def save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── Fig 1: Encoder × Aid F1 heatmap ───────────────────────────────────────────

def fig1_heatmap(soft_kl: dict, out: Path):
    aids = list(AIDS_SHORT.keys())
    encs = [e for e in ENCODER_ORDER if e in soft_kl]

    matrix = pd.DataFrame(
        [[soft_kl[enc]["aids"].get(aid, {}).get("macro_f1_mean", np.nan) for aid in aids]
         for enc in encs],
        index=[ENCODER_LABELS[e] for e in encs],
        columns=[AIDS_SHORT[a] for a in aids],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix,
        annot=True, fmt=".2f",
        cmap="YlGn",
        vmin=0.35, vmax=0.75,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"label": "Macro-F1", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Encoder × Mobility Aid — Macro-F1\n5-fold CV · Soft-KL · seed=42",
                 fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=25, ha="left")
    plt.yticks(rotation=0)
    save(fig, out / "fig1_encoder_f1_heatmap.pdf")


# ── Fig 2: Brier soft — soft_kl vs hard_ce ────────────────────────────────────

def fig2_brier(soft_kl: dict, hard_ce: dict, out: Path):
    encs = [e for e in ENCODER_ORDER if e in soft_kl and e in hard_ce]
    rows = []
    for enc in encs:
        rows.append({
            "Encoder": ENCODER_LABELS[enc],
            "Brier Score (soft)": soft_kl[enc]["overall"]["brier_soft_mean_across_aids"],
            "Method": "Soft-KL (ours)",
        })
        rows.append({
            "Encoder": ENCODER_LABELS[enc],
            "Brier Score (soft)": hard_ce[enc]["overall"]["brier_soft_mean_across_aids"],
            "Method": "Hard-CE (baseline)",
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(
        data=df, x="Encoder", y="Brier Score (soft)", hue="Method",
        palette={"Soft-KL (ours)": PALETTE_BLUE, "Hard-CE (baseline)": PALETTE_RED},
        alpha=0.88, edgecolor="white", linewidth=0.8,
        ax=ax,
    )

    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333",
            )

    ax.set_title("Calibration Quality: Soft-KL vs Hard-CE\n"
                 "Brier score against human vote distributions — lower is better",
                 fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Brier Score (soft) ↓")
    ax.set_ylim(0, 0.32)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="", frameon=True, loc="upper left")
    sns.despine(ax=ax)
    save(fig, out / "fig2_brier_comparison.pdf")


# ── Fig 3: Latency vs F1 scatter ──────────────────────────────────────────────

ENCODER_MARKERS = {
    "clip-vit-b32":   "o",
    "clip-vit-b16":   "s",
    "clip-vit-l14":   "^",
    "dinov2-base":    "D",
    "dinov2-large":   "v",
    "siglip2-base":   "P",
    "siglip2-so400m": "*",
    "vit-b16-sup":    "X",
}


def fig3_latency(soft_kl: dict, hard_ce: dict, latency: dict, out: Path):
    lat_map = {r["encoder"]: r["mean_ms"] for r in latency["encoders"]}

    rows = []
    for enc in ENCODER_ORDER:
        if enc not in lat_map:
            continue
        if enc in soft_kl:
            rows.append({
                "enc_key": enc,
                "Encoder": ENCODER_LABELS[enc],
                "Latency (ms)": lat_map[enc],
                "Macro-F1": soft_kl[enc]["overall"]["macro_f1_mean_across_aids"],
                "Method": "Soft-KL",
            })
        if enc in hard_ce:
            rows.append({
                "enc_key": enc,
                "Encoder": ENCODER_LABELS[enc],
                "Latency (ms)": lat_map[enc],
                "Macro-F1": hard_ce[enc]["overall"]["macro_f1_mean_across_aids"],
                "Method": "Hard-CE",
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {"Soft-KL": PALETTE_BLUE, "Hard-CE": PALETTE_RED}

    # Draw connector lines between Soft-KL and Hard-CE for the same encoder
    for enc_key, grp in df.groupby("enc_key"):
        if len(grp) == 2:
            x = grp["Latency (ms)"].values
            y = grp["Macro-F1"].values
            ax.plot(x, y, color="#aaa", linewidth=0.9, zorder=1, linestyle="--")

    # Draw points: shape = encoder, color = method
    for _, row in df.iterrows():
        ax.scatter(
            row["Latency (ms)"], row["Macro-F1"],
            marker=ENCODER_MARKERS[row["enc_key"]],
            color=palette[row["Method"]],
            s=110, zorder=3, edgecolors="white", linewidths=0.7,
        )

    # ── Legend: two sections ──────────────────────────────────────────────────
    # Section 1 — Method (color)
    method_handles = [
        mpatches.Patch(color=PALETTE_BLUE, label="Soft-KL"),
        mpatches.Patch(color=PALETTE_RED,  label="Hard-CE"),
    ]
    # Section 2 — Encoder (marker shape, neutral color)
    from matplotlib.lines import Line2D
    enc_handles = [
        Line2D([0], [0], marker=ENCODER_MARKERS[enc], color="w",
               markerfacecolor="#555", markersize=7, label=ENCODER_LABELS[enc])
        for enc in ENCODER_ORDER if enc in lat_map
    ]
    legend1 = ax.legend(handles=method_handles, title="Method", loc="upper left",
                        frameon=True, fontsize=8.5)
    ax.add_artist(legend1)
    ax.legend(handles=enc_handles, title="Encoder", loc="lower right",
              frameon=True, fontsize=7.5, ncol=2)

    ax.set_xlabel("Feature Extraction Latency (ms/image, A100)")
    ax.set_ylabel("Macro-F1 (5-fold CV)")
    ax.set_title("Accuracy vs Speed Trade-off\n"
                 "Dashed lines connect the same encoder across methods",
                 fontsize=11, pad=10)
    ax.set_xlim(52, 112)
    sns.despine(ax=ax)
    save(fig, out / "fig3_latency_vs_f1.pdf")


# ── Fig 4: Zero-shot vs probe ──────────────────────────────────────────────────

def fig4_zeroshot(soft_kl: dict, hard_ce: dict, zeroshot_dir: Path, out: Path):
    rows = []

    best_soft = max(soft_kl, key=lambda e: soft_kl[e]["overall"]["macro_f1_mean_across_aids"])
    best_hard = max(hard_ce, key=lambda e: hard_ce[e]["overall"]["macro_f1_mean_across_aids"])

    rows.append({
        "Method": f"Soft-KL\n({ENCODER_LABELS[best_soft]})",
        "Macro-F1": soft_kl[best_soft]["overall"]["macro_f1_mean_across_aids"],
        "Type": "Trained probe",
    })
    rows.append({
        "Method": f"Hard-CE\n({ENCODER_LABELS[best_hard]})",
        "Macro-F1": hard_ce[best_hard]["overall"]["macro_f1_mean_across_aids"],
        "Type": "Trained probe",
    })

    vlm_labels = {
        "llava-1.5-7b":  "LLaVA-1.5-7B\n(zero-shot)",
        "qwen2.5-vl-7b": "Qwen2.5-VL-7B\n(zero-shot)",
        "qwen3-vl-8b":   "Qwen3-VL-8B\n(zero-shot)",
    }
    for model_dir in sorted(zeroshot_dir.iterdir()):
        p = model_dir / "zero_shot_results.json"
        if not p.exists(): continue
        d = json.load(open(p))
        rows.append({
            "Method": vlm_labels.get(d["model"], d["model"]),
            "Macro-F1": d["overall"]["macro_f1_mean_across_aids"],
            "Type": "Zero-shot VLM",
        })

    df = pd.DataFrame(rows)
    palette = {"Trained probe": PALETTE_BLUE, "Zero-shot VLM": PALETTE_RED}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = sns.barplot(
        data=df, x="Method", y="Macro-F1", hue="Type",
        palette=palette, alpha=0.88, edgecolor="white", linewidth=0.8,
        dodge=False, ax=ax,
    )

    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.004,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222",
            )

    best_probe_f1 = df[df["Type"] == "Trained probe"]["Macro-F1"].max()
    ax.axhline(best_probe_f1, color=PALETTE_BLUE, linestyle="--",
               linewidth=1.2, alpha=0.6, label=f"Best probe ({best_probe_f1:.3f})")

    ax.set_title("Trained Probe vs Zero-Shot VLMs\nOverall Macro-F1 (5 mobility aids)",
                 fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.35, 0.68)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.legend(title="", frameon=True, loc="lower right")
    sns.despine(ax=ax)
    save(fig, out / "fig4_zeroshot_vs_probe.pdf")


# ── Fig 5: Vote entropy histogram ─────────────────────────────────────────────

def fig5_entropy(tallies_json: str, out: Path):
    df = pd.read_json(tallies_json)
    aids = list(AIDS_SHORT.keys())
    max_h = np.log(3)

    all_rows = []
    for aid in aids:
        sub = df[df["MobilityAid"] == aid][["p_no", "p_unsure", "p_yes"]].dropna()
        for _, row in sub.iterrows():
            p = np.array([row["p_no"], row["p_unsure"], row["p_yes"]]) + 1e-9
            h = scipy_entropy(p) / max_h
            all_rows.append({"Aid": AIDS_SHORT[aid], "Normalised Entropy": h})

    df_h = pd.DataFrame(all_rows)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), sharey=True)
    for ax, aid in zip(axes, aids):
        sub = df_h[df_h["Aid"] == AIDS_SHORT[aid]]["Normalised Entropy"]
        sns.histplot(
            sub, bins=12, ax=ax,
            color=PALETTE_GREEN, alpha=0.8, edgecolor="white", linewidth=0.6,
        )
        ax.axvline(sub.mean(), color=PALETTE_RED, linestyle="--",
                   linewidth=1.5, label=f"μ={sub.mean():.2f}")
        ax.set_title(AIDS_SHORT[aid], fontsize=9, pad=6)
        ax.set_xlabel("Norm. Entropy", fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=7.5, frameon=False)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Count")
    fig.suptitle(
        "Vote Distribution Entropy per (Image × Mobility Aid)\n"
        "H = 0: full consensus   ·   H = 1: maximum disagreement",
        fontsize=10, y=1.03,
    )
    save(fig, out / "fig5_entropy_histogram.pdf")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_dir",       default="results/cv")
    parser.add_argument("--zeroshot_dir", default="results/zero_shot")
    parser.add_argument("--latency_json", default="results/latency/latency_results.json")
    parser.add_argument("--tallies_json", required=True)
    parser.add_argument("--output_dir",   default="results/figures")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading results …")
    cv_dir  = Path(args.cv_dir)
    soft_kl = load_cv(cv_dir, "soft_kl")
    hard_ce = load_cv(cv_dir, "hard_ce")
    latency = json.load(open(args.latency_json))
    print(f"  soft_kl: {len(soft_kl)} encoders | hard_ce: {len(hard_ce)} encoders")

    print("\nGenerating figures …")
    fig1_heatmap(soft_kl, out)
    fig2_brier(soft_kl, hard_ce, out)
    fig3_latency(soft_kl, hard_ce, latency, out)
    fig4_zeroshot(soft_kl, hard_ce, Path(args.zeroshot_dir), out)
    fig5_entropy(args.tallies_json, out)

    print(f"\nDone — {out}/")


if __name__ == "__main__":
    main()
