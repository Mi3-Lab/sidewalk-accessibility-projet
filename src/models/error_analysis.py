#!/usr/bin/env python3
"""
Walking cane error analysis: why Hard-CE fails and Soft-KL succeeds.

Extracts DINOv2-large features for all training images, then runs Hard-CE vs
Soft-KL for Walking cane specifically, logging per-image predictions across
5-fold CV. Produces:
  - results/error_analysis/walking_cane_votes.png   — vote distributions for 7 'no' images
  - results/error_analysis/walking_cane_preds.png   — Hard-CE vs Soft-KL per image
  - results/error_analysis/walking_cane_summary.json

Usage:
    python src/models/error_analysis.py \
        --encoder    dinov2-large \
        --tallies    data/processed/tallies_firebase.json \
        --images_dir data/images/sidewalk-images \
        --output_dir results/error_analysis
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from train import (
    AIDS, CLASS3, ENCODERS, LABEL_MAP, RANDOM_STATE,
    extract_encoder_features, load_encoder, set_seed,
)

TARGET_AID = "Walking cane"
N_FOLDS    = 5
LR         = 1e-3
EPOCHS     = 50
WD         = 1e-4


# ── Loss functions ────────────────────────────────────────────────────────────

def soft_kl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits, dim=1)
    return torch.mean(torch.sum(-targets * log_p, dim=1))


def hard_ce_loss(logits: torch.Tensor, hard_labels: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    ce = nn.CrossEntropyLoss(reduction="none")(logits, hard_labels)
    return (ce * weights).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train_probe(X_train: np.ndarray, y_soft: np.ndarray,
                y_hard: np.ndarray, w_hard: np.ndarray,
                loss_type: str, device: torch.device) -> nn.Linear:
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train)
    X_t    = torch.tensor(X_s, dtype=torch.float32).to(device)
    probe  = nn.Linear(X_s.shape[1], 3).to(device)
    opt    = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WD)

    if loss_type == "soft_kl":
        y_t = torch.tensor(y_soft, dtype=torch.float32).to(device)
    else:
        y_t = torch.tensor(y_hard, dtype=torch.long).to(device)
        w_t = torch.tensor(w_hard, dtype=torch.float32).to(device)

    probe.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = probe(X_t)
        if loss_type == "soft_kl":
            loss = soft_kl_loss(logits, y_t)
        else:
            loss = hard_ce_loss(logits, y_t, w_t)
        loss.backward()
        opt.step()

    return probe, scaler


def predict(probe: nn.Linear, scaler, X_test: np.ndarray,
            device: torch.device) -> tuple:
    X_s = scaler.transform(X_test)
    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
    probe.eval()
    with torch.no_grad():
        logits = probe(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.argmax(axis=1), probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",    default="dinov2-large", choices=list(ENCODERS))
    parser.add_argument("--tallies",    default="data/processed/tallies_firebase.json")
    parser.add_argument("--images_dir", default="data/images/sidewalk-images")
    parser.add_argument("--output_dir", default="results/error_analysis")
    parser.add_argument("--seed",       type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tallies ──────────────────────────────────────────────────────────
    tallies = pd.DataFrame(json.load(open(args.tallies)))
    df = tallies[tallies["MobilityAid"] == TARGET_AID].copy()
    df["label_int"] = df["argmax_label"].map(LABEL_MAP)
    df = df[df["label_int"].notna()].copy()
    df["label_int"] = df["label_int"].astype(int)

    # entropy-based sample weight (used by Hard-CE)
    def entropy(row):
        import math
        H_max = math.log(3)
        ps = [row["p_no"], row["p_unsure"], row["p_yes"]]
        h = -sum(p * math.log(max(p, 1e-10)) for p in ps)
        return 1.0 - h / H_max
    df["sample_weight"] = df.apply(entropy, axis=1)

    # soft distribution target
    df["dist"] = df.apply(lambda r: [r["p_no"], r["p_unsure"], r["p_yes"]], axis=1)

    print(f"\n{TARGET_AID} dataset: {len(df)} samples")
    dist = df["label_int"].value_counts().sort_index()
    for i, cls in enumerate(CLASS3):
        n = dist.get(i, 0)
        print(f"  {cls:8s}: {n} ({100*n/len(df):.0f}%)")

    no_df = df[df["label_int"] == 0].copy()  # the 7 "no" images
    print(f"\nThe {len(no_df)} '{TARGET_AID}' NO images (borderline):")
    for _, row in no_df.sort_values("p_no", ascending=False).iterrows():
        bar_no  = "█" * int(row["p_no"]  * 20)
        bar_yes = "█" * int(row["p_yes"] * 20)
        margin_flag = " (AMBIGUOUS)" if row["margin"] < 0.05 else ""
        print(f"  ImageID={int(row['ImageID']):6d}  "
              f"p_no={row['p_no']:.3f} {bar_no:<10}  "
              f"p_yes={row['p_yes']:.3f} {bar_yes:<10}  "
              f"margin={row['margin']:.3f}{margin_flag}")

    # ── Image path resolution ─────────────────────────────────────────────────
    images_dir = Path(args.images_dir)
    image_files = list(images_dir.glob("*.png"))

    def find_images(image_id):
        return [p for p in image_files if f"-{image_id}-" in p.name]

    df["paths"] = df["ImageID"].apply(find_images)
    df = df.explode("paths").dropna(subset=["paths"])
    df["path_str"] = df["paths"].apply(str)

    missing = df[df["path_str"].apply(lambda p: not Path(p).exists())]
    if not missing.empty:
        print(f"\nWARNING: {len(missing)} images not found on disk.")
    df = df[df["path_str"].apply(lambda p: Path(p).exists())]
    print(f"\nImages on disk: {len(df)} rows ({df['ImageID'].nunique()} unique panoramas)")

    # ── Extract features ──────────────────────────────────────────────────────
    print(f"\nLoading encoder: {args.encoder}")
    model, processor, device, enc_type = load_encoder(args.encoder)

    paths = df["path_str"].tolist()
    X = extract_encoder_features(model, processor, device, paths, enc_type)
    print(f"Features: {X.shape}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 5-fold CV with per-image logging ─────────────────────────────────────
    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stratify by panorama-level label
    pano_label = (df.groupby("ImageID")["label_int"]
                  .agg(lambda x: x.mode().iloc[0])
                  .reset_index())
    pano_label.columns = ["ImageID", "label_int"]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    unique_panos = pano_label["ImageID"].values
    pano_y       = pano_label["label_int"].values

    # Per-image prediction storage
    results = {loss: {"y_pred": np.full(len(df), -1, int),
                      "y_prob": np.zeros((len(df), 3))}
               for loss in ("hard_ce", "soft_kl")}

    fold_metrics = {loss: [] for loss in ("hard_ce", "soft_kl")}

    print(f"\nRunning {N_FOLDS}-fold CV ...")
    for fold, (tr_pano_idx, te_pano_idx) in enumerate(skf.split(unique_panos, pano_y)):
        tr_ids = set(unique_panos[tr_pano_idx])
        te_ids = set(unique_panos[te_pano_idx])

        tr_mask = df["ImageID"].isin(tr_ids).values
        te_mask = df["ImageID"].isin(te_ids).values

        X_tr = X[tr_mask]; X_te = X[te_mask]
        y_hard_tr = df["label_int"].values[tr_mask]
        y_soft_tr = np.array(df["dist"].tolist())[tr_mask]
        w_hard_tr = df["sample_weight"].values[tr_mask].astype(float)
        y_te      = df["label_int"].values[te_mask]

        for loss_type in ("hard_ce", "soft_kl"):
            warnings.filterwarnings("ignore")
            probe, scaler = train_probe(
                X_tr, y_soft_tr, y_hard_tr, w_hard_tr,
                loss_type, probe_device
            )
            y_pred, y_prob = predict(probe, scaler, X_te, probe_device)

            results[loss_type]["y_pred"][te_mask] = y_pred
            results[loss_type]["y_prob"][te_mask]  = y_prob

            f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
            bal = balanced_accuracy_score(y_te, y_pred)
            fold_metrics[loss_type].append({"fold": fold, "f1": f1, "bal": bal})

        no_in_fold = (df["label_int"].values[te_mask] == 0).sum()
        print(f"  Fold {fold}: {no_in_fold} no-samples in test set")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n── Final CV metrics ───────────────────────────────────────────")
    print(f"{'Loss':<12} {'Macro F1':>10} {'Bal Acc':>10}")
    print("-"*35)
    summary = {}
    for loss_type in ("hard_ce", "soft_kl"):
        f1s  = [m["f1"]  for m in fold_metrics[loss_type]]
        bals = [m["bal"] for m in fold_metrics[loss_type]]
        print(f"{loss_type:<12} {np.mean(f1s):>10.3f}±{np.std(f1s):.3f} "
              f"{np.mean(bals):>10.3f}±{np.std(bals):.3f}")
        summary[loss_type] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std":  float(np.std(f1s)),
            "bal_acc_mean":  float(np.mean(bals)),
            "bal_acc_std":   float(np.std(bals)),
        }

    trivial_f1 = f1_score(df["label_int"].values,
                          np.full(len(df), 2, int),  # all-yes predictor
                          average="macro", zero_division=0)
    print(f"\nTrivial baseline (all yes): Macro F1 = {trivial_f1:.3f}")
    summary["trivial_baseline_f1"] = float(trivial_f1)

    # ── Per no-image breakdown ────────────────────────────────────────────────
    print("\n── Predictions on 7 'no' walking cane images ─────────────────")
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    print(f"{'ImageID':>8}  {'p_no':>6}  {'margin':>8}  {'Hard-CE':>10}  {'Soft-KL':>10}")
    print("-"*55)

    no_pred_records = []
    for _, row in no_df.sort_values("p_no", ascending=False).iterrows():
        img_id = int(row["ImageID"])
        idx    = np.where(df["ImageID"].values == img_id)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        hce_pred = inv_map.get(results["hard_ce"]["y_pred"][i], "?")
        skl_pred = inv_map.get(results["soft_kl"]["y_pred"][i], "?")
        hce_ok   = "✓" if results["hard_ce"]["y_pred"][i] == 0 else "✗"
        skl_ok   = "✓" if results["soft_kl"]["y_pred"][i] == 0 else "✗"
        print(f"{img_id:>8}  {row['p_no']:>6.3f}  {row['margin']:>8.3f}  "
              f"{hce_pred:>8}{hce_ok}  {skl_pred:>8}{skl_ok}")
        no_pred_records.append({
            "image_id":     img_id,
            "p_no":         float(row["p_no"]),
            "p_yes":        float(row["p_yes"]),
            "margin":       float(row["margin"]),
            "hard_ce_pred": hce_pred,
            "soft_kl_pred": skl_pred,
            "hard_ce_correct": hce_pred == "p_no",
            "soft_kl_correct": skl_pred == "p_no",
        })

    summary["no_image_preds"] = no_pred_records

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = output_dir / "walking_cane_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {json_path}")

    # ── Figure 1: vote distributions of the 7 'no' images ────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, len(no_df), figsize=(14, 3.5))
        colors = {"p_no": "#d62728", "p_unsure": "#ff7f0e", "p_yes": "#2ca02c"}

        for ax, (_, row) in zip(axes, no_df.sort_values("p_no", ascending=False).iterrows()):
            vals   = [row["p_no"], row["p_unsure"], row["p_yes"]]
            clrs   = [colors["p_no"], colors["p_unsure"], colors["p_yes"]]
            bars   = ax.bar(CLASS3, vals, color=clrs, edgecolor="white", linewidth=0.5)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"ID {int(row['ImageID'])}\nmargin={row['margin']:.3f}",
                         fontsize=8)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)

        patches = [mpatches.Patch(color=colors[k], label=k.replace("p_",""))
                   for k in ("p_no", "p_unsure", "p_yes")]
        fig.legend(handles=patches, loc="upper right", fontsize=8)
        fig.suptitle(f"{TARGET_AID} — Vote distributions of 7 'no'-labeled images\n"
                     f"5 of 7 have margin < 0.05 (nearly tied)", fontsize=10)
        plt.tight_layout()
        fig1_path = output_dir / "walking_cane_votes.png"
        plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure → {fig1_path}")

        # ── Figure 2: Hard-CE vs Soft-KL predictions on the 7 'no' images ──
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        x = np.arange(len(no_pred_records))
        labels_x = [f"ID {r['image_id']}\nmargin={r['margin']:.3f}" for r in no_pred_records]
        hce_correct = [1 if r["hard_ce_correct"] else 0 for r in no_pred_records]
        skl_correct = [1 if r["soft_kl_correct"] else 0 for r in no_pred_records]

        width = 0.35
        bars1 = ax2.bar(x - width/2, hce_correct, width, label="Hard-CE",
                        color="#1f77b4", alpha=0.8)
        bars2 = ax2.bar(x + width/2, skl_correct, width, label="Soft-KL",
                        color="#ff7f0e", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_x, fontsize=8)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Wrong (pred yes)", "Correct (pred no)"])
        ax2.set_title(f"{TARGET_AID}: Hard-CE vs Soft-KL predictions on 'no' images\n"
                      f"Hard-CE macro F1={summary['hard_ce']['macro_f1_mean']:.3f}  "
                      f"Soft-KL macro F1={summary['soft_kl']['macro_f1_mean']:.3f}  "
                      f"Trivial={trivial_f1:.3f}", fontsize=9)
        ax2.legend()
        plt.tight_layout()
        fig2_path = output_dir / "walking_cane_preds.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure → {fig2_path}")

    except ImportError:
        print("matplotlib not available — skipping figures")


if __name__ == "__main__":
    main()
