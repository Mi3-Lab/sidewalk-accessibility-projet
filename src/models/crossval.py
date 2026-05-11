#!/usr/bin/env python3
"""
Panorama-level K-fold cross-validation for sidewalk accessibility models.

Split is performed at the ImageID (panorama) level so that all crops from
the same panorama always land in the same fold — preventing data leakage.

Usage:
    python src/models/crossval.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --encoder      dinov2-base \
        --n_folds      5 \
        --output_dir   results/cv/dinov2-base

Run all 8 encoders sequentially:
    for enc in clip-vit-b32 clip-vit-b16 clip-vit-l14 \
               dinov2-base dinov2-large \
               siglip2-base siglip2-so400m vit-b16-sup; do
        python src/models/crossval.py --encoder $enc \
            --tallies_json data/processed/tallies_firebase.json \
            --images_dir   data/images/sidewalk-images \
            --output_dir   results/cv/$enc
    done
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from train import (
    AIDS,
    CLASS3,
    ENCODERS,
    LABEL_MAP,
    LOSS_TYPE,
    PROBE_EPOCHS,
    PROBE_LR,
    PROBE_WD,
    RANDOM_STATE,
    SOFT_COLS,
    extract_encoder_features,
    find_image_path,
    load_encoder,
    set_seed,
    train_probe_hard,
    train_probe_soft,
)



# ── Linear probe helpers ──────────────────────────────────────────────────────

def predict_probe(probe: nn.Linear, X: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = probe(torch.tensor(X, dtype=torch.float32).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def predict_proba(probe: nn.Linear, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return softmax probability vectors (N, 3)."""
    with torch.no_grad():
        logits = probe(torch.tensor(X, dtype=torch.float32).to(device))
        probs  = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


# ── Calibration metrics ───────────────────────────────────────────────────────

def brier_score_soft(y_prob: np.ndarray, y_soft: np.ndarray) -> float:
    """MSE between predicted distribution and human vote distribution.

    Lower = model better reproduces human perception uncertainty.
    Meaningful only when y_soft has non-trivial spread (not one-hot).
    """
    return float(np.mean(np.sum((y_prob - y_soft) ** 2, axis=1)))


def brier_score_hard(y_prob: np.ndarray, y_true: np.ndarray, n_classes: int = 3) -> float:
    """Standard Brier score: MSE between predicted probs and one-hot hard labels."""
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: weighted mean |accuracy − confidence| across confidence bins."""
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct      = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


# ── Class distribution report ─────────────────────────────────────────────────

def report_class_distribution(df: pd.DataFrame) -> dict:
    dist: dict[str, dict] = {}
    for aid in AIDS:
        sub = df[df["MobilityAid"] == aid]
        counts = sub["argmax_label"].value_counts().to_dict()
        dist[aid] = {
            "total":  len(sub),
            "no":     counts.get("p_no",     0),
            "unsure": counts.get("p_unsure", 0),
            "yes":    counts.get("p_yes",    0),
        }
    return dist


# ── Per-aid K-fold CV ─────────────────────────────────────────────────────────

def crossval_aid(
    model,
    processor,
    device: torch.device,
    df_aid: pd.DataFrame,
    aid: str,
    enc_type: str,
    n_folds: int,
    loss_type: str = LOSS_TYPE,
) -> dict:
    """
    Run n_folds CV for a single mobility aid.
    Folds are defined over unique ImageIDs (panoramas).
    All crops/rows from the same ImageID travel together.
    """
    df_aid = df_aid.copy()
    df_aid["label_int"] = df_aid["argmax_label"].map(LABEL_MAP)

    unique_panos = df_aid["ImageID"].unique()
    n = len(unique_panos)

    if n < n_folds:
        print(f"  [{aid}] only {n} panoramas < {n_folds} folds; using leave-one-out.")
        n_folds = n

    # One representative label per panorama for stratification
    pano_label = (
        df_aid.groupby("ImageID")["label_int"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    pano_label.columns = ["ImageID", "label_int"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    y_arr = pano_label["label_int"].values

    fold_metrics: list[dict] = []
    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (tr_pano_idx, te_pano_idx) in enumerate(
        skf.split(y_arr, y_arr)
    ):
        tr_pano_ids = pano_label.iloc[tr_pano_idx]["ImageID"].values
        te_pano_ids = pano_label.iloc[te_pano_idx]["ImageID"].values

        tr_df = df_aid[df_aid["ImageID"].isin(tr_pano_ids)]
        te_df = df_aid[df_aid["ImageID"].isin(te_pano_ids)]

        if tr_df.empty or te_df.empty:
            print(f"  [{aid}] fold {fold_idx}: empty split — skipping.")
            continue

        set_seed(RANDOM_STATE + fold_idx)   # unique but deterministic per fold
        t0 = time.perf_counter()

        X_train = extract_encoder_features(
            model, processor, device, tr_df["path"].tolist(), enc_type
        )
        X_test = extract_encoder_features(
            model, processor, device, te_df["path"].tolist(), enc_type
        )

        y_test  = te_df["label_int"].values
        w_train = tr_df["sample_weight"].values

        scaler  = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        if loss_type == "soft_kl":
            y_soft = tr_df[SOFT_COLS].values.astype(np.float32)
            probe  = train_probe_soft(X_train, y_soft, w_train, X_train.shape[1], probe_device)
        else:
            y_train = tr_df["label_int"].values
            probe   = train_probe_hard(X_train, y_train, w_train, X_train.shape[1], probe_device)

        y_prob = predict_proba(probe, X_test, probe_device)   # (N, 3) softmax
        y_pred = y_prob.argmax(axis=1)
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        y_soft_test = te_df[SOFT_COLS].values.astype(np.float32)  # human distribution

        elapsed = time.perf_counter() - t0
        m = {
            "fold":            fold_idx,
            "n_train_panos":   len(tr_pano_ids),
            "n_test_panos":    len(te_pano_ids),
            "n_train_samples": len(tr_df),
            "n_test_samples":  len(te_df),
            "accuracy":        float(accuracy_score(y_test, y_pred)),
            "macro_f1":        float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "balanced_acc":    float(balanced_accuracy_score(y_test, y_pred)),
            "brier_soft":      brier_score_soft(y_prob, y_soft_test),
            "brier_hard":      brier_score_hard(y_prob, y_test),
            "ece":             expected_calibration_error(y_prob, y_test),
            "elapsed_s":       round(elapsed, 2),
        }
        fold_metrics.append(m)
        print(
            f"  [{aid}] fold {fold_idx+1}/{n_folds} | "
            f"test_panos={len(te_pano_ids)} | "
            f"macro_f1={m['macro_f1']:.3f} | "
            f"bal_acc={m['balanced_acc']:.3f} | "
            f"brier_soft={m['brier_soft']:.3f} | "
            f"ece={m['ece']:.3f}"
        )
        if _WANDB_AVAILABLE and wandb.run is not None:
            aid_key = aid.lower().replace(" ", "_")
            wandb.log({
                f"fold/{aid_key}/macro_f1":    m["macro_f1"],
                f"fold/{aid_key}/balanced_acc": m["balanced_acc"],
                f"fold/{aid_key}/brier_soft":  m["brier_soft"],
                f"fold/{aid_key}/brier_hard":  m["brier_hard"],
                f"fold/{aid_key}/ece":         m["ece"],
            })

    if not fold_metrics:
        return {}

    def _mean(key): return float(np.mean([m[key] for m in fold_metrics]))
    def _std(key):  return float(np.std( [m[key] for m in fold_metrics]))

    return {
        "aid":                aid,
        "n_folds":            len(fold_metrics),
        "macro_f1_mean":      _mean("macro_f1"),
        "macro_f1_std":       _std("macro_f1"),
        "balanced_acc_mean":  _mean("balanced_acc"),
        "balanced_acc_std":   _std("balanced_acc"),
        "accuracy_mean":      _mean("accuracy"),
        "accuracy_std":       _std("accuracy"),
        "brier_soft_mean":    _mean("brier_soft"),
        "brier_soft_std":     _std("brier_soft"),
        "brier_hard_mean":    _mean("brier_hard"),
        "brier_hard_std":     _std("brier_hard"),
        "ece_mean":           _mean("ece"),
        "ece_std":            _std("ece"),
        "folds":              fold_metrics,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panorama-level K-fold CV for sidewalk accessibility models."
    )
    parser.add_argument("--tallies_json", required=True)
    parser.add_argument("--images_dir",   required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument(
        "--encoder",
        default="clip-vit-b32",
        choices=list(ENCODERS),
        help=f"Vision encoder. Options: {list(ENCODERS)}",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--loss_type",
        default=LOSS_TYPE,
        choices=["soft_kl", "hard_ce"],
        help="soft_kl: KL divergence on vote distribution (default). hard_ce: argmax + entropy weight (ablation).",
    )
    parser.add_argument("--wandb_project", default="", help="W&B project name. Empty string disables W&B.")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(
        lambda x: find_image_path(x, args.images_dir)
    )
    tallies = tallies[tallies["path"].notna()]

    class_dist = report_class_distribution(tallies)
    print(f"\nEncoder: {args.encoder}  ({ENCODERS[args.encoder][0]})")
    print(f"Images found: {len(tallies)} rows\n")
    print("Class distribution per aid:")
    for aid, d in class_dist.items():
        print(
            f"  {aid:<30}  yes={d['yes']:3d}  "
            f"unsure={d['unsure']:3d}  no={d['no']:3d}  total={d['total']:3d}"
        )
    print()

    # ── Load encoder ─────────────────────────────────────────────────────────
    model, processor, device, enc_type = load_encoder(args.encoder)

    # ── CV config ────────────────────────────────────────────────────────────
    config = {
        "encoder":      args.encoder,
        "hf_id":        ENCODERS[args.encoder][0],
        "enc_type":     enc_type,
        "loss_type":    args.loss_type,
        "n_folds":      args.n_folds,
        "lr":           PROBE_LR,
        "epochs":       PROBE_EPOCHS,
        "weight_decay": PROBE_WD,
        "random_state": RANDOM_STATE,
    }


    # ── Init W&B run (one per encoder) ──────────────────────────────────────────
    if _WANDB_AVAILABLE and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.loss_type}/{args.encoder}",
            group=f"cv_{args.loss_type}",
            tags=[args.loss_type, "cv", "linear_probe", args.encoder],
            config=config,
        )

    all_results: dict = {
        "config":             config,
        "class_distribution": class_dist,
        "aids":               {},
    }

    # ── Run CV per aid ───────────────────────────────────────────────────────
    for aid in AIDS:
        df_aid = tallies[tallies["MobilityAid"] == aid]
        if df_aid.empty:
            print(f"[{aid}] no data — skipping.")
            continue
        print(
            f"\n── {aid} "
            f"({len(df_aid)} rows, {df_aid['ImageID'].nunique()} panoramas) ──"
        )
        summary = crossval_aid(
            model, processor, device,
            df_aid, aid, enc_type, args.n_folds, args.loss_type,
        )
        if summary:
            all_results["aids"][aid] = summary
            print(
                f"  → macro_f1 = {summary['macro_f1_mean']:.3f}"
                f" ± {summary['macro_f1_std']:.3f}"
                f"  |  bal_acc = {summary['balanced_acc_mean']:.3f}"
                f" ± {summary['balanced_acc_std']:.3f}"
            )
            if _WANDB_AVAILABLE and wandb.run is not None:
                aid_key = aid.lower().replace(" ", "_")
                wandb.summary[f"{aid_key}/macro_f1_mean"]    = round(summary["macro_f1_mean"],     4)
                wandb.summary[f"{aid_key}/bal_acc_mean"]     = round(summary["balanced_acc_mean"], 4)
                wandb.summary[f"{aid_key}/brier_soft_mean"]  = round(summary["brier_soft_mean"],   4)
                wandb.summary[f"{aid_key}/brier_hard_mean"]  = round(summary["brier_hard_mean"],   4)
                wandb.summary[f"{aid_key}/ece_mean"]         = round(summary["ece_mean"],          4)

    # ── Aggregate across aids ────────────────────────────────────────────────
    if all_results["aids"]:
        aids_vals = all_results["aids"].values()
        all_results["overall"] = {
            "macro_f1_mean_across_aids":   float(np.mean([v["macro_f1_mean"]    for v in aids_vals])),
            "macro_f1_std_across_aids":    float(np.std( [v["macro_f1_mean"]    for v in aids_vals])),
            "brier_soft_mean_across_aids": float(np.mean([v["brier_soft_mean"]  for v in aids_vals])),
            "brier_hard_mean_across_aids": float(np.mean([v["brier_hard_mean"]  for v in aids_vals])),
            "ece_mean_across_aids":        float(np.mean([v["ece_mean"]         for v in aids_vals])),
        }

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = output_dir / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    if _WANDB_AVAILABLE and wandb.run is not None and "overall" in all_results:
        ov = all_results["overall"]
        wandb.summary["overall/macro_f1_mean"]   = round(ov["macro_f1_mean_across_aids"],   4)
        wandb.summary["overall/macro_f1_std"]    = round(ov["macro_f1_std_across_aids"],    4)
        wandb.summary["overall/brier_soft_mean"] = round(ov["brier_soft_mean_across_aids"], 4)
        wandb.summary["overall/brier_hard_mean"] = round(ov["brier_hard_mean_across_aids"], 4)
        wandb.summary["overall/ece_mean"]        = round(ov["ece_mean_across_aids"],        4)
        wandb.finish()

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n── Summary table ──────────────────────────────────────────────────")
    print(f"{'Aid':<30} {'macro_f1':>10}  {'bal_acc':>10}  {'brier_soft':>11}  {'brier_hard':>11}  {'ece':>7}")
    print("-" * 88)
    for aid, v in all_results["aids"].items():
        print(
            f"{aid:<30} {v['macro_f1_mean']:>10.3f}"
            f" ±{v['macro_f1_std']:.3f}"
            f"  {v['balanced_acc_mean']:>10.3f}"
            f" ±{v['balanced_acc_std']:.3f}"
            f"  {v['brier_soft_mean']:>11.4f}"
            f"  {v['brier_hard_mean']:>11.4f}"
            f"  {v['ece_mean']:>7.4f}"
        )
    if "overall" in all_results:
        ov = all_results["overall"]
        print(
            f"\n{'Overall (mean across aids)':<30} "
            f"{ov['macro_f1_mean_across_aids']:>10.3f}"
            f" ±{ov['macro_f1_std_across_aids']:.3f}"
            f"  {'brier_soft':>11}={ov['brier_soft_mean_across_aids']:.4f}"
            f"  {'brier_hard':>11}={ov['brier_hard_mean_across_aids']:.4f}"
            f"  ece={ov['ece_mean_across_aids']:.4f}"
        )
    print("──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
