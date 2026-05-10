#!/usr/bin/env python3
"""
Panorama-level K-fold cross-validation for sidewalk accessibility models.

Split is performed at the ImageID (panorama) level so that all crops from
the same panorama always land in the same fold — preventing data leakage
between train and test.

Usage:
    python src/models/crossval.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --model_type   clip \
        --model_name   openai/clip-vit-base-patch32 \
        --n_folds      5 \
        --output_dir   results/cv_clip
"""

import argparse
import json
import sys
import time
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

from train import (
    AIDS,
    CLASS3,
    extract_features,
    extract_vision_features,
    find_image_path,
    load_clip_model,
    load_vision_model,
)

RANDOM_STATE = 42


# ── feature extraction ───────────────────────────────────────────────────────

def get_features(model, processor, device, paths, model_type):
    if model_type == "clip":
        return extract_features(model, processor, device, paths)
    return extract_vision_features(model, processor, device, paths)


# ── linear probe ─────────────────────────────────────────────────────────────

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    lr: float,
    epochs: int,
    device: torch.device,
) -> nn.Linear:
    probe = nn.Linear(X_train.shape[1], len(CLASS3)).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    w_t = torch.tensor(w_train, dtype=torch.float32).to(device)

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = criterion(probe(X_t), y_t)
        (losses * w_t).mean().backward()
        optimizer.step()

    probe.eval()
    return probe


def predict_probe(probe: nn.Linear, X: np.ndarray, device: torch.device) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = probe(X_t)
    return logits.argmax(dim=1).cpu().numpy()


# ── per-aid CV ───────────────────────────────────────────────────────────────

def crossval_aid(
    model,
    processor,
    device: torch.device,
    df_aid: pd.DataFrame,
    aid: str,
    model_type: str,
    n_folds: int,
    lr: float,
    epochs: int,
) -> dict:
    """
    Run n_folds CV for a single mobility aid.

    Folds are defined over unique ImageIDs (panoramas).  All crops/rows
    that belong to the same ImageID always travel together.
    """
    label_map = {c: i for i, c in enumerate(CLASS3)}  # no→0, unsure→1, yes→2
    df_aid = df_aid.copy()
    df_aid["label_int"] = df_aid["argmax_label"].map(label_map)

    unique_panos = df_aid["ImageID"].unique()
    n = len(unique_panos)

    if n < n_folds:
        print(f"  [{aid}] only {n} panoramas < {n_folds} folds; using leave-one-out.")
        n_folds = n

    # Assign a "representative" label per panorama for stratification
    pano_label = (
        df_aid.groupby("ImageID")["label_int"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    pano_label.columns = ["ImageID", "label_int"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    X_arr  = pano_label["label_int"].values   # dummy — used only for stratification
    y_arr  = pano_label["label_int"].values

    fold_metrics: list[dict] = []

    for fold_idx, (tr_pano_idx, te_pano_idx) in enumerate(skf.split(X_arr, y_arr)):
        tr_pano_ids = pano_label.iloc[tr_pano_idx]["ImageID"].values
        te_pano_ids = pano_label.iloc[te_pano_idx]["ImageID"].values

        tr_df = df_aid[df_aid["ImageID"].isin(tr_pano_ids)]
        te_df = df_aid[df_aid["ImageID"].isin(te_pano_ids)]

        if tr_df.empty or te_df.empty:
            print(f"  [{aid}] fold {fold_idx}: empty split — skipping.")
            continue

        t0 = time.perf_counter()

        X_train = get_features(model, processor, device, tr_df["path"].tolist(), model_type)
        X_test  = get_features(model, processor, device, te_df["path"].tolist(), model_type)

        y_train = tr_df["label_int"].values
        y_test  = te_df["label_int"].values
        w_train = tr_df["sample_weight"].values

        scaler  = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probe = train_probe(X_train, y_train, w_train, lr, epochs, probe_device)
        y_pred = predict_probe(probe, X_test, probe_device)

        elapsed = time.perf_counter() - t0
        m = {
            "fold":             fold_idx,
            "n_train_panos":    len(tr_pano_ids),
            "n_test_panos":     len(te_pano_ids),
            "n_train_samples":  len(tr_df),
            "n_test_samples":   len(te_df),
            "accuracy":         float(accuracy_score(y_test, y_pred)),
            "macro_f1":         float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "balanced_acc":     float(balanced_accuracy_score(y_test, y_pred)),
            "elapsed_s":        round(elapsed, 2),
        }
        fold_metrics.append(m)
        print(
            f"  [{aid}] fold {fold_idx+1}/{n_folds} | "
            f"test panos={len(te_pano_ids)} | "
            f"macro_f1={m['macro_f1']:.3f} | "
            f"bal_acc={m['balanced_acc']:.3f}"
        )

    if not fold_metrics:
        return {}

    macro_f1s    = [m["macro_f1"]    for m in fold_metrics]
    bal_accs     = [m["balanced_acc"] for m in fold_metrics]
    accuracies   = [m["accuracy"]     for m in fold_metrics]

    summary = {
        "aid":            aid,
        "n_folds":        len(fold_metrics),
        "macro_f1_mean":  float(np.mean(macro_f1s)),
        "macro_f1_std":   float(np.std(macro_f1s)),
        "balanced_acc_mean": float(np.mean(bal_accs)),
        "balanced_acc_std":  float(np.std(bal_accs)),
        "accuracy_mean":  float(np.mean(accuracies)),
        "accuracy_std":   float(np.std(accuracies)),
        "folds":          fold_metrics,
    }
    return summary


# ── class distribution helper ─────────────────────────────────────────────────

def report_class_distribution(df: pd.DataFrame) -> dict:
    label_map = {"p_no": "no", "p_unsure": "unsure", "p_yes": "yes"}
    dist: dict[str, dict] = {}
    for aid in AIDS:
        sub = df[df["MobilityAid"] == aid]
        counts = sub["argmax_label"].map(label_map).value_counts().to_dict()
        total = len(sub)
        dist[aid] = {
            "total": total,
            "no":    counts.get("no", 0),
            "unsure": counts.get("unsure", 0),
            "yes":   counts.get("yes", 0),
        }
    return dist


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panorama-level K-fold CV for sidewalk accessibility models."
    )
    parser.add_argument("--tallies_json", required=True)
    parser.add_argument("--images_dir",   required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument(
        "--model_name", default="openai/clip-vit-base-patch32",
        help="HuggingFace model ID or timm model name",
    )
    parser.add_argument(
        "--model_type", default="clip",
        choices=["clip", "resnet50", "vit_base_patch16_224", "convnext_base", "dinov2"],
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--lr",     type=float, default=1e-3, help="Linear probe learning rate")
    parser.add_argument("--epochs", type=int,  default=50,   help="Linear probe training epochs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(
        lambda x: find_image_path(x, args.images_dir)
    )
    tallies = tallies[tallies["path"].notna()]

    class_dist = report_class_distribution(tallies)
    print("\nClass distribution per aid:")
    for aid, d in class_dist.items():
        print(f"  {aid:30s}  yes={d['yes']:3d}  unsure={d['unsure']:3d}  no={d['no']:3d}  total={d['total']:3d}")
    print()

    # ── load model ───────────────────────────────────────────────────────────
    if args.model_type == "clip":
        model, processor, device = load_clip_model(args.model_name)
    elif args.model_type == "dinov2":
        from transformers import AutoModel, AutoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(args.model_name).to(device).eval()
        processor = AutoProcessor.from_pretrained(args.model_name)
    else:
        model, processor, device = load_vision_model(args.model_type)

    # ── run CV per aid ───────────────────────────────────────────────────────
    config = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "n_folds":    args.n_folds,
        "lr":         args.lr,
        "epochs":     args.epochs,
        "random_state": RANDOM_STATE,
    }

    all_results: dict = {
        "config":           config,
        "class_distribution": class_dist,
        "aids":             {},
    }

    for aid in AIDS:
        df_aid = tallies[tallies["MobilityAid"] == aid]
        if df_aid.empty:
            print(f"[{aid}] no data — skipping.")
            continue
        print(f"\n── {aid} ({len(df_aid)} rows, {df_aid['ImageID'].nunique()} panoramas) ──")
        summary = crossval_aid(
            model, processor, device,
            df_aid, aid, args.model_type,
            args.n_folds, args.lr, args.epochs,
        )
        if summary:
            all_results["aids"][aid] = summary
            print(
                f"  → macro_f1 = {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}"
                f"  |  bal_acc = {summary['balanced_acc_mean']:.3f} ± {summary['balanced_acc_std']:.3f}"
            )

    # ── aggregate across aids ────────────────────────────────────────────────
    if all_results["aids"]:
        f1_means = [v["macro_f1_mean"] for v in all_results["aids"].values()]
        all_results["overall"] = {
            "macro_f1_mean_across_aids": float(np.mean(f1_means)),
            "macro_f1_std_across_aids":  float(np.std(f1_means)),
        }

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = output_dir / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── print summary table ───────────────────────────────────────────────────
    print("\n── Summary table ──────────────────────────────────────────────────────")
    print(f"{'Aid':<30} {'macro_f1':>10} {'±':>3} {'bal_acc':>10} {'±':>3}")
    print("-" * 60)
    for aid, v in all_results["aids"].items():
        print(
            f"{aid:<30} {v['macro_f1_mean']:>10.3f} {'±':>3} {v['macro_f1_std']:>4.3f}"
            f"   {v['balanced_acc_mean']:>10.3f} {'±':>3} {v['balanced_acc_std']:>4.3f}"
        )
    if "overall" in all_results:
        ov = all_results["overall"]
        print(
            f"\n{'Overall (mean across aids)':<30} "
            f"{ov['macro_f1_mean_across_aids']:>10.3f} ± {ov['macro_f1_std_across_aids']:>4.3f}"
        )
    print("────────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
