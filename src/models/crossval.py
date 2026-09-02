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



# ── Frozen-encoder feature cache ──────────────────────────────────────────────

def load_feature_npz(npz_path: str, key: str) -> dict:
    """Load precomputed features (see masked_features.py) as {path: vector}.

    Lets the same folds be run over whole-image and sidewalk-masked features
    without re-encoding, so the only thing that differs is what the probe sees.
    """
    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise SystemExit(f"{npz_path} has no array '{key}' (has {list(data.keys())})")
    vecs = data[key]
    cache = {}
    for k, v in zip(data["keys"], vecs):
        cache[str(Path(str(k)).resolve())] = v
    return cache


def build_feature_cache(
    model, processor, device: torch.device, paths: list, enc_type: str
) -> dict:
    """Encode each unique image once and return {path: feature vector}.

    The encoder is frozen, so a given image yields the same feature vector in
    every fold, seed and mobility aid. Encoding once instead of per fold is what
    makes repeated-seed CV cheap enough to run.
    """
    unique_paths = sorted({str(p) for p in paths})
    feats = extract_encoder_features(
        model, processor, device, unique_paths, enc_type
    )
    return {p: feats[i] for i, p in enumerate(unique_paths)}


def _features_for(paths: list, feat_cache: dict) -> np.ndarray:
    """Look up cached features for paths, preserving their order.

    Falls back to the resolved absolute path so a cache built elsewhere (e.g. a
    precomputed .npz) matches regardless of how the path was spelled.
    """
    out = []
    for p in paths:
        key = str(p)
        if key not in feat_cache:
            key = str(Path(str(p)).resolve())
        out.append(feat_cache[key])
    return np.stack(out)


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


def entropy_correlation(y_prob: np.ndarray, y_soft: np.ndarray) -> float:
    """Pearson r between the model's prediction entropy and the entropy of the
    human vote distribution, across test items.

    Brier asks whether the predicted distribution is right. This asks something
    different and arguably more operational: does the model know *which scenes
    people disagree about*? A planner that can flag contested edges is useful
    even when its point estimate is off, and a Hard-CE model trained on argmax
    labels has no way to represent that structure at all.

    It is the metric the recent soft-label literature leads with, and this
    dataset is unusually well placed to measure it: with 141-240 votes per
    (image, aid) the human entropy estimate is far past the saturation point
    where the metric is reported to converge (N~20-50).
    """
    eps = 1e-12
    h_model = -(np.clip(y_prob, eps, 1) * np.log(np.clip(y_prob, eps, 1))).sum(axis=1)
    h_human = -(np.clip(y_soft, eps, 1) * np.log(np.clip(y_soft, eps, 1))).sum(axis=1)
    if h_model.std() < eps or h_human.std() < eps:
        return float("nan")     # no spread to correlate (e.g. a degenerate fold)
    return float(np.corrcoef(h_model, h_human)[0, 1])


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
    seed: int = RANDOM_STATE,
    feat_cache: dict | None = None,
    rank: int = 0,
    weight_decay: float = PROBE_WD,
) -> dict:
    """
    Run n_folds CV for a single mobility aid.
    Folds are defined over unique ImageIDs (panoramas).
    All crops/rows from the same ImageID travel together.

    seed controls both the fold partition and the probe initialisation, so
    repeating the run with different seeds measures partition sensitivity —
    the stability check a single seed=42 run cannot provide.

    feat_cache maps image path -> encoder feature vector. The encoder is frozen,
    so features are identical across folds and seeds; passing a prebuilt cache
    avoids re-encoding the same images once per fold.
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

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
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

        set_seed(seed + fold_idx)   # unique but deterministic per (seed, fold)
        t0 = time.perf_counter()

        X_train = _features_for(tr_df["path"].tolist(), feat_cache)
        X_test  = _features_for(te_df["path"].tolist(), feat_cache)

        y_test  = te_df["label_int"].values
        w_train = tr_df["sample_weight"].values

        scaler  = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Probe init uses `seed` itself, not seed+fold_idx: this reproduces the
        # original single-seed behaviour exactly at seed=42 while still varying
        # the initialisation across seeds.
        if loss_type == "soft_kl":
            y_soft = tr_df[SOFT_COLS].values.astype(np.float32)
            probe  = train_probe_soft(
                X_train, y_soft, w_train, X_train.shape[1], probe_device,
                seed=seed, rank=rank, weight_decay=weight_decay,
            )
        else:
            y_train = tr_df["label_int"].values
            probe   = train_probe_hard(
                X_train, y_train, w_train, X_train.shape[1], probe_device,
                seed=seed, rank=rank, weight_decay=weight_decay,
            )

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
            "entropy_corr":    entropy_correlation(y_prob, y_soft_test),
            "elapsed_s":       round(elapsed, 2),
        }
        fold_metrics.append(m)
        print(
            f"  [{aid}] fold {fold_idx+1}/{n_folds} | "
            f"test_panos={len(te_pano_ids)} | "
            f"macro_f1={m['macro_f1']:.3f} | "
            f"bal_acc={m['balanced_acc']:.3f} | "
            f"brier_soft={m['brier_soft']:.3f} | "
            f"ece={m['ece']:.3f} | "
            f"H-corr={m['entropy_corr']:.3f}"
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

    def _mean(key): return float(np.nanmean([m[key] for m in fold_metrics]))
    def _std(key):  return float(np.nanstd( [m[key] for m in fold_metrics]))

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
        "entropy_corr_mean":  _mean("entropy_corr"),
        "entropy_corr_std":   _std("entropy_corr"),
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
    parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE,
        help="First seed for the fold partition and probe init.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=PROBE_WD,
        help="L2 on the probe. The pipeline default of 1e-4 is close to no "
             "regularisation for a d->3 map fit to 52 points; sweeps put the "
             "Soft-KL optimum near 3e-1 and the Hard-CE optimum near 1e1.",
    )
    parser.add_argument(
        "--features_npz", default="",
        help="Use precomputed features from masked_features.py instead of "
             "encoding here. Lets whole-image and sidewalk-masked features be "
             "compared on identical folds.",
    )
    parser.add_argument(
        "--feature_key", default="masked", choices=["masked", "whole"],
        help="Which array to read from --features_npz.",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Low-rank bottleneck for the probe (0 = the original full-rank d->3 "
             "map used for every published number). A rank of 16-64 cuts soft "
             "Brier ~30%% and raises macro F1; see results/coupled/.",
    )
    parser.add_argument(
        "--n_seeds", type=int, default=1,
        help="Number of consecutive seeds to run (seed, seed+1, ...). "
             ">1 reports mean±std across seeds, measuring partition sensitivity.",
    )
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.n_seeds)]
    set_seed(args.seed)

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

    # ── Encode every unique image once, reuse across folds/seeds/aids ────────
    t_cache = time.perf_counter()
    if args.features_npz:
        feat_cache = load_feature_npz(args.features_npz, args.feature_key)
        print(f"Loaded '{args.feature_key}' features from {args.features_npz}")
    else:
        feat_cache = build_feature_cache(
            model, processor, device, tallies["path"].tolist(), enc_type
        )
    print(
        f"Feature cache: {len(feat_cache)} unique images encoded "
        f"in {time.perf_counter() - t_cache:.1f}s\n"
    )

    # ── CV config ────────────────────────────────────────────────────────────
    config = {
        "encoder":      args.encoder,
        "hf_id":        ENCODERS[args.encoder][0],
        "enc_type":     enc_type,
        "loss_type":    args.loss_type,
        "n_folds":      args.n_folds,
        "lr":           PROBE_LR,
        "epochs":       PROBE_EPOCHS,
        "random_state": args.seed,
        "seeds":        seeds,
        "rank":         args.rank,
        "weight_decay": args.weight_decay,
        "features_npz": args.features_npz,
        "feature_key":  args.feature_key if args.features_npz else "encoder",
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

    # ── Run CV per seed ──────────────────────────────────────────────────────
    per_seed: dict[int, dict] = {}

    for seed in seeds:
        if len(seeds) > 1:
            print(f"\n{'═' * 70}\n seed {seed}  ({seeds.index(seed) + 1}/{len(seeds)})\n{'═' * 70}")

        seed_results: dict = {
            "config":             {**config, "seed": seed},
            "class_distribution": class_dist,
            "aids":               {},
        }

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
                seed=seed, feat_cache=feat_cache, rank=args.rank,
                weight_decay=args.weight_decay,
            )
            if summary:
                seed_results["aids"][aid] = summary
                print(
                    f"  → macro_f1 = {summary['macro_f1_mean']:.3f}"
                    f" ± {summary['macro_f1_std']:.3f}"
                    f"  |  bal_acc = {summary['balanced_acc_mean']:.3f}"
                    f" ± {summary['balanced_acc_std']:.3f}"
                )

        # Aggregate across aids for this seed
        if seed_results["aids"]:
            aids_vals = seed_results["aids"].values()
            seed_results["overall"] = {
                "macro_f1_mean_across_aids":   float(np.mean([v["macro_f1_mean"]    for v in aids_vals])),
                "macro_f1_std_across_aids":    float(np.std( [v["macro_f1_mean"]    for v in aids_vals])),
                "brier_soft_mean_across_aids": float(np.mean([v["brier_soft_mean"]  for v in aids_vals])),
                "brier_hard_mean_across_aids": float(np.mean([v["brier_hard_mean"]  for v in aids_vals])),
                "ece_mean_across_aids":        float(np.mean([v["ece_mean"]         for v in aids_vals])),
                "entropy_corr_mean_across_aids": float(np.nanmean([v["entropy_corr_mean"] for v in aids_vals])),
            }

        per_seed[seed] = seed_results

        if len(seeds) > 1:
            with open(output_dir / f"cv_results_seed{seed}.json", "w") as f:
                json.dump(seed_results, f, indent=2)

    # The first seed keeps the canonical filename so existing summarisers and
    # the numbers already reported in the paper stay reproducible.
    all_results = per_seed[seeds[0]]

    if _WANDB_AVAILABLE and wandb.run is not None:
        for aid, summary in all_results["aids"].items():
            aid_key = aid.lower().replace(" ", "_")
            wandb.summary[f"{aid_key}/macro_f1_mean"]    = round(summary["macro_f1_mean"],     4)
            wandb.summary[f"{aid_key}/bal_acc_mean"]     = round(summary["balanced_acc_mean"], 4)
            wandb.summary[f"{aid_key}/brier_soft_mean"]  = round(summary["brier_soft_mean"],   4)
            wandb.summary[f"{aid_key}/brier_hard_mean"]  = round(summary["brier_hard_mean"],   4)
            wandb.summary[f"{aid_key}/ece_mean"]         = round(summary["ece_mean"],          4)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = output_dir / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Across-seed stability (the reviewer's actual question) ───────────────
    if len(seeds) > 1:
        metrics = ["macro_f1", "brier_soft", "brier_hard", "ece", "entropy_corr"]
        stability: dict = {"seeds": seeds, "n_seeds": len(seeds), "per_aid": {}}

        for aid in AIDS:
            vals = {
                m: [per_seed[s]["aids"][aid][f"{m}_mean"]
                    for s in seeds if aid in per_seed[s]["aids"]]
                for m in metrics
            }
            if not vals["macro_f1"]:
                continue
            stability["per_aid"][aid] = {
                f"{m}_mean_over_seeds": float(np.mean(vals[m])) for m in metrics
            } | {
                f"{m}_std_over_seeds": float(np.std(vals[m])) for m in metrics
            } | {
                f"{m}_per_seed": vals[m] for m in metrics
            }

        overall = {
            m: [per_seed[s]["overall"][f"{m}_mean_across_aids"]
                for s in seeds if "overall" in per_seed[s]]
            for m in metrics
        }
        stability["overall"] = {
            f"{m}_mean_over_seeds": float(np.mean(overall[m])) for m in metrics
        } | {
            f"{m}_std_over_seeds": float(np.std(overall[m])) for m in metrics
        } | {
            f"{m}_per_seed": overall[m] for m in metrics
        }

        ms_path = output_dir / "cv_results_multiseed.json"
        with open(ms_path, "w") as f:
            json.dump(stability, f, indent=2)
        print(f"Across-seed stability saved → {ms_path}")

        print(f"\n── Stability across {len(seeds)} seeds ({seeds[0]}–{seeds[-1]}) ──")
        for m in metrics:
            print(
                f"  {m:<12} = {stability['overall'][f'{m}_mean_over_seeds']:.4f}"
                f" ± {stability['overall'][f'{m}_std_over_seeds']:.4f}"
            )

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
