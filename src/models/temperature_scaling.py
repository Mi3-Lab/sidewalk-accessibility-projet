#!/usr/bin/env python3
"""
Temperature-scaled Hard-CE baseline for sidewalk accessibility.

For each encoder × aid × fold:
  1. Split train fold 80/20 at panorama level → train2 / calibration.
  2. Train Hard-CE probe on train2.
  3. Get raw logits on calibration split; optimise temperature T to minimise NLL.
  4. Apply T-scaled probabilities to test fold; report calibrated Brier soft.

Answers reviewer question: "Have you tried post-hoc calibration of Hard-CE?"

Usage:
    python src/models/temperature_scaling.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --encoder      dinov2-large \
        --output_dir   results/cv/hard_ce_ts/dinov2-large

Run all 8 encoders:
    for enc in clip-vit-b32 clip-vit-b16 clip-vit-l14 \
               dinov2-base dinov2-large \
               siglip2-base siglip2-so400m vit-b16-sup; do
        python src/models/temperature_scaling.py --encoder $enc \
            --tallies_json data/processed/tallies_firebase.json \
            --images_dir   data/images/sidewalk-images \
            --output_dir   results/cv/hard_ce_ts/$enc
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
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from crossval import _features_for, build_feature_cache
from sklearn.preprocessing import StandardScaler

from train import (
    AIDS,
    CLASS3,
    ENCODERS,
    LABEL_MAP,
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
)
from crossval import brier_score_soft, brier_score_hard, expected_calibration_error


# ── Temperature scaling ───────────────────────────────────────────────────────

def get_logits(probe: nn.Linear, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return raw logits (N, 3) without any activation."""
    with torch.no_grad():
        logits = probe(torch.tensor(X, dtype=torch.float32).to(device))
    return logits.cpu().numpy()


def calibrated_probs(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T and return softmax probabilities (N, 3)."""
    scaled = logits / T
    # Numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def nll_loss(T: float, logits: np.ndarray, y_true: np.ndarray) -> float:
    """Negative log-likelihood at temperature T on calibration data."""
    probs = calibrated_probs(logits, max(T, 1e-3))
    # Clip for numerical safety
    p_true = probs[np.arange(len(y_true)), y_true].clip(1e-9, 1.0)
    return float(-np.mean(np.log(p_true)))


def tune_temperature(logits_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Find optimal T in [0.05, 20.0] minimising NLL on calibration set."""
    result = minimize_scalar(
        nll_loss,
        args=(logits_cal, y_cal),
        bounds=(0.05, 20.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


# ── Per-fold helper ───────────────────────────────────────────────────────────

CAL_SPLIT = 0.20   # fraction of train panoramas used for temperature calibration


def crossval_ts_aid(
    model,
    processor,
    device: torch.device,
    df_aid: pd.DataFrame,
    aid: str,
    enc_type: str,
    n_folds: int,
    seed: int = RANDOM_STATE,
    feat_cache: dict | None = None,
) -> dict:
    """5-fold CV with temperature-scaled Hard-CE for one aid.

    seed controls the fold partition, the train2/calibration split and the probe
    init, so repeating over seeds measures how stable the temperature-scaling
    result actually is. feat_cache holds frozen-encoder features, which never
    change across folds or seeds."""
    df_aid = df_aid.copy()
    df_aid["label_int"] = df_aid["argmax_label"].map(LABEL_MAP)

    unique_panos = df_aid["ImageID"].unique()
    n = len(unique_panos)
    if n < n_folds:
        n_folds = n

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

    for fold_idx, (tr_pano_idx, te_pano_idx) in enumerate(skf.split(y_arr, y_arr)):
        tr_pano_ids = pano_label.iloc[tr_pano_idx]["ImageID"].values
        te_pano_ids = pano_label.iloc[te_pano_idx]["ImageID"].values

        te_df = df_aid[df_aid["ImageID"].isin(te_pano_ids)]
        if te_df.empty:
            continue

        # Split train panoramas into train2 / calibration at panorama level
        rng = np.random.default_rng(seed + fold_idx)
        n_cal_panos = max(1, int(len(tr_pano_ids) * CAL_SPLIT))
        cal_pano_ids = set(rng.choice(tr_pano_ids, size=n_cal_panos, replace=False))
        tr2_pano_ids = set(tr_pano_ids) - cal_pano_ids

        tr2_df = df_aid[df_aid["ImageID"].isin(tr2_pano_ids)]
        cal_df = df_aid[df_aid["ImageID"].isin(cal_pano_ids)]

        if tr2_df.empty or cal_df.empty:
            continue

        set_seed(seed + fold_idx)
        t0 = time.perf_counter()

        # Features come from the cache: the encoder is frozen, so they are
        # identical across folds and seeds.
        X_tr2 = _features_for(tr2_df["path"].tolist(), feat_cache)
        X_cal = _features_for(cal_df["path"].tolist(), feat_cache)
        X_te  = _features_for(te_df["path"].tolist(),  feat_cache)

        scaler = StandardScaler(with_mean=False)
        X_tr2 = scaler.fit_transform(X_tr2)
        X_cal = scaler.transform(X_cal)
        X_te  = scaler.transform(X_te)

        y_tr2  = tr2_df["label_int"].values
        w_tr2  = tr2_df["sample_weight"].values
        y_cal  = cal_df["label_int"].values
        y_te   = te_df["label_int"].values
        y_soft_te = te_df[SOFT_COLS].values.astype(np.float32)

        # Train Hard-CE probe on train2
        probe = train_probe_hard(X_tr2, y_tr2, w_tr2, X_tr2.shape[1], probe_device)

        # Tune temperature on calibration split
        logits_cal = get_logits(probe, X_cal, probe_device)
        T_opt = tune_temperature(logits_cal, y_cal)

        # Evaluate on test fold: uncalibrated and calibrated
        logits_te    = get_logits(probe, X_te, probe_device)
        probs_raw    = calibrated_probs(logits_te, 1.0)    # T=1 → original Hard-CE
        probs_scaled = calibrated_probs(logits_te, T_opt)  # temperature-scaled

        elapsed = time.perf_counter() - t0
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

        m = {
            "fold":              fold_idx,
            "temperature":       round(T_opt, 4),
            "n_cal_panos":       len(cal_pano_ids),
            "n_tr2_panos":       len(tr2_pano_ids),
            "n_test_panos":      len(te_pano_ids),
            "n_test_samples":    len(te_df),
            # uncalibrated
            "brier_soft_raw":    brier_score_soft(probs_raw,    y_soft_te),
            "brier_hard_raw":    brier_score_hard(probs_raw,    y_te),
            "ece_raw":           expected_calibration_error(probs_raw,    y_te),
            # temperature-scaled
            "brier_soft_ts":     brier_score_soft(probs_scaled, y_soft_te),
            "brier_hard_ts":     brier_score_hard(probs_scaled, y_te),
            "ece_ts":            expected_calibration_error(probs_scaled, y_te),
            "elapsed_s":         round(elapsed, 2),
        }
        fold_metrics.append(m)
        print(
            f"  [{aid}] fold {fold_idx+1}/{n_folds} | "
            f"T={T_opt:.3f} | "
            f"brier_soft_raw={m['brier_soft_raw']:.3f} "
            f"→ ts={m['brier_soft_ts']:.3f}"
        )

    if not fold_metrics:
        return {}

    def _mean(key): return float(np.mean([m[key] for m in fold_metrics]))
    def _std(key):  return float(np.std( [m[key] for m in fold_metrics]))

    return {
        "aid":                  aid,
        "n_folds":              len(fold_metrics),
        "temperature_mean":     _mean("temperature"),
        "temperature_std":      _std("temperature"),
        # uncalibrated Hard-CE (sanity check — should match hard_ce CV results)
        "brier_soft_raw_mean":  _mean("brier_soft_raw"),
        "brier_soft_raw_std":   _std("brier_soft_raw"),
        "brier_hard_raw_mean":  _mean("brier_hard_raw"),
        "brier_hard_raw_std":   _std("brier_hard_raw"),
        "ece_raw_mean":         _mean("ece_raw"),
        "ece_raw_std":          _std("ece_raw"),
        # temperature-scaled Hard-CE
        "brier_soft_ts_mean":   _mean("brier_soft_ts"),
        "brier_soft_ts_std":    _std("brier_soft_ts"),
        "brier_hard_ts_mean":   _mean("brier_hard_ts"),
        "brier_hard_ts_std":    _std("brier_hard_ts"),
        "ece_ts_mean":          _mean("ece_ts"),
        "ece_ts_std":           _std("ece_ts"),
        "folds":                fold_metrics,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temperature-scaled Hard-CE CV for sidewalk accessibility."
    )
    parser.add_argument("--tallies_json", required=True)
    parser.add_argument("--images_dir",   required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument(
        "--encoder",
        default="dinov2-large",
        choices=list(ENCODERS),
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--n_seeds", type=int, default=1,
        help="Number of consecutive seeds. >1 reports mean±std across independent "
             "fold partitions, which is what tells us whether the temperature-scaling "
             "result is stable or a single-draw artefact.",
    )
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.n_seeds)]
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(
        lambda x: find_image_path(x, args.images_dir)
    )
    tallies = tallies[tallies["path"].notna()]

    print(f"\nEncoder: {args.encoder}  (temperature-scaled Hard-CE)")
    print(f"Images found: {len(tallies)} rows\n")

    model, processor, device, enc_type = load_encoder(args.encoder)

    feat_cache = build_feature_cache(
        model, processor, device, tallies["path"].tolist(), enc_type
    )
    print(f"Feature cache: {len(feat_cache)} unique images\n")

    config = {
        "encoder":      args.encoder,
        "hf_id":        ENCODERS[args.encoder][0],
        "enc_type":     enc_type,
        "loss_type":    "hard_ce_ts",
        "n_folds":      args.n_folds,
        "cal_split":    CAL_SPLIT,
        "lr":           PROBE_LR,
        "epochs":       PROBE_EPOCHS,
        "weight_decay": PROBE_WD,
        "random_state": args.seed,
        "seeds":        seeds,
    }

    per_seed: dict[int, dict] = {}

    for seed in seeds:
        if len(seeds) > 1:
            print(f"\n{'═' * 60}\n seed {seed}\n{'═' * 60}")

        seed_results: dict = {"config": {**config, "seed": seed}, "aids": {}}

        for aid in AIDS:
            df_aid = tallies[tallies["MobilityAid"] == aid]
            if df_aid.empty:
                print(f"[{aid}] no data — skipping.")
                continue
            print(
                f"\n── {aid} "
                f"({len(df_aid)} rows, {df_aid['ImageID'].nunique()} panoramas) ──"
            )
            summary = crossval_ts_aid(
                model, processor, device,
                df_aid, aid, enc_type, args.n_folds,
                seed=seed, feat_cache=feat_cache,
            )
            if summary:
                seed_results["aids"][aid] = summary
                print(
                    f"  → brier_soft_raw = {summary['brier_soft_raw_mean']:.3f}"
                    f" ± {summary['brier_soft_raw_std']:.3f}"
                    f"  |  brier_soft_ts = {summary['brier_soft_ts_mean']:.3f}"
                    f" ± {summary['brier_soft_ts_std']:.3f}"
                    f"  |  T = {summary['temperature_mean']:.3f}"
                )

        per_seed[seed] = seed_results

    all_results = per_seed[seeds[0]]
    out_path = output_dir / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    if len(seeds) > 1:
        import numpy as _np
        keys = ["brier_soft_raw", "brier_soft_ts", "temperature"]
        stability: dict = {"seeds": seeds, "n_seeds": len(seeds), "per_aid": {}}
        for aid in AIDS:
            vals = {k: [per_seed[s]["aids"][aid][f"{k}_mean"]
                        for s in seeds if aid in per_seed[s]["aids"]] for k in keys}
            if not vals["brier_soft_raw"]:
                continue
            stability["per_aid"][aid] = {
                **{f"{k}_mean_over_seeds": float(_np.mean(vals[k])) for k in keys},
                **{f"{k}_std_over_seeds":  float(_np.std(vals[k]))  for k in keys},
                **{f"{k}_per_seed":        vals[k]                  for k in keys},
            }
        agg = {k: [v for aid in stability["per_aid"]
                   for v in [stability["per_aid"][aid][f"{k}_mean_over_seeds"]]] for k in keys}
        stability["overall"] = {
            **{f"{k}_mean_over_seeds": float(_np.mean(agg[k])) for k in keys},
            **{f"{k}_std_over_seeds":  float(_np.std(agg[k]))  for k in keys},
        }
        raw = stability["overall"]["brier_soft_raw_mean_over_seeds"]
        ts  = stability["overall"]["brier_soft_ts_mean_over_seeds"]
        stability["overall"]["ts_brier_reduction_pct"] = 100.0 * (raw - ts) / raw if raw else 0.0

        ms_path = output_dir / "cv_results_multiseed.json"
        with open(ms_path, "w") as f:
            json.dump(stability, f, indent=2)
        print(f"Across-seed stability saved → {ms_path}")
        print(f"\n── Temperature scaling over {len(seeds)} seeds ──")
        print(f"  Brier raw Hard-CE = {raw:.4f}")
        print(f"  Brier T-scaled    = {ts:.4f}")
        print(f"  reduction         = {stability['overall']['ts_brier_reduction_pct']:.1f}%"
              f"   [paper claims ~34%]")


if __name__ == "__main__":
    main()
