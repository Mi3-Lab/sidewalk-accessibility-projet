#!/usr/bin/env python3
"""
Semi-supervised expansion of the 52-panorama soft-label set.

The training set carries dense human vote distributions but covers only 52
scenes. This script tests whether a pool of *unlabelled* Project Sidewalk
images can extend that coverage: a probe trained on the real votes assigns
soft pseudo-distributions to the pool, the unreliable ones are filtered out,
and the probe is retrained on real + pseudo rows.

Evaluation is always against held-out REAL human votes, never against the
pseudo-labels themselves — otherwise the experiment would be circular in the
same way the routing evaluation was criticised for.

Two filters gate a pseudo-label:

  entropy      the predicted distribution must be confident enough
               (normalised entropy <= --max_entropy)

  ps_consistency  the predicted argmax must not contradict the Project Sidewalk
               annotator label for that image, which is an independent human
               signal: a CurbRamp scene must not be pseudo-labelled "no", and a
               NoCurbRamp scene must not be pseudo-labelled "yes"

Usage:
    python src/models/semisup_expand.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --pool_csv     data/generalization/test_images.csv \
        --encoder      dinov2-large \
        --n_folds 5 --n_seeds 5 \
        --output_dir   results/semisup/dinov2-large
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from crossval import (
    _features_for,
    brier_score_soft,
    build_feature_cache,
    expected_calibration_error,
    predict_proba,
)
from train import (
    AIDS,
    CLASS3,
    ENCODERS,
    LABEL_MAP,
    RANDOM_STATE,
    SOFT_COLS,
    extract_encoder_features,
    find_image_path,
    load_encoder,
    set_seed,
    train_probe_soft,
)

HMAX = np.log(len(CLASS3))          # max entropy of a 3-class distribution
PS_LABEL_COL = "ps_label"


# ── Pseudo-label construction ────────────────────────────────────────────────

def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    """Row-wise entropy of a distribution, scaled to [0, 1]."""
    p = np.clip(probs, 1e-12, 1.0)
    return (-(p * np.log(p)).sum(axis=1)) / HMAX


def ps_consistent(pred_class: np.ndarray, ps_labels: np.ndarray) -> np.ndarray:
    """Keep pseudo-labels that do not contradict the Project Sidewalk label.

    CurbRamp scenes must not be called impassable, NoCurbRamp scenes must not
    be called passable. "unsure" is compatible with either.
    """
    is_curb_ramp = ps_labels == "CurbRamp"
    says_no  = pred_class == LABEL_MAP["p_no"]
    says_yes = pred_class == LABEL_MAP["p_yes"]
    return np.where(is_curb_ramp, ~says_no, ~says_yes)


def build_pseudo_rows(
    probs: np.ndarray,
    pool_df: pd.DataFrame,
    aid: str,
    max_entropy: float,
    use_ps_consistency: bool,
    max_pseudo: int,
    pseudo_weight: float,
) -> tuple[pd.DataFrame, dict]:
    """Filter pool predictions into pseudo-labelled training rows.

    Returns the surviving rows and a breakdown of how many each filter cut.
    """
    ent  = normalised_entropy(probs)
    pred = probs.argmax(axis=1)

    keep = ent <= max_entropy
    n_after_entropy = int(keep.sum())

    if use_ps_consistency and PS_LABEL_COL in pool_df.columns:
        keep &= ps_consistent(pred, pool_df[PS_LABEL_COL].values)
    n_after_ps = int(keep.sum())

    idx = np.flatnonzero(keep)
    if max_pseudo > 0 and len(idx) > max_pseudo:
        # keep the most confident ones
        idx = idx[np.argsort(ent[idx])[:max_pseudo]]

    rows = pool_df.iloc[idx].copy()
    rows["MobilityAid"] = aid
    rows[SOFT_COLS] = probs[idx]
    # same entropy weighting the real rows use, scaled down: a pseudo-label is
    # never worth as much as a real vote distribution
    rows["sample_weight"] = pseudo_weight * (1.0 - ent[idx])
    rows["label_int"] = pred[idx]

    return rows, {
        "pool_size":          len(pool_df),
        "after_entropy":      n_after_entropy,
        "after_ps_filter":    n_after_ps,
        "used":               len(idx),
    }


# ── One fold ─────────────────────────────────────────────────────────────────

def evaluate(probe, X, y_int, y_soft, device) -> dict:
    prob = predict_proba(probe, X, device)
    pred = prob.argmax(axis=1)
    return {
        "macro_f1":   float(f1_score(y_int, pred, average="macro", zero_division=0)),
        "brier_soft": brier_score_soft(prob, y_soft),
        "ece":        expected_calibration_error(prob, y_int),
    }


def run_fold(
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    aid: str,
    feat_cache: dict,
    pool_cache: dict,
    device: torch.device,
    args,
    seed: int,
) -> dict:
    """Train baseline and semi-supervised probes on one fold, score both on real votes."""
    X_train = _features_for(tr_df["path"].tolist(), feat_cache)
    X_test  = _features_for(te_df["path"].tolist(), feat_cache)
    X_pool  = _features_for(pool_df["image_path"].tolist(), pool_cache)

    scaler  = StandardScaler(with_mean=False).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    X_pool_s = scaler.transform(X_pool)

    y_train_soft = tr_df[SOFT_COLS].values.astype(np.float32)
    w_train      = tr_df["sample_weight"].values
    y_test_int   = te_df["label_int"].values
    y_test_soft  = te_df[SOFT_COLS].values.astype(np.float32)

    # 1. baseline probe: real votes only
    base = train_probe_soft(
        X_train, y_train_soft, w_train, X_train.shape[1], device, seed=seed
    )
    base_metrics = evaluate(base, X_test, y_test_int, y_test_soft, device)

    # 2. pseudo-label the pool with the baseline probe, then filter
    pool_probs = predict_proba(base, X_pool_s, device)
    pseudo_rows, filter_stats = build_pseudo_rows(
        pool_probs, pool_df, aid,
        args.max_entropy, not args.no_ps_consistency,
        args.max_pseudo, args.pseudo_weight,
    )

    if len(pseudo_rows) == 0:
        return {"baseline": base_metrics, "semisup": base_metrics,
                "filter": filter_stats, "note": "no pseudo-labels survived filtering"}

    # 3. retrain on real train + surviving pseudo rows
    X_pseudo = _features_for(pseudo_rows["image_path"].tolist(), pool_cache)
    X_pseudo = scaler.transform(X_pseudo)

    X_aug = np.vstack([X_train, X_pseudo])
    y_aug = np.vstack([y_train_soft, pseudo_rows[SOFT_COLS].values.astype(np.float32)])
    w_aug = np.concatenate([w_train, pseudo_rows["sample_weight"].values])

    semi = train_probe_soft(X_aug, y_aug, w_aug, X_aug.shape[1], device, seed=seed)
    semi_metrics = evaluate(semi, X_test, y_test_int, y_test_soft, device)

    return {"baseline": base_metrics, "semisup": semi_metrics, "filter": filter_stats}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tallies_json", required=True)
    p.add_argument("--images_dir",   required=True)
    p.add_argument("--pool_csv",     required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--encoder", default="dinov2-large", choices=list(ENCODERS))
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=RANDOM_STATE)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--max_entropy",  type=float, default=0.6,
                   help="Max normalised entropy for a pseudo-label to be kept.")
    p.add_argument("--no_ps_consistency", action="store_true",
                   help="Disable the Project Sidewalk label agreement filter.")
    p.add_argument("--max_pseudo",    type=int,   default=0,
                   help="Cap on pseudo rows per aid per fold (0 = no cap).")
    p.add_argument("--pseudo_weight", type=float, default=0.3,
                   help="Weight multiplier applied to pseudo rows vs real votes.")
    p.add_argument("--pool_cities", default="",
                   help="Comma-separated city filter for the pool (default: all).")
    args = p.parse_args()

    seeds = [args.seed + i for i in range(args.n_seeds)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Real labelled data ───────────────────────────────────────────────────
    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(lambda x: find_image_path(x, args.images_dir))
    tallies = tallies[tallies["path"].notna()].copy()
    tallies["label_int"] = tallies["argmax_label"].map(LABEL_MAP)

    # ── Unlabelled pool ──────────────────────────────────────────────────────
    pool = pd.read_csv(args.pool_csv)
    if args.pool_cities:
        wanted = {c.strip() for c in args.pool_cities.split(",")}
        pool = pool[pool["city"].isin(wanted)]
    pool = pool[pool["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    print(f"Encoder      : {args.encoder}")
    print(f"Real rows    : {len(tallies)} ({tallies['ImageID'].nunique()} panoramas)")
    print(f"Pool images  : {len(pool)}")
    print(f"Seeds        : {seeds[0]}..{seeds[-1]}   Folds: {args.n_folds}")
    print(f"Filters      : max_entropy={args.max_entropy}"
          f"  ps_consistency={not args.no_ps_consistency}"
          f"  pseudo_weight={args.pseudo_weight}\n")

    # ── Encode once ──────────────────────────────────────────────────────────
    model, processor, enc_device, enc_type = load_encoder(args.encoder)
    t0 = time.perf_counter()
    feat_cache = build_feature_cache(model, processor, enc_device,
                                     tallies["path"].tolist(), enc_type)
    pool_cache = build_feature_cache(model, processor, enc_device,
                                     pool["image_path"].tolist(), enc_type)
    print(f"Encoded {len(feat_cache)} labelled + {len(pool_cache)} pool images "
          f"in {time.perf_counter() - t0:.1f}s\n")

    # ── Experiment ───────────────────────────────────────────────────────────
    records: list[dict] = []

    for seed in seeds:
        set_seed(seed)
        for aid in AIDS:
            df_aid = tallies[tallies["MobilityAid"] == aid]
            if df_aid.empty:
                continue

            pano_label = (df_aid.groupby("ImageID")["label_int"]
                          .agg(lambda x: x.mode().iloc[0]).reset_index())
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
            y_arr = pano_label["label_int"].values

            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(y_arr, y_arr)):
                tr_ids = pano_label.iloc[tr_idx]["ImageID"].values
                te_ids = pano_label.iloc[te_idx]["ImageID"].values
                tr_df = df_aid[df_aid["ImageID"].isin(tr_ids)]
                te_df = df_aid[df_aid["ImageID"].isin(te_ids)]
                if tr_df.empty or te_df.empty:
                    continue

                res = run_fold(tr_df, te_df, pool, aid,
                               feat_cache, pool_cache, device, args, seed)
                records.append({"seed": seed, "aid": aid, "fold": fold_idx, **res})

            done = [r for r in records if r["seed"] == seed and r["aid"] == aid]
            if done:
                d_brier = np.mean([r["semisup"]["brier_soft"] - r["baseline"]["brier_soft"]
                                   for r in done])
                d_f1 = np.mean([r["semisup"]["macro_f1"] - r["baseline"]["macro_f1"]
                                for r in done])
                used = np.mean([r["filter"]["used"] for r in done])
                print(f"  seed {seed} | {aid:<22} "
                      f"Δbrier={d_brier:+.4f}  Δf1={d_f1:+.4f}  pseudo_used={used:.0f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    def agg(key: str, metric: str) -> tuple[float, float]:
        vals = [r[key][metric] for r in records]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "config": vars(args) | {"seeds": seeds, "n_records": len(records)},
        "pool":   {"n_images": len(pool),
                   "cities": pool["city"].value_counts().to_dict()},
        "overall": {},
        "per_aid": {},
    }

    for metric in ("macro_f1", "brier_soft", "ece"):
        b_m, b_s = agg("baseline", metric)
        s_m, s_s = agg("semisup", metric)
        summary["overall"][metric] = {
            "baseline_mean": b_m, "baseline_std": b_s,
            "semisup_mean":  s_m, "semisup_std":  s_s,
            "delta":         s_m - b_m,
        }

    for aid in AIDS:
        rs = [r for r in records if r["aid"] == aid]
        if not rs:
            continue
        summary["per_aid"][aid] = {}
        for metric in ("macro_f1", "brier_soft", "ece"):
            b = float(np.mean([r["baseline"][metric] for r in rs]))
            s = float(np.mean([r["semisup"][metric]  for r in rs]))
            summary["per_aid"][aid][metric] = {
                "baseline": b, "semisup": s, "delta": s - b,
            }
        summary["per_aid"][aid]["pseudo_used_mean"] = float(
            np.mean([r["filter"]["used"] for r in rs])
        )

    with open(out_dir / "semisup_results.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print("\n── Semi-supervised expansion, scored on held-out REAL votes ──")
    print(f"{'metric':<12}{'baseline':>12}{'+pseudo':>12}{'delta':>12}")
    print("-" * 48)
    for metric, v in summary["overall"].items():
        arrow = "better" if (
            (metric in ("brier_soft", "ece") and v["delta"] < 0)
            or (metric == "macro_f1" and v["delta"] > 0)
        ) else "worse"
        print(f"{metric:<12}{v['baseline_mean']:>12.4f}{v['semisup_mean']:>12.4f}"
              f"{v['delta']:>+12.4f}  {arrow}")
    print(f"\nSaved → {out_dir / 'semisup_results.json'}")


if __name__ == "__main__":
    main()
