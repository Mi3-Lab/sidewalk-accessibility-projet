#!/usr/bin/env python3
"""
Choose which unlabelled images are worth spending a real human vote on.

The dataset's binding constraint is not image availability, since Project Sidewalk
has millions. It is that each new soft label costs a mobility-aid user several
minutes of unpaid attention. So the question is not "how do we get more votes" but
"which handful of images buys the most per vote".

Three acquisition strategies, each answering a different objection:

  coverage   greedy k-centre (core-set) over frozen-encoder feature space, seeded
             with the 52 already-labelled scenes. Directly targets the reviewers'
             objection that 52 scenes cover too little of the visual input space:
             each pick is the pool image furthest from everything labelled so far.

  entropy    highest predictive entropy under the current probe, averaged over
             aids. Targets scenes the model finds ambiguous.

  aid_spread largest disagreement *between aids* (std of p_yes across the five
             probes). Targets scenes that discriminate mobility aids from each
             other, which is what the per-aid claim rests on.

  hybrid     coverage selection restricted to the most uncertain part of the pool.

Outputs a ranked CSV and a trials.json fragment in the schema the study app
already consumes, so a shortlist can go straight in front of participants.

Usage:
    python src/models/active_select.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --pool_csv     data/generalization/test_images.csv \
        --encoder      dinov2-large \
        --strategy     coverage \
        --n_select     20 \
        --output_dir   results/active_selection
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from crossval import _features_for, build_feature_cache, predict_proba
from train import (
    AIDS,
    CLASS3,
    ENCODERS,
    LABEL_MAP,
    RANDOM_STATE,
    SOFT_COLS,
    find_image_path,
    load_encoder,
    set_seed,
    train_probe_soft,
)

HMAX = np.log(len(CLASS3))
AID_KEY = {a: a.lower().replace(" ", "_") for a in AIDS}


# ── Acquisition strategies ───────────────────────────────────────────────────

def greedy_kcenter(
    pool_feats: np.ndarray, labelled_feats: np.ndarray, n_select: int
) -> tuple[list[int], list[float]]:
    """Greedy k-centre: repeatedly take the pool point furthest from everything
    already covered, starting from the labelled set.

    Returns the chosen pool indices and, for each, the distance it closed —
    that distance is the coverage gap the new annotation would fill.
    """
    # distance from every pool point to the nearest labelled point
    d = np.linalg.norm(
        pool_feats[:, None, :] - labelled_feats[None, :, :], axis=2
    ).min(axis=1)

    chosen: list[int] = []
    gaps:   list[float] = []
    for _ in range(min(n_select, len(pool_feats))):
        i = int(np.argmax(d))
        chosen.append(i)
        gaps.append(float(d[i]))
        # everything now also has the freshly chosen point as a potential centre
        d = np.minimum(d, np.linalg.norm(pool_feats - pool_feats[i], axis=1))
    return chosen, gaps


def per_aid_predictions(
    tallies: pd.DataFrame,
    feat_cache: dict,
    pool_paths: list,
    pool_cache: dict,
    device: torch.device,
    seed: int,
) -> dict:
    """Train one probe per aid on all labelled data, predict the whole pool."""
    from sklearn.preprocessing import StandardScaler

    X_pool_raw = _features_for(pool_paths, pool_cache)
    preds: dict[str, np.ndarray] = {}

    for aid in AIDS:
        df_aid = tallies[tallies["MobilityAid"] == aid]
        if df_aid.empty:
            continue
        X = _features_for(df_aid["path"].tolist(), feat_cache)
        scaler = StandardScaler(with_mean=False).fit(X)
        probe = train_probe_soft(
            scaler.transform(X),
            df_aid[SOFT_COLS].values.astype(np.float32),
            df_aid["sample_weight"].values,
            X.shape[1], device, seed=seed,
        )
        preds[aid] = predict_proba(probe, scaler.transform(X_pool_raw), device)

    return preds


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tallies_json", required=True)
    p.add_argument("--images_dir",   required=True)
    p.add_argument("--pool_csv",     required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--encoder", default="dinov2-large", choices=list(ENCODERS))
    p.add_argument("--strategy", default="coverage",
                   choices=["coverage", "entropy", "aid_spread", "hybrid"])
    p.add_argument("--n_select", type=int, default=20,
                   help="How many images to shortlist for real annotation.")
    p.add_argument("--hybrid_frac", type=float, default=0.3,
                   help="For --strategy hybrid: fraction of the pool, most "
                        "uncertain first, that coverage selection runs over.")
    p.add_argument("--pool_cities", default="",
                   help="Comma-separated city filter (default: all).")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ─────────────────────────────────────────────────────────────────
    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(lambda x: find_image_path(x, args.images_dir))
    tallies = tallies[tallies["path"].notna()].copy()

    pool = pd.read_csv(args.pool_csv)
    if args.pool_cities:
        wanted = {c.strip() for c in args.pool_cities.split(",")}
        pool = pool[pool["city"].isin(wanted)]
    pool = pool[pool["image_path"].apply(lambda q: Path(q).exists())].reset_index(drop=True)

    print(f"Encoder   : {args.encoder}")
    print(f"Labelled  : {tallies['ImageID'].nunique()} scenes")
    print(f"Pool      : {len(pool)} images")
    print(f"Strategy  : {args.strategy}   selecting {args.n_select}\n")

    # ── Encode ───────────────────────────────────────────────────────────────
    model, processor, enc_device, enc_type = load_encoder(args.encoder)
    feat_cache = build_feature_cache(model, processor, enc_device,
                                     tallies["path"].tolist(), enc_type)
    pool_cache = build_feature_cache(model, processor, enc_device,
                                     pool["image_path"].tolist(), enc_type)

    labelled_feats = np.stack([feat_cache[k] for k in sorted(feat_cache)])
    pool_feats     = _features_for(pool["image_path"].tolist(), pool_cache)

    # ── Model predictions (needed by every strategy except pure coverage) ────
    preds = per_aid_predictions(
        tallies, feat_cache, pool["image_path"].tolist(), pool_cache, device, args.seed
    )
    p_yes = np.stack([preds[a][:, LABEL_MAP["p_yes"]] for a in AIDS if a in preds], axis=1)
    mean_probs = np.mean([preds[a] for a in preds], axis=0)
    entropy = (-(np.clip(mean_probs, 1e-12, 1) * np.log(np.clip(mean_probs, 1e-12, 1)))
               .sum(axis=1)) / HMAX
    aid_spread = p_yes.std(axis=1)

    # ── Select ───────────────────────────────────────────────────────────────
    if args.strategy == "coverage":
        idx, gaps = greedy_kcenter(pool_feats, labelled_feats, args.n_select)
        score = gaps
    elif args.strategy == "entropy":
        idx = list(np.argsort(-entropy)[: args.n_select])
        score = [float(entropy[i]) for i in idx]
    elif args.strategy == "aid_spread":
        idx = list(np.argsort(-aid_spread)[: args.n_select])
        score = [float(aid_spread[i]) for i in idx]
    else:  # hybrid
        k = max(args.n_select, int(len(pool) * args.hybrid_frac))
        cand = np.argsort(-entropy)[:k]
        sub_idx, gaps = greedy_kcenter(pool_feats[cand], labelled_feats, args.n_select)
        idx = [int(cand[i]) for i in sub_idx]
        score = gaps

    sel = pool.iloc[idx].copy()
    sel["rank"]        = range(1, len(sel) + 1)
    sel["score"]       = score
    sel["entropy"]     = entropy[idx]
    sel["aid_spread"]  = aid_spread[idx]
    for j, aid in enumerate([a for a in AIDS if a in preds]):
        sel[f"p_yes_{AID_KEY[aid]}"] = p_yes[idx, j]

    csv_path = out_dir / f"selection_{args.strategy}.csv"
    sel.to_csv(csv_path, index=False)

    # ── trials.json fragment for the study app ───────────────────────────────
    trials = []
    for _, r in sel.iterrows():
        city = str(r["city"]).replace("_", " ").title()
        trials.append({
            "id":  r["image_id"],
            "src": f"assets/images/{Path(r['image_path']).name}",
            "alt": f"{city} sidewalk scene.",
            "caption": f"{city} sidewalk. Judge only what you can see in this scene.",
            "model_p_yes": round(float(np.mean([r[f"p_yes_{AID_KEY[a]}"]
                                                for a in AIDS if a in preds])), 3),
            "ps_class": "yes" if r.get("ps_label") == "CurbRamp" else "no",
            "per_aid_p_yes": {
                AID_KEY[a]: round(float(r[f"p_yes_{AID_KEY[a]}"]), 3)
                for a in AIDS if a in preds
            },
            "selection": {"strategy": args.strategy,
                          "rank": int(r["rank"]),
                          "score": round(float(r["score"]), 4)},
        })

    trials_path = out_dir / f"trials_{args.strategy}.json"
    with open(trials_path, "w") as f:
        json.dump({"imageTrials": trials}, f, indent=2)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"{'#':>3}  {'image':<22}{'city':<12}{'score':>8}{'entropy':>9}{'spread':>8}  PS")
    print("-" * 70)
    for _, r in sel.iterrows():
        print(f"{r['rank']:>3}  {r['image_id']:<22}{r['city']:<12}"
              f"{r['score']:>8.4f}{r['entropy']:>9.3f}{r['aid_spread']:>8.3f}  {r.get('ps_label','')}")

    print(f"\nCity mix of the shortlist: {sel['city'].value_counts().to_dict()}")
    print(f"Pool city mix            : {pool['city'].value_counts().to_dict()}")
    print(f"\nSaved → {csv_path}")
    print(f"Saved → {trials_path}  (drop into the study app's data/trials.json)")


if __name__ == "__main__":
    main()
