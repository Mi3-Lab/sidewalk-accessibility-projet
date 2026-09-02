#!/usr/bin/env python3
"""
Leave-one-city-out: does the probe transfer, and do sidewalk-masked features
transfer better than whole-image ones?

The published geographic-transfer result does not survive an encoder swap (all
five cities drop to chance under SigLIP2), which suggests the probe may be keying
on city-level scene appearance rather than surface condition. Mask-pooling the
walkable region was the proposed fix; in-domain it changes nothing much, which is
expected — within a fixed set of cities, scene context is legitimately
informative. The claim only bites across cities.

This tests it directly on the real soft labels, rather than through the binary
CurbRamp proxy the transfer evaluation uses. The 52 panoramas span nine cities;
each fold holds out one city entirely and trains on the rest.

Both feature variants come from the same .npz, so the only thing that differs
between the two runs is what the probe sees.

Usage:
    python src/models/leave_one_city.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --features_npz results/features/dinov2-large_masked.npz \
        --output_dir   results/leave_one_city/dinov2-large
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from crossval import (
    _features_for,
    brier_score_soft,
    entropy_correlation,
    expected_calibration_error,
    load_feature_npz,
    predict_proba,
)
from train import (
    AIDS,
    LABEL_MAP,
    PROBE_WD,
    RANDOM_STATE,
    SOFT_COLS,
    find_image_path,
    set_seed,
    train_probe_hard,
    train_probe_soft,
)

# gsv-<city>-<paneid>-<x>-<y>.png ; a few files use an underscore in the city
CITY_RE = re.compile(r"^gsv-([a-z_]+)-\d")


def city_of(path) -> str:
    m = CITY_RE.match(Path(str(path)).name.lower())
    if not m:
        return "unknown"
    # la_piedad and lapiedad appear in both spellings
    return m.group(1).replace("_", "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tallies_json", required=True)
    p.add_argument("--images_dir",   required=True)
    p.add_argument("--features_npz", required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--weight_decay", type=float, default=PROBE_WD)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--seed",    type=int, default=RANDOM_STATE)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(lambda x: find_image_path(x, args.images_dir))
    tallies = tallies[tallies["path"].notna()].copy()
    tallies["label_int"] = tallies["argmax_label"].map(LABEL_MAP)
    tallies["city"] = tallies["path"].apply(city_of)

    counts = tallies.groupby("city")["ImageID"].nunique().sort_values(ascending=False)
    print("Panoramas per city:")
    for c, n in counts.items():
        print(f"  {c:<14} {n}")
    cities = [c for c, n in counts.items() if n >= 2 and c != "unknown"]
    print(f"\nHolding out {len(cities)} cities with >=2 panoramas\n")

    records = []
    for key in ("whole", "masked"):
        cache = load_feature_npz(args.features_npz, key)
        for loss in ("soft_kl", "hard_ce"):
            for city in cities:
                for seed in range(args.seed, args.seed + args.n_seeds):
                    set_seed(seed)
                    for aid in AIDS:
                        df = tallies[tallies["MobilityAid"] == aid]
                        tr = df[df["city"] != city]
                        te = df[df["city"] == city]
                        if len(tr) < 5 or te.empty:
                            continue

                        X_tr = _features_for(tr["path"].tolist(), cache)
                        X_te = _features_for(te["path"].tolist(), cache)
                        sc = StandardScaler(with_mean=False).fit(X_tr)
                        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

                        w = tr["sample_weight"].values
                        if loss == "soft_kl":
                            probe = train_probe_soft(
                                X_tr, tr[SOFT_COLS].values.astype(np.float32), w,
                                X_tr.shape[1], device, seed=seed,
                                weight_decay=args.weight_decay)
                        else:
                            probe = train_probe_hard(
                                X_tr, tr["label_int"].values, w,
                                X_tr.shape[1], device, seed=seed,
                                weight_decay=args.weight_decay)

                        pr = predict_proba(probe, X_te, device)
                        y_soft = te[SOFT_COLS].values.astype(np.float32)
                        records.append({
                            "features": key, "loss": loss, "city": city,
                            "aid": aid, "seed": seed, "n_test": len(te),
                            "brier_soft": brier_score_soft(pr, y_soft),
                            "macro_f1": float(f1_score(te["label_int"].values, pr.argmax(1),
                                                       average="macro", zero_division=0)),
                            "entropy_corr": entropy_correlation(pr, y_soft),
                        })
            print(f"  {key}/{loss} done")

    df = pd.DataFrame(records)
    summary = {}
    for (key, loss), g in df.groupby(["features", "loss"]):
        summary[f"{key}/{loss}"] = {
            "brier_soft": float(g["brier_soft"].mean()),
            "macro_f1":   float(g["macro_f1"].mean()),
            "entropy_corr": float(np.nanmean(g["entropy_corr"])),
            "per_city": {c: float(gc["brier_soft"].mean())
                         for c, gc in g.groupby("city")},
        }

    with open(out / "leave_one_city.json", "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "records": records}, f, indent=2)

    print(f"\n{'variant':<22}{'Brier':>10}{'F1':>9}{'H-corr':>10}")
    print("-" * 51)
    for k in sorted(summary):
        v = summary[k]
        print(f"{k:<22}{v['brier_soft']:>10.4f}{v['macro_f1']:>9.3f}{v['entropy_corr']:>10.3f}")

    for loss in ("soft_kl", "hard_ce"):
        w = summary[f"whole/{loss}"]["brier_soft"]
        m = summary[f"masked/{loss}"]["brier_soft"]
        print(f"\n{loss}: masking changes held-out-city Brier by {m - w:+.4f} "
              f"({'better' if m < w else 'worse'})")

    print(f"\nSaved → {out / 'leave_one_city.json'}")


if __name__ == "__main__":
    main()
