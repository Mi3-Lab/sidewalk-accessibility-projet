#!/usr/bin/env python3
"""
Compute soft Brier scores for VLM zero-shot predictions.

VLMs output hard "yes"/"no"/"unsure" responses with no calibrated probabilities.
We map responses to one-hot vectors and compute Brier_soft:
    Brier_soft = mean_i sum_j (pred_j - true_j)^2
where true_j is the human vote distribution [p_no, p_unsure, p_yes].

This matches the brier_soft metric in crossval.py and train.py, making
probe vs VLM comparisons directly comparable.

Usage:
    python src/models/compute_vlm_brier.py \
        --tallies    data/processed/tallies_firebase.json \
        --images_dir data/images/sidewalk-images \
        --results    results/zero_shot/qwen3-vl-8b/zero_shot_results.json \
                     results/zero_shot/qwen2.5-vl-7b/zero_shot_results.json \
                     results/zero_shot/llava-1.5-7b/zero_shot_results.json \
        --output     results/zero_shot/vlm_brier_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from train import AIDS, LABEL_MAP, find_image_path

CLASS3     = ["no", "unsure", "yes"]
RESP_MAP   = {"yes": 2, "no": 0, "unsure": 1}   # response string → class index


def parse_response(raw: str) -> int:
    """Map a VLM raw response string to a class index (0=no, 1=unsure, 2=yes)."""
    raw = raw.strip().lower()
    if raw.startswith("yes"):
        return 2
    if raw.startswith("no"):
        return 0
    if "unsure" in raw or "uncertain" in raw or "unclear" in raw:
        return 1
    # fall back: look for the first keyword anywhere
    for kw, idx in [("yes", 2), ("no", 0), ("unsure", 1)]:
        if kw in raw:
            return idx
    return 2  # default to "yes" (majority class)


def one_hot(idx: int, n: int = 3) -> np.ndarray:
    v = np.zeros(n)
    v[idx] = 1.0
    return v


def brier_soft(y_pred_probs: np.ndarray, y_true_soft: np.ndarray) -> float:
    """MSE between predicted probability vector and human vote distribution."""
    return float(np.mean(np.sum((y_pred_probs - y_true_soft) ** 2, axis=1)))


def load_tallies_ordered(tallies_path: str, images_dir: str) -> pd.DataFrame:
    """
    Load tallies in the same order zero_shot.py uses:
      pd.read_json → filter images found on disk → iterate per aid.
    """
    df = pd.read_json(tallies_path)
    df["path"] = df["ImageID"].apply(lambda x: find_image_path(x, images_dir))
    df = df[df["path"].notna()].copy()
    df["label_int"] = df["argmax_label"].map(LABEL_MAP)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute soft Brier scores for VLM zero-shot predictions."
    )
    parser.add_argument("--tallies",    default="data/processed/tallies_firebase.json")
    parser.add_argument("--images_dir", default="data/images/sidewalk-images")
    parser.add_argument("--results",    nargs="+", required=True,
                        help="Path(s) to zero_shot_results.json files.")
    parser.add_argument("--output",     default="results/zero_shot/vlm_brier_comparison.json")
    args = parser.parse_args()

    tallies = load_tallies_ordered(args.tallies, args.images_dir)
    print(f"Tallies loaded: {len(tallies)} rows, {tallies['ImageID'].nunique()} panoramas\n")

    output: dict = {"models": {}}

    for result_path in args.results:
        result_path = Path(result_path)
        with open(result_path) as f:
            res = json.load(f)

        model_name = res.get("model", result_path.parent.name)
        print(f"=== {model_name} ===")

        aid_briers = {}
        for aid in AIDS:
            df_aid = tallies[tallies["MobilityAid"] == aid].copy()
            if df_aid.empty:
                continue

            raw_responses = res.get("aids", {}).get(aid, {}).get("raw_responses", [])
            if len(raw_responses) != len(df_aid):
                print(f"  [{aid}] WARNING: {len(raw_responses)} responses vs "
                      f"{len(df_aid)} tallies rows — skipping")
                continue

            # Predicted one-hot vectors
            y_pred = np.array([one_hot(parse_response(r)) for r in raw_responses])

            # Ground truth soft distributions [p_no, p_unsure, p_yes]
            y_true_soft = df_aid[["p_no", "p_unsure", "p_yes"]].values

            bs = brier_soft(y_pred, y_true_soft)
            macro_f1 = res["aids"][aid].get("macro_f1", float("nan"))
            aid_briers[aid] = {"brier_soft": round(bs, 4), "macro_f1": round(macro_f1, 4)}
            print(f"  [{aid:<22}] Brier_soft={bs:.4f}   Macro F1={macro_f1:.3f}")

        overall_brier = float(np.mean([v["brier_soft"] for v in aid_briers.values()]))
        overall_f1    = float(np.mean([v["macro_f1"]   for v in aid_briers.values()]))
        print(f"  OVERALL                    Brier_soft={overall_brier:.4f}   Macro F1={overall_f1:.3f}\n")

        output["models"][model_name] = {
            "per_aid":      aid_briers,
            "overall_brier_soft": round(overall_brier, 4),
            "overall_macro_f1":   round(overall_f1, 4),
        }

    # ── Comparison table ──────────────────────────────────────────────────────
    print("=== Summary comparison table ===")
    print(f"{'Model':<28} {'Macro F1':>10} {'Brier_soft':>12}")
    print("-" * 54)

    # Probe numbers from cv_results (hardcoded from existing results)
    probes = [
        ("DINOv2-large  [Soft-KL probe]", 0.613, 0.075),
        ("CLIP-B/32     [Hard-CE probe]", 0.617, 0.231),
    ]
    for name, f1, bs in probes:
        print(f"{name:<28} {f1:>10.3f} {bs:>12.4f}")

    for m, v in output["models"].items():
        print(f"{m:<28} {v['overall_macro_f1']:>10.3f} {v['overall_brier_soft']:>12.4f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
