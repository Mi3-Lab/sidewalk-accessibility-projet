#!/usr/bin/env python3
"""
Upload all CV results to a single W&B run.

Creates one run per (loss_type, run_tag) with:
  - Encoder comparison table (Table 1 of the paper)
  - Per-aid bar charts across encoders
  - Per-fold scatter (variance analysis)
  - Full config logged as run metadata

Usage:
    python src/models/log_wandb.py \
        --results_dir results/cv/soft_kl \
        --loss_type   soft_kl \
        --wandb_project sidewalk-accessibility

    # After ablation:
    python src/models/log_wandb.py \
        --results_dir results/cv/hard_ce \
        --loss_type   hard_ce \
        --wandb_project sidewalk-accessibility
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import wandb

AIDS = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]

ENCODER_ORDER = [
    "clip-vit-b32",
    "clip-vit-b16",
    "clip-vit-l14",
    "dinov2-base",
    "dinov2-large",
    "siglip2-base",
    "siglip2-so400m",
    "vit-b16-sup",
]


def load_results(results_dir: Path) -> dict[str, dict]:
    data = {}
    for enc in ENCODER_ORDER:
        path = results_dir / enc / "cv_results.json"
        if path.exists():
            with open(path) as f:
                data[enc] = json.load(f)
        else:
            print(f"  [skip] {enc} — no cv_results.json found")
    return data


def build_encoder_table(results: dict[str, dict]) -> wandb.Table:
    """Main comparison table: encoder × overall + per-aid F1 + calibration."""
    cols = (
        ["encoder"]
        + [a.replace(" ", "_") + "_f1" for a in AIDS]
        + ["overall_f1", "overall_std", "brier_soft", "brier_hard", "ece"]
    )
    table = wandb.Table(columns=cols)
    for enc, d in results.items():
        row = [enc]
        for aid in AIDS:
            v = d["aids"].get(aid, {})
            row.append(round(v.get("macro_f1_mean", float("nan")), 3))
        ov = d.get("overall", {})
        row.append(round(ov.get("macro_f1_mean_across_aids",   float("nan")), 3))
        row.append(round(ov.get("macro_f1_std_across_aids",    float("nan")), 3))
        row.append(round(ov.get("brier_soft_mean_across_aids", float("nan")), 4))
        row.append(round(ov.get("brier_hard_mean_across_aids", float("nan")), 4))
        row.append(round(ov.get("ece_mean_across_aids",        float("nan")), 4))
        table.add_data(*row)
    return table


def build_fold_table(results: dict[str, dict]) -> wandb.Table:
    """Per-fold detail table for variance analysis."""
    cols = ["encoder", "aid", "fold", "macro_f1", "balanced_acc", "n_test_panos"]
    table = wandb.Table(columns=cols)
    for enc, d in results.items():
        for aid, v in d["aids"].items():
            for fold in v.get("folds", []):
                table.add_data(
                    enc, aid,
                    fold["fold"],
                    round(fold["macro_f1"],    3),
                    round(fold["balanced_acc"], 3),
                    fold["n_test_panos"],
                )
    return table


def build_balanced_acc_table(results: dict[str, dict]) -> wandb.Table:
    """Balanced accuracy comparison table."""
    cols = ["encoder"] + [a.replace(" ", "_") + "_bal_acc" for a in AIDS] + ["overall_bal_acc"]
    table = wandb.Table(columns=cols)
    for enc, d in results.items():
        row = [enc]
        bal_accs = []
        for aid in AIDS:
            v = d["aids"].get(aid, {})
            ba = v.get("balanced_acc_mean", float("nan"))
            row.append(round(ba, 3))
            if not (ba != ba):  # nan check
                bal_accs.append(ba)
        row.append(round(sum(bal_accs) / len(bal_accs), 3) if bal_accs else float("nan"))
        table.add_data(*row)
    return table


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",    required=True, help="e.g. results/cv/soft_kl")
    parser.add_argument("--loss_type",      required=True, choices=["soft_kl", "hard_ce"])
    parser.add_argument("--wandb_project",  default="sidewalk-accessibility")
    parser.add_argument("--run_name",       default=None,  help="Custom run name (auto-generated if omitted)")
    args = parser.parse_args()

    _seed(42)
    results_dir = Path(args.results_dir)
    run_name = args.run_name or f"cv_{args.loss_type}_all_encoders"

    print(f"Loading results from {results_dir} …")
    results = load_results(results_dir)
    if not results:
        print("No results found. Exiting.")
        return
    print(f"Loaded {len(results)} encoders: {list(results)}")

    # ── Collect hyperconfig from first result ─────────────────────────────────
    first = next(iter(results.values()))
    cfg   = first.get("config", {})

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        group="cross_validation",
        tags=[args.loss_type, "cv", "linear_probe"],
        config={
            "loss_type":    args.loss_type,
            "n_folds":      cfg.get("n_folds", 5),
            "lr":           cfg.get("lr"),
            "epochs":       cfg.get("epochs"),
            "weight_decay": cfg.get("weight_decay"),
            "random_state": cfg.get("random_state", 42),
            "seed_strategy": "RANDOM_STATE + fold_idx per fold",
            "encoders":     list(results),
            "n_encoders":   len(results),
        },
    )

    # ── Tables ────────────────────────────────────────────────────────────────
    print("Building tables …")
    run.log({
        "encoder_comparison/macro_f1":    build_encoder_table(results),
        "encoder_comparison/balanced_acc": build_balanced_acc_table(results),
        "fold_detail":                     build_fold_table(results),
    })

    # ── Scalar summary per encoder (shows up in W&B charts) ───────────────────
    for enc, d in results.items():
        ov = d.get("overall", {})
        run.summary[f"{enc}/overall_macro_f1"]     = round(ov.get("macro_f1_mean_across_aids",   float("nan")), 4)
        run.summary[f"{enc}/overall_macro_f1_std"] = round(ov.get("macro_f1_std_across_aids",    float("nan")), 4)
        run.summary[f"{enc}/brier_soft"]           = round(ov.get("brier_soft_mean_across_aids", float("nan")), 4)
        run.summary[f"{enc}/brier_hard"]           = round(ov.get("brier_hard_mean_across_aids", float("nan")), 4)
        run.summary[f"{enc}/ece"]                  = round(ov.get("ece_mean_across_aids",        float("nan")), 4)
        for aid, v in d["aids"].items():
            aid_key = aid.lower().replace(" ", "_")
            run.summary[f"{enc}/{aid_key}_f1"]       = round(v["macro_f1_mean"],     4)
            run.summary[f"{enc}/{aid_key}_bal_acc"]  = round(v["balanced_acc_mean"], 4)
            run.summary[f"{enc}/{aid_key}_brier_soft"] = round(v.get("brier_soft_mean", float("nan")), 4)
            run.summary[f"{enc}/{aid_key}_ece"]      = round(v.get("ece_mean",        float("nan")), 4)

    # ── Best encoder summary ──────────────────────────────────────────────────
    ranked = sorted(
        [(enc, d["overall"]["macro_f1_mean_across_aids"])
         for enc, d in results.items() if "overall" in d],
        key=lambda x: x[1], reverse=True,
    )
    run.summary["best_encoder"]    = ranked[0][0]
    run.summary["best_overall_f1"] = round(ranked[0][1], 4)

    print("\nEncoder ranking (overall macro-F1):")
    for rank, (enc, f1) in enumerate(ranked, 1):
        print(f"  #{rank}  {enc:<22}  {f1:.4f}")

    run.finish()
    print(f"\nW&B run: {run.url}")


if __name__ == "__main__":
    main()
