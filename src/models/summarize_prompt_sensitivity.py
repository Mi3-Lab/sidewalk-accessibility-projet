#!/usr/bin/env python3
"""
Summarize prompt sensitivity analysis results.

Reads results/prompt_sensitivity/<model>/<variant>/zero_shot_results.json
and produces a summary JSON + prints a markdown table.

Usage:
    python src/models/summarize_prompt_sensitivity.py \
        --results_dir results/prompt_sensitivity \
        --output results/prompt_sensitivity/summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = ["direct", "descriptive", "cot"]
VARIANT_LABELS = {
    "direct":      "Direct",
    "descriptive": "Descriptive",
    "cot":         "Chain-of-thought",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/prompt_sensitivity")
    parser.add_argument("--output",      default="results/prompt_sensitivity/summary.json")
    args = parser.parse_args()

    base = Path(args.results_dir)
    summary = {}

    model_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    for model_dir in model_dirs:
        model_key = model_dir.name
        summary[model_key] = {}

        for variant in VARIANTS:
            p = model_dir / variant / "zero_shot_results.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            overall = d.get("overall", {})
            summary[model_key][variant] = {
                "macro_f1":    round(overall.get("macro_f1_mean_across_aids",     0), 4),
                "balanced_acc": round(overall.get("balanced_acc_mean_across_aids", 0), 4),
                "parse_failures": overall.get("parse_failures", 0),
            }

    # Print markdown table
    print("\n## Prompt Sensitivity — Overall Macro-F1\n")
    print(f"{'Model':<22} {'Direct':>10} {'Descriptive':>13} {'CoT':>8} {'Range':>8} {'Best':>15}")
    print("-" * 82)

    for model_key, variants in summary.items():
        f1s = {v: variants[v]["macro_f1"] for v in variants}
        if not f1s:
            continue
        values = list(f1s.values())
        rng = max(values) - min(values)
        best_v = max(f1s, key=f1s.get)
        print(
            f"{model_key:<22}"
            f" {f1s.get('direct', float('nan')):>10.3f}"
            f" {f1s.get('descriptive', float('nan')):>13.3f}"
            f" {f1s.get('cot', float('nan')):>8.3f}"
            f" {rng:>8.3f}"
            f" {VARIANT_LABELS[best_v]:>15}"
        )

    print()

    # Per-aid breakdown
    print("## Per-Aid Macro-F1 by Variant\n")
    for model_key, variants_data in summary.items():
        if not variants_data:
            continue
        print(f"### {model_key}\n")
        # Re-read for per-aid data
        aids_seen = set()
        aid_rows = {}
        for variant in VARIANTS:
            p = base / model_key / variant / "zero_shot_results.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            for aid, metrics in d.get("aids", {}).items():
                aids_seen.add(aid)
                if aid not in aid_rows:
                    aid_rows[aid] = {}
                aid_rows[aid][variant] = round(metrics.get("macro_f1", 0), 3)

        aids = sorted(aids_seen)
        header = f"{'Aid':<25} {'Direct':>8} {'Descriptive':>13} {'CoT':>8}"
        print(header)
        print("-" * 58)
        for aid in aids:
            row = aid_rows.get(aid, {})
            print(
                f"{aid:<25}"
                f" {row.get('direct', float('nan')):>8.3f}"
                f" {row.get('descriptive', float('nan')):>13.3f}"
                f" {row.get('cot', float('nan')):>8.3f}"
            )
        print()

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
