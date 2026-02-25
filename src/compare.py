#!/usr/bin/env python3
"""
Compare accuracies across models.
"""

import json
import os
from pathlib import Path

def compare_accuracies(models_dir):
    results = {}
    for subdir in Path(models_dir).iterdir():
        if subdir.is_dir():
            meta_path = subdir / "meta.json"
            if meta_path.exists():
                meta = json.load(open(meta_path))
                aid = meta['aid']
                macro_f1 = meta.get('macro_f1', meta.get('accuracy', 0))  # fallback to accuracy if macro_f1 not present
                balanced_acc = meta.get('balanced_accuracy', macro_f1)  # fallback to macro_f1 if not present
                backbone = meta['backbone']
                if aid not in results:
                    results[aid] = {}
                results[aid][backbone] = {
                    "macro_f1": macro_f1,
                    "balanced_accuracy": balanced_acc
                }
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    args = parser.parse_args()
    results = compare_accuracies(args.models_dir)
    print(json.dumps(results, indent=2))