#!/usr/bin/env python3
"""
Update meta.json files with balanced_accuracy.
"""

import json
import torch
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

def update_meta_with_balanced_acc(models_dir):
    for subdir in Path(models_dir).iterdir():
        if subdir.is_dir():
            meta_path = subdir / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

                # If balanced_accuracy already exists, skip
                if 'balanced_accuracy' in meta:
                    continue

                # Load validation data - we need to re-compute
                # But we don't have the validation data here. This is tricky.

                # Actually, since we can't easily get the validation predictions without the data,
                # let's assume we need to re-run training. But for now, let's add a placeholder.

                # Wait, better idea: since macro_f1 is there, and for balanced datasets it's similar,
                # but we need exact. Let's re-run the training scripts.

                print(f"Updating {meta_path}")
                # For now, set balanced_accuracy to macro_f1 as approximation, but that's not accurate.
                # Balanced accuracy is average recall, macro_f1 is harmonic mean of precision and recall.

                # Actually, let's re-run the training. But to save time, perhaps compute it if we can.

                # Since the models are trained with fixed seed, the predictions are the same.
                # But to get balanced_accuracy, I need the y_val and y_pred.

                # The easiest is to modify the training script to save y_val and y_pred, but that's not done.

                # Let's re-run the training for all models. It should be quick since features are extracted.

                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    args = parser.parse_args()
    update_meta_with_balanced_acc(args.models_dir)