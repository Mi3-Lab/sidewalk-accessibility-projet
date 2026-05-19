#!/usr/bin/env python3
"""
Single-image inference API for sidewalk accessibility.

This is the robot-facing interface: given one sidewalk image, return per-aid
accessibility probability distributions [p_no, p_unsure, p_yes].  A robot or
autonomous vehicle planning a drop-off can call this to score each candidate
sidewalk segment in real time.

Typical deployment latency (A100 40GB, batch=1):
  DINOv2-large  ~78ms   ← best accuracy/speed trade-off
  CLIP ViT-B/32 ~62ms   ← smallest footprint

Usage (CLI):
    python src/models/infer.py \
        --image      path/to/sidewalk.jpg \
        --encoder    dinov2-large \
        --checkpoint results/models/dinov2-large \
        --format     json

Usage (library):
    from src.models.infer import SidewalkAccessibilityModel
    model = SidewalkAccessibilityModel("dinov2-large", "results/models/dinov2-large")
    result = model.predict("sidewalk.jpg")
    # result["Manual wheelchair"]["p_yes"] → 0.73
    # result["Manual wheelchair"]["passable"] → True (p_yes ≥ threshold)

Output JSON per aid:
    {
      "Walking cane": {
        "p_no": 0.12, "p_unsure": 0.15, "p_yes": 0.73,
        "passable": true,
        "confidence": "high"    # high: H < 0.5, medium: 0.5–0.9, low: H ≥ 0.9
      },
      ...
    }
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import torch
import torch.nn as nn

from train import (
    AIDS,
    CLASS3,
    ENCODERS,
    RANDOM_STATE,
    extract_encoder_features,
    load_encoder,
    set_seed,
)

PASSABILITY_THRESHOLD = 0.50   # p_yes ≥ this → passable
HIGH_CONF_ENTROPY     = 0.50   # H < this → high confidence
MED_CONF_ENTROPY      = 0.90   # H < this → medium confidence


def _entropy(p: np.ndarray) -> float:
    """Normalised Shannon entropy in [0, 1]."""
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


class SidewalkAccessibilityModel:
    """
    Loaded once, called for every sidewalk image.

    Args:
        encoder_key:     One of the 8 supported encoders (see ENCODERS in train.py).
        checkpoint_dir:  Output of train_final_model.py (contains per-aid probe.pth).
        threshold:       p_yes threshold for "passable" binary flag.
    """

    def __init__(
        self,
        encoder_key: str = "dinov2-large",
        checkpoint_dir: str = "results/models/dinov2-large",
        threshold: float = PASSABILITY_THRESHOLD,
        seed: int = RANDOM_STATE,
    ):
        set_seed(seed)
        self.encoder_key    = encoder_key
        self.checkpoint_dir = Path(checkpoint_dir)
        self.threshold      = threshold

        print(f"Loading encoder: {encoder_key} ({ENCODERS[encoder_key][0]})")
        self.model, self.processor, self.device, self.enc_type = load_encoder(encoder_key)
        self.probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._probes:  dict[str, nn.Linear]          = {}
        self._scalers: dict[str, object]              = {}
        self._load_probes()

    def _load_probes(self) -> None:
        loaded = []
        for aid in AIDS:
            aid_key     = aid.lower().replace(" ", "_")
            scaler_path = self.checkpoint_dir / aid_key / "scaler.joblib"
            probe_path  = self.checkpoint_dir / aid_key / "probe.pth"

            if not scaler_path.exists() or not probe_path.exists():
                print(f"  WARNING: no checkpoint for '{aid}' — run train_final_model.py first.")
                continue

            scaler = joblib.load(scaler_path)
            # Infer feature dim from scaler
            feature_dim = scaler.scale_.shape[0]
            probe = nn.Linear(feature_dim, len(CLASS3))
            probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
            probe = probe.to(self.probe_device).eval()

            self._scalers[aid] = scaler
            self._probes[aid]  = probe
            loaded.append(aid)

        if loaded:
            print(f"Probes loaded for: {loaded}")
        else:
            raise FileNotFoundError(
                f"No checkpoints found in {self.checkpoint_dir}. "
                "Run: python src/models/train_final_model.py"
            )

    def predict(self, image_path: str) -> dict:
        """
        Predict accessibility for a single sidewalk image.

        Returns:
            Dict mapping each mobility aid to its probability distribution
            and derived passability flag.
        """
        t0 = time.perf_counter()

        features = extract_encoder_features(
            self.model, self.processor, self.device,
            [image_path], self.enc_type,
        )   # (1, D)

        results = {}
        for aid, probe in self._probes.items():
            scaler  = self._scalers[aid]
            X_sc    = scaler.transform(features)
            X_t     = torch.tensor(X_sc, dtype=torch.float32).to(self.probe_device)

            with torch.no_grad():
                probs = torch.softmax(probe(X_t), dim=1).cpu().numpy()[0]

            p_no, p_unsure, p_yes = float(probs[0]), float(probs[1]), float(probs[2])
            h = _entropy(probs)
            confidence = "high" if h < HIGH_CONF_ENTROPY else ("medium" if h < MED_CONF_ENTROPY else "low")

            results[aid] = {
                "p_no":       round(p_no,    4),
                "p_unsure":   round(p_unsure, 4),
                "p_yes":      round(p_yes,   4),
                "passable":   bool(p_yes >= self.threshold),
                "confidence": confidence,
                "entropy":    round(h, 4),
            }

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"predictions": results, "latency_ms": elapsed_ms}

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        """Predict for multiple images (batched feature extraction, one result per image)."""
        t0 = time.perf_counter()

        features = extract_encoder_features(
            self.model, self.processor, self.device,
            image_paths, self.enc_type,
        )   # (N, D)

        batch_results = []
        for i in range(len(image_paths)):
            feat_i = features[i:i+1]
            img_results = {}
            for aid, probe in self._probes.items():
                scaler = self._scalers[aid]
                X_sc   = scaler.transform(feat_i)
                X_t    = torch.tensor(X_sc, dtype=torch.float32).to(self.probe_device)
                with torch.no_grad():
                    probs = torch.softmax(probe(X_t), dim=1).cpu().numpy()[0]
                p_no, p_unsure, p_yes = float(probs[0]), float(probs[1]), float(probs[2])
                h = _entropy(probs)
                img_results[aid] = {
                    "p_no":       round(p_no,    4),
                    "p_unsure":   round(p_unsure, 4),
                    "p_yes":      round(p_yes,   4),
                    "passable":   bool(p_yes >= self.threshold),
                    "confidence": "high" if h < HIGH_CONF_ENTROPY else ("medium" if h < MED_CONF_ENTROPY else "low"),
                    "entropy":    round(h, 4),
                }
            batch_results.append(img_results)

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        per_img  = round(total_ms / max(len(image_paths), 1), 1)
        print(f"Batch of {len(image_paths)}: {total_ms}ms total ({per_img}ms/image)")
        return batch_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict sidewalk accessibility for a single image."
    )
    parser.add_argument("--image",      required=True, help="Path to sidewalk image.")
    parser.add_argument("--encoder",    default="dinov2-large", choices=list(ENCODERS))
    parser.add_argument("--checkpoint", default="results/models/dinov2-large",
                        help="Directory with per-aid probe checkpoints.")
    parser.add_argument("--threshold",  type=float, default=PASSABILITY_THRESHOLD)
    parser.add_argument(
        "--format", default="table", choices=["table", "json"],
        help="Output format. 'table': human-readable. 'json': machine-readable.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    model = SidewalkAccessibilityModel(args.encoder, args.checkpoint, args.threshold, args.seed)
    output = model.predict(args.image)
    predictions = output["predictions"]

    if args.format == "json":
        print(json.dumps(output, indent=2))
        return

    # Human-readable table
    print(f"\nSidewalk: {args.image}")
    print(f"Encoder:  {args.encoder}  |  Latency: {output['latency_ms']}ms")
    print(f"Threshold: p_yes ≥ {args.threshold}\n")
    print(f"{'Mobility Aid':<25} {'p_no':>7} {'p_unsure':>9} {'p_yes':>7} {'Passable':>10} {'Confidence':>12}")
    print("─" * 73)
    for aid, r in predictions.items():
        passable_str = "✓ YES" if r["passable"] else "✗ NO"
        print(
            f"{aid:<25} {r['p_no']:>7.3f} {r['p_unsure']:>9.3f} {r['p_yes']:>7.3f}"
            f" {passable_str:>10} {r['confidence']:>12}"
        )
    print("─" * 73)


if __name__ == "__main__":
    main()
