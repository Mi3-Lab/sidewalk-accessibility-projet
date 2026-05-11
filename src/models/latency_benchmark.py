#!/usr/bin/env python3
"""
Inference latency benchmark for all encoders and zero-shot VLMs.

Measures per-image feature extraction time (encoders) and per-image
inference time (VLMs) on GPU. Reports mean ± std over N_RUNS iterations.

Usage:
    python src/models/latency_benchmark.py \
        --images_dir data/images/sidewalk-images \
        --output_dir results/latency

    # VLMs only (skips encoders):
    python src/models/latency_benchmark.py \
        --images_dir data/images/sidewalk-images \
        --output_dir results/latency \
        --vlms_only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from PIL import Image

from train import ENCODERS, extract_encoder_features, load_encoder

WARMUP_RUNS = 5
N_RUNS      = 20

VLM_CONFIGS = {
    "llava-1.5-7b":  ("llava-hf/llava-1.5-7b-hf",         "llava"),
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct",      "qwen25vl"),
    "qwen3-vl-8b":   ("Qwen/Qwen3-VL-8B-Instruct",        "qwen3vl"),
}

PROMPT = (
    "Is the sidewalk in this image accessible to someone using a manual wheelchair? "
    "Answer with one word: yes, no, or unsure."
)


# ── Timing helpers ─────────────────────────────────────────────────────────────

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(fn, warmup: int = WARMUP_RUNS, runs: int = N_RUNS) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms":   round(float(np.mean(times)),   2),
        "std_ms":    round(float(np.std(times)),    2),
        "min_ms":    round(float(np.min(times)),    2),
        "max_ms":    round(float(np.max(times)),    2),
        "n_runs":    runs,
    }


# ── Encoder benchmark ──────────────────────────────────────────────────────────

def benchmark_encoder(enc_key: str, img_path: str) -> dict:
    print(f"\n[encoder] {enc_key} …")
    model, processor, device, enc_type = load_encoder(enc_key)

    def run():
        extract_encoder_features(model, processor, device, [img_path], enc_type)

    result = measure(run)
    result["encoder"]  = enc_key
    result["device"]   = str(device)
    result["hf_id"]    = ENCODERS[enc_key][0]
    print(f"  {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms/image  (GPU: {device})")

    # Free VRAM before next model
    del model
    torch.cuda.empty_cache()
    return result


# ── VLM benchmark ──────────────────────────────────────────────────────────────

def benchmark_vlm_llava(hf_id: str, img: Image.Image) -> dict:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(hf_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    def run():
        inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8, do_sample=False)

    result = measure(run, warmup=3, runs=10)
    del model; torch.cuda.empty_cache()
    return result


def benchmark_vlm_qwen(hf_id: str, model_type: str, img: Image.Image) -> dict:
    from transformers import AutoProcessor
    if model_type == "qwen25vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as CLS
    else:
        from transformers import Qwen3VLForConditionalGeneration as CLS

    processor = AutoProcessor.from_pretrained(hf_id)
    model = CLS.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    img_small = img.resize((448, 448))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_small},
        {"type": "text",  "text":  PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def run():
        inputs = processor(text=[text], images=[img_small], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8, do_sample=False)

    result = measure(run, warmup=3, runs=10)
    del model; torch.cuda.empty_cache()
    return result


def benchmark_vlm(vlm_key: str, img: Image.Image) -> dict:
    hf_id, model_type = VLM_CONFIGS[vlm_key]
    print(f"\n[vlm] {vlm_key} …")

    if model_type == "llava":
        result = benchmark_vlm_llava(hf_id, img)
    else:
        result = benchmark_vlm_qwen(hf_id, model_type, img)

    result["model"]     = vlm_key
    result["hf_id"]     = hf_id
    result["device"]    = "cuda"
    print(f"  {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms/image")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", default="results/latency")
    parser.add_argument("--vlms_only",  action="store_true")
    parser.add_argument("--encoders_only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick one real image for benchmarking
    images_dir = Path(args.images_dir)
    img_paths  = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png"))
    if not img_paths:
        print(f"No images found in {images_dir}"); return
    img_path = str(img_paths[0])
    img      = Image.open(img_path).convert("RGB")
    print(f"Benchmark image: {img_path}  ({img.size[0]}×{img.size[1]})")
    print(f"Warmup: {WARMUP_RUNS} runs | Measure: {N_RUNS} runs\n")

    results = {"encoders": [], "vlms": []}

    # ── Encoders ─────────────────────────────────────────────────────────────
    if not args.vlms_only:
        print("=" * 60)
        print("ENCODER LATENCY (feature extraction, 1 image, GPU)")
        print("=" * 60)
        for enc_key in ENCODERS:
            try:
                r = benchmark_encoder(enc_key, img_path)
                results["encoders"].append(r)
            except Exception as e:
                print(f"  [error] {enc_key}: {e}")

    # ── VLMs ─────────────────────────────────────────────────────────────────
    if not args.encoders_only:
        print("\n" + "=" * 60)
        print("VLM LATENCY (full inference, 1 image, GPU)")
        print("=" * 60)
        for vlm_key in VLM_CONFIGS:
            try:
                r = benchmark_vlm(vlm_key, img)
                results["vlms"].append(r)
            except Exception as e:
                print(f"  [error] {vlm_key}: {e}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = output_dir / "latency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Encoder latency (ms/image, A100) ─────────────────────────────")
    print(f"  {'encoder':<22} {'mean':>8} {'std':>7}")
    print("  " + "-" * 40)
    for r in sorted(results["encoders"], key=lambda x: x["mean_ms"]):
        print(f"  {r['encoder']:<22} {r['mean_ms']:>7.1f}ms ±{r['std_ms']:.1f}")

    if results["vlms"]:
        print("\n── VLM latency (ms/image, A100) ────────────────────────────────")
        print(f"  {'model':<22} {'mean':>8} {'std':>7}")
        print("  " + "-" * 40)
        for r in sorted(results["vlms"], key=lambda x: x["mean_ms"]):
            print(f"  {r['model']:<22} {r['mean_ms']:>7.1f}ms ±{r['std_ms']:.1f}")


if __name__ == "__main__":
    main()
