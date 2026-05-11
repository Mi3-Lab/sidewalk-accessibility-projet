#!/usr/bin/env python3
"""
Zero-shot VLM evaluation for sidewalk accessibility classification.

Evaluates open-source VLMs zero-shot on the same task as the linear probe.
No fine-tuning — the model is prompted directly with the accessibility question.

Supported models:
  llava-1.5-7b   | llava-hf/llava-1.5-7b-hf          (~14 GB VRAM float16)
  llava-1.6-7b   | llava-hf/llava-v1.6-mistral-7b-hf  (~14 GB VRAM float16)
  qwen2.5-vl-7b  | Qwen/Qwen2.5-VL-7B-Instruct        (~14 GB VRAM bfloat16)
  qwen2.5-vl-3b  | Qwen/Qwen2.5-VL-3B-Instruct        (~6  GB VRAM bfloat16)

If VRAM is tight, install bitsandbytes and pass --load_in_4bit:
    pip install bitsandbytes

Usage:
    python src/models/zero_shot.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --model        llava-1.5-7b \
        --output_dir   results/zero_shot/llava-1.5-7b \
        --wandb_project sidewalk-accessibility
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from train import AIDS, LABEL_MAP, SOFT_COLS, find_image_path

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS = {
    "llava-1.5-7b":         ("llava-hf/llava-1.5-7b-hf",              "llava"),
    "llava-1.6-7b":         ("llava-hf/llava-v1.6-mistral-7b-hf",     "llava"),
    "qwen2.5-vl-7b":        ("Qwen/Qwen2.5-VL-7B-Instruct",           "qwen25vl"),
    "qwen2.5-vl-3b":        ("Qwen/Qwen2.5-VL-3B-Instruct",           "qwen25vl"),
    "qwen3-vl-8b":          ("Qwen/Qwen3-VL-8B-Instruct",             "qwen3vl"),
    "qwen3-vl-8b-thinking": ("Qwen/Qwen3-VL-8B-Thinking",             "qwen3vl_thinking"),
    "qwen3-vl-32b":         ("Qwen/Qwen3-VL-32B-Instruct",            "qwen3vl"),
    "qwen3.6-27b":          ("Qwen/Qwen3.6-27B",                      "qwen36"),
    "qwen3.6-27b-fp8":     ("Qwen/Qwen3.6-27B-FP8",                  "qwen36_fp8"),
}

AID_PHRASE = {
    "Walking cane":        "a walking cane",
    "Walker":              "a walker",
    "Mobility scooter":    "a mobility scooter",
    "Manual wheelchair":   "a manual wheelchair",
    "Motorized wheelchair": "a motorized wheelchair",
}

PROMPTS = {
    "descriptive": (
        "You are evaluating sidewalk and street accessibility for people with mobility impairments.\n"
        "Question: Is the sidewalk or path shown in this image accessible to someone using {aid}?\n"
        "Answer with exactly one word — yes, no, or unsure — and nothing else."
    ),
    "direct": (
        "Can a person using {aid} pass through the sidewalk in this image?\n"
        "Answer with one word: yes, no, or unsure."
    ),
    "cot": (
        "Look at this sidewalk image carefully. Consider the surface condition, obstacles, "
        "curb cuts, width, and any barriers.\n"
        "Question: Is this sidewalk accessible to someone using {aid}?\n"
        "Think briefly, then end your answer with one word on the last line: yes, no, or unsure."
    ),
}

PROMPT = PROMPTS["descriptive"]

ANSWER_TO_INT = {"no": 0, "unsure": 1, "yes": 2}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_key: str, load_in_4bit: bool = False):
    hf_id, model_type = MODELS[model_key]
    print(f"Loading {model_key} ({hf_id}) …")

    bnb_cfg = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("  4-bit quantization enabled")
        except ImportError:
            print("  bitsandbytes not installed — falling back to float16")

    # Use Flash Attention 2 if available (2-3x faster on A100)
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("  Flash Attention 2 enabled")
    except ImportError:
        attn_impl = "eager"

    if model_type == "llava":
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(hf_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_impl,
            quantization_config=bnb_cfg,
        )

    elif model_type == "qwen25vl":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(hf_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            quantization_config=bnb_cfg,
        )

    elif model_type in ("qwen3vl", "qwen3vl_thinking"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(hf_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            quantization_config=bnb_cfg,
        )

    elif model_type == "qwen36":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(hf_id)
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            quantization_config=bnb_cfg,
        )

    elif model_type == "qwen36_fp8":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(hf_id)
        # FP8 weights are pre-quantized — no torch_dtype or quantization_config needed
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id,
            device_map="auto",
            attn_implementation=attn_impl,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    print(f"  Loaded. Device map: {getattr(model, 'hf_device_map', 'N/A')}")
    return model, processor, model_type


# ── Batch inference ────────────────────────────────────────────────────────────

def batch_llava(model, processor, images: list, prompt: str) -> list[str]:
    conversations = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        for _ in images
    ]
    texts = [processor.apply_chat_template(c, add_generation_prompt=True) for c in conversations]
    inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return [
        r.strip() for r in
        processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    ]


def batch_qwen2vl(model, processor, images: list, prompt: str) -> list[str]:
    messages_batch = [
        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        for img in images
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return [
        r.strip() for r in
        processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    ]


def batch_qwen3vl(model, processor, images: list, prompt: str) -> list[str]:
    # Qwen3-VL: one image at a time using file:// paths to avoid processor hang
    import tempfile, os
    results = []
    for img in images:
        img_resized = img.resize((448, 448))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
            img_resized.save(tmp_path, format="JPEG")
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{tmp_path}"},
                {"type": "text",  "text":  prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img_resized], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            response = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
        finally:
            os.unlink(tmp_path)
        results.append(response)
    return results


def batch_qwen3vl_thinking(model, processor, images: list, prompt: str) -> list[str]:
    """Qwen3-VL Thinking variant — uses more tokens, strips <think>...</think> block."""
    import tempfile, os
    results = []
    for img in images:
        img_resized = img.resize((448, 448))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
            img_resized.save(tmp_path, format="JPEG")
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{tmp_path}"},
                {"type": "text",  "text":  prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img_resized], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            response = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            # Extract the answer after </think> block
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
        finally:
            os.unlink(tmp_path)
        results.append(response)
    return results


def batch_qwen36(model, processor, images: list, prompt: str) -> list[str]:
    """Qwen3.6 — uses AutoModelForImageTextToText with apply_chat_template returning tensors."""
    import tempfile, os
    results = []
    for img in images:
        img_resized = img.resize((448, 448))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
            img_resized.save(tmp_path, format="JPEG")
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "url": tmp_path},  # path direto, sem file://
                {"type": "text",  "text": prompt},
            ]}]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            response = processor.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
        finally:
            os.unlink(tmp_path)
        results.append(response)
    return results


def generate_batch(model, processor, model_type: str, images: list, prompt: str) -> list[str]:
    if model_type == "llava":
        return batch_llava(model, processor, images, prompt)
    elif model_type == "qwen25vl":
        return batch_qwen2vl(model, processor, images, prompt)
    elif model_type == "qwen3vl":
        return batch_qwen3vl(model, processor, images, prompt)
    elif model_type == "qwen3vl_thinking":
        return batch_qwen3vl_thinking(model, processor, images, prompt)
    elif model_type in ("qwen36", "qwen36_fp8"):
        return batch_qwen36(model, processor, images, prompt)
    raise ValueError(model_type)


def parse_answer(response: str) -> int:
    """Extract yes/no/unsure from response. Defaults to unsure if ambiguous."""
    text = response.lower().strip()
    # Look for the first matching keyword
    for word in re.split(r"\W+", text):
        if word in ANSWER_TO_INT:
            return ANSWER_TO_INT[word]
    # Partial matches as fallback
    if "yes" in text:
        return 2
    if "no" in text:
        return 0
    return 1  # unsure


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "n_samples":    int(len(y_true)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tallies_json",  required=True)
    parser.add_argument("--images_dir",    required=True)
    parser.add_argument("--model",         required=True, choices=list(MODELS))
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--batch_size",    type=int, default=8,
                        help="Images per forward pass (default: 8 for A100 40GB)")
    parser.add_argument("--load_in_4bit",  action="store_true",
                        help="Load model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--prompt_variant", default="descriptive",
                        choices=list(PROMPTS.keys()),
                        help="Prompt variant for sensitivity analysis (default: descriptive)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            run_name = f"zero_shot/{args.model}"
            if args.prompt_variant != "descriptive":
                run_name += f"/{args.prompt_variant}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                group="zero_shot",
                tags=["zero_shot", args.model, args.prompt_variant],
                config={
                    "model": args.model,
                    "hf_id": MODELS[args.model][0],
                    "prompt_variant": args.prompt_variant,
                },
            )
        except ImportError:
            print("wandb not installed — skipping W&B logging")

    # ── Load data ────────────────────────────────────────────────────────────
    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(
        lambda x: find_image_path(x, args.images_dir)
    )
    tallies = tallies[tallies["path"].notna()].copy()
    tallies["label_int"] = tallies["argmax_label"].map(LABEL_MAP)
    print(f"Dataset: {len(tallies)} rows, {tallies['ImageID'].nunique()} panoramas")

    # ── Load model ───────────────────────────────────────────────────────────
    model, processor, model_type = load_model(args.model, load_in_4bit=args.load_in_4bit)

    # ── Evaluate per aid ─────────────────────────────────────────────────────
    prompt_template = PROMPTS[args.prompt_variant]
    print(f"Prompt variant: {args.prompt_variant}")

    all_results: dict = {
        "model":          args.model,
        "hf_id":          MODELS[args.model][0],
        "prompt_variant": args.prompt_variant,
        "prompt_template": prompt_template,
        "aids":           {},
    }
    parse_failures = 0

    for aid in AIDS:
        df_aid = tallies[tallies["MobilityAid"] == aid].copy()
        if df_aid.empty:
            print(f"\n[{aid}] no data — skipping.")
            continue

        prompt = prompt_template.format(aid=AID_PHRASE[aid])
        print(f"\n── {aid} ({len(df_aid)} samples) ──")

        y_true, y_pred, raw_responses = [], [], []
        rows = list(df_aid.itertuples())

        for batch_start in tqdm(range(0, len(rows), args.batch_size), desc=aid):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            images, labels = [], []
            for row in batch_rows:
                try:
                    images.append(Image.open(row.path).convert("RGB"))
                    labels.append(int(row.label_int))
                except Exception as e:
                    print(f"  [warn] could not load {row.path}: {e}")

            if not images:
                continue

            responses = generate_batch(model, processor, model_type, images, prompt)

            for response, label in zip(responses, labels):
                pred = parse_answer(response)
                if not any(w in response.lower() for w in ("yes", "no", "unsure")):
                    parse_failures += 1
                y_true.append(label)
                y_pred.append(pred)
                raw_responses.append(response)

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        metrics = compute_metrics(y_true_arr, y_pred_arr)
        print(
            f"  macro_f1={metrics['macro_f1']:.3f}  "
            f"bal_acc={metrics['balanced_acc']:.3f}  "
            f"n={metrics['n_samples']}"
        )

        all_results["aids"][aid] = {
            **metrics,
            "raw_responses": raw_responses,
        }

        if wandb_run is not None:
            aid_key = aid.lower().replace(" ", "_")
            wandb_run.summary[f"{aid_key}/macro_f1"]    = round(metrics["macro_f1"],     4)
            wandb_run.summary[f"{aid_key}/balanced_acc"] = round(metrics["balanced_acc"], 4)

    # ── Overall ──────────────────────────────────────────────────────────────
    if all_results["aids"]:
        vals = all_results["aids"].values()
        all_results["overall"] = {
            "macro_f1_mean_across_aids":  float(np.mean([v["macro_f1"]     for v in vals])),
            "balanced_acc_mean_across_aids": float(np.mean([v["balanced_acc"] for v in vals])),
            "parse_failures": parse_failures,
        }
        ov = all_results["overall"]
        print(
            f"\nOverall — macro_f1={ov['macro_f1_mean_across_aids']:.3f}  "
            f"bal_acc={ov['balanced_acc_mean_across_aids']:.3f}  "
            f"parse_failures={parse_failures}"
        )
        if wandb_run is not None:
            wandb_run.summary["overall/macro_f1"]    = round(ov["macro_f1_mean_across_aids"],     4)
            wandb_run.summary["overall/balanced_acc"] = round(ov["balanced_acc_mean_across_aids"], 4)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = output_dir / "zero_shot_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
