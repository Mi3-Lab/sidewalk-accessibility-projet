#!/usr/bin/env bash
# Zero-shot VLM evaluation — A100 40GB, float16/bfloat16 sem quantização.
#
# Usage:
#   MODEL=llava-1.5-7b   ./run_zero_shot.sh
#   MODEL=llava-1.6-7b   ./run_zero_shot.sh
#   MODEL=qwen2.5-vl-7b  ./run_zero_shot.sh
#   MODEL=qwen3-vl-8b    ./run_zero_shot.sh   ← mais recente

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
OUTDIR="results/zero_shot"
WANDB_PROJECT="${WANDB_PROJECT:-sidewalk-accessibility}"
MODEL="${MODEL:-llava-1.5-7b}"

mkdir -p "$OUTDIR/$MODEL"
LOG="$OUTDIR/${MODEL}.log"

echo "========================================="
echo " Zero-shot eval — $MODEL"
echo " $(date)"
echo "========================================="

BATCH_SIZE="${BATCH_SIZE:-8}"

$PYTHON src/models/zero_shot.py \
    --tallies_json  "$TALLIES" \
    --images_dir    "$IMAGES" \
    --model         "$MODEL" \
    --output_dir    "$OUTDIR/$MODEL" \
    --wandb_project "$WANDB_PROJECT" \
    --batch_size    "$BATCH_SIZE" \
    2>&1 | tee "$LOG"

echo ""
echo "========================================="
echo " Done — $(date)"
echo " Results: $OUTDIR/$MODEL/zero_shot_results.json"
echo "========================================="
