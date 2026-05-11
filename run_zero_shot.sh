#!/usr/bin/env bash
# Zero-shot VLM evaluation — A100 40GB, float16/bfloat16 sem quantização.
#
# Usage:
#   MODEL=llava-1.5-7b          ./run_zero_shot.sh
#   MODEL=qwen2.5-vl-7b         ./run_zero_shot.sh
#   MODEL=qwen3-vl-8b           ./run_zero_shot.sh   (~16GB VRAM)
#   MODEL=qwen3-vl-8b-thinking  ./run_zero_shot.sh   (~16GB VRAM, gera <think> interno)
#   MODEL=qwen3-vl-32b          ./run_zero_shot.sh   (~64GB VRAM — precisa 2×A100)
#   MODEL=qwen3.6-27b           ./run_zero_shot.sh   (~54GB VRAM — precisa 2×A100)
#   MODEL=qwen3.6-27b-fp8       ./run_zero_shot.sh   (~27GB VRAM — 1×A100, quase igual ao bf16)

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

# Qwen3-VL and Qwen3.6 variants run one image at a time (processor API limitation)
BATCH_SIZE="${BATCH_SIZE:-8}"
if [[ "$MODEL" == qwen3-vl-* || "$MODEL" == qwen3.* ]]; then
    BATCH_SIZE=1
fi

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
