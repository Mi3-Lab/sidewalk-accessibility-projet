#!/usr/bin/env bash
# Latency benchmark — all 8 encoders + 3 VLMs on A100.
#
# Usage:
#   ./run_latency.sh                  # encoders + VLMs
#   ENCODERS_ONLY=1 ./run_latency.sh  # encoders only (faster)
#   VLMS_ONLY=1     ./run_latency.sh  # VLMs only

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

IMAGES="data/images/sidewalk-images"
OUTDIR="results/latency"
LOG="$OUTDIR/latency.log"

mkdir -p "$OUTDIR"

EXTRA_FLAGS=""
[[ "${ENCODERS_ONLY:-0}" == "1" ]] && EXTRA_FLAGS="--encoders_only"
[[ "${VLMS_ONLY:-0}"     == "1" ]] && EXTRA_FLAGS="--vlms_only"

echo "========================================="
echo " Latency benchmark — $(date)"
echo "========================================="

$PYTHON src/models/latency_benchmark.py \
    --images_dir "$IMAGES" \
    --output_dir "$OUTDIR" \
    $EXTRA_FLAGS \
    2>&1 | tee "$LOG"

echo ""
echo "========================================="
echo " Done — results/latency/latency_results.json"
echo "========================================="
