#!/usr/bin/env bash
# Evaluate trained probes on out-of-distribution cities.
#
# Prerequisites:
#   1. Trained model checkpoint in results/models/<encoder>/
#      (run run_routing_demo.sh or train_final_model.py first)
#   2. data/generalization/test_images.csv populated with image paths + ps_label
#      (use src/generalization/download_city_images.py to create it)
#
# Usage:
#   ./run_generalization.sh [encoder] [checkpoint_dir]
#   ./run_generalization.sh dinov2-large results/models/dinov2-large
#   ./run_generalization.sh clip-vit-b32 results/models/clip-vit-b32

set -euo pipefail

ENCODER="${1:-dinov2-large}"
CHECKPOINT_DIR="${2:-results/models/${ENCODER}}"
TEST_IMAGES="data/generalization/test_images.csv"
OUTPUT_DIR="results/generalization/${ENCODER}"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

echo "============================================"
echo " Generalization Evaluation — $(date)"
echo " Encoder:    $ENCODER"
echo " Checkpoint: $CHECKPOINT_DIR"
echo " Test CSV:   $TEST_IMAGES"
echo " Output:     $OUTPUT_DIR"
echo "============================================"

if [[ ! -f "$TEST_IMAGES" ]]; then
    echo "ERROR: $TEST_IMAGES not found."
    echo "Run src/generalization/download_city_images.py first."
    exit 1
fi

python src/generalization/evaluate_generalization.py \
    --encoder    "$ENCODER" \
    --checkpoint "$CHECKPOINT_DIR" \
    --test_images "$TEST_IMAGES" \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "Results → $OUTPUT_DIR/agreement_summary.json"
echo "          $OUTPUT_DIR/predictions.csv"
