#!/usr/bin/env bash
#SBATCH --job-name=downstream
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/downstream_%j.out
#SBATCH --error=logs/downstream_%j.err

# Re-run the remaining DINOv2-large-dependent results under a new encoder.
#
# Repeated-seed CV showed DINOv2-large's Soft-KL lead was a seed-42 artefact, so
# every downstream number computed with it has to be recomputed, not just the
# routing. That means:
#   - geographic transfer across 5 cities (including the Taipei near-chance result
#     the paper leans on as its headline limitation)
#   - the walking-cane error analysis behind contribution C2
#
# Expects results/models/<encoder>/ to exist — run_routing_encoder.sbatch step 1
# creates it, so submit this with a dependency on that job:
#
#   sbatch --partition=gpu --dependency=afterok:<routing_jobid> \
#          --export=ALL,ENCODER=siglip2-so400m run_downstream_encoder.sbatch

set -euo pipefail

# The routing pipeline needs osmnx, which only the repo venv has. The venv also
# carries the newer torch/transformers the routing code was written against, so
# the whole pipeline runs on one stack rather than mixing interpreters.
PYTHON="/data/wesleyferreiramaia/sidewalk-accessibility-project/venv/bin/python"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

ENCODER="${ENCODER:-siglip2-so400m}"
MODEL_DIR="results/models/$ENCODER"
TEST_IMAGES="data/generalization/test_images.csv"

echo "========================================="
echo " Downstream re-run — encoder: $ENCODER"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: $MODEL_DIR not found — the routing job trains it in step 1."
    exit 1
fi

# ── 1. Geographic transfer, 5 cities ─────────────────────────────────────────
echo ""
echo ">>> Step 1: generalization ($ENCODER)"
$PYTHON src/generalization/evaluate_generalization.py \
    --encoder     "$ENCODER" \
    --checkpoint  "$MODEL_DIR" \
    --test_images "$TEST_IMAGES" \
    --output_dir  "results/generalization_$ENCODER"

# ── 2. Walking-cane error analysis (contribution C2) ─────────────────────────
echo ""
echo ">>> Step 2: walking-cane error analysis ($ENCODER)"
$PYTHON src/models/error_analysis.py \
    --encoder    "$ENCODER" \
    --output_dir "results/error_analysis_$ENCODER"

echo ""
echo "========================================="
echo " DONE — $ENCODER — $(date)"
echo "========================================="
