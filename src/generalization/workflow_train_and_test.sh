#!/bin/bash
#SBATCH --job-name=gen_workflow
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/workflow_%j.log
#SBATCH --error=logs/workflow_%j.err

# Complete Workflow: Train Final Model + Run Generalization Test
# 
# Task 5.7: Generate checkpoints and evaluate on test images
# Follows paper methodology exactly (Section 3)
#
# Usage:
#   sbatch src/generalization/workflow_train_and_test.sh
#
# Logs:
#   logs/workflow_*.log

set -e

PROJECT_DIR="/home/wesleyferreiramaia/data/sidewalk-accessibility-project"
ENCODER="${ENCODER:-dinov2-large}"

echo "════════════════════════════════════════════════════════════════"
echo "WORKFLOW: Train Final Model + Generalization Test"
echo "════════════════════════════════════════════════════════════════"
echo "Job ID: $SLURM_JOB_ID"
echo "Encoder: $ENCODER"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd "${PROJECT_DIR}"

# Create logs directory
mkdir -p logs

# ─── STEP 1: Train Final Model ───────────────────────────────────────
echo "[STEP 1] Training final model checkpoints..."
echo "─────────────────────────────────────────────────────────────────"

python -u src/models/train_final_model.py \
    --encoder "${ENCODER}" \
    --output_dir "results/models/${ENCODER}" \
    --loss_type soft_kl \
    --seed 42

if [ ! -d "results/models/${ENCODER}/walking_cane" ]; then
    echo "❌ Training failed: checkpoints not created"
    exit 1
fi

echo "✅ Step 1 complete"
echo ""

# ─── STEP 2: Run Generalization Test ─────────────────────────────────
echo "[STEP 2] Running generalization evaluation..."
echo "─────────────────────────────────────────────────────────────────"

python -u src/generalization/evaluate_generalization.py \
    --encoder "${ENCODER}" \
    --checkpoint "results/models/${ENCODER}" \
    --test_images "data/generalization/test_images.csv" \
    --output_dir "results/generalization/${ENCODER}" \
    --use_wandb \
    --wandb_project "sidewalk-generalization" \
    --wandb_run_name "complete_workflow_${ENCODER}_$(date +%Y%m%d_%H%M%S)"

echo "✅ Step 2 complete"
echo ""

# ─── SUMMARY ─────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "✅ WORKFLOW COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  • results/models/${ENCODER}/ (checkpoints)"
echo "  • results/generalization/${ENCODER}/ (predictions)"
echo ""
echo "Next: Check predictions.csv and agreement_summary.json"
echo ""
