#!/bin/bash
#SBATCH --job-name=generalization_task57
#SBATCH --partition=gpu           # Ajuste conforme seu cluster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/generalization_%j.log
#SBATCH --error=logs/generalization_%j.err

# Task 5.7 Generalization Test — SLURM Job Submission
# 
# Usage:
#   sbatch src/generalization/run_generalization_slurm.sh
#
# Or with parameters:
#   sbatch --job-name=gen_test_sf src/generalization/run_generalization_slurm.sh
#
# Check status:
#   squeue -u $USER
#   tail -f logs/generalization_*.log

set -e

# Configuration
ENCODER="${ENCODER:-dinov2-large}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/models/dinov2-large}"
PROJECT_DIR="/home/wesleyferreiramaia/data/sidewalk-accessibility-project"

# Ensure logs directory exists
mkdir -p "${PROJECT_DIR}/logs"

echo "════════════════════════════════════════════════════════════════"
echo "TASK 5.7 GENERALIZATION TEST — SLURM JOB"
echo "════════════════════════════════════════════════════════════════"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo ""

# Change to project directory
cd "${PROJECT_DIR}"

# Load environment (adjust as needed for your cluster)
module load cuda/11.8 2>/dev/null || true
module load python/3.11 2>/dev/null || true

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install W&B if not present
pip install -q wandb 2>/dev/null || true

# Set W&B offline mode (optional, for offline clusters)
# export WANDB_MODE=offline

# Run the generalization test
echo "Running generalization evaluation with W&B tracking..."
echo ""

python -u src/generalization/evaluate_generalization.py \
    --encoder "${ENCODER}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --use_wandb \
    --wandb_project "sidewalk-generalization" \
    --wandb_run_name "gen_${ENCODER}_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Job completed at: $(date)"
echo "Log file: logs/generalization_${SLURM_JOB_ID}.log"
echo "════════════════════════════════════════════════════════════════"
