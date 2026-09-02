#!/usr/bin/env bash
#SBATCH --job-name=wd-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/wd_sweep_%j.out
#SBATCH --error=logs/wd_sweep_%j.err

# Is the low-rank gain actually just regularisation?
#
# The coupled-probe run showed a rank-16 bottleneck cuts soft Brier by ~33%
# (0.0828 -> 0.0553 on DINOv2-large, 10/10 seeds), and the lowrank_peraid control
# showed essentially none of that comes from sharing across aids — it is capacity
# control. The pipeline trains a d=1024 -> 3 map on 52 points with weight_decay
# 1e-4, which is close to no regularisation at all.
#
# So the obvious question: does plain L2 at a sensible strength get the same
# thing, without any architectural change? If it does, the fix is one
# hyperparameter, not a new head — and, more seriously, the paper's whole
# 8-encoder comparison was run at an under-regularised operating point.
#
# Submit:
#   sbatch --partition=gpu --export=ALL,ENCODER=dinov2-large run_wd_sweep.sbatch

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
ENCODER="${ENCODER:-dinov2-large}"
OUTROOT="results/wd_sweep/$ENCODER"
N_SEEDS="${N_SEEDS:-10}"

echo "========================================="
echo " Weight-decay sweep — $ENCODER   seeds $N_SEEDS"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for wd in 1e-4 1e-3 1e-2 1e-1 1e0 1e1; do
    echo ""
    echo ">>> [weight_decay=$wd] $(date +%H:%M:%S)"
    mkdir -p "$OUTROOT/wd$wd"
    $PYTHON src/models/coupled_probe.py \
        --tallies_json  "$TALLIES" \
        --images_dir    "$IMAGES" \
        --encoder       "$ENCODER" \
        --n_folds 5 --n_seeds "$N_SEEDS" --rank 16 \
        --weight_decay  "$wd" \
        --output_dir    "$OUTROOT/wd$wd" \
        2>&1 | tee "$OUTROOT/wd${wd}.log"
done

echo ""
echo "DONE — $(date)"
