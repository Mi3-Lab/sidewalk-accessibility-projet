#!/usr/bin/env bash
#SBATCH --job-name=masked-feat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/masked_feat_%j.out
#SBATCH --error=logs/masked_feat_%j.err

# Does the probe do better when it sees the sidewalk instead of the whole scene?
#
# Motivation: mask coverage measured on the 52 training images averages 0.45, so
# roughly 55% of what the probe currently receives is sky, buildings, vehicles and
# vegetation. Two failures point at that mattering:
#   - geographic transfer collapses to chance under an encoder swap, which is what
#     scene-appearance shortcutting would look like
#   - entropy correlation with human disagreement is ~0.05
#
# Features are precomputed once per encoder and stored with both variants in the
# same file, so the whole-image and mask-pooled runs differ in nothing except
# which array they read.
#
# Submit:
#   sbatch --partition=gpu --export=ALL,ENCODER=dinov2-large run_masked_features.sbatch

set -euo pipefail

PYTHON="/data/wesleyferreiramaia/sidewalk-accessibility-project/venv/bin/python"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

ENCODER="${ENCODER:-dinov2-large}"
TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
FEAT="results/features/${ENCODER}_masked.npz"
OUTROOT="results/masked_cv/$ENCODER"
N_SEEDS="${N_SEEDS:-10}"
WD="${WD:-1e-4}"

echo "========================================="
echo " Mask-pooled features — $ENCODER  seeds $N_SEEDS"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── 1. Precompute both feature variants, training images + transfer cities ───
if [[ ! -f "$FEAT" ]]; then
    echo ""
    echo ">>> Extracting features ($ENCODER)"
    mkdir -p results/features
    $PYTHON src/models/masked_features.py \
        --images_dir "$IMAGES" \
        --encoder    "$ENCODER" \
        --extra_dirs data/generalization/images \
        --output     "$FEAT"
else
    echo ">>> Features already present: $FEAT"
fi

# ── 2. Same folds, same seeds, only the feature definition differs ───────────
for key in whole masked; do
    for loss in soft_kl hard_ce; do
        echo ""
        echo ">>> [$key / $loss] $(date +%H:%M:%S)"
        mkdir -p "$OUTROOT/$key/$loss"
        $PYTHON src/models/crossval.py \
            --encoder       "$ENCODER" \
            --tallies_json  "$TALLIES" \
            --images_dir    "$IMAGES" \
            --features_npz  "$FEAT" \
            --feature_key   "$key" \
            --output_dir    "$OUTROOT/$key/$loss" \
            --n_folds 5 --seed 42 --n_seeds "$N_SEEDS" \
            --loss_type     "$loss" \
            --wandb_project "" \
            2>&1 | tee "$OUTROOT/$key/${loss}.log"
    done
done

echo ""
echo "DONE — $(date)"
