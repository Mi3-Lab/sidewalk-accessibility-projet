#!/usr/bin/env bash
#SBATCH --job-name=temp-scale
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/temp_scaling_%j.out
#SBATCH --error=logs/temp_scaling_%j.err

# Repeated-seed temperature-scaled Hard-CE, all 8 encoders.
#
# The paper claims post-hoc temperature scaling cuts Hard-CE Brier by ~34% while
# leaving a >2x gap to Soft-KL. That was measured at seed=42 only, and the
# repeated-seed CV already showed one single-seed claim (DINOv2-large leading
# Soft-KL) fail to replicate. This checks whether the temperature-scaling claim
# holds up the same way.
#
# Writes to results/cv_multiseed_ts/, leaving results/cv/hard_ce_ts/ untouched.
#
# Submit:
#   sbatch --partition=gpu --export=ALL,CHUNK=0,NCHUNKS=1 run_temp_scaling.sbatch

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
OUTROOT="results/cv_multiseed_ts"
N_SEEDS="${N_SEEDS:-10}"
SEED=42

CHUNK="${CHUNK:-0}"
NCHUNKS="${NCHUNKS:-1}"

ENCODERS=(
    clip-vit-b32
    clip-vit-b16
    clip-vit-l14
    dinov2-base
    dinov2-large
    siglip2-base
    siglip2-so400m
    vit-b16-sup
)

echo "========================================="
echo " Temperature scaling multiseed — chunk $CHUNK/$NCHUNKS"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
echo " seeds $SEED..$((SEED + N_SEEDS - 1))"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for i in "${!ENCODERS[@]}"; do
    if (( i % NCHUNKS != CHUNK )); then
        continue
    fi
    enc="${ENCODERS[$i]}"
    echo ""
    echo ">>> [$enc] started $(date +%H:%M:%S)"
    mkdir -p "$OUTROOT/$enc"

    $PYTHON src/models/temperature_scaling.py \
        --tallies_json "$TALLIES" \
        --images_dir   "$IMAGES" \
        --encoder      "$enc" \
        --n_folds      5 \
        --seed         "$SEED" \
        --n_seeds      "$N_SEEDS" \
        --output_dir   "$OUTROOT/$enc" \
        2>&1 | tee "$OUTROOT/${enc}.log"

    echo ">>> [$enc] done $(date +%H:%M:%S)"
done

echo ""
echo "CHUNK $CHUNK DONE — $(date)"
