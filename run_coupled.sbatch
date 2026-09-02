#!/usr/bin/env bash
#SBATCH --job-name=coupled
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/coupled_%j.out
#SBATCH --error=logs/coupled_%j.err

# Count-based losses and cross-aid coupling, on several encoders.
#
# Two properties of this dataset the current setup ignores:
#   - each target is estimated from 141-240 votes (binomial SE ~0.035), yet the
#     loss treats the empirical distribution as exact and re-weights by an ad-hoc
#     entropy term. The count-based losses use the votes directly.
#   - the five aids correlate at 0.886, yet five independent probes are trained
#     on 52 points each. The lowrank head shares a subspace across aids.
#
# The `lowrank_peraid` head is the control: same bottleneck, no sharing. Without
# it a lowrank win is ambiguous between "cross-aid transfer helps" and "less
# capacity helps on 52 points", and getting that distinction wrong is how the
# DINOv2-large claim ended up in the paper.
#
# Submit:
#   sbatch --partition=gpu --export=ALL,CHUNK=0,NCHUNKS=1 run_coupled.sbatch

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
OUTROOT="results/coupled"
N_SEEDS="${N_SEEDS:-10}"
RANK="${RANK:-16}"

CHUNK="${CHUNK:-0}"
NCHUNKS="${NCHUNKS:-1}"

ENCODERS=(
    dinov2-large
    clip-vit-b32
    siglip2-so400m
    clip-vit-l14
)

echo "========================================="
echo " Coupled probe — chunk $CHUNK/$NCHUNKS   rank $RANK   seeds $N_SEEDS"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
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

    $PYTHON src/models/coupled_probe.py \
        --tallies_json "$TALLIES" \
        --images_dir   "$IMAGES" \
        --encoder      "$enc" \
        --n_folds 5 --n_seeds "$N_SEEDS" --rank "$RANK" \
        --output_dir   "$OUTROOT/$enc" \
        2>&1 | tee "$OUTROOT/${enc}.log"

    echo ">>> [$enc] done $(date +%H:%M:%S)"
done

# Rank sweep on one encoder: is the gain specific to a bottleneck size?
if (( CHUNK == 0 )); then
    for r in 4 8 32 64; do
        echo ""
        echo ">>> [dinov2-large rank=$r] $(date +%H:%M:%S)"
        mkdir -p "$OUTROOT/rank_sweep/r$r"
        $PYTHON src/models/coupled_probe.py \
            --tallies_json "$TALLIES" \
            --images_dir   "$IMAGES" \
            --encoder      dinov2-large \
            --n_folds 5 --n_seeds "$N_SEEDS" --rank "$r" \
            --output_dir   "$OUTROOT/rank_sweep/r$r" \
            2>&1 | tee "$OUTROOT/rank_sweep/r${r}.log"
    done
fi

echo ""
echo "CHUNK $CHUNK DONE — $(date)"
