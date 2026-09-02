#!/usr/bin/env bash
#SBATCH --job-name=cv-ms
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cv_ms_%j.out
#SBATCH --error=logs/cv_ms_%j.err

# Repeated-seed panorama-level CV over all 8 encoders and both loss types.
#
# Answers the CoRL reviewer objection that a single seed=42 5-fold split over
# 52 panoramas cannot show statistical stability: each encoder is re-run over
# N_SEEDS independent fold partitions and we report mean±std across them.
#
# The cluster QOS caps submitted jobs per user (gpu=4), so instead of a 16-task
# array this script takes a slice of the 16 (loss, encoder) configs and runs
# them sequentially. Feature caching keeps each config to a few minutes.
#
# Writes to results/cv_multiseed/ so the published results/cv/ numbers stay put.
#
# Submit one slice per job, e.g. 4 slices across the gpu partition:
#   for c in 0 1 2 3; do
#     sbatch --partition=gpu --export=ALL,CHUNK=$c,NCHUNKS=4 run_cv_multiseed.sbatch
#   done
#
# Gather: python src/models/summarize_multiseed.py --results_dir results/cv_multiseed

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
RANK="${RANK:-0}"
# rank 0 keeps the published full-rank probe; a positive rank inserts the
# bottleneck and writes to a separate tree so both can be compared.
if [[ "$RANK" == "0" ]]; then
    OUTROOT="results/cv_multiseed"
else
    OUTROOT="results/cv_multiseed_rank$RANK"
fi
N_FOLDS=5
SEED=42
N_SEEDS="${N_SEEDS:-10}"

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
LOSSES=(soft_kl hard_ce)

echo "========================================="
echo " CV multiseed — chunk $CHUNK/$NCHUNKS"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
echo " seeds $SEED..$((SEED + N_SEEDS - 1))  folds $N_FOLDS  rank $RANK"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

i=0
for loss in "${LOSSES[@]}"; do
    for enc in "${ENCODERS[@]}"; do
        if (( i % NCHUNKS == CHUNK )); then
            echo ""
            echo ">>> [$loss / $enc] started $(date +%H:%M:%S)"
            mkdir -p "$OUTROOT/$loss/$enc"

            $PYTHON src/models/crossval.py \
                --encoder       "$enc" \
                --tallies_json  "$TALLIES" \
                --images_dir    "$IMAGES" \
                --output_dir    "$OUTROOT/$loss/$enc" \
                --n_folds       "$N_FOLDS" \
                --loss_type     "$loss" \
                --seed          "$SEED" \
                --n_seeds       "$N_SEEDS" \
                --rank          "$RANK" \
                --wandb_project "" \
                2>&1 | tee "$OUTROOT/$loss/${enc}.log"

            echo ">>> [$loss / $enc] done $(date +%H:%M:%S)"
        fi
        i=$((i + 1))
    done
done

echo ""
echo "CHUNK $CHUNK DONE — $(date)"
