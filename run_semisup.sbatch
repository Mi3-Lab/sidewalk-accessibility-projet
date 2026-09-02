#!/usr/bin/env bash
#SBATCH --job-name=semisup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/semisup_%j.out
#SBATCH --error=logs/semisup_%j.err

# Semi-supervised expansion beyond the 52 labelled panoramas.
#
# Each config: train on real votes, pseudo-label the 484-image Project Sidewalk
# pool, filter by confidence + agreement with the independent PS annotator
# label, retrain, and score on held-out REAL votes. Evaluation never touches
# the pseudo-labels, so the experiment does not repeat the circularity the
# reviewers flagged in the routing evaluation.
#
# Configs 0-2  main comparison across encoders
# Configs 3-7  entropy-threshold ablation on the paper's main encoder, so the
#              filter strictness is an ablation rather than an arbitrary constant
# Config  8    drops the PS-label filter, to show whether it does any work
#
# The cluster QOS caps submitted jobs per user, so this takes a slice of the 9
# configs rather than running a 9-task array.
#
# Submit, e.g. 2 slices:
#   for c in 0 1; do
#     sbatch --partition=cenvalarc.gpu --export=ALL,CHUNK=$c,NCHUNKS=2 run_semisup.sbatch
#   done

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
POOL="data/generalization/test_images.csv"
OUTROOT="results/semisup"
N_SEEDS="${N_SEEDS:-10}"

CHUNK="${CHUNK:-0}"
NCHUNKS="${NCHUNKS:-1}"

# config: "<encoder>|<extra args>|<output subdir>"
CONFIGS=(
    "dinov2-large||dinov2-large"
    "clip-vit-b32||clip-vit-b32"
    "siglip2-so400m||siglip2-so400m"
    "dinov2-large|--max_entropy 0.30|ablation_entropy/ent0.30"
    "dinov2-large|--max_entropy 0.45|ablation_entropy/ent0.45"
    "dinov2-large|--max_entropy 0.60|ablation_entropy/ent0.60"
    "dinov2-large|--max_entropy 0.75|ablation_entropy/ent0.75"
    "dinov2-large|--max_entropy 0.90|ablation_entropy/ent0.90"
    "dinov2-large|--no_ps_consistency|ablation_no_ps_filter"
)

echo "========================================="
echo " Semisup — chunk $CHUNK/$NCHUNKS"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)  seeds $N_SEEDS"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for i in "${!CONFIGS[@]}"; do
    if (( i % NCHUNKS != CHUNK )); then
        continue
    fi

    IFS='|' read -r enc extra subdir <<< "${CONFIGS[$i]}"
    out="$OUTROOT/$subdir"
    mkdir -p "$out"

    echo ""
    echo ">>> [config $i: $enc ${extra:-defaults}] started $(date +%H:%M:%S)"

    # shellcheck disable=SC2086  # $extra is an intentional argument list
    $PYTHON src/models/semisup_expand.py \
        --tallies_json "$TALLIES" \
        --images_dir   "$IMAGES" \
        --pool_csv     "$POOL" \
        --encoder      "$enc" \
        --n_folds      5 \
        --n_seeds      "$N_SEEDS" \
        --output_dir   "$out" \
        $extra \
        2>&1 | tee "$out/run.log"

    echo ">>> [config $i] done $(date +%H:%M:%S)"
done

echo ""
echo "CHUNK $CHUNK DONE — $(date)"
