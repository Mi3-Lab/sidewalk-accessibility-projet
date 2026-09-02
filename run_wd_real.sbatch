#!/usr/bin/env bash
#SBATCH --job-name=wd-real
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/wd_real_%j.out
#SBATCH --error=logs/wd_real_%j.err

# Confirm the weight-decay result in the *real* pipeline.
#
# The sweep that showed the 3.2x ratio to be an artefact of an untuned
# hyperparameter was run in the coupled-probe harness (a reimplementation). It
# reproduced the published 3.2x at wd=1e-4, which is reassuring, but the numbers
# that go in a paper have to come from crossval.py itself: per-aid
# StratifiedKFold, the real training loop, all 8 encoders.
#
# The first pass covered 1e-4 / 3e-1 / 1e1 and showed the real pipeline does not
# share the harness's optima: Soft-KL is flat between 3e-1 and 1e1 (0.0416 vs
# 0.0407) and Hard-CE reaches only 0.0830 at 1e1, against 0.0515 in the harness.
# Hard-CE's optimum is therefore not yet bracketed, so the fair tuned-vs-tuned
# ratio cannot be stated from the real pipeline yet. This pass extends the grid
# upward (3e0, 3e1, 1e2) to bracket it.
#
# The number to read off is best Soft-KL against best Hard-CE, each at its own
# best setting. Do not quote the harness's 1.4x: that came from the
# reimplementation, and the real pipeline's first pass already puts the ratio
# nearer 2x.
#
# Submit:
#   sbatch --partition=gpu --export=ALL,CHUNK=0,NCHUNKS=1 run_wd_real.sbatch

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
OUTROOT="results/cv_wd_real"   # same tree; new wd values
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

echo "========================================="
echo " Weight decay in the real pipeline — chunk $CHUNK/$NCHUNKS  seeds $N_SEEDS"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

i=0
for wd in 3e1 1e2 3e0; do
    for loss in soft_kl hard_ce; do
        for enc in "${ENCODERS[@]}"; do
            if (( i % NCHUNKS == CHUNK )); then
                out="$OUTROOT/wd$wd/$loss/$enc"
                mkdir -p "$out"
                echo ""
                echo ">>> [wd=$wd $loss $enc] $(date +%H:%M:%S)"
                $PYTHON src/models/crossval.py \
                    --encoder       "$enc" \
                    --tallies_json  "$TALLIES" \
                    --images_dir    "$IMAGES" \
                    --output_dir    "$out" \
                    --n_folds 5 --seed 42 --n_seeds "$N_SEEDS" \
                    --loss_type     "$loss" \
                    --weight_decay  "$wd" \
                    --wandb_project "" \
                    > "$out/run.log" 2>&1
                tail -3 "$out/run.log" | head -2
            fi
            i=$((i + 1))
        done
    done
done

echo ""
echo "CHUNK $CHUNK DONE — $(date)"
