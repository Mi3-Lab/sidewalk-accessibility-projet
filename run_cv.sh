#!/usr/bin/env bash
# Run 5-fold panorama-level cross-validation for all 8 encoders.
# Logs are saved per encoder + combined log.
#
# Usage:
#   chmod +x run_cv.sh
#   ./run_cv.sh
#
# Results are saved to:
#   results/cv/<encoder>/cv_results.json   ← machine-readable
#   results/cv/<encoder>.log               ← per-encoder log
#   results/cv_run.log                     ← combined log

set -euo pipefail

# Python 3.11 with torch 2.9.1+cu128
PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

# Redirect HuggingFace model cache to /data (avoid home quota)
export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
OUTDIR="results/cv"
COMBINED_LOG="results/cv_run.log"
N_FOLDS=5

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

# Loss type: "soft_kl" (paper method) or "hard_ce" (ablation baseline)
# Override via env var:  LOSS_TYPE=hard_ce ./run_cv.sh
LOSS_TYPE="${LOSS_TYPE:-soft_kl}"

# W&B project name — set to "" to disable wandb tracking
WANDB_PROJECT="${WANDB_PROJECT:-sidewalk-accessibility}"

mkdir -p "$OUTDIR/$LOSS_TYPE"
echo "" > "$COMBINED_LOG"
echo "=========================================" | tee -a "$COMBINED_LOG"
echo " CoRL CV run — $(date)"                   | tee -a "$COMBINED_LOG"
echo " Folds: $N_FOLDS"                         | tee -a "$COMBINED_LOG"
echo " Encoders: ${ENCODERS[*]}"                | tee -a "$COMBINED_LOG"
echo "=========================================" | tee -a "$COMBINED_LOG"

for enc in "${ENCODERS[@]}"; do
    echo ""                                                    | tee -a "$COMBINED_LOG"
    echo ">>> [$enc] started at $(date +%H:%M:%S)"            | tee -a "$COMBINED_LOG"

    $PYTHON src/models/crossval.py \
        --encoder        "$enc" \
        --tallies_json   "$TALLIES" \
        --images_dir     "$IMAGES" \
        --output_dir     "$OUTDIR/$LOSS_TYPE/$enc" \
        --n_folds        "$N_FOLDS" \
        --loss_type      "$LOSS_TYPE" \
        --wandb_project  "$WANDB_PROJECT" \
        2>&1 | tee "$OUTDIR/$LOSS_TYPE/${enc}.log" | tee -a "$COMBINED_LOG"

    echo ">>> [$enc] done at $(date +%H:%M:%S)"               | tee -a "$COMBINED_LOG"
done

echo ""                                                        | tee -a "$COMBINED_LOG"
echo "========================================="                       | tee -a "$COMBINED_LOG"
echo " ALL ENCODERS DONE — $(date)"                                   | tee -a "$COMBINED_LOG"
echo " Loss type: $LOSS_TYPE"                                         | tee -a "$COMBINED_LOG"
echo " Run summary:"                                                   | tee -a "$COMBINED_LOG"
$PYTHON src/models/summarize_cv.py --results_dir "$OUTDIR/$LOSS_TYPE" | tee -a "$COMBINED_LOG"
echo "========================================="                       | tee -a "$COMBINED_LOG"

# ── Upload all results to ONE W&B run ─────────────────────────────────────────
if [[ -n "$WANDB_PROJECT" ]]; then
    echo ""                                                            | tee -a "$COMBINED_LOG"
    echo ">>> Uploading results to W&B project: $WANDB_PROJECT"       | tee -a "$COMBINED_LOG"
    $PYTHON src/models/log_wandb.py \
        --results_dir  "$OUTDIR/$LOSS_TYPE" \
        --loss_type    "$LOSS_TYPE" \
        --wandb_project "$WANDB_PROJECT" \
        2>&1 | tee -a "$COMBINED_LOG"
fi
