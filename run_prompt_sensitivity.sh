#!/usr/bin/env bash
# Prompt sensitivity analysis — 3 prompt variants × 3 VLMs.
#
# Usage:
#   ./run_prompt_sensitivity.sh                      # all models × all variants
#   MODEL=qwen3-vl-8b ./run_prompt_sensitivity.sh    # one model, all variants
#   VARIANT=direct ./run_prompt_sensitivity.sh        # all models, one variant
#
# Results: results/prompt_sensitivity/<model>/<variant>/zero_shot_results.json
# Summary: results/prompt_sensitivity/summary.json

set -euo pipefail

PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
OUTBASE="results/prompt_sensitivity"
BATCH_SIZE="${BATCH_SIZE:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-sidewalk-accessibility}"

# Which models and variants to run
MODELS=("llava-1.5-7b" "qwen2.5-vl-7b" "qwen3-vl-8b")
VARIANTS=("direct" "descriptive" "cot")

# Allow env-var override for single model or single variant
[[ -n "${MODEL:-}" ]]   && MODELS=("$MODEL")
[[ -n "${VARIANT:-}" ]] && VARIANTS=("$VARIANT")

echo "========================================="
echo " Prompt sensitivity — $(date)"
echo " Models: ${MODELS[*]}"
echo " Variants: ${VARIANTS[*]}"
echo "========================================="

for MODEL_KEY in "${MODELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        OUTDIR="$OUTBASE/$MODEL_KEY/$VARIANT"

        # Skip if already done
        if [[ -f "$OUTDIR/zero_shot_results.json" ]]; then
            echo "[skip] $MODEL_KEY / $VARIANT — already exists"
            continue
        fi

        echo ""
        echo "-----------------------------------------"
        echo " $MODEL_KEY  |  variant=$VARIANT"
        echo "-----------------------------------------"

        mkdir -p "$OUTDIR"
        LOG="$OUTDIR/run.log"

        # Qwen3-VL uses batch_size=1 (sequential inference)
        BS=$BATCH_SIZE
        [[ "$MODEL_KEY" == "qwen3-vl-8b" ]] && BS=1

        $PYTHON src/models/zero_shot.py \
            --tallies_json  "$TALLIES" \
            --images_dir    "$IMAGES" \
            --model         "$MODEL_KEY" \
            --output_dir    "$OUTDIR" \
            --batch_size    "$BS" \
            --prompt_variant "$VARIANT" \
            --wandb_project "$WANDB_PROJECT" \
            2>&1 | tee "$LOG"

        echo "  → $OUTDIR/zero_shot_results.json"
    done
done

echo ""
echo "========================================="
echo " Generating summary …"
echo "========================================="

$PYTHON src/models/summarize_prompt_sensitivity.py \
    --results_dir "$OUTBASE" \
    --output "$OUTBASE/summary.json"

echo ""
echo "========================================="
echo " Done — $OUTBASE/"
echo "========================================="
