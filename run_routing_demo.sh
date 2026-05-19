#!/usr/bin/env bash
# Full routing demo pipeline:
#   1. Train final per-aid probes on all data (DINOv2-large, soft-KL)
#   2. Run accessibility-aware routing demo on Pittsburgh/Oakland
#   3. Save figures to results/routing/
#
# Usage:
#   chmod +x run_routing_demo.sh
#   ./run_routing_demo.sh
#
# Requires: pip install osmnx

set -euo pipefail

# Use venv Python if active, otherwise fall back to system Python 3.11
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python3.11"
else
    PYTHON="/home/wesleyferreiramaia/python311/bin/python3.11"
fi
export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

ENCODER="dinov2-large"
LOSS_TYPE="soft_kl"
TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
MODEL_DIR="results/models/$ENCODER"
CV_RESULTS="results/cv/$LOSS_TYPE/$ENCODER/cv_results.json"
ROUTING_DIR="results/routing"
# Pittsburgh PA (Forbes/Murray corridor) — shifted west from Oakland to capture
# the densest NoCurbRamp label cluster (67 labels in bbox vs 32 at Oakland center),
# producing meaningfully different routes per mobility aid.
LAT=40.444
LON=-79.956
DIST=800
SEED=42

echo "========================================="
echo " Routing Demo Pipeline — $(date)"
echo " Encoder: $ENCODER  |  Loss: $LOSS_TYPE"
echo "========================================="

# ── Step 1: Train final checkpoints (full dataset) ────────────────────────────
echo ""
echo ">>> Step 1: Training final probes (all data, $ENCODER, $LOSS_TYPE)"
$PYTHON src/models/train_final_model.py \
    --encoder      "$ENCODER" \
    --tallies_json "$TALLIES" \
    --images_dir   "$IMAGES" \
    --output_dir   "$MODEL_DIR" \
    --loss_type    "$LOSS_TYPE" \
    --seed         "$SEED"

echo ">>> Step 1 done — checkpoints in $MODEL_DIR/"

# ── Step 1.5: Download PS labels and score OSM edges ─────────────────────────
PS_LABELS="data/routing/pittsburgh/ps_labels.geojson"
EDGE_SCORES="$ROUTING_DIR/edge_scores.json"

if [[ ! -f "$PS_LABELS" ]]; then
    echo ""
    echo ">>> Step 1.5a: Downloading Project Sidewalk Pittsburgh labels (no auth)"
    $PYTHON src/routing/fetch_ps_labels.py \
        --output "$PS_LABELS"
else
    echo ">>> Step 1.5a: PS labels already downloaded ($PS_LABELS)"
fi

echo ""
echo ">>> Step 1.5b: Scoring OSM edges from PS labels (${LAT},${LON} r=${DIST}m)"
$PYTHON src/routing/score_osm_edges.py \
    --ps_labels "$PS_LABELS" \
    --output    "$EDGE_SCORES" \
    --lat       "$LAT" \
    --lon       "$LON" \
    --dist      "$DIST"

# ── Step 2: Routing demo ──────────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Running routing demo (Pittsburgh PA — ${LAT},${LON} r=${DIST}m)"
$PYTHON src/routing/demo.py \
    --edge_scores "$EDGE_SCORES" \
    --cv_results "$CV_RESULTS" \
    --checkpoint "$MODEL_DIR" \
    --encoder    "$ENCODER" \
    --lat        "$LAT" \
    --lon        "$LON" \
    --dist       "$DIST" \
    --output_dir "$ROUTING_DIR" \
    --seed       "$SEED"

echo ""
echo "========================================="
echo " DONE — $(date)"
echo " Figures → $ROUTING_DIR/"
echo "========================================="
