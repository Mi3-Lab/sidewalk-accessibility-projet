#!/usr/bin/env bash
#SBATCH --job-name=routing-enc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/routing_%j.out
#SBATCH --error=logs/routing_%j.err

# Re-run the routing pipeline under a chosen encoder.
#
# Why: DINOv2-large was picked for every routing experiment because it led
# Soft-KL at seed 42. Over 10 seeds it is the *worst* Soft-KL encoder
# (F1 0.475 ± 0.054 vs 0.613 published), so that justification is gone.
#
# SigLIP2-SO400M is used here, but note it is NOT a clear winner: per-seed it
# takes the best Brier 3/10 times against CLIP-L/14's 4/10, and the paired
# difference is +0.0005 ± 0.0150. The top encoders are statistically tied, and
# all encoder-level gaps (~0.005) sit an order of magnitude below the binomial
# noise on the training target itself (95% CI ±0.068). Treat this as "one member
# of the tied group", not "the best encoder".
#
# Results go to results/routing_<encoder>/, leaving the published
# results/routing/ (DINOv2-large) untouched so the two can be compared.
#
# The PS-label edge scores (step 2) do not depend on the encoder, so the
# existing edge_scores.json is reused. The GSV thumbnails are cached on disk,
# so both encoders score the same pixels and the comparison is not confounded
# by panoramas expiring between runs.
#
# Submit:
#   sbatch --partition=gpu --export=ALL,ENCODER=siglip2-so400m run_routing_encoder.sbatch

set -euo pipefail

# The routing pipeline needs osmnx, which only the repo venv has. The venv also
# carries the newer torch/transformers the routing code was written against, so
# the whole pipeline runs on one stack rather than mixing interpreters.
PYTHON="/data/wesleyferreiramaia/sidewalk-accessibility-project/venv/bin/python"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

ENCODER="${ENCODER:-siglip2-so400m}"
LOSS_TYPE="${LOSS_TYPE:-soft_kl}"
TALLIES="data/processed/tallies_firebase.json"
IMAGES="data/images/sidewalk-images"
MODEL_DIR="results/models/$ENCODER"
OUTDIR="results/routing_$ENCODER"
PS_LABELS="data/routing/pittsburgh/ps_labels.geojson"
BASE_EDGES="results/routing/edge_scores.json"     # PS-label scores, encoder-independent
FULL_EDGES="$OUTDIR/edge_scores_full.json"
# The 956 graph (2046 nodes / 2618 edges, centred on -79.9551) is the one the
# paper's numbers come from, and the one the edge_scores files are keyed to.
# pittsburgh_graph.graphml is a different, smaller Oakland extract (1920 edges);
# pairing it with these scores silently drops most edges to constant priors.
GRAPH_CACHE="results/routing/pittsburgh_graph_956.graphml"
LAT=40.444
LON=-79.956
DIST=800
SEED=42
N_PAIRS="${N_PAIRS:-10000}"

mkdir -p "$OUTDIR"

echo "========================================="
echo " Routing re-run — encoder: $ENCODER"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── 1. Final per-aid probes on all 52 scenes ─────────────────────────────────
echo ""
echo ">>> Step 1: final probes ($ENCODER, $LOSS_TYPE)"
$PYTHON src/models/train_final_model.py \
    --encoder      "$ENCODER" \
    --tallies_json "$TALLIES" \
    --images_dir   "$IMAGES" \
    --output_dir   "$MODEL_DIR" \
    --loss_type    "$LOSS_TYPE" \
    --seed         "$SEED"

# ── 2. Fill prior edges with model inference on cached GSV thumbnails ────────
echo ""
echo ">>> Step 2: GSV edge scoring ($ENCODER)"
$PYTHON src/routing/score_gsv_edges.py \
    --edge_scores    "$BASE_EDGES" \
    --ps_labels      "$PS_LABELS" \
    --checkpoint     "$MODEL_DIR" \
    --encoder        "$ENCODER" \
    --lat "$LAT" --lon "$LON" --dist "$DIST" \
    --pano_cache_dir cache/panos \
    --output         "$FULL_EDGES" \
    --seed           "$SEED"

# ── 3. Per-aid routes for the canonical OD pair ──────────────────────────────
echo ""
echo ">>> Step 3: routing demo"
$PYTHON src/routing/demo.py \
    --edge_scores "$FULL_EDGES" \
    --cv_results  "results/cv_multiseed/$LOSS_TYPE/$ENCODER/cv_results.json" \
    --checkpoint  "$MODEL_DIR" \
    --encoder     "$ENCODER" \
    --graph_cache "$GRAPH_CACHE" \
    --lat "$LAT" --lon "$LON" --dist "$DIST" \
    --output_dir  "$OUTDIR" \
    --seed        "$SEED"

# ── 4. Monte Carlo over random OD pairs ──────────────────────────────────────
echo ""
echo ">>> Step 4: Monte Carlo ($N_PAIRS OD pairs)"
$PYTHON src/routing/monte_carlo.py \
    --graph_cache  "$GRAPH_CACHE" \
    --edge_scores  "$FULL_EDGES" \
    --n_pairs      "$N_PAIRS" \
    --barrier_cost 8 \
    --min_dist     200 \
    --seed         "$SEED" \
    --output_dir   "$OUTDIR/monte_carlo"

echo ""
echo "========================================="
echo " DONE — $ENCODER — $(date)"
echo " Results → $OUTDIR/"
echo "========================================="
