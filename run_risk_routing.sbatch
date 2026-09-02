#!/usr/bin/env bash
#SBATCH --job-name=risk-route
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=logs/risk_route_%j.out
#SBATCH --error=logs/risk_route_%j.err

# Does calibration actually reach the planner?
#
# The paper argues calibration is the operative metric "because the routing
# objective is linear in p_yes". But a linear function of one scalar is the
# weakest possible consumer of a calibrated distribution: it cannot tell a
# genuinely split scene ([0.45, 0.05, 0.45]) from a confidently middling one
# ([0.10, 0.45, 0.45]) — and that bimodality is exactly what soft-label training
# exists to preserve. Worse, edge scoring stored only p_yes, so p_no and
# p_unsure were discarded before the planner ever saw them.
#
# Step 1 regenerates edge scores keeping the full three-class distribution, for
# both the Soft-KL and Hard-CE checkpoints. Step 2 routes the same OD pairs under
# four objectives and measures how much the two score sets disagree.
#
# The prediction that makes the paper's thesis testable rather than asserted:
# under `linear` the Soft-KL and Hard-CE routes should agree closely, because the
# objective only needs the mean roughly right; under the risk objectives they
# should diverge, because tail behaviour depends on the whole distribution.
# If they agree everywhere, calibration does not reach the planner and the
# paper's causal story does not hold.
#
# Submit:
#   sbatch --partition=gpu run_risk_routing.sbatch

set -euo pipefail

PYTHON="/data/wesleyferreiramaia/sidewalk-accessibility-project/venv/bin/python"

export HF_HOME="/data/wesleyferreiramaia/.cache/huggingface"
export HF_HUB_CACHE="/data/wesleyferreiramaia/.cache/huggingface/hub"

cd /data/wesleyferreiramaia/sidewalk-accessibility-project

OUT="results/risk_routing"
BASE_EDGES="results/routing/edge_scores.json"
PS_LABELS="data/routing/pittsburgh/ps_labels.geojson"
GRAPH="results/routing/pittsburgh_graph_956.graphml"
LAT=40.444
LON=-79.956
DIST=800
N_PAIRS="${N_PAIRS:-2000}"

mkdir -p "$OUT"

echo "========================================="
echo " Risk-sensitive routing — $(date)"
echo " node $(hostname)  job ${SLURM_JOB_ID:-none}"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── 1. Edge scores that keep [p_no, p_unsure, p_yes] ─────────────────────────
# The GSV thumbnails are on disk from the earlier run, so both checkpoints score
# identical pixels and any divergence comes from the model, not from panoramas
# expiring between runs.
for variant in soft:results/models/dinov2-large hard:results/models/dinov2-large-hard-ce; do
    tag="${variant%%:*}"
    ckpt="${variant##*:}"
    if [[ ! -d "$ckpt" ]]; then
        echo "SKIP $tag — checkpoint missing: $ckpt"
        continue
    fi
    if [[ -f "$OUT/edge_scores_${tag}_dist.json" ]]; then
        echo ">>> $tag edge scores already present"
        continue
    fi
    echo ""
    echo ">>> Step 1 [$tag]: edge scoring with full distributions"
    $PYTHON src/routing/score_gsv_edges.py \
        --edge_scores    "$BASE_EDGES" \
        --ps_labels      "$PS_LABELS" \
        --checkpoint     "$ckpt" \
        --encoder        dinov2-large \
        --lat "$LAT" --lon "$LON" --dist "$DIST" \
        --pano_cache_dir cache/panos \
        --output         "$OUT/edge_scores_${tag}_dist.json" \
        --seed 42
done

# ── 2. Route under four objectives, compare Soft-KL vs Hard-CE routes ────────
echo ""
echo ">>> Step 2: risk-sensitive routing comparison"
$PYTHON src/routing/risk_planner.py \
    --graph_cache      "$GRAPH" \
    --edge_scores_soft "$OUT/edge_scores_soft_dist.json" \
    --edge_scores_hard "$OUT/edge_scores_hard_dist.json" \
    --n_pairs          "$N_PAIRS" \
    --output_dir       "$OUT"

echo ""
echo "DONE — $(date)"
