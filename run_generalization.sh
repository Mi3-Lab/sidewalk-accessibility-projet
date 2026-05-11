#!/bin/bash
# Generalization Test Pipeline
# Task 5.7: Evaluate model on new cities (out-of-distribution transfer)
#
# Usage:
#   ./run_generalization.sh [encoder] [checkpoint_dir]
#   ./run_generalization.sh dinov2-large results/models/dinov2-large
#   ./run_generalization.sh clip-vit-b32 results/models/clip-vit-b32
#
# Prerequisites:
#   1. Trained model checkpoint in checkpoint_dir
#   2. Test images CSV in data/generalization/test_images.csv
#   3. Images should exist locally in data/generalization/images/

set -e

# Configuration
ENCODER="${1:-dinov2-large}"
CHECKPOINT_DIR="${2:-results/models/dinov2-large}"
TEST_IMAGES="data/generalization/test_images.csv"
OUTPUT_DIR="results/generalization/${ENCODER}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}GENERALIZATION TEST PIPELINE — Task 5.7${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Encoder:       $ENCODER"
echo "Checkpoint:    $CHECKPOINT_DIR"
echo "Test images:   $TEST_IMAGES"
echo "Output:        $OUTPUT_DIR"
echo ""

# Step 1: Validate test data exists
echo -e "${BLUE}[Step 1/3] Validating test data...${NC}"
if [ ! -f "$TEST_IMAGES" ]; then
    echo -e "${YELLOW}⚠️  No test images CSV found at $TEST_IMAGES${NC}"
    echo "    Create one with:"
    echo "    python src/generalization/prepare_test_data.py --create_template --output $TEST_IMAGES"
    echo ""
    echo "    Then populate it with image paths and Project Sidewalk labels"
    exit 1
fi
python src/generalization/prepare_test_data.py --validate --images_csv "$TEST_IMAGES"

# Step 2: Run evaluation
echo ""
echo -e "${BLUE}[Step 2/3] Running generalization evaluation...${NC}"
python src/generalization/evaluate_generalization.py \
    --encoder "$ENCODER" \
    --checkpoint "$CHECKPOINT_DIR" \
    --test_images "$TEST_IMAGES" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

# Step 3: Summary
echo ""
echo -e "${BLUE}[Step 3/3] Generating summary...${NC}"
echo ""
echo -e "${GREEN}✅ Results saved to:${NC}"
echo "   - $OUTPUT_DIR/predictions.csv"
echo "   - $OUTPUT_DIR/agreement_summary.json"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "   1. Review $OUTPUT_DIR/agreement_summary.json for accuracy stats"
echo "   2. Analyze predictions.csv for per-image results"
echo "   3. If accuracy is low, investigate edge cases in results"
echo "   4. Add findings to paper Section 5.7 (Generalization)"
echo ""
