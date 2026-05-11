#!/bin/bash
# Quick start: Initialize generalization test infrastructure
# Run this once to set up template CSV and folder structure

set -e

echo "🚀 Setting up generalization test infrastructure..."

# Create directories
mkdir -p data/generalization/images
mkdir -p results/generalization

# Create template CSV
python src/generalization/prepare_test_data.py \
    --create_template \
    --output data/generalization/test_images.csv \
    --n_samples 100

echo ""
echo "✅ Infrastructure ready! Next steps:"
echo ""
echo "1. Populate data/generalization/test_images.csv with:"
echo "   - image_path: local path to test image"
echo "   - ps_label: CurbRamp or NoCurbRamp (from Project Sidewalk)"
echo "   - city: city name (e.g., 'Washington DC')"
echo ""
echo "2. Download test images to data/generalization/images/"
echo ""
echo "3. Run evaluation:"
echo "   ./run_generalization.sh dinov2-large results/models/dinov2-large"
echo ""
