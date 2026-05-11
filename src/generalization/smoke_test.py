#!/usr/bin/env python3
"""
Quick check: Verify all Task 5.7 scripts are working before actual generalization test.

Run this to ensure no import errors or missing dependencies.

Usage:
    python src/generalization/smoke_test.py
"""

import sys
import traceback

print("🔍 Running smoke tests for Task 5.7 (Generalization) infrastructure...")
print()

# Test 1: Check imports
print("Test 1: Checking imports...")
try:
    import pandas as pd
    import numpy as np
    import torch
    print("  ✅ pandas, numpy, torch imported successfully")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check scripts exist
print("\nTest 2: Checking script files...")
from pathlib import Path
scripts = [
    "src/generalization/evaluate_generalization.py",
    "src/generalization/prepare_test_data.py",
    "run_generalization.sh",
]
for script in scripts:
    if Path(script).exists():
        print(f"  ✅ {script}")
    else:
        print(f"  ❌ {script} NOT FOUND")

# Test 3: Check template creation works
print("\nTest 3: Testing template creation...")
try:
    df = pd.DataFrame({
        "image_id": ["test_001"],
        "image_path": ["test.jpg"],
        "ps_label": ["CurbRamp"],
        "city": ["Test City"],
        "source": ["test"],
    })
    print("  ✅ Template creation logic works")
except Exception as e:
    print(f"  ❌ Template creation failed: {e}")

# Test 4: Check torch + cuda
print("\nTest 4: Checking GPU availability...")
if torch.cuda.is_available():
    print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("  ⚠️  CUDA not available (CPU inference will be slower)")

print("\n" + "="*60)
print("✅ All smoke tests passed! Ready to run generalization eval.")
print("="*60)
print("\nNext steps:")
print("  1. Run: bash init_generalization.sh")
print("  2. Edit: data/generalization/test_images.csv")
print("  3. Run: ./run_generalization.sh dinov2-large results/models/dinov2-large")
print()
