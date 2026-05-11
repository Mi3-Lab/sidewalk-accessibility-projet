#!/usr/bin/env python3
"""
Prepare test images for generalization evaluation.

This script helps you:
1. Create a CSV template for test images from new cities
2. Download/organize images from Project Sidewalk or other sources
3. Extract structural labels from PSW API (if available)

Usage:
    # Step 1: Create empty template
    python src/generalization/prepare_test_data.py \
        --create_template \
        --output data/generalization/test_images.csv

    # Step 2: Populate template with images and labels
    # (manually edit CSV or use PSW API)

    # Step 3: Validate
    python src/generalization/prepare_test_data.py \
        --validate \
        --images_csv data/generalization/test_images.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm


def create_template(output_path: str, n_samples: int = 100):
    """Create empty template CSV for test images."""
    template_df = pd.DataFrame({
        "image_id": [f"gen_{i:04d}" for i in range(n_samples)],
        "image_path": [""] * n_samples,
        "ps_label": [""] * n_samples,  # CurbRamp, NoCurbRamp, Unmarked, etc.
        "city": [""] * n_samples,
        "lat": [0.0] * n_samples,
        "lon": [0.0] * n_samples,
        "source": [""] * n_samples,  # PSW, manual, etc.
        "notes": [""] * n_samples,
    })
    
    template_df.to_csv(output_path, index=False)
    print(f"✅ Template created: {output_path}")
    print("\nTo populate:")
    print("  1. Add image_path (local path to .jpg)")
    print("  2. Add ps_label (from Project Sidewalk: CurbRamp, NoCurbRamp, etc.)")
    print("  3. Add city, lat/lon, source")


def validate_test_data(images_csv: str):
    """Validate that test images exist and have required fields."""
    df = pd.read_csv(images_csv)
    
    print(f"\nValidating {images_csv}...")
    print(f"  Total rows: {len(df)}")
    
    # Check required columns
    required_cols = ["image_path", "ps_label", "city"]
    for col in required_cols:
        if col not in df.columns:
            print(f"  ⚠️  Missing column: {col}")
    
    # Check image files exist
    missing = []
    for idx, row in df.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.exists():
            missing.append(row["image_path"])
    
    if missing:
        print(f"  ⚠️  {len(missing)} image files not found:")
        for path in missing[:5]:
            print(f"      {path}")
        if len(missing) > 5:
            print(f"      ... and {len(missing) - 5} more")
    else:
        print(f"  ✅ All {len(df)} image files exist")
    
    # Check ps_label values
    valid_labels = {"CurbRamp", "NoCurbRamp", "Unmarked", ""}
    invalid_labels = set(df["ps_label"].unique()) - valid_labels
    if invalid_labels:
        print(f"  ⚠️  Unknown ps_label values: {invalid_labels}")
    else:
        print(f"  ✅ ps_label values valid")
    
    # Check cities
    cities = df["city"].unique()
    print(f"  Cities: {', '.join(cities)}")
    
    print()


def download_psw_data(city_name: str, output_dir: str, label_type: str = "CurbRamp"):
    """
    Attempt to download images from Project Sidewalk API for a given city.
    
    Note: This requires understanding the PSW API schema. 
    For now, this is a placeholder to show the structure.
    
    Docs: https://projectsidewalk.org/api
    """
    print(f"\n⚠️  PSW API download not yet implemented.")
    print(f"    To use this, you need to:")
    print(f"    1. Register with Project Sidewalk API")
    print(f"    2. Get region_id for {city_name}")
    print(f"    3. Implement API calls in this script")
    print(f"\n    For now, download images manually from:")
    print(f"    https://projectsidewalk.org/map")


def main():
    parser = argparse.ArgumentParser(description="Prepare test data for generalization eval.")
    parser.add_argument("--create_template", action="store_true", help="Create empty template CSV")
    parser.add_argument("--output", type=str, default="data/generalization/test_images.csv")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of rows in template")
    parser.add_argument("--validate", action="store_true", help="Validate existing CSV")
    parser.add_argument("--images_csv", type=str, help="Path to CSV to validate")
    parser.add_argument("--download_psw", type=str, help="Download from PSW for city (not yet implemented)")
    
    args = parser.parse_args()
    
    if args.create_template:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        create_template(args.output, args.n_samples)
    
    elif args.validate:
        if not args.images_csv:
            print("Error: --images_csv required for validation")
            return
        validate_test_data(args.images_csv)
    
    elif args.download_psw:
        download_psw_data(args.download_psw, "data/generalization/images/")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
