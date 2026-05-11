#!/usr/bin/env python3
"""
Download or prepare test images for generalization evaluation.

The easiest way to get test images:

OPTION A: Project Sidewalk Web Interface (RECOMMENDED FOR REAL DATA)
─────────────────────────────────────────────────────────────────
1. Go to https://projectsidewalk.org/map
2. Choose a city NOT in training (e.g., San Francisco, Portland, New York, Boston)
3. Pan/zoom to regions with sidewalks
4. Click "Download Map Data" → Images
5. Select a bounding box or region
6. Export images as zip
7. Extract to: data/generalization/images/
8. Update data/generalization/test_images.csv with paths

OPTION B: Demo Mode (FOR TESTING THE PIPELINE)
───────────────────────────────────────────────
Run this script with --mode demo to:
- Copy sample images from existing dataset
- Create test CSV
- Let you verify the pipeline works before real data

Usage:
    # Create demo dataset (quick test)
    python src/generalization/download_test_images.py --mode demo

    # Instructions for real download
    python src/generalization/download_test_images.py --help
"""

import argparse
from pathlib import Path
import shutil
import pandas as pd
import random


def create_demo_dataset(images_dir: str, csv_path: str, n_samples: int = 30):
    """
    Create a demo dataset by copying existing images.
    
    This is for pipeline testing only — real generalization test
    requires images from new cities.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing images in data/images/
    source_image_dir = Path("data/images")
    if not source_image_dir.exists():
        print(f"⚠️  No existing images in {source_image_dir}")
        print("    Create empty images and CSV for manual population")
        _create_empty_structure(images_dir, csv_path, n_samples)
        return
    
    # Collect available images
    available_images = list(source_image_dir.glob("**/*.jpg")) + \
                       list(source_image_dir.glob("**/*.png"))
    
    if not available_images:
        print(f"⚠️  No .jpg/.png files found in {source_image_dir}")
        _create_empty_structure(images_dir, csv_path, n_samples)
        return
    
    # Sample images
    sampled = random.sample(available_images, min(n_samples, len(available_images)))
    
    # Copy to demo directory
    print(f"📋 Copying {len(sampled)} demo images...")
    rows = []
    for idx, src_path in enumerate(sampled):
        dst_name = f"demo_{idx:04d}.jpg"
        dst_path = images_dir / dst_name
        shutil.copy2(src_path, dst_path)
        
        rows.append({
            "image_id": f"demo_{idx:04d}",
            "image_path": str(dst_path),
            "ps_label": "Unmarked",  # Unknown for demo
            "city": "Demo",
            "lat": 0.0,
            "lon": 0.0,
            "source": "demo_copy",
            "notes": f"Copied from {src_path.name} for pipeline testing",
        })
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Demo dataset created:")
    print(f"   Images: {images_dir}/ ({len(sampled)} files)")
    print(f"   CSV: {csv_path}")
    print()
    print("⚠️  NOTE: This is for testing only.")
    print("   For real generalization test, download from Project Sidewalk:")
    print("   https://projectsidewalk.org/map")
    print()


def _create_empty_structure(images_dir: str, csv_path: str, n_samples: int):
    """Create empty CSV and image directory structure."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create template
    template_df = pd.DataFrame({
        "image_id": [f"gen_{i:04d}" for i in range(n_samples)],
        "image_path": [""] * n_samples,
        "ps_label": [""] * n_samples,
        "city": [""] * n_samples,
        "lat": [0.0] * n_samples,
        "lon": [0.0] * n_samples,
        "source": [""] * n_samples,
        "notes": [""] * n_samples,
    })
    
    template_df.to_csv(csv_path, index=False)
    
    print(f"✅ Empty structure created:")
    print(f"   Images dir: {images_dir}/")
    print(f"   CSV template: {csv_path}")
    print()
    print("📥 Next steps:")
    print("   1. Download images from Project Sidewalk: https://projectsidewalk.org/map")
    print("   2. Save them to: data/generalization/images/")
    print("   3. Edit CSV: data/generalization/test_images.csv")
    print("   4. Fill in image_path, city, ps_label, etc.")
    print()


def print_help():
    """Print detailed help."""
    help_text = """
GENERALIZATION TEST DATA — QUICK START
═════════════════════════════════════════

STEP 1: Get images from new cities (NOT in training set)
────────────────────────────────────────────────────────

Training cities: Washington DC, Seattle, Chicago, Columbus
Test cities: San Francisco, Portland, New York, Boston, etc.

Option A — Web Download (RECOMMENDED)
  1. Go to https://projectsidewalk.org/map
  2. Choose a city (NOT DC/Seattle/Chicago/Columbus)
  3. Click "Download Map Data" or take screenshots
  4. Save 50+ images to: data/generalization/images/
  
Option B — Demo Mode (for testing)
  $ python src/generalization/download_test_images.py --mode demo
  
  This copies existing images for pipeline testing (NOT real generalization).


STEP 2: Update CSV metadata
─────────────────────────────

Edit data/generalization/test_images.csv:
  • image_path: full path to image file
  • city: San Francisco, Portland, etc.
  • ps_label: From Project Sidewalk (CurbRamp, NoCurbRamp, or Unmarked)
  • lat, lon: (optional) coordinates

Example:
  image_id,image_path,ps_label,city,lat,lon,source,notes
  gen_0001,data/generalization/images/sf_001.jpg,CurbRamp,San Francisco,37.7749,-122.4194,PSW,Curb cut visible
  gen_0002,data/generalization/images/sf_002.jpg,NoCurbRamp,San Francisco,37.7755,-122.4193,PSW,No curb cut


STEP 3: Run evaluation
──────────────────────

$ ./run_generalization.sh dinov2-large results/models/dinov2-large

This will:
  1. Load images from CSV
  2. Extract features using frozen encoder
  3. Predict probabilities for each aid
  4. Save results to results/generalization/dinov2-large/predictions.csv


STEP 4: Analyze predictions qualitatively
──────────────────────────────────────────

Manual review of 30 sampled predictions:
  • 10 high-confidence "yes" (prob_yes > 0.8)
  • 10 high-confidence "no" (prob_yes < 0.2)
  • 10 uncertain (0.4 < prob_yes < 0.6)

Check: Do predictions visually make sense?

Write findings in paper/manuscript.tex Section 5.7


API / Programmatic Option (Advanced)
────────────────────────────────────

Project Sidewalk API: https://api.projectsidewalk.org/docs
  
This requires:
  • Registering for API key
  • Understanding PSW schema
  • Implementing calls to fetch region images

We haven't implemented this yet, but feel free to contribute!
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="Download or prepare test images for generalization test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "help"],
        default="help",
        help="Mode: 'demo' for quick pipeline test, 'help' for instructions"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/generalization/images",
        help="Directory to save images"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/generalization/test_images.csv",
        help="CSV path for metadata"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of demo images to create"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        create_demo_dataset(args.images_dir, args.csv, args.n_samples)
    else:
        print_help()


if __name__ == "__main__":
    main()
