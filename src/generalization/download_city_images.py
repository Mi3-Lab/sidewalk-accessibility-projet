#!/usr/bin/env python3
"""
Download Project Sidewalk images for generalization testing.

Fetches CurbRamp and NoCurbRamp labels from a PS city instance, then downloads
the corresponding Google Street View panorama thumbnail for each label using the
Maps streetviewpixels API (no API key required, works for non-expired panoramas).

Creates a CSV with columns: image_id, image_path, city, ps_label
suitable for direct input to evaluate_generalization.py.

Usage (Pittsburgh, uses pre-downloaded labels if available):
    python src/generalization/download_city_images.py \
        --city      pittsburgh \
        --ps_url    https://sidewalk-pittsburgh.cs.washington.edu \
        --ps_labels data/routing/pittsburgh/ps_labels.geojson \
        --output_dir data/generalization \
        --n_per_class 75

Usage (Washington DC, fetches labels first):
    python src/generalization/download_city_images.py \
        --city      dc \
        --ps_url    https://sidewalk-dc.cs.washington.edu \
        --lat_min   38.88 --lat_max 38.94 \
        --lon_min   -77.06 --lon_max -76.96 \
        --output_dir data/generalization \
        --n_per_class 75

Known PS city URLs (all public, no auth required):
    Pittsburgh : https://sidewalk-pittsburgh.cs.washington.edu  (not in training set)
    DC         : https://sidewalk-dc.cs.washington.edu          (not in training set)
    Seattle    : https://sidewalk-seattle.cs.washington.edu     (IN training set)
    Chicago    : https://sidewalk-chicago.cs.washington.edu     (IN training set)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# GSV thumbnail endpoint — no API key required; returns black image for expired panos
GSV_THUMB_URL = "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"

# Minimum bytes for a real (non-expired) thumbnail
MIN_REAL_IMAGE_BYTES = 5_000

# PS label types we care about for generalization testing
TARGET_LABEL_TYPES = {"CurbRamp", "NoCurbRamp"}

# Mapping to evaluate_generalization.py expected values
LABEL_TYPE_MAP = {"CurbRamp": "CurbRamp", "NoCurbRamp": "NoCurbRamp"}


def fetch_ps_labels(ps_url: str, lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float) -> list[dict]:
    """Download PS labels via the public v3 API (no auth)."""
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    endpoint = f"{ps_url.rstrip('/')}/v3/api/rawLabels"
    print(f"Querying {endpoint}  bbox={bbox}")
    r = requests.get(endpoint, params={"bbox": bbox, "filetype": "json"}, timeout=120)
    r.raise_for_status()
    return r.json().get("features", [])


def load_ps_labels_from_geojson(path: str) -> list[dict]:
    """Load labels from an existing PS GeoJSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("features", [])


def features_to_label_dicts(features: list[dict]) -> list[dict]:
    """Convert GeoJSON features to flat dicts with the fields we need."""
    labels = []
    for feat in features:
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        lt = props.get("label_type", "")
        if lt not in TARGET_LABEL_TYPES:
            continue
        pano_id = props.get("pano_id") or props.get("gsv_panorama_id")
        if not pano_id:
            continue
        labels.append({
            "label_id":   props.get("label_id"),
            "label_type": lt,
            "pano_id":    pano_id,
            "heading":    props.get("heading", 0.0),
            "pitch":      props.get("pitch", -10.0),
            "lat":        coords[1] if len(coords) > 1 else None,
            "lon":        coords[0] if len(coords) > 0 else None,
            # Quality signal: labels validated by multiple annotators
            "agree_count": props.get("agree_count", 0),
        })
    return labels


def download_thumbnail(pano_id: str, heading: float, pitch: float,
                       out_path: Path, session: requests.Session,
                       size: int = 512) -> bool:
    """
    Download a GSV panorama thumbnail for the given pano + orientation.
    Returns True if a real (non-expired) image was saved.
    """
    params = {
        "panoid":    pano_id,
        "cb_client": "maps_sv.tactile",
        "w":         size,
        "h":         size,
        "yaw":       round(heading, 4),
        "pitch":     round(pitch, 4),
        "thumbfov":  90,
    }
    try:
        r = session.get(GSV_THUMB_URL, params=params, timeout=20)
        if r.status_code != 200:
            return False
        if len(r.content) < MIN_REAL_IMAGE_BYTES:
            return False
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False


def balanced_sample(labels: list[dict], n_per_class: int) -> list[dict]:
    """Sample up to n_per_class labels per label_type, preferring high agree_count."""
    from collections import defaultdict
    by_type: dict[str, list] = defaultdict(list)
    for lbl in labels:
        by_type[lbl["label_type"]].append(lbl)

    result = []
    for lt, items in by_type.items():
        # Sort by agree_count descending (higher quality first)
        items.sort(key=lambda x: x["agree_count"], reverse=True)
        result.extend(items[:n_per_class] if n_per_class > 0 else items)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download PS images for generalization testing (no API key)."
    )
    parser.add_argument("--city",       required=True,
                        help="City name tag (used in image filenames and CSV).")
    parser.add_argument("--ps_url",     required=True,
                        help="Base URL of the PS city instance.")
    parser.add_argument("--ps_labels",  default=None,
                        help="Path to existing PS GeoJSON (skip API query if provided).")
    parser.add_argument("--lat_min",    type=float, default=None)
    parser.add_argument("--lat_max",    type=float, default=None)
    parser.add_argument("--lon_min",    type=float, default=None)
    parser.add_argument("--lon_max",    type=float, default=None)
    parser.add_argument("--n_per_class", type=int, default=75,
                        help="Max images per class (CurbRamp/NoCurbRamp). 0 = no limit.")
    parser.add_argument("--output_dir", default="data/generalization")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Thumbnail width/height in pixels.")
    parser.add_argument("--sleep_ms",   type=int, default=80,
                        help="Sleep between requests to avoid rate limits.")
    parser.add_argument("--append_csv", default=None,
                        help="Existing CSV to append to (for multi-city datasets).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images" / args.city
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load or fetch PS labels ─────────────────────────────────────────────
    if args.ps_labels and Path(args.ps_labels).exists():
        print(f"Loading PS labels from: {args.ps_labels}")
        features = load_ps_labels_from_geojson(args.ps_labels)
    else:
        if any(v is None for v in [args.lat_min, args.lat_max, args.lon_min, args.lon_max]):
            print("ERROR: --lat_min/lat_max/lon_min/lon_max required when not using --ps_labels")
            sys.exit(1)
        features = fetch_ps_labels(
            args.ps_url, args.lat_min, args.lat_max, args.lon_min, args.lon_max
        )

    all_labels = features_to_label_dicts(features)
    from collections import Counter
    counts = Counter(l["label_type"] for l in all_labels)
    print(f"PS labels loaded: {len(all_labels)} CurbRamp/NoCurbRamp")
    for lt, n in sorted(counts.items()):
        print(f"  {lt:<15}: {n}")

    if not all_labels:
        print("No CurbRamp/NoCurbRamp labels found. Check bbox or GeoJSON path.")
        sys.exit(1)

    # ── 2. Balanced sample ─────────────────────────────────────────────────────
    sampled = balanced_sample(all_labels, args.n_per_class)
    sample_counts = Counter(l["label_type"] for l in sampled)
    print(f"\nSampled (n_per_class={args.n_per_class}):")
    for lt, n in sorted(sample_counts.items()):
        print(f"  {lt:<15}: {n}")
    print(f"Total to attempt: {len(sampled)}")

    # ── 3. Download thumbnails ─────────────────────────────────────────────────
    session = requests.Session()
    session.headers["User-Agent"] = "sidewalk-accessibility-research/1.0"

    records   = []
    skipped   = 0
    cached    = 0

    for lbl in tqdm(sampled, desc=f"Downloading {args.city}"):
        label_id  = lbl["label_id"]
        label_type = lbl["label_type"]
        pano_id    = lbl["pano_id"]
        heading    = lbl["heading"]
        pitch      = lbl["pitch"]

        fname = f"{args.city}_{label_id}.jpg"
        fpath = images_dir / fname

        if fpath.exists() and fpath.stat().st_size >= MIN_REAL_IMAGE_BYTES:
            cached += 1
        else:
            ok = download_thumbnail(pano_id, heading, pitch, fpath,
                                    session, size=args.image_size)
            if not ok:
                skipped += 1
                continue
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000)

        records.append({
            "image_id":   f"{args.city}_{label_id}",
            "image_path": str(fpath),
            "city":       args.city,
            "ps_label":   LABEL_TYPE_MAP[label_type],
        })

    # ── 4. Report download stats ───────────────────────────────────────────────
    downloaded = len(records) - cached
    print(f"\nResults: {len(records)} images saved")
    print(f"  Downloaded: {downloaded}, Cached: {cached}, Skipped (expired): {skipped}")
    final_counts = Counter(r["ps_label"] for r in records)
    for lt, n in sorted(final_counts.items()):
        print(f"  {lt:<15}: {n}")

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    if args.append_csv and Path(args.append_csv).exists():
        # Load existing rows and append
        import pandas as pd
        existing = pd.read_csv(args.append_csv).to_dict("records")
        existing_ids = {r["image_id"] for r in existing}
        new_records = [r for r in records if r["image_id"] not in existing_ids]
        all_records = existing + new_records
        csv_path = Path(args.append_csv)
    else:
        all_records = records
        csv_path = output_dir / "test_images.csv"

    fieldnames = ["image_id", "image_path", "city", "ps_label"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nCSV → {csv_path}  ({len(all_records)} rows)")
    print(f"\nNext step:")
    print(f"  python src/generalization/evaluate_generalization.py \\")
    print(f"      --encoder    dinov2-large \\")
    print(f"      --checkpoint results/models/dinov2-large \\")
    print(f"      --test_images {csv_path} \\")
    print(f"      --output_dir  results/generalization/dinov2-large")


if __name__ == "__main__":
    main()
