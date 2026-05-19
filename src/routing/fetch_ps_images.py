#!/usr/bin/env python3
"""
Download Project Sidewalk images from a city instance for a geographic bounding box.

Queries the PS public label API, downloads the associated sidewalk image for each
label point, and saves a CSV with (image_id, image_path, lat, lon, ps_label, city).

This CSV feeds directly into score_osm_edges.py → real p_yes scores for the routing demo.

Usage (Pittsburgh/Oakland, default bbox):
    python src/routing/fetch_ps_images.py \
        --ps_url     https://sidewalk-pittsburgh.cs.uw.edu \
        --output_dir data/routing/pittsburgh \
        --n_images   300

Usage (custom bbox):
    python src/routing/fetch_ps_images.py \
        --ps_url   https://sidewalk-pittsburgh.cs.uw.edu \
        --lat_min  40.430 --lat_max 40.460 \
        --lon_min  -79.970 --lon_max -79.920 \
        --output_dir data/routing/pittsburgh

Configuration:
    The script uses the Project Sidewalk public REST API (no auth required).
    Known city base URLs:
        Pittsburgh : https://sidewalk-pittsburgh.cs.uw.edu
        Seattle    : https://sidewalk-seattle.cs.uw.edu
        Chicago    : https://sidewalk-chicago.cs.washington.edu
        DC         : https://sidewalk-dc.cs.uw.edu
        Columbus   : https://sidewalk-columbus.cs.uw.edu

    Images are fetched from the PS /label/{id}/image endpoint (wraps GSV).
    If that returns 404, pass --gsv_key YOUR_KEY to fall back to GSV Static API.

Output CSV columns:
    image_id, image_path, lat, lon, ps_label, heading, city
    ps_label: CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | NoSidewalk | Other
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

# Oakland, Pittsburgh PA — same bbox as the routing graph (r=800m from 40.4432,-79.9433)
DEFAULT_LAT_MIN = 40.432
DEFAULT_LAT_MAX = 40.455
DEFAULT_LON_MIN = -79.963
DEFAULT_LON_MAX = -79.925

# PS label types to download — all sidewalk-related
SIDEWALK_LABEL_TYPES = {
    "CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem",
    "NoSidewalk", "Crosswalk", "Signal",
}

# GSV Static API image size
GSV_SIZE = "512x512"
GSV_FOV  = 90


def fetch_labels(ps_url: str, lat_min: float, lat_max: float,
                 lon_min: float, lon_max: float) -> list[dict]:
    """
    Query PS label API for a bounding box.
    Returns list of label dicts with lat, lng, label_type, panorama_id, heading, etc.
    """
    endpoint = f"{ps_url.rstrip('/')}/v2/access/attributesWithLabels"
    params = {
        "lat1": lat_min, "lng1": lon_min,
        "lat2": lat_max, "lng2": lon_max,
    }
    print(f"Querying: {endpoint}")
    print(f"  Bbox: ({lat_min},{lon_min}) → ({lat_max},{lon_max})")

    try:
        r = requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as e:
        print(f"  HTTP error {e.response.status_code}. Trying fallback endpoint …")
        return _fetch_labels_fallback(ps_url, lat_min, lat_max, lon_min, lon_max)
    except Exception as e:
        print(f"  Error: {e}")
        return []

    # PS API returns {"type":"FeatureCollection","features":[...]}
    if isinstance(data, dict) and "features" in data:
        features = data["features"]
        labels = []
        for f in features:
            props = f.get("properties", {})
            geom  = f.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            labels.append({
                "label_id":    props.get("label_id") or props.get("labelId"),
                "label_type":  props.get("label_type") or props.get("labelType", "Unknown"),
                "lat":         coords[1] if len(coords) > 1 else props.get("lat"),
                "lng":         coords[0] if len(coords) > 0 else props.get("lng"),
                "panorama_id": props.get("gsv_panorama_id") or props.get("panoramaId"),
                "heading":     props.get("photographer_heading") or props.get("heading", 0),
                "pitch":       props.get("photographer_pitch")   or props.get("pitch",   0),
                "zoom":        props.get("zoom", 3),
                "city":        props.get("city", "pittsburgh"),
            })
        print(f"  Found {len(labels)} labels.")
        return labels

    # Flat list format (older PS API versions)
    if isinstance(data, list):
        print(f"  Found {len(data)} labels (flat format).")
        return data

    print(f"  Unexpected API response format: {type(data)}")
    return []


def _fetch_labels_fallback(ps_url: str, lat_min: float, lat_max: float,
                            lon_min: float, lon_max: float) -> list[dict]:
    """Try alternative PS label endpoints."""
    for path in ["/v2/labels", "/api/labels"]:
        endpoint = f"{ps_url.rstrip('/')}{path}"
        params = {"lat1": lat_min, "lng1": lon_min, "lat2": lat_max, "lng2": lon_max}
        try:
            r = requests.get(endpoint, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    print(f"  Fallback succeeded: {endpoint} ({len(data)} labels)")
                    return data
        except Exception:
            pass
    print("  All PS API endpoints failed. Check --ps_url or network access.")
    return []


def download_image_ps(ps_url: str, label_id, output_path: Path,
                       session: requests.Session) -> bool:
    """Download image crop from PS /label/{id}/image endpoint."""
    url = f"{ps_url.rstrip('/')}/label/{label_id}/image"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
            output_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def download_image_gsv(lat: float, lng: float, heading: float,
                        output_path: Path, api_key: str,
                        session: requests.Session) -> bool:
    """Download image from Google Street View Static API."""
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size":     GSV_SIZE,
        "location": f"{lat},{lng}",
        "heading":  heading,
        "fov":      GSV_FOV,
        "pitch":    -10,
        "key":      api_key,
    }
    try:
        r = session.get(url, params=params, timeout=20)
        if r.status_code == 200 and not r.content.startswith(b"<"):
            output_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Project Sidewalk images for OSM edge scoring."
    )
    parser.add_argument("--ps_url",     required=True,
                        help="Base URL of PS city instance, e.g. https://sidewalk-pittsburgh.cs.uw.edu")
    parser.add_argument("--output_dir", default="data/routing/pittsburgh")
    parser.add_argument("--lat_min",    type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat_max",    type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon_min",    type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon_max",    type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--n_images",   type=int,   default=300,
                        help="Max images to download (0 = all).")
    parser.add_argument("--gsv_key",    default=None,
                        help="Google Street View Static API key (fallback if PS image endpoint fails).")
    parser.add_argument("--label_types", nargs="+",
                        default=list(SIDEWALK_LABEL_TYPES),
                        help="PS label types to include.")
    parser.add_argument("--sleep_ms",  type=int, default=100,
                        help="Sleep between image downloads (ms). Avoids rate limits.")
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    images_dir  = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    label_types = set(args.label_types)

    # ── 1. Fetch label metadata ───────────────────────────────────────────────
    labels = fetch_labels(
        args.ps_url, args.lat_min, args.lat_max, args.lon_min, args.lon_max
    )
    if not labels:
        print("No labels found. Check --ps_url and bbox arguments.")
        sys.exit(1)

    # Filter to requested label types
    labels = [l for l in labels if l.get("label_type") in label_types]
    print(f"After type filter ({', '.join(sorted(label_types))}): {len(labels)} labels")

    # Filter to valid GPS
    labels = [l for l in labels if l.get("lat") and l.get("lng")]
    print(f"After GPS filter: {len(labels)} labels")

    if args.n_images > 0:
        labels = labels[:args.n_images]
        print(f"Capped at {args.n_images}: {len(labels)} labels")

    # ── 2. Download images ────────────────────────────────────────────────────
    session = requests.Session()
    session.headers["User-Agent"] = "sidewalk-accessibility-research/1.0"

    records     = []
    downloaded  = 0
    skipped     = 0

    for i, lbl in enumerate(labels):
        label_id = lbl.get("label_id") or i
        lat      = float(lbl["lat"])
        lng      = float(lbl["lng"])
        heading  = float(lbl.get("heading") or 0)
        ps_label = lbl.get("label_type", "Unknown")
        city     = lbl.get("city", "pittsburgh")

        fname  = f"ps_{city}_{label_id}.jpg"
        fpath  = images_dir / fname

        if fpath.exists():
            skipped += 1
        else:
            ok = download_image_ps(args.ps_url, label_id, fpath, session)
            if not ok and args.gsv_key:
                ok = download_image_gsv(lat, lng, heading, fpath, args.gsv_key, session)
            if ok:
                downloaded += 1
            else:
                # Record without image so GPS is at least preserved for snapping
                pass
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000)

        if fpath.exists():
            records.append({
                "image_id":  f"ps_{city}_{label_id}",
                "image_path": str(fpath),
                "lat":        lat,
                "lon":        lng,
                "heading":    heading,
                "ps_label":   ps_label,
                "city":       city,
            })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(labels)} — {downloaded} downloaded, {skipped} cached")

    # ── 3. Save CSV ───────────────────────────────────────────────────────────
    csv_path = output_dir / "ps_labels.csv"
    fieldnames = ["image_id", "image_path", "lat", "lon", "heading", "ps_label", "city"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDone: {len(records)} images with valid paths")
    print(f"  Downloaded: {downloaded}, Cached: {skipped}")
    print(f"  CSV → {csv_path}")
    print(f"\nNext step:")
    print(f"  python src/routing/score_osm_edges.py \\")
    print(f"      --ps_csv   {csv_path} \\")
    print(f"      --checkpoint results/models/dinov2-large \\")
    print(f"      --output   results/routing/edge_scores.json")


if __name__ == "__main__":
    main()
