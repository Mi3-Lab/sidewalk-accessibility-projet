#!/usr/bin/env python3
"""
Download Project Sidewalk accessibility labels for a city bounding box.

Uses the public PS v3 API (no authentication required) to download GPS-tagged
accessibility labels: CurbRamp, NoCurbRamp, Obstacle, SurfaceProblem, NoSidewalk.

These labels ARE ground-truth accessibility assessments by trained annotators
walking the streets. They feed directly into score_osm_edges.py for the routing demo.

Usage (Pittsburgh/Oakland — default):
    python src/routing/fetch_ps_labels.py \
        --output data/routing/pittsburgh/ps_labels.geojson

Usage (custom bbox):
    python src/routing/fetch_ps_labels.py \
        --ps_url  https://sidewalk-pittsburgh.cs.washington.edu \
        --lat_min 40.430 --lat_max 40.460 \
        --lon_min -79.970 --lon_max -79.920 \
        --output  data/routing/pittsburgh/ps_labels.geojson

Known city URLs (all public, no auth required):
    Pittsburgh : https://sidewalk-pittsburgh.cs.washington.edu  ← ROUTING DEMO
    Seattle    : https://sidewalk-seattle.cs.washington.edu     (in training set)
    Chicago    : https://sidewalk-chicago.cs.washington.edu     (in training set)
    Columbus   : https://sidewalk-columbus.cs.washington.edu    (in training set)
    DC         : https://sidewalk-dc.cs.washington.edu
    Amsterdam  : https://sidewalk-amsterdam.cs.washington.edu
"""

import argparse
import json
import sys
from pathlib import Path

import requests

PS_PITTSBURGH_URL = "https://sidewalk-pittsburgh.cs.washington.edu"

# Oakland, Pittsburgh PA — same bbox as routing graph (r=800m from 40.4432,-79.9433)
DEFAULT_LAT_MIN = 40.430
DEFAULT_LAT_MAX = 40.460
DEFAULT_LON_MIN = -79.965
DEFAULT_LON_MAX = -79.925

SIDEWALK_LABEL_TYPES = {
    "CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem",
    "NoSidewalk", "Crosswalk", "Signal",
}


def fetch_labels(ps_url: str, lat_min: float, lat_max: float,
                 lon_min: float, lon_max: float) -> dict:
    """
    Fetch labels from PS public v3 API.
    Returns a GeoJSON FeatureCollection.
    """
    # Format: minLng,minLat,maxLng,maxLat
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    endpoint = f"{ps_url.rstrip('/')}/v3/api/rawLabels"
    params = {"bbox": bbox, "filetype": "json"}

    print(f"Querying {endpoint}")
    print(f"  bbox: {bbox}  ({lat_min},{lon_min}) → ({lat_max},{lon_max})")

    r = requests.get(endpoint, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(
        description="Download Project Sidewalk labels for OSM edge scoring (no auth required)."
    )
    parser.add_argument("--ps_url",  default=PS_PITTSBURGH_URL,
                        help="Base URL of PS city instance.")
    parser.add_argument("--lat_min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat_max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon_min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon_max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--output",  default="data/routing/pittsburgh/ps_labels.geojson",
                        help="Output GeoJSON path.")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    data = fetch_labels(args.ps_url, args.lat_min, args.lat_max,
                        args.lon_min, args.lon_max)

    features = data.get("features", [])
    print(f"Downloaded {len(features)} labels")

    # Summary by label type
    from collections import Counter
    types = Counter(f["properties"]["label_type"] for f in features)
    for lt, n in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {lt:<20} {n:>5}")

    with open(args.output, "w") as f:
        json.dump(data, f)
    print(f"\nSaved → {args.output}  ({Path(args.output).stat().st_size // 1024} KB)")
    print(f"\nNext step:")
    print(f"  python src/routing/score_osm_edges.py \\")
    print(f"      --ps_labels {args.output} \\")
    print(f"      --output    results/routing/edge_scores.json")


if __name__ == "__main__":
    main()
