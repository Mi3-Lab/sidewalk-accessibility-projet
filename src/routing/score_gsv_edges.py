#!/usr/bin/env python3
"""
Score ALL OSM pedestrian edges via DINOv2-large inference on GSV thumbnails.

Takes the existing edge_scores.json (40.4% PS-label coverage, 59.6% prior)
and replaces every "prior" edge with a real model prediction.

Strategy
--------
  For each "prior" edge (no PS label within 50m):
    1. Find the nearest PS labels by haversine distance.
    2. Try to download a GSV thumbnail using that label's pano_id.
    3. Try up to MAX_PANO_TRIES nearest labels (some panos are expired).
    4. If a valid image is found: run DINOv2-large inference → p_yes.
    5. If all nearby panos are expired: keep the original prior.

Rationale: The GSV thumbnail from a nearby labeled location captures the
general neighbourhood character (surface quality, urban density, road type)
relevant to accessibility — substantially better than a fixed population-level
prior, even when the pano is 50–200m from the exact edge midpoint.

Usage
-----
    python src/routing/score_gsv_edges.py \
        --edge_scores   results/routing/edge_scores.json \
        --ps_labels     data/routing/pittsburgh/ps_labels.geojson \
        --checkpoint    results/models/dinov2-large \
        --lat 40.444 --lon -79.956 --dist 800 \
        --output        results/routing/edge_scores_full.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

try:
    import osmnx as ox
except ImportError:
    print("ERROR: osmnx not installed. pip install osmnx")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).parent))   # for score_osm_edges imports

GSV_THUMB_URL   = "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
MIN_IMAGE_BYTES = 5_000
MAX_PANO_TRIES  = 8    # try this many nearest labels before giving up


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def edge_midpoint(G, u, v, k):
    data = G[u][v][k]
    if "geometry" in data:
        coords = list(data["geometry"].coords)
        mid    = coords[len(coords) // 2]
        return float(mid[1]), float(mid[0])
    return float((G.nodes[u]["y"] + G.nodes[v]["y"]) / 2), \
           float((G.nodes[u]["x"] + G.nodes[v]["x"]) / 2)


def fetch_thumbnail(pano_id: str, heading: float, pitch: float,
                    session: requests.Session, size: int = 512) -> bytes | None:
    """Download a GSV thumbnail by pano_id. Returns bytes or None if expired."""
    try:
        r = session.get(GSV_THUMB_URL, params={
            "panoid":    pano_id,
            "cb_client": "maps_sv.tactile",
            "w":         size,
            "h":         size,
            "yaw":       round(heading, 2),
            "pitch":     round(pitch, 2),
            "thumbfov":  90,
        }, timeout=20)
        if r.status_code != 200 or len(r.content) < MIN_IMAGE_BYTES:
            return None
        return r.content
    except Exception:
        return None


def load_ps_pano_pool(ps_geojson_path: str) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Extract (lat, lon, pano_id, heading) for all labels in the PS GeoJSON.
    Returns arrays for fast vectorised nearest-neighbour search.
    """
    with open(ps_geojson_path) as f:
        features = json.load(f)["features"]

    lats, lons, panos, headings = [], [], [], []
    seen_panos: set[str] = set()

    for feat in features:
        props  = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        pano_id = props.get("pano_id")
        if not pano_id:
            continue
        lat = coords[1]
        lon = coords[0]
        heading = props.get("heading", 0.0)
        pitch   = props.get("pitch", -10.0)

        # Keep only unique pano_ids but allow multiple headings (different angles)
        lats.append(lat)
        lons.append(lon)
        panos.append((pano_id, pitch))
        headings.append(heading)

    print(f"Loaded {len(lats)} pano entries ({len({p for p,_ in panos})} unique pano_ids)")
    return np.array(lats), np.array(lons), panos, np.array(headings)


def main():
    parser = argparse.ArgumentParser(
        description="Fill routing edge_scores.json gaps with GSV model inference."
    )
    parser.add_argument("--edge_scores", default="results/routing/edge_scores.json")
    parser.add_argument("--ps_labels",   default="data/routing/pittsburgh/ps_labels.geojson")
    parser.add_argument("--checkpoint",  default="results/models/dinov2-large")
    parser.add_argument("--encoder",     default="dinov2-large")
    parser.add_argument("--lat",  type=float, default=40.444)
    parser.add_argument("--lon",  type=float, default=-79.956)
    parser.add_argument("--dist", type=int,   default=800)
    parser.add_argument("--output", default="results/routing/edge_scores_full.json")
    parser.add_argument("--sleep_ms",   type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size",  type=int, default=512)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    # ── 1. Load existing edge scores ─────────────────────────────────────────
    print(f"Loading edge scores from {args.edge_scores} …")
    with open(args.edge_scores) as f:
        existing = json.load(f)

    meta_orig   = existing.get("__meta__", {})
    edge_scores = existing["edges"]
    prior_keys  = [k for k, v in edge_scores.items() if v.get("source") == "prior"]
    ps_keys     = [k for k, v in edge_scores.items() if v.get("source") != "prior"]
    print(f"  Total: {len(edge_scores)}  |  PS-label: {len(ps_keys)}  |  Prior: {len(prior_keys)}")

    # ── 2. Load OSM graph for midpoints ──────────────────────────────────────
    print(f"\nLoading OSM graph ({args.lat}, {args.lon}), r={args.dist}m …")
    G = ox.graph_from_point((args.lat, args.lon), dist=args.dist,
                            network_type="walk", simplify=True).to_undirected()

    edge_midpoints: dict[str, tuple] = {}
    for u, v, k in G.edges(keys=True):
        key = f"{u}_{v}_{k}"
        edge_midpoints[key] = edge_midpoint(G, u, v, k)

    to_infer = [k for k in prior_keys if k in edge_midpoints]
    print(f"  Edges to fill: {len(to_infer)}")

    # ── 3. Build pano pool from PS labels ────────────────────────────────────
    print(f"\nBuilding pano pool from {args.ps_labels} …")
    pool_lats, pool_lons, pool_panos, pool_headings = load_ps_pano_pool(args.ps_labels)

    # ── 4. Download thumbnails (nearest-pano strategy) ───────────────────────
    print(f"\nFetching GSV thumbnails for {len(to_infer)} prior edges …")
    session = requests.Session()
    session.headers["User-Agent"] = "sidewalk-accessibility-research/1.0"

    downloaded: dict[str, str]  = {}   # edge_key → temp image path
    pano_cache: dict[str, bytes] = {}  # pano_id → bytes (avoid re-downloading)
    skipped   = 0
    n_cached  = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for edge_key in tqdm(to_infer, desc="GSV fetch"):
            mid_lat, mid_lon = edge_midpoints[edge_key]

            # Vectorised haversine to all pool entries
            dphi  = np.radians(pool_lats - mid_lat)
            dlon  = np.radians(pool_lons - mid_lon)
            phi1  = np.radians(mid_lat)
            phi2  = np.radians(pool_lats)
            a     = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
            dists = 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

            # Try MAX_PANO_TRIES nearest labels
            nearest_idxs = np.argsort(dists)[:MAX_PANO_TRIES]
            found = False
            for idx in nearest_idxs:
                pano_id, pitch = pool_panos[idx]
                heading        = float(pool_headings[idx])

                # Use cached bytes if already downloaded
                if pano_id in pano_cache:
                    img_bytes = pano_cache[pano_id]
                    n_cached += 1
                else:
                    img_bytes = fetch_thumbnail(pano_id, heading, pitch, session, args.size)
                    pano_cache[pano_id] = img_bytes   # cache even None
                    if args.sleep_ms > 0 and img_bytes is not None:
                        time.sleep(args.sleep_ms / 1000)

                if img_bytes is not None:
                    fpath = Path(tmpdir) / f"{edge_key}.jpg"
                    fpath.write_bytes(img_bytes)
                    downloaded[edge_key] = str(fpath)
                    found = True
                    break

            if not found:
                skipped += 1

        print(f"\n  Downloaded: {len(downloaded)}, Skipped (all panos expired): {skipped}")
        print(f"  Pano cache hits: {n_cached}")

        # ── 5. Batch inference ────────────────────────────────────────────────
        print(f"\nLoading DINOv2-large model …")
        from infer import SidewalkAccessibilityModel
        model = SidewalkAccessibilityModel(
            encoder_key=args.encoder,
            checkpoint_dir=args.checkpoint,
            seed=args.seed,
        )

        inferred: dict[str, dict] = {}
        img_keys  = list(downloaded.keys())
        img_paths = [downloaded[k] for k in img_keys]

        print(f"Inference on {len(img_paths)} images (batch_size={args.batch_size}) …")
        for i in tqdm(range(0, len(img_paths), args.batch_size), desc="Inference"):
            batch_keys  = img_keys[i : i + args.batch_size]
            batch_paths = img_paths[i : i + args.batch_size]
            results     = model.predict_batch(batch_paths)
            for key, preds in zip(batch_keys, results):
                inferred[key] = {
                    aid.lower().replace(" ", "_"): round(r["p_yes"], 4)
                    for aid, r in preds.items()
                }

    # ── 6. Merge ──────────────────────────────────────────────────────────────
    from score_osm_edges import AIDS_KEYS, PRIOR_P_YES

    updated    = dict(edge_scores)
    n_replaced = 0
    n_fallback = 0

    for edge_key in prior_keys:
        if edge_key in inferred:
            scores = inferred[edge_key]
            updated[edge_key] = {
                **{k: scores.get(k, PRIOR_P_YES[k]) for k in AIDS_KEYS},
                "n_images": 1,
                "source":   "inference",
            }
            n_replaced += 1
        else:
            n_fallback += 1

    pct_real = 100 * (len(ps_keys) + n_replaced) / len(updated)

    print(f"\n=== Merge Summary ===")
    print(f"  PS-label edges (unchanged) : {len(ps_keys)}")
    print(f"  Inferred from GSV          : {n_replaced}  ({100*n_replaced/len(prior_keys):.1f}% of prior)")
    print(f"  Kept as prior (all expired): {n_fallback}")
    print(f"  Final real coverage        : {pct_real:.1f}%  (was 40.4%)")

    print("\nPer-aid mean p_yes (all edges, final):")
    for aid_key in AIDS_KEYS:
        vals = [v[aid_key] for v in updated.values() if aid_key in v]
        print(f"  {aid_key:<25}  {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    meta_new = {
        **meta_orig,
        "gsv_inference": {
            "encoder":         args.encoder,
            "checkpoint":      args.checkpoint,
            "edges_inferred":  n_replaced,
            "edges_fallback":  n_fallback,
            "final_real_pct":  round(pct_real, 1),
            "strategy":        "nearest_ps_pano",
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"__meta__": meta_new, "edges": updated}, f)

    size_kb = out_path.stat().st_size // 1024
    print(f"\nFull edge scores → {out_path}  ({size_kb} KB)")
    print(f"\nNext:")
    print(f"  mkdir -p results/routing/full_coverage")
    print(f"  python src/routing/demo.py \\")
    print(f"      --edge_scores {out_path} \\")
    print(f"      --lat {args.lat} --lon {args.lon} \\")
    print(f"      --output_dir results/routing/full_coverage")


if __name__ == "__main__":
    main()
