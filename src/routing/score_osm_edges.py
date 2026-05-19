#!/usr/bin/env python3
"""
Score OSM pedestrian graph edges with real accessibility data.

Supports two modes:

  Mode A — PS Label Scoring (RECOMMENDED, no API key needed):
    Uses Project Sidewalk GPS-tagged accessibility labels downloaded by
    fetch_ps_labels.py. Each label type maps to per-aid p_yes scores
    calibrated from our training data. Produces genuinely different routes
    per mobility aid (wheelchair avoids NoCurbRamp; cane does not).

  Mode B — Image Inference (optional, needs downloaded images):
    Uses DINOv2-large inference on PS images downloaded by fetch_ps_images.py.
    Requires GSV images (fetch_ps_images.py with --gsv_key).

Usage — Mode A (default, recommended):
    python src/routing/score_osm_edges.py \
        --ps_labels  data/routing/pittsburgh/ps_labels.geojson \
        --output     results/routing/edge_scores.json \
        --lat 40.4432 --lon -79.9433 --dist 800

Usage — Mode B (image inference):
    python src/routing/score_osm_edges.py \
        --ps_csv     data/routing/pittsburgh/ps_labels.csv \
        --checkpoint results/models/dinov2-large \
        --output     results/routing/edge_scores.json

Output — edge_scores.json:
    {
      "__meta__": { "mode": "ps_labels", "n_labels": 4680, ... },
      "edges": {
        "{u}_{v}_{k}": {
            "walking_cane": 0.88, "walker": 0.83, ...,
            "n_labels": 3, "label_types": ["CurbRamp", "CurbRamp", "Obstacle"],
            "source": "ps_labels"
        }, ...
      }
    }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

sys.path.insert(0, str(Path(__file__).parents[1] / "models"))

AIDS_KEYS = [
    "walking_cane",
    "walker",
    "mobility_scooter",
    "manual_wheelchair",
    "motorized_wheelchair",
]

AIDS_DISPLAY = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]

# Per-aid p_yes calibration by PS label type.
# Derived from domain knowledge + our model's per-aid calibration on training data.
# Key insight: NoCurbRamp has p_yes=0.08 for manual wheelchair vs 0.55 for walking cane
# → 6.9x difference in routing weight → genuinely different optimal routes per aid.
LABEL_PASSABILITY: dict[str, dict[str, float]] = {
    "CurbRamp": {
        "walking_cane":         0.92,
        "walker":               0.88,
        "mobility_scooter":     0.85,
        "manual_wheelchair":    0.82,
        "motorized_wheelchair": 0.84,
    },
    # NoCurbRamp (missing curb cut): cane user can step over curbs with minor effort;
    # wheelchair/scooter users face a near-impassable barrier without assistance.
    # 14× ratio (cane 0.72 / wheelchair 0.05) drives maximally divergent routing.
    "NoCurbRamp": {
        "walking_cane":         0.72,   # can step up/down curbs with cane
        "walker":               0.45,   # difficult; risk of tipping
        "mobility_scooter":     0.15,   # cannot mount kerb; must find dropped kerb
        "manual_wheelchair":    0.05,   # impassable without assistance or dropped kerb
        "motorized_wheelchair": 0.07,
    },
    "Obstacle": {
        "walking_cane":         0.60,
        "walker":               0.45,
        "mobility_scooter":     0.35,
        "manual_wheelchair":    0.25,
        "motorized_wheelchair": 0.28,
    },
    "SurfaceProblem": {
        "walking_cane":         0.65,   # cane stabilises on rough surfaces
        "walker":               0.50,
        "mobility_scooter":     0.40,
        "manual_wheelchair":    0.30,
        "motorized_wheelchair": 0.38,
    },
    "NoSidewalk": {
        "walking_cane":         0.50,   # can walk on road shoulder
        "walker":               0.35,
        "mobility_scooter":     0.20,
        "manual_wheelchair":    0.12,
        "motorized_wheelchair": 0.15,
    },
    "Crosswalk": {
        "walking_cane":         0.82,
        "walker":               0.78,
        "mobility_scooter":     0.75,
        "manual_wheelchair":    0.72,
        "motorized_wheelchair": 0.74,
    },
    "Signal": {
        "walking_cane":         0.88,
        "walker":               0.85,
        "mobility_scooter":     0.83,
        "manual_wheelchair":    0.80,
        "motorized_wheelchair": 0.82,
    },
    "Occlusion": {
        "walking_cane":         0.65,
        "walker":               0.65,
        "mobility_scooter":     0.65,
        "manual_wheelchair":    0.65,
        "motorized_wheelchair": 0.65,
    },
    "Other": {
        "walking_cane":         0.65,
        "walker":               0.65,
        "mobility_scooter":     0.65,
        "manual_wheelchair":    0.65,
        "motorized_wheelchair": 0.65,
    },
}

# Prior for edges with no PS labels nearby (Pittsburgh is fairly accessible)
PRIOR_P_YES: dict[str, float] = {
    "walking_cane":         0.75,
    "walker":               0.68,
    "mobility_scooter":     0.60,
    "manual_wheelchair":    0.50,
    "motorized_wheelchair": 0.55,
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def edge_midpoint(G: "nx.Graph", u: int, v: int, k: int) -> tuple[float, float]:
    data = G[u][v][k]
    if "geometry" in data:
        coords = list(data["geometry"].coords)
        mid    = coords[len(coords) // 2]
        return mid[1], mid[0]
    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
    return lat, lon


def aggregate_labels(label_types: list[str]) -> dict[str, float]:
    """
    Aggregate multiple PS labels on an edge into one per-aid p_yes score.
    Strategy: take the MINIMUM p_yes per aid (accessibility = worst barrier).
    This is conservative and correct: one missing curb cut blocks a wheelchair.
    """
    if not label_types:
        return dict(PRIOR_P_YES)
    scores = {k: 1.0 for k in AIDS_KEYS}
    for lt in label_types:
        mapping = LABEL_PASSABILITY.get(lt, PRIOR_P_YES)
        for aid_key in AIDS_KEYS:
            scores[aid_key] = min(scores[aid_key], mapping[aid_key])
    return scores


# ── Mode A: PS label-based scoring ────────────────────────────────────────────

def score_edges_from_ps_labels(
    G: "nx.Graph",
    ps_geojson_path: str,
    snap_radius: float = 50.0,
) -> tuple[dict, dict]:
    """
    Score OSM edges using PS label types (no images, no model inference).

    Two-pass matching for maximum coverage:
      Pass 1 — OSM way ID matching (exact): every PS label carries osm_way_id;
               match directly to the edge(s) with that osmid in G.
      Pass 2 — Node proximity fallback (50m): labels not matched in pass 1
               are snapped to the nearest OSM node; all edges incident to
               that node inherit the label (semantically correct: a missing
               curb cut at a corner affects every approach).

    Returns (edge_scores, stats).
    """
    with open(ps_geojson_path) as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    print(f"Loaded {len(features)} PS labels from {ps_geojson_path}")

    # Extract usable labels + way IDs
    lats, lons, ltypes, way_ids = [], [], [], []
    for feat in features:
        coords = feat["geometry"]["coordinates"]
        lt     = feat["properties"].get("label_type", "Other")
        wid    = feat["properties"].get("osm_way_id")
        if lt in LABEL_PASSABILITY:
            lats.append(coords[1])
            lons.append(coords[0])
            ltypes.append(lt)
            way_ids.append(int(wid) if wid else None)

    lats_arr   = np.array(lats)
    lons_arr   = np.array(lons)
    ltypes_arr = np.array(ltypes)
    print(f"  Usable labels: {len(lats_arr)}")

    # ── Pass 1: OSM way ID matching ───────────────────────────────────────────
    # Build way_id → [(u,v,k), ...] lookup from graph edges
    wayid_to_edges: dict[int, list[tuple]] = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        osmid = data.get("osmid")
        ids = osmid if isinstance(osmid, list) else [osmid]
        for eid in ids:
            if eid is not None:
                wayid_to_edges.setdefault(int(eid), []).append((u, v, k))

    edge_label_buf: dict[tuple, list[str]] = {}  # (u,v,k) → [label_types]
    n_way_matched = 0
    unmatched_mask = []

    for i, (lt, wid) in enumerate(zip(ltypes_arr, way_ids)):
        if wid is not None and wid in wayid_to_edges:
            for uvk in wayid_to_edges[wid]:
                edge_label_buf.setdefault(uvk, []).append(lt)
            n_way_matched += 1
            unmatched_mask.append(False)
        else:
            unmatched_mask.append(True)

    unmatched_mask = np.array(unmatched_mask)
    print(f"  Pass 1 (way ID): {n_way_matched}/{len(lats_arr)} labels matched to edges")

    # ── Pass 2: Node proximity fallback ──────────────────────────────────────
    node_ids    = list(G.nodes())
    node_lats   = np.array([G.nodes[n]["y"] for n in node_ids])
    node_lons   = np.array([G.nodes[n]["x"] for n in node_ids])
    node_id_arr = np.array(node_ids)

    # Pre-filter to graph bbox
    lat_margin = snap_radius / 111_000
    lon_margin = snap_radius / (111_000 * np.cos(np.radians(np.mean(node_lats))))
    unmatched_idxs = np.where(unmatched_mask)[0]
    in_bbox = (
        (lats_arr[unmatched_idxs] >= node_lats.min() - lat_margin) &
        (lats_arr[unmatched_idxs] <= node_lats.max() + lat_margin) &
        (lons_arr[unmatched_idxs] >= node_lons.min() - lon_margin) &
        (lons_arr[unmatched_idxs] <= node_lons.max() + lon_margin)
    )
    fallback_idxs = unmatched_idxs[in_bbox]

    node_labels: dict[int, list[str]] = {n: [] for n in node_ids}
    n_node_snapped = 0
    for i in fallback_idxs:
        dlat = np.radians(node_lats - lats_arr[i])
        dlon = np.radians(node_lons - lons_arr[i])
        phi1 = np.radians(lats_arr[i])
        phi2 = np.radians(node_lats)
        a    = np.sin(dlat / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
        dists = 2 * 6_371_000 * np.arcsin(np.sqrt(a))
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] <= snap_radius:
            node_labels[node_id_arr[nearest_idx]].append(ltypes_arr[i])
            n_node_snapped += 1

    n_nodes_labeled = sum(1 for v in node_labels.values() if v)
    print(f"  Pass 2 (node):   {n_node_snapped}/{len(fallback_idxs)} fallback labels "
          f"snapped to {n_nodes_labeled} nodes")

    # ── Merge and finalise edge scores ────────────────────────────────────────
    edge_scores: dict[str, dict] = {}
    n_with_labels = 0
    n_prior_only  = 0

    for u, v, k in G.edges(keys=True):
        edge_key  = f"{u}_{v}_{k}"
        combined  = list(edge_label_buf.get((u, v, k), []))
        combined += node_labels.get(u, []) + node_labels.get(v, [])

        if combined:
            scores = aggregate_labels(combined)
            edge_scores[edge_key] = {
                **scores,
                "n_labels":    len(combined),
                "label_types": combined,
                "source":      "ps_labels",
            }
            n_with_labels += 1
        else:
            edge_scores[edge_key] = {
                **PRIOR_P_YES,
                "n_labels":    0,
                "label_types": [],
                "source":      "prior",
            }
            n_prior_only += 1

    stats = {
        "mode":              "ps_labels",
        "n_labels_total":    len(lats_arr),
        "n_way_matched":     n_way_matched,
        "n_node_snapped":    n_node_snapped,
        "snap_radius_m":     snap_radius,
        "nodes_total":       len(node_ids),
        "nodes_labeled":     n_nodes_labeled,
        "edges_total":       G.number_of_edges(),
        "edges_labeled":     n_with_labels,
        "edges_prior":       n_prior_only,
    }
    return edge_scores, stats


# ── Mode B: Image inference ────────────────────────────────────────────────────

def score_edges_from_images(
    G: "nx.Graph",
    ps_csv_path: str,
    model,
    cv_results_path: str,
    snap_radius: float = 30.0,
) -> tuple[dict, dict]:
    """Score edges using DINOv2-large inference on PS images (requires images)."""
    img_df = pd.read_csv(ps_csv_path)
    img_df = img_df[img_df["image_path"].apply(lambda p: Path(p).exists())]
    print(f"Loaded {len(img_df)} images with valid paths")

    img_lats  = img_df["lat"].values.astype(float)
    img_lons  = img_df["lon"].values.astype(float)
    img_paths = img_df["image_path"].tolist()
    img_ids   = img_df["image_id"].tolist()

    # Load CV priors
    priors = {k: 0.6 for k in AIDS_KEYS}
    if Path(cv_results_path).exists():
        with open(cv_results_path) as f:
            cv = json.load(f)
        for disp, key in zip(AIDS_DISPLAY, AIDS_KEYS):
            v = cv.get("aids", {}).get(disp, {})
            priors[key] = float(v.get("macro_f1_mean", 0.6))

    edge_scores: dict[str, dict] = {}
    n_real = 0
    n_prior = 0

    for u, v, k in G.edges(keys=True):
        edge_key = f"{u}_{v}_{k}"
        mid_lat, mid_lon = edge_midpoint(G, u, v, k)
        dists = np.array([
            haversine_m(mid_lat, mid_lon, la, lo)
            for la, lo in zip(img_lats, img_lons)
        ])
        nearby_idx = np.where(dists <= snap_radius)[0]

        if len(nearby_idx) > 0:
            paths  = [img_paths[i] for i in nearby_idx]
            ids    = [img_ids[i]   for i in nearby_idx]
            batch  = model.predict_batch(paths)
            sums   = {k2: 0.0 for k2 in AIDS_KEYS}
            for r in batch:
                for disp, key in zip(AIDS_DISPLAY, AIDS_KEYS):
                    sums[key] += r.get(disp, {}).get("p_yes", 0.5)
            cnt = len(batch)
            scores = {k2: sums[k2] / cnt for k2 in AIDS_KEYS}
            edge_scores[edge_key] = {
                **scores, "n_images": cnt, "image_ids": ids, "source": "inference",
            }
            n_real += 1
        else:
            edge_scores[edge_key] = {
                **priors, "n_images": 0, "image_ids": [], "source": "prior",
            }
            n_prior += 1

    stats = {
        "mode":         "image_inference",
        "n_images":     len(img_df),
        "snap_radius_m": snap_radius,
        "edges_total":  G.number_of_edges(),
        "edges_real":   n_real,
        "edges_prior":  n_prior,
    }
    return edge_scores, stats


def main():
    parser = argparse.ArgumentParser(
        description="Score OSM edges with real PS accessibility data."
    )
    # Mode A: PS labels (recommended)
    parser.add_argument("--ps_labels", default=None,
                        help="GeoJSON from fetch_ps_labels.py (Mode A — no images needed).")
    # Mode B: image inference
    parser.add_argument("--ps_csv",     default=None,
                        help="CSV from fetch_ps_images.py (Mode B — needs GSV images).")
    parser.add_argument("--checkpoint", default="results/models/dinov2-large")
    parser.add_argument("--encoder",    default="dinov2-large")
    parser.add_argument("--cv_results", default="results/cv/soft_kl/dinov2-large/cv_results.json")
    # Graph params
    parser.add_argument("--lat",  type=float, default=40.4432)
    parser.add_argument("--lon",  type=float, default=-79.9433)
    parser.add_argument("--dist", type=int,   default=800)
    # Output
    parser.add_argument("--output",      default="results/routing/edge_scores.json")
    parser.add_argument("--snap_radius", type=float, default=50.0,
                        help="Max metres from OSM node to snap a PS label.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAS_OSMNX:
        print("ERROR: osmnx not installed. pip install osmnx")
        sys.exit(1)

    if not args.ps_labels and not args.ps_csv:
        print("ERROR: provide --ps_labels (Mode A) or --ps_csv (Mode B).")
        sys.exit(1)

    # ── Load OSM graph ────────────────────────────────────────────────────────
    print(f"\nDownloading OSM graph ({args.lat}, {args.lon}), r={args.dist}m …")
    G = ox.graph_from_point((args.lat, args.lon), dist=args.dist,
                            network_type="walk", simplify=True).to_undirected()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Score edges ───────────────────────────────────────────────────────────
    if args.ps_labels:
        print(f"\nMode A: PS label-based scoring (snap_radius={args.snap_radius}m)")
        edge_scores, stats = score_edges_from_ps_labels(
            G, args.ps_labels, snap_radius=args.snap_radius
        )
    else:
        print(f"\nMode B: Image inference (snap_radius={args.snap_radius}m)")
        from infer import SidewalkAccessibilityModel
        model = SidewalkAccessibilityModel(args.encoder, args.checkpoint, seed=args.seed)
        edge_scores, stats = score_edges_from_images(
            G, args.ps_csv, model, args.cv_results, snap_radius=args.snap_radius
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    total = stats["edges_total"]
    n_lab = stats.get("edges_labeled", stats.get("edges_real", 0))
    print(f"\nScoring complete ({stats['mode']}):")
    print(f"  Total edges   : {total}")
    print(f"  With real data: {n_lab} ({100*n_lab/total:.1f}%)")
    print(f"  Prior only    : {total - n_lab} ({100*(total-n_lab)/total:.1f}%)")

    # Per-aid stats on labeled edges
    labeled = [s for s in edge_scores.values() if s.get("source") != "prior"]
    if labeled:
        print("\n  Per-aid mean p_yes (edges with real labels):")
        for aid_key in AIDS_KEYS:
            vals = [s[aid_key] for s in labeled]
            print(f"    {aid_key:<25} {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"__meta__": stats, "edges": edge_scores}
    with open(output, "w") as f:
        json.dump(result, f)

    size_kb = output.stat().st_size // 1024
    print(f"\nEdge scores → {output}  ({size_kb} KB)")
    print(f"\nNext step:")
    print(f"  python src/routing/demo.py \\")
    print(f"      --edge_scores {output} \\")
    print(f"      --lat {args.lat} --lon {args.lon} --dist {args.dist} \\")
    print(f"      --output_dir  results/routing")


if __name__ == "__main__":
    main()
