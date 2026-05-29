#!/usr/bin/env python3
"""
Fetch Street View images for route trials in the user study.

For each mobility aid:
  1. Runs accessibility-aware Dijkstra (bc=8) and standard Dijkstra on the
     Pittsburgh OSM graph using real DINOv2-large edge scores.
  2. Samples 3 waypoints evenly along each route.
  3. Finds the nearest Project Sidewalk label for each waypoint (already has
     pano_id + heading + pitch — no GSV API key needed).
  4. Downloads the image via streetviewpixels (same as our generalization pipeline).
  5. Saves images to data/images/ and prints the trials.json snippet.

Usage:
    cd apps/accessibility-study
    python fetch_route_images.py

    # Dry-run (show waypoints without downloading):
    python fetch_route_images.py --dry-run

    # Custom search radius for nearest PS label (default 60m):
    python fetch_route_images.py --radius 80

Outputs:
    data/images/route_{aid}_{type}_scene{n}.jpg   (type = acc | std)
    Prints a trials.json snippet for the routeTrials block.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
import requests

# ── Paths (relative to this script's directory) ───────────────────────────────
HERE        = Path(__file__).parent
PROJECT     = HERE.parent.parent
GRAPH_PATH  = PROJECT / "results" / "routing" / "pittsburgh_graph.graphml"
SCORES_PATH = PROJECT / "results" / "routing" / "edge_scores_full.json"
PS_LABELS   = PROJECT / "data" / "routing" / "pittsburgh" / "ps_labels_fresh.geojson"
OUT_DIR     = HERE / "data" / "images"

# ── Constants ─────────────────────────────────────────────────────────────────
BARRIER_COST = 8.0
SEED         = 42
N_SCENES     = 3        # images per route
IMG_SIZE     = "640x640"
GSV_URL      = "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"

AIDS = [
    ("Walking cane",          "walking_cane"),
    ("Walker",                "walker"),
    ("Mobility scooter",      "mobility_scooter"),
    ("Manual wheelchair",     "manual_wheelchair"),
    ("Motorized wheelchair",  "motorized_wheelchair"),
]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing in degrees from point 1 → point 2."""
    dλ = math.radians(lon2 - lon1)
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ── Graph helpers ─────────────────────────────────────────────────────────────

def load_graph() -> nx.MultiDiGraph:
    G = nx.read_graphml(str(GRAPH_PATH))
    # Cast numeric attributes stored as strings
    for n, d in G.nodes(data=True):
        d["x"] = float(d["x"])
        d["y"] = float(d["y"])
    for u, v, k, d in G.edges(keys=True, data=True):
        d["length"] = float(d.get("length", 1.0))
    return G


def assign_scores(G: nx.MultiDiGraph, scores: dict) -> None:
    for u, v, k in G.edges(keys=True):
        entry = scores.get(f"{u}_{v}_{k}") or scores.get(f"{v}_{u}_{k}") or {}
        for aid_label, aid_key in AIDS:
            G[u][v][k][f"p_yes_{aid_key}"] = float(entry.get(aid_key, 0.5))


def routing_weight(G: nx.MultiDiGraph, aid_key: str):
    def weight_fn(u, v, data):
        # MultiDiGraph passes the dict of the best edge
        best = min(
            (G[u][v][k] for k in G[u][v]),
            key=lambda d: d.get("length", 1.0),
        )
        length = best.get("length", 1.0)
        p_yes  = best.get(f"p_yes_{aid_key}", 0.5)
        return length * (1.0 + BARRIER_COST * (1.0 - p_yes))
    return weight_fn


def pick_od(G: nx.MultiDiGraph) -> tuple:
    rng   = np.random.default_rng(SEED)
    nodes = list(G.nodes())
    xs    = np.array([G.nodes[n]["x"] for n in nodes])
    ys    = np.array([G.nodes[n]["y"] for n in nodes])
    cx, cy = xs.mean(), ys.mean()
    orig_cands = [n for n, x, y in zip(nodes, xs, ys) if x < cx and y < cy]
    dest_cands = [n for n, x, y in zip(nodes, xs, ys) if x > cx and y > cy]
    return str(rng.choice(orig_cands)), str(rng.choice(dest_cands))


def best_waypoints(route: list, G: nx.MultiDiGraph,
                   ps_labels: list[dict], n: int = N_SCENES,
                   max_radius_m: float = 200.0,
                   excluded_panos: set[str] | None = None) -> list[dict]:
    """
    For every edge midpoint along the route, find the nearest PS label.
    Return the n edge midpoints with the closest labels, spread across
    early/mid/late thirds of the route to avoid clustering.
    excluded_panos: pano_ids already used by a prior route (e.g. the std
    route for this aid) that should not be reused.
    """
    candidates = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        hdg = bearing(G.nodes[u]["y"], G.nodes[u]["x"],
                      G.nodes[v]["y"], G.nodes[v]["x"])
        lbls = nearest_labels(lat, lon, ps_labels, max_radius_m, top_k=30)
        if lbls:
            dist = haversine_m(lat, lon, lbls[0]["lat"], lbls[0]["lon"])
            # Deduplicate pano_ids within fallback list
            seen = set()
            unique_lbls = []
            for lb in lbls:
                if lb["pano_id"] not in seen:
                    seen.add(lb["pano_id"])
                    unique_lbls.append(lb)
            candidates.append({
                "edge_idx": i,
                "lat": lat, "lon": lon, "heading": hdg,
                "label": unique_lbls[0],
                "fallbacks": unique_lbls[1:],
                "dist_m": dist,
            })

    if not candidates:
        return []

    # Divide route into n thirds; pick best candidate from each third,
    # enforcing unique pano_ids across all selected segments and excluded_panos.
    # Falls back to allowing excluded panos only if no fresh pano is available.
    total = len(route) - 1
    selected = []
    used_panos: set[str] = set(excluded_panos or [])
    fresh_used: set[str] = set()  # tracks only panos we picked in this call

    for seg in range(n):
        lo = (seg * total) // n
        hi = ((seg + 1) * total) // n
        seg_cands = [c for c in candidates if lo <= c["edge_idx"] < hi]
        if not seg_cands:
            seg_cands = candidates

        seg_cands_sorted = sorted(seg_cands, key=lambda c: c["dist_m"])
        chosen = None

        # First pass: prefer panos not in used_panos at all
        for cand in seg_cands_sorted:
            all_lbls = [cand["label"]] + cand.get("fallbacks", [])
            for lbl in all_lbls:
                if lbl["pano_id"] not in used_panos:
                    chosen = {**cand, "label": lbl,
                               "fallbacks": [l for l in all_lbls
                                             if l["pano_id"] != lbl["pano_id"]]}
                    break
            if chosen:
                break

        # Second pass: if nothing fresh, allow excluded (cross-route) panos
        # but still avoid within-route duplicates
        if not chosen:
            for cand in seg_cands_sorted:
                all_lbls = [cand["label"]] + cand.get("fallbacks", [])
                for lbl in all_lbls:
                    if lbl["pano_id"] not in fresh_used:
                        chosen = {**cand, "label": lbl,
                                   "fallbacks": [l for l in all_lbls
                                                 if l["pano_id"] != lbl["pano_id"]]}
                        break
                if chosen:
                    break

        if chosen:
            used_panos.add(chosen["label"]["pano_id"])
            fresh_used.add(chosen["label"]["pano_id"])
            selected.append(chosen)

    return selected


# ── PS label lookup ───────────────────────────────────────────────────────────

def load_ps_labels(path: Path) -> list[dict]:
    with open(path) as f:
        g = json.load(f)
    labels = []
    for feat in g["features"]:
        p = feat["properties"]
        if not p.get("pano_id"):
            continue
        lon, lat = feat["geometry"]["coordinates"]
        labels.append({
            "pano_id": p["pano_id"],
            "heading": float(p.get("heading", 0)),
            "pitch":   float(p.get("pitch", -10)),
            "lat": lat,
            "lon": lon,
            "label_type": p.get("label_type", ""),
        })
    return labels


def nearest_labels(lat: float, lon: float, labels: list[dict],
                   max_radius_m: float = 60.0, top_k: int = 10) -> list[dict]:
    """Return up to top_k nearest labels within radius, sorted by distance."""
    candidates = []
    for lbl in labels:
        d = haversine_m(lat, lon, lbl["lat"], lbl["lon"])
        if d <= max_radius_m:
            candidates.append((d, lbl))
    candidates.sort(key=lambda x: x[0])
    return [lbl for _, lbl in candidates[:top_k]]


def nearest_label(lat: float, lon: float, labels: list[dict],
                  max_radius_m: float = 60.0) -> dict | None:
    results = nearest_labels(lat, lon, labels, max_radius_m, top_k=1)
    return results[0] if results else None


# ── Image download ────────────────────────────────────────────────────────────

MIN_IMAGE_BYTES = 10_000  # < 10 KB → expired/black panorama


def download_image(pano_id: str, heading: float, pitch: float, out_path: Path,
                   size: str = IMG_SIZE, retries: int = 3) -> bool:
    params = {
        "panoid": pano_id,
        "cb_client": "maps_sv.tactile",
        "w": size.split("x")[0],
        "h": size.split("x")[1],
        "yaw": f"{heading:.1f}",
        "pitch": f"{pitch:.1f}",
        "thumbfov": "90",
    }
    for attempt in range(retries):
        try:
            r = requests.get(GSV_URL, params=params, timeout=15)
            if r.status_code == 200 and len(r.content) >= MIN_IMAGE_BYTES:
                out_path.write_bytes(r.content)
                return True
            if r.status_code == 200 and len(r.content) < MIN_IMAGE_BYTES:
                return False  # expired pano — no point retrying same pano
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print waypoints without downloading images")
    parser.add_argument("--radius", type=float, default=60.0,
                        help="Max search radius (m) for nearest PS label")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading graph …")
    G = load_graph()

    print("Loading edge scores …")
    with open(SCORES_PATH) as f:
        raw = json.load(f)
    scores = raw.get("edges", raw)
    assign_scores(G, scores)

    print("Loading PS labels …")
    ps_labels = load_ps_labels(PS_LABELS)
    print(f"  {len(ps_labels)} labels available")

    origin, dest = pick_od(G)
    print(f"OD pair: {origin} → {dest}")
    print(f"  origin: lat={G.nodes[origin]['y']:.5f}  lon={G.nodes[origin]['x']:.5f}")
    print(f"  dest:   lat={G.nodes[dest]['y']:.5f}  lon={G.nodes[dest]['x']:.5f}")

    trial_images: dict[str, dict] = {}  # aid_key → {acc: [...], std: [...]}

    for aid_label, aid_key in AIDS:
        print(f"\n── {aid_label} ──")
        trial_images[aid_key] = {"acc": [], "std": []}

        # Track panos used by the std route so acc route can avoid them
        std_panos_used: set[str] = set()

        for route_type, weight in [("std", "length"),
                                   ("acc", routing_weight(G, aid_key))]:
            try:
                route = nx.shortest_path(G, origin, dest, weight=weight)
            except nx.NetworkXNoPath:
                print(f"  [{route_type}] No path found — skipping")
                continue

            excluded = std_panos_used if route_type == "acc" else None
            waypoints = best_waypoints(route, G, ps_labels, N_SCENES, args.radius,
                                       excluded_panos=excluded)
            print(f"  [{route_type}] {len(route)} nodes → {len(waypoints)} scenes found")

            if not waypoints:
                print(f"    no PS labels within {args.radius}m of any edge midpoint")
                continue

            for i, wp in enumerate(waypoints, start=1):
                label = wp["label"]
                # Prefer the PS auditor's heading (already facing the sidewalk);
                # fall back to route travel direction if PS heading is near zero.
                heading = label["heading"] if label["heading"] > 1.0 else wp["heading"]
                fname   = f"route_{aid_key}_{route_type}_scene{i}.jpg"
                fpath   = OUT_DIR / fname

                print(f"    scene {i}: pano={label['pano_id']}  "
                      f"hdg={heading:.0f}°  pitch={label['pitch']:.0f}°  "
                      f"nearest_label={wp['dist_m']:.0f}m"
                      + (" [DRY RUN]" if args.dry_run else ""))

                if not args.dry_run:
                    ok = download_image(label["pano_id"], heading, label["pitch"], fpath)
                    if not ok:
                        # pano expired — try fallbacks.
                        # For acc routes, prefer panos not already used in the std route,
                        # but fall back to std panos if no fresh one is available.
                        fallbacks = wp.get("fallbacks", [])
                        fresh_fbs = [fb for fb in fallbacks
                                     if fb["pano_id"] not in std_panos_used]
                        std_fbs   = [fb for fb in fallbacks
                                     if fb["pano_id"] in std_panos_used]
                        ordered = (fresh_fbs + std_fbs) if route_type == "acc" else fallbacks
                        for fb in ordered:
                            fb_heading = fb["heading"] if fb["heading"] > 1.0 else wp["heading"]
                            print(f"      pano expired, trying fallback {fb['pano_id']}")
                            ok = download_image(fb["pano_id"], fb_heading, fb["pitch"], fpath)
                            if ok:
                                label = fb
                                heading = fb_heading
                                break
                    status = "✓" if ok else "FAILED (all panos expired)"
                    print(f"      → {fname}  [{status}]")

                trial_images[aid_key][route_type].append({
                    "src":     f"../../apps/accessibility-study/data/images/{fname}",
                    "id":      fname.replace(".jpg", ""),
                    "pano_id": label["pano_id"],
                    "heading": round(heading, 1),
                    "lat":     round(wp["lat"], 6),
                    "lon":     round(wp["lon"], 6),
                })
                if route_type == "std":
                    std_panos_used.add(label["pano_id"])

    # ── Print trials.json snippet ─────────────────────────────────────────────
    print("\n\n── trials.json snippet (routeTrials) ──\n")
    for aid_label, aid_key in AIDS:
        imgs = trial_images.get(aid_key, {})
        acc  = imgs.get("acc", [])
        std  = imgs.get("std", [])
        if not acc and not std:
            continue
        print(f'  // {aid_label}')
        print(f'  // accessible route scenes: {[x["id"] for x in acc]}')
        print(f'  // standard route scenes:   {[x["id"] for x in std]}')

    print("\nDone. Review images in:", OUT_DIR)
    print("Then update apps/accessibility-study/data/trials.json with the new image paths.")


if __name__ == "__main__":
    main()
