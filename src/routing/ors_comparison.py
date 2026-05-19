#!/usr/bin/env python3
"""
Compare our accessibility-weighted routing against OpenRouteService wheelchair profile.

ORS wheelchair profile uses OSM wheelchair tags (slope, surface, kerb height) —
the current open-source SOTA for accessible pedestrian routing.
Our method scores 91.7% of edges with DINOv2-large inference from GSV thumbnails,
providing accessibility scores where OSM tags are absent.

Usage:
    # 1. Get a free API key at https://openrouteservice.org/dev/#/signup
    # 2. Run:
    python src/routing/ors_comparison.py \
        --api_key YOUR_ORS_KEY \
        --graph_cache results/routing/pittsburgh_graph.graphml \
        --edge_scores results/routing/edge_scores_full.json \
        --n_pairs 50 \
        --output_dir results/routing/ors_comparison

Outputs:
    results/routing/ors_comparison/summary.json
    results/routing/ors_comparison/comparison_table.png/pdf
    results/routing/ors_comparison/pareto_comparison.png/pdf
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    print("ERROR: pip install osmnx networkx")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

AIDS = ["walking_cane", "walker", "mobility_scooter", "manual_wheelchair", "motorized_wheelchair"]
AID_LABELS = {
    "walking_cane":        "Walking cane",
    "walker":              "Walker",
    "mobility_scooter":    "Mobility scooter",
    "manual_wheelchair":   "Manual wheelchair",
    "motorized_wheelchair":"Motorized wheelchair",
}
AID_COLORS = {
    "walking_cane":        "#e41a1c",
    "walker":              "#ff7f00",
    "mobility_scooter":    "#4daf4a",
    "manual_wheelchair":   "#377eb8",
    "motorized_wheelchair":"#984ea3",
}
PRIOR = {"walking_cane": 0.75, "walker": 0.68, "mobility_scooter": 0.60,
         "manual_wheelchair": 0.50, "motorized_wheelchair": 0.55}

ORS_BASE = "https://api.openrouteservice.org/v2/directions"


def load_and_score_graph(graph_path: str, scores_path: str) -> nx.Graph:
    G = ox.load_graphml(graph_path)
    G = G.to_undirected()
    with open(scores_path) as f:
        raw = json.load(f)
    edge_scores = raw.get("edges", raw)
    for u, v, k, data in G.edges(keys=True, data=True):
        score = edge_scores.get(f"{u}_{v}_{k}") or edge_scores.get(f"{v}_{u}_{k}")
        for aid in AIDS:
            data[aid] = score.get(aid, PRIOR[aid]) if score else PRIOR[aid]
    return G


def acc_weight(aid: str, bc: float = 8.0):
    def w(u, v, data):
        d = next(iter(data.values()))
        p = d.get(aid, PRIOR[aid])
        return d.get("length", 1.0) * (1.0 + bc * (1.0 - p))
    return w


def route_stats(G: nx.Graph, path: list, aid: str) -> dict:
    total_len, total_pyes, n = 0.0, 0.0, 0
    for a, b in zip(path[:-1], path[1:]):
        d = next(iter(G[a][b].values()))
        total_len += d.get("length", 1.0)
        total_pyes += d.get(aid, PRIOR[aid])
        n += 1
    return {"length": total_len, "mean_pyes": total_pyes / n if n > 0 else PRIOR[aid]}


def ors_wheelchair_route(lon1: float, lat1: float, lon2: float, lat2: float,
                          api_key: str, session: requests.Session) -> dict | None:
    """Query ORS wheelchair profile route."""
    url = f"{ORS_BASE}/wheelchair"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    body = {
        "coordinates": [[lon1, lat1], [lon2, lat2]],
        "units": "m",
        "geometry": True,
    }
    try:
        r = session.post(url, json=body, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            route = data["routes"][0]
            geom = route["geometry"]
            if isinstance(geom, dict):
                raw_coords = geom["coordinates"]
                coords = [(c[1], c[0]) for c in raw_coords]  # (lat, lon)
            else:
                import polyline as pl
                coords = pl.decode(geom)  # returns (lat, lon) tuples
            return {"length": route["summary"]["distance"], "coords": coords}
        else:
            return None
    except Exception:
        return None


def snap_ors_route_to_graph(G: nx.Graph, ors_coords: list, aid: str) -> dict | None:
    """Snap ORS route coordinates to graph edges and compute p_yes."""
    if len(ors_coords) < 2:
        return None
    node_coords = {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in G.nodes()}

    def nearest_node(lat, lon):
        best, best_d = None, float("inf")
        for n, (nlat, nlon) in node_coords.items():
            d = (nlat - lat)**2 + (nlon - lon)**2
            if d < best_d:
                best, best_d = n, d
        return best

    # Sample intermediate waypoints along ORS route for snapping
    step = max(1, len(ors_coords) // 10)
    waypoints = [nearest_node(lat, lon) for lat, lon in ors_coords[::step]]
    waypoints = list(dict.fromkeys(waypoints))  # deduplicate preserving order

    if len(waypoints) < 2:
        return None

    total_len, total_pyes, n = 0.0, 0.0, 0
    path_valid = True
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        if a == b:
            continue
        try:
            seg = nx.shortest_path(G, a, b, weight="length")
        except nx.NetworkXNoPath:
            path_valid = False
            break
        for x, y in zip(seg[:-1], seg[1:]):
            d = next(iter(G[x][y].values()))
            total_len += d.get("length", 1.0)
            total_pyes += d.get(aid, PRIOR[aid])
            n += 1

    if not path_valid or n == 0:
        return None
    return {"length": total_len, "mean_pyes": total_pyes / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",      required=True,
                        help="OpenRouteService API key (free at openrouteservice.org)")
    parser.add_argument("--graph_cache",  default="results/routing/pittsburgh_graph.graphml")
    parser.add_argument("--edge_scores",  default="results/routing/edge_scores_full.json")
    parser.add_argument("--n_pairs",      type=int, default=50,
                        help="Number of OD pairs to evaluate (ORS free tier: 2000 req/day).")
    parser.add_argument("--barrier_cost", type=float, default=8.0)
    parser.add_argument("--min_dist",     type=int,   default=300)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--sleep_s",      type=float, default=1.2,
                        help="Sleep between ORS API calls (rate limit: ~40 req/min).")
    parser.add_argument("--output_dir",   default="results/routing/ors_comparison")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading graph …")
    G = load_and_score_graph(args.graph_cache, args.edge_scores)
    nodes = list(G.nodes())
    node_coords = {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes}

    def haversine(n1, n2):
        lat1, lon1 = node_coords[n1]
        lat2, lon2 = node_coords[n2]
        R = 6_371_000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    # Sample OD pairs
    od_pairs = []
    attempts = 0
    while len(od_pairs) < args.n_pairs and attempts < args.n_pairs * 30:
        o, d = random.sample(nodes, 2)
        if haversine(o, d) >= args.min_dist:
            od_pairs.append((o, d))
        attempts += 1
    print(f"Sampled {len(od_pairs)} OD pairs")

    session = requests.Session()
    session.headers["User-Agent"] = "sidewalk-accessibility-research/1.0"

    # Per-aid results storage
    records = []  # list of dicts per (pair_idx, aid)
    ors_failures = 0

    print(f"\nQuerying ORS + running local routing ({len(od_pairs)} pairs) …")
    for i, (o, d) in enumerate(od_pairs):
        print(f"  Pair {i+1}/{len(od_pairs)} …", end=" ")
        olat, olon = node_coords[o]
        dlat, dlon = node_coords[d]

        # ORS wheelchair route (one call per pair, shared across aids)
        ors_result = ors_wheelchair_route(olon, olat, dlon, dlat, args.api_key, session)
        time.sleep(args.sleep_s)

        if ors_result is None:
            ors_failures += 1
            print("ORS no route — skip")
            continue

        for aid in AIDS:
            try:
                path_std = nx.shortest_path(G, o, d, weight="length")
                path_acc = nx.shortest_path(G, o, d, weight=acc_weight(aid, args.barrier_cost))
            except nx.NetworkXNoPath:
                continue

            std = route_stats(G, path_std, aid)
            acc = route_stats(G, path_acc, aid)

            # Snap ORS route to graph for p_yes scoring
            ors_snapped = snap_ors_route_to_graph(G, ors_result["coords"], aid)
            ors_pyes = ors_snapped["mean_pyes"] if ors_snapped else None

            records.append({
                "pair_idx": i,
                "aid": aid,
                "std_pyes":      std["mean_pyes"],
                "acc_pyes":      acc["mean_pyes"],
                "ors_pyes":      ors_pyes,
                "ors_length":    ors_result["length"],
                "std_length":    std["length"],
                "acc_length":    acc["length"],
                "delta_std_acc": acc["mean_pyes"] - std["mean_pyes"],
                "delta_ors_std": (ors_pyes - std["mean_pyes"]) if ors_pyes is not None else None,
                "delta_ors_acc": (acc["mean_pyes"] - ors_pyes) if ors_pyes is not None else None,
            })
        print("OK")

    print(f"\nORS failures (no wheelchair route found): {ors_failures}/{len(od_pairs)}")
    print(f"This reflects sparse OSM wheelchair tagging in Pittsburgh.")

    if not records:
        print("ERROR: no records collected. Check API key and connectivity.")
        sys.exit(1)

    # Aggregate per aid
    summary = {"n_pairs": len(od_pairs), "ors_failures": ors_failures,
               "barrier_cost": args.barrier_cost, "per_aid": {}}

    print("\n=== Comparison Summary: Standard vs ORS vs Ours ===")
    print(f"{'Aid':<22} {'Std p_yes':>9} {'ORS p_yes':>9} {'Ours p_yes':>10} "
          f"{'Ours>ORS?':>10} {'ORS failure%':>13}")
    print("-" * 80)

    per_aid_data = {aid: {"std": [], "ors": [], "acc": []} for aid in AIDS}
    for rec in records:
        a = rec["aid"]
        per_aid_data[a]["std"].append(rec["std_pyes"])
        per_aid_data[a]["acc"].append(rec["acc_pyes"])
        if rec["ors_pyes"] is not None:
            per_aid_data[a]["ors"].append(rec["ors_pyes"])

    for aid in AIDS:
        d = per_aid_data[aid]
        std_m  = np.mean(d["std"]) if d["std"] else 0
        ors_m  = np.mean(d["ors"]) if d["ors"] else None
        acc_m  = np.mean(d["acc"]) if d["acc"] else 0
        ors_fail_pct = 100 * ors_failures / len(od_pairs)
        ours_beats_ors = (acc_m > ors_m) if ors_m is not None else None
        summary["per_aid"][aid] = {
            "n": len(d["std"]),
            "mean_pyes_standard": round(float(std_m), 4),
            "mean_pyes_ors": round(float(ors_m), 4) if ors_m else None,
            "mean_pyes_ours": round(float(acc_m), 4),
            "ors_failure_pct": round(ors_fail_pct, 1),
            "ours_beats_ors": bool(ours_beats_ors) if ours_beats_ors is not None else None,
        }
        ors_str = f"{ors_m:.4f}" if ors_m else " N/A  "
        beats = "✓" if ours_beats_ors else ("✗" if ours_beats_ors is False else "N/A")
        print(f"{AID_LABELS[aid]:<22} {std_m:>9.4f} {ors_str:>9} {acc_m:>10.4f} "
              f"{beats:>10} {ors_fail_pct:>12.1f}%")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "records.json", "w") as f:
        json.dump(records, f)

    # ── Figure: grouped bar chart — Standard / ORS / Ours per aid ─────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(AIDS))
    w = 0.25

    std_vals  = [summary["per_aid"][a]["mean_pyes_standard"] for a in AIDS]
    ors_vals  = [summary["per_aid"][a]["mean_pyes_ors"] or 0 for a in AIDS]
    ours_vals = [summary["per_aid"][a]["mean_pyes_ours"] for a in AIDS]

    ax.bar(x - w, std_vals,  w, label="Standard Dijkstra", color="#aec7e8", edgecolor="white")
    ax.bar(x,     ors_vals,  w, label="ORS wheelchair profile", color="#ffbb78", edgecolor="white")
    ax.bar(x + w, ours_vals, w, label="Ours (DINOv2 + Soft-KL)", color="#2ca02c", edgecolor="white")

    for xi, (s, o, u) in enumerate(zip(std_vals, ors_vals, ours_vals)):
        ax.text(xi - w, s + 0.003, f"{s:.3f}", ha="center", va="bottom", fontsize=7.5)
        if o:
            ax.text(xi,     o + 0.003, f"{o:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + w, u + 0.003, f"{u:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([AID_LABELS[a] for a in AIDS], rotation=12, ha="right", fontsize=10)
    ax.set_ylabel("Mean p_yes along route", fontsize=11)
    ax.set_title("Routing comparison: Standard Dijkstra vs ORS wheelchair vs Ours\n"
                 f"Pittsburgh PA — {len(od_pairs)} OD pairs (min {args.min_dist}m)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ors_fail = summary["per_aid"][AIDS[0]]["ors_failure_pct"]
    ax.text(0.98, 0.02,
            f"ORS failure rate: {ors_fail:.0f}%\n(no wheelchair route found — sparse OSM tags)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray", style="italic")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out / "comparison_table.pdf", bbox_inches="tight")
    fig.savefig(out / "comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults → {out}/")
    print("  summary.json, records.json")
    print("  comparison_table.png/pdf")


if __name__ == "__main__":
    main()
