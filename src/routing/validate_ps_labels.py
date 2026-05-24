#!/usr/bin/env python3
"""
Independent route quality validation using Project Sidewalk label severity.

For the 40 OD pairs in the ORS comparison (seed=42, min_dist=300m), compares:
  - Standard Dijkstra (shortest path)
  - Accessibility-weighted Dijkstra (our method, beta_c=8)

Using PS labels as INDEPENDENT ground truth:
  - PS severity scores (1–5) were NEVER used in routing optimisation.
  - PS label types were used for 40.4% of edge costs (via lookup table),
    but not for the 51.3% of inference-scored edges — noted as a caveat.
  - ORS uses OSM wheelchair tags only, making the ORS comparison fully
    independent of PS labels.

Primary metrics (per route, per pair):
  - n_neg_labels : count of Obstacle / SurfaceProblem / NoCurbRamp labels
                   snapped within snap_radius metres of route edges
  - mean_severity: mean severity (1–5) of those negative labels

Usage:
    source venv/bin/activate
    python src/routing/validate_ps_labels.py \
        --graph         results/routing/pittsburgh_graph.graphml \
        --edge_scores   results/routing/edge_scores_full.json \
        --ps_labels     data/routing/pittsburgh/ps_labels.geojson \
        --ors_records   results/routing/ors_comparison/records.json \
        --output_dir    results/routing/ps_label_validation \
        --snap_radius   50 \
        --barrier_cost  8 \
        --seed          42 \
        --min_dist      300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import networkx as nx
import osmnx as ox
from scipy.stats import wilcoxon

AIDS = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]
AIDS_KEYS = [
    "walking_cane",
    "walker",
    "mobility_scooter",
    "manual_wheelchair",
    "motorized_wheelchair",
]
AID_DISPLAY_TO_KEY = dict(zip(AIDS, AIDS_KEYS))

NEGATIVE_LABEL_TYPES = {"Obstacle", "SurfaceProblem", "NoCurbRamp", "NoSidewalk"}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_graph(graph_path: str, edge_scores_path: str) -> nx.Graph:
    G = ox.load_graphml(graph_path)
    G = G.to_undirected()
    with open(edge_scores_path) as f:
        raw = json.load(f)
    edge_scores = raw.get("edges", raw)
    for u, v, k, data in G.edges(keys=True, data=True):
        key = f"{u}_{v}_{k}"
        scores = edge_scores.get(key, {})
        for aid_key in AIDS_KEYS:
            data[aid_key] = scores.get(aid_key, 0.6)
        data["source"] = scores.get("source", "prior")
    return G


def load_ps_labels(ps_path: str) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    with open(ps_path) as f:
        geojson = json.load(f)
    lats, lons, ltypes, sevs = [], [], [], []
    for feat in geojson["features"]:
        lt = feat["properties"].get("label_type", "")
        if lt not in NEGATIVE_LABEL_TYPES:
            continue
        coords = feat["geometry"]["coordinates"]
        sev = feat["properties"].get("severity")
        if sev is None:
            continue
        lats.append(coords[1])
        lons.append(coords[0])
        ltypes.append(lt)
        sevs.append(float(sev))
    return np.array(lats), np.array(lons), ltypes, np.array(sevs)


def sample_od_pairs(G: nx.Graph, n: int, min_dist: float, seed: int) -> list[tuple]:
    random.seed(seed)
    np.random.seed(seed)
    nodes = list(G.nodes())
    node_coords = {nd: (G.nodes[nd]["y"], G.nodes[nd]["x"]) for nd in nodes}
    pairs, attempts = [], 0
    while len(pairs) < n and attempts < n * 30:
        o, d = random.sample(nodes, 2)
        lat1, lon1 = node_coords[o]
        lat2, lon2 = node_coords[d]
        if haversine_m(lat1, lon1, lat2, lon2) >= min_dist:
            pairs.append((o, d))
        attempts += 1
    return pairs


def acc_weight(aid_key: str, beta_c: float):
    def _w(u, v, data):
        p = data.get(aid_key, 0.6)
        length = data.get("length", 1.0)
        return length * (1 + beta_c * (1 - p))
    return _w


def route_edge_midpoints(G: nx.Graph, path: list) -> list[tuple[float, float]]:
    midpoints = []
    for u, v in zip(path[:-1], path[1:]):
        data = G[u][v]
        k = list(data.keys())[0]
        d = data[k]
        if "geometry" in d:
            coords = list(d["geometry"].coords)
            mid = coords[len(coords) // 2]
            midpoints.append((mid[1], mid[0]))
        else:
            lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
            lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
            midpoints.append((lat, lon))
    return midpoints


def ps_labels_on_route(
    midpoints: list[tuple],
    ps_lats: np.ndarray,
    ps_lons: np.ndarray,
    ps_ltypes: list,
    ps_sevs: np.ndarray,
    snap_radius: float,
) -> dict:
    if not midpoints or len(ps_lats) == 0:
        return {"n_neg": 0, "mean_sev": np.nan, "label_types": []}

    seen_labels = set()
    neg_sevs = []
    neg_types = []

    for mlat, mlon in midpoints:
        dists = np.array([haversine_m(mlat, mlon, la, lo) for la, lo in zip(ps_lats, ps_lons)])
        nearby = np.where(dists <= snap_radius)[0]
        for idx in nearby:
            if idx not in seen_labels:
                seen_labels.add(idx)
                neg_sevs.append(ps_sevs[idx])
                neg_types.append(ps_ltypes[idx])

    return {
        "n_neg":      len(neg_sevs),
        "mean_sev":   float(np.mean(neg_sevs)) if neg_sevs else 0.0,
        "label_types": neg_types,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate routing quality using PS label severity scores."
    )
    parser.add_argument("--graph",        default="results/routing/pittsburgh_graph.graphml")
    parser.add_argument("--edge_scores",  default="results/routing/edge_scores_full.json")
    parser.add_argument("--ps_labels",    default="data/routing/pittsburgh/ps_labels.geojson")
    parser.add_argument("--ors_records",  default="results/routing/ors_comparison/records.json",
                        help="ORS comparison records to get the 40 routable pair indices.")
    parser.add_argument("--output_dir",   default="results/routing/ps_label_validation")
    parser.add_argument("--snap_radius",  type=float, default=50.0)
    parser.add_argument("--barrier_cost", type=float, default=8.0)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--min_dist",     type=int, default=300)
    parser.add_argument("--n_pairs",      type=int, default=50)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load graph ────────────────────────────────────────────────────────────
    print("Loading graph + edge scores …")
    G = load_graph(args.graph, args.edge_scores)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Load PS labels (negative types only, with severity) ───────────────────
    print("Loading PS labels …")
    ps_lats, ps_lons, ps_ltypes, ps_sevs = load_ps_labels(args.ps_labels)
    print(f"  {len(ps_lats)} negative labels with severity")
    from collections import Counter
    print(" ", dict(Counter(ps_ltypes)))

    # ── Reproduce the same 50 OD pairs ───────────────────────────────────────
    print(f"\nReproducing OD pairs (seed={args.seed}, min_dist={args.min_dist}m) …")
    od_pairs = sample_od_pairs(G, args.n_pairs, args.min_dist, args.seed)
    print(f"  Sampled {len(od_pairs)} pairs")

    # Identify which pair indices were routable by ORS
    with open(args.ors_records) as f:
        records_ors = json.load(f)
    routable_pair_idxs = set(r["pair_idx"] for r in records_ors if r.get("ors_pyes") is not None)
    print(f"  ORS-routable pairs: {len(routable_pair_idxs)}")

    # ── Run routing for each ORS-routable pair, all aids ─────────────────────
    print(f"\nRunning routing + PS label matching (snap_radius={args.snap_radius}m) …")
    node_coords = {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in G.nodes()}

    records = []
    for i, (o, d) in enumerate(od_pairs):
        if i not in routable_pair_idxs:
            continue

        for aid_disp, aid_key in zip(AIDS, AIDS_KEYS):
            try:
                path_std = nx.shortest_path(G, o, d, weight="length")
                path_acc = nx.shortest_path(G, o, d, weight=acc_weight(aid_key, args.barrier_cost))
            except nx.NetworkXNoPath:
                continue

            mid_std = route_edge_midpoints(G, path_std)
            mid_acc = route_edge_midpoints(G, path_acc)

            ps_std = ps_labels_on_route(mid_std, ps_lats, ps_lons, ps_ltypes, ps_sevs, args.snap_radius)
            ps_acc = ps_labels_on_route(mid_acc, ps_lats, ps_lons, ps_ltypes, ps_sevs, args.snap_radius)

            # Route length
            len_std = sum(G[u][v][list(G[u][v].keys())[0]].get("length", 0) for u, v in zip(path_std[:-1], path_std[1:]))
            len_acc = sum(G[u][v][list(G[u][v].keys())[0]].get("length", 0) for u, v in zip(path_acc[:-1], path_acc[1:]))

            records.append({
                "pair_idx":    i,
                "aid":         aid_disp,
                "std_n_neg":   ps_std["n_neg"],
                "acc_n_neg":   ps_acc["n_neg"],
                "std_sev":     ps_std["mean_sev"],
                "acc_sev":     ps_acc["mean_sev"],
                "std_length":  len_std,
                "acc_length":  len_acc,
                "delta_n_neg": ps_std["n_neg"] - ps_acc["n_neg"],   # positive = ours has fewer
                "delta_sev":   ps_std["mean_sev"] - ps_acc["mean_sev"],  # positive = ours has lower sev
            })

        print(f"  Pair {i} done")

    print(f"\n{len(records)} records ({len(records)//len(AIDS)} pairs × {len(AIDS)} aids)")

    # ── Per-aid summary ───────────────────────────────────────────────────────
    print("\n── Per-aid results ──────────────────────────────────────────────────────")
    print(f"{'Aid':<22} │ {'Std n_neg':>10} {'Acc n_neg':>10} │ {'Std sev':>8} {'Acc sev':>8} │ {'p (Wilcox)':>12}")
    print("─" * 80)

    per_aid_results = {}
    for aid_disp in AIDS:
        recs = [r for r in records if r["aid"] == aid_disp]
        if not recs:
            continue

        std_neg  = [r["std_n_neg"] for r in recs]
        acc_neg  = [r["acc_n_neg"] for r in recs]
        std_sev  = [r["std_sev"]   for r in recs]
        acc_sev  = [r["acc_sev"]   for r in recs]
        diffs    = [r["delta_sev"] for r in recs]

        # Wilcoxon signed-rank on severity differences
        nz = [(a, b) for a, b in zip(std_sev, acc_sev) if a != b]
        if len(nz) >= 3:
            _, p = wilcoxon([x[0] for x in nz], [x[1] for x in nz], alternative="greater")
        else:
            p = np.nan

        per_aid_results[aid_disp] = {
            "std_n_neg_mean": float(np.mean(std_neg)),
            "acc_n_neg_mean": float(np.mean(acc_neg)),
            "std_sev_mean":   float(np.mean(std_sev)),
            "acc_sev_mean":   float(np.mean(acc_sev)),
            "delta_sev_mean": float(np.mean(diffs)),
            "wilcox_p_sev":   float(p) if not np.isnan(p) else None,
            "n_pairs":        len(recs),
        }

        p_str = f"{p:.3f}" if not np.isnan(p) else "  n/a"
        print(f"{aid_disp:<22} │ {np.mean(std_neg):>10.2f} {np.mean(acc_neg):>10.2f} │ "
              f"{np.mean(std_sev):>8.3f} {np.mean(acc_sev):>8.3f} │ {p_str:>12}")

    # ── Overall (mean across aids) ────────────────────────────────────────────
    all_delta_n   = [r["delta_n_neg"] for r in records]
    all_delta_sev = [r["delta_sev"]   for r in records]
    all_std_neg   = [r["std_n_neg"]   for r in records]
    all_acc_neg   = [r["acc_n_neg"]   for r in records]
    all_std_sev   = [r["std_sev"]     for r in records]
    all_acc_sev   = [r["acc_sev"]     for r in records]

    nz_all = [(a, b) for a, b in zip(all_std_sev, all_acc_sev) if a != b]
    if len(nz_all) >= 3:
        _, p_all = wilcoxon([x[0] for x in nz_all], [x[1] for x in nz_all], alternative="greater")
    else:
        p_all = np.nan

    pct_pairs_improved_n = 100 * sum(1 for d in all_delta_n if d > 0) / len(all_delta_n)
    pct_pairs_improved_s = 100 * sum(1 for d in all_delta_sev if d > 0) / len(all_delta_sev)

    print("─" * 80)
    print(f"{'Mean (all aids)':<22} │ {np.mean(all_std_neg):>10.2f} {np.mean(all_acc_neg):>10.2f} │ "
          f"{np.mean(all_std_sev):>8.3f} {np.mean(all_acc_sev):>8.3f} │ {p_all:>12.3f}")
    print(f"\n% pairs where our route has fewer negative labels: {pct_pairs_improved_n:.1f}%")
    print(f"% pairs where our route has lower mean severity:    {pct_pairs_improved_s:.1f}%")
    print(f"Mean reduction in negative label count:   {np.mean(all_delta_n):.2f}")
    print(f"Mean reduction in mean severity:          {np.mean(all_delta_sev):.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        "n_pairs":               len(routable_pair_idxs),
        "n_records":             len(records),
        "snap_radius_m":         args.snap_radius,
        "barrier_cost":          args.barrier_cost,
        "mean_std_n_neg":        float(np.mean(all_std_neg)),
        "mean_acc_n_neg":        float(np.mean(all_acc_neg)),
        "mean_std_sev":          float(np.mean(all_std_sev)),
        "mean_acc_sev":          float(np.mean(all_acc_sev)),
        "mean_delta_n_neg":      float(np.mean(all_delta_n)),
        "mean_delta_sev":        float(np.mean(all_delta_sev)),
        "pct_improved_n_neg":    pct_pairs_improved_n,
        "pct_improved_sev":      pct_pairs_improved_s,
        "wilcox_p_sev_all":      float(p_all) if not np.isnan(p_all) else None,
        "per_aid":               per_aid_results,
        "independence_note": (
            "PS severity scores (1-5) were never used in routing optimisation. "
            "PS label types were used for 40.4% of edge costs (lookup table). "
            "The severity metric is orthogonal to the optimisation input."
        ),
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "records.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nResults saved → {out}/")


if __name__ == "__main__":
    main()
