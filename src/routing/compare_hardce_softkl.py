#!/usr/bin/env python3
"""
Cross-evaluation: Soft-KL vs Hard-CE routing on the canonical Pittsburgh OD pair.

Uses the exact same graph-attribute approach as demo.py to ensure consistent
edge lookups and default values.

Key question: does Hard-CE routing select a route that is objectively WORSE
than the Soft-KL route, when both are evaluated by the calibrated Soft-KL
p_yes metric?

Usage:
    python src/routing/compare_hardce_softkl.py \
        --edge_scores_skl results/routing/edge_scores_full.json \
        --edge_scores_hce results/routing/edge_scores_hard_ce.json \
        --graph_cache     results/routing/pittsburgh_graph.graphml \
        --barrier_cost    8 \
        --output          results/routing/hard_ce_comparison/cross_eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    print("ERROR: osmnx not installed")
    sys.exit(1)

AIDS = ["Walking cane", "Walker", "Mobility scooter", "Manual wheelchair", "Motorized wheelchair"]
BARRIER_COST = 8.0
ORIGIN_NODE  = 1454917652
DEST_NODE    = 12187161148


def _edge_attr(G, u, v, key, default=None):
    for attr_dict in G[u][v].values():
        return attr_dict.get(key, default)
    return default


def load_and_assign_scores(G: nx.Graph, edge_scores_path: str, suffix: str) -> None:
    """Load edge scores and write them as graph attributes with a distinguishing suffix."""
    with open(edge_scores_path) as f:
        raw = json.load(f)
    edges = raw.get("edges", raw)

    for u, v, k in G.edges(keys=True):
        score = edges.get(f"{u}_{v}_{k}") or edges.get(f"{v}_{u}_{k}")
        for aid in AIDS:
            aid_key = aid.lower().replace(" ", "_")
            p = score.get(aid_key, 0.5) if score else 0.5
            G[u][v][k][f"p_yes_{aid_key}_{suffix}"] = float(p)


def routing_weight(aid_key: str, score_suffix: str, barrier_cost: float):
    def weight(u, v, data):
        if not data:
            return float("inf")
        attr = next(iter(data.values()), {}) if isinstance(next(iter(data.values()), {}), dict) else data
        length = attr.get("length", 1.0)
        p_yes  = attr.get(f"p_yes_{aid_key}_{score_suffix}", 0.5)
        return length * (1.0 + barrier_cost * (1.0 - p_yes))
    return weight


def route_stats(route: list, G: nx.Graph, score_suffix: str, aid_key: str):
    pairs = [(u, v) for u, v in zip(route[:-1], route[1:]) if G.has_edge(u, v)]
    length = sum(_edge_attr(G, u, v, "length", 1.0) for u, v in pairs)
    p_yes  = float(np.mean([_edge_attr(G, u, v, f"p_yes_{aid_key}_{score_suffix}", 0.5)
                             for u, v in pairs]))
    return float(length), p_yes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_scores_skl", default="results/routing/edge_scores_full.json")
    parser.add_argument("--edge_scores_hce", default="results/routing/edge_scores_hard_ce.json")
    parser.add_argument("--graph_cache",     default="results/routing/pittsburgh_graph.graphml")
    parser.add_argument("--barrier_cost",    type=float, default=BARRIER_COST)
    parser.add_argument("--output", default="results/routing/hard_ce_comparison/cross_eval.json")
    args = parser.parse_args()

    print("Loading graph …")
    G = ox.load_graphml(args.graph_cache)
    G = G.to_undirected()

    print("Assigning Soft-KL scores …")
    load_and_assign_scores(G, args.edge_scores_skl, "skl")
    print("Assigning Hard-CE scores …")
    load_and_assign_scores(G, args.edge_scores_hce, "hce")

    origin, dest = ORIGIN_NODE, DEST_NODE
    print(f"\nOrigin: {origin}  |  Dest: {dest}\n")

    results = []
    for aid in AIDS:
        aid_key = aid.lower().replace(" ", "_")

        # Standard route (distance-only, identical regardless of scoring)
        route_std = nx.shortest_path(G, origin, dest, weight="length")

        # Soft-KL optimal route
        w_skl = routing_weight(aid_key, "skl", args.barrier_cost)
        route_skl = nx.shortest_path(G, origin, dest, weight=w_skl)

        # Hard-CE optimal route
        w_hce = routing_weight(aid_key, "hce", args.barrier_cost)
        route_hce = nx.shortest_path(G, origin, dest, weight=w_hce)

        # Evaluate every route under BOTH scoring systems
        len_std,  _         = route_stats(route_std, G, "skl", aid_key)
        len_skl,  p_skl_skl = route_stats(route_skl, G, "skl", aid_key)  # ground truth
        len_hce,  p_hce_skl = route_stats(route_hce, G, "skl", aid_key)  # Hard-CE route, calibrated view

        _,        p_std_skl = route_stats(route_std, G, "skl", aid_key)
        _,        p_skl_hce = route_stats(route_skl, G, "hce", aid_key)  # Soft-KL route by Hard-CE (inflated)
        _,        p_hce_hce = route_stats(route_hce, G, "hce", aid_key)  # Hard-CE route self-reported

        same_route = (route_skl == route_hce)
        calibrated_gap = p_skl_skl - p_hce_skl  # positive → Soft-KL route is better

        result = {
            "aid":              aid,
            "std_length_m":     round(len_std, 1),
            "skl_length_m":     round(len_skl, 1),
            "hce_length_m":     round(len_hce, 1),
            "skl_delta_dist_pct": round(100 * (len_skl - len_std) / len_std, 1),
            "hce_delta_dist_pct": round(100 * (len_hce - len_std) / len_std, 1),
            # Calibrated (Soft-KL) evaluation of each route — ground truth
            "p_yes_std_by_skl": round(p_std_skl, 4),
            "p_yes_skl_by_skl": round(p_skl_skl, 4),
            "p_yes_hce_by_skl": round(p_hce_skl, 4),
            # Hard-CE self-reported (inflated) evaluation
            "p_yes_skl_by_hce": round(p_skl_hce, 4),
            "p_yes_hce_by_hce": round(p_hce_hce, 4),
            "calibrated_gap":   round(calibrated_gap, 4),
            "same_route":       same_route,
        }
        results.append(result)

        print(f"[{aid}]")
        print(f"  Lengths:  std={len_std:.0f}m  SKL={len_skl:.0f}m (+{len_skl-len_std:.0f}m)  HCE={len_hce:.0f}m (+{len_hce-len_std:.0f}m)")
        print(f"  Calibrated (SKL) p_yes:  std={p_std_skl:.4f}  SKL_route={p_skl_skl:.4f}  HCE_route={p_hce_skl:.4f}  gap={calibrated_gap:+.4f}")
        print(f"  Hard-CE self-report:      SKL_route={p_skl_hce:.4f}  HCE_route={p_hce_hce:.4f}")
        print(f"  Same route? {same_route}")
        print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {args.output}")

    print("\n── Cross-evaluation Summary ──────────────────────────────────────────")
    print(f"{'Aid':<22} {'Std':>6} {'SKL':>6} {'HCE':>6} {'Gap':>8}  {'Same?'}")
    print(f"{'':22} {'p_yes':>6} {'p_yes':>6} {'p_yes':>6} {'SKL-HCE':>8}  {'(route)'}")
    print("-" * 68)
    for r in results:
        print(f"{r['aid']:<22} {r['p_yes_std_by_skl']:>6.4f} {r['p_yes_skl_by_skl']:>6.4f} "
              f"{r['p_yes_hce_by_skl']:>6.4f} {r['calibrated_gap']:>+8.4f}  "
              f"{'yes' if r['same_route'] else 'NO – differ'}")


if __name__ == "__main__":
    main()
