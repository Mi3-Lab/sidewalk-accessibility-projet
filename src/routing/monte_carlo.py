#!/usr/bin/env python3
"""
Monte Carlo routing evaluation over the Pittsburgh pedestrian graph.

Samples N random OD pairs and compares:
  - Standard Dijkstra (distance only)
  - Accessibility-weighted Dijkstra (our method, bc=8)

Produces:
  results/routing/monte_carlo/summary.json
  results/routing/monte_carlo/monte_carlo_boxplot.pdf/.png
  results/routing/monte_carlo/monte_carlo_pareto.pdf/.png
  results/routing/monte_carlo/monte_carlo_heatmap.pdf/.png

Usage:
    python src/routing/monte_carlo.py \
        --graph_cache results/routing/pittsburgh_graph.graphml \
        --edge_scores results/routing/edge_scores_full.json \
        --n_pairs 10000 \
        --barrier_cost 8 \
        --min_dist 200 \
        --output_dir results/routing/monte_carlo
"""

import argparse
import json
import random
import sys
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
    print("ERROR: osmnx/networkx not installed.")
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


def load_and_score_graph(graph_path: str, scores_path: str) -> nx.Graph:
    G = ox.load_graphml(graph_path)
    G = G.to_undirected()
    with open(scores_path) as f:
        raw = json.load(f)
    edge_scores = raw.get("edges", raw)

    matched = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        score = edge_scores.get(f"{u}_{v}_{k}") or edge_scores.get(f"{v}_{u}_{k}")
        if score:
            for aid in AIDS:
                data[aid] = score.get(aid, PRIOR[aid])
            matched += 1
        else:
            for aid in AIDS:
                data[aid] = PRIOR[aid]
    print(f"  Edges matched to scores: {matched}/{G.number_of_edges()} "
          f"({100*matched/G.number_of_edges():.1f}%)")
    return G


def route_length_and_pyes(G: nx.Graph, path: list, aid: str) -> tuple[float, float]:
    total_len = 0.0
    total_pyes = 0.0
    n = 0
    for a, b in zip(path[:-1], path[1:]):
        edge_data = G[a][b]
        # MultiGraph: take first key
        key = next(iter(edge_data))
        d = edge_data[key]
        total_len += d.get("length", 1.0)
        total_pyes += d.get(aid, PRIOR[aid])
        n += 1
    mean_pyes = total_pyes / n if n > 0 else PRIOR[aid]
    return total_len, mean_pyes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_cache",  default="results/routing/pittsburgh_graph.graphml")
    parser.add_argument("--edge_scores",  default="results/routing/edge_scores_full.json")
    parser.add_argument("--n_pairs",      type=int,   default=10000)
    parser.add_argument("--barrier_cost", type=float, default=8.0)
    parser.add_argument("--min_dist",     type=int,   default=200,
                        help="Minimum straight-line distance (m) between OD nodes.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output_dir",   default="results/routing/monte_carlo")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading graph and edge scores …")
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

    def acc_weight(aid):
        bc = args.barrier_cost
        def w(u, v, data):
            # MultiGraph: data is dict of {key: attr_dict}
            d = next(iter(data.values()))
            p = d.get(aid, PRIOR[aid])
            return d.get("length", 1.0) * (1.0 + bc * (1.0 - p))
        return w

    # Sample OD pairs with min distance constraint
    print(f"Sampling {args.n_pairs} OD pairs (min_dist={args.min_dist}m) …")
    od_pairs = []
    attempts = 0
    while len(od_pairs) < args.n_pairs and attempts < args.n_pairs * 20:
        o, d = random.sample(nodes, 2)
        if haversine(o, d) >= args.min_dist:
            od_pairs.append((o, d))
        attempts += 1

    print(f"  Sampled {len(od_pairs)} pairs (after {attempts} attempts)")

    # Run routing for all pairs × all aids
    results = {aid: {"delta_pyes": [], "delta_dist": [], "dist_std": [],
                     "pyes_std": [], "pyes_acc": [], "improved": 0, "total": 0}
               for aid in AIDS}

    print(f"Running routing ({len(od_pairs)} pairs × {len(AIDS)} aids) …")
    failed = 0
    for i, (o, d) in enumerate(od_pairs):
        if i % 1000 == 0:
            print(f"  {i}/{len(od_pairs)} …")
        for aid in AIDS:
            try:
                path_std = nx.shortest_path(G, o, d, weight="length")
                path_acc = nx.shortest_path(G, o, d, weight=acc_weight(aid))
                len_std,  pyes_std  = route_length_and_pyes(G, path_std, aid)
                len_acc,  pyes_acc  = route_length_and_pyes(G, path_acc, aid)
                delta_p = pyes_acc - pyes_std
                delta_d = (len_acc - len_std) / max(len_std, 1.0)
                results[aid]["delta_pyes"].append(delta_p)
                results[aid]["delta_dist"].append(delta_d)
                results[aid]["dist_std"].append(len_std)
                results[aid]["pyes_std"].append(pyes_std)
                results[aid]["pyes_acc"].append(pyes_acc)
                results[aid]["total"] += 1
                if delta_p > 0.001:
                    results[aid]["improved"] += 1
            except nx.NetworkXNoPath:
                failed += 1

    print(f"  Done. No-path failures: {failed}")

    # Compute summary statistics
    summary = {"n_pairs": len(od_pairs), "barrier_cost": args.barrier_cost,
               "seed": args.seed, "min_dist_m": args.min_dist, "per_aid": {}}
    print("\n=== Monte Carlo Summary ===")
    print(f"{'Aid':<22} {'Mean Δp_yes':>11} {'Median Δp_yes':>13} "
          f"{'% improved':>11} {'Mean Δdist%':>11}")
    print("-" * 72)
    for aid in AIDS:
        r = results[aid]
        dp = np.array(r["delta_pyes"])
        dd = np.array(r["delta_dist"]) * 100  # percent
        pct = 100 * r["improved"] / r["total"] if r["total"] > 0 else 0
        summary["per_aid"][aid] = {
            "n_routed": r["total"],
            "pct_improved": round(pct, 1),
            "mean_delta_pyes": round(float(np.mean(dp)), 4),
            "median_delta_pyes": round(float(np.median(dp)), 4),
            "p25_delta_pyes": round(float(np.percentile(dp, 25)), 4),
            "p75_delta_pyes": round(float(np.percentile(dp, 75)), 4),
            "mean_delta_dist_pct": round(float(np.mean(dd)), 2),
            "median_delta_dist_pct": round(float(np.median(dd)), 2),
            "mean_pyes_std": round(float(np.mean(r["pyes_std"])), 4),
            "mean_pyes_acc": round(float(np.mean(r["pyes_acc"])), 4),
        }
        print(f"{AID_LABELS[aid]:<22} {np.mean(dp):>+11.4f} {np.median(dp):>+13.4f} "
              f"{pct:>10.1f}% {np.mean(dd):>+10.1f}%")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Figure 1: Boxplot of Δp_yes per aid ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = list(range(len(AIDS)))
    boxes = []
    for i, aid in enumerate(AIDS):
        dp = results[aid]["delta_pyes"]
        bp = ax.boxplot(dp, positions=[i], widths=0.5, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                        boxprops={"facecolor": AID_COLORS[aid], "alpha": 0.75})
        boxes.append(mpatches.Patch(color=AID_COLORS[aid],
                                    label=f"{AID_LABELS[aid]} ({summary['per_aid'][aid]['pct_improved']:.0f}% improved)"))

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([AID_LABELS[a] for a in AIDS], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Δ Mean p_yes (accessible − standard)", fontsize=11)
    ax.set_title(f"Monte Carlo routing evaluation — {len(od_pairs):,} random OD pairs, Pittsburgh PA\n"
                 f"Accessibility-weighted Dijkstra (bc={args.barrier_cost}) vs. standard Dijkstra",
                 fontsize=10)
    ax.legend(handles=boxes, loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out / "monte_carlo_boxplot.pdf", bbox_inches="tight")
    fig.savefig(out / "monte_carlo_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Pareto scatter (mean Δp_yes vs mean Δdist%) per aid ─────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for aid in AIDS:
        s = summary["per_aid"][aid]
        ax.scatter(s["mean_delta_dist_pct"], s["mean_delta_pyes"],
                   color=AID_COLORS[aid], s=120, zorder=5,
                   label=AID_LABELS[aid], edgecolors="white", linewidths=1)
        ax.annotate(AID_LABELS[aid].split()[0],
                    (s["mean_delta_dist_pct"], s["mean_delta_pyes"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Mean Δ distance (%)", fontsize=11)
    ax.set_ylabel("Mean Δ p_yes (accessibility gain)", fontsize=11)
    ax.set_title(f"Accessibility gain vs. distance overhead\n"
                 f"{len(od_pairs):,} OD pairs, Pittsburgh PA (bc={args.barrier_cost})",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.xaxis.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out / "monte_carlo_pareto.pdf", bbox_inches="tight")
    fig.savefig(out / "monte_carlo_pareto.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: % pairs improved per aid (bar chart) ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    pcts = [summary["per_aid"][aid]["pct_improved"] for aid in AIDS]
    colors = [AID_COLORS[aid] for aid in AIDS]
    bars = ax.bar(range(len(AIDS)), pcts, color=colors, edgecolor="white", width=0.55)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(AIDS)))
    ax.set_xticklabels([AID_LABELS[a] for a in AIDS], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("OD pairs where accessible > standard (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title(f"Route improvement rate across {len(od_pairs):,} random OD pairs", fontsize=10)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out / "monte_carlo_improvement_rate.pdf", bbox_inches="tight")
    fig.savefig(out / "monte_carlo_improvement_rate.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults → {out}/")
    print(f"  summary.json")
    print(f"  monte_carlo_boxplot.png/pdf")
    print(f"  monte_carlo_pareto.png/pdf")
    print(f"  monte_carlo_improvement_rate.png/pdf")


if __name__ == "__main__":
    main()
