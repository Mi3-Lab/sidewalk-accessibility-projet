#!/usr/bin/env python3
"""
Risk-sensitive routing over calibrated per-aid distributions, and the experiment
that tests whether calibration actually matters to the planner.

**The gap this addresses.** The probe predicts [p_no, p_unsure, p_yes] per aid,
but edge scoring stored only p_yes and the planner used
`cost = length * (1 + beta * (1 - p_yes))`. A linear function of one scalar
cannot distinguish a genuinely split scene ([0.45, 0.05, 0.45]) from a
confidently middling one ([0.10, 0.45, 0.45]) — and that bimodality is precisely
what soft-label training exists to preserve. So the paper argues calibration is
the operative metric while feeding the planner an objective that is largely
insensitive to it.

**Objectives.** All four stay Dijkstra-solvable:

  linear      length * (1 + beta * (1 - p_yes))
              The current objective. Expected-value; needs only the mean.

  logbarrier  length + lam * (-log p_yes)
              Chance-constrained relaxation: additive in -log p, so the route
              maximises the product of per-edge passability. Unlike `linear`,
              the penalty diverges as p_yes -> 0, so a single likely-impassable
              edge cannot be averaged away by a long stretch of good ones.

  risk_split  length * (1 + beta_no * p_no + beta_unsure * p_unsure)
              Separates "known impassable" from "unknown". Under `linear` these
              are identical, since 1 - p_yes = p_no + p_unsure. With
              beta_no > beta_unsure the planner avoids known barriers hard and
              treats uncertainty as a milder, verifiable cost — which matches
              the deployment story of onboard verification at the drop-off.

  bottleneck  minimise the worst p_no along the route, ties broken by length.
              CVaR in the tail limit: a route is only as good as its worst edge.

**The experiment.** For each objective, route the same OD pairs twice — once
with Soft-KL edge scores, once with Hard-CE — and measure how much the two route
sets differ (Jaccard over edge sets). The prediction that makes the paper's
thesis testable rather than asserted:

    under `linear` the two should agree closely, because the objective only needs
    the mean roughly right; under the risk objectives they should diverge,
    because tail behaviour depends on the whole distribution, which Hard-CE
    destroys.

The PS-label and prior edges are identical between the two score files, so any
divergence comes purely from the model-inferred edges.

Usage:
    python src/routing/risk_planner.py \
        --graph_cache  results/routing/pittsburgh_graph_956.graphml \
        --edge_scores_soft results/routing/edge_scores_full.json \
        --edge_scores_hard results/routing/edge_scores_hard_ce.json \
        --n_pairs 2000 --output_dir results/risk_routing
"""

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import networkx as nx
    import osmnx as ox
except ImportError:
    print("ERROR: needs networkx + osmnx (use the repo venv)")
    sys.exit(1)

AIDS_KEYS = ["walking_cane", "walker", "mobility_scooter",
             "manual_wheelchair", "motorized_wheelchair"]

OBJECTIVES = ["linear", "logbarrier", "risk_split", "bottleneck"]

# Edges scored from PS labels or population priors carry no model distribution.
# Their p_yes is kept and the remaining mass is split using the dataset-wide
# ratio of "no" to "unsure" votes, so they stay comparable across objectives.
# They are identical in both score files, so they cannot create divergence.
UNSURE_SHARE_OF_REMAINDER = 0.25


def edge_distribution(data: dict, aid: str) -> tuple[float, float, float]:
    """Return (p_no, p_unsure, p_yes) for one edge and aid."""
    dist = data.get("dist") or {}
    if aid in dist:
        p_no, p_unsure, p_yes = dist[aid]
        return float(p_no), float(p_unsure), float(p_yes)
    p_yes = float(data.get(aid, 0.5))
    rem = max(0.0, 1.0 - p_yes)
    p_unsure = rem * UNSURE_SHARE_OF_REMAINDER
    return rem - p_unsure, p_unsure, p_yes


def make_weight(objective: str, aid: str, params: dict):
    """Edge weight function for the additive objectives."""
    eps = 1e-6

    def _w(u, v, data):
        length = float(data.get("length", 1.0))
        p_no, p_unsure, p_yes = edge_distribution(data, aid)

        if objective == "linear":
            return length * (1.0 + params["beta"] * (1.0 - p_yes))
        if objective == "logbarrier":
            return length + params["lam"] * (-np.log(max(p_yes, eps)))
        if objective == "risk_split":
            return length * (1.0 + params["beta_no"] * p_no
                             + params["beta_unsure"] * p_unsure)
        raise ValueError(objective)

    return _w


def bottleneck_path(G, source, target, aid: str):
    """Minimise the worst p_no along the route, breaking ties by length.

    Standard minimax (widest-path) Dijkstra: the cost of a path is the maximum
    edge risk on it, not the sum, so one bad edge cannot be offset by good ones.
    """
    best = {source: (0.0, 0.0)}          # node -> (worst p_no, length)
    prev: dict = {}
    pq = [(0.0, 0.0, source)]

    while pq:
        risk, dist, node = heapq.heappop(pq)
        if node == target:
            break
        if (risk, dist) > best.get(node, (float("inf"), float("inf"))):
            continue
        for nbr in G.neighbors(node):
            data = min(G[node][nbr].values(), key=lambda d: float(d.get("length", 1.0)))
            p_no, _, _ = edge_distribution(data, aid)
            cand = (max(risk, p_no), dist + float(data.get("length", 1.0)))
            if cand < best.get(nbr, (float("inf"), float("inf"))):
                best[nbr] = cand
                prev[nbr] = node
                heapq.heappush(pq, (cand[0], cand[1], nbr))

    if target not in prev and target != source:
        raise nx.NetworkXNoPath(f"no path {source}->{target}")

    path, node = [target], target
    while node != source:
        node = prev[node]
        path.append(node)
    return path[::-1]


def route(G, source, target, objective: str, aid: str, params: dict):
    if objective == "bottleneck":
        return bottleneck_path(G, source, target, aid)
    return nx.shortest_path(G, source, target,
                            weight=make_weight(objective, aid, params))


def edge_set(path) -> set:
    return {frozenset((a, b)) for a, b in zip(path[:-1], path[1:])}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_graph(graph_path: str, scores_path: str):
    G = ox.load_graphml(graph_path).to_undirected()
    with open(scores_path) as f:
        raw = json.load(f)
    scores = raw.get("edges", raw)

    matched = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        s = scores.get(f"{u}_{v}_{k}") or scores.get(f"{v}_{u}_{k}")
        if s:
            for aid in AIDS_KEYS:
                data[aid] = s.get(aid, 0.5)
            if "dist" in s:
                data["dist"] = s["dist"]
            matched += 1
        else:
            for aid in AIDS_KEYS:
                data[aid] = 0.5
    pct = 100 * matched / G.number_of_edges()
    print(f"  {Path(scores_path).name}: {matched}/{G.number_of_edges()} edges matched ({pct:.1f}%)")
    if pct < 50:
        raise SystemExit(f"ERROR: only {pct:.1f}% matched — graph and scores disagree.")
    n_dist = sum(1 for _, _, d in G.edges(data=True) if d.get("dist"))
    print(f"    with full distribution: {n_dist} ({100*n_dist/G.number_of_edges():.1f}%)")
    return G


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph_cache", default="results/routing/pittsburgh_graph_956.graphml")
    p.add_argument("--edge_scores_soft", default="results/routing/edge_scores_full.json")
    p.add_argument("--edge_scores_hard", default="results/routing/edge_scores_hard_ce.json")
    p.add_argument("--output_dir", default="results/risk_routing")
    p.add_argument("--n_pairs",  type=int, default=2000)
    p.add_argument("--min_dist", type=float, default=200)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--beta",        type=float, default=8.0)
    p.add_argument("--lam",         type=float, default=200.0)
    p.add_argument("--beta_no",     type=float, default=12.0)
    p.add_argument("--beta_unsure", type=float, default=3.0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = {"beta": args.beta, "lam": args.lam,
              "beta_no": args.beta_no, "beta_unsure": args.beta_unsure}

    print("Loading graphs …")
    G_soft = load_graph(args.graph_cache, args.edge_scores_soft)
    G_hard = load_graph(args.graph_cache, args.edge_scores_hard)

    rng = np.random.default_rng(args.seed)
    nodes = list(G_soft.nodes())
    coords = {n: (float(G_soft.nodes[n]["y"]), float(G_soft.nodes[n]["x"])) for n in nodes}

    def haversine(n1, n2):
        (la1, lo1), (la2, lo2) = coords[n1], coords[n2]
        p1, p2 = np.radians(la1), np.radians(la2)
        dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * 6_371_000 * np.arcsin(np.sqrt(a))

    pairs = []
    while len(pairs) < args.n_pairs:
        s, t = rng.choice(len(nodes), 2, replace=False)
        s, t = nodes[s], nodes[t]
        if haversine(s, t) >= args.min_dist:
            pairs.append((s, t))
    print(f"\n{len(pairs)} OD pairs (min {args.min_dist} m apart)\n")

    results: dict = {obj: {aid: [] for aid in AIDS_KEYS} for obj in OBJECTIVES}
    failures = 0

    for s, t in pairs:
        for obj in OBJECTIVES:
            for aid in AIDS_KEYS:
                try:
                    r_soft = edge_set(route(G_soft, s, t, obj, aid, params))
                    r_hard = edge_set(route(G_hard, s, t, obj, aid, params))
                except (nx.NetworkXNoPath, nx.NodeNotFound, SystemExit):
                    failures += 1
                    continue
                results[obj][aid].append({
                    "jaccard":   jaccard(r_soft, r_hard),
                    "identical": r_soft == r_hard,
                })

    summary: dict = {"config": vars(args), "n_pairs": len(pairs),
                     "failures": failures, "per_objective": {}}

    print(f"{'objective':<14}{'mean Jaccard':>15}{'identical routes':>20}")
    print("-" * 52)
    for obj in OBJECTIVES:
        js, ident = [], []
        for aid in AIDS_KEYS:
            js += [r["jaccard"] for r in results[obj][aid]]
            ident += [r["identical"] for r in results[obj][aid]]
        summary["per_objective"][obj] = {
            "mean_jaccard": float(np.mean(js)) if js else float("nan"),
            "pct_identical": float(100 * np.mean(ident)) if ident else float("nan"),
            "n": len(js),
            "per_aid": {
                aid: {
                    "mean_jaccard": float(np.mean([r["jaccard"] for r in results[obj][aid]]))
                    if results[obj][aid] else float("nan"),
                    "pct_identical": float(100 * np.mean([r["identical"] for r in results[obj][aid]]))
                    if results[obj][aid] else float("nan"),
                }
                for aid in AIDS_KEYS
            },
        }
        v = summary["per_objective"][obj]
        print(f"{obj:<14}{v['mean_jaccard']:>15.4f}{v['pct_identical']:>19.1f}%")

    with open(out / "risk_routing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lin = summary["per_objective"]["linear"]["mean_jaccard"]
    print("\nSoft-KL vs Hard-CE routes, agreement by objective:")
    print(f"  linear (the paper's objective): {lin:.4f}")
    for obj in OBJECTIVES[1:]:
        v = summary["per_objective"][obj]["mean_jaccard"]
        print(f"  {obj:<28}: {v:.4f}   ({v - lin:+.4f} vs linear)")
    print("\nA *lower* Jaccard under the risk objectives is the predicted result: it")
    print("means the choice of training loss changes the route, i.e. calibration")
    print("reaches the planner. Agreement everywhere would mean it does not.")
    print(f"\nSaved → {out / 'risk_routing_summary.json'}")


if __name__ == "__main__":
    main()
