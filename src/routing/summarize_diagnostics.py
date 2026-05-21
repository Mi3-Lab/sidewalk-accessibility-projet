#!/usr/bin/env python3
"""Summarize routing diagnostics for paper tables.

This script is intentionally lightweight: it consumes existing JSON artifacts
and emits compact CSV/JSON summaries that can be copied into the manuscript or
supplement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

try:
    import osmnx as ox
except ImportError:  # pragma: no cover - optional CLI dependency
    ox = None


AIDS = [
    "walking_cane",
    "walker",
    "mobility_scooter",
    "manual_wheelchair",
    "motorized_wheelchair",
]

AID_LABELS = {
    "walking_cane": "Walking cane",
    "walker": "Walker",
    "mobility_scooter": "Mobility scooter",
    "manual_wheelchair": "Manual wheelchair",
    "motorized_wheelchair": "Motorized wheelchair",
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_edge_scores(path: Path, graph_path: Path | None = None) -> dict:
    raw = json.loads(path.read_text())
    all_edges = raw["edges"]
    edges = all_edges

    graph_match = None
    if graph_path is not None:
        if ox is None:
            raise RuntimeError("osmnx is required for --graph")
        graph = ox.load_graphml(graph_path).to_undirected()
        filtered = {}
        missing = 0
        for u, v, k in graph.edges(keys=True):
            key = f"{u}_{v}_{k}"
            rev_key = f"{v}_{u}_{k}"
            if key in all_edges:
                filtered[key] = all_edges[key]
            elif rev_key in all_edges:
                filtered[rev_key] = all_edges[rev_key]
            else:
                missing += 1
        graph_match = {
            "graph_path": str(graph_path),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "matched_edges": len(filtered),
            "missing_edges": missing,
            "matched_pct": round(100.0 * len(filtered) / graph.number_of_edges(), 1),
        }
        edges = filtered
    sources: dict[str, int] = {}
    per_source: dict[str, dict[str, list[float]]] = {}

    for edge in edges.values():
        src = edge.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        per_source.setdefault(src, {aid: [] for aid in AIDS})
        for aid in AIDS:
            per_source[src][aid].append(float(edge[aid]))

    total = len(edges)
    source_rows = [
        {
            "source": src,
            "n_edges": n,
            "pct_edges": round(100.0 * n / total, 1),
        }
        for src, n in sorted(sources.items())
    ]

    aid_rows = []
    for src in sorted(per_source):
        for aid in AIDS:
            vals = per_source[src][aid]
            aid_rows.append(
                {
                    "source": src,
                    "aid": AID_LABELS[aid],
                    "n_edges": len(vals),
                    "mean_p_yes": round(mean(vals), 4),
                    "p10": round(percentile(vals, 0.10), 4),
                    "p50": round(percentile(vals, 0.50), 4),
                    "p90": round(percentile(vals, 0.90), 4),
                    "pct_lt_0_50": round(100.0 * sum(v < 0.50 for v in vals) / len(vals), 1),
                    "pct_ge_0_80": round(100.0 * sum(v >= 0.80 for v in vals) / len(vals), 1),
                }
            )

    all_rows = []
    for aid in AIDS:
        vals = [float(edge[aid]) for edge in edges.values()]
        all_rows.append(
            {
                "aid": AID_LABELS[aid],
                "n_edges": len(vals),
                "mean_p_yes": round(mean(vals), 4),
                "p10": round(percentile(vals, 0.10), 4),
                "p50": round(percentile(vals, 0.50), 4),
                "p90": round(percentile(vals, 0.90), 4),
                "pct_lt_0_50": round(100.0 * sum(v < 0.50 for v in vals) / len(vals), 1),
                "pct_ge_0_80": round(100.0 * sum(v >= 0.80 for v in vals) / len(vals), 1),
            }
        )

    return {
        "meta": raw.get("__meta__", {}),
        "total_edges": total,
        "graph_match": graph_match,
        "sources": source_rows,
        "per_source_aid": aid_rows,
        "all_edges_by_aid": all_rows,
    }


def summarize_beta(paths: list[Path]) -> list[dict]:
    rows = []
    for path in sorted(paths, key=lambda p: float(json.loads(p.read_text())["barrier_cost"])):
        raw = json.loads(path.read_text())
        beta = float(raw["barrier_cost"])
        per_aid = raw["per_aid"]
        rows.append(
            {
                "beta_c": int(beta) if beta.is_integer() else beta,
                "n_pairs": raw["n_pairs"],
                "mean_pct_improved": round(mean(v["pct_improved"] for v in per_aid.values()), 1),
                "mean_delta_p_yes": round(mean(v["mean_delta_pyes"] for v in per_aid.values()), 4),
                "mean_delta_dist_pct": round(mean(v["mean_delta_dist_pct"] for v in per_aid.values()), 2),
                "median_delta_dist_pct": round(mean(v["median_delta_dist_pct"] for v in per_aid.values()), 2),
                "min_pct_improved": round(min(v["pct_improved"] for v in per_aid.values()), 1),
                "max_pct_improved": round(max(v["pct_improved"] for v in per_aid.values()), 1),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_scores", type=Path, default=Path("results/routing/edge_scores_full.json"))
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--beta_summaries", type=Path, nargs="*", default=[])
    parser.add_argument("--output_dir", type=Path, default=Path("results/routing/diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    edge_summary = summarize_edge_scores(args.edge_scores, args.graph)
    (args.output_dir / "edge_score_summary.json").write_text(json.dumps(edge_summary, indent=2))
    write_csv(args.output_dir / "edge_source_counts.csv", edge_summary["sources"])
    write_csv(args.output_dir / "edge_score_by_source_aid.csv", edge_summary["per_source_aid"])
    write_csv(args.output_dir / "edge_score_by_aid.csv", edge_summary["all_edges_by_aid"])

    if args.beta_summaries:
        beta_rows = summarize_beta(args.beta_summaries)
        (args.output_dir / "beta_sensitivity_summary.json").write_text(json.dumps(beta_rows, indent=2))
        write_csv(args.output_dir / "beta_sensitivity_summary.csv", beta_rows)

    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
