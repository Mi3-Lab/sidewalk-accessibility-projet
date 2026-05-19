#!/usr/bin/env python3
"""
Plot Pareto curve: accessibility gain vs distance overhead for each barrier_cost value.

Reads routing_results.json from each ablation_bc{N} directory and produces:
  results/routing/pareto_barrier_cost.pdf / .png

Usage:
    python src/routing/plot_barrier_cost_pareto.py \
        --results_dir results/routing \
        --barrier_costs 1 2 4 8 16 \
        --output results/routing/pareto_barrier_cost
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

AIDS = [
    "Walking cane",
    "Walker",
    "Mobility scooter",
    "Manual wheelchair",
    "Motorized wheelchair",
]

AID_COLORS = {
    "Walking cane":         "#e41a1c",
    "Walker":               "#ff7f00",
    "Mobility scooter":     "#4daf4a",
    "Manual wheelchair":    "#377eb8",
    "Motorized wheelchair": "#984ea3",
}

AID_MARKERS = {
    "Walking cane":         "o",
    "Walker":               "s",
    "Mobility scooter":     "^",
    "Manual wheelchair":    "D",
    "Motorized wheelchair": "P",
}


def load_sweep(results_dir: Path, barrier_costs: list[int]) -> dict[str, list[dict]]:
    """Returns {aid: [{bc, delta_dist_pct, delta_access}, ...]}"""
    data: dict[str, list[dict]] = {aid: [] for aid in AIDS}
    for bc in barrier_costs:
        p = results_dir / f"ablation_bc{bc}" / "routing_results.json"
        if not p.exists():
            print(f"WARNING: {p} not found — skipping bc={bc}")
            continue
        with open(p) as f:
            rows = json.load(f)
        for row in rows:
            aid = row["aid"]
            std_l = row["std_length"]
            delta_dist_pct = (row["delta_length"] / std_l * 100) if std_l > 0 else 0.0
            data[aid].append({
                "bc":             bc,
                "delta_dist_pct": delta_dist_pct,
                "delta_access":   row["delta_access"],
            })
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",   default="results/routing")
    parser.add_argument("--barrier_costs", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--selected_bc",   type=int,  default=8,
                        help="barrier_cost used in the main paper — highlighted with a star.")
    parser.add_argument("--output",        default="results/routing/pareto_barrier_cost")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = load_sweep(results_dir, sorted(args.barrier_costs))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for aid in AIDS:
        pts = sorted(data[aid], key=lambda x: x["bc"])
        if not pts:
            continue

        xs = [p["delta_dist_pct"] for p in pts]
        ys = [p["delta_access"]   for p in pts]
        color  = AID_COLORS[aid]
        marker = AID_MARKERS[aid]

        ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.8, zorder=2)
        ax.scatter(xs, ys, color=color, marker=marker, s=60, zorder=3, edgecolors="white", linewidths=0.5)

        # Label bc values at each point (only on first aid to avoid clutter, or all small)
        for p in pts:
            xp = p["delta_dist_pct"]
            yp = p["delta_access"]
            bc = p["bc"]
            # Small label offset: above for most, below for crowded cases
            ax.annotate(
                f"{bc}",
                xy=(xp, yp),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=6,
                color=color,
                alpha=0.8,
            )

        # Star for selected bc
        sel = next((p for p in pts if p["bc"] == args.selected_bc), None)
        if sel:
            ax.scatter(
                sel["delta_dist_pct"], sel["delta_access"],
                color=color, marker="*", s=220, zorder=4,
                edgecolors="white", linewidths=0.5,
            )

    # Legend
    handles = [
        mlines.Line2D([], [], color=AID_COLORS[a], marker=AID_MARKERS[a],
                      markersize=7, linewidth=1.5, label=a)
        for a in AIDS
    ]
    star_handle = mlines.Line2D([], [], color="grey", marker="*", markersize=10,
                                linestyle="None", label=f"bc={args.selected_bc} (paper)")
    ax.legend(handles=handles + [star_handle], fontsize=8, loc="upper left",
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_xlabel("Route length overhead  Δdist (%)", fontsize=10)
    ax.set_ylabel("Accessibility gain  Δp_yes", fontsize=10)
    ax.set_title(
        f"Pareto Tradeoff: Accessibility vs Distance\n"
        f"barrier_cost ∈ {{{', '.join(str(b) for b in sorted(args.barrier_costs))}}}  |  "
        f"Pittsburgh PA, n=2618 edges, real PS labels",
        fontsize=10,
    )
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    out = Path(args.output)
    fig.savefig(str(out) + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pareto figure → {out}.png")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'barrier_cost':>14}", end="")
    for aid in AIDS:
        ak = aid.replace(" ", "_")[:14]
        print(f"  {ak:>14}", end="")
    print()
    print("-" * (14 + 16 * len(AIDS)))

    all_bcs = sorted(args.barrier_costs)
    for bc in all_bcs:
        marker = " ★" if bc == args.selected_bc else "  "
        print(f"{bc:>14}{marker}", end="")
        for aid in AIDS:
            pts = [p for p in data[aid] if p["bc"] == bc]
            if pts:
                p = pts[0]
                print(f"  {p['delta_access']:>+.3f}/{p['delta_dist_pct']:>+.0f}%", end="")
            else:
                print(f"  {'—':>12}", end="")
        print()

    print("\nCells: Δp_yes / Δdist%")

    # ── Save sweep summary JSON ───────────────────────────────────────────────
    summary = {}
    for bc in all_bcs:
        summary[str(bc)] = {}
        for aid in AIDS:
            pts = [p for p in data[aid] if p["bc"] == bc]
            if pts:
                summary[str(bc)][aid] = {
                    "delta_access":   round(pts[0]["delta_access"], 4),
                    "delta_dist_pct": round(pts[0]["delta_dist_pct"], 1),
                }
    out_json = results_dir / "pareto_barrier_cost_summary.json"
    with open(out_json, "w") as f:
        import json as _json
        _json.dump(summary, f, indent=2)
    print(f"Summary JSON → {out_json}")


if __name__ == "__main__":
    main()
