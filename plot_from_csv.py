"""
Render conductance-tester figures from saved viz CSVs — no re-simulation.

experiments.py --visualize writes, per experiment:
  results/<EXP>_viz_nodes.csv : algorithm, graph_name, node, x, y, value,
                                marked, marked_reason, is_source
  results/<EXP>_viz_edges.csv : graph_name, u, v

This script reads those and draws one figure per (algorithm, graph) into
results/figures/<EXP>/. Each figure colours nodes by `value`:
  - BTT: total walks ending at the node; red ring = node locally rejected.
  - FV:  average fraction of walks ending at the node; black ring = source.

Usage:
    python plot_from_csv.py --experiment E1
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load(experiment: str):
    nodes_path = os.path.join(RESULTS_DIR, f"{experiment}_viz_nodes.csv")
    edges_path = os.path.join(RESULTS_DIR, f"{experiment}_viz_edges.csv")
    if not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
        raise SystemExit(f"missing viz CSVs for {experiment}; run experiments.py "
                         f"--experiment {experiment} --visualize first")
    nodes = list(csv.DictReader(open(nodes_path)))
    edges = list(csv.DictReader(open(edges_path)))
    return nodes, edges


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def plot_experiment(experiment: str) -> None:
    nodes, edges = _load(experiment)
    out_dir = os.path.join(RESULTS_DIR, "figures", experiment)
    os.makedirs(out_dir, exist_ok=True)

    # edges grouped by graph
    edges_by_graph: Dict[str, List[Tuple]] = defaultdict(list)
    for e in edges:
        edges_by_graph[e["graph_name"]].append((e["u"], e["v"]))

    # nodes grouped by (algorithm, graph)
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in nodes:
        groups[(r["algorithm"], r["graph_name"])].append(r)

    for (algo, graph), rows in sorted(groups.items()):
        pos = {r["node"]: (float(r["x"]), float(r["y"])) for r in rows}
        values = {r["node"]: float(r["value"]) for r in rows}
        marked = {r["node"]: _truthy(r["marked"]) for r in rows}
        reason = rows[0]["marked_reason"]

        fig, ax = plt.subplots(figsize=(6, 5))

        # edges
        for (u, v) in edges_by_graph.get(graph, []):
            if u in pos and v in pos:
                x0, y0 = pos[u]; x1, y1 = pos[v]
                ax.plot([x0, x1], [y0, y1], color="0.7", linewidth=0.6, zorder=1)

        # nodes
        order = list(pos.keys())
        xs = [pos[k][0] for k in order]
        ys = [pos[k][1] for k in order]
        cs = [values[k] for k in order]
        vmax = max(cs) if cs else 1.0
        norm = [c / vmax if vmax > 0 else 0.0 for c in cs]
        sc = ax.scatter(xs, ys, c=norm, cmap="viridis", s=300, zorder=2,
                        edgecolors="none")

        # marked rings
        ring_color = "red" if reason == "local_reject" else "black"
        ring_x = [pos[k][0] for k in order if marked[k]]
        ring_y = [pos[k][1] for k in order if marked[k]]
        if ring_x:
            ax.scatter(ring_x, ring_y, facecolors="none", edgecolors=ring_color,
                       linewidths=2.0, s=360, zorder=3)

        # labels (integer value)
        for k in order:
            vx, vy = pos[k]
            label = f"{values[k]:.0f}" if values[k] >= 1 else f"{values[k]:.2g}"
            ax.text(vx, vy, label, fontsize=7, ha="center", va="center", zorder=4)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("walks ending at node" if algo == "BTT"
                       else "avg walk fraction ending at node")
        ring_label = ("red ring = local reject" if reason == "local_reject"
                      else "black ring = source")
        ax.set_title(f"{algo} — {graph}\n({ring_label})")
        ax.axis("off")
        fig.tight_layout()
        out = os.path.join(out_dir, f"{algo.lower()}_{graph}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="E1")
    args = p.parse_args()
    plot_experiment(args.experiment)


if __name__ == "__main__":
    main()
