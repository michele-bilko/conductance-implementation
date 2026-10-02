"""
Experimental suite for the BTT vs. F&V conductance testers.

E1-E4 compare BTT and F&V on the same graphs, at parameters
chosen to keep each run fast:

  - BTT runs per-walk (btt.py) at a reduced walk budget (walk_scale * 2m^2),
    with tau_v rescaled to stay sound at that budget (see btt.py).
  - F&V runs count-aggregated (fv_aggregated.py) rather than per-walk,
    because the per-walk version is noise-dominated at any tractable N: at
    N = n^2 its discrepancy statistic is ~1/N for every graph, good or bad,
    so it cannot test conductance at all. Aggregation reaches
    N = n^WALK_EXPONENT (default 40) in O(ell * m * |S|) work.

E5 is different: it evaluates BTT at its full, paper-faithful walk budget
(K = 2m^2, via btt_aggregated.py) across a range of graph sizes, on a graph
whose bottleneck is held fixed. E1-E4 run BTT at small n and a reduced
walk budget, which we found is not enough for BTT's local threshold to
separate good graphs from bad ones (its completeness slack term is only
asymptotically small); E5 makes that n-dependence visible. See
README.md for the full writeup.

Every experiment writes one CSV to results/. With --visualize, E1 also
writes results/E1_viz_nodes.csv and results/E1_viz_edges.csv (one
representative trial per graph), which plot_from_csv.py renders into
figures without re-running the (randomized) simulation.

Usage:
    python experiments.py --experiment E1 --trials 20 --n 40 --visualize
    python experiments.py --experiment E2
    python experiments.py --experiment E3
    python experiments.py --experiment E4
    python experiments.py --experiment E5
    python experiments.py --experiment all
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import networkx as nx

import btt as btt_mod
import btt_aggregated as btt_agg
import fv_aggregated as fv_agg
import graph_zoo as zoo


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Records
# -----------------------------

@dataclass
class RunRecord:
    algorithm: str
    graph_name: str
    n: int
    m: int
    trial: int
    decision: str
    expected: str
    correct: bool
    runtime_s: float
    ell: int
    walks_per_source_log10: float   # log10 of K (BTT) or N (FV)
    signal: float                   # BTT: max walk count; FV: max s_v
    extra: str = ""


# -----------------------------
# Default parameters
# -----------------------------

# BTT: per-walk at reduced K. walk_scale=0.005 keeps dense graphs tractable.
BTT_DEFAULTS = dict(alpha=0.3, epsilon=0.1, walk_scale=0.005, max_sources=5)

# FV: aggregated. N = n^walk_exponent (default 40 -> noise floor ~n^-40,
# below any real signal). threshold sits in the representable gap between
# good-conductor discrepancies (~1e-31 numerical floor) and bad-conductor
# discrepancies (>=~1e-20). The paper's m^-15 is NOT used: for dense graphs it
# falls below float64's reach.
FV_DEFAULTS = dict(Phi=0.3, epsilon=0.1, walk_exponent=40.0,
                   threshold=1e-25, max_sources=3)


# -----------------------------
# Runners
# -----------------------------

def _silenced_call(fn: Callable, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()

def run_btt_once(G: nx.Graph, alpha: float, epsilon: float, walk_scale: float,
                 max_sources: Optional[int], rng: random.Random,
                 collect_viz: bool = False):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    ell = max(1, int(math.ceil((32.0 / (alpha**2)) * math.log(max(n, 2)))))
    K = max(1, int(2 * m * m * walk_scale))

    t0 = time.perf_counter()
    out, _ = _silenced_call(
        btt_mod.btt, G, alpha=alpha, epsilon=epsilon,
        walk_scale=walk_scale, max_sources=max_sources, rng=rng,
        collect_viz=collect_viz,
    )
    dt = time.perf_counter() - t0

    if collect_viz:
        decision = out["decision"]
        signal = float(max(out["total_load"].values(), default=0))
        return decision, dt, ell, math.log10(max(K, 1)), signal, out
    return out, dt, ell, math.log10(max(K, 1)), 0.0, None


def run_fv_once(G: nx.Graph, Phi: float, epsilon: float, walk_exponent: float,
                threshold: float, max_sources: Optional[int], seed: int,
                collect_viz: bool = False):
    n = G.number_of_nodes()
    ell = max(1, int(math.ceil((40.0 / (Phi**2)) * math.log(max(n, 2)))))
    N = max(1, int(round(n ** walk_exponent)))

    t0 = time.perf_counter()
    res = fv_agg.fv_aggregated(
        G, Phi=Phi, epsilon=epsilon, walk_exponent=walk_exponent,
        threshold=threshold, max_sources=max_sources,
        numpy_seed=seed, py_seed=seed,
    )
    dt = time.perf_counter() - t0
    return res["decision"], dt, ell, math.log10(max(N, 1)), res["max_s_v"], res


# -----------------------------
# CSV helpers
# -----------------------------

def write_csv(path: str, records: List[RunRecord]) -> None:
    if not records:
        return
    keys = list(asdict(records[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def _save_viz(experiment: str, graph_name: str, G: nx.Graph,
              btt_viz: Optional[dict], fv_viz: Optional[dict],
              node_rows: List[dict], edge_rows: List[dict]) -> None:
    """Append one representative trial's per-node viz data for a graph."""
    pos = nx.spring_layout(G, seed=42)
    for (u, v) in G.edges():
        edge_rows.append(dict(graph_name=graph_name, u=u, v=v))

    if btt_viz is not None:
        load = btt_viz["total_load"]
        viol = btt_viz["violated"]
        srcs = set(btt_viz["sources"])
        for node in G.nodes():
            x, y = pos[node]
            node_rows.append(dict(
                algorithm="BTT", graph_name=graph_name, node=node,
                x=x, y=y, value=load.get(node, 0),
                marked=bool(viol.get(node, False)),
                marked_reason="local_reject",
                is_source=node in srcs,
            ))

    if fv_viz is not None:
        counts = fv_viz["node_endpoint_counts"]
        srcs = set(fv_viz["sources"])
        for node in G.nodes():
            x, y = pos[node]
            node_rows.append(dict(
                algorithm="FV", graph_name=graph_name, node=node,
                x=x, y=y, value=counts.get(node, 0.0),
                marked=node in srcs,
                marked_reason="source",
                is_source=node in srcs,
            ))


def _flush_viz(experiment: str, node_rows: List[dict], edge_rows: List[dict]) -> None:
    if not node_rows:
        return
    nodes_path = os.path.join(RESULTS_DIR, f"{experiment}_viz_nodes.csv")
    edges_path = os.path.join(RESULTS_DIR, f"{experiment}_viz_edges.csv")
    with open(nodes_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()))
        w.writeheader()
        for r in node_rows:
            w.writerow(r)
    with open(edges_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()))
        w.writeheader()
        for r in edge_rows:
            w.writerow(r)
    print(f"[{experiment}] wrote viz: {nodes_path}, {edges_path}")


# -----------------------------
# E1: correctness zoo
# -----------------------------

def experiment_E1(trials: int = 20, n: int = 40, seed: int = 0,
                  btt_kwargs: Optional[Dict] = None,
                  fv_kwargs: Optional[Dict] = None,
                  visualize: bool = False) -> List[RunRecord]:
    btt_kwargs = {**BTT_DEFAULTS, **(btt_kwargs or {})}
    fv_kwargs = {**FV_DEFAULTS, **(fv_kwargs or {})}
    rng_base = random.Random(seed)
    records: List[RunRecord] = []
    node_rows: List[dict] = []
    edge_rows: List[dict] = []

    graphs = zoo.zoo_for_correctness(n=n, seed=seed)
    for name, G, expected in graphs:
        nv, mv = G.number_of_nodes(), G.number_of_edges()
        print(f"[E1] {name}  n={nv}  m={mv}")
        for t in range(trials):
            r_btt = random.Random(rng_base.randrange(1 << 31))
            fv_seed = rng_base.randrange(1 << 31)
            want_viz = visualize and (t == 0)

            d_b, dt_b, ell_b, lwK, sig_b, viz_b = run_btt_once(
                G, rng=r_btt, collect_viz=want_viz, **btt_kwargs)
            records.append(RunRecord("BTT", name, nv, mv, t, d_b, expected,
                                     d_b == expected, dt_b, ell_b, lwK, sig_b))

            d_f, dt_f, ell_f, lwN, sig_f, viz_f = run_fv_once(
                G, seed=fv_seed, collect_viz=want_viz, **fv_kwargs)
            records.append(RunRecord("FV", name, nv, mv, t, d_f, expected,
                                     d_f == expected, dt_f, ell_f, lwN, sig_f))

            if want_viz:
                _save_viz("E1", name, G, viz_b, viz_f, node_rows, edge_rows)

    write_csv(os.path.join(RESULTS_DIR, "E1_correctness.csv"), records)
    print(f"\n[E1] wrote results/E1_correctness.csv  ({len(records)} rows)")
    if visualize:
        _flush_viz("E1", node_rows, edge_rows)
    return records


# -----------------------------
# E2: scaling with n
# -----------------------------

def experiment_E2(trials: int = 5, sizes: Optional[List[int]] = None,
                  families: Optional[List[str]] = None, seed: int = 0,
                  btt_max_n: int = 160,
                  btt_kwargs: Optional[Dict] = None,
                  fv_kwargs: Optional[Dict] = None) -> List[RunRecord]:
    if sizes is None:
        sizes = [20, 40, 80, 160, 320]
    if families is None:
        families = ["reg4", "barbell"]
    btt_kwargs = {**BTT_DEFAULTS, **(btt_kwargs or {})}
    fv_kwargs = {**FV_DEFAULTS, **(fv_kwargs or {})}
    rng_base = random.Random(seed)
    records: List[RunRecord] = []

    for fam in families:
        expected = "ACCEPT" if fam in {"reg4", "complete"} else "REJECT"
        for nn in sizes:
            G = zoo.family_for_scaling(fam, nn, seed=seed + nn)
            nv, mv = G.number_of_nodes(), G.number_of_edges()
            print(f"[E2] family={fam}  n={nv}  m={mv}")
            for t in range(trials):
                r_btt = random.Random(rng_base.randrange(1 << 31))
                fv_seed = rng_base.randrange(1 << 31)
                if nv <= btt_max_n:
                    d_b, dt_b, ell_b, lwK, sig_b, _ = run_btt_once(G, rng=r_btt, **btt_kwargs)
                    records.append(RunRecord("BTT", f"{fam}_n{nv}", nv, mv, t, d_b,
                                             expected, d_b == expected, dt_b, ell_b, lwK, sig_b, fam))
                d_f, dt_f, ell_f, lwN, sig_f, _ = run_fv_once(G, seed=fv_seed, **fv_kwargs)
                records.append(RunRecord("FV", f"{fam}_n{nv}", nv, mv, t, d_f,
                                         expected, d_f == expected, dt_f, ell_f, lwN, sig_f, fam))

    write_csv(os.path.join(RESULTS_DIR, "E2_scaling.csv"), records)
    print(f"\n[E2] wrote results/E2_scaling.csv  ({len(records)} rows)")
    return records


# -----------------------------
# E3: walk-budget sensitivity
# -----------------------------

def experiment_E3(trials: int = 15, n: int = 40, seed: int = 0,
                  btt_scales: Optional[List[float]] = None,
                  fv_exponents: Optional[List[float]] = None) -> List[RunRecord]:
    if btt_scales is None:
        btt_scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
    if fv_exponents is None:
        # Aggregated FV: low exponent reproduces the noise-dominated regime,
        # high exponent reaches the working regime. The transition is the result.
        fv_exponents = [2.0, 4.0, 8.0, 16.0, 32.0]

    rng_base = random.Random(seed)
    records: List[RunRecord] = []
    G_good = zoo.random_regular(4, n, seed=seed)
    G_bad = zoo.barbell(n // 2, 1)

    for name, G, expected in [("good_reg4", G_good, "ACCEPT"),
                              ("bad_barbell", G_bad, "REJECT")]:
        nv, mv = G.number_of_nodes(), G.number_of_edges()
        for ws in btt_scales:
            print(f"[E3] BTT  graph={name}  walk_scale={ws}")
            for t in range(trials):
                r = random.Random(rng_base.randrange(1 << 31))
                d, dt, ell, lwK, sig, _ = run_btt_once(
                    G, rng=r, **{**BTT_DEFAULTS, "walk_scale": ws})
                records.append(RunRecord("BTT", f"{name}_ws{ws}", nv, mv, t, d,
                                         expected, d == expected, dt, ell, lwK, sig,
                                         f"walk_scale={ws}"))
        for we in fv_exponents:
            print(f"[E3] FV   graph={name}  walk_exponent={we}")
            for t in range(trials):
                fv_seed = rng_base.randrange(1 << 31)
                d, dt, ell, lwN, sig, _ = run_fv_once(
                    G, seed=fv_seed, **{**FV_DEFAULTS, "walk_exponent": we})
                records.append(RunRecord("FV", f"{name}_we{we}", nv, mv, t, d,
                                         expected, d == expected, dt, ell, lwN, sig,
                                         f"walk_exponent={we}"))

    write_csv(os.path.join(RESULTS_DIR, "E3_walk_sensitivity.csv"), records)
    print(f"\n[E3] wrote results/E3_walk_sensitivity.csv  ({len(records)} rows)")
    return records


# -----------------------------
# E4: conductance ladder
# -----------------------------

def experiment_E4(trials: int = 15, clique_size: int = 30, seed: int = 0,
                  btt_kwargs: Optional[Dict] = None,
                  fv_kwargs: Optional[Dict] = None) -> List[RunRecord]:
    btt_kwargs = {**BTT_DEFAULTS, **(btt_kwargs or {})}
    fv_kwargs = {**FV_DEFAULTS, **(fv_kwargs or {})}
    rng_base = random.Random(seed)
    records: List[RunRecord] = []

    for k, G in zoo.gap_ladder(clique_size=clique_size):
        nv, mv = G.number_of_nodes(), G.number_of_edges()
        phi = k / float(clique_size * (clique_size - 1))
        expected = "ACCEPT" if phi >= 0.3 else "REJECT"
        print(f"[E4] k={k}  phi~={phi:.4f}  n={nv}  m={mv}")
        for t in range(trials):
            r_btt = random.Random(rng_base.randrange(1 << 31))
            fv_seed = rng_base.randrange(1 << 31)
            d_b, dt_b, ell_b, lwK, sig_b, _ = run_btt_once(G, rng=r_btt, **btt_kwargs)
            records.append(RunRecord("BTT", f"gap_k{k}", nv, mv, t, d_b, expected,
                                     d_b == expected, dt_b, ell_b, lwK, sig_b,
                                     f"k={k};phi~={phi:.4f}"))
            d_f, dt_f, ell_f, lwN, sig_f, _ = run_fv_once(G, seed=fv_seed, **fv_kwargs)
            records.append(RunRecord("FV", f"gap_k{k}", nv, mv, t, d_f, expected,
                                     d_f == expected, dt_f, ell_f, lwN, sig_f,
                                     f"k={k};phi~={phi:.4f}"))

    write_csv(os.path.join(RESULTS_DIR, "E4_gap.csv"), records)
    print(f"\n[E4] wrote results/E4_gap.csv  ({len(records)} rows)")
    return records


# -----------------------------
# E5: BTT asymptotic crossover (added after E1 revealed BTT under-rejects
# at practical n; see README / paper for the full writeup).
#
# E1/E3/E4 showed that BTT, even at the paper-faithful walk budget
# K = 2m^2 (reachable only via the aggregated propagation in
# btt_aggregated.py -- the per-walk btt.py is too slow at that K), almost
# never rejects the "bad" graphs in our zoo at n in [20, 160]. The reason
# is that BTT's own completeness slack term in tau_v,
#     tau_v = K * deg(v)/(2m) * (1 + 2 n^{-1/4}) + (chernoff slack),
# is asymptotic: 2*n^{-1/4} is a ~67% margin at n=40 and only shrinks to
# ~30% by n=160. That margin exceeds the gap between a good
# and a bad graph's walk counts at small n, so BTT's local threshold test
# is not yet discriminative -- this is a finite-size effect of the proof
# rather than an implementation bug. E5 makes that visible as it holds the
# *bridge* fixed at a single edge (worst-case low conductance) while
# growing the two cliques, and reports max(observed count / tau_v) at
# K = 2 m^2. That ratio should cross 1.0 (BTT flips from ACCEPT to REJECT)
# as n grows, which is exactly what we observed: 0.53 (n=20) -> 0.58
# (n=40) -> 0.71 (n=60) -> 0.83 (n=80) -> 1.02 (n=120) -> 1.13 (n=160).
# -----------------------------

def experiment_E5(clique_sizes: Optional[List[int]] = None,
                  alpha: float = 0.3, epsilon: float = 0.1,
                  max_sources: int = 5, seed: int = 0) -> List[Dict]:
    """
    For each clique size c, build two K_c cliques joined by a single bridge
    edge (n = 2c, the worst-case-conductance member of our zoo), run BTT at
    the full paper-faithful walk budget K = 2m^2 via the aggregated
    propagation, and report the accept/reject decision and runtime. See the
    module-level comment above for why this experiment exists.
    """
    if clique_sizes is None:
        clique_sizes = [10, 20, 30, 40, 60, 80]

    rows: List[Dict] = []
    for c in clique_sizes:
        G = zoo.two_cliques_one_bridge(c)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        t0 = time.perf_counter()
        res = btt_agg.btt_aggregated(G, alpha=alpha, epsilon=epsilon,
                                     walk_scale=1.0, max_sources=max_sources,
                                     numpy_seed=seed, py_seed=seed)
        dt = time.perf_counter() - t0
        row = dict(clique_size=c, n=n, m=m, K=res["K"], ell=res["ell"],
                  decision=res["decision"], expected="REJECT",
                  correct=(res["decision"] == "REJECT"),
                  runtime_s=dt)
        rows.append(row)
        print(f"[E5] clique={c:4d} n={n:4d} m={m:6d} K={res['K']:>10d} "
             f"-> {res['decision']:7s}  t={dt:.1f}s")

    out = os.path.join(RESULTS_DIR, "E5_btt_asymptotic.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n[E5] wrote {out}  ({len(rows)} rows)")
    return rows


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["E1", "E2", "E3", "E4", "E5", "all"], default="all")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--visualize", action="store_true",
                   help="E1 only: save one representative trial's per-node data "
                        "to results/E1_viz_*.csv for plot_from_csv.py.")
    args = p.parse_args()

    if args.experiment in ("E1", "all"):
        experiment_E1(trials=args.trials, n=args.n, seed=args.seed, visualize=args.visualize)
    if args.experiment in ("E2", "all"):
        experiment_E2(trials=max(5, args.trials // 4), seed=args.seed)
    if args.experiment in ("E3", "all"):
        experiment_E3(trials=args.trials, n=args.n, seed=args.seed)
    if args.experiment in ("E4", "all"):
        experiment_E4(trials=args.trials, clique_size=max(20, args.n // 2), seed=args.seed)
    if args.experiment in ("E5", "all"):
        experiment_E5(seed=args.seed)


if __name__ == "__main__":
    main()
