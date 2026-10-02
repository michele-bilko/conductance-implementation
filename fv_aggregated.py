"""
Count-aggregated F&V (Fichtenberger–Vasudev 2018) for arbitrarily large N.

Background (from paper)
----------
The per-walk fv.py loops over each of N = n^100 walks per source. A direct simulation of this is not possible.
Through the first experiment (E1) we found that at the
tractable per-walk setting N = n^2, the discrepancy statistic

    s_v = Σ_u ( Ŵ^ℓ_{v,u} − deg(u)/(2m) )^2

is dominated by sampling noise. Its expected value for *any* graph is
≈ (1 − ‖π‖^2) / N ≈ 1/N, because each Ŵ^ℓ_{v,u} is a Binomial(N, ·)/N
estimate with variance ~1/N. At N = n^2 = 1600 the noise floor is ~6e-4,
which overwhelms the ‖W^ℓ(v,·) − π‖^2 signal for every graph in the zoo.
This helped me understand why the paper uses N = n^100 as it pushes the noise floor (n^{-100})
far below the m^{-15} signal scale.

The fix is by using a multinomial trick I also used in btt_aggregated.py. The walk
endpoint distribution after ℓ steps is obtained by propagating a *count
vector* per source (not individual walks) so we can take N as large as
we like (N = n^100 included) in O(ℓ · m · |S|) work. At large N the count
split is dominated by its deterministic mean (Gaussian fallback), so s_v
converges to the *exact* ‖W^ℓ − π‖^2 with negligible sampling noise — i.e.
the paper's regime, where the m^{-15} threshold has meaning.

This module reuses _multinomial_split from btt_aggregated.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Optional

import networkx as nx
import numpy as np

from btt_aggregated import _multinomial_split

Node = Hashable


def _endpoint_distribution_aggregated(
    G: nx.Graph,
    source: Node,
    ell: int,
    N: int,
    rng: np.random.Generator,
) -> Dict[Node, int]:
    """
    Return the count vector c[u] = number of the N length-ℓ lazy walks from
    `source` that end at u, computed by count aggregation (no per-walk loop).
    """
    # c[v] = number of walks currently at v.
    c: Dict[Node, int] = defaultdict(int)
    c[source] = N

    for _ in range(ell):
        nxt: Dict[Node, int] = defaultdict(int)
        for v, count in c.items():
            if count <= 0:
                continue
            neighbors = list(G.neighbors(v))
            d = len(neighbors)
            if d == 0:
                nxt[v] += count
                continue
            probs = [0.5] + [0.5 / d] * d        # lazy walk: stay, else uniform nbr
            targets = [v] + neighbors
            parts = _multinomial_split(count, probs, rng)
            for t, k in zip(targets, parts):
                if k > 0:
                    nxt[t] += int(k)
        c = nxt
    return c


def fv_aggregated(
    G: nx.Graph,
    Phi: float = 0.3,
    epsilon: float = 0.1,
    N: Optional[int] = None,
    walk_exponent: float = 100.0,
    threshold: Optional[float] = None,
    threshold_power: float = 15.0,
    max_sources: Optional[int] = None,
    numpy_seed: int = 0,
    py_seed: int = 0,
    skip_bfs: bool = True,
) -> Dict:
    """
    Aggregated F&V conductance tester. Returns a dict with the decision plus
    diagnostics. Defaults to the paper's N = n^100 and threshold m^-15.

    Parameters
    ----------
    N : if given, used directly. Otherwise N = round(n ** walk_exponent).
        Counts may be arbitrarily large (Python big ints + Gaussian split).
    threshold : if given, used directly. Otherwise m^(-threshold_power).
        For diagnosing the noise regime, pass threshold = kappa / N.
    skip_bfs : the per-walk fv.py builds a BFS tree and a diameter check. The
        aggregation does not need the tree for the walks; we keep an optional
        diameter reject for completeness but default to skipping it, since at
        the parameter settings of interest the diameter check is not the point (quite arbitrary).

    Returns
    -------
    dict with: decision, reason, n, m, ell, N (as str if huge), num_sources,
    max_s_v, threshold, noise_floor_1_over_N, per_source_s.
    """
    rng_np = np.random.default_rng(numpy_seed)
    rng_py = random.Random(py_seed)

    nodes = list(G.nodes())
    n = len(nodes)
    m = G.number_of_edges()
    if n == 0 or m == 0:
        return {"decision": "ACCEPT", "reason": "trivial", "n": n, "m": m,
                "ell": 0, "N": 0, "num_sources": 0, "max_s_v": 0.0,
                "threshold": 0.0, "noise_floor_1_over_N": 0.0,
                "per_source_s": {}}

    deg = dict(G.degree())
    two_m = 2.0 * m
    ell = max(1, int(math.ceil((40.0 / (Phi**2)) * math.log(n))))
    if N is None:
        N = int(round(n ** walk_exponent))
    N = max(N, 1)

    # Phase 2: degree-weighted source sampling (paper Alg.1 line 6, with eps).
    S: List[Node] = [
        v for v in nodes
        if rng_py.random() < min(1.0, 1e4 * deg[v] / (2.0 * epsilon * m))
    ]
    if not S:
        S = [rng_py.choice(nodes)]
    if max_sources is not None and len(S) > max_sources:
        S = rng_py.sample(S, k=max_sources)
    if len(S) > 1e5 / epsilon:
        return {"decision": "REJECT", "reason": "S_too_large", "n": n, "m": m,
                "ell": ell, "N": N, "num_sources": len(S), "max_s_v": float("inf"),
                "threshold": 0.0, "noise_floor_1_over_N": 1.0 / N,
                "per_source_s": {}}

    # Stationary distribution.
    pi = {u: deg[u] / two_m for u in nodes}

    # Phase 3 + 4: per-source endpoint distribution and discrepancy s_v.
    per_source_s: Dict[Node, float] = {}
    node_endpoint_counts: Dict[Node, float] = {u: 0.0 for u in nodes}
    for s in S:
        c = _endpoint_distribution_aggregated(G, s, ell, N, rng_np)
        disc = 0.0
        # Σ_u (c[u]/N − π_u)^2. Vertices with c[u]=0 contribute π_u^2.
        seen = set()
        for u, cu in c.items():
            What = cu / N
            disc += (What - pi[u]) ** 2
            seen.add(u)
            # Accumulate fraction of walks ending at u, averaged over sources.
            node_endpoint_counts[u] += What / max(len(S), 1)
        for u in nodes:
            if u not in seen:
                disc += pi[u] ** 2
        per_source_s[s] = disc

    max_s = max(per_source_s.values()) if per_source_s else 0.0

    if threshold is None:
        threshold = m ** (-threshold_power)

    decision = "ACCEPT" if all(sv <= threshold for sv in per_source_s.values()) else "REJECT"
    reason = "accept" if decision == "ACCEPT" else "discrepancy"

    return {
        "decision": decision, "reason": reason,
        "n": n, "m": m, "ell": ell, "N": N, "num_sources": len(S),
        "max_s_v": max_s, "threshold": threshold,
        "noise_floor_1_over_N": 1.0 / N,
        "per_source_s": {str(k): v for k, v in per_source_s.items()},
        "node_endpoint_counts": node_endpoint_counts,  # avg walk fraction per node
        "sources": list(S),
    }


if __name__ == "__main__":
    import argparse, json, time
    import graph_zoo as zoo

    p = argparse.ArgumentParser()
    p.add_argument("--family", default="reg4",
                   choices=["reg4", "reg6", "reg8", "barbell", "complete",
                            "path", "two_cliques"])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--Phi", type=float, default=0.3)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--walk_exponent", type=float, default=100.0,
                   help="N = n^walk_exponent. Default 100 = paper-faithful.")
    p.add_argument("--N", type=int, default=None, help="Set N directly.")
    p.add_argument("--threshold_power", type=float, default=15.0)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold directly (e.g. kappa/N to probe noise).")
    p.add_argument("--max_sources", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph_seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.family in ("reg4", "reg6", "reg8"):
        d = {"reg4": 4, "reg6": 6, "reg8": 8}[args.family]
        G = zoo.random_regular(d, args.n, seed=args.graph_seed)
    elif args.family == "barbell":
        G = zoo.barbell(args.n // 2, 1)
    elif args.family == "complete":
        G = zoo.complete(args.n)
    elif args.family == "path":
        G = zoo.path(args.n)
    elif args.family == "two_cliques":
        G = zoo.two_cliques_one_bridge(args.n // 2)
    else:
        raise SystemExit(f"unknown family {args.family}")

    t0 = time.perf_counter()
    result = fv_aggregated(
        G, Phi=args.Phi, epsilon=args.epsilon,
        N=args.N, walk_exponent=args.walk_exponent,
        threshold=args.threshold, threshold_power=args.threshold_power,
        max_sources=args.max_sources,
        numpy_seed=args.seed, py_seed=args.seed,
    )
    result["runtime_s"] = time.perf_counter() - t0
    result["family"] = args.family
    result["graph_seed"] = args.graph_seed
    result["seed"] = args.seed
    if isinstance(result["N"], int) and result["N"] > 2**62:
        result["N"] = str(result["N"])
    print(json.dumps(result, default=str))
    if args.out is not None:
        with open(args.out, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
