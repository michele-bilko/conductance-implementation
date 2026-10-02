"""
Count-aggregated BTT (Algorithm 1 + Algorithm 2) for arbitrarily large K.

The naive btt.py loops over each of K = 2m^2 walks per source per step.
For the paper-faithful K = 2m^2 — let alone for the F&V-style N = n^100 —
this is unresonable (on my machine at least). The CONGEST algorithm itself never simulates individual
walks: it propagates (q, count, i) tuples along edges, and the count is
exactly the multinomial split of the previous step's count.

So we can implement the *same* random process in O(ell · m · |Q|) work
*regardless of K*, by using multinomial sampling on the count vector at
each (source, vertex, step). This is what btt_aggregated implements.

For counts that fit in int64 (~9.2e18), we use numpy.random.multinomial.
For larger counts, we fall back to a Gaussian approximation:
  X_i = round(count * p_i + N(0, sqrt(count * p_i * (1 - p_i))))
which is essentially exact for huge counts (relative error vanishes).

The Gaussian fallback breaks the *exact* coupling with the per-walk
simulation but preserves all the moments the analysis relies on. For
counts > 10^15 the empirical difference is unnoticeable.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np

Node = Hashable
SourceId = Hashable

INT64_MAX = 2**62  # leave headroom under numpy's int64 ceiling


def _multinomial_split(count: int, probs: List[float],
                       rng: np.random.Generator) -> List[int]:
    """
    Sample a multinomial split of `count` across len(probs) categories.

    For `count` <= INT64_MAX uses numpy.random.multinomial exactly.
    For larger `count` uses a Gaussian approximation per category, then
    repairs the total so the splits sum to count.
    """
    if count == 0:
        return [0] * len(probs)
    if count <= INT64_MAX:
        return rng.multinomial(int(count), probs).tolist()

    # Gaussian approximation. Means and variances of the marginal binomials.
    out = []
    used = 0
    rem_count = count
    rem_prob = 1.0
    for i, p in enumerate(probs):
        if i == len(probs) - 1:
            # Force the last category to absorb the remainder.
            out.append(rem_count)
            used += rem_count
            break
        if rem_prob <= 0:
            out.append(0)
            continue
        local_p = p / rem_prob
        local_p = min(max(local_p, 0.0), 1.0)
        mean = rem_count * local_p
        var = rem_count * local_p * (1.0 - local_p)
        # Sample, clamp to [0, rem_count].
        x = int(round(mean + rng.standard_normal() * math.sqrt(max(var, 0.0))))
        x = max(0, min(rem_count, x))
        out.append(x)
        used += x
        rem_count -= x
        rem_prob -= p
    return out


def move_walks_one_step_aggregated(
    G: nx.Graph,
    W: Dict[Node, Dict[SourceId, int]],
    ell: int,
    step: int,
    epsilon: float,
    rng: np.random.Generator,
    congestion_cap_factor: float = 5500.0,
) -> Tuple[Dict[Node, Dict[SourceId, int]],
           Dict[Node, Dict[SourceId, int]], bool]:
    """
    Count-aggregated version of Algorithm 2. Same semantics, same return
    signature as btt.move_walks_one_step, but never iterates per-walk.

    For each vertex v and each source q with count c stationed at v:
      - Build the lazy-walk transition probabilities: P(stay)=1/2,
        P(to each neighbor)=1/(2*deg(v)).
      - Sample (c_stay, c_to_w1, c_to_w2, ...) via _multinomial_split.

    Then aggregate per-edge tuples per destination as in the paper.
    """
    D: Dict[Node, Dict[SourceId, Dict[Node, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    C: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))

    for v in G.nodes():
        if v not in W:
            continue
        neighbors = list(G.neighbors(v))
        d = len(neighbors)

        for q, count in W[v].items():
            if count <= 0:
                continue
            if step == ell:
                # Last step: these walks end here.
                C[v][q] += count
                continue
            if d == 0:
                # Isolated vertex: walks stay.
                D[v][q][v] += count
                continue
            # Lazy-walk distribution: [stay, n1, n2, ..., n_d]
            probs = [0.5] + [0.5 / d] * d
            targets = [v] + neighbors
            parts = _multinomial_split(count, probs, rng)
            for t, c in zip(targets, parts):
                if c > 0:
                    D[v][q][t] += int(c)

    # Congestion check, same rule as btt.py.
    congested = False
    max_tuples = (congestion_cap_factor / epsilon) if epsilon > 0 else math.inf
    for v in D:
        per_dest_sources: Dict[Node, int] = defaultdict(int)
        for q, dest_counts in D[v].items():
            for dest, cnt in dest_counts.items():
                if cnt > 0:
                    per_dest_sources[dest] += 1
        if per_dest_sources and max(per_dest_sources.values(), default=0) > max_tuples:
            congested = True
            break

    W_next: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))
    if not congested:
        for v in D:
            for q, dest_counts in D[v].items():
                for dest, cnt in dest_counts.items():
                    if cnt:
                        W_next[dest][q] += cnt

    return W_next, C, congested


def btt_aggregated(
    G: nx.Graph,
    alpha: float = 0.3,
    epsilon: float = 0.1,
    K: Optional[int] = None,
    walk_scale: float = 1.0,
    max_sources: Optional[int] = None,
    numpy_seed: int = 0,
    py_seed: int = 0,
    verbose: bool = False,
) -> Dict:
    """
    Distributed-Graph-Conductance-Test (BTT) using count aggregation.

    Returns a dict:
      {
        "decision": "ACCEPT" | "REJECT",
        "reason":   "tau_violation" | "congestion" | "accept",
        "n": n, "m": m,
        "ell": ell, "K": K, "num_sources": len(Q),
        "violating_vertices": [...],   # only if rejected for tau
        "max_count_seen": int,         # largest C_v[q] value at termination
      }

    By default K = 2m^2 (paper-faithful). Set walk_scale to reduce uniformly
    or pass K explicitly to set it directly. Counts can be arbitrarily large.
    """
    rng_np = np.random.default_rng(numpy_seed)
    rng_py = random.Random(py_seed)

    nodes = list(G.nodes())
    n = len(nodes)
    m = G.number_of_edges()
    if n == 0 or m == 0:
        return {"decision": "ACCEPT", "reason": "trivial",
                "n": n, "m": m, "ell": 0, "K": 0, "num_sources": 0,
                "violating_vertices": [], "max_count_seen": 0}

    deg = dict(G.degree())
    ell = max(1, int(math.ceil((32.0 / (alpha**2)) * math.log(n))))
    if K is None:
        K = int(round(2 * m * m * walk_scale))
    K = max(K, 1)

    K_theoretical = 2 * m * m
    # See btt.btt() for rationale: decompose τ_v into rescaled mean + Chernoff slack.
    chernoff_c = 3.0
    tau = {
        v: K * deg[v] / (2.0 * m) * (1.0 + 2.0 * n ** (-0.25))
           + chernoff_c * math.sqrt(K * deg[v] / (2.0 * m))
        for v in nodes
    }

    # Sample source set Q with p_v = 5000 deg(v) / (eps * 2m).
    Q: List[Node] = []
    for v in nodes:
        p_v = min(1.0, 5000.0 * deg[v] / (epsilon * 2.0 * m))
        if rng_py.random() < p_v:
            Q.append(v)
    if not Q:
        Q = [rng_py.choice(nodes)]
    if max_sources is not None and len(Q) > max_sources:
        Q = rng_py.sample(Q, k=max_sources)

    W: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))
    for q in Q:
        W[q][q] = K

    C_total: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))
    congested = False

    for step in range(1, ell + 1):
        W, C_step, congested = move_walks_one_step_aggregated(
            G, W, ell, step, epsilon, rng_np)
        for v, qcounts in C_step.items():
            for q, cnt in qcounts.items():
                C_total[v][q] += cnt
        if congested:
            break
        if verbose and step % max(1, ell // 10) == 0:
            print(f"  [agg] step {step}/{ell}  |W|={sum(len(d) for d in W.values())}")

    if congested:
        return {"decision": "REJECT", "reason": "congestion",
                "n": n, "m": m, "ell": ell, "K": K, "num_sources": len(Q),
                "violating_vertices": [], "max_count_seen": 0}

    violating: List[Node] = []
    max_count = 0
    for v in nodes:
        for q, count in C_total[v].items():
            if count > max_count:
                max_count = count
            if count > tau[v]:
                violating.append(v)
                break  # one source is enough; v rejects locally

    decision = "REJECT" if violating else "ACCEPT"
    reason = "tau_violation" if violating else "accept"
    return {"decision": decision, "reason": reason,
            "n": n, "m": m, "ell": ell, "K": K, "num_sources": len(Q),
            "violating_vertices": violating, "max_count_seen": int(max_count)}


# -----------------------------
# CLI for SCC use
# -----------------------------

if __name__ == "__main__":
    import argparse, json, time
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="reg4",
                   choices=["reg4", "reg6", "reg8", "barbell", "complete",
                            "path", "two_cliques"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--walk_scale", type=float, default=1.0,
                   help="K = walk_scale * 2*m^2. Default 1.0 = paper-faithful.")
    p.add_argument("--K", type=int, default=None,
                   help="Set K directly; overrides walk_scale.")
    p.add_argument("--max_sources", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph_seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="Append JSON line to this path; useful for SCC job arrays.")
    args = p.parse_args()

    # Build the graph.
    import graph_zoo as zoo
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
    result = btt_aggregated(
        G, alpha=args.alpha, epsilon=args.epsilon,
        K=args.K, walk_scale=args.walk_scale,
        max_sources=args.max_sources,
        numpy_seed=args.seed, py_seed=args.seed,
        verbose=True,
    )
    result["runtime_s"] = time.perf_counter() - t0
    result["family"] = args.family
    result["graph_seed"] = args.graph_seed
    result["seed"] = args.seed
    print(json.dumps(result, default=str))

    if args.out is not None:
        with open(args.out, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
