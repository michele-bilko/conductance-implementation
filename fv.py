"""
F&V tester — Fichtenberger & Vasudev, "A Two-Sided Error Distributed Property
Tester for Conductance" (MFCS 2018).
Paper: https://doi.org/10.4230/LIPIcs.MFCS.2018.19

This module is a sequential (single-process) simulation of the CONGEST
algorithm's five phases (diameter check, edge counting, source sampling,
random walks, and aggregated decision). It reproduces the tester's
*decisions* faithfully so we can study its behavior on real graphs; it does
not model actual distributed message passing or round timing. Phases that
are a convergecast-then-broadcast on a spanning tree in the real algorithm
(edge counting, discrepancy aggregation) are computed here as a direct sum,
since that produces the same numbers without simulating individual rounds.

Important: this module runs N literal random walks per source in a Python
loop, which is only tractable for N up to a few thousand. The paper's tester
needs N = n^100 for its noise floor to sit far below the true signal — at any
per-walk-loop-tractable N (e.g. N = n^2), the discrepancy statistic here is
dominated by its own sampling noise on every graph, good or bad, and the
tester cannot distinguish them (see fv_aggregated.py's docstring for the
math). For real experiments, use `fv_aggregated.fv_aggregated()` instead,
which reproduces the exact same random process via count propagation and
reaches the paper's true walk budget in practice.
"""

import math
import random
from collections import defaultdict
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import networkx as nx

Node = Hashable


def build_bfs_tree(G: nx.Graph, Phi: float) -> Tuple[nx.DiGraph, Node, bool]:
    """
    Phase 0 (Algorithm 3, BFS diameter check).

    In the real algorithm, every vertex starts as its own candidate root and
    forwards the smallest root ID it has seen; after
        D = (6 / Phi) * ln(m)
    rounds a unique root has been elected if the diameter is at most D, and
    every vertex knows its parent/children in the resulting BFS tree.
    Rejecting here means "diameter too large to be a good conductor."

    This function computes the same outcome by building a BFS tree from
    the minimum-ID vertex up to depth D, and reject if it does not reach
    every vertex.
    """
    n = G.number_of_nodes()
    if n == 0:
        return nx.DiGraph(), None, True  # type: ignore[return-value]

    m = G.number_of_edges()
    D = max(1, int(math.ceil((6.0 / Phi) * math.log(max(m, 2)))))
    root: Node = min(G.nodes())

    bfs_tree = nx.bfs_tree(G, source=root, depth_limit=D)
    rejected = bfs_tree.number_of_nodes() < n
    return bfs_tree, root, rejected


def aggregate_sum(G: nx.Graph, bfs_tree: nx.DiGraph, f: Callable[[Node], float]) -> float:
    """
    Phase 1 / Phase 4 (Algorithm 4, AggregateSum).

    In the real algorithm this is a convergecast of f(v) up the BFS tree
    followed by a broadcast of the total back down, so every vertex ends up
    knowing sum_v f(v). Since we only need the final number here, we compute
    it directly; `bfs_tree` is accepted for signature symmetry with the
    paper but is not otherwise needed once you're summing centrally.
    """
    return float(sum(f(v) for v in G.nodes()))


def random_walk_phase(
    G: nx.Graph,
    S: List[Node],
    ell: int,
    N: int,
    m: int,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[Node, Dict[Node, float]], bool]:
    """
    Phase 3 (Algorithm 2, RandomWalk).

    For each source v in S, simulate N independent lazy random walks of
    length ell, record their endpoints, and compute
        What^ell_{v,u} = (# walks from v ending at u) / N
        s_{v,u}        = (What^ell_{v,u} - deg(u) / (2m))^2.

    Also checks the paper's line-14 condition: if What^ell_{v,u} <= 2 m^{-2}
    for some u, the real algorithm rejects immediately (this can fire almost
    every time on small graphs at modest N, so the caller decides whether to
    enforce it via `enforce_line14`).

    Returns (s_vu, rejected_line14) where s_vu[v][u] = s_{v,u}.
    """
    if rng is None:
        rng = random

    deg: Dict[Node, int] = dict(G.degree())
    stationary: Dict[Node, float] = {v: deg[v] / (2.0 * m) for v in G.nodes()}
    neighbors: Dict[Node, List[Node]] = {v: list(G.neighbors(v)) for v in G.nodes()}

    s_vu: Dict[Node, Dict[Node, float]] = {}
    rejected_line14 = False

    for s in S:
        endpoint_counts: Dict[Node, int] = defaultdict(int)

        for _ in range(N):
            v: Node = s
            for _ in range(ell):
                if neighbors[v] and rng.random() >= 0.5:
                    v = rng.choice(neighbors[v])
                # else: lazy step, stay at v
            endpoint_counts[v] += 1

        s_vu[s] = {}
        for v in G.nodes():
            w_hat = endpoint_counts.get(v, 0) / float(N)
            if w_hat <= 2.0 * (m ** -2):
                rejected_line14 = True
            s_vu[s][v] = (w_hat - stationary[v]) ** 2

    return s_vu, rejected_line14


def fv(
    G: nx.Graph,
    Phi: float = 0.3,
    epsilon: float = 0.1,
    walk_exponent: float = 4.0,
    enforce_line14: bool = True,
    threshold_power: float = 15.0,
    max_sources: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Algorithm 1 (TestConductance), Fichtenberger-Vasudev.

    Parameters:
    - Phi: conductance parameter.
    - epsilon: distance parameter.
    - walk_exponent: simulate N = n^walk_exponent walks per source instead of
      the paper's n^100, since n^100 individual walks cannot be simulated in
      a loop. See the module docstring for why this matters.
    - enforce_line14: whether to enforce the RandomWalk line-14 rejection
      (What^ell_{v,u} <= 2 m^{-2}). This is overly strict on small
      experimental graphs at modest N, so it defaults to True (paper-exact)
      but is easy to turn off for exploration.
    - threshold_power: uses threshold m^{-threshold_power} in phase 5. The
      theorem uses 15.0; smaller values are useful on small demo graphs
      where m^{-15} underflows float64.

    Runs the five phases in order (diameter check, edge count, source
    sampling, random walks, aggregate-and-decide) and returns "ACCEPT" or
    "REJECT".
    """
    if rng is None:
        rng = random

    nodes: List[Node] = list(G.nodes())
    n = len(nodes)
    m = G.number_of_edges()

    if n == 0 or m == 0:
        return "ACCEPT"

    deg: Dict[Node, int] = dict(G.degree())

    ell = max(1, int(math.ceil((40.0 / (Phi**2)) * math.log(n))))
    N = max(1, int(round(n ** walk_exponent)))

    # Phase 0: BFS diameter check.
    bfs_tree, root, bfs_rejected = build_bfs_tree(G, Phi)
    if bfs_rejected:
        return "REJECT"

    # Phase 1: count edges via convergecast/broadcast (computed directly here).
    m_computed = aggregate_sum(G, bfs_tree, lambda v: deg[v] / 2.0)

    # Phase 2: sample sources S, each vertex marking itself with probability
    # proportional to its volume: p = min(1, 10^4 * deg(v) / (2 * epsilon * m)).
    S: List[Node] = [
        v for v in nodes
        if rng.random() < min(1.0, 1e4 * deg[v] / (2.0 * epsilon * m))
    ]
    if not S:
        S = [rng.choice(nodes)]
    if max_sources is not None and len(S) > max_sources:
        S = rng.sample(S, k=max_sources)
    if len(S) > 1e5 / epsilon:
        return "REJECT"

    # Phase 3: random walks (Algorithm 2).
    s_vu, rejected_line14 = random_walk_phase(G, S, ell, N, m, rng=rng)
    if enforce_line14 and rejected_line14:
        return "REJECT"

    # Phase 4: aggregate discrepancies via AggregateSum.
    sv: Dict[Node, float] = {
        s: aggregate_sum(G, bfs_tree, lambda v, s=s: s_vu[s][v]) for s in S
    }

    # Phase 5: global decision. Every source's discrepancy must be at most
    # the threshold for the tester to accept.
    threshold = m_computed ** (-threshold_power)
    return "ACCEPT" if all(sv[s] <= threshold for s in S) else "REJECT"


if __name__ == "__main__":
    # Demo: run F&V once on a good and a bad conductor at a walk budget
    # small enough to finish quickly (N = n^2.5, a few thousand walks per
    # source). Per the module docstring, this is expected to be unable to
    # tell the two graphs apart — the discrepancy statistic here is
    # dominated by ~1/N sampling noise regardless of graph structure. This
    # demo exists to show that failure mode directly, not to hide it: run
    # fv_aggregated.py instead to see F&V work correctly at a walk budget
    # large enough for the true signal to emerge.
    demo_graphs = {
        "good conductor (4-regular, n=20)": nx.random_regular_graph(4, 20, seed=42),
        "bad conductor (barbell: two K10 joined by 1 edge)": nx.barbell_graph(10, 1),
    }
    print("Per-walk F&V at a tractable N (expected to be noise-dominated -- see fv_aggregated.py):")
    for name, G in demo_graphs.items():
        G = nx.convert_node_labels_to_integers(G)
        decision = fv(
            G, Phi=0.3, epsilon=0.1, walk_exponent=2.5,
            enforce_line14=False, threshold_power=1.0, max_sources=3,
            rng=random.Random(0),
        )
        print(f"  {name:48s} -> {decision}")
