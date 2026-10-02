"""
BTT tester — Batu, Trehan & Trehan, "All You Need are Random Walks: Fast and
Simple Distributed Conductance Testing" (2023).
Paper: https://doi.org/10.48550/arXiv.2305.14178

This module is a sequential (single-process) simulation of the CONGEST
algorithm described in the paper's Algorithm 1 (source sampling + decision)
and Algorithm 2 (moving one step of every walk while avoiding congestion).
It is meant to reproduce the tester's *decisions* faithfully so we can study
its behavior on real graphs, not to model actual distributed message
passing or round timing.

Two things to know before using this module:

1. Walk budget. The paper's tester runs K = 2m^2 random walks per source.
   Simulating that many individual walks in a Python loop is only tractable
   for small graphs. `walk_scale` lets us run at a fraction of K for quick
   experiments; the local threshold tau_v is rescaled to stay sound at any
   K (see the comment inside `btt()`). To run at the true paper-faithful
   K = 2m^2 on graphs too large for the per-walk loop, use
   `btt_aggregated.btt_aggregated()` instead, which reproduces the exact
   same random process via count propagation instead of per-walk simulation.

2. Visualization. `btt()` accepts `collect_viz=True` to return per-vertex
   walk counts and reject flags alongside the decision. `experiments.py`
   uses this to write the CSVs that `plot_from_csv.py` renders into figures.
   There is no standalone plotting function in this module; all
   visualization goes through that CSV pipeline so figures are reproducible
   from saved data without re-running the (randomized) simulation.
"""

import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx

Node = Hashable
SourceId = Hashable


def _lazy_random_step(G: nx.Graph, v: Node, rng: random.Random) -> Node:
    """
    One step of the lazy random walk M used throughout the paper.

    Stay at v with probability 1/2; otherwise move to a uniformly random
    neighbor of v. A degree-0 vertex has no neighbors to move to, so it
    always stays.
    """
    neighbors: List[Node] = list(G.neighbors(v))
    if not neighbors or rng.random() < 0.5:
        return v
    return rng.choice(neighbors)


def move_walks_one_step(
    G: nx.Graph,
    W: Dict[Node, Dict[SourceId, int]],
    ell: int,
    step: int,
    epsilon: float,
    rng: Optional[random.Random] = None,
    congestion_cap_factor: float = 5500.0,
) -> Tuple[Dict[Node, Dict[SourceId, int]], Dict[Node, Dict[SourceId, int]], bool]:
    """
    Algorithm 2 (Move-Walks-At-v), simulated globally for all vertices at once.

    Parameters mirror the paper's notation:
    - W: for each vertex v, W[v][q] is the number of length-i walks from
         source q currently stationed at v (the tuples (q, k, i) in W_v).
    - ell: target walk length L in the paper.
    - step: the current step i (1-indexed).
    - epsilon: distance parameter used in the congestion bound 5500/epsilon.

    Returns (W_next, C, congested):
    - W_next: W after moving every unfinished walk one step.
    - C: for each vertex v, C[v][q] is the number of walks from q that
         finished at v on this step (only nonempty when step == ell).
    - congested: True if any single edge would need to carry more than
         5500/epsilon tuples this round, which the paper treats as an abort
         (the vertex that would send that message rejects immediately).
    """
    if rng is None:
        rng = random

    # D_v in the paper: for each sender v, D[v][q][dest] is k in (q, k, dest).
    D: Dict[Node, Dict[SourceId, Dict[Node, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    # C_v in the paper: for each v, C[v][q] is count in (q, count) of walks that ended at v.
    C: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))

    # Lines 6-10 of Algorithm 2: loop over tuples in W_v for every vertex.
    for v in G.nodes():
        if v not in W:
            continue
        for q, count in W[v].items():
            if step != ell:
                # Not the last step: draw next destinations for k lazy walks.
                for _ in range(count):
                    dest = _lazy_random_step(G, v, rng)
                    D[v][q][dest] += 1
            else:
                # Last step: these walks end at v.
                C[v][q] += count

    # Congestion check (lines 13-15 of Algorithm 2): would any edge carry
    # more than 5500/epsilon tuples this round?
    congested = False
    max_tuples_per_message = congestion_cap_factor / epsilon if epsilon > 0 else float("inf")
    for v in D:
        per_dest_sources: Dict[Node, int] = defaultdict(int)
        for q, dest_counts in D[v].items():
            for dest, cnt in dest_counts.items():
                if cnt > 0:
                    per_dest_sources[dest] += 1
        if per_dest_sources and max(per_dest_sources.values(), default=0) > max_tuples_per_message:
            congested = True
            break

    # Build next W from D, unless congestion already forces a reject.
    W_next: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))
    if not congested:
        for v in D:
            for q, dest_counts in D[v].items():
                for dest, cnt in dest_counts.items():
                    if cnt:
                        W_next[dest][q] += cnt

    return W_next, C, congested


def btt(
    G: nx.Graph,
    alpha: float = 0.3,
    epsilon: float = 0.1,
    walk_scale: float = 1.0,
    max_sources: Optional[int] = None,
    rng: Optional[random.Random] = None,
    collect_viz: bool = False,
):
    """
    Algorithm 1 (Distributed-Graph-Conductance-Test), Batu-Trehan-Trehan.

    Parameters follow the paper's notation as closely as possible:
    - alpha: conductance parameter.
    - epsilon: distance parameter.
    - ell = (32 / alpha^2) * log(n): walk length.
    - K = 2 * m^2: number of walks per source, scaled by `walk_scale` (see
      the module docstring for why you would want to).
    - tau_v: local rejection threshold at vertex v.

    Each vertex independently joins the source set Q with probability
        p_v = 5000 * deg(v) / (epsilon * 2m),
    then every source in Q starts K lazy walks of length ell, advanced via
    `move_walks_one_step`. At the end, every vertex v checks each (q, count)
    it received and REJECTS if count > tau_v.

    Returns the decision string "ACCEPT" or "REJECT". If collect_viz=True,
    returns a dict instead:
        {
            "decision": "ACCEPT" | "REJECT",
            "total_load": {vertex: total walks ending there, all sources summed},
            "violated": {vertex: True if it locally rejected},
            "sources": list of sampled source vertices,
            "congested": whether the run aborted due to congestion,
        }
    """
    if rng is None:
        rng = random

    nodes: List[Node] = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return "ACCEPT"

    m = G.number_of_edges()
    if m == 0:
        # An edgeless graph has zero conductance, but the tester's behavior
        # is undefined there (deg(v) = 0 for all v makes tau_v = 0). We
        # accept trivially rather than reject a graph the paper doesn't
        # define the tester on.
        return "ACCEPT"

    deg: Dict[Node, int] = dict(G.degree())

    # ell = (32 / alpha^2) * log(n).
    ell = max(1, int(math.ceil((32.0 / (alpha**2)) * math.log(n))))

    # K = 2 * m^2 at walk_scale=1.0 (paper-faithful); scaled down otherwise.
    K_theoretical = 2 * m * m
    K = max(1, int(K_theoretical * walk_scale))

    # Local thresholds.
    #
    # The paper's threshold (defined for K = 2m^2 only) is
    #     tau_v = m * deg(v) * (1 + 2 n^{-1/4}).
    #
    # For K != 2m^2, the mean of the walk count E[X_{u,v}] = K * deg(v)/(2m)
    # scales linearly in K, but its Chernoff concentration slack scales as
    # sqrt(K). Naively rescaling the paper's tau linearly in K would shrink
    # that slack too fast and make the threshold unsound at small K. Instead
    # we decompose tau_v into a rescaled mean term plus an explicit
    # concentration term and rescale each correctly:
    #
    #   mean term:          K * deg(v)/(2m) * (1 + 2 n^{-1/4})   [paper's own
    #                                                              completeness
    #                                                              slack on the mean]
    #   concentration term: chernoff_c * sqrt(K * deg(v)/(2m))   [~3 std devs]
    #
    # At K = 2m^2 the concentration term is small relative to the mean term
    # for the paper's regime, so this reduces to (and is slightly looser
    # than) the paper's exact formula; at smaller K it stays sound.
    chernoff_c = 3.0
    tau = {
        v: K * deg[v] / (2.0 * m) * (1.0 + 2.0 * n ** (-0.25))
           + chernoff_c * math.sqrt(K * deg[v] / (2.0 * m))
        for v in nodes
    }

    # Line 8: degree-weighted sampling of the source set Q.
    sources: List[Node] = []
    for v in nodes:
        p_v = min(1.0, 5000.0 * deg[v] / (epsilon * 2.0 * m))
        if rng.random() < p_v:
            sources.append(v)
    if not sources:
        # Very small graphs / parameters can sample zero sources; force at
        # least one so the test is still meaningful.
        sources = [rng.choice(nodes)]
    if max_sources is not None and len(sources) > max_sources:
        sources = rng.sample(sources, k=max_sources)

    # Lines 9-10: initialize W_v at each source and run Algorithm 2 for ell rounds.
    W: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))
    for q in sources:
        W[q][q] = K

    C_total: Dict[Node, Dict[SourceId, int]] = defaultdict(lambda: defaultdict(int))
    congested = False

    for step in range(1, ell + 1):
        W, C_step, congested = move_walks_one_step(
            G=G, W=W, ell=ell, step=step, epsilon=epsilon, rng=rng,
        )
        for v, qcounts in C_step.items():
            for q, cnt in qcounts.items():
                C_total[v][q] += cnt
        if congested:
            break

    if congested:
        decision = "REJECT"
    else:
        decision = "ACCEPT"
        for v in nodes:
            for q, count in C_total[v].items():
                if count > tau[v]:
                    decision = "REJECT"
                    break
            if decision == "REJECT":
                break

    if not collect_viz:
        return decision

    # Per-vertex diagnostics for the visualization pipeline (see
    # plot_from_csv.py): total walk load summed over sources, and whether
    # this vertex locally rejected.
    total_load: Dict[Node, int] = {}
    violated: Dict[Node, bool] = {}
    for v in nodes:
        loads = list(C_total[v].values())
        total_load[v] = sum(loads)
        violated[v] = (not congested) and any(c > tau[v] for c in loads)
    return {
        "decision": decision,
        "total_load": total_load,
        "violated": violated,
        "sources": list(sources),
        "congested": congested,
    }


if __name__ == "__main__":
    # Demo: run BTT once on a good and a bad conductor at a small n.
    # Expect BOTH to come back ACCEPT here but this is not a bug. At graph
    # sizes this small, BTT's completeness slack term (1 + 2 n^{-1/4}) is
    # large enough to overtake the entire signal a modest bottleneck
    # produces; the local threshold only becomes discriminative at larger
    # n. See experiment_E5 in experiments.py (and README.md) for the sweep
    # that shows the crossover, and btt_aggregated.py to run BTT at its
    # full paper-faithful walk budget on larger graphs.
    demo_graphs = {
        "good conductor (4-regular, n=30)": nx.random_regular_graph(4, 30, seed=42),
        "bad conductor (barbell: two K15 joined by 1 edge)": nx.barbell_graph(15, 1),
    }
    print("BTT at small n (expected to accept both -- see experiment_E5 for why):")
    for name, G in demo_graphs.items():
        G = nx.convert_node_labels_to_integers(G)
        decision = btt(
            G, alpha=0.3, epsilon=0.1, walk_scale=0.05, max_sources=5,
            rng=random.Random(0),
        )
        print(f"  {name:48s} -> {decision}")
