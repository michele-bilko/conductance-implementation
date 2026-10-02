"""
Graph constructors used by the experimental suite.

Each constructor returns a fresh `networkx.Graph` with integer node labels
in [0, n). For reproducibility every constructor takes an optional `seed`.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import networkx as nx


def _relabel(G: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(G)


# -----------------------------
# Good conductors
# -----------------------------

def complete(n: int) -> nx.Graph:
    return _relabel(nx.complete_graph(n))


def random_regular(d: int, n: int, seed: int = 0) -> nx.Graph:
    # nx requires n*d even
    if (n * d) % 2 == 1:
        n += 1
    return _relabel(nx.random_regular_graph(d, n, seed=seed))


def erdos_renyi_dense(n: int, c: float = 2.0, seed: int = 0) -> nx.Graph:
    """G(n, p) with p = c * ln(n) / n. For c >= 1 this is connected w.h.p."""
    p = min(1.0, c * math.log(max(n, 2)) / n)
    return _relabel(nx.erdos_renyi_graph(n, p, seed=seed))


# -----------------------------
# Bad conductors
# -----------------------------

def barbell(clique_size: int, path_length: int) -> nx.Graph:
    return _relabel(nx.barbell_graph(clique_size, path_length))


def two_cliques_one_bridge(clique_size: int, seed: int = 0) -> nx.Graph:
    """
    Two disjoint K_{clique_size} joined by exactly one edge.
    Bottleneck conductance ~ 1 / (clique_size*(clique_size-1)).
    """
    G = nx.disjoint_union(nx.complete_graph(clique_size),
                          nx.complete_graph(clique_size))
    # Bridge from a node in the first clique to a node in the second.
    G.add_edge(0, clique_size)
    return _relabel(G)


def two_cliques_k_bridges(clique_size: int, k: int, seed: int = 0) -> nx.Graph:
    """
    Two K_{clique_size} joined by k disjoint bridge edges between distinct vertex pairs.
    Bottleneck conductance ~ k / (clique_size*(clique_size-1)).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > clique_size:
        raise ValueError("Cannot have more bridge edges than vertices per side")

    G = nx.disjoint_union(nx.complete_graph(clique_size),
                          nx.complete_graph(clique_size))
    # Map left vertices 0..n-1, right vertices n..2n-1. Connect i <-> n+i for i=0..k-1.
    for i in range(k):
        G.add_edge(i, clique_size + i)
    return _relabel(G)


def path(n: int) -> nx.Graph:
    return _relabel(nx.path_graph(n))


def lollipop(clique_size: int, tail_length: int) -> nx.Graph:
    return _relabel(nx.lollipop_graph(clique_size, tail_length))


def planted_partition(n: int, n_blocks: int = 2, p_in: float = 0.5,
                      p_out: float = 0.01, seed: int = 0) -> nx.Graph:
    """
    Stochastic block model with `n_blocks` equal-sized blocks.
    p_out small => weak inter-cluster mixing => low conductance.
    """
    sizes = [n // n_blocks] * n_blocks
    sizes[0] += n - sum(sizes)
    probs = [[p_in if i == j else p_out for j in range(n_blocks)]
             for i in range(n_blocks)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    # Connect any tiny disconnected components by linking them to vertex 0;
    # otherwise neither tester is well-defined.
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        anchor = next(iter(comps[0]))
        for comp in comps[1:]:
            G.add_edge(anchor, next(iter(comp)))
    return _relabel(G)


# -----------------------------
# Zoo for experiments
# -----------------------------

def zoo_for_correctness(n: int = 50, seed: int = 0) -> List[Tuple[str, nx.Graph, str]]:
    """
    Returns a list of (name, graph, expected_label) for E1.
    expected_label is "ACCEPT" or "REJECT" — the ground truth the
    algorithms are *supposed* to return at sensible parameter settings.
    """
    return [
        # Good conductors --------------------------
        (f"complete_n{n}",            complete(n),                              "ACCEPT"),
        (f"reg4_n{n}",                random_regular(4, n, seed=seed),          "ACCEPT"),
        (f"reg6_n{n}",                random_regular(6, n, seed=seed),          "ACCEPT"),
        (f"reg8_n{n}",                random_regular(8, n, seed=seed),          "ACCEPT"),
        (f"er_dense_n{n}",            erdos_renyi_dense(n, c=2.0, seed=seed),   "ACCEPT"),

        # Bad conductors ---------------------------
        (f"barbell_{n//2}_1",         barbell(n // 2, 1),                       "REJECT"),
        (f"barbell_{n//2}_5",         barbell(n // 2, 5),                       "REJECT"),
        (f"two_cliques_bridge_n{n//2}", two_cliques_one_bridge(n // 2),         "REJECT"),
        (f"path_n{n}",                path(n),                                  "REJECT"),
        (f"lollipop_{n//2}_{n//2}",   lollipop(n // 2, n // 2),                 "REJECT"),
        (f"sbm_2blocks_n{n}",         planted_partition(n, 2, 0.5, 0.005, seed),"REJECT"),
    ]


def family_for_scaling(family: str, n: int, seed: int = 0) -> nx.Graph:
    """Used by E2 — same family, varying n."""
    if family == "reg4":
        return random_regular(4, n, seed=seed)
    if family == "barbell":
        return barbell(n // 2, 1)
    if family == "complete":
        return complete(n)
    if family == "path":
        return path(n)
    raise ValueError(f"Unknown family: {family}")


def gap_ladder(clique_size: int = 30, ks: List[int] | None = None) -> List[Tuple[int, nx.Graph]]:
    """E4: two K_{clique_size} joined by k bridge edges, k swept."""
    if ks is None:
        ks = [1, 2, 3, 5, 8, 12, 18, 25, clique_size]
    # Clamp to valid range: at least 1, at most clique_size bridge edges.
    ks = sorted(set(min(max(k, 1), clique_size) for k in ks))
    return [(k, two_cliques_k_bridges(clique_size, k)) for k in ks]
