# Distributed Conductance Testing: BTT vs. Fichtenberger–Vasudev

Directed study under Professor Dora Erdős, Boston University.

This repository implements and benchmarks two CONGEST-model algorithms that
test whether a graph is a good "conductor" (well-mixed, expander-like) or not, using only short random walks and O(log n) rounds of
communication:

- **F&V** — Fichtenberger & Vasudev, *A Two-Sided Error Distributed Property
  Tester for Conductance*, MFCS 2018.
  ([paper](https://doi.org/10.4230/LIPIcs.MFCS.2018.19))
- **BTT** — Batu, Trehan & Trehan, *All You Need are Random Walks: Fast and
  Simple Distributed Conductance Testing*, 2023.
  ([paper](https://doi.org/10.48550/arXiv.2305.14178))

Both papers reduce conductance testing to "run random walks, send only
counts to avoid congestion." F&V does this with a spanning tree that
aggregates a global discrepancy statistic; BTT removes the spanning tree
entirely and has each vertex decide locally against a threshold. The two
full write-ups in [`docs/`](docs/) cover the algorithms, the comparison, and
what running both of them actually revealed:

- [`docs/Directed_Study_Summary.pdf`](docs/Directed_Study_Summary.pdf) —
  the accessible version: background, both algorithms explained, and what
  the implementation work showed.
- [`docs/Implementation_Analysis.pdf`](docs/Implementation_Analysis.pdf) —
  the technical version: a code walkthrough, the E1–E5 experiment results,
  and the root-cause analysis behind the main finding below.

## Findings

Once F&V is run at a walk budget large enough for its aggregated
discrepancy statistic to clear its own sampling noise, it correctly
distinguishes every good/bad conductor pair tested here. BTT, despite being
provably correct and architecturally simpler (no spanning tree, no global
aggregation), essentially never rejects the "bad" graphs in this test suite
at practical sizes (n ≈ 20–80) even when run at its full paper-specified walk
budget. BTT's local threshold has a completeness slack
term that is only *asymptotically* small, and at these sizes that slack
exceeds the actual signal a modest bottleneck produces. `experiment_E5`
(below) demonstrates this directly by holding a graph's bottleneck fixed and
growing it, and the same threshold statistic flips from wrong to right once n
is large enough. See `docs/Implementation_Analysis.pdf` for the full
derivation and numbers.

## Repository layout

```
btt.py                 BTT tester (Algorithm 1 + 2), per-walk simulation.
fv.py                  F&V tester (Algorithm 1–4), per-walk simulation.
btt_aggregated.py      BTT via count propagation (multinomial/Gaussian split)
                        instead of simulating each walk individually — lets
                        BTT run at its true walk budget K = 2m^2.
fv_aggregated.py       F&V via the same count-propagation trick — lets F&V
                        reach N = n^100-scale walk budgets, which is
                        necessary (not optional) for it to work at all.
graph_zoo.py           Graph constructors used across all experiments: good
                        conductors (complete, random-regular, dense
                        Erdős–Rényi) and bad conductors (barbells, lollipops,
                        two cliques joined by k bridges, stochastic block
                        models, paths).
experiments.py         The experiment suite (E1–E5) and CLI. See below.
plot_from_csv.py       Renders per-(algorithm, graph) figures from the CSVs
                        experiments.py writes, without re-running the
                        (randomized) simulation.
docs/                  The two write-ups referenced above.
```

## Why two implementations of each algorithm?

`btt.py`/`fv.py` are direct, per-walk translations of each paper's
pseudocode which simulate every individual random walk in a loop, which is
easy to verify against the paper line-by-line but only tractable at a small
walk budget. `btt_aggregated.py`/`fv_aggregated.py` reproduce the *exact
same random process* by propagating count vectors through a multinomial
split at each step (the same trick both papers use internally to avoid
CONGEST edge congestion), which makes the papers' true walk budgets
(K = 2m², N up to n^100) computationally reachable. For F&V at any per-walk-loop-tractable N, its discrepancy statistic is dominated by its own sampling noise on every graph, so aggregation is what makes the tester work at all. For BTT, aggregation is what makes it possible to check whether the tester's local threshold statistic is discriminative at a given n (see `experiment_E5`).

## Running the experiments

```bash
pip install -r requirements.txt

# All experiments, default settings (writes CSVs to results/):
python experiments.py --experiment all

# Individual experiments:
python experiments.py --experiment E1 --trials 20 --n 40 --visualize
python experiments.py --experiment E2   # scaling with n
python experiments.py --experiment E3   # walk-budget sensitivity
python experiments.py --experiment E4   # conductance-gap ladder
python experiments.py --experiment E5   # BTT's asymptotic crossover (see above)

# Render figures from E1's --visualize output:
python plot_from_csv.py --experiment E1
```

| Experiment | What it measures |
|---|---|
| **E1** | Decision accuracy for both testers across the full graph zoo. |
| **E2** | How accuracy and runtime scale with n, on a good and a bad graph family. |
| **E3** | Sensitivity to the walk budget (BTT's `walk_scale`, F&V's `walk_exponent`). |
| **E4** | A conductance-gap ladder: two cliques joined by k bridge edges, k swept from 1 to the clique size. |
| **E5** | BTT at its full paper-faithful walk budget (via `btt_aggregated`), on a fixed worst-case bottleneck, as n grows — isolates whether BTT's under-rejection in E1–E4 is a walk-budget problem or an n problem. |

Each of `btt.py`, `fv.py`, `btt_aggregated.py`, and `fv_aggregated.py` can
also be run standalone (`python btt.py`, etc.) for a quick sanity check on
one good and one bad graph.

## Larger-scale runs

`btt_aggregated.py` and `fv_aggregated.py` both expose a CLI
(`python btt_aggregated.py --help`) intended for batch/cluster use — e.g. on
BU's Shared Computing Cluster — since their count-propagation approach
scales far better than the per-walk implementations.

## Credits

Besides the papers themselves I also made use of Claude code to help with implementation specifics (specifically to figure out how to implement the aggregations for each algorithm) and used Cursor to write the detailed comments in most of the files (comments are a mix of my own work, modified Cursor comments, and Cursor's actual words). The formatting for the README was also created with Claude (though the words themselves are mostly mine). The test suite, graph zoo, and plot from csv were all generated (at least in part) by Cursor. Parts of the implementation code shown here were debugged using Claude code as well, but the majority of the implementation was done without LLM tools. That being said, this implementation would not have been possible without Claude code and Cursor and it would be dishonest to say otherwise. However, our findings, readings, and analysis of the papers were done during the Spring 2026 semester at Boston University without the assistance of LLMs. Any questions about implementation speciifcs can be directed to me at mbilko@bu.edu. 
