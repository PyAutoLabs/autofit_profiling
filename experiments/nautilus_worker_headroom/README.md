# How much fit time can two-worker scheduling recover?

Keep the native scheduler on this phase 0a fixture. Across 14 traced fits, an ideal two-worker schedule could save at most **0.39–0.51% of complete fit wall time** with the measured per-proposal service times. The predeclared threshold for a live scheduling experiment was 5%. Even removing the entire submit-to-collection interval would save only 0.65–1.44%. [Phase issue and decision rule](https://github.com/PyAutoLabs/autofit_profiling/issues/28).

The source was a 5D analytic Gaussian with two persistent likelihood workers, native update interval 200 and logical batches of 100. Fixed and parameter-correlated costs each have three seeds per adapter. The synthetic random-cost case has one. Each row below gives the range across its seeds. All percentages in the last three columns use the full **traced** fit as denominator.

| Adapter | Cost | Fits | Worker occupancy inside maps | Entire map | Ideal scheduling ceiling | Single-point chunk prediction |
|---|---|---:|---:|---:|---:|---:|
| direct | fixed | 3 | 29.7–32.7% | 0.65–0.72% | 0.44–0.51% | 0.003–0.005% |
| direct | correlated | 3 | 44.1–45.6% | 0.76–0.81% | 0.43–0.44% | 0.013–0.015% |
| direct | random | 1 | 68.3% | 1.38% | 0.44% | 0.033% |
| pyautofit | fixed | 3 | 54.6–56.0% | 0.92–0.96% | 0.41–0.43% | 0.007–0.014% |
| pyautofit | correlated | 3 | 59.0–60.7% | 0.98–1.01% | 0.40–0.41% | 0.010–0.019% |
| pyautofit | random | 1 | 72.6% | 1.44% | 0.39% | 0.023% |

## Reading the estimates

Worker occupancy is the sum of elapsed per-proposal service times divided by the capacity of two workers over the observed map interval. It includes time a worker might be descheduled during service. The map occupies a small fraction of the total fit, so even low occupancy can coexist with little wall time to recover.

For each batch, the replay measures wall time from the common submission timestamp to collection of the ordered results. With two workers and indivisible proposals, the earliest possible completion is bounded below by the longer of the slowest individual service time and half the sum of all service times. Subtracting that floor from the map interval deliberately lets dispatch, queue gaps and collection overhead vanish. The resulting bound is optimistic and can exceed the gain physically attainable by chunk scheduling. Summation uses disjoint map intervals and keeps parent work, proposal order and the complete-batch adaptation boundary fixed.

The last column compares two zero-overhead queue simulations. Each assigns the next contiguous chunk to the earliest available worker. One uses the native CPython chunk size of 13, the other uses one proposal per chunk. Each simulation reuses the same recorded service intervals and assumes they are independent of assignment. The predicted reductions span 0.003–0.033% of full-fit wall. Actual fine-grained dispatch would incur overhead. The records provide no measured speedup for either a changed scheduler or asynchronous adaptation.

## Inputs and verification

The manifest selects all 14 included two-worker traced fits. All 356 logical batches and 35,600 proposal IDs reconciled against the native likelihood counts. The two original compressed examples remain in the baseline results. The other 12 original traces are included in this directory so the full replay runs from a clean checkout. Every input is checked against its compressed SHA-256 in the pinned manifest. The replay also checks timestamp order, nonoverlap, one result per proposal, assignment of contiguous native chunks to individual workers and the two-worker lower bound. [Baseline manifest and controls](../../results/analytic_gaussian/nautilus/pilot_runs.json) · [Per-fit replay including raw nanosecond totals](result.json).

Run from the producer root with Python 3.10 or later.

```sh
python3 experiments/nautilus_worker_headroom/replay.py
python3 experiments/nautilus_worker_headroom/replay.py --examples-only
```

An optional `--archive PATH` verifies original run-ID directories against the same manifest. The replay starts no workers or samplers. The conclusion applies to the recorded cheap two-worker fits on the shared WSL host. An expensive or more variable target requires its own map share before extrapolation. The phase 0b dimension and accuracy work remains open. [Canonical profiling producer task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
