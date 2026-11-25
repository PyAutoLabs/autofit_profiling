# Why fresh-sample efficiency falls at 30 dimensions

On the saved seed-0 path at fresh-only ESS 10,000, the correlated Gaussian needed 1.17 likelihood calls per effective sample at 5D, 1.64 at 10D and 98.02 at 30D. In the three independently explored 30D fits the cost was 109.73–156.04. A few large weights within each shell dominate the ESS denominator. In the independent fits the highest-weight 1% of draws in each shell together contributed 94.85–95.61% of total squared posterior weight. [Registered phase 0b diagnostic](https://github.com/PyAutoLabs/autofit_profiling/issues/36).

| Dimension and exploration | Earlier calls | Fresh calls | Fresh ESS | Calls per ESS | Top 1% share of squared weight | Hindsight saving | Split-half saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D / saved seed 0 | 4,600 | 11,700 | 10,020.8 | 1.17 | 4.31% | 0.1651% | +0.05% |
| 10D / saved seed 0 | 6,400 | 16,500 | 10,062.2 | 1.64 | 12.99% | 0.1279% | -0.06% |
| 30D / saved seed 0 | 34,100 | 980,300 | 10,001.1 | 98.02 | 94.05% | 0.0060% | -9.48% |
| 30D / seed-0 bounds, resample 1 | 34,100 | 636,100 | 10,000.7 | 63.61 | 92.52% | 0.0015% | -12.55% |
| 30D / seed-0 bounds, resample 2 | 34,100 | 835,600 | 10,001.7 | 83.55 | 93.79% | 0.0007% | -17.36% |
| 30D / independent seed 1 | 33,700 | 1,248,500 | 10,000.1 | 124.85 | 94.98% | 0.0012% | -10.09% |
| 30D / independent seed 2 | 34,600 | 1,097,600 | 10,003.0 | 109.73 | 94.85% | 0.0005% | -15.36% |
| 30D / independent seed 3 | 34,200 | 1,560,600 | 10,001.0 | 156.04 | 95.61% | 0.0003% | -8.48% |

Earlier calls are total native calls minus fresh rows. They are excluded from the fresh-sample cost. The three rows using seed-0 bounds share one exploration. The final three use different learned bounds. The 5D and 10D rows each cover a single exploration. These are direct native CPU runs on the analytic fixture. [Original 5D/10D records](../nautilus_30d_diagnostics/evidence/attempts.json) · [Seed-0 30D continuations](../nautilus_30d_evidence_check/result.json) · [Independent 30D endpoints](../nautilus_30d_matched_ess/result.json).

## What the allocation comparison measures

Kish ESS is the square of the sum of the weights divided by the sum of squared weights. To measure concentration, take the largest likelihoods within each shell (round 1% up to a whole draw), weight them by the same shell factor as the other rows, and sum their squared weights across shells. These shares describe the ESS denominator. They do not measure posterior probability or rejected proposals.

Let a shell have estimated volume V, likelihood mean m, second raw likelihood moment s, and fraction p of the fresh likelihood calls. Under fixed shell volumes and moments, evidence is Z = Σ V m. Predicted fresh calls per ESS are Σ(V² s / p) / Z². For equal cost per call the continuous minimum occurs at p proportional to V √s. This follows directly from Cauchy–Schwarz. Native per-shell minimums, integer batches, changes in volume estimates and the cost of acquiring the moments are omitted from the optimum. These assumptions make it an optimistic retrospective comparison of allocation on these shells.

We reconstructed each historical ESS and log evidence using the complete fresh sections of all shells. The hindsight calculation then uses those same rows to find the optimum allocation. Across all six 30D rows the predicted saving is at most 0.0061%. This near-agreement is consistent with native Nautilus already directing later draws toward shells with large contributions to the ESS denominator. These endpoints leave open the gains from changing the proposal distribution, learned geometry or the number of calls needed to learn those bounds.

As a sensitivity check, estimate allocation proportions from the first half of each shell and score both that allocation and the observed proportions on the second half. Both use the final recorded shell volumes. The native proportions also reflect adaptation to the full run. This calculation tests sensitivity to the observed likelihood moments. A live allocation change would require a separate experiment. The early-half allocation predicts 8.48–17.36% more calls per ESS across the six 30D cases. A negative saving denotes an increase.

## Decision

Retain native shell allocation for this fixture. A candidate worth testing is a better proposal *within* a learned shell. A bounded next diagnostic can use retained positions to ask whether quantities available before evaluating the likelihood (for example, distance in the bound geometry) identify where the high weights occur on held-out draws. Any later proposal change needs its own support, importance-weight and evidence checks. No algorithm change or new campaign was made here. Wider phase 0b remains open.

## Reproduction

Run the following command with NumPy and SciPy from a checkout containing the three source receipts. The compact input contains counts and sums of likelihoods and squared likelihoods in log space, plus the top-tail sum for each shell. All eight cases are required. [Compact inputs and source hashes](moments.json) · [Per-case numbers](result.json).

```sh
python experiments/nautilus_shell_efficiency/analyze.py
```

To re-extract the input use Python 3.11 or later with h5py and pass `--extract PATH_MAP.json`. Map each case key in moments.json to its retained HDF5 checkpoint. Extraction reads only the arrays past each stored exploration cutoff, checks native shell counts, volumes, mean likelihoods, ESS and the aggregate phase receipt, and records the checkpoint hash. The two earlier low-dimensional files still carry the historical false discard flag. Their fresh-only shell summaries and explicit cutoff counts reconcile; neither source file was modified. [Canonical profiling task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
