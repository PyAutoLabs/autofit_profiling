# Scalar weight updates in Nautilus ellipsoid fitting

Updating a single weight by index reduced the wall time to reconstruct eleven retained outer bounds in every paired block, with a **median reduction of 14.4%**. The 57 native ellipsoid calls captured in preflight returned bitwise-identical centers and matrices under the candidate. [Full timing and parity receipt](result.json)

## Candidate

Nautilus 1.0.5 updates every weight after choosing point `j`. Its native expression makes an integer index array and a Boolean mask for each accepted update.

```python
u = u * (1 - a) + a * (np.arange(n_points) == j)
```

The candidate performs the same scaling in the existing weight array and adds the increment at the chosen index.

```python
u *= 1 - a
u[j] += a
```

The weights are local to the solver. The replacement preserves the selected point, iteration order, and the matrix and covariance calculations. The harness validates the native source hash and substitutes only this expression in a temporary copy. The production library and fitted neural models remain as measured in the earlier experiments.

## Matched timing

Each row sums one call to the native outer mixture constructor per retained bound, using the eleven 400 × 10 arrays from RAL job 415196, seed 2. Forward and reverse blocks alternate. Each individual bound sees the candidate run first in three blocks and second in three. Wall and process CPU measurements enclose construction only. Parity checks and JSON writes occur outside the timers.

| Block | Native wall (s) | Candidate wall (s) | Wall reduction | Process CPU reduction |
|---|---:|---:|---:|---:|
| 1 | 3.287 | 2.781 | 15.4% | 15.3% |
| 2 | 3.577 | 2.882 | 19.4% | 19.4% |
| 3 | 3.839 | 2.862 | 25.5% | 24.7% |
| 4 | 3.788 | 3.281 | 13.4% | 13.4% |
| 5 | 4.547 | 4.474 | 1.6% | 1.6% |
| 6 | 5.981 | 5.808 | 2.9% | 4.5% |

Median process CPU fell 14.4%. Summing all six blocks gives an 11.7% reduction in wall time. Both variants slowed substantially in the final two blocks on the shared WSL2 laptop, where their differences narrowed. The six blocks repeat the same inputs and provide no independent-seed evidence.

## Numerical checks and scope

In preflight the harness records the shape, dtype and SHA-256 of each solver input and each of its three outputs. It compares the full ordered sequence for both variants, and checks each input again after return. The first two outer bounds made 20 and 19 calls; the next nine made two each. Every preflight and timed final geometry had identical dimension masks, membership on retained points, log volume and available ellipsoid matrices and center relative to the native reconstruction. The preflight also checked the original checkpoint using the tolerances from [PR #19](https://github.com/PyAutoLabs/autofit_profiling/pull/19). Random generator state was preserved.

These 57 calls cover the outer mixture constructor. [Whole-fit attribution](https://github.com/PyAutoLabs/autofit_profiling/pull/21) found 101 ellipsoid calls across all bound-building paths and 18.6–19.1% of fit time in the routine. The remaining calls have a different input mix. The next test is a direct native-versus-candidate whole-fit comparison with checkpoint reopening and exact ordered posterior, evidence, ESS, shell and likelihood-count parity. The present experiment provides no end-to-end timing or production default change.

## Reproduction

Use the extracted job 415196 pilot directory and the prerequisite [replay instructions and package pins](../nautilus_bound_geometry/README.md). The committed receipt records the numerical libraries, thread counts and measured harness hash. From the producer root:

```sh
python experiments/nautilus_ellipsoid_weight_update/compare.py \
    /path/to/parent-cpu-diagnostics-ral-01/pilot output/ellipsoid-weight-comparison
```

Choose a fresh output directory. A parity or input failure leaves a halted receipt with any previously recorded progress. This off-board experiment belongs to [issue #22](https://github.com/PyAutoLabs/autofit_profiling/issues/22).
