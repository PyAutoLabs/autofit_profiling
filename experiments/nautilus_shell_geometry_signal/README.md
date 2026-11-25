# Can the existing Nautilus bounds identify high-weight draws before evaluating the likelihood?

On the correlated 30D Gaussian, proximity to the centre of the learned ellipsoid identified 64–84% of held-out squared posterior weight in about 10% of fresh draws. This held across three independently learned sets of bounds. The existing neural ensemble’s margin captured 7–11% at the same selection rate. Only ellipsoid distance cleared the registered threshold for a proposal experiment. [Phase protocol](https://github.com/PyAutoLabs/autofit_profiling/issues/38).

| Seed | Score | Held-out draws | Selected | Recall of highest-weight 1% | Recall / random | Squared weight captured | Capture / random |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Ellipsoid | 624,250 | 9.88% | 79.87% | 8.08× | 84.06% | 8.53× |
| 2 | Ellipsoid | 548,800 | 10.02% | 82.71% | 8.26× | 83.41% | 8.38× |
| 3 | Ellipsoid | 780,300 | 9.93% | 66.35% | 6.69× | 64.32% | 6.48× |
| 1 | Neural | 624,250 | 9.93% | 9.03% | 0.91× | 11.05% | 1.11× |
| 2 | Neural | 548,800 | 10.01% | 7.59% | 0.76× | 7.13% | 0.71× |
| 3 | Neural | 780,300 | 9.97% | 9.66% | 0.97× | 11.15% | 1.12× |

Each random comparator samples the same expected number from each shell as that score selected on held-out data. The registered threshold was twice random on **both** metrics in **each** seed. The neural margin measures distance in score from a learned membership cutoff. These results assess its value for ranking points already drawn within shells. The native neural membership test remains in use. [Per-seed metrics and matched baselines](result.json).

## How the replay works

The three native checkpoints had each reached fresh-only ESS 10,000. We retained every fresh point after the stored exploration cutoff, split each shell in temporal order, set each score’s top-decile threshold from the first half, and applied it unchanged to the second. The frozen models came from exploration. The scoring functions take point coordinates and saved model parameters. Held-out likelihoods enter only to evaluate the rankings. [Independent 30D endpoints](../nautilus_30d_matched_ess/README.md) · [Earlier within-shell concentration diagnostic](../nautilus_shell_efficiency/README.md).

The ellipsoid score is negative squared radius in the enclosing ellipsoid’s native coordinates, taking the largest score across components. The neural score uses the largest native ensemble prediction minus its stored threshold among component ellipsoids containing the point. Points without a score rank last. At a cutoff tie each tied point receives its fractional inclusion probability. Each seed had 19 shells with a trained component and one initial shell without geometry. This initial shell contributed at the neutral 10% rate, including its 224,350 held-out draws in seed 3. All 60 shells and 1,953,350 held-out draws enter the table.

Tail recall counts the top 1% of held-out likelihoods within each shell, rounding the number up and splitting boundary ties. Each point in a shell has the same volume factor, so likelihood and posterior weight have the same rank there. To measure global squared-weight capture, give each held-out point weight proportional to its likelihood times native shell volume divided by that shell’s held-out count. The reported fraction is the share of the sum of **squared** weights, the denominator of importance-sampling ESS. A share of squared weight does not directly give a fraction of evidence or a runtime gain.

## Practical implication

The ellipsoid score warrants a bounded experiment that gives high-ranked points more likelihood evaluations while keeping a known, positive inclusion probability everywhere. Compare the reweighted estimate against native sampling on the same frozen bounds. Check analytic evidence, variance, ESS per likelihood call and wall time including scoring. Seed 3 is an important case because 35.68% of squared weight fell outside the selected subset. All measurements here used archived draws from one Gaussian fixture. Native sampling settings stay in production.

## Reproduction

Run the compact replay with NumPy and SciPy from the repository root.

```sh
python experiments/nautilus_shell_geometry_signal/analyze.py
```

To rebuild the [per-shell receipts](shells.json), use Python 3.11 or later, Nautilus 1.0.5, h5py and the three retained HDF5 files. The argument is a directory containing `seed1/checkpoint.hdf5`, `seed2/checkpoint.hdf5` and `seed3/checkpoint.hdf5`. Run with the original compatible scikit-learn environment for the archived networks.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/nautilus_shell_geometry_signal/analyze.py --extract /path/to/checkpoints
```

Extraction checks all three SHA-256 identities against the earlier shell diagnostic and reconstructs native ESS, log evidence and row counts. An independent HDF5 calculation using the stored ellipsoid precision matrix, sorted cutoffs and direct linear weights reproduced the ellipsoid selection counts, tail hits and squared-weight capture for every shell. Native neural membership also agreed on 7,343 sampled point-component pairs. This local diagnostic belongs to the [canonical profiling producer task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
