# Does giving native bound training more cores shorten a Nautilus fit?

Keep serial training on these workloads. Two and four in-process training threads made the complete PyAutoFit fit slower in all 12 paired comparisons. On the Gaussian profile, median paired wall time rose 38.6% with two threads and 79.8% with four. On its two-mode control it rose 40.8% and 104.6%. Every pair gave the same ordered proposals, likelihoods, evidence, and byte-identical saved posterior sample table. [Registered experiment and decision rule](https://github.com/PyAutoLabs/autofit_profiling/issues/40).

## Complete fits

Times are seconds for `search.fit`, including native output, checkpointing and pool setup and shutdown. Positive changes mean longer fits. Percentages above are medians of the three within-seed relative changes per fixture.

| Fixture | Seed | Native serial | 2 threads | 4 threads | 2-thread change | 4-thread change |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian profile | 1 | 33.83 | 47.87 | 72.21 | +41.5% | +113.4% |
| Gaussian profile | 2 | 54.30 | 75.26 | 97.65 | +38.6% | +79.8% |
| Gaussian profile | 3 | 86.29 | 97.30 | 109.96 | +12.8% | +27.4% |
| Two-mode control | 1 | 85.24 | 120.00 | 174.42 | +40.8% | +104.6% |
| Two-mode control | 2 | 99.96 | 290.17 | 223.67 | +190.3% | +123.8% |
| Two-mode control | 3 | 122.40 | 169.64 | 183.32 | +38.6% | +49.8% |

## Where the time went

CPU time divided by elapsed training time estimates the average number of cores used by the process during that stage. Four worker threads achieved a median of about 1.3 cores on each fixture. Training consumed more CPU-seconds to produce the same result and also took longer on the clock. The medians below are calculated separately for each metric.

| Fixture | Workers | Median training wall (s) | Median training CPU (core-seconds) | Median training CPU / wall | Other bound geometry (s) | Rest of fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian profile | 1 | 21.96 | 21.84 | 0.99 | 9.88 | 22.47 |
| Gaussian profile | 2 | 40.82 | 49.54 | 1.21 | 10.36 | 24.07 |
| Gaussian profile | 4 | 62.70 | 76.52 | 1.30 | 10.46 | 21.52 |
| Two-mode control | 1 | 47.00 | 46.61 | 0.99 | 14.25 | 38.71 |
| Two-mode control | 2 | 103.95 | 106.77 | 1.13 | 20.94 | 44.75 |
| Two-mode control | 4 | 137.91 | 179.93 | 1.30 | 14.80 | 30.60 |

The workers ran in one Python process. The timing includes the native scikit-learn fits and the ordered map. These measurements do not isolate interpreter contention, numerical-library overhead or host competition. The four CPU cores were available through WSL affinity on a shared laptop. Other activity varied, particularly during the 290-second two-mode fit. The size of the slowdown is machine-dependent. The earlier spawn-based experiment measured a different execution path and had mixed whole-fit results. [Earlier process-pool experiment](https://github.com/PyAutoLabs/autofit_profiling/issues/16).

## Execution and parity

We reused the complete [Gaussian-profile and signed-centre fits](../nautilus_bound_cadence/README.md) and ran their native-cadence arm in fresh output directories. All 18 fits used 200 live points, native update interval 200, batch size 100, four native neural networks, the same stopping configuration and a serial likelihood worker. Each ensemble still trained the same four networks with ordered seeds. Only the training map changed. The single-worker arm used the native serial map, while the other arms used one persistent ordered thread executor per fit. BLAS and OpenMP had one thread each. The primary clock includes lazy thread startup inside training plus executor creation and shutdown. Python interpreter startup and imports are outside that clock for all arms.

Cases ran one at a time on the same four distinct physical cores visible to WSL (Intel Core i9-10885H, logical CPUs 0, 2, 4 and 6). Each worker count occupied each order position once per fixture across seeds 1–3. The control reverses the profile order within each seed. Streaming hashes covered the ordered arrays of proposed points and evaluated log likelihoods. The checks also compared calls, bound builds, mode discovery, evidence, ESS and posterior summaries. Separately, we read all 18 completed output receipts and compared the saved posterior sample files byte-for-byte by SHA-256 across all 12 pairs. The registered threshold required every pair to improve, a median saving of at least 5% on both fixtures, and exact parity.

## Reproduction

Use the pinned PyAutoFit and PyAutoNerves revisions and dependency versions in the [machine-readable receipt](result.json), with those two checkouts on `PYTHONPATH`. The [measured driver](run.py) imports the baseline arm from the cadence experiment. From the producer checkout, choose four logical CPU IDs on distinct physical cores and give the supervisor a fresh output path.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0,2,4,6 \
  python experiments/nautilus_bound_core_scaling/run.py /path/to/fresh-output
```

The supervisor limits each child to 300 seconds, records per-case JSON and fit logs in the output directory, and writes the final paired comparisons to `result.json`. Timing output is host-specific. This local check belongs to the [canonical profiling producer task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
