# Do CPU workers feeding a single GPU shorten a complete Nautilus fit?

On this RTX 2060 laptop, two CPU feeder threads gave no repeatable whole-fit speedup on either workload. With the 100-point Gaussian profile, the feeder was 14.6% faster than the full-batch GPU control on seed 1 and 7.4% slower on seed 2. With 64 copies of the observations it was 2.6% faster and 11.2% slower, respectively. Neither workload met the registered requirement of at least 5% faster on both seeds and a win over the serial split control. Retain native serial training and the existing execution defaults. [Phase protocol](https://github.com/PyAutoLabs/autofit_profiling/issues/42).

| Likelihood observations | Seed | CPU only | GPU, full batch | GPU, serial half batches | GPU, worker-fed half batches |
|---|---:|---:|---:|---:|---:|
| 100 | 1 | 44.76 s | 70.12 s | 52.16 s | 59.86 s |
| 100 | 2 | 47.88 s | 55.01 s | 47.50 s | 59.07 s |
| 6,400 | 1 | 40.35 s | 46.52 s | 44.16 s | 45.32 s |
| 6,400 | 2 | 48.57 s | 38.93 s | 55.70 s | 43.28 s |

The primary clock includes the complete PyAutoFit `search.fit`, first objective compilation, worker lifecycle, normal output and checkpointing. The receipt also includes elapsed time for the fresh Python process from launch to exit. The CPU baseline uses the same FP64 JAX objective as the GPU arms. The 6,400-point diagnostic repeats the original data 64 times and averages the log likelihood, increasing the work per call while preserving the mathematical target. It is a synthetic cost change on the same model. [All case results and paired comparisons](result.json).

## Where the fit spent its time

Full-width GPU likelihood evaluation took 0.50–0.72 s per complete fit, or 0.78–1.61% of wall time. This includes the prior transform, first compilation, device dispatch and collection through host-visible likelihood values. At 100 observations the median warm logical batch took 3.99–4.01 ms with a serial GPU host and 5.51–6.70 ms with feeders. At 6,400 observations it took 6.76–7.00 ms and 8.07–8.16 ms, respectively. The CPU-only warm batch medians were 0.48 ms at 100 observations and 1.11–1.35 ms at 6,400. Feeding could also affect read-only sampler geometry. The faster whole-fit times in individual pairs cannot be attributed to it from the end-to-end clock alone.

The full-width GPU runs spent 15.6–25.3 s in native neural training. The measurements used one shared WSL laptop and two seeds with reversed run orders. The first fit also built the Matplotlib font cache. Device-wide one-second polling recorded 91–99 MiB at peak across the GPU runs. The kernels were short relative to that sampling interval. Per-batch timings provide the finer measurement. This pilot leaves GPU-heavy observational models, CPU/GPU co-evaluation and multi-GPU execution open.

## Execution and numerical checks

All 16 fits ran sequentially with the same four physical CPU cores available, one numerical-library thread, 200 live points, an update interval of 200, logical batches of 100 and four native networks trained serially. The controller owned all RNG state and waited for the complete logical batch before admitting results. CPU-only and full-width GPU evaluated one vector of 100. Both half-width arms evaluated two ordered vectors of 50. For the feeder, two persistent CPU threads applied the native prior transform to the preselected proposals. They also split read-only neural membership checks on candidate arrays of at least 1,000 points. The controller alone owned the GPU and called the existing PyAutoFit objective. The two sub-batches could be outstanding within one logical batch. The result does not measure speculative proposal generation across adaptive boundaries.

The feeder uses PyAutoFit’s existing `chunk_slices` contract. It repeats the last physical row to maintain a fixed device width and discards padded outputs before admission. Each scored fit used one cached compiled batched objective. All native batches happened to be full in these fits. Separate CPU and RTX 2060 probes exercised lengths 1, 17, 50, 99, 100 and 137 at both widths against the canonical `evaluate_in_chunks` helper and a NumPy reference, including points near the fitted peak. [Existing variable-batch task](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/bug/autofit/vmap_jit_recompiles_per_nautilus_batch_length.md) · [Hardware probes](probes.json).

All 119,200 admitted values passed the independent FP64 likelihood audit (absolute tolerance 1e-8, relative tolerance 5e-12). The largest absolute difference was 2.8e-9. All proposal hashes matched within each seed and workload. Paired log evidence agreed within 7.2e-15 and marginal means within 2.4e-12 posterior standard deviations. Native ESS was about 1,301 and 1,381 for seeds 1 and 2. All 16 native completion markers and per-fit likelihood counts reconciled.

A preliminary attempt ended on the first batch because the cache counter was read from the wrapper. The corrected driver reads the compiled callable through `__wrapped__`. All scored cases started in fresh directories. PyAutoFit reported a nonfatal visualization warm-up warning in all arms. Quick updates were disabled and normal fit output completed.

## Reproduction

The measured host used an Intel i9-10885H, one NVIDIA GeForce RTX 2060 Max-Q (6 GiB), WSL2, Python 3.12.10, JAX and jaxlib 0.12.0.1, Nautilus 1.0.5, NumPy 2.2.6, SciPy 1.17.1 and scikit-learn 1.5.1. Make the pinned PyAutoFit and PyAutoNerves checkouts available on `PYTHONPATH` in a CUDA-capable JAX environment. Choose four CPU IDs on distinct physical cores. Use a new output directory. [PyAutoFit revision](https://github.com/PyAutoLabs/PyAutoFit/tree/911fe05e1c6b570df51789f0ab5b2cbf54c1f43a) · [PyAutoNerves revision](https://github.com/PyAutoLabs/PyAutoNerves/tree/7f3055b7f3d1967f372bdb41ee732135bca0e68d).

```sh
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false
JAX_PLATFORMS=cpu python experiments/nautilus_gpu_feed/probe.py cpu-probe.json
JAX_PLATFORMS=cuda python experiments/nautilus_gpu_feed/probe.py cuda-probe.json
taskset -c 0,2,4,6 python experiments/nautilus_gpu_feed/pilot.py /path/to/fresh-output
python experiments/nautilus_gpu_feed/compare.py /path/to/fresh-output
```

The supervisor enforces one platform per process, retains the case logs and emits device names, software version, CPU time, sampled device usage, model counts, posterior summaries and fit times. This local phase uses the [canonical profiling producer task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
