# Ellipsoid fitting consumed 18.6–19.1% of the complete native fit

On this local 10D Gaussian, the minimum-volume enclosing ellipsoid routine took 7.88 and 7.37 seconds in the two traced fits. Each made 101 calls across eleven bound builds. Across all four runs the ordered posterior, 23,500 likelihood calls, 12 shells, log evidence (-12.37447679) and ESS (10038.50761) matched exactly, including after reopening each native checkpoint. The posterior hash also matches the archived [RAL seed-2 reference](https://github.com/PyAutoLabs/autofit_profiling/blob/04b33ab7be4d976c43beaaaedf09892f96703974/experiments/nautilus_spawn_training_pool/diagnostics_10d/reference.json).

| Run | Whole native fit (s) | Sampling (s) | All bound builds (s) | Neural training inside builds (s) | Ellipsoid fitting inside builds (s) | Ellipsoid share of whole fit |
|---|---:|---:|---:|---:|---:|---:|
| 1-control | 40.43 | 40.40 | — | — | — | — |
| 2-traced | 42.34 | 42.31 | 30.22 | 20.34 | 7.88 | 18.6% |
| 3-traced | 38.63 | 38.60 | 28.55 | 19.36 | 7.37 | 19.1% |
| 4-control | 39.94 | 39.90 | — | — | — | — |

The table's bound-build time includes neural and ellipsoid work. Ellipsoid fitting consumed 80.2% and 80.6% of the measured build time after subtracting neural training and prediction. The first two builds made 24 and 23 ellipsoid calls, followed by six per build for each of the remaining nine. The earlier [outer-mixture replay](https://github.com/PyAutoLabs/autofit_profiling/pull/19) accounted for 57 calls on these inputs. Whole-bound measurement captures additional calls.

The traced fit was 4.7% slower than its control in the first pair and 3.3% faster in the reverse pair. With one seed and a shared laptop, these timings cannot resolve a small instrumentation penalty. No timing correction was applied. Both direct measurements put ellipsoid fitting near one fifth of this native workload. Halving the routine’s time would save about 9% of whole-fit time if the sampling trajectory and other costs stayed fixed. A bounded kernel experiment on the retained point arrays is now justified before changing sampler code.

The pilot finished in 166.6 seconds on an Intel Core i9-10885H under WSL2. Peak sampled child-process-group RSS was 161.7 MiB. Every fit completed within the registered caps. [Individual results and per-bound call counts](result.json)

## Measurement

The diagnostic runs four fresh native Nautilus fits on the project's analytic correlated Gaussian. Every fit uses the same 10 dimensions and seed 2, 400 live points, a batch of 100, all four native neural networks, native HDF5 checkpoints, discarded exploration samples and a 10,000 ESS target. The likelihood and training paths are serial. The parent process supervises each fresh process group with a 180-second and 4096 MiB cap inside a 720-second aggregate allowance.

The whole-fit clock starts immediately before constructing the sampler and ends when its ordered posterior arrays have been materialized. It includes the writes made during the fit. Imports, creation of the target, subsequent hashing and checkpoint reopening sit outside that clock. The denominator therefore describes a complete **native** fit on this fixture. PyAutoFit plotting and results output are separate costs.

The probed path times each native bound build. While that build is executing, an exclusive wall timer records the minimum-volume enclosing ellipsoid routine, neural training, neural prediction and the remaining build work. Its wrappers are removed on exit. The control path runs the same configuration with all probes off. The two repetitions in forward and reverse order are timing controls at one seed.

## Reproduction

Use Linux, Python 3.10.12 and the package versions in [the receipt](result.json). The producer checks the source hash for Nautilus 1.0.5. The numerical libraries recorded in each receipt use one native thread. From the repository root:

```sh
python experiments/nautilus_whole_fit_geometry/run.py output/whole-fit-geometry-replay
```

Provide a fresh directory. The driver records partial progress in `result.json` and stops on the first failed or capped child or a parity mismatch. Per-run logs, checkpoints and the numeric receipts stay under that directory. The committed compact receipt includes the four outcomes and all measured build times. This engineering experiment remains outside Pulse. [Issue #20](https://github.com/PyAutoLabs/autofit_profiling/issues/20)
