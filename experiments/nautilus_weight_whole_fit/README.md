# Scalar weight updates preserved the posterior. Whole-fit timing was inconclusive

On the 10D seed-2 fixture, the scalar update was 4.1% slower than native when it ran second and 15.2% faster when it ran first. Both comparisons passed exact scientific parity. The pair effects changed sign, so the registered decision rule leaves the end-to-end speed benefit unresolved. [All four fit receipts](result.json)

| Run order | Native fit | Candidate fit | Candidate wall-time change relative to native |
|---|---:|---:|---:|
| 1. Native then candidate | 41.48 s | 43.17 s | +4.1% |
| 2. Candidate then native | 45.08 s | 38.22 s | −15.2% |

Every fit made **101 enclosing-ellipsoid calls** with the same ordered sequence of input shapes, and **23,500 likelihood evaluations**. The ordered posterior hash (`919674e04195da25f5d6646743b2ce8aa0089fc5642e7cda523dd72991eae260`), log evidence (-12.37447679), ESS (10038.50761), 12 shells and exploration flags matched the frozen native reference exactly. All four checkpoints reopened under unmodified Nautilus with the same scientific signature and zero additional likelihood evaluations. The four cases completed in 174.7 s on the shared WSL2 laptop, with a maximum sampled process-group RSS of 161.5 MiB.

The native controls themselves differed by 3.60 s. This small pilot cannot attribute the variation to execution order or a specific hardware cause. [PR #23](https://github.com/PyAutoLabs/autofit_profiling/pull/23) measured a consistent direction for the isolated outer-bound reconstruction on retained inputs. The whole-fit result does not support changing the production solver. If development continues, the next discriminating experiment is a pre-registered paired comparison on a quieter CPU allocation with multiple seeds and separate kernel and total-fit timings. The present study made no allocation or campaign request.

## What was timed

Each fresh process used Nautilus 1.0.5, 400 live points, four native neural networks, batches of 100 and an ESS target of 10,000. The 10D correlated Gaussian, seed 2, discarded exploration samples, one numerical thread and native checkpoints match the original native experiment. The timed interval starts before constructing `Sampler` and ends after materializing its ordered posterior. It includes checkpoint writes. Imports, loading the experimental solver, result hashing and checkpoint verification occur outside the clock. Both variants count invocations through the same lightweight wrapper. The candidate replaces the enclosing-ellipsoid routine only for that process and restores the original callable afterward.

The native–candidate–candidate–native order produces one pair in each direction. Positive reductions mean the candidate completed sooner. The decision rule gives an inconclusive timing result when the pair signs disagree. Repeated fits use the same seed to check numerical parity and control workload variation. They supply no cross-seed reliability estimate.

The supervisor limited each fresh process group to 180 seconds and 4096 MiB. The four-fit limit was 720 seconds. Any incomplete fit, cap or science mismatch stops the next case. The compact receipt retains the science values, solver call count and ordered shape digest, times and supervision. The four checkpoints and child logs are in the retained raw output.

## Reproduce

Use Linux and the exact Python and dependency versions in [result.json](result.json). This branch contains the [candidate loader](../nautilus_ellipsoid_weight_update/compare.py). The runner also verifies two files from the pinned [whole-fit reference directory](https://github.com/PyAutoLabs/autofit_profiling/tree/d8709a7d536a61f278a311b67ac99ea66f61702f/experiments/nautilus_whole_fit_geometry), `run.py` and `result.json`. Pass their directory in a local checkout of that revision as the second argument.

```sh
python experiments/nautilus_weight_whole_fit/compare_fits.py \
    output/whole-fit-weight-pilot \
    /path/to/baseline-checkout/experiments/nautilus_whole_fit_geometry
```

Use a fresh output directory. The trial is an off-board CPU engineering experiment tracked by [issue #24](https://github.com/PyAutoLabs/autofit_profiling/issues/24).
