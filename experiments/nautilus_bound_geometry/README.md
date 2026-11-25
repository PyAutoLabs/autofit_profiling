# CPU cost of the Nautilus outer-bound geometry

Across eleven saved 10D outer bounds, the native minimum-volume enclosing ellipsoid routine accounted for **97.3% and 97.4%** of the outer mixture constructor's measured wall time in two local passes. Each pass made **57 ellipsoid fits**. The first two bounds alone needed 39 of them. This identifies a candidate for finer whole-fit profiling on the serial CPU path. [Machine-readable measurements](result.json)

## What the original fits measured

RAL job 415196 ran the same seed-2 target twice in its ordinary serial arm, with 400 live points and four neural networks. The two final checkpoints were byte-identical. Each trace recorded eleven calls to `NautilusBound.compute`. Subtracting directly nested events, including neural training and prediction, reconciles exactly to the recorded exclusive time for every call. The traces do not time the inner geometry operations individually. [Parent CPU protocol](https://github.com/PyAutoLabs/autofit_profiling/blob/04b33ab7be4d976c43beaaaedf09892f96703974/experiments/nautilus_spawn_training_pool/parent_cpu/PROTOCOL.md)

| Bound builds | Exclusive geometry on RAL, first / reverse control | Saved cube dimensions per outer mixture | Ellipsoid fits in local reconstruction |
|---|---:|---:|---:|
| 1 | 0.822 / 1.520 s | 10 of 10 | 20 |
| 2 | 0.702 / 1.384 s | 9 of 10 | 19 |
| 3–11 combined | 3.301 / 6.035 s | 0 of 10 in each | 2 per build |
| **All eleven** | **4.825 / 8.939 s** | | **57** |

Each outer union contained one mixture with 400 saved points. Builds 1 and 2 took 31.6% and 32.5% of all exclusive bound-construction time on RAL. Builds 3–11 still account for most of that total. The native mixture constructor repeatedly fits candidate ellipsoids when choosing which coordinates to draw from the unit cube. The recorded input dimensionality of each ellipsoid fit is in the receipt.

## What the local replay measured

The replay calls only `UnitCubeEllipsoidMixture.compute` on the saved point arrays, outside the sampler. All eleven reconstructed masks and membership vectors agreed exactly with the checkpoints. The centers, geometry matrices, Cholesky factors, inverse factors and log volumes matched at relative tolerance `1e-8` and absolute tolerance `1e-10`, where present. Every call left its random generator state unchanged. The existing posterior and model checks remain evidence from the original fits.

After an untimed preflight excluded from the result, the replay ran a forward and a reverse pass. Each measured bound had one plain call and one call with a timer around `minimum_volume_enclosing_ellipsoid`. The order within each pair alternated and flipped on the reverse pass. Input loading, geometry checks and output writing were outside the timers.

| Local pass | Plain constructor | Probed constructor | Inside minimum-volume ellipsoid routine | Share of probed constructor |
|---|---:|---:|---:|---:|
| Forward | 13.313 s | 8.371 s | 8.149 s | 97.3% |
| Reverse | 9.303 s | 7.759 s | 7.557 s | 97.4% |

The laptop was shared during measurement, including a concurrent readiness refresh. The spread between plain and probed calls prevents a precise estimate of timer overhead. The local NumPy/SciPy versions and machine differ from RAL. The percentages describe time inside the **local outer mixture constructor**. The original sampler also constructs the initial ellipsoid union, attempts a split, selects training points and may test the outer volume. A whole-bound timer around the enclosing-ellipsoid calls during a matched native fit would establish their share of that wider cost before any kernel change.

## Reproduce

Use the original job 415196 `output/parent-cpu-diagnostics-ral-01/pilot/` directory from the retained parent CPU archive. The archive SHA-256 is `8ff46ff18d6a9f982b192eea7a67b8d0ad09089d53cd88c7a18c1c4034eb8be6`. The script verifies both `child.json` files and both final checkpoints against frozen hashes. The raw archive is held with the project evidence. Only a compact receipt is committed here.

Use Python 3.10.12 on Linux with NumPy 1.26.4, SciPy 1.15.3, h5py 3.10.0, threadpoolctl 3.6.0 and Nautilus 1.0.5 to reproduce this local configuration. The source-package hash matches the RAL receipt. The script sets native thread pools to one and verifies the pinned Nautilus Python files before loading the geometry module. An extracted wheel's `nautilus/` directory may also be passed with `--nautilus-source`.

```sh
python experiments/nautilus_bound_geometry/replay.py \
    /path/to/parent-cpu-diagnostics-ral-01/pilot geometry-replay.json
```

Choose a fresh output filename. Failure of an input, timing-accounting or geometry check stops the replay before it publishes a result. This is an off-board engineering diagnostic within [issue #18](https://github.com/PyAutoLabs/autofit_profiling/issues/18).
