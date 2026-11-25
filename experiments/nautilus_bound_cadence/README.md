# Nautilus bound-update cadence on complete PyAutoFit fits

This local pilot compares the native interval of 200 with an interval of 400 at 200 live points and four neural networks. Each row is a pair of complete fits. The two fixtures share the same simulated 100-pixel Gaussian-profile data. The second uses a signed centre and evaluates the profile at its absolute value. Its positive and negative modes have exactly equal prior and posterior mass and its evidence equals that of the first fixture. [Registered comparison and decision rules](https://github.com/PyAutoLabs/autofit_profiling/issues/26).

## Paired results

Values in paired columns run from default to doubled interval. A positive wall reduction means the doubled interval finished sooner. Peak columns give the first sampler evaluation within one log-likelihood unit of the independently fitted maximum. The symmetric rows show positive / negative peaks.

| Fixture | Seed | Wall (s) | Reduction | Bound builds | Evaluations | Default peak calls | Doubled peak calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| profile | 1 | 60.64 → 39.58 | 34.7% | 26 → 18 | 7,500 → 9,300 | 4278 | 6020 |
| profile | 2 | 53.55 → 39.89 | 25.5% | 28 → 19 | 7,400 → 9,200 | 4043 | 5608 |
| symmetric | 1 | 98.98 → 68.11 | 31.2% | 25 → 18 | 7,700 → 9,300 | 4737 / 4740 | 6235 / 5561 |
| symmetric | 2 | 94.12 → 63.61 | 32.4% | 26 → 18 | 7,500 → 9,300 | 5136 / 4427 | 6047 / 5486 |

| Fixture | Seed | Arm | Training (s) | Geometry (s) | Log evidence | ESS | Positive-mode mass |
|---|---:|---|---:|---:|---:|---:|---:|
| profile | 1 | default | 22.76 | 10.38 | -57.5115 | 1301 | — |
| profile | 1 | doubled | 16.66 | 8.13 | -57.4934 | 1451 | — |
| profile | 2 | doubled | 15.58 | 7.96 | -57.4815 | 1570 | — |
| profile | 2 | default | 22.69 | 10.11 | -57.4743 | 1381 | — |
| symmetric | 1 | doubled | 35.53 | 10.36 | -57.4991 | 1563 | 0.490 |
| symmetric | 1 | default | 51.47 | 14.94 | -57.4734 | 1416 | 0.520 |
| symmetric | 2 | default | 46.75 | 14.00 | -57.4966 | 1405 | 0.496 |
| symmetric | 2 | doubled | 33.96 | 9.70 | -57.4785 | 1650 | 0.501 |

## Decision

The doubled interval finished sooner in 4 of four pairs. In 3 pairs at least one peak took over 25% more evaluations to reach. The registered pilot decision is **keep the native default**. All eight runs recovered the expected peaks and met the evidence tolerance. Both symmetric seeds met the mode-mass gate in both arms. Every pair met the folded posterior-mean gate.

For the Gaussian profile the doubled interval required 1,800 extra evaluations for each seed. Dividing the paired wall saving by those extra calls gives about 11.7 ms and 7.6 ms per additional evaluation respectively. In a simple serial cost model, adding that much work to every likelihood call would use up the observed savings. This extrapolation assumes the sampling path and other costs stay fixed. Lensing fits and batched GPU execution need their own measurements.

## Method and reproduction

Each case ran in a fresh local WSL process on an Intel Core i9-10885H with one sampler core, one numerical thread, NumPy, batch 100, native checkpointing, normal PyAutoFit output and target ESS 500. Seeds 1 and 2 were paired within each fixture. The runner reversed arm order between seeds and between fixtures. Wall is the elapsed time of the full search.fit call. Training and geometry are timed by the existing exclusive component wrappers. The count and discovery metrics cover sampler evaluations only. Reaching the threshold counts the position within the batch. Elapsed seconds to the threshold were outside the timing protocol.

Least squares gave a peak log likelihood of −38.17109. Tensor quadrature over all three uniform prior ranges gave log evidence −57.45410 at orders 32 and 48. The signed prior folds exactly onto the original positive prior. Posterior checks use normalized sample weights, sign mass and marginal means after folding the centre. The two seeds form a bounded engineering pilot. The symmetric control has two equal peaks and leaves recovery of faint secondary modes on observational data open.

The eight fits ran with [driver revision b8c5549](https://github.com/PyAutoLabs/autofit_profiling/blob/b8c5549171304ef0d333808c880cc82ee36cd2eb/experiments/nautilus_bound_cadence/compare.py). The runner requires the PyAutoFit and PyAutoNerves checkouts and dependency versions in [the receipt](result.json) on the Python import path. From the producer checkout run the following command with a fresh output path. The supervisor caps each child at 300 seconds and writes the ordered results, gates and individual fit logs under that path.

```sh
python experiments/nautilus_bound_cadence/compare.py output/bound_cadence_pilot
```

The comparison is an off-board experiment under the [canonical profiling producer task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md). [Application breakdown](https://github.com/PyAutoLabs/autofit_profiling/blob/e64e3a7e5c8f9d9b3e4aba3fded2c0ea04b09dbf/results/gaussian_x1/nautilus/breakdown.json).
