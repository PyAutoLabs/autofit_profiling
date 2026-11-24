# 30D conditional evidence check

All three planned resampling streams on the seed-0 learned bounds reached fresh-sample ESS 10,000. The analytic log evidence is -36.767688153. The original exploration-inclusive estimate had signed error -0.908090. The three completed fresh-only errors range from -0.048116 to -0.027755. Their mean is -0.034966 and their sample standard deviation is 0.011406. This spread describes repetitions on the same exploration. [Phase issue](https://github.com/PyAutoLabs/autofit_profiling/issues/30).

| Resampling stream | Log evidence | Signed error | Relative evidence error | Fresh ESS | Fresh likelihood calls | Calls in this follow-up | Min fresh count per shell | Max standardized mean error | Relative covariance error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| saved | -36.796714 | -0.029026 | -2.86% | 10,001.1 | 980,300 | 861,400 | 200 | 0.0201 | 0.0384 |
| seed1 | -36.815804 | -0.048116 | -4.70% | 10,000.7 | 636,100 | 636,100 | 1000 | 0.0282 | 0.0433 |
| seed2 | -36.795443 | -0.027755 | -2.74% | 10,001.7 | 835,600 | 835,600 | 900 | 0.0182 | 0.0445 |

Relative evidence error is `exp(estimated logZ − analytic logZ) − 1`. The mean diagnostic is the largest component error in analytic posterior standard deviations. The covariance diagnostic is the Frobenius norm of the error divided by that of the analytic covariance.

## What changed

The original seed-0 fit finished exploration with 34,100 likelihood calls and no subsequently sampled points. The earlier fresh-only attempt timed out at 120 seconds. Its last persisted copy had 118,900 fresh calls, ESS 2,633.6 and log-evidence error −0.049803. For the saved stream this follow-up copied that file, restored the exclusion flag, matched its archived ordered posterior hash and then wrote the corrected mode to the copy with native Nautilus. It resumed at 153,000 total calls. The other two streams used copies of the same original exploration and changed the shared RNG state before drawing fresh samples. The source checkpoints retained their original SHA-256 hashes. [Earlier protocol and timeout](../nautilus_30d_diagnostics/README.md).

Every run used Nautilus 1.0.5, 400 live points, native bound cadence 400, batch size 100 and four neural networks. Exploration stayed closed. All learned bounds and their exploration cutoffs were retained. Each run stopped at ESS ≥10,000 with at least 100 fresh samples per shell. Each final checkpoint reopened directly with native Nautilus, reproducing the full ordered posterior hash, log evidence, ESS, moments and counts exactly. The independent reader checked the analytic answer in a covariance eigenbasis, reconstructed log evidence and ESS from the post-exploration likelihood rows and stored shell volumes, and re-evaluated one fresh sample in each shell. [Numerical checks](verification.json) · [Per-stream results and supervision](result.json).

## Interpretation

All three fresh-only estimates were low by 2.74–4.70% on the evidence scale. The original exploration-inclusive estimate was low by 59.67%. The three conditional runs specified by phase 0b.2 are complete. The comparison changes both sample selection and sample count. It cannot assign the original discrepancy uniquely to exploration reuse. Independent full-search seeds are needed to quantify variability from learning the bounds and to assess evidence uncertainty calibration. The protocol specified a stopping condition and an analytic comparison. It set no numerical accuracy acceptance band. Wider phase 0b remains open.

## Reproduction

Use the PyAutoFit and PyAutoNerves revisions, Nautilus source hash and scientific package versions in the receipt. Place the two source checkouts on PYTHONPATH and include this repository's scripts/misc. Use the original direct 30D `case-719639ab504cacbebb25-attempt-001/checkpoint.hdf5` and the retained `evidence-d30-rsaved/checkpoint.hdf5` from the earlier timeout. Their exact checksums are verified before and after the run. Reproducing the saved stream requires the retained partial binary. Local raw checkpoints are outside git.

```sh
python experiments/nautilus_30d_evidence_check/check.py ORIGINAL.hdf5 PARTIAL.hdf5 FRESH_OUTPUT_DIRECTORY
python experiments/nautilus_30d_evidence_check/verify.py FRESH_OUTPUT_DIRECTORY
```

The three single-process children ran concurrently on the local CPU with one numerical thread each, a 1,800-second cap and 4 GiB sampled process-group RSS limit per child. Elapsed times are supervision records for this accuracy check. [Domain profiling task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
