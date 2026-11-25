# Independent 30D evidence at matched ESS

At the registered ESS endpoint, seeds 1 and 2 gave evidence below the analytic value. Seed 3 gave evidence above it. Signed log-evidence errors ran from -0.018021 to +0.005755 against analytic log evidence -36.767688153. The continuations reached their endpoints in 49.5 minutes of shared wall time. [Phase protocol](https://github.com/PyAutoLabs/autofit_profiling/issues/34).

| Independent seed | Prior checkpoint error in evidence | Final log evidence | Final logZ error | Final error in evidence | Final fresh ESS | Additional calls |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.42% | -36.770081 | -0.002393 | -0.24% | 10,000.1 | 174,400 |
| 2 | -1.76% | -36.785710 | -0.018021 | -1.79% | 10,003.0 | 7,500 |
| 3 | -0.35% | -36.761933 | +0.005755 | +0.58% | 10,001.0 | 624,100 |

Two estimates changed sign between the original time cap and the ESS endpoint. Seed 1 moved from above to below the analytic value, and seed 3 from below to above. The mean signed log-evidence error was -0.004887, with sample standard deviation 0.012083 across three independent runs. These describe this fixture and these seeds. Relative evidence error is exp(estimated logZ − analytic logZ) − 1.

## Comparison with exploration and the shared-bound experiment

At the end of exploration the same three seeds had log-evidence errors of −0.8924, −0.9557 and −0.8886. Fresh-only sampling reduced the absolute discrepancy for every seed. The earlier 40-minute attempt ended at ESS 8,147, 9,904 and 6,920 and was kept as a separate partial result. Its files were read only. [Previous independent exploration attempt](../nautilus_30d_independent/README.md).

The historical seed-0 study gave errors of −0.0290, −0.0481 and −0.0278 across three continuations sharing a single exploration. Here each reported endpoint uses separately learned bounds and its own continuing random stream. One continuation per exploration mixes variation from learning bounds with variation in fresh sampling. The sample of three is too small to calibrate an evidence uncertainty or establish a general bias. [Conditional seed-0 comparison](../nautilus_30d_evidence_check/README.md).

## Verification

The source hashes and copied starting estimates matched PR #33. Exploration exclusion remained enabled, no new bounds were built, the per-shell exploration cutoffs remained fixed and all three original hashes matched after the run. Each native HDF5 checkpoint reopened with exact agreement on ordered posterior, evidence, ESS and moments. A separate reader verified the analytic integral by eigendecomposition, rebuilt log evidence and ESS from 3,906,700 fresh likelihood rows and stored shell volumes, and checked one fresh likelihood in each shell. [Independent verification](verification.json) · [Per-seed receipts, hashes and runtime pins](result.json).

## Reproduce

Use the PyAutoFit and PyAutoNerves checkouts and numerical package versions named in the receipt. Place both checkouts on PYTHONPATH. Pass the raw directory from PR #33 containing seed1, seed2 and seed3, along with a fresh output path. The program verifies each input checkpoint hash before advancing a copy. The exact continuation requires the retained source HDF5 files.

```sh
python experiments/nautilus_30d_matched_ess/continue_runs.py RAW_FROM_PR_33 FRESH_OUTPUT_DIRECTORY
```

The three spawned local CPU processes had one numerical thread each, native update cadence 400, four neural networks and a 4 GiB sampled RSS cap per process. The supervisor shared a single 3,600-second deadline and reserved 30 seconds for shutdown. Each seed stopped at its first native ESS ≥10,000 with at least 100 fresh samples per shell. The original timeout attempts retain their own receipts. The wider phase 0b dimension and workload campaign remains open. [Canonical profiling task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
