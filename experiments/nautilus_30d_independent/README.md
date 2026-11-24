# 30D evidence with independently learned bounds

At the three terminal checkpoints, seed 1 was above the analytic evidence and seeds 2 and 3 were below it. **None reached the registered fresh-only ESS of 10,000** within 2,400 seconds per seed. Rows marked partial describe the last verified sample set for that seed. They remain outside the endpoint comparison. [Phase protocol](https://github.com/PyAutoLabs/autofit_profiling/issues/32).

| Exploration seed | Exploration logZ error | Fresh samples | Fresh ESS | Fresh logZ | Fresh logZ error | Evidence error | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -0.892372 | 1,074,100 | 8,147.4 | -36.763453 | +0.004235 | +0.42% | Partial, terminal |
| 2 | -0.955682 | 1,090,100 | 9,904.2 | -36.785425 | -0.017737 | -1.76% | Partial, terminal |
| 3 | -0.888639 | 936,500 | 6,920.3 | -36.771196 | -0.003508 | -0.35% | Partial, terminal |

Analytic log evidence is -36.767688153. Signed log error is estimate minus truth. Relative evidence error is exp(log error) − 1. Each partial run stopped at its wall limit; no fit was retried. Its effective sample size and estimate can change with further sampling. The checkpoints belong to different stages of convergence. They supply no calibrated bias estimate.

## What repeated across explorations

Each seed began empty with a fresh RNG and no input checkpoint. All three learned distinct outer ellipsoid centres and shape matrices, also distinct from those in the historical seed-0 checkpoint. Their exploration-only errors all lay between −0.956 and −0.888 in log evidence. The original seed-0 exploration had error −0.908. Fresh sampling substantially narrowed that discrepancy in all three new runs. The earlier three fully sampled continuations gave errors between −0.0481 and −0.0278 on a single shared set of bounds. The comparison above uses the independently learned bounds once each. [Earlier conditional results](../nautilus_30d_evidence_check/README.md).

## Verification and reproduction

Each native checkpoint was read without advancing the RNG. The audit verified the exploration-exclusion flag, learned geometry, position/likelihood row counts, per-shell ESS and mean likelihood, total calls and fresh counts. It recomputed log evidence and ESS directly from fresh likelihood rows and stored shell volumes, compared both with a native reopening, and verified the inspected file hash stayed unchanged. The audit receipt identifies whether each estimate used the terminal file or an earlier checked copy. [Checkpoint hashes and estimates](checkpoints.json) · [Supervisor outcomes and runtime pins](result.json).

The target, prior, Nautilus 1.0.5, 400 live points, four networks, native update interval 400, batch 100 and f_live 0.01 match the earlier fixture. The first native run stops at the exploration transition. The second continues the same RNG with fresh-only samples, requiring ESS ≥10,000 and at least 100 fresh samples per shell. All three children used one numerical thread each and a 4 GiB RSS cap on the shared laptop.

Use the library pins in the receipt and the input checkpoint from the original direct 30D case, case-719639ab504cacbebb25-attempt-001. The only use of seed 0 is a read-only geometry comparison. The supervisor writes separate checkpoints and exit diagnostics for each seed. A nonzero exit with an incomplete receipt indicates that the endpoint or verification was missed. The following commands reproduce the run and audit a closed checkpoint.

```sh
python experiments/nautilus_30d_independent/run.py FRESH_OUTPUT_DIRECTORY --reference SEED0.hdf5
python experiments/nautilus_30d_independent/read_saved.py FRESH_OUTPUT_DIRECTORY/seed1/checkpoint.hdf5 FRESH_OUTPUT_DIRECTORY/seed1/exploration.json -36.76768815337412
```

This local phase 0b evidence remains outside the Pulse dashboard. [Canonical profiling task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
