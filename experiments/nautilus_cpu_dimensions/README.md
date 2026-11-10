# Nautilus experimental dimension pilot

This is an off-board engineering experiment for [issue #10](https://github.com/PyAutoLabs/autofit_profiling/issues/10), following the [phase-0a baseline](../../wiki/campaigns/nautilus_cpu_baseline.md). It is not a registered Pulse campaign or a scored inference result. Nothing under `experiments/` is ingested by `build_catalogue.py`. Promotion requires a later human decision.

The frozen [protocol](protocol.json) has 36 cases: dimensions 5/10/30 × three seeds × direct/PyAutoFit × tracing off/on. All use 400 live points, ESS target 500, batch 100, four neural networks and one worker with native threads capped at one. Checkpoint and update policies match explicitly. Dimension 5 with 400 live points is a new bridge cell, not a speed comparison with phase 0a's 200-live-point cell.

## Running

Use an isolated environment with the dependencies pinned in protocol.json and activate the detached library paths. From the repo root:

```bash
python scripts/analytic_gaussian/nautilus/campaign.py identity --output output/dimensions-identity.json
python scripts/analytic_gaussian/nautilus/campaign.py plan --identity output/dimensions-identity.json --output output/dimensions-plan.json
python scripts/analytic_gaussian/nautilus/campaign.py run --plan output/dimensions-plan.json --output output/dimensions-run
python scripts/analytic_gaussian/nautilus/experiment_report.py output/dimensions-run --output experiments/nautilus_cpu_dimensions/evidence
```

`plan` uses only the standard library. `identity` imports the runtime, computes analytic truth digests and captures source/content, package, Python, host and affinity identity. Commit measurement code before this step. Pass `identity --preflight-seconds N` to charge preceding timed runtime checks to the 90-minute execution budget. A plan is immutable: rerun the same `run` command to reconcile an interrupted driver, never regenerate the matrix on resume.

Each attempt has a unique directory. Running attempts are durably recorded before launch. Resume refuses a live supervisor, retains interrupted attempts and charges their full timeout conservatively. A previously failed/capped child halts instead of triggering an automatic retry. A selected success is revalidated before resume. The driver holds a nonblocking file lock. It does not run outside the invoking session or schedule future work.

Per-case limit is 120 seconds; total admitted execution plus recorded preflight is at most 90 minutes. The Linux supervisor samples process-group RSS every 50 ms and kills the group at 4 GiB. This is a sampled operational guard, not a kernel-enforced memory quota: sub-interval spikes can be missed and shared pages may be counted more than once. A cap is `memory_cap`, not an inferred OS OOM. At least twice the requested cap must be available. The driver refuses new cases if one-minute load exceeds max(1, affinity CPUs / 4); this observable quiet-host check is not an exclusive scheduler reservation. Do not interfere with other work; noisy or unavailable hosts leave unrun evidence rather than a claimed timing result.

Timeout, memory cap, source mismatch, changed target/configuration or accounting failure stops escalation. Every planned case remains in coverage, including unrun, failed, interrupted and missing results. No failed seed is silently replaced. Traced results require digest-checked timelines and reconciled counts. Missing/partial pairs do not produce overhead ratios; duplicate selected identities fail closed. The historical phase-0a report rejects multiple dimensions rather than reusing its old catalogue IDs; use this experiment exporter for the new matrix.

## Interpretation

Report full adapter wall and subprocess wall separately. Worker times overlap; do not sum them into parent wall. Ratios include host noise and are not corrected. Standardized mean error divides by analytic posterior marginal standard deviation; covariance error is the relative Frobenius norm. ESS and logZ error are diagnostics, not a calibrated scientific acceptance test at three seeds. Native logZ uncertainty and proposal-rejection counts remain unavailable.

Full phase 0b remains incomplete: 60/100/200-dimensional probes, harder targets, worker scaling and parity-admitted lens fixtures need later bounded work. Historical Insight correlated_nd failures concern another family, prior and source pin. Neither these measurements nor their failures are pooled with that campaign. No optimization, release or merge is authorized by running this experiment.
