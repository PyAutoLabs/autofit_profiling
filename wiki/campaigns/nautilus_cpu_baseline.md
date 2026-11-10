# Nautilus CPU baseline — phase 0a

Measure the same analytic Gaussian through direct Nautilus and PyAutoFit, then inspect
where parent and worker time goes. This is an exploratory CPU baseline for
[issue #8](https://github.com/PyAutoLabs/autofit_profiling/issues/8). The dimension and
real-target campaign is phase 0b; these local runs do not complete that campaign.

## Local pilot — 10 November 2026

[Interactive report](../../results/analytic_gaussian/nautilus/report.html),
[all attempt records](../../results/analytic_gaussian/nautilus/pilot_runs.json),
[paired summaries](../../results/analytic_gaussian/nautilus/summary.json).

The selected baseline contains **52 successful runs**: the 48 three-seed cases plus
four variable-cost checks. All 26 trace/control pairs have identical posterior hashes
and likelihood counts. Direct and PyAutoFit also match the native run-call schedule
and posterior for every matched case. All 26 detailed timelines reconcile.

The desktop shutdown interrupted one attempt. Ten earlier successful runs used an
older PyAutoFit revision and are explicitly excluded, not deleted. During the
shutdown another session advanced the shared PyAutoFit checkout, including plotting
changes. Provenance caught this; affected cases were repeated. The final baseline
pins PyAutoFit `bdb7af0e9c339ad07850679fd6ac168aacc7d18c`, PyAutoNerves
`9b204c90c2ce05c5bb0cf3e9800ed13f005c6c56` and Nautilus 1.0.5. Read-only detached
library checkouts now prevent shared-main changes during this task. The measurement
source is commit `7110619`; subsequent changes tighten reporting/accounting checks.
The report compares revision/content identities, not checkout locations.

Hardware: Intel Core i9-10885H, WSL Linux exposing four cores/eight logical CPUs;
Linux filesystem, NumPy float64, native math-library thread counts set to one.
Every run carries package versions, source hashes, actual affinity and host load.
The host was not reserved exclusively, including other local agent work and a Heart
refresh. Timing scatter is substantial; the following are exploratory observations.

Control timings (seconds; three independent seeds, median [minimum–maximum]):

| Execution | Workers | Fixed cost | Parameter-correlated cost |
|---|---:|---:|---:|
| Direct Nautilus | 1 | 8.74 [7.46–10.94] | 8.53 [7.45–9.62] |
| Direct Nautilus | 2 | 10.12 [9.14–11.79] | 8.44 [7.03–10.74] |
| PyAutoFit | 1 | 12.47 [10.91–12.72] | 10.26 [9.62–20.42] |
| PyAutoFit | 2 | 11.81 [11.72–13.57] | 11.66 [10.51–14.60] |

The median observed traced/control wall ratio over the 12 main pairs per execution
path is 0.948 direct (range 0.662–1.144) and 0.955 PyAutoFit (0.679–2.114). These noisy
ratios do **not** show negative instrumentation overhead or a reliable speedup.
No correction is subtracted. A quiet-host repeat is needed to bound small overheads.
The slow traced outlier also shows inflated neural, geometry and checkpoint times;
this pilot cannot attribute its extra time uniquely to instrumentation or contention.

Median component shares from the three fixed-cost, one-worker traced runs:

| Component | Direct | PyAutoFit |
|---|---:|---:|
| Neural training | 51.9% | 37.4% |
| Bound geometry | 32.4% | 24.2% |
| Visualization | not applicable | 18.0% |
| Native checkpoint I/O | 11.6% | 8.7% |
| Prior conversion | 0.03% | 2.3% |
| Likelihood evaluation/dispatch/wait | 0.5% | 1.1% |

Shares are medians of each run's component/wall ratio. Exported component seconds are
independent medians and need not sum exactly to the median wall. Worker service times
overlap and are not added to this table. On this cheap target the evidence does not
support a useful likelihood-pool speedup: most time is in the serial parent.

The variable-cost witness is present: for the direct two-worker path, median p95
service time grows from 0.026 ms (fixed) to 0.109 ms (correlated), and median batch
worker-tail wait grows from 0.032 ms to 0.069 ms. PyAutoFit's corresponding values are
0.082→0.143 ms and 0.115→0.142 ms. These are small absolute tails, not a promised
end-to-end scheduling gain.

All selected cases use 2,500–2,600 sampler likelihood evaluations. ESS is
1,257–1,335 (the stopping rule also requires exploration completion), logZ minus
analytic truth ranges from −0.0618 to −0.0430, and the largest absolute posterior-mean
error is 0.0321. Three seeds and one Gaussian do not establish general scientific
accuracy, mode recovery, dimensional robustness or calibrated evidence uncertainty.

**Next bounded optimization proposal:** investigate batching native checkpoint
updates while keeping each on-disk snapshot internally consistent, with mandatory
flushes at bound transitions and clean exit. On these measured shares, even removing
all checkpoint time would cap ideal speedup at about **1.13x direct / 1.10x PyAutoFit**;
a real change will achieve less. Preserve neural bounds, all submitted points and
resume behavior, and measure recoverable lost work under injected interruption.
Changing PyAutoFit's full-output interval alone does not suppress Nautilus's internal
per-batch shell writes. This is a proposal, not an implemented speedup. Neural training
and geometry remain the larger upstream targets; a safe dedicated sampler pool is
still a separately reviewed phase-2 experiment.

Runtime validation passed target/cost invariance, worker-death detection with and
without tracing, timeout recording, and interrupted/resumed searches for both
execution paths with one/two workers. Resumed posteriors match uninterrupted ones.
Receipts are included with the result records. Larger dimensions and real targets
remain **phase 0b, not run**.

## Run a case

Use Python 3.12+ with the workspace PyAutoFit/PyAutoNerves environment, Nautilus 1.0.5,
scikit-learn, NumPy, SciPy and h5py. Activate the task/workspace environment first,
then run from this repository:

```bash
source activate.sh
python scripts/analytic_gaussian/nautilus/cpu_baseline.py \
  --adapter pyautofit --workers 2 --seed 0 --cost-profile correlated \
  --trace on --output output/cpu-baseline-example
```

Each command launches a fresh process, sets native thread limits to one before
imports, and refuses an existing output directory. Choose `--adapter direct`,
`--workers 1` or `--trace off` for the paired cases. `--trace off` retains only a
common coarse monitor (constructor checks, sampler.run timing and checkpoint count);
it disables stage and per-proposal instrumentation.

The primary configuration is five nonlinear dimensions, 200 live points, four neural
networks, batches of 100, ESS target 500 and `f_live=0.01`. Both paths use checkpoints,
explicit resolved constructor/stopping settings, an output update interval of one
million likelihood calls, and serial sampler work. Two-worker cases use persistent
likelihood workers and PyAutoFit's existing worker-initializer/watchdog wrapper.
These are controlled settings; they are not a comparison of package defaults.

The direct likelihood evaluates the analytic density without constructing PyAutoFit
model instances. The wrapped run retains search.fit output and result work. Its
parameter order and prior transform are checked against the direct target. Both
sample independent standard-normal priors and the same normalized correlated Gaussian
likelihood. Evidence and posterior moments are known analytically, with no finite-prior
truncation approximation.

## Pilot and verification

```bash
python -m pytest -q scripts/misc/test
python scripts/analytic_gaussian/nautilus/check_runtime.py output/cpu-runtime-checks
python scripts/analytic_gaussian/nautilus/pilot.py --output output/cpu-pilot
python scripts/analytic_gaussian/nautilus/report.py output/cpu-pilot \
  --export results/analytic_gaussian/nautilus
python scripts/misc/tooling/build_catalogue.py
```

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1` before the standalone runtime
checks. The case/pilot launcher sets them itself. Runtime checks require the full
stack; the regular tooling tests remain stdlib-only apart from pytest.

The pilot runs 48 cases: two execution paths, one/two workers, fixed/parameter-correlated
CPU work, three seeds, with/without detailed tracing. Four additional cases exercise
independent variable-cost work. Cases run sequentially, with trace/control order
alternated across seeds. The default budget is 30 minutes, with 120 seconds per case.
Failures stop the pilot and unrun cases remain in `pilot.json`; a missing witness is
not a successful result. Calibration/smoke results are separate from the pilot.

CPU work uses a fixed number of scalar operations. Parameter-correlated cost is 16x
larger for points with first coordinate above 0.5. The independent variable-cost
case uses a deterministic point hash, separate from sampler/training RNG: duplicate
points deliberately have the same cost. Every cost policy returns the identical
numerical likelihood. There are no sleeps or first-finished subsets.

## What the outputs mean

- `request.json`, `manifest.json`, `result.json` and `run.log` preserve the requested
  case, actual configuration and terminal outcome. Exact source revisions, code hashes,
  package versions, CPU allocation and native thread settings are recorded. Installed
  Nautilus source is hashed; no upstream commit is inferred from a version label.
- `wall_s` covers the complete adapter operation, including pool setup and result work.
  `process_wall_s` additionally includes cold imports and provenance/report extraction.
  `sampler_run_s` isolates native run calls. The manifest records run-call limits and
  checkpoint write counts so a cadence difference is visible.
- Parent component times are exclusive. Worker service/CPU times overlap and must not
  be added to parent wall. Worker occupancy is relative to likelihood-evaluation
  windows, not the whole fit. Pool-construction timing does not measure the completion
  of every worker initializer. Parent and child peak RSS are process maxima, not total
  simultaneous memory.
- Worker events return with existing chunk results. Their serialization is included
  in the measured tracing overhead. Queue wait includes waiting inside a map chunk;
  separate serialization and dequeue times are unavailable. Detailed traces have a
  200,000-event cap. Hard process termination can lose buffered events, which is a
  missing witness, never complete accounting.
- Proposal IDs are `(batch, row)` within one fresh run. Submitted/completed IDs and
  native `n_like` must reconcile. Counts refer to sampler calls; PyAutoFit diagnostic
  evaluations outside the sampler are excluded. Trace/control posterior hashes and likelihood counts
  must match before the report accepts the comparison. A checkpoint-resume check
  compares retained native state and the final posterior with an uninterrupted run.
- Three seeds support an engineering pilot, not a scored statistical acceptance
  claim. Evidence error and posterior moments are reported against analytic truth.
  Calibrated logZ uncertainty and proposal rejection rates are not provided by this
  harness; they remain missing diagnostics for the wider campaign.

`report.html` includes a table and actual worker service intervals from the first
eight batches of the correlated-cost cases. Compact run records retain failures and
missing cases; only successful measurements are exported to `profiling-summary@2`.
The original gaussian_x1 evidence is preserved.

## Phase 0b handoff

Before campaign execution, resolve the canonical task with PyAutoPulse; link it to
this issue and the epic. Coordinate full-search validation with PyAutoInsight. Do not
resume paused campaigns or bypass accepted fixture/parity blockers.

Use `--dimension 10`, then 30/60/100/200 where affordable, keeping sampling accuracy
criteria and cost profiles explicit. `n_live` must exceed the dimension (for example,
`--dimension 200 --n-live 400`); parameter changes are named experiments, not an
unchanged-setting speed comparison. Source/linear inversion coefficients are not
sampler dimensions. Add the accepted lens ladder only after its domain task and
runtime witnesses are in place. No GPU or HPC jobs are submitted by these commands.

For a new campaign, pin dependencies in read-only checkouts before starting. The pilot
now halts if source identities differ between cases; a report also rejects mixed pins
or differing within-group settings. Superseded successful rows remain in the evidence
with `included_in_analysis: false`.
