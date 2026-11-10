# Nautilus CPU baseline — phase 0a

Measure the same analytic Gaussian through direct Nautilus and PyAutoFit, then inspect
where parent and worker time goes. This is an exploratory CPU baseline for
[issue #8](https://github.com/PyAutoLabs/autofit_profiling/issues/8). The dimension and
real-target campaign is phase 0b; these local runs do not complete that campaign.

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
  windows, not the whole fit. Parent and child peak RSS are process maxima, not total
  simultaneous memory.
- Worker events return with existing chunk results. Their serialization is included
  in the measured tracing overhead. Queue wait includes waiting inside a map chunk;
  separate serialization and dequeue times are unavailable. Detailed traces have a
  200,000-event cap. Hard process termination can lose buffered events, which is a
  missing witness, never complete accounting.
- Proposal IDs are `(batch, row)` within one fresh run. Submitted/completed IDs and
  native `n_like` must reconcile. Trace/control posterior hashes and likelihood counts
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
