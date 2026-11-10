# Spawn-based Nautilus training experiment

This is the approved bounded experiment for [issue16](https://github.com/PyAutoLabs/autofit_profiling/issues/16), outside Pulse. No production API, installed-library source, default, catalogue or release changes. Merge remains human. The branch depends on unmerged diagnostic PR13; it does not require the plotting/checkpoint candidate library APIs.

The candidate supplies two persistent **spawn** workers only to the native `NeuralNetworkEmulator.train` pool argument. All four native networks, seeds0..3, default neural settings and result order remain intact. Likelihood evaluation, geometry and sampler adaptation remain serial. Workers have one native numerical thread and import neither PyAutoFit nor JAX. The original classmethod descriptor is restored when the scope exits.

The baseline uses the same ordered measurement shim in the parent process, executing the unchanged native training function. Metrics include parent training wall, individual network wall/CPU time, estimated pickle payload sizes, pool startup and teardown, sampler.run, full fit and total workflow wall. Workflow timing starts before pool construction and ends after teardown; subprocess wall additionally includes interpreter/import/receipt overhead. Worker times overlap and must not be summed into parent critical-path time. Payload sizes exclude transport framing. Instrumentation overhead is not subtracted.

## Frozen admission and budget

[Protocol](protocol.json): maximum18 children,120s and4096MiB sampled process-group RSS per child,1800s total including preflight and between-case waits. All children are supervised process groups; failures, time/memory caps, identity/parity mismatch or leaked processes stop the attempt. No retries or overwritten attempt directories.

Before the attempt, collect three one-minute-load readings below1.0 at least30s apart, with at least8192MiB available. This pre-launch wait is recorded separately and must be recent (last reading within120s). During the attempt retain load<=max(1, affinity CPUs/4), memory>=8192MiB. Up to60s between-case admission waiting is charged to the1800s budget; it does not relax the threshold or repeat a child. The failure snapshot and missing coverage are retained.

The six correctness children precede every timing case: native serial/two-spawn MLP coefficient/prediction preflight; uninterrupted serial D5/seed0 control; training-map interruption/resume; post-checkpoint interruption/resume. Resumes use a copy of the retained checkpoint and must match the control exactly. Training-worker entry is observed before parent pause; workers can advance before the group kill. Report that observed work separately from likelihood work, not as an instantaneous or universal loss bound.

Then twelve timing children cover D5/D10 × three seeds × serial/two-spawn training in AB/BA/AB order. Every pair must preserve exact posterior arrays/order, likelihood count, ESS, shells and trained-model digests. Each saved checkpoint is reopened without evaluating a likelihood and must match the in-memory result without changing checkpoint bytes.

Use400 live points,batch100,ESS10000,four default networks,n_shell1,discard_exploration=True,no synthetic likelihood work,one likelihood worker,one numerical thread,native checkpoint policy and full-update interval1000000. Search corner plotting alone is disabled equally; other fit output remains. These settings are not pooled with historical phase0a measurements.

## Execution

Activate the isolated task runtime with pinned PyAutoFit bdb7af0, PyAutoNerves9b204c9 and Nautilus1.0.5 source digest from protocol.json. Commit clean sources before launching. Do not install into or modify the retained environment. With the approved quiet-host JSONL receipt:

```bash
python scripts/analytic_gaussian/nautilus/training_pool_benchmark.py run \
  --output output/training-pool-01 --quiet-log <quiet-host-receipt.jsonl>
```

Do not run individual scientific children independently of the frozen supervisor: they belong to the single shared budget. No monitor/subscription or process remains after the invoking chat finishes.

## Validation and results

Tooling tests, Ruff checks and unchanged catalogue validation gate publication; the budgeted preflight and recovery witnesses supply actual runtime evidence. The pilot is not yet run. No performance claim. Positive or negative end-to-end results and every failed/unrun case will be retained here; a positive result would require a separate production-integration plan.
