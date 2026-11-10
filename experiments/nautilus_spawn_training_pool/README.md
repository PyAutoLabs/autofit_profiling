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

Tooling tests, Ruff checks and unchanged catalogue validation gate publication; the budgeted preflight and recovery witnesses supply actual runtime evidence. Attempt01 at producer `c9f6d2ea5def8fc71b368402aca606d35918178b` stopped after **96.82s**: six correctness children passed, the first serial timing child failed its post-fit checkpoint verification, and eleven children remain unrun. [Compact receipt](attempt01.json). No complete performance pair and no speedup claim. Raw data (~17MiB), quiet-host receipt and logs remain in the task worktree.

The fixed training corpus produced exactly identical model coefficients/normalization and predictions with serial and two-spawn execution. Both D5/seed0 crash-resume witnesses matched the uninterrupted serial control exactly:16,300 likelihood evaluations and ESS10,020.771588733165. Training interruption observed seeds0/1 entering with an unfinished training map, checkpoint at800 completed evaluations; post-checkpoint interruption retained100. Both had zero lost completed likelihood evaluations and zero likelihood evaluations in flight at their controlled pause. Workers could advance before termination; these witnesses are not a universal loss bound. All process groups were cleaned up. Peak sampled group RSS was556.9MiB; longest child25.00s.

The first wrapped fit completed, but PyAutoFit's normal finalization had already discarded `search_internal/checkpoint.hdf5`. The harness incorrectly tried reopening that deleted resume pathname. This is a verification harness error, not evidence of sampler or pool failure; the failed case remains failed.

The correction retains a hard link immediately after successful native `Sampler.run`, before normal Fit finalization removes the original pathname. Native checkpoint writes, final archive creation and cleanup remain unchanged. Verification reopens the retained checkpoint after workflow timing and checks exact posterior/state plus unchanged bytes. Retention wall time is separately recorded and remains included in full-fit/total-workflow wall equally for both arms; sampler.run timing excludes it. Atomic replacement preserves a prior retained inode if a later retention fails. No performance timing is subtracted. Two regression tests exercise native-path deletion, inode replacement, absent source and source-path protection;145 tooling tests pass. Scientific rerun has **not** been launched: the approved stop-on-first-failure rule requires a fresh human decision for another18-child/120s/4GiB/1800s attempt, with the same quiet-host gate. Both attempts must be retained. Merge stays human; outside Pulse.

## Approved RAL execution

The user approved moving real timing to RAL and the concrete allocation amendment. No further laptop quiet-host attempt is planned. The previously approved corrected local run was never launched. [RAL protocol](protocol_ral.json) is versioned separately from the retained local protocol; all18 scientific cases and the120s/4096MiB child and1800s total limits are unchanged.

The batch wrapper requests one exclusive node on partition `ral`, four physical cores for the experiment,8GiB memory and35minutes of scheduler time. Whole-node exclusivity reserves all node CPUs; only four are exposed to the experiment. There is no array, GPU request, mail notification, automatic requeue or second attempt. Five scheduler minutes provide verification/cleanup margin, not extra scientific time.

Admission reads live `scontrol` job/node/partition/step records and `squeue` node occupancy. It checks the numeric job/step, owner, running state, partition, whole-node CPU allocation, lack of another running/suspended/completing job, TRES, restart policy and scheduler deadline. Kernel affinity and physical core topology must show one logical CPU on each of four distinct physical cores. Linux memory-cgroup ancestry must identify this job, enforce8GiB and leave at least4GiB headroom. Host free RAM and SLURM environment variables alone cannot admit a run. Both cgroup v1 and v2 conventional mount layouts are supported; unavailable fields or namespaced/nonstandard layouts stop, rather than being guessed.

The initial allocation identity is frozen into plan.json and checked around every child. Each child independently checks it. Spawned workers verify inherited affinity; the supervisor samples every child-group thread's affinity as well as RSS. All inspection overhead is charged to the scientific budget. RAL's global host load is retained as context, not used as the laptop admission threshold. Core binding does not eliminate OS or shared-filesystem variation.

The correct Slurm step fields (`NodeList`, `Nodes`, `Tasks`, `CPUs`) were grounded against [SchedMD's23.02 formatter](https://github.com/SchedMD/slurm/blob/slurm-23-02-7-1/src/api/job_step_info.c). Allocation/binding semantics follow [sbatch](https://slurm.schedmd.com/sbatch.html#OPT_exclusive) and [srun](https://slurm.schedmd.com/srun.html#OPT_cpu-bind). **The installed RAL version and actual records have not yet been inspected.** Synthetic scheduler/cgroup tests establish rejection behavior, not live RAL compatibility. Discovery is required before submission; do not count a mock fixture as a runtime witness.

### Stage after SSH authentication works

1. Read-only discovery: SLURM version, `ral` partition/node policy and CPU topology, supported binding/cgroup interfaces, storage, and runtime versions. Do not submit a probe job. Confirm that a CPU-only exclusive allocation with this binding and memory accounting is supported. If the site cannot establish the requested isolation, stop and report the exact discrepancy.
2. Create an isolated task git worktree under the RAL profiling area at this PR's validated producer commit. Prepare separate clean pinned Fit/Nerves worktrees; do not switch shared checkouts or update the shared environment. [runtime.json](runtime.json) carries the approved Python/scientific versions and full reference inventory for staging. The harness checks Python3.12.10 and the eight recorded core scientific package versions; the complete actual distribution inventory is then frozen and checked across every case. Reconcile other scientific runtime dependencies against the reference inventory during staging before submission; unrelated tooling metadata may differ and must be recorded. An incompatible scientific stack stops staging for a concrete decision.
3. In the isolated producer's ignored `output/ral-environment.sh`, write exactly five reviewed shell-quoted exports: `NAUTILUS_ROOT`, `NAUTILUS_PYTHON`, `NAUTILUS_FIT`, `NAUTILUS_NERVES`, `NAUTILUS_PRODUCER_SHA`. This file bridges hpc/sync's remote shell, which does not forward local exports. No secrets belong here. Verify paths, imports and exact source hashes, and retain its receipt. Do not install PyAuto packages over the shared stack.
4. Point the local ignored `hpc/sync.conf` at the isolated remote checkout (HPC_BASE/PROJECT_NAME), not canonical RAL main. After live checks and exact-head CI pass, call `hpc/sync submit --cpu submit_nautilus_training_pool` **once**. This existing helper submits from hpc/batch_cpu so relative log paths resolve. Preserve the returned job ID immediately. A unique output/training-pool-ral-01 directory rejects repeat execution; never remove it to retry.
5. Inspect this job with the existing job/accounting tools in the active chat. On completion, pull output/training-pool-ral-01 and this job's stdout/stderr. Preserve checkpoints and hashes, explicitly checking the generic sync exclusions when collecting raw state. Obtain final `sacct` rows for the batch and srun step; the wrapper's accounting-at-exit receipt may still say RUNNING. Verify no job processes remain. Never cancel or reorder another task's jobs.

The wrapper preserves nonzero exit status, captures CPU/filesystem and accounting snapshots, and keeps caches inside the retained task output. The corrected checkpoint hard-link retention remains active. Report only within-RAL paired timings after all gates pass; do not pool laptop and RAL measurements or claim cross-CPU bitwise parity.

### Current validation and blocker

Local validation covers190 tooling tests, including scheduler/cgroup rejection, worker/thread affinity, budget exhaustion/cleanup, version pins, unchanged scientific protocol, and batch success/failure propagation plus duplicate-execution refusal. No RAL scientific runtime evidence exists yet. SSH through the existing `euclid_jump`/`jump_finan` route still rejects authentication at the jump host. The human will restore that route. No cluster files changed and no job submitted. The approved one-job authorization remains unused; further permission is not needed merely to resume staging/submission after these gates pass. Outside Pulse; human merge.
