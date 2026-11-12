# Parent CPU slowdown after pooled training

Approved follow-up to the completed 10D diagnostic, outside Pulse. Merge stays human.
[Approved plan](https://github.com/PyAutoLabs/PyAutoMind/blob/279ec2cd9/draft/feature/autofit/nautilus_parent_cpu_slowdown.md).

The retained job415180 receipts show similar parent likelihood CPU before training
(0.08627s serial, 0.08738s pooled for eight batches). After the last training call,
168 batches use 1.81169s versus 3.39046s, averaged over two fits per mode. This
localizes the slowdown but does not establish its cause.

## Fixed comparison

Protocol `nautilus-parent-cpu-diagnostics@1`: twelve fresh 10D/seed2 fits in order
**A B C D E F F E D C B A**. Two observations per arm, balanced by position only.

| Arm | Native training | Parent placement | Intervention |
|---|---|---|---|
| A | Serial | Four allocated CPUs | Baseline |
| B | Two persistent spawned workers | Shared four CPUs | Pooled baseline |
| C | Same as B | Shared four CPUs | Close/reap workers after native training call11 |
| D | Serial | Four allocated CPUs | ForkingPickler roundtrip of each returned network |
| E | Serial | Lowest allocated CPU | Parent placement control |
| F | Two spawned workers on next two CPUs | Same CPU as E | Disjoint parent/worker placement |

C closes after the result and native training timer finish, in a separate phase.
An unexpected twelfth training call fails; no restart or serial fallback. D uses
multiprocessing's effective default pickle protocol, verifies exact values and
records array layout/alignment. E/F narrow all existing parent threads after worker
startup and restore surviving thread masks in finally. No CPU outside the allocation
is used. Every arm has the same observation policy.

## Science, admission and limits

The original runtime pins, four native networks/seeds0–3, numerical thread limits,
likelihood, sampling configuration, checkpointing and fit output are unchanged.
Only the search corner is disabled equally, as in the reference. No installed
library edits, synthetic likelihood work, extra capture, training replay or warmup.
Exact reference: `../diagnostics_10d/reference.json`, SHA256
`5d424a9c4c3344330d6fefbe0a391b0dc5b121cb45bc2af468af79ff1e99ede0`.
Require identical posterior, model, target, checkpoint and phase counts before timing
interpretation; never loosen tolerances or replace the reference.

One exclusive `ral` job on `euclid-ral-compute-1`, four bound physical cores, 8GiB,
35 minutes (the reservation covers the whole node). At most twelve supervised
case process groups, including workers: 120s and 4096MiB each; 1800s aggregate
including admission/verification, with 180s bootstrap and 120s cleanup reserve.
Stop on first failure. No retry, requeue, replacement job or automatic merge.
Fresh `parent-cpu-diagnostics-ral-01` claim and dedicated environment; retain all
old and new outputs. Clean up only the newly owned temporary runtime cache.

## Measurements and interpretation

Nested exclusive wall, parent-process CPU and main-thread CPU reconcile exactly.
Coarse phase boundaries record placement, context switches, faults and per-thread
CPU; there are no per-likelihood or per-prediction procfs observations. GC callbacks
are passive. Runtime, native library/thread and readable cgroup counters are saved
before/after the fit and around training. Worker CPU snapshots separate training
from intervals between training calls and the long post-training tail. Missing
counters are explicit; system/cgroup counters may include other activity.

Observation snapshots, model layout/serialization and shutdown have separate
phases. Instrumentation and passive GC callback overhead are still included in
workflow cost; observation allocations can affect GC cadence. These are diagnostic
fits, not an uninstrumented performance claim. Parent process CPU includes its
management threads; worker CPU remains separate. Do not sum overlapping wall times.

C versus B tests worker lifetime; D versus A tests serialization; E versus A tests
parent pinning; F versus E tests pooled training with disjoint placement. F versus B
changes both placement components. Closing workers may change scheduling as well as
worker activity. Two observations per arm cannot establish a general speed ranking.
If inconclusive, report it and propose no default change.

Candidate, observation and phase receipts are retained before exact parity is
accepted. Scientific outputs stay on witnessed shared storage; runtime/source are
node-local. Tests, zero-science structural smoke, lint/format/shell/catalogue checks,
author review, exact-head CI, current readiness and a fresh runtime restore precede
the single submission. Earlier 18-case and 17-case protocols remain unchanged.
