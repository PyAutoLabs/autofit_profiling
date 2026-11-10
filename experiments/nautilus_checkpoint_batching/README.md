# Checkpoint batching experiment

The explicitly approved corrected attempt stopped at the host-load guard after one successful controlled interruption. Total elapsed time was 46.74s including preflight; the first child took 44.94s and peaked at 203.6 MiB sampled process-group RSS. All 19 remaining children, including the resume child and every timing comparison, are unrun. **There is still no checkpoint speedup measurement.**

The saved checkpoint remained byte-identical to its recorded hash: 4,600 recoverable evaluations versus 4,800 completed, with 200 completed evaluations lost and zero in flight at this controlled hook. The uninterrupted control reached 16,300 evaluations and ESS 10,020.771588733165. Resume was not launched in this attempt; no crash/resume witness is fully completed by attempt02.

The one-minute host-load admission limit was 2.0 for eight available CPUs. Load before the first child was 1.7075; the guard rejected the next child. Its exact failing snapshot was not persisted, so this report does not attribute the load increase to a process or claim a precise value. Limits remained 20 children, 120s/4096 MiB each and 1800s aggregate. No retry followed the stop, and the guard was not relaxed.

Measured producer15afe55 contains exactly the approved corrected harness from daf0cc6, plus a test portability adjustment and the attempt01 report; library e16fba3ee was unchanged. Frozen source identities, configuration, control and status are in attempt02.json. Both raw attempts and all worktrees remain retained, outside Pulse; merge remains human.

## First attempt and correction

The first bounded attempt stopped after 38.12s because the resume receipt could not serialize a restored NumPy integer. One controlled interruption completed; its resume process failed while writing its receipt. All other 18 children, including every performance comparison, remain unrun. No checkpoint speedup has been measured.

The single-worker deferred interruption preserved the published file at 4,600 likelihood evaluations, with 4,800 completed in memory: 200 completed evaluations lost, zero in flight at this controlled boundary. The prior checkpoint's SHA256 remained unchanged after process-group termination.

The resumed computation reached the exact uninterrupted control result before the receipt error. A separate read-only check reopened the completed resumed checkpoint, evaluated no likelihood, changed no checkpoint bytes, and confirmed exact posterior hash/count/ESS equality: 16,300 evaluations, ESS 10,020.771588733165, nine shells. This diagnostic is not a passed pilot case and does not establish the unrun atomic-publication or two-worker witnesses.

The harness now converts the restored count to a Python integer before JSON serialization (daf0cc6). Its regression reproduced the failure before the fix; all 125 tooling tests pass afterwards. A subsequent test-only correction keeps this regression compatible with the standard-library tooling CI environment; no sampler or measurement behavior changed. No automatic retry followed the stop. Both corrected code and stopped raw data are retained.

The opt-in library policy keeps interval1 native. Larger intervals publish full native snapshots atomically, retain native full-write points, and flush on normal return. Sampling settings, default behavior and identifiers are unchanged. This is protection against process interruption, not a guarantee for power loss or network filesystems.

Measured library e16fba3ee; initial producer d108719. Complete frozen identities, settings, limits, status and unrun cases are in attempt01.json. The original matrix is four crash/resume witnesses followed by six paired D5/D10 seed comparisons; n_live400, n_batch100, n_eff10000, four default networks, discard_exploration=True. Limits are 20 children, 120s/4096MiB each, 1800s total including preflight. All four correctness witnesses must pass before timing comparisons. No further pilot is authorized by this report.

Validation: full library suite 4302 passed, 2 skipped, 1 expected upstream failure; eight script and two notebook smoke cases passed. Tooling Ruff check/format and unchanged catalogue validation pass. Exact library head e16fba3ee passed Python 3.12, Python 3.13 and no-JAX CI (run42867205132), plus Docs. The corrected producer15afe55 passed tooling CI (run42867935308); final report-head CI is recorded on the PR. Author review only.

[Library PR](https://github.com/PyAutoLabs/PyAutoFit/pull/1765) · [Experiment PR](https://github.com/PyAutoLabs/autofit_profiling/pull/15). Outside Pulse; merge remains human. The profiling branch depends on unmerged diagnostic PR13 and the library-first merge gate. Worktrees and raw data remain retained.
