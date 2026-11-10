# Checkpoint batching experiment

**The completed pilot is a negative performance result: interval 5 was slower than native interval 1 in all six pairs. Keep the native default.** All four controlled interruption/resume witnesses passed and every pair retained exactly the same posterior, likelihood count, ESS and shell count. This result does not justify promoting the optional full-snapshot policy as a speed improvement.

The approved quiet-host attempt03 completed all 20 children in 931.93s (15 min 32 s), including preflight, within the 1800s budget. The slowest child took 88.91s against 120s; peak sampled process-group RSS was 493.1 MiB against 4096 MiB. No case failed, no retry occurred, and none remain unrun. The pre-launch wait took about 19 min 21 s and is recorded separately; final load readings were 0.9980, 0.7402, 0.5732. The in-run load guard remained 2.0.

## Matched timing results

Ratios below are candidate interval 5 / baseline interval 1; values above 1 mean slower. Three seeds per dimension, AB/BA/AB order, identical sampling/output settings. These are exploratory local measurements, not a statistical confidence estimate or a comparison with the historical phase0a settings.

| Dimension | Seed | Sampler1 (s) | Sampler5 (s) | Ratio | Full fit1 (s) | Full fit5 (s) | Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 26.77 | 33.65 | 1.257 | 32.25 | 37.53 | 1.164 |
| 5 | 1 | 22.41 | 28.44 | 1.269 | 26.41 | 32.14 | 1.217 |
| 5 | 2 | 26.08 | 33.72 | 1.293 | 30.24 | 38.07 | 1.259 |
| 10 | 0 | 50.60 | 72.85 | 1.440 | 57.96 | 81.27 | 1.402 |
| 10 | 1 | 44.03 | 53.38 | 1.212 | 53.43 | 61.75 | 1.156 |
| 10 | 2 | 42.52 | 63.26 | 1.488 | 48.45 | 71.05 | 1.466 |

Sampler time rose 21.2–48.8% (median 28.1%); full fit time rose 15.6–46.6% (median 23.8%). `sampler.run` includes checkpoint publication and the candidate normal-return flush. Full fit time includes wrapper output except search corner plotting, disabled identically in both arms. Process times additionally include startup and receipt checks; all are retained in attempt03.json.

## Why this policy costs more

The native policy uses incremental shell updates. The candidate batches notifications but replaces each scheduled update with a complete atomic snapshot, including fsync. Measured checkpoint time rose 2.81–4.44x, and bytes passed to write syscalls during checkpoint calls rose 2.37–2.62x. Kernel write_bytes counts also rose. This supports full-snapshot persistence cost as the main explanation for the slowdown, rather than a changed sampling trajectory. It does not isolate serialization, fsync and replacement costs individually.

| Dimension | Seed | Checkpoint1 (s) | Checkpoint5 (s) | Full writes1/5 | Write syscall bytes1/5 (MiB) |
|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 3.09 | 8.67 | 10/42 | 40.46/97.84 |
| 5 | 1 | 2.38 | 7.89 | 10/42 | 41.39/98.10 |
| 5 | 2 | 3.07 | 9.10 | 10/42 | 40.76/97.90 |
| 10 | 0 | 4.68 | 20.76 | 12/55 | 88.25/211.84 |
| 10 | 1 | 4.10 | 14.41 | 12/57 | 90.97/223.53 |
| 10 | 2 | 4.07 | 18.07 | 13/58 | 91.48/239.62 |

The counters record top-level persistence time without double-counting nested calls. `/proc/self/io` measures the sampler process during persistence calls; it is not a physical-device bandwidth measurement. Full-snapshot logical-byte sums exclude native incremental updates and must not be interpreted as total I/O. Instrumentation overhead is not subtracted; brief RSS spikes may be missed. No durability guarantee is made for power loss or network filesystems.

## Recovery witnesses

| Workers | Interrupted phase | Saved evaluations | Completed evaluations | Lost completed | In flight | Exact resumed control |
|---:|---|---:|---:|---:|---:|---|
| 1 | deferred | 4600 | 4800 | 200 | 0 | yes |
| 1 | atomic | 4600 | 5100 | 500 | 0 | yes |
| 2 | deferred | 4600 | 4800 | 200 | 0 | yes |
| 2 | atomic | 4600 | 5100 | 500 | 0 | yes |

Every interrupted published file retained its recorded SHA256. Resume restored the saved count and finished with the exact uninterrupted control posterior/count/ESS. The controlled hooks had no likelihood work in flight; the 200/500 losses are observations at those hooks, not a universal bound on arbitrary interruption loss. Partial unpublished atomic-write files and all raw data are retained.

## Provenance and validation

Measured producer 5206f0b and library e16fba3ee. The harness is unchanged from the approved correction daf0cc6. See attempt03.json for complete source/package hashes, frozen settings, per-case metrics, exact parity and resource receipts; quiet_host_admission03.json records the pre-launch wait. Settings remain 400 live points, batch 100, ESS 10000, four default networks, discard_exploration=True, one timing worker and one numerical-library thread.

Library validation remains 4302 passes / 2 skips / 1 expected upstream failure, 10 smoke cases, and exact-head Python 3.12/3.13/no-JAX CI 42867205132 plus Docs green. Harness validation remains 125 tooling tests with Ruff/catalogue gates. This update adds result data/documentation only. Author review, no independent review claimed.

[Library PR](https://github.com/PyAutoLabs/PyAutoFit/pull/1765) · [Experiment PR](https://github.com/PyAutoLabs/autofit_profiling/pull/15). Outside Pulse; defaults unchanged; merge remains human. The experiment depends on unmerged diagnostic PR13 and the library-first merge gate. All three attempts/worktrees remain retained. This completes the approved bounded checkpoint experiment, not the wider phase 0b or 30D evidence-accuracy work.

## Retained earlier attempts

The explicitly approved corrected attempt stopped at the host-load guard after one successful controlled interruption. Total elapsed time was 46.74s including preflight; the first child took 44.94s and peaked at 203.6 MiB sampled process-group RSS. All 19 remaining children, including the resume child and every timing comparison, are unrun. **There is still no checkpoint speedup measurement.**

The saved checkpoint remained byte-identical to its recorded hash: 4,600 recoverable evaluations versus 4,800 completed, with 200 completed evaluations lost and zero in flight at this controlled hook. The uninterrupted control reached 16,300 evaluations and ESS 10,020.771588733165. Resume was not launched in this attempt; no crash/resume witness is fully completed by attempt02.

The one-minute host-load admission limit was 2.0 for eight available CPUs. Load before the first child was 1.7075; the guard rejected the next child. Its exact failing snapshot was not persisted, so this report does not attribute the load increase to a process or claim a precise value. Limits remained 20 children, 120s/4096 MiB each and 1800s aggregate. No retry followed the stop, and the guard was not relaxed.

Measured producer15afe55 contains exactly the approved corrected harness from daf0cc6, plus a test portability adjustment and the attempt01 report; library e16fba3ee was unchanged. Frozen source identities, configuration, control and status are in attempt02.json. Both raw attempts and all worktrees remain retained, outside Pulse; merge remains human.

## First attempt and correction

The first bounded attempt stopped after 38.12s because the resume receipt could not serialize a restored NumPy integer. One controlled interruption completed; its resume process failed while writing its receipt. All other 18 children, including every performance comparison, remain unrun. No checkpoint speedup has been measured.

The single-worker deferred interruption preserved the published file at 4,600 likelihood evaluations, with 4,800 completed in memory: 200 completed evaluations lost, zero in flight at this controlled boundary. The prior checkpoint's SHA256 remained unchanged after process-group termination.

The resumed computation reached the exact uninterrupted control result before the receipt error. A separate read-only check reopened the completed resumed checkpoint, evaluated no likelihood, changed no checkpoint bytes, and confirmed exact posterior hash/count/ESS equality: 16,300 evaluations, ESS 10,020.771588733165, nine shells. This diagnostic is not a passed pilot case and does not establish the unrun atomic-publication or two-worker witnesses.

The harness now converts the restored count to a Python integer before JSON serialization (daf0cc6). Its regression reproduced the failure before the fix; all 125 tooling tests pass afterwards. A subsequent test-only correction keeps this regression compatible with the standard-library tooling CI environment; no sampler or measurement behavior changed. No automatic retry followed the stop. Both corrected code and stopped raw data are retained.

The opt-in library policy keeps interval 1 native. Larger intervals publish full native snapshots atomically, retain native full-write points, and flush on normal return. Sampling settings, default behavior and identifiers are unchanged. This is protection against process interruption, not a guarantee for power loss or network filesystems.

Measured library e16fba3ee; initial producer d108719. Complete frozen identities, settings, limits, status and unrun cases are in attempt01.json. The original matrix is four crash/resume witnesses followed by six paired D5/D10 seed comparisons; n_live400, n_batch100, n_eff10000, four default networks, discard_exploration=True. Limits are 20 children, 120s/4096MiB each, 1800s total including preflight. All four correctness witnesses must pass before timing comparisons. No further pilot is authorized by this report.

Validation: full library suite 4302 passed, 2 skipped, 1 expected upstream failure; eight script and two notebook smoke cases passed. Tooling Ruff check/format and unchanged catalogue validation pass. Exact library head e16fba3ee passed Python 3.12, Python 3.13 and no-JAX CI (run42867205132), plus Docs. The corrected producer15afe55 passed tooling CI (run42867935308); final report-head CI is recorded on the PR. Author review only.

[Library PR](https://github.com/PyAutoLabs/PyAutoFit/pull/1765) · [Experiment PR](https://github.com/PyAutoLabs/autofit_profiling/pull/15). Outside Pulse; merge remains human. The profiling branch depends on unmerged diagnostic PR13 and the library-first merge gate. Worktrees and raw data remain retained.
