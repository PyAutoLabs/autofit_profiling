# RAL training-pool pilot: complete, mixed speed result

Job 415139 completed with exit code 0:0 on 2026-11-12 in 9 minutes 51 seconds at measured producer `db516b50cd8d6ea1301704bc39e8d14819024968`. All 18 cases passed: six correctness checks followed by twelve timing cases, forming six paired comparisons. Exact posterior/state, trained-model and target parity passed in every pair; both crash/resume witnesses matched control. Every timing checkpoint reopened exactly with the original prior. [Full compact receipt](ral_attempt03.json).

Two persistent spawned training workers reduced median paired full-workflow time by 17.9% in 5D (1.217× speedup), but increased it by 2.4% in 10D (0.976×). Individual changes ranged from 36.2% faster to 7.1% slower in 5D, and 1.0% faster to 19.9% slower in 10D. Native four-network training wall time decreased in all six pairs, but startup, transport and other fit work prevented consistent end-to-end gains. This is a mixed experimental result, insufficient to change production defaults.

## Paired timings

Full-workflow time includes pool startup and shutdown plus the complete fit; it excludes interpreter/import and supervisor overhead. Percent change is `(pool / serial - 1) * 100`; negative is faster. Sampler speedup is serial sampler.run wall divided by pooled sampler.run wall, so greater than 1 is faster. The reported dimension summaries are medians of paired ratios/changes, not ratios of separately aggregated durations.

| Dimension | Seed | Serial workflow (s) | Pool workflow (s) | Workflow change | Sampler speedup |
|---|---|---|---|---|---|
| 5 | 0 | 26.794 | 22.011 | -17.9% | 1.393× |
| 5 | 1 | 26.843 | 17.135 | -36.2% | 1.766× |
| 5 | 2 | 16.613 | 17.796 | +7.1% | 1.024× |
| 10 | 0 | 29.560 | 30.284 | +2.4% | 1.038× |
| 10 | 1 | 30.286 | 29.977 | -1.0% | 1.080× |
| 10 | 2 | 31.520 | 37.799 | +19.9% | 0.921× |

Median sampler speedup was 1.393× in 5D and 1.038× in 10D. Two-worker pool startup was about 1.33–1.41 seconds per fit. Do not sum overlapping worker durations into elapsed runtime. Three seeds per dimension on one exclusive RAL allocation are a small pilot; this is not a confidence interval, a general CPU speedup claim or a 30D result.

## Controls, limits and cleanup

Same four native networks, seeds/settings, serial likelihood and adaptation, native checkpoint policy, output filesystem and frozen packages in both arms. D5/D10 × seeds 0/1/2, alternating AB/BA/AB. 400 live points, ESS 10000, no synthetic work. Search corner plotting disabled equally; other fit output retained. No installed-library edits.

Compute bootstrap 23.56 seconds; scientific pilot 565.95 seconds against the 1800-second cap. Maximum child 47.92 seconds against 120 seconds; maximum sampled child-process-group RSS 594.50 MiB against 4096 MiB. Scheduler reports a four-CPU bound step and whole-node exclusivity (252 CPUs reserved), 8 GiB job memory and 35-minute cap. All child descendants reaped; compute runtime cache cleanup complete. All raw outputs and earlier failed jobs 415132/415136 plus diagnostic 415135 are retained.

## Correction and validation

The earlier failure compared Gaussian physical coordinates with a checkpoint reopened using identity prior. The verification code now passes the captured native prior unchanged and retains exact checks and the forbidden likelihood callback. The non-identity-prior regression rejects the old behavior. Offline retained-state validation passed with no sampling/training/likelihood and unchanged checkpoint bytes; the new complete pilot provides the previously missing live parity evidence.

244 tooling tests, lint/format, shell and unchanged catalogue checks pass; measured-producer CI 43206509078 green. CI 43206309695 exposed a procfs ESRCH race in a process-reaping test, corrected before the final artifact and job. Author review only. Existing scoped Heart RED development override applies; no release/merge/failed-CI bypass.

The approved single fresh pilot is complete and its allowance consumed. No further compute is scheduled or authorized by this record. Outside Pulse; PR #17 remains for human review and merge. No production default changed.
