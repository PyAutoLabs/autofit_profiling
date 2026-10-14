# EP / graphical-model baselines and the analytic parity benchmark (B4b findings)

Cells: [`scripts/gaussian_1d_ep/ep/baseline.py`](../../scripts/gaussian_1d_ep/ep/baseline.py),
[`scripts/gaussian_1d_ep/graphical/baseline.py`](../../scripts/gaussian_1d_ep/graphical/baseline.py)
(shared harness [`shared.py`](../../scripts/gaussian_1d_ep/shared.py), config beside) and
[`scripts/analytic_gaussian/parity/benchmark.py`](../../scripts/analytic_gaussian/parity/benchmark.py)
· rows: `results/gaussian_1d_ep/{ep,graphical}/N{3,10,30}.json` (schema
`autofit-profiling-baseline`) and `results/analytic_gaussian/parity/seed0_n5.json`
(`autofit-profiling-parity`) · catalogue: `dashboard/catalogue.json` (`profiling-summary@2`,
Pulse instance `fit`; each row exports a `runtime` wall and a `memory` `host_peak_rss` record).

Ported on 2026-10-14 (search-extensibility B4b, autofit_profiling#4) from
`autofit_workspace_developer/{ep,graphical}` (baselines committed in `07b92ace`, 2026-05-20;
the "postfix" re-runs in `98f260e5` / `27f0a3a9`, 2026-06-01). The model, priors, searches,
EP settings, config and sanity rules are unchanged; the port simulates the seeded datasets
into the gitignored `output/` on every run, starts every fit from an empty output directory
(never a resume) and writes one result row per cell. The developer packages are not deleted
here; that is a separate Gut-backed task after this port merges.

## Host pin (decision taken, flagged)

The epic's release-sweep host pin is `autolens_profiling/hpc/release_sweep.conf`
(`euclid-ral-gpu-2`, an A100 node). It does not fit CPU Dynesty runs, and nothing on the
PyAutoFit side has a pin. B4b pins to the **same laptop host B4a used** —
`DESKTOP-H143S82` (WSL2, 8 cores), recorded as `platform.node()` in every row — so Pulse's
same-host comparison rule pairs these records with B4a's. Cross-machine reproduction on RAL
is a later campaign.

## Reproduction (2026-10-14, laptop CPU)

Stack: PyAutoFit `df6d218f1` (main, tag 2026.10.12.1, clean checkout, sha checked before and
after every fit), PyAutoNerves `7f3055b7f`, JAX 0.10.2 on CPU with x64 on (float64),
dynesty 2.1.5, numpy 2.2.6, scipy 1.17.1, Python 3.12.10. One fit per fresh process, run
serially (graphical N3 → N30, then EP N3 → N30). Wall is `time.perf_counter` around the fit
call only. The reference is the committed developer `profiles/baseline.json` (laptop CPU,
PyAutoFit of 2026-05-20, host not recorded); the drift rule is the epic's
`runtime-drift-2x-1ms` (ratio = reproduced / committed; ≥ 2 drifted, ≤ 0.5 improved,
otherwise flat; 1 ms floor).

| method | N | wall s | committed s | ratio | drift | postfix s / ratio | peak RSS MiB | committed MiB | truth logL sum (exact match) | max logL | sanity (committed) | output MB |
|---|---|---:|---:|---:|---|---|---:|---:|---|---:|---|---:|
| graphical | 3 | 6.0 | 22.8 | 0.263 | improved | 5.6 / 1.061 flat | 420 | 556 | −128.397 (yes) | −125.191 | PASS (PASS) | 0.8 |
| graphical | 10 | 7.1 | 50.7 | 0.141 | improved | 14.9 / 0.477 improved | 446 | 844 | −472.253 (yes) | −519.740 | PASS (PASS) | 2.2 |
| graphical | 30 | 19.0 | 132.8 | 0.143 | improved | — | 515 | 2979 | −1500.561 (yes) | −1658.694 | **FAIL (FAIL)** | 6.3 |
| EP | 3 | 37.1 | 251.9 | 0.147 | improved | 76.3 / 0.486 improved | 353 | 599 | −128.397 (yes) | — | PASS (PASS) | 2.2 |
| EP | 10 | 152.9 | 566.9 | 0.270 | improved | — | 403 | 672 | −472.253 (yes) | — | PASS (PASS) | 9.0 |
| EP | 30 | 842.8 | 1758.7 | 0.479 | improved | — | 642 | 1123 | −1500.561 (yes) | — | **FAIL (FAIL)** | 27.1 |

**Correctness reproduces exactly.** Every `truth_log_likelihood_sum` equals the committed
float bit-for-bit (−128.39683874532219 / −472.2530881754556 / −1500.5605499266132), so the
seeded simulator regenerates the same datasets. Every sanity verdict matches the committed
one, including both N30 failures, which are reported here rather than hidden:

- graphical N30 fails only the max-log-likelihood rule: max logL −1658.694 against the
  threshold −1650.561 (truth sum − 5 nats × 30); committed −1658.655. All 61 parameter
  recovery checks pass. The joint 61-parameter DynestyStatic fit at `nlive=50` stops short
  of the likelihood peak.
- EP N30 fails only the shared centre: 49.26 ± 0.110 against truth 50 (|z| = 6.68); all 60
  per-dataset normalization and sigma checks pass. The EP mean-field centre message is
  over-confident at N = 30.

Graphical max logL at N3 / N10 (−125.191 / −519.740 vs committed −125.143 / −519.326) differs
in the third decimal because the developer fits are unseeded.

**Runtime does not reproduce within 2× — it is faster.** Against `baseline.json` all six
cells are outside the policy on the `improved` side (ratios 0.14–0.48), so the witness
"N3 and N10 within `runtime-drift-2x-1ms`" is **not met as written**. The committed
baselines predate the library changes behind the developer "postfix" re-runs
(2026-06-01; the 2026-07 scoping put per-factor visualisation at ~48 % of EP runtime and
truncnorm prior transforms at 25–38 %): against those, graphical N3 is flat (1.06×) while graphical N10 and
EP N3 are still faster (0.48×, 0.49×). The baselines have moved; nothing here is a
regression. The new rows are the baselines going forward.

**Caveat on the seconds.** Another session's CPU work (a PyAutoFit pytest run and seeded
Dynesty runs) shared the 8-core host during these measurements (load average 2.5 → 4.7,
growing through the EP N30 fit). The ratios are therefore conservative upper bounds — the
direction (improved) is not in doubt, but EP N30's 842.8 s is likely inflated, and its EP
N10 → N30 scaling (5.5× for 3× datasets) should not be read until a quiet-host repeat.
Peak RSS is the fit process's `ru_maxrss` (KiB); how the developer harness's peak was
taken (process scope, simulator in-process or not) is not recorded, so the memory columns
are information, not a like-for-like pair (graphical N30: 515 vs 2979 MiB).

## Analytic Gaussian parity benchmark (seed 0, N = 5)

The CI parity script `autofit_workspace_test/scripts/graphical/analytic_gaussian.py`
(autofit_workspace_test#91) run unchanged from the sibling checkout
(`8b86eca55`), timed as one child process:

| leg | column | cells passed | max a | max b | column s |
|---|---|---:|---:|---:|---:|
| A (σ = 10 known) | minimal EP | 6/6 | 0.000 | 0.000 | 0.0 |
| A | autofit graphical | 6/6 | 0.065 | 0.058 | 24.5 |
| A | autofit EP | 6/6 | 0.000 | 0.000 | 1.9 |
| B (σ ~ TruncGauss(10, 5, 0, 100)) | minimal EP | 7/7 | 0.078 | 0.146 | 0.0 |
| B | autofit graphical | 7/7 | 0.050 | 0.041 | 44.4 |
| B | autofit EP | 7/7 | 0.079 | 0.146 | 56.8 |

`PARITY: PASS (41/41)`, both leg-B hard caps pass, wall 130.0 s, peak RSS 332 MiB (the
docstring's "~85–130 s locally"; an earlier quiet probe the same day took 104.9 s). The
closed-form leg-B scatter quantiles are q05/q50/q95 = 2.9797 / 5.9814 / **12.1536**:
`sigma_q95` matches the value banked in the benchmark's own docstrings, `[q05, q95] =
[2.98, 12.15]`.

**Flag:** the B4b prompt states the banked value as `sigma_q95 = 12.18`. No source carries
12.18; the benchmark docstrings bank 12.15 and the closed form gives 12.1536. The cell
checks against 12.15 (two decimals) and records the full value.

Suite coverage: `analytic_reference.py`, `analytic_ep_minimal.py` and `analytic_autofit.py`
are exercised through the parity script; `analytic_gaussian_collapse.py` is a separate CI
smoke script, not profiled; `analytic_gaussian_priors.py` is **parked `NEEDS_FIX`**
(2026-09-30, loggaussian moments projection, PyAutoFit#1656) and is recorded as parked, not
run.

## What this does not claim

- No verdict: every record is `unreviewed`; the Brain's profiling conductor judges.
- No Pulse comparison: the drift notes live in each row's `reference`, because the
  developer baselines carry no host and so cannot form a record pair.
- No hotspot port: the developer `N10_hotspots.txt` / cProfile summaries are historical
  (absolute machine paths, pre-fix code) and are not copied; EP-loop profiling is epic 2.
