# Nautilus 30D evidence and output investigation

Reusing exploration samples is the leading explanation for the negative evidence error in the [dimension pilot](../nautilus_cpu_dimensions/README.md). Fresh samples from the same learned bounds substantially reduced the discrepancy. The 30D follow-up timed out before its ESS target, so this is supporting evidence rather than a completed accuracy validation.

The latest instrumented full30D replay completed in114.77seconds and produced a figure. It attributes substantial cost to drawing495axes, repeated weighted quantiles, plotting-library dispatch and work outside individual panel calls. No speedup was applied, and the earlier full30D timeout is retained: single-run variation is large. See the [attribution report](attribution_evidence_01/report.html) and the [earlier comparison](plot_evidence_03/report.html).

This is experimental phase 0b.2 for [issue #12](https://github.com/PyAutoLabs/autofit_profiling/issues/12), outside Pulse. Full phase 0b remains incomplete. [Open the report](evidence/report.html), [all attempt receipts](evidence/attempts.json), or the [frozen protocol](protocol.json).

## Evidence results

Each input is the original direct seed-0 checkpoint. Fresh sampling explicitly excludes exploration points, requests ESS 10,000 and at least 100 new samples per shell, and uses the original likelihood, prior, constructor settings and learned bounds. These are conditional resampling results, not independent full-search seeds.

| Dimension | Original logZ error | Fresh logZ error | Fresh ESS | Additional likelihood calls | Outcome |
|---|---:|---:|---:|---:|---|
| 5 | −0.057994 | +0.012717 | 10,020.8 | 11,700 | Completed |
| 10 | −0.147382 | −0.017798 | 10,062.2 | 16,500 | Completed |
| 30 | −0.908090 | −0.049803 | 2,633.6 | 118,900 | **Partial checkpoint only; timed out** |

The 30D row comes from a separate read-only inspection of the last persisted checkpoint. It remains ineligible as a success and may lag the killed process. No additional calls were made to inspect it. Its standardized mean error is 0.0455 and relative covariance error 0.0860, but neither moments nor ESS certify evidence accuracy. There are no independent repetitions or uncertainty calibration at 30D.

For the completed 5D/10D runs, including the exploration samples again in the final estimates gives errors −0.002583 and −0.044351. Discarding exploration removes one known bias source; it does not guarantee that a single realization is closer to truth. Additional sampling also changes the precision, so these observations do not quantify the full bias separately from sampling variation.

The prior investigation independently checked the Gaussian convolution evidence, stored likelihoods and pointwise Bayes identity to numerical precision. All original pilot controls had zero post-exploration samples. Nautilus 1.0.5's `Sampler.run` docstring explicitly identifies discarding exploration as required for unbiased estimates.

## Boundaries and missing work

The local driver spent **159.24 seconds**, including 5.14 seconds of runtime preflight. The 5D and 10D children took 14.94 and 16.51 seconds. The first 30D child hit its unchanged 120-second cap at 120.07 seconds; sampled process-group RSS peaked at about 279 MiB. The driver stopped immediately. The two additional 30D conditional repetitions and the wrapped output replay are **unrun**. No cap increase, retry, larger dimension or remote submission followed.

Original checkpoint/result hashes passed before and after all attempts. Around 93 MB of raw checkpoints, logs and validation copies remain locally under ignored `output/nautilus-30d-diagnostics/`. No production results, catalogue or Pulse records changed. The input archives are also retained.

## Checkpoint mode persistence

This investigation exposed a diagnostic-harness persistence defect. Setting `sampler.discard_exploration = True` updates in-memory shell counts, but Nautilus's incremental `write_shell_update` does not persist that flag. The measured 30D copy therefore contained fresh-only counts with its old `discard_exploration=False` metadata. Calling `posterior()` directly after reopening it failed with inconsistent array lengths.

The [partial-checkpoint receipt](evidence/partial-checkpoint.json) records the recovery: set the flag to true **in memory only**, recompute native shell statistics and read the estimate. Stored fresh counts matched the dataset lengths minus the exploration boundaries. The raw checkpoint hash stayed unchanged; no repaired file replaced it.

The final harness now performs a full native checkpoint write immediately after setting the exclusion mode, before sampling. A [30D round-trip smoke check](evidence/mode-roundtrip.json) confirms that a copied original checkpoint reopens with exclusion enabled, zero selected samples, and unchanged likelihood count 34,100. No sampling occurred in this check.

Measurements were made at commit `3cf17eab18ead7fff72d3bf0de394c8c22188303`. The persistence correction, extra tests and report were added afterwards. **No timed experiments were rerun against the correction.** The completed estimates were calculated from the correct in-memory state and are unaffected by the persisted flag defect. Future runs use the corrected code and receive a new provenance identity.

## Wrapped timeout: original evidence and initial pilot

The original wrapped 30D checkpoint matched the completed direct checkpoint in every dataset and attribute. It reached the same saved sampling result, then had only about 14.5 seconds of its 120-second process budget left for final output. It has no completed-fit receipt.

A preceding read-only replay of its 33,189 thresholded CSV samples took 21.59 seconds for `SamplesPDF.summary()` under cProfile, mostly repeated parameter-array construction. Drawing 100 samples took 0.009 seconds, and drawing plus the empty analysis's latent calculation took 0.026 seconds. These diagnostic timings include profiler overhead and are not end-to-end benchmarks. The original log had already passed its first summary write: the last latent message alone does not locate where it was interrupted. Plotting and subsequent output remain candidates, not established causes.

The new output-only replay rejects any likelihood evaluation, checks the posterior hash, and durably records conversion, summary, CSV, latent, visualization and finalization stages. Its guards have unit tests and its runtime hooks passed preflight, but **the actual replay was not reached** because the earlier evidence case hit the stop rule. The separately authorized continuation below isolates that replay without another resampling attempt. No library optimization is included here.

## Authorized output-only continuation

The human selected **“Continue the wrapped-output investigation”** after the first pilot stopped. A [separate protocol](output_protocol.json) and plan contain exactly one output replay, with the same 120-second/4096-MiB child caps and a tighter 180-second aggregate budget. No evidence case can be admitted in this scope. The original six-case plan and its halted/unrun coverage are unchanged.

The replay used measurement commit `e701a77f83d7c58ba49a534e4aa180f8db28395c` and the same pinned libraries/packages. It hit the wall cap at **120.06 seconds**; total charged runtime was **123.19 seconds**, including **2.95 seconds** of preflight. Sampled process-group peak RSS was **526 MiB**. There was no retry or cap increase.

| Output stage | Inclusive wall seconds | Result |
|---|---:|---|
| Restore sampler / native `run` | 0.0063 for `run` | Already satisfied original stopping criteria |
| Convert native samples | 9.902 | Completed |
| Build sample summary | 10.572 | Completed |
| Write samples | 5.263 | Completed |
| Complete sample-output parent stage | 25.772 | Completed; includes the stages above |
| Latent-output parent stage | 0.0165 | Completed |
| Draw 100 latent samples | 0.0077 | Completed; nested within latent output |
| Visualization | — | **Entered at trace time26.767s; still active at termination** |
| Profiling/summary after visualization and post-fit output | — | Not reached |

Nested stage times overlap and must not be summed. The broad visualization stage includes graphics setup: the log records a new Matplotlib font cache during it. This is one instrumented local replay, not a warm-cache steady-state benchmark, and the trace does not separate plot construction, font setup and rendering. No exact visualization duration is reported for the incomplete stage.

This establishes that visualization can exhaust the output-only budget even after sample and latent handling have finished. The latent stage is negligible in this replay. The original timeout's exact instruction remains unobserved; the replay identifies a material contributor without manufacturing a completed original fit.

Both native sampler evaluation and direct wrapper likelihood calls were blocked. The copied checkpoint is [byte-identical to its original input](output_evidence/checkpoint-integrity.json) after termination; original inputs also passed their hash checks. The standard success receipt is absent because the process was killed, and the attempt remains ineligible as success. Around45MB of additional raw output is retained under ignored `output/nautilus-output-only/`.

Source inspection identifies the likely plotting target for a later experiment: the pinned weighted-search configuration enables `corner_anesthetic`, uses compact plots above10parameters, and skips the full corner only **above30parameters**. This30D model is still eligible for the full corner. The current trace measures `SearchUpdater.visualize`, not that plot function individually. A future bounded plot-subset/configuration comparison should isolate it; no plotting behavior or library code was changed here.

To prepare this separately authorized scope, add `--output-only` to the `prepare` command below and use a new plan/output directory. It cannot resume the halted evidence pilot. Detailed [attempts](output_evidence/attempts.json), [durable stages](output_evidence/output-d30-rsaved-stages.jsonl), and [portable plan](output_evidence/plan.json) are retained.

## Approved plotting comparison: stopped by host-load guard

The human approved three sequential output replays and explicitly allowed **one final-summary timing likelihood call per case**. Native sampling remains forbidden; the harness permits that direct call only inside `SearchUpdater._profile_and_summarize`, counts it separately, and checks the point/value against the stored maximum likelihood. Earlier zero-call experiments retain their original constraints.

The [frozen protocol](plot_protocol.json) compares no corner, a six-parameter compact corner (indices 0, 6, 12, 17, 23, 29), and a full 30D compact corner. Both rendered variants use a 12×12-inch, 100-dpi canvas, diagonal KDE and lower-triangle histograms, with all original samples and weights. Durable hooks distinguish axis construction, posterior plotting and PNG export. Required images, actual geometry and swallowed plotting errors are checked explicitly.

Measurement code was frozen at `51663d11fc6a873d9681899608bd352a294293e5`. Runtime/checkpoint/font-cache preflight succeeded in **6.438 seconds**, but the driver then halted with **`RuntimeError: host load exceeds guard` before launching any child**. Its guard limits one-minute load to 2.0 on this eight-CPU affinity. The exact rejected load was not persisted; a later observation cannot substitute for it.

**All three cases are unrun.** No sampler or final-summary evaluations occurred in this continuation, no figure was produced, and there are no new timing comparisons or successful finalization witnesses. Total charged time was 6.438 seconds including preflight. The approved stop rule was honored without retry, cap change or environment change. See the [comparison report](plot_evidence/report.html), [halted state](plot_evidence/attempts.json) and [portable plan](plot_evidence/plan.json).

The small raw plan/state archive is retained under `output/nautilus-plot-comparison/`; earlier raw archives are preserved. This report and its driver-stop display were added after the frozen measurement commit. The runtime plotting path remains unvalidated end to end. A further attempt requires a new human compute decision and separate plan/output identity; PR #13 remains open for human review, outside Pulse.

## Fresh plotting attempt: harness error before summary timing

The human authorized one fresh attempt with “I approve, continue”, retaining the earlier host-guard halt. The unchanged harness at `9dbc272a1bb498d0d2b6d4e904484c3d84e7914f` passed preflight in3.067seconds and launched the no-corner case. That child failed after18.982seconds; total charged time was22.364seconds. Both plotted cases remain unrun under the stop-on-first-failure rule.

This was a **diagnostic harness defect**, not a PyAutoFit or sampling failure: the wrapper declared `timed_summary(instance, ...)`, so Python's bound updater argument collided with PyAutoFit's separate `instance=` keyword. It raised `TypeError: ...timed_summary() got multiple values for argument 'instance'` before entering the summary timing gate. No likelihood evaluation occurred, and the copied checkpoint is byte-identical to its original. Sample output completed in14.805seconds, latent processing in0.0127seconds, and no-corner visualization in0.0150seconds. These are inclusive diagnostic stage observations; finalization did not complete and the case is not a success.

The wrapper's bound argument is now named `updater`, preserving PyAutoFit's keyword unchanged. A regression test exercises the real keyword calling convention and verifies that the guard and method restoration still work. This correction was made **after** the failed measurement; no timed replay has been run against it. There is still no end-to-end plotting or successful finalization witness.

See [attempt02 report](plot_evidence_02/report.html), [attempt receipts](plot_evidence_02/attempts.json), [checkpoint integrity](plot_evidence_02/checkpoint-integrity.json) and [portable plan](plot_evidence_02/plan.json). Raw data remain under `output/nautilus-plot-comparison-02/`; the first comparison's guard-halted state and every earlier experiment remain unchanged. The same120s/4096MiB child and420s aggregate caps apply to any separately authorized next attempt; no automatic retry or merge.

## Corrected plotting comparison: two successes, full30D capped

The human selected **“Run corrected comparison once”** after the wrapper fix. Attempt03 used clean committed code `ab896cdc36061b33998bae9c214d6a47c7773b5d`, the same pinned environment, unchanged original posterior and identical approved caps. The driver charged **242.32seconds**, including4.518seconds of preflight, within its420-second budget. It stopped after the third case timed out; no retry followed.

| Plot policy | Outcome | Child process seconds | Axis construction | Posterior plotting | PNG export |
|---|---|---:|---:|---:|---:|
| No corner | Success | 54.054 | — | — | — |
| Six selected parameters,12×12in | Success | 62.278 | 0.436s | 2.649s | 1.005s |
| All30parameters,12×12in | Timeout | 120.092 | 9.997s | 74.739s | Entered; unfinished |

Times are inclusive instrumented stage measurements, not additive component totals. Child process time also includes imports, checkpoint copy, provenance/integrity checks and posterior verification after wrapped output. The wrapped replay itself completed in37.470seconds without the corner and42.681seconds for six parameters. These single realizations are not steady-state speedup estimates.

Both successes completed normal wrapped finalization, preserved the posterior hash and native count34100, and made **exactly one separate summary timing evaluation** at the stored maximum-likelihood point. No native sampling occurred. The six-parameter PNG has1200×1200pixels,27axes including diagonal twins, and every one of the33984original selected samples with its original weight. Its parameter indices are0,6,12,17,23,29; the library labels all six axes `x`, so the index list identifies them. The figure was visually inspected and its dimensions/hash verified.

The30D case constructed495axes on the same12×12-inch,100-dpi figure. It finished posterior plotting, entered PNG export at trace109.261seconds, then reached the process cap. No PNG was produced; final summary timing and post-fit output were not reached. It has no success receipt and stays ineligible as success. All three copied checkpoints are byte-identical to the original; original input hashes passed before/after every child. Sampled peak process-group RSS was347MiB,365MiB and526MiB respectively, with the usual shared-page/detection-interval limitations.

**Interpretation:** disabling the corner allows full finalization, and a six-parameter corner completes under the same guard. At a fixed small canvas, the full30D plotting path remains expensive before export begins. Shrinking raster dimensions alone was insufficient for this cap. The earlier default-canvas replay had a different font-cache state and only broad visualization tracing, so this is not a matched large-versus-small-canvas benchmark or proof of the original timeout's exact instruction. The run does not resolve30D evidence accuracy or establish production defaults.

See the [latest report and figure](plot_evidence_03/report.html), [attempt receipts](plot_evidence_03/attempts.json), [checkpoint integrity](plot_evidence_03/checkpoint-integrity.json) and [portable plan](plot_evidence_03/plan.json). About156MiB of raw data are retained in `output/nautilus-plot-comparison-03/`, alongside the45MiB failed attempt02 and all previous archives. No Pulse records, library code or production configuration changed. Further sampling, optimization or timing experiments need a separately scoped decision; PR13 remains for human review/merge.

## Full30D attribution: drawing and repeated panel work

The human approved the proposed attribution step with **“continue”**. Code was frozen at `375fa0be68e9c65436b466398eb560bf838650ba`. A supervised synthetic2D/30-point preflight exercised the real plotting aliases, panel counts, numerical hooks, deferred axis handling and PNG path without a likelihood evaluation. It passed within its20-second limit. Then exactly one full30D replay ran with the unchanged small-canvas policy and original posterior.

The child **succeeded in114.77seconds**, within120s/4096MiB caps; total charged118.76seconds included3.881seconds of preflight, below180seconds. Sampled peak process-group RSS was1037MiB. The replay completed finalization, retained the posterior hash/native34100likelihood count, and made exactly one separately authorized summary timing call with zero new sampler calls. The copied checkpoint is byte-identical to the original.

| Measured work | Calls | Seconds | Meaning |
|---|---:|---:|---|
| Figure drawing | 1 | 21.519 | Rasterizing the495axes and their artists |
| Plot loop outside timed panel calls | 1 | 13.914 | Residual: includes pair selection/cleanup, labels and other loop work; not individually attributed |
| Histogram panel dispatch outside histogram kernel | 435 | 10.793 | Plot-accessor/pandas overhead and other wrapper work; not one identified operation |
| Weighted quantiles inside histograms | 1,740 | 10.407 | Repeated conversion, sorting and interpolation |
| Diagonal KDE kernel, including its60quantiles | 30 | 5.916 | Density computation and associated drawing setup |
| Axis construction | 1 | 5.018 | Creating the figure/axes structure |
| Deferred shared-axis handling | 1 | 4.794 | Already deferred to one pass by the pinned library |
| 2D histogram binning | 435 | 0.785 | NumPy histogram calculation |
| Histogram artist construction | 435 | 0.472 | pcolormesh creation; later rendering counted above |
| PNG encoding and writing | 1 | 0.095 | Image save after canvas drawing |

These selected rows do not overlap but are not a complete end-to-end budget. The [raw attribution receipt](attribution_evidence_01/attribution.json) also includes nested inclusive totals, which **must not be added together**. Timer self time excludes timed children. Both timer totals exclude measured snapshot I/O;947snapshot writes recorded4.112seconds of overhead, with the final write's own cost absent from its receipt. Other instrumentation overhead remains. Broad durable plotting time was47.039seconds, while the nested plotting total excluding snapshot I/O was43.009seconds. The entire visualization stage took78.884seconds.

The full figure has1200×1200pixels,495axes and all33984original samples/weights. Visual inspection confirms that labels overlap heavily: this small full-corner canvas is diagnostic, not a readable production default. The earlier full30D run spent74.739seconds in plotting and hit the cap during export; this run completed without any algorithmic optimization. Different observed timing despite the same pinned inputs/settings, plus probe overhead, means **this is attribution evidence, not a measured speedup**. Repeated matched trials are required before judging an optimization.

The numerical result narrows the next choices. Histogram binning and file writing are small contributors in this realization. Repeated quantiles are a concrete candidate for per-plot/per-column reuse, but any reuse must preserve weighting, quantile semantics and pair-specific finite-value handling. The total histogram-quantile opportunity is about10.4seconds here before cache overhead; it cannot remove the much larger drawing and wrapper/loop costs. Rendering and pandas/loop overhead need further targeted attribution before claiming a behavior-preserving fix. Existing shared-axis deferral and avoidance of a duplicate pyplot draw are already present and must not be counted as new work.

See the [report and full figure](attribution_evidence_01/report.html), [attempt receipt](attribution_evidence_01/attempts.json), [integrity witness](attribution_evidence_01/checkpoint-integrity.json), [synthetic preflight witness](attribution_evidence_01/synthetic-success.json) and [frozen protocol](attribution_protocol.json). About56MiB raw data remain in `output/nautilus-plot-attribution-01/`, with the separate preflight archive and every previous experiment retained. No sampling optimization, library/default change, Pulse registration or merge occurred. Fullphase0b and30Devidence accuracy remain incomplete.

## Reproduce

Use the phase-0b.1 pinned environment: PyAutoFit `bdb7af0e9c339ad07850679fd6ac168aacc7d18c`, PyAutoNerves `9b204c90c2ce05c5bb0cf3e9800ed13f005c6c56`, and the [third-party requirements](../nautilus_cpu_dimensions/requirements.txt). Activate the pinned library sources and commit measurement code before preparing a new plan. The runtime preflight verifies dependency, package, target, host, CPU-affinity and thread identities against the input archive; the new producer identity is recorded separately.

```bash
python scripts/analytic_gaussian/nautilus/diagnose.py prepare --archive PATH_TO_ORIGINAL_PILOT --output output/diagnostics/plan.json
python scripts/analytic_gaussian/nautilus/diagnose.py run --plan output/diagnostics/plan.json --output output/diagnostics/pilot
python scripts/analytic_gaussian/nautilus/diagnose_report.py output/diagnostics/pilot --output output/diagnostics/report
```

Execution refuses an existing output directory. It has six fixed planned cases, native threads capped at one, 120 seconds and 4096 MiB per child, and a 900-second aggregate budget including preflight. Memory/load guards precede every attempt. Caps and interruptions cannot be promoted to success. A new run is a new compute decision; these commands do not authorize extending the stopped pilot.

The published plan removes local paths while retaining input hashes, case identities, runtime pins and the original local-plan digest. It is a provenance record, not a runnable replacement for the path-bearing plan.

## Validation and readiness

The 111-test tooling suite covers plan tampering, source/input identity, the resumed exclusion semantics, stage persistence, patch restoration, no-likelihood enforcement even when an exception is swallowed, timeout precedence over child success, no retry/budget reset, and missing-case reporting. Runtime preflight restored all three archived posterior hashes and verified that exclusion leaves zero selected samples. The post-measurement checkpoint-mode smoke passed without sampling. Ruff and the unchanged catalogue check are required alongside the tooling suite; no notebooks are generated for these standalone diagnostic scripts.

[Authoritative Heart readiness](evidence/heart-readiness.json) remains RED: **PyAutoGalaxy: CI failure**; **PyAutoLens: CI failure**. The live phase-0b development override and approval of this bounded member permit the development PR. Merge remains a separate human decision; no release or readiness-clearance claim.
