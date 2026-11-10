# Nautilus 30D evidence and output investigation

Reusing exploration samples is the leading explanation for the negative evidence error in the [dimension pilot](../nautilus_cpu_dimensions/README.md). Fresh samples from the same learned bounds substantially reduced the discrepancy. The 30D follow-up timed out before its ESS target, so this is supporting evidence rather than a completed accuracy validation.

A separately authorized output-only replay also timed out inside visualization, after sample and latent output had completed. See the [output replay report](output_evidence/report.html) and the measured stages below.

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

The 98-test tooling suite covers plan tampering, source/input identity, the resumed exclusion semantics, stage persistence, patch restoration, no-likelihood enforcement even when an exception is swallowed, timeout precedence over child success, no retry/budget reset, and missing-case reporting. Runtime preflight restored all three archived posterior hashes and verified that exclusion leaves zero selected samples. The post-measurement checkpoint-mode smoke passed without sampling. Ruff and the unchanged catalogue check are required alongside the tooling suite; no notebooks are generated for these standalone diagnostic scripts.

[Authoritative Heart readiness](evidence/heart-readiness.json) remains RED: **PyAutoGalaxy: CI failure**; **PyAutoLens: CI failure**. The live phase-0b development override and approval of this bounded member permit the development PR. Merge remains a separate human decision; no release or readiness-clearance claim.
