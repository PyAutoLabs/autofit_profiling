# 10D diagnostic attempt01: stopped on cross-run exact parity

The implementation passed validation, but the first RAL capture failed its comparison with the retained reference. No replay or traced-fit timings were collected. The study therefore does not yet explain the earlier10D end-to-end slowdown.

Job415179 ran on euclid-ral-compute-10-2, whereas reference job415139 ran on euclid-ral-compute-1. Both had the same pinned Python, complete distribution inventory, Fit/Nerves/native Nautilus sources, thread settings, sampling configuration and four-core affinity. The protocol required bitwise cross-run equality but did not restrict the new allocation to the reference node. That was an omission in experiment admission, not a RAL installation failure.

## What ran

Measured source0958bc28705541f6ff07fa4ad79bbfaf94ccdaaf; CI43220440830 green.292 local tests, lint/format/shell/catalogue checks and zero-science worker/descriptor smoke passed. Pinned-runtime validation passed3.91s; fresh artifact13 restore passed65.78s with zero science and cleanup. Compute bootstrap passed25.80s.

The capture fit completed normal output and prior-preserving live checkpoint verification, then rejected the historical posterior checksum. The job stopped immediately: one failed capture,12 replays and four traced fits unrun. Scheduler FAILED1:0 in70s; scientific supervisor42.645s; capture child36.976s; sampled process-group peak241.81MiB. No descendants remained and temporary compute-cache cleanup completed. The one-job authorization is consumed.

## Read-only checkpoint comparison

Both retained checkpoints were reopened with the original Gaussian prior, with sampling, training and likelihood calls forbidden. All forbidden-call counters remained zero and checkpoint bytes were unchanged. The old reconstructed posterior and all11 old model digests exactly reproduce the retained reference, validating this inspection path. It is read-only reconstruction, not a retroactive live-validation claim.

| Check | Reference415139 | Capture415179 |
|---|---:|---:|
| Likelihood calls | 23,500 | 23,500 |
| Shells | 12 | 12 |
| Effective samples | 10038.507608002912 | 10038.507608002512 |
| Training calls | 11 | 11 |
| All44 network iteration counts | identical | identical |
| Posterior/model hashes | reference | different |

Initial cube points in shells0,1,2 are exactly equal, yet shell0 likelihood values differ by at most1.07e-14. First-model normalization and coefficients differ at roughly1e-15–1e-14. Across all saved shells, the largest coordinate difference is1.56e-13 and the largest likelihood difference is3.89e-9. All array shapes match. No tolerance was introduced and the exact gate correctly rejected these differences.

The different node plus the tiny discrepancies on identical initial points make cross-node numerical variation the leading explanation. The retained evidence does not isolate CPU instructions, BLAS dispatch or another floating-point mechanism; it does not establish causation or guarantee that node pinning will pass. No new performance conclusion follows.

## Retention and next step

16,627,876bytes of raw output are retained remotely and locally, including the final checkpoint, fit outputs and22 NPY input files. The compressed archive is11,478,684bytes; its checksum and detailed inspection are in attempt01.json. All previous experiments remain retained.

One harness limitation became visible: the historical parity check happens before the successful candidate result and corpus manifest are written. The raw inputs and checkpoint survive, but capture timing/model receipts are not durable on that path. The proposed amendment saves explicitly unverified candidate receipts before historical validation and retains strict admission for every subsequent case.

The proposed next job would request the original node euclid-ral-compute-1, verify its hostname at admission, and retain CPU/BLAS metadata before any fit. It would use the same17-child matrix, limits, pins and exact comparisons, a new output claim, green CI and a fresh runtime restore. No replacement run or implementation of that amendment has been authorized or started. Outside Pulse; merge remains human.
