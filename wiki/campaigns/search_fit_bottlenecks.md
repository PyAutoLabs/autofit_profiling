# Epic 1 — where one `search.fit` spends its time (fast likelihood)

The ranked bottleneck table for epic 1 of the PyAutoFit profiling campaign (Pulse task
`autofit_profiling_bootstrap`, campaign `autofit`; search-extensibility B4a,
autofit_profiling#2): one `af.Nautilus(...).fit` on gaussian_x1, a likelihood fast enough
that the overhead has nowhere to hide. Epic 2 (the EP loop) is **not** opened: it waits
for epic 1's top findings to ship.

This table is a producer measurement and a ranking of *where time goes*, not a verdict
on what to change. The Brain's profiling conductor judges; fixes take the normal Mind
route as one-PR prompts.

## Measurement

Rows: [`results/gaussian_x1/nautilus/breakdown.json`](../../results/gaussian_x1/nautilus/breakdown.json)
(cell [`search_fit_breakdown.py`](../../scripts/gaussian_x1/nautilus/search_fit_breakdown.py),
findings [`search_fit_breakdown.md`](search_fit_breakdown.md)). Laptop CPU (WSL2, 8
cores), numpy fp64, PyAutoFit `911fe05e1` (2026.10.12.1+1), nautilus 1.0.5,
`n_live=200`, `n_networks=4`, `seed=1`, one core, library output defaults; median of 3
fresh fits; wall 28.4 s (repeats 25.8 / 28.4 / 28.9 s); 7 501 likelihood calls;
exclusive component times.

## Ranked table

| rank | component | owner | seconds | share of wall |
|---:|---|---|---:|---:|
| 1 | NN ensemble training (`NeuralNetworkEmulator.train`, 26 bound updates) | nautilus | 11.87 | 43.4 % |
| 2 | checkpoint I/O (`Sampler.write` / `write_shell_update`, 103 HDF5 writes) | nautilus, switched on by PyAutoFit's `filepath=checkpoint.hdf5` | 7.09 | 24.5 % |
| 3 | bound geometry (`NautilusBound.compute` minus NN) | nautilus | 5.61 | 20.4 % |
| 4 | visualisation (`SearchUpdater.visualize`) | PyAutoFit | 1.16 | 4.1 % |
| 5 | sampler bookkeeping (rest of `Sampler.run`) | nautilus | 0.67 | 2.3 % |
| 6 | samples output (pre/post-fit output, samples, latents, summary) | PyAutoFit | 0.49 | 1.7 % |
| 7 | likelihood (`Sampler.evaluate_likelihood`: prior transform + `Fitness`) | PyAutoFit + user | 0.32 | 1.1 % |
| 8 | other (setup, result construction — unattributed remainder) | PyAutoFit | 0.29 | 1.0 % |
| 9 | proposal sampling (`Sampler.sample_shell`) | nautilus | 0.22 | 0.8 % |

PyAutoFit-attributable time (ranks 2's switch, 4, 6, 7's wrapper, 8) is dominated by the
checkpoint write, which is the only PyAutoFit lever above 5 %: the sampler's own
machinery (ranks 1, 3, 5, 9) is 67 % of wall and was ruled on 2026-07-09 (findings §5).

## `likelihood_share` cross-check (B2/B3 rows, autofit_inference)

`likelihood_share` (= `per_call_s × likelihood_evals / wall`, basis `estimated`) from the
B3 wave-1 pilot rows, gaussian_x3 (10 free parameters), laptop CPU, PyAutoFit `0dbf258c`
(pre-A2). Read from autofit_inference `dashboard/summary.json` (inference-summary@1) at
commit [`637713b2`](https://github.com/PyAutoLabs/autofit_inference/blob/637713b253ab149f87a08ea0b02fcb009b4aa24e/dashboard/summary.json)
(branch `feature/search-ext-b3-postmerge-fixes`, PR #7; values identical on main
`900648b5`). `catalogue/search_catalogue.json` at the same commit carries no
`likelihood_share` field (it aggregates legs by task), so the per-run summary records are
the source.

| search | backend | rows with a share | median share | range |
|---|---|---:|---:|---:|
| Nautilus | numpy | 11 | 1.9 % | 1.1–3.4 % |
| Nautilus | JAX CPU | 12 | 0.01 % | 0.00–0.02 % |
| DynestyStatic | numpy | 4 | 32.7 % | 19.0–52.3 % |
| DynestyStatic | JAX CPU | 4 | 4.5 % | 2.6–7.3 % |
| DynestyDynamic | numpy | 4 | 26.1 % | 21.1–53.1 % |
| DynestyDynamic | JAX CPU | 4 | 2.5 % | 1.4–4.4 % |
| SMC | JAX CPU | 3 | 0.01 % | 0.01–0.03 % |

The Nautilus numpy pilot share (1.9 %, 10 parameters) agrees with this cell's measured
1.1 % (3 parameters): on a fast likelihood the likelihood is ~1–2 % of a Nautilus
`search.fit` and the remaining ~98 % is search and wrapper overhead. Under JAX the
likelihood share is ~10⁻⁴, so every row above it in the ranked table is the whole cost.
Searches without a share (optimisers, emcee, zeus, NUTS) recorded none in wave 1.

## Candidate one-PR prompts (for the human / profiling conductor; not filed here)

1. Nautilus checkpoint cadence under PyAutoFit (rank 2): measure the write cost against
   `iterations_per_full_update` / an opt-out, preserving resume.
2. Visualisation on a fast likelihood (rank 4): the end-of-fit visualise is 1.2 s here.
3. Bound geometry (rank 3) and NN training (rank 1) stay nautilus-side and ruled
   (findings §5); revisit only with new evidence.

Filed as a Pulse campaign entry: PyAutoPulse `tasks/autofit_profiling_bootstrap.md`
(check-in 2026-10-14).
