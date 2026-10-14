# Nautilus `search.fit` breakdown — gaussian_x1 (findings)

Cell: [`scripts/gaussian_x1/nautilus/search_fit_breakdown.py`](../../scripts/gaussian_x1/nautilus/search_fit_breakdown.py)
· rows: [`results/gaussian_x1/nautilus/breakdown.json`](../../results/gaussian_x1/nautilus/breakdown.json)
(the `search.fit`) and [`breakdown_bare_sampler.json`](../../results/gaussian_x1/nautilus/breakdown_bare_sampler.json)
(the reproduction) · catalogue: `dashboard/catalogue.json` (`profiling-summary@2`, Pulse instance `fit`).

Re-homed on 2026-10-14 (search-extensibility B4a, autofit_profiling#2) from
`autofit_workspace_developer/searches_minimal/{nautilus_profile.py,nautilus_bottleneck_findings.md}`
(research task Jammy2211/autofit_workspace_developer#18, verdict 2026-07-09). §1–§5 below
are the original findings, kept verbatim in substance; §0 is the B4a re-measurement.

## 0. B4a re-measurement (2026-10-14, laptop CPU)

Laptop `DESKTOP-H143S82` (WSL2, 8 cores), numpy fp64, PyAutoFit `911fe05e1`
(2026.10.12.1+1), nautilus 1.0.5, scikit-learn 1.5.1, `n_live=200`, `n_networks=4`,
`seed=1`, one core; median of 3 fresh fits. Component times are exclusive (a nested call
is charged to the innermost component).

**Reproduction (bare `nautilus.Sampler`, the original measurement):** wall 22.5 s,
16 400 likelihood calls in every repeat, log Z −57.455 (the original reference value).

| component | findings §1 | B4a | Δ |
|---|---:|---:|---:|
| NN ensemble training | 51.7 % | 53.8 % | +2.1 pp |
| bound geometry | 25.2 % | 25.3 % | +0.1 pp |
| likelihood (incl. transform) | 14.3 % | 11.6 % | −2.7 pp |
| proposal sampling | 5.2 % | 5.7 % | +0.5 pp |

All three gated shares (NN training, bound geometry, likelihood) are within the cell's
stated tolerance of ±10 percentage points; the cell exits non-zero when they are not.

**The full `af.Nautilus(...).fit`** (what epic 1 profiles): wall 28.4 s, 7 501 likelihood
calls (PyAutoFit's default `n_eff=500` stops earlier than bare nautilus's 10 000), log Z
−57.511. Ranked: NN training 43.4 %, **nautilus checkpoint I/O 24.5 %**, bound geometry
20.4 %, PyAutoFit visualisation 4.1 %, sampler bookkeeping 2.3 %, PyAutoFit samples output
1.7 %, likelihood 1.1 %, other 1.0 %, proposal sampling 0.8 %. The checkpoint I/O is new:
the bare sampler writes no checkpoint, whereas PyAutoFit always hands nautilus
`filepath=checkpoint.hdf5`, and nautilus's 103 HDF5 writes cost ~69 ms each here. The
ranked epic-1 table, joined with autofit_inference's `likelihood_share`, is
[`search_fit_bottlenecks.md`](search_fit_bottlenecks.md).

The likelihood itself is cheaper under PyAutoFit (~43 µs/call through `Fitness` and the
vectorised prior, against ~160 µs/call for the bare script's `instance_from_vector`
callback), so the sampler-internal likelihood share (1.2 %) is *not* a reproduction of
the 14.3 % row — that comparison is recorded in the row for information only.

## 1. Where the time goes (Phase 1, 2026-07)

With a fast likelihood, Nautilus wall time is almost entirely sampler overhead,
and it is **not only the neural networks**:

| component | 3-param Gaussian fit | fast 10-d likelihood (5 µs/call) |
|---|---:|---:|
| NN ensemble training | 51.7 % | 40.2 % |
| bound geometry (ellipsoid unions + 1000-pt volume MC) | 25.2 % | 38.2 % |
| proposal sampling | 5.2 % | 18.0 % |
| likelihood (incl. transform) | 14.3 % | **0.6 %** |

nautilus 1.0.5, `n_live=200`, `seed=1`, laptop CPU (WSL2, 8 cores). Wall times vary
±40 % between sessions (thermal/contention); every comparison is within-run. The fast
10-d scenario was not re-homed (it is not a PyAutoFit `search.fit`).

Mechanics (all in nautilus source): each bound update trains one
`NeuralNetworkEmulator` **per ellipsoid** — `n_networks=4` sklearn MLPs
(hidden layers 100/50/20, `tol=0`, early stop after 10 flat epochs), each
pinned to a single thread (`neural.py`, `threadpool_limits(1)`). Training
parallelises only across the ensemble via the sampler pool (`pool_s`).

**PyAutoFit wrapper finding** (`autofit/non_linear/search/nest/nautilus/search.py`):
`fit_x1_cpu` — the JAX/GPU path — passes `pool=None`, so in exactly the
fast-GPU-likelihood regime every MLP trains serially.

## 2. Zero-code knobs (Phase 2)

All configs preserved quality (log Z within 0.02 nats of baseline both
scenarios, ESS ~10 000, reference log Z −57.5 for the Gaussian fit):

| config | fast 10-d | 3-param fit | note |
|---|---:|---:|---|
| `n_networks=0` (pure ellipsoid bounds) | **3.2×** | **2.0×** | +1–10 % evals; log Z −57.451 vs −57.455 |
| `n_update=2*n_live` (retrain half as often) | 3.0× | — | quality intact |
| `pool=(None, 4)` (parallel MLP training) | 2.3× | 1.1× | `os.fork()` + JAX threads ⇒ deadlock risk |
| `n_networks=2` | 2.2× | — | |
| small MLP `(32,)` | 2.0× | 1.0× | |
| `vectorized=True` (batched callback) | 2.0× | — | the hook a vmapped GPU likelihood needs |
| `n_networks=0` + `n_update=2x` combined | 2.7× | 1.9× | ≈ `n_networks=0` alone; they don't stack |

`n_networks=0` is an explicitly supported nautilus mode (`bounds/neural.py:74`): bounds
fall back to the ellipsoid union, i.e. MultiNest-style region sampling. Its cost is
sampling efficiency — here only +1 % (10-d) to +10–27 % (3-param) more likelihood calls.

**Caveat that decided against it (see §5):** evals-to-ML on the 3-param fit degraded
4278 → 6481 (+50 %) — the sampler takes materially longer to *find the peak* without the
NN bounds, even on an easy unimodal problem. On real curved/degenerate lens posteriors
at 20–30 parameters the ellipsoid-only efficiency loss is expected to be much larger.

## 3. JAX/GPU MLP swap prototype (Phase 3) — negative on CPU

A single jitted JAX program replacing the sklearn ensemble (vmap over the 4 networks,
full-batch Adam, padded shapes, jitted predict) matched sklearn's quality but was
**4–6× slower** on CPU (fast 10-d: 5.3 s → 24–28 s NN time; 3-param fit: 37 s → 118 s+).
NN training is ≤ 52 % of wall, so even free training beats it by at most ~2× — less than
`n_networks=0` delivers for zero code. **Parked**; revisit only if a GPU-node measurement
of a real lens fit contradicts this.

## 4. Can nautilus be JAX-ified wholesale? No.

The sampler's control flow is adaptive and host-side by design (data-dependent while
loops, a growing point store, per-bound sklearn objects, HDF5 checkpointing). Porting
that to XLA is writing a different sampler, which is what NSS already is. The long-term
answer remains a JAX-native sampler swap.

## 5. Verdict (human decision 2026-07-09)

The `n_networks=0` route was **rejected by the maintainer**: for real lens-model sampling
the NN bounds are what keep Nautilus's sampling efficiency high. No PyAutoFit config
change and no GPU validation follow-up were filed. What stands:

1. **The overhead anatomy (§1, reproduced in §0) is the reference** for Nautilus speed-up
   work: NN training 40–54 %, bound geometry 20–38 %; they must be attacked together for
   a win beyond ~2×.
2. **Do not wire a process pool into the JAX path** (`pool=(None, N)`): modest win, and
   `os.fork()` under a JAX-threaded process risks deadlock.
3. **The JAX/GPU MLP swap is parked with a measured negative** (§3).
4. **The long-term answer remains a JAX-native sampler swap** (§4).
5. If Nautilus overhead work is revisited, the next targets in order: bound geometry
   (~0.2–0.9 s/bound; profile `Union.compute`/split vs the 1000-pt volume MC first), then
   early-stopping-preserving training parallelism without process forks. B4a adds a
   third, PyAutoFit-side target: the per-update nautilus checkpoint write (§0).
