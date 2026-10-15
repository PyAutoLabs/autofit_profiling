# autofit_profiling — Agent Instructions

This repo is the single home for **PyAutoFit profiling runs and results**: where the time in
a `search.fit` goes (the per-phase breakdown of a fit — likelihood calls, sampler overhead,
output and visualization), and baselines for expectation propagation (EP) and graphical
models. It is the sibling of
[`autofit_inference`](https://github.com/PyAutoLabs/autofit_inference), which owns
*which search finds the right answer*, and the PyAutoFit counterpart of
[`autolens_profiling`](https://github.com/PyAutoLabs/autolens_profiling).

It is a collection of standalone scripts, **not** an installable package — there is no
`pyproject.toml`. These are the canonical, agent-agnostic instructions; `README.md` is the
human-facing overview.

**Status:** born 2026-10-07 (PyAutoMind#492, phase B1 of the `search-extensibility`
epic). B4a (2026-10-14) added the first cell — the Nautilus `search.fit` breakdown on
gaussian_x1 — the `profiling-summary@2` exporter and the Pages board; the repo reports to
PyAutoPulse as instance `fit`. B4b (2026-10-14) ported the EP / graphical-model baselines
(`scripts/gaussian_1d_ep/`) and added the analytic Gaussian parity cell
(`scripts/analytic_gaussian/`); findings in `wiki/campaigns/ep_graphical_baselines.md`.

## Where this repo is going

The work arrives in later phases of the `search-extensibility` epic, whose ledger is
`PyAutoMind/draft/research/autofit/search_extensibility_epic.md`
([on GitHub](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md)):

- **B4a** — the `search.fit` breakdown exporter and the first bottleneck table. From B4a
  this repo reports to [PyAutoPulse](https://github.com/PyAutoLabs/PyAutoPulse) as instance
  `fit` (`profiling-summary@2`). This adopts Pulse's task `autofit_profiling_bootstrap`.
- **B4b** — the EP / graphical-model baseline port (shipped; pinned to the B4a laptop host).

Do not add scripts, results or exporters outside the phase that owns them.
PyAutoPulse validates the published summary contract; the Brain's profiling conductor
judges it. Nothing here issues a verdict.

## Repository Structure

```
ruff.toml        lint config AND the root sentinel (leaf scripts walk up to it)
activate.sh      RAL shared venv + PYTHONPATH (PyAutoNerves, PyAutoFit only)
config/          PyAutoConf overrides (library output defaults kept on purpose)
scripts/<dataset>/<search>/<cell>.py   measurement cells, e.g. gaussian_x1/nautilus/search_fit_breakdown.py
scripts/gaussian_1d_ep/{ep,graphical}/baseline.py  EP / graphical N3/N10/N30 (shared.py, config/ beside)
scripts/analytic_gaussian/parity/benchmark.py     runs autofit_workspace_test's analytic parity script
scripts/misc/profiling/                stdlib helpers shared by cells (exclusive component timer)
scripts/misc/tooling/build_catalogue.py  results/ -> dashboard/{catalogue.json,index.html,state.json}
scripts/misc/test/                     stdlib-only tests (no PyAuto install)
results/         committed result rows, one file per cell variant (schemas autofit-profiling-breakdown,
                 autofit-profiling-baseline, autofit-profiling-parity; build_catalogue.py documents each)
dashboard/       generated: catalogue.json (profiling-summary@2), index.html, state.json
wiki/campaigns/  findings pages; search_fit_bottlenecks.md is epic 1's ranked table
hpc/             hpc/sync (laptop-side RAL driver) + the batch_cpu submit template
```

## Producing a result and publishing it

1. `source activate.sh`, run a cell (e.g. `python scripts/gaussian_x1/nautilus/search_fit_breakdown.py`);
   it writes its row under `results/`. Laptop CPU cells need no RAL.
2. `python scripts/misc/tooling/build_catalogue.py --validate-with ../PyAutoPulse` renders
   `dashboard/` and validates the catalogue with the Pulse reader; commit rows and
   `dashboard/` together. The renderer is deterministic (its stamp is the newest
   evidence time), so `--check` is the lint gate.
3. On merge, `.github/workflows/pages_dashboard.yml` publishes `dashboard/` to Pages
   (`https://pyautolabs.github.io/autofit_profiling/`) and dispatches `pulse-refresh` to
   PyAutoPulse (payload repo / sha / `dashboard/catalogue.json`; needs the org secret
   `PAT_PYAUTOLABS`, warns and skips without it). `dashboard/state.json` is this
   project's own badge feed; the Brain cockpit reads the PyAutoPulse organ feed.

Every catalogue record is declared `unreviewed`: the producer measures, the Brain's
profiling conductor judges.

## RAL

The RAL copy of this repo is a **git clone** at `/mnt/ral/jnightin/profiling/autofit_profiling`.
Code goes over with `git pull` on the login node; results come back with `hpc/sync pull`
(`cp hpc/sync.conf.example hpc/sync.conf` first; `sync.conf` is gitignored). Runs here are
CPU runs on `--partition=ral` — there is no `cpu` partition on RAL, and CPU arrays never go
on `gpu`. The PyAuto* libraries resolve from the shared checkouts on `PYTHONPATH`
(`source activate.sh`) and are updated with `HPCPullPyAuto`, never pip-installed. Full
detail in [`hpc/README.md`](hpc/README.md).

## Testing

The PR gate is `.github/workflows/lint.yml` on Python 3.12: `ruff check .`,
`ruff format --check .`, `pytest -q scripts/misc/test`, `build_catalogue.py --check
--validate-with _pulse` (PyAutoPulse checked out beside) and a `lychee` link check over
`README.md` and `AGENTS.md`. It installs no PyAuto library and runs no measurement cell.

## Related Repos

- `../autofit_inference` — PyAutoFit search benchmarking.
- `../PyAutoFit` — the library being profiled (plus `../PyAutoNerves` on `PYTHONPATH`).
- `../autolens_profiling` — the PyAutoLens sibling this repo is modelled on.
- `../PyAutoPulse` — the cross-project profiling dashboard.

## Task Workflows

Keep `ruff check .` and `ruff format --check .` clean, do not commit machine-specific
absolute paths, and flag any change that affects PyAutoFit itself in your PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->

<!-- repos_sync:filing:begin -->
## Where to file

Questions, help with code or an analysis, ideas, bug reports and results from a
user or collaborator — or an agent acting for one — go to
<https://github.com/orgs/PyAutoLabs/discussions> in the matching category
(Help & Questions, Ideas & Proposals, Bugs & Errors, Show and tell;
Announcements is maintainers-only), never to this repo's Issues. An agent never
runs `gh issue create` for such a report: it drafts the title, category and
body and hands them to the human (sessions cannot create Discussions). Only the
development flow — Mind prompt → `/start_dev` → `/create_issue` → one issue per
task → PR — opens issues here. Why: `PyAutoMind/policy/community_surface.md`.
<!-- repos_sync:filing:end -->

<!-- repos_sync:standards:begin -->
## Shared standards

Before changing a shared interface, consult the applicable
[organism standard](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/docs/standards.md)
on demand, identify affected consumers, and validate their adoption. Change
generated guidance at its canonical source and regenerate.
<!-- repos_sync:standards:end -->
