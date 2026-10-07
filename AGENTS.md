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

**Status:** skeleton born 2026-10-07 (PyAutoMind#492), phase B1 of the
`search-extensibility` epic. There is no harness, no exporter and no result yet.

## Where this repo is going

The work arrives in later phases of the `search-extensibility` epic, whose ledger is
`PyAutoMind/draft/research/autofit/search_extensibility_epic.md`
([on GitHub](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md)):

- **B4a** — the `search.fit` breakdown exporter and the first bottleneck table. From B4a
  this repo reports to [PyAutoPulse](https://github.com/PyAutoLabs/PyAutoPulse) as instance
  `fit` (`profiling-summary@2`). This adopts Pulse's task `autofit_profiling_bootstrap`.
- **B4b** — the EP / graphical-model baseline port.

Until then, do not add scripts, results or exporters outside the phase that owns them.
PyAutoPulse validates the published summary contract; the Brain's profiling conductor
judges it. Nothing here issues a verdict.

## Repository Structure

```
ruff.toml        lint config AND the root sentinel (leaf scripts walk up to it)
activate.sh      RAL shared venv + PYTHONPATH (PyAutoNerves, PyAutoFit only)
hpc/             hpc/sync (laptop-side RAL driver) + the batch_cpu submit template
```

`scripts/`, `results/` and the summary exporter arrive in B4a.

## RAL

The RAL copy of this repo is a **git clone** at `/mnt/ral/jnightin/autofit_profiling`.
Code goes over with `git pull` on the login node; results come back with `hpc/sync pull`
(`cp hpc/sync.conf.example hpc/sync.conf` first; `sync.conf` is gitignored). Runs here are
CPU runs on `--partition=ral` — there is no `cpu` partition on RAL, and CPU arrays never go
on `gpu`. The PyAuto* libraries resolve from the shared checkouts on `PYTHONPATH`
(`source activate.sh`) and are updated with `HPCPullPyAuto`, never pip-installed. Full
detail in [`hpc/README.md`](hpc/README.md).

## Testing

The PR gate is `.github/workflows/lint.yml` on Python 3.12: `ruff check .`,
`ruff format --check .` and a `lychee` link check over `README.md` and `AGENTS.md`. It
installs no PyAuto library and runs no script.

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
