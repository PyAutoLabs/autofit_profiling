autofit_profiling
=================

Profiling runs and results for [PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit) — where
the time in a `search.fit` goes, and baselines for expectation propagation and graphical
models.

It is the sibling of [autofit_inference](https://github.com/PyAutoLabs/autofit_inference)
(PyAutoFit search benchmarking) and the PyAutoFit counterpart of
[autolens_profiling](https://github.com/PyAutoLabs/autolens_profiling).

**Status:** the first cell — where the time in one Nautilus `search.fit` goes on a fast
1-D Gaussian likelihood — landed in B4a of the
[search-extensibility epic](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md),
with its ranked table in [`wiki/campaigns/search_fit_bottlenecks.md`](wiki/campaigns/search_fit_bottlenecks.md).
The repo reports to [PyAutoPulse](https://github.com/PyAutoLabs/PyAutoPulse) as instance
`fit` (`dashboard/catalogue.json`, `profiling-summary@2`); its own board is published to
GitHub Pages from `dashboard/`. B4b added the expectation-propagation / graphical-model
baselines (N = 3, 10, 30 datasets) and the analytic Gaussian parity benchmark; their
reproduction table is [`wiki/campaigns/ep_graphical_baselines.md`](wiki/campaigns/ep_graphical_baselines.md).

Agents and contributors: start with [`AGENTS.md`](AGENTS.md).
