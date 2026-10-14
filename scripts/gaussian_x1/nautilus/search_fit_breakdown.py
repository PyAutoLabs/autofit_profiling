"""
gaussian_x1 · Nautilus · ``search.fit`` breakdown
=================================================

Where the wall time of one PyAutoFit ``search.fit`` goes, on a likelihood fast enough
that the overhead has nowhere to hide. Axis: ``breakdown``.

Ported from ``autofit_workspace_developer/searches_minimal/nautilus_profile.py``
(scenario 1, the 3-parameter Gaussian fit; pre-delete commit recorded in the
autofit_workspace_developer retirement PR). The original timed a bare
``nautilus.Sampler``; this cell times the full ``af.Nautilus(...).fit(model, analysis)``
so the PyAutoFit side of the fit (samples output, visualisation, checkpointing, setup
and result construction) is measured too. The findings it reproduces live in
``wiki/campaigns/search_fit_breakdown.md``.

Components (exclusive wall time — a nested call is charged to the innermost one, see
``scripts/misc/profiling/exclusive_timer.py``):

    nn_training          nautilus NeuralNetworkEmulator.train (sklearn MLP ensembles)
    bound_geometry       NautilusBound.compute minus NN training (ellipsoid unions, volume MC)
    proposal_sampling    Sampler.sample_shell
    likelihood           Sampler.evaluate_likelihood (prior transform + PyAutoFit Fitness)
    checkpoint_io        Sampler.write / write_shell_update (nautilus HDF5 checkpoint)
    sampler_bookkeeping  the rest of Sampler.run (shell updates, status, bookkeeping)
    samples_output       PyAutoFit pre/post-fit output, samples, latents, summary
    visualisation        PyAutoFit SearchUpdater.visualize
    other                wall minus all of the above (setup, result construction)

Two variants, both run by default:

    search_fit    the epic-1 measurement: one ``af.Nautilus(...).fit`` → ``breakdown.json``
    bare_sampler  the findings page's own measurement re-run (bare ``nautilus.Sampler``,
                  no PyAutoFit, no checkpoint file) → ``breakdown_bare_sampler.json``.
                  Its shares of wall must reproduce the findings page's NN training,
                  bound geometry and likelihood shares within ``TOLERANCE``; the exit
                  status is non-zero when they do not.

Deterministic in its sampling (``seed=1``: every repeat makes the same likelihood calls);
wall times vary between sessions (the findings page measured ±40 % on this laptop),
so each variant reports the median over ``--repeats`` fresh fits.

Run (laptop CPU; no RAL):

    source activate.sh
    python scripts/gaussian_x1/nautilus/search_fit_breakdown.py [--repeats 3]

Writes ``results/gaussian_x1/nautilus/breakdown*.json``; then rebuild the catalogue with
``python scripts/misc/tooling/build_catalogue.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "ruff.toml").is_file():
            return parent
    raise RuntimeError("ruff.toml root sentinel not found above this script")


ROOT = _repo_root()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "misc")]

import autofit as af
import numpy as np
from profiling.exclusive_timer import ExclusiveTimer, shares, within_tolerance

CELL = "gaussian_x1/nautilus/search_fit_breakdown"
RESULT = ROOT / "results" / "gaussian_x1" / "nautilus" / "breakdown.json"
OUTPUT = ROOT / "output" / "gaussian_x1" / "nautilus" / "search_fit_breakdown"

SEARCH_SETTINGS = {"n_live": 200, "n_networks": 4, "seed": 1, "number_of_cores": 1}

SAMPLER_COMPONENTS = (
    "nn_training",
    "bound_geometry",
    "proposal_sampling",
    "likelihood",
    "checkpoint_io",
    "sampler_bookkeeping",
)
FIT_COMPONENTS = ("samples_output", "visualisation")
COMPONENTS = SAMPLER_COMPONENTS + FIT_COMPONENTS

# The anatomy this cell re-homes: nautilus_bottleneck_findings.md §1, 3-parameter
# Gaussian fit column, shares of the bare sampler's wall. Compared with this cell's
# sampler-internal shares (component / Sampler.run time).
FINDINGS_SHARES = {"nn_training": 0.517, "bound_geometry": 0.252, "likelihood": 0.143}
TOLERANCE = 0.10  # absolute, per share (10 percentage points)
FINDINGS_SOURCE = (
    "autofit_workspace_developer/searches_minimal/nautilus_bottleneck_findings.md §1 "
    "(nautilus 1.0.5, n_live=200, seed=1, bare Sampler, laptop CPU)"
)


# ----------------------------------------------------------------------------- data ---


class Gaussian:
    def __init__(self, centre=30.0, normalization=1.0, sigma=5.0):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_from(self, xvalues):
        return (self.normalization / (self.sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * np.square((xvalues - self.centre) / self.sigma)
        )


def make_dataset():
    """The findings page's 1-D Gaussian (identical data and seed)."""
    np.random.seed(1)
    xvalues = np.arange(100)
    data = Gaussian(centre=50.0, normalization=25.0, sigma=10.0).model_data_from(xvalues)
    noise_map = np.full(data.shape, 0.01)
    data = data + np.random.normal(0.0, 0.01, data.shape)
    return xvalues, data, noise_map


class AnalysisGaussianX1(af.Analysis):
    def __init__(self, xvalues, data, noise_map):
        super().__init__()
        self.xvalues = xvalues
        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        model_data = instance.model_data_from(xvalues=self.xvalues)
        chi_squared = np.sum(((self.data - model_data) / self.noise_map) ** 2.0)
        return float(-0.5 * chi_squared)


def make_model():
    model = af.Model(Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
    model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
    return model


# ------------------------------------------------------------------ instrumentation ---


def instrument(timer: ExclusiveTimer):
    """Monkey-patch the timed methods; returns a callable that restores them."""
    from autofit.non_linear.search.abstract_search import NonLinearSearch
    from autofit.non_linear.search.updater import SearchUpdater
    from nautilus import Sampler
    from nautilus.bounds import NautilusBound
    from nautilus.neural import NeuralNetworkEmulator

    patches = []

    def patch(owner, name, component, classmethod_=False):
        original = owner.__dict__[name]
        func = original.__func__ if classmethod_ else original
        wrapped = timer.wrap(component, func)
        setattr(owner, name, classmethod(wrapped) if classmethod_ else wrapped)
        patches.append((owner, name, original))

    patch(NeuralNetworkEmulator, "train", "nn_training", classmethod_=True)
    patch(NautilusBound, "compute", "bound_geometry", classmethod_=True)
    patch(Sampler, "sample_shell", "proposal_sampling")
    patch(Sampler, "evaluate_likelihood", "likelihood")
    patch(Sampler, "write", "checkpoint_io")
    patch(Sampler, "write_shell_update", "checkpoint_io")
    patch(Sampler, "run", "sampler_bookkeeping")
    for name in ("_save_samples", "_compute_latent_samples", "_profile_and_summarize"):
        patch(SearchUpdater, name, "samples_output")
    patch(SearchUpdater, "visualize", "visualisation")
    for name in ("pre_fit_output", "post_fit_output"):
        patch(NonLinearSearch, name, "samples_output")

    def restore():
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)

    return restore


# ----------------------------------------------------------------------- provenance ---


def _git(path: Path, *args) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def software():
    import autofit
    import nautilus
    import scipy
    import sklearn

    fit_dir = Path(autofit.__file__).resolve().parents[1]
    sw = {
        "PyAutoFit": _git(fit_dir, "rev-parse", "HEAD"),
        "nautilus-sampler": nautilus.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
    }
    try:
        import autonerves

        sw["PyAutoNerves"] = _git(
            Path(autonerves.__file__).resolve().parents[1], "rev-parse", "HEAD"
        )
    except ImportError:
        pass
    # A source checkout's autofit.__version__ is stale (it is stamped at release), so the
    # library version is the nearest release tag of the checkout that was measured.
    tag = _git(fit_dir, "describe", "--tags", "--abbrev=0")
    return (
        {k: v for k, v in sw.items() if v},
        tag or autofit.__version__,
        _git(fit_dir, "status", "--porcelain"),
    )


# --------------------------------------------------------------------------- runner ---


def run_once(timer: ExclusiveTimer, xvalues, data, noise_map):
    shutil.rmtree(OUTPUT, ignore_errors=True)  # a fresh fit, never a resume
    model = make_model()
    analysis = AnalysisGaussianX1(xvalues=xvalues, data=data, noise_map=noise_map)
    search = af.Nautilus(
        path_prefix=str(Path("gaussian_x1") / "nautilus"),
        name="search_fit_breakdown",
        verbose=False,
        **SEARCH_SETTINGS,
    )
    calls = {"n": 0}
    original = analysis.log_likelihood_function

    def counted(instance):
        calls["n"] += 1
        return original(instance)

    analysis.log_likelihood_function = counted

    timer.reset()
    t0 = time.perf_counter()
    result = search.fit(model=model, analysis=analysis)
    wall = time.perf_counter() - t0

    seconds = shares(timer.totals, wall, COMPONENTS)
    sampler_run = sum(seconds[c] for c in SAMPLER_COMPONENTS)
    samples = result.samples
    return {
        "wall_s": wall,
        "components_s": seconds,
        "sampler_run_s": sampler_run,
        "likelihood_function_calls": calls["n"],
        "component_calls": dict(sorted(timer.counts.items())),
        "max_log_likelihood": float(max(samples.log_likelihood_list)),
        "log_evidence": float(samples.log_evidence),
    }


def run_bare_once(timer: ExclusiveTimer, xvalues, data, noise_map):
    """The findings page's own measurement: a bare ``nautilus.Sampler`` (no PyAutoFit
    search, no checkpoint file), same data, priors, ``n_live`` and seed."""
    from nautilus import Sampler

    model = make_model()
    calls = {"n": 0}

    def prior_transform(cube):
        return np.array(model.vector_from_unit_vector(cube))

    def log_likelihood(params):
        calls["n"] += 1
        instance = model.instance_from_vector(vector=params)
        residual = data - instance.model_data_from(xvalues=xvalues)
        return float(-0.5 * np.sum((residual / noise_map) ** 2.0))

    sampler = Sampler(
        prior=prior_transform,
        likelihood=log_likelihood,
        n_dim=model.prior_count,
        n_live=SEARCH_SETTINGS["n_live"],
        n_networks=SEARCH_SETTINGS["n_networks"],
        seed=SEARCH_SETTINGS["seed"],
    )
    timer.reset()
    t0 = time.perf_counter()
    sampler.run(verbose=False)
    wall = time.perf_counter() - t0
    seconds = shares(timer.totals, wall, SAMPLER_COMPONENTS)
    _, _, log_l = sampler.posterior()
    return {
        "wall_s": wall,
        "components_s": seconds,
        "sampler_run_s": wall,
        "likelihood_function_calls": calls["n"],
        "component_calls": dict(sorted(timer.counts.items())),
        "max_log_likelihood": float(np.max(log_l)),
        "log_evidence": float(sampler.log_z),
    }


def summarise(repeats, components):
    keys = (*components, "other")
    wall = statistics.median(r["wall_s"] for r in repeats)
    comp = {c: statistics.median(r["components_s"][c] for r in repeats) for c in keys}
    share = {
        c: statistics.median(r["components_s"][c] / r["wall_s"] for r in repeats) for c in keys
    }
    internal = {
        c: statistics.median(r["components_s"][c] / r["sampler_run_s"] for r in repeats)
        for c in SAMPLER_COMPONENTS
    }
    return wall, comp, share, internal


VARIANTS = {
    "search_fit": {
        "result": RESULT,
        "setup_id": "gaussian_x1/nautilus/search_fit",
        "label": "af.Nautilus search.fit (PyAutoFit, checkpointing and output on)",
        "runner": run_once,
        "components": COMPONENTS,
        "method_id": "search-fit-exclusive-component-wall",
    },
    "bare_sampler": {
        "result": RESULT.with_name("breakdown_bare_sampler.json"),
        "setup_id": "gaussian_x1/nautilus/bare_sampler",
        "label": "bare nautilus.Sampler (the findings page's measurement; no PyAutoFit, no checkpoint)",
        "runner": run_bare_once,
        "components": SAMPLER_COMPONENTS,
        "method_id": "bare-sampler-exclusive-component-wall",
    },
}


def measure(variant, n_repeats, timer, dataset):
    spec = VARIANTS[variant]
    repeats = []
    for i in range(n_repeats):
        row = spec["runner"](timer, *dataset)
        print(
            f"{variant} repeat {i + 1}/{n_repeats}: wall {row['wall_s']:.2f} s, "
            f"{row['likelihood_function_calls']} likelihood calls"
        )
        repeats.append(row)
    wall, comp, share, internal = summarise(repeats, spec["components"])
    calls = sorted({r["likelihood_function_calls"] for r in repeats})
    # The findings page reports shares of the bare sampler's wall; the search.fit
    # variant is compared on the same sampler-internal basis, for information only.
    basis = share if variant == "bare_sampler" else internal
    ok, deltas = within_tolerance(basis, FINDINGS_SHARES, TOLERANCE)
    return {
        "variant": variant,
        "label": spec["label"],
        "setup_id": spec["setup_id"],
        "method_id": spec["method_id"],
        "wall_s": wall,
        "components_s": comp,
        "shares_of_wall": share,
        "sampler_internal_shares": internal,
        "likelihood_function_calls": calls[0] if len(calls) == 1 else calls,
        "deterministic_calls": len(calls) == 1,
        "reference": {
            "source": FINDINGS_SOURCE,
            "shares": FINDINGS_SHARES,
            "compared": "shares_of_wall"
            if variant == "bare_sampler"
            else "sampler_internal_shares",
            "tolerance_abs": TOLERANCE,
            "deltas": deltas,
            "within_tolerance": ok,
            "gates_reproduction": variant == "bare_sampler",
        },
        "repeats": repeats,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args(argv)

    af.conf.instance.push(new_path=ROOT / "config", output_path=ROOT / "output")
    dataset = make_dataset()
    timer = ExclusiveTimer()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    restore = instrument(timer)
    try:
        measured = [measure(v, args.repeats, timer, dataset) for v in variants]
    finally:
        restore()

    sw, library_version, dirty = software()
    measured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = 0
    for m in measured:
        row = {
            "schema": "autofit-profiling-breakdown",
            "version": 1,
            "cell": CELL,
            "script": f"scripts/{CELL}.py",
            "axis": "breakdown",
            "unit": "s",
            "setup": {
                "id": m.pop("setup_id"),
                "label": m.pop("label"),
                "dataset": "gaussian_x1",
                "dataset_description": "1-D Gaussian, 100 pixels, noise 0.01, numpy seed 1 (findings page scenario 1)",
                "model": "Gaussian (centre, normalization, sigma), 3 free parameters, uniform priors",
                "search": "Nautilus",
                "search_settings": SEARCH_SETTINGS,
                "likelihood": "numpy, no JAX",
                "config": "config/general.yaml (library output defaults; version check off)",
            },
            "method": {
                "id": m.pop("method_id"),
                "statistic": "median",
                "repetitions": args.repeats,
                "warmup": "none; each repeat is a fresh fit (output removed, never a resume)",
                "synchronization": "none (numpy, single process)",
                "cache_state": "warm process: imports and first-touch costs fall in the first repeat",
                "timer": "time.perf_counter, exclusive per component (scripts/misc/profiling/exclusive_timer.py)",
            },
            **m,
            "identity": {
                "device": "cpu",
                "backend": "numpy",
                "precision": "float64",
                "library_version": library_version,
                "software": sw,
            },
            "provenance": {
                "host": platform.node(),
                "measured_at": measured_at,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "pyautofit_checkout_dirty": bool(dirty),
            },
        }
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "variant",
                        "wall_s",
                        "shares_of_wall",
                        "sampler_internal_shares",
                        "reference",
                    )
                },
                indent=2,
            )
        )
        if row["reference"]["gates_reproduction"] and not row["reference"]["within_tolerance"]:
            status = 1
        if not args.no_write:
            path = VARIANTS[row["variant"]]["result"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, indent=2) + "\n")
            print(f"wrote {path.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
