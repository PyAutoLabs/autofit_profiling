"""Shared pieces of the gaussian_1d_ep EP / graphical baseline cells.

Ported from ``autofit_workspace_developer/{ep,graphical}/{simulator,util,sanity}.py``
(byte-identical between the two packages at commit ``07b92ace``, last touched in
``98f260e5`` / ``27f0a3a9``). What changed in the port, and nothing else:

- datasets are simulated into the gitignored ``output/gaussian_1d_ep/dataset/`` on every
  run (seeded, so identical to the committed developer datasets) instead of a committed
  ``dataset/`` tree, and the per-dataset quick-look PNG is no longer drawn;
- the profile writer is replaced by one result row per cell variant under ``results/``
  carrying host, device, backend, precision, software revisions and the sanity verdict;
- every fit starts from an empty output directory (never a resume).

The simulator, the model, the priors, the searches and the sanity rules are unchanged.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import autofit as af
import numpy as np
from autofit import to_dict

CELL_DIR = Path(__file__).resolve().parent
ROOT = CELL_DIR.parents[1]
DATASET = "gaussian_1d_ep"
SAMPLE_NAME = "toy__gaussian_x1__low_snr"
SAMPLE_PATH = ROOT / "output" / DATASET / "dataset" / SAMPLE_NAME
CONFIG_PATH = CELL_DIR / "config"
N_VALUES = (3, 10, 30)

# Toy 1D Gaussian truth -- shared across all datasets in the sample. The fits recover
# `centre` as a shared parameter; `normalization` and `sigma` are dataset-specific (but
# identical in truth here). Verbatim from the developer simulator.
TRUE_CENTRE = 50.0
TRUE_NORMALIZATION = 0.5
TRUE_SIGMA = 5.0
N_PIXELS = 100
SIGNAL_TO_NOISE_RATIO = 25.0
NOISE_SIGMA = 1.0 / SIGNAL_TO_NOISE_RATIO

DATASET_DESCRIPTION = (
    "N independent 1-D Gaussians (100 pixels, S/N 25, noise 0.04) sharing centre 50; "
    "dataset i is numpy default_rng(seed=i); simulated by scripts/gaussian_1d_ep/shared.py"
)
MODEL_DESCRIPTION = (
    "per dataset af.ex.Gaussian with a shared GaussianPrior(50, 30) centre and "
    "TruncatedGaussianPrior normalization (0.5, 2, >0) and sigma (5, 5, >0): 1 + 2N free parameters"
)


# ------------------------------------------------------------------------ simulator ---


def simulate_dataset(dataset_index: int, dataset_path: Path) -> float:
    xvalues = np.arange(N_PIXELS, dtype=float)
    gaussian = af.ex.Gaussian(
        centre=TRUE_CENTRE, normalization=TRUE_NORMALIZATION, sigma=TRUE_SIGMA
    )
    model_data = gaussian.model_data_from(xvalues=xvalues)

    rng = np.random.default_rng(seed=dataset_index)
    noise = rng.normal(0.0, NOISE_SIGMA, N_PIXELS)
    data = model_data + noise
    noise_map = NOISE_SIGMA * np.ones(N_PIXELS)

    dataset_path.mkdir(parents=True, exist_ok=True)
    af.util.numpy_array_to_json(array=data, file_path=dataset_path / "data.json", overwrite=True)
    af.util.numpy_array_to_json(
        array=noise_map, file_path=dataset_path / "noise_map.json", overwrite=True
    )

    # Truth-evaluated log likelihood: what a perfect-knowledge oracle would compute for
    # THIS noisy realisation. A fit's max log likelihood should sit at or slightly above.
    analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=False)
    truth_log_likelihood = float(analysis.log_likelihood_function(instance=gaussian))

    ground_truth = {
        "centre": TRUE_CENTRE,
        "normalization": TRUE_NORMALIZATION,
        "sigma": TRUE_SIGMA,
        "n_pixels": N_PIXELS,
        "signal_to_noise_ratio": SIGNAL_TO_NOISE_RATIO,
        "noise_sigma": NOISE_SIGMA,
        "truth_log_likelihood": truth_log_likelihood,
    }
    (dataset_path / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    try:
        (dataset_path / "model.json").write_text(json.dumps(to_dict(gaussian), indent=2))
    except (TypeError, ValueError):
        pass
    return truth_log_likelihood


def simulate(total_datasets: int = max(N_VALUES)) -> list[float]:
    """Write ``dataset_0 .. dataset_{total-1}`` under SAMPLE_PATH; return the truth logLs."""
    return [simulate_dataset(i, SAMPLE_PATH / f"dataset_{i}") for i in range(total_datasets)]


def load_datasets(n: int):
    """Load the first ``n`` datasets as (data_list, noise_map_list, truths)."""
    data_list, noise_map_list, truths = [], [], []
    for i in range(n):
        path = SAMPLE_PATH / f"dataset_{i}"
        data_list.append(af.util.numpy_array_from_json(file_path=path / "data.json"))
        noise_map_list.append(af.util.numpy_array_from_json(file_path=path / "noise_map.json"))
        truths.append(json.loads((path / "ground_truth.json").read_text()))
    return data_list, noise_map_list, truths


def make_model_list(n: int):
    """The developer model: one Gaussian per dataset, a single shared centre prior."""
    centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)
    model_list = []
    for _ in range(n):
        gaussian = af.Model(af.ex.Gaussian)
        gaussian.centre = centre_shared_prior
        gaussian.normalization = af.TruncatedGaussianPrior(mean=0.5, sigma=2.0, lower_limit=0.0)
        gaussian.sigma = af.TruncatedGaussianPrior(mean=5.0, sigma=5.0, lower_limit=0.0)
        model_list.append(af.Collection(gaussian=gaussian))
    return centre_shared_prior, model_list


# --------------------------------------------------------------------------- sanity ---


def check_parameter_recovery(name, recovered_mean, recovered_sigma, truth, k=3.0):
    """Pass iff |recovered_mean - truth| <= k * recovered_sigma."""
    if recovered_sigma is None or recovered_sigma == 0.0:
        passed = abs(recovered_mean - truth) < 1e-6
        msg = (
            f"{name}: mean={recovered_mean:.4g} truth={truth:.4g} "
            f"sigma=0 (deterministic) -> {'PASS' if passed else 'FAIL'}"
        )
        return passed, msg
    z = abs(recovered_mean - truth) / recovered_sigma
    passed = z <= k
    msg = (
        f"{name}: mean={recovered_mean:.4g} ± {recovered_sigma:.4g} "
        f"truth={truth:.4g} |z|={z:.2f} (k={k}) -> {'PASS' if passed else 'FAIL'}"
    )
    return passed, msg


def check_max_log_likelihood(max_ll, truth_ll_sum, n_datasets, tolerance_per_dataset=5.0):
    """Pass iff max_ll >= truth_ll_sum - n_datasets * tolerance."""
    threshold = truth_ll_sum - n_datasets * tolerance_per_dataset
    passed = max_ll >= threshold
    msg = (
        f"max_log_likelihood: {max_ll:.3f} vs truth_sum={truth_ll_sum:.3f} "
        f"(N={n_datasets}, tol={tolerance_per_dataset}/dset, "
        f"threshold={threshold:.3f}) -> {'PASS' if passed else 'FAIL'}"
    )
    return passed, msg


def summarise_sanity(results, header):
    """Print the PASS/FAIL table (never raises) and return (aggregate, checks)."""
    n_pass = sum(1 for p, _ in results if p)
    print(f"\n=== {header} ({n_pass}/{len(results)} pass) ===")
    for passed, msg in results:
        print(("  ok  " if passed else "  !!  ") + msg)
    aggregate = n_pass == len(results)
    print(f"SANITY: {'PASS' if aggregate else 'FAIL'} ({n_pass}/{len(results)} checks)\n")
    checks = [{"passed": bool(p), "message": m} for p, m in results]
    return aggregate, checks


# ------------------------------------------------------------------------- measuring ---


def setup_autofit(method: str) -> Path:
    """Point autofit at the cell config and a per-method output root; return that root."""
    output_root = ROOT / "output" / DATASET / method
    af.conf.instance.push(new_path=CONFIG_PATH, output_path=output_root)
    return output_root


def fresh_output(output_root: Path, run_name: str) -> Path:
    run_dir = output_root / SAMPLE_NAME / run_name
    shutil.rmtree(run_dir, ignore_errors=True)  # a fresh fit, never a resume
    return run_dir


def peak_rss_kib() -> int:
    """Process high-water RSS. ``ru_maxrss`` is KiB on Linux (the measured host)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def output_dir_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


DRIFT_POLICY = "runtime-drift-2x-1ms"
DRIFT_RATIO = 2.0
DRIFT_FLOOR_S = 0.001


def drift(reference_s: float, measured_s: float) -> dict:
    """The epic's ``runtime-drift-2x-1ms`` rule (autolens_profiling build_dashboard.py)."""
    ratio = measured_s / reference_s
    delta = abs(measured_s - reference_s)
    if ratio >= DRIFT_RATIO and delta >= DRIFT_FLOOR_S:
        status = "drifted"
    elif ratio <= 1.0 / DRIFT_RATIO and delta >= DRIFT_FLOOR_S:
        status = "improved"
    else:
        status = "flat"
    return {"policy": DRIFT_POLICY, "ratio": round(ratio, 3), "status": status}


# ------------------------------------------------------------------------ provenance ---


def _git(path: Path, *args) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def software():
    """(software mapping, library version, PyAutoFit checkout dirty?) of the measured stack."""
    import autofit
    import dynesty
    import jax
    import scipy

    fit_dir = Path(autofit.__file__).resolve().parents[1]
    sw = {
        "PyAutoFit": _git(fit_dir, "rev-parse", "HEAD"),
        "dynesty": dynesty.__version__,
        "jax": jax.__version__,
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
    # A source checkout's autofit.__version__ is stale (stamped at release), so the
    # library version is the nearest release tag of the checkout that was measured.
    tag = _git(fit_dir, "describe", "--tags", "--abbrev=0")
    dirty = _git(fit_dir, "status", "--porcelain", "--untracked-files=no")
    return {k: v for k, v in sw.items() if v}, tag or autofit.__version__, bool(dirty)


def load_average() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def precision() -> str:
    import jax

    return "float64" if jax.config.jax_enable_x64 else "float32"


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def baseline_row(
    *,
    method: str,
    n: int,
    label: str,
    search: str,
    search_settings: dict,
    wall_s: float,
    rss_kib: int,
    disk_bytes: int,
    correctness: dict,
    reference: dict,
    sw_before: dict,
    load_before: list[float],
) -> dict:
    """One ``autofit-profiling-baseline`` v1 result row (see build_catalogue.py)."""
    sw, library_version, dirty = software()
    if sw.get("PyAutoFit") != sw_before.get("PyAutoFit"):
        raise RuntimeError(
            "PyAutoFit checkout moved during the measurement "
            f"({sw_before.get('PyAutoFit')} -> {sw.get('PyAutoFit')}); rerun"
        )
    import jax

    return {
        "schema": "autofit-profiling-baseline",
        "version": 1,
        "cell": f"{DATASET}/{method}/baseline",
        "script": f"scripts/{DATASET}/{method}/baseline.py",
        "setup": {
            "id": f"{DATASET}/{method}/N{n}",
            "label": label,
            "dataset": DATASET,
            "dataset_description": DATASET_DESCRIPTION,
            "model": MODEL_DESCRIPTION,
            "n_datasets": n,
            "free_parameters": 1 + 2 * n,
            "data_pixels": N_PIXELS * n,
            "search": search,
            "search_settings": search_settings,
            "likelihood": "af.ex.Analysis(use_jax=True), af.FactorGraphModel(use_jax=True)",
            "config": f"scripts/{DATASET}/config (verbatim from the developer packages)",
        },
        "method": {
            "id": "fresh-process-single-fit",
            "statistic": "single",
            "repetitions": 1,
            "warmup": "none; one fit per fresh process from an empty output directory (never a resume)",
            "synchronization": "none (JAX CPU, results read back to Python floats)",
            "cache_state": "cold process; JAX compilation falls inside the timed fit",
            "timer": "time.perf_counter around the fit call only (simulation, imports and sanity excluded)",
        },
        "wall_s": wall_s,
        "peak_rss_kib": rss_kib,
        "output_dir_bytes": disk_bytes,
        "correctness": correctness,
        "reference": reference,
        "identity": {
            "device": "cpu",
            "backend": "jax",
            "precision": precision(),
            "library_version": library_version,
            "software": sw,
        },
        "provenance": {
            "host": platform.node(),
            "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "jax_platform": jax.default_backend(),
            "pyautofit_checkout_dirty": dirty,
            # Host load (1/5/15-min) just before and just after the fit: a contended host
            # inflates wall times, so the reader can tell a quiet baseline from a busy one.
            "load_average_before": load_before,
            "load_average_after": load_average(),
        },
    }


def reference_block(source: str, ref: dict, wall_s: float, correctness: dict) -> dict:
    """Compare a reproduction against the committed developer baseline for the same N."""
    block = {
        "source": source,
        "wall_s": ref["wall_time_s"],
        "peak_rss_kb": ref["peak_rss_kb"],
        "truth_log_likelihood_sum": ref["truth_log_likelihood_sum"],
        "sanity_pass": ref["sanity_pass"],
        "wall_drift": drift(ref["wall_time_s"], wall_s),
        "truth_log_likelihood_sum_matches": (
            correctness["truth_log_likelihood_sum"] == ref["truth_log_likelihood_sum"]
        ),
        "sanity_pass_matches": correctness["sanity_pass"] == ref["sanity_pass"],
    }
    if "postfix_wall_time_s" in ref:
        block["postfix_wall_s"] = ref["postfix_wall_time_s"]
        block["postfix_source"] = ref["postfix_source"]
        block["postfix_wall_drift"] = drift(ref["postfix_wall_time_s"], wall_s)
    return block


def run_all(script: Path, ns, extra=()) -> int:
    """Run each N in its own fresh process (peak RSS is per process), serially."""
    status = 0
    for n in ns:
        cmd = [sys.executable, str(script), "--n", str(n), *extra]
        print(f"\n### {' '.join(cmd[1:])}", flush=True)
        status |= subprocess.run(cmd, check=False).returncode
    return status
