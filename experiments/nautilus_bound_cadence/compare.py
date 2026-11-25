"""Paired search.fit comparisons for issue #26. Run from a fresh output directory."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gaussian_x1" / "nautilus"))

import numpy as np
import search_fit_breakdown as baseline
from nautilus import Sampler
from scipy.optimize import least_squares
from scipy.special import logsumexp

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
ARMS = ("default", "doubled")
FIXTURES = ("profile", "symmetric")


def reference(dataset):
    """Integrate the uniform prior by tensor quadrature, refined around the fitted peak."""
    x, data, noise = dataset

    def residual(params):
        return (data - baseline.Gaussian(*params).model_data_from(x)) / noise

    fit = least_squares(residual, [50, 25, 10], bounds=([0, 0, 1e-8], [100, 50, 50]))
    widths = np.sqrt(np.diag(np.linalg.inv(fit.jac.T @ fit.jac)))
    log_evidences = []
    for order in (16, 32, 48):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        axes = []
        for peak, width, upper in zip(fit.x, widths, (100, 50, 50)):
            edges = np.unique(
                np.clip([0, peak - 12 * width, peak, peak + 12 * width, upper], 0, upper)
            )
            grid, mass = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                grid.extend((hi + lo + (hi - lo) * nodes) / 2)
                mass.extend(weights * (hi - lo) / (2 * upper))
            axes.append((np.asarray(grid), np.asarray(mass)))
        (centres, wc), (amps, wa), (sigmas, ws) = axes
        parts = []
        for centre, w in zip(centres, wc):
            basis = np.exp(-0.5 * ((x - centre) / sigmas[:, None]) ** 2) / (
                sigmas[:, None] * np.sqrt(2 * np.pi)
            )
            residuals = (data - amps[:, None, None] * basis) / noise
            ll = -0.5 * np.sum(residuals**2, axis=-1)
            parts.append(np.log(w) + logsumexp(ll, b=wa[:, None] * ws))
        log_evidences.append(float(logsumexp(parts)))
    if abs(log_evidences[-1] - log_evidences[-2]) > 0.001:
        raise RuntimeError("Evidence quadrature failed to converge")
    return {
        "best_log_likelihood": -fit.cost,
        "maximum": fit.x.tolist(),
        "orders": [16, 32, 48],
        "log_evidences": log_evidences,
    }


class SignedGaussian(baseline.Gaussian):
    """Identical positive- and negative-centre peaks under a uniform signed prior."""

    def model_data_from(self, xvalues):
        return baseline.Gaussian(abs(self.centre), self.normalization, self.sigma).model_data_from(
            xvalues
        )


def run_one(directory, fixture, arm, seed, ref):
    if (directory / "output").exists():
        raise FileExistsError("Each child needs a fresh output directory")
    baseline.af.conf.instance.push(new_path=ROOT / "config", output_path=directory / "output")
    model = baseline.make_model()
    if fixture == "symmetric":
        model = baseline.af.Model(
            SignedGaussian,
            normalization=model.normalization,
            sigma=model.sigma,
            centre=baseline.af.UniformPrior(lower_limit=-100, upper_limit=100),
        )
    names = list(model.parameter_names)
    centre_index = names.index("centre")
    search = baseline.af.Nautilus(
        name="cadence",
        verbose=False,
        n_live=200,
        n_networks=4,
        number_of_cores=1,
        seed=seed,
        n_update=None if arm == "default" else 400,
    )
    analysis = baseline.AnalysisGaussianX1(*baseline.make_dataset())
    trace = {"calls": 0, "peak_calls": {}, "native": {}}
    timer = baseline.ExclusiveTimer()
    restore = baseline.instrument(timer)
    evaluate = Sampler.evaluate_likelihood
    threshold = ref["best_log_likelihood"] - 1.0

    def observed(sampler, points):
        result = evaluate(sampler, points)
        if not trace["native"]:
            trace["native"] = {
                "n_update": int(sampler.n_update),
                "n_batch": int(sampler.n_batch),
                "n_networks": int(sampler.n_networks),
                "n_live": int(sampler.n_live),
            }
        for index in np.flatnonzero(np.asarray(result[0]) >= threshold):
            sign = (
                "negative"
                if fixture == "symmetric" and points[index, centre_index] < 0.5
                else "positive"
            )
            trace["peak_calls"].setdefault(sign, trace["calls"] + int(index) + 1)
        trace["calls"] += len(points)
        return result

    try:
        with patch.object(Sampler, "evaluate_likelihood", observed):
            start = time.perf_counter()
            result = search.fit(model=model, analysis=analysis)
            wall = time.perf_counter() - start
    finally:
        restore()
    samples = result.samples
    weights = np.array(samples.weight_list)
    weights /= weights.sum()
    params = np.asarray(samples.parameter_lists)
    positive_mass = float(weights[params[:, centre_index] > 0].sum())
    params[:, centre_index] = abs(params[:, centre_index])
    mean = weights @ params
    std = np.sqrt(weights @ (params - mean) ** 2)
    expected_interval = 200 if arm == "default" else 400
    if (
        trace["native"]["n_update"] != expected_interval
        or trace["calls"] != samples.samples_info["total_samples"]
    ):
        raise RuntimeError("Sampler cadence or evaluation accounting mismatch")
    return {
        "fixture": fixture,
        "arm": arm,
        "seed": seed,
        "wall_s": wall,
        **trace,
        "bounds": timer.counts["bound_geometry"],
        "training_s": timer.totals["nn_training"],
        "geometry_s": timer.totals["bound_geometry"],
        "log_evidence": float(samples.log_evidence),
        "max_log_likelihood": float(max(samples.log_likelihood_list)),
        "ess": float(1 / (weights @ weights)),
        "parameter_names": names,
        "folded_mean": mean.tolist(),
        "folded_std": std.tolist(),
        "positive_mass": positive_mass,
    }


def compare(base, candidate, ref):
    modes = ("positive", "negative") if base["fixture"] == "symmetric" else ("positive",)
    checks = {}
    for row in (base, candidate):
        arm = row["arm"]
        checks[f"{arm}_evidence"] = abs(row["log_evidence"] - ref["log_evidences"][-1]) <= 0.5
        checks[f"{arm}_peaks"] = all(mode in row["peak_calls"] for mode in modes)
        if len(modes) == 2:
            checks[f"{arm}_mode_mass"] = 0.35 <= row["positive_mass"] <= 0.65
    checks["peak_discovery"] = all(
        mode in base["peak_calls"]
        and mode in candidate["peak_calls"]
        and candidate["peak_calls"][mode] <= base["peak_calls"][mode] * 1.25
        for mode in modes
    )
    # Prior creation order can differ between models. Compare within each fixture by name.
    delta = (np.asarray(candidate["folded_mean"]) - base["folded_mean"]) / base["folded_std"]
    checks["folded_means"] = bool(np.all(abs(delta) <= 0.3))
    return {
        "fixture": base["fixture"],
        "seed": base["seed"],
        "checks": checks,
        "mean_shifts_in_baseline_sd": delta.tolist(),
        "wall_reduction_pct": 100 * (1 - candidate["wall_s"] / base["wall_s"]),
    }


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", nargs=3, metavar=("FIXTURE", "ARM", "SEED"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.case:
        fixture, arm, seed = args.case
        ref = json.loads((directory.parent / "reference.json").read_text())
        write(directory / "case.json", run_one(directory, fixture, arm, int(seed), ref))
        return
    directory.mkdir(parents=True, exist_ok=False)
    ref = reference(baseline.make_dataset())
    write(directory / "reference.json", ref)
    plan = [
        (fixture, seed, arm)
        for i, fixture in enumerate(FIXTURES)
        for seed in (1, 2)
        for arm in (ARMS if (i + seed) % 2 else ARMS[::-1])
    ]
    software, release, dirty = baseline.software()
    receipt = {
        "reference": ref,
        "software": software,
        "release": release,
        "dirty_library": dirty,
        "python": platform.python_version(),
        "threads": 1,
        "plan": plan,
        "runs": [],
        "pairs": [],
        "status": "running",
    }
    for fixture, seed, arm in plan:
        case = directory / f"{fixture}-{seed}-{arm}"
        case.mkdir()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(case),
            "--case",
            fixture,
            arm,
            str(seed),
        ]
        try:
            with (case / "fit.log").open("w") as log:
                subprocess.run(
                    cmd,
                    env={**os.environ, **dict.fromkeys(THREAD_VARS, "1")},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    check=True,
                )
        except (subprocess.SubprocessError, OSError) as error:
            receipt.update(status="incomplete", failure=str(error), failed_case=case.name)
            write(directory / "result.json", receipt)
            raise
        row = json.loads((case / "case.json").read_text())
        receipt["runs"].append(row)
        print(case.name, round(row["wall_s"], 2), flush=True)
        if len(receipt["runs"]) % 2 == 0:
            pair = {r["arm"]: r for r in receipt["runs"][-2:]}
            receipt["pairs"].append(compare(pair["default"], pair["doubled"], ref))
        write(directory / "result.json", receipt)
    passed = all(
        p["wall_reduction_pct"] > 0 and all(p["checks"].values()) for p in receipt["pairs"]
    )
    receipt.update(status="complete", pilot_win=passed)
    write(directory / "result.json", receipt)


if __name__ == "__main__":
    main()
