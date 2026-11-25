"""Audit all admitted likelihoods against NumPy and compare full-fit timings by seed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_draws(raw, row, x, data, noise):
    case = raw / f"r{row['repeats']}-s{row['seed']}-{row['arm']}"
    with np.load(case / "draws.npz") as saved:
        points = saved["points"]
        values = saved["log_l"]
    assert points.shape == (row["calls"], 3) and len(values) == row["calls"]
    assert hashlib.sha256(points.tobytes()).hexdigest() == row["proposal_sha256"]
    bounds = {"centre": 100, "normalization": 50, "sigma": 50}
    physical = dict(
        zip(row["names"], (points * np.array([bounds[n] for n in row["names"]])).T, strict=True)
    )
    sigma = physical["sigma"][:, None]
    prediction = (
        physical["normalization"][:, None]
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - physical["centre"][:, None]) / sigma) ** 2)
    )
    expected = -0.5 * np.square((data - prediction) / noise).sum(axis=1)
    np.testing.assert_allclose(values, expected, atol=1e-8, rtol=5e-12)
    info = json.loads(next(case.rglob("samples_info.json")).read_text())
    assert row["logz"] == info["log_evidence"] and row["calls"] == info["total_samples"]
    assert row["native"] == {"n_live": 200, "n_update": 200, "n_networks": 4, "n_batch": 100}
    assert row["ess"] >= 500 and len(list(case.rglob(".completed"))) == 1
    assert (
        row["objective_cache_sizes"] == [1]
        and row["device_chunks"] * row["width"] == row["calls"] + row["padding"]
    )
    if row["arm"] == "feed":
        assert row["membership_jobs"] > 0 and row["wait_for_prepare_s"] > 0
    return float(np.max(np.abs(expected - values)))


def compare(raw):
    runs = json.loads((raw / "runs.json").read_text())
    groups = {(row["repeats"], row["seed"], row["arm"]): row for row in runs}
    if len(runs) != 16 or len(groups) != 16:
        raise ValueError("Expected four distinct arms per seed and data size")
    x = np.arange(100)
    noise = 0.01
    data = 25 / (10 * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * ((x - 50) / 10) ** 2
    ) + np.random.RandomState(1).normal(0, noise, 100)
    for row in runs:
        row["max_abs_numpy_error"] = audit_draws(raw, row, x, data, noise)
    pairs = []
    for repeats in (1, 64):
        for seed in (1, 2):
            rows = {arm: groups[repeats, seed, arm] for arm in ("cpu", "gpu", "split", "feed")}
            gpu = rows["gpu"]
            feed = rows["feed"]
            flags = {}
            for arm, row in rows.items():
                flags[arm] = {
                    "trajectory_equal": row["proposal_sha256"] == gpu["proposal_sha256"],
                    "logz_delta": row["logz"] - gpu["logz"],
                    "max_mean_shift_sd": float(
                        np.max(np.abs(np.array(row["mean"]) - gpu["mean"]) / gpu["std"])
                    ),
                    "ess_relative_delta": row["ess"] / gpu["ess"] - 1,
                }
            parity = all(
                abs(f["logz_delta"]) <= 0.1
                and f["max_mean_shift_sd"] <= 0.1
                and abs(f["ess_relative_delta"]) <= 0.1
                for f in flags.values()
            )
            saved = {
                arm: 100 * (1 - feed["fit_wall_s"] / rows[arm]["fit_wall_s"])
                for arm in ("cpu", "gpu", "split")
            }
            pairs.append(
                dict(
                    repeats=repeats,
                    seed=seed,
                    parity=flags,
                    parity_pass=parity,
                    feed_saving_pct=saved,
                    advances=bool(parity and saved["gpu"] >= 5 and saved["split"] > 0),
                )
            )
    return dict(
        runs=runs,
        pairs=pairs,
        advance={str(r): all(p["advances"] for p in pairs if p["repeats"] == r) for r in (1, 64)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.raw), indent=2, allow_nan=False))
