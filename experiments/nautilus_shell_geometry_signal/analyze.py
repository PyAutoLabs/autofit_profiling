"""Replay frozen 30D bound scores on chronological held-out samples within each shell."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
NAMES = ("ellipsoid", "neural")
CHUNK = 4096


def read(path):
    return json.loads(path.read_text())


def cutoff(values, quota):
    """Choose a threshold and the inclusion probability of points tied there."""
    index = len(values) - math.ceil(quota)
    threshold = np.partition(values, index)[index]
    above = np.count_nonzero(values > threshold)
    tied = np.count_nonzero(values == threshold)
    return threshold, (quota - above) / tied


def inclusion(values, threshold, tie_probability):
    return (values > threshold).astype(float) + (values == threshold) * tie_probability


def bound_scores(points, children):
    radius = np.full(len(points), -np.inf)
    neural = np.full(len(points), -np.inf)
    for child in children:
        transformed = child.outer_bound.transform(points)
        r2 = np.einsum("ij,ij->i", transformed, transformed)
        radius = np.maximum(radius, -r2)
        inside = r2 < 1.0
        if child.emulator is not None and np.any(inside):
            pred = child.emulator.predict(transformed[inside]) - child.score_predict_min
            neural[inside] = np.maximum(neural[inside], pred)
    if np.isnan(radius).any() or np.isnan(neural).any():
        raise ValueError("Unusable model output")
    return radius, neural


def shell_scores(dataset, start, children):
    chunks = [
        bound_scores(dataset[i : i + CHUNK], children) for i in range(start, len(dataset), CHUNK)
    ]
    return {name: np.concatenate(parts) for name, parts in zip(NAMES, zip(*chunks), strict=True)}


def evaluate(scores, split, tail, squared):
    threshold, tie = cutoff(scores[:split], split / 10)
    chosen = inclusion(scores[split:], threshold, tie)
    return {
        "cutoff": float(threshold) if np.isfinite(threshold) else None,
        "tie_probability": float(tie),
        "selected": float(chosen.sum()),
        "tail_selected": float(chosen @ tail),
        "squared_weight_fraction": float(chosen @ squared),
    }


def extract(path, source, expected):
    import h5py
    from nautilus.bounds import NeuralBound

    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["checkpoint_sha256"]:
        raise ValueError(f"Unexpected checkpoint: {path}")
    shells = []
    full_first, full_second = [], []
    with h5py.File(path, "r") as file:
        g = file["sampler"]
        a = g.attrs
        if not a["_discard_exploration"] or a["n_like"] != expected["n_like"]:
            raise ValueError("Wrong sampling stage or likelihood count")
        for i, cut in enumerate(a["shell_end_exp"]):
            points = g[f"points_{i}"]
            log_l = g[f"log_l_{i}"][cut:]
            n = len(log_l)
            split = n // 2
            if n != len(points) - cut or n != expected["fresh_samples_per_shell"][i] or n < 100:
                raise ValueError("Shell rows do not reconcile")
            b = file[f"bound_{i}"]
            children = [
                NeuralBound.read(b[f"neural_bound_{j}"])
                for j in range(b.attrs.get("n_neural_bounds", 0))
            ]
            scores = shell_scores(points, cut, children)
            # Labels first enter the ranking evaluation here, after the frozen scores are computed.
            test = log_l[split:]
            tail_n = math.ceil(len(test) / 100)
            tail = inclusion(test, *cutoff(test, tail_n))
            log_second = logsumexp(2 * test)
            squared = np.exp(2 * test - log_second)
            volume = a["shell_log_v"][i]
            full_first.append(logsumexp(log_l) + volume - np.log(n))
            full_second.append(logsumexp(2 * log_l) + 2 * (volume - np.log(n)))
            shells.append(
                {
                    "shell": i,
                    "components": len(children),
                    "train_n": split,
                    "test_n": len(test),
                    "tail_n": tail_n,
                    "log_test_squared_mass": float(log_second + 2 * (volume - np.log(len(test)))),
                    "scores": {
                        name: evaluate(scores[name], split, tail, squared) for name in NAMES
                    },
                }
            )
    logz = logsumexp(full_first)
    ess = np.exp(2 * logz - logsumexp(full_second))
    np.testing.assert_allclose([logz, ess], [expected["logZ"], expected["n_eff"]], rtol=1e-10)
    return {"checkpoint_sha256": digest, "shells": shells}


def summarize(case):
    shells = case["shells"]
    weights = np.array([s["log_test_squared_mass"] for s in shells])
    weights = np.exp(weights - logsumexp(weights))
    sizes = np.array([s["test_n"] for s in shells])
    tails = np.array([s["tail_n"] for s in shells])
    results = {}
    for name in NAMES:
        rows = [s["scores"][name] for s in shells]
        rates = np.array([r["selected"] for r in rows]) / sizes
        capture = float(weights @ [r["squared_weight_fraction"] for r in rows])
        neutral = float(weights @ rates)
        tail_recall = sum(r["tail_selected"] for r in rows) / tails.sum()
        tail_neutral = float(tails @ rates / tails.sum())
        results[name] = {
            "selected_pct": 100 * float(sizes @ rates / sizes.sum()),
            "squared_capture_pct": 100 * capture,
            "squared_random_pct": 100 * neutral,
            "squared_lift": capture / neutral,
            "tail_recall_pct": 100 * tail_recall,
            "tail_random_pct": 100 * tail_neutral,
            "tail_lift": tail_recall / tail_neutral,
            "passes": bool(capture >= 2 * neutral and tail_recall >= 2 * tail_neutral),
        }
    return {
        "train_n": sum(s["train_n"] for s in shells),
        "test_n": int(sizes.sum()),
        "shells": len(shells),
        "scores": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract", type=Path, help="raw PR #35 directory containing the three seed subdirectories"
    )
    args = parser.parse_args()
    prior = read(EXPERIMENTS / "nautilus_shell_efficiency/moments.json")
    endpoints = read(EXPERIMENTS / "nautilus_30d_matched_ess/result.json")["runs"]
    if args.extract:
        cases = {
            r["stream"]: extract(
                args.extract / r["stream"] / "checkpoint.hdf5",
                prior["independent-" + r["stream"]],
                r["result"]["final"],
            )
            for r in endpoints
        }
        output = cases
    else:
        cases = read(HERE / "shells.json")
        if set(cases) != {r["stream"] for r in endpoints}:
            raise ValueError("Missing held-out seed")
        results = {}
        for r in endpoints:
            name = r["stream"]
            if (
                cases[name]["checkpoint_sha256"]
                != prior["independent-" + name]["checkpoint_sha256"]
            ):
                raise ValueError("Checkpoint identity differs from PR #37")
            result = summarize(cases[name])
            if result["train_n"] + result["test_n"] != r["result"]["final"]["selected_samples"]:
                raise ValueError("Missing fresh samples")
            results[name] = result
        output = {
            "runs": results,
            "advance": {
                name: all(r["scores"][name]["passes"] for r in results.values()) for name in NAMES
            },
        }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
