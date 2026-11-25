"""Extract shell likelihood moments, or replay the compact phase 0b receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def read(path):
    return json.loads(path.read_text())


def expected_runs():
    prior = read(EXPERIMENTS / "nautilus_30d_diagnostics/evidence/attempts.json")
    for row in prior["attempts"]:
        if row["case"] in ("evidence-d5-rsaved", "evidence-d10-rsaved"):
            yield row["case"], row["child"]["fresh_only"]
    for exp, prefix in (
        ("nautilus_30d_evidence_check", "conditional"),
        ("nautilus_30d_matched_ess", "independent"),
    ):
        for row in read(EXPERIMENTS / exp / "result.json")["runs"]:
            yield prefix + "-" + row["stream"], row["result"]["final"]


def moment(ll):
    return {"n": len(ll), "log_sum": float(logsumexp(ll)), "log_sum_sq": float(logsumexp(2 * ll))}


def extract(path, expected):
    import h5py

    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    shells = []
    with h5py.File(path, "r") as checkpoint:
        group = checkpoint["sampler"]
        a = group.attrs
        if int(a["n_like"]) != expected["n_like"]:
            raise ValueError("Wrong checkpoint call count")
        np.testing.assert_allclose(a["shell_log_v"], expected["shell_log_v"], rtol=0, atol=1e-12)
        for i, cut in enumerate(a["shell_end_exp"]):
            ll = group[f"log_l_{i}"][cut:]
            n = len(ll)
            if n < 100 or n != expected["fresh_samples_per_shell"][i] or n != a["shell_n"][i]:
                raise ValueError("Fresh sample count disagrees with phase receipt")
            full = moment(ll)
            np.testing.assert_allclose(
                full["log_sum"] - np.log(n), a["shell_log_l"][i], atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(
                np.exp(2 * full["log_sum"] - full["log_sum_sq"]), a["shell_n_eff"][i], rtol=1e-10
            )
            k = (n + 99) // 100
            top = np.partition(ll, n - k)[-k:]
            shells.append(
                {
                    "log_v": float(a["shell_log_v"][i]),
                    "top_n": k,
                    "top_log_sum_sq": float(logsumexp(2 * top)),
                    "first": moment(ll[: n // 2]),
                    "last": moment(ll[n // 2 :]),
                }
            )
        return {
            "checkpoint_sha256": digest,
            "persisted_discard_flag": bool(a["_discard_exploration"]),
            "shells": shells,
        }


def moments(shells, part):
    counts, first, second = [], [], []
    for shell in shells:
        m = [shell["first"], shell["last"]] if part == "full" else [shell[part]]
        n = sum(x["n"] for x in m)
        counts.append(n)
        first.append(logsumexp([x["log_sum"] for x in m]) - np.log(n))
        second.append(logsumexp([x["log_sum_sq"] for x in m]) - np.log(n))
    return np.array(counts), np.array(first), np.array(second)


def allocation(log_v, log_second):
    # Cauchy-Schwarz minimizer of sum(V_h^2 E[L_h^2] / p_h).
    log_share = log_v + 0.5 * log_second
    return log_share - logsumexp(log_share)


def cost(log_v, log_first, log_second, log_share):
    log_z = logsumexp(log_v + log_first)
    return float(np.exp(logsumexp(2 * log_v + log_second - log_share) - 2 * log_z))


def analyze(run, expected):
    shells = run["shells"]
    log_v = np.array([s["log_v"] for s in shells])
    n, first, second = moments(shells, "full")
    if n.tolist() != expected["fresh_samples_per_shell"]:
        raise ValueError("Shell counts changed")
    np.testing.assert_allclose(log_v, expected["shell_log_v"], rtol=0, atol=1e-12)
    log_z = logsumexp(log_v + first)
    observed = cost(log_v, first, second, np.log(n / n.sum()))
    ess = float(n.sum() / observed)
    np.testing.assert_allclose([log_z, ess], [expected["logZ"], expected["n_eff"]], rtol=1e-11)
    if n.sum() != expected["selected_samples"]:
        raise ValueError("Total selected samples changed")
    optimum = cost(log_v, first, second, allocation(log_v, second))
    if optimum > observed + 1e-9:
        raise ValueError("Optimum worse than feasible reference")
    _, _, first_half_second = moments(shells, "first")
    last_n, last_first, last_second = moments(shells, "last")
    historical = cost(log_v, last_first, last_second, np.log(last_n / last_n.sum()))
    predicted = cost(log_v, last_first, last_second, allocation(log_v, first_half_second))
    # Each row in a shell has unnormalized posterior weight V_h * L / n_h.
    top = np.array([shell["top_log_sum_sq"] for shell in shells])
    top_sq = logsumexp(2 * (log_v - np.log(n)) + top)
    all_sq = logsumexp(2 * log_v + second - np.log(n))
    concentration = float(100 * np.exp(top_sq - all_sq))
    if not 0 <= concentration <= 100:
        raise ValueError("Invalid squared-weight share")
    return {
        "fresh_calls": int(n.sum()),
        "earlier_calls": expected["n_like"] - int(n.sum()),
        "ess": ess,
        "logZ": float(log_z),
        "logZ_error": expected["logZ_error"],
        "calls_per_ess": observed,
        "top_count": sum(shell["top_n"] for shell in shells),
        "top_squared_weight_pct": concentration,
        "optimal_calls_per_ess": optimum,
        "same_data_saving_pct": 100 * (1 - optimum / observed),
        "held_out_native_calls_per_ess": historical,
        "held_out_predicted_calls_per_ess": predicted,
        "held_out_saving_pct": 100 * (1 - predicted / historical),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract",
        type=Path,
        metavar="PATH_MAP",
        help="JSON mapping case names to original local checkpoints",
    )
    args = parser.parse_args()
    expected = dict(expected_runs())
    if args.extract:
        paths = read(args.extract)
        if paths.keys() != expected.keys():
            raise ValueError("Incomplete checkpoint map")
        inputs = {k: extract(Path(paths[k]), v) for k, v in expected.items()}
        for k, v in expected.items():
            analyze(inputs[k], v)
        print(json.dumps(inputs, indent=2, allow_nan=False))
    else:
        inputs = read(HERE / "moments.json")
        if inputs.keys() != expected.keys():
            raise ValueError("Incomplete moment set")
        print(
            json.dumps(
                {k: analyze(inputs[k], v) for k, v in expected.items()}, indent=2, allow_nan=False
            )
        )


if __name__ == "__main__":
    main()
