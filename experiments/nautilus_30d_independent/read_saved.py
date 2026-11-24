"""Recover a fresh-only evidence estimate from a closed native checkpoint after a cap."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from run import geometry
from scipy.special import logsumexp


def inspect_checkpoint(path, exploration, truth):
    boundary = json.loads(exploration.read_text())
    if geometry(path) != boundary["geometry"]:
        raise ValueError("Learned geometry differs from this seed's boundary")
    weights = []
    with h5py.File(path, "r") as file:
        g = file["sampler"]
        a = dict(g.attrs)
        if not a["explored"] or not a["_discard_exploration"]:
            raise ValueError("Expected a fresh-only checkpoint")
        for i, cut in enumerate(a["shell_end_exp"]):
            likelihoods = g[f"log_l_{i}"]
            if len(likelihoods) != len(g[f"points_{i}"]):
                raise ValueError("Mismatched position and likelihood counts")
            ll = likelihoods[cut:]
            if len(ll) != a["shell_n"][i] or len(ll) < 100:
                raise ValueError("Incomplete shell")
            z = logsumexp(ll)
            ess = np.exp(2 * z - logsumexp(2 * ll))
            if not np.isclose(
                z - np.log(len(ll)), a["shell_log_l"][i], rtol=0, atol=1e-10
            ) or not np.isclose(ess, a["shell_n_eff"][i], rtol=1e-9):
                raise ValueError("Native shell summary differs from stored likelihoods")
            weights.append(ll + a["shell_log_v"][i] - np.log(len(ll)))
        calls = int(a["n_like"])
    weights = np.concatenate(weights)
    if len(weights) != calls - boundary["estimate"]["n_like"]:
        raise ValueError("Fresh counts do not reconcile")
    z = float(logsumexp(weights))
    ess = float(np.exp(2 * z - logsumexp(2 * weights)))
    if not np.isfinite(z) or not np.isfinite(ess):
        raise ValueError("Nonfinite estimate")
    # Native construction reopens the existing file. No sampling or writes are requested.
    from nautilus import Sampler
    from profiling.nautilus_manifest import RunConfig
    from profiling.nautilus_target import CorrelatedGaussianTarget

    target = CorrelatedGaussianTarget(30)
    native = Sampler(
        target.prior,
        target,
        filepath=str(path),
        **RunConfig(dimension=30, n_live=400).constructor(),
    )
    if (
        not native.discard_exploration
        or native.n_like != calls
        or abs(z - native.log_z) > 1e-10
        or abs(ess - native.n_eff) > 1e-5
        or abs(truth - target.log_evidence) > 1e-12
    ):
        raise ValueError("Native and stored-row diagnostics disagree")
    return {
        "total_calls": calls,
        "fresh_samples": len(weights),
        "ess": ess,
        "logZ": z,
        "logZ_error": z - truth,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("exploration", type=Path)
    parser.add_argument("truth", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_checkpoint(args.checkpoint, args.exploration, args.truth),
            indent=2,
            allow_nan=False,
        )
    )
