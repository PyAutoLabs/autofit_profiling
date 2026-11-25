"""Independently check evidence and ESS from the completed HDF5 shell likelihoods."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import logsumexp, ndtri


def verify(directory):
    receipt = json.loads((directory / "result.json").read_text())
    if receipt["status"] != "complete" or [r["stream"] for r in receipt["runs"]] != [
        "saved",
        "seed1",
        "seed2",
    ]:
        raise ValueError("All three conditional streams are required")
    dimension = 30
    axis = np.arange(dimension)
    covariance = 0.6 ** abs(axis[:, None] - axis)
    mean = np.linspace(-0.5, 0.5, dimension)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    rotated = vectors.T @ mean
    normalizer = dimension * np.log(2 * np.pi)
    truth = -0.5 * (
        normalizer + np.log1p(eigenvalues).sum() + (rotated**2 / (1 + eigenvalues)).sum()
    )
    results = []
    for row in receipt["runs"]:
        measured = row["result"]["final"]
        weights, max_likelihood_difference = [], 0.0
        with h5py.File(directory / row["stream"] / "checkpoint.hdf5", "r") as file:
            group = file["sampler"]
            if not group.attrs["_discard_exploration"]:
                raise ValueError("Checkpoint includes exploration")
            for i, (cut, volume) in enumerate(
                zip(group.attrs["shell_end_exp"], group.attrs["shell_log_v"], strict=True)
            ):
                ll = group[f"log_l_{i}"][cut:]
                if len(ll) < 100 or len(ll) != measured["fresh_samples_per_shell"][i]:
                    raise ValueError("Wrong fresh shell count")
                weights.append(volume + ll - np.log(len(ll)))
                # Re-evaluate a deterministic interior point of each fresh shell in a separate basis.
                j = len(ll) // 2
                delta = ndtri(group[f"points_{i}"][cut + j]) - mean
                value = -0.5 * (
                    normalizer
                    + np.log(eigenvalues).sum()
                    + ((vectors.T @ delta) ** 2 / eigenvalues).sum()
                )
                difference = abs(value - ll[j])
                if not np.isfinite(difference):
                    raise ValueError("Non-finite likelihood comparison")
                max_likelihood_difference = max(max_likelihood_difference, difference)
        log_weights = np.concatenate(weights)
        log_z = float(logsumexp(log_weights))
        ess = float(np.exp(2 * log_z - logsumexp(2 * log_weights)))
        if (
            len(log_weights) != measured["selected_samples"]
            or abs(truth - row["result"]["target_logZ"]) > 1e-12
            or abs(log_z - measured["logZ"]) > 1e-10
            or abs(ess - measured["n_eff"]) > 1e-5
            or max_likelihood_difference > 1e-9
        ):
            raise ValueError(f"Independent check failed for {row['stream']}")
        results.append(
            {
                "stream": row["stream"],
                "fresh_rows": len(log_weights),
                "logZ": log_z,
                "ess": ess,
                "max_shell_probe_log_likelihood_difference": float(max_likelihood_difference),
            }
        )
    return {"analytic_logZ_eigenbasis": float(truth), "streams": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2, allow_nan=False))
