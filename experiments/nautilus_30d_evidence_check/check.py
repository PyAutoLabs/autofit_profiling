"""Complete three conditional 30D evidence streams from the hash-pinned local archives."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "misc"))

from profiling.nautilus_diagnostics import discard_saved_exploration, estimate, fresh_counts, sha256
from profiling.nautilus_manifest import RunConfig, fingerprint
from profiling.nautilus_resources import atomic_json, supervise

EVIDENCE = ROOT / "experiments/nautilus_30d_diagnostics/evidence"
STREAMS = ("saved", "seed1", "seed2")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TIMEOUT_S = 1800
MEMORY_BYTES = 4096 * 1024**2


def load(path):
    return json.loads(path.read_text())


def records():
    plan = load(EVIDENCE / "plan.json")
    case = next(row for row in plan["cases"] if row["id"] == "evidence-d30-rsaved")
    return plan, case, load(EVIDENCE / "partial-checkpoint.json")


def check_sources(original, partial):
    _, case, prior = records()
    if (
        sha256(original) != case["checkpoint_sha256"]
        or sha256(partial) != prior["checkpoint_sha256"]
    ):
        raise ValueError("The input checkpoint differs from its published hash")


def run_stream(original, partial, output, stream):
    import numpy as np
    from autofit.non_linear.search.nest.nautilus.search import _large_attribute_hdf5
    from nautilus import Sampler
    from profiling.nautilus_target import CorrelatedGaussianTarget

    _, case, prior = records()
    target = CorrelatedGaussianTarget(30)
    if fingerprint(target.truth()) != case["target_sha256"]:
        raise ValueError("Analytic target changed")
    checkpoint = output / "checkpoint.hdf5"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    shutil.copyfile(partial if stream == "saved" else original, checkpoint)
    config = RunConfig(dimension=30, n_live=400, trace=False).constructor()

    def reopen():
        return Sampler(target.prior, target, filepath=str(checkpoint), **config)

    with _large_attribute_hdf5():
        sampler = reopen()
        if stream == "saved":
            discard_saved_exploration(sampler)
            previous = prior["estimate"]
            expected_digest = previous["posterior_sha256"]
            expected_calls = previous["n_like"]
        else:
            expected_digest = case["original_posterior_sha256"]
            expected_calls = case["original_n_like"]
            if any(fresh_counts(sampler)):
                raise ValueError("Original checkpoint already has fresh samples")
        start = estimate(sampler, target)
        if start["posterior_sha256"] != expected_digest or start["n_like"] != expected_calls:
            raise ValueError("Starting posterior or likelihood count differs")
        if stream != "saved":
            seed = int(stream.removeprefix("seed"))
            sampler.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
            discard_saved_exploration(sampler)
        cuts = sampler.shell_end_exp.copy()
        num_bounds = len(sampler.bounds)
        if (sampler.n_update, sampler.n_live, sampler.n_batch, sampler.n_networks) != (
            400,
            400,
            100,
            4,
        ):
            raise ValueError("Unexpected native sampler settings")
        sampler.write(str(checkpoint), overwrite=True)
        last = 0.0
        update = sampler.write_shell_update

        def persist(*args, **kwargs):
            nonlocal last
            update(*args, **kwargs)
            now = time.monotonic()
            if now - last >= 30:
                atomic_json(
                    output / "progress.json",
                    {"n_like": int(sampler.n_like), "ess": float(sampler.n_eff)},
                )
                last = now

        with (
            patch.object(sampler, "add_bound", side_effect=RuntimeError("Learned bounds changed")),
            patch.object(sampler, "write_shell_update", persist),
        ):
            converged = sampler.run(
                n_eff=10000, n_shell=100, discard_exploration=True, verbose=False
            )
        if not converged or sampler.n_eff < 10000 or min(fresh_counts(sampler)) < 100:
            raise ValueError("Fresh-only stopping conditions failed")
        final = estimate(sampler, target)
        native = reopen()
        check = estimate(native, target)
        if (
            not native.discard_exploration
            or len(native.bounds) != num_bounds
            or not np.array_equal(native.shell_end_exp, cuts)
        ):
            raise ValueError("Bound count, exploration boundary or exclusion mode changed")
        if final != check or sum(fresh_counts(native)) != native.n_like - case["original_n_like"]:
            raise ValueError("Native checkpoint failed posterior, evidence or count verification")
    atomic_json(
        output / "complete.json",
        {
            "stream": stream,
            "target_logZ": target.log_evidence,
            "source_posterior_sha256": expected_digest,
            "starting_n_like": expected_calls,
            "additional_calls_this_run": final["n_like"] - expected_calls,
            "bounds": num_bounds,
            "native_reopen_exact": True,
            "final": final,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "partial", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--child", choices=STREAMS)
    args = parser.parse_args()
    original, partial, output = (p.resolve() for p in (args.original, args.partial, args.output))
    if args.child:
        run_stream(original, partial, output, args.child)
        return
    from profiling.nautilus_baseline import provenance
    from profiling.nautilus_campaign import environment_key

    check_sources(original, partial)
    plan, case, previous = records()
    actual = environment_key(provenance())
    expected = plan["environment"]
    for key in ("nautilus_source_sha256", "packages", "python"):
        if actual[key] != expected[key]:
            raise ValueError(f"Phase 0b.2 runtime mismatch: {key}")
    for source in ("PyAutoFit", "PyAutoNerves"):
        if actual["sources"][source] != expected["sources"][source]:
            raise ValueError(f"Phase 0b.2 source mismatch: {source}")
    output.mkdir(parents=True, exist_ok=False)

    def launch(stream):
        folder = output / stream
        folder.mkdir()
        cmd = [
            sys.executable,
            __file__,
            str(original),
            str(partial),
            str(folder),
            "--child",
            stream,
        ]
        start = time.monotonic()
        with (folder / "run.log").open("w") as log:
            child = subprocess.Popen(
                cmd,
                env={**os.environ, **dict.fromkeys(THREADS, "1")},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt = supervise(child, TIMEOUT_S, MEMORY_BYTES, interval=0.5)
        receipt["elapsed_s"] = time.monotonic() - start
        atomic_json(folder / "supervision.json", receipt)
        result = folder / "complete.json"
        return {
            "stream": stream,
            "supervision": receipt,
            "result": load(result) if result.exists() else None,
        }

    with ThreadPoolExecutor(max_workers=3) as executor:
        runs = list(executor.map(launch, STREAMS))
    check_sources(original, partial)
    complete = all(row["supervision"]["returncode"] == 0 and row["result"] for row in runs)
    atomic_json(
        output / "result.json",
        {
            "status": "complete" if complete else "incomplete",
            "original_sha256": case["checkpoint_sha256"],
            "partial_sha256": previous["checkpoint_sha256"],
            "inputs_unchanged": True,
            "environment": actual,
            "original_logZ_error": case["original_logZ_error"],
            "prior_partial": previous["estimate"],
            "timeout_s": TIMEOUT_S,
            "memory_limit_bytes_per_child": MEMORY_BYTES,
            "runs": runs,
        },
    )
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
