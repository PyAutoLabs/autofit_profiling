"""Run and verify independent 30D Nautilus explorations against the analytic evidence."""

import argparse
import hashlib
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
sys.path.insert(0, str(ROOT / "scripts/misc"))
sys.path.insert(0, str(ROOT / "experiments/nautilus_30d_evidence_check"))

from check import MEMORY_BYTES, THREADS, load, records
from profiling.nautilus_diagnostics import discard_saved_exploration, estimate, fresh_counts, sha256
from profiling.nautilus_manifest import RunConfig, fingerprint
from profiling.nautilus_resources import atomic_json, supervise

SEEDS = (1, 2, 3)
TIMEOUT_S = 2400


def geometry(path):
    """Hash learned outer ellipsoid centres and matrices; omit cached proposals and RNG."""
    import h5py
    import numpy as np

    digest = hashlib.sha256()
    count = 0

    def visit(name, node):
        nonlocal count
        if (
            "outer_bound/" in name
            and isinstance(node, h5py.Group)
            and "A" in node.attrs
            and "c" in node.attrs
        ):
            count += 1
            digest.update(name.encode() + b"\0")
            for key in ("A", "c"):
                x = np.asarray(node.attrs[key], dtype="<f8")
                digest.update(np.asarray(x.shape, dtype="<i8").tobytes())
                digest.update(x.tobytes())

    with h5py.File(path, "r") as file:
        file.visititems(visit)
    if count == 0:
        raise ValueError("No learned outer ellipsoids found")
    return {"ellipsoids": count, "sha256": digest.hexdigest()}


def sample(folder, seed):
    import numpy as np
    from autofit.non_linear.search.nest.nautilus.search import _large_attribute_hdf5
    from nautilus import Sampler
    from profiling.nautilus_target import CorrelatedGaussianTarget

    _, case, _ = records()
    target = CorrelatedGaussianTarget(30)
    if fingerprint(target.truth()) != case["target_sha256"]:
        raise ValueError("30D target differs from the baseline")
    path = folder / "checkpoint.hdf5"
    if path.exists():
        raise FileExistsError("Independent exploration requires a fresh checkpoint path")
    config = RunConfig(dimension=30, n_live=400, seed=seed, trace=False).constructor()

    def open_sampler():
        return Sampler(target.prior, target, filepath=str(path), **config)

    with _large_attribute_hdf5():
        sampler = open_sampler()
        if sampler.explored or sampler.n_like or sampler.bounds:
            raise ValueError("Sampler was not empty at initialization")
        # Stop exactly at the native f_live transition, before fresh draws begin.
        ok = sampler.run(f_live=0.01, n_shell=0, n_eff=0, discard_exploration=False, verbose=False)
        if not ok or not sampler.explored or any(fresh_counts(sampler)):
            raise ValueError("Exploration boundary was missed")
        boundary = estimate(sampler, target)
        cuts = sampler.shell_end_exp.copy()
        learned = geometry(path)
        shutil.copyfile(path, folder / "exploration.hdf5")
        atomic_json(folder / "exploration.json", {"estimate": boundary, "geometry": learned})
        discard_saved_exploration(sampler)
        sampler.write(str(path), overwrite=True)
        update = sampler.write_shell_update
        previous = 0.0

        def record_progress(*args, **kwargs):
            nonlocal previous
            update(*args, **kwargs)
            now = time.monotonic()
            if now - previous >= 30:
                atomic_json(
                    folder / "progress.json",
                    {"n_like": int(sampler.n_like), "ess": float(sampler.n_eff)},
                )
                previous = now

        with (
            patch.object(sampler, "add_bound", side_effect=ValueError("Unexpected new bound")),
            patch.object(sampler, "write_shell_update", record_progress),
        ):
            ok = sampler.run(n_eff=10000, n_shell=100, discard_exploration=True, verbose=False)
        if not ok or sampler.n_eff < 10000 or min(fresh_counts(sampler)) < 100:
            raise ValueError("Fresh-only sampling criteria failed")
        final = estimate(sampler, target)
        native = open_sampler()
        if (
            not native.discard_exploration
            or not np.array_equal(cuts, native.shell_end_exp)
            or geometry(path) != learned
            or estimate(native, target) != final
            or final["selected_samples"] != native.n_like - boundary["n_like"]
        ):
            raise ValueError("Final checkpoint did not reproduce the fixed-bound result")
    atomic_json(
        folder / "complete.json",
        {
            "seed": seed,
            "constructor": config,
            "target_logZ": target.log_evidence,
            "geometry": learned,
            "exploration": boundary,
            "final": final,
            "native_reopen_exact": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path, help="archived seed-0 exploration, read only")
    parser.add_argument("--child", type=int, choices=SEEDS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.child:
        sample(output, args.child)
        return
    from profiling.nautilus_baseline import provenance
    from profiling.nautilus_campaign import environment_key
    from verify import verify

    plan, case, _ = records()
    if not args.reference or sha256(args.reference) != case["checkpoint_sha256"]:
        parser.error("Supply the pinned seed-0 checkpoint with --reference")
    reference = geometry(args.reference)
    actual = environment_key(provenance())
    expected = plan["environment"]
    for key in ("nautilus_source_sha256", "packages", "python"):
        if actual[key] != expected[key]:
            raise ValueError(f"Runtime mismatch: {key}")
    for source in ("PyAutoFit", "PyAutoNerves"):
        if actual["sources"][source] != expected["sources"][source]:
            raise ValueError(f"Source mismatch: {source}")
    output.mkdir(parents=True, exist_ok=False)

    def launch(seed):
        folder = output / f"seed{seed}"
        folder.mkdir()
        start = time.monotonic()
        cmd = [sys.executable, __file__, str(folder), "--child", str(seed)]
        with (folder / "run.log").open("w") as log:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, **dict.fromkeys(THREADS, "1")},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            supervision = supervise(proc, TIMEOUT_S, MEMORY_BYTES, interval=0.5)
        supervision["elapsed_s"] = time.monotonic() - start
        atomic_json(folder / "supervision.json", supervision)
        file = folder / "complete.json"
        return {
            "stream": folder.name,
            "supervision": supervision,
            "result": load(file) if file.is_file() else None,
        }

    with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        runs = list(executor.map(launch, SEEDS))
    complete = all(r["supervision"]["returncode"] == 0 and r["result"] for r in runs)
    hashes = {reference["sha256"]} | {
        r["result"]["geometry"]["sha256"] for r in runs if r["result"]
    }
    valid = complete and len(hashes) == 4 and sha256(args.reference) == case["checkpoint_sha256"]
    atomic_json(
        output / "result.json",
        {
            "status": "complete" if valid else "incomplete",
            "runs": runs,
            "seed0_geometry": reference,
            "seed0_file_sha256": case["checkpoint_sha256"],
            "environment": actual,
            "timeout_s": TIMEOUT_S,
            "rss_limit_bytes_per_child": MEMORY_BYTES,
        },
    )
    if not valid:
        raise SystemExit("Completion, geometry independence or input-integrity check failed")
    atomic_json(output / "verification.json", verify(output, streams=[r["stream"] for r in runs]))


if __name__ == "__main__":
    main()
