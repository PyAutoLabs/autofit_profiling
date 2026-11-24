"""Continue the three PR #33 terminal checkpoints within one local hour."""

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
PREDECESSOR = ROOT / "experiments/nautilus_30d_independent"
sys.path.insert(0, str(PREDECESSOR))
from profiling.nautilus_diagnostics import estimate, fresh_counts, sha256
from profiling.nautilus_manifest import RunConfig
from profiling.nautilus_resources import atomic_json, supervise
from read_saved import inspect_checkpoint
from run import MEMORY_BYTES, THREADS, geometry, load

BUDGET_S = 3600
GRACE_S = 30


def continue_seed(source, output, row, truth, deadline):
    import numpy as np
    from autofit.non_linear.search.nest.nautilus.search import _large_attribute_hdf5
    from nautilus import Sampler
    from profiling.nautilus_target import CorrelatedGaussianTarget

    name = row["stream"]
    source_file = source / name / "checkpoint.hdf5"
    path = output / "checkpoint.hdf5"
    if path.exists():
        raise FileExistsError(path)
    shutil.copyfile(source_file, path)
    if sha256(path) != row["checkpoint_sha256"]:
        raise ValueError(f"Wrong source checkpoint for {name}")
    exploration = output / "exploration.json"
    atomic_json(exploration, row["exploration"])
    before = inspect_checkpoint(path, exploration, truth)
    if before != row["estimate"]:
        raise ValueError("Copied checkpoint differs from archived audit")
    seed = int(name.removeprefix("seed"))
    target = CorrelatedGaussianTarget(30)
    if abs(truth - target.log_evidence) > 1e-12:
        raise ValueError("Analytic evidence changed")
    config = RunConfig(dimension=30, n_live=400, seed=seed, trace=False).constructor()

    def reopen():
        return Sampler(target.prior, target, filepath=str(path), **config)

    with _large_attribute_hdf5():
        sampler = reopen()
        cuts = sampler.shell_end_exp.copy()
        if (
            not sampler.discard_exploration
            or not sampler.explored
            or sampler.n_like != before["total_calls"]
        ):
            raise ValueError("Resume state mismatch")
        update = sampler.write_shell_update
        last = 0.0

        def persist(*args, **kwargs):
            nonlocal last
            update(*args, **kwargs)
            now = time.monotonic()
            if now - last >= 30:
                atomic_json(
                    output / "progress.json",
                    {
                        "elapsed_from_launch_s": now - (deadline - BUDGET_S),
                        "n_like": int(sampler.n_like),
                        "ess": float(sampler.n_eff),
                    },
                )
                last = now

        remaining = max(0.0, deadline - time.monotonic() - GRACE_S)
        with (
            patch.object(sampler, "add_bound", side_effect=ValueError("Unexpected bound update")),
            patch.object(sampler, "write_shell_update", persist),
        ):
            done = sampler.run(
                n_eff=10000, n_shell=100, discard_exploration=True, verbose=False, timeout=remaining
            )
        final = estimate(sampler, target)
        completed = bool(done and sampler.n_eff >= 10000 and min(fresh_counts(sampler)) >= 100)
        native = reopen()
        if (
            not native.discard_exploration
            or not np.array_equal(cuts, native.shell_end_exp)
            or estimate(native, target) != final
            or geometry(path) != row["exploration"]["geometry"]
            or final["selected_samples"] != native.n_like - row["exploration"]["estimate"]["n_like"]
        ):
            raise ValueError("Saved result failed native reopening or fixed-bound verification")
    atomic_json(
        output / "outcome.json",
        {
            "seed": seed,
            "completed": completed,
            "initial": before,
            "final": final,
            "target_logZ": truth,
            "additional_calls": final["n_like"] - before["total_calls"],
            "geometry": row["exploration"]["geometry"],
            "native_reopen_exact": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--child", choices=("seed1", "seed2", "seed3"))
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    prior = load(PREDECESSOR / "checkpoints.json")
    rows = prior["seeds"]
    if args.child:
        row = next(r for r in rows if r["stream"] == args.child)
        continue_seed(source, output, row, prior["analytic_logZ"], args.deadline)
        return
    from profiling.nautilus_baseline import provenance
    from profiling.nautilus_campaign import environment_key
    from verify import verify

    start = time.monotonic()
    deadline = start + BUDGET_S
    expected = load(PREDECESSOR / "result.json")["environment"]
    actual = environment_key(provenance())
    for key in ("packages", "nautilus_source_sha256", "python"):
        if actual[key] != expected[key]:
            raise ValueError(f"Runtime mismatch: {key}")
    for repo in ("PyAutoFit", "PyAutoNerves"):
        if actual["sources"][repo] != expected["sources"][repo]:
            raise ValueError(f"Library source mismatch: {repo}")
    output.mkdir(parents=True, exist_ok=False)

    def launch(row):
        name = row["stream"]
        folder = output / name
        folder.mkdir()
        cmd = [
            sys.executable,
            __file__,
            str(source),
            str(folder),
            "--child",
            name,
            "--deadline",
            str(deadline),
        ]
        with (folder / "run.log").open("w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, **dict.fromkeys(THREADS, "1")},
            )
            supervision = supervise(proc, deadline - time.monotonic(), MEMORY_BYTES, interval=0.25)
        elapsed = time.monotonic() - start
        atomic_json(folder / "supervision.json", supervision)
        result = folder / "outcome.json"
        return {
            "stream": name,
            "elapsed_from_shared_start_s": elapsed,
            "supervision": supervision,
            "result": load(result) if result.exists() else None,
        }

    with ThreadPoolExecutor(max_workers=3) as executor:
        runs = list(executor.map(launch, rows))
    elapsed_s = time.monotonic() - start
    unchanged = all(
        sha256(source / r["stream"] / "checkpoint.hdf5") == r["checkpoint_sha256"] for r in rows
    )
    success = unchanged and all(
        r["supervision"]["returncode"] == 0 and r["result"] and r["result"]["completed"]
        for r in runs
    )
    atomic_json(
        output / "result.json",
        {
            "status": "complete" if success else "incomplete",
            "runs": runs,
            "source_hashes": {r["stream"]: r["checkpoint_sha256"] for r in rows},
            "inputs_unchanged": unchanged,
            "environment": actual,
            "shared_budget_s": BUDGET_S,
            "supervised_elapsed_s": elapsed_s,
            "grace_s": GRACE_S,
            "memory_limit_bytes_per_child": MEMORY_BYTES,
        },
    )
    if success:
        atomic_json(
            output / "verification.json", verify(output, streams=[r["stream"] for r in runs])
        )
    else:
        raise SystemExit(
            "One or more streams incomplete or unverified; inspect their saved checkpoints."
        )


if __name__ == "__main__":
    main()
