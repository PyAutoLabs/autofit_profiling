"""Time complete native PyAutoFit fits with ordered, in-process neural training workers."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from nautilus import Sampler
from nautilus.neural import NeuralNetworkEmulator
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nautilus_bound_cadence"))
import compare as cadence

WORKERS = (1, 2, 4)
PARITY = (
    "trajectory_sha256",
    "native",
    "calls",
    "bounds",
    "peak_calls",
    "log_evidence",
    "max_log_likelihood",
    "ess",
    "parameter_names",
    "folded_mean",
    "folded_std",
    "positive_mass",
)


def run_case(directory, fixture, workers, seed, reference):
    evaluate = Sampler.evaluate_likelihood
    native_train = NeuralNetworkEmulator.train.__func__
    trajectory = hashlib.sha256()
    training_cpu = 0.0
    lifecycle = 0.0
    threads = threadpool_info()
    if any(lib["num_threads"] != 1 for lib in threads):
        raise RuntimeError("Expected single-thread numerical libraries")

    def observed(sampler, points):
        output = evaluate(sampler, points)
        for array in (points, output[0]):
            trajectory.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return output

    start = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    lifecycle += time.perf_counter() - start

    def train(cls, *args, **kwargs):
        nonlocal training_cpu
        if kwargs["pool"] is not None:
            raise RuntimeError("The baseline already supplies a training pool")
        start_cpu = time.process_time()
        kwargs["pool"] = pool
        emulator = native_train(cls, *args, **kwargs)
        training_cpu += time.process_time() - start_cpu
        return emulator

    try:
        with (
            patch.object(NeuralNetworkEmulator, "train", classmethod(train)),
            patch.object(Sampler, "evaluate_likelihood", observed),
        ):
            row = cadence.run_one(directory, fixture, "default", seed, reference)
    finally:
        start = time.perf_counter()
        if pool is not None:
            pool.shutdown(wait=True)
        lifecycle += time.perf_counter() - start

    if threadpool_info() != threads:
        raise RuntimeError("Numerical library thread settings changed")
    return {
        **row,
        "training_workers": workers,
        "fit_including_pool_s": row["wall_s"] + lifecycle,
        "pool_setup_teardown_s": lifecycle,
        "training_process_cpu_s": training_cpu,
        "trajectory_sha256": trajectory.hexdigest(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "thread_limits": {lib["prefix"]: lib["num_threads"] for lib in threads},
    }


def comparisons(runs):
    pairs = []
    for row in runs:
        if row["training_workers"] == 1:
            continue
        base = next(
            r
            for r in runs
            if r["fixture"] == row["fixture"]
            and r["seed"] == row["seed"]
            and r["training_workers"] == 1
        )
        pairs.append(
            {
                "fixture": row["fixture"],
                "seed": row["seed"],
                "workers": row["training_workers"],
                "mismatched_fields": [key for key in PARITY if row[key] != base[key]],
                "wall_saved_pct": 100
                * (1 - row["fit_including_pool_s"] / base["fit_including_pool_s"]),
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", nargs=3, metavar=("FIXTURE", "WORKERS", "SEED"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    ref = json.loads((HERE.parent / "nautilus_bound_cadence/result.json").read_text())["reference"]
    if args.case:
        fixture, workers, seed = args.case
        result = run_case(directory, fixture, int(workers), int(seed), ref)
        cadence.write(directory / "case.json", result)
        return
    cpus = sorted(os.sched_getaffinity(0))
    topology = Path("/sys/devices/system/cpu")
    cores = {
        (topology / f"cpu{cpu}/topology/thread_siblings_list").read_text().strip() for cpu in cpus
    }
    if len(cpus) != 4 or len(cores) != 4:
        parser.error("Use taskset to give each fit the same four distinct physical cores")
    directory.mkdir(parents=True, exist_ok=False)
    software, release, dirty = cadence.baseline.software()
    runs = []
    receipt = {
        "status": "running",
        "software": software,
        "release": release,
        "dirty_library": dirty,
        "platform": platform.platform(),
        "affinity": cpus,
        "runs": runs,
    }
    for seed in (1, 2, 3):
        for index, fixture in enumerate(cadence.FIXTURES):
            rotation = WORKERS[seed - 1 :] + WORKERS[: seed - 1]
            for workers in rotation if index == 0 else rotation[::-1]:
                case = directory / f"{fixture}-{seed}-{workers}"
                case.mkdir()
                command = [
                    sys.executable,
                    __file__,
                    str(case),
                    "--case",
                    fixture,
                    str(workers),
                    str(seed),
                ]
                with (case / "fit.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=300,
                        env={**os.environ, **dict.fromkeys(cadence.THREAD_VARS, "1")},
                    )
                row = json.loads((case / "case.json").read_text())
                runs.append(row)
                cadence.write(directory / "result.json", receipt)
                print(case.name, round(row["fit_including_pool_s"], 2), flush=True)
    pairs = comparisons(runs)
    median = {
        str(n): {
            fixture: float(
                np.median(
                    [
                        p["wall_saved_pct"]
                        for p in pairs
                        if p["fixture"] == fixture and p["workers"] == n
                    ]
                )
            )
            for fixture in cadence.FIXTURES
        }
        for n in (2, 4)
    }
    receipt.update(
        status="complete",
        pairs=pairs,
        median_wall_saved_pct=median,
        qualifies={
            str(n): all(
                not p["mismatched_fields"] and p["wall_saved_pct"] > 0
                for p in pairs
                if p["workers"] == n
            )
            and min(median[str(n)].values()) >= 5
            for n in (2, 4)
        },
    )
    cadence.write(directory / "result.json", receipt)


if __name__ == "__main__":
    main()
