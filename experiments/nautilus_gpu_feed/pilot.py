"""Run the local four-arm comparison sequentially in fresh processes."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def monitor_gpu(stream, samples):
    for line in stream:
        fields = line.strip().split(",")
        if len(fields) == 2 and all(f.strip().isdigit() for f in fields):
            samples.append((time.time(), *map(int, fields)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    raw = args.directory.resolve()
    raw.mkdir(parents=True, exist_ok=False)
    cores = os.sched_getaffinity(0)
    topology = Path("/sys/devices/system/cpu")
    if (
        len(cores) != 4
        or len({(topology / f"cpu{c}/topology/thread_siblings_list").read_text() for c in cores})
        != 4
    ):
        parser.error("Pin the pilot to four different physical cores with taskset")
    order = ("gpu", "feed", "split", "cpu")
    runs = []
    for seed in (1, 2):
        for repeats in (1, 64):
            for arm in order if (seed == 1) == (repeats == 1) else order[::-1]:
                name = f"r{repeats}-s{seed}-{arm}"
                folder = raw / name
                env = {
                    **os.environ,
                    "JAX_PLATFORMS": "cpu" if arm == "cpu" else "cuda",
                    "JAX_ENABLE_X64": "true",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "JAX_ENABLE_COMPILATION_CACHE": "false",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                }
                samples = []
                monitor = None
                if arm != "cpu":
                    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
                    monitor = subprocess.Popen(
                        [
                            smi,
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                            "-l",
                            "1",
                        ],
                        stdout=subprocess.PIPE,
                        text=True,
                    )
                    reader = threading.Thread(target=monitor_gpu, args=(monitor.stdout, samples))
                    reader.start()
                try:
                    tic = time.perf_counter()
                    with (raw / (name + ".log")).open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(HERE / "run.py"),
                                str(folder),
                                arm,
                                str(seed),
                                str(repeats),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=600,
                            env=env,
                        )
                    process_wall = time.perf_counter() - tic
                finally:
                    if monitor is not None:
                        monitor.terminate()
                        monitor.wait(timeout=10)
                        reader.join()
                        monitor.stdout.close()
                row = json.loads((folder / "case.json").read_text())
                row["fresh_process_s"] = process_wall
                if monitor is not None:
                    usage = [
                        (u, m) for t, u, m in samples if row["start_epoch"] <= t <= row["end_epoch"]
                    ]
                    row["gpu_samples"] = len(usage)
                    row["device_util_mean_pct"] = (
                        statistics.mean(u for u, _ in usage) if usage else None
                    )
                    row["peak_device_memory_mib"] = max((m for _, m in usage), default=None)
                runs.append(row)
                (raw / "runs.json").write_text(json.dumps(runs, indent=2, allow_nan=False) + "\n")
                print(name, round(row["fit_wall_s"], 2), row["objective_cache_sizes"], flush=True)


if __name__ == "__main__":
    main()
