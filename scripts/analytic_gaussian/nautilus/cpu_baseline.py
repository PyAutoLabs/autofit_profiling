"""Run one fresh, supervised CPU baseline. See wiki/campaigns/nautilus_cpu_baseline.md."""

import argparse
import dataclasses
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from profiling.nautilus_manifest import RunConfig
from profiling.nautilus_resources import atomic_json, memory_available, supervise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("direct", "pyautofit"), default="direct")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument(
        "--cost-profile", choices=("none", "fixed", "random", "correlated"), default="fixed"
    )
    parser.add_argument("--work", type=int, default=100)
    parser.add_argument("--trace", choices=("on", "off"), default="on")
    parser.add_argument("--n-live", type=int, default=200)
    parser.add_argument("--n-eff", type=int, default=500)
    parser.add_argument("--n-batch", type=int, default=100)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--memory-limit-mib", type=int, help="Linux process-group RSS cap")
    parser.add_argument(
        "--expected-plan", type=Path, help="verify frozen environment before sampling"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.memory_limit_mib is not None and args.memory_limit_mib < 1:
        parser.error("memory limit must be positive")
    config = RunConfig(
        adapter=args.adapter,
        workers=args.workers,
        seed=args.seed,
        dimension=args.dimension,
        cost_profile=args.cost_profile,
        work=args.work,
        trace=args.trace == "on",
        n_live=args.n_live,
        n_eff=args.n_eff,
        n_batch=args.n_batch,
        checkpoint=not args.no_checkpoint,
    )
    directory = args.output.resolve()
    if args.child:
        try:
            from profiling.nautilus_baseline import measure

            expected = (
                json.loads(args.expected_plan.read_text())["environment"]
                if args.expected_plan
                else None
            )
            result = measure(config, directory, expected_environment=expected)
        except Exception as error:
            traceback.print_exc()
            result = {
                "status": "error",
                "config": dataclasses.asdict(config),
                "error": f"{type(error).__name__}: {error}",
            }
        (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "success" else 1
    memory_limit = args.memory_limit_mib * 1024**2 if args.memory_limit_mib else None
    if memory_limit and memory_available() < 2 * memory_limit:
        raise RuntimeError("unsafe host memory: require twice the requested RSS cap available")
    directory.mkdir(parents=True, exist_ok=False)  # Never silently reuse a checkpoint.
    env = os.environ.copy()
    env.update(
        {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
    )
    env["MPLBACKEND"] = "Agg"
    request = {
        "config": dataclasses.asdict(config),
        "timeout_s": args.timeout,
        "memory_limit_bytes": memory_limit,
        "command": sys.argv,
        "status": "requested",
    }
    atomic_json(directory / "request.json", request)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    start = time.perf_counter()
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            supervision = supervise(process, args.timeout, memory_limit)
        except KeyboardInterrupt:
            supervision = {"returncode": 130, "termination_reason": "interrupted"}
        code = supervision["returncode"]
        if code or not (directory / "result.json").exists():
            previous = directory / "result.json"
            previous_row = json.loads(previous.read_text()) if previous.exists() else {}
            if previous.exists():
                previous.replace(directory / "child_result.json")
            atomic_json(
                directory / "result.json",
                {
                    "status": supervision["termination_reason"]
                    or (
                        previous_row.get("status")
                        if previous_row.get("status") != "success"
                        else None
                    )
                    or "process_failure",
                    "config": dataclasses.asdict(config),
                    "returncode": code,
                    "error": previous_row.get("error"),
                },
            )
    result = json.loads((directory / "result.json").read_text())
    result["process_wall_s"] = time.perf_counter() - start
    result["supervision"] = supervision
    atomic_json(directory / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(directory),
                "wall_s": result.get("wall_s"),
                "error": result.get("error"),
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
