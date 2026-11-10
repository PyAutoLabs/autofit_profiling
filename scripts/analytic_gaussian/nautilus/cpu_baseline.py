"""Run one fresh, supervised CPU baseline. See wiki/campaigns/nautilus_cpu_baseline.md."""

import argparse
import dataclasses
import json
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
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

            result = measure(config, directory)
        except Exception as error:
            traceback.print_exc()
            result = {
                "status": "error",
                "config": dataclasses.asdict(config),
                "error": f"{type(error).__name__}: {error}",
            }
        (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "success" else 1
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
        "command": sys.argv,
        "status": "requested",
    }
    (directory / "request.json").write_text(json.dumps(request, indent=2))
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
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
        if code and not (directory / "result.json").exists():
            (directory / "result.json").write_text(
                json.dumps(
                    {
                        "status": "timeout" if code == 124 else "process_failure",
                        "config": dataclasses.asdict(config),
                        "returncode": code,
                    },
                    indent=2,
                )
            )
    result = json.loads((directory / "result.json").read_text())
    result["process_wall_s"] = time.perf_counter() - start
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
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
