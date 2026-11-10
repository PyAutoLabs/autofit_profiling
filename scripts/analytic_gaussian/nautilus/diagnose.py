"""Prepare/run the approved copied-checkpoint investigation; no automatic retries."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts/misc"))
from profiling.nautilus_campaign import host_snapshot
from profiling.nautilus_diagnostics import (
    LIMITS,
    check_inputs,
    child,
    prepare,
    stage_summary,
    validate_plan,
)
from profiling.nautilus_resources import atomic_json, memory_available, supervise


def run(plan, root):
    validate_plan(plan)
    # Exclusive creation prevents duplicate/restarted execution and budget reset.
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "plan.json", plan)
    state = {"status": "running", "attempts": [], "spent_s": plan["preflight_s"]}
    atomic_json(root / "state.json", state)
    start = time.monotonic()
    try:
        for case in plan["cases"]:
            spent = plan["preflight_s"] + time.monotonic() - start
            if spent + LIMITS["timeout_s"] > LIMITS["budget_s"]:
                state.update(status="halted", reason="aggregate budget")
                break
            snapshot = host_snapshot()
            if memory_available() < 2 * LIMITS["memory_mib"] * 1024**2:
                raise RuntimeError("insufficient safe host memory")
            if snapshot["load"][0] > max(1.0, len(snapshot["affinity"]) / 4):
                raise RuntimeError("host load exceeds guard")
            check_inputs(plan)
            directory = root / case["id"]
            directory.mkdir()
            row = {"case": case["id"], "status": "running", "host": snapshot}
            state["attempts"].append(row)
            atomic_json(root / "state.json", state)
            with (directory / "run.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "child",
                        "--plan",
                        str(root / "plan.json"),
                        "--case",
                        case["id"],
                        "--output",
                        str(directory),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                tick = time.monotonic()
                supervision = supervise(
                    process, LIMITS["timeout_s"], LIMITS["memory_mib"] * 1024**2
                )
            row.update(
                process_wall_s=time.monotonic() - tick,
                supervision=supervision,
                stages=stage_summary(directory / "stages.jsonl"),
            )
            receipt = directory / "child.json"
            if receipt.exists():
                row["child"] = json.loads(receipt.read_text())
            row["status"] = supervision["termination_reason"] or (
                "success"
                if supervision["returncode"] == 0
                and row.get("child", {}).get("status") == "success"
                else "error"
            )
            check_inputs(plan)  # Also check after capped/failed children.
            row["original_unchanged"] = True
            atomic_json(directory / "result.json", row)
            state["spent_s"] = plan["preflight_s"] + time.monotonic() - start
            atomic_json(root / "state.json", state)
            print(f"{case['id']}: {row['status']} ({row['process_wall_s']:.2f}s)", flush=True)
            if row["status"] != "success":
                state.update(status="halted", reason=row["status"])
                break
        else:
            state["status"] = "complete"
    except BaseException as error:
        state.update(status="halted", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["spent_s"] = plan["preflight_s"] + time.monotonic() - start
        state["unrun"] = [
            c["id"] for c in plan["cases"] if c["id"] not in {a["case"] for a in state["attempts"]}
        ]
        atomic_json(root / "state.json", state)
    return 0 if state["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "child"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    if args.command == "prepare":
        if args.output.exists():
            raise ValueError("refusing to overwrite frozen plan")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, prepare(args.archive.resolve()))
        return 0
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.command == "run":
        return run(plan, args.output.resolve())
    case = next(c for c in plan["cases"] if c["id"] == args.case)
    try:
        result = child(plan, case, args.output.resolve())
    except Exception as error:
        traceback.print_exc()
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    atomic_json(args.output / "child.json", result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
