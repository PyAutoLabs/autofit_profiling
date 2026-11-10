"""Explicit plan/run doors for the off-board dimension experiment; never submits jobs."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from profiling.nautilus_campaign import (
    compare_pair,
    environment_key,
    host_snapshot,
    make_plan,
    new_state,
    next_case,
    process_token,
    recover,
    row_identity,
    save_state,
    validate_plan,
    verify_result,
)
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_resources import atomic_json, memory_available


def identity(dimensions):
    # Scientific imports only at this explicit preflight door, never plan mode.
    start = time.monotonic()
    from profiling.nautilus_baseline import provenance
    from profiling.nautilus_target import CorrelatedGaussianTarget

    return {
        "environment": environment_key(provenance()),
        "truth_digests": {
            str(d): fingerprint(CorrelatedGaussianTarget(d).truth()) for d in dimensions
        },
        "preflight_s": time.monotonic() - start,
    }


def run(plan, root):
    validate_plan(plan)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another experiment driver holds this directory") from exc
        existing = root / "plan.json"
        if existing.exists():
            if json.loads(existing.read_text()) != plan:
                raise ValueError("immutable plan differs from saved plan")
        else:
            if (root / "state.json").exists():
                raise ValueError("state without plan")
            atomic_json(existing, plan)
        state_path = root / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else new_state(plan)
        recover(state, plan, root)
        session = {"start": host_snapshot(), "status": "running"}
        state["sessions"].append(session)
        save_state(root, state)
        successes = []
        cases = {c["id"]: c for c in plan["cases"]}
        for case_id, attempt_id in state["selected"].items():
            previous = json.loads((root / attempt_id / "result.json").read_text())
            if previous["status"] != "success":
                raise ValueError("selected attempt is not successful")
            verify_result(cases[case_id], previous, root / attempt_id)
            successes.append(previous)
        last = time.monotonic()
        while True:
            state["spent_s"] += time.monotonic() - last
            last = time.monotonic()
            case = next_case(state, plan)
            if case is None:
                break
            # Refuse unsafe/contended starts, never cancel another workload.
            cap = plan["limits"]["memory_mib"] * 1024**2
            snapshot = host_snapshot()
            if memory_available() < 2 * cap:
                state["halt_reason"] = "insufficient safe host memory"
                break
            if snapshot["load"][0] > max(1.0, len(snapshot["affinity"]) / 4):
                state["halt_reason"] = "host not quiet; no new case admitted"
                break
            number = 1 + sum(a["case_id"] == case["id"] for a in state["attempts"])
            attempt = {
                "id": f"{case['id']}-attempt-{number:03}",
                "case_id": case["id"],
                "status": "running",
                "host_before": snapshot,
            }
            state["attempts"].append(attempt)
            save_state(root, state)
            config = case["config"]
            command = [
                sys.executable,
                str(Path(__file__).with_name("cpu_baseline.py")),
                "--output",
                str(root / attempt["id"]),
                "--timeout",
                str(plan["limits"]["timeout_s"]),
                "--expected-plan",
                str(root / "plan.json"),
                "--memory-limit-mib",
                str(plan["limits"]["memory_mib"]),
            ]
            for key in (
                "adapter",
                "workers",
                "seed",
                "dimension",
                "cost_profile",
                "work",
                "n_live",
                "n_eff",
                "n_batch",
            ):
                command += ["--" + key.replace("_", "-"), str(config[key])]
            command += ["--trace", "on" if config["trace"] else "off"]
            if not config["checkpoint"]:
                command.append("--no-checkpoint")
            # Outer log is retained even when child preflight fails before mkdir.
            with (root / (attempt["id"] + ".log")).open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                attempt.update(pid=process.pid, process_token=process_token(process.pid))
                save_state(root, state)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait(timeout=10)
                    raise
            result_path = root / attempt["id"] / "result.json"
            row = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {"status": "missing"}
            )
            attempt.update(status=row["status"], returncode=code, host_after=host_snapshot())
            if code == 0 and row["status"] == "success":
                try:
                    verify_result(case, row, result_path.parent)
                    for previous in successes:
                        if row_identity(previous, seed=True) == row_identity(row, seed=True):
                            if previous["manifest"]["config"]["trace"] == config["trace"]:
                                raise ValueError("duplicate selected run")
                            compare_pair(previous, row)
                        if row_identity(
                            previous, adapter=False, seed=True, trace=True
                        ) == row_identity(row, adapter=False, seed=True, trace=True):
                            compare_pair(previous, row)
                    state["selected"][case["id"]] = attempt["id"]
                    successes.append(row)
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    attempt.update(status="validation_failure", reason=str(exc))
            if attempt["status"] != "success" or code:
                state["halt_reason"] = attempt["status"] + "; no escalation or automatic retry"
            state["spent_s"] += time.monotonic() - last
            last = time.monotonic()
            save_state(root, state)
            print(
                json.dumps(
                    {
                        "dimension": config["dimension"],
                        "adapter": config["adapter"],
                        "seed": config["seed"],
                        "trace": config["trace"],
                        "status": attempt["status"],
                        "wall_s": row.get("wall_s"),
                    }
                ),
                flush=True,
            )
        session.update(
            status="stopped" if state["halt_reason"] else "complete", end=host_snapshot()
        )
        save_state(root, state)
        print(
            json.dumps(
                {
                    "selected": len(state["selected"]),
                    "expected": len(plan["cases"]),
                    "halt_reason": state["halt_reason"],
                    "spent_s": state["spent_s"],
                }
            ),
            flush=True,
        )
        return 0 if len(state["selected"]) == len(plan["cases"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("identity")
    probe.add_argument("--output", required=True, type=Path)
    probe.add_argument("--dimensions", nargs="+", type=int, default=[5, 10, 30])
    probe.add_argument(
        "--preflight-seconds",
        type=float,
        default=0,
        help="additional timed runtime checks charged to the pilot budget",
    )
    plan = commands.add_parser("plan")
    plan.add_argument("--identity", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--memory-mib", type=int, default=4096)
    execute = commands.add_parser("run")
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        return run(json.loads(args.plan.read_text()), args.output.resolve())
    if args.output.exists():
        raise FileExistsError("refusing to replace existing identity/plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "identity":
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[key] = "1"
        value = identity(args.dimensions)
        if args.preflight_seconds < 0:
            raise ValueError("preflight seconds cannot be negative")
        value["preflight_s"] += args.preflight_seconds
    else:
        value = make_plan(json.loads(args.identity.read_text()), memory_mib=args.memory_mib)
    atomic_json(args.output, value)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
