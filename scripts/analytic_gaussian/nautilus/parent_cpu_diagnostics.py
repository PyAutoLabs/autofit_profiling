"""One approved 12-fit parent CPU study; RAL only, exact parity, no retries."""

import argparse
import contextlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_pool_benchmark as base
from profiling import nautilus_allocation as allocation
from profiling.nautilus_campaign import environment_key
from profiling.nautilus_diagnostics import assert_environment
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_parent_cpu import (
    Observer,
    ParentTrace,
    ParentTransport,
    verify_phase_counts,
)
from profiling.nautilus_plot_comparison import replaced
from profiling.nautilus_resources import atomic_json
from profiling.nautilus_training_diagnostics import TrainingScope, runtime_metadata
from training_diagnostics import (
    REFERENCE_NODE,
    REFERENCE_SHA256,
    check_allocation,
    cpu_identity,
    exact_fit,
    reference,
    validate_node,
)

PROTOCOL = "nautilus-parent-cpu-diagnostics@1"
LIMITS = {"children": 12, "timeout_s": 120, "memory_mib": 4096, "budget_s": 1800}


def matrix():
    return [
        {
            "id": f"fit-{i:02d}-{arm}",
            "phase": "fit",
            "arm": arm,
            "dimension": 10,
            "seed": 2,
            "training_workers": 2 if arm in "BCF" else 0,
        }
        for i, arm in enumerate("ABCDEFFEDCBA")
    ]


def validate(plan):
    if plan["sha256"] != fingerprint({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("frozen diagnostic plan changed")
    if (
        plan["protocol"] != PROTOCOL
        or plan["cases"] != matrix()
        or plan["limits"] != LIMITS
        or plan["required_host"] != REFERENCE_NODE
        or plan["reference_sha256"] != REFERENCE_SHA256
        or plan["configuration"] != base.configuration(matrix()[0]).matching()
        or plan["allocation"]["identity"]["policy"] != allocation.POLICY
    ):
        raise ValueError("unapproved diagnostic protocol/configuration/allocation")
    validate_node(plan["allocation"])
    reference()


def fit(case, directory):
    atomic_json(
        directory / "parent-runtime-before-fit.json",
        {"status": "before_fit", "cpu_identity": cpu_identity(), "runtime": runtime_metadata()},
    )
    trace = ParentTrace()
    observer = Observer(trace)
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(trace)
            for name in ("wrapped_fit", "verify_checkpoint", "retain_checkpoint"):
                trace.patch(base, name, "full_fit" if name == "wrapped_fit" else name)
            stack.enter_context(trace.scope("workflow"))
            stack.enter_context(observer)
            observer.snapshot("before_fit", runtime=True)
            stack.enter_context(
                replaced(
                    base,
                    "TrainingTransport",
                    lambda workers: ParentTransport(workers, case["arm"], observer),
                )
            )
            stack.enter_context(
                replaced(
                    base,
                    "TrainingScope",
                    lambda transport: TrainingScope(
                        transport,
                        timer=trace,
                        before=transport.before_training,
                        after=transport.after_training,
                    ),
                )
            )
            result = base.performance(case, directory)
            observer.snapshot("after_fit", runtime=True)
    finally:
        # Keep observations even if sampling/parity/accounting fails; never mark verified here.
        atomic_json(directory / "observations.json", observer.receipt())
        atomic_json(
            directory / "phase-events.json", {"status": "unverified", "events": trace.events}
        )
    atomic_json(
        directory / "candidate.json",
        {**result, "status": "unverified", "historical_reference_verified": False},
    )
    result["phases"] = trace.receipt()
    result["observations"] = observer.receipt()
    atomic_json(
        directory / "candidate.json",
        {**result, "status": "unverified", "historical_reference_verified": False},
    )
    exact_fit(result, reference())
    verify_phase_counts(result["phases"])
    if (
        len(result["transport"]["calls"]) != 11
        or result["transport"]["close_count"] != 1
        or result["transport"]["closed"] is not True
    ):
        raise ValueError("training lifecycle differs")
    result["historical_reference_verified"] = True
    return result


def child(plan, case, directory):
    from profiling.nautilus_baseline import provenance

    validate(plan)
    if case not in matrix():
        raise ValueError("case outside approved matrix")
    check_allocation(plan)
    assert_environment(environment_key(provenance()), plan["environment"])
    result = fit(case, directory)
    assert_environment(environment_key(provenance()), plan["environment"])
    check_allocation(plan)
    return {"status": "success", **result}


def run(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "attempts": []}
    atomic_json(root / "state.json", state)
    try:
        from profiling.nautilus_baseline import provenance

        initial = allocation.snapshot()
        validate_node(initial)
        os.environ["PYAUTO_TRAINING_AFFINITY"] = json.dumps(initial["identity"]["affinity"])
        identity = provenance()
        base.validate_sources(identity)
        base.validate_runtime(identity)
        if any(s["dirty"] for s in identity["sources"].values()):
            raise ValueError("measurement sources must be clean")
        plan = {
            "protocol": PROTOCOL,
            "required_host": REFERENCE_NODE,
            "cases": matrix(),
            "limits": LIMITS,
            "configuration": base.configuration(matrix()[0]).matching(),
            "reference_sha256": REFERENCE_SHA256,
            "allocation": initial,
            "environment": environment_key(identity),
        }
        plan["sha256"] = fingerprint(plan)
        validate(plan)
        atomic_json(root / "plan.json", plan)
        for case in matrix():
            admission = check_allocation(plan)
            if time.monotonic() - started + LIMITS["timeout_s"] > LIMITS["budget_s"]:
                raise RuntimeError("aggregate budget exhausted before child")
            directory = root / case["id"]
            directory.mkdir()
            row = {"case": case["id"], "status": "running", "allocation": admission}
            state["attempts"].append(row)
            atomic_json(root / "state.json", state)
            with (directory / "run.log").open("w") as log:
                tick = time.monotonic()
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
                # Identical per-child caps to the original benchmark; no change to its matrix.
                receipt = base.supervise(
                    process,
                    directory,
                    False,
                    initial["identity"]["affinity"],
                    started + LIMITS["budget_s"],
                )
            row.update(process_wall_s=time.monotonic() - tick, supervision=receipt)
            path = directory / "child.json"
            if path.exists():
                row["child"] = json.loads(path.read_text())
            row["status"] = receipt["termination_reason"] or (
                "success"
                if receipt["returncode"] == 0 and row.get("child", {}).get("status") == "success"
                else "error"
            )
            atomic_json(directory / "result.json", row)
            if row["status"] != "success":
                raise RuntimeError(f"{case['id']}: {row['status']}")
            exact_fit(row["child"], reference())
            verify_phase_counts(row["child"]["phases"])
            assert_environment(environment_key(provenance()), plan["environment"])
            check_allocation(plan)
            state["spent_s"] = time.monotonic() - started
            if state["spent_s"] > LIMITS["budget_s"]:
                raise RuntimeError("aggregate budget exhausted during verification")
            atomic_json(root / "state.json", state)
            print(f"{case['id']}: success ({row['process_wall_s']:.2f}s)", flush=True)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="halted", reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        state["spent_s"] = time.monotonic() - started
        done = {r["case"] for r in state["attempts"]}
        state["unrun"] = [c["id"] for c in matrix() if c["id"] not in done]
        atomic_json(root / "state.json", state)
    return 0 if state["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "child"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--execution", choices=("ral",), default="ral")
    args = parser.parse_args()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    if args.command == "run":
        return run(args.output.resolve())
    try:
        plan = json.loads(args.plan.read_text())
        case = next(c for c in plan["cases"] if c["id"] == args.case)
        result = child(plan, case, args.output.resolve())
    except Exception as error:
        traceback.print_exc()
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    atomic_json(args.output / "child.json", result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
