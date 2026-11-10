"""One bounded, off-Pulse AB/BA/AB comparison; never retries a stopped run."""

import argparse
import contextlib
import copy
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts/misc"))
from profiling.nautilus_campaign import environment_key, host_snapshot
from profiling.nautilus_diagnostics import (
    StageLog,
    assert_environment,
    check_inputs,
    estimate,
    output_replay,
    runtime_identity,
    sha256,
    stage_summary,
)
from profiling.nautilus_diagnostics import (
    validate_plan as validate_input_plan,
)
from profiling.nautilus_manifest import fingerprint, source_pins
from profiling.nautilus_resources import atomic_json, memory_available, supervise

LIMITS = {"timeout_s": 240, "memory_mib": 4096, "budget_s": 1800}
ORDER = ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"]
PROTOCOL = "nautilus-compact-array-experiment@1"


def validate(plan):
    if plan["sha256"] != fingerprint({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("frozen plan changed")
    if plan["protocol"] != PROTOCOL or plan["limits"] != LIMITS:
        raise ValueError("unapproved protocol or limits")
    if [c["backend"] for c in plan["cases"]] != ORDER:
        raise ValueError("unapproved case order")
    for index, case in enumerate(plan["cases"]):
        if (
            case["id"] != f"pair-{index // 2 + 1}-{ORDER[index]}"
            or case["arm"] != "output"
            or case["original_n_like"] != 34100
            or case["dimension"] != 30
            or case["summary_timing_call_limit"] != 1
            or case["original_selected_samples"] != 33984
            or case.get("attribution", False)
            or case["plot_policy"]
            != {"name": "thirty-small", "indices": list(range(30)), "figsize": [12, 12], "dpi": 100}
        ):
            raise ValueError("unapproved case policy")


def prepare(input_plan, baseline_fit):
    from nautilus import Sampler
    from profiling.nautilus_baseline import git_identity, provenance
    from profiling.nautilus_manifest import RunConfig
    from profiling.nautilus_plot_comparison import font_cache
    from profiling.nautilus_target import CorrelatedGaussianTarget

    old = json.loads(input_plan.read_text())
    validate_input_plan(old)
    if old["scope"] != "plot-attribution":
        raise ValueError("expected the frozen original attribution input")
    check_inputs(old)
    actual = provenance()
    baseline_source = git_identity(baseline_fit)
    if baseline_source["dirty"] or any(s["dirty"] for s in actual["sources"].values()):
        raise ValueError("all measured sources must be committed and clean")
    candidate = environment_key(actual)
    baseline = copy.deepcopy(candidate)
    baseline["sources"]["PyAutoFit"] = source_pins({"PyAutoFit": baseline_source})["PyAutoFit"]
    assert_environment(runtime_identity(baseline), runtime_identity(old["environment"]))
    template = copy.deepcopy(old["cases"][0])
    template.pop("attribution", None)
    target = CorrelatedGaussianTarget(30)
    sampler = Sampler(
        target.prior,
        target,
        filepath=template["checkpoint"],
        **RunConfig(dimension=30, n_live=400, trace=False).constructor(),
    )
    restored = estimate(sampler, target)
    if (
        restored["posterior_sha256"] != template["original_posterior_sha256"]
        or fingerprint(target.truth()) != template["target_sha256"]
    ):
        raise ValueError("saved posterior/target differs")
    plan = {
        "protocol": PROTOCOL,
        "limits": LIMITS,
        "font_cache": font_cache(),
        "environments": {"baseline": baseline, "candidate": candidate},
        "source_roots": {
            "baseline": str(baseline_fit.resolve()),
            "candidate": actual["sources"]["PyAutoFit"]["root"],
            "nerves": actual["sources"]["PyAutoNerves"]["root"],
        },
        "input_plan": str(input_plan.resolve()),
        "input_plan_sha256": sha256(input_plan),
        "cases": [
            {**template, "id": f"pair-{i // 2 + 1}-{backend}", "backend": backend}
            for i, backend in enumerate(ORDER)
        ],
    }
    plan["sha256"] = fingerprint(plan)
    validate(plan)
    return plan


def child(plan, case, directory):
    from profiling.nautilus_baseline import provenance
    from profiling.nautilus_plot_comparison import font_cache, replaced
    from profiling.nautilus_target import CorrelatedGaussianTarget

    validate(plan)
    check_inputs(plan)
    expected = plan["environments"][case["backend"]]
    assert_environment(environment_key(provenance()), expected)
    if font_cache() != plan["font_cache"]:
        raise ValueError("font cache changed")
    target = CorrelatedGaussianTarget(30)
    if fingerprint(target.truth()) != case["target_sha256"]:
        raise ValueError("target changed")
    checkpoint = directory / "checkpoint.hdf5"
    shutil.copyfile(case["checkpoint"], checkpoint)
    log = StageLog(directory / "stages.jsonl")
    adapter_calls = []
    with contextlib.ExitStack() as stack:
        if case["backend"] == "candidate":
            from autofit.non_linear.plot import compact_corner, nest_plotters

            original = nest_plotters.corner_anesthetic

            def array_corner(*args, **kwargs):
                return original(*args, compact_backend="array", **kwargs)

            stack.enter_context(replaced(nest_plotters, "corner_anesthetic", array_corner))
            draw = compact_corner.draw

            @functools.wraps(draw)
            def witnessed(*args, **kwargs):
                adapter_calls.append(True)
                with log.stage("array_posterior_plotting"):
                    return draw(*args, **kwargs)

            stack.enter_context(replaced(compact_corner, "draw", witnessed))
        result = output_replay(case, checkpoint, target, directory, log)
    if len(adapter_calls) != (case["backend"] == "candidate"):
        raise ValueError("candidate unexpectedly fell back")
    if sha256(checkpoint) != case["checkpoint_sha256"]:
        raise ValueError("replay changed copied checkpoint")
    assert_environment(environment_key(provenance()), expected)
    check_inputs(plan)
    return {"status": "success", "adapter_calls": len(adapter_calls), **result}


def run(input_plan, baseline_fit, root):
    start = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "attempts": [], "unrun": list(range(6))}
    atomic_json(root / "state.json", state)
    try:
        plan = prepare(input_plan, baseline_fit)
        atomic_json(root / "plan.json", plan)
        state["preflight_s"] = time.monotonic() - start
        for case in plan["cases"]:
            if time.monotonic() - start + LIMITS["timeout_s"] > LIMITS["budget_s"]:
                raise RuntimeError("aggregate budget exhausted")
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
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [plan["source_roots"][case["backend"]], plan["source_roots"]["nerves"]]
            )
            with (directory / "run.log").open("w") as output:
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
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
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
            check_inputs(plan)
            atomic_json(directory / "result.json", row)
            if row["status"] != "success":
                raise RuntimeError(f"{case['id']}: {row['status']}")
            if len(state["attempts"]) % 2 == 0:
                a, b = state["attempts"][-2:]
                if a["child"]["plot"]["png"]["sha256"] != b["child"]["plot"]["png"]["sha256"]:
                    raise ValueError("paired PNGs differ; exact parity required")
            state["spent_s"] = time.monotonic() - start
            atomic_json(root / "state.json", state)
            print(f"{case['id']}: {row['process_wall_s']:.2f}s", flush=True)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="halted", reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        state["spent_s"] = time.monotonic() - start
        state["unrun"] = list(range(len(state["attempts"]), 6))
        atomic_json(root / "state.json", state)
    return 0 if state["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "child"))
    parser.add_argument("--input-plan", type=Path)
    parser.add_argument("--baseline-fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    if args.command == "run":
        return run(args.input_plan, args.baseline_fit, args.output.resolve())
    plan = json.loads(args.plan.read_text())
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
