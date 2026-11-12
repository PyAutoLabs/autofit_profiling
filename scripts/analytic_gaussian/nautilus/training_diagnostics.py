"""One approved 17-child RAL diagnostic; no local scientific-run mode or retry."""

import argparse
import contextlib
import json
import os
import platform
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
from profiling.nautilus_diagnostics import assert_environment, sha256
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_plot_comparison import replaced
from profiling.nautilus_resources import atomic_json
from profiling.nautilus_training_diagnostics import (
    CorpusWriter,
    DiagnosticTransport,
    PhaseTrace,
    TrainingScope,
    array_identity,
    forbid_sampling,
    runtime_metadata,
    validate_corpus,
)

PROTOCOL = "nautilus-training-diagnostics@2-reference-node"
REFERENCE_NODE = "euclid-ral-compute-1"
LIMITS = {"children": 17, "timeout_s": 120, "memory_mib": 4096, "budget_s": 1800}
REFERENCE = ROOT / "experiments/nautilus_spawn_training_pool/diagnostics_10d/reference.json"
REFERENCE_SHA256 = "5d424a9c4c3344330d6fefbe0a391b0dc5b121cb45bc2af468af79ff1e99ede0"


def reference():
    if sha256(REFERENCE) != REFERENCE_SHA256:
        raise ValueError("retained reference changed")
    return json.loads(REFERENCE.read_text())


def matrix():
    rows = [{"id": "capture", "phase": "capture", "training_workers": 0, "bound": False}]
    arms = {"P": (0, False), "S": (1, False), "U": (2, False), "B": (2, True)}
    for repeat, order in enumerate(("PSUB", "BUSP", "UPBS")):
        for arm in order:
            workers, bound = arms[arm]
            rows.append(
                {
                    "id": f"replay-{repeat}-{arm}",
                    "phase": "replay",
                    "arm": arm,
                    "repeat": repeat,
                    "training_workers": workers,
                    "bound": bound,
                }
            )
    for i, workers in enumerate((0, 2, 2, 0)):
        rows.append(
            {
                "id": f"fit-{i}-w{workers}",
                "phase": "fit",
                "pair": i // 2,
                "training_workers": workers,
                "bound": False,
            }
        )
    return [{"dimension": 10, "seed": 2, **r} for r in rows]


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


def validate_node(receipt):
    if receipt["identity"]["host"] != REFERENCE_NODE:
        raise ValueError("diagnostic requires the original reference node")


def check_allocation(plan):
    current = allocation.snapshot()
    validate_node(current)
    allocation.verify_identity(plan["allocation"]["identity"], current)
    return current


def exact_fit(result, expected):
    base.exact_result(result, expected)
    if (
        result["target_sha256"] != expected["target_sha256"]
        or result["transport"]["models"] != expected["transport"]["models"]
        or result["checkpoint_reopen_equal"] is not True
    ):
        raise ValueError("target, ordered models or checkpoint parity differs")


def cpu_identity():
    allowed = set(os.sched_getaffinity(0))
    fields = {
        "processor",
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "microcode",
        "flags",
    }
    try:
        processors = []
        for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
            row = {
                key.strip(): value.strip()
                for line in block.splitlines()
                if ":" in line
                for key, value in [line.split(":", 1)]
            }
            if int(row.get("processor", -1)) in allowed:
                processors.append({k: v for k, v in row.items() if k in fields})
        return {"uname": platform.uname()._asdict(), "processors": processors}
    except (OSError, ValueError) as error:
        return {"unavailable": str(error), "uname": platform.uname()._asdict()}


def require_verified_capture(capture):
    if (
        capture.get("status") != "success"
        or capture.get("historical_reference_verified") is not True
    ):
        raise ValueError("replay requires a successful historically verified capture")
    exact_fit(capture, reference())


def fit(case, directory):
    atomic_json(
        directory / "parent-runtime-before-fit.json",
        {
            "status": "before_fit",
            "cpu_identity": cpu_identity(),
            "runtime": runtime_metadata(),
        },
    )
    trace = PhaseTrace() if case["phase"] == "fit" else None
    corpus = CorpusWriter(directory / "corpus") if case["phase"] == "capture" else None
    with contextlib.ExitStack() as stack:
        if trace:
            stack.enter_context(trace)
            for name in ("wrapped_fit", "verify_checkpoint", "retain_checkpoint"):
                trace.patch(base, name, "full_fit" if name == "wrapped_fit" else name)
            stack.enter_context(trace.scope("workflow"))
        stack.enter_context(replaced(base, "TrainingTransport", DiagnosticTransport))
        stack.enter_context(
            replaced(
                base, "TrainingScope", lambda transport: TrainingScope(transport, corpus, trace)
            )
        )
        result = base.performance(case, directory)
    atomic_json(
        directory / "candidate.json",
        {**result, "status": "unverified", "historical_reference_verified": False},
    )
    if corpus:
        result["corpus_sha256"] = corpus.finish()
        if [r["model_sha256"] for r in corpus.rows] != result["transport"]["models"]:
            raise ValueError("capture model order differs")
    if trace:
        result["phases"] = trace.receipt()
    atomic_json(
        directory / "candidate.json",
        {**result, "status": "unverified", "historical_reference_verified": False},
    )
    exact_fit(result, reference())
    result["historical_reference_verified"] = True
    return result


def replay(case, directory):
    corpus = directory.parent / "capture/corpus"
    capture = json.loads((directory.parent / "capture/child.json").read_text())
    require_verified_capture(capture)
    manifest, arrays = validate_corpus(corpus)
    if (
        capture["status"] != "success"
        or manifest["sha256"] != capture["corpus_sha256"]
        or [row["model_sha256"] for row, _, _ in arrays] != reference()["transport"]["models"]
    ):
        raise ValueError("corpus differs from verified capture")
    from nautilus.neural import NeuralNetworkEmulator

    # Equal eager loading policy in every arm, outside all training/lifecycle timers.
    started = time.monotonic()
    with forbid_sampling() as forbidden:
        with (
            DiagnosticTransport(case["training_workers"], case["bound"]) as transport,
            TrainingScope(transport),
        ):
            for row, x, y in arrays:
                before = [array_identity(x), array_identity(y)]
                NeuralNetworkEmulator.train(x, y)
                if transport.models[-1] != row["model_sha256"]:
                    raise ValueError("replay model differs from capture")
                if before != [array_identity(x), array_identity(y)]:
                    raise ValueError("replay mutated training inputs")
    elapsed = time.monotonic() - started
    final, _ = validate_corpus(corpus)
    if final != manifest:
        raise ValueError("retained corpus changed during replay")
    return {
        "transport": transport.receipt(),
        "workflow_s": elapsed,
        "corpus_sha256": manifest["sha256"],
        "forbidden_calls": forbidden,
        "exact_models": True,
        "inputs_unchanged": True,
    }


def child(plan, case, directory):
    from profiling.nautilus_baseline import provenance

    validate(plan)
    if case not in matrix():
        raise ValueError("case outside approved matrix")
    check_allocation(plan)
    assert_environment(environment_key(provenance()), plan["environment"])
    result = replay(case, directory) if case["phase"] == "replay" else fit(case, directory)
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
            if case["phase"] != "replay":
                exact_fit(row["child"], reference())
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
