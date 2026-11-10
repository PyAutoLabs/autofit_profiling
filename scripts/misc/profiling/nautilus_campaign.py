"""Off-board experiment manifests, identity and crash-safe attempt accounting."""

import copy
import json
import math
import os
import platform
import time
from pathlib import Path

from profiling.nautilus_manifest import RunConfig, assert_matching, fingerprint, source_pins
from profiling.nautilus_resources import atomic_json

PROTOCOL = "nautilus-cpu-experiment@1"
TARGET = {"family": "ar-gaussian", "correlation": 0.6, "prior": "independent N(0,1)"}


def environment_key(env):
    return {
        "sources": source_pins(env["sources"]),
        "nautilus_source_sha256": env["nautilus_source_sha256"],
        "packages": env["packages"],
        "distribution_versions": env["distribution_versions"],
        "python": env["python"],
        "host": env["host"],
        "platform": env["platform"],
        "affinity": env["affinity"],
        "thread_env": env["thread_env"],
    }


def row_identity(row, *, adapter=True, seed=False, trace=False):
    """Complete grouping identity; runtime target truth and host are never implicit."""
    manifest = row["manifest"]
    config = copy.deepcopy(manifest["config"])
    if not adapter:
        config.pop("adapter")
    if not seed:
        config.pop("seed")
    if not trace:
        config.pop("trace")
    return fingerprint(
        {
            "config": config,
            "target": manifest["target_sha256"],
            "environment": environment_key(manifest["environment"]),
        }
    )


def make_plan(
    identity, dimensions=(5, 10, 30), seeds=(0, 1, 2), budget_s=5400, timeout_s=120, memory_mib=4096
):
    if len(set(dimensions)) != len(dimensions) or len(set(seeds)) != len(seeds):
        raise ValueError("duplicate dimensions/seeds")
    if not dimensions or not seeds or list(dimensions) != sorted(dimensions):
        raise ValueError("dimensions must be nonempty and ascending; seeds nonempty")
    if any(not math.isfinite(v) or v <= 0 for v in (budget_s, timeout_s, memory_mib)):
        raise ValueError("resource limits must be finite and positive")
    if not 0 <= identity["preflight_s"] < budget_s:
        raise ValueError("preflight exhausted budget")
    cases = []
    for dimension in dimensions:
        target_digest = identity["truth_digests"][str(dimension)]
        for seed in seeds:
            for adapter in ("direct", "pyautofit"):
                for trace in (False, True) if seed % 2 == 0 else (True, False):
                    config = RunConfig(
                        adapter=adapter, dimension=dimension, seed=seed, trace=trace, n_live=400
                    ).to_dict()
                    logical = {
                        "protocol": PROTOCOL,
                        "target": TARGET,
                        "truth_sha256": target_digest,
                        "config": config,
                        "environment": identity["environment"],
                    }
                    cases.append({"id": "case-" + fingerprint(logical)[:20], **logical})
    plan = {
        "protocol": PROTOCOL,
        "purpose": "experimental, not a Pulse campaign",
        "environment": identity["environment"],
        "cases": cases,
        "limits": {"budget_s": budget_s, "timeout_s": timeout_s, "memory_mib": memory_mib},
        "preflight_s": identity["preflight_s"],
    }
    plan["sha256"] = fingerprint(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    payload = {k: v for k, v in plan.items() if k != "sha256"}
    if plan["sha256"] != fingerprint(payload) or plan["protocol"] != PROTOCOL:
        raise ValueError("experiment plan changed or unsupported protocol")
    for key, maximum in (("budget_s", 5400), ("timeout_s", 120), ("memory_mib", 4096)):
        value = plan["limits"][key]
        if not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("resource limit outside experimental protocol")
    if not isinstance(plan["limits"]["memory_mib"], int):
        raise ValueError("memory_mib must be an integer")
    ids = [case["id"] for case in plan["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case identity")
    expected = {
        (d, seed, adapter, trace)
        for d in (5, 10, 30)
        for seed in (0, 1, 2)
        for adapter in ("direct", "pyautofit")
        for trace in (False, True)
    }
    actual = set()
    for case in plan["cases"]:
        config = RunConfig(**case["config"])
        actual.add((config.dimension, config.seed, config.adapter, config.trace))
        fixed = RunConfig(
            dimension=config.dimension,
            seed=config.seed,
            adapter=config.adapter,
            trace=config.trace,
            n_live=400,
        )
        if config != fixed or case["target"] != TARGET or case["protocol"] != PROTOCOL:
            raise ValueError("settings differ from the frozen experimental protocol")
        logical = {k: v for k, v in case.items() if k != "id"}
        if case["id"] != "case-" + fingerprint(logical)[:20]:
            raise ValueError("case identity does not match configuration")
        assert_matching(plan["environment"], case["environment"])
    if actual != expected or len(plan["cases"]) != len(expected):
        raise ValueError("incomplete or expanded experimental matrix")


def verify_result(case, row, directory=None):
    if row["status"] != "success":
        return
    manifest = row["manifest"]
    assert_matching(case["config"], manifest["config"])
    assert_matching(RunConfig(**case["config"]).matching(), manifest["matching"])
    assert_matching(case["environment"], environment_key(manifest["environment"]))
    if manifest["target_sha256"] != case["truth_sha256"]:
        raise ValueError("target truth mismatch")
    if fingerprint(manifest["target"]) != case["truth_sha256"]:
        raise ValueError("target truth digest mismatch")
    if not row["run_calls"] or not row["run_calls"][-1]["converged"]:
        raise ValueError("missing stopping witness")
    if case["config"]["trace"]:
        if not row.get("timeline") or not row.get("trace"):
            raise ValueError("trace-incomplete")
        if row["trace"]["completed"] != row["n_like"]:
            raise ValueError("trace-incomplete")
        if directory is not None:
            import gzip
            import hashlib

            from profiling.nautilus_trace import reconcile

            path = Path(directory) / "timeline.json.gz"
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["timeline"]["sha256"]
            ):
                raise ValueError("trace-incomplete")
            timeline = json.loads(gzip.decompress(path.read_bytes()))
            reconcile(timeline["events"], timeline["batches"])


def compare_pair(left, right):
    assert_matching(left["manifest"]["matching"], right["manifest"]["matching"])
    for key in ("posterior_sha256", "n_like", "run_calls", "resolved_constructor"):
        if left[key] != right[key]:
            raise ValueError(f"numerical/cadence mismatch: {key}")


def process_token(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def new_state(plan):
    return {
        "plan_sha256": plan["sha256"],
        "spent_s": plan["preflight_s"],
        "halt_reason": None,
        "attempts": [],
        "selected": {},
        "sessions": [],
    }


def recover(state, plan, root):
    """Conservatively charge interrupted attempts their full reserved timeout."""
    if state["plan_sha256"] != plan["sha256"]:
        raise ValueError("resume plan differs")
    for attempt in state["attempts"]:
        if attempt["status"] != "running":
            continue
        if (
            attempt.get("pid")
            and attempt.get("process_token")
            and process_token(attempt["pid"]) == attempt["process_token"]
        ):
            raise RuntimeError("previous supervisor still alive; do not launch a duplicate")
        attempt["status"] = "interrupted"
        attempt["reason"] = "driver stopped before authoritative terminal receipt"
        state["spent_s"] += plan["limits"]["timeout_s"]
        # An interrupted attempt is retained even if a child wrote success; do not
        # silently promote it or retry a completed failed measurement.
        result = Path(root) / attempt["id"] / "result.json"
        if result.exists():
            row = json.loads(result.read_text())
            attempt["child_status"] = row["status"]
            if row["status"] != "success":
                state["halt_reason"] = "recovered child " + row["status"]
    return state


def next_case(state, plan):
    if state["halt_reason"]:
        return None
    remaining = plan["limits"]["budget_s"] - state["spent_s"]
    if remaining < plan["limits"]["timeout_s"]:
        state["halt_reason"] = "budget exhausted; remaining cases unrun"
        return None
    return next((c for c in plan["cases"] if c["id"] not in state["selected"]), None)


def host_snapshot():
    return {
        "host": platform.node(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "load": list(os.getloadavg()),
        "time": time.time(),
    }


def save_state(root, state):
    atomic_json(Path(root) / "state.json", state)
