"""Parent CPU study safety/diagnostic contracts; no sampling or network training."""

import contextlib
import gc
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from profiling import nautilus_parent_cpu as p
from profiling import nautilus_training_diagnostics as d
from profiling.nautilus_manifest import fingerprint

SOURCE = (
    Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/parent_cpu_diagnostics.py"
)
spec = importlib.util.spec_from_file_location("parent_cpu_benchmark", SOURCE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def frozen():
    plan = {
        "protocol": b.PROTOCOL,
        "required_host": b.REFERENCE_NODE,
        "cases": b.matrix(),
        "limits": b.LIMITS.copy(),
        "reference_sha256": b.REFERENCE_SHA256,
        "configuration": b.base.configuration(b.matrix()[0]).matching(),
        "allocation": {"identity": {"policy": b.allocation.POLICY, "host": b.REFERENCE_NODE}},
    }
    plan["sha256"] = fingerprint(plan)
    return plan


def test_matrix_and_old_contracts():
    b.validate(frozen())
    assert len(b.matrix()) == b.LIMITS["children"] == 12
    assert "".join(r["arm"] for r in b.matrix()) == "ABCDEFFEDCBA"
    assert all(r["dimension"] == 10 and r["seed"] == 2 for r in b.matrix())
    assert b.LIMITS == {"children": 12, "timeout_s": 120, "memory_mib": 4096, "budget_s": 1800}
    import training_diagnostics as prior

    assert len(prior.matrix()) == 17
    assert len(b.base.matrix()) == 18


@pytest.mark.parametrize(
    "key,value",
    [
        ("cases", []),
        ("limits", {}),
        ("required_host", "other"),
        ("reference_sha256", "0" * 64),
        ("protocol", "other"),
        ("configuration", {}),
    ],
)
def test_rehashed_changes_rejected(key, value):
    plan = frozen()
    plan[key] = value
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="unapproved"):
        b.validate(plan)


def test_wrong_node_and_tampering():
    plan = frozen()
    plan["cases"].reverse()
    with pytest.raises(ValueError, match="changed"):
        b.validate(plan)
    plan = frozen()
    plan["allocation"]["identity"]["host"] = "wrong"
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="original reference node"):
        b.validate(plan)


def test_three_channel_nested_accounting_and_corruption():
    trace = p.ParentTrace(
        iter([0, 10, 30, 100]).__next__, iter([0, 2, 7, 20]).__next__, iter([0, 1, 4, 10]).__next__
    )
    with trace.scope("root"), trace.scope("child"):
        pass
    rows = trace.receipt()["totals"]
    assert rows["root"]["exclusive_thread_ns"] == 7
    assert rows["child"]["exclusive_thread_ns"] == 3
    assert rows["root"]["exclusive_cpu_ns"] == 15
    trace.events[0]["exclusive_thread_ns"] += 1
    with pytest.raises(ValueError, match="thread accounting"):
        trace.receipt()


def test_negative_thread_time_and_restore_after_exception():
    trace = p.ParentTrace(thread=iter([2, 1]).__next__)
    with pytest.raises(ValueError, match="negative"):
        with trace.scope("bad"):
            pass
    before = gc.callbacks[:]
    trace = p.ParentTrace()
    with pytest.raises(RuntimeError):
        with p.Observer(trace) as observer:
            observer.on_gc("start", {"generation": 0})
            raise RuntimeError("stop")
    assert gc.callbacks == before and trace.observer is None
    assert observer.collections[0]["info"]["generation"] == 0


def test_gc_overflow_is_not_silently_ignored():
    observer = p.Observer(p.ParentTrace())
    observer.collections = [{}] * 20000
    observer.on_gc("start", {})
    with pytest.raises(ValueError, match="GC event cap"):
        observer.receipt()


def test_coarse_observations_only(monkeypatch):
    trace = p.ParentTrace()
    observer = p.Observer(trace)
    monkeypatch.setattr(p, "process_snapshot", lambda pid: {"pid": pid})
    with observer:
        with trace.scope("full_fit"):
            with trace.scope("NeuralNetworkEmulator.predict"):
                pass
            with trace.scope("Sampler.evaluate_likelihood"):
                pass
    assert len(observer.samples) == 2
    assert trace.receipt()["totals"]["observation"]["calls"] == 2


def test_proc_parser_handles_parentheses_and_missing_process():
    fields = ["0"] * 40
    for i in (7, 9, 11, 12, 19, 36):
        fields[i] = str(i)
    result = p.proc_stat("42 (name (with parens)) " + " ".join(fields))
    assert result == {
        "user_ticks": 11,
        "system_ticks": 12,
        "minor_faults": 7,
        "major_faults": 9,
        "start_ticks": 19,
        "cpu": 36,
    }
    fields[11] = "-1"
    with pytest.raises(ValueError):
        p.proc_stat("42 (x) " + " ".join(fields))
    assert "unavailable" in p.process_snapshot(999999999)


def observer_stub():
    return SimpleNamespace(
        trace=SimpleNamespace(scope=lambda phase: contextlib.nullcontext()),
        snapshot=lambda *a, **kw: None,
    )


def test_early_shutdown_once_and_later_dispatch_forbidden(monkeypatch):
    monkeypatch.setattr(p.os, "sched_getaffinity", lambda tid: {0, 1, 2, 3})
    exits = []
    monkeypatch.setattr(d.DiagnosticTransport, "__exit__", lambda *a: exits.append(1))
    transport = p.ParentTransport(2, "C", observer_stub())
    transport.calls = [{}] * 10
    transport.after_training()
    assert not exits
    transport.calls.append({})
    transport.after_training()
    transport.__exit__()
    assert exits == [1] and transport.closed and transport.close_count == 1
    with pytest.raises(ValueError, match="unexpected training"):
        transport.before_training()
    with pytest.raises(ValueError, match="dispatch"):
        transport.map(None, [])


def test_disjoint_masks_and_affinity_restoration_on_failure(monkeypatch):
    original = set(os.sched_getaffinity(0))
    allowed = set(sorted(original)[:4])
    if len(allowed) != 4:
        pytest.skip("requires four allowed CPUs; no scientific work")
    os.sched_setaffinity(0, allowed)
    monkeypatch.setattr(d.DiagnosticTransport, "__enter__", lambda self: self)
    monkeypatch.setattr(d.DiagnosticTransport, "__exit__", lambda *a: None)
    try:
        transport = p.ParentTransport(2, "F", observer_stub())
        assert transport.masks == [[cpu] for cpu in sorted(allowed)[1:3]]
        with pytest.raises(RuntimeError):
            with transport:
                assert os.sched_getaffinity(0) == {min(allowed)}
                raise RuntimeError("stop")
        assert os.sched_getaffinity(0) == allowed
    finally:
        os.sched_setaffinity(0, original)


def test_roundtrip_protocol_and_numeric_values():
    np = pytest.importorskip("numpy")
    network = SimpleNamespace(
        coefs_=[np.arange(24.0).reshape(4, 6)], intercepts_=[np.arange(6.0)], n_iter_=12
    )
    restored, row = p.pickle_roundtrip(network)
    assert restored is not network
    assert row["default_protocol"] == pickle.DEFAULT_PROTOCOL
    assert row["effective_protocol"] == bytes([128, pickle.DEFAULT_PROTOCOL]).hex()
    np.testing.assert_array_equal(restored.coefs_[0], network.coefs_[0])
    assert row["before"][0]["strides"] == [48, 8]


def test_training_callbacks_outside_native_timer_and_restoration(monkeypatch):
    order = []

    class Emulator:
        @classmethod
        def train(cls, *args, **kwargs):
            order.append("native")
            return "model"

    original = vars(Emulator)["train"]
    monkeypatch.setitem(
        sys.modules, "nautilus.neural", SimpleNamespace(NeuralNetworkEmulator=Emulator)
    )
    monkeypatch.setattr(d, "model_digest", lambda model: order.append("digest") or "digest")
    t = SimpleNamespace(models=[], training_wall_s=0)
    with d.TrainingScope(
        t, before=lambda: order.append("before"), after=lambda: order.append("after")
    ):
        Emulator.train([], [])
    assert order == ["before", "native", "digest", "after"]
    assert vars(Emulator)["train"] is original


def test_phase_count_mismatch_stops_interpretation():
    with pytest.raises(ValueError, match="phase count differs"):
        p.verify_phase_counts({"totals": {}})


def test_fail_stop_and_duplicate_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("PYAUTO_TRAINING_AFFINITY", "[]")
    identity = {"sources": {"producer": {"dirty": False}}}
    monkeypatch.setitem(
        sys.modules, "profiling.nautilus_baseline", SimpleNamespace(provenance=lambda: identity)
    )
    monkeypatch.setattr(
        b.allocation,
        "snapshot",
        lambda: {
            "identity": {
                "affinity": [0, 1, 2, 3],
                "policy": b.allocation.POLICY,
                "host": b.REFERENCE_NODE,
            }
        },
    )
    monkeypatch.setattr(b, "check_allocation", lambda plan: {})
    monkeypatch.setattr(b.base, "validate_sources", lambda i: None)
    monkeypatch.setattr(b.base, "validate_runtime", lambda i: None)
    monkeypatch.setattr(b, "environment_key", lambda i: {})
    calls = []
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **kw: calls.append(a) or object())
    monkeypatch.setattr(
        b.base, "supervise", lambda *a: {"termination_reason": "timeout", "returncode": -15}
    )
    root = tmp_path / "run"
    assert b.run(root) == 1
    state = json.loads((root / "state.json").read_text())
    assert len(calls) == 1 and state["status"] == "halted" and len(state["unrun"]) == 11
    with pytest.raises(FileExistsError):
        b.run(root)


def test_runtime_allowlist_and_wrapper_identity(tmp_path):
    path = SOURCE.parents[3] / "hpc/tools/nautilus_runtime_cache.py"
    spec = importlib.util.spec_from_file_location("parent_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    identity = runtime.experiment_identity(SOURCE.parents[3], "parent-cpu-diagnostics")
    runtime.validate_experiment(
        {"experiment": identity}, SOURCE.parents[3], "parent-cpu-diagnostics"
    )
    with pytest.raises(ValueError):
        runtime.validate_experiment(
            {"experiment": identity}, SOURCE.parents[3], "training-diagnostics"
        )
    wrapper = (SOURCE.parents[3] / identity["wrapper"]).read_text()
    assert "--nodelist=euclid-ral-compute-1" in wrapper
    assert 'mkdir "$receipt"' in wrapper and "--no-requeue" in wrapper
    assert "parent-cpu-diagnostics-ral-01" in wrapper
    assert "ral-parent-cpu-environment-01.sh" in wrapper


def test_candidate_survives_exact_parity_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(p.ParentTrace, "__enter__", lambda self: self)
    monkeypatch.setattr(b, "runtime_metadata", lambda: {})
    monkeypatch.setattr(b, "cpu_identity", lambda: {})
    monkeypatch.setattr(p.Observer, "snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(b.base, "performance", lambda *a: b.reference())
    monkeypatch.setattr(b, "exact_fit", lambda *a: (_ for _ in ()).throw(ValueError("parity")))
    with pytest.raises(ValueError, match="parity"):
        b.fit(b.matrix()[0], tmp_path)
    receipt = json.loads((tmp_path / "candidate.json").read_text())
    assert receipt["historical_reference_verified"] is False
    assert receipt["status"] == "unverified"
    assert (tmp_path / "observations.json").is_file()
    assert (tmp_path / "phase-events.json").is_file()


def test_aggregate_deadline_prevents_first_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("PYAUTO_TRAINING_AFFINITY", "[]")
    identity = {"sources": {"producer": {"dirty": False}}}
    monkeypatch.setitem(
        sys.modules, "profiling.nautilus_baseline", SimpleNamespace(provenance=lambda: identity)
    )
    monkeypatch.setattr(
        b.allocation,
        "snapshot",
        lambda: {
            "identity": {
                "affinity": [0, 1, 2, 3],
                "policy": b.allocation.POLICY,
                "host": b.REFERENCE_NODE,
            }
        },
    )
    monkeypatch.setattr(b, "check_allocation", lambda plan: {})
    monkeypatch.setattr(b.base, "validate_sources", lambda i: None)
    monkeypatch.setattr(b.base, "validate_runtime", lambda i: None)
    monkeypatch.setattr(b, "environment_key", lambda i: {})
    monkeypatch.setattr(
        b.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched past deadline")
    )
    ticks = iter([0, 1790, 1791])
    monkeypatch.setattr(b.time, "monotonic", ticks.__next__)
    root = tmp_path / "run"
    assert b.run(root) == 1
    state = json.loads((root / "state.json").read_text())
    assert state["attempts"] == [] and len(state["unrun"]) == 12
