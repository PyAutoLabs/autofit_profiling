"""Frozen diagnostic contracts and negative gates; never runs a sampler or network fit."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from profiling import nautilus_training_diagnostics as d
from profiling.nautilus_manifest import fingerprint

SOURCE = Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/training_diagnostics.py"
spec = importlib.util.spec_from_file_location("diagnostic_benchmark", SOURCE)
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


def test_frozen_matrix_and_original_18_case_contract():
    b.validate(frozen())
    rows = b.matrix()
    assert len(rows) == 17
    assert [r["phase"] for r in rows] == ["capture"] + ["replay"] * 12 + ["fit"] * 4
    assert "".join(r["arm"] for r in rows if r["phase"] == "replay") == "PSUBBUSPUPBS"
    assert [r["training_workers"] for r in rows[-4:]] == [0, 2, 2, 0]
    assert all(r["dimension"] == 10 and r["seed"] == 2 for r in rows)
    assert len(b.base.matrix()) == b.base.LIMITS["children"] == 18
    with pytest.raises(ValueError):
        b.base.TrainingTransport(1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "other"),
        ("limits", {}),
        ("cases", []),
        ("configuration", {}),
        ("reference_sha256", "0" * 64),
    ],
)
def test_rehashed_protocol_changes_still_fail(field, value):
    plan = frozen()
    plan[field] = value
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="unapproved"):
        b.validate(plan)


def test_manifest_tampering_and_reference_changes(tmp_path, monkeypatch):
    plan = frozen()
    plan["cases"].reverse()
    with pytest.raises(ValueError, match="changed"):
        b.validate(plan)
    path = tmp_path / "ref"
    path.write_text("{}")
    monkeypatch.setattr(b, "REFERENCE", path)
    with pytest.raises(ValueError, match="reference changed"):
        b.reference()


@pytest.mark.parametrize(
    "key",
    [
        "posterior_sha256",
        "n_like",
        "n_eff",
        "shells",
        "explored",
        "discard_exploration",
        "target_sha256",
        "checkpoint_reopen_equal",
    ],
)
def test_fit_parity_is_strict(key):
    expected = b.reference()
    actual = copy.deepcopy(expected)
    actual[key] = None
    with pytest.raises(ValueError):
        b.exact_fit(actual, expected)


def test_ordered_model_parity_is_strict():
    expected = b.reference()
    actual = copy.deepcopy(expected)
    actual["transport"]["models"].reverse()
    with pytest.raises(ValueError, match="ordered models"):
        b.exact_fit(actual, expected)


@pytest.mark.parametrize(
    "workers,bound,masks",
    [
        (0, False, []),
        (1, False, [[2, 4, 6, 8]]),
        (2, False, [[2, 4, 6, 8]] * 2),
        (2, True, [[2], [4]]),
    ],
)
def test_worker_masks(workers, bound, masks):
    assert d.assignments(workers, bound, [2, 4, 6, 8]) == masks


@pytest.mark.parametrize(
    "workers,bound,allowed",
    [
        (3, False, [1, 2, 3, 4]),
        (1, True, [1, 2, 3, 4]),
        (2, True, [1, 1, 2, 3]),
        (2, True, [3, 2, 1, 0]),
    ],
)
def test_invalid_worker_masks(workers, bound, allowed):
    with pytest.raises(ValueError):
        d.assignments(workers, bound, allowed)


def test_worker_receipt_rejects_missing_threads_placement_and_processes():
    row = {
        "pid": 123,
        "pgid": os.getpgrp(),
        "affinity": [1],
        "start_method": "spawn",
        "autofit_imported": False,
        "jax_imported": False,
        "libraries": [{"num_threads": 1}],
    }
    d.validate_workers([row], [[1]])
    for key, value in (
        ("affinity", [2]),
        ("libraries", []),
        ("start_method", "fork"),
        ("autofit_imported", True),
        ("libraries", [{"num_threads": 2}]),
    ):
        with pytest.raises(ValueError):
            d.validate_workers([{**row, key: value}], [[1]])
    with pytest.raises(ValueError):
        d.validate_workers([row, row], [[1], [2]])


def events():
    return [
        {
            "dispatch_start_ns": 0,
            "collect_finish_ns": 100,
            "networks": [
                {
                    "seed": i,
                    "pid": i % 2,
                    "submit_ns": 0,
                    "start_ns": start,
                    "finish_ns": finish,
                    "collect_ns": 99,
                    "cpu_ns": 10,
                }
                for i, (start, finish) in enumerate(((10, 30), (15, 20), (40, 60), (50, 70)))
            ],
        }
    ]


def test_overlap_aware_union_including_contained_intervals():
    result = d.occupancy(events())
    assert result["any_worker_busy_ns"] == 50
    assert result["no_worker_busy_ns"] == 50
    assert result["workers"]["0"]["busy_ns"] == 40
    assert result["workers"]["1"]["busy_ns"] == 25


@pytest.mark.parametrize("fault", ["order", "timeline", "same_worker_overlap", "cpu"])
def test_bad_network_accounting(fault):
    rows = events()
    nets = rows[0]["networks"]
    if fault == "order":
        nets.reverse()
    elif fault == "timeline":
        nets[0]["collect_ns"] = 5
    elif fault == "same_worker_overlap":
        nets[2]["start_ns"] = 20
    else:
        nets[0]["cpu_ns"] = -1
    with pytest.raises(ValueError):
        d.occupancy(rows)


def test_nested_wall_and_cpu_reconcile_without_clamping():
    wall, cpu = iter([0, 10, 30, 100]), iter([0, 2, 7, 20])
    trace = d.PhaseTrace(lambda: next(wall), lambda: next(cpu))
    with trace.scope("root"), trace.scope("child"):
        pass
    result = trace.receipt()
    assert result["totals"]["child"]["exclusive_wall_ns"] == 20
    assert result["totals"]["root"]["exclusive_wall_ns"] == 80
    assert result["totals"]["root"]["exclusive_cpu_ns"] == 15
    trace.events[0]["exclusive_cpu_ns"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        trace.receipt()


def test_negative_clock_and_open_trace_fail():
    trace = d.PhaseTrace(iter([2, 1]).__next__, iter([0, 1]).__next__)
    with pytest.raises(ValueError, match="negative"):
        with trace.scope("bad"):
            pass
    trace = d.PhaseTrace()
    with trace.scope("still open"):
        with pytest.raises(ValueError, match="incomplete"):
            trace.receipt()


def test_descriptor_restoration_and_exception_accounting():
    class Owner:
        @classmethod
        def method(cls):
            raise RuntimeError("intentional")

        @staticmethod
        def static():
            return 4

    originals = dict(vars(Owner))
    trace = d.PhaseTrace()
    trace.patch(Owner, "method", "method")
    trace.patch(Owner, "static", "static")
    with pytest.raises(RuntimeError, match="intentional"):
        with trace.scope("root"):
            assert Owner.static() == 4
            Owner.method()
    trace.__exit__()
    assert vars(Owner)["method"] is originals["method"]
    assert vars(Owner)["static"] is originals["static"]
    assert trace.receipt()["accounting"] == "exact"


def make_corpus(root):
    np = pytest.importorskip("numpy")
    writer = d.CorpusWriter(root)
    x, y = np.arange(80, dtype=float).reshape(8, 10), np.arange(8, dtype=float)
    row = writer.before(x, y)
    writer.after(row, x, y, "a" * 64)
    writer.finish()
    return x, y


def test_numpy_corpus_roundtrip_and_layout(tmp_path):
    np = pytest.importorskip("numpy")
    root = tmp_path / "corpus"
    x, y = make_corpus(root)
    manifest, loaded = d.validate_corpus(root)
    np.testing.assert_array_equal(x, loaded[0][1])
    np.testing.assert_array_equal(y, loaded[0][2])
    assert manifest["calls"][0]["model_sha256"] == "a" * 64
    other = d.CorpusWriter(tmp_path / "fortran")
    x = np.asfortranarray(x)
    row = other.before(x, y)
    other.after(row, x, y, "b" * 64)
    other.finish()
    assert d.validate_corpus(other.root)[1][0][1].flags.f_contiguous


@pytest.mark.parametrize(
    "fault", ["bytes", "missing", "extra", "manifest", "order", "settings", "shape", "size"]
)
def test_corrupt_corpus_is_rejected(tmp_path, monkeypatch, fault):
    root = tmp_path / "corpus"
    make_corpus(root)
    if fault == "bytes":
        (root / "000-x.npy").write_bytes(b"bad")
    elif fault == "missing":
        (root / "000-y.npy").unlink()
    elif fault == "extra":
        (root / "extra").touch()
    elif fault == "size":
        monkeypatch.setattr(d, "CORPUS_BYTES", 1)
    else:
        path = root / "manifest.json"
        value = json.loads(path.read_text())
        if fault == "manifest":
            value["sha256"] = "0" * 64
        else:
            if fault == "order":
                value["calls"][0]["index"] = 1
            elif fault == "settings":
                value["calls"][0]["settings"]["n_networks"] = 2
            else:
                value["calls"][0]["arrays"]["x"]["shape"] = [10, 8]
            value["sha256"] = fingerprint({k: v for k, v in value.items() if k != "sha256"})
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        d.validate_corpus(root)


def test_empty_incomplete_mutated_and_object_corpus(tmp_path):
    np = pytest.importorskip("numpy")
    writer = d.CorpusWriter(tmp_path / "corpus")
    with pytest.raises(ValueError, match="empty"):
        writer.finish()
    x, y = np.zeros((2, 10)), np.zeros(2)
    row = writer.before(x, y)
    with pytest.raises(ValueError, match="incomplete"):
        writer.finish()
    x[0, 0] = 1
    with pytest.raises(ValueError, match="mutated"):
        writer.after(row, x, y, "a" * 64)
    with pytest.raises(ValueError, match="object"):
        d.array_identity(np.array([object()]))


def test_replay_forbidden_entrypoints_and_restoration(monkeypatch):
    class Sampler:
        def run(self):
            return "original"

        def evaluate_likelihood(self):
            pass

    class Target:
        def __call__(self):
            pass

    monkeypatch.setitem(sys.modules, "nautilus", SimpleNamespace(Sampler=Sampler))
    monkeypatch.setitem(
        sys.modules, "profiling.nautilus_target", SimpleNamespace(CorrelatedGaussianTarget=Target)
    )
    for call in (
        lambda: Sampler().run(),
        lambda: Sampler().evaluate_likelihood(),
        lambda: Target()(),
    ):
        with pytest.raises(RuntimeError, match="forbids"):
            with d.forbid_sampling():
                call()
        assert Sampler().run() == "original"
    with d.forbid_sampling() as counts:
        pass
    assert counts == {"sampler_run": 0, "evaluate_likelihood": 0, "target": 0}


def test_training_scope_restores_and_detects_changed_settings(monkeypatch):
    class Emulator:
        @classmethod
        def train(cls, *args, **kwargs):
            return "model"

    monkeypatch.setitem(
        sys.modules, "nautilus.neural", SimpleNamespace(NeuralNetworkEmulator=Emulator)
    )
    monkeypatch.setattr(d, "model_digest", lambda model: "a" * 64)
    original = vars(Emulator)["train"]
    transport = SimpleNamespace(models=[], training_wall_s=0)
    with d.TrainingScope(transport):
        assert Emulator.train([], []) == "model"
        with pytest.raises(ValueError):
            Emulator.train([], [], n_networks=2)
    assert vars(Emulator)["train"] is original
    assert transport.models == ["a" * 64]


def test_run_stops_on_first_child_failure_and_preserves_unrun(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYAUTO_TRAINING_AFFINITY", "[]"
    )  # Restore the environment after mocked run.
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
    assert len(calls) == 1
    assert state["status"] == "halted"
    assert len(state["unrun"]) == 16
    assert state["attempts"][0]["status"] == "timeout"
    with pytest.raises(FileExistsError):
        b.run(root)


def test_wrong_node_rejected_in_frozen_plan_and_live_admission(monkeypatch):
    plan = frozen()
    plan["allocation"]["identity"]["host"] = "euclid-ral-compute-10-2"
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="original reference node"):
        b.validate(plan)
    monkeypatch.setattr(b.allocation, "snapshot", lambda: plan["allocation"])
    monkeypatch.setattr(
        b.allocation, "verify_identity", lambda *a: pytest.fail("admitted wrong node")
    )
    with pytest.raises(ValueError, match="original reference node"):
        b.check_allocation(frozen())


def test_rehashed_node_requirement_cannot_change():
    plan = frozen()
    plan["required_host"] = "another-host"
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="unapproved"):
        b.validate(plan)


@pytest.mark.parametrize(
    "status,verified",
    [("unverified", False), ("error", True), ("success", False), ("success", None)],
)
def test_replay_refuses_unverified_capture_before_loading_or_training(
    tmp_path, monkeypatch, status, verified
):
    capture = {**b.reference(), "status": status, "historical_reference_verified": verified}
    directory = tmp_path / "capture"
    directory.mkdir()
    (directory / "child.json").write_text(json.dumps(capture))
    monkeypatch.setattr(b, "validate_corpus", lambda *a: pytest.fail("loaded rejected corpus"))
    with pytest.raises(ValueError, match="historically verified"):
        b.replay(b.matrix()[1], tmp_path / "replay-0-P")


@pytest.mark.parametrize("mismatch", [False, True])
def test_candidate_corpus_and_runtime_survive_historical_parity_failure(
    tmp_path, monkeypatch, mismatch
):
    class Corpus:
        def __init__(self, root):
            self.root = root
            root.mkdir()
            self.rows = [{"model_sha256": h} for h in b.reference()["transport"]["models"]]

        def finish(self):
            (self.root / "manifest.json").write_text(json.dumps(self.rows))
            return "a" * 64

    monkeypatch.setattr(b, "CorpusWriter", Corpus)
    monkeypatch.setattr(b, "runtime_metadata", lambda: {"libraries": ["retained metadata"]})
    monkeypatch.setattr(b, "cpu_identity", lambda: {"model name": "fixture"})
    result = copy.deepcopy(b.reference())
    result.pop("status")
    if mismatch:
        result["posterior_sha256"] = "0" * 64

    def performance(case, directory):
        assert (directory / "parent-runtime-before-fit.json").exists()
        return result

    monkeypatch.setattr(b.base, "performance", performance)
    if mismatch:
        with pytest.raises(ValueError, match="sampling parity"):
            b.fit(b.matrix()[0], tmp_path)
    else:
        assert b.fit(b.matrix()[0], tmp_path)["historical_reference_verified"] is True
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["status"] == "unverified"
    assert candidate["historical_reference_verified"] is False
    assert candidate["corpus_sha256"] == "a" * 64
    assert candidate["transport"]["models"] == b.reference()["transport"]["models"]
    assert (tmp_path / "corpus/manifest.json").is_file()
