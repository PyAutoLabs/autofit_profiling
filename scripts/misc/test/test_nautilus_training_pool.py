"""Admission, isolation and ordered-transport contracts without a science stack."""

import contextlib
import functools
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from profiling import nautilus_training_worker as worker
from profiling.nautilus_manifest import fingerprint

SOURCE = (
    Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/training_pool_benchmark.py"
)
spec = importlib.util.spec_from_file_location("training_benchmark", SOURCE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def plan():
    value = {
        "protocol": benchmark.PROTOCOL,
        "cases": benchmark.matrix(),
        "configurations": [benchmark.configuration(c).matching() for c in benchmark.matrix()],
        "limits": benchmark.LIMITS.copy(),
        "quiet_admission": [
            {"time": t, "load": [0.5, 1, 1], "available_mib": 8192} for t in (0, 30, 60)
        ],
    }
    value["sha256"] = fingerprint(value)
    return value


def test_retained_checkpoint_survives_fit_finalization(tmp_path):
    source, retained = tmp_path / "native.hdf5", tmp_path / "retained.hdf5"
    source.write_bytes(b"first native state")
    benchmark.retain_checkpoint(source, retained)
    assert source.samefile(retained)
    source.unlink()  # Fit discards the resume file after writing its final archive.
    assert retained.read_bytes() == b"first native state"
    source.write_bytes(b"later native state")
    benchmark.retain_checkpoint(source, retained)
    source.unlink()
    assert retained.read_bytes() == b"later native state"


def test_retention_does_not_destroy_source_or_hide_missing_checkpoint(tmp_path):
    source = tmp_path / "native.hdf5"
    source.write_bytes(b"state")
    with pytest.raises(ValueError, match="separate path"):
        benchmark.retain_checkpoint(source, source)
    assert source.read_bytes() == b"state"
    retained = tmp_path / "retained"
    retained.write_bytes(b"previous state")
    with pytest.raises(FileNotFoundError):
        benchmark.retain_checkpoint(tmp_path / "missing", retained)
    assert retained.read_bytes() == b"previous state"


@pytest.mark.parametrize("fault", [None, "state", "write", "likelihood"])
def test_checkpoint_reopen_keeps_nonidentity_prior_and_read_only_contract(
    tmp_path, monkeypatch, fault
):
    retained = tmp_path / "checkpoint.json"
    retained.write_text(json.dumps({"points": [0.25, 0.75], "n_like": 100}))
    original = retained.read_bytes()
    calls = []

    class Sampler:
        def __init__(self, prior, likelihood, filepath, **kwargs):
            self.prior = prior
            self.state = json.loads(Path(filepath).read_text())
            calls.append(kwargs)
            if fault == "state":
                self.state["n_like"] += 1
            if fault == "write":
                Path(filepath).write_bytes(original + b" ")
            if fault == "likelihood":
                likelihood([0.5])

        def run(self, *args, **kwargs):
            pytest.fail("verification must never sample")

        def receipt(self):
            arrays = ([self.prior(x) for x in self.state["points"]], [0.2, 0.8], [-2, -1])
            return {
                "posterior_sha256": hashlib.sha256(json.dumps(arrays).encode()).hexdigest(),
                "n_like": self.state["n_like"],
                "n_eff": 2.0,
                "shells": 1,
                "explored": True,
                "discard_exploration": True,
            }

    # Non-identity coordinates are essential: the former lambda x:x fails here.
    live = object.__new__(Sampler)
    live.prior = lambda x: 4 * x - 2
    live.state = json.loads(original)
    expected = live.receipt()
    monkeypatch.setitem(sys.modules, "nautilus", SimpleNamespace(Sampler=Sampler))
    monkeypatch.setitem(
        sys.modules,
        "autofit.non_linear.search.nest.nautilus.search",
        SimpleNamespace(_large_attribute_hdf5=contextlib.nullcontext),
    )
    monkeypatch.setattr(benchmark, "posterior", lambda sampler: sampler.receipt())
    config = benchmark.configuration(benchmark.matrix()[6])
    errors = {
        "state": "sampling parity differs",
        "write": "changed bytes",
        "likelihood": "must not evaluate likelihood",
    }
    if fault:
        with pytest.raises((ValueError, RuntimeError), match=errors[fault]):
            benchmark.verify_checkpoint(live, config, retained, expected)
    else:
        assert (
            benchmark.verify_checkpoint(live, config, retained, expected)
            == hashlib.sha256(original).hexdigest()
        )
        assert calls == [config.constructor()]
        assert retained.read_bytes() == original


@pytest.mark.parametrize(
    "mutation", ["order", "workers", "networks", "ess", "discard", "cap", "quiet", "hash"]
)
def test_frozen_protocol_cannot_be_expanded(mutation):
    value = plan()
    benchmark.validate(value)
    if mutation == "order":
        value["cases"].reverse()
    elif mutation == "workers":
        value["cases"][-1]["training_workers"] = 4
    elif mutation == "networks":
        value["configurations"][-1]["constructor"]["n_networks"] = 0
    elif mutation == "ess":
        value["configurations"][-1]["stopping"]["n_eff"] = 500
    elif mutation == "discard":
        value["configurations"][-1]["stopping"]["discard_exploration"] = False
    elif mutation == "cap":
        value["limits"]["timeout_s"] += 1
    elif mutation == "quiet":
        value["quiet_admission"][-1]["time"] = 40
    else:
        value["sha256"] = "bad"
    if mutation != "hash":
        value["sha256"] = fingerprint({k: v for k, v in value.items() if k != "sha256"})
    with pytest.raises(ValueError):
        benchmark.validate(value)


def test_correctness_cases_gate_interleaved_timing_matrix():
    rows = benchmark.matrix()
    assert len(rows) == 18
    assert [r["phase"] for r in rows[:6]] == [
        "preflight",
        "control",
        "interrupt",
        "resume",
        "interrupt",
        "resume",
    ]
    assert all(r["phase"] == "performance" for r in rows[6:])
    assert [r["training_workers"] for r in rows[6:12]] == [0, 2, 2, 0, 0, 2]


def test_planning_imports_no_scientific_packages():
    code = (
        "import runpy,sys; runpy.run_path(sys.argv[1]); "
        "assert not {'numpy','sklearn','nautilus','autofit','jax'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-S", "-c", code, str(SOURCE)], check=True, timeout=10)


def fake_network(x, y, kwargs, seed):
    return {"seed": seed}


def test_native_training_map_keeps_all_four_seeds_in_order(monkeypatch):
    monkeypatch.setitem(sys.modules, "nautilus.neural", SimpleNamespace(train_network=fake_network))
    transport = worker.TrainingTransport(0)
    function = functools.partial(fake_network, SimpleNamespace(shape=(64, 5)), [0], {})
    result = transport.map(function, range(4))
    assert result == [{"seed": i} for i in range(4)]
    assert len(transport.calls[0]["networks"]) == 4
    assert transport.calls[0]["input_pickle_bytes"] > 0
    with pytest.raises(ValueError):
        transport.map(function, [0, 1])
    with pytest.raises(ValueError):
        transport.map(str, range(4))


def test_training_hook_restores_original_descriptor_after_failure(monkeypatch):
    class Emulator:
        @classmethod
        def train(cls, *args, **kwargs):
            assert kwargs["n_networks"] == 4
            assert kwargs["pool"] is transport
            raise RuntimeError("training failed")

    transport = worker.TrainingTransport(0)
    monkeypatch.setitem(
        sys.modules, "nautilus.neural", SimpleNamespace(NeuralNetworkEmulator=Emulator)
    )
    original = vars(Emulator)["train"]
    with pytest.raises(RuntimeError, match="training failed"), worker.TrainingScope(transport):
        Emulator.train([], [])
    assert vars(Emulator)["train"] is original
    with worker.TrainingScope(transport), pytest.raises(ValueError):
        Emulator.train([], [], n_networks=0)
    assert vars(Emulator)["train"] is original


@pytest.mark.parametrize("workers", [-1, 1, 3, 4])
def test_training_worker_count_is_bounded(workers):
    with pytest.raises(ValueError):
        worker.TrainingTransport(workers)


@pytest.mark.parametrize(
    "condition,reason",
    [("marker", "intentional_checkpoint_kill"), ("memory", "memory_cap"), ("timeout", "timeout")],
)
def test_supervisor_cleans_group_at_all_stop_conditions(tmp_path, monkeypatch, condition, reason):
    process = SimpleNamespace(pid=42, returncode=None, poll=lambda: None)
    cleaned = []
    monkeypatch.setattr(benchmark, "terminate_group", lambda p: cleaned.append(p.pid))
    monkeypatch.setattr(benchmark, "live_group", lambda pid: [])
    monkeypatch.setattr(
        benchmark, "group_rss", lambda pid: 5000 * 1024**2 if condition == "memory" else 1
    )
    ticks = iter(range(0, 2000, 121))
    monkeypatch.setattr(
        benchmark.time, "monotonic", lambda: next(ticks) if condition == "timeout" else 0
    )
    if condition == "marker":
        (tmp_path / "paused.json").write_text("{}")
    result = benchmark.supervise(process, tmp_path, True)
    assert result["termination_reason"] == reason
    assert cleaned == [42]


def test_admission_wait_is_bounded_and_charged(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark.time, "sleep", lambda t: clock.__setitem__(0, clock[0] + t))
    monkeypatch.setattr(benchmark, "memory_available", lambda: 9000 * 1024**2)
    monkeypatch.setattr(
        benchmark, "host_snapshot", lambda: {"load": [3, 3, 3], "affinity": list(range(8))}
    )
    ok, reason, host, wait = benchmark.admit(0)
    assert not ok and wait == 60 and "admission" in reason
    clock[0] = 1681
    ok, reason, _, _ = benchmark.admit(0)
    assert not ok and "budget" in reason


def test_live_group_accepts_reaped_or_zombie_but_detects_live(tmp_path):
    for pid, state, group in [(10, "S", 42), (11, "Z", 42), (12, "S", 43)]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(f"{pid} (name with ) spaces) {state} 1 {group} 0")
    assert benchmark.live_group(42, tmp_path) == [10]


def test_completed_attempt_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        benchmark.run(tmp_path, None)


@pytest.mark.parametrize("source", ["PyAutoFit", "PyAutoNerves", "nautilus"])
def test_wrong_baseline_source_is_rejected(source):
    identity = {
        "sources": {key: {"revision": value} for key, value in benchmark.SOURCE_PINS.items()},
        "nautilus_source_sha256": benchmark.NAUTILUS_SOURCE,
    }
    benchmark.validate_sources(identity)
    if source == "nautilus":
        identity["nautilus_source_sha256"] = "different"
    else:
        identity["sources"][source]["revision"] = "different"
    with pytest.raises(ValueError):
        benchmark.validate_sources(identity)


def test_worker_failure_propagates_without_resubmission(monkeypatch):
    monkeypatch.setitem(sys.modules, "nautilus.neural", SimpleNamespace(train_network=fake_network))
    submitted = []

    class Failed:
        def result(self):
            raise RuntimeError("worker failed")

    class Executor:
        def submit(self, func, native, seed, path):
            submitted.append(seed)
            return Failed()

    transport = worker.TrainingTransport(2)
    transport.executor = Executor()
    function = functools.partial(fake_network, SimpleNamespace(shape=(64, 5)), [0], {})
    with pytest.raises(RuntimeError, match="worker failed"):
        transport.map(function, range(4))
    assert submitted == [0, 1, 2, 3]
    assert transport.calls == []


@pytest.mark.parametrize(
    "field", ["posterior_sha256", "n_like", "n_eff", "shells", "explored", "discard_exploration"]
)
def test_exact_sampling_parity_rejects_any_changed_field(field):
    result = dict(
        posterior_sha256="abc",
        n_like=100,
        n_eff=10000.0,
        shells=9,
        explored=True,
        discard_exploration=True,
    )
    benchmark.exact_result(result, result)
    other = {**result, field: None}
    with pytest.raises(ValueError):
        benchmark.exact_result(result, other)


def test_ral_protocol_cannot_import_laptop_admission():
    value = plan()
    value["protocol"] = benchmark.RAL_PROTOCOL
    value.pop("quiet_admission")
    value["allocation"] = {"identity": {"policy": benchmark.allocation.POLICY.copy()}}
    value["sha256"] = fingerprint({k: v for k, v in value.items() if k != "sha256"})
    benchmark.validate(value)
    value["quiet_admission"] = []
    value["sha256"] = fingerprint({k: v for k, v in value.items() if k != "sha256"})
    with pytest.raises(ValueError, match="mixed"):
        benchmark.validate(value)


def test_ral_allocation_commands_are_charged_to_aggregate_budget(monkeypatch):
    clock = [1675.0]
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark, "host_snapshot", lambda: {"load": [90, 90, 90]})

    def inspect(plan):
        clock[0] += 10
        return {"verified": True}

    monkeypatch.setattr(benchmark, "check_allocation", inspect)
    ok, reason, host, wait = benchmark.admit_ral(0, {})
    assert not ok and wait == 10 and "budget" in reason
    assert host["allocation"]["verified"]
    clock[0] = 0
    assert benchmark.admit_ral(0, {})[0]  # Host-wide load is evidence, not RAL admission.


def test_ral_without_verified_allocation_launches_no_children(tmp_path, monkeypatch):
    import json

    def reject():
        raise ValueError("no verified scheduler allocation")

    monkeypatch.setattr(benchmark.allocation, "snapshot", reject)
    root = tmp_path / "unadmitted"
    assert benchmark.run(root, None, "ral") == 1
    state = json.loads((root / "state.json").read_text())
    assert state["status"] == "halted" and state["attempts"] == []
    assert len(state["unrun"]) == 18


def test_ral_requires_approved_python_and_scientific_versions():
    import json

    ref = json.loads(
        (benchmark.ROOT / "experiments/nautilus_spawn_training_pool/runtime.json").read_text()
    )
    identity = {"python": ref["python"] + " other compiler", "packages": ref["packages"].copy()}
    benchmark.validate_runtime(identity)
    identity["packages"]["scikit-learn"] = "different"
    with pytest.raises(ValueError, match="dependencies"):
        benchmark.validate_runtime(identity)


@pytest.mark.parametrize("failure", ["affinity", "aggregate"])
def test_ral_supervisor_cleans_up_on_new_admission_failures(tmp_path, monkeypatch, failure):
    import time

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True
    )
    kwargs = {}
    if failure == "affinity":

        def reject(*args):
            raise ValueError("escaped")

        monkeypatch.setattr(benchmark.allocation, "verify_process_affinity", reject)
        kwargs["affinity"] = [0, 1, 2, 3]
    else:
        kwargs["budget_deadline"] = time.monotonic() - 1
    result = benchmark.supervise(process, tmp_path, False, **kwargs)
    assert result["termination_reason"] == (
        "affinity_mismatch" if failure == "affinity" else "aggregate_budget"
    )
    assert not result["live_descendants"] and process.poll() is not None


def test_ral_protocol_receipt_matches_executable_matrix_and_limits():
    import json

    value = json.loads(
        (benchmark.ROOT / "experiments/nautilus_spawn_training_pool/protocol_ral.json").read_text()
    )
    assert value["protocol"] == benchmark.RAL_PROTOCOL
    assert value["cases"] == benchmark.matrix()
    assert value["limits"] == benchmark.LIMITS
    assert value["allocation_policy"] == benchmark.allocation.POLICY
