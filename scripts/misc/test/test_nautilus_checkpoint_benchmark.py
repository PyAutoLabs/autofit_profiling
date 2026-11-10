"""Fixed-budget admission, ordering and intentional-kill supervision."""

import importlib.util
from pathlib import Path

import pytest
from profiling.nautilus_manifest import fingerprint

SOURCE = Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/checkpoint_benchmark.py"
spec = importlib.util.spec_from_file_location("checkpoint_benchmark", SOURCE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def plan():
    result = {
        "protocol": benchmark.PROTOCOL,
        "limits": benchmark.LIMITS.copy(),
        "cases": benchmark.matrix(),
        "configurations": [benchmark.configuration(c).matching() for c in benchmark.matrix()],
    }
    result["sha256"] = fingerprint(result)
    return result


@pytest.mark.parametrize(
    "change", ["budget", "order", "networks", "ess", "discard", "workers", "interval", "hash"]
)
def test_changed_protocol_rejected(change):
    p = plan()
    benchmark.validate(p)
    if change == "budget":
        p["limits"]["budget_s"] += 1
    elif change == "order":
        p["cases"].reverse()
    elif change == "networks":
        p["configurations"][0]["constructor"]["n_networks"] = 0
    elif change == "ess":
        p["configurations"][0]["stopping"]["n_eff"] = 500
    elif change == "discard":
        p["configurations"][0]["stopping"]["discard_exploration"] = False
    elif change == "workers":
        p["cases"][0]["workers"] = 4
    elif change == "interval":
        p["cases"][-1]["interval"] = 10
    else:
        p["sha256"] = "bad"
    if change != "hash":
        p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError):
        benchmark.validate(p)


def test_all_resume_witnesses_precede_performance():
    cases = benchmark.matrix()
    assert len(cases) == 20
    assert [c["phase"] for c in cases[:8]] == ["interrupt", "resume"] * 4
    assert all(c["phase"] == "performance" for c in cases[8:])
    assert {(c["workers"], c["mode"]) for c in cases[:8]} == {
        (1, "deferred"),
        (1, "atomic"),
        (2, "deferred"),
        (2, "atomic"),
    }


@pytest.mark.parametrize(
    "condition,reason",
    [("marker", "intentional_checkpoint_kill"), ("memory", "memory_cap"), ("timeout", "timeout")],
)
def test_supervisor_kills_group_at_boundary_or_cap(tmp_path, monkeypatch, condition, reason):
    class Process:
        pid = 42
        returncode = None

        def poll(self):
            return None

    process = Process()
    killed = []
    monkeypatch.setattr(benchmark, "terminate_group", lambda proc: killed.append(proc.pid))
    monkeypatch.setattr(
        benchmark, "group_rss", lambda pid: 5000 * 1024**2 if condition == "memory" else 1
    )
    times = iter([0.0, 121.0] if condition == "timeout" else [0.0, 0.0])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(times))
    if condition == "marker":
        (tmp_path / "paused.json").write_text("{}")
    result = benchmark.supervise_interruption(process, tmp_path)
    assert result["termination_reason"] == reason
    assert killed == [42]


def test_completed_directory_cannot_be_reused(tmp_path):
    with pytest.raises(FileExistsError):
        benchmark.run(tmp_path)
