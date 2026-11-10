"""Diagnostic safety, resumable-sampler semantics, and incomplete event evidence."""

import copy
import json
from types import SimpleNamespace

import pytest
from profiling.nautilus_diagnostics import (
    CASES,
    LIMITS,
    PROTOCOL,
    StageLog,
    assert_environment,
    check_inputs,
    discard_saved_exploration,
    fresh_counts,
    runtime_identity,
    sha256,
    stage_summary,
    validate_plan,
)
from profiling.nautilus_manifest import fingerprint


def plan():
    value = {
        "protocol": PROTOCOL,
        "limits": LIMITS.copy(),
        "preflight_s": 1,
        "cases": [
            {"arm": a, "dimension": d, "resample_seed": s}
            for a, d, s in [("evidence", d, s) for d, s in CASES] + [("output", 30, None)]
        ],
    }
    value["sha256"] = fingerprint(value)
    return value


@pytest.mark.parametrize("mutation", ["limits", "matrix", "digest", "budget"])
def test_modified_plan_rejected(mutation):
    p = plan()
    validate_plan(p)
    if mutation == "limits":
        p["limits"]["timeout_s"] = 121
    elif mutation == "matrix":
        p["cases"][-1]["dimension"] = 60
    elif mutation == "budget":
        p["preflight_s"] = 901
    else:
        p["sha256"] = "wrong"
    if mutation != "digest":
        p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_plan(p)


def test_original_integrity_is_enforced(tmp_path):
    p = tmp_path / "original.hdf5"
    p.write_bytes(b"retained evidence")
    case = {
        "id": "a",
        "checkpoint": str(p),
        "result": str(p),
        "checkpoint_sha256": sha256(p),
        "result_sha256": sha256(p),
    }
    check_inputs({"cases": [case]})
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        check_inputs({"cases": [case]})


def test_source_producer_may_differ_but_runtime_pins_must_match():
    expected = {"sources": {"producer": "old", "PyAutoFit": "pin"}, "packages": {"numpy": "x"}}
    actual = copy.deepcopy(expected)
    actual["sources"]["producer"] = "new"
    assert_environment(runtime_identity(actual), runtime_identity(expected))
    actual["sources"]["PyAutoFit"] = "changed"
    with pytest.raises(ValueError, match="environment changed"):
        assert_environment(runtime_identity(actual), runtime_identity(expected))


def test_discard_must_recalculate_selected_shell_counts():
    # Models the critical native API: assigning the property updates shell_n;
    # passing a flag to run on an already explored sampler does not.
    class Sampler:
        explored = True
        points = [[0, 1, 2], [0, 1]]
        shell_end_exp = [2, 1]
        shell_n = [3, 2]

        @property
        def discard_exploration(self):
            return self.discard

        @discard_exploration.setter
        def discard_exploration(self, value):
            self.discard = value
            self.shell_n = fresh_counts(self)

    sampler = Sampler()
    discard_saved_exploration(sampler)
    assert sampler.shell_n == [1, 1]
    bad = SimpleNamespace(explored=True, points=[[0, 1]], shell_end_exp=[2], shell_n=[2])
    with pytest.raises(ValueError, match="exclusion"):
        discard_saved_exploration(bad)
    sampler.explored = False
    with pytest.raises(ValueError, match="not finished"):
        discard_saved_exploration(sampler)


def test_durable_stages_preserve_unfinished_and_errors(tmp_path):
    p = tmp_path / "stages.jsonl"
    log = StageLog(p)
    with log.stage("summary"):
        pass
    with pytest.raises(ValueError), log.stage("failure"):
        raise ValueError("test")
    log.write(id=2, stage="plot", event="enter")
    with p.open("a") as f:
        f.write('{"partial":')
    result = stage_summary(p)
    assert [r["status"] for r in result["completed"]] == ["exit", "error"]
    assert [r["stage"] for r in result["incomplete"]] == ["plot"]


def test_patches_restore_after_exception(tmp_path):
    class Example:
        def call(self):
            raise ValueError("expected")

    original = Example.call
    log = StageLog(tmp_path / "stages.jsonl")
    with pytest.raises(ValueError), log.patch(Example, "call"):
        Example().call()
    assert Example.call is original
    assert stage_summary(log.path)["completed"][0]["status"] == "error"


def test_timeout_cannot_promote_a_child_success_or_run_next_case(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/diagnose.py"
    spec = importlib.util.spec_from_file_location("diagnostic_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = plan()
    for i, case in enumerate(p["cases"]):
        case["id"] = str(i)
    p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})
    monkeypatch.setattr(module, "host_snapshot", lambda: {"load": [0], "affinity": [0]})
    monkeypatch.setattr(module, "memory_available", lambda: 10 * 1024**3)
    monkeypatch.setattr(module, "check_inputs", lambda p: None)
    calls = []

    def launch(command, **kwargs):
        directory = Path(command[-1])
        (directory / "child.json").write_text('{"status":"success"}')
        calls.append(command)
        return object()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(
        module, "supervise", lambda *a: {"returncode": 124, "termination_reason": "timeout"}
    )
    output = tmp_path / "pilot"
    assert module.run(p, output) == 1
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "halted"
    assert state["attempts"][0]["status"] == "timeout"
    assert len(calls) == 1 and len(state["unrun"]) == 5
    with pytest.raises(FileExistsError):
        module.run(p, output)  # No restart silently resets the budget.


@pytest.mark.parametrize("route", ["native", "wrapper"])
def test_output_replay_rejects_even_a_swallowed_likelihood_attempt(tmp_path, monkeypatch, route):
    import sys
    from types import ModuleType

    import profiling.nautilus_baseline as baseline
    from profiling.nautilus_diagnostics import output_replay

    def owner(names):
        return type("Fake", (), {n: lambda *a, **k: None for n in names})

    classes = {
        "autofit.non_linear.paths.directory": (
            "DirectoryPaths",
            owner(["save_samples", "save_samples_summary"]),
        ),
        "autofit.non_linear.samples.pdf": (
            "SamplesPDF",
            owner(["summary", "samples_drawn_randomly_via_pdf_from"]),
        ),
        "autofit.non_linear.search.abstract_search": (
            "NonLinearSearch",
            owner(["samples_from", "pre_fit_output", "post_fit_output"]),
        ),
        "autofit.non_linear.search.updater": (
            "SearchUpdater",
            owner(
                ["_save_samples", "_compute_latent_samples", "visualize", "_profile_and_summarize"]
            ),
        ),
    }

    class FakeSampler:
        def __init__(self, **kwargs):
            self.n_like = 34100

        def evaluate_likelihood(self):
            raise AssertionError("original evaluation must never run")

        def run(self):
            pass

    classes["nautilus"] = ("Sampler", FakeSampler)
    for name, (attr, cls) in classes.items():
        module = ModuleType(name)
        setattr(module, attr, cls)
        monkeypatch.setitem(sys.modules, name, module)
    original = FakeSampler.evaluate_likelihood

    def replay(*args):
        sampler = FakeSampler()
        with pytest.raises(RuntimeError, match="attempted likelihood"):
            if route == "native":
                sampler.evaluate_likelihood()
            else:
                args[1]([0.0] * 30)

    monkeypatch.setattr(baseline, "run_wrapped", replay)
    with pytest.raises(ValueError, match="invalid"):
        output_replay(
            {"original_n_like": 34100},
            tmp_path / "copy.hdf5",
            None,
            tmp_path,
            StageLog(tmp_path / "stages.jsonl"),
        )
    assert FakeSampler.evaluate_likelihood is original


def test_report_keeps_timeout_and_unrun_cases_separate(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/diagnose_report.py"
    spec = importlib.util.spec_from_file_location("diagnostic_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = plan()
    p["interpretation"] = "conditional repetitions"
    for i, case in enumerate(p["cases"]):
        case.update(
            id=str(i),
            result=f"archive/{i}/result.json",
            checkpoint=f"archive/{i}/checkpoint.hdf5",
            original_logZ_error=-0.9,
        )
    p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})
    root = tmp_path / "input"
    (root / "0").mkdir(parents=True)
    row = {
        "case": "0",
        "status": "timeout",
        "child": {"status": "success", "fresh_only": {"logZ_error": 0}},
        "stages": {"completed": [], "incomplete": []},
    }
    (root / "0/result.json").write_text(json.dumps(row))
    (root / "plan.json").write_text(json.dumps(p))
    (root / "state.json").write_text(
        json.dumps({"status": "halted", "spent_s": 120, "attempts": [row]})
    )
    monkeypatch.setattr(module, "check_inputs", lambda p: None)
    out = tmp_path / "report"
    module.export(root, out)
    rows = json.loads((out / "summary.json").read_text())["coverage"]
    assert rows[0]["fresh_logZ_error"] is None
    assert rows[0]["eligible_as_success"] is False
    assert [r["status"] for r in rows] == ["timeout"] + ["unrun"] * 5
    assert "partial-checkpoint.json" not in (out / "report.html").read_text()


def test_output_only_plan_cannot_admit_evidence_or_larger_budget():
    from profiling.nautilus_diagnostics import OUTPUT_LIMITS

    p = plan()
    p.update(scope="output-only", limits=OUTPUT_LIMITS.copy(), cases=[p["cases"][-1]])

    def seal():
        p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})

    seal()
    validate_plan(p)
    p["cases"][0]["arm"] = "evidence"
    seal()
    with pytest.raises(ValueError, match="matrix"):
        validate_plan(p)
    p["cases"][0]["arm"] = "output"
    p["limits"]["budget_s"] = 900
    seal()
    with pytest.raises(ValueError, match="limits"):
        validate_plan(p)
