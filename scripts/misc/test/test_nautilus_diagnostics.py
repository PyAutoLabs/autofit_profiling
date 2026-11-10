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
