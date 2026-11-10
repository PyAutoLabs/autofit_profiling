"""Experimental identity, interrupted accounting, guarded subprocesses and report gates."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from profiling.nautilus_campaign import (
    environment_key,
    make_plan,
    new_state,
    next_case,
    process_token,
    recover,
    row_identity,
    validate_plan,
    verify_result,
)
from profiling.nautilus_experiment_report import summarize_rows
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_resources import atomic_json, group_rss, supervise


@pytest.fixture
def row():
    path = (
        Path(__file__).resolve().parents[3] / "results/analytic_gaussian/nautilus/pilot_runs.json"
    )
    records = json.loads(path.read_text())["runs"]
    return copy.deepcopy(
        next(
            r["result"]
            for r in records
            if r["case"].get("included_in_analysis", True)
            and r["result"].get("status") == "success"
        )
    )


@pytest.fixture
def identity(row):
    return {
        "environment": environment_key(row["manifest"]["environment"]),
        "truth_digests": {str(d): f"truth-{d}" for d in (5, 10, 30)},
        "preflight_s": 2.0,
    }


def test_plan_complete_deterministic_and_dimension_unique(identity):
    plan = make_plan(identity)
    assert plan == make_plan(identity)
    assert len({c["id"] for c in plan["cases"]}) == 36
    assert {c["config"]["dimension"] for c in plan["cases"]} == {5, 10, 30}
    assert {c["config"]["n_live"] for c in plan["cases"]} == {400}
    plan["cases"][0]["config"]["dimension"] = 100
    with pytest.raises(ValueError, match="changed"):
        validate_plan(plan)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": (0, 0)},
        {"dimensions": (10, 5)},
        {"budget_s": 6000},
        {"timeout_s": float("nan")},
        {"memory_mib": 4097},
        {"timeout_s": 0},
    ],
)
def test_invalid_plan_refused(identity, kwargs):
    with pytest.raises(ValueError):
        make_plan(identity, **kwargs)


def test_resume_retains_interruption_and_conservatively_charges_budget(identity, tmp_path):
    plan = make_plan(identity)
    state = new_state(plan)
    case = next_case(state, plan)
    state["attempts"].append({"id": "old", "case_id": case["id"], "status": "running"})
    recover(state, plan, tmp_path)
    assert state["attempts"][0]["status"] == "interrupted"
    assert state["spent_s"] == 122
    assert next_case(state, plan) == case
    recover(state, plan, tmp_path)
    assert state["spent_s"] == 122  # never double-charge on a second resume
    state["spent_s"] = 5300
    assert next_case(state, plan) is None
    assert "budget" in state["halt_reason"]


def test_resume_refuses_live_supervisor_and_modified_plan(identity, tmp_path):
    plan = make_plan(identity)
    state = new_state(plan)
    state["attempts"] = [
        {
            "id": "a",
            "status": "running",
            "pid": os.getpid(),
            "process_token": process_token(os.getpid()),
        }
    ]
    with pytest.raises(RuntimeError, match="still alive"):
        recover(state, plan, tmp_path)
    state["plan_sha256"] = "changed"
    with pytest.raises(ValueError, match="differs"):
        recover(state, plan, tmp_path)


@pytest.mark.parametrize(
    "status", ["timeout", "memory_cap", "error", "process_failure", "missing", "interrupted"]
)
def test_failed_recovered_child_never_retried(identity, tmp_path, status):
    plan = make_plan(identity)
    state = new_state(plan)
    state["attempts"] = [{"id": "old", "status": "running"}]
    (tmp_path / "old").mkdir()
    atomic_json(tmp_path / "old/result.json", {"status": status})
    recover(state, plan, tmp_path)
    assert next_case(state, plan) is None
    assert status in state["halt_reason"]


def test_dimension_settings_target_and_host_are_identity(row):
    original = row_identity(row)
    for section, key, value in [
        ("config", "dimension", 30),
        ("config", "n_live", 400),
        ("environment", "host", "another"),
    ]:
        other = copy.deepcopy(row)
        other["manifest"][section][key] = value
        assert row_identity(other) != original
    other = copy.deepcopy(row)
    other["manifest"]["target_sha256"] = "another target"
    assert row_identity(other) != original
    other = copy.deepcopy(row)
    for source in other["manifest"]["environment"]["sources"].values():
        source["root"] = "/relocated"
    assert row_identity(other) == original


def report_row(row, dimension=5, trace=False):
    row = copy.deepcopy(row)
    row["manifest"]["config"].update(dimension=dimension, trace=trace)
    row["supervision"] = {"sampled_group_peak_rss_bytes": 10}
    return row


def test_multi_dimension_grouping_pairs_and_duplicates(row):
    rows = [report_row(row, d, t) for d in (5, 10) for t in (False, True)]
    result = summarize_rows(rows)
    assert len(result["groups"]) == 4
    assert len(result["trace_control_pairs"]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        summarize_rows(rows + rows[:1])


def test_incomplete_pair_has_no_overhead_and_mismatch_rejected(row):
    on, off = report_row(row, trace=True), report_row(row)
    result = summarize_rows([on])
    assert result["trace_control_pairs"] == []
    assert len(result["incomplete_pairs"]) == 1
    on["n_like"] += 1
    with pytest.raises(ValueError, match="n_like"):
        summarize_rows([on, off])


def test_result_validation_checks_truth_and_missing_trace(row):
    row = report_row(row, trace=True)
    case = {
        "config": row["manifest"]["config"],
        "truth_sha256": row["manifest"]["target_sha256"],
        "environment": environment_key(row["manifest"]["environment"]),
    }
    row["trace"] = None
    with pytest.raises(ValueError, match="trace-incomplete"):
        verify_result(case, row)
    row["manifest"]["target"]["mean"][0] += 1
    with pytest.raises(ValueError, match="truth digest"):
        verify_result(case, row)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process group guard")
@pytest.mark.parametrize("cap, timeout, expected", [(None, 0.1, "timeout"), (1, 5, "memory_cap")])
def test_supervisor_terminates_actual_process_group(tmp_path, cap, timeout, expected):
    child_pid = tmp_path / "child"
    code = (
        "import subprocess,time,pathlib; p=subprocess.Popen(['sleep','30']); pathlib.Path("
        + repr(str(child_pid))
        + ").write_text(str(p.pid)); time.sleep(30)"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    result = supervise(process, timeout, cap)
    assert result["termination_reason"] == expected
    assert process.poll() is not None
    if child_pid.exists():
        path = Path("/proc") / child_pid.read_text() / "stat"
        if path.exists():
            assert path.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_rss_group_reader_handles_spaces_in_process_name(tmp_path):
    (tmp_path / "100").mkdir()
    fields = ["S", "1", "77"] + ["0"] * 18 + ["42"]
    (tmp_path / "100/stat").write_text("100 (name with ) spaces) " + " ".join(fields))
    assert group_rss(77, tmp_path) == 42 * os.sysconf("SC_PAGE_SIZE")
    assert group_rss(78, tmp_path) == 0


def test_plan_fingerprint_ignores_json_formatting(identity, tmp_path):
    plan = make_plan(identity)
    atomic_json(tmp_path / "plan.json", plan)
    reloaded = json.loads((tmp_path / "plan.json").read_text())
    assert fingerprint(reloaded) == fingerprint(plan)
    validate_plan(reloaded)


def test_resealed_incomplete_matrix_is_still_rejected(identity):
    plan = make_plan(identity)
    plan["cases"].pop()
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="matrix"):
        validate_plan(plan)


def test_plan_command_needs_no_scientific_site_packages(identity, tmp_path):
    path = tmp_path / "identity.json"
    atomic_json(path, identity)
    script = Path(__file__).resolve().parents[3] / "scripts/analytic_gaussian/nautilus/campaign.py"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            "plan",
            "--identity",
            str(path),
            "--output",
            str(tmp_path / "plan.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads((tmp_path / "plan.json").read_text())["cases"]) == 36


def test_empty_experiment_reports_all_missing_pairs(identity, tmp_path):
    plan = make_plan(identity)
    root = tmp_path / "run"
    root.mkdir()
    atomic_json(root / "plan.json", plan)
    state = new_state(plan)
    state["halt_reason"] = "host not quiet"
    atomic_json(root / "state.json", state)
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/analytic_gaussian/nautilus/experiment_report.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(script), str(root), "--output", str(tmp_path / "report")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "report/summary.json").read_text())
    assert summary["counts"] == {"unrun": 36}
    assert summary["expected_pairs"] == 18
    assert len(summary["incomplete_pairs"]) == 18
    assert summary["trace_control_pairs"] == []
