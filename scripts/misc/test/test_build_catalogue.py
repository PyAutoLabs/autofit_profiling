import copy
import json
import shutil
from pathlib import Path

import build_catalogue as bc
import pytest

REPO = Path(__file__).resolve().parents[3]
# CI checks Pulse out at ./_pulse; a workspace has it beside this repo.
PULSE = next(
    (p for p in (REPO / "_pulse", REPO.parent / "PyAutoPulse") if (p / "pulse").is_dir()),
    REPO / "_pulse",
)


def _row(variant="search_fit", **components):
    components = components or {"nn_training": 6.0, "likelihood": 1.0, "other": 3.0}
    return {
        "schema": bc.RESULT_SCHEMA,
        "version": 1,
        "variant": variant,
        "axis": "breakdown",
        "unit": "s",
        "setup": {
            "id": f"gaussian_x1/nautilus/{variant}",
            "label": variant,
            "dataset": "gaussian_x1",
            "search": "Nautilus",
            "search_settings": {"n_live": 200, "n_networks": 4, "seed": 1, "number_of_cores": 1},
            "likelihood": "numpy, no JAX",
            "config": "config/general.yaml",
        },
        "method": {
            "id": f"{variant}-method",
            "statistic": "median",
            "repetitions": 3,
            "warmup": "none",
            "synchronization": "none",
            "cache_state": "warm process",
        },
        "wall_s": 10.0,
        "components_s": components,
        "shares_of_wall": {k: v / 10.0 for k, v in components.items()},
        "likelihood_function_calls": 100,
        "reference": {
            "shares": {"nn_training": 0.5},
            "deltas": {"nn_training": 0.1},
            "compared": "shares_of_wall",
            "tolerance_abs": 0.1,
            "within_tolerance": True,
            "gates_reproduction": variant == "bare_sampler",
        },
        "identity": {
            "device": "cpu",
            "backend": "numpy",
            "precision": "float64",
            "library_version": "2026.10.12.1",
            "software": {"PyAutoFit": "0" * 40, "nautilus-sampler": "1.0.5"},
        },
        "provenance": {"host": "test-host", "measured_at": "2026-10-14T10:00:00Z"},
    }


def _run_row(schema, setup_id, sanity_pass=True):
    """A whole-run row (EP / graphical baseline or analytic parity)."""
    row = {
        "schema": schema,
        "version": 1,
        "setup": {
            "id": setup_id,
            "label": setup_id,
            "dataset": setup_id.split("/")[0],
            "n_datasets": 3,
            "free_parameters": 7,
            "data_pixels": 300,
            "search": "DynestyStatic",
            "search_settings": {"nlive": 100, "sample": "rwalk", "kl_tol": 0.05},
            "likelihood": "jax",
            "config": "cell config",
        },
        "method": {
            "id": "fresh-process-single-fit",
            "statistic": "single",
            "repetitions": 1,
            "warmup": "none",
            "synchronization": "none",
            "cache_state": "cold process",
        },
        "wall_s": 37.0,
        "peak_rss_kib": 361472,
        "correctness": {
            "sanity_pass": sanity_pass,
            "truth_log_likelihood_sum": -128.4,
            "parity": "PASS",
            "parity_cells_passed": 41,
            "parity_cells_total": 41,
            "sigma_q95": 12.1536,
            "sigma_q95_banked": 12.15,
            "sigma_q95_matches_banked": True,
            "legs": {},
        },
        "reference": {
            "wall_s": 251.9,
            "sanity_pass": sanity_pass,
            "truth_log_likelihood_sum_matches": True,
            "wall_drift": {"policy": "runtime-drift-2x-1ms", "ratio": 0.147, "status": "improved"},
        },
        "suite": {"parked": {"script": "p.py", "reason": "NEEDS_FIX"}},
        "identity": {
            "device": "cpu",
            "backend": "jax",
            "precision": "float64",
            "library_version": "2026.10.12.1",
            "software": {"PyAutoFit": "0" * 40, "dynesty": "2.1.5"},
        },
        "provenance": {"host": "test-host", "measured_at": "2026-10-14T11:00:00Z"},
    }
    return row


def test_baseline_row_exports_runtime_and_memory_records_only(root):
    _write(root, "n3.json", _run_row(bc.BASELINE_SCHEMA, "gaussian_1d_ep/ep/N3"))
    doc, _ = bc.build(root)
    got = {(r["axis"], r["metric"], r["unit"]) for r in doc["records"]}
    assert got == {("runtime", "wall", "s"), ("memory", "host_peak_rss", "KiB")}
    rss = next(r for r in doc["records"] if r["axis"] == "memory")
    assert rss["measurement"] == {"host_peak_rss": 361472}
    assert rss["evidence"]["fragment"] == "/peak_rss_kib"
    setup = doc["setups"][0]
    assert setup["model"] == "gaussian, centre shared across datasets"
    assert setup["configuration"]["n_datasets"] == {"value": 3, "unit": "count"}
    assert setup["configuration"]["sample"] == {"value": "rwalk", "unit": "name"}
    assert setup["configuration"]["kl_tol"] == {"value": 0.05, "unit": "dimensionless"}
    assert doc["comparisons"] == [] and len(doc["selections"]) == 2


def test_failed_sanity_is_exported_and_named_never_hidden(root):
    _write(root, "n30.json", _run_row(bc.BASELINE_SCHEMA, "gaussian_1d_ep/ep/N30", False))
    doc, rows = bc.build(root)
    assert len(doc["records"]) == 2
    assert all(
        "sanity FAIL (committed baseline FAIL)" in r["validation"]["reason"] for r in doc["records"]
    )
    assert all(r["validation"]["status"] == "unreviewed" for r in doc["records"])
    assert "FAIL (FAIL)" in bc.render_html(rows, doc)


def test_parity_row_reason_carries_the_parity_verdict(root):
    _write(root, "p.json", _run_row(bc.PARITY_SCHEMA, "analytic_gaussian/parity/seed0_n5"))
    doc, rows = bc.build(root)
    assert {r["axis"] for r in doc["records"]} == {"runtime", "memory"}
    assert all("parity PASS (41/41)" in r["validation"]["reason"] for r in doc["records"])
    assert "sigma_q95 12.1536" in bc.render_html(rows, doc)


def test_host_contention_is_named_in_the_record_reason(root):
    row = _run_row(bc.BASELINE_SCHEMA, "gaussian_1d_ep/ep/N30")
    row["provenance"]["host_contention"] = "load 4.7 on 8 cores"
    _write(root, "n30.json", row)
    doc, _ = bc.build(root)
    assert all("host contended" in r["validation"]["reason"] for r in doc["records"])


@pytest.mark.skipif(not (PULSE / "pulse" / "summary.py").is_file(), reason="no PyAutoPulse beside")
def test_mixed_schemas_validate_with_the_pulse_reader(root):
    _write(root, "a.json", _row())
    _write(root, "n3.json", _run_row(bc.BASELINE_SCHEMA, "gaussian_1d_ep/graphical/N3"))
    _write(root, "p.json", _run_row(bc.PARITY_SCHEMA, "analytic_gaussian/parity/seed0_n5"))
    assert bc.validate(bc.render(root), PULSE) == "ok"


@pytest.fixture
def root(tmp_path):
    (tmp_path / "results" / "x").mkdir(parents=True)
    return tmp_path


def _write(root, name, row):
    (root / "results" / "x" / name).write_text(json.dumps(row))


def test_one_record_per_component_plus_wall_and_a_selection_each(root):
    _write(root, "a.json", _row())
    doc, rows = bc.build(root)
    assert len(rows) == 1 and len(doc["setups"]) == 1
    assert sorted(r["metric"] for r in doc["records"]) == [
        "likelihood",
        "nn_training",
        "other",
        "wall",
    ]
    assert {r["axis"] for r in doc["records"] if r["metric"] == "wall"} == {"runtime"}
    assert {r["axis"] for r in doc["records"] if r["metric"] != "wall"} == {"breakdown"}
    assert doc["coverage"]["expected"]["cells"] == len(doc["selections"]) == 4
    assert all(r["validation"]["status"] == "unreviewed" for r in doc["records"])
    assert doc["comparisons"] == []


def test_zero_inapplicable_component_is_a_slot_not_a_zero_record(root):
    _write(root, "a.json", _row("bare_sampler", nn_training=5.0, checkpoint_io=0.0, other=5.0))
    doc, _ = bc.build(root)
    assert "checkpoint_io" not in {r["metric"] for r in doc["records"]}
    slot = next(s for s in doc["selections"] if s["metric"] == "checkpoint_io")
    assert slot["status"] == "inapplicable" and slot["record_id"] is None
    assert doc["coverage"]["observed"]["selected"] == len(doc["selections"]) - 1


def test_non_result_json_is_ignored_and_render_is_deterministic(root):
    _write(root, "a.json", _row())
    _write(root, "other.json", {"schema": "something-else"})
    first, second = bc.render(root), bc.render(root)
    assert first == second
    assert json.loads(first["catalogue.json"])["generated_at"] == "2026-10-14T10:00:00Z"


def test_check_mode_reports_stale_then_current(root, capsys):
    _write(root, "a.json", _row())
    assert bc.main(["--root", str(root), "--check"]) == 1
    assert bc.main(["--root", str(root)]) == 0
    assert bc.main(["--root", str(root), "--check"]) == 0
    row = _row()
    row["wall_s"] = 11.0
    _write(root, "a.json", row)
    assert bc.main(["--root", str(root), "--check"]) == 1


def test_committed_catalogue_is_current():
    assert bc.main(["--root", str(REPO), "--check"]) == 0


@pytest.mark.skipif(not (PULSE / "pulse" / "summary.py").is_file(), reason="no PyAutoPulse beside")
def test_committed_results_validate_with_the_pulse_reader(root):
    shutil.copytree(REPO / "results", root / "results", dirs_exist_ok=True)
    assert bc.validate(bc.render(root), PULSE) == "ok"


@pytest.mark.skipif(not (PULSE / "pulse" / "summary.py").is_file(), reason="no PyAutoPulse beside")
def test_pulse_reader_refuses_a_broken_catalogue(root):
    _write(root, "a.json", _row())
    outputs = bc.render(root)
    doc = json.loads(outputs["catalogue.json"])
    broken = copy.deepcopy(doc)
    broken["records"][0]["unit"] = "MiB"
    outputs["catalogue.json"] = json.dumps(broken)
    with pytest.raises(ValueError):
        bc.validate(outputs, PULSE)
