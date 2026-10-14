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
