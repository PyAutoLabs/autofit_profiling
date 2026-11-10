"""Plot-scope admission, restoration, and output witnesses; no scientific imports."""

import copy
import struct

import pytest
from profiling.nautilus_diagnostics import PROTOCOL, validate_plan
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_plot_comparison import LIMITS, POLICIES, replaced, verify_plot


def comparison_plan():
    p = {
        "protocol": PROTOCOL,
        "scope": "plot-comparison",
        "limits": LIMITS.copy(),
        "preflight_s": 1,
        "cases": [
            {
                "arm": "output",
                "dimension": 30,
                "resample_seed": None,
                "plot_policy": copy.deepcopy(policy),
                "summary_timing_call_limit": 1,
            }
            for policy in POLICIES
        ],
    }
    p["sha256"] = fingerprint(p)
    return p


@pytest.mark.parametrize(
    "change", ["indices", "dimension", "arm", "order", "budget", "size", "dpi"]
)
def test_scope_cannot_change_approved_comparison(change):
    p = comparison_plan()
    validate_plan(p)
    if change == "indices":
        p["cases"][1]["plot_policy"]["indices"] = [0, 1]
    elif change == "dimension":
        p["cases"][0]["dimension"] = 60
    elif change == "arm":
        p["cases"][0]["arm"] = "evidence"
    elif change == "order":
        p["cases"].reverse()
    elif change == "budget":
        p["limits"]["budget_s"] = 421
    elif change == "size":
        p["cases"][1]["plot_policy"]["figsize"] = [120, 120]
    else:
        p["cases"][1]["plot_policy"]["dpi"] = 300
    p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_plan(p)


def test_inherited_plot_method_restored_on_error():
    class Parent:
        def plot(self):
            return "original"

    class Child(Parent):
        pass

    with pytest.raises(RuntimeError), replaced(Child, "plot", lambda self: "replacement"):
        assert Child().plot() == "replacement"
        raise RuntimeError("exit")
    assert "plot" not in vars(Child)
    assert Child().plot() == "original"


def receipt():
    return {"errors": [], "search_plot_calls": 1, "input_samples": 33984, "figures": []}


def test_swallowed_plot_error_is_failure_even_if_png_exists(tmp_path):
    r = receipt()
    r["errors"] = ["Unable to produce corner_anesthetic visual"]
    with pytest.raises(ValueError, match="swallowed"):
        verify_plot(POLICIES[1], r, tmp_path, 33984)


def test_no_corner_prohibits_any_image(tmp_path):
    r = receipt()
    verify_plot(POLICIES[0], r, tmp_path, 33984)
    (tmp_path / "unexpected.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="unexpectedly"):
        verify_plot(POLICIES[0], r, tmp_path, 33984)


def test_required_image_and_geometry_are_verified(tmp_path):
    r = receipt()
    with pytest.raises(ValueError, match="missing"):
        verify_plot(POLICIES[1], r, tmp_path, 33984)
    r["figures"] = [{"figsize": [12.0, 12.0], "dpi": 100.0, "axes": 27}]
    p = tmp_path / "corner.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1200, 1200))
    verify_plot(POLICIES[1], r, tmp_path, 33984)
    assert r["png"]["dimensions"] == [1200, 1200]
    r["figures"][0]["axes"] = 36
    with pytest.raises(ValueError, match="geometry"):
        verify_plot(POLICIES[1], r, tmp_path, 33984)


def test_plot_report_preserves_incomplete_stages_and_verifies_png(tmp_path):
    import hashlib
    import json

    from profiling.nautilus_plot_comparison import export_plots

    plan = {"cases": [{"id": p["name"], "plot_policy": p} for p in POLICIES]}
    state = {
        "spent_s": 150,
        "attempts": [
            {"case": "no-corner", "status": "success", "process_wall_s": 30},
            {"case": "six-small", "status": "success", "process_wall_s": 40},
            {
                "case": "thirty-small",
                "status": "timeout",
                "process_wall_s": 120,
                "stages": {"completed": [], "incomplete": [{"stage": "axes_construction"}]},
            },
        ],
    }
    root = tmp_path / "raw"
    (root / "six-small").mkdir(parents=True)
    png = root / "six-small/figure.png"
    png.write_bytes(b"PNG fixture")
    receipt = {
        "policy": POLICIES[1],
        "png": {"path": "figure.png", "sha256": hashlib.sha256(png.read_bytes()).hexdigest()},
    }
    (root / "six-small/plot-receipt.json").write_text(json.dumps(receipt))
    dest = tmp_path / "report"
    dest.mkdir()
    document = export_plots(plan, state, root, dest)
    assert "unfinished stages axes_construction" in document
    assert "timeout" in document
    assert (dest / "six-small.png").read_bytes() == png.read_bytes()
    png.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="PNG changed"):
        export_plots(plan, state, root, dest)


@pytest.mark.parametrize("limit", [0, 2])
def test_plan_rejects_changed_summary_allowance(limit):
    p = comparison_plan()
    p["cases"][0]["summary_timing_call_limit"] = limit
    p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError, match="summary timing"):
        validate_plan(p)


def test_summary_gate_counts_and_rejects_out_of_scope_or_extra_calls():
    from profiling.nautilus_plot_comparison import SummaryTimingGate

    calls, rejected = [], []

    def target(parameters):
        calls.append(parameters)
        return 3.5

    def forbidden(parameters):
        rejected.append(parameters)
        raise RuntimeError("forbidden")

    gate = SummaryTimingGate(target, forbidden, 1)
    with pytest.raises(RuntimeError, match="forbidden"):
        gate.evaluate([0])
    with gate.summary():
        assert gate.evaluate([1]) == 3.5
        with pytest.raises(RuntimeError, match="forbidden"):
            gate.evaluate([2])
    with pytest.raises(RuntimeError, match="forbidden"):
        gate.evaluate([3])
    assert calls == [[1]]
    assert rejected == [[0], [2], [3]]
    assert (gate.calls, gate.parameters, gate.value, gate.active) == (1, [1], 3.5, False)


def test_original_zero_call_gate_and_exception_cleanup():
    from profiling.nautilus_plot_comparison import SummaryTimingGate

    def forbidden(parameters):
        raise RuntimeError("forbidden")

    gate = SummaryTimingGate(lambda p: pytest.fail("target called"), forbidden)
    with pytest.raises(RuntimeError, match="forbidden"), gate.summary():
        gate.evaluate([1])
    assert not gate.active
    assert gate.calls == 0
