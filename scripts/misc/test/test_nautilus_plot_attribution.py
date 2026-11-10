import copy
import json

import pytest
from profiling.nautilus_diagnostics import PROTOCOL, validate_plan
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_plot_attribution import LIMITS, Timers, export_attribution
from profiling.nautilus_plot_comparison import POLICIES


def seal(plan):
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    return plan


def plan():
    return seal(
        {
            "protocol": PROTOCOL,
            "scope": "plot-attribution",
            "limits": LIMITS.copy(),
            "preflight_s": 5,
            "synthetic_preflight": {"status": "success"},
            "cases": [
                {
                    "arm": "output",
                    "dimension": 30,
                    "resample_seed": None,
                    "plot_policy": copy.deepcopy(POLICIES[-1]),
                    "summary_timing_call_limit": 1,
                    "attribution": True,
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "change",
    ["budget", "timeout", "memory", "policy", "cases", "seed", "hooks", "witness", "call_limit"],
)
def test_attribution_scope_rejects_expansion(change):
    p = plan()
    validate_plan(p)
    if change in ("budget", "timeout", "memory"):
        key = {"budget": "budget_s", "timeout": "timeout_s", "memory": "memory_mib"}[change]
        p["limits"][key] += 1
    elif change == "policy":
        p["cases"][0]["plot_policy"] = POLICIES[1]
    elif change == "cases":
        p["cases"] *= 2
    elif change == "seed":
        p["cases"][0]["resample_seed"] = 1
    elif change == "hooks":
        p["cases"][0]["attribution"] = False
    elif change == "witness":
        p["synthetic_preflight"]["status"] = "error"
    else:
        p["cases"][0]["summary_timing_call_limit"] = 2
    with pytest.raises(ValueError):
        validate_plan(seal(p))


def test_timers_separate_nested_time_and_snapshot_io(tmp_path, monkeypatch):
    import profiling.nautilus_plot_attribution as module

    now, snapshots = [0.0], []

    def write(path, receipt):
        snapshots.append(copy.deepcopy(receipt))
        now[0] += 1.0

    monkeypatch.setattr(module, "atomic_json", write)
    timers = Timers(tmp_path / "receipt.json", clock=lambda: now[0])
    with timers.measure("outer", durable=True):
        now[0] += 2
        with timers.measure("inner", durable=True):
            now[0] += 3
        now[0] += 4
    assert timers.totals["outer"]["inclusive_s"] == 9
    assert timers.totals["outer"]["self_s"] == 6
    assert timers.totals["outer/inner"]["inclusive_s"] == 3
    assert timers.logging_s == 4
    assert snapshots[1]["active"][-1]["name"] == "inner"
    assert snapshots[-1]["active"] == []


def test_failure_and_interrupted_snapshot_are_not_success(tmp_path):
    timers = Timers(tmp_path / "receipt.json")
    with pytest.raises(RuntimeError), timers.measure("plot_2d", durable=True):
        with timers.measure("panel_hist_2d", durable=True):
            pending = json.loads(timers.path.read_text())
            assert pending["active"][-1]["name"] == "panel_hist_2d"
            assert "plot_2d/panel_hist_2d" not in pending["completed"]
            raise RuntimeError("failed")
    assert timers.totals["plot_2d/panel_hist_2d"]["errors"] == 1
    with pytest.raises(ValueError, match="panel count"):
        timers.verify_panels(2)


def test_report_keeps_active_frames_and_completed_only_totals(tmp_path):
    root = tmp_path / "raw"
    (root / "case").mkdir(parents=True)
    dest = tmp_path / "report"
    dest.mkdir()
    data = {
        "active": [{"name": "savefig"}, {"name": "figure_draw"}],
        "logging_s": 0.5,
        "completed": {
            "plot_2d/panel_hist_2d": {"calls": 435, "inclusive_s": 70, "self_s": 10, "errors": 0}
        },
    }
    (root / "case/attribution.json").write_text(json.dumps(data))
    document = export_attribution(root, dest, "case")
    assert "savefig, figure_draw" in document
    assert "435" in document
    assert "Unfinished work is absent" in document
    assert json.loads((dest / "attribution.json").read_text()) == data
