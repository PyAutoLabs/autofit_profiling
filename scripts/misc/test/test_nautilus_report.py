"""An excluded successful measurement must survive in evidence, not aggregates."""

import json
import subprocess
import sys
from pathlib import Path


def test_superseded_run_retained_but_not_aggregated(tmp_path):
    pilot = tmp_path / "pilot"
    (pilot / "old").mkdir(parents=True)
    case = {
        "id": "old",
        "status": "success",
        "included_in_analysis": False,
        "workers": 2,
        "trace": "on",
        "cost_profile": "correlated",
        "seed": 0,
    }
    (pilot / "pilot.json").write_text(json.dumps({"cases": [case]}))
    (pilot / "old" / "result.json").write_text(
        json.dumps({"status": "success", "evidence": "old source"})
    )
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analytic_gaussian/nautilus/report.py"),
            str(pilot),
            "--export",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads((output / "summary.json").read_text())["groups"] == []
    assert (
        json.loads((output / "pilot_runs.json").read_text())["runs"][0]["result"]["evidence"]
        == "old source"
    )
