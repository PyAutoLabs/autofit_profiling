"""Export portable evidence, retaining caps and unrun cases; never publishes to Pulse."""

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts/misc"))
from profiling.nautilus_diagnostics import check_inputs, stage_summary, validate_plan
from profiling.nautilus_resources import atomic_json


def export(root, destination):
    plan = json.loads((root / "plan.json").read_text())
    validate_plan(plan)
    check_inputs(plan)
    state = json.loads((root / "state.json").read_text())
    if state["status"] not in ("complete", "halted"):
        raise ValueError("cannot export a running experiment")
    attempts = {a["case"]: a for a in state["attempts"]}
    if len(attempts) != len(state["attempts"]):
        raise ValueError("duplicate attempts")
    rows, cells = [], []
    for case in plan["cases"]:
        result = attempts.get(case["id"])
        if result is not None:
            path = root / case["id"] / "result.json"
            if not path.exists() or json.loads(path.read_text()) != result:
                raise ValueError("attempt receipt differs from state")
            if result["stages"] != stage_summary(root / case["id"] / "stages.jsonl"):
                raise ValueError("stage receipt differs from durable events")
        status = result["status"] if result else "unrun"
        if status == "running":
            status = "interrupted"
        child = result.get("child", {}) if result else {}
        if status == "success" and child.get("status") != "success":
            raise ValueError("success without child witness")
        before = case["original_logZ_error"]
        after = child.get("fresh_only", {}).get("logZ_error") if status == "success" else None
        row = {
            "case": case["id"],
            "status": status,
            "original_logZ_error": before,
            "fresh_logZ_error": after,
            "eligible_as_success": status == "success",
        }
        rows.append(row)
        cells.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    case["id"],
                    status,
                    f"{before:.6f}",
                    "—" if after is None else f"{after:.6f}",
                )
            )
            + "</tr>"
        )
    portable = {
        **plan,
        "original_plan_sha256": plan["sha256"],
        "note": "Paths removed for publication; original_plan_sha256 identifies the unmodified local execution plan.",
    }
    portable.pop("sha256")
    portable["cases"] = [
        {
            **{k: v for k, v in c.items() if k not in ("checkpoint", "result")},
            "archived_case": Path(c["result"]).parent.name,
        }
        for c in plan["cases"]
    ]
    partial = None
    partial_path = root / "partial-checkpoint.json"
    if partial_path.exists():
        from profiling.nautilus_diagnostics import sha256

        partial = json.loads(partial_path.read_text())
        if partial.get("eligible_as_success") is not False or partial.get("status") != "incomplete":
            raise ValueError("partial checkpoint must remain incomplete")
        if partial["checkpoint_sha256"] != sha256(root / "evidence-d30-rsaved/checkpoint.hdf5"):
            raise ValueError("partial checkpoint changed")
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "plan.json", portable)
    atomic_json(destination / "attempts.json", state)
    atomic_json(
        destination / "summary.json",
        {"coverage": rows, "spent_s": state["spent_s"], "interpretation": plan["interpretation"]},
    )
    for name in ("partial-checkpoint.json",):
        if (root / name).exists():
            shutil.copyfile(root / name, destination / name)
    for case in plan["cases"]:
        event_path = root / case["id"] / "stages.jsonl"
        if event_path.exists():
            shutil.copyfile(event_path, destination / (case["id"] + "-stages.jsonl"))
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Nautilus evidence diagnostics</title><style>
body{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#172333}
h1{font-size:30px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ccd4de}
a{color:#125db3}</style><h1>Nautilus evidence diagnostics</h1>
<p>Experimental phase 0b.2 · outside Pulse · fixed bounds from the original seed-0 exploration.</p>
<table><tr><th>Case</th><th>Status</th><th>Original logZ error</th><th>Completed fresh logZ error</th></tr>"""
    document += "".join(cells) + "</table>"
    if partial is not None:
        estimate = partial["estimate"]
        document += (
            f"<p><strong>Incomplete 30D checkpoint:</strong> fresh logZ error "
            f"{estimate['logZ_error']:+.6f}, ESS {estimate['n_eff']:.1f} of 10,000. "
            "This supports the exploration-reuse hypothesis, but is not a completed accuracy result. "
            "The wrapped output replay remains unrun.</p>"
        )
    document += f"<p>Charged runtime: {state['spent_s']:.2f} seconds including preflight. Stopped after the first 120-second cap; no retries or larger limits.</p>"
    document += """<p>The 30D last-checkpoint inspection is separate incomplete evidence, not a successful sample. See <a href="partial-checkpoint.json">its receipt</a> and the <a href="../README.md">findings</a> for the checkpoint-mode correction and limitations.</p>
<p><a href="attempts.json">All attempts</a> · <a href="plan.json">Portable plan</a> · <a href="summary.json">Coverage</a></p></html>"""
    if not partial_path.exists():
        document = document.replace(
            '<a href="partial-checkpoint.json">its receipt</a>', "the retained raw checkpoint"
        )
    if plan.get("scope") == "output-only":
        attempt = state["attempts"][0] if state["attempts"] else None
        result = attempt.get("child", {}) if attempt else {}
        stages = attempt.get("stages", {}).get("completed", []) if attempt else []
        document = (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Nautilus wrapped-output replay</title>"
            "<style>body{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 24px}"
            "td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>"
            "<h1>Nautilus wrapped-output replay</h1>"
            "<p>Experimental continuation outside Pulse. One copied original30D checkpoint; "
            "120s child cap,4GiB memory cap,180s aggregate including preflight.</p>"
            f"<p>Status: <strong>{html.escape(attempt['status'] if attempt else 'unrun')}</strong>. "
            f"Charged time: {state['spent_s']:.2f}s. "
            f"Added likelihood calls: {html.escape(str(result.get('new_likelihood_calls', 'unverified')))}.</p>"
            "<table><tr><th>Stage</th><th>Inclusive wall seconds</th><th>Exit</th></tr>"
        )
        for stage in stages:
            document += (
                f"<tr><td>{html.escape(stage['stage'])}</td><td>{stage['wall_s']:.4f}</td>"
                f"<td>{html.escape(stage['status'])}</td></tr>"
            )
        unfinished = attempt.get("stages", {}).get("incomplete", []) if attempt else []
        for stage in unfinished:
            document += (
                f"<tr><td>{html.escape(stage['stage'])}</td><td>—</td>"
                "<td>unfinished at termination</td></tr>"
            )
        document += (
            "</table><p>Nested stage times overlap; do not sum them. "
            "This replay excludes the original search and does not replace its timeout receipt.</p>"
            "<p><a href='attempts.json'>Complete receipts</a> · <a href='plan.json'>Portable plan</a> · "
            "<a href='../README.md'>Findings and interpretation</a></p></html>"
        )
    if plan.get("scope") == "plot-comparison":
        from profiling.nautilus_plot_comparison import export_plots

        document = export_plots(plan, state, root, destination)
    (destination / "report.html").write_text(document)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.output)
