"""Export all attempt receipts and an off-board HTML report, never a Pulse catalogue."""

import argparse
import gzip
import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from profiling.nautilus_campaign import validate_plan, verify_result
from profiling.nautilus_experiment_report import summarize_rows
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_resources import atomic_json
from report import portable, timeline_svg


def export(root, output):
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "state.json").read_text())
    validate_plan(plan)
    if state["plan_sha256"] != plan["sha256"]:
        raise ValueError("state/plan mismatch")
    cases = {c["id"]: c for c in plan["cases"]}
    attempts = {a["id"]: a for a in state["attempts"]}
    if len(attempts) != len(state["attempts"]):
        raise ValueError("duplicate attempt identity")
    if not set(state["selected"]).issubset(cases):
        raise ValueError("selected case outside plan")
    selected, records, coverage = [], [], []
    for case_id, case in cases.items():
        attempt_id = state["selected"].get(case_id)
        status = "unrun"
        if attempt_id:
            attempt = attempts[attempt_id]
            if attempt["case_id"] != case_id or attempt["status"] != "success":
                raise ValueError("invalid selected attempt")
            row = json.loads((root / attempt_id / "result.json").read_text())
            if row["status"] != "success":
                raise ValueError("selected result is not successful")
            verify_result(case, row, root / attempt_id)
            selected.append(row)
            status = "success"
        else:
            previous = [a for a in state["attempts"] if a["case_id"] == case_id]
            if previous:
                status = previous[-1]["status"]
        coverage.append(
            {
                "case_id": case_id,
                "config": case["config"],
                "status": status,
                "selected_attempt": attempt_id,
                "unrun_reason": state["halt_reason"] if status == "unrun" else None,
            }
        )
    for attempt in state["attempts"]:
        path = root / attempt["id"] / "result.json"
        row = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        records.append({"attempt": portable(attempt), "result": portable(row)})
    summary = summarize_rows(selected)
    expected_pairs = set()
    for case in plan["cases"]:
        config = dict(case["config"])
        config.pop("trace")
        expected_pairs.add(
            fingerprint(
                {
                    "config": config,
                    "target": case["truth_sha256"],
                    "environment": case["environment"],
                }
            )
        )
    represented = {
        p["pair_id"] for p in summary["trace_control_pairs"] + summary["incomplete_pairs"]
    }
    for pair_id in sorted(expected_pairs - represented):
        summary["incomplete_pairs"].append(
            {"pair_id": pair_id, "present_trace_modes": [], "status": "no successful members"}
        )
    summary["expected_pairs"] = len(expected_pairs)
    summary.update(
        coverage=coverage,
        counts=dict(Counter(c["status"] for c in coverage)),
        attempts=len(records),
        halt_reason=state["halt_reason"],
        spent_s=state["spent_s"],
    )
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", portable(plan))
    atomic_json(output / "attempts.json", {"state": portable(state), "records": records})
    atomic_json(output / "summary.json", summary)
    content = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Nautilus dimension experiment</title>',
        "<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:36px auto;padding:20px;color:#152b40}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}svg{width:100%}h1{font-size:30px}</style>",
        "<h1>Nautilus dimension experiment</h1><p>Exploratory local evidence · outside Pulse · phase 0b remains incomplete.</p>",
        f"<p><b>{len(selected)} / {len(cases)} cases selected</b> · {len(summary['trace_control_pairs'])} matched tracing pairs · {len(summary['incomplete_pairs'])} incomplete pairs.</p>",
        f"<p>Stop reason: {html.escape(str(state['halt_reason'] or 'completed'))}. Budget used: {state['spent_s']:.1f} s.</p>",
        "<h2>Complete adapter wall</h2><p>Median [minimum–maximum]; three seeds are engineering evidence, not a confidence interval.</p>",
        "<table><tr><th>Dimensions</th><th>Adapter</th><th>Trace</th><th>Seeds</th><th>Wall (s)</th><th>logZ error</th></tr>",
    ]
    for group in sorted(
        summary["groups"],
        key=lambda g: (g["config"]["dimension"], g["config"]["adapter"], g["config"]["trace"]),
    ):
        c, m = group["config"], group["metrics"]
        w, z = m["wall_s"], m["logZ_error"]
        content.append(
            f"<tr><td>{c['dimension']}</td><td>{c['adapter']}</td><td>{c['trace']}</td><td>{group['n']}</td><td>{w['median']:.2f} [{w['min']:.2f}–{w['max']:.2f}]</td><td>[{z['min']:.3f}, {z['max']:.3f}]</td></tr>"
        )
    content.append(
        "</table><h2>Coverage</h2><p>Every planned case is retained. Capped timings are censored; no automatic retry, resource increase or extrapolated higher-dimensional success.</p><table><tr><th>D</th><th>Adapter</th><th>Seed</th><th>Trace</th><th>Status</th></tr>"
    )
    for item in coverage:
        c = item["config"]
        content.append(
            f"<tr><td>{c['dimension']}</td><td>{c['adapter']}</td><td>{c['seed']}</td><td>{c['trace']}</td><td>{item['status']}</td></tr>"
        )
    content.append(
        "</table><p>See summary.json for standardized posterior/covariance errors, ESS, calls/ESS, memory and stage shares. Unknown logZ uncertainty and proposal-rejection diagnostics remain unavailable. No catalogue or scientific acceptance is implied.</p>"
    )
    # One compact example timeline, all selected traces were verified above.
    for case_id, attempt_id in state["selected"].items():
        if cases[case_id]["config"]["trace"]:
            path = root / attempt_id / "timeline.json.gz"
            shutil.copyfile(path, output / "example-timeline.json.gz")
            content.append("<h2>Example: first eight batches</h2>")
            content.append(timeline_svg(json.loads(gzip.decompress(path.read_bytes()))))
            break
    content.append("</html>")
    (output / "report.html").write_text("".join(content))
    print(json.dumps({"counts": summary["counts"], "pairs": len(summary["trace_control_pairs"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)
