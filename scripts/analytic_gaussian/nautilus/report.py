"""Summarise a pilot, verify trace/control pairs, export compact evidence and timeline."""

import argparse
import gzip
import hashlib
import html
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from profiling.nautilus_manifest import assert_matching


def portable(row):
    """Remove machine paths from committed evidence; exact raw manifest stays local."""
    if isinstance(row, dict):
        return {
            key: portable(value)
            for key, value in row.items()
            if key
            not in ("root", "executable", "output_filesystem_path", "command", "driver_output")
        }
    if isinstance(row, list):
        return [portable(value) for value in row]
    return row


def timeline_svg(events):
    selected = [e for e in events["events"] if e["id"][0] < 8]
    if not selected:
        return "<p>No events.</p>"
    pids = sorted({e["pid"] for e in selected})
    start = min(e["submit_ns"] for e in selected)
    end = max(e["collect_ns"] for e in selected)
    scale = 850 / max(1, end - start)
    out = [
        f'<svg viewBox="0 0 1000 {len(pids) * 45 + 40}" role="img" aria-label="Worker service intervals, first eight batches">'
    ]
    for i, pid in enumerate(pids):
        y = 20 + i * 45
        out.append(f'<text x="0" y="{y + 15}">worker {i + 1}</text>')
        for event in selected:
            if event["pid"] == pid:
                x = 130 + (event["start_ns"] - start) * scale
                width = max(0.2, (event["finish_ns"] - event["start_ns"]) * scale)
                out.append(
                    f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="20" fill="#2563a6"><title>batch/row {event["id"]}; {(event["finish_ns"] - event["start_ns"]) * 1e-6:.3f} ms</title></rect>'
                )
    out.append(
        f'<text x="130" y="{len(pids) * 45 + 30}">Elapsed {(end - start) * 1e-9:.4f} s; gaps include parent work between batches.</text></svg>'
    )
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    args.export.mkdir(parents=True, exist_ok=False)
    pilot = json.loads((args.pilot / "pilot.json").read_text())
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    records = []
    source_pins = set()
    for case in pilot["cases"]:
        path = args.pilot / case["id"] / "result.json"
        row = json.loads(path.read_text()) if path.exists() else {"status": case["status"]}
        records.append({"case": portable(case), "result": portable(row)})
        if row["status"] != "success":
            continue
        config = row["manifest"]["config"]
        key = (config["adapter"], config["workers"], config["cost_profile"])
        groups[(*key, config["trace"])].append(row)
        pairs[(*key, config["seed"])][config["trace"]] = row
        source_pins.add(json.dumps(row["manifest"]["environment"]["sources"], sort_keys=True))
    if len(source_pins) > 1:
        raise ValueError("source pins changed across pilot")
    overhead = []
    for key, pair in sorted(pairs.items()):
        if len(pair) != 2:
            continue
        on, off = pair[True], pair[False]
        assert_matching(on["manifest"]["matching"], off["manifest"]["matching"])
        if on["posterior_sha256"] != off["posterior_sha256"] or on["n_like"] != off["n_like"]:
            raise ValueError(f"tracing changed posterior or calls: {key}")
        overhead.append(
            {
                "adapter": key[0],
                "workers": key[1],
                "cost": key[2],
                "seed": key[3],
                "trace_wall_ratio": on["wall_s"] / off["wall_s"],
                "posterior_identical": True,
            }
        )
    (args.export / "pilot_runs.json").write_text(
        json.dumps(
            {
                "schema": "nautilus-cpu-pilot",
                "version": 1,
                "runs": records,
                "instrumentation_pairs": overhead,
            },
            indent=2,
        )
    )
    summary = []
    for key, rows in sorted(groups.items()):
        adapter, workers, cost, traced = key
        wall = statistics.median(r["wall_s"] for r in rows)
        components = {
            name: statistics.median(r["components_s"][name] for r in rows)
            for name in rows[0]["components_s"]
        }
        summary.append(
            {
                "adapter": adapter,
                "workers": workers,
                "cost": cost,
                "traced": traced,
                "n": len(rows),
                "wall_s": wall,
                "wall_range_s": [min(r["wall_s"] for r in rows), max(r["wall_s"] for r in rows)],
                "n_like_range": [min(r["n_like"] for r in rows), max(r["n_like"] for r in rows)],
                "logZ_error_range": [
                    min(r["log_evidence_error"] for r in rows),
                    max(r["log_evidence_error"] for r in rows),
                ],
                "components_s": components,
            }
        )
        variant = f"{adapter}-w{workers}-{cost}-{'trace' if traced else 'control'}"
        manifest = rows[0]["manifest"]
        env = manifest["environment"]
        settings = {k: v for k, v in manifest["config"].items() if isinstance(v, (str, int, float))}
        doc = {
            "schema": "autofit-profiling-breakdown",
            "version": 1,
            "status": "success",
            "variant": variant,
            "axis": "breakdown",
            "unit": "s",
            "setup": {
                "id": f"analytic_gaussian/nautilus/{variant}",
                "label": variant,
                "dataset": "analytic_gaussian",
                "catalogue_model": "correlated_gaussian",
                "free_parameters": manifest["config"]["dimension"],
                "data_pixels": 0,
                "instrument_reason": "analytic density; no data pixels or instrument",
                "search": "Nautilus",
                "likelihood": "numpy fp64 normalized correlated Gaussian",
                "search_settings": settings,
                "config": "explicit matched baseline (see pilot_runs.json)",
            },
            "method": {
                "id": f"nautilus-cpu-{variant}",
                "statistic": "median",
                "repetitions": len(rows),
                "warmup": "none; fresh process",
                "synchronization": "complete logical batch",
                "cache_state": "cold Python process; shared OS/font cache",
            },
            "wall_s": wall,
            "components_s": components,
            "shares_of_wall": {k: v / wall for k, v in components.items()},
            "likelihood_function_calls": statistics.median(r["n_like"] for r in rows),
            "reference": None,
            "identity": {
                "device": "cpu",
                "backend": "numpy",
                "precision": "float64",
                "library_version": env["sources"]["PyAutoFit"]["revision"],
                "software": {
                    **env["packages"],
                    "PyAutoFit": env["sources"]["PyAutoFit"]["revision"],
                },
            },
            "provenance": {
                "host": env["host"],
                "measured_at": max(r["manifest"]["measured_at"] for r in rows),
            },
            "pilot_evidence": "pilot_runs.json",
        }
        (args.export / f"{variant}.json").write_text(json.dumps(doc, indent=2))
    (args.export / "summary.json").write_text(
        json.dumps({"groups": summary, "paired_tracing": overhead}, indent=2)
    )
    content = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Nautilus CPU baseline</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}svg{width:100%}</style><h1>Nautilus CPU profiling baseline · phase 0a</h1><p>Exploratory local CPU evidence. No scored accuracy or production-speedup verdict.</p><table><tr><th>Adapter</th><th>Workers</th><th>Cost</th><th>Tracing</th><th>Seeds</th><th>Median wall (s)</th><th>logZ error range</th></tr>'
    ]
    for g in summary:
        content.append(
            f"<tr><td>{g['adapter']}</td><td>{g['workers']}</td><td>{g['cost']}</td><td>{g['traced']}</td><td>{g['n']}</td><td>{g['wall_s']:.3f}</td><td>{html.escape(str(g['logZ_error_range']))}</td></tr>"
        )
    content.append(
        "</table><p>Detailed trace/control pairs have identical posterior hashes and likelihood counts. Ratios include measurement noise; no assumed timing correction is applied. Failed and unrun cells are retained in pilot_runs.json.</p>"
    )
    for case in pilot["cases"]:
        if (
            case["status"] == "success"
            and case["workers"] == 2
            and case["trace"] == "on"
            and case["cost_profile"] == "correlated"
            and case["seed"] == 0
        ):
            source = args.pilot / case["id"] / "timeline.json.gz"
            events = json.loads(gzip.decompress(source.read_bytes()))
            destination = args.export / f"{case['adapter']}-correlated-timeline.json.gz"
            shutil.copyfile(source, destination)
            content.append(f"<h2>{case['adapter']}: first eight likelihood batches</h2>")
            content.append(timeline_svg(events))
    content.append("</html>")
    (args.export / "report.html").write_text("".join(content))
    print(
        json.dumps(
            {
                "successful": sum(c["status"] == "success" for c in pilot["cases"]),
                "planned": len(pilot["cases"]),
                "paired": len(overhead),
                "groups": len(groups),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
