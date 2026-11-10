"""Export compact plotting evidence without admitting an unmatched timing arm."""

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair"], {})[row["backend"]] = row
    result = []
    for number, arms in sorted(grouped.items()):
        if set(arms) != {"baseline", "candidate"}:
            continue
        a, b = arms["baseline"], arms["candidate"]
        if a["png_sha256"] != b["png_sha256"]:
            raise ValueError("paired images differ")
        result.append(
            {
                "pair": number,
                "process_speedup": a["process_s"] / b["process_s"],
                "wrapper_speedup": a["wrapper_s"] / b["wrapper_s"],
                "plot_speedup": a["plot_s"] / b["plot_s"],
            }
        )
    return result


def report(root, destination):
    state = json.loads((root / "state.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    rows = []
    for attempt in state["attempts"]:
        if attempt["status"] != "success":
            continue
        case = next(c for c in plan["cases"] if c["id"] == attempt["case"])
        child = attempt["child"]
        if (
            child["new_sampler_likelihood_calls"] != 0
            or child["summary_timing_calls"] != 1
            or child["posterior"]["posterior_sha256"] != case["original_posterior_sha256"]
        ):
            raise ValueError("sampling/summary invariant differs")
        for path in (Path(case["checkpoint"]), root / case["id"] / "checkpoint.hdf5"):
            if digest(path) != case["checkpoint_sha256"]:
                raise ValueError("checkpoint bytes changed")
        png = child["plot"]["png"]
        if digest(root / case["id"] / png["path"]) != png["sha256"]:
            raise ValueError("PNG bytes changed")

        stage_seconds = {}
        for stage in attempt["stages"]["completed"]:
            name = stage["stage"]
            stage_seconds[name] = stage_seconds.get(name, 0.0) + stage["wall_s"]
        rows.append(
            {
                "case": case["id"],
                "pair": int(case["id"].split("-")[1]),
                "backend": case["backend"],
                "process_s": attempt["process_wall_s"],
                "wrapper_s": stage_seconds["wrapped_replay"],
                "plot_s": stage_seconds["search_corner"],
                "png_export_s": stage_seconds["png_export"],
                "axes_s": stage_seconds["axes_construction"],
                "peak_rss_mib": attempt["supervision"]["sampled_group_peak_rss_bytes"] / 1024**2,
                "png_sha256": png["sha256"],
                "checkpoint_sha256": case["checkpoint_sha256"],
                "posterior_sha256": case["original_posterior_sha256"],
                "new_sampler_calls": 0,
                "summary_timing_calls": 1,
            }
        )
    paired = pairs(rows)
    summary = {
        "status": state["status"],
        "reason": state.get("reason"),
        "spent_s": state["spent_s"],
        "protocol": plan["protocol"],
        "limits": plan["limits"],
        "measured_environments": plan["environments"],
        "plan_sha256": plan["sha256"],
        "rows": rows,
        "pairs": paired,
        "unrun": [plan["cases"][i]["id"] for i in state["unrun"]],
        "interpretation": "exploratory paired output replays, not sampler or statistical-efficiency speedups",
    }
    if paired:
        values = [p["process_speedup"] for p in paired]
        summary["process_speedup"] = {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "complete_pairs": len(values),
        }
    previous = root.parent / "compact-array-01/state.json"
    if previous.exists():
        summary["retained_preflight_failure"] = json.loads(previous.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    image_case = next(c for c in plan["cases"] if c["backend"] == "candidate")
    attempt = next(r for r in state["attempts"] if r["case"] == image_case["id"])
    shutil.copyfile(
        root / image_case["id"] / attempt["child"]["plot"]["png"]["path"],
        destination / "compact_corner.png",
    )
    lines = [
        "# Compact corner plotting experiment",
        "",
        "The array renderer reduced full replay time in both completed pairs while producing byte-identical PNGs. "
        "Five cases completed; the host-load guard stopped the final candidate case. "
        "The third baseline is unpaired and is excluded from every speedup summary.",
        "",
        "| Pair | Backend | Process (s) | Wrapped output (s) | Corner plot (s) | Peak RSS (MiB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair']} | {row['backend']} | {row['process_s']:.2f} | "
            f"{row['wrapper_s']:.2f} | {row['plot_s']:.2f} | {row['peak_rss_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Completed-pair process ratios (baseline / candidate): "
            + ", ".join(f"{p['process_speedup']:.3f}x" for p in paired)
            + ".",
            "The median/range in summary.json use only those two pairs. Timing variation is material; "
            "this is exploratory evidence and does not justify changing the default.",
            "",
            "All five cases retained all 33,984 weighted samples, the native count of 34,100, unchanged "
            "posterior/checkpoint hashes, zero sampler likelihood calls and exactly one separately authorized "
            "final-summary timing call. Both matched pairs have identical PNG bytes. "
            "The canvas remains 12x12 inches at 100 dpi with every cross-parameter panel and unchanged typography.",
            "",
            f"Aggregate elapsed time: {state['spent_s']:.2f}s, including preflight, against 1800s. "
            "Every completed child stayed within 240s/4096 MiB. Memory is sampled process-group RSS; "
            "brief spikes can be missed. The start guard uses one-minute host load <=2 for the eight-CPU affinity. "
            "The precise failing host snapshot was not persisted. No automatic retry followed it.",
            "",
            "The first attempt stopped before any child because the activation omitted auxiliary package "
            "metadata paths. No installed dependency changed. The corrected attempt was explicitly approved; "
            "both attempts and all raw data remain retained.",
            "",
            "Measured library: e57611dc6; producer: 3bce28c; baseline: bdb7af0e9. "
            "A later library guard sends non-string labels back to pandas; the measured string-label drawing path "
            "is unchanged. See summary.json for complete frozen source/package identities.",
            "",
            "[Library PR](https://github.com/PyAutoLabs/PyAutoFit/pull/1764) · "
            "[Experiment PR](https://github.com/PyAutoLabs/autofit_profiling/pull/14). "
            "Outside Pulse. Default backend remains anesthetic; merge remains human. "
            "This does not resolve 30D evidence accuracy or complete the wider phase 0b.",
            "",
            "![Measured compact corner plot](compact_corner.png)",
            "",
        ]
    )
    (destination / "README.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report(args.root, args.destination)
