"""Replay two-worker phase 0a batches using their recorded service times."""

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = ROOT / "results" / "analytic_gaussian" / "nautilus"
MANIFEST_SHA256 = "7370d4a1a86d216e7c100c1b1f601ec329230bb84a5821ae451870dfbc04ec60"
sys.path.insert(0, str(ROOT / "scripts" / "misc"))

from profiling.nautilus_trace import reconcile


def greedy_makespan(durations, chunk):
    """Assign the next contiguous chunk to the first of two available workers."""
    free = [0, 0]
    for i in range(0, len(durations), chunk):
        worker = 0 if free[0] <= free[1] else 1
        free[worker] += sum(durations[i : i + chunk])
    return max(free)


def replay_batch(events):
    events = sorted(events, key=lambda e: e["id"][1])
    submits = {e["submit_ns"] for e in events}
    collects = {e["collect_ns"] for e in events}
    if len(submits) != 1 or len(collects) != 1 or len({e["pid"] for e in events}) != 2:
        raise ValueError("Expected one pooled map with two observed workers")
    # CPython Pool.map_async defaults to ceil(n / (4 * workers)).
    chunk = math.ceil(len(events) / 8)
    for i in range(0, len(events), chunk):
        if len({e["pid"] for e in events[i : i + chunk]}) != 1:
            raise ValueError("Observed assignments split a native-sized chunk")
    durations = [e["finish_ns"] - e["start_ns"] for e in events]
    window = max(collects) - min(submits)
    span = max(e["finish_ns"] for e in events) - min(e["start_ns"] for e in events)
    floor = max(max(durations), (sum(durations) + 1) // 2)
    native = greedy_makespan(durations, chunk)
    single = greedy_makespan(durations, 1)
    if not (floor <= span <= window and floor <= min(native, single)):
        raise ValueError("Impossible two-worker schedule")
    return dict(
        map_ns=window,
        service_ns=sum(durations),
        floor_ns=floor,
        edge_ns=window - span,
        native_model_ns=native,
        single_model_ns=single,
    )


def replay(record, compressed):
    result, case = record["result"], record["case"]
    sha = hashlib.sha256(compressed).hexdigest()
    if sha != result["timeline"]["sha256"]:
        raise ValueError(f"Wrong timeline for {case['id']}")
    trace = json.loads(gzip.decompress(compressed))
    batches = trace["batches"]
    by_batch = reconcile(trace["events"], batches)
    ordered = sorted(batches, key=lambda b: b["start_ns"])
    if len(by_batch) != len(batches) or any(
        a["finish_ns"] > b["start_ns"] for a, b in zip(ordered, ordered[1:])
    ):
        raise ValueError("Missing, duplicate or overlapping batch windows")
    if len(trace["events"]) != result["n_like"]:
        raise ValueError("Proposal count disagrees with native likelihood count")
    windows = [replay_batch(by_batch[b["id"]]) for b in ordered]
    totals = {key: sum(batch[key] for batch in windows) for key in windows[0]}
    wall_ns = result["wall_s"] * 1e9
    if totals["map_ns"] > wall_ns:
        raise ValueError("Map windows exceed full fit")
    return {
        "id": case["id"],
        "adapter": case["adapter"],
        "cost": case["cost_profile"],
        "seed": case["seed"],
        "native_update": result["resolved_constructor"]["n_update"],
        "source_sha256": sha,
        "fit_wall_s": result["wall_s"],
        "batches": len(windows),
        "proposals": len(trace["events"]),
        **totals,
        "observed_occupancy_pct": 50 * totals["service_ns"] / totals["map_ns"],
        "all_map_pct_fit": 100 * totals["map_ns"] / wall_ns,
        "scheduling_ceiling_pct_fit": 100 * (totals["map_ns"] - totals["floor_ns"]) / wall_ns,
        "single_vs_native_model_pct_fit": 100
        * (totals["native_model_ns"] - totals["single_model_ns"])
        / wall_ns,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="read original run-ID directories")
    parser.add_argument(
        "--examples-only", action="store_true", help="replay the two original public examples"
    )
    args = parser.parse_args()
    raw = (BASELINE / "pilot_runs.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Phase 0a manifest changed")
    examples = {
        hashlib.sha256(p.read_bytes()).hexdigest(): p for p in BASELINE.glob("*-timeline.json.gz")
    }
    rows = []
    for record in json.loads(raw)["runs"]:
        case = record["case"]
        if not (
            case.get("included_in_analysis") and case["workers"] == 2 and case["trace"] == "on"
        ):
            continue
        if case["status"] != "success":
            raise ValueError(f"Incomplete selected case: {case['id']}")
        path = examples.get(record["result"]["timeline"]["sha256"])
        if args.examples_only and path is None:
            continue
        if args.archive:
            path = args.archive / case["id"] / "timeline.json.gz"
        elif path is None:
            path = HERE / "timelines" / (case["id"] + ".json.gz")
        rows.append(replay(record, path.read_bytes()))
    if len(rows) != (2 if args.examples_only else 14):
        raise ValueError("Unexpected selected case count")
    print(json.dumps({"manifest_sha256": MANIFEST_SHA256, "runs": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
