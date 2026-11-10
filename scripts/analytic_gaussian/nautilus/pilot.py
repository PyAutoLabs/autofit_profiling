"""Bounded sequential pilot; retains failures and unrun cells. Never submits compute."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from profiling.nautilus_manifest import source_pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=1800)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--work", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for seed, cost, workers, adapter in itertools.product(
        range(3), ("fixed", "correlated"), (1, 2), ("direct", "pyautofit")
    ):
        for trace in ("off", "on") if seed % 2 == 0 else ("on", "off"):
            cases.append(
                {
                    "adapter": adapter,
                    "workers": workers,
                    "seed": seed,
                    "cost_profile": cost,
                    "trace": trace,
                }
            )
    for adapter, trace in itertools.product(("direct", "pyautofit"), ("off", "on")):
        cases.append(
            {"adapter": adapter, "workers": 2, "seed": 0, "cost_profile": "random", "trace": trace}
        )
    for i, case in enumerate(cases):
        case.update(id=f"run-{i:03}", status="unrun")

    def save():
        (args.output / "pilot.json").write_text(
            json.dumps(
                {
                    "budget_s": args.budget,
                    "timeout_s": args.timeout,
                    "work": args.work,
                    "cases": cases,
                },
                indent=2,
            )
        )

    save()
    expected_sources = None
    start = time.monotonic()
    for case in cases:
        remaining = args.budget - (time.monotonic() - start)
        if remaining < args.timeout:
            break
        command = [
            sys.executable,
            str(HERE / "cpu_baseline.py"),
            "--output",
            str(args.output / case["id"]),
            "--timeout",
            str(args.timeout),
            "--work",
            str(args.work),
        ]
        for key in ("adapter", "workers", "seed", "cost_profile", "trace"):
            command.extend(["--" + key.replace("_", "-"), str(case[key])])
        case["host_load_before"] = list(os.getloadavg())
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        case["returncode"] = result.returncode
        case["driver_output"] = result.stdout[-2000:] + result.stderr[-2000:]
        output = args.output / case["id"] / "result.json"
        row = json.loads(output.read_text()) if output.exists() else {"status": "driver_failure"}
        case["status"] = row["status"]
        if row["status"] == "success":
            actual_sources = source_pins(row["manifest"]["environment"]["sources"])
            if expected_sources is not None and expected_sources != actual_sources:
                case["status"] = "source_mismatch"
                case["included_in_analysis"] = False
                case["exclusion_reason"] = "source revision/content changed between pilot cases"
            expected_sources = actual_sources
        save()
        print(
            case["id"],
            case["adapter"],
            case["workers"],
            case["cost_profile"],
            case["trace"],
            case["status"],
            round(row.get("wall_s", 0), 2),
            flush=True,
        )
        if case["status"] != "success":
            print("Pilot halted on failure; remaining cases retained as unrun.", flush=True)
            break
    return 0 if all(c["status"] == "success" for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
