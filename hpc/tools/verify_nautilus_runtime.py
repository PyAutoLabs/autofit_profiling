"""Import/identity validation only; never evaluates a likelihood or trains a network."""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path


def validate_inventory(identity, reference, manifest):
    def normalize(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    expected = {
        normalize(k): v
        for k, v in reference["reference_distribution_versions"].items()
        if normalize(k) not in ("autofit", "autonerves")
    }
    actual = {normalize(k): v for k, v in identity["distribution_versions"].items()}
    if actual != expected or len(actual) != 90:
        raise ValueError("complete third-party distribution inventory differs")
    if identity["python"] != manifest["python_build"]:
        raise ValueError("Python binary build differs from staged runtime")
    if identity["sources"]["producer"]["revision"] != manifest["sources"]["autofit_profiling"]:
        raise ValueError("producer revision differs from staged runtime")
    if any(row["dirty"] for row in identity["sources"].values()):
        raise ValueError("restored measurement sources are dirty")


def validate_capacity(snapshot):
    available = min(
        row["limit"] - row["used"]
        for row in snapshot["memory"]["ancestors"]
        if row["limit"] is not None
    )
    if available < 4 * 1024**3:
        raise ValueError("insufficient cgroup memory before science")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execution", choices=("ral", "validate-only"), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = args.cache / "sources/autofit_profiling"
    os.chdir(source)
    sys.path.insert(0, str(source / "scripts/misc"))
    from profiling import nautilus_allocation as allocation
    from profiling.nautilus_baseline import provenance

    spec = importlib.util.spec_from_file_location(
        "training_benchmark",
        source / "scripts/analytic_gaussian/nautilus/training_pool_benchmark.py",
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    reference = json.loads(
        (source / "experiments/nautilus_spawn_training_pool/runtime.json").read_text()
    )
    identity = provenance()
    benchmark.validate_sources(identity)
    benchmark.validate_runtime(identity)
    validate_inventory(identity, reference, manifest)
    receipt = {
        "status": "pass",
        "identity": identity,
        "elapsed_s": time.monotonic() - started,
        "scope": "imports and source/runtime validation only; zero sampling/training",
    }
    if args.execution == "ral":
        receipt["allocation"] = allocation.snapshot()
        validate_capacity(receipt["allocation"])
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Runtime validation passed in {receipt['elapsed_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
