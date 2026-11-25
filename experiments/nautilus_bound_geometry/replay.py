"""Replay the outer mixture constructor on the frozen serial controls from RAL job 415196."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from contextlib import nullcontext
from importlib.metadata import distribution
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import h5py
import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

CASES = {
    "fit-00-A": "397f6c0ac6217755f05e89717c96e4494a4df16694e6ab3cf43d333f7b6a5d98",
    "fit-11-A": "8567a0485d82f592206c3d26382b97e09aeb4bb22413f8d173f7c0d6835443a5",
}
CHECKPOINT_SHA = "752fbc1390babd98838eb1bb4c28c9d36420e512c6eb4da201a3217f6931cbf3"
SOURCE_SHA = "61e91f8fb85cf4a38877cd1c65bb0b111589bde77f5f74acb22a90f3ff2ec45c"


def verify_file(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")


def load_basic(source):
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != SOURCE_SHA:
        raise ValueError("Nautilus source differs from the job 415196 pin")
    # Load only the native geometry module. This replay has no sampler or neural workload.
    spec = importlib.util.spec_from_file_location("native_basic", source / "bounds/basic.py")
    basic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic)
    return basic


def load_inputs(pilot, basic):
    inputs = {}
    timeline = []
    for case, receipt_sha in CASES.items():
        child_path = pilot / case / "child.json"
        checkpoint = pilot / case / "final-checkpoint.hdf5"
        verify_file(child_path, receipt_sha)
        verify_file(checkpoint, CHECKPOINT_SHA)
        child = json.loads(child_path.read_text())
        events = child["phases"]["events"]
        builds = [event for event in events if event["phase"] == "NautilusBound.compute"]
        if len(builds) != 11 or not child["historical_reference_verified"]:
            raise ValueError(f"{case} has an incomplete reference trajectory")
        with h5py.File(checkpoint, "r") as file:
            for ordinal, event in enumerate(builds, 1):
                group = file[f"bound_{ordinal}/outer_bound"]
                if len(group.attrs["log_v_all"]) != 1 or group.attrs["enlarge_per_dim"] != 1.1:
                    raise ValueError(f"Unexpected outer union at bound {ordinal}")
                points = group["points_bound_0"][:]
                if points.shape != (400, 10):
                    raise ValueError(f"Unexpected geometry input at bound {ordinal}")
                expected = basic.UnitCubeEllipsoidMixture.read(group["bound_0"])
                inputs[ordinal] = points, expected  # The checkpoints have identical SHA-256.
                nested = Counter()
                for entry in events:
                    if entry["parent"] == event["id"]:
                        nested[entry["phase"]] += entry["finish_ns"] - entry["start_ns"]
                inclusive = event["finish_ns"] - event["start_ns"]
                if event["exclusive_wall_ns"] + sum(nested.values()) != inclusive:
                    raise ValueError(f"{case} bound {ordinal} timing does not reconcile")
                timeline.append(
                    {
                        "case": case,
                        "ordinal": ordinal,
                        "cube_dims": int(expected.dim_cube.sum()),
                        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                        "geometry_wall_s": event["exclusive_wall_ns"] / 1e9,
                        "geometry_parent_cpu_s": event["exclusive_cpu_ns"] / 1e9,
                        "inclusive_s": inclusive / 1e9,
                        "nested_s": {name: ns / 1e9 for name, ns in nested.items()},
                    }
                )
    return inputs, timeline


def check_geometry(bound, expected, points):
    np.testing.assert_array_equal(bound.dim_cube, expected.dim_cube)
    np.testing.assert_allclose(bound.log_v, expected.log_v, rtol=1e-8, atol=1e-10)
    if (bound.ellipsoid is None) != (expected.ellipsoid is None):
        raise ValueError("Ellipsoid presence differs from checkpoint")
    if bound.ellipsoid is not None:
        for field in ("c", "A", "B", "B_inv"):
            np.testing.assert_allclose(
                getattr(bound.ellipsoid, field),
                getattr(expected.ellipsoid, field),
                rtol=1e-8,
                atol=1e-10,
            )
    np.testing.assert_array_equal(bound.contains(points), expected.contains(points))


def measure(basic, points, expected, probed):
    durations = []
    dimensions = []
    original = basic.minimum_volume_enclosing_ellipsoid

    def timed(array, *args, **kwargs):
        start = perf_counter_ns()
        result = original(array, *args, **kwargs)
        durations.append(perf_counter_ns() - start)
        dimensions.append(array.shape[1])
        return result

    scope = patch.object(basic, "minimum_volume_enclosing_ellipsoid", timed)
    rng = np.random.default_rng(2)
    state = rng.bit_generator.state
    with scope if probed else nullcontext():
        start = perf_counter_ns()
        bound = basic.UnitCubeEllipsoidMixture.compute(points, enlarge_per_dim=1.1, rng=rng)
        elapsed = perf_counter_ns() - start
    if rng.bit_generator.state != state:
        raise ValueError("Geometry replay consumed random draws")
    check_geometry(bound, expected, points)
    return {
        "probed": probed,
        "wall_s": elapsed / 1e9,
        "mvee_calls": len(durations) if probed else None,
        "mvee_dims": dimensions if probed else None,
        "mvee_s": sum(durations) / 1e9 if probed else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path, help="Extracted parent-cpu-diagnostics-ral-01/pilot")
    parser.add_argument("output", type=Path, help="New JSON receipt")
    parser.add_argument(
        "--nautilus-source", type=Path, help="Optional pinned nautilus package path"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a fresh path.")
    source = args.nautilus_source or Path(distribution("nautilus-sampler").locate_file("nautilus"))
    basic = load_basic(source)
    inputs, timeline = load_inputs(args.pilot, basic)
    measured = []
    with threadpool_limits(limits=1):
        libraries = [
            {
                key: lib.get(key)
                for key in ("internal_api", "num_threads", "version", "architecture")
            }
            for lib in threadpool_info()
        ]
        for points, expected in inputs.values():
            measure(basic, points, expected, False)
        for block, order in enumerate((list(inputs), list(reversed(inputs)))):
            for ordinal in order:
                modes = (False, True) if (ordinal + block) % 2 else (True, False)
                for probed in modes:
                    row = measure(basic, *inputs[ordinal], probed)
                    measured.append({"block": block + 1, "ordinal": ordinal, **row})
    result = {
        "source_sha256": SOURCE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "child_sha256": CASES,
        "local_environment": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "native_threads": libraries,
        },
        "ral_builds": timeline,
        "preflight_passed": list(inputs),
        "local_probes": measured,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
