"""Paired ablation of the one-hot weight update in the pinned Nautilus MVEE solver."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from importlib.metadata import distribution, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns, process_time_ns
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/nautilus_bound_geometry"))
sys.path.insert(0, str(ROOT / "scripts/misc"))
from profiling.nautilus_resources import atomic_json
from replay import CHECKPOINT_SHA, SOURCE_SHA, check_geometry, load_basic, load_inputs

ORIGINAL = "            u = u * (1 - a) + a * (np.arange(n_points) == j)"
REPLACEMENT = "            u *= 1 - a\n            u[j] += a"
BLOCKS = 6


def load_candidate(source, directory):
    text = (source / "bounds/basic.py").read_text()
    if text.count(ORIGINAL) != 1:
        raise ValueError("Expected exactly one native weight update")
    path = directory / "basic_candidate.py"
    path.write_text(text.replace(ORIGINAL, REPLACEMENT))
    spec = importlib.util.spec_from_file_location("mvee_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(array):
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def geometry(bound, points):
    result = {
        "mask": identity(bound.dim_cube),
        "membership": identity(bound.contains(points)),
        "log_volume": identity(np.asarray(bound.log_v)),
    }
    if bound.ellipsoid is not None:
        for field in ("c", "A", "B", "B_inv"):
            result[field] = identity(getattr(bound.ellipsoid, field))
    return result


def construct(module, points, rng):
    return module.UnitCubeEllipsoidMixture.compute(points, enlarge_per_dim=1.1, rng=rng)


def preflight(module, points, expected):
    calls = []
    kernel = module.minimum_volume_enclosing_ellipsoid

    def record(array, *args, **kwargs):
        before = identity(array)
        output = kernel(array, *args, **kwargs)
        if before != identity(array):
            raise ValueError("Solver modified its input")
        calls.append({"input": before, "outputs": [identity(x) for x in output]})
        return output

    rng = np.random.default_rng(2)
    state = rng.bit_generator.state
    with patch.object(module, "minimum_volume_enclosing_ellipsoid", record):
        bound = construct(module, points, rng)
    if state != rng.bit_generator.state:
        raise ValueError("Solver consumed random draws")
    check_geometry(bound, expected, points)
    return calls, geometry(bound, points)


def run(inputs, native, candidate, state, receipt):
    modules = {"native": native, "candidate": candidate}
    reference = {}
    for ordinal, (points, expected) in inputs.items():
        points.setflags(write=False)
        state["current"] = {"phase": "parity", "ordinal": ordinal}
        calls, signature = preflight(native, points, expected)
        other_calls, other_signature = preflight(candidate, points, expected)
        if calls != other_calls or signature != other_signature:
            raise ValueError(f"Bitwise solver or bound parity failed at bound {ordinal}")
        reference[ordinal] = signature
        state["parity"].append(
            {
                "ordinal": ordinal,
                "call_count": len(calls),
                "call_sequence_sha256": hashlib.sha256(
                    json.dumps(calls, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
        atomic_json(receipt, state)
    if sum(row["call_count"] for row in state["parity"]) != 57:
        raise ValueError("Incomplete retained solver path")

    for block in range(BLOCKS):
        ordinals = list(inputs) if block % 2 == 0 else list(reversed(inputs))
        for ordinal in ordinals:
            order = list(modules) if (block + ordinal) % 2 else list(reversed(modules))
            points, _ = inputs[ordinal]
            for name in order:
                state["current"] = {"block": block + 1, "ordinal": ordinal, "variant": name}
                rng = np.random.default_rng(2)
                rng_before = rng.bit_generator.state
                cpu = process_time_ns()
                wall = perf_counter_ns()
                bound = construct(modules[name], points, rng)
                elapsed = perf_counter_ns() - wall
                cpu_elapsed = process_time_ns() - cpu
                if (
                    rng.bit_generator.state != rng_before
                    or geometry(bound, points) != reference[ordinal]
                ):
                    raise ValueError(f"Timed geometry differs at bound {ordinal}")
                state["timings"].append(
                    {**state["current"], "wall_ns": elapsed, "cpu_ns": cpu_elapsed}
                )
            atomic_json(receipt, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path, help="Retained job 415196 pilot directory")
    parser.add_argument("output", type=Path, help="Fresh directory for result.json")
    parser.add_argument("--nautilus-source", type=Path, help="Pinned nautilus package directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = args.output / "result.json"
    state = {
        "status": "running",
        "source_sha256": SOURCE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "h5py", "threadpoolctl")},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "parity": [],
        "timings": [],
    }
    atomic_json(receipt, state)
    try:
        source = args.nautilus_source or Path(
            distribution("nautilus-sampler").locate_file("nautilus")
        )
        native = load_basic(source)
        inputs, _ = load_inputs(args.pilot, native)
        with TemporaryDirectory() as tmp, threadpool_limits(1):
            candidate = load_candidate(source, Path(tmp))
            state["threads"] = [
                {k: lib.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                for lib in threadpool_info()
            ]
            if any(lib["num_threads"] != 1 for lib in state["threads"]):
                raise RuntimeError("Numerical thread limit was not applied")
            run(inputs, native, candidate, state, receipt)
        state.pop("current", None)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="halted", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(receipt, state)


if __name__ == "__main__":
    main()
