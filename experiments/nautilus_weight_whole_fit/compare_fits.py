"""Paired whole native fits for the retained-array scalar weight candidate."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/misc"))
from profiling.nautilus_resources import atomic_json, supervise

BASELINE_SHA = "081a2bda9a61e24e2c82663b6b4711461e86ddd924c8e81ba5bf66daad573286"
REFERENCE_SHA = "24f24946e38c2a4c1875f8c1d7aa8199c9d4ce89a87e170b1557f362e6b5ae36"
CANDIDATE_SHA = "965c8cd737f52049761facf7523627f7d1f35803afd2d383bfb41d2f40d1c96c"
ORDER = ("native", "candidate", "candidate", "native")
THREADS = dict.fromkeys(
    (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ),
    "1",
)


def verified(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")
    return data


def load_module(path, expected, name):
    verified(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(directory):
    return json.loads(verified(directory / "result.json", REFERENCE_SHA))


@contextmanager
def solver(variant, candidate):
    import nautilus
    from nautilus.bounds import basic

    calls = []
    with TemporaryDirectory() as tmp:
        kernel = basic.minimum_volume_enclosing_ellipsoid
        if variant == "candidate":
            module = candidate.load_candidate(Path(nautilus.__file__).parent, Path(tmp))
            kernel = module.minimum_volume_enclosing_ellipsoid

        def counted(points, *args, **kwargs):
            calls.append(points.shape)
            return kernel(points, *args, **kwargs)

        with patch.object(basic, "minimum_volume_enclosing_ellipsoid", counted):
            yield calls


def check_native_checkpoint(base, directory, expected):
    from nautilus import Sampler
    from profiling.nautilus_target import CorrelatedGaussianTarget

    path = directory / "checkpoint.hdf5"
    if not path.is_file():
        raise FileNotFoundError(path)

    def forbidden(*args, **kwargs):
        raise ValueError("Native checkpoint reopening attempted a likelihood evaluation")

    restored = Sampler(
        CorrelatedGaussianTarget.prior,
        forbidden,
        pool=None,
        filepath=str(path),
        **base.CONFIG.constructor(),
    )
    if base.signature(restored, restored.posterior()) != expected:
        raise ValueError("Unpatched native checkpoint result differs")


def child(directory, baseline_dir, variant):
    base = load_module(baseline_dir / "run.py", BASELINE_SHA, "whole_fit_baseline")
    candidate = load_module(
        ROOT / "experiments/nautilus_ellipsoid_weight_update/compare.py",
        CANDIDATE_SHA,
        "weight_candidate",
    )
    ref = reference(baseline_dir)
    expected = ref["runs"][0]["science"]
    with solver(variant, candidate) as calls:
        base.one_fit(directory, "control")
    fit = json.loads((directory / "fit.json").read_text())
    if fit["science"] != expected or not fit["checkpoint_reopen_equal"]:
        raise ValueError("Whole-fit science differs from the frozen native control")
    traced = ref["runs"][1]["builds"]
    if len(calls) != sum(b["calls"]["ellipsoid"] for b in traced):
        raise ValueError("Unexpected number of enclosing-ellipsoid calls")
    check_native_checkpoint(base, directory, expected)
    fit.update(variant=variant, solver_call_shapes=calls, native_reopen_equal=True)
    atomic_json(directory / "verified.json", fit)


def run(output, baseline_dir):
    ref = reference(baseline_dir)
    verified(baseline_dir / "run.py", BASELINE_SHA)
    candidate = ROOT / "experiments/nautilus_ellipsoid_weight_update/compare.py"
    verified(candidate, CANDIDATE_SHA)
    versions = {name: version(name) for name in ref["versions"]}
    if versions != ref["versions"] or platform.python_version() != ref["python"]:
        raise ValueError("Runtime differs from the registered whole-fit versions")
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "result.json"
    state = {
        "status": "running",
        "baseline_sha256": BASELINE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "candidate_loader_sha256": CANDIDATE_SHA,
        "nautilus_source_sha256": ref["nautilus_source_sha256"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration": ref["configuration"],
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "limits": ref["limits"],
        "runs": [],
        "unrun": list(ORDER),
    }
    atomic_json(receipt, state)
    started = time.monotonic()
    try:
        for i, variant in enumerate(ORDER, 1):
            directory = output / f"{i}-{variant}"
            directory.mkdir()
            row = {"case": directory.name, "variant": variant, "load_average": os.getloadavg()}
            state["runs"].append(row)
            remaining = state["limits"]["total_s"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Aggregate allowance exhausted")
            with (directory / "run.log").open("w") as log:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        str(directory),
                        str(baseline_dir),
                        "--child",
                        variant,
                    ],
                    env={**os.environ, **THREADS},
                    start_new_session=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                state["unrun"].pop(0)
                row["supervision"] = supervise(
                    proc,
                    min(state["limits"]["child_s"], remaining),
                    state["limits"]["group_mib"] * 1024**2,
                )
            if row["supervision"]["returncode"] != 0:
                raise RuntimeError(f"{directory.name} failed; see its log and supervisor receipt")
            row.update(json.loads((directory / "verified.json").read_text()))
            shapes = row.pop("solver_call_shapes")
            row["solver_call_count"] = len(shapes)
            row["solver_shapes_sha256"] = hashlib.sha256(json.dumps(shapes).encode()).hexdigest()
            if row["solver_shapes_sha256"] != state["runs"][0]["solver_shapes_sha256"]:
                raise ValueError(f"{directory.name} solver input shape sequence differs")
            if time.monotonic() - started > state["limits"]["total_s"]:
                raise TimeoutError("Aggregate allowance exhausted")
            row["status"] = "success"
            atomic_json(receipt, state)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="halted", reason=f"{type(error).__name__}: {error}")
        if state["runs"] and "status" not in state["runs"][-1]:
            state["runs"][-1]["status"] = "failed"
        raise
    finally:
        state["elapsed_s"] = time.monotonic() - started
        atomic_json(receipt, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Fresh output directory")
    parser.add_argument(
        "baseline_dir", type=Path, help="Pinned whole-fit experiment directory from PR #21"
    )
    parser.add_argument("--child", choices=("native", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.output, args.baseline_dir, args.child)
    else:

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, interrupted)
        run(args.output, args.baseline_dir)
