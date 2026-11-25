"""Bounded local whole-fit attribution for the 10D serial Nautilus diagnostic."""

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/misc"))
from profiling.exclusive_timer import ExclusiveTimer
from profiling.nautilus_manifest import RunConfig
from profiling.nautilus_resources import atomic_json, supervise

ORDER = ("control", "traced", "traced", "control")
SOURCE_SHA = "61e91f8fb85cf4a38877cd1c65bb0b111589bde77f5f74acb22a90f3ff2ec45c"
CONFIG = RunConfig(dimension=10, seed=2, n_live=400, n_eff=10000, cost_profile="none", work=0)
STOP = {**CONFIG.stopping(), "discard_exploration": True}
THREAD_LIMITS = dict.fromkeys(
    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1"
)


@contextmanager
def trace_builds(rows):
    from nautilus.bounds import NautilusBound, basic
    from nautilus.neural import NeuralNetworkEmulator

    compute = vars(NautilusBound)["compute"].__func__
    train = vars(NeuralNetworkEmulator)["train"].__func__

    def traced(cls, *args, **kwargs):
        timer = ExclusiveTimer()
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    basic,
                    "minimum_volume_enclosing_ellipsoid",
                    timer.wrap("ellipsoid", basic.minimum_volume_enclosing_ellipsoid),
                )
            )
            stack.enter_context(
                patch.object(
                    NeuralNetworkEmulator, "train", classmethod(timer.wrap("train", train))
                )
            )
            stack.enter_context(
                patch.object(
                    NeuralNetworkEmulator,
                    "predict",
                    timer.wrap("predict", NeuralNetworkEmulator.predict),
                )
            )
            bound = timer.wrap("other_build", compute)(cls, *args, **kwargs)
        rows.append({"exclusive_s": timer.totals, "calls": timer.counts})
        return bound

    with patch.object(NautilusBound, "compute", classmethod(traced)):
        yield


def signature(sampler, posterior):
    import numpy as np

    digest = hashlib.sha256()
    for array in posterior:
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "posterior_sha256": digest.hexdigest(),
        "n_like": int(sampler.n_like),
        "ess": float(sampler.n_eff),
        "log_z": float(sampler.log_z),
        "shells": len(sampler.bounds),
        "explored": bool(sampler.explored),
        "discard_exploration": bool(sampler.discard_exploration),
    }


def one_fit(directory, mode):
    import nautilus
    from nautilus import Sampler
    from profiling.nautilus_target import CorrelatedGaussianTarget
    from threadpoolctl import threadpool_info, threadpool_limits

    source = Path(nautilus.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != SOURCE_SHA:
        raise ValueError("Nautilus source differs from the registered pin")

    target = CorrelatedGaussianTarget(10, "none", 0)
    checkpoint = directory / "checkpoint.hdf5"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    args = {**CONFIG.constructor(), "filepath": str(checkpoint), "pool": None}
    builds = []
    with threadpool_limits(1):
        libraries = [
            {
                key: lib.get(key)
                for key in ("internal_api", "version", "num_threads", "architecture")
            }
            for lib in threadpool_info()
        ]
        if any(lib["num_threads"] != 1 for lib in libraries):
            raise RuntimeError("Thread cap was not applied")
        with trace_builds(builds) if mode == "traced" else nullcontext():
            start = time.perf_counter()
            sampler = Sampler(target.prior, target, **args)
            initialized = time.perf_counter()
            finished = sampler.run(**STOP)
            sampled = time.perf_counter()
            posterior = sampler.posterior()
            end = time.perf_counter()
        result = signature(sampler, posterior)
        if not finished or not sampler.explored or sampler.n_eff < CONFIG.n_eff:
            raise ValueError("Incomplete native fit")

        def forbidden(*args, **kwargs):
            raise AssertionError("Reopen evaluated the likelihood")

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        reopened = Sampler(target.prior, forbidden, **args)
        if signature(reopened, reopened.posterior()) != result:
            raise ValueError("Reopened checkpoint differs from the final result")
    if mode == "traced":
        if len(builds) != len(sampler.bounds) - 1 or not builds:
            raise ValueError("Bound build count differs from final checkpoint")
        for row in builds:
            if min(row["calls"].get(k, 0) for k in ("ellipsoid", "train", "predict")) < 1:
                raise ValueError("A bound build missed a required timer")
            if min(row["exclusive_s"].values()) < 0:
                raise ValueError("Negative exclusive time")
        if sum(sum(row["exclusive_s"].values()) for row in builds) > sampled - initialized:
            raise ValueError("Bound time exceeds sampling time")
    atomic_json(
        directory / "fit.json",
        {
            "science": result,
            "checkpoint_reopen_equal": True,
            "native_threads": libraries,
            "fit_s": end - start,
            "constructor_s": initialized - start,
            "sampling_s": sampled - initialized,
            "posterior_s": end - sampled,
            "builds": builds,
        },
    )


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "configuration": {"constructor": CONFIG.constructor(), "stopping": STOP},
        "nautilus_source_sha256": SOURCE_SHA,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            n: version(n)
            for n in (
                "nautilus-sampler",
                "numpy",
                "scipy",
                "scikit-learn",
                "h5py",
                "threadpoolctl",
                "joblib",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "limits": {"child_s": 180, "group_mib": 4096, "total_s": 720},
        "runs": [],
        "unrun": list(ORDER),
    }
    receipt = output / "result.json"
    atomic_json(receipt, state)
    start = time.monotonic()
    for i, mode in enumerate(ORDER):
        directory = output / f"{i + 1}-{mode}"
        directory.mkdir()
        row = {"case": directory.name}
        try:
            remaining = 720 - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError("Aggregate budget exhausted")
            with (directory / "run.log").open("w") as log:
                child = subprocess.Popen(
                    [sys.executable, __file__, str(directory), "--child", mode],
                    env={**os.environ, **THREAD_LIMITS},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state["unrun"].pop(0)
                row["supervision"] = supervise(child, min(180, remaining), 4096 * 1024**2)
            if row["supervision"]["returncode"] != 0:
                raise RuntimeError(f"{directory.name}: see run.log and supervision")
            row.update(json.loads((directory / "fit.json").read_text()))
            if i and row["science"] != state["runs"][0]["science"]:
                raise ValueError("Exact local sampling parity failed")
            if time.monotonic() - start > 720:
                raise TimeoutError("Aggregate budget exhausted")
            row["status"] = "success"
        except BaseException as error:
            state["status"] = "halted"
            row.update(status="failed", error=str(error))
            raise
        finally:
            state["runs"].append(row)
            state["elapsed_s"] = time.monotonic() - start
            atomic_json(receipt, state)
    state["status"] = "complete"
    atomic_json(receipt, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="Fresh output directory for checkpoints and receipt"
    )
    parser.add_argument("--child", choices=("control", "traced"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        one_fit(args.output, args.child)
    else:

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, interrupted)
        run(args.output)
