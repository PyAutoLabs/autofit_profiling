"""Bounded off-Pulse checkpoint witnesses and paired CPU pilot; no retries."""

import argparse
import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts/misc"))
from profiling.nautilus_campaign import environment_key, host_snapshot
from profiling.nautilus_diagnostics import assert_environment, sha256
from profiling.nautilus_manifest import RunConfig, fingerprint
from profiling.nautilus_plot_comparison import replaced
from profiling.nautilus_resources import (
    atomic_json,
    group_rss,
    memory_available,
    supervise,
    terminate_group,
)

LIMITS = {"timeout_s": 120, "memory_mib": 4096, "budget_s": 1800}
PROTOCOL = "nautilus-checkpoint-batching-experiment@1"


def matrix():
    cases = []
    for workers in (1, 2):
        for mode in ("deferred", "atomic"):
            witness = f"workers-{workers}-{mode}"
            for phase in ("interrupt", "resume"):
                cases.append(
                    {
                        "id": f"{witness}-{phase}",
                        "phase": phase,
                        "witness": witness,
                        "mode": mode,
                        "workers": workers,
                        "dimension": 5,
                        "seed": 0,
                        "interval": 5,
                    }
                )
    for dimension in (5, 10):
        for seed in range(3):
            for interval in (1, 5) if seed % 2 == 0 else (5, 1):
                cases.append(
                    {
                        "id": f"d{dimension}-s{seed}-i{interval}",
                        "phase": "performance",
                        "workers": 1,
                        "dimension": dimension,
                        "seed": seed,
                        "interval": interval,
                    }
                )
    return cases


class Config(RunConfig):
    def stopping(self):
        return {**super().stopping(), "discard_exploration": True}


def configuration(case):
    return Config(
        adapter="pyautofit",
        workers=case["workers"],
        dimension=case["dimension"],
        seed=case["seed"],
        n_live=400,
        n_batch=100,
        n_eff=10000,
        trace=False,
        cost_profile="none",
        work=0,
    )


def validate(plan):
    if plan["sha256"] != fingerprint({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("frozen plan changed")
    if plan["protocol"] != PROTOCOL or plan["limits"] != LIMITS or plan["cases"] != matrix():
        raise ValueError("unapproved matrix/protocol/limits")
    if plan["configurations"] != [configuration(c).matching() for c in plan["cases"]]:
        raise ValueError("unapproved sampling configuration")


def posterior(sampler):
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for array in sampler.posterior():
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "posterior_sha256": digest.hexdigest(),
        # HDF5 restoration gives this scalar numpy.int64; fresh runs use int.
        "n_like": int(sampler.n_like),
        "n_eff": float(sampler.n_eff),
        "shells": len(sampler.bounds),
    }


def native(case, path, stop_mode=None, marker=None, expected_start=None):
    import h5py
    from autofit.non_linear.parallel import fork_context
    from autofit.non_linear.search.nest.nautilus.checkpoint_policy import BatchedCheckpointSampler
    from autofit.non_linear.search.nest.nautilus.search import (
        _large_attribute_hdf5,
        _LikelihoodWorkerPool,
    )
    from nautilus import Sampler
    from nautilus.pool import initialize_worker
    from profiling.nautilus_target import CorrelatedGaussianTarget

    target = CorrelatedGaussianTarget(case["dimension"], "none", 0)
    config = configuration(case)
    pool_context = contextlib.nullcontext(None)
    if case["workers"] > 1:
        pool_context = _LikelihoodWorkerPool(
            fork_context().Pool(case["workers"], initializer=initialize_worker, initargs=(target,))
        )
    with pool_context as pool, _large_attribute_hdf5(), contextlib.ExitStack() as stack:
        obj = BatchedCheckpointSampler(
            target.prior,
            target,
            filepath=path,
            pool=(pool, None) if pool else None,
            checkpoint_interval_batches=5,
            **config.constructor(),
        )

        if expected_start is not None and obj.n_like != expected_start:
            raise ValueError("resume did not restore the published likelihood count")

        def pause(mode, temporary=None):
            with h5py.File(path, "r") as saved:
                recoverable = int(saved["sampler"].attrs["n_like"])
            atomic_json(
                marker,
                {
                    "mode": mode,
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha256(path),
                    "recoverable_n_like": recoverable,
                    "completed_n_like": obj.n_like,
                    "lost_completed_evaluations": obj.n_like - recoverable,
                    "pending_notifications": obj._pending_checkpoint_batches,
                    "in_flight_evaluations": 0,
                    "temporary": temporary,
                },
            )
            os.kill(os.getpid(), signal.SIGSTOP)
            raise RuntimeError("supervisor failed to terminate paused witness")

        if stop_mode == "deferred":
            update = obj.write_shell_update

            def deferred(*args, **kwargs):
                result = update(*args, **kwargs)
                if obj.explored and obj._pending_checkpoint_batches == 2:
                    pause("deferred")
                return result

            stack.enter_context(replaced(obj, "write_shell_update", deferred))
        elif stop_mode == "atomic":
            write = Sampler.write

            def atomic(instance, filepath, **kwargs):
                if instance is obj and obj.explored and obj._pending_checkpoint_batches >= 5:
                    bound = obj.bounds[0]
                    original = bound.write

                    def partial(group):
                        original(group)
                        pause("atomic", str(filepath))

                    with replaced(bound, "write", partial):
                        return write(instance, filepath, **kwargs)
                return write(instance, filepath, **kwargs)

            stack.enter_context(replaced(Sampler, "write", atomic))
        if not obj.run(**config.stopping()):
            raise RuntimeError("native control did not converge")
        return posterior(obj)


def io_counts():
    return {
        key: int(value)
        for key, value in (
            line.split(":") for line in Path("/proc/self/io").read_text().splitlines()
        )
        if key in ("wchar", "syscw", "write_bytes")
    }


def wrapped_fit(config, target, directory, interval):
    import autofit as af

    names = [f"x{i:03}" for i in range(config.dimension)]
    model = af.Collection(**{name: af.GaussianPrior(mean=0.0, sigma=1.0) for name in names})

    class Analysis(af.Analysis):
        def log_likelihood_function(self, instance):
            return target([getattr(instance, name) for name in names])

    # Parameter order checked rather than assumed from Collection/dict ordering.
    import numpy as np

    cube = np.linspace(0.2, 0.8, config.dimension)
    np.testing.assert_allclose(
        model.vector_from_unit_vector(cube), target.prior(cube), rtol=1e-13, atol=1e-13
    )
    instance = model.instance_from_vector(np.arange(config.dimension, dtype=float))
    np.testing.assert_array_equal(
        [getattr(instance, name) for name in names], np.arange(config.dimension)
    )
    constructor = {key: value for key, value in config.constructor().items() if key != "n_dim"}
    search = af.Nautilus(
        **constructor,
        **config.stopping(),
        number_of_cores=config.workers,
        checkpoint=config.checkpoint,
        checkpoint_interval_batches=interval,
        iterations_per_full_update=config.update_interval,
        path_prefix=str(directory / "fit"),
        name="baseline",
    )
    search.fit(model=model, analysis=Analysis(use_jax=False))


def performance(case, directory):
    import autofit as af
    from autofit.non_linear.search.nest.nautilus.checkpoint_policy import BatchedCheckpointSampler
    from nautilus import Sampler
    from profiling.nautilus_target import CorrelatedGaussianTarget

    owner = Sampler if case["interval"] == 1 else BatchedCheckpointSampler
    captured, writes = [], []
    metrics = {
        "sampler_run_s": 0.0,
        "checkpoint_s": 0.0,
        "full_writes": 0,
        "shell_notifications": 0,
        "io": dict.fromkeys(io_counts(), 0),
    }
    config = configuration(case)
    target = CorrelatedGaussianTarget(case["dimension"], "none", 0)
    original_init = Sampler.__init__

    @functools.wraps(original_init)
    def capture(obj, *args, **kwargs):
        expected = config.constructor()
        if any(kwargs.get(key) != value for key, value in expected.items()):
            raise ValueError("native constructor differs from frozen configuration")
        original_init(obj, *args, **kwargs)
        captured.append(obj)

    depth = 0

    def persistence(name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(obj, *args, **kwargs):
            nonlocal depth
            metrics["full_writes" if name == "write" else "shell_notifications"] += 1
            outer = depth == 0
            if outer:
                before, tick = io_counts(), time.perf_counter()
            depth += 1
            try:
                result = original(obj, *args, **kwargs)
                if name == "write":
                    writes.append({"n_like": obj.n_like, "bytes": Path(args[0]).stat().st_size})
                return result
            finally:
                depth -= 1
                if outer:
                    metrics["checkpoint_s"] += time.perf_counter() - tick
                    after = io_counts()
                    for key in before:
                        metrics["io"][key] += after[key] - before[key]

        return timed

    run = owner.run

    def timed_run(*args, **kwargs):
        tick = time.perf_counter()
        try:
            return run(*args, **kwargs)
        finally:
            metrics["sampler_run_s"] += time.perf_counter() - tick

    with contextlib.ExitStack() as stack:
        stack.enter_context(replaced(Sampler, "__init__", capture))
        stack.enter_context(replaced(af.Nautilus, "plot_results", lambda *a, **kw: None))
        stack.enter_context(replaced(owner, "run", timed_run))
        for name in ("write", "write_shell_update"):
            stack.enter_context(replaced(owner, name, persistence(name)))
        tick = time.perf_counter()
        wrapped_fit(config, target, directory, case["interval"])
        metrics["full_fit_s"] = time.perf_counter() - tick
    if len(captured) != 1 or type(captured[0]) is not owner:
        raise ValueError("wrong sampler/policy constructed")
    result = posterior(captured[0])
    if result["n_eff"] < 10000 or not captured[0].discard_exploration:
        raise ValueError("wrong stopping/exploration policy")
    return {
        **result,
        "metrics": metrics,
        "writes": writes,
        "target_sha256": fingerprint(target.truth()),
        "plot_policy": "no search corner in either arm; all other fit output retained",
    }


def child(plan, case, directory):
    from profiling.nautilus_baseline import provenance

    assert_environment(environment_key(provenance()), plan["environment"])
    validate(plan)
    if case["phase"] == "performance":
        result = performance(case, directory)
    elif case["phase"] == "interrupt":
        control = native(case, directory / "control.hdf5")
        atomic_json(directory / "control.json", control)
        native(case, directory / "checkpoint.hdf5", case["mode"], directory / "paused.json")
        raise RuntimeError("interruption hook was never reached")
    else:
        interrupted = directory.parent / (case["witness"] + "-interrupt")
        marker = json.loads((interrupted / "paused.json").read_text())
        checkpoint = interrupted / "checkpoint.hdf5"
        if sha256(checkpoint) != marker["checkpoint_sha256"]:
            raise ValueError("previous checkpoint changed during interrupted write")
        # Keep interrupted artifacts intact: resume a new copy.
        import shutil

        shutil.copyfile(checkpoint, directory / "checkpoint.hdf5")
        result = native(
            case, directory / "checkpoint.hdf5", expected_start=marker["recoverable_n_like"]
        )
        control = json.loads((interrupted / "control.json").read_text())
        if result != control:
            raise ValueError("resume differs from uninterrupted matching control")
        result = {**result, "recovery": marker, "control_equal": True}
    assert_environment(environment_key(provenance()), plan["environment"])
    return {"status": "success", **result}


def supervise_interruption(process, directory):
    start, peak, reason = time.monotonic(), 0, None
    try:
        while process.poll() is None:
            peak = max(peak, group_rss(process.pid))
            if peak > LIMITS["memory_mib"] * 1024**2:
                reason = "memory_cap"
                break
            if time.monotonic() - start >= LIMITS["timeout_s"]:
                reason = "timeout"
                break
            if (directory / "paused.json").exists():
                reason = "intentional_checkpoint_kill"
                break
            time.sleep(0.05)
    finally:
        terminate_group(process)
    return {
        "termination_reason": reason,
        "returncode": process.returncode,
        "sampled_group_peak_rss_bytes": peak,
        "rss_sample_interval_s": 0.05,
    }


def run(root):
    start = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "attempts": []}
    atomic_json(root / "state.json", state)
    try:
        from profiling.nautilus_baseline import provenance

        actual = provenance()
        if any(s["dirty"] for s in actual["sources"].values()):
            raise ValueError("measurement source must be committed and clean")
        plan = {
            "protocol": PROTOCOL,
            "limits": LIMITS,
            "cases": matrix(),
            "configurations": [configuration(c).matching() for c in matrix()],
            "environment": environment_key(actual),
        }
        plan["sha256"] = fingerprint(plan)
        validate(plan)
        atomic_json(root / "plan.json", plan)
        state["preflight_s"] = time.monotonic() - start
        for case in plan["cases"]:
            if time.monotonic() - start + LIMITS["timeout_s"] > LIMITS["budget_s"]:
                raise RuntimeError("aggregate budget exhausted")
            snapshot = host_snapshot()
            if memory_available() < 2 * LIMITS["memory_mib"] * 1024**2:
                raise RuntimeError("insufficient safe host memory")
            if snapshot["load"][0] > max(1.0, len(snapshot["affinity"]) / 4):
                raise RuntimeError("host load exceeds guard")
            directory = root / case["id"]
            directory.mkdir()
            row = {"case": case["id"], "status": "running", "host": snapshot}
            state["attempts"].append(row)
            atomic_json(root / "state.json", state)
            with (directory / "run.log").open("w") as output:
                tick = time.monotonic()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "child",
                        "--plan",
                        str(root / "plan.json"),
                        "--case",
                        case["id"],
                        "--output",
                        str(directory),
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                supervision = (
                    supervise_interruption(process, directory)
                    if case["phase"] == "interrupt"
                    else supervise(process, LIMITS["timeout_s"], LIMITS["memory_mib"] * 1024**2)
                )
            row.update(process_wall_s=time.monotonic() - tick, supervision=supervision)
            if case["phase"] == "interrupt":
                row["status"] = (
                    "success"
                    if supervision["termination_reason"] == "intentional_checkpoint_kill"
                    else "error"
                )
                if row["status"] == "success":
                    row["marker"] = json.loads((directory / "paused.json").read_text())
                    if sha256(directory / "checkpoint.hdf5") != row["marker"]["checkpoint_sha256"]:
                        raise ValueError("interrupted publication changed previous checkpoint")
            else:
                receipt = directory / "child.json"
                if receipt.exists():
                    row["child"] = json.loads(receipt.read_text())
                row["status"] = supervision["termination_reason"] or (
                    "success"
                    if supervision["returncode"] == 0
                    and row.get("child", {}).get("status") == "success"
                    else "error"
                )
            assert_environment(environment_key(provenance()), plan["environment"])
            atomic_json(directory / "result.json", row)
            if row["status"] != "success":
                raise RuntimeError(f"{case['id']}: {row['status']}")
            if case["phase"] == "performance" and len(state["attempts"]) % 2 == 0:
                a, b = [r["child"] for r in state["attempts"][-2:]]
                if any(a[k] != b[k] for k in ("posterior_sha256", "n_like", "n_eff")):
                    raise ValueError("paired sampling trajectories differ")
            state["spent_s"] = time.monotonic() - start
            atomic_json(root / "state.json", state)
            print(f"{case['id']}: {row['status']} ({row['process_wall_s']:.2f}s)", flush=True)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="halted", reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        state["spent_s"] = time.monotonic() - start
        done = {r["case"] for r in state["attempts"]}
        state["unrun"] = [c["id"] for c in matrix() if c["id"] not in done]
        atomic_json(root / "state.json", state)
    return 0 if state["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "child"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    if args.command == "run":
        return run(args.output.resolve())
    plan = json.loads(args.plan.read_text())
    case = next(c for c in plan["cases"] if c["id"] == args.case)
    try:
        result = child(plan, case, args.output.resolve())
    except Exception as error:
        traceback.print_exc()
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    atomic_json(args.output / "child.json", result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
