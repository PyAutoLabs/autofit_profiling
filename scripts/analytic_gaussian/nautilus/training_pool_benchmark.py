"""One approved, bounded off-Pulse trial of two spawned training workers."""

import argparse
import contextlib
import functools
import hashlib
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
from profiling.nautilus_resources import atomic_json, group_rss, memory_available, terminate_group
from profiling.nautilus_training_worker import TrainingScope, TrainingTransport, model_digest

PROTOCOL = "nautilus-spawn-training-experiment@1"
LIMITS = {"children": 18, "timeout_s": 120, "memory_mib": 4096, "budget_s": 1800}
SOURCE_PINS = {
    "PyAutoFit": "bdb7af0e9c339ad07850679fd6ac168aacc7d18c",
    "PyAutoNerves": "9b204c90c2ce05c5bb0cf3e9800ed13f005c6c56",
}
NAUTILUS_SOURCE = "61e91f8fb85cf4a38877cd1c65bb0b111589bde77f5f74acb22a90f3ff2ec45c"


def validate_sources(identity):
    if any(
        identity["sources"][name]["revision"] != revision for name, revision in SOURCE_PINS.items()
    ):
        raise ValueError("approved read-only library source pins differ")
    if identity["nautilus_source_sha256"] != NAUTILUS_SOURCE:
        raise ValueError("approved native Nautilus source differs")


def matrix():
    rows = [
        {"id": "preflight", "phase": "preflight", "training_workers": 2},
        {"id": "control", "phase": "control", "training_workers": 0},
    ]
    for mode in ("training", "checkpoint"):
        for phase in ("interrupt", "resume"):
            rows.append(
                {
                    "id": f"{mode}-{phase}",
                    "phase": phase,
                    "mode": mode,
                    "training_workers": 2,
                }
            )
    for dimension in (5, 10):
        for seed in range(3):
            for workers in (0, 2) if seed % 2 == 0 else (2, 0):
                rows.append(
                    {
                        "id": f"d{dimension}-s{seed}-w{workers}",
                        "phase": "performance",
                        "dimension": dimension,
                        "seed": seed,
                        "training_workers": workers,
                    }
                )
    return [{"dimension": 5, "seed": 0, **row} for row in rows]


class Config(RunConfig):
    def stopping(self):
        return {**super().stopping(), "discard_exploration": True}


def configuration(case):
    return Config(
        adapter="pyautofit",
        dimension=case["dimension"],
        seed=case["seed"],
        workers=1,
        trace=False,
        cost_profile="none",
        work=0,
        n_live=400,
        n_batch=100,
        n_eff=10000,
    )


def validate(plan):
    if plan["sha256"] != fingerprint({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("frozen protocol changed")
    if plan["protocol"] != PROTOCOL or plan["cases"] != matrix() or plan["limits"] != LIMITS:
        raise ValueError("unapproved protocol, matrix or limits")
    if plan["configurations"] != [configuration(c).matching() for c in matrix()]:
        raise ValueError("sampling configuration changed")
    validate_quiet(plan["quiet_admission"])


def validate_quiet(readings):
    if len(readings) != 3:
        raise ValueError("three quiet readings required")
    for i, row in enumerate(readings):
        if row["load"][0] >= 1.0 or row["available_mib"] < 8192:
            raise ValueError("host was not quiet or lacked memory")
        if i and row["time"] - readings[i - 1]["time"] < 30:
            raise ValueError("quiet readings must be at least30s apart")


def posterior(sampler):
    import numpy as np

    digest = hashlib.sha256()
    for array in sampler.posterior():
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "posterior_sha256": digest.hexdigest(),
        "n_like": int(sampler.n_like),
        "n_eff": float(sampler.n_eff),
        "shells": len(sampler.bounds),
        "explored": bool(sampler.explored),
        "discard_exploration": bool(sampler.discard_exploration),
    }


def exact_result(left, right):
    for key in ("posterior_sha256", "n_like", "n_eff", "shells", "explored", "discard_exploration"):
        if left[key] != right[key]:
            raise ValueError(f"sampling parity differs: {key}")


def preflight(directory):
    import numpy as np
    from nautilus.neural import NeuralNetworkEmulator

    rng = np.random.default_rng(49)
    x = rng.normal(size=(64, 5))
    y = x[:, 0] + 0.2 * x[:, 1]
    query = rng.normal(size=(19, 5))
    results = []
    original = vars(NeuralNetworkEmulator)["train"]
    for workers in (0, 2):
        with TrainingTransport(workers) as transport, TrainingScope(transport):
            model = NeuralNetworkEmulator.train(x, y)
        assert vars(NeuralNetworkEmulator)["train"] is original
        results.append(
            {
                "model_sha256": model_digest(model),
                "prediction_sha256": hashlib.sha256(model.predict(query).tobytes()).hexdigest(),
                "transport": transport.receipt(),
            }
        )
    if any(results[0][key] != results[1][key] for key in ("model_sha256", "prediction_sha256")):
        raise ValueError("native training coefficients or predictions differ")
    return {
        "training_equal": True,
        "corpus_sha256": hashlib.sha256(x.tobytes() + y.tobytes() + query.tobytes()).hexdigest(),
        "results": results,
    }


def native(case, directory, expected_start=None):
    import h5py
    from autofit.non_linear.search.nest.nautilus.search import _large_attribute_hdf5
    from nautilus import Sampler
    from profiling.nautilus_target import CorrelatedGaussianTarget

    config = configuration(case)
    target = CorrelatedGaussianTarget(case["dimension"], "none", 0)
    path = directory / "checkpoint.hdf5"
    workflow = time.perf_counter()
    events = directory / "worker-events"
    events.mkdir()
    transport = TrainingTransport(case["training_workers"], events=events)
    with (
        transport,
        TrainingScope(transport),
        _large_attribute_hdf5(),
        contextlib.ExitStack() as stack,
    ):
        sampler = Sampler(target.prior, target, pool=None, filepath=path, **config.constructor())
        if expected_start is not None and int(sampler.n_like) != expected_start:
            raise ValueError("resume did not restore saved likelihood count")

        def pause(active=()):
            if not path.exists():
                return
            with h5py.File(path, "r") as saved:
                recoverable = int(saved["sampler"].attrs["n_like"])
            atomic_json(
                directory / "paused.json",
                {
                    "mode": case["mode"],
                    "checkpoint_sha256": sha256(path),
                    "recoverable_n_like": recoverable,
                    "completed_n_like": int(sampler.n_like),
                    "lost_completed_evaluations": int(sampler.n_like) - recoverable,
                    "likelihood_in_flight": 0,
                    "training_map_in_flight": bool(active),
                    "training_tasks_observed_started": list(active),
                    "worker_pids": [r["pid"] for r in transport.worker_info],
                    "training_note": "entry observed before parent pause; workers can advance before group kill",
                },
            )
            os.kill(os.getpid(), signal.SIGSTOP)
            raise RuntimeError("supervisor failed to terminate paused parent")

        if case["phase"] == "interrupt":
            if case["mode"] == "training":
                transport.on_active = pause
            else:
                original = Sampler.write

                def write(obj, filepath, *args, **kwargs):
                    result = original(obj, filepath, *args, **kwargs)
                    if obj is sampler and Path(filepath) == path:
                        pause()
                    return result

                stack.enter_context(replaced(Sampler, "write", write))
        tick = time.perf_counter()
        if not sampler.run(**config.stopping()):
            raise RuntimeError("native search did not converge")
        run_s = time.perf_counter() - tick
        if case["phase"] == "interrupt":
            raise RuntimeError("interruption hook was not reached")
        result = posterior(sampler)
    return {
        **result,
        "sampler_run_s": run_s,
        "workflow_s": time.perf_counter() - workflow,
        "transport": transport.receipt(),
        "checkpoint_sha256": sha256(path),
    }


def wrapped_fit(config, directory):
    import autofit as af
    import numpy as np
    from profiling.nautilus_target import CorrelatedGaussianTarget

    target = CorrelatedGaussianTarget(config.dimension, "none", 0)
    names = [f"x{i:03}" for i in range(config.dimension)]
    model = af.Collection(**{name: af.GaussianPrior(mean=0.0, sigma=1.0) for name in names})

    class Analysis(af.Analysis):
        def log_likelihood_function(self, instance):
            return target([getattr(instance, name) for name in names])

    cube = np.linspace(0.2, 0.8, config.dimension)
    np.testing.assert_allclose(model.vector_from_unit_vector(cube), target.prior(cube), rtol=1e-13)
    instance = model.instance_from_vector(np.arange(config.dimension, dtype=float))
    np.testing.assert_array_equal(
        [getattr(instance, name) for name in names], np.arange(config.dimension)
    )
    search = af.Nautilus(
        **{k: v for k, v in config.constructor().items() if k != "n_dim"},
        **config.stopping(),
        number_of_cores=1,
        checkpoint=True,
        iterations_per_full_update=config.update_interval,
        path_prefix=str(directory / "fit"),
        name="training",
    )
    search.fit(model=model, analysis=Analysis(use_jax=False))
    return fingerprint(target.truth())


def retain_checkpoint(source, destination):
    """Keep the native inode after Fit's normal finalization deletes its pathname."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("retained checkpoint must have a separate path")
    pending = destination.with_name(destination.name + ".pending")
    os.link(source, pending)
    os.replace(pending, destination)


def performance(case, directory):
    import autofit as af
    from nautilus import Sampler

    config = configuration(case)
    objects = []
    timings = {"sampler_run_s": 0.0, "checkpoint_retention_s": 0.0}
    retained = directory / "final-checkpoint.hdf5"
    original_init, original_run = Sampler.__init__, Sampler.run

    @functools.wraps(original_init)
    def capture(obj, *args, **kwargs):
        if any(kwargs.get(k) != v for k, v in config.constructor().items()):
            raise ValueError("native constructor configuration differs")
        original_init(obj, *args, **kwargs)
        if obj.pool_l is not None or obj.pool_s is not None:
            raise ValueError("likelihood/geometry pool unexpectedly enabled")
        if obj.neural_network_kwargs:
            raise ValueError("native neural defaults were changed")
        objects.append(obj)

    def timed_run(*args, **kwargs):
        if any(kwargs.get(k) != v for k, v in config.stopping().items()):
            raise ValueError("native stopping configuration differs")
        tick = time.perf_counter()
        try:
            value = original_run(*args, **kwargs)
        finally:
            timings["sampler_run_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        retain_checkpoint(args[0].filepath, retained)
        timings["checkpoint_retention_s"] += time.perf_counter() - tick
        return value

    workflow = time.perf_counter()
    with TrainingTransport(case["training_workers"]) as transport, TrainingScope(transport):
        with contextlib.ExitStack() as stack:
            stack.enter_context(replaced(Sampler, "__init__", capture))
            stack.enter_context(replaced(Sampler, "run", timed_run))
            stack.enter_context(replaced(af.Nautilus, "plot_results", lambda *args, **kw: None))
            tick = time.perf_counter()
            truth = wrapped_fit(config, directory)
            timings["full_fit_s"] = time.perf_counter() - tick
    timings["workflow_s"] = time.perf_counter() - workflow
    if len(objects) != 1 or type(objects[0]) is not Sampler:
        raise ValueError("expected one unchanged native Sampler")
    sampler = objects[0]
    result = posterior(sampler)
    if result["n_eff"] < 10000 or not result["discard_exploration"]:
        raise ValueError("stopping/exploration policy differs")
    # Reopen the saved native state without running or evaluating a likelihood.
    saved_hash = sha256(retained)

    def forbidden(*args):
        raise RuntimeError("checkpoint read must not evaluate likelihood")

    from autofit.non_linear.search.nest.nautilus.search import _large_attribute_hdf5

    with _large_attribute_hdf5():
        reopened = Sampler(lambda x: x, forbidden, filepath=str(retained), **config.constructor())
        exact_result(posterior(reopened), result)
    if sha256(retained) != saved_hash:
        raise ValueError("read-only checkpoint reopen changed bytes")
    return {
        **result,
        "metrics": timings,
        "transport": transport.receipt(),
        "target_sha256": truth,
        "checkpoint_sha256": saved_hash,
        "checkpoint_reopen_equal": True,
        "plot_policy": "search corner disabled equally; all other fit output retained",
    }


def child(plan, case, directory):
    from importlib.metadata import version

    from profiling.nautilus_baseline import provenance

    validate(plan)
    assert_environment(environment_key(provenance()), plan["environment"])
    if version("nautilus-sampler") != "1.0.5":
        raise ValueError("only the pinned native training schema is supported")
    if case["phase"] == "preflight":
        result = preflight(directory)
    elif case["phase"] == "performance":
        result = performance(case, directory)
    elif case["phase"] == "resume":
        import shutil

        old = directory.parent / (case["mode"] + "-interrupt")
        marker = json.loads((old / "paused.json").read_text())
        if sha256(old / "checkpoint.hdf5") != marker["checkpoint_sha256"]:
            raise ValueError("interrupted checkpoint changed")
        shutil.copyfile(old / "checkpoint.hdf5", directory / "checkpoint.hdf5")
        result = native(case, directory, marker["recoverable_n_like"])
        control = json.loads((directory.parent / "control/child.json").read_text())
        exact_result(result, control)
        result.update(control_equal=True, recovery=marker)
    else:
        result = native(case, directory)
    assert_environment(environment_key(provenance()), plan["environment"])
    return {"status": "success", **result}


def live_group(pgid, proc=Path("/proc")):
    alive = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                alive.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return alive


def supervise(process, directory, intentional):
    started, peak, reason = time.monotonic(), 0, None
    try:
        while process.poll() is None:
            peak = max(peak, group_rss(process.pid))
            if peak > LIMITS["memory_mib"] * 1024**2:
                reason = "memory_cap"
                break
            if time.monotonic() - started >= LIMITS["timeout_s"]:
                reason = "timeout"
                break
            if intentional and (directory / "paused.json").exists():
                reason = "intentional_checkpoint_kill"
                break
            time.sleep(0.05)
    finally:
        terminate_group(process)
    # Allow the OS to finish termination, never re-run a child or a task.
    deadline = time.monotonic() + 2
    while live_group(process.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    leaked = live_group(process.pid)
    return {
        "returncode": process.returncode,
        "termination_reason": "leaked_descendants" if leaked else reason,
        "live_descendants": leaked,
        "sampled_group_peak_rss_bytes": peak,
        "rss_sample_interval_s": 0.05,
    }


def admit(started):
    tick = time.monotonic()
    while True:
        host = host_snapshot()
        host["available_mib"] = memory_available() / 1024**2
        wait = time.monotonic() - tick
        if time.monotonic() - started + LIMITS["timeout_s"] > LIMITS["budget_s"]:
            return False, "aggregate budget exhausted", host, wait
        if host["available_mib"] >= 8192 and host["load"][0] <= max(1, len(host["affinity"]) / 4):
            return True, None, host, wait
        if wait >= 60:
            return False, "host admission unavailable after60s", host, wait
        time.sleep(min(1, 60 - wait))


def run(root, quiet_path):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "attempts": []}
    atomic_json(root / "state.json", state)
    try:
        from profiling.nautilus_baseline import provenance

        quiet = [json.loads(line) for line in quiet_path.read_text().splitlines()][-3:]
        validate_quiet(quiet)
        if not 0 <= time.time() - quiet[-1]["time"] <= 120:
            raise ValueError("quiet admission is not recent")
        identity = provenance()
        validate_sources(identity)
        if any(s["dirty"] for s in identity["sources"].values()):
            raise ValueError("measurement sources must be clean and committed")
        plan = {
            "protocol": PROTOCOL,
            "cases": matrix(),
            "configurations": [configuration(c).matching() for c in matrix()],
            "limits": LIMITS,
            "quiet_admission": quiet,
            "environment": environment_key(identity),
        }
        plan["sha256"] = fingerprint(plan)
        validate(plan)
        atomic_json(root / "plan.json", plan)
        state["preflight_s"] = time.monotonic() - started
        for case in plan["cases"]:
            ok, reason, host, wait = admit(started)
            state["last_admission"] = {"host": host, "wait_s": wait, "reason": reason}
            if not ok:
                raise RuntimeError(reason)
            directory = root / case["id"]
            directory.mkdir()
            row = {"case": case["id"], "status": "running", "host": host, "admission_wait_s": wait}
            state["attempts"].append(row)
            atomic_json(root / "state.json", state)
            with (directory / "run.log").open("w") as log:
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
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                receipt = supervise(process, directory, case["phase"] == "interrupt")
            row.update(process_wall_s=time.monotonic() - tick, supervision=receipt)
            if case["phase"] == "interrupt":
                row["status"] = (
                    "success"
                    if receipt["termination_reason"] == "intentional_checkpoint_kill"
                    else "error"
                )
                if row["status"] == "success":
                    row["marker"] = json.loads((directory / "paused.json").read_text())
                    if sha256(directory / "checkpoint.hdf5") != row["marker"]["checkpoint_sha256"]:
                        raise ValueError("interrupted checkpoint changed bytes")
            else:
                path = directory / "child.json"
                if path.exists():
                    row["child"] = json.loads(path.read_text())
                row["status"] = receipt["termination_reason"] or (
                    "success"
                    if receipt["returncode"] == 0
                    and row.get("child", {}).get("status") == "success"
                    else "error"
                )
            assert_environment(environment_key(provenance()), plan["environment"])
            atomic_json(directory / "result.json", row)
            if row["status"] != "success":
                raise RuntimeError(f"{case['id']}: {row['status']}")
            if case["phase"] == "performance" and len(state["attempts"]) % 2 == 0:
                a, b = [r["child"] for r in state["attempts"][-2:]]
                exact_result(a, b)
                if (
                    a["target_sha256"] != b["target_sha256"]
                    or a["transport"]["models"] != b["transport"]["models"]
                ):
                    raise ValueError("paired target or trained models differ")
            state["spent_s"] = time.monotonic() - started
            atomic_json(root / "state.json", state)
            print(f"{case['id']}: {row['status']} ({row['process_wall_s']:.2f}s)", flush=True)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="halted", reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        state["spent_s"] = time.monotonic() - started
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
    parser.add_argument("--quiet-log", type=Path)
    args = parser.parse_args()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    if args.command == "run":
        return run(args.output.resolve(), args.quiet_log)
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
