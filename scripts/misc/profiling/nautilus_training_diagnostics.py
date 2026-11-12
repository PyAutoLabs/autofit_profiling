"""Diagnostic-only capture, ordered training transport and exclusive phase timing.

Science imports are lazy so safety/protocol tests run in the tooling-only CI.
No installed source or original training-pool benchmark contract is changed.
"""

import concurrent.futures
import contextlib
import functools
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

from profiling.nautilus_diagnostics import sha256
from profiling.nautilus_manifest import fingerprint
from profiling.nautilus_resources import atomic_json
from profiling.nautilus_training_worker import TrainingTransport, model_digest, ping

CORPUS_SCHEMA = "nautilus-native-training-corpus@1"
CORPUS_BYTES = 64 * 1024**2
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def runtime_metadata():
    from threadpoolctl import threadpool_info

    libraries = threadpool_info()
    if not libraries or any(p["num_threads"] != 1 for p in libraries):
        raise ValueError("missing native library metadata or native thread count differs")
    if any(os.environ.get(k) != "1" for k in THREAD_ENV):
        raise ValueError("native thread environment differs")
    for library in libraries:
        if not all(k in library for k in ("filepath", "version", "internal_api", "num_threads")):
            raise ValueError("incomplete native library metadata")
    affinity = sorted(os.sched_getaffinity(0))
    frequency = {}
    for cpu in affinity:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        try:
            frequency[str(cpu)] = int(path.read_text())
        except (OSError, ValueError):
            frequency[str(cpu)] = None
    try:
        cpu_line = Path("/proc/stat").read_text().splitlines()[0].split()
        steal_ticks = int(cpu_line[8])
    except (OSError, ValueError, IndexError):
        steal_ticks = None
    return {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "affinity": affinity,
        "start_method": multiprocessing.get_start_method(),
        "libraries": libraries,
        "thread_environment": {k: os.environ[k] for k in THREAD_ENV},
        "frequency_khz_observation": frequency,
        "system_steal_ticks_observation": steal_ticks,
        "cache_counters": "unavailable: no privileged counters requested",
        "autofit_imported": "autofit" in sys.modules,
        "jax_imported": "jax" in sys.modules,
    }


def worker_initialize(ready, release, assignments):
    # Select a fixed mask before importing numerical libraries.
    affinity = assignments.get(timeout=30)
    if not set(affinity) <= set(os.sched_getaffinity(0)):
        raise ValueError("worker assignment escaped inherited allocation")
    os.sched_setaffinity(0, affinity)
    from nautilus.neural import train_network  # noqa: F401

    ready.put(runtime_metadata())
    if not release.wait(30):
        raise RuntimeError("parent did not release diagnostic worker")


def observed_cpu():
    try:
        return int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[36])
    except (OSError, ValueError, IndexError):
        return None


def network_task(function, seed, allowed):
    affinity = sorted(os.sched_getaffinity(0))
    if affinity not in allowed:
        raise ValueError("network affinity changed")
    start_cpu = observed_cpu()
    started = time.monotonic_ns()
    cpu = time.process_time_ns()
    network = function(seed)
    cpu_ns = time.process_time_ns() - cpu
    finished = time.monotonic_ns()
    return network, {
        "seed": seed,
        "pid": os.getpid(),
        "start_ns": started,
        "finish_ns": finished,
        "cpu_ns": cpu_ns,
        "affinity": affinity,
        "cpu_at_start": start_cpu,
        "cpu_at_finish": observed_cpu(),
        "output_pickle_bytes": len(pickle.dumps(network)),
    }


def assignments(workers, bound, allowed):
    if workers not in (0, 1, 2) or (bound and workers != 2):
        raise ValueError("unapproved diagnostic worker configuration")
    if len(set(allowed)) != 4 or sorted(allowed) != allowed:
        raise ValueError("four distinct sorted allocation CPUs required")
    return [[cpu] for cpu in allowed[:2]] if bound else [allowed[:] for _ in range(workers)]


def validate_workers(rows, masks):
    if len(rows) != len(masks) or len({r["pid"] for r in rows}) != len(masks):
        raise ValueError("missing or duplicate initialized workers")
    if sorted(r["affinity"] for r in rows) != sorted(masks):
        raise ValueError("initialized worker affinity differs")
    for row in rows:
        if (
            row["start_method"] != "spawn"
            or row["pgid"] != os.getpgrp()
            or row["autofit_imported"]
            or row["jax_imported"]
            or not row["libraries"]
            or any(p["num_threads"] != 1 for p in row["libraries"])
        ):
            raise ValueError("unexpected diagnostic worker runtime")


def occupancy(calls):
    """Per-worker busy time plus the interval union; overlapping work is not summed as elapsed."""
    workers = defaultdict(lambda: {"busy_ns": 0, "cpu_ns": 0, "tasks": 0})
    union, windows = 0, 0
    for call in calls:
        events = call["networks"]
        if [e["seed"] for e in events] != list(range(4)):
            raise ValueError("missing/reordered network events")
        windows += call["collect_finish_ns"] - call["dispatch_start_ns"]
        by_pid = defaultdict(list)
        for e in events:
            if not (
                call["dispatch_start_ns"]
                <= e["submit_ns"]
                <= e["start_ns"]
                <= e["finish_ns"]
                <= e["collect_ns"]
                <= call["collect_finish_ns"]
            ):
                raise ValueError("invalid network event timeline")
            if e["cpu_ns"] < 0:
                raise ValueError("negative network CPU time")
            by_pid[e["pid"]].append(e)
            row = workers[str(e["pid"])]
            row["busy_ns"] += e["finish_ns"] - e["start_ns"]
            row["cpu_ns"] += e["cpu_ns"]
            row["tasks"] += 1
        for rows in by_pid.values():
            ordered = sorted(rows, key=lambda e: e["start_ns"])
            if any(a["finish_ns"] > b["start_ns"] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("overlapping tasks on the same worker")
        end = 0
        for event in sorted(events, key=lambda e: e["start_ns"]):
            if event["finish_ns"] > end:
                union += event["finish_ns"] - max(end, event["start_ns"])
            end = max(end, event["finish_ns"])
    if union < 0 or union > windows:
        raise ValueError("invalid interval-union accounting")
    return {
        "workers": dict(workers),
        "any_worker_busy_ns": union,
        "dispatch_collect_windows_ns": windows,
        "no_worker_busy_ns": windows - union,
    }


class DiagnosticTransport(TrainingTransport):
    def __init__(self, workers, bound=False):
        super().__init__(0)  # Preserve the original transport's public 0/2 contract.
        self.workers = workers
        self.allowed = sorted(os.sched_getaffinity(0))
        self.masks = assignments(workers, bound, self.allowed)
        self.bound = bound
        self.assign_queue = None

    def __enter__(self):
        from nautilus.neural import train_network  # noqa: F401

        self.parent_info = runtime_metadata()
        if not self.workers:
            return self
        started = time.monotonic()
        context = multiprocessing.get_context("spawn")
        self.ready, self.release, self.assign_queue = (
            context.Queue(),
            context.Event(),
            context.Queue(),
        )
        for mask in self.masks:
            self.assign_queue.put(mask)
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=worker_initialize,
            initargs=(self.ready, self.release, self.assign_queue),
        )
        try:
            futures = [self.executor.submit(ping) for _ in self.masks]
            self.worker_info = [self.ready.get(timeout=30) for _ in self.masks]
            validate_workers(self.worker_info, self.masks)
            self.release.set()
            for future in futures:
                future.result(timeout=30)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        self.startup_s = time.monotonic() - started
        return self

    def __exit__(self, *exc):
        try:
            super().__exit__(*exc)
        finally:
            if self.assign_queue is not None:
                self.assign_queue.close()
                self.assign_queue.join_thread()

    def map(self, function, seeds):
        from nautilus.neural import train_network

        seeds = list(seeds)
        if getattr(function, "func", None) is not train_network or seeds != list(range(4)):
            raise ValueError("changed native training function or seeds")
        payload = sum(len(pickle.dumps((function, seed))) for seed in seeds)
        start = time.monotonic_ns()
        submissions, results = [], []
        allowed = self.masks if self.workers else [self.allowed]
        if self.executor is None:
            for seed in seeds:
                submitted = time.monotonic_ns()
                network, event = network_task(function, seed, allowed)
                event.update(submit_ns=submitted, collect_ns=time.monotonic_ns())
                results.append((network, event))
            dispatched = time.monotonic_ns()
        else:
            for seed in seeds:
                submitted = time.monotonic_ns()
                submissions.append(
                    (submitted, self.executor.submit(network_task, function, seed, allowed))
                )
            dispatched = time.monotonic_ns()
            for submitted, future in submissions:
                network, event = future.result()
                event.update(submit_ns=submitted, collect_ns=time.monotonic_ns())
                results.append((network, event))
        self.calls.append(
            {
                "input_pickle_bytes": payload,
                "networks": [r[1] for r in results],
                "dispatch_start_ns": start,
                "dispatch_finish_ns": dispatched,
                "collect_finish_ns": time.monotonic_ns(),
                "training_rows": int(function.args[0].shape[0]),
                "dimension": int(function.args[0].shape[1]),
            }
        )
        return [r[0] for r in results]

    def receipt(self):
        return {
            **super().receipt(),
            "bound": self.bound,
            "parent_info": self.parent_info,
            "occupancy": occupancy(self.calls),
        }


def array_identity(array):
    if array.dtype.hasobject or array.ndim not in (1, 2):
        raise ValueError("object/invalid training array")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
        "sha256": hashlib.sha256(array.tobytes(order="A")).hexdigest(),
    }


class CorpusWriter:
    def __init__(self, root):
        self.root, self.rows, self.size = root, [], 0
        root.mkdir(exist_ok=False)

    def before(self, x, y):
        import numpy as np

        index = len(self.rows)
        row = {
            "index": index,
            "settings": {"n_networks": 4, "neural_network_kwargs": {}},
            "arrays": {},
        }
        for name, array in (("x", x), ("y", y)):
            identity = array_identity(array)
            if not (array.flags.c_contiguous or array.flags.f_contiguous):
                raise ValueError("noncontiguous corpus array cannot preserve layout")
            path = self.root / f"{index:03}-{name}.npy"
            if self.size + array.nbytes + 512 > CORPUS_BYTES:
                raise ValueError("corpus exceeds64MiB")
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            self.size += path.stat().st_size
            row["arrays"][name] = {**identity, "file": path.name, "file_sha256": sha256(path)}
        self.rows.append(row)
        return row

    def after(self, row, x, y, digest):
        for name, array in (("x", x), ("y", y)):
            if array_identity(array) != {
                k: v for k, v in row["arrays"][name].items() if not k.startswith("file")
            }:
                raise ValueError("native training mutated capture inputs")
        row["model_sha256"] = digest

    def finish(self):
        if not self.rows or any("model_sha256" not in r for r in self.rows):
            raise ValueError("empty/incomplete corpus")
        manifest = {"schema": CORPUS_SCHEMA, "calls": self.rows}
        manifest["sha256"] = fingerprint(manifest)
        atomic_json(self.root / "manifest.json", manifest)
        validate_corpus(self.root)
        return manifest["sha256"]


def validate_corpus(root):
    import numpy as np

    files = list(root.iterdir())
    if (
        any(p.is_symlink() or not p.is_file() for p in files)
        or sum(p.stat().st_size for p in files) > CORPUS_BYTES
    ):
        raise ValueError("invalid/oversize corpus files")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("sha256") != fingerprint({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("corpus manifest checksum differs")
    if manifest["schema"] != CORPUS_SCHEMA or not manifest["calls"]:
        raise ValueError("empty/wrong corpus schema")
    expected_files, loaded = {"manifest.json"}, []
    for index, row in enumerate(manifest["calls"]):
        if row["index"] != index or row["settings"] != {
            "n_networks": 4,
            "neural_network_kwargs": {},
        }:
            raise ValueError("corpus call order/settings differ")
        arrays = {}
        for name in ("x", "y"):
            entry = row["arrays"][name]
            filename = f"{index:03}-{name}.npy"
            if entry["file"] != filename or sha256(root / filename) != entry["file_sha256"]:
                raise ValueError("corpus file checksum/name differs")
            expected_files.add(filename)
            array = np.load(root / filename, allow_pickle=False)
            if array_identity(array) != {
                k: v for k, v in entry.items() if not k.startswith("file")
            }:
                raise ValueError("corpus array identity differs")
            arrays[name] = array
        if (
            arrays["x"].ndim != 2
            or arrays["x"].shape[1] != 10
            or arrays["y"].shape != (len(arrays["x"]),)
        ):
            raise ValueError("corpus training dimensions differ")
        if len(row["model_sha256"]) != 64:
            raise ValueError("missing corpus model digest")
        loaded.append((row, arrays["x"], arrays["y"]))
    if {p.name for p in files} != expected_files:
        raise ValueError("unexpected corpus files")
    return manifest, loaded


class TrainingScope:
    def __init__(self, transport, corpus=None, timer=None, before=None, after=None):
        self.transport, self.corpus, self.timer = transport, corpus, timer
        self.before, self.after = before, after

    def scope(self, name):
        return self.timer.scope(name) if self.timer else contextlib.nullcontext()

    def __enter__(self):
        from nautilus.neural import NeuralNetworkEmulator

        self.owner = NeuralNetworkEmulator
        self.original = vars(self.owner)["train"]
        original = self.original.__func__

        def train(cls, x, y, n_networks=4, neural_network_kwargs=None, pool=None):
            if n_networks != 4 or neural_network_kwargs or pool is not None:
                raise ValueError("changed neural settings or sampler pool")
            if self.before:
                self.before()
            row = self.corpus.before(x, y) if self.corpus else None
            tick = time.monotonic()
            # PhaseTrace wraps the native method below this boundary: digest/corpus
            # overhead is separate from neural_train in traced full fits.
            model = original(cls, x, y, n_networks=4, neural_network_kwargs={}, pool=self.transport)
            self.transport.training_wall_s += time.monotonic() - tick
            with self.scope("model_validation"):
                digest = model_digest(model)
                self.transport.models.append(digest)
                if self.corpus:
                    self.corpus.after(row, x, y, digest)
            if self.after:
                self.after()
            return model

        self.owner.train = classmethod(train)
        return self

    def __exit__(self, *exc):
        self.owner.train = self.original


class PhaseTrace:
    def __init__(self, wall=time.monotonic_ns, cpu=time.process_time_ns):
        self.wall, self.cpu = wall, cpu
        self.stack, self.events, self.patches = [], [], []

    @contextlib.contextmanager
    def scope(self, phase):
        if len(self.events) >= 200000:
            raise ValueError("phase event cap exceeded")
        event = {
            "id": len(self.events),
            "phase": phase,
            "parent": self.stack[-1]["id"] if self.stack else None,
            "start_ns": self.wall(),
            "cpu_start_ns": self.cpu(),
            "child_wall_ns": 0,
            "child_cpu_ns": 0,
        }
        self.events.append(event)
        self.stack.append(event)
        try:
            yield
        finally:
            event["cpu_ns"] = self.cpu() - event["cpu_start_ns"]
            event["finish_ns"] = self.wall()
            wall = event["finish_ns"] - event["start_ns"]
            event["exclusive_wall_ns"] = wall - event.pop("child_wall_ns")
            event["exclusive_cpu_ns"] = event["cpu_ns"] - event.pop("child_cpu_ns")
            self.stack.pop()
            if min(event["exclusive_wall_ns"], event["exclusive_cpu_ns"]) < 0:
                raise ValueError("negative exclusive phase accounting")
            if self.stack:
                self.stack[-1]["child_wall_ns"] += wall
                self.stack[-1]["child_cpu_ns"] += event["cpu_ns"]

    def patch(self, owner, name, phase):
        original = vars(owner)[name]
        descriptor = type(original) if isinstance(original, (classmethod, staticmethod)) else None
        function = original.__func__ if descriptor else original

        @functools.wraps(function)
        def call(*args, **kwargs):
            with self.scope(phase):
                return function(*args, **kwargs)

        setattr(owner, name, descriptor(call) if descriptor else call)
        self.patches.append((owner, name, original))

    def __enter__(self):
        from autofit.non_linear.search.abstract_search import NonLinearSearch
        from autofit.non_linear.search.updater import SearchUpdater
        from nautilus import Sampler
        from nautilus.bounds import NautilusBound
        from nautilus.neural import NeuralNetworkEmulator

        try:
            for owner, names in (
                (
                    Sampler,
                    ("run", "evaluate_likelihood", "sample_shell", "write", "write_shell_update"),
                ),
                (NautilusBound, ("compute",)),
                (NeuralNetworkEmulator, ("train", "predict")),
                (NonLinearSearch, ("pre_fit_output", "post_fit_output")),
                (
                    SearchUpdater,
                    (
                        "visualize",
                        "_save_samples",
                        "_compute_latent_samples",
                        "_profile_and_summarize",
                    ),
                ),
            ):
                for name in names:
                    self.patch(owner, name, f"{owner.__name__}.{name}")
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def receipt(self):
        if self.stack or not self.events:
            raise ValueError("incomplete phase trace")
        totals = defaultdict(lambda: {"calls": 0, "exclusive_wall_ns": 0, "exclusive_cpu_ns": 0})
        for event in self.events:
            row = totals[event["phase"]]
            row["calls"] += 1
            for key in ("exclusive_wall_ns", "exclusive_cpu_ns"):
                row[key] += event[key]
        roots = [e for e in self.events if e["parent"] is None]
        for metric, duration in (
            ("wall", lambda e: e["finish_ns"] - e["start_ns"]),
            ("cpu", lambda e: e["cpu_ns"]),
        ):
            if sum(r[f"exclusive_{metric}_ns"] for r in totals.values()) != sum(
                duration(e) for e in roots
            ):
                raise ValueError("phase accounting does not reconcile")
        return {
            "totals": dict(totals),
            "events": self.events,
            "accounting": "exact",
            "residual_definition": "exclusive Sampler.run/full_fit/workflow time is unassigned work including instrumentation overhead; worker CPU is separate",
        }


@contextlib.contextmanager
def forbid_sampling():
    from nautilus import Sampler

    from profiling.nautilus_plot_comparison import replaced
    from profiling.nautilus_target import CorrelatedGaussianTarget

    calls = {"sampler_run": 0, "evaluate_likelihood": 0, "target": 0}

    def forbidden(name):
        def call(*args, **kwargs):
            calls[name] += 1
            raise RuntimeError(f"training replay forbids {name}")

        return call

    with contextlib.ExitStack() as stack:
        for owner, name, key in (
            (Sampler, "run", "sampler_run"),
            (Sampler, "evaluate_likelihood", "evaluate_likelihood"),
            (CorrelatedGaussianTarget, "__call__", "target"),
        ):
            stack.enter_context(replaced(owner, name, forbidden(key)))
        yield calls
    if any(calls.values()):
        raise ValueError("forbidden replay calls occurred")
