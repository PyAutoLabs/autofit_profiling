"""Experimental ordered training transport; imports no science at module load."""

import concurrent.futures
import hashlib
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path

from profiling.nautilus_resources import atomic_json


def initialize(ready, release):
    # Environment is set before this fresh interpreter imports scientific code.
    from nautilus.neural import train_network  # noqa: F401
    from threadpoolctl import threadpool_info

    info = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "start_method": multiprocessing.get_start_method(),
        "native_threads": [p["num_threads"] for p in threadpool_info()],
        "autofit_imported": "autofit" in sys.modules,
        "jax_imported": "jax" in sys.modules,
    }
    ready.put(info)
    if not release.wait(30):
        raise RuntimeError("parent did not release initialized training workers")


def ping():
    return os.getpid()


def run_network(function, seed, event_path=None):
    started = time.perf_counter()
    cpu = time.process_time()
    if event_path is not None:
        atomic_json(Path(event_path), {"pid": os.getpid(), "seed": seed})
    network = function(seed)
    result = {
        "seed": seed,
        "pid": os.getpid(),
        "wall_s": time.perf_counter() - started,
        "cpu_s": time.process_time() - cpu,
        "output_pickle_bytes": len(pickle.dumps(network)),
    }
    return network, result


def model_digest(model):
    """Exact learned parameters, normalization and per-network iteration count."""
    digest = hashlib.sha256()
    for array in [model.mean, model.scale] + [
        a for n in model.neural_networks for a in n.coefs_ + n.intercepts_
    ]:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    digest.update(str([int(n.n_iter_) for n in model.neural_networks]).encode())
    return digest.hexdigest()


class TrainingTransport:
    """Supply only NeuralNetworkEmulator.train's existing ordered map seam."""

    def __init__(self, workers, events=None, on_active=None):
        if workers not in (0, 2):
            raise ValueError("only serial or two spawned training workers are approved")
        self.workers = workers
        self.events = events
        self.on_active = on_active
        self.executor = None
        self.ready = None
        self.release = None
        self.calls = []
        self.worker_info = []
        self.models = []
        self.training_wall_s = 0.0
        self.startup_s = 0.0
        self.teardown_s = 0.0

    def __enter__(self):
        if not self.workers:
            return self
        started = time.perf_counter()
        context = multiprocessing.get_context("spawn")
        self.ready, self.release = context.Queue(), context.Event()
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=2,
            mp_context=context,
            initializer=initialize,
            initargs=(self.ready, self.release),
        )
        try:
            futures = [self.executor.submit(ping) for _ in range(2)]
            self.worker_info = [self.ready.get(timeout=30) for _ in range(2)]
            if len({r["pid"] for r in self.worker_info}) != 2:
                raise RuntimeError("two distinct workers did not initialize")
            for info in self.worker_info:
                if (
                    info["start_method"] != "spawn"
                    or info["pgid"] != os.getpgrp()
                    or info["autofit_imported"]
                    or info["jax_imported"]
                    or not info["native_threads"]
                    or any(n != 1 for n in info["native_threads"])
                ):
                    raise RuntimeError(f"unexpected training worker runtime: {info}")
            self.release.set()
            for future in futures:
                future.result(timeout=30)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        self.startup_s = time.perf_counter() - started
        return self

    def __exit__(self, *exc):
        started = time.perf_counter()
        if self.release is not None:
            self.release.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if self.ready is not None:
            self.ready.close()
            self.ready.join_thread()
        self.teardown_s = time.perf_counter() - started
        for info in self.worker_info:
            try:
                os.kill(info["pid"], 0)
            except ProcessLookupError:
                continue
            raise RuntimeError("training worker remains after executor shutdown")

    def map(self, function, seeds):
        from nautilus.neural import train_network

        seeds = list(seeds)
        if getattr(function, "func", None) is not train_network or seeds != list(range(4)):
            raise ValueError("training pool received non-native training or changed seeds")
        call = len(self.calls)
        payload_bytes = sum(len(pickle.dumps((function, seed))) for seed in seeds)
        if self.executor is None:
            results = [run_network(function, seed) for seed in seeds]
        else:
            paths = [
                str(self.events / f"train-{call}-{seed}.json") if self.events else None
                for seed in seeds
            ]
            futures = [
                self.executor.submit(run_network, function, seed, path)
                for seed, path in zip(seeds, paths)
            ]
            if self.on_active is not None:
                # The parent is in an unfinished training map. Record which
                # tasks reported entry and are not yet collected; workers may
                # advance between this observation and the supervisor's kill.
                while not any(f.done() for f in futures):
                    active = [
                        seed for seed, path in zip(seeds, paths) if path and Path(path).exists()
                    ]
                    if active:
                        self.on_active(active)
                        break
                    time.sleep(0.005)
            results = [future.result() for future in futures]
        if [r[1]["seed"] for r in results] != seeds:
            raise RuntimeError("network result order changed")
        self.calls.append(
            {
                "input_pickle_bytes": payload_bytes,
                "networks": [r[1] for r in results],
                "training_rows": int(function.args[0].shape[0]),
                "dimension": int(function.args[0].shape[1]),
            }
        )
        return [r[0] for r in results]

    def receipt(self):
        return {
            "workers": self.workers,
            "worker_info": self.worker_info,
            "startup_s": self.startup_s,
            "teardown_s": self.teardown_s,
            "training_wall_s": self.training_wall_s,
            "models": self.models,
            "calls": self.calls,
            "payload_note": "pickle size estimate for function/seed and returned model; excludes transport framing",
        }


class TrainingScope:
    def __init__(self, transport):
        self.transport = transport

    def __enter__(self):
        from nautilus.neural import NeuralNetworkEmulator

        self.owner = NeuralNetworkEmulator
        self.original = vars(self.owner)["train"]
        transport = self.transport
        original = self.original.__func__

        def train(cls, x, y, n_networks=4, neural_network_kwargs=None, pool=None):
            if n_networks != 4 or neural_network_kwargs or pool is not None:
                raise ValueError("changed neural settings or nonserial sampler pool")
            started = time.perf_counter()
            result = original(cls, x, y, n_networks=4, neural_network_kwargs={}, pool=transport)
            transport.training_wall_s += time.perf_counter() - started
            transport.models.append(model_digest(result))
            return result

        self.owner.train = classmethod(train)
        return self

    def __exit__(self, *exc):
        self.owner.train = self.original
