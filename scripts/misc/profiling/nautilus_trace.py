"""Harness-only timing; preserve native map batching and worker-death watchdog.

Worker events ride back with the existing chunk result; no logging IPC channel or
per-task Fitness pickling is introduced. This extra payload is deliberately included
in the measured tracing overhead. This module's accounting is stdlib-only.
"""

import functools
import os
import resource
import time
from collections import defaultdict

from profiling.exclusive_timer import ExclusiveTimer


def _timed_worker(job):
    from nautilus.pool import likelihood_worker

    ident, submitted, parameters = job
    start = time.monotonic_ns()
    cpu = time.process_time_ns()
    value = likelihood_worker(parameters)
    return value, {
        "id": ident,
        "pid": os.getpid(),
        "submit_ns": submitted,
        "start_ns": start,
        "finish_ns": time.monotonic_ns(),
        "cpu_ns": time.process_time_ns() - cpu,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


class _Result:
    def __init__(self, result, trace):
        self.result, self.trace = result, trace

    def get(self, timeout=None):
        rows = self.result.get(timeout=timeout)
        collected = time.monotonic_ns()
        for _, event in rows:
            event["collect_ns"] = collected
            self.trace.events.append(event)
        return [value for value, _ in rows]


class _PoolProxy:
    def __init__(self, pool, trace):
        self.pool, self.trace = pool, trace

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def map_async(self, func, iterable):
        values = list(iterable)
        submitted = time.monotonic_ns()
        jobs = [(self.trace.next_id(), submitted, value) for value in values]
        return _Result(self.pool.map_async(_timed_worker, jobs), self.trace)


def reconcile(events, batches):
    """Reject malformed, missing and duplicate events; don't hide negative timing."""
    seen = set()
    by_batch = defaultdict(list)
    for event in events:
        ident = tuple(event["id"])
        if ident in seen:
            raise ValueError(f"duplicate proposal {ident}")
        seen.add(ident)
        if not (
            event["submit_ns"] <= event["start_ns"] <= event["finish_ns"] <= event["collect_ns"]
        ):
            raise ValueError(f"invalid timeline {ident}")
        if event["cpu_ns"] < 0:
            raise ValueError("negative CPU duration")
        by_batch[ident[0]].append(event)
    by_worker = defaultdict(list)
    for event in events:
        by_worker[event["pid"]].append(event)
    for rows in by_worker.values():
        ordered = sorted(rows, key=lambda event: event["start_ns"])
        if any(left["finish_ns"] > right["start_ns"] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("overlapping evaluations on one worker")
    expected = {(batch["id"], i) for batch in batches for i in range(batch["count"])}
    if seen != expected:
        raise ValueError(
            f"proposal mismatch: missing={len(expected - seen)}, unexpected={len(seen - expected)}"
        )
    for batch in batches:
        rows = by_batch[batch["id"]]
        if rows and not (
            batch["start_ns"]
            <= min(e["submit_ns"] for e in rows)
            <= max(e["collect_ns"] for e in rows)
            <= batch["finish_ns"]
        ):
            raise ValueError("events outside batch")
    return by_batch


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(events, batches):
    by_batch = reconcile(events, batches)
    latency = [(e["finish_ns"] - e["start_ns"]) * 1e-9 for e in events]
    queue = [(e["start_ns"] - e["submit_ns"]) * 1e-9 for e in events]
    workers = defaultdict(list)
    for event in events:
        workers[event["pid"]].append(event)
    spans = []
    for batch in batches:
        rows = by_batch[batch["id"]]
        last_per_worker = defaultdict(int)
        for e in rows:
            last_per_worker[e["pid"]] = max(last_per_worker[e["pid"]], e["finish_ns"])
        finishes = list(last_per_worker.values())
        spans.append(
            {
                "id": batch["id"],
                "count": len(rows),
                "wall_s": (batch["finish_ns"] - batch["start_ns"]) * 1e-9,
                "worker_tail_s": (max(finishes) - min(finishes)) * 1e-9
                if len(finishes) > 1
                else None,
            }
        )
    evaluation_window = sum(b["finish_ns"] - b["start_ns"] for b in batches) * 1e-9
    occupancy = {}
    for pid, rows in workers.items():
        service = sum(e["finish_ns"] - e["start_ns"] for e in rows) * 1e-9
        occupancy[str(pid)] = {
            "calls": len(rows),
            "service_s": service,
            "cpu_s": sum(e["cpu_ns"] for e in rows) * 1e-9,
            "idle_in_evaluation_windows_s": evaluation_window - service,
            "occupancy_in_evaluation_windows": service / evaluation_window
            if evaluation_window
            else None,
            "peak_rss_kib": max(e["peak_rss_kib"] for e in rows),
        }
    return {
        "submitted": len(events),
        "completed": len(events),
        "accounting": "complete",
        "latency_s": {f"p{int(p * 100)}": percentile(latency, p) for p in (0.5, 0.95, 0.99)},
        "queue_wait_s": {f"p{int(p * 100)}": percentile(queue, p) for p in (0.5, 0.95, 0.99)},
        "workers": occupancy,
        "batches": spans,
        "worker_tail_definition": "last completion minus first worker's last completion in each batch; one worker => unavailable",
        "unavailable": [
            "serialization isolated from dispatch",
            "individual chunk dequeue timestamps",
            "worker initializer completion (worker_startup times pool construction)",
        ],
    }


class Trace:
    def __init__(self):
        self.timer = ExclusiveTimer()
        self.events, self.batches, self.patches = [], [], []
        self.batch_id, self.row = -1, 0
        self.parent_pid = os.getpid()
        self.stage_events = []

    def next_id(self):
        if len(self.events) + self.row > 200000:
            raise RuntimeError("detailed trace limit exceeded; use coarser instrumentation")
        ident = [self.batch_id, self.row]
        self.row += 1
        return ident

    def patch(self, owner, name, wrapper):
        original = owner.__dict__[name]
        self.patches.append((owner, name, original))
        setattr(owner, name, wrapper(original))

    def stage(self, component):
        def decorate(original):
            is_class = isinstance(original, classmethod)
            func = original.__func__ if is_class else original
            timed = self.timer.wrap(component, func)

            @functools.wraps(func)
            def call(*args, **kwargs):
                start = time.monotonic_ns()
                try:
                    return timed(*args, **kwargs)
                finally:
                    self.stage_events.append(
                        {"stage": component, "start_ns": start, "finish_ns": time.monotonic_ns()}
                    )

            return classmethod(call) if is_class else call

        return decorate

    def __enter__(self):
        from multiprocessing.pool import Pool

        from autofit.non_linear.search.abstract_search import NonLinearSearch
        from autofit.non_linear.search.nest.nautilus.search import _LikelihoodWorkerPool
        from autofit.non_linear.search.updater import SearchUpdater
        from nautilus import Sampler
        from nautilus.bounds import NautilusBound
        from nautilus.neural import NeuralNetworkEmulator

        def pool_map(original):
            @functools.wraps(original)
            def call(pool, func, iterable):
                real = pool._pool
                pool._pool = _PoolProxy(real, self)
                try:
                    return original(pool, func, iterable)
                finally:
                    pool._pool = real

            return call

        self.patch(_LikelihoodWorkerPool, "map", pool_map)

        def evaluate(original):
            @functools.wraps(original)
            def call(sampler, points):
                self.batch_id += 1
                self.row = 0
                batch = {"id": self.batch_id, "count": len(points), "start_ns": time.monotonic_ns()}
                likelihood, prior = sampler.likelihood, sampler.prior
                sampler.prior = self.timer.wrap("prior_transform", prior)
                if sampler.pool_l is None:

                    def scalar(parameters):
                        submitted = time.monotonic_ns()
                        ident = self.next_id()
                        start = time.monotonic_ns()
                        cpu = time.process_time_ns()
                        value = likelihood(parameters)
                        finish = time.monotonic_ns()
                        self.events.append(
                            {
                                "id": ident,
                                "pid": os.getpid(),
                                "submit_ns": submitted,
                                "start_ns": start,
                                "finish_ns": finish,
                                "collect_ns": time.monotonic_ns(),
                                "cpu_ns": time.process_time_ns() - cpu,
                                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            }
                        )
                        return value

                    sampler.likelihood = scalar
                try:
                    return original(sampler, points)
                finally:
                    sampler.likelihood, sampler.prior = likelihood, prior
                    batch["finish_ns"] = time.monotonic_ns()
                    self.batches.append(batch)

            return call

        self.patch(Sampler, "evaluate_likelihood", evaluate)
        for owner, name, component in (
            (Pool, "__init__", "worker_startup"),
            (Sampler, "run", "sampler_bookkeeping"),
            (Sampler, "evaluate_likelihood", "evaluation_dispatch_wait"),
            (Sampler, "sample_shell", "proposal_sampling"),
            (Sampler, "write", "checkpoint_io"),
            (Sampler, "write_shell_update", "checkpoint_io"),
            (NautilusBound, "compute", "bound_geometry"),
            (NeuralNetworkEmulator, "train", "nn_training"),
            (NeuralNetworkEmulator, "predict", "nn_prediction"),
            (SearchUpdater, "visualize", "visualisation"),
            (SearchUpdater, "_save_samples", "samples_output"),
            (SearchUpdater, "_compute_latent_samples", "samples_output"),
            (SearchUpdater, "_profile_and_summarize", "samples_output"),
            (NonLinearSearch, "pre_fit_output", "samples_output"),
            (NonLinearSearch, "post_fit_output", "samples_output"),
        ):
            self.patch(owner, name, self.stage(component))
        return self

    def __exit__(self, *args):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()
