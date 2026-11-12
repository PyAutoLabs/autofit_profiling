"""Passive parent/worker observations and six diagnostic-only interventions.

No scientific imports at module load. No numerical-library or GC settings change.
"""

import contextlib
import gc
import os
import pickle
import time
from collections import defaultdict
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

from profiling.nautilus_training_diagnostics import (
    DiagnosticTransport,
    PhaseTrace,
    array_identity,
    runtime_metadata,
)

ARMS = "ABCDEF"
TRAINING_CALLS = 11
COARSE = {
    "Sampler.evaluate_likelihood",
    "Sampler.sample_shell",
    "NautilusBound.compute",
    "NonLinearSearch.pre_fit_output",
    "NonLinearSearch.post_fit_output",
    "SearchUpdater._save_samples",
    "SearchUpdater._compute_latent_samples",
    "SearchUpdater._profile_and_summarize",
    "SearchUpdater.visualize",
}


def proc_stat(text):
    fields = text.rsplit(")", 1)[1].split()
    result = {
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "minor_faults": int(fields[7]),
        "major_faults": int(fields[9]),
        "start_ticks": int(fields[19]),
        "cpu": int(fields[36]),
    }
    if min(result.values()) < 0:
        raise ValueError("negative process counter")
    return result


def process_snapshot(pid):
    root = Path(f"/proc/{pid}")
    try:
        row = proc_stat((root / "stat").read_text())
        status = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines())
        row.update(
            pid=pid,
            affinity=sorted(os.sched_getaffinity(pid)),
            voluntary_switches=int(status["voluntary_ctxt_switches"]),
            involuntary_switches=int(status["nonvoluntary_ctxt_switches"]),
        )
        row["threads"] = {}
        for path in sorted((root / "task").iterdir()):
            try:
                row["threads"][path.name] = proc_stat((path / "stat").read_text())
            except (OSError, ValueError, IndexError) as error:
                row["threads"][path.name] = {"unavailable": str(error)}
        return row
    except (OSError, ValueError, IndexError, KeyError) as error:
        return {"pid": pid, "unavailable": f"{type(error).__name__}: {error}"}


def cgroup_snapshot():
    """Optional Linux v1/v2 counters, explicitly labelled; never writes configuration."""
    observations = {}
    try:
        groups = Path("/proc/self/cgroup").read_text().splitlines()
        for line in groups:
            _, controllers, relative = line.split(":", 2)
            if ".." in Path(relative).parts:
                continue
            mounts = (
                [Path("/sys/fs/cgroup")]
                if not controllers
                else [
                    Path("/sys/fs/cgroup") / controllers,
                    Path("/sys/fs/cgroup/cpu,cpuacct"),
                    Path("/sys/fs/cgroup/cpu"),
                ]
            )
            for mount in mounts:
                for name in ("cpu.stat", "cpu.max", "cpu.cfs_quota_us", "cpu.cfs_period_us"):
                    path = mount / relative.lstrip("/") / name
                    try:
                        observations[str(path)] = path.read_text()
                    except OSError:
                        pass
    except (OSError, ValueError) as error:
        observations["unavailable"] = str(error)
    return observations or {"unavailable": "CPU cgroup counters not readable"}


class ParentTrace(PhaseTrace):
    """Separate nested wall, process and main-thread CPU accounting."""

    def __init__(
        self, wall=time.monotonic_ns, cpu=time.process_time_ns, thread=time.thread_time_ns
    ):
        super().__init__(wall, cpu)
        self.thread = thread
        self.observer = None

    @contextlib.contextmanager
    def scope(self, phase):
        observe = self.observer if phase in COARSE else None
        if observe:
            observe.snapshot(phase + ":before")
        if len(self.events) >= 200000:
            raise ValueError("phase event cap exceeded")
        event = {
            "id": len(self.events),
            "phase": phase,
            "parent": self.stack[-1]["id"] if self.stack else None,
            "start_ns": self.wall(),
            "cpu_start_ns": self.cpu(),
            "thread_start_ns": self.thread(),
            "child_wall_ns": 0,
            "child_cpu_ns": 0,
            "child_thread_ns": 0,
        }
        self.events.append(event)
        self.stack.append(event)
        try:
            yield
        finally:
            event["thread_ns"] = self.thread() - event["thread_start_ns"]
            event["cpu_ns"] = self.cpu() - event["cpu_start_ns"]
            event["finish_ns"] = self.wall()
            durations = {
                "wall": event["finish_ns"] - event["start_ns"],
                "cpu": event["cpu_ns"],
                "thread": event["thread_ns"],
            }
            self.stack.pop()
            for metric, duration in durations.items():
                event[f"exclusive_{metric}_ns"] = duration - event.pop(f"child_{metric}_ns")
                if event[f"exclusive_{metric}_ns"] < 0:
                    raise ValueError("negative exclusive phase accounting")
                if self.stack:
                    self.stack[-1][f"child_{metric}_ns"] += duration
            if observe:
                observe.snapshot(phase + ":after")

    def receipt(self):
        receipt = super().receipt()
        totals = receipt["totals"]
        for row in totals.values():
            row["exclusive_thread_ns"] = 0
        for event in self.events:
            totals[event["phase"]]["exclusive_thread_ns"] += event["exclusive_thread_ns"]
        if sum(r["exclusive_thread_ns"] for r in totals.values()) != sum(
            e["thread_ns"] for e in self.events if e["parent"] is None
        ):
            raise ValueError("thread accounting does not reconcile")
        receipt["channels"] = ["wall", "process_cpu", "main_thread_cpu"]
        return receipt


class Observer:
    def __init__(self, trace):
        self.trace = trace
        self.transport = None
        self.samples, self.collections = [], []
        self.gc_error = None

    def on_gc(self, phase, info):
        # Do not inspect objects or collect here; keep callback allocation bounded.
        if len(self.collections) >= 20000:
            self.gc_error = "GC event cap exceeded"
            return
        self.collections.append(
            {
                "phase": phase,
                "info": dict(info),
                "wall_ns": time.monotonic_ns(),
                "cpu_ns": time.process_time_ns(),
                "thread_ns": time.thread_time_ns(),
            }
        )

    def __enter__(self):
        self.trace.observer = self
        gc.callbacks.append(self.on_gc)
        return self

    def __exit__(self, *exc):
        gc.callbacks.remove(self.on_gc)
        self.trace.observer = None

    def snapshot(self, label, runtime=False):
        with self.trace.scope("observation"):
            if len(self.samples) >= 5000:
                raise ValueError("observation cap exceeded")
            transport = self.transport
            row = {
                "label": label,
                "wall_ns": time.monotonic_ns(),
                "completed_training_calls": len(transport.calls) if transport else 0,
                "parent": process_snapshot(os.getpid()),
                "workers": [process_snapshot(r["pid"]) for r in transport.worker_info]
                if transport and not transport.closed
                else [],
                "workers_closed": transport.closed if transport else None,
            }
            if runtime:
                row.update(
                    runtime=runtime_metadata(),
                    cgroup=cgroup_snapshot(),
                    gc={
                        "enabled": gc.isenabled(),
                        "counts": gc.get_count(),
                        "thresholds": gc.get_threshold(),
                        "stats": gc.get_stats(),
                    },
                )
            self.samples.append(row)

    def receipt(self):
        if self.gc_error:
            raise ValueError(self.gc_error)
        return {
            "snapshots": self.samples,
            "gc_events": self.collections,
            "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
            "note": "Boundary observations only; missing counters explicit. Worker CPU and system/cgroup counters are separate from parent CPU. GC callbacks are passive and their overhead remains included.",
        }


def network_layout(network):
    return [
        {
            **array_identity(a),
            "strides": list(a.strides),
            "c_contiguous": bool(a.flags.c_contiguous),
            "f_contiguous": bool(a.flags.f_contiguous),
            "alignment_mod64": int(a.ctypes.data % 64),
        }
        for a in network.coefs_ + network.intercepts_
    ]


def pickle_roundtrip(network):
    before = network_layout(network)
    payload = ForkingPickler.dumps(network)  # Same default protocol as multiprocessing queues.
    restored = ForkingPickler.loads(payload)
    after = network_layout(restored)
    if [{k: r[k] for k in ("shape", "dtype", "sha256")} for r in before] != [
        {k: r[k] for k in ("shape", "dtype", "sha256")} for r in after
    ] or int(network.n_iter_) != int(restored.n_iter_):
        raise ValueError("serialized network values changed")
    return restored, {
        "before": before,
        "after": after,
        "bytes": len(payload),
        "effective_protocol": bytes(payload[:2]).hex(),
        "default_protocol": pickle.DEFAULT_PROTOCOL,
    }


class ParentTransport(DiagnosticTransport):
    def __init__(self, workers, arm, observer):
        if arm not in ARMS or workers != (2 if arm in "BCF" else 0):
            raise ValueError("unapproved parent CPU arm")
        super().__init__(workers)
        self.arm, self.observer = arm, observer
        self.allocated = self.allowed[:]
        if arm == "F":
            self.masks = [[cpu] for cpu in self.allocated[1:3]]
        self.closed = False
        self.close_count = 0
        self.saved_affinities = {}
        self.layouts = []
        observer.transport = self

    def __enter__(self):
        try:
            super().__enter__()
            if self.arm in "EF":
                # Include executor management threads, created before narrowing.
                for path in Path("/proc/self/task").iterdir():
                    tid = int(path.name)
                    self.saved_affinities[tid] = os.sched_getaffinity(tid)
                    os.sched_setaffinity(tid, [self.allocated[0]])
                if not self.workers:
                    self.allowed = [self.allocated[0]]
            self.observer.snapshot("pool_ready", runtime=True)
            return self
        except BaseException:
            self.__exit__()
            raise

    def close(self):
        if not self.closed:
            # Mark first: exception cleanup must not attempt a second shutdown.
            self.closed = True
            self.close_count += 1
            with self.observer.trace.scope("worker_shutdown"):
                super().__exit__()

    def __exit__(self, *exc):
        try:
            self.close()
        finally:
            for tid, mask in self.saved_affinities.items():
                try:
                    os.sched_setaffinity(tid, mask)
                except ProcessLookupError:
                    if tid == os.getpid():
                        raise
            self.saved_affinities.clear()

    def before_training(self):
        if self.closed or len(self.calls) >= TRAINING_CALLS:
            raise ValueError("unexpected training after final native call")
        self.observer.snapshot("before_training", runtime=True)

    def after_training(self):
        self.observer.snapshot("after_training", runtime=True)
        if self.arm == "C" and len(self.calls) == TRAINING_CALLS:
            self.close()
            self.observer.snapshot("after_early_shutdown", runtime=True)

    def map(self, function, seeds):
        if self.closed or len(self.calls) >= TRAINING_CALLS:
            raise ValueError("dispatch after final training or shutdown")
        networks = super().map(function, seeds)
        with self.observer.trace.scope("model_layout_serialization"):
            if self.arm == "D":
                results = [pickle_roundtrip(n) for n in networks]
                networks = [r[0] for r in results]
                self.layouts.append([r[1] for r in results])
            else:
                self.layouts.append([{"received": network_layout(n)} for n in networks])
        return networks

    def receipt(self):
        return {
            **super().receipt(),
            "arm": self.arm,
            "close_count": self.close_count,
            "closed": self.closed,
            "allocated_affinity": self.allocated,
            "worker_masks": self.masks,
            "layouts": self.layouts,
        }


def verify_phase_counts(receipt):
    expected = {
        "Sampler.evaluate_likelihood": 235,
        "Sampler.sample_shell": 235,
        "Sampler.write": 13,
        "Sampler.write_shell_update": 235,
        "NautilusBound.compute": 11,
        "NeuralNetworkEmulator.train": 11,
        "NeuralNetworkEmulator.predict": 4511,
    }
    for phase, count in expected.items():
        if receipt["totals"].get(phase, {}).get("calls") != count:
            raise ValueError(f"phase count differs: {phase}")
    # No new science may hide in output or an intervention.
    for phase in (
        "NonLinearSearch.pre_fit_output",
        "NonLinearSearch.post_fit_output",
        "SearchUpdater.visualize",
        "SearchUpdater._save_samples",
        "SearchUpdater._compute_latent_samples",
        "SearchUpdater._profile_and_summarize",
    ):
        if receipt["totals"].get(phase, {}).get("calls") != 1:
            raise ValueError(f"output phase count differs: {phase}")
