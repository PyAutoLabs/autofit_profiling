"""Runtime implementation for the phase-0a CPU baseline (imports inside child only)."""

import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from profiling.nautilus_manifest import assert_matching, fingerprint


def git_identity(path):
    root = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True
    ).strip()

    def git(*args):
        return subprocess.check_output(["git", "-C", root, *args], text=True).strip()

    digest = hashlib.sha256()
    for name in sorted(
        set(git("ls-files", "--cached", "--others", "--exclude-standard").splitlines())
    ):
        if Path(name).suffix not in (".py", ".yaml", ".toml"):
            continue
        file = Path(root) / name
        digest.update(name.encode())
        digest.update(file.read_bytes() if file.is_file() else b"<missing>")
    return {
        "root": root,
        "revision": git("rev-parse", "HEAD"),
        "dirty": git("status", "--porcelain"),
        "code_sha256": digest.hexdigest(),
    }


def provenance():
    import autofit
    import autonerves
    import nautilus

    packages = (
        "numpy",
        "scipy",
        "scikit-learn",
        "h5py",
        "nautilus-sampler",
        "jax",
        "jaxlib",
        "threadpoolctl",
    )
    src = Path(nautilus.__file__).parent
    digest = hashlib.sha256()
    for p in sorted(src.rglob("*.py")):
        digest.update(str(p.relative_to(src)).encode())
        digest.update(p.read_bytes())
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "distribution_versions": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "sources": {
            "PyAutoFit": git_identity(Path(autofit.__file__).parent),
            "PyAutoNerves": git_identity(Path(autonerves.__file__).parent),
            "producer": git_identity(Path(__file__).parent),
        },
        "nautilus_source_sha256": digest.hexdigest(),
        "nautilus_source_sha": None,
        "nautilus_sha_note": "installed wheel source hashed; upstream git SHA unverified",
        "host": platform.node(),
        "cpu": platform.processor(),
        "cores": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


class Monitor:
    """Common coarse monitoring in both control and detailed-trace runs."""

    def __init__(self, config):
        self.config = config
        self.sampler = None
        self.runs = []
        self.writes = 0
        self.run_wall = 0.0
        self.originals = {}

    def __enter__(self):
        from nautilus import Sampler

        original_init = Sampler.__init__
        self.originals = {
            name: getattr(Sampler, name)
            for name in ("__init__", "run", "write", "write_shell_update")
        }

        def init(sampler, *args, **kwargs):
            bound = inspect.signature(original_init).bind(sampler, *args, **kwargs)
            bound.apply_defaults()
            self.constructor_arguments = {
                k: v for k, v in bound.arguments.items() if k in self.config.constructor()
            }
            assert_matching(self.config.constructor(), self.constructor_arguments)
            original_init(sampler, *args, **kwargs)
            self.sampler = sampler
            self.resolved = {
                key: getattr(sampler, key) for key in self.config.constructor() if key != "seed"
            }
            self.resolved["seed"] = self.constructor_arguments["seed"]
            assert_matching(self.config.constructor(), self.resolved)
            workers = 0 if sampler.pool_l is None else sampler.pool_l.size
            if (
                workers != (self.config.workers if self.config.workers > 1 else 0)
                or sampler.pool_s is not None
            ):
                raise ValueError("sampler/likelihood pool topology mismatch")
            if bool(sampler.filepath) != self.config.checkpoint:
                raise ValueError("checkpoint policy mismatch")

        def run(sampler, *args, **kwargs):
            bound = inspect.signature(self.originals["run"]).bind(sampler, *args, **kwargs)
            bound.apply_defaults()
            got = {key: bound.arguments[key] for key in self.config.stopping()}
            assert_matching(self.config.stopping(), got)
            self.runs.append(
                {"before": sampler.n_like, "n_like_max": bound.arguments["n_like_max"]}
            )
            t = time.perf_counter()
            try:
                done = self.originals["run"](sampler, *args, **kwargs)
                self.runs[-1].update(after=sampler.n_like, converged=bool(done))
                return done
            finally:
                self.run_wall += time.perf_counter() - t

        Sampler.__init__, Sampler.run = init, run
        for name in ("write", "write_shell_update"):

            def write(*args, _name=name, **kwargs):
                self.writes += 1
                return self.originals[_name](*args, **kwargs)

            setattr(Sampler, name, write)
        return self

    def __exit__(self, *args):
        from nautilus import Sampler

        for name, func in self.originals.items():
            setattr(Sampler, name, func)


def run_direct(config, target, directory):
    from autofit.non_linear.parallel import fork_context
    from autofit.non_linear.search.nest.nautilus.search import (
        _large_attribute_hdf5,
        _LikelihoodWorkerPool,
    )
    from nautilus import Sampler
    from nautilus.pool import initialize_worker

    pool_context = contextlib.nullcontext(None)
    if config.workers > 1:
        pool_context = _LikelihoodWorkerPool(
            fork_context().Pool(config.workers, initializer=initialize_worker, initargs=(target,))
        )
    with pool_context as pool, _large_attribute_hdf5():
        sampler = Sampler(
            prior=target.prior,
            likelihood=target,
            pool=(pool, None) if pool else None,
            filepath=str(directory / "checkpoint.hdf5") if config.checkpoint else None,
            **config.constructor(),
        )
        finished = False
        while not finished:
            finished = sampler.run(
                **config.stopping(), n_like_max=sampler.n_like + config.update_interval
            )
        sampler.posterior()  # materialization included in complete adapter wall
    return sampler


def run_wrapped(config, target, directory):
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
        iterations_per_full_update=config.update_interval,
        path_prefix=str(directory / "fit"),
        name="baseline",
    )
    search.fit(model=model, analysis=Analysis(use_jax=False))


def measure(config, directory):
    import numpy as np
    from threadpoolctl import threadpool_info

    from profiling.nautilus_target import CorrelatedGaussianTarget
    from profiling.nautilus_trace import Trace, summarize

    if sys.platform != "linux":
        raise RuntimeError("phase-0a fork baseline currently validated on Linux only")
    before = provenance()
    target = CorrelatedGaussianTarget(config.dimension, config.cost_profile, config.work)
    truth = target.truth()
    trace = Trace() if config.trace else None
    manifest = {
        "config": config.to_dict(),
        "matching": config.matching(),
        "configuration_sha256": fingerprint(config.matching()),
        "target": truth,
        "target_sha256": fingerprint(truth),
        "environment": before,
        "measured_at": datetime.now(UTC).isoformat(),
        "output_filesystem_path": str(directory),
        "instrumentation": "coarse monitor + detailed events"
        if trace
        else "coarse monitor only; no stage/worker events",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        with Monitor(config) as monitor:
            with trace if trace else contextlib.nullcontext():
                start = time.perf_counter()
                if config.adapter == "direct":
                    run_direct(config, target, directory)
                else:
                    run_wrapped(config, target, directory)
                wall = time.perf_counter() - start
    except BaseException:
        if trace:
            partial = {
                "unit": "monotonic_ns",
                "accounting": "incomplete",
                "events": trace.events,
                "batches": trace.batches,
                "submitted": sum(b["count"] for b in trace.batches),
                "collected": len(trace.events),
                "note": "Unreturned chunk events are unavailable; no retries or partial admission.",
            }
            (directory / "incomplete_trace.json").write_text(json.dumps(partial))
        raise
    sampler = monitor.sampler
    points, log_weights, log_l = sampler.posterior()
    weights = np.exp(log_weights)
    mean = np.sum(weights[:, None] * points, axis=0)
    covariance = (points - mean).T @ (weights[:, None] * (points - mean))
    after = provenance()
    if (
        before["sources"] != after["sources"]
        or before["nautilus_source_sha256"] != after["nautilus_source_sha256"]
    ):
        raise RuntimeError("source changed during measurement")
    row = {
        "status": "success",
        "manifest": manifest,
        "wall_s": wall,
        "sampler_run_s": monitor.run_wall,
        "resolved_constructor": monitor.resolved,
        "sampler_defaults": {
            "neural_network_kwargs": sampler.neural_network_kwargs,
            "pass_dict": sampler.pass_dict,
            "blobs_dtype": str(sampler.blobs_dtype),
        },
        "run_calls": monitor.runs,
        "checkpoint_writes": monitor.writes,
        "n_like": int(sampler.n_like),
        "n_eff": float(sampler.n_eff),
        "log_evidence": float(sampler.log_z),
        "log_evidence_error": float(sampler.log_z - target.log_evidence),
        "posterior_mean": mean.tolist(),
        "posterior_covariance": covariance.tolist(),
        "posterior_mean_error": (mean - target.posterior_mean).tolist(),
        "posterior_sha256": hashlib.sha256(
            points.tobytes() + log_weights.tobytes() + log_l.tobytes()
        ).hexdigest(),
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "children_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "threadpools": threadpool_info(),
        "components_s": {},
        "trace": None,
        "unavailable": [
            "logZ uncertainty: upstream 1.0.5 exposes no calibrated per-run error here",
            "proposal acceptance rate: requires proposal/rejection counter beyond likelihood batches",
        ],
    }
    if not monitor.runs or not monitor.runs[-1]["converged"]:
        raise RuntimeError("sampler did not reach stopping criteria")
    if trace:
        row["trace"] = summarize(trace.events, trace.batches)
        if row["trace"]["completed"] != sampler.n_like:
            raise ValueError("traced count does not equal native n_like")
        row["components_s"] = dict(trace.timer.totals)
        remainder = wall - sum(row["components_s"].values())
        if remainder < -1e-6:
            raise ValueError("parent exclusive timing exceeds adapter wall")
        row["components_s"]["other"] = remainder
        events = {
            "unit": "monotonic_ns",
            "events": trace.events,
            "batches": trace.batches,
            "stages": trace.stage_events,
        }
        import gzip

        payload = json.dumps(events, separators=(",", ":")).encode()
        path = directory / "timeline.json.gz"
        path.write_bytes(gzip.compress(payload, mtime=0))
        row["timeline"] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return row
