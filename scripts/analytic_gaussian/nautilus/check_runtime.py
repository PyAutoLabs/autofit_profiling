"""Numerical target, resume and real worker-death checks; requires the runtime stack."""

import contextlib
import hashlib
import json
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))

import numpy as np
from nautilus import Sampler
from nautilus.pool import initialize_worker
from profiling.nautilus_baseline import Monitor, run_direct, run_wrapped
from profiling.nautilus_manifest import RunConfig
from profiling.nautilus_target import CorrelatedGaussianTarget


def fingerprint(sampler):
    return hashlib.sha256(b"".join(x.tobytes() for x in sampler.posterior())).hexdigest()


def die(_):
    os.kill(os.getpid(), signal.SIGKILL)


def target_checks():
    points = np.array([[-1.0, 0.0, 1.0], [0.2, 0.3, 0.4]])
    reference = CorrelatedGaussianTarget(3, "none")
    rng_before = np.random.get_state()
    for cost in ("none", "fixed", "random", "correlated"):
        target = CorrelatedGaussianTarget(3, cost, 100)
        np.testing.assert_array_equal(
            [target(p) for p in points], [reference.value(p) for p in points]
        )
    rng_after = np.random.get_state()
    assert all(np.array_equal(x, y) for x, y in zip(rng_before, rng_after))
    # Independent scipy density check and conjugate identities.
    from scipy.stats import multivariate_normal

    np.testing.assert_allclose(
        reference.log_evidence,
        multivariate_normal.logpdf(reference.mean, cov=reference.covariance + np.eye(3)),
    )
    np.testing.assert_allclose(
        (np.eye(3) + reference.precision) @ reference.posterior_covariance, np.eye(3), atol=1e-14
    )


def resume_check(adapter, workers, directory):
    config = RunConfig(
        adapter=adapter,
        workers=workers,
        n_live=60,
        n_eff=100,
        dimension=3,
        work=0,
        trace=False,
        update_interval=200,
    )
    target = CorrelatedGaussianTarget(config.dimension, "none", 0)
    run = run_direct if adapter == "direct" else run_wrapped
    full = directory / "full"
    split = directory / "split"
    full.mkdir(parents=True)
    split.mkdir(parents=True)
    with Monitor(config) as monitor:
        run(config, target, full)
    expected = fingerprint(monitor.sampler)
    original = Sampler.run
    saved = {}

    def interrupted(sampler, *args, **kwargs):
        original(sampler, *args, **kwargs)
        saved["n_like"] = sampler.n_like
        saved["checkpoint"] = sampler.filepath
        raise InterruptedError("injected stop after completed native checkpoint")

    Sampler.run = interrupted
    try:
        try:
            run(config, target, split)
        except InterruptedError:
            pass
        else:
            raise AssertionError("interruption was not injected")
    finally:
        Sampler.run = original
    assert Path(saved["checkpoint"]).exists()
    with Monitor(config) as resumed:
        run(config, target, split)
    assert resumed.runs[0]["before"] == saved["n_like"], "did not resume native checkpoint"
    assert fingerprint(resumed.sampler) == expected, "resume changed posterior/log weights"
    return {
        "adapter": adapter,
        "workers": workers,
        "checkpoint_n_like": int(saved["n_like"]),
        "final_n_like": int(resumed.sampler.n_like),
        "posterior_sha256": expected,
    }


def worker_death_check(traced=False):
    from autofit.non_linear.search.nest.nautilus import search as module
    from profiling.nautilus_trace import Trace

    old = module.LIKELIHOOD_POOL_POLL_INTERVAL
    module.LIKELIHOOD_POOL_POLL_INTERVAL = 0.05
    start = time.monotonic()
    try:
        with Trace() if traced else contextlib.nullcontext():
            with module._LikelihoodWorkerPool(
                multiprocessing.get_context("fork").Pool(
                    2, initializer=initialize_worker, initargs=(die,)
                )
            ) as pool:
                try:
                    pool.map(die, [0, 1])
                except Exception as error:
                    assert "worker" in str(error).lower()
                else:
                    raise AssertionError("worker death was not detected")
    finally:
        module.LIKELIHOOD_POOL_POLL_INTERVAL = old
    assert time.monotonic() - start < 10
    return {"status": "detected", "wall_s": time.monotonic() - start}


def main():
    directory = Path(sys.argv[1]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    target_checks()
    results = {
        "target": "pass",
        "worker_death": {"control": worker_death_check(), "traced": worker_death_check(True)},
        "resume": [],
    }
    for adapter in ("direct", "pyautofit"):
        for workers in (1, 2):
            results["resume"].append(
                resume_check(adapter, workers, directory / f"{adapter}-{workers}")
            )
            (directory / "checks.json").write_text(json.dumps(results, indent=2))
            print(adapter, workers, "resume PASS", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
