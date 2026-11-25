"""One complete PyAutoFit search with a bounded, ordered host-to-device feeder."""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from autofit.non_linear.objective import chunk_slices
from nautilus import Sampler
from nautilus.bounds import NeuralBound

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gaussian_x1/nautilus"))
import search_fit_breakdown as baseline

ARMS = ("cpu", "gpu", "split", "feed")
clock = time.perf_counter


class Profile(baseline.af.Analysis):
    def __init__(self, repeats):
        super().__init__(use_jax=True)
        x, data, noise = baseline.make_dataset()
        self.x, self.data, self.noise = (jnp.asarray(np.tile(a, repeats)) for a in (x, data, noise))
        self.repeats = repeats

    def log_likelihood_function(self, instance):
        prediction = (
            instance.normalization
            / (instance.sigma * jnp.sqrt(2 * jnp.pi))
            * jnp.exp(-0.5 * ((self.x - instance.centre) / instance.sigma) ** 2)
        )
        return -0.5 * jnp.sum(((self.data - prediction) / self.noise) ** 2) / self.repeats


def prepare(prior, points, pad):
    start = clock()
    physical = prior(points.copy())
    if pad:
        physical = np.concatenate((physical, np.repeat(physical[-1:], pad, axis=0)))
    return physical, clock() - start


class Feed:
    def __init__(self, arm, pool):
        self.width = 50 if arm in ("split", "feed") else 100
        self.pool = pool
        self.points = []
        self.values = []
        self.times = []
        self.wait_s = self.collect_s = self.prepare_s = self.geometry_s = 0.0
        self.geometry_jobs = self.padded = self.chunks = 0
        self.cache_sizes = set()
        self.native = {}

    def evaluate(self, sampler, points):
        start = clock()
        if not sampler.vectorized or sampler.pool_s or sampler.pool_l or sampler.blobs_dtype:
            raise RuntimeError(
                "The pilot requires the native vectorized, single-owner scalar likelihood"
            )
        self.native = {
            key: getattr(sampler, key) for key in ("n_live", "n_update", "n_batch", "n_networks")
        }
        spans = chunk_slices(len(points), self.width)
        if not 1 <= len(spans) <= 2:
            raise RuntimeError("Unexpected logical batch size")
        jobs = []
        if self.pool is not None:
            jobs = [
                self.pool.submit(prepare, sampler.prior, points[lo:hi], pad)
                for lo, hi, pad in spans
            ]
        pending = []
        for i, (lo, hi, pad) in enumerate(spans):
            wait = clock()
            if jobs:
                physical, cost = jobs[i].result()
                self.wait_s += clock() - wait
            else:
                physical, cost = prepare(sampler.prior, points[lo:hi], pad)
            self.prepare_s += cost
            pending.append(sampler.likelihood(physical))
            self.padded += pad
        collect = clock()
        values = np.concatenate(
            [
                np.asarray(result)[: hi - lo]
                for result, (lo, hi, _) in zip(pending, spans, strict=True)
            ]
        )
        self.collect_s += clock() - collect
        self.cache_sizes.add(sampler.likelihood.func.__self__._call._cache_size())
        if len(values) != len(points) or not np.isfinite(values).all():
            raise ValueError("Missing or non-finite likelihood")
        sampler.n_like += len(values)
        self.points.append(points.copy())
        self.values.append(values.copy())
        self.chunks += len(spans)
        self.times.append(clock() - start)
        return values, None

    def membership(self, original, bound, points):
        if self.pool is None or len(points) < 1000:
            return original(bound, points)
        start = clock()
        middle = len(points) // 2
        jobs = [
            self.pool.submit(original, bound, part) for part in (points[:middle], points[middle:])
        ]
        mask = np.concatenate([job.result() for job in jobs])
        self.geometry_s += clock() - start
        self.geometry_jobs += len(jobs)
        return mask


def run(directory, arm, seed, repeats):
    device = jax.devices()[0]
    if device.platform != ("cpu" if arm == "cpu" else "gpu") or not jax.config.jax_enable_x64:
        raise RuntimeError(f"Unexpected backend or precision: {device}")
    baseline.af.conf.instance.push(new_path=ROOT / "config", output_path=directory / "output")
    model = baseline.make_model()
    analysis = Profile(repeats)
    search = baseline.af.Nautilus(
        name="gpu_feed", verbose=False, n_live=200, number_of_cores=1, seed=seed
    )
    timer = baseline.ExclusiveTimer()
    start_epoch = time.time()
    process_start = time.process_time()
    start = clock()
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2)) if arm == "feed" else None
        feed = Feed(arm, pool)

        def evaluate(sampler, points):
            return feed.evaluate(sampler, points)

        original = NeuralBound.contains

        def contains(bound, points):
            return feed.membership(original, bound, points)

        stack.enter_context(patch.object(Sampler, "evaluate_likelihood", evaluate))
        stack.enter_context(patch.object(NeuralBound, "contains", contains))
        stack.callback(baseline.instrument(timer))
        result = search.fit(model=model, analysis=analysis)
    wall = clock() - start
    cpu_s = time.process_time() - process_start
    end_epoch = time.time()
    samples = result.samples
    weight = np.asarray(samples.weight_list)
    weight /= weight.sum()
    parameters = np.asarray(samples.parameter_lists)
    mean = weight @ parameters
    points = np.concatenate(feed.points)
    values = np.concatenate(feed.values)
    if len(values) != samples.samples_info["total_samples"] or not feed.times:
        raise RuntimeError("Native accounting mismatch")
    np.savez(directory / "draws.npz", points=points, log_l=values)
    return dict(
        arm=arm,
        seed=seed,
        repeats=repeats,
        device=device.device_kind,
        fit_wall_s=wall,
        cpu_s=cpu_s,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        native=feed.native,
        calls=len(values),
        batches=len(feed.times),
        device_chunks=feed.chunks,
        width=feed.width,
        padding=feed.padded,
        objective_cache_sizes=sorted(feed.cache_sizes),
        first_batch_s=feed.times[0],
        warm_batch_median_s=float(np.median(feed.times[1:])),
        evaluation_wall_s=sum(feed.times),
        prepare_worker_s=feed.prepare_s,
        wait_for_prepare_s=feed.wait_s,
        collect_s=feed.collect_s,
        parallel_membership_s=feed.geometry_s,
        membership_jobs=feed.geometry_jobs,
        training_s=timer.totals["nn_training"],
        bounds=timer.counts["bound_geometry"],
        proposal_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        logz=float(samples.log_evidence),
        ess=float(1 / (weight @ weight)),
        mean=mean.tolist(),
        std=np.sqrt(weight @ (parameters - mean) ** 2).tolist(),
        names=list(model.parameter_names),
        affinity=sorted(os.sched_getaffinity(0)),
        jax=jax.__version__,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("seed", type=int)
    parser.add_argument("repeats", type=int, choices=(1, 64))
    opts = parser.parse_args()
    opts.directory.mkdir(parents=True, exist_ok=False)
    row = run(opts.directory.resolve(), opts.arm, opts.seed, opts.repeats)
    (opts.directory / "case.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
