"""Check CPU/GPU likelihoods and tail masking before the timed fits."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from autofit.non_linear.objective import evaluate_in_chunks
from run import Profile, baseline, prepare


def probe():
    x, data, noise = baseline.make_dataset()
    rng = np.random.default_rng(4702)
    theta = rng.uniform([1, 0.1, 1], [99, 45, 49], size=(137, 3))
    theta[:21] = rng.normal([50.034, 25.063, 10.0], [0.04, 0.08, 0.04], size=(21, 3))
    expected = np.array(
        [
            -0.5 * np.sum(((data - baseline.Gaussian(*p).model_data_from(x)) / noise) ** 2)
            for p in theta
        ]
    )
    rows = []
    for repeats in (1, 64):
        analysis = Profile(repeats)
        for width in (50, 100):
            fun = jax.jit(
                jax.vmap(
                    lambda p, analysis=analysis: analysis.log_likelihood_function(
                        baseline.Gaussian(*p)
                    )
                )
            )
            for length in (1, 17, 50, 99, 100, 137):
                vals = np.asarray(evaluate_in_chunks(fun, theta[:length], width))
                np.testing.assert_allclose(vals, expected[:length], rtol=5e-12, atol=1e-8)
                host = []
                for start in range(0, length, width):
                    real = theta[start : min(start + width, length)]
                    padded, _ = prepare(np.asarray, real, width - len(real))
                    host.extend(np.asarray(fun(padded))[: len(real)])
                np.testing.assert_allclose(host, vals, rtol=5e-12, atol=1e-8)
            assert fun._cache_size() in (1, 2)  # NumPy and explicitly placed JAX inputs
            rows.append(
                dict(
                    repeats=repeats,
                    width=width,
                    max_abs_error=float(np.max(np.abs(vals - expected))),
                    lengths=[1, 17, 50, 99, 100, 137],
                )
            )
    return dict(
        device=jax.devices()[0].device_kind,
        backend=jax.default_backend(),
        fp64=jax.config.jax_enable_x64,
        checks=rows,
    )


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(probe(), indent=2) + "\n")
