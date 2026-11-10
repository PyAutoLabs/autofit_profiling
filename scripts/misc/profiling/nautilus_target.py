"""Analytic Gaussian with standard-normal priors and reproducible real CPU work."""

import hashlib
import math

import numpy as np
from scipy.special import ndtri


class CorrelatedGaussianTarget:
    def __init__(self, dimension=5, cost_profile="fixed", work=100):
        self.dimension = dimension
        self.cost_profile = cost_profile
        self.work = work
        self.mean = np.linspace(-0.5, 0.5, dimension)
        self.covariance = 0.6 ** np.abs(
            np.subtract.outer(np.arange(dimension), np.arange(dimension))
        )
        self.precision = np.linalg.solve(self.covariance, np.eye(dimension))
        self.normalization = -0.5 * (
            dimension * math.log(2 * math.pi) + np.linalg.slogdet(self.covariance)[1]
        )
        total = self.covariance + np.eye(dimension)
        self.log_evidence = float(
            -0.5
            * (
                dimension * math.log(2 * math.pi)
                + np.linalg.slogdet(total)[1]
                + self.mean @ np.linalg.solve(total, self.mean)
            )
        )
        self.posterior_covariance = np.linalg.solve(
            np.eye(dimension) + self.precision, np.eye(dimension)
        )
        self.posterior_mean = self.posterior_covariance @ self.precision @ self.mean
        self.work_sink = 0.0

    @staticmethod
    def prior(cube):
        return ndtri(np.asarray(cube))

    def value(self, parameters):
        delta = np.asarray(parameters) - self.mean
        return float(self.normalization - 0.5 * delta @ self.precision @ delta)

    def __call__(self, parameters):
        # Stateless independent workload randomness, keyed by the point, so control
        # and traced execution do identical work even if scheduling changes. Duplicate
        # points deliberately get identical cost. No sampler/training RNG is consumed.
        if self.cost_profile == "random":
            key = hashlib.blake2b(
                np.asarray(parameters, dtype="<f8").tobytes(), digest_size=8
            ).digest()
            multiplier = 1 + int.from_bytes(key, "little") % 16
        elif self.cost_profile == "correlated":
            multiplier = 16 if parameters[0] > 0.5 else 1
        else:
            multiplier = 1
        count = 0 if self.cost_profile == "none" else self.work * multiplier
        value = 0.5
        for i in range(count):
            value = math.sin(value + i * 0.001)
        self.work_sink = value
        return self.value(parameters)

    def truth(self):
        return {
            "log_evidence": self.log_evidence,
            "mean": self.posterior_mean.tolist(),
            "covariance": self.posterior_covariance.tolist(),
            "prior": "independent N(0,1)",
            "likelihood_mean": self.mean.tolist(),
            "likelihood_covariance": self.covariance.tolist(),
        }
