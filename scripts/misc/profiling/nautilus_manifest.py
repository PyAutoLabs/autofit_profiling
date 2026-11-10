"""Dependency-free run configuration and matching checks."""

import hashlib
import json
from dataclasses import asdict, dataclass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RunConfig:
    adapter: str = "direct"
    workers: int = 1
    seed: int = 0
    dimension: int = 5
    cost_profile: str = "fixed"
    work: int = 100
    trace: bool = True
    n_live: int = 200
    n_eff: int = 500
    n_batch: int = 100
    checkpoint: bool = True
    update_interval: int = 1000000

    def __post_init__(self):
        if self.adapter not in ("direct", "pyautofit") or self.workers not in (1, 2):
            raise ValueError("expected direct/pyautofit and 1/2 workers")
        if self.dimension < 2 or self.n_live <= self.dimension or min(self.n_eff, self.n_batch) < 1:
            raise ValueError("invalid dimension/live points/stopping configuration")
        if self.cost_profile not in ("none", "fixed", "random", "correlated") or self.work < 0:
            raise ValueError("invalid CPU workload")
        if self.update_interval < 3 * self.n_live:
            raise ValueError("update interval below PyAutoFit's effective floor")

    def constructor(self):
        return {
            "n_dim": self.dimension,
            "n_live": self.n_live,
            "n_update": self.n_live,
            "n_points_min": self.dimension + 50,
            "n_like_new_bound": 10 * self.n_live,
            "enlarge_per_dim": 1.1,
            "split_threshold": 100,
            "n_networks": 4,
            "n_batch": self.n_batch,
            "vectorized": False,
            "seed": self.seed,
        }

    def stopping(self):
        return {
            "f_live": 0.01,
            "n_shell": 1,
            "n_eff": self.n_eff,
            "discard_exploration": False,
            "verbose": False,
        }

    def matching(self):
        return {
            "constructor": self.constructor(),
            "stopping": self.stopping(),
            "pool": {
                "likelihood_workers": self.workers if self.workers > 1 else 0,
                "sampler_workers": 0,
                "start_method": "fork" if self.workers > 1 else None,
            },
            "checkpoint": self.checkpoint,
            "update_interval": self.update_interval,
            "cost_profile": self.cost_profile,
            "work": self.work,
        }

    def to_dict(self):
        return asdict(self)


def assert_matching(left, right):
    if left != right:
        differences = [key for key in left.keys() | right.keys() if left.get(key) != right.get(key)]
        raise ValueError(f"unmatched baseline settings: {differences}")


def source_pins(sources):
    """Compare code identity, not where an equivalent read-only checkout lives."""
    return {
        name: {key: source[key] for key in ("revision", "code_sha256")}
        for name, source in sources.items()
    }
