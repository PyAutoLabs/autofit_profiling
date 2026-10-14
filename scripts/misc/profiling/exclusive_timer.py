"""Exclusive (self) wall-time accounting for monkey-patched methods (stdlib only).

A breakdown cell wraps a handful of library methods. Some of them nest — a Nautilus
bound update (``NautilusBound.compute``) trains the neural-network emulators
(``NeuralNetworkEmulator.train``) inside it, and a PyAutoFit output update may evaluate
likelihoods. Timing each wrapper inclusively would count the inner time twice, so the
timer keeps a call stack and charges every interval to the innermost open component
only. The components' exclusive totals therefore never sum past the wall time, and
``other`` (wall minus the instrumented total) is never negative by construction.
"""

from __future__ import annotations

import functools
import time


class ExclusiveTimer:
    """Charge wall time to the innermost active component."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._stack: list[list] = []  # [component, start, child_time]

    def reset(self):
        self.totals.clear()
        self.counts.clear()
        self._stack.clear()

    def enter(self, component: str):
        self._stack.append([component, self.clock(), 0.0])

    def exit(self):
        component, start, child = self._stack.pop()
        inclusive = self.clock() - start
        self.totals[component] = self.totals.get(component, 0.0) + inclusive - child
        self.counts[component] = self.counts.get(component, 0) + 1
        if self._stack:
            self._stack[-1][2] += inclusive

    def wrap(self, component: str, func):
        """Return ``func`` wrapped so its exclusive time is charged to ``component``."""

        @functools.wraps(func)
        def timed(*args, **kwargs):
            self.enter(component)
            try:
                return func(*args, **kwargs)
            finally:
                self.exit()

        return timed


def shares(totals: dict[str, float], wall: float, order) -> dict[str, float]:
    """Exclusive seconds per component in ``order`` plus ``other`` = wall − instrumented."""
    seconds = {name: float(totals.get(name, 0.0)) for name in order}
    seconds["other"] = max(wall - sum(seconds.values()), 0.0)
    return seconds


def within_tolerance(measured: dict[str, float], reference: dict[str, float], tol: float):
    """``(ok, deltas)``: each reference share reproduced within ``tol`` (absolute, 0–1)."""
    deltas = {k: measured.get(k, 0.0) - v for k, v in reference.items()}
    return all(abs(d) <= tol for d in deltas.values()), deltas
