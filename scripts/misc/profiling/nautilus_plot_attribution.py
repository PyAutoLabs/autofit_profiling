"""Scoped plotting attribution, preserving numerical and rendering behavior."""

import contextlib
import functools
import json
import time
from pathlib import Path

from profiling.nautilus_plot_comparison import replaced
from profiling.nautilus_resources import atomic_json

LIMITS = {"budget_s": 180, "timeout_s": 120, "memory_mib": 4096}


class Timers:
    """Nested completed timings; durable active frames survive process termination."""

    def __init__(self, path, clock=time.monotonic):
        self.path, self.clock = Path(path), clock
        self.start = clock()
        self.frames, self.totals = [], {}
        self.logging_s, self.writes = 0.0, 0

    def snapshot(self):
        tick = self.clock()
        atomic_json(
            self.path,
            {
                "schema": "nautilus-plot-attribution@1",
                "elapsed_s": tick - self.start,
                "completed": self.totals,
                "active": [
                    {"name": f["name"], "entered_s": f["start"] - self.start} for f in self.frames
                ],
                "logging_s": self.logging_s,
                "writes": self.writes,
                "note": "Nested inclusive times overlap. Self time excludes timed children. Both exclude snapshot I/O. Active frames and unfinished-panel counters are incomplete; last write's own I/O is not in its receipt.",
            },
        )
        self.logging_s += self.clock() - tick
        self.writes += 1

    @contextlib.contextmanager
    def measure(self, name, durable=False):
        frame = {"name": name, "start": self.clock(), "io": self.logging_s, "children": 0.0}
        key = "/".join([f["name"] for f in self.frames] + [name])
        self.frames.append(frame)
        if durable:
            self.snapshot()
        error = False
        try:
            yield
        except BaseException:
            error = True
            raise
        finally:
            elapsed = self.clock() - frame["start"] - (self.logging_s - frame["io"])
            self.frames.pop()
            row = self.totals.setdefault(
                key, {"calls": 0, "errors": 0, "inclusive_s": 0.0, "self_s": 0.0}
            )
            row["calls"] += 1
            row["errors"] += int(error)
            row["inclusive_s"] += elapsed
            row["self_s"] += max(0.0, elapsed - frame["children"])
            if self.frames:
                self.frames[-1]["children"] += elapsed
            if durable:
                self.snapshot()

    def has(self, name):
        return any(f["name"] == name for f in self.frames)

    def verify_panels(self, dimension):
        for kind, expected in (
            ("hist_2d", dimension * (dimension - 1) // 2),
            ("kde_1d", dimension),
        ):
            rows = [v for k, v in self.totals.items() if k.endswith("/panel_" + kind)]
            if sum(r["calls"] for r in rows) != expected or any(r["errors"] for r in rows):
                raise ValueError(f"unexpected completed {kind} panel count")


@contextlib.contextmanager
def attribution_scope(directory, dimension=30):
    import anesthetic
    import anesthetic.plot as plot
    import matplotlib.image as mpl_image
    import numpy as np
    from anesthetic.plotting import PlotAccessor
    from anesthetic.plotting._matplotlib import hist
    from anesthetic.samples import NestedSamples
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from pandas.plotting._matplotlib import core as pandas_core

    timers = Timers(Path(directory) / "attribution.json")

    def wrapper(original, label, durable=False, condition=None):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            if condition is not None and not condition():
                return original(*args, **kwargs)
            with timers.measure(label, durable):
                result = original(*args, **kwargs)
                if label == "plot_2d":
                    timers.verify_panels(dimension)
                return result

        return measured

    original_dispatch = PlotAccessor.__call__

    @functools.wraps(original_dispatch)
    def dispatch(*args, **kwargs):
        kind = kwargs.get("kind")
        if not timers.has("plot_2d") or kind not in ("hist_2d", "kde_1d"):
            return original_dispatch(*args, **kwargs)
        with timers.measure("panel_" + kind, durable=True):
            return original_dispatch(*args, **kwargs)

    def histogram():
        return timers.has("hist_kernel")

    def exporting():
        return timers.has("savefig")

    hooks = [
        (anesthetic, "make_2d_axes", "make_axes", True, None),
        (NestedSamples, "plot_2d", "plot_2d", True, None),
        (hist, "hist_plot_2d", "hist_kernel", False, None),
        (hist, "kde_plot_1d", "kde_kernel", False, None),
        (plot, "quantile", "quantile", False, None),
        (np, "histogram2d", "histogram2d", False, histogram),
        (Axes, "pcolormesh", "pcolormesh", False, histogram),
        (pandas_core, "handle_shared_axes", "shared_axes", True, None),
        (plot.AxesDataFrame, "set_labels", "set_labels", True, None),
        (Axes, "legend", "legend", False, None),
        (Figure, "savefig", "savefig", True, None),
        (FigureCanvasAgg, "draw", "agg_draw", True, exporting),
        (Figure, "draw", "figure_draw", True, exporting),
        (mpl_image, "imsave", "image_write", True, exporting),
    ]
    with contextlib.ExitStack() as stack:
        stack.enter_context(replaced(PlotAccessor, "__call__", dispatch))
        for owner, name, label, durable, condition in hooks:
            original = getattr(
                owner, name
            )  # Fail preflight on unavailable hook, never silently omit.
            stack.enter_context(replaced(owner, name, wrapper(original, label, durable, condition)))
        timers.snapshot()
        try:
            yield timers
        finally:
            timers.snapshot()


def synthetic_preflight(directory):
    """Deterministic tiny plot through real aliases; no likelihood or sampling."""
    import anesthetic
    import numpy as np
    from anesthetic import NestedSamples
    from autofit.non_linear.plot.nest_plotters import _deferred_shared_axes
    from matplotlib import pyplot as plt

    x = np.linspace(-2, 2, 30)
    samples = NestedSamples(
        np.column_stack([x, x * x + x / 3]), columns=["x", "y"], weights=np.ones(30)
    )
    original = anesthetic.make_2d_axes
    with attribution_scope(directory, dimension=2) as timers:
        fig, axes = anesthetic.make_2d_axes(["x", "y"], upper=False, figsize=(3, 3), dpi=60)
        with _deferred_shared_axes():
            samples.plot_2d(axes, kinds={"diagonal": "kde_1d", "lower": "hist_2d"})
        axes.set_labels({"x": "x", "y": "y"})
        fig.savefig(Path(directory) / "synthetic.png")
        plt.close(fig)
        timers.verify_panels(2)
    if anesthetic.make_2d_axes is not original:
        raise ValueError("plot hook not restored")
    labels = {k.split("/")[-1] for k in timers.totals}
    if (
        not {
            "quantile",
            "histogram2d",
            "pcolormesh",
            "shared_axes",
            "figure_draw",
            "agg_draw",
            "image_write",
        }
        <= labels
    ):
        raise ValueError("synthetic preflight did not exercise all critical aliases")
    atomic_json(
        Path(directory) / "success.json",
        {"status": "success", "panels": {"hist_2d": 1, "kde_1d": 2}, "labels": sorted(labels)},
    )


def export_attribution(root, destination, case_id):
    """Publish partial counters without promoting an interrupted replay."""
    import html
    import shutil

    path = Path(root) / case_id / "attribution.json"
    if not path.exists():
        return "<p>No attribution receipt: child did not reach the plotting hooks.</p>"
    receipt = json.loads(path.read_text())
    shutil.copyfile(path, Path(destination) / "attribution.json")
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['calls']}</td><td>{v['inclusive_s']:.4f}</td><td>{v['self_s']:.4f}</td><td>{v['errors']}</td></tr>"
        for k, v in sorted(receipt["completed"].items())
    )
    active = ", ".join(f["name"] for f in receipt["active"]) or "none"
    return (
        "<h2>Plot attribution</h2><p>Nested inclusive times overlap; do not add rows. "
        "Self time excludes timed children; both exclude recorded snapshot I/O. "
        "Unfinished work is absent from completed totals.</p>"
        f"<p>Active at last snapshot: {html.escape(active)}. Recorded logging: {receipt['logging_s']:.4f}s.</p>"
        "<table><tr><th>Call path</th><th>Completed calls</th><th>Inclusive seconds</th><th>Self seconds</th><th>Errors</th></tr>"
        + rows
        + "</table><p><a href='attribution.json'>Raw attribution receipt</a></p>"
    )


def run_synthetic_preflight(directory):
    import subprocess
    import sys

    from profiling.nautilus_resources import supervise

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    command = Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/diagnose.py"
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(command), "synthetic", "--output", str(directory)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        supervision = supervise(process, 20, 4096 * 1024**2)
    atomic_json(directory / "supervision.json", supervision)
    if supervision["returncode"] != 0 or supervision["termination_reason"]:
        raise ValueError(
            "synthetic preflight failed; see retained log/supervision; no replay launched"
        )
    witness = json.loads((directory / "success.json").read_text())
    if witness["status"] != "success":
        raise ValueError("synthetic preflight did not succeed")
    return {**witness, "supervision": supervision}
