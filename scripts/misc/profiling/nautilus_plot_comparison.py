"""Child-local plot policies and durable witnesses; no library/default changes."""

import contextlib
import copy
import functools
import hashlib
import json
import logging
import struct
from pathlib import Path

from profiling.nautilus_resources import atomic_json

LIMITS = {"budget_s": 420, "timeout_s": 120, "memory_mib": 4096}
POLICIES = [
    {"name": "no-corner", "indices": [], "figsize": None, "dpi": None},
    {"name": "six-small", "indices": [0, 6, 12, 17, 23, 29], "figsize": [12, 12], "dpi": 100},
    {"name": "thirty-small", "indices": list(range(30)), "figsize": [12, 12], "dpi": 100},
]
STYLE = {"kinds": {"diagonal": "kde_1d", "lower": "hist_2d"}, "upper": False}


class SummaryTimingGate:
    """One explicitly authorized target call inside final-summary timing only."""

    def __init__(self, target, forbidden, limit=0):
        if limit not in (0, 1):
            raise ValueError("summary timing limit must be zero or one")
        self.target, self.forbidden, self.limit = target, forbidden, limit
        self.active = False
        self.calls = 0
        self.parameters = None
        self.value = None

    @contextlib.contextmanager
    def summary(self):
        if self.active:
            raise RuntimeError("nested summary timing scope")
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def evaluate(self, parameters):
        if not self.active or self.calls >= self.limit:
            return self.forbidden(parameters)
        self.calls += 1
        self.parameters = list(parameters)
        self.value = self.target(parameters)
        return self.value


def font_cache():
    import matplotlib
    from matplotlib import font_manager

    font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(matplotlib.get_cachedir()).glob("fontlist-*.json")
    }
    if not files:
        raise ValueError("font cache was not established")
    return {
        "policy": "font cache initialized in charged preflight, shared unchanged by sequential children",
        "files": files,
    }


@contextlib.contextmanager
def replaced(owner, name, replacement):
    owned = name in vars(owner)
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        if owned:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def png_dimensions(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("not a PNG with an IHDR")
    return list(struct.unpack(">II", header[16:24]))


def verify_plot(policy, receipt, directory, expected_samples):
    if receipt["errors"]:
        raise ValueError(f"plot error (possibly swallowed): {receipt['errors']}")
    if receipt["search_plot_calls"] != 1 or receipt["input_samples"] != expected_samples:
        raise ValueError("plot invocation/sample count differs")
    files = sorted(Path(directory).rglob("*.png"))
    if not policy["indices"]:
        if files or receipt["figures"]:
            raise ValueError("no-corner case unexpectedly produced a figure")
        return
    if len(files) != 1 or len(receipt["figures"]) != 1:
        raise ValueError("missing or extra PNG/figure")
    figure = receipt["figures"][0]
    n = len(policy["indices"])
    if (
        figure["figsize"] != policy["figsize"]
        or figure["dpi"] != policy["dpi"]
        or figure["axes"] != n * (n + 1) // 2 + n
    ):
        raise ValueError("unexpected figure geometry")
    if png_dimensions(files[0]) != [1200, 1200]:
        raise ValueError("unexpected PNG dimensions")
    receipt["png"] = {
        "path": str(files[0].relative_to(directory)),
        "dimensions": png_dimensions(files[0]),
        "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "bytes": files[0].stat().st_size,
    }


@contextlib.contextmanager
def plot_scope(case, directory, log):
    import anesthetic
    import autofit as af
    from anesthetic.samples import NestedSamples
    from autofit.non_linear.plot.nest_plotters import corner_anesthetic
    from matplotlib.figure import Figure

    policy = case["plot_policy"]
    receipt = {
        "policy": copy.deepcopy(policy),
        "style": copy.deepcopy(STYLE),
        "search_plot_calls": 0,
        "input_samples": None,
        "figures": [],
        "errors": [],
    }
    receipt_path = directory / "plot-receipt.json"

    def save():
        atomic_json(receipt_path, receipt)

    class Errors(logging.Handler):
        def emit(self, record):
            if "Unable to produce" in record.getMessage():
                receipt["errors"].append(record.getMessage())
                save()

    logger = logging.getLogger("autofit.non_linear.plot.plot_util")
    handler = Errors()
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    def instrument(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            try:
                with log.stage(label):
                    if name == "savefig":
                        fig = args[0]
                        receipt["figures"].append(
                            {
                                "figsize": fig.get_size_inches().tolist(),
                                "dpi": float(fig.dpi),
                                "axes": len(fig.axes),
                            }
                        )
                        save()
                    result = original(*args, **kwargs)
                return result
            except BaseException as error:
                receipt["errors"].append(f"{label}: {type(error).__name__}: {error}")
                save()
                raise

        return wrapped

    def search_plot(search, samples):
        receipt["search_plot_calls"] += 1
        receipt["input_samples"] = len(samples.sample_list)
        save()
        with log.stage("search_corner"):
            if policy["indices"]:
                corner_anesthetic(
                    samples,
                    path=search.paths.image_path / "search",
                    filename="diagnostic_corner",
                    format="png",
                    parameter_indices=policy["indices"],
                    figsize=policy["figsize"],
                    dpi=policy["dpi"],
                    **copy.deepcopy(STYLE),
                )
            verify_plot(policy, receipt, directory, case["original_selected_samples"])
            save()

    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(replaced(af.Nautilus, "plot_results", search_plot))
            for owner, name, label in (
                (anesthetic, "make_2d_axes", "axes_construction"),
                (NestedSamples, "plot_2d", "posterior_plotting"),
                (Figure, "savefig", "png_export"),
            ):
                stack.enter_context(replaced(owner, name, instrument(owner, name, label)))
            yield receipt
            verify_plot(policy, receipt, directory, case["original_selected_samples"])
            save()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def export_plots(plan, state, root, destination):
    """Keep incomplete cases visible and publish only hash-verified completed PNGs."""
    import html
    import shutil

    attempts = {a["case"]: a for a in state["attempts"]}
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Nautilus plotting comparison</title><style>
body{font:17px/1.6 system-ui;max-width:1050px;margin:36px auto;padding:0 24px;color:#172333}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #ccd4de}
img{max-width:100%;height:auto}summary{cursor:pointer}h1{font-size:30px}
</style><h1>Nautilus plotting comparison</h1>
<p>Experimental · outside Pulse · unchanged posterior · one frozen run per policy.</p>"""
    document += f"<p>Charged runtime: {state['spent_s']:.2f}s, including preflight. Stage times are inclusive and overlap.</p>"
    document += "<table><tr><th>Policy</th><th>Status</th><th>Process seconds</th><th>Axes</th><th>Plotting</th><th>PNG export</th></tr>"
    details = []
    for case in plan["cases"]:
        attempt = attempts.get(case["id"], {})
        stages = attempt.get("stages", {"completed": [], "incomplete": []})
        times = {s["stage"]: s["wall_s"] for s in stages["completed"] if s["status"] == "exit"}

        def timing(name, times=times):
            return f"{times[name]:.3f}" if name in times else "—"

        cells = [
            case["plot_policy"]["name"],
            attempt.get("status", "unrun"),
            f"{attempt['process_wall_s']:.3f}" if "process_wall_s" in attempt else "—",
            timing("axes_construction"),
            timing("posterior_plotting"),
            timing("png_export"),
        ]
        document += "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>"
        receipt_path = root / case["id"] / "plot-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["policy"] != case["plot_policy"]:
                raise ValueError("plot receipt policy differs")
            atomic_json(destination / (case["id"] + "-plot.json"), receipt)
            if "png" in receipt:
                png = root / case["id"] / receipt["png"]["path"]
                if hashlib.sha256(png.read_bytes()).hexdigest() != receipt["png"]["sha256"]:
                    raise ValueError("plot PNG changed")
                filename = case["id"] + ".png"
                shutil.copyfile(png, destination / filename)
                details.append(
                    f"<h2>{html.escape(case['plot_policy']['name'])}</h2><img src='{filename}' alt='Diagnostic corner plot'>"
                )
        unfinished = [s["stage"] for s in stages["incomplete"]]
        if unfinished:
            details.append(
                f"<p>{html.escape(case['id'])}: unfinished stages {html.escape(', '.join(unfinished))}.</p>"
            )
    document += "</table><p>The full30D small-canvas figure tests cost, not a recommended readable default. No production settings changed.</p>"
    document += "".join(details)
    document += "<p><a href='attempts.json'>All receipts</a> · <a href='plan.json'>Portable plan</a> · <a href='../README.md'>Findings</a></p></html>"
    return document
