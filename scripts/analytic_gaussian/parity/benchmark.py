"""Analytic Gaussian benchmark cell: closed form vs minimal EP vs autofit graphical vs autofit EP.

Profiles the CI parity script of the conjugate hierarchical Gaussian benchmark,
``autofit_workspace_test/scripts/graphical/analytic_gaussian.py`` (autofit_workspace_test#91;
seed 0, N = 5 datasets, 20 points each), as one ``runtime`` + ``memory`` + correctness record.
The benchmark scripts stay in autofit_workspace_test (their CI home); this cell runs the
parity script unchanged from the sibling checkout, times it and parses its own verdicts:

- per-leg, per-column agreement with the closed form (cells passed, worst a and b, where
  ``a = |dmean| / std_ref`` and ``b = |std / std_ref - 1|``), the hard caps and the final
  ``PARITY: PASS|FAIL (k/n)`` line;
- the closed-form leg-B scatter quantiles, including the analytic upper limit ``sigma_q95``,
  checked against the value banked in the benchmark docstrings (12.15 on seed 0);
- the script's own per-column runtimes (information only; the record is the whole script).

Suite members: ``analytic_reference.py`` (closed form), ``analytic_ep_minimal.py`` (minimal
EP) and ``analytic_autofit.py`` (shared autofit wiring) are imported by the parity script;
``analytic_gaussian_collapse.py`` is a separate CI smoke script, not profiled here; and
``analytic_gaussian_priors.py`` is parked ``NEEDS_FIX`` in ``config/build/no_run.yaml`` and is
recorded as parked, never run.

Usage::

    source activate.sh
    python scripts/analytic_gaussian/parity/benchmark.py

``autofit_workspace_test`` is found beside this repo (canonical ``fit/`` layout or a flat
task bundle) or at ``$AUTOFIT_WORKSPACE_TEST``. Writes
``results/analytic_gaussian/parity/seed0_n5.json`` (schema ``autofit-profiling-parity``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").is_file():
            return parent
    raise RuntimeError("ruff.toml root sentinel not found above this script")


ROOT = _repo_root()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "misc")]

PARITY_SCRIPT = "scripts/graphical/analytic_gaussian.py"
PARKED_SCRIPT = "graphical/analytic_gaussian_priors.py"
RESULT = ROOT / "results" / "analytic_gaussian" / "parity" / "seed0_n5.json"
COLUMNS = ("minimal EP", "autofit graphical", "autofit EP")

# Banked in the benchmark's own docstrings (analytic_gaussian.py "q05/q50/q95 on seed 0 =
# 3.0/6.0/12.2" and analytic_gaussian{,_collapse}.py "[q05, q95] = [2.98, 12.15]").
BANKED = {"sigma_q05": 2.98, "sigma_q95": 12.15, "decimals": 2}
BANKED_SOURCE = (
    "autofit_workspace_test/scripts/graphical/analytic_gaussian.py and "
    "analytic_gaussian_collapse.py docstrings ([q05, q95] = [2.98, 12.15], seed 0)"
)
DOCUMENTED_RUNTIME = "the script runs in ~85-130 s locally against the 300 s CI cap (its docstring)"

ROW_RE = re.compile(r"(PASS|FAIL) a=([0-9.]+) b=([0-9.]+)")
QUANT_RE = re.compile(r"q05/q50/q95 = ([0-9.]+)/([0-9.]+)/([0-9.]+)")
RUNTIME_RE = re.compile(r"(closed form|minimal EP|autofit graphical|autofit EP) ([0-9.]+)s")
CELLS_RE = re.compile(r"cells passed: (\d+)/(\d+)")
PARITY_RE = re.compile(r"^PARITY: (PASS|FAIL) \((\d+)/(\d+)\)", re.M)
CAP_RE = re.compile(r"\[(PASS|FAIL)\] hard cap: (.+)")


def workspace_test() -> Path:
    env = os.environ.get("AUTOFIT_WORKSPACE_TEST")
    for cand in ([Path(env)] if env else []) + [ROOT.parent / "autofit_workspace_test"]:
        if (cand / PARITY_SCRIPT).is_file():
            return cand.resolve()
    raise SystemExit(
        "autofit_workspace_test not found beside this repo; set $AUTOFIT_WORKSPACE_TEST"
    )


def parse(stdout: str) -> dict:
    """Split the parity script's report into legs and read its verdicts."""
    parts = re.split(r"^Leg ([AB]) ", stdout, flags=re.M)
    legs = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        cols = {c: {"passed": 0, "cells": 0, "max_a": 0.0, "max_b": 0.0} for c in COLUMNS}
        for line in body.splitlines():
            hits = ROW_RE.findall(line)
            if len(hits) != len(COLUMNS):
                continue
            for col, (verdict, a, b) in zip(COLUMNS, hits):
                c = cols[col]
                c["cells"] += 1
                c["passed"] += verdict == "PASS"
                c["max_a"] = max(c["max_a"], float(a))
                c["max_b"] = max(c["max_b"], float(b))
        cells = CELLS_RE.search(body)
        runtimes = re.search(r"runtimes: (.+)", body)
        leg = {
            "columns": cols,
            "cells_passed": int(cells.group(1)) if cells else None,
            "cells_total": int(cells.group(2)) if cells else None,
            "hard_caps": [{"passed": v == "PASS", "message": m} for v, m in CAP_RE.findall(body)],
            "column_runtimes_s": (
                {k: float(v) for k, v in RUNTIME_RE.findall(runtimes.group(1))} if runtimes else {}
            ),
        }
        quant = QUANT_RE.search(body.splitlines()[0])
        if quant:
            leg["sigma_quantiles"] = dict(
                zip(("sigma_q05", "sigma_q50", "sigma_q95"), map(float, quant.groups()))
            )
        legs[name] = leg
    parity = PARITY_RE.search(stdout)
    return {
        "legs": legs,
        "parity": parity.group(1) if parity else None,
        "parity_cells_passed": int(parity.group(2)) if parity else None,
        "parity_cells_total": int(parity.group(3)) if parity else None,
    }


def parked_reason(ws: Path) -> str | None:
    for line in (ws / "config" / "build" / "no_run.yaml").read_text().splitlines():
        if line.strip().startswith(f"- {PARKED_SCRIPT}"):
            return line.split("#", 1)[1].strip() if "#" in line else "listed in no_run.yaml"
    return None


def _git(path: Path, *args) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def software(ws: Path):
    import autofit
    import dynesty
    import numpy
    import scipy

    fit_dir = Path(autofit.__file__).resolve().parents[1]
    sw = {
        "PyAutoFit": _git(fit_dir, "rev-parse", "HEAD"),
        "autofit_workspace_test": _git(ws, "rev-parse", "HEAD"),
        "dynesty": dynesty.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
    }
    try:
        import autonerves

        sw["PyAutoNerves"] = _git(
            Path(autonerves.__file__).resolve().parents[1], "rev-parse", "HEAD"
        )
    except ImportError:
        pass
    tag = _git(fit_dir, "describe", "--tags", "--abbrev=0")
    dirty = _git(fit_dir, "status", "--porcelain", "--untracked-files=no")
    return {k: v for k, v in sw.items() if v}, tag or autofit.__version__, bool(dirty)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--no-write", action="store_true", help="print only")
    parser.add_argument("--log", type=Path, help="also save the parity script's output here")
    args = parser.parse_args(argv)

    ws = workspace_test()
    sw_before, _, _ = software(ws)
    env = {**os.environ, "PYAUTO_SKIP_WORKSPACE_VERSION_CHECK": "1"}
    load_before = [round(x, 2) for x in os.getloadavg()]
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, PARITY_SCRIPT], cwd=ws, env=env, capture_output=True, text=True
    )
    wall_s = time.perf_counter() - t0
    rss_kib = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if args.log:
        args.log.write_text(proc.stdout + proc.stderr)
    parsed = parse(proc.stdout)
    if parsed["parity"] is None:
        print(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"parity script produced no PARITY line (exit {proc.returncode})")

    sw, library_version, dirty = software(ws)
    if sw.get("PyAutoFit") != sw_before.get("PyAutoFit"):
        raise SystemExit("PyAutoFit checkout moved during the measurement; rerun")

    q = parsed["legs"]["B"]["sigma_quantiles"]
    scale = 10 ** BANKED["decimals"]
    sigma_q95_matches = round(q["sigma_q95"] * scale) == round(BANKED["sigma_q95"] * scale)
    sigma_q05_matches = round(q["sigma_q05"] * scale) == round(BANKED["sigma_q05"] * scale)

    row = {
        "schema": "autofit-profiling-parity",
        "version": 1,
        "cell": "analytic_gaussian/parity/benchmark",
        "script": "scripts/analytic_gaussian/parity/benchmark.py",
        "setup": {
            "id": "analytic_gaussian/parity/seed0_n5",
            "label": "analytic Gaussian parity: closed form | minimal EP | autofit graphical | autofit EP",
            "dataset": "analytic_gaussian",
            "dataset_description": (
                "conjugate hierarchical Gaussian, seed 0, N = 5 datasets of 20 points, "
                "truth mu = 50, sigma = 10, noise s_i = linspace(3, 8), simulated in memory"
            ),
            "model": (
                "leg A: mu, x_1..x_5 with sigma = 10 known (6 parameters); leg B: sigma ~ "
                "TruncatedGaussian(10, 5, 0, 100) added (7 parameters)"
            ),
            "n_datasets": 5,
            "free_parameters": 7,
            "data_pixels": 100,
            "search": "DynestyStatic (graphical) + LaplaceOptimiser(projection='moments') (EP)",
            "search_settings": {
                "nlive": 100,
                "sample": "unif",
                "maxcall": 30000,
                "seed": 0,
                "kl_tol": 0.0001,
                "max_steps": 30,
            },
            "likelihood": "numpy (GaussianDataAnalysis, no JAX)",
            "config": "autofit_workspace_test config (the parity script's CI home)",
            "benchmark_script": f"autofit_workspace_test/{PARITY_SCRIPT}",
        },
        "method": {
            "id": "parity-script-subprocess-wall",
            "statistic": "single",
            "repetitions": 1,
            "warmup": "none; one run of the parity script in a fresh child process",
            "synchronization": "none (numpy, single process)",
            "cache_state": "cold process; imports inside the timed interval",
            "timer": "time.perf_counter around the whole child process; peak RSS is the child's ru_maxrss",
        },
        "wall_s": wall_s,
        "peak_rss_kib": rss_kib,
        "exit_code": proc.returncode,
        "correctness": {
            **parsed,
            "sanity_pass": parsed["parity"] == "PASS" and proc.returncode == 0,
            "sanity_rule": "the parity script's own PARITY verdict (tolerances never loosened)",
            "sigma_q95": q["sigma_q95"],
            "sigma_q95_banked": BANKED["sigma_q95"],
            "sigma_q95_matches_banked": sigma_q95_matches,
            "sigma_q05_matches_banked": sigma_q05_matches,
            "banked_source": BANKED_SOURCE,
        },
        "suite": {
            "parity_script": f"{PARITY_SCRIPT} (run)",
            "imported": ["analytic_reference.py", "analytic_ep_minimal.py", "analytic_autofit.py"],
            "not_profiled": ["analytic_gaussian_collapse.py (separate CI smoke script)"],
            "parked": {"script": PARKED_SCRIPT, "status": "not run", "reason": parked_reason(ws)},
        },
        "reference": {"documented_runtime": DOCUMENTED_RUNTIME},
        "identity": {
            "device": "cpu",
            "backend": "numpy",
            "precision": "float64",
            "library_version": library_version,
            "software": sw,
        },
        "provenance": {
            "host": platform.node(),
            "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "pyautofit_checkout_dirty": dirty,
            "load_average_before": load_before,
            "load_average_after": [round(x, 2) for x in os.getloadavg()],
        },
    }
    c = row["correctness"]
    print(
        f"parity {c['parity']} ({c['parity_cells_passed']}/{c['parity_cells_total']}); wall "
        f"{wall_s:.1f} s; peak RSS {rss_kib / 1024:.0f} MiB; sigma_q95 {q['sigma_q95']:.4f} "
        f"(banked {BANKED['sigma_q95']}, match {sigma_q95_matches})"
    )
    for leg, body in parsed["legs"].items():
        cols = ", ".join(
            f"{k} {v['passed']}/{v['cells']} (max a {v['max_a']:.3f}, b {v['max_b']:.3f})"
            for k, v in body["columns"].items()
        )
        print(f"  leg {leg}: {cols}; runtimes {body['column_runtimes_s']}")
    if not args.no_write:
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(json.dumps(row, indent=2) + "\n")
        print(f"wrote {RESULT.relative_to(ROOT)}")
    return 0 if c["sanity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
