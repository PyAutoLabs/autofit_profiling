"""Build the profiling-summary@2 catalogue and the Pages board (stdlib only).

Run ``python scripts/misc/tooling/build_catalogue.py [--check] [--validate-with PULSE]``.

Reads every committed result row under ``results/**/*.json`` and writes, deterministically:

- ``dashboard/catalogue.json`` — the PyAutoPulse read contract ``profiling-summary`` v2
  (setups, records, explicit selections). Three result-row schemas are read, each on the
  existing axes (no contract extension):

  - ``autofit-profiling-breakdown`` (B4a): one ``breakdown`` record per measured component
    plus the ``runtime`` wall;
  - ``autofit-profiling-baseline`` (B4b, EP / graphical N3/N10/N30): a ``runtime`` wall and a
    ``memory`` ``host_peak_rss`` record, with the sanity verdict and the reproduction note
    against the committed developer baseline kept in the row;
  - ``autofit-profiling-parity`` (B4b, analytic Gaussian benchmark): the same two records,
    with the closed-form / minimal-EP / graphical / EP agreement kept in the row.

  The
  shape follows autolens_profiling's ``build_catalogue.py``; nothing here judges a
  timing, picks a "latest" row or infers acceptance — every row is ``unreviewed``.
- ``dashboard/index.html`` — a minimal page listing each cell's breakdown.
- ``dashboard/state.json`` — the project badge feed (PyAutoBrain ``board/_state.py`` v1);
  the Brain cockpit reads the PyAutoPulse organ feed, not this one.

``--check`` exits 1 when any committed output differs from a fresh render (the lint gate).
``--validate-with PATH`` validates the catalogue with the Pulse checkout's own reader
(``pulse.summary.classify``) before anything is written.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = "autofit-profiling-breakdown"
BASELINE_SCHEMA = "autofit-profiling-baseline"
PARITY_SCHEMA = "autofit-profiling-parity"
RESULT_SCHEMAS = (RESULT_SCHEMA, BASELINE_SCHEMA, PARITY_SCHEMA)
PROJECT = "autofit_profiling"
REPO_URL = "https://github.com/PyAutoLabs/autofit_profiling"
PAGES_URL = "https://pyautolabs.github.io/autofit_profiling/"
OUTPUTS = ("catalogue.json", "index.html", "state.json")

# Components whose exact-zero value means "this variant has no such phase", declared
# as an inapplicable slot instead of a zero-cost record (REFERENCE.md: missing cells
# belong in selections, not zero-valued records).
INAPPLICABLE_REASON = {
    "checkpoint_io": "the bare sampler is run without a checkpoint file, so it writes none",
}

# Model names per dataset (the setup's short `model` field; the row carries the long form).
MODEL_NAME = {
    "gaussian_x1": "gaussian",
    "gaussian_1d_ep": "gaussian, centre shared across datasets",
    "analytic_gaussian": "conjugate hierarchical gaussian",
}

LIMITATIONS = [
    "Laptop CPU only, pinned to one host (DESKTOP-H143S82, WSL2) so Pulse's same-host rule pairs every record; the epic's release-sweep pin (euclid-ral-gpu-2, an A100 node) does not fit CPU Dynesty runs. Cross-machine reproduction on RAL is a later campaign.",
    "Wall times vary up to ±40 % between sessions on this host, so breakdown shares are the comparable quantity, not seconds; the EP / graphical / parity rows are single fresh-process fits (repetitions 1).",
    "Component times are exclusive monkey-patched wall-clock intervals; 'other' is wall minus the instrumented total, not a measured phase.",
    "The EP / graphical rows carry a producer-side reproduction note against the committed autofit_workspace_developer baselines (runtime-drift-2x-1ms, in each row's `reference`); it is not a Pulse comparison, because those baselines recorded no host and no record pair exists.",
    "The N30 EP and graphical rows reproduce the committed sanity FAIL; their wall and memory are real measurements of a fit whose parameter recovery fails, not of a correct fit.",
    "Every record is unreviewed: the producer declares measurements, the Brain's profiling conductor judges them; nothing here is a verdict.",
    "No temporal comparisons: one baseline per setup.",
]


def load_rows(root: Path):
    rows = []
    for path in sorted((root / "results").rglob("*.json")):
        doc = json.loads(path.read_text())
        if isinstance(doc, dict) and doc.get("schema") in RESULT_SCHEMAS:
            rows.append((path.relative_to(root).as_posix(), doc))
    return rows


def _unit(value):
    if isinstance(value, str):
        return "name"
    if isinstance(value, int) and not isinstance(value, bool):
        return "count"
    return "dimensionless"


def _setup(doc):
    if doc["schema"] != RESULT_SCHEMA:
        return _run_setup(doc)
    s = doc["setup"]
    config = {
        "search": {"value": s["search"], "unit": "name"},
        "variant": {"value": doc["variant"], "unit": "name"},
        "likelihood": {"value": s["likelihood"], "unit": "name"},
        "free_parameters": {"value": 3, "unit": "count"},
        "data_pixels": {"value": 100, "unit": "count"},
        "output_config": {"value": s["config"], "unit": "name"},
    }
    for key, value in s["search_settings"].items():
        unit = "count" if key in ("n_live", "n_networks", "number_of_cores") else "dimensionless"
        config[key] = {"value": value, "unit": unit}
    return {
        "id": s["id"],
        "label": s["label"],
        "dataset": s["dataset"],
        "model": "gaussian",
        "instrument": None,
        "unknowns": {"instrument": "toy 1-D dataset; no instrument"},
        "configuration_id": f"{s['id']}@{doc['version']}",
        "configuration": dict(sorted(config.items())),
    }


def _run_setup(doc):
    """Setup of a whole-run row (EP / graphical baseline, analytic parity)."""
    s = doc["setup"]
    config = {
        "search": {"value": s["search"], "unit": "name"},
        "likelihood": {"value": s["likelihood"], "unit": "name"},
        "n_datasets": {"value": s["n_datasets"], "unit": "count"},
        "free_parameters": {"value": s["free_parameters"], "unit": "count"},
        "data_pixels": {"value": s["data_pixels"], "unit": "count"},
        "output_config": {"value": s["config"], "unit": "name"},
    }
    for key, value in s["search_settings"].items():
        config.setdefault(key, {"value": value, "unit": _unit(value)})
    return {
        "id": s["id"],
        "label": s["label"],
        "dataset": s["dataset"],
        "model": MODEL_NAME.get(s["dataset"], s["dataset"]),
        "instrument": None,
        "unknowns": {"instrument": "toy 1-D dataset; no instrument"},
        "configuration_id": f"{s['id']}@{doc['version']}",
        "configuration": dict(sorted(config.items())),
    }


def _identity(doc):
    ident = doc["identity"]
    return {
        "device": ident["device"],
        "backend": ident["backend"],
        "precision": ident["precision"],
        "library_version": ident["library_version"],
        "software": dict(sorted(ident["software"].items())),
    }


def _method(doc, suffix):
    m = doc["method"]
    return {
        "id": f"{m['id']}/{suffix}",
        "statistic": m["statistic"],
        "repetitions": m["repetitions"],
        "warmup": m["warmup"],
        "synchronization": m["synchronization"],
        "cache_state": m["cache_state"],
    }


_REASON = "producer measurement on one laptop host; awaits the profiling conductor's triage"


def _reason(doc):
    if doc["schema"] == RESULT_SCHEMA:
        return _REASON
    reason = _reason_core(doc)
    if doc["provenance"].get("host_contention"):
        reason += "; host contended during the run (see provenance.host_contention)"
    return reason


def _reason_core(doc):
    c = doc["correctness"]
    verdict = "PASS" if c["sanity_pass"] else "FAIL"
    if doc["schema"] == BASELINE_SCHEMA:
        committed = "PASS" if doc["reference"]["sanity_pass"] else "FAIL"
        return f"{_REASON}; sanity {verdict} (committed baseline {committed})"
    return f"{_REASON}; parity {c['parity']} ({c['parity_cells_passed']}/{c['parity_cells_total']})"


def _record(doc, path, setup_id, axis, metric, value, method, fragment, unit="s"):
    prov = doc["provenance"]
    return {
        "id": f"{setup_id}/{axis}/{metric}",
        "setup_id": setup_id,
        "run_id": f"{setup_id}@{prov['measured_at']}",
        "axis": axis,
        "metric": metric,
        "unit": unit,
        "measurement": {metric: value},
        "identity": _identity(doc),
        "method": method,
        "provenance": {
            "host": prov["host"],
            "measured_at": prov["measured_at"],
            "has_provenance": True,
            "qualified": True,
        },
        "validation": {"status": "unreviewed", "reason": _reason(doc)},
        "evidence": {"path": path, "fragment": fragment},
    }


def _selection(record, status="unreviewed", reason="the cell's committed median row"):
    return {
        "id": f"select/{record['id']}",
        "setup_id": record["setup_id"],
        "axis": record["axis"],
        "metric": record["metric"],
        "unit": record["unit"],
        "method_id": record["method"]["id"],
        "identity": record["identity"],
        "host": record["provenance"]["host"],
        "status": status,
        "record_id": record["id"],
        "reason": reason,
    }


def build(root: Path):
    rows = load_rows(root)
    setups, records, selections = [], [], []
    for path, doc in rows:
        setup = _setup(doc)
        sid = setup["id"]
        setups.append(setup)
        wall = _record(
            doc, path, sid, "runtime", "wall", doc["wall_s"], _method(doc, "wall"), "/wall_s"
        )
        records.append(wall)
        selections.append(_selection(wall))
        if doc["schema"] != RESULT_SCHEMA:
            rss = _record(
                doc,
                path,
                sid,
                "memory",
                "host_peak_rss",
                doc["peak_rss_kib"],
                _method(doc, "peak_rss"),
                "/peak_rss_kib",
                unit="KiB",
            )
            records.append(rss)
            selections.append(_selection(rss))
            continue
        method = _method(doc, "components")
        for metric, value in doc["components_s"].items():
            if value == 0 and metric in INAPPLICABLE_REASON:
                slot = _selection(
                    _record(doc, path, sid, "breakdown", metric, 0, method, None),
                    "inapplicable",
                    INAPPLICABLE_REASON[metric],
                )
                slot["record_id"] = None
                selections.append(slot)
                continue
            rec = _record(
                doc, path, sid, "breakdown", metric, value, method, f"/components_s/{metric}"
            )
            records.append(rec)
            selections.append(_selection(rec))
    measured = sorted(r["provenance"]["measured_at"] for r in records)
    latest = measured[-1] if measured else None
    catalogue = {
        "schema": "profiling-summary",
        "version": 2,
        "project": PROJECT,
        "scope": "pyautofit-profiling",
        # Deterministic: the render stamp is the newest evidence time, never the clock,
        # so --check is stable and an unchanged result set renders byte-identically.
        "generated_at": latest or "1970-01-01T00:00:00Z",
        "evidence_updated_at": latest,
        **({} if latest else {"evidence_updated_at_reason": "no result rows yet"}),
        "valid_until": None,
        "producer_revision": None,
        "comparison_policy": {"id": "no-temporal-comparisons"},
        "coverage": {
            "expected": {"cells": len(selections)},
            "observed": {
                "setups": len(setups),
                "records": len(records),
                "selected": sum(1 for s in selections if s["record_id"]),
            },
            "excluded": [],
        },
        "setups": setups,
        "records": records,
        "selections": selections,
        "hazards": [],
        "recommendations": [],
        "comparisons": [],
        "limitations": LIMITATIONS,
    }
    return catalogue, rows


def render_state(catalogue):
    n = catalogue["coverage"]["observed"]
    return {
        "schema_version": 1,
        "organ": PROJECT,
        "repo": PROJECT,
        "status": "grey" if not n["records"] else "green",
        "headline": (
            f"{n['setups']} setups · {n['records']} records (search.fit breakdown, "
            "EP / graphical baselines, analytic parity) — baseline only, not a verdict"
        ),
        "updated": catalogue["generated_at"],
        "pages_url": PAGES_URL,
        "items": [],
    }


def _pct(x):
    return f"{100 * x:.1f} %"


def render_html(rows, catalogue):
    out = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>autofit_profiling</title>",
        "<style>",
        ":root{--bg:#fff;--fg:#1b1f24;--muted:#57606a;--line:#d0d7de;--bar:#3b6fb6}",
        "@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--line:#30363d;--bar:#6ea0e6}}",
        "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:960px;padding:16px}",
        "table{border-collapse:collapse;width:100%;margin:8px 0 24px}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}",
        "td.n{text-align:right;font-variant-numeric:tabular-nums}.bar{background:var(--bar);height:10px}",
        "p.m,li{color:var(--muted)}a{color:var(--bar)}",
        "</style></head><body>",
        "<h1>autofit_profiling</h1>",
        "<p>Where the time in one PyAutoFit <code>search.fit</code> goes, and the expectation-propagation "
        "/ graphical-model baselines. Producer page for the "
        '<a href="https://pyautolabs.github.io/PyAutoPulse/">PyAutoPulse</a> instance <code>fit</code> '
        '(<a href="catalogue.json">catalogue.json</a>, <code>profiling-summary@2</code>). '
        "Measurements are declared, not judged.</p>",
    ]
    for path, doc in rows:
        if doc["schema"] != RESULT_SCHEMA:
            continue
        s = doc["setup"]
        ref = doc["reference"]
        out.append(f"<h2>{html.escape(s['id'])}</h2>")
        out.append(
            f"<p class=m>{html.escape(s['label'])} · wall {doc['wall_s']:.2f} s (median of "
            f"{doc['method']['repetitions']}) · {doc['likelihood_function_calls']} likelihood calls · "
            f"{html.escape(doc['identity']['device'])}/{html.escape(doc['identity']['backend'])} · "
            f"PyAutoFit {html.escape(doc['identity']['library_version'])} · measured "
            f"{html.escape(doc['provenance']['measured_at'])} · "
            f'<a href="{REPO_URL}/blob/main/{html.escape(path)}">result row</a></p>'
        )
        out.append(
            "<table><tr><th>component</th><th>seconds</th><th>share of wall</th><th></th></tr>"
        )
        ranked = sorted(doc["components_s"].items(), key=lambda kv: -kv[1])
        for name, sec in ranked:
            share = doc["shares_of_wall"][name]
            out.append(
                f"<tr><td>{html.escape(name)}</td><td class=n>{sec:.2f}</td><td class=n>{_pct(share)}</td>"
                f'<td style="width:40%"><div class=bar style="width:{100 * share:.1f}%"></div></td></tr>'
            )
        out.append("</table>")
        verdict = "within" if ref["within_tolerance"] else "outside"
        role = "reproduction gate" if ref["gates_reproduction"] else "information only"
        out.append(
            f"<p class=m>Findings-page shares ({ref['compared']}, {role}): "
            + ", ".join(
                f"{html.escape(k)} {_pct(v)} (Δ {100 * ref['deltas'][k]:+.1f} pp)"
                for k, v in ref["shares"].items()
            )
            + f" — {verdict} ±{100 * ref['tolerance_abs']:.0f} pp.</p>"
        )
    out.extend(_render_baselines([(p, d) for p, d in rows if d["schema"] == BASELINE_SCHEMA]))
    out.extend(_render_parity([(p, d) for p, d in rows if d["schema"] == PARITY_SCHEMA]))
    out.append("<h2>Limitations</h2><ul>")
    out.extend(f"<li>{html.escape(x)}</li>" for x in catalogue["limitations"])
    out.append("</ul>")
    out.append(
        f"<p class=m>Rendered from {len(rows)} result rows by "
        f'<a href="{REPO_URL}/blob/main/scripts/misc/tooling/build_catalogue.py">build_catalogue.py</a>; '
        f"evidence updated {html.escape(str(catalogue['evidence_updated_at']))}.</p>"
    )
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def _yes(flag):
    return "yes" if flag else "NO"


def _render_baselines(rows):
    if not rows:
        return []
    out = [
        "<h2>EP / graphical-model baselines (gaussian_1d_ep)</h2>",
        "<p class=m>N 1-D Gaussians sharing one centre, fitted by expectation propagation "
        "(<code>factor_graph.optimise</code>, per-factor DynestyStatic + Laplace cavity) and by one "
        "joint DynestyStatic graphical fit. One fresh-process fit per cell, reproduced against the "
        "committed <code>autofit_workspace_developer</code> baselines with the "
        "<code>runtime-drift-2x-1ms</code> rule (producer note, not a Pulse comparison).</p>",
        "<table><tr><th>cell</th><th>wall s</th><th>committed s</th><th>ratio</th><th>drift</th>"
        "<th>peak RSS MiB</th><th>truth logL sum</th><th>sanity (committed)</th></tr>",
    ]
    for path, doc in rows:
        ref, c = doc["reference"], doc["correctness"]
        d = ref["wall_drift"]
        sanity = "PASS" if c["sanity_pass"] else "FAIL"
        committed = "PASS" if ref["sanity_pass"] else "FAIL"
        out.append(
            f'<tr><td><a href="{REPO_URL}/blob/main/{html.escape(path)}">'
            f"{html.escape(doc['setup']['id'])}</a></td>"
            f"<td class=n>{doc['wall_s']:.1f}</td><td class=n>{ref['wall_s']:.1f}</td>"
            f"<td class=n>{d['ratio']:.2f}</td><td>{html.escape(d['status'])}</td>"
            f"<td class=n>{doc['peak_rss_kib'] / 1024:.0f}</td>"
            f"<td class=n>{c['truth_log_likelihood_sum']:.3f} "
            f"(match {_yes(ref['truth_log_likelihood_sum_matches'])})</td>"
            f"<td>{sanity} ({committed})</td></tr>"
        )
    out.append("</table>")
    first = rows[0][1]
    out.append(
        f"<p class=m>{html.escape(first['identity']['device'])}/{html.escape(first['identity']['backend'])}"
        f"/{html.escape(first['identity']['precision'])} · PyAutoFit "
        f"{html.escape(first['identity']['software'].get('PyAutoFit', '?')[:10])} · host "
        f"{html.escape(str(first['provenance']['host']))}.</p>"
    )
    return out


def _render_parity(rows):
    out = []
    for path, doc in rows:
        c = doc["correctness"]
        out.append("<h2>Analytic Gaussian benchmark (closed form vs EP vs graphical)</h2>")
        out.append(
            f"<p class=m>{html.escape(doc['setup']['label'])} · wall {doc['wall_s']:.1f} s · peak RSS "
            f"{doc['peak_rss_kib'] / 1024:.0f} MiB · PARITY {html.escape(str(c['parity']))} "
            f"({c['parity_cells_passed']}/{c['parity_cells_total']}) · sigma_q95 "
            f"{c['sigma_q95']:.4f} (banked {c['sigma_q95_banked']}, match "
            f"{_yes(c['sigma_q95_matches_banked'])}) · "
            f'<a href="{REPO_URL}/blob/main/{html.escape(path)}">result row</a></p>'
        )
        out.append(
            "<table><tr><th>leg</th><th>column</th><th>cells passed</th><th>max a</th>"
            "<th>max b</th><th>column s</th></tr>"
        )
        for leg, body in c["legs"].items():
            for col, v in body["columns"].items():
                sec = body["column_runtimes_s"].get(col)
                out.append(
                    f"<tr><td>{html.escape(leg)}</td><td>{html.escape(col)}</td>"
                    f"<td class=n>{v['passed']}/{v['cells']}</td><td class=n>{v['max_a']:.3f}</td>"
                    f"<td class=n>{v['max_b']:.3f}</td>"
                    f"<td class=n>{'' if sec is None else f'{sec:.1f}'}</td></tr>"
                )
        out.append("</table>")
        parked = doc["suite"]["parked"]
        out.append(
            f"<p class=m>Parked, not run: <code>{html.escape(parked['script'])}</code> "
            f"({html.escape(str(parked['reason']))}).</p>"
        )
    return out


def render(root: Path):
    catalogue, rows = build(root)
    return {
        "catalogue.json": json.dumps(catalogue, indent=2) + "\n",
        "index.html": render_html(rows, catalogue),
        "state.json": json.dumps(render_state(catalogue), indent=2) + "\n",
    }


def validate(outputs, pulse: Path):
    sys.path.insert(0, str(pulse.resolve()))
    from pulse.summary import classify

    outcome, errors = classify(json.loads(outputs["catalogue.json"]), ("profiling-summary", 2))
    if outcome != "ok":
        raise ValueError(f"Pulse v2 validation {outcome}: " + "; ".join(errors[:20]))
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--validate-with", type=Path, help="PyAutoPulse checkout whose reader validates the feed"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    outputs = render(root)
    if args.validate_with:
        print(f"pulse validator: {validate(outputs, args.validate_with)} (profiling-summary@2)")
    out = root / "dashboard"
    stale = [n for n in OUTPUTS if not (out / n).is_file() or (out / n).read_text() != outputs[n]]
    if args.check:
        if stale:
            print(f"catalogue: STALE ({', '.join(stale)}); run build_catalogue.py")
            return 1
        print("catalogue: current")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (out / name).write_text(outputs[name])
    doc = json.loads(outputs["catalogue.json"])
    print(
        f"catalogue: wrote {len(doc['setups'])} setups, {len(doc['records'])} records, "
        f"{len(doc['selections'])} selections"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
