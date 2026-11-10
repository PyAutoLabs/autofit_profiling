"""Dimension-safe experimental summaries; never emits profiling-summary records."""

import math
import statistics
from collections import defaultdict

from profiling.nautilus_campaign import compare_pair, row_identity


def diagnostics(row):
    truth = row["manifest"]["target"]
    mean_error = row["posterior_mean_error"]
    covariance = row["posterior_covariance"]
    reference = truth["covariance"]
    return {
        "max_abs_mean_error": max(abs(v) for v in mean_error),
        "max_abs_standardized_mean_error": max(
            abs(v) / math.sqrt(reference[i][i]) for i, v in enumerate(mean_error)
        ),
        "covariance_relative_frobenius_error": math.sqrt(
            sum(
                (a - b) ** 2
                for ar, br in zip(covariance, reference, strict=True)
                for a, b in zip(ar, br, strict=True)
            )
            / sum(b * b for br in reference for b in br)
        ),
        "logZ_error": row["log_evidence_error"],
        "n_like": row["n_like"],
        "n_eff": row["n_eff"],
        "calls_per_ess": row["n_like"] / row["n_eff"],
        "wall_s": row["wall_s"],
        "sampled_group_peak_rss_bytes": row["supervision"]["sampled_group_peak_rss_bytes"],
    }


def summarize_rows(rows):
    groups, pairs, adapters = defaultdict(list), defaultdict(dict), defaultdict(dict)
    seen = set()
    for row in rows:
        full = row_identity(row, seed=True, trace=True)
        if full in seen:
            raise ValueError("duplicate selected case")
        seen.add(full)
        config = row["manifest"]["config"]
        pairs[row_identity(row, seed=True)][config["trace"]] = row
        adapters[row_identity(row, adapter=False, seed=True, trace=True)][config["adapter"]] = row
        groups[row_identity(row, trace=True)].append(row)
    paired, missing, adapter_pairs = [], [], 0
    for key, pair in pairs.items():
        if set(pair) != {False, True}:
            missing.append(
                {"pair_id": key, "present_trace_modes": sorted(pair), "status": "incomplete"}
            )
            continue
        on, off = pair[True], pair[False]
        compare_pair(on, off)
        paired.append(
            {
                "pair_id": key,
                "config": on["manifest"]["config"],
                "trace_wall_ratio": on["wall_s"] / off["wall_s"],
                "posterior_and_calls_identical": True,
            }
        )
    for pair in adapters.values():
        if set(pair) == {"direct", "pyautofit"}:
            compare_pair(pair["direct"], pair["pyautofit"])
            adapter_pairs += 1
    summaries = []
    for key, group in groups.items():
        config = dict(group[0]["manifest"]["config"])
        config.pop("seed")
        metrics = [diagnostics(row) for row in group]
        components = {
            name: statistics.median(row["components_s"][name] / row["wall_s"] for row in group)
            for name in group[0]["components_s"]
        }
        summaries.append(
            {
                "group_id": key,
                "config": config,
                "n": len(group),
                "seeds": [row["manifest"]["config"]["seed"] for row in group],
                "metrics": {
                    name: {
                        "median": statistics.median(m[name] for m in metrics),
                        "min": min(m[name] for m in metrics),
                        "max": max(m[name] for m in metrics),
                    }
                    for name in metrics[0]
                },
                "median_component_shares": components,
            }
        )
    return {
        "groups": summaries,
        "trace_control_pairs": paired,
        "incomplete_pairs": missing,
        "matching_cross_adapter_pairs": adapter_pairs,
        "qualification": "experimental; unreviewed; no scientific acceptance or speedup verdict",
    }
