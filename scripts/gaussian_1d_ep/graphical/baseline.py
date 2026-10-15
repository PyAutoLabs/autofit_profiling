"""Graphical baseline: shared Gaussian centre across N datasets via one joint Dynesty fit.

Ported from ``autofit_workspace_developer/graphical/fit.py`` (B4b of the
search-extensibility epic). The model, priors and search (one ``DynestyStatic``,
``sample="rwalk"``, ``nlive`` from the cell config, on ``factor_graph.global_prior_model``)
are unchanged; only the measurement harness differs (see ``shared.py``).

Usage::

    source activate.sh
    python scripts/gaussian_1d_ep/graphical/baseline.py --n all   # N3, N10, N30 serially
    python scripts/gaussian_1d_ep/graphical/baseline.py --n 3     # one cell

Each N writes ``results/gaussian_1d_ep/graphical/N<n>.json`` (schema
``autofit-profiling-baseline``). Sanity: parameter recovery (|z| <= 3 against the median
PDF and the mean 1-sigma error) for the shared centre and every normalization and sigma,
plus the max-log-likelihood rule (max logL >= truth sum - 5 nats per dataset).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").is_file():
            return parent
    raise RuntimeError("ruff.toml root sentinel not found above this script")


ROOT = _repo_root()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "misc"), str(Path(__file__).resolve().parents[1])]

import autofit as af
import shared

METHOD = "graphical"
SEARCH = "DynestyStatic (joint)"
# nlive / walks / bound come from scripts/gaussian_1d_ep/config/non_linear/nest.yaml.
SEARCH_SETTINGS = {
    "nlive": 50,
    "sample": "rwalk",
    "walks": 5,
    "bound": "multi",
    "number_of_cores": 1,
}

# Committed developer baselines (laptop CPU, PyAutoFit of 2026-05-20, host unrecorded),
# copied verbatim from autofit_workspace_developer/graphical/profiles/{baseline,postfix_*}.json
# at 07b92ace / 27f0a3a9 so the comparison survives the retirement of that package.
REFERENCE_SOURCE = "autofit_workspace_developer/graphical/profiles/baseline.json @ 07b92ace3b3afa51afb914c17972f6fcbe17d994"
_POSTFIX = "autofit_workspace_developer/graphical/profiles/postfix_N{n}_summary.json @ 27f0a3a9fac4484e71b2aeaeb877a9c77204fc4f"
REFERENCE = {
    3: {
        "wall_time_s": 22.7954490000011,
        "peak_rss_kb": 569152,
        "max_log_likelihood": -125.14263334344666,
        "truth_log_likelihood_sum": -128.39683874532219,
        "sanity_pass": True,
        "postfix_wall_time_s": 5.640972900000634,
        "postfix_source": _POSTFIX.format(n=3),
    },
    10: {
        "wall_time_s": 50.74623470000006,
        "peak_rss_kb": 864280,
        "max_log_likelihood": -519.3259230865922,
        "truth_log_likelihood_sum": -472.2530881754556,
        "sanity_pass": True,
        "postfix_wall_time_s": 14.934478999999556,
        "postfix_source": _POSTFIX.format(n=10),
    },
    30: {
        "wall_time_s": 132.83731699999953,
        "peak_rss_kb": 3050040,
        "max_log_likelihood": -1658.6547285227005,
        "truth_log_likelihood_sum": -1500.5605499266132,
        "sanity_pass": False,
    },
}


def measure(n: int) -> dict:
    sw_before, _, _ = shared.software()
    output_root = shared.setup_autofit(METHOD)
    run_name = f"N{n}"
    shared.simulate(n)
    data_list, noise_map_list, truths = shared.load_datasets(n)

    analysis_list = [
        af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)
        for data, noise_map in zip(data_list, noise_map_list)
    ]
    _, model_list = shared.make_model_list(n)
    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model, analysis=analysis)
        for model, analysis in zip(model_list, analysis_list)
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    run_dir = shared.fresh_output(output_root, run_name)
    search = af.DynestyStatic(
        path_prefix=Path(shared.SAMPLE_NAME),
        name=run_name,
        sample="rwalk",
        number_of_cores=1,
    )

    load_before = shared.load_average()
    t0 = time.perf_counter()
    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)
    wall_s = time.perf_counter() - t0
    rss_kib = shared.peak_rss_kib()

    max_log_likelihood = float(result.samples.max_log_likelihood_sample.log_likelihood)
    truth_ll_sum = float(sum(g["truth_log_likelihood"] for g in truths))
    median_instance = result.samples.median_pdf()
    errors_instance = result.samples.errors_at_sigma(sigma=1.0)

    sanity = []
    # Shared centre -- pulled from the first factor; identical across all i.
    centre_lower, centre_upper = errors_instance[0].gaussian.centre
    sanity.append(
        shared.check_parameter_recovery(
            "shared centre",
            float(median_instance[0].gaussian.centre),
            float((centre_lower + centre_upper) / 2.0),
            truths[0]["centre"],
        )
    )
    for i, gt in enumerate(truths):
        nl, nu = errors_instance[i].gaussian.normalization
        sanity.append(
            shared.check_parameter_recovery(
                f"dset_{i} normalization",
                float(median_instance[i].gaussian.normalization),
                float((nl + nu) / 2.0),
                gt["normalization"],
            )
        )
        sl, su = errors_instance[i].gaussian.sigma
        sanity.append(
            shared.check_parameter_recovery(
                f"dset_{i} sigma",
                float(median_instance[i].gaussian.sigma),
                float((sl + su) / 2.0),
                gt["sigma"],
            )
        )
    sanity.append(shared.check_max_log_likelihood(max_log_likelihood, truth_ll_sum, n))
    sanity_pass, checks = shared.summarise_sanity(sanity, header=f"graphical N={n}")

    correctness = {
        "truth_log_likelihood_sum": truth_ll_sum,
        "max_log_likelihood": max_log_likelihood,
        "sanity_pass": sanity_pass,
        "sanity_rule": (
            "every parameter |median - truth| <= 3 x mean 1-sigma error, and "
            "max logL >= truth logL sum - 5 nats per dataset"
        ),
        "sanity_checks": checks,
    }
    reference = shared.reference_block(REFERENCE_SOURCE, REFERENCE[n], wall_s, correctness)
    reference["max_log_likelihood"] = REFERENCE[n]["max_log_likelihood"]
    return shared.baseline_row(
        method=METHOD,
        n=n,
        label=f"joint DynestyStatic search.fit on factor_graph.global_prior_model, {n} datasets",
        search=SEARCH,
        search_settings={**SEARCH_SETTINGS, "n_datasets": n},
        wall_s=wall_s,
        rss_kib=rss_kib,
        disk_bytes=shared.output_dir_bytes(run_dir),
        correctness=correctness,
        reference=reference,
        sw_before=sw_before,
        load_before=load_before,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--n", default="all", help="3, 10, 30 or all (each N in its own process)")
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args(argv)
    if args.n == "all":
        extra = ["--no-write"] if args.no_write else []
        return shared.run_all(Path(__file__), shared.N_VALUES, extra)
    n = int(args.n)
    row = measure(n)
    ref = row["reference"]
    print(
        f"graphical N={n}: wall {row['wall_s']:.1f} s vs {ref['wall_s']:.1f} s "
        f"({ref['wall_drift']['status']}, x{ref['wall_drift']['ratio']}); peak RSS "
        f"{row['peak_rss_kib'] / 1024:.0f} MiB; max logL {row['correctness']['max_log_likelihood']:.3f} "
        f"(committed {ref['max_log_likelihood']:.3f}); truth logL sum match "
        f"{ref['truth_log_likelihood_sum_matches']}; sanity {row['correctness']['sanity_pass']} "
        f"(committed {ref['sanity_pass']})"
    )
    if not args.no_write:
        shared.write_row(shared.ROOT / "results" / shared.DATASET / METHOD / f"N{n}.json", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
