"""EP baseline: shared Gaussian centre across N datasets via per-factor Dynesty + Laplace EP.

Ported from ``autofit_workspace_developer/ep/fit.py`` (B4b of the search-extensibility
epic). The model, priors, search (``DynestyStatic(nlive=100, sample="rwalk")`` per
factor), ``LaplaceOptimiser`` cavity updates, ``EPHistory(kl_tol=0.05)`` and
``max_steps=5`` are unchanged; only the measurement harness differs (see ``shared.py``).

Usage::

    source activate.sh
    python scripts/gaussian_1d_ep/ep/baseline.py --n all     # N3, N10, N30 serially
    python scripts/gaussian_1d_ep/ep/baseline.py --n 3       # one cell

Each N writes ``results/gaussian_1d_ep/ep/N<n>.json`` (schema
``autofit-profiling-baseline``). Sanity compares the EP mean-field posterior of the
shared centre and each factor's normalization and sigma with the truth (|z| <= 3); EP
produces no joint max log likelihood, so the max-LL check is skipped, as in the original.
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

METHOD = "ep"
SEARCH = "DynestyStatic+LaplaceOptimiser (EP)"
SEARCH_SETTINGS = {
    "nlive": 100,
    "sample": "rwalk",
    "optimiser": "LaplaceOptimiser",
    "kl_tol": 0.05,
    "max_steps": 5,
}

# Committed developer baselines (laptop CPU, PyAutoFit of 2026-05-20, host unrecorded),
# copied verbatim from autofit_workspace_developer/ep/profiles/{baseline,postfix_N3_summary}.json
# at 07b92ace / 98f260e5 so the comparison survives the retirement of that package.
REFERENCE_SOURCE = "autofit_workspace_developer/ep/profiles/baseline.json @ 07b92ace3b3afa51afb914c17972f6fcbe17d994"
REFERENCE = {
    3: {
        "wall_time_s": 251.8654706999987,
        "peak_rss_kb": 613128,
        "truth_log_likelihood_sum": -128.39683874532219,
        "sanity_pass": True,
        "postfix_wall_time_s": 76.32506300000023,
        "postfix_source": "autofit_workspace_developer/ep/profiles/postfix_N3_summary.json @ 98f260e5334d344a0d34b4430777b67f46f663b4",
    },
    10: {
        "wall_time_s": 566.8557853000002,
        "peak_rss_kb": 688632,
        "truth_log_likelihood_sum": -472.2530881754556,
        "sanity_pass": True,
    },
    30: {
        "wall_time_s": 1758.7042256000004,
        "peak_rss_kb": 1149596,
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
    centre_shared_prior, model_list = shared.make_model_list(n)

    run_dir = shared.fresh_output(output_root, run_name)
    paths = af.DirectoryPaths(path_prefix=Path(shared.SAMPLE_NAME), name=run_name)
    search = af.DynestyStatic(paths=paths, nlive=100, sample="rwalk")
    analysis_factor_list = [
        af.AnalysisFactor(
            prior_model=model, analysis=analysis, optimiser=search, name=f"dataset_{i}"
        )
        for i, (model, analysis) in enumerate(zip(model_list, analysis_list))
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)
    laplace = af.LaplaceOptimiser()

    load_before = shared.load_average()
    t0 = time.perf_counter()
    result = factor_graph.optimise(
        optimiser=laplace,
        paths=paths,
        ep_history=af.EPHistory(kl_tol=0.05),
        max_steps=5,
    )
    wall_s = time.perf_counter() - t0
    rss_kib = shared.peak_rss_kib()

    mean_field = result.updated_ep_mean_field.mean_field
    sanity = []
    centre_msg = mean_field[centre_shared_prior]
    sanity.append(
        shared.check_parameter_recovery(
            "shared centre", float(centre_msg.mean), float(centre_msg.sigma), truths[0]["centre"]
        )
    )
    for i, (model, gt) in enumerate(zip(model_list, truths)):
        norm_msg = mean_field[model.gaussian.normalization]
        sanity.append(
            shared.check_parameter_recovery(
                f"dset_{i} normalization",
                float(norm_msg.mean),
                float(norm_msg.sigma),
                gt["normalization"],
            )
        )
        sig_msg = mean_field[model.gaussian.sigma]
        sanity.append(
            shared.check_parameter_recovery(
                f"dset_{i} sigma", float(sig_msg.mean), float(sig_msg.sigma), gt["sigma"]
            )
        )
    sanity_pass, checks = shared.summarise_sanity(sanity, header=f"EP N={n}")

    correctness = {
        "truth_log_likelihood_sum": float(sum(g["truth_log_likelihood"] for g in truths)),
        "max_log_likelihood": None,
        "max_log_likelihood_reason": "EP returns marginal messages, not a joint max-likelihood sample",
        "sanity_pass": sanity_pass,
        "sanity_rule": "every parameter |mean - truth| <= 3 sigma of its EP mean-field message",
        "sanity_checks": checks,
    }
    return shared.baseline_row(
        method=METHOD,
        n=n,
        label=f"factor_graph.optimise EP, {n} datasets (per-factor DynestyStatic + Laplace cavity)",
        search=SEARCH,
        search_settings={**SEARCH_SETTINGS, "n_datasets": n},
        wall_s=wall_s,
        rss_kib=rss_kib,
        disk_bytes=shared.output_dir_bytes(run_dir),
        correctness=correctness,
        reference=shared.reference_block(REFERENCE_SOURCE, REFERENCE[n], wall_s, correctness),
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
        f"EP N={n}: wall {row['wall_s']:.1f} s vs {ref['wall_s']:.1f} s "
        f"({ref['wall_drift']['status']}, x{ref['wall_drift']['ratio']}); peak RSS "
        f"{row['peak_rss_kib'] / 1024:.0f} MiB; truth logL sum match "
        f"{ref['truth_log_likelihood_sum_matches']}; sanity {row['correctness']['sanity_pass']} "
        f"(committed {ref['sanity_pass']})"
    )
    if not args.no_write:
        shared.write_row(shared.ROOT / "results" / shared.DATASET / METHOD / f"N{n}.json", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
