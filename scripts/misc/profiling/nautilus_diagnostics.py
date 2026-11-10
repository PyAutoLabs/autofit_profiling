"""Bounded, copied-checkpoint diagnostics, deliberately outside Pulse."""

import contextlib
import copy
import functools
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from profiling.nautilus_campaign import environment_key
from profiling.nautilus_manifest import RunConfig, fingerprint
from profiling.nautilus_resources import atomic_json

PROTOCOL = "nautilus-copied-checkpoint-diagnostics@1"
LIMITS = {"budget_s": 900, "timeout_s": 120, "memory_mib": 4096}
OUTPUT_LIMITS = {"budget_s": 180, "timeout_s": 120, "memory_mib": 4096}
# These are conditional repetitions with fixed learned bounds, NOT new explorations.
CASES = [(5, None), (10, None), (30, None), (30, 1), (30, 2)]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_identity(environment):
    value = copy.deepcopy(environment)
    value["sources"].pop("producer")
    return value


def assert_environment(actual, expected):
    if actual != expected:
        differences = sorted(
            k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k)
        )
        raise ValueError(f"environment changed: {differences}")


def check_inputs(plan):
    for case in plan["cases"]:
        for key in ("checkpoint", "result"):
            if sha256(case[key]) != case[key + "_sha256"]:
                raise ValueError(f"original {key} changed: {case['id']}")


def validate_plan(plan):
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if plan["sha256"] != fingerprint(body):
        raise ValueError("plan changed")
    if plan.get("scope") not in (None, "output-only", "plot-comparison"):
        raise ValueError("unauthorized scope")
    limits = OUTPUT_LIMITS if plan.get("scope") == "output-only" else LIMITS
    if plan.get("scope") == "plot-comparison":
        from profiling.nautilus_plot_comparison import LIMITS as plot_limits
        from profiling.nautilus_plot_comparison import POLICIES

        limits = plot_limits
        if [c.get("plot_policy") for c in plan["cases"]] != POLICIES:
            raise ValueError("unauthorized plot policies")
    call_limit = 1 if plan.get("scope") == "plot-comparison" else 0
    if any(c.get("summary_timing_call_limit", 0) != call_limit for c in plan["cases"]):
        raise ValueError("unauthorized summary timing calls")
    if plan["protocol"] != PROTOCOL or plan["limits"] != limits:
        raise ValueError("unauthorized protocol or limits")
    expected = [("evidence", d, seed) for d, seed in CASES] + [("output", 30, None)]
    if plan.get("scope") == "output-only":
        expected = [("output", 30, None)]
    if plan.get("scope") == "plot-comparison":
        expected = [("output", 30, None)] * 3
    if [(c["arm"], c["dimension"], c["resample_seed"]) for c in plan["cases"]] != expected:
        raise ValueError("unauthorized case matrix")
    if not 0 <= plan["preflight_s"] < limits["budget_s"]:
        raise ValueError("preflight exhausted budget")


def prepare(archive, *, output_only=False, plot_comparison=False):
    start = time.monotonic()
    output_only = output_only or plot_comparison
    from profiling.nautilus_baseline import provenance
    from profiling.nautilus_target import CorrelatedGaussianTarget

    environment = environment_key(provenance())
    old_plan = json.loads((archive / "plan.json").read_text())
    assert_environment(runtime_identity(environment), runtime_identity(old_plan["environment"]))
    inputs = {}
    for result in archive.glob("case-*/result.json"):
        row = json.loads(result.read_text())
        if row.get("status") != "success":
            continue
        config = row["manifest"]["config"]
        if config["adapter"] != "direct" or config["trace"] or config["seed"] != 0:
            continue
        dimension = config["dimension"]
        if output_only and dimension != 30:
            continue
        if config != RunConfig(dimension=dimension, n_live=400, trace=False).to_dict():
            raise ValueError("archive settings differ")
        if row["manifest"]["target_sha256"] != fingerprint(
            CorrelatedGaussianTarget(dimension).truth()
        ):
            raise ValueError("target changed")
        if dimension in inputs:
            raise ValueError("duplicate archived control")
        checkpoint = result.parent / "checkpoint.hdf5"
        inputs[dimension] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "result": str(result.resolve()),
            "result_sha256": sha256(result),
            "target_sha256": row["manifest"]["target_sha256"],
            "original_n_like": row["n_like"],
            "original_logZ_error": row["log_evidence_error"],
            "original_posterior_sha256": row["posterior_sha256"],
        }
    # Read-only runtime preflight: validate native restoration and every hook before attempts.
    from autofit.non_linear.paths.directory import DirectoryPaths
    from autofit.non_linear.samples.pdf import SamplesPDF
    from autofit.non_linear.search.abstract_search import NonLinearSearch
    from autofit.non_linear.search.updater import SearchUpdater
    from nautilus import Sampler

    for owner, names in (
        (DirectoryPaths, ("save_samples", "save_samples_summary")),
        (SamplesPDF, ("summary", "samples_drawn_randomly_via_pdf_from")),
        (
            SearchUpdater,
            ("_save_samples", "_compute_latent_samples", "visualize", "_profile_and_summarize"),
        ),
        (NonLinearSearch, ("samples_from", "pre_fit_output", "post_fit_output")),
    ):
        for name in names:
            if not callable(getattr(owner, name, None)):
                raise ValueError(f"missing output hook: {owner.__name__}.{name}")
    for dimension, entry in inputs.items():
        target = CorrelatedGaussianTarget(dimension)
        sampler = Sampler(
            target.prior,
            target,
            filepath=entry["checkpoint"],
            **RunConfig(dimension=dimension, n_live=400, trace=False).constructor(),
        )
        restored = estimate(sampler, target)
        entry["original_selected_samples"] = restored["selected_samples"]
        if restored["posterior_sha256"] != entry["original_posterior_sha256"] or any(
            fresh_counts(sampler)
        ):
            raise ValueError("checkpoint restoration differs from archived receipt")
        discard_saved_exploration(sampler)
        if sum(sampler.shell_n) != 0:
            raise ValueError("preflight exploration exclusion failed")
    cases = []
    matrix = (
        [("output", 30, None)]
        if output_only
        else [("evidence", d, s) for d, s in CASES] + [("output", 30, None)]
    )
    for arm, dimension, seed in matrix:
        cases.append(
            {
                "id": f"{arm}-d{dimension}-r{seed if seed is not None else 'saved'}",
                "arm": arm,
                "dimension": dimension,
                "resample_seed": seed,
                **inputs[dimension],
            }
        )
    plan = {
        "protocol": PROTOCOL,
        "limits": LIMITS,
        "environment": environment,
        "cases": cases,
        "preflight_s": time.monotonic() - start,
        "interpretation": "conditional on seed-0 learned bounds; not independent full searches",
    }
    if output_only:
        plan.update(
            scope="output-only",
            limits=OUTPUT_LIMITS,
            cases=[c for c in cases if c["arm"] == "output"],
            interpretation="Output-only follow-up authorized after halted pilot; no new likelihood evaluations",
        )
    if plot_comparison:
        from profiling.nautilus_plot_comparison import LIMITS as plot_limits
        from profiling.nautilus_plot_comparison import POLICIES, font_cache

        plan.update(
            scope="plot-comparison",
            limits=plot_limits,
            cases=[
                {
                    **cases[0],
                    "id": "plot-" + policy["name"],
                    "plot_policy": copy.deepcopy(policy),
                    "summary_timing_call_limit": 1,
                }
                for policy in POLICIES
            ],
            font_cache=font_cache(),
            interpretation="Three output-only plot policies; original posterior and non-plot output held fixed",
        )
        plan["preflight_s"] = time.monotonic() - start
    plan["sha256"] = fingerprint(plan)
    validate_plan(plan)
    return plan


class StageLog:
    """Entry records survive termination; incomplete stages remain visible."""

    def __init__(self, path):
        self.path = Path(path)
        self.start = time.monotonic()
        self.next_id = 0

    def write(self, **event):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"t_s": time.monotonic() - self.start, **event}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @contextlib.contextmanager
    def stage(self, name):
        ident = self.next_id
        self.next_id += 1
        self.write(id=ident, stage=name, event="enter")
        try:
            yield
        except BaseException:
            self.write(id=ident, stage=name, event="error")
            raise
        else:
            self.write(id=ident, stage=name, event="exit")

    @contextlib.contextmanager
    def patch(self, owner, name, label=None):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.stage(label or name):
                return original(*args, **kwargs)

        setattr(owner, name, wrapped)
        try:
            yield
        finally:
            setattr(owner, name, original)


def stage_summary(path):
    pending, completed = {}, []
    if not Path(path).exists():
        return {"completed": [], "incomplete": []}
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed writer may leave its last line incomplete.
        if event["event"] == "enter":
            pending[event["id"]] = event
        else:
            begin = pending.pop(event["id"])
            completed.append(
                {
                    "stage": event["stage"],
                    "status": event["event"],
                    "wall_s": event["t_s"] - begin["t_s"],
                }
            )
    return {"completed": completed, "incomplete": list(pending.values())}


def fresh_counts(sampler):
    return [len(p) - int(end) for p, end in zip(sampler.points, sampler.shell_end_exp)]


def discard_saved_exploration(sampler):
    if not sampler.explored:
        raise ValueError("checkpoint has not finished exploration")
    sampler.discard_exploration = True  # run's argument alone is ineffective on resume.
    if not sampler.discard_exploration or list(sampler.shell_n) != fresh_counts(sampler):
        raise ValueError("exploration exclusion did not take effect")


def estimate(sampler, target):
    import numpy as np

    points, log_weights, log_l = sampler.posterior()
    weights = np.exp(log_weights)
    mean = weights @ points
    centered = points - mean
    covariance = (centered * weights[:, None]).T @ centered
    digest = hashlib.sha256()
    for array in (points, log_weights, log_l):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    return {
        "logZ": float(sampler.log_z),
        "logZ_error": float(sampler.log_z - target.log_evidence),
        "n_eff": float(sampler.n_eff),
        "posterior_sha256": digest.hexdigest(),
        "n_like": int(sampler.n_like),
        "selected_samples": int(sum(sampler.shell_n)),
        "fresh_samples_per_shell": fresh_counts(sampler),
        "max_standardized_mean_error": float(
            np.max(
                np.abs(mean - target.posterior_mean) / np.sqrt(np.diag(target.posterior_covariance))
            )
        ),
        "covariance_relative_error": float(
            np.linalg.norm(covariance - target.posterior_covariance)
            / np.linalg.norm(target.posterior_covariance)
        ),
        "shell_log_v": sampler.shell_log_v.tolist(),
    }


def evidence(case, checkpoint, target, log):
    import numpy as np
    from autofit.non_linear.search.nest.nautilus.search import _large_attribute_hdf5
    from nautilus import Sampler

    config = RunConfig(dimension=case["dimension"], n_live=400, trace=False)
    with _large_attribute_hdf5():
        sampler = Sampler(target.prior, target, filepath=str(checkpoint), **config.constructor())
        before = estimate(sampler, target)
        if before["posterior_sha256"] != case["original_posterior_sha256"]:
            raise ValueError("input checkpoint posterior differs")
        if sampler.n_like != case["original_n_like"] or any(fresh_counts(sampler)):
            raise ValueError("unexpected input checkpoint state")
        # Change the state of the shared generator, not the object used by every bound.
        if case["resample_seed"] is not None:
            sampler.rng.bit_generator.state = np.random.default_rng(
                case["resample_seed"]
            ).bit_generator.state
        discard_saved_exploration(sampler)
        # Incremental native shell writes omit the discard flag.
        # Persist the changed mode first so a capped copy can be reopened correctly.
        sampler.write(str(checkpoint), overwrite=True)
        with log.stage("fresh_sampling"):
            converged = sampler.run(
                n_eff=10000, n_shell=100, discard_exploration=True, verbose=False
            )
        if not converged or sampler.n_eff < 10000 or min(fresh_counts(sampler)) < 100:
            raise ValueError("fresh sampling did not meet stopping criteria")
        fresh = estimate(sampler, target)
        sampler.discard_exploration = False
        combined = estimate(sampler, target)
    return {
        "before": before,
        "fresh_only": fresh,
        "including_exploration": combined,
        "stopping": {"n_eff": 10000, "n_shell": 100, "discard_exploration": True},
        "new_likelihood_calls": int(sampler.n_like - before["n_like"]),
    }


def output_replay(case, checkpoint, target, directory, log):
    from autofit.non_linear.paths.directory import DirectoryPaths
    from autofit.non_linear.samples.pdf import SamplesPDF
    from autofit.non_linear.search.abstract_search import NonLinearSearch
    from autofit.non_linear.search.updater import SearchUpdater
    from nautilus import Sampler

    from profiling.nautilus_baseline import run_wrapped

    original_init = Sampler.__init__
    original_evaluate = Sampler.evaluate_likelihood
    captured, evaluations = [], []

    def resume(instance, *args, **kwargs):
        kwargs.update(filepath=str(checkpoint), resume=True)
        original_init(instance, *args, **kwargs)
        captured.append(instance)
        if instance.n_like != case["original_n_like"]:
            raise ValueError("output replay did not load checkpoint")

    def forbidden(*args, **kwargs):
        evaluations.append(True)
        raise RuntimeError("output-only replay attempted likelihood evaluation")

    from profiling.nautilus_plot_comparison import SummaryTimingGate, replaced

    gate = SummaryTimingGate(target, forbidden, case.get("summary_timing_call_limit", 0))
    original_summary = SearchUpdater._profile_and_summarize

    def timed_summary(updater, *args, **kwargs):
        with gate.summary():
            result = original_summary(updater, *args, **kwargs)
        # Outside the likelihood timer: durable accounting must not time disk I/O as likelihood work.
        atomic_json(
            directory / "summary-timing.json",
            {
                "calls": gate.calls,
                "limit": gate.limit,
                "parameters": gate.parameters,
                "value": gate.value,
            },
        )
        return result

    class NoLikelihood:
        @property
        def prior(self):
            return target.prior

        def __call__(self, parameters):
            return gate.evaluate(parameters)

    Sampler.__init__ = resume
    Sampler.evaluate_likelihood = forbidden
    try:
        plot_receipt = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(replaced(SearchUpdater, "_profile_and_summarize", timed_summary))
            if "plot_policy" in case:
                from profiling.nautilus_plot_comparison import plot_scope

                plot_receipt = stack.enter_context(plot_scope(case, directory, log))
            for owner, names in (
                (
                    SearchUpdater,
                    (
                        "_save_samples",
                        "_compute_latent_samples",
                        "visualize",
                        "_profile_and_summarize",
                    ),
                ),
                (SamplesPDF, ("summary", "samples_drawn_randomly_via_pdf_from")),
                (NonLinearSearch, ("samples_from", "pre_fit_output", "post_fit_output")),
                (DirectoryPaths, ("save_samples", "save_samples_summary")),
                (Sampler, ("run",)),
            ):
                for name in names:
                    stack.enter_context(log.patch(owner, name))
            with log.stage("wrapped_replay"):
                run_wrapped(
                    RunConfig(adapter="pyautofit", dimension=30, n_live=400, trace=False),
                    NoLikelihood(),
                    directory,
                )
        if evaluations or len(captured) != 1 or captured[0].n_like != case["original_n_like"]:
            raise ValueError("output-only replay invalid")
        result = estimate(captured[0], target)
        # Baseline hashes include exactly the same arrays and normalized log weights.
        if result["posterior_sha256"] != case["original_posterior_sha256"]:
            raise ValueError("output replay changed posterior")
        if gate.calls != gate.limit:
            raise ValueError("summary timing call count differs from approved plan")
        if gate.limit:
            import numpy as np

            points, _, log_l = captured[0].posterior()
            best = int(np.argmax(log_l))
            if not np.array_equal(gate.parameters, points[best]) or not np.isclose(
                gate.value, log_l[best], rtol=0, atol=1e-10
            ):
                raise ValueError(
                    "summary timing point/value differs from stored maximum likelihood"
                )
        return {
            "new_likelihood_calls": gate.calls,
            "new_sampler_likelihood_calls": 0,
            "summary_timing_calls": gate.calls,
            "posterior": result,
            "plot": plot_receipt,
        }
    finally:
        Sampler.__init__ = original_init
        Sampler.evaluate_likelihood = original_evaluate


def child(plan, case, directory):
    from profiling.nautilus_baseline import provenance
    from profiling.nautilus_target import CorrelatedGaussianTarget

    check_inputs(plan)
    before = provenance()
    assert_environment(environment_key(before), plan["environment"])
    if any(s["dirty"] for s in before["sources"].values()):
        raise ValueError("measurement source must be committed and clean")
    if plan.get("scope") == "plot-comparison":
        from profiling.nautilus_plot_comparison import font_cache

        if font_cache() != plan["font_cache"]:
            raise ValueError("font-cache state changed")
    target = CorrelatedGaussianTarget(case["dimension"])
    if fingerprint(target.truth()) != case["target_sha256"]:
        raise ValueError("target changed")
    checkpoint = directory / "checkpoint.hdf5"
    shutil.copyfile(case["checkpoint"], checkpoint)
    log = StageLog(directory / "stages.jsonl")
    if case["arm"] == "evidence":
        result = evidence(case, checkpoint, target, log)
    else:
        result = output_replay(case, checkpoint, target, directory, log)
    check_inputs(plan)
    assert_environment(environment_key(provenance()), plan["environment"])
    return {"status": "success", "case": case["id"], "original_unchanged": True, **result}
