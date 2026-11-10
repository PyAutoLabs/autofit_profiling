"""Admission rejects enlarged budgets, sampling, thinning and reordered pairs."""

import copy
import importlib.util
from pathlib import Path

import pytest
from profiling.nautilus_manifest import fingerprint

SOURCE = Path(__file__).resolve().parents[2] / "analytic_gaussian/nautilus/compact_benchmark.py"
spec = importlib.util.spec_from_file_location("compact_benchmark", SOURCE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def plan():
    result = {
        "protocol": benchmark.PROTOCOL,
        "limits": benchmark.LIMITS.copy(),
        "cases": [
            {
                "id": f"pair-{i // 2 + 1}-{backend}",
                "backend": backend,
                "arm": "output",
                "dimension": 30,
                "original_n_like": 34100,
                "summary_timing_call_limit": 1,
                "original_selected_samples": 33984,
                "plot_policy": {
                    "name": "thirty-small",
                    "indices": list(range(30)),
                    "figsize": [12, 12],
                    "dpi": 100,
                },
            }
            for i, backend in enumerate(benchmark.ORDER)
        ],
    }
    result["sha256"] = fingerprint(result)
    return result


@pytest.mark.parametrize(
    "change", ["budget", "order", "sample", "timing", "plot", "sampling", "hash"]
)
def test_scope_change_rejected(change):
    p = plan()
    benchmark.validate(p)
    if change == "budget":
        p["limits"]["budget_s"] += 1
    elif change == "order":
        p["cases"].reverse()
    elif change == "sample":
        p["cases"][0]["original_selected_samples"] -= 1
    elif change == "timing":
        p["cases"][0]["summary_timing_call_limit"] += 1
    elif change == "plot":
        p["cases"][0]["plot_policy"]["indices"] = [0, 1]
    elif change == "sampling":
        p["cases"][0]["arm"] = "evidence"
    else:
        p["sha256"] = "bad"
    if change != "hash":
        p["sha256"] = fingerprint({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError):
        benchmark.validate(p)


def test_one_shot_directory_cannot_be_reused(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(FileExistsError):
        benchmark.run(None, None, root)


def test_auxiliary_metadata_paths_never_reintroduce_library_roots(tmp_path):
    names = ["PyAutoFit", "PyAutoNerves", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens", "unknown"]
    paths = [str(tmp_path / name) for name in names]
    assert benchmark.metadata_paths(benchmark.os.pathsep.join(paths)) == paths[2:5]
