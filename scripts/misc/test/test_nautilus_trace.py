import copy

import pytest
from profiling.nautilus_trace import reconcile, summarize


def fixture():
    batches = [{"id": 0, "count": 2, "start_ns": 0, "finish_ns": 100}]
    events = [
        {
            "id": [0, 0],
            "pid": 1,
            "submit_ns": 1,
            "start_ns": 2,
            "finish_ns": 90,
            "collect_ns": 95,
            "cpu_ns": 80,
            "peak_rss_kib": 100,
        },
        {
            "id": [0, 1],
            "pid": 2,
            "submit_ns": 1,
            "start_ns": 3,
            "finish_ns": 20,
            "collect_ns": 95,
            "cpu_ns": 15,
            "peak_rss_kib": 200,
        },
    ]
    return events, batches


def test_reordered_completions_and_overlap():
    events, batches = fixture()
    result = summarize(events[::-1], batches)
    assert result["accounting"] == "complete"
    assert result["completed"] == 2
    assert result["batches"][0]["worker_tail_s"] == pytest.approx(70e-9)
    assert sum(w["service_s"] for w in result["workers"].values()) > result["batches"][0]["wall_s"]


@pytest.mark.parametrize("fault", ["duplicate", "missing", "negative", "unexpected", "outside"])
def test_bad_accounting_rejected(fault):
    events, batches = fixture()
    if fault == "duplicate":
        events.append(copy.deepcopy(events[0]))
    elif fault == "missing":
        events.pop()
    elif fault == "negative":
        events[0]["start_ns"] = 91
    elif fault == "unexpected":
        events[0]["id"][0] = 1
    else:
        batches[0]["finish_ns"] = 94
    with pytest.raises(ValueError):
        reconcile(events, batches)


def test_single_worker_tail_unavailable():
    events, batches = fixture()
    events[1].update(pid=1, start_ns=91, finish_ns=92)
    assert summarize(events, batches)["batches"][0]["worker_tail_s"] is None


def test_overlapping_service_on_same_worker_rejected():
    events, batches = fixture()
    events[1]["pid"] = 1
    with pytest.raises(ValueError, match="overlapping"):
        summarize(events, batches)
