"""Offline restoration safety and lifecycle tests; no scientific packages needed."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "hpc/tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load("nautilus_runtime_cache")
v = load("verify_nautilus_runtime")


def storage_fixture(root):
    marker = root / "witness"
    marker.write_text("compute-written retained witness")
    output = root / "output"
    output.mkdir(exist_ok=True)
    return {
        "schema": "nautilus-retained-storage@1",
        "root": str(root),
        "views": [
            {"target": "/shared", "source": "gateway:/shared", "fstype": "nfs4"},
            {"target": "/shared", "source": "monitors:/volumes/shared", "fstype": "ceph"},
        ],
        "witnesses": [{"path": "witness", "sha256": c.digest(marker)}],
    }


def archive(path, entries):
    with tarfile.open(path, "w:gz") as stream:
        for name, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "../../outside"
            member.size = 4 if kind == tarfile.REGTYPE else 0
            stream.addfile(member, io.BytesIO(b"data") if member.size else None)


@pytest.mark.parametrize(
    "entry",
    [
        ("../outside", tarfile.REGTYPE),
        ("/outside", tarfile.REGTYPE),
        ("python/../../outside", tarfile.REGTYPE),
        (".owner", tarfile.REGTYPE),
        ("python/link", tarfile.SYMTYPE),
        ("python/hardlink", tarfile.LNKTYPE),
        ("python/device", tarfile.CHRTYPE),
        ("python/fifo", tarfile.FIFOTYPE),
    ],
)
def test_unsafe_archive_is_rejected_before_any_file_is_written(tmp_path, entry):
    packed = tmp_path / "runtime.tar.gz"
    archive(packed, [("python/valid", tarfile.REGTYPE), entry])
    root = tmp_path / "cache"
    root.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        c.safe_extract(packed, root)
    assert list(root.iterdir()) == []


def test_safe_files_restore_and_duplicates_fail(tmp_path):
    packed = tmp_path / "runtime.tar.gz"
    archive(packed, [("python/bin/python", tarfile.REGTYPE)])
    root = tmp_path / "cache"
    root.mkdir()
    c.safe_extract(packed, root)
    assert (root / "python/bin/python").read_bytes() == b"data"
    archive(packed, [("python/one", tarfile.REGTYPE)] * 2)
    with pytest.raises(ValueError, match="unsafe"):
        c.safe_extract(packed, root)


def test_tampering_is_detected_before_extraction_or_interpreter_launch(tmp_path, monkeypatch):
    packed = tmp_path / "runtime.tar.gz"
    packed.write_bytes(b"changed")
    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setattr(c, "safe_extract", lambda *args: pytest.fail("extracted tampered data"))
    with pytest.raises(ValueError, match="checksum"):
        c.prepare(root, packed, "0" * 64, "a" * 40, tmp_path, "validate-only")


@pytest.mark.parametrize(
    "local_fs,output_fs", [("nfs4", "nfs4"), ("tmpfs", "nfs4"), ("ext4", "ext4")]
)
def test_storage_policy_rejects_network_runtime_memory_runtime_or_local_science(
    tmp_path, monkeypatch, local_fs, output_fs
):
    monkeypatch.setattr(
        c,
        "filesystem",
        lambda p: (
            {"fstype": local_fs}
            if p == tmp_path
            else {"target": "/shared", "source": "gateway:/shared", "fstype": output_fs}
        ),
    )
    with pytest.raises(ValueError):
        c.check_storage(tmp_path, tmp_path / "output", 100, storage_fixture(tmp_path))


def test_missing_storage_and_low_capacity_fail(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        c.make_cache(tmp_path / "missing", "owner")
    monkeypatch.setattr(
        c,
        "filesystem",
        lambda p: (
            {"fstype": "ext4"}
            if p == tmp_path
            else {"target": "/shared", "source": "gateway:/shared", "fstype": "nfs4"}
        ),
    )
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=1024))
    with pytest.raises(ValueError, match="capacity"):
        c.check_storage(tmp_path, tmp_path / "output", 100, storage_fixture(tmp_path))


@pytest.mark.parametrize("mutation", ["owner", "mode", "parent", "symlink"])
def test_cleanup_refuses_unowned_paths(tmp_path, mutation):
    root = c.make_cache(tmp_path, "owner")
    parent = tmp_path
    if mutation == "owner":
        (root / ".owner").write_text("other")
    elif mutation == "mode":
        root.chmod(0o755)
    elif mutation == "parent":
        parent = tmp_path.parent
    else:
        (root / ".owner").unlink()
        (tmp_path / "marker").write_text("owner")
        (root / ".owner").symlink_to(tmp_path / "marker")
    with pytest.raises(ValueError, match="unowned"):
        c.cleanup_cache(root, parent, "owner")
    assert root.exists()


@pytest.fixture
def supervision(tmp_path, monkeypatch):
    receipt = tmp_path / "nfs-output"
    receipt.mkdir()
    packed = tmp_path / "runtime.tar.gz"
    packed.write_bytes(b"fixture")
    args = SimpleNamespace(
        receipt=receipt,
        local_parent=tmp_path,
        archive=packed,
        sha256="a" * 64,
        producer="b" * 40,
        execution="ral",
        started=c.uptime(),
    )
    monkeypatch.setattr(c, "check_storage", lambda *args: {"scientific_output": {"fstype": "nfs4"}})
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    monkeypatch.setattr(c.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(c, "stop_descendants", lambda: None)
    contract_path = tmp_path / "storage-contract.json"
    c.write_json(contract_path, storage_fixture(tmp_path))
    args.storage_contract = contract_path
    args.storage_sha256 = c.digest(contract_path)
    monkeypatch.setattr(
        c,
        "filesystem",
        lambda p: {
            "fstype": "ext4" if p == tmp_path else "ceph",
            "target": "/shared",
            "source": "fixture",
        },
    )
    calls = []

    def runner(command, seconds, log, **kwargs):
        calls.append((command, seconds, kwargs))
        c.write_json(
            receipt / "runtime-validation.json",
            {
                "status": "pass",
                "allocation": {"remaining_job_s": 2000},
            },
        )
        return 0

    monkeypatch.setattr(c, "run_command", runner)
    return args, calls


def test_success_keeps_science_on_nfs_and_removes_only_cache(supervision):
    args, calls = supervision
    assert c.supervise(args) == 0
    assert len(calls) == 2
    assert calls[1][0][-2:] == ["--output", str(args.receipt / "pilot")]
    assert calls[1][1] == 1800
    assert calls[1][2]["cwd"].name == "autofit_profiling"
    result = json.loads((args.receipt / "bootstrap.json").read_text())
    assert result["sampling_started"] and result["cleanup"] == "complete"
    assert not Path(result["cache"]).exists()
    assert args.archive.exists() and args.receipt.exists()


def test_validate_only_never_launches_science(supervision):
    args, calls = supervision
    args.execution = "validate-only"
    assert c.supervise(args) == 0
    assert len(calls) == 1
    assert not json.loads((args.receipt / "bootstrap.json").read_text())["sampling_started"]


@pytest.mark.parametrize(
    "failure", ["bootstrap", "timeout", "scheduler-time", "validation", "science"]
)
def test_failure_stops_without_retry_and_preserves_exit_status(supervision, monkeypatch, failure):
    args, calls = supervision
    good = c.run_command

    def runner(command, seconds, log, **kwargs):
        result = good(command, seconds, log, **kwargs)
        if failure == "bootstrap":
            return 17
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, seconds)
        if failure == "scheduler-time":
            c.write_json(
                args.receipt / "runtime-validation.json",
                {
                    "status": "pass",
                    "allocation": {"remaining_job_s": 1919},
                },
            )
        if failure == "validation":
            c.write_json(args.receipt / "runtime-validation.json", {"status": "failed"})
        return 19 if failure == "science" and len(calls) == 2 else result

    monkeypatch.setattr(c, "run_command", runner)
    code = c.supervise(args)
    assert code == {"bootstrap": 17, "science": 19}.get(failure, 1)
    assert len(calls) == (2 if failure == "science" else 1)
    result = json.loads((args.receipt / "bootstrap.json").read_text())
    assert result["sampling_started"] == (failure == "science")
    assert result["cleanup"] == "complete"


def test_elapsed_bootstrap_blocks_science(supervision, monkeypatch):
    args, calls = supervision
    monkeypatch.setattr(c, "uptime", lambda: args.started + 181)
    assert c.supervise(args) != 0
    assert len(calls) == 1
    assert not json.loads((args.receipt / "bootstrap.json").read_text())["sampling_started"]


def test_cleanup_failure_is_not_reported_as_success(supervision, monkeypatch):
    args, _ = supervision
    monkeypatch.setattr(
        c, "stop_descendants", lambda: (_ for _ in ()).throw(RuntimeError("live child"))
    )
    assert c.supervise(args) != 0
    result = json.loads((args.receipt / "bootstrap.json").read_text())
    assert "live child" in result["cleanup"]
    assert Path(result["cache"]).exists()


@pytest.mark.parametrize("change", ["package", "extra", "build", "producer", "dirty"])
def test_runtime_identity_mismatch_fails(change):
    packages = {f"package{i}": "1" for i in range(90)}
    identity = {
        "distribution_versions": dict(packages),
        "python": "exact build",
        "sources": {"producer": {"revision": "a" * 40, "dirty": ""}},
    }
    manifest = {"python_build": "exact build", "sources": {"autofit_profiling": "a" * 40}}
    reference = {"reference_distribution_versions": packages}
    v.validate_inventory(identity, reference, manifest)
    if change == "package":
        identity["distribution_versions"]["package1"] = "2"
    elif change == "extra":
        identity["distribution_versions"]["extra"] = "1"
    elif change == "build":
        identity["python"] = "different build"
    elif change == "producer":
        identity["sources"]["producer"]["revision"] = "b" * 40
    else:
        identity["sources"]["producer"]["dirty"] = " M changed.py"
    with pytest.raises(ValueError):
        v.validate_inventory(identity, reference, manifest)


def test_process_cleanup_reaps_descendants_in_separate_sessions(tmp_path):
    script = f"""
import ctypes, importlib.util, subprocess, sys
spec=importlib.util.spec_from_file_location('cache', {str(TOOLS / "nautilus_runtime_cache.py")!r})
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True)
assert p.pid in c.descendants()
c.stop_descendants()
assert not c.descendants()
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


@pytest.mark.parametrize("headroom", [4 * 1024**3 - 1, 4 * 1024**3])
def test_memory_admission_checks_most_restrictive_ancestor(headroom):
    snapshot = {
        "memory": {
            "ancestors": [
                {"limit": 8 * 1024**3, "used": 100},
                {"limit": 10 * 1024**3, "used": 10 * 1024**3 - headroom},
            ]
        }
    }
    if headroom < 4 * 1024**3:
        with pytest.raises(ValueError, match="cgroup memory"):
            v.validate_capacity(snapshot)
    else:
        v.validate_capacity(snapshot)


@pytest.mark.parametrize("view_index", [0, 1])
def test_witnessed_nfs_and_ceph_views_accept_same_retained_output(
    tmp_path, monkeypatch, view_index
):
    contract = storage_fixture(tmp_path)
    observed = {"runtime": {"fstype": "ext4"}, "scientific_output": contract["views"][view_index]}
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * c.GIB))
    assert c.check_storage(tmp_path, tmp_path / "output", 100, contract, observed) == observed


@pytest.mark.parametrize("change", ["source", "target", "type", "witness", "traversal", "outside"])
def test_shared_filesystem_type_alone_cannot_admit_wrong_storage(tmp_path, monkeypatch, change):
    contract = storage_fixture(tmp_path)
    view = dict(contract["views"][1])
    output = tmp_path / "output"
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * c.GIB))
    if change in ("source", "target"):
        view[change] = "unwitnessed"
    elif change == "type":
        view["fstype"] = "ext4"
        contract["views"] = [view]  # Even a manifest cannot admit an ephemeral local disk.
    elif change == "witness":
        (tmp_path / "witness").write_text("different data")
    elif change == "traversal":
        contract["witnesses"][0]["path"] = "../outside"
    else:
        output = tmp_path.parent
    with pytest.raises(ValueError):
        c.check_storage(
            tmp_path,
            output,
            100,
            contract,
            {
                "runtime": {"fstype": "ext4"},
                "scientific_output": view,
            },
        )


def test_changed_storage_contract_hash_rejects_before_admission(tmp_path):
    path = tmp_path / "contract.json"
    c.write_json(path, storage_fixture(tmp_path))
    original = c.digest(path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        c.storage_contract(path, original)


def test_storage_rejection_retains_observed_mount_and_never_launches_child(
    supervision, monkeypatch
):
    args, calls = supervision

    def reject(*args):
        raise ValueError("unwitnessed storage")

    monkeypatch.setattr(c, "check_storage", reject)
    assert c.supervise(args) == 1
    assert not calls
    result = json.loads((args.receipt / "bootstrap.json").read_text())
    assert result["filesystems"]["scientific_output"]["fstype"] == "ceph"
    assert not result["sampling_started"]
    assert result["cleanup"] == "complete"


def test_artifact_cannot_override_the_admitted_storage_contract(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.write_bytes(b"fixture")
    root = tmp_path / "cache"
    root.mkdir()
    c.write_json(root / "storage-contract.json", {"identity": "approved"})

    def extract(*args):
        c.write_json(
            root / "manifest.json",
            {
                "schema": c.SCHEMA,
                "sources": {**c.SOURCE_PINS, "autofit_profiling": "a" * 40},
                "storage_contract": {"identity": "different"},
            },
        )

    monkeypatch.setattr(c, "safe_extract", extract)
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: pytest.fail("interpreter launched"))
    with pytest.raises(ValueError, match="contracts differ"):
        c.prepare(root, archive, c.digest(archive), "a" * 40, tmp_path, "validate-only")


def test_experiment_is_allowlisted_and_binds_entrypoint_wrapper_and_hashes(tmp_path):
    for name, paths in c.EXPERIMENTS.items():
        for relative in paths.values():
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
    manifest = {"experiment": c.experiment_identity(tmp_path, "training-diagnostics")}
    c.validate_experiment(manifest, tmp_path, "training-diagnostics")
    with pytest.raises(ValueError, match="identity"):
        c.validate_experiment(manifest, tmp_path, "training-pool")
    with pytest.raises(ValueError, match="unapproved"):
        c.experiment_identity(tmp_path, "../../arbitrary")
    (tmp_path / c.EXPERIMENTS["training-diagnostics"]["entrypoint"]).write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        c.validate_experiment(manifest, tmp_path, "training-diagnostics")


def test_diagnostic_wrapper_has_separate_claim_and_fixed_experiment():
    path = TOOLS.parent / "batch_cpu/submit_nautilus_training_diagnostics"
    text = path.read_text()
    assert "output/training-diagnostics-ral-02" in text
    assert "output/ral-diagnostics-environment-02.sh" in text
    assert "--experiment training-diagnostics --execution ral" in text
    assert 'mkdir "$receipt"' in text
    assert 'mkdir -p "$receipt"' not in text
    assert "#SBATCH --partition=ral" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --time=00:35:00" in text
    assert "training-pool-ral-03" not in text
    assert "#SBATCH --nodelist=euclid-ral-compute-1" in text
    assert "training-diagnostics-ral-01" not in text
