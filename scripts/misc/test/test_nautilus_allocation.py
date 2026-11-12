"""Synthetic scheduler/cgroup fixtures; these are not live RAL receipts."""

import copy
import os
from pathlib import Path

import pytest
from profiling import nautilus_allocation as a
from profiling import nautilus_training_worker as worker


def scheduler():
    # Field names follow SchedMD Slurm23.02 print records; live site check is mandatory.
    job = a.fields(
        "JobId=42 JobState=RUNNING Partition=ral NumNodes=1 NumTasks=1 CPUs/Task=4 "
        "UserId=test(1000) Requeue=0 Restarts=0 OverSubscribe=NO NodeList=node1 NumCPUs=64 "
        "AllocTRES=cpu=64,mem=8G,node=1,billing=64 TimeLimit=00:35:00"
    )
    node = a.fields("NodeName=node1 CPUTot=64 CPUEfctv=64 CPUAlloc=64")
    partition = a.fields("PartitionName=ral OverSubscribe=NO")
    step = a.fields(
        "StepId=42.0 NodeList=node1 Nodes=1 Tasks=1 CPUs=4 State=RUNNING Partition=ral UserId=1000"
    )
    return [job, node, partition, step, "42", "42", "0", "node1", 1000]


def test_scheduler_requires_actual_whole_node_and_bound_step():
    a.validate_scheduler(*scheduler())


@pytest.mark.parametrize(
    "record,key,value",
    [
        (0, "JobId", "99"),
        (0, "Partition", "gpu"),
        (0, "NumNodes", "2"),
        (0, "NumTasks", "2"),
        (0, "CPUs/Task", "8"),
        (0, "UserId", "other(1)"),
        (0, "Requeue", "1"),
        (0, "Restarts", "1"),
        (0, "JobState", "PENDING"),
        (0, "OverSubscribe", "YES"),
        (0, "NumCPUs", "4"),
        (0, "NodeList", "login"),
        (0, "TimeLimit", "00:40:00"),
        (0, "AllocTRES", "cpu=64,mem=8G,node=1,gres/gpu=1"),
        (0, "AllocTRES", "cpu=64,mem=16G,node=1"),
        (1, "CPUAlloc", "4"),
        (2, "OverSubscribe", "FORCE:1"),
        (3, "StepId", "42.batch"),
        (3, "CPUs", "8"),
        (3, "Tasks", "2"),
        (3, "NodeList", "node2"),
        (3, "State", "PENDING"),
        (3, "UserId", "9"),
    ],
)
def test_scheduler_rejects_unsupported_allocations(record, key, value):
    rows = scheduler()
    rows[record][key] = value
    with pytest.raises(ValueError):
        a.validate_scheduler(*rows)


def test_another_job_or_missing_fields_rejects_admission():
    rows = scheduler()
    rows[4] = "42\n51"
    with pytest.raises(ValueError, match="another job"):
        a.validate_scheduler(*rows)
    rows = scheduler()
    del rows[0]["AllocTRES"]
    with pytest.raises(ValueError):
        a.validate_scheduler(*rows)


def memory_tree(tmp_path, version):
    base = tmp_path / "cgroup"
    root = base if version == 2 else base / "memory"
    job = root / "slurm" / "job_42"
    leaf = job / "step_0" / "task_0"
    leaf.mkdir(parents=True)
    limit = "memory.max" if version == 2 else "memory.limit_in_bytes"
    used = "memory.current" if version == 2 else "memory.usage_in_bytes"
    for path in (job, leaf):
        (path / limit).write_text(str(8 * a.GIB))
        (path / used).write_text(str(a.GIB))
    proc = tmp_path / "membership"
    proc.write_text(("0::" if version == 2 else "7:memory:") + "/slurm/job_42/step_0/task_0\n")
    return proc, base, job, leaf, limit, used


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_checks_job_and_tighter_ancestor_headroom(tmp_path, version):
    proc, base, job, leaf, limit, used = memory_tree(tmp_path, version)
    receipt = a.cgroup_memory("42", proc, base)
    assert receipt["job_limit"] == 8 * a.GIB
    (leaf / used).write_text(str(5 * a.GIB))
    with pytest.raises(ValueError, match="insufficient"):
        a.cgroup_memory("42", proc, base)
    (leaf / used).write_text(str(a.GIB))
    (job / limit).write_text(str(16 * a.GIB))
    with pytest.raises(ValueError, match="exactly8GiB"):
        a.cgroup_memory("42", proc, base)


def test_cgroup_missing_job_or_accounting_is_not_host_memory(tmp_path):
    proc, base, job, leaf, limit, used = memory_tree(tmp_path, 2)
    with pytest.raises(ValueError, match="does not belong"):
        a.cgroup_memory("999", proc, base)
    (job / used).unlink()
    with pytest.raises(FileNotFoundError):
        a.cgroup_memory("42", proc, base)


def test_affinity_must_cover_four_distinct_physical_cores(tmp_path):
    for cpu in range(4):
        path = tmp_path / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "physical_package_id").write_text("0")
        (path / "core_id").write_text(str(cpu))
    assert len(a.topology(range(4), tmp_path)) == 4
    (tmp_path / "cpu3/topology/core_id").write_text("2")
    with pytest.raises(ValueError, match="four physical cores"):
        a.topology(range(4), tmp_path)


def test_cannot_enter_ral_mode_from_environment_variable_alone(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_STEP_ID", "batch")
    with pytest.raises(ValueError, match="numbered srun"):
        a.snapshot()


def test_allocation_identity_changes_stop_following_case():
    expected = {"job_id": "42", "affinity": [0, 1, 2, 3]}
    a.verify_identity(expected, {"identity": copy.deepcopy(expected)})
    with pytest.raises(ValueError, match="changed"):
        a.verify_identity(expected, {"identity": {**expected, "job_id": "43"}})


def test_training_worker_rejects_changed_cpu_binding(monkeypatch):
    monkeypatch.setenv("PYAUTO_TRAINING_AFFINITY", "[0, 1, 2, 3]")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    assert worker.check_affinity() == [0, 1, 2, 3]
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 4})
    with pytest.raises(RuntimeError, match="affinity"):
        worker.check_affinity()


def test_supervision_checks_every_thread_not_just_parent(tmp_path, monkeypatch):
    process = tmp_path / "20"
    (process / "task/20").mkdir(parents=True)
    (process / "task/21").mkdir()
    (process / "stat").write_text("20 (python) S 1 20 0")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0} if pid == 20 else {8})
    with pytest.raises(ValueError, match="escaped"):
        a.verify_process_affinity(20, [0, 1, 2, 3], tmp_path)


def test_batch_script_has_no_mail_array_requeue_or_gpu_request():
    source = Path(__file__).resolve().parents[3] / "hpc/batch_cpu/submit_nautilus_training_pool"
    directives = [line for line in source.read_text().splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --partition=ral" in directives
    assert "#SBATCH --exclusive" in directives
    assert "#SBATCH --no-requeue" in directives
    assert "#SBATCH --time=00:35:00" in directives
    assert not any(any(x in line for x in ("--mail", "--array", "--gres")) for line in directives)


@pytest.mark.parametrize("exit_code", [0, 17])
@pytest.mark.parametrize("staged", [False, True])
def test_batch_wrapper_preserves_exit_status_and_refuses_second_execution(
    tmp_path, exit_code, staged
):
    import hashlib
    import shlex
    import subprocess

    source = Path(__file__).resolve().parents[3] / "hpc/batch_cpu/submit_nautilus_training_pool"
    binary = tmp_path / "bin"
    binary.mkdir()
    sha = "a" * 40
    git = binary / "git"
    git.write_text("""#!/bin/sh
case "$*" in
  *status*) exit 0 ;;
  *fit*rev-parse*) echo bdb7af0e9c339ad07850679fd6ac168aacc7d18c ;;
  *nerves*rev-parse*) echo 9b204c90c2ce05c5bb0cf3e9800ed13f005c6c56 ;;
  *) echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
esac
""")
    git.chmod(0o755)
    for name in ("scontrol", "sacct", "findmnt", "lscpu"):
        path = binary / name
        path.write_text("#!/bin/sh\necho fixture\n")
        path.chmod(0o755)
    helper = tmp_path / "hpc/tools/nautilus_runtime_cache.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("fixture")
    runner = binary / "srun"
    runner.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$NAUTILUS_ROOT/output/srun-calls"\n'
        + f"exit {exit_code}\n"
    )
    runner.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + ":" + os.environ["PATH"],
        "NAUTILUS_ROOT": str(tmp_path),
        "NAUTILUS_PRODUCER_SHA": sha,
        "NAUTILUS_ARCHIVE": "/fixture/runtime.tar.gz",
        "NAUTILUS_ARCHIVE_SHA256": "b" * 64,
        "NAUTILUS_STORAGE_CONTRACT": "/fixture/storage.json",
        "NAUTILUS_STORAGE_SHA256": "c" * 64,
        "NAUTILUS_BOOTSTRAP_SHA256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "NAUTILUS_WRAPPER_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "SLURM_JOB_ID": "42",
        "SLURM_JOB_PARTITION": "ral",
        "SLURM_RESTART_COUNT": "0",
    }
    env.pop("SLURM_SUBMIT_DIR", None)
    if staged:
        submit = tmp_path / "hpc/batch_cpu"
        submit.mkdir(parents=True)
        (tmp_path / "output").mkdir()
        keys = [key for key in env if key.startswith("NAUTILUS_")]
        exports = "".join(f"export {key}={shlex.quote(env[key])}\n" for key in keys)
        (tmp_path / "output/ral-environment.sh").write_text(exports)
        for key in keys:
            env.pop(key)
        env["SLURM_SUBMIT_DIR"] = str(submit)
    result = subprocess.run(["bash", str(source)], env=env, capture_output=True, timeout=15)
    assert result.returncode == exit_code, result.stderr.decode()
    receipt = tmp_path / "output/training-pool-ral-02"
    assert (receipt / "wrapper_exit_code").read_text().strip() == str(exit_code)
    again = subprocess.run(["bash", str(source)], env=env, capture_output=True, timeout=15)
    assert again.returncode != 0
    calls = (tmp_path / "output/srun-calls").read_text().splitlines()
    assert len(calls) == 1
    assert "--cpu-bind=verbose,cores" in calls[0]
    assert "nautilus_runtime_cache.py supervise" in calls[0]
    assert "--execution ral" in calls[0]
    assert (receipt / "wrapper_exit_code").read_text().strip() == str(exit_code)


@pytest.mark.parametrize("suffix", ["", ",gres/gpu=1", ",gres/gpu:a100=1"])
def test_ral22_legacy_tres_keeps_accelerator_rejection(suffix):
    rows = scheduler()
    rows[0]["TRES"] = rows[0].pop("AllocTRES") + suffix
    if suffix:
        with pytest.raises(ValueError, match="accelerator"):
            a.validate_scheduler(*rows)
    else:
        a.validate_scheduler(*rows)


def test_conflicting_new_and_legacy_tres_are_not_silently_selected():
    rows = scheduler()
    rows[0]["TRES"] = rows[0]["AllocTRES"] + ",gres/gpu=1"
    with pytest.raises(ValueError, match="conflicting"):
        a.validate_scheduler(*rows)
