"""Strict, read-only admission for the approved single-node RAL experiment.

Supports scontrol's key=value records and Linux cgroup v1/v2. Site support must
be established by read-only discovery before submission; missing fields fail.
"""

import datetime
import os
import re
import socket
import subprocess
import time
from pathlib import Path

GIB = 1024**3
POLICY = {
    "partition": "ral",
    "nodes": 1,
    "tasks": 1,
    "physical_cores": 4,
    "job_memory_bytes": 8 * GIB,
    "job_seconds": 2100,
    "exclusive": True,
}


def command(*args):
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def fields(text):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S+)", text))


def duration(text):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", text)
    if not match:
        raise ValueError("unverifiable scheduler time limit")
    day, hour, minute, second = (int(x or 0) for x in match.groups())
    return day * 86400 + hour * 3600 + minute * 60 + second


def bytes_value(text):
    match = re.fullmatch(r"(\d+)([KMGT]?)", text)
    if not match:
        raise ValueError("unverifiable scheduler memory")
    number, unit = match.groups()
    return int(number) * 1024 ** (" KMGT".index(unit) if unit else 0)


def topology(affinity, sys_cpu=Path("/sys/devices/system/cpu")):
    rows = []
    for cpu in affinity:
        root = sys_cpu / f"cpu{cpu}" / "topology"
        rows.append(
            {
                "cpu": cpu,
                "socket": int((root / "physical_package_id").read_text()),
                "core": int((root / "core_id").read_text()),
            }
        )
    if len(rows) != 4 or len({(r["socket"], r["core"]) for r in rows}) != 4:
        raise ValueError("require one logical CPU on each of four physical cores")
    return rows


def cgroup_memory(job_id, proc=Path("/proc/self/cgroup"), base=Path("/sys/fs/cgroup")):
    """Read every applicable memory ancestor; host-wide free RAM is insufficient."""
    entries = [line.split(":", 2) for line in proc.read_text().splitlines()]
    unified = next((p for _, controllers, p in entries if not controllers), None)
    if unified is not None:
        root, path, limit_name, used_name = base, unified, "memory.max", "memory.current"
    else:
        path = next((p for _, c, p in entries if "memory" in c.split(",")), None)
        if path is None:
            raise ValueError("missing memory cgroup membership")
        root, limit_name, used_name = (
            base / "memory",
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
        )
    parts = Path(path).parts
    if ".." in parts:
        raise ValueError("unresolvable cgroup namespace")
    job_part = next((p for p in parts if p in (f"job_{job_id}", f"job_{job_id}.scope")), None)
    if job_part is None:
        raise ValueError("memory cgroup does not belong to scheduler job")
    leaf = root / path.lstrip("/")
    rows = []
    current = leaf
    while True:
        limit_file = current / limit_name
        if limit_file.exists():
            raw = limit_file.read_text().strip()
            limit = int(raw) if raw != "max" else None
            used = int((current / used_name).read_text())
            rows.append({"path": str(current), "limit": limit, "used": used})
        if current == root:
            break
        current = current.parent
    job = next((r for r in rows if Path(r["path"]).name == job_part), None)
    if job is None or job["limit"] != POLICY["job_memory_bytes"]:
        raise ValueError("job cgroup must enforce exactly8GiB")
    finite = [r for r in rows if r["limit"] is not None]
    if not finite or min(r["limit"] - r["used"] for r in finite) < 4 * GIB:
        raise ValueError("insufficient cgroup memory for one4GiB child")
    return {"membership": path, "ancestors": rows, "job_limit": job["limit"]}


def validate_scheduler(job, node, partition, step, running, job_id, step_id, hostname, uid):
    if (
        job.get("JobId") != job_id
        or job.get("JobState") != "RUNNING"
        or job.get("Partition") != "ral"
        or job.get("NumNodes") != "1"
        or job.get("NumTasks") != "1"
        or job.get("CPUs/Task") != "4"
        or not job.get("UserId", "").endswith(f"({uid})")
        or job.get("Requeue") != "0"
        or job.get("Restarts") != "0"
    ):
        raise ValueError("unexpected scheduler job identity/resources/restart policy")
    if job.get("OverSubscribe", job.get("Shared")) not in ("NO", "0"):
        raise ValueError("job is not exclusive")
    if partition.get("PartitionName") != "ral" or partition.get("OverSubscribe") not in (
        "NO",
        "EXCLUSIVE",
    ):
        raise ValueError("partition isolation policy is unverifiable")
    if node.get("NodeName") != hostname or job.get("NodeList") != hostname:
        raise ValueError("job is not on the current single compute node")
    total = int(node.get("CPUEfctv", node.get("CPUTot", "0")))
    if (
        total < 4
        or int(job.get("NumCPUs", "0")) != total
        or int(node.get("CPUAlloc", "0")) != total
    ):
        raise ValueError("whole-node CPU allocation not established")
    if set(running.split()) != {job_id}:
        raise ValueError("another job shares the compute node")
    tres = dict(item.split("=", 1) for item in job.get("AllocTRES", "").split(",") if "=" in item)
    if (
        int(tres.get("cpu", "0")) != total
        or tres.get("node") != "1"
        or bytes_value(tres.get("mem", "")) != 8 * GIB
        or any(k.startswith("gres/") and v != "0" for k, v in tres.items())
    ):
        raise ValueError("unexpected allocation TRES, memory or accelerator")
    if duration(job.get("TimeLimit", "")) != POLICY["job_seconds"]:
        raise ValueError("scheduler cap differs from35minutes")
    if (
        step.get("StepId") != f"{job_id}.{step_id}"
        or step.get("NodeList") != hostname
        or step.get("Nodes") != "1"
        or step.get("Tasks") != "1"
        or step.get("CPUs") != "4"
        or step.get("State") != "RUNNING"
        or step.get("Partition") != "ral"
        or step.get("UserId") != str(uid)
    ):
        raise ValueError("step does not reserve exactly four CPUs on this node")


def snapshot():
    job_id, step_id = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    if not job_id.isdigit() or not step_id.isdigit():
        raise ValueError("RAL execution requires a numbered srun job step")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise ValueError("restarted jobs are not authorized")
    host = socket.gethostname().split(".")[0]
    job = fields(command("scontrol", "show", "job", "--oneliner", job_id))
    node = fields(command("scontrol", "show", "node", "--oneliner", host))
    partition = fields(command("scontrol", "show", "partition", "--oneliner", "ral"))
    step = fields(command("scontrol", "show", "step", "--oneliner", f"{job_id}.{step_id}"))
    running = command("squeue", "--noheader", "--nodes", host, "--states", "R,CG,S", "--format=%A")
    validate_scheduler(job, node, partition, step, running, job_id, step_id, host, os.getuid())
    remaining = datetime.datetime.fromisoformat(job["EndTime"]).timestamp() - time.time()
    if remaining < 125:
        raise ValueError("insufficient scheduler time for a120s child and cleanup")
    affinity = sorted(os.sched_getaffinity(0))
    cores = topology(affinity)
    memory = cgroup_memory(job_id)
    identity = {
        "job_id": job_id,
        "step_id": step_id,
        "host": host,
        "affinity": affinity,
        "topology": cores,
        "cgroup": memory["membership"],
        "job_memory_bytes": memory["job_limit"],
        "node_cpus": int(job["NumCPUs"]),
        "policy": POLICY,
    }
    return {
        "identity": identity,
        "timestamp": time.time(),
        "remaining_job_s": remaining,
        "memory": memory,
        "job": job,
        "node": node,
        "partition": partition,
        "step": step,
        "running_jobs": running.split(),
    }


def verify_identity(expected, current):
    if current["identity"] != expected:
        raise ValueError("allocation or CPU binding changed")


def verify_process_affinity(pgid, allowed, proc=Path("/proc")):
    """Sample every thread in the child process group, including spawned workers."""
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) != pgid or stat[0] == "Z":
                continue
            for thread in (entry / "task").iterdir():
                if not set(os.sched_getaffinity(int(thread.name))) <= set(allowed):
                    raise ValueError("child thread escaped approved CPU affinity")
        except (FileNotFoundError, ProcessLookupError):
            continue
