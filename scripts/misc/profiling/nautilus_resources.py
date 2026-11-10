"""Linux process-group supervision; stdlib only, no scientific imports."""

import json
import math
import os
import signal
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def memory_available(proc=Path("/proc")):
    for line in (proc / "meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux MemAvailable is required")


def group_rss(pgid, proc=Path("/proc")):
    """Sum RSS in a fresh session; shared pages may be counted more than once."""
    rss = 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces or parentheses; fields after the last ')'.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rss


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(process, timeout, memory_limit_bytes=None, interval=0.05):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    start = time.monotonic()
    peak = 0
    reason = None
    try:
        while process.poll() is None:
            if memory_limit_bytes is not None:
                peak = max(peak, group_rss(process.pid))
                if peak > memory_limit_bytes:
                    reason = "memory_cap"
                    break
            if time.monotonic() - start >= timeout:
                reason = "timeout"
                break
            time.sleep(interval)
    finally:
        # Also kills descendants if the leader exited leaving one behind, or on
        # KeyboardInterrupt / SIGTERM translated to an exception by the caller.
        terminate_group(process)
    return {
        "returncode": 124 if reason == "timeout" else 125 if reason else process.returncode,
        "termination_reason": reason,
        "sampled_group_peak_rss_bytes": peak if memory_limit_bytes is not None else None,
        "memory_limit_bytes": memory_limit_bytes,
        "rss_sample_interval_s": interval,
        "rss_note": "process group sum; shared pages may repeat; sub-interval spikes may be missed",
    }
