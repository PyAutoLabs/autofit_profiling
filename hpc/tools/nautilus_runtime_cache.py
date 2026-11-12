"""Bounded, offline local runtime restoration for the one approved RAL job.

Bootstrap uses the node's /usr/bin/python3 (3.9+), never the slow NFS Python.
The artifact contains python/, site-packages/, sources/ and manifest.json.
Only venv path wiring is regenerated. Outputs remain on verified shared storage.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

GIB = 1024**3
BOOTSTRAP_SECONDS = 180
SCIENCE_SECONDS = 1800
CLEANUP_SECONDS = 120
MAX_ARCHIVE_BYTES = 4 * GIB
MAX_UNPACKED_BYTES = 8 * GIB
SCHEMA = "nautilus-local-runtime@2"
SOURCE_PINS = {
    "PyAutoFit": "bdb7af0e9c339ad07850679fd6ac168aacc7d18c",
    "PyAutoNerves": "9b204c90c2ce05c5bb0cf3e9800ed13f005c6c56",
}


def uptime():
    return float(Path("/proc/uptime").read_text().split()[0])


def write_json(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def filesystem(path):
    result = subprocess.run(
        ["findmnt", "--target", str(path), "--json", "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = json.loads(result.stdout)["filesystems"]
    if len(rows) != 1:
        raise ValueError("ambiguous filesystem")
    return rows[0]


def storage_contract(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError("retained storage contract checksum differs")
    contract = json.loads(path.read_text())
    if contract["schema"] != "nautilus-retained-storage@1" or not contract["witnesses"]:
        raise ValueError("invalid retained storage contract")
    return contract


def check_storage(local, output, archive_bytes, contract, observed=None):
    observed = observed or {"runtime": filesystem(local), "scientific_output": filesystem(output)}
    local_fs, output_fs = observed["runtime"], observed["scientific_output"]
    # tmpfs would charge the runtime to the same 8GiB job memory limit.
    if local_fs["fstype"] not in ("ext4", "xfs"):
        raise ValueError("cache requires local ext4/xfs storage, not network or tmpfs")
    root = Path(contract["root"]).resolve(strict=True)
    view = {key: output_fs[key] for key in ("target", "source", "fstype")}
    if (
        output_fs["fstype"] not in ("nfs", "nfs4", "ceph")
        or view not in contract["views"]
        or not output.resolve(strict=True).is_relative_to(root)
    ):
        raise ValueError("output is not the witnessed retained shared storage")
    for witness in contract["witnesses"]:
        relative = Path(witness["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe retained storage witness path")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or digest(path) != witness["sha256"]:
            raise ValueError("retained storage witness differs")
    if not 0 < archive_bytes <= MAX_ARCHIVE_BYTES:
        raise ValueError("invalid runtime artifact size")
    if shutil.disk_usage(local).free < MAX_UNPACKED_BYTES + 2 * archive_bytes + GIB:
        raise ValueError("insufficient local disk capacity")
    return observed


def safe_extract(archive, destination):
    """Validate all headers first; never follow an archive link while writing."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > 200000 or sum(m.size for m in members) > MAX_UNPACKED_BYTES:
            raise ValueError("runtime archive exceeds extraction limits")
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] not in ("python", "site-packages", "sources", "manifest.json")
                or str(path) in names
                or not (member.isdir() or member.isfile())
                or member.size < 0
            ):
                raise ValueError("unsafe runtime archive entry")
            names.add(str(path))
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                path.mkdir(exist_ok=True)
            elif member.isfile():
                with bundle.extractfile(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                path.chmod(0o755 if member.mode & 0o111 else 0o644)


def make_cache(parent, token):
    parent = parent.resolve(strict=True)
    root = Path(tempfile.mkdtemp(prefix="nautilus-training-", dir=parent))
    root.chmod(0o700)
    (root / ".owner").write_text(token)
    return root


def cleanup_cache(root, parent, token):
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or root.parent != parent.resolve(strict=True)
        or not root.name.startswith("nautilus-training-")
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or (root / ".owner").is_symlink()
        or (root / ".owner").read_text() != token
    ):
        raise ValueError("refusing cleanup of an unowned local cache")
    shutil.rmtree(root)


def environment(root):
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYAUTO_TRAINING_AFFINITY"):
        env.pop(name, None)
    env.update(
        PYTHONPATH=os.pathsep.join(str(root / "sources" / name) for name in SOURCE_PINS),
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLBACKEND="Agg",
        JAX_PLATFORMS="cpu",
        NUMBA_CACHE_DIR=str(root / "caches/numba"),
        MPLCONFIGDIR=str(root / "caches/matplotlib"),
    )
    return env


def prepare(root, archive, expected_hash, producer, receipt, execution):
    local_archive = root / "artifact.tar.gz"
    shutil.copyfile(archive, local_archive)
    if digest(local_archive) != expected_hash:
        raise ValueError("runtime artifact checksum differs")
    safe_extract(local_archive, root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["sources"] != {
        **SOURCE_PINS,
        "autofit_profiling": producer,
    }:
        raise ValueError("runtime manifest/source pins differ")
    if manifest["storage_contract"] != json.loads((root / "storage-contract.json").read_text()):
        raise ValueError("artifact and admitted retained storage contracts differ")
    source = root / "sources/autofit_profiling"
    if digest(source / "hpc/tools/nautilus_runtime_cache.py") != digest(Path(__file__)):
        raise ValueError("bootstrap differs from frozen producer")
    python = root / "python/bin/python3"
    subprocess.run(
        [str(python), "-I", "-m", "venv", "--without-pip", str(root / "venv")], check=True
    )
    target = root / "venv/lib/python3.12/site-packages"
    if list(target.iterdir()):
        raise ValueError("new venv unexpectedly contains packages")
    target.rmdir()
    (root / "site-packages").rename(target)
    for name in ("numba", "matplotlib"):
        (root / "caches" / name).mkdir(parents=True)
    subprocess.run(
        [
            str(root / "venv/bin/python"),
            str(source / "hpc/tools/verify_nautilus_runtime.py"),
            "--cache",
            str(root),
            "--receipt",
            str(receipt / "runtime-validation.json"),
            "--execution",
            execution,
        ],
        cwd=source,
        env=environment(root),
        check=True,
    )


def descendants():
    rows = {}
    for item in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = item.read_text().rsplit(")", 1)[1].split()
            rows[int(item.parent.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            pass
    owned = {os.getpid()}
    while True:
        more = {pid for pid, parent in rows.items() if parent in owned}
        if more <= owned:
            return owned - {os.getpid()}
        owned |= more


def stop_descendants():
    # The supervisor is a subreaper: even orphaned spawn workers remain owned.
    for sig, seconds in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
        for pid in descendants():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            if not descendants():
                return
            time.sleep(0.05)
    raise RuntimeError("owned processes remain; cache retained for safety")


def run_command(args, seconds, log, **kwargs):
    if seconds <= 0:
        raise TimeoutError("bootstrap/scientific deadline exhausted")
    with log.open("w") as stream:
        process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        return process.wait(timeout=seconds)


def supervise(args):
    receipt = args.receipt.resolve(strict=True)
    parent = args.local_parent.resolve(strict=True)
    token = f"{os.getuid()}:{os.getpid()}:{time.time_ns()}"
    root, result = None, {"status": "preparing", "sampling_started": False}
    code = 1
    # PR_SET_CHILD_SUBREAPER is Linux-specific, as is the existing RAL admission.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError("cannot establish process ownership for cache cleanup")

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, BOOTSTRAP_SECONDS - (uptime() - args.started)))
    try:
        # Retain observations BEFORE admission can reject them (job415132 witness).
        result["filesystems"] = {}
        result["filesystems"]["runtime"] = filesystem(parent)
        result["filesystems"]["scientific_output"] = filesystem(receipt)
        write_json(receipt / "bootstrap.json", result)
        contract = storage_contract(args.storage_contract, args.storage_sha256)
        check_storage(parent, receipt, args.archive.stat().st_size, contract, result["filesystems"])
        result["storage_contract_sha256"] = args.storage_sha256
        root = make_cache(parent, token)
        write_json(root / "storage-contract.json", contract)
        result["cache"] = str(root)
        write_json(receipt / "bootstrap.json", result)
        code = run_command(
            [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).resolve()),
                "prepare",
                "--cache",
                str(root),
                "--archive",
                str(args.archive),
                "--sha256",
                args.sha256,
                "--producer",
                args.producer,
                "--receipt",
                str(receipt),
                "--execution",
                args.execution,
            ],
            BOOTSTRAP_SECONDS - (uptime() - args.started),
            receipt / "bootstrap.log",
        )
        if code:
            raise RuntimeError(f"bootstrap exited {code}; no sampling started")
        elapsed = uptime() - args.started
        if elapsed >= BOOTSTRAP_SECONDS:
            raise TimeoutError("bootstrap exceeded180s from wrapper entry")
        result.update(status="validated", bootstrap_s=elapsed)
        validation = json.loads((receipt / "runtime-validation.json").read_text())
        if validation["status"] != "pass":
            raise ValueError("runtime validation did not pass")
        if args.execution == "ral":
            if validation["allocation"]["remaining_job_s"] < SCIENCE_SECONDS + CLEANUP_SECONDS:
                raise ValueError("insufficient remaining scheduler time for science and cleanup")
            signal.setitimer(signal.ITIMER_REAL, 0)
            result.update(status="running", sampling_started=True)
            write_json(receipt / "bootstrap.json", result)
            source = root / "sources/autofit_profiling"
            code = run_command(
                [
                    str(root / "venv/bin/python"),
                    str(source / "scripts/analytic_gaussian/nautilus/training_pool_benchmark.py"),
                    "run",
                    "--execution",
                    "ral",
                    "--output",
                    str(receipt / "pilot"),
                ],
                SCIENCE_SECONDS,
                receipt / "experiment.log",
                cwd=source,
                env=environment(root),
            )
        result["status"] = "complete" if code == 0 else "failed"
    except BaseException as error:
        code = code if code else 1
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # No second signal may interrupt ownership checks and remove a live cache.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop_descendants()
            if root is not None:
                cleanup_cache(root, parent, token)
            result["cleanup"] = "complete"
        except BaseException as error:
            code = code or 1
            result.update(status="failed", cleanup=f"{type(error).__name__}: {error}")
        result.update(exit_code=code, elapsed_s=uptime() - args.started)
        write_json(receipt / "bootstrap.json", result)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("supervise", "prepare"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execution", choices=("ral", "validate-only"), required=True)
    parser.add_argument("--local-parent", type=Path, default=Path("/tmp"))
    parser.add_argument("--started", type=float)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--storage-contract", type=Path)
    parser.add_argument("--storage-sha256")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.cache, args.archive, args.sha256, args.producer, args.receipt, args.execution)
        return 0
    if args.started is None:
        parser.error("--started from wrapper entry is mandatory")
    if args.storage_contract is None or args.storage_sha256 is None:
        parser.error("frozen --storage-contract and --storage-sha256 are mandatory")
    return supervise(args)


if __name__ == "__main__":
    sys.exit(main())
