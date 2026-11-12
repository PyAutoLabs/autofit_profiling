"""Package a validated private login-host runtime; never runs scientific work.

Requires the cache layout prepared in the approved setup: python/, venv/ and
sources/{autofit_profiling,PyAutoFit,PyAutoNerves}. The new producer must be clean.
The result is an immutable, checksum-addressed artifact on retained NFS storage.
"""

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from nautilus_runtime_cache import EXPERIMENTS, SCHEMA, SOURCE_PINS, digest, experiment_identity


def build(cache, validation, destination, storage, experiment="training-pool"):
    identity = json.loads(validation.read_text())
    if identity["status"] != "pass":
        raise ValueError("passing login-host validation required")
    identity = identity["identity"]
    producer = cache / "sources/autofit_profiling"
    pins = {**SOURCE_PINS, "autofit_profiling": identity["sources"]["producer"]["revision"]}
    for name, sha in pins.items():
        source = cache / "sources" / name
        if (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            != sha
        ):
            raise ValueError("source revision changed after validation")
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
            raise ValueError("source became dirty after validation")
        if (source / ".git/objects/info/alternates").exists():
            raise ValueError("source clone depends on external object storage")
    manifest = {
        "schema": SCHEMA,
        "experiment": experiment_identity(producer, experiment),
        "sources": pins,
        "python_build": identity["python"],
        "distribution_versions": identity["distribution_versions"],
        "nautilus_source_sha256": identity["nautilus_source_sha256"],
        "source_identities": identity["sources"],
        "storage_contract": json.loads(storage.read_text()),
    }
    destination.mkdir(parents=True, exist_ok=True)
    # Assemble on local disk; only one sequential artifact write traverses NFS.
    with tempfile.TemporaryDirectory(prefix="nautilus-training-image-", dir=cache) as temp:
        temp = Path(temp)
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        archive = temp / "runtime.tar.gz"
        with tarfile.open(archive, "w:gz", dereference=True, compresslevel=1) as bundle:
            for source, name in (
                (cache / "python", "python"),
                (cache / "venv/lib/python3.12/site-packages", "site-packages"),
                (cache / "sources", "sources"),
                (temp / "manifest.json", "manifest.json"),
            ):
                for link in source.rglob("*") if source.is_dir() else ():
                    if link.is_symlink() and not link.resolve().is_relative_to(cache):
                        raise ValueError("runtime input links outside private cache")
                bundle.add(source, arcname=name)
        checksum = digest(archive)
        target = destination / f"{checksum}.tar.gz"
        # Exclusive create: never overwrite any previous runtime artifact.
        with archive.open("rb") as source, target.open("xb") as output:
            import shutil

            shutil.copyfileobj(source, output)
        if digest(target) != checksum:
            raise ValueError("retained artifact checksum differs after transfer")
        receipt = {
            **manifest,
            "archive": str(target),
            "archive_sha256": checksum,
            "archive_bytes": target.stat().st_size,
            "bootstrap_sha256": digest(producer / "hpc/tools/nautilus_runtime_cache.py"),
            "wrapper_sha256": digest(producer / EXPERIMENTS[experiment]["wrapper"]),
            "storage_contract_path": str(storage),
            "storage_contract_sha256": digest(storage),
        }
        (destination / f"{checksum}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--storage-contract", type=Path, required=True)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="training-pool")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.cache.resolve(),
                args.validation,
                args.destination,
                args.storage_contract,
                args.experiment,
            ),
            indent=2,
        )
    )
