# SPDX-License-Identifier: Apache-2.0
"""Record current source identity separately from installed CUDA dependencies."""

import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from .common import check_sources, check_vllm, device_info, save


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    installed = Path(check_vllm())
    versions = {}
    for name in ("vllm", "torch", "triton", "apache-tvm-ffi", "deep-gemm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "no distribution metadata"
    binaries = []
    for package in (installed / "vllm", installed / "deep_gemm"):
        if package.exists():
            for path in sorted(package.rglob("*.so")):
                binaries.append(dict(path=str(path), sha256=sha256(path)))

    def command(argv):
        result = subprocess.run(argv, text=True, capture_output=True, check=False)
        return dict(
            command=argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    save(
        args.output,
        dict(
            source=check_sources(),
            installed_versions=versions,
            binary_build_commit="unknown; not asserted to be current HEAD",
            binaries=binaries,
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            device=device_info(),
            gpu=command(["nvidia-smi", "-q"]),
            nvcc=command(["/usr/local/cuda/bin/nvcc", "--version"]),
            compile_flags_source="vllm_dsv41_indexer/native.py",
        ),
    )


if __name__ == "__main__":
    main()
