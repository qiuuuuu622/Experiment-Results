# SPDX-License-Identifier: Apache-2.0
"""Freeze a P3 run's Python, kernels, native headers, binary identity and command."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import tarfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("output")
p.add_argument("--command-file", required=True)
p.add_argument("--native-package")
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
assert not (out / "source.tgz").exists(), "A frozen run is immutable"
files = sorted(
    x
    for directory in ("vllm_dsv41_indexer", "benchmarks/kernels/dsv41_indexer")
    for x in (root / directory).rglob("*")
    if x.is_file() and "__pycache__" not in str(x)
)
records = []
with tarfile.open(out / "source.tgz", "w:gz") as tar:
    for f in files:
        relative = str(f.relative_to(root))
        tar.add(f, arcname=relative)
        records.append(
            dict(path=relative, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        )
installed = Path(
    importlib.metadata.distribution("vllm").locate_file("vllm/third_party/deep_gemm")
)
native = Path(a.native_package) if a.native_package else installed
native_records = []
with tarfile.open(out / "native-include.tgz", "w:gz") as tar:
    for f in sorted((native / "include").rglob("*")):
        if f.is_file():
            tar.add(f, arcname=str(f.relative_to(native)))
            native_records.append(
                dict(
                    path=str(f.relative_to(native)),
                    sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                )
            )
for f in sorted(installed.glob("*.so")):
    native_records.append(
        dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    )
command = Path(a.command_file).read_text()
(out / "run.sh").write_text(command)
(out / "source.json").write_text(
    json.dumps(
        dict(
            captured_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            files=records,
            native_package=str(native),
            native_files=native_records,
            command=command,
        ),
        indent=2,
    )
    + "\n"
)
print(out, flush=True)
