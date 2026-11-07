"""Verify the branch export without importing CUDA packages or running kernels."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "reruns"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "SHA256SUMS.json").read_text())
    expected = {item["path"] for item in manifest["files"]}
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and not IGNORED.intersection(p.relative_to(ROOT).parts)
        and p != ROOT / "SHA256SUMS.json"
        and p.name != ".DS_Store"
    }
    if actual != expected:
        raise RuntimeError({"missing": sorted(expected - actual), "extra": sorted(actual - expected)})
    for item in manifest["files"]:
        p = ROOT / item["path"]
        if p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
            raise RuntimeError(f"Release file mismatch: {item['path']}")
    provenance = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text())
    for item in provenance["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Source snapshot mismatch: {item['path']}")
    subprocess.run([sys.executable, str(ROOT / "experiments/verify_repository.py")], check=True)
    print(f"Verified {len(expected)} release files and {len(provenance['files'])} source files")


if __name__ == "__main__":
    main()
