"""Verify public Top-k evidence and recompute statistics without CUDA."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1] / "research/topk-h20"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and p != ROOT / "MANIFEST.json"
              and "__pycache__" not in p.parts}
    assert expected == actual, (expected - actual, actual - expected)
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert digest(path) == entry["sha256"], path
        assert path.stat().st_size == entry["bytes"], path
    export = json.loads((ROOT / "PUBLIC_EXPORT.json").read_text())
    for entry in export["files"]:
        path = ROOT / entry["path"]
        assert digest(path) == entry["public_sha256"], path
        assert path.stat().st_size == entry["bytes"], path
    sources = json.loads((ROOT / "SOURCE_FINAL.json").read_text())["files"]
    for name, sha in sources.items():
        assert digest(ROOT / name) == sha, name
    checked = []
    for stage, archive in [
        ("", "round1-source.tar.gz"),
        ("round2", "round2-final-source.tar.gz"),
        ("round3", "round3-source.tar.gz"),
        ("final-v1", "final-v1-source.tar.gz"),
    ]:
        with tarfile.open(ROOT / "snapshots" / archive) as tf:
            hashes = {m.name.removeprefix("./"): hashlib.sha256(tf.extractfile(m).read()).hexdigest()
                      for m in tf if m.isfile()}
        for result in sorted((ROOT / "results" / stage).glob("gpu*.json")):
            data = json.loads(result.read_text())
            if data.get("args", {}).get("phase") not in ("benchmark", "verify"):
                continue
            assert data["status"] == "completed", result
            assert data["environment_before"]["sources"] == data["environment_after"]["sources"], result
            for name, sha in data["environment_before"]["sources"].items():
                assert hashes[name] == sha, (result, name)
            checked.append(result.relative_to(ROOT).as_posix())
    frozen = json.loads((ROOT / "results/SNAPSHOT_CHECK.json").read_text())
    assert sorted(checked) == sorted(frozen["checked"])
    # The collector writes outputs, so recompute in a disposable copy only.
    with tempfile.TemporaryDirectory(prefix="topk-publication-") as tmp:
        copy = Path(tmp) / "run"
        shutil.copytree(ROOT, copy)
        run = subprocess.run([sys.executable, str(copy / "collect_results.py")],
                             check=True, capture_output=True, text=True)
        for name in ("AUDIT.json", "FINAL_SUMMARY.json"):
            assert (copy / "results" / name).read_bytes() == (ROOT / "results" / name).read_bytes(), name
    print(f"Verified {len(export['files'])} public mappings, {len(sources)} final source files, "
          f"{len(checked)} frozen runner identities; raw-sample summaries recomputed exactly")


if __name__ == "__main__":
    main()
