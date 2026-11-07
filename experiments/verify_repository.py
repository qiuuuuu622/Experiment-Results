"""Verify every published artifact against the public SHA256 manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SHA256SUMS.json").read_text())
expected = {x["path"] for x in manifest["files"]}
actual = {str(p.relative_to(root)) for p in root.rglob("*")
          if p.is_file() and ".git" not in p.parts
          and p.name != "SHA256SUMS.json"}
assert actual == expected, dict(missing=sorted(expected-actual), extra=sorted(actual-expected))
for item in manifest["files"]:
    path = root / item["path"]
    assert path.stat().st_size == item["bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
print(f"Verified {len(expected)} published files")

# Internal stage digests retain private-run provenance; check the explicit
# public artifact mapping after privacy-preserving export instead.
dependencies = root / "PUBLIC_DEPENDENCIES.json"
if dependencies.exists():
    entries = json.loads(dependencies.read_text())["files"]
    for item in entries:
        artifact = root / item["public_path"]
        assert artifact.is_file(), artifact
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == item["public_sha256"], artifact
    print(f"Verified {len(entries)} public evidence mappings")
