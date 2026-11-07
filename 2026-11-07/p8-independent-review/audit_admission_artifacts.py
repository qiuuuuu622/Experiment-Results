"""Independent CPU audit; checks stored evidence bytes, never runs GPU kernels."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--verification", nargs="*", default=[])
args = parser.parse_args()
manifest_path = Path(args.manifest)
data = json.loads(manifest_path.read_text())
issues = []
checked = {}


def check_artifact(value, label):
    if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
        issues.append(dict(label=label, error="Missing artifact path/hash"))
        return
    name, expected = value["path"], value["sha256"]
    path = Path(name)
    if not path.is_file():
        issues.append(dict(label=label, path=name, error="Artifact not found"))
        return
    if name not in checked:
        checked[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if checked[name] != expected:
        issues.append(dict(label=label, path=name, error="SHA256 differs"))


if data.get("status") != "completed" or data.get("bundle_failures"):
    issues.append(dict(label="manifest", error="Incomplete or failed bundles"))
records = data.get("records", [])
for index, record in enumerate(records):
    label = f"record[{index}] {record.get('candidate')}"
    check_artifact(record.get("source_manifest"), label + " source")
    for item in record.get("performance_artifacts", []):
        check_artifact(item, label + " performance")
    for item in record.get("correctness_artifacts", []):
        path, value = item.rsplit(":", 1)
        check_artifact(dict(path=path, sha256=value), label + " correctness")
    native = record.get("native_identity")
    if native is None:
        native = (
            data.get("environments", {})
            .get(record.get("environment_digest"), {})
            .get("native_identity")
        )
    check_artifact(native, label + " native")
    if set(record.get("orders", {})) != {"forward", "reverse"}:
        issues.append(dict(label=label, error="Missing forward/reverse"))
    if record.get("candidate", "").startswith(
        ("grouped_gluon", "shared_consumer")
    ) and not record.get("parameters"):
        issues.append(
            dict(label=label, error="Measured grouped constructor parameters missing")
        )
verification = []
for name in args.verification:
    report = json.loads(Path(name).read_text())
    verification.append(
        dict(
            path=name,
            status=report.get("status"),
            checks=len(report.get("checks", [])),
            skipped=report.get("skipped", []),
        )
    )
    if report.get("status") != "passed" or report.get("skipped"):
        issues.append(
            dict(label=name, error="Verification failed or skipped decisions")
        )
result = dict(
    status="passed" if not issues else "incomplete",
    scope=(
        "CPU artifact integrity only; actual kernel binding/native runtime identity "
        "still require explicit verification report"
    ),
    manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    admitted_records=len(records),
    candidates=dict(collections.Counter(r["candidate"] for r in records)),
    checked_unique_artifacts=len(checked),
    verification=verification,
    issues=issues,
)
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: result[k]
            for k in ("status", "admitted_records", "checked_unique_artifacts")
        }
    )
)
print("issues", len(issues))
