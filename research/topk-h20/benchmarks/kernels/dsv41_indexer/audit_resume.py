# SPDX-License-Identifier: Apache-2.0
"""Read-only provenance and sampling audit of archived P4/P5/P6 evidence."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sample_records(value, location="root"):
    if isinstance(value, dict):
        if "samples_us" in value:
            yield location, value
        for key, child in value.items():
            yield from sample_records(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from sample_records(child, f"{location}[{index}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path("results/indexer/20261106-tp4")
    audit = dict(status="audited", phases={}, archives=[])
    for phase in ["p4", "p5", "p6"]:
        records = []
        for path in sorted((root / phase).glob("*.json")):
            data = json.loads(path.read_text())
            samples = list(sample_records(data))
            violations = []
            noise = []
            for loc, item in samples:
                if (
                    len(item["samples_us"]) != 30
                    or item.get("warmup") != 20
                    or item.get("inner") != 10
                ):
                    violations.append(loc)
                if item.get("coefficient_of_variation", 0) > 0.1:
                    noise.append(
                        dict(
                            location=loc, extra_rounds=len(item.get("extra_rounds", []))
                        )
                    )
            records.append(
                dict(
                    path=str(path),
                    sha256=sha(path.read_bytes()),
                    status=data.get("status"),
                    checks=len(data.get("checks", [])),
                    ragged=len(data.get("ragged", [])),
                    cases=len(data.get("cases", [])),
                    measurement_nodes=len(samples),
                    sampling_violations=violations,
                    high_cv=noise,
                    log_exists=path.with_suffix(".log").exists(),
                )
            )
        audit["phases"][phase] = records
    for path in sorted(root.glob("p[456]/*.tgz")):
        members = []
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                data = archive.extractfile(member).read()
                current = Path(member.name)
                if member.name.startswith("scripts/"):
                    current = Path("benchmarks/kernels/dsv41_indexer") / current.name
                members.append(
                    dict(
                        path=member.name,
                        sha256=sha(data),
                        current_exists=current.exists(),
                        current_equal=current.exists()
                        and sha(current.read_bytes()) == sha(data),
                    )
                )
        audit["archives"].append(
            dict(path=str(path), sha256=sha(path.read_bytes()), files=members)
        )
    expected = {
        "p4/source-final.tgz": (root / "p4/checkpoint.sha256").read_text().split()[0],
        "p5/source.tgz": json.loads((root / "p5/checkpoint.json").read_text())[
            "source_tar_sha256"
        ],
        "p5/gluon-source.tgz": (root / "p5/gluon-source-sha256.txt")
        .read_text()
        .splitlines()[-1]
        .split()[0],
    }
    audit["anchored_archive_hashes"] = [
        dict(
            path=key,
            expected=value,
            actual=sha((root / key).read_bytes()),
            match=sha((root / key).read_bytes()) == value,
        )
        for key, value in expected.items()
    ]
    p6 = json.loads((root / "p6/source-sha256.json").read_text())["files"]
    p6archive = next(
        x for x in audit["archives"] if x["path"].endswith("p6/source-snapshot.tgz")
    )
    archived = {x["path"]: x["sha256"] for x in p6archive["files"]}
    audit["p6_manifest_vs_snapshot"] = [
        dict(
            **x,
            archive_path=(
                "scripts/06_topk.sh" if x["path"].endswith("06_topk.sh") else x["path"]
            ),
            archive_match=archived.get(
                "scripts/06_topk.sh" if x["path"].endswith("06_topk.sh") else x["path"]
            )
            == x["sha256"],
        )
        for x in p6
    ]
    Path(args.output).write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                archives=len(audit["archives"]),
                anchored_hashes_match=all(
                    x["match"] for x in audit["anchored_archive_hashes"]
                ),
                p6_manifest_match=all(
                    x["archive_match"] for x in audit["p6_manifest_vs_snapshot"]
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
