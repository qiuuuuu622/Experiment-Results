"""Recompute summary and audit counts from all completed experiment outputs."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = []
checks = []
failures = []
benchmarks = []
for path in sorted((ROOT / "results").rglob("*.json")):
    if path.name.startswith("trace-"): continue
    data = json.loads(path.read_text())
    if not isinstance(data,dict) or not isinstance(data.get("args"),dict):continue
    phase = data["args"].get("phase")
    if phase not in ("compile","verify","benchmark"):continue
    if data["status"] != "completed":
        failures.append(dict(path=path.relative_to(ROOT).as_posix(),status=data["status"],error=data.get("error")))
        continue
    assert data["environment_before"]["sources"]==data["environment_after"]["sources"],path
    if phase=="verify":
        for c in data["checks"]:
            assert c["status"]=="passed"
            checks.append(c)
    elif phase=="benchmark":
        benchmarks.append(path.relative_to(ROOT).as_posix())
        for c in data["cases"]:
            assert c["same_input_bytes_before_after"]
            for name,check in c["correctness"].items():
                if name!="native":assert check["stable_exact"],(path,name)
            for m in c["measurements"]:
                rounds=[m["metric"]]+m["metric"].get("extra_rounds",[])
                for r in rounds:
                    assert r["warmup"]>=20 and r["repeats"]>=30 and r["inner"]>=10
                    assert len(r["samples_us"])==len(r["wall_samples_us"])==r["repeats"]
                    assert abs(statistics.median(r["samples_us"])-r["p50_us"])<1e-8
                records.append(dict(file=path.relative_to(ROOT).as_posix(),rows=c["rows"],width=c["width"],
                                    distribution=c.get("distribution","random"),score_sha256=c["score_sha256"],
                                    visible_lengths=c.get("visible_lengths"),variant=m["variant"],
                                    execution=m["execution"],order=m["order"],rounds=rounds))

final=collections.defaultdict(lambda:collections.defaultdict(list))
hashes=collections.defaultdict(set)
for r in records:
    if "/final-v1/" not in r["file"]:continue
    key=(r["rows"],r["width"],r["execution"])
    final[key][r["variant"]].extend(r["rounds"])
    hashes[key].add(r["score_sha256"])
summary=[]
for key,variants in sorted(final.items()):
    assert len(hashes[key])==1,key
    entry=dict(rows=key[0],width=key[1],execution=key[2],score_sha256=next(iter(hashes[key])),variants={})
    for name,rounds in variants.items():
        samples=[s for r in rounds for s in r["samples_us"]]
        walls=[s for r in rounds for s in r["wall_samples_us"]]
        entry["variants"][name]=dict(p50_us=statistics.median(samples),p90_us=sorted(samples)[int(.9*(len(samples)-1))],
                                     min_round_p50_us=min(r["p50_us"] for r in rounds),max_round_p50_us=max(r["p50_us"] for r in rounds),
                                     max_cv=max(r["coefficient_of_variation"] for r in rounds),
                                     wall_p50_us=statistics.median(walls),rounds=len(rounds),samples=len(samples))
    summary.append(entry)
audit=dict(status="passed",completed_benchmark_files=len(benchmarks),measurement_records=len(records),
           measurement_rounds=sum(len(r["rounds"]) for r in records),
           gpu_samples=sum(len(m["samples_us"]) for r in records for m in r["rounds"]),
           verification_configurations=len(checks),verification_stages=sum(len(c["stages"]) for c in checks),
           retained_failures=failures,final_input_hashes_exact=True,
           source_identity_checks="before/after all completed runner processes",scope="selector only; native tie policy not asserted deterministic")
(ROOT/"results/AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
(ROOT/"results/FINAL_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in audit.items() if k!="retained_failures"},indent=2))
print('retained failures:',len(failures))
for e in summary:
    if e["execution"]=="graph":
        print(e["rows"],e["width"],{k:round(v["p50_us"],3) for k,v in e["variants"].items()})
