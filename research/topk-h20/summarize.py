"""Summarize every retained sample, preserving order and runtime identity."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
summary = []
for path in sorted((ROOT / "results").glob("gpu*-benchmark.json")):
    report = json.loads(path.read_text())
    for case in report["cases"]:
        for mode in ("eager", "graph"):
            variants = {}
            for rec in case["measurements"]:
                if rec["execution"] != mode: continue
                m = rec["metric"]
                rounds = [m] + m.get("extra_rounds", [])
                samples = [s for r in rounds for s in r["samples_us"]]
                wall = [s for r in rounds for s in r["wall_samples_us"]]
                variants.setdefault(rec["variant"], {})[rec["order"]] = dict(
                    p50_us=statistics.median(samples), p90_us=sorted(samples)[int(.9*(len(samples)-1))],
                    wall_p50_us=statistics.median(wall), min_round_p50_us=min(r["p50_us"] for r in rounds),
                    max_round_p50_us=max(r["p50_us"] for r in rounds), max_cv=max(r["coefficient_of_variation"] for r in rounds),
                    rounds=len(rounds), samples=len(samples))
            if not variants: continue
            stable = {k:v for k,v in variants.items() if k != "native" and set(v)=={"forward","reverse"}}
            winner = min(stable, key=lambda k:max(x["max_round_p50_us"] for x in stable[k].values())) if stable else None
            summary.append(dict(source=path.relative_to(ROOT).as_posix(),status=report["status"],
                                gpu=report["environment_before"]["gpu_uuid"],rows=case["rows"],width=case["width"],
                                execution=mode,variants=variants,stable_winner=winner))
(ROOT / "results/SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
for row in summary:
    if row["execution"] != "graph":continue
    print(row["rows"],row["width"],"winner",row["stable_winner"],
          {k:round(max(x["max_round_p50_us"] for x in v.values()),2) for k,v in row["variants"].items()})
