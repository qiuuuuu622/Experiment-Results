import json
import math
import statistics
from pathlib import Path

root=Path(__file__).resolve().parents[1]
run=root/'final-v3'
records=[]
for role in ('full','source','consumer'):
    reports=[json.loads((run/f'{role}-{order}.json').read_text()) for order in ('forward','reverse')]
    assert all(x['status']=='completed' for x in reports)
    assert len(reports[0]['cases'])==len(reports[1]['cases'])==(7 if role=='full' else 8)
    for a,b in zip(reports[0]['cases'],reports[1]['cases']):
        assert a['case']==b['case']
        item=dict(case=a['case'],metrics={},correctness=[a['checks'],b['checks']],
                  planning=[a['planning'],b['planning']],candidate_stats=a['candidate_stats'])
        for mode in ('graph','eager'):
            item['metrics'][mode]={}
            for name,t in a['timings'][mode].items():
                rounds=[]
                for c in (a,b):
                    base=c['timings'][mode][name]
                    if base['coefficient_of_variation']>.1:assert len(base['extra_rounds'])==3
                    rounds += [base]+base.get('extra_rounds',[])
                for t in rounds:
                    assert (t['warmup'],t['repeats'],t['inner'])==(20,30,10)
                    assert len(t['samples_us'])==len(t['wall_samples_us'])==30
                    assert math.isclose(statistics.median(t['samples_us']),t['p50_us'],rel_tol=1e-9)
                    assert math.isclose(statistics.pstdev(t['samples_us'])/statistics.mean(t['samples_us']),t['coefficient_of_variation'],rel_tol=1e-9)
                    assert math.isclose(statistics.median(t['wall_samples_us']),t['wall_p50_us'],rel_tol=1e-9)
                item['metrics'][mode][name]=dict(
                    p50_us=max(t['p50_us'] for t in rounds),
                    p90_us=max(t['p90_us'] for t in rounds),
                    wall_p50_us=max(t['wall_p50_us'] for t in rounds),
                    max_cv=max(t['coefficient_of_variation'] for t in rounds),
                    stable_all_rounds=all(t['coefficient_of_variation']<=.1 for t in rounds),
                    rounds=len(rounds),peak_allocated_bytes=max(t['peak_allocated_bytes'] for t in rounds))
        t=item['metrics']['graph']
        baseline=min(t['deepgemm_pipeline']['p50_us'],t['current_keys_major_pipeline']['p50_us'])
        item['ant_speedup_over_best_baseline']=baseline/t['ant_pipeline']['p50_us']
        order_tests=[]
        for order,c in zip(('forward','reverse'),(a,b)):
            timings=c['timings']['graph']
            def all_rounds(name):
                base=timings[name+'_pipeline']
                return [base]+base.get('extra_rounds',[])
            baselines=[t for name in ('deepgemm','current_keys_major') for t in all_rounds(name)]
            ant=all_rounds('ant')
            baseline_lower=min(t['p50_us']-2*1.253*statistics.pstdev(t['samples_us'])/math.sqrt(30) for t in baselines)
            ant_upper=max(t['p50_us']+2*1.253*statistics.pstdev(t['samples_us'])/math.sqrt(30) for t in ant)
            ratio=min(t['p50_us'] for t in baselines)/max(t['p50_us'] for t in ant)
            stable=all(t['coefficient_of_variation']<=.1 for t in baselines+ant)
            order_tests.append(dict(order=order,conservative_speedup=ratio,
                noise_guarded_speedup=baseline_lower/ant_upper,
                stable_all_rounds=stable,passes_5pct=stable and baseline_lower/ant_upper>=1.05))
        item['per_order_admission_checks']=order_tests
        item['ant_stable_5pct_win']=all(t['passes_5pct'] for t in order_tests)
        records.append(item)
(run/'SUMMARY.json').write_text(json.dumps(dict(status='completed',cases=records,
    estimator='conservative maximum P50/P90/wall across both orders and all CV retry rounds'),indent=2)+'\n')
lines=['| Role | Shape | Query rows | Original context | Actual max K | DG P50 µs | Current P50 µs | Ant P50 µs | Ant/best speedup |',
       '|---|---|---|---|---:|---:|---:|---:|---:|']
for r in records:
 c=r['case'];t=r['metrics']['graph']
 lines.append('|'+ '|'.join(map(str,[c['role'],c['label'],c['query_lens'],c['contexts'],max(c['key_counts']),
      *[round(t[n+'_pipeline']['p50_us'],2) for n in ['deepgemm','current_keys_major','ant']],
      round(r['ant_speedup_over_best_baseline'],3)]))+'|')
(run/'TABLE.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines));print('stable5pctwins',sum(r['ant_stable_5pct_win'] for r in records))
