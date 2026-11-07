import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
summary=[]
for suite in ('small','large','mixed'):
    runs=[json.loads((root/f'{suite}-{order}.json').read_text()) for order in ('forward','reverse')]
    assert all(run['status']=='completed' for run in runs)
    for index in range(len(runs[0]['cases'])):
        pair=[run['cases'][index] for run in runs]
        record={'case':pair[0]['case'],'metrics':{},'correctness':[p['checks'] for p in pair]}
        for execution in ('graph','eager'):
            record['metrics'][execution]={}
            for name in pair[0]['timings'][execution]:
                rounds=[]
                for p in pair:
                    t=p['timings'][execution][name]
                    assert (t['warmup'],t['repeats'],t['inner'])==(20,30,10)
                    assert len(t['samples_us'])==len(t['wall_samples_us'])==30
                    if t['coefficient_of_variation']>.1:
                        assert len(t['extra_rounds'])==3
                    rounds += [t]+t.get('extra_rounds',[])
                record['metrics'][execution][name]={
                    'conservative_p50_us':max(t['p50_us'] for t in rounds),
                    'conservative_p90_us':max(t['p90_us'] for t in rounds),
                    'conservative_wall_p50_us':max(t['wall_p50_us'] for t in rounds),
                    'max_cv':max(t['coefficient_of_variation'] for t in rounds),
                    'all_rounds_stable':all(t['coefficient_of_variation']<=.1 for t in rounds),
                    'round_count':len(rounds),
                    'peak_allocated_bytes':max(t['peak_allocated_bytes'] for t in rounds),
                }
        summary.append(record)
(root/'SUMMARY.json').write_text(json.dumps({'selection':'max P50 over both orders and all retry rounds','cases':summary},indent=2)+'\n')
lines=['| Q | Req | Index K | Mode | DG µs | Current µs | Ant µs | Ant/current |',
       '|---:|---:|---:|---|---:|---:|---:|---:|']
for r in summary:
    c=r['case'];t=r['metrics']['graph']
    v=[t[n+'_pipeline']['conservative_p50_us'] for n in ['deepgemm','current_keys_major','ant_keys_major']]
    lines.append(f"|{c['rows']}|{c['requests']}|{c['index_keys']}|{c['mode']}|{v[0]:.2f}|{v[1]:.2f}|{v[2]:.2f}|{v[2]/v[1]:.3f}|")
(root/'TABLE.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
