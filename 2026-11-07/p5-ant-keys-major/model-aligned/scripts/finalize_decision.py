import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
main=root/'final-v3'
isolated=root/'isolated-source'
comparison=[]
for order in ('forward','reverse'):
    run=json.loads((isolated/f'source-{order}.json').read_text())
    assert run['status']=='completed' and len(run['cases'])==1
    case=run['cases'][0]
    t=case['timings']['graph']
    value={n:t[n+'_pipeline']['p50_us'] for n in ['deepgemm','current_keys_major','ant']}
    comparison.append(dict(order=order,pipeline_p50_us=value,
        ant_speedup_over_best=min(value['deepgemm'],value['current_keys_major'])/value['ant'],
        max_cv=max(t[n+'_pipeline']['coefficient_of_variation'] for n in value)))
for directory in (main,isolated):
    before=json.loads((directory/'runtime-before.json').read_text())
    after=json.loads((directory/'runtime-after.json').read_text())
    assert before['source_files']==after['source_files']
    assert before['native_files']==after['native_files']
    assert before['source_archive_sha256']==after['source_archive_sha256']
value=dict(status='research_complete_no_dispatch_change',
    scope='P5b dedicated Ant scorer; P5c real perquery compact prefill remains separate',
    primary_model_shapes=23,primary_orders=2,
    primary_candidate=dict(role='source',rows=4096,original_context=262144,ratio=1),
    isolated_same_core_source=True,isolated_probe=comparison,
    admitted_shapes=[],reason='Only primary >=5% candidate did not reproduce; isolated gain <1%',
    causal_attribution='unknown; no controlled isolation of device, layout, clock and thermal factors',
    original_ant_sha256='0bdde39dd1ddb4805492badbd838aa40a51a7dc79f29c070d27b661fc3f66b3f',
    adapted_core_sha256='957557aa4ac9944073ae4e2c2a370775df446f93437fa82de821b3051d3184c9',
    model_driver_sha256='f422bdd3a5593e031c16e9270f0e93fb3dd343402c4eacd13cfe087987c7ef49',
    primary_source_archive_sha256=json.loads((main/'runtime-before.json').read_text())['source_archive_sha256'])
(root/'DECISION.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
