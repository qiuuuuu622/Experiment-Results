import ast
import hashlib
import json
from pathlib import Path

repo=Path(__file__).resolve().parents[6]
root=Path(__file__).resolve().parents[1]
driver=repo/'benchmarks/kernels/dsv41_indexer/benchmark_prefill_model.py'
module=ast.parse(driver.read_text())
cases=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
config=json.loads((root/'architecture/model-config.json').read_text())['text_config']
rows=[]
for role in ('full','source','consumer'):
    ratio=2 if role=='full' else 1
    for name,queries,contexts in cases:
        if role=='full' and name=='initial_32k':continue
        width=(max(contexts)//ratio+63)//64*64
        capacity=(1<<30)//(4*width)
        chunks=[min(capacity,q-i) for q in queries for i in range(0,q,capacity)]
        visible=[(end-q+j+1)//ratio for q,end in zip(queries,contexts) for j in range(q)]
        rows.append(dict(name=name,role=role,layers=[2,8,14] if role=='full' else [20] if role=='source' else [24,28,32,36],
            queries_per_request=queries,prefixes_original_tokens=[e-q for e,q in zip(contexts,queries)],
            logical_contexts=contexts,ratio=ratio,actual_index_keys=[e//ratio for e in contexts],
            index_heads=32,index_dim=128,topk=512,candidate_blocks=2048,candidate_block_size=8,
            visible_scores=sum(visible),allocated_scores=sum(queries)*width,
            score_arena_bytes=max(chunks)*width*4,query_chunk_rows=chunks,
            additional_source_blockmax_bytes=max(chunks)*width//8*4 if role=='source' else 0,
            common_key_prepare_bytes=len(queries)*width*132,
            conservative_resident_estimate_bytes=3*max(chunks)*width*4+2*(1<<30),
            bypass=False,workload='architecture-aligned synthetic module fixture; not a deployment default'))
assert len(rows)==23 and all(x['score_arena_bytes']<=1<<30 for x in rows)
(root/'MATRIX.json').write_text(json.dumps(dict(model_config_sha256=hashlib.sha256((root/'architecture/model-config.json').read_bytes()).hexdigest(),
    cases=rows,excluded=[dict(role='full',case='initial_32k',reason='extra long-initial points specifically cover source/consumer candidate cutoff; staged matrix')],
    short_context_bypass=[dict(ratio=r,logical_context=c,eager_bypass=c//r<=512,capture_bypass=False) for r,c in [(1,512),(1,513),(2,1024),(2,1025),(2,1026)]]),indent=2)+'\n')
print(len(rows), max(x['score_arena_bytes'] for x in rows))
