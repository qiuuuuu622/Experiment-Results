"""Read-only publication gate; incomplete final runs must not pass."""
import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(record, *, retry=False):
    assert (record['warmup'],record['repeats'],record['inner']) == (20,30,10)
    samples=record['samples_us'];wall=record['wall_samples_us']
    assert len(samples)==len(wall)==30
    assert all(math.isfinite(x) and x>0 for x in samples+wall)
    cv=statistics.pstdev(samples)/statistics.mean(samples)
    expected={'p50_us':statistics.median(samples),
              'p90_us':sorted(samples)[int(.9*(len(samples)-1))],
              'coefficient_of_variation':cv,'wall_p50_us':statistics.median(wall)}
    for key,value in expected.items():
        assert math.isclose(record[key],value,rel_tol=1e-9,abs_tol=1e-12),(key,record[key],value)
    if not retry and cv>.1:assert len(record.get('extra_rounds',[]))==3
    return 1+sum(measure(x,retry=True) for x in record.get('extra_rounds',[]))


def audit(root):
    directory=root/'p5c-perquery/final-v1'
    before_path=directory/'runtime-before.json'
    before=json.loads(before_path.read_text())
    archive=directory/'frozen/source.tar.gz'
    assert digest(archive)==before['source_archive_sha256']=='b4c1a3dbda6778ea48d226b44ff77809dce604a50fd63b9b3aac15184e83e41b'
    with tarfile.open(archive) as tar:
        archived={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                  for m in tar if m.isfile()}
    for path,sha in before['source_files'].items():assert archived[path]==sha,path
    assert archived['vllm_dsv41_indexer/perquery_prefill.py']=='77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c'
    expected={'a':['initial_2k','initial_8k','initial_32k','steady_16k'],
              'b':['steady_64k','steady_256k','tail_1m','ragged']}
    files=[];rounds=records=0;seen={};candidate_hash_differences=[]
    for group,shapes in expected.items():
        after=json.loads((directory/f'runtime-after-{group}.json').read_text())
        assert before['source_files']==after['source_files']
        assert before['native_files']==after['native_files']
        for order in ('forward','reverse'):
            path=directory/f'group-{group}-{order}.json'
            report=json.loads(path.read_text())
            assert report['status']=='completed' and report['purpose']=='performance'
            assert report['identity_sha256']==digest(before_path)
            assert report['args']['reverse']==(order=='reverse')
            assert [case['case'] for case in report['cases']]==shapes
            for case in report['cases']:
                assert case['heads']==32 and case['head_dim']==128 and case['ratio']==1
                assert case['source_generation_in_consumer_pipeline'] is False
                for workspace in case['workspaces'].values():
                    assert workspace['score_bytes']<=1<<30
                    chunks=workspace['chunks'];assert chunks[0][0]==0 and chunks[-1][1]==case['rows']
                    assert all(a[1]==b[0] for a,b in zip(chunks,chunks[1:]))
                phases={(r['variant'],r['phase'],r['execution']) for r in case['records']}
                for variant in ('full_deepgemm_mask','full_keys_major_mask','direct','gathered_wgmma'):
                    for execution in ('eager','graph'):assert (variant,'pipeline',execution) in phases
                for record in case['records']:
                    rounds+=measure(record['measurement']);records+=1
                key=case['case'];stable={k:case[k] for k in ('query_lens','contexts','rows','candidate_sha256')}
                if key in seen:
                    previous=seen[key]
                    assert all(previous[k]==stable[k] for k in ('query_lens','contexts','rows')),key
                    if previous['candidate_sha256']!=stable['candidate_sha256']:
                        candidate_hash_differences.append(dict(case=key,forward=previous['candidate_sha256'],reverse=stable['candidate_sha256']))
                else:seen[key]=stable
            files.append(dict(path=str(path.relative_to(root)),sha256=digest(path)))
    assert len(seen)==8
    result=dict(status='pending_candidate_identity' if candidate_hash_differences else 'passed',candidate_hash_differences=candidate_hash_differences,shapes=8,orders=2,measurement_records=records,
                sample_rounds=rounds,gpu_samples=30*rounds,wall_samples=30*rounds,
                source_archive_sha256=digest(archive),source_files=len(before['source_files']),
                native_files=len(before['native_files']),files=files,
                limits='CPU sample/hash/workspace coverage gate; not independent numerical recomputation or TP4 acceptance')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:result=audit(args.root)
    except (AssertionError,FileNotFoundError,KeyError) as error:
        result=dict(status='not_ready_or_failed',reason=repr(error))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 1)
