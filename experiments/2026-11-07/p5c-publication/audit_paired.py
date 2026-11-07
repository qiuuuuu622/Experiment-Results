"""Independent publication audit of same-object paired supplement, not old tensor recovery."""
import hashlib
import json
from pathlib import Path
from audit_final import measure


def audit(root):
    base = root / 'p5c-perquery'
    inventory = json.loads((base / 'INVENTORY.json').read_text())
    for entry in inventory['files']:
        path = base / entry['path']
        assert path.stat().st_size == entry['bytes'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
    paired = base / 'paired-v1'
    cases = []; records = rounds = 0; changed = []
    keys = {'q', 'weights', 'cache', 'table', 'requests', 'lengths',
            'source_q', 'source_weights', 'candidate_raw', 'candidate_canonical'}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for group in ('a', 'b'):
        before_path = paired / f'runtime-before-{group}.json'
        before = json.loads(before_path.read_text())
        after = json.loads((paired / f'runtime-after-{group}.json').read_text())
        for key in ('source_files', 'native_files', 'versions'):
            assert before[key] == after[key], key
        report = json.loads((paired / f'group-{group}.json').read_text())
        assert report['status'] == 'completed'
        assert report['old_raw_canonical_hash_reconstructed'] is False
        assert report['identity_sha256'] == sha(before_path)
        assert report['runner_sha256'] == sha(paired / 'paired_pipeline.py')
        assert report['frozen_core']['sha256'] == '77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c'
        for case in report['cases']:
            assert set(case['input_before']) == keys
            assert case['input_before'] == case['input_after']
            assert case['input_bytes_unchanged']
            repetitions = case['source_repeats']
            assert len(repetitions) == 3
            canonical = case['input_before']['candidate_canonical']
            assert all(x['canonical'] == canonical for x in repetitions)
            raw_changed = len({x['raw']['sha256'] for x in repetitions}) > 1
            assert case['source_raw_order_changed'] == raw_changed
            if raw_changed: changed.append(case['case'])
            if case['paired_pipeline']:
                object_ids = case['path_object_identity']
                for field in ('candidate_data_ptr', 'q_data_ptr', 'cache_data_ptr'):
                    assert len({x[field] for x in object_ids.values()}) == 1
                combos = {(x['order'], x['execution'], x['variant']) for x in case['records']}
                expected = {(o,e,v) for o in ('forward','reverse') for e in ('eager','graph') for v in ('full_deepgemm_mask','full_keys_major_mask','direct','gathered_wgmma')}
                assert combos == expected and len(case['records']) == 16
            else:
                assert not case['records']
            for record in case['records']:
                rounds += measure(record['measurement']); records += 1
            cases.append(case['case'])
    assert len(cases) == len(set(cases)) == 8
    assert records == rounds == 64
    return dict(status='passed', inventory_files=len(inventory['files']), canonical_cases=8,
                paired_cases=4, records=records, rounds=rounds, samples=rounds*30,
                input_byte_fields_checked=sorted(keys), same_objects_and_unchanged_inputs=True,
                source_three_canonical_sets_identical=True, unordered_source_changed_cases=changed,
                old_raw_canonical_hash_reconstructed=False,
                limits='Independent supplementary rerun; old performance outputs lacked canonical hashes and their unsaved tensors are not reconstructed')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    result = audit(root)
    Path(__file__).with_name('PAIRED_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
