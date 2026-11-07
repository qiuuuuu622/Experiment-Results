"""Read-only verification of archived four-rank P5c correctness evidence."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent.parent / 'p5-perquery-tp4'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((base / 'MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert sha(base / name) == expected, name
run = base / 'final-v1'
before = json.loads((run / 'runtime-before.json').read_text())
after = json.loads((run / 'runtime-after.json').read_text())
for key in ('source_files', 'native_files', 'versions'):
    assert before[key] == after[key], key
assert sha(run / 'frozen/source.tar.gz') == before['source_archive_sha256']
ranks = [json.loads((run / f'rank{i}.json').read_text()) for i in range(4)]
checks = rows = 0
for rank, report in enumerate(ranks):
    assert report['status'] == 'completed' and report['rank'] == rank
    assert report['world_size'] == 4
    assert report['provenance_before'] == report['provenance_after']
    assert len({d['uuid'] for d in report['devices']}) == 4
    assert report['core_sha256'] == '77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c'
    assert len(report['cases']) == 3
    for case in report['cases']:
        identities = case['identities']
        assert len({d['physical_table_sha256'] for d in identities}) == 4
        assert len({d['logical_cache_sha256'] for d in identities}) == 1
        assert all(d['logical_fields'] == identities[0]['logical_fields'] for d in identities)
        assert not case['collectives_send_physical_pages']
        assert len(case['phases']) == 7
        for phase in case['phases']:
            assert phase['graph_and_eager_variants'] == 12
            assert phase['exact_rows_per_variant'] == sum(case['sizes'])
            checks += 12
            rows += 12 * sum(case['sizes'])
for c in range(3):
    for p in range(7):
        phases = [r['cases'][c]['phases'][p] for r in ranks]
        assert len({x['logical_output_sha256'] for x in phases}) == 1
        assert len({x['local_physical_table_sha256'] for x in phases}) == 4
summary = json.loads((base / 'summary.json').read_text())
assert checks == summary['phase_variant_execution_rank_checks'] == 1008
assert rows == summary['logical_row_comparisons'] == 86016
assert not summary['performance_claim']
result = dict(status='passed', manifest_files=len(manifest['files']), comparisons=checks,
              logical_row_comparisons=rows, distinct_gpu_uuids=4,
              identical_logical_inputs_distinct_physical_tables=True,
              source_native_versions_unchanged=True, performance_claim=False,
              limits='CPU artifact consistency check; GPU computations are reported by frozen verifier')
output = Path(__file__).with_name('TP4_AUDIT.json')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
