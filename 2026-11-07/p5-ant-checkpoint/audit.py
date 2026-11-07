"""CPU-only inventory of the completed synthetic P5b screening evidence."""
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOTS = ['p5-ant-keys-major', 'p5-ant-grid', 'p5-ant-review']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(path.read_text())

files = {}
for root in ROOTS:
    for path in sorted((BASE / root).rglob('*')):
        if path.is_file() and 'model-aligned' not in path.parts:
            files[str(path.relative_to(BASE))] = {'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size}
checks, issues = [], []
def check(label, ok, detail=None):
    entry = dict(check=label, passed=bool(ok))
    if detail is not None:
        entry['detail'] = detail
    checks.append(entry)
    if not ok:
        issues.append(entry)

p = BASE / ROOTS[0]
identity = read(p / 'runtime-identity.json')
archive = p / 'frozen/source.tar.gz'
check('frozen source archive identity', sha(archive.read_bytes()) == identity['source_archive_sha256'])
with tarfile.open(archive) as t:
    members = {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
for name, expected in identity['source_files'].items():
    check('frozen source: ' + name, name in members and sha(members[name]) == expected)
for root in ROOTS[1:]:
    manifest = read(BASE / root / 'MANIFEST.json')
    for name, expected in manifest['files'].items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        target = BASE / root / name
        check('manifest: ' + root + '/' + name, target.is_file() and sha(target.read_bytes()) == expected)
for name, expected in read(BASE / 'p5-ant-review/BENCHMARK_REVIEW_SOURCE.json')['files'].items():
    check('reviewed frozen source: ' + name, name in members and sha(members[name]) == expected)
for entry in read(p / 'upstream/SOURCE.json')['files']:
    check('upstream: ' + entry['path'], sha((p / 'upstream' / entry['path']).read_bytes()) == entry['sha256'])
check('grid snapshot is identical', sha((BASE / 'p5-ant-grid/frozen-p5-source.tar.gz').read_bytes()) == identity['source_archive_sha256'])

measurement_count = retry_count = 0
cases = []
def measurement(m, label):
    global measurement_count, retry_count
    rounds = [m] + m.get('extra_rounds', [])
    check(label + ' noise retries', m['coefficient_of_variation'] <= .1 or len(rounds) == 4)
    retry_count += len(rounds) - 1
    for r in rounds:
        measurement_count += 1
        check(label + ' sample protocol', r['warmup'] == 20 and r['repeats'] == 30 and r['inner'] == 10 and len(r['samples_us']) == 30 and len(r['wall_samples_us']) == 30)

for suite in ['small', 'large', 'mixed']:
    forward = read(p / (suite + '-forward.json'))
    reverse = read(p / (suite + '-reverse.json'))
    check(suite + ' forward/reverse cases', sorted(json.dumps(c['case'], sort_keys=True) for c in forward['cases']) == sorted(json.dumps(c['case'], sort_keys=True) for c in reverse['cases']))
    for result in [forward, reverse]:
        check(suite + ' completed', result['status'] == 'completed')
        for name, expected in result['source'].items():
            check(suite + ' runtime source: ' + name, sha(members[name]) == expected)
        for c in result['cases']:
            for execution, variants in c['timings'].items():
                for variant, m in variants.items():
                    measurement(m, suite + '/' + execution + '/' + variant)
    for c in forward['cases']:
        item = dict(c['case'])
        item['suite'] = suite
        item['role_scope'] = 'synthetic API mode; no model layer bound'
        item['candidate_layout'] = 'request_shared external random blocks' if item['mode'] == 'consumer' else 'none'
        item['architecture_exclusion'] = 'ratio2 consumer is not an actual model role' if item['mode'] == 'consumer' and item['ratio'] == 2 else None
        cases.append(item)
check('23 synthetic shapes', len(cases) == 23)
grid_records = 0
for path in sorted((BASE / 'p5-ant-grid').glob('gpu*-shape*.json')):
    result = read(path)
    check(path.name + ' completed', result['status'] == 'completed')
    check(path.name + ' snapshot', result['input_snapshot_sha256'] == identity['source_archive_sha256'])
    for name, expected in result['source_sha256'].items():
        data = (BASE / 'p5-ant-grid/benchmark_ant_grid.py').read_bytes() if name.endswith('/benchmark_ant_grid.py') else members.get(name)
        check(path.name + ' source: ' + name, data is not None and sha(data) == expected)
    for record in result['records']:
        grid_records += 1
        runs = record['runs']
        measurement(dict(runs[0], extra_rounds=runs[1:]), path.name + '/' + record['variant'])
check('192 grid variant/phase/order records', grid_records == 192)
for stem in ['edges', 'edges-chunked']:
    data = read(p / (stem + '.json'))
    check(stem + ' PTX hash', sha((p / (stem + '.ptx')).read_bytes()) == data['ptx_sha256'])
    check(stem + ' passed', data['status'] == 'passed')

architecture = p / 'model-aligned/architecture/model-config.json'
config = read(architecture)['text_config']
roles = []
for layer in config['index_source_layer_ids']:
    role = 'source' if layer == config['candidate_source_layer_id'] else ('consumer' if layer > config['candidate_source_layer_id'] else 'full')
    roles.append(dict(layer=layer, ratio=config['compress_ratios'][layer], role=role))
limitations = [
    'Synthetic 23-shape suite and 4 independent single-GPU grid cases, not the model-aligned main matrix and not TP4 latency.',
    'Synthetic consumer candidates are random request_shared external blocks, not per-query candidates produced by model source layer 20.',
    'Ratio2 consumer is an API stress case, not an actual model role; model ratio2 full layers are 2/8/14, ratio1 source is 20, ratio1 consumers are 24/28/32/36.',
    'runtime-identity.json records 588 native file hashes post-run; this checkpoint checks source archive bytes but does not independently rerun import or rehash the remote binary/header tree.',
    'Native wheel binary build commit is unknown; version 0.32.0 does not establish current HEAD provenance.',
    'verify-v2/v3/v4 and edges historical JSON lack their own frozen source hash; retain as historical correctness evidence without inventing per-run exact binding.',
    'CPU review is not GPU verification. New model-aligned results are intentionally excluded because the P5 owner is still producing them.',
]
report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='synthetic_checkpoint_audited' if not issues else 'audit_issues', gpu_experiments_performed=False, source_files_modified=False, published=False, evidence_files=files, checks=checks, issues=issues, synthetic_cases=cases, grid_records=grid_records, measured_rounds=measurement_count, extra_noise_rounds=retry_count, native_identity=dict(capture=identity['native_identity_capture'], file_count=len(identity['native_files']), versions=identity['versions'], binary_build_commit=identity['binary_build_commit']), architecture_reference=dict(path=str(architecture.relative_to(BASE)), sha256=sha(architecture.read_bytes()), roles=roles, use='scope interpretation only; not model-aligned experiment completion'), limitations=limitations)
(OUT / 'INVENTORY.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(dict(files=len(files), checks=len(checks), issues=len(issues), measured_rounds=measurement_count, extra_noise_rounds=retry_count, grid_records=grid_records)))
if issues:
    print(json.dumps(issues, indent=2)); raise SystemExit(1)
