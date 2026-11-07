"""Explicit final publication export, never edits private evidence or pushes git."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--book-sha', required=True)
parser.add_argument('--stages', nargs='+', default=['p5c-perquery', 'p5-perquery-tp4', 'p5-perquery-review', 'p5c-publication'])
parser.add_argument('--execute', action='store_true')
parser.add_argument('--exclude-book', action='store_true')
args = parser.parse_args()
# Previously reviewed publication sanitizer; use the same archive treatment.
code = Path(__file__).with_name('publication_sanitizer.py').read_text()
code = code.replace('return {k:shape(x)for k,x in v.items()}', 'return [(shape(k),shape(x)) for k,x in v.items()]')
code = code.replace("'GPU-UUID-REDACTED',s)", "lambda m: 'GPU-PSEUDONYM-' + hashlib.sha256(m.group().lower().encode()).hexdigest()[:24],s)")
code = code.replace('redact physical GPU UUID; model and device index retained', 'deterministic GPU UUID pseudonym; preserve equality and distinct four-card identities')
exec(code)
book_source = SRC / 'H20优化方案书.md'
assert sha(book_source.read_bytes()) == args.book_sha, 'Root book changed or not approved'
assert re.search(r'\[x\].*P5c', book_source.read_text()), 'Inspect final approved book completion explicitly'
selected = []
for stage in args.stages:
    assert (BASE / stage).is_dir(), stage
    selected.extend(f for f in sorted((BASE / stage).rglob('*')) if f.is_file() and '__pycache__' not in f.parts)
if not args.exclude_book:
    selected.append(book_source)
assert all(f.stat().st_size <= 100 * 1024 * 1024 for f in selected), 'Oversized file'
print(json.dumps(dict(files=len(selected), bytes=sum(f.stat().st_size for f in selected), execute=args.execute)))
if not args.execute:
    raise SystemExit(0)
dep = json.loads((DST / 'PUBLIC_DEPENDENCIES.json').read_text())
historical = None
if not args.exclude_book:
    book = DST / 'H20优化方案书.md'
    historical = DST / '2026-11-07/p5c-publication/PLAN_AT_P5B_PUBLICATION.md'
    assert not historical.exists(), 'Refuse repeat book export'
    historical.parent.mkdir(parents=True, exist_ok=True)
    historical.write_bytes(book.read_bytes())
    for entry in dep['files']:
        if entry['public_path'] == 'H20优化方案书.md':
            assert entry['public_sha256'] == sha(historical.read_bytes())
            entry['public_path'] = str(historical.relative_to(DST))
for f in selected:
    raw = f.read_bytes(); private_hashes[f] = sha(raw)
    relative = f.relative_to(BASE) if f.is_relative_to(BASE) else Path(f.name)
    public_rel = Path('2026-11-07') / relative if f.is_relative_to(BASE) else relative
    if f.name in ('SHA256SUMS.json', 'SHA256SUMS', 'FINAL-SHA256.json'):
        public_rel = public_rel.with_name('ORIGINAL_PRIVATE_' + f.name)
    target = DST / public_rel
    data, edits = sanitize_archive(raw, str(public_rel)) if f.name.endswith(('.tar.gz', '.tar', '.tgz')) else sanitize_text(raw)
    if re.search(r'(^|/)(LICENSE|COPYING|NOTICE)([./_-]|$)', str(relative), re.I):
        assert data == raw, 'License changed'
    assert not target.exists() or f == book_source or target.read_bytes() == data, 'Refuse overwrite different artifact ' + str(target)
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    item = dict(source=replacement + '/' + str(f.relative_to(SRC)), public_path=str(public_rel), original_sha256=sha(raw), public_sha256=sha(data), bytes=len(data))
    origins.append(item)
    if edits or public_rel.name != f.name:
        changes.append(dict(path=str(public_rel), original_sha256=sha(raw), public_sha256=sha(data), changes=edits + ([dict(action='Retain private checksum provenance; root public checksum authoritative')] if public_rel.name != f.name else [])))
for f, h in private_hashes.items():
    assert sha(f.read_bytes()) == h, 'Private evidence changed during export ' + str(f)
keys = {(x['reference'], x['private_original_sha256'], x['public_path']) for x in dep['files']}
for item in origins:
    entry = dict(reference=item['source'], private_original_sha256=item['original_sha256'], public_path=item['public_path'], public_sha256=item['public_sha256'])
    key = (entry['reference'], entry['private_original_sha256'], entry['public_path'])
    if key not in keys:
        dep['files'].append(entry); keys.add(key)
# Preserve the exact repo-relative references used by the final admission.
manifest_path = BASE / 'p5-perquery-review/admission.json'
manifest = json.loads(manifest_path.read_text())
origin_map = {x['source']: x for x in origins}
review = json.loads((BASE / 'p5-perquery-review/REVIEW-MANIFEST.json').read_text())
references = manifest['artifact_sha256'] | review['artifact_sha256'] | review['evidence_dependencies']
for reference, expected in references.items():
    private = SRC / reference
    assert sha(private.read_bytes()) == expected, reference
    item = origin_map[replacement + '/' + reference]
    entry = dict(reference=reference, private_original_sha256=expected,
                 public_path=item['public_path'], public_sha256=item['public_sha256'])
    key = (entry['reference'], expected, entry['public_path'])
    if key not in keys:
        dep['files'].append(entry); keys.add(key)
dep['scope'] = 'Completed indexer studies and final P5c per-query consumer: public and original private evidence mappings'
(DST / 'PUBLIC_DEPENDENCIES.json').write_text(json.dumps(dep, indent=2, ensure_ascii=False) + '\n')
export['changes'].extend(changes)
export['p5c_final_export'] = dict(scope='Final per-query compact consumer research, real four-GPU correctness, independent admission/binding review; no model serving claims', files=origins, source_files_verified=len(selected), json_nonstring_invariance_checks=numeric_checks, private_originals_modified=False, gpu_uuid_policy='SHA256(lowercase original UUID) first24 hex, prefixed GPU-PSEUDONYM; preserve distinct devices without publishing raw inventory', previous_root_plan_preserved=str(historical.relative_to(DST)) if historical else None, root_book_pending=args.exclude_book)
export['private_originals_modified'] = False
(DST / 'PUBLIC_EXPORT.json').write_text(json.dumps(export, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(dict(exported=len(origins), mutations=len(changes), mappings=len(dep['files']))))
