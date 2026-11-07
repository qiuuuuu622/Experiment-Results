"""Stage immutable P5c owner/TP4 artifacts only; no book/dispatch/commit/push."""
from pathlib import Path
code = Path(__file__).with_name('publication_sanitizer.py').read_text()
code = code.replace('return {k:shape(x)for k,x in v.items()}', 'return [(shape(k),shape(x)) for k,x in v.items()]')
code = code.replace("'GPU-UUID-REDACTED',s)", "lambda m: 'GPU-PSEUDONYM-' + hashlib.sha256(m.group().lower().encode()).hexdigest()[:24],s)")
code = code.replace('redact physical GPU UUID; model and device index retained', 'deterministic GPU UUID pseudonym; preserve equality and distinct four-card identities')
exec(code)
for stage in ('p5c-perquery', 'p5-perquery-tp4'):
    for f in sorted((BASE / stage).rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts:
            continue
        raw = f.read_bytes(); private_hashes[f] = sha(raw)
        relative = Path('2026-11-07') / f.relative_to(BASE)
        if f.name in ('SHA256SUMS.json', 'SHA256SUMS', 'FINAL-SHA256.json'):
            relative = relative.with_name('ORIGINAL_PRIVATE_' + f.name)
        data, edits = sanitize_archive(raw, str(relative)) if f.name.endswith(('.tgz','.tar.gz','.tar')) else sanitize_text(raw)
        target = DST / relative
        assert not target.exists() or target.read_bytes() == data, target
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        origins.append(dict(source=replacement + '/' + str(f.relative_to(SRC)), public_path=str(relative), original_sha256=sha(raw), public_sha256=sha(data), bytes=len(data), changes=edits))
for f, expected in private_hashes.items():
    assert sha(f.read_bytes()) == expected
result = dict(status='staged_not_published', files=origins, private_originals_modified=False,
              excluded=['active dispatch review', 'root book pending approval'],
              note='Final export will include these exact bytes and authoritative public mappings/checksum; no commit or push yet')
(BASE / 'p5c-publication/STAGED_EXPORT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(dict(staged_files=len(origins), bytes=sum(x['bytes'] for x in origins))))
