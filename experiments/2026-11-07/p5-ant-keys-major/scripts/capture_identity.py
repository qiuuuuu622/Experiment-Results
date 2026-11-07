import hashlib
import importlib.metadata
import json
import sys
import tarfile
from pathlib import Path

root = Path.cwd()
out = Path(sys.argv[1])
archive = out / 'frozen/source.tar.gz'
files = {}
with tarfile.open(archive) as tar:
    for member in tar:
        if member.isfile():
            sha = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            path = root / member.name
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha, member.name
            files[member.name] = sha
from benchmarks.kernels.dsv41_indexer.common import check_vllm
check_vllm()
from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init
_lazy_init()
dg = _import_deep_gemm()
dgroot = Path(dg.__file__).parent
vllmroot = Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
native = list(dgroot.rglob('*.py')) + list(dgroot.rglob('*.so')) + list(dgroot.rglob('*.cuh')) + list(dgroot.rglob('*.h'))
native += [vllmroot/'utils/deep_gemm.py', vllmroot/'_C_stable_libtorch.abi3.so']
identity = dict(source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                source_archive_matches_post_run=True, source_files=files,
                native_identity_capture='post-run; baseline wrappers independently checked at each process start',
                imported_deep_gemm=str(dg.__file__), imported_C=str(dg._C.__file__),
                native_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(native)) if p.is_file()],
                versions={p:importlib.metadata.version(p) for p in ['torch','triton','vllm','apache-tvm-ffi']},
                binary_build_commit='unknown; wheel name is not current HEAD provenance')
(out/'runtime-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
