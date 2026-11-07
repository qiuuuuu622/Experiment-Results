import copy,gzip,hashlib,io,json,re,tarfile
from pathlib import Path
SRC=Path('/workspace/vllm_dsv41_flash_opt')
DST=SRC.parent/'Experiment-Results'
BASE=SRC/'results/indexer/20261107-resume'
mappings={'p7':'p7-tp4-row-sharding','p8':'p8-final-dispatch','p7-review-p3':'p7-independent-review','p8-cpu-review':'p8-independent-review'}
oldmappings={'p3':'p3-full-source','p8-grouped-final-v1':'p8-grouped-final-v1','p8-prefill-final-v1':'p8-prefill-final-v1'}
export=json.loads((DST/'PUBLIC_EXPORT.json').read_text())
changes=[];origins=[];private_hashes={};numeric_checks=0
local_account=SRC.parts[2]
private_prefix=str(SRC);replacement='/workspace/vllm_dsv41_flash_opt'
def sha(b):return hashlib.sha256(b).hexdigest()
def sanitize_text(data):
 global numeric_checks
 try:s=data.decode('utf-8')
 except UnicodeDecodeError:return data,[]
 if '\x00' in s:return data,[]
 old=s; counts=[]
 for pattern in (private_prefix,json.dumps(private_prefix,ensure_ascii=True)[1:-1]):
  n=s.count(pattern)
  if n:s=s.replace(pattern,replacement);counts.append({'action':'replace personal local checkout prefix with /workspace/vllm_dsv41_flash_opt','occurrences':n})
 # Personal checkout paths outside this one repository, if any.
 s,n=re.subn('/'+'Users/'+re.escape(local_account)+r'(?=/)', '/workspace/local-user',s)
 if n:counts.append({'action':'replace remaining personal home prefix with /workspace/local-user','occurrences':n})
 s,n=re.subn(r'interactive-[A-Za-z0-9-]+','execution-host-redacted',s)
 if n:counts.append({'action':'redact unrelated scheduler hostname; SSH alias/device role retained','occurrences':n})
 s,n=re.subn(r'GPU-[0-9a-fA-F-]{30,}','GPU-UUID-REDACTED',s)
 if n:counts.append({'action':'redact physical GPU UUID; model and device index retained','occurrences':n})
 if s!=old:
  try:
   a,b=json.loads(old),json.loads(s)
   def shape(v):
    if isinstance(v,str):return '<string>'
    if isinstance(v,list):return [shape(x)for x in v]
    if isinstance(v,dict):return {k:shape(x)for k,x in v.items()}
    return v
   assert shape(a)==shape(b),'JSON non-string values changed'
   numeric_checks+=1
  except json.JSONDecodeError:pass
 return s.encode(),counts

def sanitize_archive(data,container):
 try:t=tarfile.open(fileobj=io.BytesIO(data),mode='r:*')
 except tarfile.TarError:return sanitize_text(data)
 entries=[];details=[]
 with t:
  for member in t:
   info=copy.copy(member)
   if info.name.startswith('/') or '..' in Path(info.name).parts:raise RuntimeError('Unsafe tar path '+container+'!'+info.name)
   payload=t.extractfile(member).read() if member.isfile() else None
   if '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc'):
    details.append({'member':info.name,'action':'omit interpreter cache only','original_sha256':sha(payload)if payload else None});continue
   if payload is not None:
    if info.name.endswith(('.tgz','.tar.gz','.tar')):new,edits=sanitize_archive(payload,container+'!'+info.name)
    else:new,edits=sanitize_text(payload)
    if re.search(r'(^|/)(LICENSE|COPYING|NOTICE)([./_-]|$)',info.name,re.I):assert new==payload,'License text must stay unchanged'
    if edits:details.append({'member':info.name,'original_sha256':sha(payload),'public_sha256':sha(new),'changes':edits})
    payload=new;info.size=len(payload)
   # Do not publish a personal workstation account from tar owner metadata.
   if info.uname==local_account:info.uname='';info.uid=0;details.append({'member':info.name,'action':'omit personal tar owner metadata'})
   entries.append((info,payload))
 if not details:return data,[]
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT)as out:
  for info,payload in entries:out.addfile(info,io.BytesIO(payload)if payload is not None else None)
 return gzip.compress(buf.getvalue(),mtime=0),details

