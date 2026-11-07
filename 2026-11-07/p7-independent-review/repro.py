import json
import torch
from vllm import _custom_ops
r,l=128,4096
scores=torch.zeros((r,l),device='cuda')
lengths=torch.full((r,1),l,device='cuda',dtype=torch.int32)
full=torch.empty((r,512),device='cuda',dtype=torch.int32)
parts=torch.empty_like(full)
workspace=torch.empty(1<<20,device='cuda',dtype=torch.uint8)
records=[]
for repeat in range(5):
    torch.ops._C.persistent_topk(scores,lengths,full,workspace,512,l)
    for start in range(0,r,32):
        torch.ops._C.cooperative_topk(scores[start:start+32],lengths[start:start+32],parts[start:start+32],workspace,512,l)
    a,b=full.sort(1).values,parts.sort(1).values
    records.append(dict(repeat=repeat,mismatching_rows=int((a!=b).any(1).sum()),persistent_stable_lowest512=bool((a==torch.arange(512,device='cuda')).all()),cooperative_stable_lowest512=bool((b==torch.arange(512,device='cuda')).all()),persistent_row0=a[0,:20].tolist(),cooperative_row0=b[0,:20].tolist()))
print(json.dumps(records,indent=2))
