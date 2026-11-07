import json
import sys
from pathlib import Path
from benchmarks.kernels.dsv41_indexer.common import check_vllm
check_vllm()
import torch
from benchmarks.kernels.dsv41_indexer.benchmark_prefill_ant import SUITES
from benchmarks.kernels.dsv41_indexer.verify_prefill import make_case
from vllm_dsv41_indexer.prefill_keys_major_ant import AntPrefillIndexer
rows = []
for suite, shapes in SUITES.items():
    for shape in shapes:
        n, requests, keys, ratio, mode = shape
        inputs, candidates = make_case(n, requests, keys, ratio)
        if mode == 'short_prefix':
            inputs.lengths.copy_(torch.arange(1,n+1,device='cuda',dtype=torch.int32))
        if mode == 'reversed_visibility':
            inputs.lengths.copy_(inputs.lengths.flip(0))
        op = AntPrefillIndexer(inputs, candidate_blocks=candidates if mode=='consumer' else None,
                               source=mode=='source',selector='ant',native_gather=False)
        op.prepare()
        op.metadata()
        rows.append(dict(suite=suite,shape=shape,
            visible_index_key_counts=inputs.lengths.tolist(),
            actual_scored_key_counts=op.lengths.tolist(),
            actual_score_count=int(op.lengths.sum()),
            candidate_block_counts=op.state.counts.tolist() if op.compact else None,
            chunks=op.chunks,planned_grid=[p.grid_size for p in op.ant_plans.values()]))
Path(sys.argv[1]).write_text(json.dumps(dict(same_seed_and_frozen_driver=True,cases=rows),indent=2)+'\n')
