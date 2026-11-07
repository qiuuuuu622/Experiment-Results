"""Isolated H20 Top-k experiment; native baseline and Ant cluster scheduling."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def identity():
    import torch
    import triton
    import importlib.metadata as metadata
    files = {}
    for base in (ROOT / "vllm_dsv41_indexer", ROOT / "benchmarks/kernels/dsv41_indexer"):
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    files["tune_topk.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    gpu = torch.cuda.get_device_properties(0)
    return dict(torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                tvm_ffi=metadata.version("apache-tvm-ffi"), gpu_name=gpu.name,
                sm_count=gpu.multi_processor_count, gpu_uuid=str(gpu.uuid),
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), sources=files,
                nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu", "--format=csv"], text=True))


VARIANTS = {
    "native": None,
    "ant_streaming": (8, 0, "fixed"),
    "ant_c8_p18_fixed": (8, 18, "fixed"),
    "ant_c8_p18_auto_h200": (8, 18, "auto"),
    "ant_c8_p9_fixed": (8, 9, "fixed"),
    "ant_c4_p36_fixed": (4, 36, "fixed"),
    "ant_c4_p18_fixed": (4, 18, "fixed"),
    "ant_c2_p78_fixed": (2, 78, "fixed"),
    "ant_c2_p39_fixed": (2, 39, "fixed"),
    "ant_c8_direct": (8, 512, "fixed"),
    "ant_c4_direct": (4, 512, "fixed"),
    "ant_c2_direct": (2, 512, "fixed"),
    "stream_hist2": (8, 0, "fixed", 2, 0),
    "stream_hist4": (8, 0, "fixed", 4, 0),
    "stream_warpagg": (8, 0, "fixed", 1, 1),
}
LOADED = {}


def load_variant(size, pool, groups=1, aggregate=0):
    key = (size, pool, groups, aggregate)
    if key in LOADED:
        return LOADED[key]
    from tvm_ffi.cpp import load_inline
    root = ROOT / "vllm_dsv41_indexer"
    source = """#include <tvm/ffi/function.h>
#include <deepseek_v4/topk_v2.cuh>
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk, sglang::TopKKernel<true>::transform_paged);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, sglang::TopKKernel<true>::plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(floor, sglang::TopKKernel<true>::get_static_cluster_floor);
"""
    flags = ["-std=c++20", "-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a",
             "-DSGL_CUDA_ARCH=900", "-DSGL_TOPK_DET_TIES=1", f"-DSGL_TOPK_V2_MAX_C8_OCC2={pool}",
             "-DSGL_TOPK_V2_MAX_C16_OCC1=0", f"-DDSV41_EXPERIMENT_CLUSTER_SIZE={size}",
             f"-DDSV41_STREAM_HIST_GROUPS={groups}", f"-DDSV41_STREAM_WARP_AGG={aggregate}"]
    digest = hashlib.sha256((source + repr(flags)).encode())
    for p in sorted((root / "third_party").rglob("*")):
        if p.is_file(): digest.update(p.read_bytes())
    result = load_inline("dsv41_tune_" + digest.hexdigest()[:16], cuda_sources=source,
                        extra_cuda_cflags=flags,
                        extra_include_paths=[str(root / "third_party/ant/csrc"), str(root / "third_party/ant/include")])
    LOADED[key] = result
    return result


class Selector:
    def __init__(self, name, scores, lengths, k):
        import torch
        self.name, self.scores, self.lengths, self.k = name, scores, lengths, k
        self.output = torch.empty((scores.shape[0], k), dtype=torch.int32, device="cuda")
        self.metadata = torch.zeros((scores.shape[0] + 1, 2), dtype=torch.int32, device="cuda")
        self.needs_plan = False
        if name == "native":
            from benchmarks.kernels.dsv41_indexer.inputs import VllmSelector
            self.native = VllmSelector(scores.device, scores.shape[0]) if k == 512 else None
            return
        size, pool, mode, *options = VARIANTS[name]
        self.kernel = load_variant(size, pool, *options)
        self.floor = self.kernel.floor(scores.shape[0])
        self.threshold = -1 if mode == "auto" else self.floor
        self.needs_plan = pool > 0 and pool < scores.shape[0] <= 512 and scores.shape[1] > self.floor
        self.prepare()

    def prepare(self):
        if self.needs_plan: self.kernel.plan(self.lengths, self.metadata, self.threshold)

    def execute(self):
        import torch
        if self.name == "native":
            if self.native is None:
                return torch.topk(self.scores, self.k, dim=-1).indices
            self.native(self.scores, self.lengths[:, None], self.output, self.scores.shape[1])
        else:
            self.kernel.topk(self.scores, self.lengths, None, self.output, 1, self.metadata, None)
        return self.output

    def forward(self):
        self.prepare()
        return self.execute()


def measure(fn, graph):
    from benchmarks.kernels.dsv41_indexer.common import measure as base
    value = base(fn, repeats=30, inner=10, graph=graph)
    if value["coefficient_of_variation"] > 0.1:
        value["extra_rounds"] = [base(fn, repeats=30, inner=10, graph=graph) for _ in range(3)]
    return value


def oracle(scores, lengths, k):
    import torch
    masked = scores.masked_fill(torch.arange(scores.shape[1], device="cuda")[None, :] >= lengths[:, None], -float("inf"))
    idx = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :k].int()
    return idx.masked_fill(torch.arange(k, device="cuda")[None, :] >= lengths[:, None], -1)


def assert_selected(out, expected):
    import torch
    if not torch.equal(out.sort(dim=1).values, expected.sort(dim=1).values):
        mismatch = (out.sort(dim=1).values != expected.sort(dim=1).values).sum().item()
        raise AssertionError(f"stable selection mismatch: {mismatch}")


def benchmark(args, report):
    import torch
    from benchmarks.kernels.dsv41_indexer.verify_topk import make_scores
    for rows in args.rows:
        for width in args.widths:
            if args.distribution == "indexer":
                from benchmarks.kernels.dsv41_indexer.inputs import make_inputs
                from vllm_dsv41_indexer.contract import IndexerShape
                from vllm_dsv41_indexer.module import IndexerInputs, NativeFullIndexer
                fixture = dict(rows=rows, compressed_length=width, rows_per_request=rows,
                               page_size=64, candidate_blocks=2048)
                q, _, w, cache, table, req, lengths, _ = make_inputs(fixture)
                inputs = IndexerInputs(q,w,cache,table,req,lengths,IndexerShape(rows,width))
                producer = NativeFullIndexer(inputs,next_n=1)
                scores = producer.score().clone()
                scores.masked_fill_(torch.arange(width,device="cuda")[None,:]>=lengths[:,None],float("nan"))
                del producer, inputs, q, w, cache, table, req
            else:
                scores = make_scores(rows, width, "random", seed=6116)
                lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
            expected = oracle(scores, lengths, 512)
            before = hashlib.sha256(scores.cpu().numpy().tobytes()).hexdigest()
            selectors = {name: Selector(name, scores, lengths, 512) for name in args.variants}
            case = dict(rows=rows, width=width, k=512, distribution=args.distribution,
                        visible_lengths=lengths.tolist(), score_sha256=before, correctness={}, measurements=[])
            for name, sel in selectors.items():
                out = sel.forward()
                if name == "native":
                    masked = scores.masked_fill(torch.arange(width,device="cuda")[None,:]>=lengths[:,None],-float("inf"))
                    boundary = torch.sort(masked, descending=True).values[:, 511]
                    assert (scores.gather(1, out.long()) >= boundary[:, None]).all().item()
                    exact = torch.equal(out.sort(1).values, expected.sort(1).values)
                else:
                    assert_selected(out, expected); exact = True
                case["correctness"][name] = dict(stable_exact=exact, metadata=sel.metadata[0].tolist(), needs_plan=sel.needs_plan)
            for order in ("forward", "reverse"):
                names = list(selectors) if order == "forward" else list(reversed(selectors))
                for mode in args.execution:
                    for name in names:
                        sel = selectors[name]
                        value = measure(sel.forward, mode == "graph")
                        case["measurements"].append(dict(order=order, execution=mode, variant=name, phase="total", metric=value))
                        print(json.dumps(dict(rows=rows,width=width,order=order,execution=mode,variant=name,p50_us=value["p50_us"])), flush=True)
            assert hashlib.sha256(scores.cpu().numpy().tobytes()).hexdigest() == before
            case["same_input_bytes_before_after"] = True
            report["cases"].append(case); save(args.output, report)
            if args.trace and rows == max(args.rows) and width == max(args.widths):
                for name in [n for n in args.variants if n in ("native", "ant_streaming", "ant_c8_p18_fixed", "ant_c4_p36_fixed", "ant_c2_p39_fixed")]:
                    sel = selectors[name]
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(5):
                            with torch.profiler.record_function(name): sel.forward()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(Path(args.output).parent / f"trace-{rows}-{width}-{name}.json"))
            del selectors, scores, lengths, expected


def verify(args, report):
    import torch
    from benchmarks.kernels.dsv41_indexer.verify_topk import make_scores
    for rows in args.rows:
        width = max(args.widths)
        for k in (512, 2048):
            for distribution in ("random", "zero", "equal_nonzero", "cutoff_ties", "tie_bin_overflow"):
                scores = make_scores(rows, width, distribution, seed=6116)
                saved_scores = scores.clone()
                full = torch.full((rows,), width, dtype=torch.int32, device="cuda")
                dynamic = torch.tensor([min(width, [0,1,7,511,513,2047,4099,width][i%8]) for i in range(rows)],dtype=torch.int32,device="cuda")
                lengths = full.clone()
                reference_full = oracle(scores, full, k)
                reference_dynamic = oracle(scores, dynamic, k)
                for name in args.variants:
                    if name == "native": continue
                    sel = Selector(name, scores, lengths, k)
                    assert_selected(sel.forward(), reference_full)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): sel.forward()
                    for stage, values, reference in [("mixed",dynamic,reference_dynamic),("zero",torch.zeros_like(full),None),("restore",full,reference_full)]:
                        lengths.copy_(values)
                        scores.copy_(saved_scores)
                        scores.masked_fill_(torch.arange(width,device="cuda")[None,:]>=lengths[:,None],float("nan"))
                        sel.output.fill_(-999)
                        graph.replay()
                        target = torch.full_like(sel.output,-1) if reference is None else reference
                        assert_selected(sel.output,target)
                    graph.reset()
                    report["checks"].append(dict(rows=rows,width=width,k=k,distribution=distribution,variant=name,stages=["eager_full","graph_mixed","graph_zero","graph_restore"],status="passed"))
                    save(args.output, report)
                print(json.dumps(dict(verified_rows=rows,width=width,k=k,distribution=distribution)),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase", choices=["compile","benchmark","verify"], required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[64,128])
    p.add_argument("--widths", type=int, nargs="+", default=[65536,131072,262144])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--execution", nargs="+", default=["graph"])
    p.add_argument("--output", required=True)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--distribution", choices=["random","indexer"], default="random")
    args=p.parse_args()
    from benchmarks.kernels.dsv41_indexer.common import check_vllm
    check_vllm()
    import importlib
    importlib.import_module("vllm._custom_ops")
    report=dict(status="running",args=vars(args),environment_before=identity(),cases=[],checks=[],cache="warm repeated calls; no L2 flush",scope="selector-only, single GPU per process; no TP4 claim",compile=[])
    save(args.output,report)
    try:
        if args.phase=="compile":
            for name in args.variants:
                if name=="native":continue
                start=time.perf_counter();size,pool,_,*options=VARIANTS[name];load_variant(size,pool,*options)
                item=dict(variant=name,seconds=time.perf_counter()-start)
                report["compile"].append(item);save(args.output,report);print(json.dumps(item),flush=True)
        elif args.phase=="benchmark":benchmark(args,report)
        else:verify(args,report)
        report["environment_after"]=identity()
        assert report["environment_before"]["sources"]==report["environment_after"]["sources"]
        report["status"]="completed"
    except BaseException as exc:
        report.update(status="failed",error=repr(exc));raise
    finally:save(args.output,report)


if __name__=="__main__":main()
