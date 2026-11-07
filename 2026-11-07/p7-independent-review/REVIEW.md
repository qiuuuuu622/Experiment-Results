# P7 native backend independent review

Reviewed current local files and the matching h20-4 working copy. No P7 owner
files were modified. Existing positive-length static performance measurements
are not invalidated; the findings prevent calling those measurements a validated
dynamic/zero-query implementation.

1. **Dynamic metadata is stale (blocking).** `IndexerRows.forward()` calls
   `self.source.forward()` without `prepare()`. `NativeFullIndexer.prepare()`
   gathers a *copy* of the request-local page table and builds scheduler metadata;
   constructor-only preparation does not follow `RowExchange.prepare()` updates
   to requests/lengths, or changes to the original physical page table. Put native
   metadata preparation inside the dynamic forward/Graph path and include its
   cost, or explicitly restrict a separate preplanned mode to immutable metadata.
2. **The native backend was not exercised by the boundary verifier (blocking).**
   `verify_tp4.py` has no full-backend/isolated-DeepGEMM option and calls
   `make_case(**case)` with its portable default. Its four-step mixed-zero,
   zero-Q, request-reorder and physical-page-permutation checks cannot establish
   native backend correctness. Add the same checks for the final native adapter.
3. **NativeFull selection violates the stable-tie contract (blocking).**
   Replicated rows >64 select `persistent_topk`; local rows <=64 select
   `cooperative_topk`. Both use atomic collection of ties. An independent
   fixed-logit GPU3 repro compares 128 rows ×4096 zero scores against four groups
   of32 rows: 127–128 row sets differ per repetition, and neither selects the
   lowest512 logical positions. Use DeepGEMM score + Ant deterministic selector
   for the optimized full path; keep the original native selector as a labeled
   baseline. Sorting the output afterward cannot repair a different selected set.
4. **Isolated DeepGEMM initialization is correctly placed in the benchmark.**
   `benchmark_tp4.py --patched-deep-gemm` calls `_lazy_init()` then `_C.init(path)`
   before factories/score calls and records header/binary hashes. That patch fixes
   the producer's empty-request Q prefetch; it does not fix findings1–3. The
   verifier needs equivalent explicit configuration. The constructor's current
   positive-length guard also needs a deliberate policy for the repaired backend.

After changing preparation/selector costs, rerun representative full/chain
forward/reverse timing. Retain prior output as static/preplanned evidence rather
than silently relabeling it. Compact/portable validations are unaffected.

`repro.py`, `run.sh`, `source.sha256` and `result.json` record the independent
selector diagnosis on h20-dsk physical GPU3; source and native binary identity
were saved before the archived run. No distributed process group or model was
started, and GPU3 was released immediately afterward.
