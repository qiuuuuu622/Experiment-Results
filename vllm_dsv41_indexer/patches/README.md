# Isolated DeepGEMM empty-query experiment

`deepgemm-sm90-empty-query-prefetch.patch` is an experimental patch against the
installed SM90 FP8 paged-MQA include snapshot recorded by P3. Its SHA256 and the
complete native include archive are in the experiment artifacts. It is not a
claim that the installed native binary was built from the vLLM baseline HEAD.

The installed scheduler skips empty requests. Its producer previously prefetched
`q_idx + 1`, which could load an empty request's Q/weights for the next non-empty
request. The patch searches for the next scheduled non-empty request before
prefetching. KV layout, FP8 arithmetic and score selection are unchanged.

All experiments copy the DeepGEMM package into an isolated run directory, apply
the patch there, and initialize that process with `_C.init(copy_path)` before the
first scoring call. The global installed package is never overwritten. P3
benchmark `--deep-gemm-package` records the selected copy explicitly; omitting
that option selects the unchanged installed package.

This patch is not automatically applied by the indexer. A caller using the
unchanged installed baseline with mixed empty and non-empty requests can still
hit the known failure; fixed native runs and unpatched baselines are reported
separately. See `repro_zero_prefetch.py` and the frozen P3 results.
