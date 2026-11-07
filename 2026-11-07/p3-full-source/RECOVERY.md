# P3 recovered evidence, 2026-11-07

The `zero-prefetch-v1/` and `verify-v1/` batches were run on SSH host `h20-4`,
physical GPU 2, before the task was reassigned to `h20-dsk` GPU 3. Their commands,
source hashes, native headers, JSON and logs were copied back after completion.
They are not results from the new host.

- `zero-prefetch-v1/original.json`: installed native reproduces errors 0.05560249
  and 4.43944454 when one/two empty rows lie between non-empty scheduled rows.
- `zero-prefetch-v1/patched.json`: isolated producer-prefetch patch brings every
  tested valid score within 9.54e-7 of the FP8 Triton implementation. Global
  site-packages remained unchanged.
- `verify-v1/result.json`: ten ratio-1/2 boundary cases passed. Each includes
  source token/block selection, four distinct consumer Q/weights, three graph
  replays, request reorder, physical page permutation, and strict logical-index
  checks. Same-score stable selections are exact.
- The real 1 GiB shared score arena passed with 1025 planned rows, chunks
  1024+1, and changed visible lengths across two graph replays. This is a real
  allocation/chunking test: 1024 rows initially have 513 valid keys and the last
  row has 262144. It is not a 1025-by-256K full-work performance measurement.

The older 2026-11-06 `budget-v1/` also remains intact. The new `full-final-v1/`
batch is in progress on `h20-dsk` GPU 3 and must not be treated as complete until
all its result files report completion.
