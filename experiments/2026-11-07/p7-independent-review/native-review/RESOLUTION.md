# Independent re-review: native P7 findings resolved

The final `native_stable` code closes the three original findings:

- `IndexerRows.forward()` invokes `source.prepare()` on every native step before
  score. `RowExchange.forward()` packs inputs and calls this path; capturing the
  complete variant therefore captures metadata/page-table preparation too.
- Both full and chain use `NativeSourceIndexer`'s deterministic selector. Full
  calls only `score()` and `select()`; chain calls full `forward()`, retaining
  both token top512 and block top2048 before publishing to four consumers.
- Verifier and benchmark accept `--full-backend native_stable` and require an
  explicit isolated DeepGEMM package. The final verifier checks lowest logical
  positions for zero-Q, two additional graph replays, and source candidate sets.

Raw evidence independently read back from h20-4:

- `verify-native-dynamic-final.json`: passed, actual world4, 320 checks per rank.
  Every record has identical logical output hashes on four ranks and four
  distinct physical-page-table hashes. The reviewed verifier's SHA256 exactly
  matches the recorded source hash.
- `native-stable-full-chain.json`: complete,144 timing records, Eager/Graph and
  forward/reverse, corrected native_stable backend. Runtime hashes match the
  correctness run. The isolated native include hash matches P3's validated patch.
- `audit.json` records these independent checks. The previous static native
  results remain separate and must not be relabeled as this dynamic version.

The original findings are closed for these measured cases. One explicit coverage
limit remains: the current4-step verifier does not simultaneously set *all
 global rows* to zero length and permute physical pages in one step. It does
cover mixed-zero lengths (including rank-local all-zero work), restoration,
request changes and later physical-page permutation. Do not claim the untested
combined case was independently measured here. No residual code defect was
identified from this limit.

No P7 owner file was modified and no further GPU run was performed during this
re-review.
