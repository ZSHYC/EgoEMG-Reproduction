# Full-frame legacy angle validity for Tables4/6

The approved full-paper design requires actual full-frame per-hand/per-angle mask coverage. Existing shared class/activity/split counts and frozen-center coverage are retained; this work joins the released angle/mask fields over every supplied row. It does not regenerate labels, alter model inputs or establish PDF/current-release equivalence.

## Interfaces and decisions

- New `egoemg_repro.semantic_frames.collect_frame_validity(root, *, upstream, transfer_receipt_sha256, chunk_rows=65536)` returns a public aggregate projection and a private receipt. CLI writes one new private JSON without overwriting.
- Reuse existing semantic definitions, strict field specifications, ordinary-file opening and small-source snapshot helpers. Read the12 registered fields once in bounded chunks, compute complete per-field SHA during that same pass, require the original transfer ledger hashes and unchanged nine file attributes. Validate canonical boolean bytes and class/split domains before counting.
- Count shared synchronized rows once; count left/right hand frames separately. Finger masks are generated validity AND finite angle; wrist masks are independent wrist validity AND finite degree-to-radian conversion, before any cleanup. This explicitly describes the existing legacy reproduction mask; official ordinary-path wrist masking differs and stays documented.
- Export overall/hand/split/class/activity aggregates, 22 finite/valid counts and all20/all2/all22/any valid-hand-frame counts. Preserve all60 definitions, raw50 and unassigned rows. No participants/episode IDs, paths, raw labels, model errors or train/val window identities enter the public projection.
- Inputs share the old semantic schema and transfer ledger. Pre-flight: the server owns complete12fields; local only metadata, so run the actual read-only scan on the server. No extra payload copies/downloads. Source and all live model jobs remain unchanged.

## Tasks

## Task 1: write hand-counted8row RED tests for independent hand/wrist masks, nonfinite-before-cleanup, chunk invariance/privacy, full-payload tampering and noncanonical bool/class/split rejection. Implement minimal bounded scanner; run the new tests plus existing semantic definition/coverage/center regression tests. Expected: RED missing implementation, then all relevant checks pass.
## Task 2: freeze a small isolated server source capsule and run the complete66,161,725-row CPU scan under an actual exit recorder. Expected: each12field complete hash matches original transfer, all input nine attributes unchanged, no Torch/CUDA/model work and no source mutation.
## Task 3: a fresh process independently recounts the full arrays with a distinct vectorized grouping formulation, and compares every integer aggregate plus old shared157record audit and old frozen-center point checks. Expected: exact integers and complete source hashes, all actual owners exit. No duplicated model training or broader old project suite/review.
## Task 4: one fresh review of this new statistical unit only, accept/fix material findings with targeted evidence, export public JSON/Markdown and update Table4/6 scope/status; commit/push and independently verify GitHub ref. Expected: actual evidence and source bindings match; completed model setting count stays10.

Source policy remains official `f7cbd048`; the provided legacy labels explicitly use axis-angle projection and are not relabeled as the PDF landmark IK output. Nominal wrist limits, units calibration and true annotation error are not inferred from coverage.

## Execution evidence

- [x] Task1: initial8actualRED; then91passed/1Torch-optional skip.
- [x] Task2: corrected8member producer capsule, all66,161,725rows/12.306GB in46.316s/exit0; initial missing-data.py capsule exit1 before opening large labels retained.
- [x] Task3: independent column-bincount full recount51.171s/exit0, all12hashes/allintegers/157oldrecords/33,508oldcenter digests exact, all6owners closed.
- [x] Task4 review/fix/export: one review0C/1I/0M; optimized verifier protection2RED→GREEN, final93passed/1optionalskip, remaining counter AST exact, publicJSON/Markdown bound to actual evidence. Publication/ref check remains an explicit final gate until logged.

Actual execution source and published replay source differ only by explicit rejection of optimized Python before input access. The original normal-interpreter run and immutable capsule remain intact; no scientific replay is retagged or repeated for this nonnumeric safety change.
