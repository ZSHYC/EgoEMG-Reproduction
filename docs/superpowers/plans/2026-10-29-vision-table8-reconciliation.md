# Vision Table 8 reconciliation plan

**Goal:** Publish the seven actual visual baseline profiles alongside the PDF Table8 values, with exact parameter counts and explicit configuration differences.

**Spec:** Requirements2/5/7 and Table8 of the approved [full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md). This is a deterministic evidence analysis, not a new training component or recipe decision.

**Architecture:** Read retained construction schemas and immutable completed ResNet state tensors, then independently reconcile their parameter-name/shape sums with the original shared head formula. Preserve the original paper values, actual run policies and per-model evidence identities in separate fields.

**Tech stack:** Existing Python/JSON/CSV and CPU-only checkpoint reader; no new dependencies.

## Constraints

- Pinned upstream remainsf7cbd048b7092139852270491d3859cd472e0311. Original models, active jobs, checkpoints and prior reports remain unchanged.
- Do not construct models, run forwards, evaluate test data, create optimizers or train. Read no actively written checkpoint payload; completed ResNet best files and already-retained construction schemas suffice.
- Parameter count is an integer sum over actual parameter tensors, excluding persistent buffers. Retain backbone/head/component counts and distinguish total from trainable/frozen.
- A classifier's original ImageNet head is not part of the EgoEMG regressor. A schema count is not a new execution or speed benchmark.
- Keep paper LR/batch/epochs/dimension separate from actual legacy or derived official-code policies; no silent correction, run relabeling, common-population inference or additional completed science setting.
- Public artifacts contain configuration/aggregate counts and content digests, without private model assets, raw weights, subjects or sample rows.

## Steps

- [x] Read PDF p.22 Table8 and pinned original wrappers; collect saved actual configurations, initialization identities and named-tensor schemas. Verify completed ResNet checkpoint identities against their published reports and stable file attributes.
- [x] Independently verify seven counts using schema products and the four-tensor head formula `(d+1)*512+(512+1)*22`; reconcile complete tensor/buffer membership, WiLoR's unused native components and its total/trainable split. Preserve unexplained paper differences. Three ResNet AdamW moment inventories additionally match every parameter shape/count and initial1e-4 LR; no optimizer was constructed.
- [x] Publish `results/dataset-statistics/vision-table8.json`, `.csv`, `.md` and weight-free `vision-table8-inventory.json` with source digests and limitations; update Table8 status and run-status. Check CSV/JSON/Markdown agreement and source stability, commit and publish. No old model suites or closed model reviews are repeated. Independent public-inventory reduction checks all2,490tensor roles/shapes against seven rows, all three formats, head formula, AdamW moments, evidence hashes and local links; no new production component/model mathematics was introduced.

## Interpretation

This can close the seven-row configuration and parameter audit for the available implementations. Complete training, new official-code ResNet recipes, WiLoR continuation, PDF-equivalent data and full-paper completion remain separate requirements.
