# Native MANO IK Full Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. This implements the existing approved whole-paper design and official-code-priority authorization; do not repeat the design approval or review the closed synthetic IK mathematics.

**Goal:** Regenerate the available released MANO poses into UmeTrack finger labels through the pinned original optimizer, preserving the original default batch membership and every input label file.

**Architecture:** A separate read-only input component binds the released manifest, complete bilateral pose bytes, original source and six contiguous worker partitions. The native execution component will call the unchanged original optimizer and save isolated angles plus final raw optimization parameters; a fresh process will reconstruct the geometry without rerunning optimization. The full execution preserves six logical shards even when hardware schedules them serially.

**Tech Stack:** Existing NumPy1.26.4, Torch2.2.2, pinned EgoEMG f7cbd048, admitted manotorch/Chumpy/MANO assets. No dependency installation or shared runtime changes.

**Spec:** [Approved whole-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements6/7 and D12 in the full-paper status. The prior two-synthetic-pose CPU probe is closed evidence, not a substitute for this work.

## Global Constraints

- Preserve all trained models, running jobs, original manifests and generated labels. Whole-paper completion remains false until all required items are verified.
- The available source contains66,161,725 rows per hand. Each original pose is float32[48]; all zero poses remain included. There is no `valid.npy` in either actual source directory; do not replace it with marker-valid or angle-valid masks.
- Preserve original default six contiguous partitions, batch10096, stride1, right then left, maximum200 outer iterations and original stopping/restart rules. Batch membership matters to joint L-BFGS line search and stopping; a smaller batch is engineering only.
- Original function uses right MANO for both hands, flat_hand_mean=True, beta=0, first3 pose values zeroed, X flip/fixed alignment and optional optimized rotation/translation. Do not substitute the separate occlusion-renderer policy or reinterpret the left-hand convention.
- Preserve20-finger-angle output and two zero wrist inputs to FK. This work does not regenerate FE/RU wrists, recover the missing markers2mano model, or prove the legacy labels/PDF population equivalent.
- CPU input admission imports no Torch or optimizer, records all input/source generations and complete pose hashes, and grants no completed label-regeneration or GPU permit.
- No new GPU reservation until real default-batch qualification, source-bound execution and one final review of the complete new component. Existing GPU0/1/7 queues and foreign GPU3/6 remain protected.

## Review Focus

1. The original batch optimizer couples frames through reduction and line search: Task1 tests compare every shard/batch index against the actual original AnchorPoseDataset initializer, including remainder rows and partial final batches.
2. Zero poses and false marker flags must not disappear: Task1 tests retain every frame and reject an unexpected optional global valid file instead of silently changing selection.
3. Zeroing global orientation must not mutate the raw pose memmap: Task1 tests retain exact raw and canonical copies, read-only mappings and unchanged source bytes/RNG.
4. Same-sized source replacement, hardlink changes, identity edits and optional-file appearance after admission must invalidate use: Task1 guards/tests cover these conditions.
5. Small engineering data or resealed reference JSON must never claim full source: Task1 tests pin actual public reference bytes and complete manifest/pose identities; Task2/3 will bind real parameters, actual exited actors and independently reconstructed geometry.

## Task1: real pose admission and exact original batch membership

**Files:** create `egoemg_repro/native_ik_inputs.py`, `tests/test_native_ik_inputs.py`.

**Interfaces:** `admit_native_ik_inputs(data, geometry, upstream, *, engineering=False, batch_size=10096, chunk_rows=262144, reference=None) -> NativeIKInputs`; `.identity` returns an independent copy; `.check()` validates the admitted live binding; `.shards` gives the original six half-open row ranges; `.batches(hand, shard)` yields `(indices, raw_poses, canonical_poses)` in original order with independent float32 copies and only canonical first3 values zeroed.

- [x] Write new real-file tests for manifest/shape/size/dtype/finite checks, complete physical hashes, exact default partition/batch order, zeros retained, read-only originals, raw/canonical separation, no Torch import, no RNG/source mutation, optional global-mask appearance, changed identity/source/generations and false full-source claims.
- [x] Run only this new suite and observe actual missing-module RED. Implement the input component; compile the actual original AnchorPoseDataset initializer in tests rather than duplicating its indexing formula as the oracle.
- [x] Run the new suite to GREEN:40passed/0skip/0.60s, actualexit0. Two additional real-file admission-race regressions first observed2fail/0.15s, then final40GREEN; no old IK/model/math suite rerun.
- [ ] Run two separate complete real input admissions, preserving the two25.406GB payloads and all actor closures; publish scoped input evidence. Neither engineering nor full-input success is a completed IK run.

## Task2: unchanged native batch optimizer and raw geometric reconstruction

**Files:** create `scripts/paper/run_native_mano_ik.py`, `egoemg_repro/native_ik_results.py`, and focused tests.

**Interfaces:** native batch producer consumes registered Task1 inputs and immutable selected `(hand,shard,batch)` membership; raw verifier consumes final angles/raw-angle/global-rotation/translation parameters, input identity and unchanged original assets.

- [ ] Write rejection tests before implementation for arbitrary batch membership, missing selected raw state, wrong hand/source/asset/policy, incomplete outputs, premature completion and mutated inputs during native consumption.
- [ ] Invoke the original `optimize_batch` without changing its objective, L-BFGS updates, batching or stop conditions. Observe final optimizer parameters and actual step/closure counts while delegating each actual call to the original method; save raw and canonical input witnesses, angles, limits and unrounded geometry statistics.
- [ ] Execute real bilateral engineering input first, then complete actual batch10096 qualification on a permitted device. Independently reconstruct MANO/UmeTrack FK/alignment and full per-frame residuals from final parameters with0 new IK updates; preserve genuine failures and closed actors. Do not rerun the closed synthetic probe.
- [ ] Complete relevant new tests and one final new-component review with one material fix phase. No old model/math review or second review.

## Task3: complete released rows, isolated labels and publication

- [ ] Queue actual full six-shard, right-then-left default execution only after exact owned predecessor success/closure and physical resource admission. Never restart solely because a poll expires or a log is quiet.
- [ ] Preserve all66,161,725 rows per hand, partial batches, raw state and isolated float32[20] label outputs; do not edit the source manifest or replace released labels. Actual serial hardware scheduling is disclosed separately from original six-GPU defaults.
- [ ] Independently reconstruct every completed batch and confirm original source/input/asset generations, exact membership/angles/parameter bindings, final counts and actual owning-process closure. Report real wall time and IK updates separately from model training.
- [ ] Publish the implementation, aggregate errors/limits/zero-pose populations/legacy-label differences and verified evidence. Full-data label generation is distinct from author annotation accuracy, wrist generation and whole-paper completion.

## Execution ruling

The user's selected official-code route leaves one applicable method: the original optimizer and original logical batch partition. A smaller batch or deduplicated pose cache could change the optimum/stopping behavior, so neither is a full-data substitute. Asset paths and the known original limits-path typo are adapted at the caller boundary, as in the closed synthetic probe. The already-approved inline execution method applies; this is an implementation plan inside that scope.
