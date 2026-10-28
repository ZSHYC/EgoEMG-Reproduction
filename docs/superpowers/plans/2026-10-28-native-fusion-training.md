# Native ViT-S fusion training implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Steps use checkbox (`- [ ]`) syntax for tracking. Perform one final fresh-context implementation review after Tasks 1–3, followed by one author fix pass; no per-task review or re-review.

**Goal:** Complete the official ViT-S + EMGFormer-S fusion experiment, including verified branch handoff, full training, independent evaluation and a public report, within the already approved full-paper reproduction.

**Architecture:** Keep the actual upstream fusion factory, branch loaders, joint reader, losses, optimizer and Lightning loops. A separate handoff verifies the completed vision last checkpoint and the completed compatible EMG validation-best export; a dedicated fit adapter supplies complete coverage, recovery and source-bound evidence. Evaluation uses the fusion validation best and preserves native metrics alongside independently recomputed angle statistics.

**Tech Stack:** Python, PyTorch 2.2.2, Lightning 2.5.0.post0, Hydra, NumPy, timm, existing bounded memmap/LMDB adapters.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), especially requirements 2, 4 and 7. This is implementation within existing authorization, not a new scientific scope or a replacement for other remaining paper items.

## Global constraints

- Fix upstream `f7cbd048b7092139852270491d3859cd472e0311`; executed official code resolves ambiguities with the PDF.
- Preserve all old model results and running immutable capsules. Do not modify vision or EMG initializer producer sources to make a handoff pass.
- Use `fusion/fusion_vits_s_center_16ch_wl7790`: 16 zero-filled slots, original eight channel positions, 7,790 input points, stride400, repeat2, all-role batch200, bf16, clip1, AdamW1e-4/wd0.01, cosine150/eta_min1e-5, 150 epochs without early stopping, top3 plus last, validation `val_mae/min`, effective fingertip weight0.02.
- Transfer vision **last**, compatible EMG **validation best**, preserving native exact-name/shape partial EMG loading (105 matched / 5 skipped), tiny normal residual weights and all optimizer groups including temporal attention in `_unassigned`.
- Full fusion population is 395,276/3,152/8,612 consumed train/val/test hand samples; test has 4,306 unique hand samples. Do not count repeat2 as independent observations.
- Require complete content admission and independent validation. Missing merged EgoEMG/ShowEE/Incre sources, old labels, whole-episode filtering, public normalization, single rank and workers0 remain explicit limits, not paper equivalence.
- Full is never inferred from an engineering receipt. CPU engineering is two epochs, batch2, four training samples; CUDA engineering is two epochs, native batch200, 400 training samples. Both still validate the complete admitted validation population.
- Registered full training has no epoch/wall-clock truncation or automatic retries. Preserve terminal evidence; inspect actual PID/start before treating any owner as stopped.
- No test evaluation before training terminates and validation selects the final best. Do not use branch handoff or engineering validation as a new scientific model result.

## Review focus

1. Completed old ViT-S fit has an older frozen reader: verify its exact known source bytes and unchanged common dependencies, never weaken the producer reader or import arbitrary caller code.
2. Source weights may be genuine but trained on future fusion validation/test input: independently reconstruct complete jitter footprints and reject any overlap or mismatched payload provenance.
3. Native component grouping leaves temporal attention unassigned: require every trainable parameter exactly once at the original rate, reject missing/duplicate/foreign tensors and altered scheduler settings.
4. A process may die between journal, checkpoint and result publication: preserve old bytes, reconcile against the last complete native checkpoint, validate all six restored states and the complete frozen validation set; a completed budget must not receive an extra update.
5. Missing RGB keys, partial final batches and repeated test consumption must retain original behavior and explicit denominators; report generation must reject engineering, wrong-best or unpaired evidence.

---

### Task 1: Verified initializer handoff and native optimizer identity

**Files:**
- Create `egoemg_repro/fusion_initializer_handoff.py`.
- Modify `egoemg_repro/fusion_literal_runtime.py` only to add optimizer verification.
- Create `tests/test_fusion_initializer_handoff.py`; extend `tests/test_fusion_literal_runtime.py`.
- Create `docs/fusion-native-training.md` and engineering/read-only evidence in `results/`.

**Interfaces:**
- `admit_fusion_initializers(cohort, *, vision_fit, vision_source_root, emg_fit, emg_export) -> FusionInitializers` is full-only. It verifies actual completed producer fits, chooses vision `checkpoints/last.ckpt`, reopens the source-fit-bound EMG best export, compares producer cohort payloads and reconstructed partitions, and rejects overlapping full jitter footprints.
- `engineering_fusion_initializers(cohort, *, vision_checkpoint, vision_sha256, emg_checkpoint, emg_sha256) -> FusionInitializers` explicitly admits trusted untrained fixtures for bounded engineering only.
- `FusionInitializers.check()` binds all producer files/sources, the admitted cohort, and its own immutable content seal. `.runtime_kwargs` provides the four branch arguments to `build_fusion_runtime`; `.identity` carries safe lineage, actual selection and overlap counts without private paths.
- `fusion_optimizer_identity(prepared, optimizer, scheduler) -> dict` verifies original groups/rates/defaults, exact trainable coverage, cosine150/1e-5 and scheduler ownership, returns a sealed name/schema identity without changing the optimizer.

- [x] Step 1: Write tests for genuine engineering handoff and scope refusal; actual overlapping interval unions including jitter-only overlap; missing/incomplete producer fit; forged export; altered pinned historical reader; mismatched metadata/partition/payload; missing, duplicated and foreign optimizer parameters; changed group name/rate/defaults and scheduler. Do not fabricate a full-trained positive.
- [x] Step 2: Run the new tests and observe missing API/behavior failures. Record the command and actual RED result in the ledger.
- [x] Step 3: Implement the above interfaces using existing producer readers, `_StatGuard`, `_admit`, `LegacyStore`, partition builders and union arithmetic. Support the known published old ViT-S reader by its exact SHA256 `e44dc947111bec3aa858fb5fc36493650cda24f2093915bcd133a2d090d070da`; a current-reader path must satisfy the current producer's own source checks. Never mutate current vision module globals or saved reports.
- [x] Step 4: Run new and related native tests, then bare light suite. Independently run the actual old ViT-S producer half on CPU with no CUDA/model construction; full two-branch admission remains pending actual EMG150 completion.
- [x] Step 5: Commit source, tests, documentation and honest partial evidence; record task completion and unresolved actual full-handoff dependency.

### Task 2: Native complete fusion fit and interruption recovery

**Files:**
- Create `egoemg_repro/fusion_literal_train.py`, `tests/test_fusion_literal_train.py`.
- Extend `docs/fusion-native-training.md`.

**Interfaces:**
- Consumes Task1 `FusionInitializers`, existing `FusionCohort` and `native_fusion_loaders`.
- `fusion_fit_policy(upstream, config, *, scope='full') -> dict` exposes original values and explicit bounded engineering differences.
- `fit_fusion(cohort, initializers, output, *, scope='full', resume=False, stop_after_epoch=None, cohort_receipt=None) -> dict` owns original callbacks/logger/loops and all six restoration classes, accepts full only with a registered input receipt and a full verified handoff.
- `read_fusion_fit(output, *, require_full=True) -> dict` independently validates durable native files, all epochs/populations, original selection and actual 150-epoch termination.

- [x] Step 1: Write tests for original policy, real CPU two-epoch native updates/complete validation, independent process restoration and exact frozen validation, changed sources/lineage, engineering upgrade rejection, partial epoch, uncommitted journal tail, stale incomplete result and completed-checkpoint finalization without any extra update.
- [x] Step 2: Run and record expected RED before implementation.
- [x] Step 3: Implement using corrected initializer fit/storage primitives as established patterns; keep fusion-specific coverage, lineage, native optimizer groups and original forward/loss. Save original callback best/last and refuse last fallback. Preserve seed-before-factory as an explicit reproducibility correction, not the original train entry order.
- [x] Step 4: Run related native fit tests and bare light suite; generate independent CPU engineering receipts in fresh processes with no scientific score.
- [x] Step 5: Commit with recorded commands/results and public execution limits.

### Task 3: Frozen selected-best evaluation and report

**Files:**
- Create `egoemg_repro/fusion_literal_eval.py`, `tests/test_fusion_literal_eval.py`.
- Extend `docs/fusion-native-training.md` and report entry points.

**Interfaces:**
- Consumes `read_fusion_fit`, the original admitted cohort and self-contained full native best state.
- `evaluate_fusion(cohort, fit, output, *, role, require_full=True) -> dict` performs native batch200/bf16 evaluation plus safe paired predictions/statistics, honoring complete coverage and repeat2.
- `verify_fusion_evaluation(cohort, fit, reference, output, *, require_full=True) -> dict` performs a genuinely new prediction pass and independently compares all counts/statistics under a fixed predeclared numerical rule.
- `publish_fusion_report(fit, evaluation, replay, output, *, require_full=True) -> dict` writes safe metrics, curves and completion only after complete source/selection/coverage verification.

- [x] Step 1: Write tests for real engineering selected-best evaluation/replay, all versus crop-present and unique versus repeated consumption, typed masks/22 angles/partial batches, unknown/duplicated samples, wrong-best and incomplete-fit refusal, test-before-freeze rejection, unpaired or engineering report rejection.
- [x] Step 2: Observe RED, then implement original test/validation behavior plus independent float64 error sums/counts and hand/split/user/joint statistics. Retain missing-crop population and original generated mask. Keep native batch-weighted radians separate from pooled degrees.
- [ ] Step 3: Run evaluation/fit/runtime/data regressions and bare light suite, then independent fresh-process CPU engineering replay. Commit honest evidence.

### Task 4: Actual GPU qualification, one review and full experiment

**Files:**
- Task-owned stage/worker/controller/observer scripts and frozen manifests in ignored scratch.
- `results/native-fusion-*-20261028.json`, final `results/original-fusion/vit-small-emg-small/` and current status documentation.

- [ ] Step 1: Revalidate the actual complete cohort under final affected sources in independent processes. Check any proposed normalization representation fix with a real changed-input regression before changing the old reader; preserve old receipts and never rewrite them.
- [ ] Step 2: On a newly observed free GPU run original batch200/bf16 bounded train, independent native restore, complete validation and evaluation/replay qualification. No smaller batch, reduced model or loosened tolerance to force admission.
- [ ] Step 3: Request one fresh-context review of the entire new implementation; grade by actual scientific/user effect. Fix accepted Critical/Important findings once with RED→GREEN and a green suite; ledger deferred minors and every ruling. No second review.
- [ ] Step 4: When compatible EMG150 has actually completed and independent best-export acceptance has passed, perform full two-branch handoff and freeze its exact lineage, input receipt, final source and GPU qualification. Launch original full150 once with resource guards.
- [ ] Step 5: Observe the same actual process to termination. Independently accept full fit; run final validation freeze, native test, new prediction replay, report/curves and safe publication. Do not mark this task or the whole paper complete while any required stage is missing.
- [ ] Step 6: Update actual model count and paper matrix only after scientific completion. Preserve all old results and wider pending paper scope; archive only this plan's workspace after its final required work is complete.

## Self-review and execution decision

The plan implements the ViT-S fusion portion of full-paper requirements 2/4/7; other models, annotation and figures retain their own outstanding entries. Each Review Focus item belongs to Task1, Task2 or Task3 tests; Task4 supplies actual data/GPU/full-budget evidence rather than synthetic full-trained positives. Existing user authorization covers this implementation, server runs and publication to the reproduction branch. Execute inline without another design-approval or execution-method question; final approval menus do not supersede that authorization.
