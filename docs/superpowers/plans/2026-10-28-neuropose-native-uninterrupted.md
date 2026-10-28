# Native uninterrupted NeuroPose implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, task by task. No per-task implementer/reviewer agents; one final whole new implementation review after actual qualification.

**Goal:** Complete the official NeuroPose fresh-training experiment with strict selected-checkpoint replay and full scientific reporting, retaining the old failed resumed-trajectory gate unchanged.

**Architecture:** New standalone training and evaluation modules consume existing immutable original factory/data/numeric/capacity primitives. Native original Lightning runs once with no continuation interface; independent readers verify actual checkpoint/journal/selection and frozen-model prediction evidence. Old modules and running capsules remain untouched.

**Tech Stack:** Existing PyTorch2.2.2/Lightning2.5.0.post0/Hydra/NumPy; original FP16 CUDA; CPU32 fixtures; Matplotlib report extra.

**Spec:** [Native uninterrupted design](../specs/2026-10-28-neuropose-native-uninterrupted-design.md), derived from the approved full-paper design.

## Global constraints

- Fixed official upstream f7cbd048b7092139852270491d3859cd472e0311; preserve all existing source/checkpoint/evidence bytes and old failed gate outcomes.
- Full NeuroPose16slots/7790points/22angles, effective fingertip.02, B256/16-mixed/single rank/workers0, all101714/1576/4306 hands, max100/val_loss/min/patience50; no model/operator/batch/normalization replacement.
- Full test requires genuine full termination, native validation-selected best, frozen complete validation native/replay, actual native test and independent new inference. No test selection or runtime fallback.
- Engineering explicitly two epochs: CUDA512train/all1576val/B256; CPU up to4train/all small validation/B2/32-true. Neither is scientific or a full-training permit.
- No resume/overwrite/implicit restart; actual source/runtime/cohort/precondition/selection identities must be rechecked, all evidence retained and public data sanitized.

## Review focus

1. Native ModelCheckpoint saves before some loop counters increment: checkpoint metadata must bind the actual complete epoch, selected score and journal without guessing from a stale native counter (Task1).
2. FP16 GradScaler can skip an attempted step: count actual optimizer executions and preserve scaler/native states; don't equate global_step with real updates (Task1).
3. Missing or corrupted later batches must fail coverage/fingerprint checks even if counts and first predictions appear valid (Tasks1/2).
4. CPU/old failed/changed-source qualifications must not open full training; freeze actual CUDA selected-model validation and new-process replay before accepting it (Task2).
5. Selected-best replacement, changed raw journal/curve, engineering promotion and test-before-freeze must fail even with resealed sidecars (Task2).

### Task1: Native fit and independently reopened best

**Files:** Create `egoemg_repro/neuropose_native_train.py`, `tests/test_neuropose_native_train.py`, `docs/neuropose-native.md`.

**Interfaces:**
- `neuropose_policy(*,scope='full') -> dict`: exact three scopes and no caller recipe overrides.
- `fit_neuropose(data,*,upstream,config,partitions,output,scope='full',preconditions=None,cohort_receipt=None,finite_probe=None,qualification=None) -> dict`.
- `read_neuropose_fit(output,*,require_full=True) -> dict`: reopen actual data/source/identity/journal/checkpoints; return real selected-best descriptor only after scope-specific terminal verification.
- `reopen_neuropose_best(output,*,require_full=True) -> PreparedNeuro`: construct the original module, strictly load the actual selected model/scaler and rebuild its data; private `_validation(prepared) -> dict` runs the whole native validation and compares its typed digest/score with the selected epoch. Task2 consumes this same prepared interface.
- Private preparation may reuse `classic_emg_train.prepare_classic_run` for ordinary CPU fixtures and `classic_emg_formal.prepare_formal_run(...,engineering=True)` for actual numeric/capacity/complete-cohort CUDA admission; it creates a separately sealed native identity, not an old formal acceptance. Full scope additionally requires Task2's raw qualification reader before construction.

- [x] Step1: Write policy and rejection tests: exact full recipe/counts/budget, two explicit engineering scopes; unknown options/resume/wrong variant/caller pass-bool refuse before data/output access; old formal source SHA unchanged.
- [x] Step2: Observe RED, implement policy/preparation and native fit. Original factory/callbacks/optimizer/scheduler, full train/validation Coverage, actual optimizer post-step hook, finite native loss, complete sample-weighted validation, atomic epoch journal and identity checkpoint packet. Hold15GiB plus checkpoint space before output/saves. Persist native callback/scaler/loop/RNG; no custom resumed trajectory handling.
- [x] Step3: Add real ordinary memmap CPU fixture tests: native two epochs, original NeuroPose parameters actually change, whole validation consumed, actual best and native last differ as appropriate, new process strict selected-best reload and complete validation digest agrees. Include invalid masks/nonfinite/partial or duplicate coverage, native checkpoint/journal corruption, interruption/incomplete termination, source/config/data mutation and engineering-default-full refusal. Read full termination through native max100/patience50 callback state; default reader cannot accept two epochs. Actual33 new native checks and60 related original reader/coverage/callback/termination checks passed; all finite/invalid/partial cases include unchanged helper coverage.
- [x] Step4: Observe RED for reader failure classes, implement strict actual journal/checkpoint/source/selection reading; run meaningful targeted native tests and related helper regressions plus bare light suite once. Preserve failures, verify actual owners exit/source hashes, publish CPU engineering evidence and commit. No GPU/full result claim. Actual33/60native and1474bare checks passed; initial bare opt-in environment failure preserved/corrected without source/assertion edits. Committed a0e4f4f source then independently strict-reloaded/full-validated the retained original CPU model in a new process; exact predictions, no CUDA/new updates, native fit files byte-preserved. Evidence `results/neuropose-native-fit-cpu-20261028.json` and validation sidecar.

### Task2: Selected-best exact replay, qualification and report

**Files:** Create `egoemg_repro/neuropose_native_eval.py`, `tests/test_neuropose_native_eval.py`; extend `docs/neuropose-native.md`.

**Interfaces:**
- `run_neuropose_pass(fit,output,*,role,kind,reference=None,profile=None,require_full=True) -> dict`.
- `read_neuropose_pass(output) -> dict`, `freeze_neuropose_validation(native,replay,output) -> dict`.
- `qualify_neuropose(fit,native,replay,output) -> dict`, `read_neuropose_qualification(output) -> dict`: only actual current-source CUDA engineering/full1576validation/new-process exact predictions; bind raw fit/pass files, preconditions/cohort/GPU/runtime, no pass boolean.
- `publish_neuropose_report(fit,native,replay,output,*,require_full=True) -> dict`, `verify_neuropose_report(replay,output,*,require_full=True) -> dict`.

- [ ] Step1: Tests first for complete native/replay CPU engineering val/test using original tuple, ReportInputLedger and ClassicAngleReport; fixed exact typed hashes/order and sample-weighted native loss; 22-angle/7790trajectory center3895, original generated mask, final partial batch. Reject changed later inputs/targets/masks/order, wrong selected weights, same-process fake replay, engineering/full promotion and test-before-profile. Observe RED then implement genuine native Trainer validation/test with observed original forward and separate inference replay; no new dtype/tolerance/model.
- [ ] Step2: Tests first for qualification raw-chain/current-source/runtime/cohort binding and refusal of CPU/old failed/summary-only artifacts. Implement pretest freeze and actual qualification reader; incomplete artifacts remain failed, no tolerance expansion or old run reinterpretation.
- [ ] Step3: Tests first for safe complete reports, trajectory/center/hands/splits/anonymous users/joints/fingers, native-v-pooled metrics, actual journal curves/best epoch, source mutation/resealed wrong-best/modified raw curve refusal, output nooverwrite and completion-last. Observe RED then implement; default full only, explicit engineering diagnostics false-scientific.
- [ ] Step4: Run the affected native/evaluation tests, related regressions and bare light suite, then an independent fresh-process retained CPU fit/prediction/report verification. Inspect curves, record actual exits/hashes, commit source and truthful evidence. No repeated old project reviews.

### Task3: Actual GPU qualification, one final review and full experiment

**Files:** Task-owned immutable stage/controller/observer manifests in ignored scratch; public `results/neuropose-native-*-20261028.json` and final scientific results/status.

- [ ] Step1: Freeze committed source and numeric/capacity/registered full input artifacts in a new server stage. Fresh resource/GPU observation; original B256/16-mixed two-epoch512train/all1576val native run; actual last/best and all ordinary state fields retained. Never downgrade model/batch/precision to pass.
- [ ] Step2: Native complete validation of actual selected best plus a separate new-process exact input/prediction replay; qualify from reopened raw artifacts and actual exited owners. Fail closed without threshold changes; preserve old failed gate and every new failure.
- [ ] Step3: Request the sole final fresh-context whole new implementation review. Grade C/I/M by actual effect, one accepted C/I author RED→GREEN fix pass with affected green suite; ledger minors/declined rulings, no second review. Source changes require affected actual qualification to match.
- [ ] Step4: After actual qualification/review, freeze original complete-cohort full identity and launch one fresh B256/16-mixed max100/patience50 native training, with resource/owned-process guards. No resume, no test during fit, no restart on observation timeout.
- [ ] Step5: Observe actual termination; independently accept full fit/best, run full validation native/replay and pretest freeze, full native test/new-process exact replay, report/curves/publication. Update scientific count only after this completes; retain all wider paper work and do not archive the active full-paper worktree.

## Execution ruling

The scope is the existing requested NeuroPose scientific experiment. Its original model, full population, precision, batch and stopping behavior are unchanged. The new strict frozen-model reload gate proves what is needed for this fresh run; it makes no resumed-training-trajectory or distributional-equivalence claim. No new user approval is required by the already authorized full-paper implementation. Execute inline and preserve the ongoing fusion/initializer/DINO/EMG2Pose/download tasks.
