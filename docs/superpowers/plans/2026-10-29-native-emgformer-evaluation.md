# Native EMGFormer Selected Evaluation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline and one fresh review of this new component only. Existing full-paper execution authorization applies; no repeat approval.

**Goal:** Evaluate the native 0.02 EMGFormer validation-selected checkpoint with exact point-target replay, freeze validation before test, and publish independently recomputed legacy results.

**Architecture:** New `emgformer_eval.py` consumes closed train44b72b6, runtime1376203 and cohort415deb9 without modifying them. Reuse generic metric/population/source/process helpers; native EMG has one consumption per hand/window, no synthetic vision visibility or vision repetitions. Preserve original forward/Trainer metrics and typed sample/input provenance.

**Tech Stack:** Existing pinned native Torch2.2.2/Lightning2.5.0.post0/Hydra, original native loaders and CPU FP32 or GPU BF16. No installs or model/loss changes.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`.

## Global Constraints

- Require completed source-bound selected-best fit, full training for scientific eligibility; incomplete/engineering fits cannot become full results.
- Native train/val/test cohort62470/1052/2792 hands, repeat1. Full evaluation B500/6workers/BF16, CPU engineering B2/0workers/FP32 and complete synthetic val/test populations. No native test-grid replacement by4306vision centers.
- Same original model forward must select its native center and exact22x1 target/generated mask. Actual unique source index/order, window, hand, split and subject come from cohort; no manual context realignment or second forward.
- Fixed pre-validation policy: exact typed prediction/target/mask/input digest replay, native sample-weighted FP32 batch MAE versus float64 pooled valid-angle MAE, native agreement atol=rtol=2e-6rad. No test-driven tolerance/checkpoint selection.
- Validation native/replay must use distinct PID/start identities and finish before frozen profile and every test pass. Test replay requires same profile/selected checkpoint/source/config/cohort and fresh process; any reference/profile/payload/metric/coverage substitution fails.
- Report all22joint/finger/wrist/hand/Gesture/User/Both/user summaries; anonymize public subject names. No crop subset is claimed without actual visibility inputs. Explicit legacy filtering/normalization/population limitations persist; no PDF equivalence claim.
- Preserve selected model bytes during inference and shut down only owned workers/hooks in all exits. No training updates for this unit; retain existing completed current-source synthetic fit for integration instead of retraining unchanged components.
- Actual GPU qualification and full200 S/M/L are downstream. Spec idle-A800 rule and existing jobs/foreign consumers/reservations remain untouched.

## Review Focus

- Fresh same-process replay, wrong-checkpoint/role/source/profile substitutions, and chronology spoofing must reject before test/model execution where applicable.
- Native full-loop loss aggregation must agree despite partial final batch or all-invalid batch; pooled denominator and joint/finger/hand/split/user counts conserve.
- Independent native source IDs cannot be replaced by duplicated visual-repeat indices or fake visibility; original same-forward center/target/mask must be observed.
- Raw payload and public report must be re-read/recomputed, not accepted by self-consistent JSON seals alone; tampered provenance/metrics/predictions/selection fail.
- Exceptions must close own six workers and restore forward hooks; completed engineering chain never grants full/scientific eligibility.

## Task 1: statistics and selected pass/profile/report lifecycle

**Files:** create `egoemg_repro/emgformer_eval.py`, `tests/test_emgformer_eval.py`.

**Interfaces:** `emgformer_statistics(parts,*,samples,batch_size,role)`; `run_emgformer_pass(cohort,fit_path,output,*,role,kind,reference=None,profile=None,require_full=True)`; `read_emgformer_pass(directory)`; `freeze_emgformer_validation(reference,replay,output)`; `read_emgformer_profile(directory)`; `build_emgformer_report(test_replay)`; `publish_emgformer_report(test_replay,output)`; `verify_emgformer_report(test_replay,output)`.

- [x] RED: exact one-consumption22x1 statistics, FP32native batch/sample weighting versus pooledFP64, partial last/all-invalid batches; hand/split/user/joint/finger conservation. Invalid shapes/dtypes/indices/order/repeat/metadata/digests/nonfinite reject.
- [x] RED: retained current-source synthetic selected checkpoint, nativeval→freshreplay→freeze→nativetest→freshreplay→report/verify;0newoptimizerupdates, exact point/output/input replay, originalbest hash unchanged.
- [x] RED: test beforefreeze, wrongprofile/fit/role/kind, sameprocessreplay, tampered raw payload/coverage/sidecar/selection/statistics/report/completion, incomplete/fullscope reject.
- [x] Implement native original pass and independent manual replay with typed payload/statistics, strict source/cohort/fit/profile chronology and worker cleanup. Reuse unchanged training point observer and generic metadata helpers only.
- [x] Run complete new evaluation suite to terminal success; do not repeat closed73/95/58/59 suites or newtrainingupdates. Commit source/tests.

## Task 2: canonical server selected-checkpoint integration

- [ ] Freeze committed current evaluation source and consume the genuine canonical server current-source CPU fit/data read-only in its original owned directory; preserve its original absolute checkpoint bindings and all original bytes/evidence.
- [ ] Execute separate val-native/val-replay/freeze/test-native/test-replay/report/verify processes under existing resource guards. Recompute exact typed outputs and report, no training updates. Collect actual exits/closure/input/source invariance and publish scoped CPU proof; fullGPU/science pending.

## Task 3: one final review/publication

- [ ] One fresh reviewer of this new unit; one material fix pass with observed RED→GREEN/full new suite, no re-review. If production source changes, fresh current-source proof required; keep historical evidence.
- [ ] Publish source/proof/docs on own branch, independently verify remote ref, update full-goal recovery ledger. No full-paper completion claim.
