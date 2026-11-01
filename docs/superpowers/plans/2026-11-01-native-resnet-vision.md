# Native ResNet Vision Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. One fresh review covers this new ResNet unit before GPU qualification; do not reopen closed ViT/WiLoR/model/data components. Existing full-paper authorization applies.

**Goal:** Execute the original RN18 vision recipe and explicitly derived RN50/152 recipes, with the native0.02 objective, original validation cadence, initialization, checkpoint recovery and final evaluation.

**Architecture:** A new ResNet protocol/runtime/training/evaluation family calls the pinned original model and Lightning factories. Reuse the closed vision point-reader/cohort through a checked data-only projection, following the existing WiLoR companion pattern. Keep original per-model provenance and cadence-aware histories separate from completed legacy runs.

**Tech Stack:** Existing canonical Python3.11/Torch2.2.2/torchvision0.17.2/Lightning2.5.0.post0, Hydra/OmegaConf, existing local ImageNet assets and legacy crops. No installations or upstream/model-mathematics changes.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`; the user's official-code-first objective includes every named model and forbids treating old legacy results as the whole-paper completion.

## Global Constraints

- Pin upstream `f7cbd048b7092139852270491d3859cd472e0311`. RN18 author experiment is `fusion/vision_resnet18`, original `ResNetVisionPose`,22 outputs, ImageNet initialization, seed42 before factory, head512/dropout0.1, unfrozen backbone.
- RN18 actual global loader batch900,8 train/eval workers,prefetch2,persistent true,pin false. The declared `val_batch_size=256` is unused by `make_data_module`; do not silently substitute it.
- RN50/152 have no independent author YAML. Compose RN18 plus explicit backbone and paper-Table8 batch overrides400/180, respectively, as in the approved larger-ResNet plan; label them derived. A backbone-only900 probe is not an author RN50/152 training recipe.
- All three use native AdamW1e-4/WD0.01, cosine150/eta1e-5, BF16,clip1,max150, validation every2 epochs, native `val_mae/min` top3+last and EarlyStopping patience30/check-on-train-end false. Preserve all resolved callback/runtime defaults.
- Effective fingertip coefficient is0.02 from native alias accumulation. Keep completed0.01 legacy results and their source identities unchanged; never change an existing checkpoint's objective on resume.
- Native images are256x256, center-only22x1 supervision, WL7790/train stride780/eval stride7790/repeat2. Closed vision legacy admission counts are202020/3152/8612 consumed samples, with4306 unique test hand centers; verify against the actual admitted identity before use.
- A data companion supplies readers only. Compare all numerical data/normalization/mask/index/augmentation fields; record loader-only differences separately. It is never permission to construct/train a ViT under a ResNet label.
- Preserve original generated center masks and missing-crop behavior. No legacy wrist cleanup, finite-value repair, cohort reduction or prediction substitution.
- Existing single-rank legacy adaptation remains explicit: retain the author six-device declaration in metadata, execute on an actually idle permitted A800, and report single-rank versus author distributed execution without claiming PDF equivalence.
- ImageNet weights must come from locally validated official torchvision assets; record actual enumeration, source/file/tensor hashes. Restore complete states without network fallback or fresh pretraining.
- Never modify running server capsules, complete reports, raw data or old results. Keep old failures; no implicit retries or foreign-process termination. Source/input generations, actual owned CUDA execution and resource reserves must be observed.
- Metadata, CPU integration and short GPU qualification add zero scientific settings. Full termination, validation selection, frozen test, independent replay and source-bound report remain necessary.

## Review Focus

- RN50/152 must not become RN18, inherit an invented author YAML, or silently use the wrong batch. Reject unknown kinds/config drift before constructors or network access.
- Validation every2 epochs must survive training, checkpoint-only reading and resume. Odd epochs cannot be relabeled as fresh complete validations, and stale metrics cannot advance patience.
- The companion projection must reject any numerical reader change, including crop normalization, jitter, repeat, target/mask semantics or typed view mutation.
- Native BatchNorm buffers, all learned tensors, optimizer/scheduler/callback/loop/RNG state and selected-best provenance must be complete; no pretrained fallback on restore or inference.
- Genuine artifact/owned-process/source failures must remain terminal. No engineering record, nominal test skip, timeout or launch receipt grants a full result.

## Task 1: exact protocol and explicit derivatives

**Files:** create `egoemg_repro/resnet_literal.py`, `tests/test_resnet_literal.py`.

**Interfaces:** `compose_resnet(upstream,variant='resnet18')`, `_validated(upstream,config)` and `resnet_protocol(upstream,config)` follow the existing strict literal-config binding style. Supported variants are exactly `resnet18|resnet50|resnet152`. Return source-bound metadata, not instantiated models or data permits.

- [x] RED: assert exact RN18 source/900/8/256/7790/780/repeat2,0.02,native150/validation2/patience30; RN50/152 explicit400/180 derivatives and correct backbones.
- [x] RED: unknown variants, wrong family, source/config/binding changes and raw-loss/monitor/cadence mutation reject before any model/data/optimizer construction.
- [x] Implement the small protocol module by native Hydra composition. Do not duplicate native loss/model mathematics.
- [x] Run only the new metadata suite in the existing pinned environment. Capture actual terminal counts/no-GPU scope and commit.

## Task 2: checked data projection and genuine runtime

**Files:** create `egoemg_repro/resnet_data.py`, `egoemg_repro/resnet_runtime.py`, `tests/test_resnet_data.py`, `tests/test_resnet_runtime.py`.

**Interfaces:** `admit_resnet_cohort(data,crops,*,upstream,config,expected=None,require_full=True)` returns a registered `ResnetCohort` exposing `datasets`, `partitions`, `identity`, `check()`. `native_resnet_loaders(cohort,*,scope)` uses the actual native RN datamodule. `build_resnet_runtime(upstream,config,*,state=None,expected_initialization=None)` returns a checked original Lightning module and construction identity.

- [ ] RED: prove the actual RN and data-only vision companion sample fields agree; permit only explicitly identified loader differences. Reject any changed numerical projection, typed reader/view/index/mask or forged admission.
- [ ] Reuse `admit_vision_cohort` and original point methods without changing them. Bind full raw/crop hashes, registered counts, projected source and actual loader declarations. Engineering scopes remain separate.
- [ ] RED: actual original three-backbone construction, native0.02 step gradients and complete named-tensor reload; invalid backbone/pretraining/state provenance must fail before fallback/network.
- [ ] Reuse `visual.py`'s backbone-specific genuine ImageNet identity helpers where applicable; preserve native factory/setup order and all model tensors/BatchNorm buffers. Derive actual parameter totals rather than equating regression models with paper backbone counts.
- [ ] Observe targeted new-suite success and a source-bound independent CPU runtime/admission reopening. No old ViT/WiLoR631M or unchanged model-math suites repeated.

## Task 3: cadence-aware native fit and point evaluation

**Files:** create `egoemg_repro/resnet_train.py`, `egoemg_repro/resnet_eval.py`, `tests/test_resnet_train.py`, `tests/test_resnet_eval.py`.

**Interfaces:** `resnet_fit_policy(upstream,config,*,scope='full')`; `fit_resnet(cohort,output,*,scope='full',resume=False,stop_after_epoch=None,cohort_receipt=None)`; `read_resnet_fit(output,*,require_full=True)`; `run_resnet_pass(cohort,fit_path,output,*,role,kind,reference=None,profile=None,require_full=True)`, validation freeze/report/verify interfaces analogous to the closed native point evaluators.

- [ ] RED: actual CPU RN18 engineering prefix through epoch2 then a fresh process to epoch4, original validation only at2/4, native best/min selection, six-domain restore and independent saved validation replay. Use explicit small synthetic data, never a benchmark claim.
- [ ] RED: missing/duplicate/partial epochs, falsely filled odd-epoch validation, wrong batch/roles/callbacks, incomplete state, changed source/cohort/selection and engineering-as-full reports reject.
- [ ] Retain native callbacks/optimizer/scheduler and helpers for complete checkpoint tensors/journals. Full mode keeps max150/early stopping and original sanity behavior. Engineering may use CPU B2/0workers or genuine GPU native batch/workers,4 epochs, with a complete-epoch2 prefix; scheduler remains150.
- [ ] GPU engineering uses two native training batches per epoch and complete registered val/test roles. No reduced validation or test population. Record eight actual optimizer updates across4 engineering epochs.
- [ ] Record every actual training epoch and only the genuine even-epoch validations; checkpoint readers must preserve native cadence/loop state and accept real early stopping/max termination without retraining.
- [ ] Reuse original metric/point-reader semantics. Freeze validation/selection before test; independent processes must reopen the same complete selected state, inputs and population. Preserve the chosen native precision/backends; distinguish trainer and default-script evaluation views if they differ.
- [ ] Observe new suites and source-bound CPU fit/evaluation integration to terminal success. Exact checkpoint restoration is distinct from a bitwise uninterrupted persistent-worker trajectory.

## Task 4: one review, genuine GPU admission and complete experiments

**Files:** deployment/evidence in this plan's ignored workspace; final reports under `results/original-resnet/`; update the existing public comparison/status documentation after actual milestones.

- [ ] One fresh review of this new unit only. Resolve material findings once with observed regression RED→GREEN and relevant checks; no second component review or unchanged-model retest.
- [ ] Stage committed sources in independent new server directories. Wait for actually available nonreserved GPUs; current native EMGFormer/EMG2Pose/geometry jobs and foreign consumers remain protected.
- [ ] For each backbone, perform original-batch actual CUDA prefix/resume, full original val/test native/fresh replay/profile/report verification and owned-consumer closure. CPU/metadata receipts cannot grant this permit.
- [ ] Only after genuine qualification enter a fresh max150 fit at its own explicit900/400/180 batch. Preserve original8workers/BF16/validation2/callbacks and full populations; no qualifier weights or hidden batch reductions.
- [ ] Collect actual terminal full fit, selected best and original complete final evaluation/replay/report, counts/curves/hardware/time provenance. Append new protocol-family results; keep all14 previously completed setting artifacts unchanged.

Ruling: this implements the already approved whole-paper model/source scope and its explicit0.02 protocol correction; no new user preference or repeated design-approval round is required. RN18+S is a separate pending native fusion task using `fusion/fusion_rn18_s_center_8ch`; the similarly named `fusion/fusion_resnet_emgfusion_center` is RN50+M/100epochs and must not substitute for it. Future fusion initializer reuse requires actual zero-overlap validation/test footprints and correct native loading, not an assumption from matching architecture names.
