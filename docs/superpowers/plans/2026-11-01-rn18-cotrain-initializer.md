# RN18-Compatible Exact Cotrain-S Initializer Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The full-paper design already authorizes this required producer. One new-unit review after implementation, one focused material fix pass; no per-task agents or closed model/operator re-review.

**Goal:** Produce the designated native cotrain-S150/0.01 validation-BEST initializer on the RN18 fusion's fixed independent holdouts, with complete data, state, training and export provenance.

**Architecture:** Reuse unchanged `emg_initializer_literal` composition/normalization and `emg_initializer_runtime` original factory/loss/optimizer. New data/cohort/fit adapters use the already accepted RN18 temporal roles, full original trajectory labels and native12worker loaders. The prior ViT-compatible cotrain producer and active EMGFormer200 baselines remain independent, immutable experiments.

**Tech Stack:** Pinned upstream f7cbd048b7092139852270491d3859cd472e0311, existing Torch2.2.2/Lightning2.5.0.post0/Hydra/NumPy stack, readonly legacy memmaps and managed owned stages. No dependency or source model edits.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements2/4/7; [RN18 fusion plan](2026-11-01-native-rn18-fusion.md), Task3. [Actual RN18 full data admission](../../../results/native-rn18-fusion-full-cohort-admission-20261101.json) proves fixed future partition contentSHA44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0 and full62470/1052/2792 roles.

## Global Constraints

- Exact released experiment `emgformer/regression_egoemg_incre_cotrain_wl12000_small`,8target-hand channels, WL12000/trainstride1200/eval12000, full native exclusive jitter, repeat1, full22×12000 labels/masks for BOTH train and validation. No center-target substitution or initializer test role.
- Source AdamW lr0.0005 and unchanged complete parameter grouping/betas/eps/weight decay; cosine150/eta_min5e-6, effective fingertip0.01, native transforms and actual resolved batch augmentation. Do not execute optional comment-only CLI augmentation overrides.
- Full150epochs/validationevery1/top3+last val_mae/min/noEarlyStopping, nativeB260 allroles and12workers train/val, pin_memoryfalse/persistenttrue/prefetch2/evalforkserver. Source four-device declaration remains recorded; actual single-rank adaptation is explicit. Preserve native global loader RNG; no private generator.
- Original cotrain data includes EgoEMG both hands and Incre right. Only admitted released legacy data is available; state this corpus limitation, never synthesize/duplicate Incre. Published per-hand/per-channel float32 normalization and physical filtered_paper→filtered alias remain explicit; no train-estimated or scalar normalization.
- Reconstruct exact RN18 fusion roles and reserve its complete validation intervals plus future test footprints. All possible jittered INPUT AND full TRAJECTORY TARGET rows must be disjoint from future holdouts. Actual producer counts62470/1052, no test; future partition contentSHA must match the already accepted44e9... family before full training.
- Reuse closed original full reader `_view`/`EmgInitializerDataset.__getitem__`, schema and normalization; new constructor validates separate partitions. Reconstruct process-local readonly views on pickle/spawn/forkserver rather than serializing native memmap caches or calling incomplete native __setstate__ on manual IO objects.
- Preserve14 completed scientific settings, original10server flows and2dense CPUflows plus protected GPU reservations. Never reuse the active nativeS200/0.02 or differently partitioned ViT initializer as this producer.
- Complete byte hashes and all possible finite targets precede any full fit. Data-only/engineering identities and arbitrary resealed receipts grant no trained-branch eligibility; process-registered admissions check independent source/config/reader snapshots.
- CPU engineeringB2/workers0/twoepochs/four train samples per epoch; CUDA engineering nativeB260/12workers/twoepochs/two train batches, full validation. Formal fit is fresh, source150scheduler/full150epochs; no qualification weights, hidden batch/epoch reduction or auto retry.
- Resume verifies model/AdamW/scheduler/callbacks/native-loop/Python-NumPy-Torch RNG state at the native restore boundary, complete validation and owned checkpoint/history. Disclose worker/iterator RNG reconstruction and no guaranteed bitwise uninterrupted persistent-worker trajectory.
- Export native validationBEST only after actual150complete/fullcohort/history/checkpoint validation. Full key/dtype/shape/hash inventory, actual source/future partition and original state; no test selection, no random substitute and no renamed old initializer. Native geometry, MANA/FK/network math remains closed.

## Review Focus

1. A full12000-input reader can still silently use center labels, mask cleanup or a different future holdout. Test native full outputs/masks/RNG and complete train-target footprint separation.
2. Manual native IO instances cannot be unpickled through the original `_open_memmaps` without its complete constructor state. New wrapper must reconstruct readonly views and retain norm/jitter/transform settings in actual worker processes.
3. Current datamodule source actually uses12workers and evalforkserver. Construction alone cannot prove worker iteration/closure; test real worker transport and full native qualification, retaining all native prefetch/persistence/global RNG behavior.
4. Resealed metadata, stale imported source, dtype/layout/filename changes or copied dataclass admissions must reject. Full finite scanning covers masked-false trajectory values, not only source centers.
5. A compatible-but-different recipe or partial trained state must not export. Full150budget, native callback-selected BEST, allsix restored domains, explicit scope and future RN18 partition must bind before handoff.

### Task 1: exact role contract and original full reader

**Files:** create `egoemg_repro/rn18_initializer_data.py`, `tests/test_rn18_initializer_data.py`.
**Interfaces:** `rn18_initializer_contract(upstream,config)->dict`; `build_rn18_initializer_partitions(store,val_fraction=.12)->dict`; `Rn18InitializerDataset(data,partitions,*,upstream,config,normalization,role)`; `native_rn18_initializer_loader(dataset,*,scope)`.

- [x] Write tests: exact cotrain150/.01/8/12000/B260/12/evalforkserver protocol; reject200baseline or altered config; no model construction at contract time. Expected source-exact metadata.
- [x] Write tests: new windows exactly RN18train/val anchors/fulljitter, train/target union excludes complete future val/test; no initializer test role; all source masks/optional labels retained. Expected full22×12000 native target shapes in both roles.
- [x] Native byte/RNG comparisons on both hands/train/val including tailjitter; actual pickle/forkserver reopening and loader metadata. Expected readOnly8channels, repeat1/sourceflags/noRGB, exact original transforms/normalization, real worker values and consumers close.
- [x] Observe missing-module RED, implement only new IO/metadata adapter reusing unchanged original reader functions. Run relevant new tests in rich CPU runtime with0skips and commit. No old model/math suites.

### Task 2: complete full-trajectory data admission and actual population

**Files:** create `egoemg_repro/rn18_initializer_cohort.py`, tests, full CPU collector and evidence.
**Interfaces:** `admit_rn18_initializer_cohort(data,*,upstream,config,normalization,expected=None,require_full=True)` returns registered `Rn18InitializerCohort(datasets,partitions,identity,check)`; reuse native loader via explicitscope.

- [x] RED for complete raw hashes, full possible jitter/target finiteness including masked-false/wrist rows, typed reader/generation/source/config/normalization mutation and unregistered/resealed receipt rejection.
- [x] Implement source-before-import/read guards, complete path/byte/schema bindings and independent live registry. Preserve closed source/hash helpers; use streaming for large actual files and content hash actual providers on check. Expected no constructor RNG/model/optimizer/CUDA changes.
- [x] Independently produce/reopen a real small numeric cohort, then stage exact committed sources for two complete actual-data passes. Expected actual62470/1052, matching future44e9... roles, all consumed payload SHA/finite trajectories equal, all original owners closed0. Data acceptance alone is no trained initializer.

### Task 3: original fit/recovery and validationBEST export

**Files:** create `egoemg_repro/rn18_initializer_train.py`, tests, new CPU collectors.
**Interfaces:** dedicated `rn18_initializer_fit_policy`, `fit_rn18_initializer`, `read_rn18_initializer_fit`, `export_rn18_initializer_best` and `read_rn18_initializer_export`; distinct format/lineage prevents old family admission.

- [x] RED exact source150/AdamW/.01/B260/12worker policy and no early stopping/test access; CPU prefix/fresh resume complete six-state snapshot and native validation/callback/top3 semantics.
- [x] Implement new data/cohort binding with unchanged native runtime/operators. Reuse closed state/checkpoint/callback primitives; preserve original training operations and global-loader RNG, explicit constructor/scope/source/receipt identities.
- [x] RED partial/forged/bad-BEST/wrong-fullpopulation/changed-futurefold/export inventory rejection; fresh CPU complete engineering result/reload/export remains engineering only. Expected exact complete branch tensors and honest loader/RNG limits, no full permit.
- [ ] Run relevant tests and separate actualCPU prefix/resume/strict raw reopening against the same new source; all owned consumers/source/current inputs closed/unchanged. Commit.

### Task 4: single review, genuine GPU qualification and fresh full150

- [ ] One fresh review of this new producer integration, one focused material RED→GREEN fix pass; no second review or closed factory/FK/MANA/model suite.
- [ ] Stage committed source, registered actual cohort and fixed qualifiers. Wait for a truly idle permitted GPU after exact preceding workflows, no interference with reservations/other users.
- [ ] Qualify actualB260/12workers/BF16/forkserver validation, fresh recovery and complete validation, export/reopen/owner/NVML closure. Expected no batch fallback or engineering-for-full acceptance.
- [ ] Fresh full150/complete train+val at original recipe; actual terminal0 and complete histories, selected native BEST and all raw source/input/checkpoint bytes verified. No success from launch metadata.

### Task 5: publication and native RN18 fusion handoff

- [ ] Publish source/protocol/curves/provenance/fulltraining/export and actual hardware/time/owner receipts; keep private tensors/data and previous14settings intact. This initializer is a producer, not a new paper model-setting score.
- [ ] Admit full validationBEST through RN18fusionTask3 with future44e9... partition, complete tensor inventory/shared raw bytes/sourcefit bindings and0futureholdout overlap. Legacy RN18 LAST eligibility remains a separate prerequisite; full RN18 fusion150/final testing is still required.
- [ ] Whole-paper D19/D20/Tables/Figures/markers/source-data/other models remain mandatory; do not mark full paper complete from this producer.

## Preflight Decisions

Task1 full-trajectory ds feedsTask2: its val view must be full despite the future center-supervised fusion role. Task2 complete actual cohort/registered identity feedsTask3: full targets and future holdout binding replace the old ViT producer's different folds/counts. Task3 native complete/export lineage feedsTask4: workers12 and actual150 need new scope and qualification. Task4 real completedBEST feedsTask5: dataset/protocol/shape compatibility alone is insufficient. Existing approved full-paper design and explicit source priority select the exact recipe; no renewed preference/approval round. Native12workers is preserved in this new route; prior closed workers0 producer is not silently relabeled or rerun.
