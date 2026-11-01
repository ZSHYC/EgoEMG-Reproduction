# Native RN18 + EMGFormer-S Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. One fresh review of this new unit after Tasks1–4, one material fix pass; no per-task agents or repeated closed-core review.

**Goal:** Complete the pinned RN18+S fusion recipe with true trained branch provenance, independent data roles, complete native training/selection and full fresh-process final evaluation.

**Architecture:** Add a separate RN18 fusion family using the unchanged native `MidFusionPoseFormer`, center reader, Lightning factory and optimizer. Reuse closed interval, crop, content-hash, state and owned-process primitives, without changing existing ViT-fusion/ResNet/EMGFormer families. A dedicated handoff rejects any trained branch overlapping this recipe's future holdouts.

**Tech Stack:** Existing Python3.11, Torch2.2.2, torchvision0.17.2, Lightning2.5.0.post0, Hydra, NumPy and legacy memmap/LMDB inputs. No packages, original source or licensed assets are changed.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements2/4/7; explicit RN18+S in D18. Existing user official-code priority selects this route; no new scientific preference or renewed authorization is needed. [Actual composition/footprint audit](../../../results/native-rn18-fusion-feasibility-20261101.json) supplies the implementation facts below.

## Global Constraints

- Pin upstream f7cbd048b7092139852270491d3859cd472e0311. Compose exactly `fusion/fusion_rn18_s_center_8ch`; similarly named `fusion/fusion_resnet_emgfusion_center` is RN50+M and is not an alias.
- Preserve all14 completed scientific settings, running immutable capsules and protected GPUs. Full-paper completion remains false.
- Original target_hand8 EMG channels, WL12000/trainstride1200/evalstride12000, repeat1, center22×1 generated targets/masks,256RGB, native full jitter and source transforms. No16-slot/7790/repeat2 substitution.
- All roles actually use global batch200 and10workers; declared val_batch_size100 is unused. Persistent workers true/prefetch2/pin_memory true. Full BF16/clip1/seed42/high matmul; original4-device declaration is retained, actual single-rank adaptation disclosed.
- Adam lr6e-4/wd1e-4/betas0.9,0.999/eps1e-8; cosine150/eta1e-5. Full150epochs, complete validation every epoch, top3+last `val_mae/min`, no EarlyStopping. Alias accumulation gives effective fingertip0.02; no legacy0.01 continuation under a changed objective.
- Executable config has freeze_vision_branch=false/freeze_vision_head=false/lock_vision_batch_norm=false, despite the experiment comment saying frozen. Follow executable flags and preserve actual trainability/BatchNorm behavior. Preserve source tiny normal residual final weights and zero bias, never force exact-zero weights.
- Declared `egoemg_unified` namespace fails in its pinned alias asset. Explicit legacy `egoemg` namespace in the same asset gives actual per-hand/per-channel float32 vectors. Record this existing legacy-only source adaptation; do not introduce scalar/unified-stat or train-estimated normalization. Filtered_paper is an IO alias of actual filtered fields; raw files remain unchanged.
- Prospective safe legacy folds contain62470/1052/2792 consumed hand samples; repeat1 means these are also unique. This is not native ViT-fusion395276/3152/8612. Verify actual full content and every possible input/target footprint before any real fit.
- Native EMGFormer12000/full-jitter partitions have zero overlap with this fusion's future val/test inputs, but that producer is still training and its completed checkpoint remains unadmitted. Native vision7790 partitions overlap603751/438725rows; prior ViT-compatible EMG initializer overlaps570349/375600rows: reject both for this fusion even though their weights are genuine.
- Legacy RN18/no-jitter12000 partitions have zero prospective overlap. A full trained historical RN18 LAST may be admitted only after independent complete producer/history/checkpoint/payload reconstruction. Its legacy0.01 objective and other recorded differences must remain explicit. Prefer this available eligible trained source over unnecessary retraining; if actual admission fails, ledger the concrete cause and create a separate clean initializer rather than weakening it or test-selecting a candidate.
- Branch handoff is native vision LAST plus compatible completed EMG validation BEST, complete native key/shape transfer and recorded source/output hashes. No random/untrained substitute, first-convolution expansion, remapping, best-for-last swap or test-based branch selection.
- CPU and CUDA engineering are clearly bounded diagnostics. Formal fit is fresh, never reuses qualification weights; full population, all150epochs, validation-best frozen before test, native/fresh replay/report independently verified. No hidden retry, batch reduction, epoch/time truncation or inferred success from a file/lock.

## Review Focus

1. Unknown config, RN50+M alias, invented freeze flags or wrong normalization/layout must reject before constructors; exact-composition and negative tests belong to Task1.
2. Native jitter's last offset, typed memmap mutations and missing crop/mask/per-hand vectors must not corrupt role coverage; Task2 compares native original samples under identical RNG and rejects generation/layout changes.
3. A valid but differently trained checkpoint can leak future holdout rows or have incomplete BN buffers. Task3 independently reopens all producer evidence, rejects known incompatible partitions and verifies complete transfer/self-contained reload.
4. Adam must not become AdamW, omit temporal-attention parameters, freeze RN18 or restore only learned tensors. Task4 checks exact native optimizer groups/updates and all six complete state domains in a fresh process.
5. A successful partial run or unpaired test must not receive full eligibility. Task4/5 reject incomplete budgets, wrong native best, changed generations and partial/duplicate populations; full test2792 unique points, generated-mask and crop populations remain separate.

## Task 1: Exact recipe contract

**Files:** create `egoemg_repro/rn18_fusion_literal.py`, `tests/test_rn18_fusion_literal.py`.
**Interfaces:** `compose_rn18_fusion(upstream)->DictConfig`; `_validated(upstream,config)->(Path,dict)`; `rn18_fusion_protocol(upstream,config)->dict` safe source-bound metadata, no data/model/optimizer construction or checkpoint permit.

- [ ] Write tests asserting exact source family,8/12000/1200/repeat1/all-role200/10workers, unused100, original Adam/cosine150/validation1/noEarlyStopping,0.02 and actual unfrozen flags; independently assert the declared norm asset/source hashes.
- [ ] Add drift/binding rejection for wrong family, featurizer8→16, window/stride/repeat/cadence/optimizer/loss/freeze/branch-path changes. Forbid Hydra instantiation to prove metadata scope.
- [ ] Observe actual missing-module RED in pinned CPU runtime; implement only composition, exact validation and safe metadata. Re-run the same tests without skips and commit actual source-bound evidence.

## Task 2: Actual native center inputs and complete cohort

**Files:** create `egoemg_repro/rn18_fusion_data.py`, `egoemg_repro/rn18_fusion_cohort.py`, their tests.
**Interfaces:** `rn18_fusion_normalization(upstream,config)->dict` actual original norm lookup with declared failure/explicit legacy namespace; `build_rn18_fusion_partitions(store,val_fraction=.12)->dict`; `Rn18FusionDataset(...)`; `admit_rn18_fusion_cohort(data,crops,*,upstream,config,expected=None,require_full=True)` returns registered cohort with datasets/partitions/identity/check; `native_rn18_fusion_loaders(cohort,*,scope)` executes native data module.

- [ ] RED: exact per-hand float32 vectors, no scalar/zero substitution; identical seeded original center-reader EMG/RGB/targets/masks/rotation and missing-crop behavior, both hands/last batches. Use existing `_PointView` and native `_getitem_center_supervised`, not a new scientific reader.
- [ ] Reuse exact closed `build_emgformer_partitions` window/jitter provider since actual audit proves identical12000 roles, but stamp a separate fusion center-target protocol. Assert complete original jitter and zero all-role footprint overlap; reject malformed fields/layout/index/mask/source or changed generations.
- [ ] Hash every consumed complete numeric/crop/metadata payload and scan all possible consumed EMG and center-label rows. Keep actual readonly typed layout snapshots independent of mutable arrays. Verify actual registered62470/1052/2792 and independent fresh-process full receipt equality.
- [ ] Run the relevant new tests/native IO collector and record exact scopes; no model/GPU/formal-training permit from data. Commit.

## Task 3: Genuine native construction and trained branch handoff

**Files:** create `egoemg_repro/rn18_fusion_runtime.py`, `egoemg_repro/rn18_fusion_handoff.py`, their tests.
**Interfaces:** `build_rn18_fusion_runtime(upstream,config,*,initializers=None,state=None,expected_initialization=None)`; `admit_rn18_fusion_initializers(cohort,vision_fit,emg_fit,output,*,vision_source_root,emg_source_root)` returns process-registered source-bound trained initializers; separate `engineering_rn18_fusion_initializers(...)` grants no full permit.

- [ ] RED: native real RN18+S construction and exact full branch transfer including BN buffers, actual trainability, tiny-normal residual, native center forward/loss/backward and strict complete self-contained reload in an empty asset cache. Do not re-review unchanged underlying network math.
- [ ] Independently reopen full trained producer histories/actual selected/last/checkpoint hashes in their known frozen reader/source. Reconstruct complete possible training footprints and common raw payload identity; reject each known incompatible family from the actual audit and all engineering/forged/partial producers.
- [ ] Record legacy RN18 LAST as a distinct source identity if admitted, including its0.01/mask/normalization/training provenance; do not relabel it as the pending0.02 standalone fit. Admit current native EMG-S only after actual complete200 fit and native validation-best proof. Export native branch tensor format only with complete name/dtype/shape/hash inventory and permitted original exact-name/shape loading.
- [ ] Observe RED→GREEN plus fresh CPU acceptance without any random formal initializer, network or CUDA. Commit.

## Task 4: Native full fit/recovery and paired final evaluation

**Files:** create `egoemg_repro/rn18_fusion_train.py`, `egoemg_repro/rn18_fusion_eval.py`, their tests.
**Interfaces:** native fit policy/`fit_rn18_fusion(...,scope,resume,stop_after_epoch,cohort_receipt)`/`read_rn18_fusion_fit(...,require_full=True)`; full val/test native and replay passes, validation profile freeze, raw report/export/verify analogous to the closed native evaluators. Dedicated CLI paths prevent relabeling existing families.

- [ ] RED: actual CPU prefix then fresh-process resume with all model/Adam/scheduler/callback/loop/RNG domains, original trainability and every parameter exactly once. CPU engineering batch2/workers0/twoepochs/4actual training samples per epoch; CUDA engineering native200/10workers/twoepochs/two training batches per epoch, full registered val/test.
- [ ] Preserve native callbacks and complete epochs/validation every1/selected min. Keep all150 full epochs and native150 scheduler; no test in training. Forbid wrong AdamW, incomplete/duplicate history, changed source/cohort/branch handoff and engineering-as-full claims.
- [ ] Full evaluation reopens selected native validation BEST; freeze actual saved-fit vs independent validation agreement before test. Save all2792 requested test predictions/targets/generated masks/crop flags/identity; preserve original source metrics alongside pooled statistics, users/splits/hands/finger groups and explicit invalid populations.
- [ ] Fresh-process raw replay and report verifier bind every source/input/checkpoint generation and complete population. Observe relevant suites/CPU collectors, commit.

## Task 5: One review, actual qualification, full experiment and publication

- [ ] One fresh-context review of this new family only, then one focused material RED→GREEN correction pass and relevant acceptance. No second review or closed ViT/WiLoR/geometry/RN backbone suite reruns.
- [ ] Stage committed sources; finish actual full data/producer admission and fresh reopening. Wait for a truly available permitted GPU and exact predecessors; keep currently queued/reserved devices protected.
- [ ] Qualify original batch200/10workers/BF16, fresh-process recovery, full validation/test/profile/report/raw verification and actual owned NVML consumer closure.
- [ ] Start a fresh native150 experiment only after qualification. Inspect the same PID/start handles through actual terminal0, reopen all training/selection evidence and execute full final independent evaluation/replay/report. No success from a launch receipt.
- [ ] Publish curves/CSV/JSON/metrics/source/provenance/hardware/time; keep14 existing artifacts unchanged and append a newly verified scientific setting only when this entire experiment closes. Update D18 at actual proved scope; D19/D20, markers/annotation and other paper requirements remain mandatory.

## Preflight and approval continuity

Task1 exact protocol feeds every consumer; Task2 must expose source-bound full target_hand8/12000 data rather than silently project the ViT16/7790 fold. Task3 trained handoff and six-domain runtime feed Task4 full fit; a checkpoint or footprint-only probe cannot satisfy either input. Task4 complete raw final chain feeds Task5 public report; engineering and formal identities remain disjoint. Existing full-paper approval and official-source priority already decide the method. This plan introduces no source-algorithm redesign, new external side effect beyond authorized server/repository work, destructive action or new user preference.
