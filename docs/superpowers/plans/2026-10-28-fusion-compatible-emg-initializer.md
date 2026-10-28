# Official EMG Initializer on the Fixed Fusion Holdouts

> **Execution:** superpowers:executing-plans inline under the approved full-paper design. One final whole-implementation review and one author fix pass; no per-task agents. Preserve the complete paper objective.

**Goal:** Train the EMG branch required by the original ViT-S fusion without using any of that fusion's validation or test input rows for gradient updates. Preserve the original initializer recipe and full trajectory supervision; do not silently reuse the incompatible old EMG checkpoint.

**Architecture:** Dedicated configuration/data/runtime/fit adapters execute the pinned original full-window reader, Lightning module, losses, optimizer, scheduler, loaders and checkpoint selection. The only scientific adaptations are the disclosed available legacy corpus, physically aliased filtered fields, independent fixed holdout roles and single-rank/workers0 execution. Existing completed experiments and healthy jobs remain frozen. A fresh complete numeric input admission precedes training; the earlier center-only fusion target scan cannot qualify full-trajectory targets.

**Specification:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements2/4/7; [completed fusion input plan](2026-10-28-native-fusion-cohort.md). This is a mandatory initializer for the original fusion, not a new completed paper setting.

## Authoritative findings

- Fixed upstream f7cbd048b7092139852270491d3859cd472e0311. Exact experiment `emgformer/regression_egoemg_incre_cotrain_wl12000_small`.
- [Actual old initializer incompatibility](../../../results/fusion-initializer-footprint-compatibility-20261028.json): old EMG training overlaps641,552new validation input rows/68of788centers and32,030testinputrows. Old results remain valid under their own recorded protocol; those weights are disallowed for this new fusion fold.
- [Readonly original recipe feasibility](../../../results/fusion-compatible-emg-recipe-feasibility-20261028.json):8channels,WL12000/trainstride1200/eval12000, full original jitter, batch260/allroles,150epochs, lr0.0005, effective fingertip0.01, no early stopping. Original datamodule constructor defaults repeat1; no repeat2 assumption.
- Original `_load_norm_stats` actually chooses separate float32 eight-dimensional mean/std vectors for each hand from `assets/per_dataset_norm_stats_repro_filtered_paper_alias.json`, SHA d4b73febf386ec7b2028639eed8bbd7363d13cbe3fe489149915bcef071ee96a. No scalar substitution or training-only claim.
- Prospective native episode-grid selection yields31,513training windows/63,026hand samples and291validation windows/582hand samples on the registered legacy release. Counts are a source-bound feasibility result, not an existing implementation or training result.
- Full fusion input source32f10bd has already completed54payload/61,379,659,366byte checks and fresh-process reopen. Its heldout partition content SHA is37623c78591368aab9ee7bcebe47850aa3eb978ce620a97309bbcbfb0211166d.

## Invariants

- Do not change upstream, old datasets/guards, original completed weights/reports, DINO-L, EMG2Pose source preparation/queue, or acquisition owners.
- Preserve original target-hand8channel architecture. The subsequent original fusion loader's8-to16 first-convolution mismatch is recorded/skipped by its native rule; never reshape or invent learned tensors.
- Compose the exact experiment, not the example CLI comment's additional augmentation override. Record resolved native transforms, effective loss aliases, optimizer/scheduler and callbacks. Original factory performs seeding before model construction.
- Training anchors stay on the original episode1200grid with original `randint(0,min(1200,episode_end-anchor_end))`; reject unsafe anchors instead of clipping jitter. Every possible full input row must be official train and outside the union of all fixed fusion validation intervals and test footprints.
- Validation uses non-jittered12000windows wholly within those existing fusion validation intervals. Do not make a second test-driven split, evaluate test targets for checkpoint selection, or train on future test rows. No initializer final-test score is required or claimed.
- Use original full-window22target construction, wrist degree-to-radian conversion, generated validity and original native transforms. No center-only replacement, NaN repair or label-based sample dropping. Empty/malformed fields must reject.
- Register all consumed numeric generations before any memmap/partition reads. Bind actual readonly layout/strides/offset and source identity. Hash complete consumed inputs and scan every possible full-window EMG **and full-trajectory target** row; strict reopen independently recomputes the identity.
- Missing Incre training corpus, legacy/PDF label differences, whole-episode FFT dependence and published pooled norm provenance stay explicit. Nothing here proves publisher authentication or PDF numerical equivalence.

## Task 1: Exact recipe, compatible partitions and native trajectory inputs

Create `egoemg_repro/emg_initializer_literal.py`, `egoemg_repro/emg_initializer_data.py`, their tests and a concise Chinese guide.

- [x] RED before implementation: exact untouched original composition/config rejection; actual native per-hand normalization selection and asset binding; original default repeat1/batch260; effective loss aliases and complete source hashes.
- [x] RED for all-role complete possible-footprint separation, episode/center/grid/jitter boundary cases, validation containment, prospective counts, invalid/empty roles and deterministic partition seals. Test an actual crossing anchor and last possible jitter offset.
- [x] Execute the unchanged native full-window `EgoEmgMemmapDataset.__getitem__` through an IO/metadata context. Reuse source metadata defaults only where the original does; alias legacy filtered names explicitly. Compare real small files against a genuine complete native fixture with identical seeded RNG, both hands, full22×12000trajectory/mask, transforms, missing optional labels and final partial native collation. No synthetic training data may enter the scientific run.
- [x] GREEN new tests plus relevant existing reader/source regressions. Publish source-bound synthetic evidence with no scientific eligibility. Do not use its finite scan to admit real data. Actual35missing-module RED;35initial native GREEN,42new plus45existing fusion checks=87passed. Independent new-process collector4fulltrajectory fixturepoints/unchanged constructionRNG/inputfiles/no model/CUDA. Real full admission/fit still pending.

## Task 2: Complete actual numeric cohort and original model/fit/recovery

Create dedicated `emg_initializer_cohort.py`, `emg_initializer_runtime.py`, `emg_initializer_train.py` and meaningful tests. Reuse verified generic storage/owned-stage primitives without changing frozen consumers.

- [x] RED then implement all consumed complete-file hashes, preopen generation guards and full possible-window numeric finiteness; independently recompute expected receipts and full population. No unused crop/image requirement. Actual native loaders use260/repeat1/global RNG and explicit workers0. Missing-module RED28failed10.51sec;GREEN28passed61.28sec. Independent synthetic real-file16payload/fulltrajectory admission and fresh-process strict reopen actualexit0/exactidentity. Actual real-corpus qualification completed separately with16payload/16,540,457,757bytes/twofreshprocesses/63,026train582val in72.07sec; sourcee0f11f1, no model/CUDA.
- [x] RED then use original `setup_runtime`/`make_lightning_module`, forward/loss/backward, AdamW and cosine callbacks unchanged. Record architecture, complete model schema, actual loss composition, all optimizer groups and source/runtime hashes. Fresh construction is random, not a fabricated pretrained state.
- [x] Native fit supports a clearly bounded engineering mode and actual150epoch mode with all admitted samples and complete validation each epoch. Save native top3/last and select actual `val_mae/min`; no early stopping or altered loss/precision/learning schedule in full mode.
- [x] Save/restore model, optimizer, scheduler, native loop/epoch state, callbacks and all RNG states in a fresh process. Compare a frozen validation pass exactly before continuing from the next complete epoch. Keep original failures; no automatic whole-fit retry or false terminal success.
- [x] Bind checkpoint lineage to the fixed partition/cohort and source. Export only the source-selected EMG-best tensor format needed by the original fusion loader, with a complete native tensor inventory and exact byte/shape checks; never rename a last checkpoint into best.

## Task 3: Real qualification, review, full150epoch execution and independent acceptance

- [ ] Run appropriate native/local checks, then an independent CPU collector using real-file fixtures and a fresh restore process. Run the complete light suite once after final implementation; repeat only after a relevant change/failure.
- [x] Stage only committed source in a new isolated server directory. Complete real legacy numeric admission and fresh-process reopen; require63,026training/582validationhand samples or stop and diagnose the prospective discrepancy. Do not reduce populations to pass capacity checks.
- [ ] On an actually available GPU, qualify native batch260/bf16 forward/backward/update, complete validation, an actual full engineering epoch and fresh-process six-state restoration. Preserve DINO-L and reserved EMG2Pose queue GPUs; inspect live availability first. Record real resource observations, not injected values or repeated-sample estimates.
- [ ] One final whole-implementation review of the committed plan diff; accept material findings, one author TDD correction pass and affected real qualification if source changes; no re-review loop. Preserve review rulings and archive only this plan's own private workspace after completion.
- [ ] Start one source-bound, resource-guarded full150epoch initializer job with durable PID/start/actual-exit evidence and no hidden time/batch/epoch truncation. Observe the same handle; an observer timeout never justifies restart.
- [ ] After actual completion, independently reopen selected-best/last/history/validation selection and initializer tensor inventory in the frozen runtime; preserve exact source/checkpoint/partition hashes. Publish actual curves/selection and disclose missing Incre/legacy limits. No final-test or additional completed-model count is invented.

## Following work remains mandatory

Load this verified EMG-best and the already trained, compatible ViT-S-last through the reviewed original fusion constructor. Complete actual fusion batch200/bf16/recovery/full150epoch/independent validation-best selection/frozen test/report. The remaining paper models, EMG2Pose experiments, preprocessing/annotation/statistics/occlusion/qualitative requirements remain in the active full-paper objective.
