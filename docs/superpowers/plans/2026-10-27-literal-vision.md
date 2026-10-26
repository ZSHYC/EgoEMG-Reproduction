# Official vision execution on the available EgoEMG release

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, preserving the already authorized execution method. Steps use checkbox syntax. This implements the approved full-paper design; no additional approval is needed for ordinary implementation or isolated experiments.

**Goal:** Train and evaluate the real DINOv2 vision models using the pinned original execution semantics, while explicitly identifying the available legacy cohort and independent-validation adaptation.

**Architecture:** New contract/data/runtime files leave all completed legacy runs and live classic sources unchanged. Reuse the original center-only memmap method, real Lightning module, losses and optimizer/callback execution. Stage S first, then the explicitly S-derived B/L recipes; reuse this family for the missing clean ViT-S+S fusion only after its branches finish.

**Tech Stack:** Python, NumPy, PyTorch2.2.2, pinned timm1.0.15, Hydra, Lightning2.5.0.post0, readonly JPEG/LMDB and eeg-server.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`; original source `f7cbd048b7092139852270491d3859cd472e0311`.

## Global constraints

- Preserve all seven completed legacy results, live VEMG source and report/calibration source snapshot. New experiments use separate paths.
- Original DINO-S is `fusion/vision_vit_small`; B/L only change its backbone and are explicitly derived, since separate author YAMLs are absent.
- Native loss aliases sum to effective fingertip0.02. Original bf16-mixed, clip1, AdamW/cosine150, val_mae/min, patience30 and top3/last callbacks remain intact. One rank/workers0 is an explicit execution adaptation, not six-rank equivalence.
- `make_data_module` passes only config.batch_size360; `_make_loader` uses it for train/val/test. The declared val_batch_size256 is unused and must not determine native FK grouping.
- Actual `WindowedEmgDataModule._build_dataset` applies repeat2 to **train, val and test**. Record unique and consumed counts separately; never relabel duplicate test samples as new unique observations.
- The current original vision lineage uses a physically merged EgoEMG/ShowEE/Incre training dataset. Available legacy EgoEMG alone does not reproduce that corpus or establish PDF equivalence.
- Keep independent train-only validation. Preserve original train jitter only for anchors whose entire possible jittered footprint remains within retained train rows and outside val/test footprints. Drop unsafe anchors prospectively; do not clamp jitter, clean labels or tune on test.
- Original point targets use generated_label_valid for all22 angles, wrist degree→radian arithmetic and normalized RGB pre-crops. Missing crop keys produce black/vision-invalid input with the original label mask unchanged; original pure vision does not combine the vision flag into that mask. Do not silently apply legacy per-angle wrist masking.
- No data, raw participant IDs, machine paths, weights or restricted assets in public reports. No scientific completion from configuration, CPU checks, short runs or pretrained loading.

## Review focus

- repeat2 must affect all roles and retain left-then-right order; Task1 tests unique/repeated identities and val/test multiplicity. The bridge already expands repeat2; later loaders must not repeat it a second time.
- A train jitter tail can cross a held-out interval although its anchor window is safe; Task1 tests every admitted offset bound and rejects altered partitions.
- Flat old-release crop files and the webcam/head-index field alias require explicit IO bridging; Task1 compares unchanged original point reads, JPEG pixels, wrist rounding, masks and RNG.
- A black missing crop must not acquire invented label filtering; Task1 preserves original output, Task2 exercises the real original loss with vision-invalid input.
- Offline resume must strict-load actual pretrained-derived full state, never silently randomize; Task2 records true transport/tensor identity and rejects wrong schema/config/source. Task3 retains genuine stop and selected-best evidence without last fallback.

### Task 1: exact configuration and literal center-data family

**Files:** create `egoemg_repro/vision_literal.py`, `egoemg_repro/vision_literal_data.py`, `tests/test_vision_literal.py`, `tests/test_vision_literal_data.py`, readonly `tools/vision_literal_census.py`; document `docs/vision-literal.md`.

**Interfaces:** `compose_vision(upstream, variant='vit_small') -> DictConfig`; `vision_protocol(upstream, config) -> dict` checks exact composition and returns safe metadata. `build_vision_partitions(store, val_fraction=.12) -> dict` creates7790/780/7790 anchors with conservative jitter footprints. `VisionLiteralDataset(data, partitions, crops, *, upstream, config, role)` returns original point tuples plus private sample provenance, in left/right/repeat order.

- [x] Write RED exact recipe/mutation tests: S/B/L derivation, 0.02, repeat2 all roles, actual360 for every loader versus unused declared val_batch_size256, bf16/clip1,150/patience30/top3 and merged-source declaration. Implement source-bound composition and safe protocol only; no model constructor.
- [x] Write RED real memmap/JPEG tests: both hands, head/webcam mapping, exact native point output and rounding, missing key, corrupt/absent database, generated-only mask, readonly bytes and provenance.
- [x] Write RED jitter/partition tests: original randint/RNG behavior, every possible augmented train span outside val/test, end-of-episode bounds and repeat2. Implement the smallest explicit legacy IO bridge around the unchanged native reader; no old-module changes. Empty roles remain length0 and reject indices before division.
- [x] Run targeted checks, actual available-data cohort/source/crop census and deterministic original-reader comparisons. Do not call this full training. CPU evidence:37 targeted,1480/3 full and841/595 light. Actual52-file stat-guarded census and18 original point reads; whole payload hashes and full JPEG decode remain separate future admission. Commit the validated data foundation with remaining tasks explicit.

### Task 2: real native module and finite optimization/reload

**Files:** create `egoemg_repro/vision_literal_runtime.py`, `tests/test_vision_literal_runtime.py`; extend guide.

**Interfaces:** consume Task1 config/data; `build_vision_runtime(upstream, config, *, state=None, expected_initialization=None)` seeds with original setup before `make_lightning_module`; returns unchanged model/module plus actual source, initialization and optimizer identity. Engineering entry uses bounded explicit scope, actual original loss and complete independent-val population.

- [x] RED wrong configuration/state/backbone/pretrained fallback and source-mismatch checks before optimization. Characterize vision-invalid input through real original forward/loss rather than assuming comments enforce filtering.
- [x] Observe the existing timm transport without changing routing; obtain actual loaded backbone tensor SHA, 256px position interpolation and22D output. Strict reload may disable only redundant pretrained transport, then must restore every parameter/buffer.
- [x] Actual CPU engineering optimization, original component groups/AdamW/cosine and clip1; implement the server bf16 engineering entry and strict fresh-process resume. Actual GPU execution waits for Task3's final whole-tranche review. Engineering size/batch differences remain explicit and cannot qualify full training.
- [x] Complete appropriate CPU integration; preserve failures. One independent review of the complete vision runtime/training tranche occurs at Task3, followed by one root RED regression fix pass; do not introduce per-task reviewers.

### Task 3: full training, native selected-best test and scientific reports

**Files:** create `egoemg_repro/vision_literal_train.py` and its tests; separate private resource controller/staging and public evidence.

**Interfaces:** consume Task1/2 identities; complete original-budget fit with no engineering limits, native callbacks, complete repeated validation and selected-best native test. Public center report keeps actual generated-mask sums/counts, hands/splits/users, unique/duplicate populations, source hashes and true termination.

- [ ] RED premature/partial/external/NaN termination, wrong best/last, missing real cohort or pretrained identity, incomplete coverage, unsafe output and resume mismatch.
- [ ] Preserve native bf16 behavior (no invented FP16 scaler),150-epoch cosine schedule, patience30/top3/last and original360 batches for train/val/test (declared val_batch_size256 is unused); workers0/single rank disclosed. Restore actual optimizer/scheduler/loop/RNG/callback state when resuming.
- [ ] Freeze a report agreement policy before test, calibrate on independent validation, and retain actual native batch aggregation. No post-test tolerance changes or checkpoint reselection.
- [ ] After final review/integration, stage frozen code and actual assets, run bounded GPU capacity/full-val/reload checks, then full S training in a new output with finite resource/log limits and no auto retry. B/L follow their own capacity and full runs.
- [ ] Publish only actual completed/audited results; update Tables3/8 and later fusion/occlusion analysis without claiming missing current merged data or PDF equality. Whole-paper goal stays active until its separate full audit passes.
