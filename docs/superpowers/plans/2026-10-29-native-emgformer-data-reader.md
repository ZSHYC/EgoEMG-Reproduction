# Native EgoEMG EMGFormer Data Reader Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task, with one fresh review of this new component. This executes the approved full-paper design.

**Goal:** Read actual legacy EMG using the official native S/M/L train/eval semantics, exposing every adaptation needed before complete input admission and full training.

**Architecture:** New `emgformer_data.py` composes the existing exact runtime contract and reuses unchanged pure legacy schema/view helpers. Build independent train/validation/test partitions on the official 12,000-point episode grid, preserve the complete original train jitter range, train full trajectories, and use genuine center-target-only native validation/test reads. No model, live capsule or old initializer changes.

**Tech Stack:** Existing NumPy/Hydra/OmegaConf/native Torch2.2.2/Lightning2.5.0.post0 and existing process-only dependencies. No installs or asset downloads.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`; approved independent train-only validation and official-code precedence. Runtime component `2026-10-29-native-emgformer-runtime.md` is already complete and must not be reimplemented.

## Global Constraints

- Pinned clean official `f7cbd048b7092139852270491d3859cd472e0311`; original factory, model, transforms, native reader math and live jobs unchanged.
- Native `egoemg_emgformer_{small,middle,large}` train input8×12000/target22×12000; native val/test input8×12000/target22×1. Dataset YAML defaults center_target_only true only for val/test; do not replace it with full-trajectory evaluation.
- Canonical dataset instances request normalization namespace `egoemg_unified`, absent in their declared `per_dataset_norm_stats_repro_filtered_paper_alias.json`. Execute and record the original KeyError. The available legacy adapter explicitly uses the same pinned asset's `egoemg` per-hand8vectors and emits source name egoemg; no canonical unified-data equivalence claim.
- Original filtered_paper request explicitly aliases actual released filtered files in the reader manifest only; raw files/manifests are not edited. Original2D generated-label mask, optional wrist invalid handling, native scalar/array wrist rounding, missing-label defaults and NumPy/Torch RNG consumption must match genuine native reads.
- WL12000/train stride1200/eval stride12000/repeat1. Preserve original test episode-grid centers and split names. Independent val is the existing deterministic12% temporal holdout inside official train; full possible jitter footprints of train, val and test must be disjoint. Drop unsafe anchors, never shorten the native jitter range.
- Original loader B500/6workers/prefetch1/persistent true/pin false/drop_last false/global RNG. CPU engineering alone may construct0worker loaders, explicitly labeled. Original concatenated hand datasets have no session_index_groups; retain the actual RandomSampler behavior.
- Dataset construction and sampled reader checks are not complete numeric admission, GPU qualification, full training or scientific completion. Full available-data admission,200epoch trainer/selection/evaluation/replay remain required downstream.

## Review Focus

- Canonical missing namespace and explicit legacy replacement must remain distinguishable, not hidden by the existing normalization helper's eval_dataset_name.
- Val/test must use native center fast path, including its float64-to-float32 wrist conversion and omission of full trajectory metadata; train retains original full-window mask/rounding.
- Jitter at episode tails and split/holdout edges must keep original exclusive randint upper bound; no overlap of possible train input with heldout windows.
- Both hand order, original evaluation centers, absent optional labels, alias conflicts, metadata defaults and nonfinite masked values must remain observable without silent repairs.
- Loader construction must preserve native500/six-worker/full RNG behavior;0worker diagnostics cannot be called a full production permit.

## Task 1: data semantics and deterministic partitions

**Files:** new `egoemg_repro/emgformer_data.py`; new `tests/test_emgformer_data.py`.

**Interfaces:** `emgformer_data_contract(upstream,config)` returns exact original dataset/loader declarations, original namespace error, explicit legacy normalization/field substitution and false scope flags. `build_emgformer_partitions(store,val_fraction=.12)` returns original test centers, independent train-only val, unchanged train jitter ranges/footprints, counts and source/partition digests.

- [ ] RED: all3configs show center flags false/true/true and canonical namespace KeyError; native B500/6worker/repeat1; invalid fraction/config rejection; exact original test grid and all-role footprint separation including last possible jitter row.
- [ ] Run focused new tests in existing CPU environment. Expected: missing new component failures before implementation.
- [ ] Implement exact contract and pure partition adapter over unchanged `partitions.build_partitions`. Expected: tests pass without model/GPU/data rewrite.

## Task 2: real native samples and loaders

**Interfaces:** `EmgformerDataset(data,partitions,*,upstream,config,normalization,role)` returns left-then-right native samples plus explicit provenance indices. `native_emgformer_loader(dataset,*,cpu_engineering=False)` returns native DataLoader with original full policy or clearly labeled zero-worker diagnostic.

- [ ] RED: compare actual native constructors on synthetic memmaps for train/val/test, both hands,3sizes and original RNG; verify center/full shapes, wrist rounding and generated masks. Test unsafe aliases/optional file absence/metadata/normalization/partition mutations, empty roles, masked NaNs preserved for later admission.
- [ ] Implement through original `_view`/native `__getitem__`, enabling the exact role center flag and actual split IDs. Do not edit reused helpers or old pipelines.
- [ ] Verify native B500 full loader attributes and genuine CPU final partial collation; all new reader tests pass. No new model construction required for this component.

## Task 3: independent reader evidence and publication

- [ ] Run a source-bound CPU reader producer and fresh replay using genuine memmaps/native constructors; retain immutable receipts and source/input hashes, no model/CUDA/updates. Where available, sample actual registered server data with explicit sampled scope; do not claim full finite-footprint admission.
- [ ] One fresh component review, one TDD fix pass for material findings, no re-review/old95-runtime-suite rerun unless modified dependencies require it.
- [ ] Publish actual semantics, normalization incompatibility/adaptation, tests and independent evidence; commit/push/ref verification. Retain all downstream full-paper requirements and zero scientific setting increment.
