# Native Fusion Joint Data Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline under the already approved full-paper execution; one final whole-plan reviewer, no per-task agents. Steps use checkbox syntax.

**Goal:** Feed real legacy EMG/RGB windows through unchanged original center-supervised fusion point methods, with original transforms, exact source-bound normalization and disjoint possible jitter footprints.

**Architecture:** A dedicated adapter reuses established readonly crop IO and original center-point reader; it supplies original16slot EMG and center RGB/22angles on its own stride400 partition. A bound normalization descriptor records the released recipe's missing-key failure and the explicit same-revision official unified-stat asset substitution. This is the joint-reader subsystem of the full fusion experiment, not full content/GPU/training admission.

**Tech Stack:** Existing LegacyStore/partitions, pinned original EgoEmgMemmapDataset methods, Hydra transforms, NumPy/PyTorch and real LMDB/JPEG fixtures.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements2/4/7; [native fusion contract/construction](2026-10-28-native-fusion-construction.md); actual normalization probe from pinned original method.

## Global Constraints

- Preserve upstream f7cbd048 and all existing frozen scientific jobs; add dedicated files only.
- WL7790/trainstride400/evalstride7790, original randint jitter for train only, left-role then right-role then repeat2 for all roles. Exclude anchors whose full possible jitter window touches held-out data; never clip jitter.
- Call original _getitem_center_supervised and _convert_to_emg2pose_layout with positions[10,12,0,1,2,4,5,6], channel_interpolate=False; zero-fill before original scalar norm, then original role transforms and transpose.
- Bind original per_dataset_norm_stats_unified.json (SHA6a0d029147fd3f4befef5c33f82c9044606de5a8b335b15ffa434ad3767701f5), original _load_norm_stats namespace egoemg_unified/filtered_paper: mean0.00011658444197544563/std10.839283499203331/epsilon1e-6. Canonical per_dataset_norm_stats.json does not contain this namespace; preserve this failure and explicit substitution rather than claiming unchanged data config.
- Actual physical legacy fields emg_left_filtered/emg_right_filtered are explicit aliases to requested filtered_paper; source sample name remains egoemg. Existing generated labels/wrist conversion/missing crop behavior unchanged, no label cleanup.
- Old legacy-only corpus is not the declared physically merged EgoEMG/ShowEE/Incre; center test masks and missing-image flags never silently drop requests. Schema must require complete8ch float32 EMG payloads plus original point fields/crops.
- Outputs remain scientific_eligible/production_ready/full_paper_complete=False; no complete content scans, trained branches, GPU, full fit or accuracy are claimed by a reader.

## Review Focus

- Missing/corrupt/short EMG, crop or center fields must fail or preserve original missing-crop semantics, never become dummy EMG or a reduced population; Task1 real field/JPEG damage cases.
- Full possible jitter footprints across held-out/split/episode boundaries must remain disjoint, including leftover0/1/17/400; Task1 independent interval/RNG comparisons.
- Zero-filled16slots must be normalized before original train rotation, retaining RNG order; Task1 bit-exact comparison to a real original dataset for both hands/three roles.
- Normalization namespace/path/stat edits and canonical config edits must fail before data use; Task1 original missing-key witness, original unified method witness and damaged descriptor/config cases.
- Head/webcam alias priority, negative/missing frames, false wrist-valid fields and repeat identities must preserve original masks and request counts; Task1 explicit real fixtures.

### Task 1: Original joint reader, normalization and jitter-safe fusion population

**Files:** Create `egoemg_repro/fusion_literal_data.py`, `tests/test_fusion_literal_data.py`, `docs/fusion-literal-data.md`; publish only actual safe normalization probe and CPU reader evidence.

**Interfaces:** `fusion_normalization(upstream)` returns a sealed safe descriptor of the exact original norm method/asset/namespace/physical aliases and substitution. `build_fusion_partitions(store,val_fraction=.12)` returns stride400 disjoint-jitter partitions with repeats. `FusionLiteralDataset(data,partitions,crops,*,upstream,config,normalization,role)` exposes role/views/windows/store/frame_mapping_field/normalization, length4*role_windows, original joint arrays and explicit fusion sample/unique/repeat/window metadata. It accepts only exact bound config, normalization and deterministic partition.

- [ ] Write real memmap/LMDB/JPEG tests for stride400 full-jitter exclusion/repeat, original normalization missing-key/success, both hands/all roles bitwise point/EMG/RGB/mask comparison with real original dataset and original transforms, matching NumPy RNG; leftover0/1/17/400; zero-filled inactive slots have normalized scalar offset; crop aliases/missing frames/invalid wrist masks; wrong partitions/config/norm and missing/wrongdtype/wrongshape/short EMG and merged inputs. Expected: new module missing RED.
- [ ] Run new native tests and preserve actual RED; implement exact signatures above, original method delegation and role transforms, no upstream/old-reader changes. Derive source-bound descriptor by executing original _load_norm_stats with the pinned file, not a new normalization algorithm. Expected: all new tests pass; unchanged fixture files.
- [ ] Run new tests plus original vision-data/partition regressions under native dependencies; run full bare light pytest. Expected: actual exit0, no CUDA. Publish safe actual probe/CPU evidence with scope boundaries, document adaptations and mandatory full-content/GPU/fit work.
- [ ] Commit the reader; run task-done full light; one fresh whole-plan review. Important findings enter one author RED→GREEN fix pass; no re-review. Preserve current training/download owners and public branch.

## Subsequent required scientific work

Full-content hashes and EMG/possible-center finite scan plus exact full population identity; task-trained validation-best branch provenance; original batch200/bf16 GPU training/recovery qualification; full150epochs/validation-best/frozen-test/independent report. The missing merged-source/author asset equivalence stays explicit. Completing this reader does not complete fusion or the paper.

## Execution decisions already supplied

The user selected the official-code route and approved data/training/full-paper work. The original center reader is the only sensible fidelity-preserving implementation; using a vision-only dummy EMG reader or a new network would not meet that request. The same-revision unified normalization file is an explicit source-grounded implementation decision needed because the canonical recipe's namespace is absent in its declared file; record cost/limitations in the execution ledger. No new user preference is required.
