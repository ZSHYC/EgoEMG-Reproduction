# Native Point Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Execute the original MANO/world/camera/self-occlusion functions on the complete admitted legacy validation/test point grids, with actual shape parameters and explicitly named camera resolution policy.

**Architecture:** One scientific CLI reads the typed point payloads produced by the completed full geometry admission; no bulk inputs are rewritten. It imports the pinned author evaluation module and calls its original mesh and intrinsics functions and `egoemg.occlusion.compute_self_occlusion`. Private per-point native arrays and a source-bound ledger support fresh-process replay and a public aggregate report.

**Tech Stack:** Existing Python/NumPy/Torch/SMPL-X/Chumpy environment, fixed upstream f7cbd048b7092139852270491d3859cd472e0311, licensed admitted MANO assets; CPU only.

**Spec:** [Approved whole-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements4,6,7; D19 in [full-paper status](../../full-paper-status.md). This implements the existing authorized scope, with no renewed preference/approval round. The existing official-code route determines the method; no alternate rasterizer is introduced.

## Global Constraints

- Preserve all14 completed scientific experiment artifacts and every running training/data job; full-paper completion remains false.
- Standard grids are1,576/4,306 unique validation/test hand points (3,152/8,612 repeated model consumptions); this is not the later1/4stride dense grid or Figure13 paired-gain result.
- Bind the exact committed numeric admission and its role-specific point-file SHA; verify metadata and calibration SHA and all expected typed arrays before model construction. No beta=0 substitution.
- Original mesh uses right MANO, `flat_hand_mean=False`, zero global orient, pose[3:48], left X mirror and reversed faces. Preserve this source behavior and disclose legacy manifest's different convention declaration; do not silently correct it.
- `author_fallback_1920x1080` is an explicit required policy for the currently absent videos. Call the original intrinsics routine, retain its calibration scaling/crop and unused distortion behavior, and record actual video dimensions unproven. Never relabel fallback as observed dimensions.
- Backfill removed NumPy names/inspect.getargspec only inside disposable processes for unchanged Chumpy; no package/environment/source patch or network download.
- Original z-buffer, 0.005m depth agreement, window_half2 and triangle-area weights execute unchanged. Invalid/zero-pose rows remain represented; unexpected native failure on a valid pose is an error.
- Private model/input paths and licensed models are never published. Run output is independent of original labels/manifests; records distinguish a two-frame probe, complete role execution, and independent replay.

## Review Focus

1. Same-byte input corruption of typed layout/hand ordering or modified role SHA must reject before geometry construction; tests pin type/shape/order and hash checks.
2. Missing actual beta, silently zero beta, or swapped hand selection must not produce a successful native result; tests require original row values and a bilateral probe.
3. Original helper exceptions returningNone must fail on otherwise valid poses; zero-pose rows are retained without invented scores.
4. Source/input generation changes during an audit or between cached role/episode uses must reject; no persistent evidence cache.
5. Explicit fallback dimensions cannot become an observed-video claim; output and CLI reject unknown/implicit resolution policies.

---

### Task 1: typed point inputs and source-bound CLI contract

**Files:** create `scripts/paper/run_native_point_geometry.py`; create `tests/test_native_point_geometry.py`.

**Interfaces:** `load_point_inputs(points:Path, admission:Path, metadata:Path, calibration:Path, role:str)->dict` validates the committed admission, exact role file content, metadata/calibration and the fixed typed arrays; `select_points(inputs:dict, mode:str)->list[tuple[str,int]]` returns complete left/right role order or deterministic first tracked, marker-valid nonzero point per hand. Modes are `complete_role` and `bilateral_probe` only.

- [x] Write tests for valid typed loading, file-hash tamper, wrong role/shape/dtype/order, nonfinite transforms/beta, deterministic bilateral selection, absent beta and unknown modes/policies. Synthetic fixtures are explicitly engineering data; they do not grant registered/full geometry results.
- [x] Run `pytest -q tests/test_native_point_geometry.py` on the pinned CPU runtime. Expected: new behavior fails before implementation, not a setup error.
- [x] Implement exact schemas/source-safe IO and the CLI `--role val|test --mode bilateral_probe|complete_role --resolution-policy author_fallback_1920x1080`, with explicit input/upstream/assets/chumpy/output paths. `load_point_inputs` performs no Torch/model construction.
- [x] Re-run the same suite; expected all current cases pass. Commit the tested input/CLI contract.

### Task 2: genuine native geometry and atomic point ledger

**Files:** extend `scripts/paper/run_native_point_geometry.py`; extend its test file; genuine probe evidence under independent server output.

**Interfaces:** `run_geometry(args)->dict` calls the pinned native functions, writes one private exact native-array packet and one ordered JSONL row per selected hand point, then a closed result. `verify_geometry(output:Path)->dict` reopens packets/source/input fingerprints and independently recomputes per-point area-weighted ratios; it never constructs a learned pose model or grants a full role from probe records.

- [x] Write boundary tests before native glue: unknown policy rejects, absent required beta rejects, zero pose yields an explicit retained row, valid-pose nativeNone raises, changed source/input packet rejects. Add a real runtime test gated on licensed local assets rather than a mock claiming mesh execution.
- [x] Run new negative cases and observe actual failures; implement the smallest IO/dispatch layer. Preserve original right-MANO/left-mirror math and native K/camera transforms; save verts_world/verts_cam/faces/visible/area_weights/projections/zbuffer and their exact typed hashes privately.
- [x] Execute a detached real bilateral probe, then a fresh-process independent probe replay. Expected: one real right-MANO geometry layer each, two actual source mesh/occlusion calls, finite778vertices/1,538faces, exact native packets/scores/selection; no learned pose models/optimizer/CUDA and all owned consumers close. This proves an integration only, not full D19/Figure13.
- [x] Run relevant unit checks and fresh raw verification, commit source/probe evidence. No re-review of the closed IK/MANO asset mathematics.

### Task 3: one review, complete standard-grid geometry and public aggregate

**Files:** source/tests above, independent managed server deployment, new `results/native-point-geometry/` reports and existing status docs.

- [ ] One fresh review of this new unit only; resolve material findings once with targeted observed RED→GREEN tests and relevant checks. No repeated core-model suite or second review.
- [ ] Stage committed sources and run complete val/test roles using all1,576/4,306 points, explicit fallback policy and actual per-frame beta. Preserve zero-pose rows and all tracking/stale/marker flags; no score-based sample selection or hidden replacement of camera/mirror conventions.
- [ ] Run independent fresh-process native replay for the same complete selected point keys, verify all packet and aggregate counts, immutable sources/inputs and actual owned-consumer closure. Report0learned-pose models/updates and real geometry layer/call counts.
- [ ] Publish source, aggregate CSV/JSON/README and actual run evidence; retain private point/native arrays. Report invalid/visible/offscreen/behind-camera populations and fallback boundaries; do not publish licensed models or dataset payloads.
- [ ] Update D19 progress only to the scope actually proved. Dense1/4stride model predictions, paired gain/bin/SEM/Spearman, actual-video projection/overlays and whole-paper numerical equivalence remain explicit outstanding requirements.

## Preflight Decisions

The author-helper route is selected by the user's existing official-code priority. A rewritten faster rasterizer would change the requested method, and waiting for absent videos would prevent the available explicitly labeled fallback experiment. The local legacy metadata lacks inline beta fields; the admitted point payload carries the actual per-episode beta, so the original function receives it directly. Source/asset pointers only adapt IO and imports; native geometry operators remain unchanged. The builder uses the already approved inline execution method, preserves existing GPU reservations, and first establishes genuine bilateral behavior before full CPU work.
