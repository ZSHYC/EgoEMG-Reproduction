# FreiHAND Official Source Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. This is source-data qualification within the approved whole-paper, official-code-priority route; do not repeat design approval or the closed archive/numeric audits.

**Goal:** Reconstruct all32,560 training and3,960 evaluation annotations with the pinned original FreiHAND geometry and compare a separate CPU implementation against that reference and the received GT.

**Architecture:** Keep original source inputs private and immutable. Execute the original five-file MANO setup on private copies and compile the pinned, unchanged geometry function/class bodies with an explicit Python3/import/file-I/O bootstrap. Run the original Chumpy reference over every row and a separate float64 SMPL-X/LBS backend over the same rows; retain full typed vertices/joints and model-free error reductions.

**Tech Stack:** Existing Python3.11 runtime, NumPy1.26.4, original Chumpy0.70, existing OpenCV/SMPL-X/Torch CPU providers, licensed MANO v1.2 files; no new package installation or GPU allocation.

**Spec:** Active user whole-paper objective and official-code-priority instruction; `docs/full-paper-status.md` D07; `results/freihand-full-source-intake-20261102.json`; `lmb-freiburg/freihand@5ea4ab9763fea0eec988a52bfa563333cb16523f` model.py, mano_utils.py, setup_mano.py and view_samples.py.

## Global Constraints

- Consume the exact completed official v2 inputs, original ordering and roles:32,560 training rows and3,960 evaluation rows. Four RGB versions do not create four independent parameter rows. Do not rewrite source JSON or recast evaluation as training.
- Existing MANO_RIGHT SHA256 is45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767. All five licensed files must match the MD5 values in the pinned official setup before constructing a model; raw assets and licensed source stay private.
- Use official `use_mean_pca=False,use_mean_pose=True`, corresponding to `use_pca=False,flat_hand_mean=False`. Keep pose48/shape10/UVroot2/weak-perspective-scale1 and recover translation through the posed reference joint9. The separate reference-bone scale JSON is not the61-vector scale.
- Original keypoints regress16 joints from the posed mesh and add the official tips744/320/443/555/672 in the original21-joint order. Do not substitute another library's returned joint array.
- The reference executes original function bodies. Bootstrap may supply a decoded trusted pickle dictionary, cPickle alias, existing NumPy/inspect compatibility aliases and isolated module paths; omit unused rendering imports and the standalone Python2 demo block. Record every adaptation and the loader's original NumPy seed1 effect inside its own process. Do not patch shared package files or change geometry arithmetic.
- CPU only. Preserve a256GiB host-memory reserve plus16GiB workspace admission and200GiB disk reserve. Existing training, data receivers, GPU queues, original MANO assets and the critical SSH connection remain protected.
- Both backends use float64 output. Backend agreement has a fixed engineering maximum absolute coordinate difference of1e-7meters, declared before real execution; do not loosen it to make a result pass. GT reconstruction errors are measured, not judged against an invented paper threshold.
- No optimizer, label replacement, markers2mano substitute, new pose-model score or whole-paper completion follows from this audit.

## Review Focus

1. Python2 imports/text-mode pickle reads must not silently replace geometry code. Source/asset rejection tests must fail before model construction; the real qualifier binds the exact original setup outputs.
2. Another library's tips and joint API can differ. Test the original21-point assembly on distinguishable vertices/regressed joints and compare every actual output coordinate against the original backend.
3. The final61-vector entries are UV and scale. Test nonzero principal points, unequal focal lengths, the joint9 translation subtraction and rejection of malformed/nonpositive scale inputs.
4. The original loader seeds NumPy and mutates its supplied dictionary. Decode a private fresh model dictionary and record process-local effects; preserve all original asset/source generations.
5. A two-row smoke run or preallocated output is not full coverage. Full receipts require all36,520 ordered source keys, both backends, all typed outputs, source generations and actual producer closure; the independent reader constructs no model.

## Task1: source-bound original bootstrap and real qualification

**Files:** private operational sources in `tmp/freihand-geometry-qualification-20261102/`; worker `tmp/freihand-official-geometry-worker-20261102.py`; focused checks `tmp/test_freihand_official_geometry_20261102.py`. No licensed code is committed.

**Interfaces:** a private request binds six pinned public sources, five original licensed files, existing package providers and all original JSON inputs. `build_original(request)` returns the source-native pose function and its private model loader; `build_independent(request)` returns a CPU float64 backend. Both consume a source row `(mano[1,61],K[3,3])` and return `(xyz[21,3],verts[778,3])` without modifying inputs.

- [x] Freeze six public source identities and verify all five licensed MD5 values. Actual five matches are retained in `actual-existing-five-input-identity.json`; no pickle/model execution occurred at that step.
- [x] Write and observe failing new checks for source/asset mismatch before construction, tip/order assembly, original camera/root arithmetic and invalid row/scale rejection. Implement only the needed orchestration/source adapters and run those checks to GREEN.
- [x] Run the unchanged original setup on private copies, bind its patched loader and original LBS operators, and verify existing package availability without installation. Preserve original models/source bytes.
- [x] Run one actual training and one actual evaluation row with both backends. Retain geometry/error arrays, source/provider identities, actual constructors and resource/exit evidence. Unexpected backend disagreement triggers diagnosis, not a new tolerance or another joint convention.
- [x] Verify both original qualification owners closed and all inputs unchanged before full execution. Qualification remains explicitly two-row engineering evidence.

## Task2: complete reference/independent passes and publication

**Files:** same frozen worker; isolated server output containing full typed arrays and ordered role/index keys; public aggregate `results/freihand-official-geometry-20261102.{json,md}` only after actual full verification.

**Interfaces:** full mode processes exactly `(training,0..32559)` then `(evaluation,0..3959)`. Save each backend's float64[36520,778,3] vertices and[36520,21,3] joints plus full per-row GT/backend errors. A fresh model-free reader consumes only the closed immutable input/output identity packet.

- [x] Execute every original source row with the original Chumpy backend and every same row with the independent CPU backend. Persist completed-row journals separately from file lengths; preserve partial outputs on failure.
- [x] Calculate per-row mean Euclidean vertex/joint error to GT in meters and millimeters, and full backend coordinate differences. Retain full input membership and raw per-row reductions; do not drop error cases or tune pose means, scale, mirroring, root or alignment.
- [x] After all producing owners exit, independently rehash complete source/assets/outputs, reopen all typed arrays and recompute every aggregate/coverage count with0 model constructions and0 optimization.
- [x] Perform one final review of this new complete geometry component after the actual full positive reference/independent execution and reader. Fix material findings in one phase; do not reopen old model/IK/annotation mathematics reviews.
- [ ] Publish scoped aggregate evidence and update D07/FreiHAND status. Even a successful audit leaves author exact source selection/mixture, missing annotator assets, remaining data sources and whole-paper reproduction pending.

## Execution ruling

This plan executes the already-authorized official-source route inline. The two real qualification rows protect the full run; they do not redefine its36,520-row target. Native IK's beta0/flat-mean/global-orientation policy is a separate original workflow and is not reused here.
