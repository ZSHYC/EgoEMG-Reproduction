# Native Qualitative Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. Existing whole-paper design and official-code-priority approval select this work; do not repeat preferences/approval or review closed model mathematics. One fresh review of the complete NEW qualitative integration, followed by one material fix pass.

**Goal:** Produce source-bound Figure7 qualitative comparisons with real RGB crops, all three frozen predictions, ground truth, actual errors and original UmeTrack geometry, keeping all required native producers and paper population limitations explicit.

**Architecture:** First reproduce the actual ordinary-stride source candidate selections and exact seeded per-group subsampling. A later inference/cache admission joins only identical original keys, then ranks by unrounded native delta, preserves source filters, generates original FK meshes and assembles visually reviewed condition examples. Existing completed dense predictions are reusable only for exactly matching keys and the same producer/input/error semantics; unavailable points require their actual inference, never nearest-neighbor or a replacement grid.

**Tech Stack:** Existing NumPy/Torch/Hydra/native EgoEMG f7cbd048, UmeTrack generic profile, existing CPU renderer/dependencies. No new training, environment installation or protected GPU rescheduling.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirement4 and Figure7 in docs/full-paper-status.md. The existing approved route resolves material choices; this plan implements it.

## Global Constraints

- Preserve Tables1–15/Figures1–13/all-model/all-dataset/annotation/preprocess scope. Qualitative examples cannot replace any missing full experiment, annotations, data or physical-camera equivalence.
- Pinned official find_fusion_wins.py defaults: splits user/gesture/both, hands left/right, seed42, max_samples_per_split500. Local default_rng(seed) is freshly reset for EACH split/hand; sort sampled native indices. Explicit0 selects all. Batch64/4workers belongs to later actual inference.
- Ordinary evaluation stride is12000 for RN18-S and7790 for ViT-S-S, windows respectively12000/7790. Preserve native episode-start grid, center-only split filtering, last-fitting-window behavior, no jitter/repeat and native_index. Do not substitute quarter strides3000/1947 or a strict full-window grid.
- RN18 ordinary grid is an exact subgrid of its dense grid;7790 is not divisible by4, so ViT-S exact intersections must be measured, not presumed. No interpolation, nearest center or key-order repair. Existing dense full pipeline stays CLOSED; do not rerun its predictions/review/tests merely to create a new figure.
- Original error uses FP32 unmasked22-angle abs.mean, radians then57.3 degrees. Preserve source tracking/not-stale/any-hand-marker filter and all rejected counts. Full-label/mask/crop populations and differences between original default evaluation and available producer-correct branch inputs must be audited, not silently changed.
- Original mesh path uses first20angles, original skin_fk_mesh/_mirror_hand_model, winding flip and per-pose median-span normalization0.09. Display view is shared within an example; do not claim physical RGB reprojection from an isolated mesh. Source helper ignores wrist display while numeric error includes22angles; disclose this.
- Native script has stale config aliases/path imports and attention patch; adapt only entry/provider/config paths to the matching fixed native runtime, preserve executable model output. StandaloneEMG8 and ViTfusion16 need their producer-correct inputs, no fake zero branches. RN18 actual compatible fusion/initializer still training; no permit from a prospective cohort or engineering.
- Hand crops are actual model input RGB, already canonicalized for left hand; do not add a second flip. EgoEMG legacy dataset CC-BY-NC4.0, UmeTrack third-party terms retained and credited; no licensed MANO weights/subject identities redistributed.
- Category labels self-occlusion/depth ambiguity/motion blur require actual input inspection and a saved selection rationale. Ranked illustrative wins are not unbiased aggregate evidence; do not copy paper example errors or retune models on test examples.

## Review Focus

1. Seed reset per split AND hand and stable native-index sort are easy to accidentally replace with one shared generator or dense-order sampling.
2. Native ordinary-grid points crossing center split boundaries must survive; exact index reference/bytes/provider/generation cannot be substituted by a resealed receipt.
3. Dense and ordinary grids, different model/input layouts and branch tensor identities can look compatible while referring to different predictions; prove every reused key and actual producer.
4. Original first20 geometry and22-angle numeric error are different views; prevent silent angle clamping, new wrist geometry, different per-model camera/view alignment or false physical overlay claims.
5. A saved render must remain bound to actual RGB/target/three predictions/mesh/error/actor closure through independent readback and publication; early report or changed raw files must not count complete.

## Task 1: ordinary-grid candidate selection and exact cached-key intersection

**Files:** create egoemg_repro/qualitative_data.py, tests/test_qualitative_data.py; later private owned double-admission launcher and public results/native-qualitative/ evidence.
**Interfaces:** qualitative_protocol(upstream,config,*,recipe,max_samples_per_split=500,seed=42)->dict; admit_qualitative_selection(data,*,upstream,config,recipe,max_samples_per_split=500,seed=42,engineering=False)->registered QualitativeSelection with copied readonly .points, copied .identity, .check(). Original index-only native view is reused without modifying closed data/model code. Actual full source is authenticated against fixed complete index proof b57de34.../manifest+metadata+split full SHA; engineering remains explicitly engineering.

- [x] Write missing-module RED tests for both normal strides, source defaults, original AST sampling oracle including cap>n/cap0/empty/reset-per-hand, exact native source membership/order/tails, crossing windows, RNG/bytes unchanged, registered object/copies/generation/schema/symlink/reference rejection. Run only this new suite.
- [x] Implement minimal byte-bound registered data component; compare tests to actual original source functions. No learned model/optimizer/geometry/CUDA permit. Watch new tests GREEN without skips.
- [x] Two fresh actual admissions for each of default500 and all0 for both recipes. Save exact private typed arrays/identities, compare current old dense point payloads by full hand/episode/start/end/center keys and report real intersection/missing counts. Close all actual actors and fresh postexit reader; publish data-only evidence.

## Task 2: actual frozen predictions, original ranking and FK visual output

- [ ] Admit actual fully completed producer triplets/default standard selections, exact raw input RGB/GT/masks/normalization, all target/model states. Reuse a dense packet only for proven identical keys and producer semantics. Produce every remaining required standard-key prediction using original CPU-supported forwarding, complete replay and no new updates.
- [ ] Source-native tracked/not-stale/marker filter, errors and stable descending delta ranking. Keep pre/post populations and raw data. Original FK helper/20finger geometry, actual numeric22errors and all samples/conditions/selection policy explicitly joined. Tests first; preserve cold/fresh actual packets and no-zero fallback.
- [ ] Visually inspect saved candidate crops and mesh renders; record deterministic rank and condition rationale for three examples. Render standalone actual data figure and table with source/model/license/selection attribution and actual unrounded statistics.

## Task 3: lifecycle, independent reconstruction, review and publication

- [ ] Integrate owned producer/fresh reader/report closure and complete current raw/source/render lineage. New tests only; one final complete NEW-unit review and one material fix phase, no second review or old core math/test repetitions.
- [ ] Actual complete source-specific Figure7 run and independent array/geometry/error/render reconstruction; all exact owners closed, source labels/models unchanged. Publish actual supported panels/metadata/limitations and update Figure7 row only to proved scope. Native RN18 and original-cohort equivalence remain pending until actually proved; no engineering substitute.
