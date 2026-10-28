# Native ViT-S Fusion Construction Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task; one final whole-plan review, no per-task agents. This implements the existing approved full-paper design, not a new user choice.

**Goal:** Construct the unchanged official sixteen-slot ViT-S fusion model with verified local branch weights, expose exact loading decisions, and restore the full model without its initializers.

**Architecture:** A source-bound configuration contract wraps the canonical training recipe underlying the released evaluation entry. A separate CPU runtime calls original setup/factory/loaders, verifies complete vision transfer and exact native partial EMG transfer, and supports strict self-contained restore. This is the construction subsystem; coupled data, complete training/GPU qualification and scientific reports remain explicit subsequent work toward the full-paper goal.

**Tech Stack:** Existing pinned Hydra/OmegaConf, PyTorch/Lightning/timm, original EgoEMG factory and loaders; existing state hashing helpers.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements2/4/7, source/initialization and resource rules; [actual five-recipe audit](../../../results/fusion-recipe-audit-20261028.md).

## Global Constraints

- Upstream remains clean at f7cbd048b7092139852270491d3859cd472e0311. Existing training source, checkpoints, downloaders and live server namespaces remain unchanged.
- Canonical construction recipe is `fusion/fusion_vits_s_center_16ch_wl7790`:16 slots with channel_interpolate=false, positions[10,12,0,1,2,4,5,6], WL7790, stride400/eval7790, repeat2, batch200 for every native role; declared val_batch_size100 is unused by original factory.
- Original bf16/high, AdamW lr1e-4/weight_decay1e-2, cosine150, clip1, effective fingertip0.02, native near-zero-normal residual initialization and source callbacks are not replaced.
- Original branch loaders remain authoritative: complete backbone/head transfer from local trained-vision-format state; EMG allowed-prefix exact-name/shape matches only, with skipped tensors recorded. No8→16 weight expansion or strict-all-EMG claim. Cotrain-S0.01 is not redefined as0.02.
- No ambient pretrained download is necessary: fresh construction requires two existing files with explicit SHA256; restore uses complete tensor state and saved identity only. Do not claim pretrained assets or training correctness from random engineering fixtures.
- Outputs of this subsystem always declare scientific_eligible=false, production_ready=false, full_paper_complete=false. No data/GPU/full-fit or accuracy qualification is inferred.

## Review Focus

- Same-shaped but incomplete vision backbone/head must fail rather than retain random missing tensors; Task2 checks all keys and byte identity.
- Native strict=False EMG shape skips must be visible and unchanged, including8→16 first convolution; Task2 verifies all matched/skipped keys against original behavior.
- Missing or modified initialization files must reject before fallback downloads; Task2 checks hashes and file stability and blocks network in real factory tests.
- Complete-state restore must work after branch files are absent and reject wrong keys/shapes/dtypes/nonfinite tensors/source identity; Task2 checks independent factory reload and prediction identity.
- Environment/private root paths and configuration edits must not leak into safe protocol metadata or become silent overrides; Task1 checks sanitized metadata and full typed config binding.

### Task 1: Exact fusion configuration and native loader contract

**Files:** Create `egoemg_repro/fusion_literal.py`, `tests/test_fusion_literal.py`.

**Interfaces:** `compose_fusion(upstream)` returns exact bound Hydra config. `_validated(upstream,config)` returns(root,resolved_original_dict). `fusion_protocol(upstream,config)` returns sealed safe metadata with original module, optimizer, callbacks, effective loss and executed loader semantics; CONTEXT_KEY=`_literal_fusion`.

- [ ] Write cases for exact native Hydra equivalence;16-slot/WL/stride/repeat/loss/optimizer/callback values; actual native datamodule batch/repeat witness; private environment path redaction; edits to model/layout/norm/initializers/trainer/types and binding; missing/unapproved upstream. Run RED before module exists.
- [ ] Implement exact source composition/binding using existing approved source/hash activation helpers. Redact only declared path-valued module initializers in public metadata; never alter actual config. Include full source hashes and distinguish declared merged corpus from unavailable data admission.
- [ ] Run new contract and existing vision/classic metadata checks with native runtime. Expected: all pass, no CUDA/model/data construction from contract. Run light suite, commit, task-done.

### Task 2: Original construction, branch transfer and self-contained restore

**Files:** Create `egoemg_repro/fusion_literal_runtime.py`, `tests/test_fusion_literal_runtime.py`, `docs/fusion-literal.md`; publish only actual sanitized CPU evidence in `results/` after execution.

**Interfaces:** `build_fusion_runtime(upstream,config,*,vision_checkpoint=None,vision_sha256=None,emg_checkpoint=None,emg_sha256=None,state=None,expected_initialization=None)` returns `PreparedFusion` with module/identity/mode/loaded_state_sha256/check(). Fresh mode requires both regular local files and hashes; restore requires state+expected_initialization exclusively. Identity records protocol/source/runtime/schema plus all matched/skipped initializer tensor names and byte hashes. Learned weights may change; check() guards config/source/assets/identity.

- [ ] Use a real native pretrained DINO-S-format engineering fixture from existing genuine-cache fixture and an explicitly random original EMG-S fixture. Record their engineering scope. Add failing tests for exact vision backbone/head transfer, complete native EMG prefix/shape loading, residual initialization, original forward/loss/optimizer, offline factory, initializer damage and strict fresh-process restore; no model substitutes.
- [ ] Run missing-module RED. Implement source-bound two-mode native factory wrapper. Hash files with bounded reads; check stamps before/after loading. Reject partial vision transfer, nonfinite/typed-invalid tensors and empty EMG featurizer/decoder transfer. Keep original EMG mismatch skips and compare every matched tensor exactly after construction. Clear initializer paths only for explicit restore and temporarily disable pretrained downloading for its native random factory before strict full-state load.
- [ ] Verify exact native near-zero residual rule rather than forcing exact zero; native center prediction shape1×22×1, real finite CPU forward/loss/backward and original optimizer component membership. Report unqualified status and all initializer-source limitations.
- [ ] Run relevant native/contract/runtime regressions and full light suite; unchanged frozen-source SHA and no CUDA. Commit, task-done, then one fresh whole-plan review. Any important fix uses RED→GREEN without re-review.

## Following required fusion work

The full-paper requirement remains the entire ViT-S+S experiment: original joint EMG/RGB point reads and augmentation; jitter-safe independent train/val/test cohort with full identity; original150epoch/native batch200/selection/optimizer and complete-state recovery; actual GPU capacity/recovery qualification; full training, frozen test and independent reports. Current merged-source and author-checkpoint availability limits remain explicit. This construction phase never counts as another completed model setting.

## Approaches already resolved by the approved request

Using the old RN18 wrapper or assuming all eight-channel S weights fit a16-channel model would change the requested official algorithm. Reimplementing a new fusion network would also lose source fidelity. The user selected official code and approved full-paper execution; the canonical released recipe plus original loading behavior is the implementation choice within that authorization. No new integration or data-version preference is required.
