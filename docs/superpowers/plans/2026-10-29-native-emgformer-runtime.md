# Native EgoEMG EMGFormer S/M/L Runtime Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. This is a component plan executing the approved full-paper design, not a new model scope or permission request.

**Goal:** Bind and execute the exact official EgoEMG S/M/L factories and effective0.02 trajectory objective, with strict state reload and original optimizer validation. Construction evidence is not a full scientific training result.

**Architecture:** Add isolated `emgformer_literal.py` and `emgformer_runtime.py`; reuse existing source/typed-tensor helpers without changing initializer, model, data or live training code. Compose the three original named YAMLs exactly and use native setup_runtime/factory/forward/loss/AdamW/cosine. The completed cotrain initializer has a different0.01/150epoch/lr5e-4 recipe and cannot be substituted.

**Tech Stack:** Existing Python/NumPy/Hydra/OmegaConf/Torch2.2.2/Lightning2.5.0.post0, no installs/downloads; local CPU engineering, later original server GPU qualification.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`, especially fixed official source, all author models and its recorded0.02 loss ruling; existing `results/official-loss-audit.json` and `results/dataset-statistics/emgformer-architecture.json`.

## Global Constraints

- Fixed original `f7cbd048b7092139852270491d3859cd472e0311`, clean tree and unchanged native math. Do not edit existing live capsules, labels, legacy checkpoints or initializer consumers.
- Original experiments `emgformer/egoemg_emgformer_{small,middle,large}`.8target-hand channels/22angles/WL12000; original3/6/8 decoder layers,4/8/12heads,256/256/384model dims,3,460,134/6,617,382/16,320,806parameters.
- Original B500/bf16/max200/lr5e-5/cosine200 eta_min1e-5/clip1 and actual callbacks remain declared. Both fingertip aliases contribute0.01 each, effective0.02. Only the private source-binding field may differ from exact resolved Hydra.
- Native normalization reads actual per-hand8vectors from declared alias stats, not estimated train statistics or16-slot fusion normalization. Record physical legacy-field alias separately in the later data component.
- Metadata/runtime objects never grant a dataset/fit permit. Old41episode versus declared unified ShowEE/Incre corpus and source-label differences remain explicit; no old scores retagged.
- User already selected official-code reproduction and approved the full-paper design. Separate native wrappers are the executor's implementation choice; reusing0.01cotrain or modifying legacy continuation would violate that approved scope.

## Review Focus

- Wrong-size or cross-recipe learned-state packet must reject before factory; test recorded identity/config/source/schema tampering.
- Numeric alias accumulation must be exercised in actual native loss, not merely copied YAML; test both0.01terms against captured same-forward native metrics.
- Original parameter order/coverage and cosine schedule must survive real optimizer construction; test missing/duplicate parameters and changed lr/eta/Tmax.
- Normalization must resolve genuine per-hand float32 arrays; test real original `_load_norm_stats` and8-versus16layout.
- Engineering CPU construction/update/reload must not be published as GPU qualification, complete training, current merged data, or PDF precision. Independent CPU proof and public projection retain false eligibility flags.

## Task 1: exact configuration and normalization contract

**Files:** Create `egoemg_repro/emgformer_literal.py`; test `tests/test_emgformer_literal.py`.

**Interfaces:** `compose_emgformer(upstream,size='small')`; `emgformer_protocol(upstream,config)`; `emgformer_normalization(upstream,config)`. Sizes are small/middle/large; protocol records raw and effective losses, native source hashes and exact trainer/loader declarations. No factory/data read here, except explicit norm function.

- [x] RED: exact three resolved configs versus direct original Hydra, only private binding; batch500/max200/lr5e-5/eta1e-5/effective0.02 and native decoder sizes; mutation and unsupported-size refusal; native8channel per-hand normalization.
- [x] Run new test file in existing local original runtime. Expected: missing component failures before implementation.
- [x] Implement strict binding/alias aggregation and normalization via original loader. Expected: all Task1 tests pass.

## Task 2: unchanged factory, complete state and optimizer

**Files:** Create `egoemg_repro/emgformer_runtime.py`; test `tests/test_emgformer_runtime.py`.

**Interfaces:** `build_emgformer_runtime(upstream,config,*,state=None,expected_initialization=None)` returns PreparedEmgformer with module/identity/check()/mode/loaded_state_sha256. `emgformer_state_sha256(state)` reuses existing full typed-state digest. `emgformer_optimizer_identity(prepared,optimizer,scheduler)` validates every native parameter once and original AdamW/cosine200.

- [x] RED: three actual native models have expected counts and full `(B,22,11490)` aligned trajectories; setup before factory and deterministic seed42; strict same-output reload; actual train-mode loss/backward/update uses both aliases; invalid lineage/schema/config/state rejects before factory; optimizer missing/duplicate/lr/scheduler/eta refusal.
- [x] Implement minimal source/runtime/version/identity-bound wrapper. Keep all scientific/data/full-fit flags false.
- [x] Run Task1+2 tests. Expected: new relevant tests pass, no CUDA or asset downloads.

## Task 3: independent CPU engineering and publication

**Files:** isolated private CPU producer/replay; publish `results/native-emgformer-runtime-cpu-20261029.json` and docs/status only after actual proof; one fresh review of this NEW component.

- [x] For each size, execute actual source factory, train-mode trajectory loss/backward and one original AdamW/cosine update on fixed synthetic full-length input, freeze complete learned state privately. Expected: finite nonzero gradients/changed state, source config0.02, no data/GPU/full-budget claim.
- [x] Fresh process strictly reloads all three and independently computes same eval predictions/typed-state digest. Expected: exact outputs and source hashes, actual owners closed, no extra optimization in replay.
- [x] One fresh new-component review, one fix pass for material findings with targeted RED→GREEN; no repeated old model internals review.
- [ ] Verify public evidence/source bindings and scope, commit/push/ref verification. Expected: this component ready; full0.02 data/cohort/200epoch/selected-best test/report and all remaining paper requirements remain pending.

## Downstream requirements retained

This plan does not shrink the full goal. Next components must admit the complete available legacy numerical footprints with independent train-only validation and source normalization, preserve native data augmentation/RNG/target masks; implement full200epoch original callbacks/complete state/selected-best output; actually qualify original batch500 on eligible GPU without disturbing active jobs; then perform original native and clearly declared matched-center tests, independent replay, per-hand/split/user/joint/gesture results and final reports. Unified missing sources and PDF label equivalence must remain distinguishable from actual available-data results. No model completion count increments from this CPU component.
