# Explicit native DINO activation checkpointing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. This is a resource adaptation within the approved full-paper design; do not ask for renewed execution approval.

**Goal:** Admit an explicitly identified native timm activation-checkpointing run after original ViT-L batch360 failed the existing GPU memory reserve.

**Architecture:** Add an opt-in boolean to the existing native fit policy and CLI. Use only pinned timm `VisionTransformer.set_grad_checkpointing`; retain model tensors, native forward/loss/optimizer and original batch360. Bind the switch and the actual recomputation source files into native checkpoint identity. Frozen live S/B and failed L directories remain untouched.

**Tech Stack:** Existing Torch2.2.2/timm1.0.15/Lightning2.5.0.post0 and genuine DINO assets.

**Spec:** [approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), [original vision plan](2026-10-27-literal-vision.md).

## Global constraints

- Original-L failure remains failed: actual minimum device free45,088,768bytes, zero complete epochs, no automatic retry.
- Default remains checkpointing off. Explicit adaptation is not author-recipe execution or measured paper GPU hours.
- No batch splitting, accumulation, precision/loss/data/mask changes, test tuning, or agreement-tolerance changes. Original360/bf16/clip1/max150/patience30 remain.
- Restore with a different switch must reject; every fit guard checks the live native switch. Do not claim bitwise interrupted/uninterrupted trajectories.
- New GPU stage and output after CPU verification and one focused independent review; independent original-L receipt still required. All usual resource limits remain.

## Review focus

- A truthy string/integer must not silently enable recomputation.
- A checkpoint resumed with the opposite switch must reject before model construction.
- Native recomputation must preserve model tensors, all gradients and post-backward RNG in a real DINO forward/backward check.
- Actual timm recomputation/helper and Torch checkpoint sources must be bound, including the use_reentrant choice made by pinned timm.
- Metadata must never recast original failed-L or short-run GPU validation as a completed scientific result.

### Task 1: explicit native policy, real backward and resume verification

**Files:** modify `egoemg_repro/vision_literal_train.py` and public execution-policy disclosure in `egoemg_repro/vision_literal_eval.py`; create `tests/test_vision_activation_checkpointing.py`, extend the actual public-report test in `tests/test_vision_literal_eval.py`; update `docs/vision-literal.md`.

**Interfaces:** `vision_fit_policy(..., activation_checkpointing=False)` and `fit_vision(..., activation_checkpointing=False)`; CLI `--activation-checkpointing`; `_configure_activation_checkpointing(prepared, enabled)` uses the actual timm class. Existing identities remain available by their original frozen source commit.

- [ ] RED policy boolean/type/default/hash tests, real native DINO forward/backward tensor+gradient+RNG test, and real two-process native checkpointed fit/resume including opposite-switch rejection.
- [ ] Run focused tests and observe missing-API failures before editing production code.
- [ ] Implement the minimal opt-in setter, policy/source binding, batch guard and CLI; no alternate checkpoint implementation.
- [ ] GREEN focused tests plus existing native fit/eval checks; complete full/light integration against a frozen source snapshot. Inspect every result and run one focused independent review, fixing any Important findings by RED→GREEN.
- [ ] Commit and publish explicit usage, scope and actual verification evidence.

### Task 2: separate actual-L capacity and full experiment

**Files:** new private source stage, manifest/output and public engineering evidence; preserve original failed-L directory.

- [ ] Rehash unchanged complete inputs and pretrained L asset, bind the new reviewed source and frozen CPU evidence, select a freshly checked free A800.
- [ ] Execute original360/bf16 short fit, fresh native restore, complete3152val and exact prediction replay with unchanged tolerances.
- [ ] Only a successful actual capacity chain can precede a fresh full150/patience30 fit. A further resource failure remains failed; diagnose without silently changing the protocol.
- [ ] Final scientific completion requires natural full termination, validation-best selection, complete native test, independent replay and verified report. Keep full-paper goal active meanwhile.
