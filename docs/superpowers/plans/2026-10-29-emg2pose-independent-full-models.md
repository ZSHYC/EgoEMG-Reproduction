# Independent full EMG2Pose M/L scheduling

**Goal:** Run the already-required M/L experiments while the existing S experiment continues, preserving the [approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), original source, full populations and scientific protocol. This is an operational execution choice, not a new model, data protocol or claim that parallel scheduling was specifically requested by the user.

**Existing behavior:** The frozen queue has a fixed S→M→L order and no adoption/skip interface. Each per-model controller refuses an existing output before source admission or GPU/model work, and atomically creates its own directory. S receives its independent full verification and `small-verified.json` before the queue attempts M. Queue failure cleanup removes only its own aggregate completion, not S or another model's artifacts.

Two valid operational approaches were considered: keep the original serial queue, which preserves its original three-model aggregate but makes M/L wait for all 250 S epochs; or use independent original controllers on additional eligible A800s, with separate per-GPU qualification and explicit aggregation of three independently verified models. The latter is the executor's choice within the existing authorization for all three full experiments and available server resources. No source/validation gate is relaxed.

## Required invariants

- Leave the original S queue, source/config, parents, data and output untouched. Do not stop, adopt or restart its live training.
- Keep production modules byte-identical to the qualified `d8800033aac46309e425e76279978b2b61315b82` capsule and official `f7cbd048`; only isolated operational wrappers/configuration differ.
- Same-model training qualifications are explicitly accelerator-UUID independent in `fit_source`: actual runtime, native recipe, batch320/12workers/BF16 and hardware name still must match. Reuse those closed genuine receipts without relabeling them. New per-UUID FP32/320+1/12worker val/test native/replay and complete engineering controller qualification must actually pass on the new cards.
- Preserve the 25,253-session full source, 250 full epochs, B320/12workers/BF16, original optimizer/scheduler, no early stopping, and both `official_literal` and `metadata_indices` evaluation populations. No full experiment deadline, reduced cohort, batch change or replay substitution.
- New full controllers use the old queue's reserved `middle` / `large` output paths as their actual outputs, with real requests and results. There are no placeholder success artifacts. Atomic directory creation and the original no-overwrite gate allow one claimant only. A losing controller must fail before training, retain its failure and never automatically retry under another output path.
- If the independent M claimant wins, the old serial aggregate will later refuse that already-owned M output after fully verifying S. Preserve the old aggregate's actual exit/failure; never rewrite it as complete. The S per-model driver/report remain intact. An eventual new aggregate may claim all three only after independently checking each full per-model completion, source identity and owner termination.
- If the old queue claims a model first, do not launch a duplicate; retain that original model as the candidate and verify its actual identity/results. No assumed adoption or synthetic completion.
- Use actual64GiB GPU prelaunch / 8GiB ongoing, 256GiB host and 15GiB disk reserve gates unchanged. Retain WiLoR's 23GiB pre-save margin after its remaining native checkpoints plus a conservative3GiB for M/L qualification/full outputs. Full inference streams sufficient statistics/digests rather than storing full prediction trajectories. Existing/foreign consumers and all raw data remain.

## Execution

- [x] Inspect actual queue/controller ordering, no-overwrite and cleanup; measure real old M/L raw checkpoints (~80MB/~196MB), current GPU/storage/host state and source bindings. No bulk input scan or new physical sharing was needed.
- [x] Finish actual new-GPU engineering chains on GPU4/L and GPU5/M, independently reopen their receipts, verify all owners closed and original source unchanged. The first metadata-only staging error and GPU3 prelaunch reserve refusal remain recorded; neither launched a model.
- [x] Verify the existing-output safety boundary on disposable ordinary files, with no Torch, GPU, source admission or scientific artifacts. Bind checks to the unchanged active controller/queue code.
- [x] Stage and launch two isolated original full controllers only after those actual qualifications and fresh resource checks; preserve current S. Report workflow launch separately from actual GPU fit/epoch completion.
- [ ] Independently collect each genuinely completed full model and its final evaluation/replay/report. Publish the eventual three-model scientific aggregate only after all three pass; keep the superseded serial aggregate's actual status visible.

The full paper objective remains17 settings/Tables1–15/Figures1–13 and the annotation/preprocessing/statistics/occlusion/qualitative work. Qualifying additional cards or launching two controllers does not increment completed scientific settings.

One fresh review of this NEW scheduling unit: 0Critical/0Important/1Minor (checklist bookkeeping, accepted). Source/model internals were not re-reviewed. The actual terminal qualification, request-binding and fresh-resource gates remain mandatory. Public staged worker: `scripts/run_emg2pose_independent_full.py`, byte-identical to the reviewed private source.

Actual execution: both new per-device qualification chains exited0 (M22:00:40UTC, L22:08:23UTC), all120 recorded owners closed; fresh CPU original-validator reopening exited0 in435.64s. One local parser failed on informational stdout after that remote success; normalization used the retained log without repeating any reader or GPU work. The two146-member full workflows launched22:17:48UTC, their8 recorded owners were live at22:18:28UTC with original full-source reads progressing. At that observation neither real model directory was claimed nor any full GPU fit verified. S remains on its original handles. See the public closed qualification and full-launch JSON receipts.
