# Native EMGFormer Training Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with one fresh review of this new component. This executes the approved full-paper design without another approval round.

**Goal:** Run and independently read original native S/M/L training with full trajectories, center-only validation, native best selection and complete checkpoint recovery.

**Architecture:** New `emgformer_train.py` consumes closed runtime1376203, reader2c93b7a and cohort415deb9. Reuse unchanged generic state/journal/callback helpers; new provenance, point-validation witness and policy cannot reuse the different150epoch initializer protocol. Preserve all native callbacks and original six-worker loader; CPU engineering is explicit.

**Tech Stack:** Existing pinned native Torch2.2.2/Lightning2.5.0.post0, Hydra/OmegaConf, original optimizer/model/loader implementations. No dependency installs or model mathematics changes.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`; all models/full training/selection/frozen evaluation remain required. This unit supplies the fit and durable selected-checkpoint interface; GPU qualification and complete test/report chain follow.

## Global Constraints

- Exact fixed upstream f7cbd048b7092139852270491d3859cd472e0311 recipes S/M/L:0.02 effective tip loss, AdamW5e-5, cosine200/eta1e-5, seed42 before native factory, BF16, B500,6workers/prefetch1/persistent true/pin false/global RNG; max200 and original ModelSummary/LRMonitor/ModelCheckpoint/EarlyStopping callbacks (val_mae/min, patience200).
- Full native training requires registered complete62470train/1052val/2792test legacy cohort and independently frozen complete numeric receipt; raw inputs are never edited. Explicit legacy normalization/fields and population limitations persist.
- Native train target22x12000/time mask12000; native val target22x1/mask1 and genuine original Lightning prediction center. Never replace point supervision with full trajectories or silently realign the native context.
- Engineering modes max2, CPUFP32/B2/0workers or CUDABF16/B500/6workers. Balanced hand train subset4CPU/1000CUDA, full validation; no science/full-fit claim.
- Full mode preserves max200, original sanity2 and all data, callbacks, sampler and workers. Engineering may pause at a complete epoch only. Native early-stop patience200 cannot shorten a finite max200 history; require complete actual200 to claim full termination.
- Native model/optimizer/scheduler/callback/loop/parent-RNG checkpoint restoration and validation replay are verified. Persistent-worker stochastic continuation is not guaranteed bitwise; never change native worker/sampler semantics to manufacture equality.
- Owned output/lock/journal/history/best/last are bound to source/config/data/model identities. Completed runs reject retraining; preserve interrupted durable history. Always shut down owned persistent loaders on exit/error.
- GPU qualification/full training/frozen test/replay/report remain downstream; no scientific count increases from CPU integration. Existing jobs/GPUs4/5 untouched, no foreign deletion.

## Review Focus

- Role-specific native tensor shape and point forward/mask/hand/index provenance must reject substitutions and duplicated or incomplete epochs.
- Balanced engineering sample IDs are original corpus IDs, not rewritten0..N; both hands must contribute and full val order must remain native.
- EarlyStopping and all native callbacks/defaults must survive fit/resume/read; callback/journal/selected checkpoint cannot be self-consistently relabeled as another scope or protocol.
- Six-worker loader recreation adds native RNG effects; exact checkpoint restoration must be distinguished from bitwise subsequent stochastic trajectory.
- Exceptions and recovery must retain source files/history and close own workers/hooks; finished checkpoints can be accepted without further optimizer updates.

## Task 1: native policy, fit witness and durable reader

**Files:** create `egoemg_repro/emgformer_train.py`, `tests/test_emgformer_train.py`.

**Interfaces:** `emgformer_fit_policy(upstream,config,*,scope='full')`, `emgformer_stage_complete(policy,records)`, `fit_emgformer(cohort,output,*,scope='full',resume=False,stop_after_epoch=None,cohort_receipt=None)`, `read_emgformer_fit(output,*,require_full=True)`; CLI `fit --size small|middle|large --data --upstream --output --scope`.

- [x] RED: all3native policies200/500/6/BF16/original callbacks/loss/optimizer and explicit engineering differences; invalid scopes/fullshorthistory rejection; balanced original IDs.
- [x] RED: actual CPU small prefix1epoch then fresh resume2epoch; full point validation, six saved state domains restored, native best/min score and original200 scheduler/ES callback; independently read source/coverage/last/best/history tampering; incomplete/engineering cannot claim full; completed run refuses retraining.
- [x] Implement original factories/callbacks/trainer with exact typed cohort provenance, point witness, complete epoch journals and native raw checkpoint reader/recovery. Full original sampler/workers remain unchanged; own worker cleanup is production lifecycle.
- [x] Run the complete new train component tests to actual terminal success, record native point integration/updates and commit. Do not repeat closed95/58/59suites or old model math.

## Task 2: source-bound CPU integration proof

- [x] Freeze committed current source and a temporary synthetic-memmap CPU driver. Actual prefix/resume in new processes, separate checkpoint-only validation; preserve all outputs/process identities and zero-CUDA scope. No extra model training beyond the required fit integration.
- [x] Publish actual native point-forward/balanced hand coverage, full checkpoint restoration/selection/loop state and source hashes. GPU/full200/scientific metrics remain pending.

## Task 3: one fresh review and publication

- [x] One fresh-context review of this component only; one material fix pass with observed RED→GREEN/relevant suite if needed, no re-review/unchanged model repetition.
- [x] Retain pre-fix immutable evidence; bound-source changes require fresh current-source integration proof. Commit/push/independent GitHub ref verification and update goal recovery ledger.
