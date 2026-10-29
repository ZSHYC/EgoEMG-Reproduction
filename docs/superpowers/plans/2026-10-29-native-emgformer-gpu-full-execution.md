# Native EMGFormer GPU Qualification and Full Execution Plan

> **For agentic workers:** Use superpowers:executing-plans inline. One fresh review covers only this operational unit; closed model/training/data/evaluation components are not reopened. Existing full-paper execution authorization applies.

**Goal:** Run the original Small EMGFormer GPU qualification, then fresh full200 training and selected-point evaluation on an actually idle A800, without changing the official recipe.

**Architecture:** A small standard-library resource helper chooses from explicitly unreserved devices. Immutable operational workers call the already closed training44b72b6, cohort415deb9 and evaluation61a7170 interfaces under the existing owned-process controller. Qualification and full fits use distinct output directories and identities.

**Tech Stack:** Existing canonical Python3.11/Torch2.2.2/Lightning2.5.0.post0, native B500/six workers/BF16; no installs or math changes.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`.

## Global Constraints

- Pinned official f7cbd048b7092139852270491d3859cd472e0311 and effective fingertip0.02. Preserve all closed runtime/data/cohort/train/eval source bytes and native callback/RNG behavior.
- Registered complete legacy Small cohort:62470 train/1052 val/2792 test unique hand samples; original12000 input,22 full training angles,22x1 point validation/test, no visual repeat/visibility substitution.
- Qualification uses existing `cuda_engineering`:B500/six workers/BF16,1000 balanced original training IDs, one-epoch prefix then fresh resume to two epochs/four total actual updates. Original full validation and test, native/fresh replay/freeze/report/verify. This is never scientific completion.
- Full fit starts independently from native initialization:200 epochs/B500/six workers/BF16/25000 updates, no qualifier weights. Selected checkpoint and seven-stage point evaluation follow the unchanged APIs. A point report does not substitute for the remaining paper trajectory/other-model requirements.
- Only GPU1/GPU3 UUIDs are initially candidates. GPU0/2/4/5/6/7 have existing jobs/reservations; a gap between an existing job's stages is not permission to take its device.
- Admission requires zero compute applications and at least65536MiB free. A foreign application blocks a device even when free memory exceeds64GiB. Never stop foreign consumers or silently lower the batch/precision/reserve.
- Controller imports no Torch; retain host256GiB/output15GiB/GPU8GiB during execution, source/input stat guards, bounded logs and PID/start ownership. At most24hours resource wait, then fail durably; no implicit retry or success inference from timeout.
- Passive observers may record original GPU forward/optimizer execution; they must return native tensors unchanged. Actual NVML worker PID/start/UUID binding and actual CUDA input/module/output are required before claiming GPU execution.
- Preserve failed attempts and old receipts. Publish actual launch/progress separately from scientific completion. Whole goal stays active until every requirement is proven.

## Review Focus

- A mostly free device that still has another process must not be selected; reserved devices remain excluded even if physically idle.
- Partial, malformed, duplicated or stale resource inventory must fail closed, including bools passed as integers and application UUIDs absent from the GPU snapshot.
- No GPU phase runs before the one fresh operational review is resolved and source-bound CPU preflight passes. No self-claimed engineering receipt can directly grant full training.
- Qualification prefix/resume and full fit must not share checkpoints or budgets; the full worker reopens original qualified raw fit/pass/profile/report before fresh full fit.
- Source/input changes, resource pressure, foreign arrivals, failed phases or lost owners must preserve actual failure and close only owned processes.

## Task 1: idle-device admission and immutable operational chain

**Files:** create `egoemg_repro/emgformer_resource.py`, `tests/test_emgformer_resource.py`, `scripts/run_native_emgformer_worker.py`, `scripts/run_native_emgformer_controller.py`; immutable staging and collection receipts in this plan's workspace or `tmp/native-emgformer-gpu-full-*-20261029.py`.

**Interface:** `select_idle_emgformer_gpu(gpus, applications, *, candidates)` receives complete typed NVML records and returns a copy of the first explicitly allowed idle record with free_mib>=65536, or None. Malformed/incomplete inventory raises ValueError. No Torch/CUDA import or hardware claims from synthetic tests.

- [x] RED: idle versus foreign/own applications, fixed reserve boundary, ordered explicit candidates, excluded reservations, immutable input/returned copy, malformed/duplicate/partial inventory.
- [x] Implement minimal helper and run the complete new helper suite. Expected:all cases pass without model/GPU construction; no old suites rerun.
- [x] Prepare one-shot operational worker/controller using existing APIs and owned-process guards. Phase order:CPU preflight;qual-prefix/resume/accept;qual val-native/val-replay/freeze/test-native/test-replay/report/verify;fresh full-fit/accept;full val-native/val-replay/freeze/test-native/test-replay/report/verify (20 phases total).
- [x] Bind all committed Python members, operational scripts, registered cohort receipt, upstream/runtime dependencies and physical inputs. Compile operational scripts and validate requests/source gates; record zero hardware claims from preparation.

## Task 2: CPU preflight and one operational review

- [x] Freeze current helper and closed core into a new private preflight-only capsule, separate from earlier proofs. Existing complete cohort producer identity is the expected receipt; new numerical admission is required by the existing fit/eval APIs, not a new qualification of unchanged mathematics.
- [x] Actual CPU preflight runs under owned-process/resource guards, rereads registered input, binds exact source/16 input bytes, and reports0 models/optimizers/updates/no CUDA. All owners must exit before collecting it. No training restart or synthetic fit recreation.
- [ ] One fresh reviewer examines helper, actual immutable operational scripts, preflight receipt and this plan. Material findings receive one observed regression RED→GREEN/fix pass; no re-review. No review of old model math or false GPU/science claims.

## Task 3: resource wait, actual qualification and complete run

- [ ] Stage a separate reviewed-source full workflow with durable execution-state recorder. Before GPU admission, rerun only necessary source/input preflight for this actual execution and wait on the same owned handle for an allowed idle GPU; do not claim that waiting is training.
- [ ] Collect actual prefix/resume/native raw acceptance, complete validation/test exact replay and report, actual GPU/PID/counters and owned worker closure. Only then enter fresh full200 fit. Failed evidence remains terminal, without automatic repeat.
- [ ] Observe the existing handle and collect full raw completion/selected checkpoint/native and independent val/test/report verification when actually finished. Publish source-bound results/timing/counters and retain legacy/PDF limitations. Do not mark this or the full goal complete before that evidence exists.

Ruling: this is routine execution of the approved whole-paper design and already implemented native recipe, so no repeated user design-approval round. Device restrictions and guarded engineering/full scope contain the cost of a mistaken scheduling assumption.
