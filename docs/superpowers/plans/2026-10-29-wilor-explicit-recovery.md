# WiLoR Explicit Recovery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. One fresh whole-unit review only; unchanged closed trainer/data/model/evaluation mathematics and suites are not reopened.

**Goal:** Recover the genuine failed WiLoR full30 from its newest complete original epoch packet, retaining honest interruption provenance and complete final evaluation.

**Architecture:** New admission/training and recovered evaluation modules reuse the closed native APIs and original Lightning restoration. New sealed lineage/result formats preserve original identity/journal bytes and distinguish prefix versus new execution; native checkpoint callback paths alone are relocated into fresh output.

**Tech Stack:** Canonical Python3.11/Torch2.2.2/Lightning2.5.0.post0; native CPU32 engineering and original B18/BF16/zero-worker full training; stdlib manifests and owned processes. No dependency installs.

**Spec:** docs/superpowers/specs/2026-10-29-wilor-explicit-recovery-design.md and the approved full-paper design2026-10-26.

**Earlier direction before cleanup (2026-10-29, superseded below):** no additional durable server storage was available; prioritize other experiments. The real six-epoch WiLoR parent and CPU recovery proof were retained. This historical direction did not pause the full-paper goal or authorize full training on volatile storage.

**Updated user priority (2026-10-29, after storage cleanup):** prioritize ViT-L, WiLoR and ViT-S+EMGFormer-S. Noncore source-admission owners are stopped with all files retained. Sufficient durable DATA space is now available. Complete and verify the bounded Task2 recovery unit first, including same-hardware CUDA UUID portability, immutable historical capsule protection and explicit rejection of unsupported recursive parents; run one current-source actual CPU proof and one fresh-context review, then start the genuine six-epoch continuation on an admitted equivalent A800. Complete Task3 evaluation/report separately while full training runs. This supersedes the earlier storage-based WiLoR deprioritization and sequencing; unchanged native model/optimizer/loss/precision/batch/budget and final scientific gates remain binding.

## Global Constraints

- Preserve closed wilor_native_train.py SHA91967a141fad6a00cf19711f8d06749e0409b21f06b8f6325a2808ae8cf011b2 and all prior model/data/eval/runtime bytes.
- Full original631631657parameters/404retained tensors/393AdamW state entries, original0.02 composite loss,30epochs/202020train/3152val/8612test/B18BF16/0workers; no implicit recipe overrides.
- Original failed full capsule remains read-only. Parent epoch5(index0) is six complete epochs/67344updates, last is five; no blind last resume. Total full target336720updates, remaining269376.
- Different explicit recovered format; no fabricated uninterrupted result or source retag. CPU/short engineering never increments scientific setting count.
- Parent raw/journal/identity/source and input guards survive the whole transaction; fresh exclusive output and owned-process execution only. Preserve host256GiB/output15GiB/GPU8GiB and native checkpoint workspace guards.
- Volatile engineering output is labeled and is not durable full-training capacity. Equivalent A800 CUDA UUID migration must retain the exact native hardware/runtime profile and actual64GiB start/8GiB running reserve; record actual device separately from immutable historical identity. Sharing does not authorize signals to other consumers. The earlier GPU0-only idle preference is superseded by the updated human core priority.

## Review Focus

- last older than selected newest complete epoch; malformed/same-step/same-summary-but-different raw packets must reject before model creation.
- Parent may already be complete/live, or its input/source/journal may change after admission. No overwrite, state loss or source-lineage laundering.
- Lightning restores after on_fit_start and constructs its iterator before on_train_start. Capture native restoration and RNG at the correct boundary, including real callback path relocation, without replaying old optimizer steps.
- A historical best may survive all remaining epochs. Every referenced top3/last path must remain readable and its original raw identity validated; preserve early-stopping history.
- A recovered fit must not pass the old uninterrupted reader, engineering must not become scientific, and test still requires exact independent frozen validation before any test consumption.

### Task 1: strict parent admission and explicit lineage

**Files:** create egoemg_repro/wilor_recovery.py and tests/test_wilor_recovery.py; doc docs/wilor-recovery.md.
**Interfaces:** admit_wilor_recovery(parent,*,checkpoint=None) -> admitted recovery object with identity, records, raw checkpoint, selected parent path, source guard, parent file hashes and callback copy map. check() rechecks immutable admitted inputs. Full parents choose newest complete matching journal packet; explicit older checkpoint only engineering. No model/optimizer/CUDA construction.

- [x] Write tests naming wrong packet/journal/source/scope/owner/callback path and stale-last failures. Observe RED before implementation. Use real retained packets on canonical server for state-heavy acceptance; small metadata negatives must fail before Torch/factory when possible.
- [x] Implement minimal guarded admission with unchanged native _validate_checkpoint and _selection semantics, sealed recovery manifest and historical identity preserved verbatim. Raw model/optimizer contents are not accepted from summary claims.
- [x] Run entire NEW admission suite and actual current full6 read-only admission,0models/0updates/noCUDA. Commit source/evidence; do not rescan old math/input absent current execution need.

### Task 2: native restoration and recovered result acceptance

**Files:** extend egoemg_repro/wilor_recovery.py; create scripts/run_wilor_recovery_worker.py for explicit phases; tests/test_wilor_recovery.py.
**Interfaces:** resume_wilor(admitted,output) -> sealed recovered result; read_wilor_recovery(output,*,require_full=True) -> recovered result; reopen_wilor_recovery(output,*,require_full=True) -> guarded selected native prepared runtime. New result contains total_updates, segment_updates, original/new owner and immutable parent lineage. Existing native fit identity/prefix journal remain verbatim.

- [x] RED tests for existing output/concurrent claim, all native restoration domains, missing callback references, RNG restoration before iterator construction, exact selected validation/native checkpoint reload, incomplete budget and false uninterrupted completion.
- [x] Implement native callback path relocation plus verified copies, CheckpointIO bootstrap loader and passive restore boundary; initialize original observer prefix records/steps and delegate original callbacks/loader/model. New result/reader validates all raw state, prefix conservation and source/input generation. Current bounded entry additionally rejects recursive recovered parents and outputs inside historical capsules, and records equivalent-hardware CUDA migration separately.
- [x] Use a new current-source CPU prefix stopped deliberately only after original first-epoch last save, then fresh-process recovery to original2epoch budget. Assert original model count631631657,2new segment updates/4total, original/fresh state domains and frozen validation agree. Current source differs from retained old CPU proof; never retag that proof. Explicit bounded volatile workspace only until durable capacity is available. Actual source503685d/bundle6a46063e5416, all three phases exit0, eight owners closed,115 members post-exit unchanged; public CPU receipt records the remaining boundaries and review work.
- [x] Run entire NEW recovery suite and independent actual raw/result reopen. Commit implementation and actual evidence. Current75bf1a1/bundlee3621ed3c564:103 checks passed; actual348.073s CPU recovery/read,1new model/2new updates/4total,117members and all6owners closed. One fresh-context bounded Task2 review:0Critical/0Important; actual full GPU gates and Task3 are explicitly outside its completion claim.

### Task 3: recovered native/replay/profile/report chain and review

**Files:** create egoemg_repro/wilor_recovery_eval.py and tests/test_wilor_recovery_eval.py; doc/results/immutable execution helpers.
**Interfaces:** run_wilor_recovery_pass/read/freeze/publish/verify wrappers mirror original public WiLoR evaluation signatures, with new recovered fit acceptance and explicit recovery report metadata. Reuse closed capture/statistics/native/precision/render helpers; no global monkeypatch of closed readers.

- [x] RED native/replay/profile/report/lineage corruption/test-before-freeze cases. Implement recovered _Evidence context and orchestration, preserve exact typed input/pred/target/mask/source IDs and original repeated population.
- [x] Run complete NEW recovery/evaluation suite; actual CPU recovered native-val/fresh-replay/freeze/native-test/fresh-replay/report/independent verify. The first five phases succeeded; the report failed only from an omitted existing dependency path. A new explicit postprocessing execution reused those passes, published and independently verified the report without repeating inference. Successful phases recorded four genuine631M models and zero optimizer updates; the failed report's final counters remain unavailable. Curves were inspected with an engineering watermark and recovery boundary. All20owners closed and120sources independently rehashed.
- [x] One fresh-context review of this new unit and actual current-source evidence; adjudicate every finding/declined item, one regression RED→GREEN fix pass, no second review. Review5713383:0Critical/0Important/1stale-plan Minor. Actual queue startup then exposed compact-versus-complete recorder identity checks;11new RED cases led to133passing new checks and actual surviving wait loops. Native training/evaluation/recovery mathematical sources remain unchanged.
- [ ] Wait for the existing durable GPU4 full continuation to finish naturally; do not restart it. The separate source-bound evaluation controller requires exact full30/336720total/269376newupdates,3152frozen validation and six-domain restoration plus owner closure before evaluation. Then preserve native16GiB inference admission,8GiB running/256GiB host/15GiB disk reserves and execute val-native/replay/freeze/test-native/replay/report/verify. Require all8612test consumptions/4306unique hands and independent report verification before adding this scientific setting. The equivalent-device ruling supersedes the historical idle-GPU0 wording; current training and queued evaluation are not a final score.

## Pre-flight interfaces and ruling

Task1 object -> Task2 native resume: same historical identity/records/raw state and pinned file/callback map; Task2 new recovered reader -> Task3 explicit evidence context, never original uninterrupted reader. Test bounded old-prefix selection is engineering only; full selection must be newest complete. Tasks share no mutable historical capsule. Routine repair is already authorized by approved full-paper design; no repeat user design approval. Current resource shortage blocks full execution, not read-only admission/implementation or explicitly bounded engineering verification.
