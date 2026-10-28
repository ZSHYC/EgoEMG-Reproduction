# Native WiLoR22-angle implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. No per-task implementation/review agents; one final whole new WiLoR implementation review after actual GPU qualification. TDD for each behavior.

**Goal:** Complete a genuinely initialized native WiLoR22-angle full legacy experiment and independently verified scientific report, disclosing the missing author YAML and explicit derivation.

**Architecture:** Separate configuration/runtime/fit/evaluation modules leave old ResNet/DINO/classic/fusion sources and live capsules immutable. Reuse their pure data/coverage/report boundaries with explicit compatibility identities, not old model permits.

**Tech Stack:** Existing Torch2.2.2/Lightning2.5.0.post0/Hydra/NumPy, separately pinned WiLoR source/dependencies, original BF16 native factory/AdamW/cosine and existing isolated report extra.

**Spec:** [WiLoR native design](../specs/2026-10-28-wilor-native-design.md), implementation detail of the already authorized full-paper design. One fresh builder check found “nothing important left”; routine code-first choices are implementer's rulings, no new user scope decision.

## Global constraints

- EgoEMGf7cbd048/WiLoRfcb91131, genuine2,564,989,533B/SHA3e97aafc pretrained asset; original1280feature/height192×256/22angles/631631657params/52499frozen. No random/demo/MANO substitute.
- ExplicitS-recipe derivation: originalmodulemapping replacement, paperB18/LR1e-5/max30, inheritedweightdecay.01/BF16/clip1/val_mae/min/top3last/p30; nativecosineeta_min1e-5 soLRconstant, effectivefingertip.02. No guesseddecreasingfloor.
- Full202020/3152/8612repeat consumers and101010/1576/4306unique, original7790/stride780/eval7790/repeat2, generatedmask/normalizedblackmissingcrops, independentjitter-safeval. No merged/PDF/sixrank equivalence.
- No overwrite/resume/implicitretry/testselection; actualsource/data/runtime/owners/checkpoints/counters/termination frozen and independently read. Existing training/dependencies/outputs untouched, weights private; public aggregates only. ≥15GiBdisk pluscheckpointworkspace/256GiBhost/freshactualGPUreserve.
- CPU2epochs/B1/32true/≤4samples andCUDA2epochs/B18/BF16/train36/all3152val are engineering only. Full requires actual current-source raw CUDA qualification; scientific count only after genuine30epochs/all evaluation/report verification.

## Review focus

1. WILOR_PATH is captured at import: reject prior incompatible module roots without replacing sys.modules (Task2).
2. Original missing/empty weight path can silently initialize random backbone: complete asset and actual404tensor match must precede optimization (Task2).
3. Replacing instead of merging module mapping and inherited constant LR/alias loss are necessary; metadata must not claim author YAML/decreasingcosine (Task1).
4. Common S-data cohort is data-only, not a WiLoR configuration/GPU/full permit; preserve generated mask/repeat/order and reject forged compatibility bindings (Task3).
5. Native teardown relocates the model toCPU; actual input/module device receipts must be captured at each forward; no engineering/prediction-before-freeze/full-best swap promotion (Task4).

### Task 1: Exact prospective configuration contract

**Files:** Create `egoemg_repro/wilor_literal.py`, `tests/test_wilor_literal.py`, `docs/wilor-native.md`.

**Interfaces:** `compose_wilor(upstream) -> DictConfig`; `wilor_protocol(upstream,config) -> dict`; private `_validated(upstream,config) -> (Path,dict)`. Consume fixed `_approved_upstream`, `_activate`, `_file_hash`, `_hash` only. Output safe recipe/model/source metadata; no model/asset/dataset/CUDA construction.

- [x] Step1: Write tests: B18allroles/unusedval256/repeat2/7790/780/7790, exact modulekeys noDINOleak, 30epoch1e-5floor/constantclaim, rawalias.02, originalcallbacks/freeze/parameter expectations, explicitinheritance/missingYAML/PDFdisagreement/falsecompletion. Parametrize alteredbatch/type/repeat/loss/precision/clip/patience/floor/module/pretrainedpath/sourcebinding rejection; privateEGOEMG_ROOTexcluded. Genuine native datamodule characterization uses controlled3row dataset→6repeat samples perrole/B18/drop_lastFalse, no model.
- [x] Step2: Run `python -m pytest tests/test_wilor_literal.py -q` in existing native env. Expected: missing-module FAIL. Implement exactsource-bound composition/validation/protocol; repeat same command. Expected: all PASS, no model construction.
- [x] Step3: Run new contract + unchanged `tests/test_vision_literal.py`; bare suite with EGOEMG_UPSTREAMunset; fresh subprocess protocol/noTorch-orCUDA import proof. Expected: checks exit0, originalsources unchanged, all true scopes retained. Save actual logs/owners/hashes and public evidence, update ledger, commit.

### Task 2: Genuine asset and original native module/optimizer

**Files:** Create `configs/wilor-assets.json`, `egoemg_repro/wilor_literal_runtime.py`, `tests/test_wilor_literal_runtime.py`.

**Interfaces:** `admit_wilor_assets(runtime,checkpoint=None,*,restore=False) -> WilorAssets` with `.check()`; `build_wilor_runtime(upstream,config,*,assets,state=None,expected_initialization=None) -> PreparedWilor` with `.module/.identity/.check()`; `wilor_optimizer_identity(prepared,optimizer,scheduler) -> dict`. Catalog copies previously verified26fixed source/config and mean/checkpoint hashes, paths supplied privately.

- [x] Step1: Tests first for partial/wrongnamespace/source/config/mean/size/SHA/refusedbool/emptyweightprefix, schema/dtype/finite/strictstate/frozen/optimizercoverage mismatch; wrongloadedmodulepath rejection beforefactory. Observe RED then implement guarded IO/admission, strict originalfactory construction/reload and native2groupAdamW/cosineconstant identity. Restore does not reopen pretraining checkpoint, requires pinned source/config/mean and complete initial lineage plus strict actual saved state.
- [x] Step2: Genuine sequential remoteCPU construction with existing actual2.56GBprivateasset: all404backbone tensors exact, nativeoriginalmodule and optimizergroups, actualfiniteforward/backward/update onordinarysmallcohort; save/release then fresh-process strictreload/fullsmallvalidation. No local large construction, GPU or finalaccuracy claim. Expected: all ownersactualexit0, immutableassets/source/state, exactfrozenpredictions; failures retained.
- [x] Step3: Affected/related/bare checks once, independentrawCPUproof collection and viewed engineering curves if generated; commit actual source/evidence. Expected: exit0/sourcehashesmatch; nofullpermit.

### Task 3: Complete data binding and native uninterrupted fit

**Files:** Create `egoemg_repro/wilor_native_train.py`, `tests/test_wilor_native_train.py`; extend guide.

**Interfaces:** `wilor_policy(*,scope='full') -> dict`; `fit_wilor(data,crops,*,upstream,config,assets,output,scope='full',qualification=None) -> dict`; `read_wilor_fit(output,*,require_full=True) -> dict`; `reopen_wilor_best(output,*,require_full=True) -> PreparedWilorFit`. Prepared exposes `.module/.datasets/.identity/.guard/.policy`, selected actualnative best/scaler and paths for Task4.

- [x] Step1: Tests first: explicit datacompatibility projection to originalS `admit_vision_cohort`/pointreaders, fullcounts/payload/finitecenters/source/RNG/layout/missingcrop, forgedbinding/nonfullrefusal. Observe RED then implement outerWiLoR binding; no helper/global edits. Numeric actual fullinput fresh hash + independent newprocess re-open beforeCUDAqualification.
- [x] Step2: Native fit tests first: exact3scopes/no callerrecipeoverride/resume/overwrite; originalcallbacks/top3last/30epochfulltermination, epochjournal/checkpoint identity and actualoptimizerexecutions/RNG/coverage, independentselected-best reload. OrdinaryCPUtinyfixture twoepochs; interruption/nonfinite/partialcoverage/wrongbest/source/journal/scaler/counters/incompletefull fail. Observe RED then implement native trainer/observer/readers. Full entry checks Task4rawcurrentCUDApermit before model/data; engineering neverfull.
- [x] Step3: New/related/bare checks and independentCPUrawfit/best proof, sourcehashes/ownerexit; commit. Expected: allpass, fullentry remainscloseduntilactualnewGPUqualification.

### Task 4: Exact evaluation/report, actualGPU qualification, one review and full experiment

**Files:** Create `egoemg_repro/wilor_native_eval.py`, `tests/test_wilor_native_eval.py`; immutable ignoredstage/controller/observer; public sanitizedresults.

**Interfaces:** `run_wilor_pass(fit,output,*,role,kind,reference=None,profile=None,require_full=True)`, `read_wilor_pass(output)`, `freeze_wilor_validation(native,replay,output)`, `qualify_wilor(fit,native,replay,output)`, `read_wilor_qualification(output)`, `publish_wilor_report(fit,native,replay,output,require_full=True)`, `verify_wilor_report(replay,output,require_full=True)`.

- [x] Step1: Tests first actualnativeCPUval/test + newprocessreplay/typedinput-predchains/lastpartialbatch/perforwarddevicereceipts; strictrawsource/selectedstate/scaler/owner/runtime/cohort/selection/startfinish; reject summarybool, oldDINO/CPUpermit andtestbeforefreeze. Observe RED then implement originalTrainer callbacks and independentnative-precision inference; noepsilonexpansion.
- [x] Step2: Tests first full22angle center/hand/split/anonymoususer/joint/finger/PD/cropall/repeatversusunique conservation; actualjournalcurve/bestmarker/lateinputchange/nooverwrite/completionlast andanotherprocess exactPNGSVGregen. Implement and run affected/related/barechecks/independentCPUproof, inspectcurve, commit.
- [x] Step3: Fresh completecontentandresourcequalification: originalB18BF16 twoepochs36train/all3152val, actualfinitebackward/updates/capacity andselectednativefullval/newprocessreplay, rawqualificationCPUread. Expected allownedexit0/currentexactsource/runtime/weights; retain failures, neverreducebatch/model/precision to pass.
- [x] Step4: SolefinalfreshwholeNEWimplementationreview; grade/rule/ledger, oneacceptedC/IRED→GREEN+affectedqualificationfixpass, nosecondreview. Launch fresh originalfull30onlyafteractualpermit/review, separateownedfit/accept/val-native/replay/freeze/test-native/replay/report/verify; immutableguards/noexistingconsumerkill.
- [ ] Step5: Actualcomplete30epochsandnativebest; entiretest8612consumed/4306unique/newprocessreplay+reportverify; publiccurves/metrics/time/evidence. Incrementcompletedsettingonlythen; allwiderpaperwork remains active.

## Self-review and execution decision

All spec sections map to Tasks1–4; common data202020 is not fusion395276. Signatures in consumers match outputs; engineering counts and restore assets explicit; no production implementation before actual RED. Builder found no unresolved material question. This plan implements already delegated official-code-first full-paper work, not a new request for scientific-method selection. Execute inline; record each additional implementation ruling and actual failures in this plan's own ledger, do not delete evidence while full experiment remains pending.

Task4 Steps1–2 evidence: source7910238; actual retained631M CPU native-val/replay/freeze/native-test/replay/report/independent PNG-SVG verify all succeeded. Last private negative-copy audit driver failed no-overwrite protection; originalFAILED1 retained, recovery-only audit COMPLETE0 without model/prediction/training rerun. All11negative cases and late generation/completion guard passed;24related owners nonlive,226source members/original fit/raw predictions/report independently rehashed. Local284native/related+server2retained+1601light checks passed; actual curves inspected; see results/wilor-native-eval-cpu-20261028.json and validation JSON. Step3 originalB18BF16 GPU/full3152 qualification, Step4 sole NEW review/full launch and Step5 full30/whole test remain unchecked; engineering proof never opens these gates.

2026-10-29 update: Step3 first actual B18/BF16 GPU chain completed exit0 (sourcef6fab48,2epochs4updates/full3152 validation/exact new-process replay/raw CPU qualification reader); all18related owners exited and224stage members/raw checkpoints/predictions rehashed. Historical proof: results/wilor-native-gpu-qualification-20261029.json. Step4 sole fresh NEW implementation review completed:0Critical/2Important accepted/1Minor deferred, with no second review. Accepted I1 loop/CUDA-RNG and I2 fit-to-pass reader bindings fixed together at8ee760f;36RED→37targetedGREEN,321native+1632light checks, actual retained CPU/GPU packet acceptance and damaged/cross-fit rejection succeeded. Fresh current-source A/B complete-input admission also succeeded. Step3 checkbox remains open because current source binding requires a NEW actual GPU qualification; Step4 checkbox includes full launch and remains open; Step5 remains open. Current disk capacity and immutable source guards still apply. Old successful proofs remain historical rather than being resealed as current.

2026-10-29 Task4 Step3 COMPLETE against post-review current production source8ee760f (capsule HEADf993556): originalB18/BF16 twoepochs/fouractualupdates/train[36,36]/fullval[3152,3152], full selectednative3152 and exact independenttyped replay, qualifier and fresh CPU raw reader all5 actualexit0. Wholeflow19:29:01.727243→19:48:55.379698UTC; independentpostexitcollector rehashed225GPU+inputsource members/all3rawcheckpoints/predictions, all18ownersclosed/currentinputgenerations unchanged. See results/wilor-native-gpu-qualification-review-20261029.json. Sole review already adjudicated; do not repeat it or prior passed checks. Step4 remains unchecked until actualfull30launch, Step5 full scientificresult notcomplete.

2026-10-29 Task4 Step4 COMPLETE: sole review plus one accepted I1/I2 fix pass was already complete; full30 owned workflow a618f2723de7/source8a83e42 started19:56:36UTC after actual current qualification9a9973a2…64e411 and real55GiB/64GiB start guards. Preflight actualexit0/179.625329485s, then actual native fit identity scopefull/max30/B18BF16/counts20202031528612/631631657parameters/404retained pretrained tensors created. Same PID/start fit worker is an actual GPU0 consumer; no complete epoch or optimizer-update count yet claimed. See results/wilor-native-full-launch-20261029.json and results/wilor-native-full-gpu-start-20261029.json. Step5 remains unchecked until genuine30epochs/all8612test4306unique/exact replay/report verification. No new review or source change.

2026-10-29 first full epoch evidence: actual202020train/3152val/11224optimizer executions committed. IndependentCPU reader loaded the genuine7,579,991,738-byte native epoch0 packet and passed the unchanged production validator, including full-scope sanity2 loop counters/176val batches, whole model/frozen digests,393AdamW moment states/11224steps and nativeCUDA RNG bytes. Actual14.799973s/exit0/no CUDA/new models/updates; postexit113frozen members and current9raw-file attributes unchanged. First private checker import failed before checkpoint load; failure and source retained, only corrected that private import in a new directory. See results/wilor-native-full-first-epoch-20261029.json. Original GPU fit continues; Step5 remains unchecked and completed scientific settings remain10.
