# Fixed-checkpoint visual-missing EMG controls

> Execute inline with superpowers:executing-plans; existing authorized full-paper execution and explicit user priority apply. One fresh review of the new integrated control workflow, no repeated closed model/geometry suites.

**Spec:** [Controlled evaluation addendum](../specs/2026-11-03-visual-missing-emg-controls-addendum.md). **Goal:** actually determine the contribution of aligned EMG under native images and full visual loss using the existing frozen models and complete paired populations.

**Architecture:** A pure CPU FP32 input transformation isolates condition changes. A native forward adapter accepts genuinely registered completed producers, preserving exact target/mask outputs. A separate owned full workflow binds existing dense evidence/inputs and original predictions, runs the six fixed conditions, replays in a new process and publishes paired results after independent closure/readback.

**Global constraints:** upstream f7cbd048b7092139852270491d3859cd472e0311; no upstream/old mathematical/runtime mutation; no weights/BN/flags/normalization/model selection changes; no optimization; same17168ViTS/11114RN18keys and all invalid/missing requests; source-exact black-image normalization and unchanged label masks; deterministic time-only half-window roll; CPU2, RAM128GiB, disk225GiB and owned-only cleanup. Existing RN18 and RN152 jobs continue at their actual handles, never restarted due to observation timeout.

### Task 1: isolated input conditions

Files `egoemg_repro/modality_controls.py`, `tests/test_modality_controls.py`.
Interface `apply_input_condition(batch, *, emg, vision, vision_mean, vision_std)` returns independent copied `batch` and explicit `condition` metadata. Choices are aligned/normalized_zero/half_window_roll and native/missing; complete original CPU float32 input required. No learned/model/full-input permit.

- [x] Tests first: manual time-index oracle catches channel/batch roll, mean/std channel broadcast including native0..255 constants, natural missing and invalid labels preserved, clone isolation/no input mutations, unchanged RNG, zero semantics including inactive layout channels, malformed nonfinite/shape/dtype/mask/normalization and unknown condition rejection.
- [x] Observe RED; minimally implement explicit conditions; new relevant suite GREEN with0skips and source-bound evidence. Original mode is an exact value-preserving copy, not a reread/re-normalization.

### Task 2: genuinely frozen native predictions

Files `egoemg_repro/modality_control_inference.py`, `tests/test_modality_control_inference.py`.
Interfaces computational `native_control_arrays(models,batch,*,emg,vision,vision_mean,vision_std)` and registered `predict_control_batch(trained,batch,**condition)`; require actual expected completed ViTS/RN18 producer registration for the latter. Preserve native center forward/22-angle outputs and raw per-row errors. No actual complete scientific permit from adapter tests.

- [x] RED for incorrect target/mask, half/bfloat dtype, train/autocast, unsupported model/forged producer, mutated state/provider and output/nonfinite repair; observable real tensor fixtures, no fake scientific model claims.
- [x] GREEN; real original frozen producer engineering calls for both hands/all6conditions, actual aligned prediction agreement and exact before/after state; zero CUDA/optimizer/updates. Reuse original source/data evidence, no old model suite or geometry re-execution. Actual45passed/0skip/4originalTransformerwarnings169.37s;160members and2testowners plus independent collector closed, proof `results/visual-missing-emg/input-and-native-control-engineering-20261103.json`. No formalB64/fullpopulation permit.

### Task 3: full owned execution and independent replay

Files `egoemg_repro/modality_control_run.py`, `scripts/paper/run_modality_controls.py`, new workflow tests.
Interfaces prepare source-bound full evidence and actual native inputs; exact keys/conditions/raw output identity; first full pass, independent fresh full replay, model-free report and postexit collector. Existing native dense readers and full source guards are consumers, not changed internals.

- [x] RED for forged scope/model/input identities, missing/reordered key/condition/mask, incomplete populations, implicit sample filtering, false original-baseline joins and source generation drift.
- [ ] GREEN; original batch64/4workers and original six split/hand tails, real full-batch/tail qualification, source/normalization/full-input bound. Save literal unmasked22-angle error and valid-label diagnostic separately; keep original natural crop flags alongside artificial vision flags.
- [ ] One fresh integrated review, one material RED→GREEN fix pass if necessary, then launch original complete populations/conditions under resources. No repeat closed mathematics; do not label qualification a complete result.

### Task 4: evidence, comparison and publication

- [ ] Actual6condition full predictions and independent replay for each available completed route; source/model unchanged and all actors physically closed. Reopen complete output and exact original baseline keys/predictions.
- [ ] Independent raw-error arithmetic and original availability join, descriptive split/hand/severity/missing distributions and participant/recording dependence audit before inferential claims. Report negative/neutral results; no test-based retuning.
- [ ] Publish anonymous proof/summary/bottom tables and visual comparison, retain private datasets/models; verify actual public immutable bytes. Update full-paper ledger without declaring whole-paper completion.

## Review focus

1. Zero normalized EMG must match native force_zero_emg and not be advertised as raw sensor zero; Task1/2 actual boundary tests.
2. Time shift must remain time-only in the same sample with native window length; Task1 manual index and batch/channel identity tests.
3. Black image must use actual0..255 native constants and label-valid rows must survive false vision mask; Task1 normalization/mask and Task3 complete population tests.
4. Fake producer or qualification source must not grant full inference; Task2 registration and Task3 real source/prefix/cross-recipe negative tests.
5. Default original predictions, state/RNG mutations, correlated windows, selection by errors and stale active processes cannot support causal/complete claims; Tasks2/3/4 respective state, full-key, dependence and actor closure checks.

Task3 implementation checkpoint2026-11-03: source/raw22RED→22GREEN, complete source30GREEN0skip, execution preflight2RED and summary6RED→GREEN; original B64+45 tail real-neural integration48GREEN/0skipinitial337.11s. Nested original-input identity10RED→10GREEN3.81s adds exact original inner admission/flags/counts and actual native RGB metadata checks. Final current58GREEN0skip/2originalTransformerwarnings336.43s;166frozenmembers and all2current+8lineage testowners+independentreaderclosed, proof `results/visual-missing-emg/full-workflow-engineering-58-20261103.json`. Actual109source requests×6=654condition predictions are engineering in two original batches, not formal all6group qualification or17168full. Full orchestration/CLI/replay/descriptive report implemented; one fresh integration review and actual full qualified runs remain. RN18 source proof is intentionally pending while its original full workflow runs; pin only the actual completed postexit proof, never ViTS or a qualification milestone.

Task3 review/launch checkpoint2026-11-03: ONE fresh integration review atcff9369 found2Important; one material fix pass (reportreceipt17RED→19GREEN; actualnativeinputtransport13RED→13GREEN), final current90GREEN0skip/2originalTransformerwarnings408.81s and all6testowners plus independentcollector physicallyclosed;168finalfrozenmembers/source10SHA exact. Review acceptance binds4runtimefiles; no secondreview. Commit09335c96bc6b deployed as a create-only detached ViTS6condition workflow at09:05UTC;09:06 actualqual-a worker2700822/start452402887+guardian2700821/start452402883 and parentcontroller2700807/start452402603+recorder2700805/start452402600 allmatchinglive. Full scope remains17168/272and6conditions, originalB64/4workers; actualformalqualification/fullpopulation/replay/report stillpending, not a task-completion claim. Existing-prediction Task4 metadata/group audit separately shows13missinggroups5positive/8negative means and exploratoryclusterintervalcrossing0; it cannot substitute controls.
