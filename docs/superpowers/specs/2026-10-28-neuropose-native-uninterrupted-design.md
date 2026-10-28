# NeuroPose native uninterrupted execution design

Implements full-paper requirements2/4/5/7 under the [approved design](2026-10-26-full-paper-reproduction-design.md). Its existing authorization covers routine implementation, server execution and publication. This is an execution-method ruling by the implementer, not a new user-selected model recipe.

## Evidence and decision

The old `canonical_fp16_eval_copy_v1` gate remains failed. Its saved NeuroPose recovery diagnosis has exact restored state and consumed inputs, but8/171380 subsequent prediction elements exceed the chosen engineering epsilon. A retained independent fresh/fresh control, with neither run using resume, exceeds it at23122/171380 elements despite equal initial state, consumed inputs and final RNG. These observations do not isolate a CUDA kernel or establish statistical equivalence. They show that failure of this local future-training-trajectory criterion does not require checkpoint restoration. Strong deterministic CUDA execution was also already attempted and rejected by the original linear-upsample backward; do not repeat that experiment or replace the operator.

The full-paper design requires native forward/backward, complete validation and strict reload before training. The paper/code does not require two stochastic training trajectories to remain close. Adopt a separately named **native uninterrupted NeuroPose** path: real bounded original training, exact selected-checkpoint reload, full frozen validation native/replay agreement, then one fresh complete training run. It does not offer checkpoint continuation or make a resumed-future-trajectory claim. Existing failed artifacts and thresholds are never changed or accepted by this path.

An empirical resumed-trajectory envelope would address a different engineering claim and require additional calibration/confirmation experiments. It is unnecessary for the paper's fresh training experiment. Unconditional acceptance is excluded: the new full entry requires its own actual CUDA qualification artifacts and fixed exact frozen-model prediction checks.

## Fidelity

- Fixed official upstream `f7cbd048b7092139852270491d3859cd472e0311`; original NeuroPose factory, operators, dropout, BatchNorm, initialization, native Lightning loss and optimizer/scheduler.
- Original16-slot input/7790points, generated22-angle trajectory/2Dmask, fixed scalar normalization, augmentation order and original batch-union FK. Effective fingertip weight remains.02.
- Full counts101714train/1576validation/4306test hand samples; native train/validation batch256, single CUDA rank/workers0/16-mixed and default enabled GradScaler, no accumulation or clipping; max100epochs and original val_loss/min/patience50. Record the existing single-rank versus declared six-rank BatchNorm limitation.
- Existing partition, complete cohort receipt, numeric/three-rung capacity evidence and code/runtime identities must be reopened. Preserve all old source files and running/frozen producer capsules.
- Full training reads no test samples. Validation alone selects native best; last is never a substitute. Complete test predictions and detailed trajectory/center statistics are required after full termination and a frozen validation profile.

## Boundaries and interfaces

New `neuropose_native_train.py` owns policy/preparation, native epochs, checkpoint/journal binding, strict fit reading and original best selection. Reuse old pure coverage, recipe/data admission, numeric/capacity readers, and termination arithmetic without modifying their modules or globals. Use the original factory and native callbacks/checkpoint format with a new identity packet. No `resume`, caller batch/precision overrides, early truncation of full scope, overwrite or implicit restart interface.

Scopes are `full`, `cuda_engineering`, `cpu_engineering`. CUDA engineering retains B256 and all1576validation samples, trains the first512samples for2epochs. CPU engineering is explicitly B2/32-true/two epochs on an ordinary small memmap corpus, with up to4train samples and its entire validation set; never CUDA, full data or scientific qualification. Neither engineering scope can be upgraded to full by editing a summary.

New `neuropose_native_eval.py` owns actual selected-best native validation/test, independent new-process replay, pretest freeze, GPU qualification and safe report/curves. Reuse `ReportInputLedger`, original tuple checks and `ClassicAngleReport` as pure helpers. Retain typed EMG/target/mask/sample/metadata/prediction digest chains, complete ordered coverage and actual native loss/metrics. Do not retain the complete multi-gigabyte test prediction tensor merely to compare it: streaming ordered typed batch digests and sufficient statistics are adequate for a genuinely new inference pass.

Frozen native versus replay predictions/inputs must match exactly under identical batching, original dtype and model state. Native loss is separately checked against complete sample-weighted batch losses using the already used fixed atol1e-7/rtol2e-6; this is not a training-trajectory comparison or a newly fitted tolerance. A failure remains failed with raw evidence preserved.

Full qualification requires completed **CUDA engineering**, its actual selected-best native full validation and a separate-process exact replay, all current sources/runtime/cohort/preconditions bound, and actual owners exited. CPU fixtures, old failed gate records, booleans or caller tolerance values cannot supply it. Full fit identities bind the qualification and original input paths; independent readers reopen their actual evidence. No hidden acceptance based on a stored `passed` flag.

## Completion and failure

Each checkpoint retains ordinary native model/optimizer/scheduler/scaler/loop/callback state plus the new identity, actual epoch journal, generator/RNG state and actual optimizer execution count. Epoch coverage must complete before callback selection/checkpoint commitment. Failed/interrupted full runs remain incomplete; no automatic restart or continuation. Native max-budget or patience termination, complete journal, original selected best and the original best file must agree before full fit is accepted.

Validation native/replay must be frozen before held-out testing. Native test and independent test replay must bind the same full selected checkpoint and ordered input population. Public reports anonymize users/private paths and separate native batch-weighted metrics, pooled degree statistics, trajectory/center, hands/splits/users/joints/fingers. Curves derive from actual journal and selected checkpoint epoch; completion is written last after raw-chain verification. Existing output directories are rejected.

There is one final fresh-context review of this new implementation, after CPU and actual CUDA qualification and before its full run. Fix accepted Critical/Important findings once with targeted RED→GREEN and affected regressions; ledger minors and all rulings. No re-review of previously completed projects or second review.

Full-paper scope and all remaining annotation/data/analysis obligations remain intact. This document is not qualification, a full NeuroPose result, or whole-paper completion.
