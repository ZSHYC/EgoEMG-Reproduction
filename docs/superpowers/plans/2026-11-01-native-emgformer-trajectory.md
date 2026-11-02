# Native EMGFormer Full Trajectory Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. Existing approved full-paper design and official-code-priority execution authorization apply; no repeat design approval. One fresh final review of this new component, one material fix pass, no review of closed model mathematics.

**Goal:** Complete the missing S/M/L full-trajectory evaluation and its hand/split/user/joint/finger outputs from the actual completed native200 validation-selected models.

**Architecture:** Preserve the canonical center-evaluation cohort, selected fits and reports. A new full-window input adapter invokes the same original reader with its supported `center_target_only=False` setting, scans all newly consumed held-out target rows, and binds the original point-cohort identity. A separate evaluator invokes the original Lightning forward, including its original context crop and `align_predictions`/`align_mask`; full raw batches support fresh-process replay and public reports.

**Tech Stack:** Existing pinned official f7cbd048..., Torch2.2.2, Lightning2.5.0.post0, NumPy1.26.4, existing immutable legacy memmaps. No dependency installation.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`, requirements2/4/7. Existing execution method is inline.

## Global Constraints

- Full objective Tables1–15/Figures1–13 remains intact. This component does not complete the whole paper or establish PDF population equivalence.
- Completed native200 selected-best sources only; no retraining, weight selection from trajectory tests or replacement by the legacy0.01 models. Original center reports remain unchanged.
- Original evaluation windows/normalization/left-then-right order/repeat1:1052 validation and2792 test hand windows on actual release. No replacement with4306 vision centers or dense sampling.
- Reader targets22×12000, generated validity12000. Float32 wrist degree→radian conversion follows the original full reader, which differs from scalar center rounding. Stored wrist-valid flag stays ignored as in original code; no repair/filtering of masked nonfinite labels.
- The old point-cohort finite scan covers held-out centers only. New admission must stream-check all held-out window target rows, preserve complete16-file physical hashes and original parent identity, and reject masked off-center nonfinite values.
- Original full-loop settings B500/6workers/prefetch1/persistentTrue/pinFalse/shuffleFalse/dropFalse/BF16. CPU engineering is explicit B2/0workers/FP32 and grants no full scientific eligibility.
- Full target alignment must use the original same-version Lightning callable and model contexts, not hand-written context or output-width constants. Preserve native batch-weighted FP32 MAE and separately pooled FP64 valid-angle statistics.
- Separate val-native/val-replay precede trajectory profile freeze and test-native/test-replay; same fit/source/normalization/windows, exact raw replay, distinct exited producer actors. Profile does not change the already selected weight or conceal prior center-test exposure.
- Report trajectory and the center of the aligned trajectory separately from the prior native point grid. Include all22 joints, fingers/PD groups, hands, Gesture/User/Both and per-user mean/SD/denominators. Retain existing preprocessing/data-version caveats.
- All data/model sources guarded before/during/after, no optimizer or CUDA in input admission, no new GPU scheduling until engineering and one final new-unit review pass. Preserve exact original job handles; never restart from quiet logs/timeouts.

## Review Focus

- Off-center and last-row NaNs hidden by false masks must fail full admission despite accepted center-only sources:Task1 finite tests.
- Read-only view dtype/stride/normalization/center flags, dataset role/window membership or process-local pickle reconstruction must not change admitted computations:Task1 binding tests.
- Native interpolation/context and full-reader wrist rounding can make aligned-trajectory centers differ from point-grid centers:Task2 same-forward tests, no false equality claim.
- Same actor/path/copied prediction, wrong selected fit/profile/source or incomplete final batch must reject before report promotion:Task2 raw/pass tests.
- Failure during loader consumption and independent report rebuilding must close owned children and leave completion pending:Task2/3 cleanup and publication tests.

## Task 1: source-bound full held-out input admission

**Files:** create `egoemg_repro/emgformer_trajectory_data.py`, `tests/test_emgformer_trajectory_data.py`.

**Interfaces:** `admit_trajectory_inputs(data, *, upstream, config, normalization, expected_point_identity=None, expected=None, chunk_rows=262144) -> TrajectoryInputs`; `.identity`, `.point_cohort`, `.datasets['val'|'test']`, `.check()`; `trajectory_loader(inputs, role, *, cpu_engineering=False)`.

- [x] Write new tests using actual synthetic memmap IO: full22×12000 targets/both hands/roles and unchanged EMG/order; all held-out row numeric checks, masked off-center/tail NaNs, exact physical bytes and RNG invariance, resealed wrong receipt, persistent reader/source mutation, pickle rebuilding and literal loader settings. Engineering fixtures never acquire full-release identity.
- [x] Run only the new data suite; expected missing-module RED. Implement a separate wrapper around original `EmgformerDataset` views; never mutate parent point-cohort readers. Use original complete physical admission and `_reader_binding` guards; stream all new full-target spans in bounded chunks. Require actual registered input objects for loaders.
- [x] Run the complete new data suite to GREEN and independently inspect actual server logs/source hashes. Commit source/tests and scoped evidence. No old model/math/reader suite reruns.

## Task 2: full native forward/raw replay/statistics lifecycle

**Files:** create `egoemg_repro/emgformer_trajectory_eval.py`, `tests/test_emgformer_trajectory_eval.py`; create guarded `scripts/paper/run_emgformer_trajectory.py`.

**Interfaces:** `run_trajectory_pass(inputs, fit, output, *, role, kind, reference=None, profile=None, require_full=True)`; `read_trajectory_pass(directory)`; `freeze_trajectory_validation(first, second, output)`; `publish_trajectory_report(test_replay, output)`; `verify_trajectory_report(test_replay, output)`.

- [x] RED native full-reader→original Lightning forward target/context/interpolation parity; raw batch coverage/typed outputs/input digests and native-vs-pooled metric conservation, joints/fingers/users. Include all-invalid batch, partial final batch and float32 wrist rounding.
- [x] RED wrong fit/profile, same actor/path replay, missing/mutated packet, incomplete source/population, worker failure cleanup and early completion marker. Implement by reusing closed source/fit/population/process helpers, not modifying old evaluation modes or core mathematics.
- [x] GREEN complete new evaluator suite. Run separate processes against an actual retained native CPU engineering fit/explicit engineering memmaps, native val/replay/freeze/test/replay/report/rebuild with0 optimizer updates. No synthetic full-source producer positive. Actual final20passed/0skip/1originalcollationwarning525.63s; closed raw/statistics21 and input28 checks remain separately recorded, not rerun. Original Trainer iterator-reset behavior additionally tested with12actual data workers and no learned inference.

## Task 3: one final review, actual complete inputs and full execution

**Files:** scoped evidence under `results/native-emgformer-trajectory/`, guide and full-paper status updates.

- [x] One fresh final review of this new data/evaluator/CLI integration, one author material fix pass with actual RED→GREEN and green new suite; no second review or old math suites. One Important fit-cohort-receipt guard omission fixed; genuine200-fit read-only generation regression RED→GREEN4.24s. Same author fix phase also retained pre-teardown worker identities after confirming actual native Trainer resets its iterator; focused RED16.20s→GREEN16.07s, then full final20GREEN525.63s.
- [x] Complete two fresh full16-file/held-out-window numeric admissions for each actual S/M/L source; exact identity/replay and actual process closure. Data admission alone grants no fit/evaluation completion. Actualsixownedstages/414.04s exit0; each16files16,540,457,757bytes, val1052/test2792 fulltargets, exacttwoidentities/JSONbytes per size. Independentpostexit143frozenfiles/16payloadfiles/3genuinefullfits rehashed and14originalactors+collectorclosed; proof `results/native-emgformer-trajectory/full-input-double-admission-20261101.json`.
- [ ] Preserve original registered GPU jobs. Queue source-frozen B500/6worker/BF16 qualification and full val/test/replay/report only on their actual assigned GPU after exact predecessors close and original reserve checks pass; no local-PC dependency or implicit retry. Queue registered under frozen983fd9338f4a, actualliveowners and exactRN50predecessor/GPU1UUID independently verified; no learnedphase started. Use complete1052valnative/freshreplay(batch500/500/52) as stronger pretestqualification beforeprofile/test, fixed2e-6policy/no reselect; actualGPUqualification remainspending.
- [ ] Collect only terminal successful full runs, raw/source/owner closure and independent report reconstruction. Publish original reports and complete numeric tables; verify remote Git ref and public immutable file bytes. Keep full-paper goal active for remaining requirements.

2026-11-02 per-size S closure: the actual shared-lineage source14a9a1e4486d/cc02413f trajectory queue has completed S seven-stage full1052val/2792test native/fresh model replay, pretest freeze, raw report and independent closed verification; S outer and all14 inner actors, all24 loader workers are closed. Same native200/25000 point-validation BEST169 is preserved. A separate CPU existing-reader pass under collectoracc4ccb42966/a8142def reopens all16 actual inputs16,540,457,757bytes and full raw/checkpoint graph with0learned/optimizer constructors,0newforwards/updates/CUDA; both additional reader actors closed0. Full trajectory16.53769621006572deg (684769448validangles), aligned center16.474382758101598deg (59532validangles). Original source6255 aligned center differs from6000 point report; no re-selection. Actual200-row curve is point-validation, not trajectory validation. S reports and public comparison appended; legacy and all previous snapshots retained. M has actually started, L follows; all3 completion and full-paper checks remain open. No original qualifiers/models/math/engineering suites or second review repeated.
