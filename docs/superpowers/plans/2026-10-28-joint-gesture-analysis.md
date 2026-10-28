# Joint and Gesture Error Analysis Implementation Plan

> **For agentic workers:** Continue approved inline execution with superpowers:executing-plans; no per-task agents or reviewers. This implements the existing approved full-paper design.

**Goal:** Produce actual selected-checkpoint per-joint and per-gesture evidence for Table12/14/15 and Figure12, keeping legacy and fixed official-formula populations explicit.

**Architecture:** Add a small pure numerical aggregator, an independent read-only selected-checkpoint collector reusing the established legacy model/data/audit APIs, and a table/scatter renderer. Run each retained S/M/L, RN18 and RN18+S exactly once without training; retain private per-sample records and publish only class/group aggregates and source digests. Full-paper science scope and missing canonical data/recipes remain open.

**Tech Stack:** Existing Python/NumPy/PyTorch/Matplotlib runtime, original pinned models, managed server stages and actual exit receipts.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements1/4/5/7; full-paper-status Table12/14/15/Figure12 and semantic-tables denominator contracts.

## Global Constraints

- Official version remainsf7cbd048b7092139852270491d3859cd472e0311; no source/model/partition/weight/normalization changes to completed runs or active jobs.
- Official `analyze_per_gesture.py` computes the arithmetic mean over22 angles without applying mask, and the standalone EMG raw token midpoint; its printed conversion is57.3deg/rad. Record that formula separately from existing aligned-center masked180/pi statistics. No zero fallback for a missing branch.
- One standalone S, RN18 and RN18+S selected by independent validation; never substitute a fusion-finetuned branch or choose checkpoints using analysis outcomes.
- Baseline all-center sums/counts and original full trajectory sums/counts must reproduce saved metrics at the already fixed1e-4degree tolerance/count equality before publishing. No tolerance override or population trimming to obtain agreement.
- Preserve all requested centers and missing-crop records; publish separate all-test/gesture-only and crop-present views. Exclude raw50 and unassigned-1 explicitly only from59-class views; retain before/excluded/after and zero-coverage definitions asnull-MAE rows.
- Family mapping comes from verified semantic definitions;33/34are asymmetric,50/51/52symmetric. No continuous-ID family inference.
- Public artifacts must never contain source paths, row indices, subject identifiers or private sample maps. Numeric row join keys and complete per-sample predictions/errors stay private.
- Actual real resource observations and ownership apply; no healthy job restarts, existing queue modifications, retries, synthetic precision results, new GPU time claims or publication of weights/data.

## Review Focus

- Different valid-angle counts: pooled MAE must use sums/counts, while official gesture mean is equal-sample; tests use deliberately unequal counts.
- Duplicate/missing/reordered sample keys or inconsistent hand/split/gesture/crop metadata: pair strictly and reject mismatch; ordering alone cannot pair models.
- Raw/unassigned/unknown classes and an entirely empty class: never fabricate0error; tests pin exclusions and missing observations.
- NaN/Inf/negative/noninteger counts or zero-count/nonzero-error: reject malformed sufficient statistics; unmasked formula requires finite real means.
- Old checkpoint/data/source drift and process/GPU contention: reuse established validation, freeze source/input identities and use one-shot owned stages; new analysis cannot alter or qualify old training.

### Task 1: Numerical aggregation with real RED/GREEN

**Files:** Create `egoemg_repro/paper_analysis.py`; Test `tests/test_paper_analysis.py`.

**Interfaces:**
- `joint_group_table(rows: list[dict]) -> dict`: each row has22element `trajectory_angle_error_sum_rad`/`trajectory_angle_valid_counts`; returns22angle and10overlapping finger/proximal/mid/distal/wrist groups, preserving radians sums/counts and180/pi means.
- `gesture_error_tables(records: dict[str,list[dict]], definitions: list[dict]) -> dict`: keys exactlyemg/vision/fusion, strict sample-key pairing; each row haskey/hand/split/gesture_class/active/crop_present,22center error sums/counts, and`official_unmasked_mean_error_rad` computed by collector. Returns all-test andgesture-only ×all/crop-present class/family records for masked pooled and official unmasked equal-sample formulas.
- `verify_metric(current: dict, baseline: dict) -> None`: fixed1e-4degree/count-exact check; no parameter for relaxing tolerance.

- [x] Write hand-calculated tests for group overlaps, unequal denominators, null empty groups and wrist-only observations; and for paired per-class/family/overall sums and equal-sample means, raw/unassigned exclusions, crop/split views.
- [x] Write duplicate/missing keys, metadata mismatch, unknownclass, malformedstats, wrong baselinecounts and public-private sentinel rejection tests. Reordered models must produce the same aggregates.
- [x] Run tests and observe missing implementation RED; implement numerical code then actual GREEN.
- [x] Run relevant existingmetrics/audit/semantic tests, record exact results and commit.

### Task 2: One actual retained-checkpoint inference pass per model

**Files:** Create `scripts/paper/collect_joint_gesture_errors.py`; private immutable request/recorder/collector manifests underignoredtmp.

**Interfaces:** Consumes existing `audit.validate_artifacts`, `train.build_training_model/build_datasets`, original precision/batch/workers, retainedlast.pt embeddedvalidation-best. Produces private recordsJSON andpublic-safe identities/baseline agreements/whole-trajectory group table in an immutable output directory. Records include input-row/class/crop metadata privately; collector creates no optimizer and runsno_grad/eval.

- [ ] Inspect actual retained run paths and verify checkpoints/configs/runtime hashes before deployment; never rely on filenames as source identity.
- [ ] Implement collector to compute legacy aligned center, raw token midpoint diagnostic,22angle trajectory sums/counts, exact ordered sample/target/crop identities in one pass. For visual models trajectory is unavailable, not one-frame substituted as a trajectory experiment.
- [ ] Run source-stable CPU fixture or original model alignment checks appropriate to new collector, then stage once with real256GiB host/15GiB disk reserves and a genuinely available GPU. Capture actual PID/start identities and exits; no rerun solely on observer timeout.
- [ ] Run S/M/L/RN18/fusion selected models, verify baseline checks and exact sample/metadata/target/valid/crop joins. Source and checkpoints remain unchanged. Collect only actual successes.

### Task 3: Public tables and scatter

**Files:** Create `scripts/paper/render_joint_gesture_analysis.py`, `results/paper-analysis/*`; Modifyfull-paper-status/run-status evidence rows.

**Interfaces:** Consumes Task1 arithmetic and Task2 privately completed outputs, emits numeric/Markdown/CSV tables andPNG/SVGFigure12. No raw user/sample/row/path fields may be copied into publicJSON.

- [ ] Generate Table12 group records and59-class Table14/15/family data directly from unrounded sufficient statistics; sort unrounded Vision−Fusion gain, include explicitempty classes andview/protocol identity.
- [ ] Plot paired Vision/Fusion scatter with pointsizeN andverifiedfamilycolors from exactly the same table. State old population and native-formula differences, no PDFN/54of59/precision substitution.
- [ ] Independently recompute totals and weighted family results; inspectPNG, repeat render for byte identity, verify public fields/identities, update coverage and publish. Do not markfull-papercomplete.
