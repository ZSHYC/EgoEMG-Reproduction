# EgoEMG Reproduction Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Run traceable official-model baselines on the accessible legacy EgoEMG release and publish code plus actual run evidence.

**Architecture:** A small reproduction package handles selective downloads, a read-only legacy memmap adapter, guarded time partitions, and explicit metrics. Model construction and kinematics come from the fixed official checkout. Training uses independent validation, training-only statistics, recorded configuration, and persistent GPU jobs.

**Tech Stack:** Python 3.11, NumPy 1.26.4, PyTorch 2.2.2, torchvision 0.17.2, Hydra 1.3, pytest, official EgoEMG commit f7cbd048b7092139852270491d3859cd472e0311.

**Spec:** ../specs/2026-10-24-egoemg-reproduction-design.md

## Global Constraints

- Do not modify existing server environments, data, or unrelated jobs.
- Leave at least 15 GiB free on each destination filesystem.
- Never commit datasets, checkpoints, credentials, private server paths, or the paper PDF.
- Keep published-checkpoint verification separate from independent-validation retraining.
- Preserve official Gesture/User/Both test identities and record exact test centers.
- Split time ranges before windows; paired hands share partitions; disable training jitter initially.
- Convert radians to degrees exactly once; retain valid error sums and counts.
- Explicitly label legacy labels and whole-episode preprocessing limitations.
- Do not claim full paper-level reproduction from a short integration run.

## Review Focus

- Wrong HTML/error pages saved as multi-GB data files must be rejected by byte/schema checks.
- Overlapping context/label ranges crossing partitions must be rejected, including both hands.
- Missing wrist labels and unit conversion must not silently alter the 22-angle denominator.
- A mismatched upstream commit, model, or checkpoint must fail before a full run.
- Interrupted downloads and training must preserve usable partial work and provenance.

### Task 1: Selective assets and isolated runtime

**Files:** `egoemg_repro/download.py`, `assets_manifest.json`, `environment.yml`, `pyproject.toml`, `scripts/bootstrap_upstream.sh`, `tests/test_download.py`.

**Interfaces:** `python -m egoemg_repro.download --root PATH [--metadata-only]` downloads only listed public files, atomically validates expected bytes, and records SHA256. The fixed upstream checkout is supplied through `--upstream PATH` to later commands.

- [x] Write failure tests for incomplete/wrong-size files and metadata-only selection, observe failure.
- [x] Implement the public Drive download inventory with normal confirmation handling, resume, byte checks, storage reserve, and no auth secrets.
- [x] Run meaningful downloader tests and retrieve the first real metadata files.
- [x] Create a separate server environment and verify pinned imports, CUDA, and bf16.
- [x] Start selective data acquisition in a persistent project session with logs.

### Task 2: Legacy adapter and guarded partitions

**Files:** `egoemg_repro/data.py`, `egoemg_repro/partitions.py`, `tests/test_data.py`, `tests/test_partitions.py`.

**Interfaces:** `LegacyStore(root)` exposes read-only `fields`, `episodes` and `field(name)` arrays. Episode records expose `index`, `id`, `subject`, `start`, `end`. `build_partitions(store, window_length=12000, train_stride=1200, eval_stride=12000, val_fraction=0.12)` returns a JSON-serializable manifest containing `windows` records (`episode`, `start`, `end`, `center`, `role`, `split`) and provenance. `LegacyDataset(root, windows, norm=None)` returns paired-hand samples with `emg:(8,W)`, `joint_angles:(22,W)` in radians, `label_valid_mask:(W,)`, `angle_valid_mask:(22,W)`, and metadata. Normalization is per-hand 8-channel mean/std computed only from retained training intervals.

- [x] Write tiny real-format fixture tests for metadata aliases, per-wrist masks, degrees-to-radians conversion, and clipped episode boundaries; observe failure.
- [x] Implement manifest-driven memmaps without loading unused large arrays, preserving existing labels and avoiding upstream v2/v3 assumptions.
- [x] Write partition tests proving fixed official test centers, no cross-role footprint overlap, paired-hand equality, and deterministic short-run handling; observe failure.
- [x] Implement deterministic validation intervals inside official train runs, purging train windows that overlap validation or fixed official test footprints.
- [x] Record coverage, discarded windows, and preprocessing limitations; verify on real metadata once files arrive.

### Task 3: Official model training and explicit evaluation

**Files:** `egoemg_repro/train.py`, `egoemg_repro/metrics.py`, `configs/emgformer_s.json`, `tests/test_metrics.py`, `tests/test_training.py`.

**Interfaces:** `python -m egoemg_repro.train --data PATH --upstream PATH --output PATH --config configs/emgformer_s.json [--max-steps N]`. Read the Task 2 data interfaces. Construct the model from official Hydra configurations, seed before model creation, align official TDS context, and use the original angle L1 plus 0.01 generic-UmeTrack fingertip distance. Save configuration, source/data/partition hashes, optimizer state, best validation checkpoint, last checkpoint, and machine-readable progress.

- [x] Write degree/valid-count aggregation and unequal-group-weight tests, observe failure.
- [x] Implement error-sum/count accumulation by hand/split/user with per-angle validity and explicit center/trajectory modes.
- [x] Write finite one-step/reload tests around the upstream model using a tiny local fixture, observe failure where dependencies permit.
- [x] Implement AdamW and cosine schedule from the selected current recipe, explicit independent validation, resume, bounded integration mode, and deterministic evaluation.
- [x] Run a real-data forward/backward check, a bounded training run, and checkpoint reload before full training.

### Task 4: Visual crops and matched baselines

**Files:** `egoemg_repro/crops.py`, `configs/resnet18.json`, `configs/fusion_rn18_s.json`, visual dataset/training extensions and focused tests.

**Interfaces:** Read per-episode official LMDB crop keys `{video_frame_idx:08d}_L/R`; retain only fixed, manifest-selected centers in a compact cache. Reuse the same partitions, units, valid denominators and evaluation centers as EMG. Use official ResNetVisionPose and MidFusionPoseFormer model implementations.

- [x] Verify a real LMDB schema and map EMG centers to source video-frame indices before full crop acquisition.
- [x] Add missing-crop and paired-center tests, then implement compact center-crop caching.
- [ ] Run bounded real vision and fusion training, then full baselines with clean initialization checkpoints.
- [ ] Report paired evaluation population and missing-crop counts explicitly.

### Task 5: Integration, review, and published evidence

**Files:** `README.md`, `docs/run-status.md`, `results/*.json`, `.github/workflows/tests.yml`, project scripts.

- [x] Run the meaningful full reproduction-package test suite, dependency checks, and real GPU integration checks.
- [x] Obtain independent code/scientific-protocol review and resolve material findings before publishing implementation.
- [x] Commit and push tested code to the user-authorized public repository; verify remote commit and public visibility.
- [x] Launch persistent full experiments only after the corresponding integration check passes; record PID/session/logs privately and publish sanitized aggregate status.
- [ ] Publish completed measured results with all provenance; identify running, blocked, or unavailable experiments honestly.

## Execution ruling

The user explicitly approved this design and requested data preparation and training. Execute without another plan-approval round. Independent modules may be delegated concurrently with disjoint file ownership; the root agent owns integration, server actions, commits, and final review. This concurrency follows the session's proactive-delegation instruction.

### Task 6: Paper guide and coverage audit

**Files:** create `docs/paper-guide.md` and `docs/reproduction-matrix.md`.

- [x] Explain paper methods/data/training/tables from the supplied PDF and pinned source, citing pages and code.
- [x] Identify each accepted experiment and remaining verification artifact; distinguish strict-paper equivalence from released-compatible experiments.
- [x] Review scientific claims, units, architecture shapes, recipe differences, and source links.

### Task 7: Repository presentation

**Files:** update `README.md`; create original `docs/assets/banner.svg`, `docs/assets/architecture.svg`, and navigation/citation material as useful.

- [x] Build a restrained academic layout with honest running status, clear quick start and navigation.
- [x] Render and visually inspect SVGs and the README preview; verify links and command blocks.
- [x] Preserve source attribution and avoid any private data or paper-figure redistribution.

### Task 8: Independent result exporter

**Files:** create `egoemg_repro/report.py`, `tests/test_report.py`, `docs/results.md`; optional reporting extras in `pyproject.toml`.

**Interfaces:** `python -m egoemg_repro.report status --run RUN --output DIR` writes a sanitized current-run summary and training/validation curves. `... export --run RUN --output DIR --model-label NAME` accepts only full completed runs with selected-best provenance, and writes sanitized JSON/Markdown plus plots. Status does not read the test set or checkpoint weights; export must reject bounded/mismatched/incomplete evidence.

- [x] Write meaningful rejection/aggregation/privacy tests, observe failure, then implement the independent reporter.
- [x] Render plots with standard plotting tools; keep degree conversion and sampling definitions unchanged.
- [x] Check runtime/recipe bytes against the running implementation before integration.

### Task 9: EMGFormer-M comparison

**Files:** create an explicit new recipe after inspecting its pinned official experiment; server run artifacts remain private.

- [x] Verify official medium architecture/config, real-data finite GPU step, bounded training and checkpoint reload.
- [x] Launch a separate full run on an available GPU, using the same frozen cohort/partitions/normalization and complete validation selection.
- [ ] Publish complete metrics only after training and final test finish.

### Task 10: Goal completion audit

- [ ] Finish every required full experiment and publish the measured results with source/data/config/partition/normalization/initialization identities.
- [ ] Verify the result exporter against actual selected checkpoints and repeated evaluation; compare matched center populations and explicit error sums/counts.
- [ ] Obtain independent scientific/code/publication review, pass local/hosted checks, and inspect public repository artifacts.
- [ ] Derive a requirement-by-requirement completion audit from both approved designs and this plan. Keep the goal active until all evidence is complete.

### Task 11: Locked-checkpoint metric audit

Checked implementation items below are supported by fixture integration and independent review; the last item requires actual completed experiment artifacts and remains open.

**Files:** new `egoemg_repro/audit.py`, `tests/test_audit.py`, documentation under `docs/results.md` or a dedicated audit guide.

- [x] Keep every executed training file/recipe unchanged. Validate completed run, selected-best checkpoint, source/config/data/partition/norm/initialization identity before an audit.
- [x] Recompute center predictions using frozen official alignment, target masks, precision and evaluation grid; compare all-label baseline MAE against the completed run's saved center result.
- [x] Accumulate error sums/counts for hand/split/user intersections and compute explicitly pooled versus per-user mean/population SD. Add the identical actual crop-present subset for EMG, vision and fusion, preserving missing counts.
- [x] Publish only sanitized metric identifiers/hashes, anonymous user groups and aggregate center sets. Test grouping, weighting, units, mismatched evidence, bounded/incomplete rejection and privacy before implementation.
- [ ] Verify on actual completed checkpoints and record this additional deterministic verification pass; never tune or choose a checkpoint based on audited test accuracy.

### Task 12: Public crop transport recovery

**Files:** new `egoemg_repro/range_download.py`, `tests/test_range_download.py`, recovery documentation in `docs/download.md`. Existing executed acquisition/training modules remain byte-identical.

- [x] Diagnose official Drive HTML responses without treating them as binary data; preserve inspected erroneous partials separately from genuine binary resumable work.
- [x] Implement explicit sequential HTTP Range recovery against the same public asset IDs, requiring exact byte range/total/body-size, bounded chunks, no account cookies/auth/proxies, existing capacity and final receipt checks.
- [ ] Independently review rejection/resume/quarantine tests and run a real missing-episode acceptance before a persistent full recovery.
- [ ] Complete and verify all 41 LMDB payloads before the real visual/fusion driver acceptance and matched-population final audits.
