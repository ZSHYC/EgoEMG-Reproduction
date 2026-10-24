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
