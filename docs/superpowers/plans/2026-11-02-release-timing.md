# Released Timing Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. One fresh review of this new analysis unit after actual full-data execution; one focused material fix pass. Do not reopen closed model/geometry/math reviews.

**Goal:** Complete the available legacy corpus's clock-field and webcam alignment diagnostics over all66,161,725rows/41episodes, preserving quantization, missingness and raw recorded deltas.

**Architecture:** A NumPy-only reader scans six immutable arrays in bounded chunks and derives exact integer histograms, counts and histogram-weighted quantiles. A fresh independent process rehashes/rebuilds the full report and verifies deterministic artifacts. This is D02/D03 descriptive analysis, not raw sensor synchronization reconstruction or a model/data change.

**Tech Stack:** Existing Python3.11/3.12, NumPy, standard library and unchanged `_StatGuard`; no Torch, new dependencies, GPU or network in the analyzer.

**Spec:** [Approved whole-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements1/4/5/7 and current D02/D03. All user authorization and inline execution choices persist; no new scientific method or user preference is introduced.

## Global Constraints

- Preserve the14scientific settings, all10live workflows, selected checkpoints and raw inputs. No new filtering, masks, timestamp repair, training or download.
- Fixed upstream f7cbd048b7092139852270491d3859cd472e0311: migration docs warn that `timestamp_us` has repeated/quantized values and episode resets; nominal model time is the2kHz row index. Describe raw differences; never use them as model dt or measured hardware synchronization error.
- Scan `timestamp_us`/`episode_index`/`frame_index` int64, `image_webcam_frame_index`/`image_webcam_delta_ms` int32, and `image_webcam_stale` bool, exact manifest shapes/sizes; inspect stale raw bytes to reject nonboolean encodings. Total six payload bytes2,183,336,925; full scope requires66,161,725rows/41contiguous metadata episodes.
- Missing frame index/stale and INT32_MAX delta candidates remain separately counted. Publish all recorded delta values and explicit stale/metadata-usable subsets; no silently discarded sentinel, inferred crop availability, video FPS, original IMU rate or physical accuracy claim.
- Consecutive differences cross chunks inside an episode, never episode boundaries. Exact counts/integer sums/histograms conserve globally and byepisode. Quantiles use linear interpolation at integer percentile ranks, preserving null for empty populations.
- Guards precede all hashes, retain source/input generations through export/verify and protect output directory identity/membership while reading. Fresh output only; no private user names, absolute collection timestamps or machine paths in public results.
- IMU, original device clocks, full videos/ZED and raw interpolation/linear synchronization remain incomplete. Reading already aligned fields cannot close those requirements or the whole paper.

## Review Focus

1. Chunk/episode boundaries must not drop or fabricate adjacent pairs, including singleton episodes and invalid image indices.
2. Source-defined stale bits and raw delta fill candidates must not be collapsed into valid-camera or zero-error claims; integer extrema must not overflow.
3. Weighted histogram quantiles/counts must retain repeats and true denominators; no average of episode means or invented sampling rate.
4. Metadata/typed-file/source changes, symlinks and output generation races must reject rather than publish mixed generations.
5. A small fixture/manifest must not receive full-corpus status, or descriptive timestamp diagnostics receive hardware synchronization/model-performance eligibility.

### Task 1: exact statistics and full source-bound reader

**Files:** create `egoemg_repro/release_timing.py`, `tests/test_release_timing.py`.
**Interfaces:** `timing_statistics(arrays,boundaries,chunk_rows)->dict` pure diagnostic with no scientific permit; `scan_release_timing(root,upstream,require_full=True,chunk_rows=1048576)->dict` actual source-bound report, default complete legacy only.

- [x] Write tests for two handwritten episodes with ties/backward timestamps, stale/index cross-cases, sentinel and chunk1-vs3 equality; exact counts, sums, p50/p95 and null populations. Add malformed dtype/layout/boundary/episode-id/stale-byte/overflow rejections and partial-as-full rejection. Run missing-module RED.
- [x] Implement exact histogram reduction and source-bound readonly full scanner; source guard before every hash. Preserve original row-index time semantics and separate full/fixture scope.
- [x] Run only new tests without skips and commit source/tests. Expected: allnewcasespass, no ML import or scientific model run.

### Task 2: deterministic export, actual full scan/rebuild and publication

**Files:** same source/tests; create `results/dataset-statistics/release-timing/{metrics.json,episodes.csv,histograms.csv,README.md,completion.json}` and `results/dataset-statistics/release-timing-validation-20261102.json`; update D02/D03 status at true scope.
**Interfaces:** `export_release_timing(root,upstream,output,require_full=True)` and `verify_release_timing(...)`; CLI `python -m egoemg_repro.release_timing export|verify --data PATH --upstream PATH --output PATH`.

- [x] RED then GREEN: immutable/fresh outputs, exact CLI reconstruction, input change during hashing/artifact generation, missing/changed artifacts and added directory entry during completion read. No changed source/model code outside this component.
- [x] Execute two separate owned CPU full-data processes, export and independent verify. Rehash all six complete payloads/metadata and require actual66,161,725rows/41episodes, typed byte-identical results, zero ML/CUDA and source generations unchanged. Inspect actual distributions and preserve all output/source hashes; check original owning processes have closed.
- [x] One fresh final review of this new component, one material RED→GREEN fix pass if needed, then repeat only affected new-unit checks/actual export. Do not mark original hardware synchronization, missing IMU/video, full Table1 or full paper complete.
- [x] Publish exact tested source/artifact bytes and independently verify remote ref plus allchanged anonymous immutable raw files. Existing core jobs continue; no restarts or new GPU jobs.
