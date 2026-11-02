# Native Joint Group and Weighting Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. The approved full-paper design already authorizes this analysis. One final review of this new component after its engineering checks, not another review of closed models or legacy analyses.

**Goal:** Produce source-bound native S/M/L Table12 joint-group analyses, distinguishing the official code's valid-angle reduction from an explicitly labeled interpretation of the PDF's split sample weighting.

**Architecture:** Reuse the closed native full-trajectory packets and the existing `joint_group_table` arithmetic. A pure reducer retains all requested windows and emits hand/split/group counts and two distinct aggregation views. A separate read-only collector authenticates the completed source graph, reduces actual native/replay packets without constructing a model, and publishes only aggregate records after independent readback and owned process closure.

**Tech Stack:** Existing Python standard library; existing NumPy/Torch CPU raw-packet reader only in collection; no installs, training, CUDA, or learned forwards.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements1/4/5/7. The completed legacy joint/gesture plan remains closed; these are new native200 sources and missing split/group weighting evidence.

## Global Constraints

- Official source remains `f7cbd048b7092139852270491d3859cd472e0311`; completed source fits, checkpoints, masks, normalization, precision, old reports and running jobs remain unchanged.
- Actual native sources are full200/25000 S/M/L, point-validation BEST169/124/142. Use full1052 validation/2792 test lineage and the original BF16 predictions, not new FP32 inference or new test selection. Alignment is12000 input/left510/right0/11490 output; derived center6255 is not old point6000.
- Table12 has10 groups and no across-user SD. Existing `JOINT_GROUPS` is authoritative. Groups overlap and are not additive.
- Pinned `_evaluate_egoemg_pooled` accumulates masked error sums and valid-time×angle counts over left/right and splits, not the number of requested windows. Its output is22-angle overall; `_evaluate_egoemg_per_group` exports per-user groups separately by hand/split. Neither is a direct published10-cell Table12 row. Do not invent a missing author reduction.
- The primary native view is valid-angle-time pooled. A separately named caption interpretation is `sum_s(N_s * pooled_group_mae_s) / sum_s(N_s)`, with `N_s` counting every requested hand/window, including fully invalid requests. This is an explicit interpretation, not proof of the authors' unpublished table script. If a nonempty split has no valid group observations, that caption group is null, not silently dropped or zero-filled.
- Public numeric outputs contain aggregates and hashes only: no private sample keys, row indices, subjects, source paths, or per-window error arrays. Preserve all requested windows; never filter requests for nicer agreement.
- Reuse existing native typed packet validation and physical SHA guards. Bind the completed independent postexit receipt, original report, source generation, producer/loader closure, native/replay population and actual selected fit. A raw tensor file alone grants no full/source eligibility.
- No repeated old training, full input qualification, model mathematics, mapping tests or closed family reviews. New raw reads must serve the new split/group reduction.
- Full-paper completion and PDF population/precision/numerical equivalence remain false; an arithmetic result cannot certify a source run.

## Review Focus

- Unequal valid-time counts and fully invalid requests must distinguish valid-angle and requested-window weights (Task1).
- A populated split with an unobserved wrist/group must retain its requests and make that caption group undefined (Task1).
- Duplicate sample keys, bad split/hand metadata and malformed22-angle statistics must fail rather than trim the population (Task1).
- Replaced source proof, changed raw generations, partial/reordered raw packets, live producer or forged engineering/full flags must fail admission (Task2).
- No private fields or sample rows may survive into public reports; a pending reader process must not be represented as closed (Tasks1–3).

## Task 1: Pure native split/group weighting

**Files:** Create `egoemg_repro/native_joint_analysis.py`; Test `tests/test_native_joint_analysis.py`.

**Interfaces:** `native_joint_group_views(records: list[dict]) -> dict`. Each private row has unique nonempty string `key`, `hand` left/right, `split` gesture/user/both, and the existing22-element `trajectory_angle_error_sum_rad`/`trajectory_angle_valid_counts`. Extra private fields are ignored, never copied. Return global `n_requested`/`n_with_any_valid_angle`; `pooled_valid_angle_time`; `by_hand`, `by_split`, `by_hand_split` cells (each with counts and `statistics=joint_group_table(rows)`); and `caption_interpretation_split_sample_weighted` containing all10 group weighted numerators, total requested denominator and missing nonempty split names. Empty cells have null MAE; arithmetic grants no source completion.

- [x] Write analytic tests with per-angle `(sum,count)` rows user `(1,1),(1,1)`, gesture `(27,9),(0,0)`, both `(0,2)`. Expected pooled29/13rad, caption1.6rad, N5/valid4; test request conservation over hand/split cells, order independence and unchanged input.
- [x] Add missing-populated-wrist, empty-split/empty-input, duplicate keys, invalid hand/split, malformed statistics and private sentinel tests. A clean subprocess must import the reducer without NumPy/Torch.
- [x] Run the new tests and observe RED before implementation; then implement using existing `joint_group_table`, retaining distinct weights, and rerun only the new tests to GREEN.
- [x] Record results and commit this independently testable arithmetic unit; it does not complete native Table12 collection.

Verification: `/home/zshyc/EgoEMG/.venv/bin/python -m pytest tests/test_native_joint_analysis.py -q` — expected and actual20passed/0skipped. Initial18missing-module RED failures and2finite-overflow RED failures were observed before the corresponding implementations. Old mapping/model suites were not rerun.

## Task 2: Authenticate and reduce the actual closed raw graph

**Files:** Create `scripts/paper/collect_native_joint_analysis.py`; extend `tests/test_native_joint_analysis.py` or a focused new collection test file. Keep source/owned deployment requests private under ignored scratch.

**Interfaces:** Collector consumes explicit `--postexit-root`, `--size small|middle|large`, `--metrics`, `--proof`, `--expected-proof-sha256`, `--upstream`, and a fresh `--output`. Full is the default; explicit `--scope cpu_engineering` accepts only non-scientific engineering source receipts. The separate `verify --private-records --pending --expected-private-sha256 --expected-pending-sha256 --output` entry uses a fresh standard-library-only process, rejects a live producer, and recomputes public aggregate cells independently of the primary reducer. It writes private per-window sufficient statistics/source generations and public-safe aggregate JSON. A completed source identity and analysis-reader completion are separate states.

- [x] Write failing engineering checks for proof/report hash substitution, changed raw bytes/generation, engineering falsely labeled full, packet gap/order/repeat/hand/split mismatch, incomplete source/actor closure, and output privacy. A malformed proof must fail before raw tensors or model constructors can be reached.
- [x] Verify expected public postexit proof SHA; its actual readback file SHA and original report bytes/logical seal; collector bundle/source; same fit/BEST/full200/25000; actual original size and extra-reader stages exit0 with physical PID/start closure. Check all newly consumed source generations before/after.
- [x] Read each original test-native and test-replay packet with the existing `read_trajectory_packet`. Bind physical raw hashes to the accepted readback, verify complete contiguous2792 indices/repeat0/left-first/right-paired metadata and typed original targets/input fingerprints. Compute each sample's22-angle sums with float64 subtraction, as existing trajectory statistics do, preserving all masks and requests.
- [x] Require native/replay sample identities, sums/counts and resulting views to agree. Recheck existing overall/hand/split/22-angle/10-group baseline counts exactly and MAE with the existing fixed `verify_metric`1e-4degree contract; no tolerance options. Keep complete private sample maps and aggregate source signatures.
- [x] Run true raw engineering fixtures with model/optimizer constructors denied and CUDA uninitialized; independently recompute aggregate cells from private sample statistics in another standard-library process. Retain explicit engineering scope, so these fixtures grant no scientific/source eligibility.
- [x] Perform exactly one final fresh-context review of this new component, one material fix pass if required, and its relevant GREEN checks. Commit before staging the actual source-bound server reader; do not request another review of closed legacy/core components.

Verification before final review: actual owned server run47passed/0skipped in17.30s on the declared existing CPU environment; reader/guardian/controller/recorder all physically closed, source hashes unchanged. Source/direct typed raw checks, actor/source substitution, cross hand/split baseline and request-population signatures are covered. Single final review found the actual private/public user alias mismatch: realistic fixture6RED/45passed, minimal exact original alias projection, then the whole new component selection51passed/0skipped17.66s on a new frozen server snapshot with all4owners physically closed. No second review; actual full three-model analysis remains pending.

## Task 3: Actual S/M/L analysis and public Table12 view

**Files:** Create `results/native-joint-analysis/` aggregate JSON/CSV/README and source/completion proof; update `docs/full-paper-status.md` Table12/D14 as evidence permits.

- [x] Wait for actual L full seven-stage and independent postexit closure; export the accepted L report/proof once. Do not replace it with the old point or legacy result.
- [ ] Stage the reviewed collector with fixed sources and all three actual accepted proofs. Use CPU-only owned stages, bounded threads/memory/disk and independent standard-library aggregate readback. Capture worker/guardian physical closure and complete source/record/report hashes; no new learned forward/update.
- [ ] Require the three models' original request/hand/split/target identities to match before one three-column table. Publish10 group rows for the primary pooled view and the separately named caption interpretation, with split request weights/valid-angle counts and original PDF reference numbers. No author-equivalence assertion or controlled legacy-to-native improvement claim.
- [ ] Verify public schema/privacy, baseline conservation and actual completion receipt; preserve prior artifacts, publish by authorized fast-forward and verify changed public bytes with rate-limit-aware reads. Keep all remaining whole-paper requirements active.
