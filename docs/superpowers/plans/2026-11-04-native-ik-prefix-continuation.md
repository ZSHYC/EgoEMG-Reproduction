# Native IK Prefix Continuation Implementation Plan

**2026-11-05 resource override:** the user prohibits physical GPUs0/1. Do not
execute the original GPU1 deployment step below. The prepared launcher was never
started and is disabled. Actual qualification and backend binding on an allowed
GPU2–7 are required before a new full continuation; the old68 batches remain
read-only historical inputs. See [repository GPU rule](../../../AGENTS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline, task by task. Existing full-paper design and explicit resume authorize this source-faithful operational recovery. One fresh integration review after the new unit is complete; no repeated old model/IK mathematics review.

**Goal:** Complete the original132,323,450-row IK workflow while preserving its68 genuinely completed batches and actual producer lineage.

**Architecture:** An admitted, generation-bound old prefix remains read-only. A new workflow computes missing original batches, publishes an explicit mixed-producer packet map, independently reconstructs all old/new packets, and uses the unchanged native finalization/assembly helpers.

**Tech Stack:** Existing NumPy1.26.4/Torch2.2.2 and admitted source/assets; no server dependency installation or shared runtime mutation.

**Spec:** [Prefix continuation addendum](../specs/2026-11-04-native-ik-prefix-continuation.md), implementing Task3 of [the original full-data plan](2026-11-01-native-mano-ik-full-data.md).

## Global Constraints

- Originalf7cbd048; six contiguous partitions/hand, right0..5 thenleft0..5, defaultbatch10096/stride1/max200 original outer iterations and exact remainder membership.
- Preserve66,161,725 rows/hand,132,323,450 total; zero poses retained; no valid-mask substitute, changed optimizer/grouping or source-label replacement.
- Keep all four existing input/runtime/results/CLI files byte-identical. New code has its own snapshot, never a retroactive original qualification.
- Full prefix is the curated originalright/shard0/batches0..67,686,528 rows; original packets and actor identities remain unchanged. The incompletebatch68 remains excluded and preserved.
- Original closedB10096 qualification, exact physicalGPU1/runtime/source/assets; genuinely idleGPU and64GiB start headroom,225GiB disk/128GiB host; foreign processes untouched.
- Engineering remains explicit, no fake scientific permit. Full wrist/markers2mano/PDF annotation equivalence remain unfinished.

## Review Focus

1. A mixed aggregate actor must not replace actual old/new per-packet producer identities; Task2 tests mutate both roles independently.
2. A generation may change between initial prefix audit, resource wait and packet consumption; Task1 tests same-sized replacement/hardlinks and Task2 tests late handoff drift.
3. A complete later packet after a hole, a partial NPZ/JSON pair or a forged audit seal must not be adopted; Task1 tests exact prefix coverage and full-proof pinning.
4. Prefix reuse must never alter the original coupled L-BFGS batch membership or omit zeros/tails; Task1 uses original selector indices and Task3 actual engineering retains every frame.
5. A stopped verifier, wrong physical GPU or unclosed old producer cannot publish labels; Task2 negative lifecycle/backend tests and Task3 physical closure audit cover this.

## Task1: authenticated read-only prefix

**Files:** create `egoemg_repro/native_ik_continuation.py`, `tests/test_native_ik_continuation.py`.

**Interfaces:** `admit_ik_prefix(inputs, original_root, *, proof, engineering=False) -> IKPrefix`; `IKPrefix.identity` returns a deep independent dict; `.check()` validates registration/live providers/generations/closed source actors; `.batch_prefix(hand, shard) -> tuple[Path,...]` returns only the contiguous admitted original packet directories. Full mode requires exact pinned real proof and actual full registered inputs. Explicit engineering can use smaller real-file fixtures but always retains false full/scientific flags.

- [x] Write RED tests for missing implementation, complete ordered prefix against the original selector, defensive copies, old live/swapped actors, missing JSON/NPZ, duplicate/reordered/holey/extra complete batches, copied or mutated identity/provider, same-size replacement/hardlinks, optional-mask appearance, and forged/resealed full audit. A structural synthetic carrier tests engineering I/O only.
- [x] Run only `python -m pytest -q tests/test_native_ik_continuation.py`; observe the intended missing-module/input failure before implementing admission.
- [x] Implement registered prefix/guards using unchanged `inspect_ik_batch`, `select_ik_batch`, `require_closed` and existing source guards. Never materialize output labels or touch old packet generations.
- [x] Run the same focused suite GREEN, record actual counts/skips and commit. Do not rerun original optimizer/math tests.

## Task2: strict continued workflow and separate geometric replay

**Files:** create `scripts/paper/run_native_mano_ik_continuation.py`, `tests/test_native_ik_continued_workflow.py`; extend only the new continuation module/tests from Task1.

**Interfaces:** `run_continued_workflow(configuration, original_root, prefix_proof, output)` creates a new root and source-bound configuration; workers produce new packets, verify all mixed packets, finalize and assemble through original helpers. The new producer summary has `receipt_actor` and an ordered per-batch `producer_actor`, `directory`, result/state hashes and a reused/new marker. `read_continued_shard(inputs, directory, *, configuration_sha256, receipt_actor, prefix)` validates the exact map and returns immutable original/new producer-directory pairs for the fresh verifier. It cannot turn an aggregate actor into a packet producer or reuse an old receipt as a new role.

- [x] Write RED for changed batch keys/configuration/runtime/source, false reused/new flags, swapped packet/aggregate actors, missing old/new packets, same-state false producer claims, late source drift, existing output, stopped/failed worker and forged completion. Preserve the original CLI rejection of mixed actor receipts.
- [x] Implement a Torch-free parent with actual owned stages and physical/resource/source gates. New production calls only unchanged `produce_ik_batch` for missing batches; fresh verification calls unchanged `verify_ik_batch` for every old/new packet with optimizer construction/steps denied.
- [x] Require exact full ordered pairs before unchanged `finalize_ik_shard`, then all twelve original shards before `assemble_ik_labels`; report reused/new counts/work separately. Bind new wrapper bytes separately from the reused original qualified scientific snapshot.
- [x] Run `python -m pytest -q tests/test_native_ik_continuation.py tests/test_native_ik_continued_workflow.py` GREEN, inspect actual output and commit.

## Task3: real mixed-lineage engineering, review, full continuation and publication

**Files:** evidence under `results/native-mano-ik/`; update this plan, original full-data plan and full-paper status.

- [x] Run a genuine small real-pose CPU engineering prefix in a separately owned process, then a new continued producer and fresh geometric reader over all six original logical shards/hand. Choose a declared small source with a within-shard partial prefix and a final partial batch; no source-data replacement, synthetic optimization or full10096 claim. Preserve actual old/new per-packet identities and read original geometry/labels byte-exact after all actors exit. Evidence: `results/native-mano-ik/mixed-prefix-native-engineering-20261105.json`; actual source1980851,62rows/24packets/1reused+23new,108owningactors+collectorclosed.
- [x] Perform ONE fresh review of this new whole integration against the spec, Task1/2 tests and real engineering. Resolve material findings in one observed RED→GREEN pass; do not re-review original closed mathematics or fake final qualification from engineering. OneImportant stable-actor handoff fixed atd22ac5b;2RED→83GREEN, actualserver83/0skip, projectNoTorch2784passed/1929skipped. See `results/native-mano-ik/continuation-integration-review-20261105.json`.
- [ ] Independently re-open the genuine full68-prefix/source/qualification once under the accepted new code. Register a detached new workflow waiting for the same qualified idleGPU1; confirm actual PID/start/resource/source bindings, not just a lock file or requested launch.
- [ ] Observe the same workflow through all original batches, all fresh geometry, complete bilateral labels and postexit closure; publish aggregate actual errors/zero populations/work and source/label hashes. No completion from queue registration or a partial producer. Keep old failures, partial files, masks and original labels.

Self-review: the two code interfaces are consistent; Task1 owns exact prefix admission/generations, Task2 owns role attribution/full ordering and physical lifecycle, Task3 owns actual native execution, independent geometry and final closure. No new model or scientific protocol is introduced; full-paper scope is unchanged.
