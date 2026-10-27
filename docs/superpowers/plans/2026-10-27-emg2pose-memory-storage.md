# Complete EMG2Pose memory storage implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Acquire and consume every byte/session of the official full EMG2Pose source on eeg-server without requiring simultaneous disk copies of TAR, HDF5 and memmap arrays, then run the three complete author experiments.

**Architecture:** Acquire exact, identity-bound HTTP ranges with bounded concurrency into an owned Linux memfd; hash and seal the complete object before exposing readers. A bounded pread file object exposes original HDF5 members, and an explicitly separate dataset storage adapter preserves the pinned official window/transform/sampler implementation. The existing persistent compressed download and old mini/conversion receipts remain intact.

**Tech Stack:** Python 3.11, requests, Linux memfd/seals/cgroup v2, tarfile, h5py 3.11, NumPy 1.26, pinned official Lightning/data factories.

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`, especially requirements 3/7 and “资源与运行”. This is a routine storage implementation under the already authorized full-paper execution; it does not redefine the scientific cohort or add a new approval gate.

## Global Constraints

- “EMG2Pose 数据集上的 S/M/L 实验，遵循其 20 维标签和独立数据协议，不能替换为小样本演示。mini 数据仅用于工程验收。”
- “存储不够时不得删除其他项目的数据。”
- “已完成实验不能因观察超时被重新启动。”
- “新实验写入独立输出目录。”
- Preserve pinned upstream `f7cbd048b7092139852270491d3859cd472e0311`; do not edit it, the existing converter, or the source files of active DINO runs.
- Builder decision: require observed initial memory for the full source, bounded buffers, and a 256 GiB reserve, considering every visible cgroup-v2 ancestor. Continue observing capacity during acquisition/read/hash; a check is not an OS reservation guarantee.
- Full source is 462,824,048,640 bytes. Strong ETag/Last-Modified identify the transport object; received SHA256 is not a publisher checksum. A complete byte cache alone is not a complete scientific corpus or experiment.
- RAM is volatile. No automatic source substitution, reduced cohort, modified mask, relaxed tolerance, or automatic training retry. Keep the persistent local xz receiver unchanged until the alternative is qualified.
- No production full allocation/launch before bounded end-to-end qualification; no other users' processes or files may be terminated/removed.

## Review Focus

1. Partial/changed/range-ignoring HTTP responses must fail without returning a ready cache (Task 1).
2. A tighter parent cgroup limit or declining free memory must refuse/stop owned acquisition and release its fd (Task 1).
3. HDF5 callbacks must not see neighbouring TAR members, writable/unsealed storage, or a shared seek cursor (Task 2).
4. IK failures, nonfinite angles, all-false fallback, jitter boundaries and session ordering must remain identical to the official converted reader (Task 3).
5. A lost cache-holder, stale PID/fd, failed child or resource pressure must never publish scientific completion or leave an unaccounted owned training child (Task 4).

## Approach and evidence

The existing local resumable xz receiver is correct but its single-flow compression/transfer is slow. A bounded server-fetch/compress/SSH test was slower, so that route was rejected. Full TAR plus converted arrays would consume about twice the source size and violate the chosen memory reserve. Direct server range probes verified the original ETag and matching existing chunk hashes; eight bounded ranges observed 6.72 MiB/s for 128 MiB, which is not a full-download throughput guarantee. A small memfd/seal probe succeeded. These are feasibility observations only (`tmp/emg2pose-memory-tar-feasibility.json`).

### Task 1: Complete byte-exact sealed-memory acquisition

**Files:** Create `egoemg_repro/emg2pose_memory_archive.py`; test `tests/test_emg2pose_memory_archive.py`.

**Interfaces:**
- Consumes unchanged `emg2pose_archive._head`, `_receive_range`, `_ASSETS` identity/protocol checks.
- Produces `available_memory_bytes() -> int`; `acquire_archive(asset: str, *, workers=8, chunk_bytes=16777216, reserve_bytes=274877906944, on_progress=None) -> SealedArchive`.
- `SealedArchive` owns one fully hashed, immutable memfd; supports a context manager, `fileno()`, `receipt`, and `close()`. No source/metadata path is written and no scientific completion is claimed. Internal `_acquire_source` permits controlled local-server engineering tests.

- [x] Write failing real-loopback HTTP tests: out-of-order ranges/short last chunk equal the complete source bytes; every range checked; final SHA256 exact; mutation prohibited by kernel seals; releasing context closes the owned fd.
- [x] Add failure cases for wrong ETag/Content-Range/200 response/interrupted body/final HEAD change; no ready object, owned fd closed on failures/callback exceptions.
- [x] Add capacity tests using real fixture proc/cgroup files: finite parent tighter than leaf, unsupported hierarchy rejected, initial budget refuses before allocation, mid-run pressure stops owned cache. Invalid numeric policy inputs fail before I/O.
- [x] Run tests RED, implement bounded download (at most `workers` responses/buffers retained), exact pwrite/readback/hash, repeated guard checks and seal verification; run GREEN with existing archive tests.
- [x] Record results/limitations in the ledger; commit only after required suite checks. Task1 code committed as `d6a40a5`; actual complete-project 1606/4, light874/689 and real mini receipt are published separately from scientific results. `task-done` additionally re-ran75checks successfully.

### Task 2: Original TAR/HDF5 range reading and full inventory

**Files:** Create `egoemg_repro/emg2pose_memory_data.py`; tests `tests/test_emg2pose_memory_data.py`; document the private API in `docs/emg2pose-memory.md`.

**Interfaces:**
- Consumes Task 1 `SealedArchive` and an inherited sealed descriptor.
- Produces `FdSlice(fd, offset, length)` with independent cursor and bounded read/readinto/seek/tell; `index_archive(fd, *, expected_bytes, expected_sha256) -> dict`; `inspect_archive(..., cohort, metadata_csv, upstream) -> dict`.
- Inventory binds the entire original source SHA, every physical member offset/size/hash, all original HDF5 objects/typed attrs, exact CSV session population/order, global row offsets and original IK/finite masks. A new format stays distinct from completed direct-memmap receipts.

- [x] First write RED cases for two concurrent cursors, slice boundary EOF, nonzero-offset real HDF5 fields/attrs, closed ownership and unsealed fd rejection.
- [x] Add hostile/ambiguous TAR cases: traversal, duplicate/normalized duplicate, links, sparse members, truncated member/trailer, missing/extra official filenames and wrong metadata/attrs.
- [x] Implement bounded original-member reads, preserve all nonconsumed fields in the sealed source, and stream original helpers for masks/blocks/field hashes; no materialized source-sized arrays or weakened old path guards.
- [x] Prove actual mini original member and HDF5 field bytes against existing source receipts; run relevant GREEN tests. Actual30HDF5/4,218,756rows,allglobal/session fields and native metadata arrays match; complete101source1651/4 and light919/689. Code committed as `cbee8fe`; native-environment task-done re-ran82checks without skips (earlier light-tier helper81/1dependency-skip is recorded separately). Native526window/model check remains Task3.

### Task 3: Storage-only adapter to the unchanged official data factory

**Files:** Create `egoemg_repro/emg2pose_memory_index.py`, `egoemg_repro/emg2pose_memory_dataset.py`, `egoemg_repro/emg2pose_memory_runtime.py`; corresponding `tests/test_emg2pose_memory_{index,dataset,runtime}.py`; extend the memory documentation. Separate index/admission from the importable official subclass so actual spawn can unpickle it without eager upstream imports in admission code.

**Interfaces:**
- Consumes Task 2 complete inventory and inherited sealed fd.
- Produces global-row field views and an explicitly named official-dataset subclass overriding storage open/read only; new config/source binding and complete runtime identity. Unchanged official metadata/block/window/transform/normalization/sampler methods remain authoritative.
- `prepare_index(fd: int, inventory: dict, output: Path, *, upstream: Path, metadata_csv: Path) -> dict` independently regenerates/compares Task2 inventory, writes only source-inventory.json/metadata.npz/manifest.json/completion.json in an atomic no-overwrite transaction, and returns the completion SHA plus content identity. It never writes source-sized arrays.
- `open_source(fd: int, index: Path, *, expected_completion_sha256: str, upstream: Path, metadata_csv: Path) -> VerifiedSource` validates the separately frozen completion SHA, all sidecars/helper pins/metadata, and actual sealed source SHA once. The context owns its own duplicate and detects closed/changed bindings; no global registry retains descriptors.
- `VerifiedSource.fileno()`, `.data_identity`, `.manifest`, `.handle_identity`, `.assert_current()`, `.close()` form the runtime proof/ownership boundary. Content identity excludes private paths, inode/fd and acquisition timing; concrete completion/file hashes remain bound in private instance provenance. Identical rebuilt caches can resume; a new handle must actually verify its source.
- `SealedTarDataset` is a top-level importable dataclass subclass of the pinned Emg2PoseDataset. Adds archive_fd and archive_handle_identity to the ordinary memmap_dir=index-directory parameters. Storage overrides expose bounded global-row field slices; open/close HDF5 per window initially. It owns a duplicate, supports close(), resets process-local views after fork and transfers the same sealed object with actual multiprocessing DupFd under spawn; no HDF5 objects are pickled.
- New `compose_config(upstream, size, index, *, source: VerifiedSource, metadata_csv)`, `build_runtime(upstream, config, *, source)`, `runtime_identity(upstream, config, *, source)`, `reload_runtime(upstream, config, state_dict, *, identity, source)` preserve the original factories/seed/norm/loss/optimizer configuration, replace only the dataset target/descriptor bindings, and require the live proof. `close_data_module(data_module)` closes owned leaf dataset descriptors after consumers finish, including shared ConcatDataset leaves. Existing direct-memmap validators are not broadened.

- [x] Write RED tests comparing all fields/windows/metadata to actual official converted fixtures, including both IK modes, invalid and all-false windows, adjacent sessions, filters, padding, jitter under restored RNG, norm and transformed tensors. Nineteen dataset cases and actual526mini windows agree with the unchanged original reader.
- [x] First prove index admission/lifecycle: no large dat arrays, failed/incomplete/no-overwrite transactions, mutated metadata/receipt/descriptor/source rejected, retained source works after caller closes, and reconstructed identical content preserves data identity despite changed paths/fd/acquisition-instance evidence. Eighteen index cases include the reproduced/fixed reused-fd ownership bug.
- [x] Implement compact mask/block sidecars and bounded per-worker HDF5 lifecycle. Bind exact source/metadata/helper/adapter/config hashes; forbid incomplete/mini receipts claiming full. Native factory28cases and all110targeted cases passed; no full-project or actual three-model success is inferred from them.
- [x] Complete actual 30-file mini inventory and compare all 526 official windows and full streamed input/target/mask digests, then actual optimization/full-state reload for S/M/L using the new reader. Actual qualified73119exit0: each2updates, all526evaluations exact in fresh process, native state restore andone further real update tostep3; allownedFDclosed/CUDAuninitialized. Never present mini as independent validation.
- [x] Run complete relevant suite, one final fresh-context review of the implementation, and record/fix justified findings once with RED/GREEN. Review on64b1885 found oneImportant finalHEAD pressure guard; two realRED cases→77GREEN. Final107sourceaf216 complete1,719/3CUDA skips and light939/691 capturedexit0; finalactual526window andS/M/L nativeCPU checks requalified. Task3 completion command: native environment pytest of test_emg2pose_memory_index.py, test_emg2pose_memory_dataset.py and test_emg2pose_memory_runtime.py (65cases), in addition to the already exited whole-project checks.

### Task 4: Guarded server cache lifecycle and three complete scientific runs

**Files:** New owned controller/CLI and integration tests, finalized after Task 3 interfaces are proven; update public results and full-paper status from actual evidence only.

**Interfaces:**
- Consumes qualified Tasks 1–3 and pinned complete source/metadata receipts.
- Produces owned cache/controller/child PID-start receipts; bounded resource observations; separate S/M/L complete selected-best train/val/test/replay/report artifacts.

- [ ] Write executable subprocess tests for inherited descriptor/source binding, holder loss, signals, child failure, memory pressure and no orphan/completion leakage. Define exact CLI/controller identity in a ledgered plan refinement before production code.
- [ ] Stage immutable tested source on server, verify all bytes, recheck current RAM/cgroup/disk/GPU capacity, qualify a bounded real acquisition/read/owned-child lifecycle.
- [ ] Start the complete source cache only after qualification; retain persistent xz progress. Verify all bytes, exact 25,253-source population and full mask/field inventory before any complete training claim.
- [ ] Qualify unchanged original S/M/L GPU batches/recovery/complete validation, then run all three complete 250-epoch experiments with independent official roles and selected-best final tests/replays.
- [ ] Publish actual complete results, Tables 11/13 and underlying counts. Keep the full-paper goal active while other required work remains.

## Self-review / execution

Tasks 1→2 share a sealed fd/complete byte identity; 2→3 share a separate complete inventory; 3→4 share a qualified native storage adapter. No existing direct-memmap validator is extended to accept another storage format. Task 4 launch details deliberately depend on measured Task 3 runtime behaviour and must be made exact before that task starts. Inline execution remains the established method; ordinary authorized storage choices do not require a new user approval.
