# Bounded Range Recovery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. This repairs the already authorized full-paper data acquisition; no per-task agents or repeated design approval.

**Goal:** Resume the entire reviewed 37,778,344,975-byte geometry selection despite temporary TLS disconnects, preserving every existing strict byte gate and partial.

**Architecture:** Add an external `BoundedRangeDownloader` around the unchanged `StrictRangeDownloader`, then use it in a new owned attempt of the existing eight-asset intake. Keep original failed attempts and library bytes. Code acceptance requires deterministic fault tests and actual renewed committed progress; full geometry completion remains a separately observed requirement of the active full-paper goal.

**Tech Stack:** Existing Python, requests, original Asset/download_one/strict range adapter, one owned local process and real disk reserve.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements4/6/7 and resource/source-identity rules. Actual failure: geometry intake exit1 on SSLEOF after 45 new verified 16MiB chunks; old prefixes remain intact.

## Global Constraints

- Preserve officialf7cbd048b7092139852270491d3859cd472e0311, all training/checkpoints, data selection and current owned server jobs.
- Do not edit `download.py` SHA5063fc0a5a0dbb1929c054a9cac86d3f90c01b375171660b6d75e2707e684e75 or `range_download.py` SHAa09919cf9f312570c9121eb8b63a43043ee6c00aa4786ee2f0f3f6004e8e1fe5.
- Keep exact206/Range/total/length/binary/identity-encoding checks, TLS verification, no ambient auth/cookies/proxies, fresh sessions and15GiB real free-disk reserve.
- Three attempts at one stalled committed offset; delays1then2seconds. Reset only after complete strict chunks advance the offset. No configurable unbounded retry or cohort reduction.
- Retry connection/timeouts/chunked transport interruption and the observed TLS unexpected-EOF case. Certificate/proxy/other SSL errors, protocol/data/resource errors and cancellation propagate without retry.
- Only validated complete chunks reach the partial; only original full-size/hash/receipt gate promotes a final file. No new scientific result from acquisition engineering.

## Review Focus

- Streaming exception after a partial response: no unverified bytes appended or falsely completed receipt.
- Many separate offsets fail twice: budget resets on verified progress, never creates a fourth request at one stalled offset.
- Certificate failures and false binary/range/length headers: fail immediately, no backoff hiding them.
- Partial shrinks/replacement/changes during backoff, cancellation or evidence callback failure: stop rather than consume a competing writer's state.
- Real renewal retains old failure/quarantine bytes, complete eight-asset selection and source hashes; live process/progress is not full geometry completion.

### Task 1: External bounded recovery and original transport fault tests

**Files:** Create `egoemg_repro/range_recovery.py`, `tests/test_range_recovery.py`.

**Interfaces:** `BoundedRangeDownloader(asset, *, chunk_bytes=CHUNK_BYTES, session_factory=None, sleep=time.sleep, on_retry=None)` is callable as `(drive_id, part)` for original `download_one`. `statistics` returns a copy of transient failure/scheduled retry/exhaustion/last-offset counters. `on_retry` receives only offset, consecutive failure number, will-retry flag, delay and exception type. Limits are fixed3/1/2. All native transport/writer behavior stays delegated.

- [ ] Add deterministic real-writer/HTTP-boundary cases: handshake EOF, streamed ConnectionError, progress-reset, exhaustion, permanent SSL/proxy/status/range/type/length/body faults, exact arbitrary-prefix resume, wrong ID, quota preservation, resource failure, partial mutation and callback/cancel propagation. Run missing-module RED.
- [ ] Implement the minimal external policy. Snapshot/check committed offset/file identity around failures/backoff; call native quota handling before the first snapshot. Never catch native DownloadError as retryable.
- [ ] Run new and existing range/download tests GREEN; verify frozen source SHA unchanged; run complete light suite and commit.

### Task 2: Public selection and actual owned full-intake renewal

**Files:** Create `assets_geometry_manifest.json`; modify `docs/download.md`; new private immutable renewal script/state/evidence under ignoredtmp.

**Interfaces:** Inventory lists exactly the original eight filename/DriveID/bytes records, old release manifest identity and shape declarations. Existing `validate_manifest_sizes`, `ensure_capacity` and `download_one` remain authoritative. Renewed intake uses Task1 callback/statistics and an independent one-shot state; old failed script/state/log remain unchanged.

- [ ] Export reviewed inventory, verify it against actual pinned release manifest and old selection; document original library delegation and fixed retry policy.
- [ ] Confirm old process actually nonlive and no actual target consumers. Revalidate preserved tails/endpoint identity, real aggregate capacity and frozen source bytes before one new launch.
- [ ] Start the entire eight-asset selection and observe actual PID/start plus new committed complete chunks or final receipt. Do not call all files acquired while the owner lives; retain actual failure if it occurs.
- [ ] Publish only sanitized actual evidence and source/selection identities, update full-paper status, run appropriate final checks and commit. Continue tracking full37.778GB separately until real completion.

## Implementation decisions within the approved spec

Whole-job retries would repeatedly hit another disconnect during this large selection. HTTP adapter connect retries alone do not cover streaming interruption after headers. The external per-offset policy handles both while preserving the already qualified strict implementation. This is the implementer's routine recovery choice; it changes neither the requested data nor scientific protocol. No new user preference or authorization is required.
