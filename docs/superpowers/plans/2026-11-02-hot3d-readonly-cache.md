# HOT3D Read-only Cache Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one fresh final review of this new transport unit and one material fix pass if needed. This implements the already approved full-paper acquisition and resource rules; no new experiment or authorization round.

**Goal:** Receive all 2,804 fixed public HOT3D train archives while reusing verified closed generations without physically copying them at each transport failure.

**Architecture:** A separate receiver resolves each manifest path through an explicit read-only cache or its new owned archives directory. Cache preparation authenticates actual closed source owners, source bytes and journals; cache admission hashes every candidate against the publisher's complete LFS SHA. New files retain the original native single-connection downloader and exact checksum, ownership and stop-on-failure behavior. Existing receivers, receipts and payload generations remain unchanged.

**Tech Stack:** Existing Python standard library, original native aria2 binary and owned-process recorder/guardian; no installs.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements6/7 and resource/source rules; the choice of physical read paths is an implementation decision within that design.

## Global Constraints

- Fixed public manifest/revision `30fe9674782f32e1e5edba98476b6ff4300132c5`, all2804/294201743360bytes; no reduction of the requested source or claim to author-mixture/numeric equivalence.
- Original `scripts/paper/hot3d_full_train_receiver.py` stays byte-identical. No scientific model/math/source audit changes, GPU use or existing job restarts.
- Reuse source payloads through regular nonsymlink physical paths. No copies, links, overwrites or deletion of old generations, metadata, controls or partials.
- Source terminal state alone is insufficient: validate exact source launch/bundle, PID+start ticks of owners/native workers, complete source-member hashes, journal lengths/digests and file generations. Cached unjournaled complete-size candidates require fresh whole SHA before use.
- Rehash all cache candidates before the first new download; preserve and guard nine file attributes and closed-source metadata throughout use. A subsequent generation flattens the physical path map and ancestry instead of pointing through implicit aliases.
- Keep original source URLs, TLS verification, native max-tries1, one connection/split,32MiB/s, checksum, continuefalse and no-overwrite options. No hidden retries, route/auth/proxy changes or denied-response bypass.
- Require remaining full bytes plus225GiB output-filesystem reserve before acquisition; preserve225GiB while downloading to protect the running RN18 workflow. This raises the logistics reserve from the old200GiB without changing source data.

## Review Focus

- Reused files outside the new root, or a second cached generation: exact original physical path/bytes/generation and ancestry must survive without copying or symlink resolution hiding a link.
- A source says terminal but an owner/native child still lives, or source metadata/payload changes: refuse reuse before download.
- A complete-size unjournaled candidate or partial with a control file: only genuinely full LFS-matching uncontrolled candidates can be reused; other partials remain untouched.
- Errors after child creation: reap only the owned native child, retain partial bytes and never silently retry or promote full completion.
- Cache candidates need not be a journal prefix; aggregate capacity, ordered journal counts and final2804/fullbyte totals must still reconcile.

### Task 1: Source-bound cache and complete receiver

**Files:** Create `egoemg_repro/hot3d_cache_intake.py`, `scripts/paper/hot3d_layered_train_receiver.py`, `tests/test_hot3d_cache_intake.py`.

**Interfaces:** `cache_from_closed_source(root, launch, assets)` returns a flattened cache descriptor with closed source handles, metadata nine-attribute/SHA pins and complete-size candidate physical paths. `load_readonly_cache(path, *, expected_sha256, assets)` verifies all metadata/actors/current whole payload hashes and returns a checked cache with `path_for(asset)` and `check()`. `receive_assets(root, assets, cache, *, binary, guard)` is a transport-only helper returning exact received counts and actual native worker identities, never full completion. `main(root)` authenticates the fixed complete manifest/source/binary/cache and publishes source acquisition completion only for all2804/294201743360bytes.

- [x] Write meaningful source/payload/link/live-owner/ancestry/capacity/real-child failure regressions first; actual missing-module RED. Small payloads and a real deterministic child stand in only for the external network binary; they cannot qualify actual HOT3D transfer.
- [x] Implement minimal helpers and new CLI, retaining the original receiver's inventory/checksum/native option semantics.
- [x] New component suite GREEN, original receiver SHA unchanged, no old model/geometry tests rerun. Commit new source/tests and exact evidence.
- [x] One fresh review of this complete new integration; re-grade findings, one material RED→GREEN fix pass, then green relevant suite. Record all declined-to-judge rulings. No second review.

### Task 2: Actual closed-source admission and explicit successor

**Files:** Private immutable preparation/launch/observer code and sanitized source/cache acquisition evidence; update full-paper status.

- [x] Reopen actual TLS833 successor: terminal failed1/907journalarchives, exact original actors all closed, all source/metadata/generations preserved. Inspect complete candidates without assuming candidate count equals journal count.
- [x] Same original failed Hub URL HEAD: frozen revision/LFS SHA/full length agree; honor errors/Retry-After without bypass. Measure full remaining byte capacity and225GiB reserve.
- [x] Stage reviewed committed source; prepare cache descriptor read-only, then perform whole cache SHA admission in a detached owned process. Start new archives only after that admission succeeds.
- [x] Verify actual original PID/start handles and first genuinely new full native archive after its native owner exits. Publish accurate proof/current progress; live or small fixture checks never mean complete2804 or numeric/author-mixture qualification.
- [ ] Receive all2,804 archives/294,201,743,360bytes and verify complete journal/publisher SHA/physical paths after the actual receiver/controller/recorder and native owners exit. The first-new-archive milestone does not complete the full-source goal.

2026-11-03 checkpoint: prior3Important fixed in ONE material pass, local37GREEN11.16s and actualserver37GREEN0skip11.68s/8frozenmembers/testowners+readerclosed; originalreceivercc94d103unchanged, commitfe46e605. Fresh907journal/908candidatepreflight canonicalHEAD302 pinnedrevision/LFS108318720 and noRetryAfter, free1.490TB vsremaining195762022400+225GiB. Newownedreceiver2710203/start452574215 underguardian2710202/start452574211, controller2710201/start452574191 andrecorder2710200/start452574188 actuallystarted09:33:56UTC. At09:37 actual908wholeSHAadmission98439720960bytes; firstnewclip002756 actualnative2714085/start452642419 exited0 andits108318720wholeSHA independentlyrechecked, noaria2 andallold9attrs/sourcebytes/actorsclosed; snapshot914journal/7new, all4ownparentsstilllive. Proof results/hot3d-readonly-cache-admission-first-native-20261103.json. Full2804 andnumeric/author-mixture remainfalse.

## Implementation rulings

The existing approved design already authorizes complete source acquisition and routine storage decisions. Copying every previous complete generation repeatedly would consume storage without obtaining new source bytes. Deleting old generations would discard preserved evidence. Explicit regular-file read paths preserve the original bytes and evidence while allowing complete acquisition; the cost if wrong is a stale/misbound source cache, addressed by actual closure, full publisher SHA and generation guards. No new scientific or user preference is introduced.

2026-11-06 operational continuation after the actual November3 timeout: original fe46 source ended failed1 with1043journalarchives/113301555200bytes; all222oldsourceactors are physically closed. Fresh cache snapshot has1044candidates/113403996160bytes (one closed future Quest3 seed). Same original failed Hub URL HEAD302 matched revision/LFS/103577600bytes and noRetryAfter; full remaining180797747200bytes plus225GiB fits. One new create-only source-bound successor uses the unchanged already-reviewed receiver/native options and0payloadcopies. Its actual1044wholeSHA admission completed2026-11-05T16:08:15UTC; all4owningactors live at16:10:25UTC, journal0/new0 while initial capacity admission proceeds. No new scientific feature or repeated unit review/test run; source hashes match the accepted37test receiver. [Actual source/cache/closure evidence](../../../results/hot3d-closed1043-cache-admission-20261106.json). First new native postexit independent hash and full2804completion remain pending; the unchecked full-source item above remains unchecked.
