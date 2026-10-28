# Full EMG2Pose Handoff Contract Correction

Execute inline under the approved full-paper design. One final whole-fix review, one author correction pass if needed. Existing full source, failed queue and all trained/qualified model artifacts stay immutable.

**Observed failure:** Full source controller9e563c5aef66 completedactualexit0 at05:38:44UTC with25,253sessions/3,043,302,266rows and exact original split counts17136/1950/6167. Waiting queueff9274687c53 then failedactualexit1 before starting any model: `source_handoff` passed the raw `emg2pose_runtime_data_identity_v1` to the later verified-source validator. Raw numeric identity deliberately lacks `archive_bytes`; `open_source` supplies that field only after validating the completed source and retained archive lineage. The same strong validator already runs after `open_source` in queue main.

**Correction:** Validate the actual raw numeric identity at the handoff boundary; preserve unchanged deep source opening and the subsequent verified full-source/archive gate before any GPU/model action. Do not invent an archive size, change role counts, weaken the verified-source gate, modify training code or re-download/re-convert data.

- [ ] Reproduce the genuine raw identity schema in a successful-terminal handoff fixture; see the intended missing-archive failure before changing code. Include negative population/scope/schema cases and prove the existing verified archive gate still rejects that raw identity.
- [ ] Make the single boundary correction in `scripts/run_emg2pose_full_queue.py`; run queue regression checks and a readonly replay against the actual completed source and original queue config. Confirm all old queue/source owners nonlive, failure files preserved, and model qualification producer hashes unchanged.
- [ ] One final whole-fix review; one author TDD correction pass if material findings, no review loop. Then create a new isolated queue stage/output and start exactly one S/M/L250epoch queue with existing frozen per-model and controller qualifications. Keep its actual handle; full source deep opening may take time and is not a vanished process. Publish source completion, original failure, correction and actual new-owner state separately; no scientific completion until full experiments finish.

The EMG initializer implementation plan remains active alongside this correction; no in-progress initializer source is overwritten or falsely called trained.
