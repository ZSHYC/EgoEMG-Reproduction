# WiLoR explicit recovery repair

This repairs execution of the already approved full-paper design, not a new scientific recipe. The original full run failed at its sixth complete epoch while saving last.ckpt because its workspace reserve was lost. Native full-scope epoch5 checkpoint was independently accepted by the unchanged production validator:6epochs/67344updates/SHA4835db65a180e4a8a389080f30728a07933af9a25aa2619b3854311774bf03eb. last.ckpt contains only5epochs. Neither the original failed capsule nor its source/evidence is overwritten.

## Approach and delegated decisions

Continue the complete original state in a new directory, using a separately identified recovery format and explicit parent lineage. A fresh restart discards real progress; silently converting the old uninterrupted result into a resumed result invents provenance. Both are excluded. User's existing official-code-first/full-paper/remote-execution authorization covers this routine reversible repair. Model, loss, optimizer, data, precision, batch, epoch30 limit, callbacks, train/validation coverage and original source hashes remain unchanged.

The new recovery entry uses native Lightning checkpoint restoration. Only callback output paths are relocated to verified copies of the original top3/last packets. A custom CheckpointIO returns the same raw native state with this documented path relocation; it does not rewrite the parent packet. A passive observer records exact restoration of model/AdamW/scheduler/callback/loop/RNG state before iterator reconstruction, then delegates native observer operations. Prefix journal bytes are copied exactly. New result format records original and new owners, total versus newly executed steps, interruption, source/input hashes and parent packet; it does not claim uninterrupted execution.

The existing closed trainer/data/evaluation source bytes are preserved. Recovered readers and evaluation orchestration live in new modules; reuse unchanged original forward, loaders, statistics, native/replay checks and rendering helpers. New reports explicitly include recovery lineage. Original read_wilor_fit continues to reject the new format; new wrappers accept only independently validated recovered fits.

## Gates and evidence

- Admit only complete, source/current-runtime-bound native epoch packets matching a verified journal prefix. For an interrupted full parent choose the newest complete packet; reject an older last when epoch6 exists, a partial journal, existing/live owner, arbitrary scope or already completed budget. A bounded engineering test may explicitly choose an older complete packet.
- Check actual parent identity/policy/cohort/native loop sources, raw native model/optimizer/scheduler/callback/loop/RNG contents, every checkpoint needed by top3/last state, full hashes and nine file attributes. Source/input admission is repeated only for this new actual execution.
- Fresh output only, exclusive attempt claim, complete private lineage before updates; preserve parent paths/bytes/attributes and owned-process cleanup on failure. Retain original host/output/GPU reserves and finite budgets; no automatic restart.
- Canonical CPU checkpoint8013690 predates the validator-only91967a1 current trainer hash. It cannot be relabeled current. A new current-source CPU prefix deliberately interrupts after the first complete native last save; a new process resumes the original second epoch. This genuinely exercises the new entry, rather than regenerating an unchanged test fixture. Compare raw restoration domains and frozen validation, and independently compare final numeric state with the retained older oracle only after proving its training operators unchanged.
- Engineering outputs may use an explicitly named /dev/shm directory under host reserve. This is volatile diagnostic storage, never full scientific checkpoint storage. Keep raw engineering evidence and export durable evidence; a full30 continuation requires sufficient durable storage (current/data has only~31GiB; additional storage question is pending).
- One fresh review of the new recovery unit after complete new tests and actual engineering proof; one Critical/Important RED→GREEN fix pass, no second review or reopening unchanged model suites.

## Completion

The repair is operationally complete only after current-source native recovery, full native/replayed validation/test/profile/report and lineage acceptance succeed. The scientific setting remains incomplete until the genuine resumed full30 plus all8612test consumptions/4306unique hands and independent report verification finish. The whole-paper goal remains active beyond this repair.
