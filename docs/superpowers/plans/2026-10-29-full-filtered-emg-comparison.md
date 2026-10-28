# Full legacy filtered EMG comparison

**Scope:** Complete the missing all-41-episode numerical comparison in D04 of the [approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md). The original official FFT outputs already exist, and the released arrays remain in use by active training. Neither is regenerated or modified in this step.

- [x] Add a bounded read-only comparison script and check independently hand-computed cases, including unequal values and signed zero. Validate full metadata coverage, exact stream length, both complete SHA256 identities and unchanged file attributes.
- [x] Stream each existing local native FFT output to a server CPU reader of the matching released array. Save per-hand/episode/channel MAE, RMSE, maximum absolute difference, unequal numeric counts and unequal float32 bit counts. No full array copy, model import, GPU allocation or training change.
- [x] Confirm actual worker termination and independently reopen receipts; reconcile whole counts, per-episode/channel reductions and the retained first-episode result. Publish actual differences and update D04, without asserting byte equality, hardware unit calibration, split-isolated filtering or PDF equivalence.

Use the existing NumPy runtime. Compare all 66,161,725 rows × 8 channels × 2 hands, retaining all values and independent hand identities. This is a deterministic data analysis; targeted numeric checks and actual full-array reads replace a repeated project test suite or model review.
