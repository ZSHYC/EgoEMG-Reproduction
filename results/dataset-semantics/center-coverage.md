# Legacy frozen-window center coverage

Actual point reads consumed all **33,508 saved windows / 67,016 hand centers** in the existing legacy partition. The operation read6,232,488 payload bytes across12 fields and verified source stability; it did not rescan or rehash all large arrays.

These are physical centers of the saved12,000-point windows, with train stride1,200/eval stride12,000 and independent temporal holdout. The new classic7,790-point cohort and an EMG-token midpoint are different populations.

| Independent role | Window centers | Hand centers | Observed classes /60 | Possible angles | Valid angles |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 31,586 | 63,172 | 51 | 1,389,784 | 1,338,448 |
| val | 526 | 1,052 | 32 | 23,144 | 22,064 |
| test | 1,396 | 2,792 | 59 | 61,424 | 59,572 |
| All roles | 33,508 | 67,016 | 60 | 1,474,352 | 1,420,084 |

The test population is the existing2,792 hand centers; no grid was tuned to paper4,278. All60 classes occur across all roles, while train/val/test contain51/32/59 observed definitions respectively. The original released split named train also contains independent-val windows: its observed52 classes do not mean the training role saw52.

| Original released split | Window centers | Hand centers | Observed classes /60 | Valid angles |
| --- | ---: | ---: | ---: | ---: |
| train | 32,112 | 64,224 | 52 | 1,360,512 |
| user | 538 | 1,076 | 51 | 23,232 |
| gesture | 754 | 1,508 | 24 | 31,918 |
| both | 104 | 208 | 17 | 4,422 |

Known60-class centers: 33,267; unassigned(-1): 241; raw50: 66. Thus 33,508 - 241 unassigned - 66 raw = 33,201 defined59-class centers. This view has 66,402 hand centers and 1,406,598 valid angles; it is not an error-analysis sample count.

Across all roles, generated validity is true at 64,552 hand centers and independent wrist validity at 64,522. Masks use first20 generated validity / last2 wrist validity AND per-angle finite before cleanup. Finger disk values are radians; wrists are converted from degrees once. This is the frozen legacy20+2 protocol, not the new classic literal2D mask.

The JSON contains actual role/split/class/activity contingency cells, per-angle finite/valid counts, hand/family/class aggregates and explicit60/59 views. A synchronized window, a hand center and a valid angle remain distinct. by_hand shares physical windows, so its window counts are not additional independent windows.

[Full synchronized-frame audit](coverage.md) is separate earlier evidence. Model errors, full-frame validity, crop availability and per-user performance were not computed by this point read; Tables14/15 and qualitative/occlusion results remain unfinished.

Execution source: [`f6c293d6308dc0ead082b7afd2c70936901d31dd`](https://github.com/ZSHYC/EgoEMG-Reproduction/commit/f6c293d6308dc0ead082b7afd2c70936901d31dd). Private audit content SHA256: `36e606f9f4112e3acfd1e4172ca80250e12f16f6b472f4ad08b64ee2557aa094`. Public JSON retains source/partition/transfer identities and coordinate+raw-point digests. Original raw identities are local-transfer hashes, not publisher-certified checksums. No participant/episode mapping or private source path is exported.
