# EMG2Pose mini: real two-update CPU acceptance

S/M/L each completed two actual original-Lightning updates and **all526 windows twice**, with exact equality after strict model/optimizer/scheduler/RNG restoration. This is engineering acceptance; **not independent validation, a selected best, completed scientific training or a paper result**.

All30 mini files are original val. Each model ran in its own CPU process with float32, batch1, workers0, no shuffle and original val transforms; intra/inter-op threads were both2. The official forward, loss, AdamW and epoch cosine were retained.

| Model | Parameters | Actual updates | Windows before/after reload | Reload equality | Checkpoint bytes |
| --- | ---: | ---: | ---: | --- | ---: |
| EMGFormer-S | 3,481,636 | 2 | 526 / 526 | exact | 41,997,632 |
| EMGFormer-M | 6,638,884 | 2 | 526 / 526 | exact | 79,926,816 |
| EMGFormer-L | 16,342,308 | 2 | 526 / 526 | exact | 196,398,992 |

Every pass contains76,585,600 prediction elements, the same number of targets and3,829,280 mask elements. Per-model JSON preserves actual valid-frame counts and before/after streaming digests for input, prediction, target, mask, window order and actual official metrics. Private metric-window means and original session/person/path fields are excluded.

Nonzero finite gradients, clipping at1 and changed parameter tensors were observed at both updates. Original `loss_weights={mae:1, landmark/fingertip:0.01}` applies to this EMG2Pose family. As max_steps2 stops within the first unfinished epoch, the original epoch scheduler remains `last_epoch=0`, `_step_count=1`; no step was invented.

Full Lightning checkpoints retain the actual optimizer/scheduler/loop/RNG state. Atomic checkpoint/JSON writes preserve15GiB; each checkpoint records actual tensor bytes, conservative serialization budget and final size. Strict restoration here supports exhaustive engineering evaluation; it is not a250-epoch continuation or final-test claim.

Execution source: [`07019609583640d685f724041b022d10b4831e61`](https://github.com/ZSHYC/EgoEMG-Reproduction/commit/07019609583640d685f724041b022d10b4831e61), official revision `f7cbd048b7092139852270491d3859cd472e0311`. The393-file runtime bundle includes the normalization asset; it is a different scope from the392-file loss-audit bundle. Exact source, data receipt, artifact and streaming hashes are in [the aggregate JSON](emg2pose-mini-optimization.json).

[Data acquisition/conversion](emg2pose-mini-data.md), [earlier one-window runtime check](emg2pose-mini-runtime.md), and [exact runnable CLI](../docs/training.md#7-emg2pose-mini-两步工程检查) are separate evidence. Full-data acquisition and three complete scientific runs remain unfinished; Tables11/13 have no new full-reproduction result from this check.
