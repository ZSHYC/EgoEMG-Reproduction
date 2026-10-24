# EgoEMG Reproduction

An independent reproduction project for **EgoEMG: A Multimodal Egocentric Dataset with Bilateral EMG and Vision for Hand Pose Estimation**.

This project follows the authors' implementation, pinned to a specific commit. It is not the official EgoEMG repository.

## Current status

**Feasibility audit and proposed design; no reproduction training result has been produced yet.**

- The paper and appendices have been reviewed.
- The user selected reproduction based on the official implementation.
- The designated GPU server is reachable and has 8 NVIDIA A800 GPUs with 80 GB each.
- Data availability, data version compatibility, and evaluation leakage are being audited before training.
- The proposed first experiments are EMGFormer-S, ResNet-18, and ResNet-18 + EMGFormer-S residual fusion. EMGFormer-M follows once the pipeline is verified.

Read the [proposed design](docs/superpowers/specs/2026-10-24-egoemg-reproduction-design.md) and [initial feasibility audit](docs/feasibility-audit.md).

## Version policy

| Artifact | Reference |
| --- | --- |
| Paper under study | [arXiv:2605.05712v1](https://arxiv.org/abs/2605.05712v1), 41 participants, over 10 hours |
| Official source | [zhenqis123/EgoEMG](https://github.com/zhenqis123/EgoEMG) |
| Audited source commit | [`f7cbd048b7092139852270491d3859cd472e0311`](https://github.com/zhenqis123/EgoEMG/tree/f7cbd048b7092139852270491d3859cd472e0311) |
| Reproduction results | Pending; synthetic or preview checks will not be reported as paper-level reproduction |

The current upstream README describes 53 participants and over 18 hours. Some released configurations also differ from the supplied paper. Every future run must record its exact source commit, dataset manifest, configuration, initialization, and evaluation protocol.

## Evaluation policy

- Keep published checkpoint verification separate from new training with independent validation.
- Never choose a new checkpoint using the final test set.
- Split continuous recordings before constructing overlapping windows; keep paired hands in the same partition.
- Fit normalization statistics only on the final training partition for new training.
- Report angles in degrees, with explicit conversion from upstream tensors in radians.
- Distinguish EMG trajectory evaluation from center-frame evaluation.
- Report the Gesture, User, and Both conditions separately, including sample counts and the definition of any aggregate.

## Attribution and assets

Paper and original implementation: Ziheng Xi, Jiayi Yu, Yitao Wang, Yanbo Duan, Jianjiang Feng, and Jie Zhou.

This repository does not redistribute the paper PDF, dataset, pretrained checkpoints, or MANO assets. Original assets remain subject to their respective terms. Upstream declares MIT for its baseline code and identifies inherited emg2pose portions under CC-BY-NC-SA-4.0; see its [license](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/LICENSE).

Original contributions to this reproduction repository are covered by [LICENSE](LICENSE).
