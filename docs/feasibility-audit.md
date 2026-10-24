# Initial feasibility audit

Audit date: 2026-10-24, Asia/Shanghai. This is a source and infrastructure audit, not an experimental result.

## Compute and storage

Read-only SSH checks succeeded on the user-designated `eeg-server`.

| Resource | Observed state |
| --- | --- |
| GPUs | 8 x NVIDIA A800-SXM4-80GB; all idle at inspection |
| Host RAM | Approximately 1 TiB total, 944 GiB available at inspection |
| NVIDIA driver | 595.91.07 |
| System Python | Python 3.10, no PyTorch |
| Existing unrelated environment | PyTorch 1.13.0 + CUDA 11.7; must not be modified |
| System filesystem free | Approximately 118 GB at inspection |
| Shared data filesystem free | Approximately 79 GB at inspection |

Compute is sufficient for the proposed baseline experiments. Storage is the immediate constraint. Availability is a snapshot, not a reservation; jobs must inspect resources again before starting. A separate environment and a project-specific data root are required. No existing datasets, environments, or other users' files will be removed to make space.

## Source and paper versions

The audited upstream commit is `f7cbd048b7092139852270491d3859cd472e0311`.

The supplied paper describes 41 participants, over 10 hours, 7,790-sample EMG windows, and filtered EMG without amplitude normalization. Current upstream documentation describes 53 participants and over 18 hours. Released-compatible EgoEMG EMGFormer configurations use 12,000-sample windows, 8 target-hand channels, and per-dataset normalization.

Consequently, running the newest default command does not by itself reproduce the supplied PDF configuration. A report must state which dataset and protocol it actually used.

## Confirmed evaluation issues

1. The latest upstream commit fixes EMG2Pose validation loaders that previously combined validation and test data for checkpoint selection. The commit explicitly notes that released checkpoints used the old configuration. Its change does not establish clean EgoEMG validation.
2. The audited EgoEMG dataset configuration assigns `user`, `gesture`, and `both` to validation, then aliases test to validation. New training needs an independent validation partition while preserving the official final test conditions.
3. The dataset loader filters windows using their center split. A context window can cross split boundaries; training jitter can move the center without rechecking the split. Window-level random splitting is therefore unsuitable.
4. Published normalization files state that statistics were computed over the full EgoEMG dataset. New clean training must recompute statistics using retained training data only.
5. Upstream angle tensors and CSV metrics use radians. Paper tables use degrees. Conversion must be explicit and applied once.
6. Training-time EMG validation and offline trajectory analysis may use different target granularity. Visual and fusion evaluation must use the same center timestamp and valid-example population.

These findings affect claims about independent generalization. Published-checkpoint verification remains useful if its provenance and selection history are reported explicitly.

## Model and dependency issues

- EgoEMG EMGFormer checkpoints use 8 target-hand channels. Some fusion configurations use a 16-slot representation created by mapping/padding one hand's 8 channels; this must not be described as concatenating the two wrists.
- The current default normalization alias and the training dataset name need compatibility checks before a run.
- A nominally short upstream training command may still trigger full evaluation unless `eval=false` is explicit.
- TensorBoard is used by the default logger but is not explicitly listed in the audited environment specification.
- Even EMG-only imports can pull in visual dependencies such as OpenCV.
- The upstream environment specifies Python 3.11.12, PyTorch 2.2.2, torchvision 0.17.2, CUDA 11.8, and NumPy 1.26.4. The existing unrelated environment is not an acceptable substitute.

## Data availability

The asset guide describes a current Baidu release and a three-episode preview, but upstream LICENSE says that the current complete release is still in preparation. Documentation alone is not proof that the current dataset is downloadable or that a particular checkpoint matches it.

Read-only inspection of the official legacy Google Drive mirror found a real v2 memmap with 41 episodes/subjects and 66,161,725 rows. The manifest and a 17,088-byte metadata file were inspected. Public download endpoints for one angle array and one legacy checkpoint returned the expected binary content type and byte length without an account login; their bodies were not downloaded. This establishes a practical source for an initial real-data attempt, not checkpoint compatibility or completed transfer.

The complete legacy memmap directory is approximately 114.46 GB, but the fields needed for EMG, angles, validity, splits, and indexing total approximately 18.39 GB (17.13 GiB). Retaining raw EMG and materializing a separate paper-filtered signal increases the estimated peak to approximately 26.86 GB. This fits the observed free space if downloads remain selective. Videos and crops are a separate storage budget and must be measured before acquisition.

The legacy schema uses webcam metadata and older filtered-field names. Current loader assumptions require a tested adapter. Legacy descriptions of angle generation also differ from the PDF's landmark-IK description, so the fact that both contain 41 subjects does not establish label equivalence. Legacy checkpoints are a smaller, differently named collection than the current README's S/M/L families. Their compatibility must be established from checkpoint metadata and shapes before evaluation. The current three-episode preview on Baidu has not been verified as anonymously downloadable. A third-party snippet is not required while a usable official legacy source is available.

An additional preprocessing limitation matters for temporal validation: the upstream paper-filtering script applies FFT filtering to a whole episode. A window guard alone does not remove this global filtering dependence. Strict new training should filter raw training and validation intervals independently and discard their boundary regions; published-checkpoint verification should retain its original preprocessing and disclose it.

Before downloading a large artifact, record the source, release/schema version, file list, byte sizes, terms, and manifest hash. Prefer an official small real-data package for integration checks. A third-party snippet is only a fallback if provenance, terms, and compatibility can be established.

No full dataset or checkpoint download and no reproduction training was performed during this initial audit.

## Evidence

- [Upstream commit and validation fix](https://github.com/zhenqis123/EgoEMG/commit/f7cbd048b7092139852270491d3859cd472e0311)
- [EgoEMG dataset configuration](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/dataset/egoemg_unified_angle_regression.yaml)
- [Dataset loader](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/datasets/egoemg_memmap_dataset.py)
- [Environment](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/environment.yml)
- [Asset guide](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/docs/ASSET_SETUP.md)
- [Official legacy download script](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/download/download_egoemg_data.sh)
- [Official legacy memmap directory](https://drive.google.com/drive/folders/1325S0iBVvbT_OnFAeiacLSRjXoWbCPsl)
- [License and release status](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/LICENSE)
