# EgoEMG reproduction design

Date: 2026-10-24, Asia/Shanghai.

Status: Proposed design, pending user review. The user has selected reproduction based on the official implementation. Design approval, implementation, and experimental results must not be inferred from that choice.

## Intent

Create an independently owned public GitHub repository that makes the EgoEMG baseline experiments traceable and repeatable. Run experiments on the user-designated `eeg-server`. Use the supplied arXiv v1 paper as the scientific reference and explicitly record differences in later source, datasets, or checkpoints.

Success means another researcher can identify the exact source, prepare the same data, execute the documented command, and recompute the reported metrics. A passing synthetic check, a tiny preview, or a loaded checkpoint is not a full reproduction of the paper's quantitative results.

The proposed primary acceptance target is a traceable reproduction of the fixed official implementation on accessible official legacy data, using explicitly recorded released-compatible settings. The initial scope does not train both 7,790- and 12,000-sample EMG recipes in parallel. Strict numerical reproduction of the PDF tables is conditional on proving dataset, label, preprocessing, and checkpoint provenance; if that evidence remains unavailable, report the completed work as a legacy-release reproduction and retain the PDF values only as reference.

## Approaches

1. **Selected by the user: pinned official implementation with a reproduction layer.** Keep model implementations traceable to upstream; add configuration overlays, data/partition validation, launch scripts, and result aggregation only where needed. This provides the clearest comparison with the authors' implementation.
2. **Considered: independent implementation from the paper.** This would help study the architecture, but the PDF leaves input-channel, alignment, loss, and training details unspecified or inconsistent. It would introduce additional implementation uncertainty before validating the published baseline.
3. **Considered: unmodified latest upstream commands.** Useful for published-checkpoint verification, but insufficient for clean new training because source, paper, dataset, and validation protocols differ.

## Initial experiment scope

The first complete baseline set is:

- EMGFormer-S for one-hand EMG-to-angle regression; add EMGFormer-M after the pipeline is verified.
- ResNet-18 for center-frame image-to-angle regression.
- ResNet-18 + EMGFormer-S residual fusion for the same center-frame target.

Use the same 22-angle semantics and official Gesture/User/Both test identities where the accessible release supplies them. Report EMG trajectory metrics separately from center-frame metrics. Evaluate both hands with the documented channel and image conventions.

Larger backbones, WiLoR fine-tuning, EMG2Pose-scale experiments, and retraining the offline markers2mano annotator are later extensions. Existing released pose labels are the initial supervision source.

## Repository and source ownership

Create `ZSHYC/EgoEMG-Reproduction` as requested. Keep upstream at commit `f7cbd048b7092139852270491d3859cd472e0311` in a separately managed checkout. Record the exact commit and any local patch hash in every run. Do not silently follow the upstream default branch.

The public repository contains original reproduction code, configuration, documentation, tests, and small aggregate results. It excludes raw signals, participant metadata, videos, model weights, private paths, SSH configuration, credentials, and the local paper PDF. Preserve all required upstream notices for any derived code and do not relicense external assets under the reproduction repository's license.

## Environment and execution

Use a project-specific environment, separate from existing server projects. Start with the upstream Python 3.11 / PyTorch 2.2.2 / torchvision 0.17.2 / NumPy 1.26.4 stack; record the actual CUDA build and lock the resolved dependencies. Verify import, GPU allocation, and bf16 execution before data-dependent experiments.

Begin with one currently idle GPU for integration and a small batch; measure memory and throughput before selecting more GPUs or changing batch size. Preserve effective batch size and scheduler semantics when scaling. The observed 8 x A800 80 GB resource is sufficient but is not reserved.

Run long jobs in a project-specific persistent session with timestamped logs, a saved command/configuration, and a clear process identifier. Do not start unbounded sweeps or occupy every GPU by default. Resume from a recorded checkpoint after interruption when supported.

Use storage checks before environment creation, download, extraction, and checkpoint writes. Leave at least 15 GiB free on the destination filesystem. Download only fields needed for the selected task. The full legacy asset set does not currently fit comfortably alongside an environment and outputs on the server; selectively acquiring the required EMG fields is estimated at 18.39 GB, or approximately 26.86 GB peak when retaining raw EMG and generating separate filtered signals. Measure videos/crops separately before acquisition. Do not delete existing files to free space.

## Data preparation and provenance

Before implementation settles on a release, verify a small real sample and its schema. Record source URL/identifier, release version, byte sizes, file/manifest checksums, episode and subject counts, angle ordering, units, sampling rates, valid masks, and official split representation.

Treat the paper's 41-person dataset, the current README's 53-person dataset, official legacy mirror, and any third-party snippet as different artifacts until hashes and metadata establish their relationship. A preview demonstrates integration only and must not be substituted for a full test cohort in headline results.

The official legacy mirror is the proposed initial source: its manifest and metadata describe 41 episodes/subjects and 66,161,725 rows, and selected public download endpoints have been checked. Implement a minimal, tested v2 schema adapter rather than pretending the legacy files are a current v3 package. Verify the first transferred sample before scheduling the remaining selective downloads. The legacy angle-generation descriptions differ from the PDF's landmark-IK description; equal participant counts do not establish label equivalence. Match any legacy checkpoint by its stored configuration and tensor shapes, not by its filename alone.

Published checkpoint verification may retain original normalization and input layout, because changing them changes the checkpoint's expected input. New clean training uses training-only normalization. Keep these two tracks separate in commands and reports.

## Independent validation for new training

Preserve the official final test center lists for Gesture/User/Both. Construct a read-only sidecar partition manifest rather than modifying the source data or relying on upstream's ambiguous `val` alias.

The manifest records release hash, source/episode/subject identity, sample ranges, train/validation/test role, input footprint, and center indices. Paired hands share the same time partition.

For each episode, inspect contiguous runs belonging to the official training portion. Reserve a deterministic contiguous validation block from sufficiently long training runs, aiming for roughly 10-15% of eligible training duration. Validation must contain complete windows. Short runs must not be forced into unusable partitions; publish coverage and excluded-window counts before launching training.

Disable training jitter initially. A retained training window's full input and supervised target footprint must remain inside its training interval and must not intersect validation or official test-window footprints. If jitter is later enabled, include its maximum displacement in the footprint checks. Never split already constructed overlapping windows at random.

Recompute normalization over the final retained training samples only. Select checkpoints using validation only. This validation measures held-out time within training users; do not describe it as an independent cross-user validation cohort. Report any change in participant/gesture coverage caused by window purging.

For strict preprocessing isolation, derive the train/validation filtered signals from raw EMG independently within their assigned intervals and discard filtering boundaries. Whole-episode FFT filtering introduces dependence across temporal partitions, which a finite window guard cannot fully eliminate. If a run intentionally uses the official prefiltered signal, label that limitation explicitly rather than claiming strict preprocessing separation. Preserve the published test preprocessing for checkpoint-verification comparisons.

If real metadata cannot support a usable independent validation partition without changing the official test set, document the limitation and resolve the partition design before launching full training.

## Models and configuration

Retain official model implementations. Audit all composed configurations before a run, including input channels, window length, filtering, normalization key, target units, loss scale, initialization, and evaluation granularity.

The paper's 7,790-sample configuration and currently released-compatible 12,000-sample EMG configuration are separate configurations. Never load a checkpoint using a different input layout merely because its model size has the same name.

The fusion residual uses both pooled EMG and visual features and starts with a zero-initialized final residual layer. A 16-slot input generated from one wrist is not evidence of bilateral EMG fusion.

## Metrics and acceptance criteria

For each real run record:

- source and patch revisions, resolved configuration and random seed;
- environment and GPU model;
- dataset and sidecar manifest hashes, retained sample/window counts;
- initialization checkpoint source and hash, or explicit random initialization;
- training and checkpoint-selection protocol;
- per-hand and per-split 22-angle MAE in degrees;
- finger and wrist components where valid labels are available;
- center-frame versus full-trajectory target definition;
- aggregation weights and valid denominators.

Recompute the pooled sample-weighted mean from error sums and valid counts. Also report per-user means and their standard deviation. If showing the upstream README's unweighted split mean, label it separately. Convert radians to degrees exactly once. Missing samples, unavailable splits, and invalid crops must be counted explicitly rather than silently dropped from a paired model comparison.

Acceptance proceeds in stages: dependency/GPU check; real-data schema check; finite forward/backward/loss check; bounded short training; checkpoint reload and deterministic evaluation; full baseline training and final test evaluation. Only the last stage supports a claim about full quantitative reproduction.

## Builder decisions and open evidence

The repository name, initial small-model set, released-compatible primary configuration, separate environment, one-GPU integration start, and training-only normalization are proposed implementation choices, not extra user requests. The user has authorized a public repository and server use. A separate builder reviewed this design; its two material questions concerned the primary configuration and the acceptance boundary when legacy labels cannot be proved equivalent to the PDF. The proposed acceptance target above makes both choices explicit for user review.

Selective legacy EMG acquisition appears feasible; exact transferred-file compatibility and visual crop costs remain under investigation. Experiment duration is not yet measured. Do not promise the paper's exact MAE or its reported GPU-hours on this hardware.

## References

- [Paper](https://arxiv.org/abs/2605.05712v1)
- [Pinned upstream](https://github.com/zhenqis123/EgoEMG/tree/f7cbd048b7092139852270491d3859cd472e0311)
- [Initial audit](../../feasibility-audit.md)
