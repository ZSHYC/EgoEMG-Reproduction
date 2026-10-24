# Fixed-center visual baselines

The adapter uses the same guarded 12,000-row manifest windows and evaluation centers as the EMG baseline. Each center yields a left and a right sample. Visual labels and per-angle validity are `(22,1)`: the existing 20 finger angles remain radians; the independently valid wrist pitch/yaw are converted from degrees once by the shared legacy adapter. The selected clean fusion recipe takes the full 12,000-row EMG window. The generic adapter can also take a shorter centered input inside the guarded footprint. It does not rebuild partitions for its shorter input.

The source video-frame number comes from `image_webcam_frame_index[center]`, with `image_head_frame_index` accepted for newer naming. `frame_index` is never a substitute. The legacy mapping is used as released; its alignment quality is not asserted to be corrected parquet ground truth. Whole-episode preprocessing and legacy-label limitations also remain.

## Acquiring and inspecting crops

`assets_crops_manifest.json` lists the public 41-episode crop inventory and exact byte lengths. The complete inventory is approximately 45.04 GB. Select episodes explicitly for a first integration check:

```bash
python -m egoemg_repro.crops download --root data/crops --inventory assets_crops_manifest.json --episodes episode_000000 --workers 2
python -m egoemg_repro.crops download --root data/crops --data data/legacy --index-root data/vision_index --inventory assets_crops_manifest.json --fields-only --workers 2
```

The second command downloads alignment fields into a separate receipt directory while signal acquisition is active. After both acquisitions finish, verify sizes and link/copy those fields into the legacy manifest's expected filenames, then incorporate their receipts into the data provenance. Do not let separate processes update one acquisition receipt concurrently. Omit `--episodes` only when the full selected crop acquisition fits, preserving the 15 GiB storage reserve. Downloads use the shared resumable, atomic, byte-validated acquisition implementation and record transfer SHA256.

A crop episode is an original directory `episode_XXXXXX.lmdb/data.mdb`, an LMDB single file `episode_XXXXXX.lmdb`, or the flattened download `episode_XXXXXX.data.mdb`. Keys are zero-padded source video-frame indices followed by `_L` or `_R`; values are 256 by 256 RGB JPEG bytes. Read handles are lazy, read-only, bounded, and local to each worker process. A missing whole episode raises an error. A missing individual key yields an ImageNet-normalized black image and `vision_valid_mask=False` while retaining that sample and center.

The official crop preparation script horizontally flips the left-hand video before cropping. The official pre-crop dataset fast path therefore performs **no additional flip**. RGB byte values are normalized with ImageNet means `(0.485,0.456,0.406)*255` and standard deviations `(0.229,0.224,0.225)*255`.

```bash
python -m egoemg_repro.crops cache --data data/legacy --partitions RUN/partitions.json --crops data/crops --output data/center_crops
```

This copies only selected JPEG keys without re-encoding them, deduplicates repeated video frames within each episode, and writes `center_cache_manifest.json` with paired sample identities, missing-key counts, and generated-cache `payload_files` byte sizes and SHA256 digests. Payload memory is bounded by one episode. The cache refuses to overwrite existing episode files; use a fresh destination when changing the selected manifest.

## Models and initialization

`VisualDataset(root, windows, crops_root, norm=None, kind='vision', emg_window_length=7790, emg_layout='target_hand')` returns core sample metadata, centered labels, per-angle masks, `vision_img`, `vision_valid_mask`, and `vision_frame_indices`. `kind='fusion'` also returns normalized target-hand EMG with the explicitly supplied length. The generic default is 7,790, while the selected clean training config supplies 12,000 and returns `(8,12000)`. The optional `emg2pose_sparse16` layout puts those channels at `[10,12,0,1,2,4,5,6]` with every other slot zero and no interpolation; it is intended for separately declared architecture/checkpoint verification.

`build_visual_model(upstream, config, vision_checkpoint=None, emg_checkpoint=None, resume_state=None)` composes the pinned official experiments and instantiates unchanged `ResNetVisionPose` or `MidFusionPoseFormer`. Vision uses the official ResNet-18 MLP and ImageNet backbone initialization by default. Fresh requested ImageNet initialization rejects a conflicting `EGOEMG_NO_PRETRAINED_DOWNLOAD` override so random weights cannot be recorded as pretrained. Explicit `vision_pretrained=false` selects recorded random initialization. Supplying a clean vision checkpoint avoids unnecessary ImageNet initialization and records the checkpoint as the actual weight source. Tests explicitly disable external pretrained downloads. Fusion requires clean reproduction checkpoints using the `{'model': state_dict}` schema. Complete vision, EMG featurizer, decoder and head schemas load strictly; partial or mismatched branches fail.

Clean fusion deliberately uses the existing official **8-channel S branch**, which accepts the independently trained 8-channel EMG model directly. It differs from the released 16-channel fusion checkpoint recipe and must be reported as an independent-validation retraining adaptation. No first-convolution expansion or unvalidated weight mapping is applied. Its input remains centered at the same evaluation point; the selected config uses the full shared 12,000-row input and the official Adam recipe (learning rate 0.0006, weight decay 0.0001, 150 epochs). The released official 8-channel fusion configuration uses decoder dropout 0.1, while the current clean EMG recipe uses 0.15; trainable weights transfer exactly and the fusion configuration governs subsequent training.

The official fusion implementation initializes the residual final layer with small normal weights (standard deviation `1e-5`). The approved reproduction design instead requires an exactly zero initial residual, so the builder explicitly zeros that final Linear layer **after** loading the clean EMG head. The source implementation is unchanged. This initialization adaptation and both checkpoint hashes are recorded in `model.reproduction_initialization`; the resolved module configuration is recorded in `model.reproduction_hydra_config`. In evaluation mode, initial fusion predictions equal the loaded vision predictor within numerical tolerance. Optimizer state is not imported from either initializer. A self-contained resume/evaluation checkpoint can pass its model state as `resume_state`; the builder then avoids pretrained downloads and original branch files, restores the entire state strictly, and preserves the learned residual without resetting it. The trainer must retain the original checkpoint initialization provenance when resuming. Initializer selection must use independent validation only; the builder does not read test scores.

`align_visual_batch(model,batch)` returns center predictions, center targets and the authoritative `(B,22,1)` per-angle mask. Missing crops do not alter that label mask. Evaluation must report both all selected labeled centers and the crop-valid matched population, with missing counts by hand/split/user. Black-image samples must not silently disappear from one model's denominator. Any training policy that excludes invalid crops must be declared separately.

The downloader and cache are available independently of training integration. Fixture tests exercise actual JPEGs stored in actual LMDBs; official-model tests require `EGOEMG_UPSTREAM` plus the pinned runtime. Fixture success is not evidence of a bounded real-data GPU run or a full baseline result.
