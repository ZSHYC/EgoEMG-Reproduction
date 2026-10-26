# Joint angle semantics

22 dimensions: 20 finger angles and 2 independent mocap wrist angles.
Finger angles are radians on disk; wrist pitch/yaw are degrees on disk. The model uses radians; reports use degrees.
Nominal model limits are not measured data ranges. Unknown mocap wrist limits remain unspecified.

| Index | Name | Disk field | Component | Disk unit | Raw validity | Nominal limit (rad) |
| ---: | --- | --- | ---: | --- | --- | --- |
| 0 | THUMB_CMC_FE | generated_joint_angles_{hand} | 0 | rad | generated_label_valid | [-0.787266, 1.287266] |
| 1 | THUMB_CMC_AA | generated_joint_angles_{hand} | 1 | rad | generated_label_valid | [-0.611799, 0.611799] |
| 2 | THUMB_MCP_FE | generated_joint_angles_{hand} | 2 | rad | generated_label_valid | [0.000000, 1.287266] |
| 3 | THUMB_IP_FE | generated_joint_angles_{hand} | 3 | rad | generated_label_valid | [-0.258539, 1.087266] |
| 4 | INDEX_MCP_AA | generated_joint_angles_{hand} | 4 | rad | generated_label_valid | [-0.337266, 0.487266] |
| 5 | INDEX_MCP_FE | generated_joint_angles_{hand} | 5 | rad | generated_label_valid | [-0.287266, 1.337266] |
| 6 | INDEX_PIP_FE | generated_joint_angles_{hand} | 6 | rad | generated_label_valid | [0.000000, 1.882462] |
| 7 | INDEX_DIP_FE | generated_joint_angles_{hand} | 7 | rad | generated_label_valid | [0.000000, 1.433663] |
| 8 | MIDDLE_MCP_AA | generated_joint_angles_{hand} | 8 | rad | generated_label_valid | [-0.487266, 0.287266] |
| 9 | MIDDLE_MCP_FE | generated_joint_angles_{hand} | 9 | rad | generated_label_valid | [-0.287266, 1.437266] |
| 10 | MIDDLE_PIP_FE | generated_joint_angles_{hand} | 10 | rad | generated_label_valid | [0.000000, 1.882462] |
| 11 | MIDDLE_DIP_FE | generated_joint_angles_{hand} | 11 | rad | generated_label_valid | [0.000000, 1.433663] |
| 12 | RING_MCP_AA | generated_joint_angles_{hand} | 12 | rad | generated_label_valid | [-0.287266, 0.287266] |
| 13 | RING_MCP_FE | generated_joint_angles_{hand} | 13 | rad | generated_label_valid | [-0.287266, 1.537266] |
| 14 | RING_PIP_FE | generated_joint_angles_{hand} | 14 | rad | generated_label_valid | [0.000000, 1.882462] |
| 15 | RING_DIP_FE | generated_joint_angles_{hand} | 15 | rad | generated_label_valid | [0.000000, 1.433663] |
| 16 | PINKY_MCP_AA | generated_joint_angles_{hand} | 16 | rad | generated_label_valid | [-0.687266, 0.487266] |
| 17 | PINKY_MCP_FE | generated_joint_angles_{hand} | 17 | rad | generated_label_valid | [-0.487266, 1.637267] |
| 18 | PINKY_PIP_FE | generated_joint_angles_{hand} | 18 | rad | generated_label_valid | [0.000000, 1.882462] |
| 19 | PINKY_DIP_FE | generated_joint_angles_{hand} | 19 | rad | generated_label_valid | [0.000000, 1.433663] |
| 20 | WRIST_FE | mocap_{hand}_wrist_pitch | scalar | deg | mocap_{hand}_wrist_angles_valid | unknown |
| 21 | WRIST_RU | mocap_{hand}_wrist_yaw | scalar | deg | mocap_{hand}_wrist_angles_valid | unknown |

Hand columns: left=0, right=1. Both hands use the same semantic index order.
Left-hand strategy in the release metadata: flip_local_z; no blanket angle sign inversion is applied.
PDF wrist signs: extension/radial deviation positive, left normal L cross F; legacy implementation equivalence is unverified.

- legacy_reproduction_angle_mask: first 20 use generated hand validity, last 2 use independent wrist validity; each angle must be finite before cleanup.
- official_literal_loader: generated hand validity is surfaced, but ordinary-path wrist validity is not passed to loss; dataset-name wrist masking is separate.
- The official wrist metric uses channel 20 of a per-joint mask for both wrist components.

Generation discrepancy: legacy metadata describes axis-angle projection; PDF B.3 describes landmark-level IK. Matching names do not prove equivalence.

Finger angles are radians on disk; wrist pitch/yaw are degrees on disk.
The model uses radians and reports use degrees; reporting units do not change disk units.
Nominal limits are model constraints, not measured legacy ranges or proof they were enforced.
Independent mocap wrist limits are unknown; generic model final limits are not assigned.
Matching dimension names does not establish label-generation or fitting-error equivalence.

Official revision: `f7cbd048b7092139852270491d3859cd472e0311`.
PDF reference: fixed transcription, SHA256 `124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725`.

Content identities:

- manifest.json: `ebcef7877444253869845b9337b7c0df2b16b0b6f55d83c58d6c3c03bae8ac09`
- metadata.npz: `cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57`
- metadata_summary.json: `34736a9ef3c311a0ddc4a060d6a996c6c850c26f90d1cb32856c9231fe87be62`
- egoemg_repro/data.py: `de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e`
- egoemg_repro/semantic_tables.py: `fe0e0e3a87888bcca763bd28e8ea894328316c36d60b2b9e3e1616c7b272d1ea`
- egoemg/constants.py: `01f4ad9d6be5b587d46d28b01c2325d584a593a63bd14c79dd0f8e2a1e561099`
- scripts/paper/generate_per_gesture_appendix.py: `590b85ee7f1675e82ca7cf02fa72464b6c3c335b068467ff8d35484fd91ecf1d`
- scripts/paper/plot_paper_figures.py: `0da286e14f434d8b66a3fa7cb297b4175a3b77f899ef1c7642409936cd755289`
- egoemg/UmeTrack/dataset/generic_hand_model.json: `2eb8029a8e1cedc2cf5b40c14c0955e2004277024df3c6304ce0bb74106a7c31`
- scripts/eval/analyze_per_gesture.py: `448e83f7a3698334d1813fda8ce0b7303672f9d95740442157ded4aa244d28a7`
- egoemg/datasets/egoemg_memmap_dataset.py: `9f8999ccc6a2d964713ff85886781ecc4d41ddb18f2986e0fbe5c6cc11236060`
- egoemg/metrics.py: `85aac28fc2cac11e4714bf6a18a52b2faeccc488d6135202aae8a8e974e95309`
- egoemg/lightning.py: `f32b97dab2bb94aff35d0513d964b15087a859caaf11f36030f79074edae1899`
- scripts/ik/batch_ik_mesh.py: `794fdfcb82a692812de4ca7a15fd4e41c0c3f3717424dd917fb05e4f9508da2c`
