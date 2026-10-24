"""Small on-disk arrays following the inspected public v2 schema."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from egoemg_repro.data import LegacyDataset, LegacyStore, compute_normalization


def make_store(root, splits=None, lengths=(40, 7), labels=('train', 'user', 'gesture', 'both')):
    root = Path(root)
    n = sum(lengths)
    if splits is None:
        splits = np.zeros(n, dtype=np.int8)
    arrays = {'frame_split_id': np.array(splits, dtype=np.int8),
              'generated_label_valid': np.ones((n, 2), dtype=bool)}
    arrays['generated_label_valid'][1, 0] = False
    for hand, offset in [('left', 0), ('right', 100)]:
        arrays[f'emg_{hand}_filtered'] = np.tile(np.arange(n, dtype=np.float32)[:, None] + offset, (1, 8))
        arrays[f'generated_joint_angles_{hand}'] = np.full((n, 20), 0.5 + offset, np.float32)
        arrays[f'mocap_{hand}_wrist_pitch'] = np.full(n, 90, np.float32)
        arrays[f'mocap_{hand}_wrist_yaw'] = np.full(n, -180, np.float32)
        arrays[f'mocap_{hand}_wrist_angles_valid'] = np.ones(n, dtype=bool)
    arrays['mocap_left_wrist_angles_valid'][2] = False
    fields = {}
    for key, arr in arrays.items():
        filename = 'custom_' + key + '.dat'
        arr.tofile(root / filename)
        fields[key] = dict(filename=filename, dtype=str(arr.dtype), shape=list(arr.shape))
    # An absent irrelevant file proves lazy access, rather than loading all fields.
    fields['unused_video_feature'] = dict(filename='absent.dat', dtype='float32', shape=[n, 32])
    (root / 'manifest.json').write_text(json.dumps(dict(format_version='egoemg_v2_memmap', total_rows=n, fields=fields, frame_split_labels=list(labels))))
    starts = np.cumsum([0] + list(lengths[:-1]))
    np.savez(root / 'metadata.npz', episode_id=np.array([f'episode_{i}' for i in range(len(lengths))], dtype='S20'),
             episode_subject=np.array([f'subject_{i}' for i in range(len(lengths))], dtype='S20'),
             episode_start_idx=starts, episode_end_idx=starts + lengths, episode_length=np.array(lengths),
             episode_webcam_video_path=np.array(['webcam.mp4'] * len(lengths), dtype='S20'),
             splits_split=np.array(labels, dtype='S20'), splits_split_id=np.arange(len(labels)))
    return arrays


def window(start=0, end=4, role='train', episode=0):
    return dict(episode=episode, start=start, end=end, center=(start + end)//2, role=role, split='train')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_store(self.root)
        self.addCleanup(self.tmp.cleanup)

    def test_real_metadata_webcam_alias_and_exclusive_episode_boundaries(self):
        store = LegacyStore(str(self.root))
        self.assertEqual(store.episodes[1], dict(index=1, id='episode_1', subject='subject_1', start=40, end=47))
        self.assertEqual(store.metadata['episode_head_video_path'][0], 'webcam.mp4')
        arr = store.field('emg_left_filtered')
        self.assertEqual(arr.shape, (47, 8))
        self.assertFalse(arr.flags.writeable)

    def test_reject_truncated_field_when_accessed(self):
        (self.root / 'custom_emg_left_filtered.dat').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'size'):
            LegacyStore(self.root).field('emg_left_filtered')

    def test_normalization_union_ignores_overlap_and_heldout_values(self):
        store = LegacyStore(self.root)
        windows = [window(0, 4), window(2, 6), window(40, 47, 'test', 1)]
        norm = compute_normalization(store, windows)
        np.testing.assert_allclose(norm['left']['mean'], [2.5] * 8)
        np.testing.assert_allclose(norm['right']['mean'], [102.5] * 8)
        np.testing.assert_allclose(norm['left']['std'], [np.sqrt(35/12)] * 8)

    def test_normalization_requires_training_samples(self):
        with self.assertRaisesRegex(ValueError, 'training'):
            compute_normalization(LegacyStore(self.root), [window(role='val')])

    def test_dataset_rejects_cross_episode_window(self):
        with self.assertRaisesRegex(ValueError, 'episode'):
            LegacyDataset(self.root, [window(38, 42)])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'real torch runtime required')
    def test_tensor_masks_units_and_both_hands_use_identical_intervals(self):
        ds = LegacyDataset(self.root, [window()])
        self.assertEqual(len(ds), 2)
        left, right = ds[0], ds[1]
        self.assertEqual(tuple(left['emg'].shape), (8, 4))
        self.assertEqual(tuple(left['joint_angles'].shape), (22, 4))
        np.testing.assert_allclose(left['joint_angles'][0], [0.5]*4)
        np.testing.assert_allclose(left['joint_angles'][20], [np.pi/2]*4)
        np.testing.assert_allclose(left['joint_angles'][21], [-np.pi]*4)
        self.assertFalse(left['angle_valid_mask'][0, 1])
        self.assertTrue(left['angle_valid_mask'][20, 1])
        self.assertFalse(left['angle_valid_mask'][20, 2])
        self.assertTrue(left['angle_valid_mask'][0, 2])
        self.assertTrue(right['angle_valid_mask'].all())
        self.assertEqual([left['hand'], right['hand']], ['left', 'right'])
        for key in ('episode', 'center', 'subject', 'split', 'start', 'end'):
            self.assertEqual(left[key], right[key])


if __name__ == '__main__':
    unittest.main()
