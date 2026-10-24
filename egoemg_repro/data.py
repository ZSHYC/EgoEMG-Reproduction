"""Read the released legacy memmaps without importing current upstream datasets.

Rows and window ends are half-open. Existing generated finger angles are radians;
wrist pitch/yaw are degrees on disk. No filtering or label regeneration occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _decoded(array):
    return [v.decode('utf-8').rstrip('\x00') if isinstance(v, bytes) else str(v) for v in array]


class LegacyStore:
    """Manifest-driven, read-only and lazy memmap access (safe to pickle workers)."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest_path = self.root / 'manifest.json'
        self.metadata_path = self.root / 'metadata.npz'
        self.manifest = json.loads(self.manifest_path.read_text())
        self._fields = self.manifest['fields']
        self._arrays = {}
        with np.load(self.metadata_path, allow_pickle=False) as metadata:
            self.metadata = {key: (_decoded(metadata[key]) if metadata[key].dtype.kind in 'SU' else metadata[key].copy()) for key in metadata.files}
        if 'episode_head_video_path' not in self.metadata and 'episode_webcam_video_path' in self.metadata:
            self.metadata['episode_head_video_path'] = list(self.metadata['episode_webcam_video_path'])
        md = self.metadata
        n = len(md['episode_id'])
        self.total_rows = int(self.manifest.get('total_rows', self._fields['frame_split_id']['shape'][0]))
        self.episodes = []
        prior_end = 0
        for i in range(n):
            start, end = int(md['episode_start_idx'][i]), int(md['episode_end_idx'][i])
            if start < prior_end or end <= start or end > self.total_rows:
                raise ValueError(f'Invalid half-open episode boundaries for episode {i}')
            if 'episode_length' in md and int(md['episode_length'][i]) != end-start:
                raise ValueError(f'Inconsistent exclusive episode end/length for episode {i}')
            self.episodes.append(dict(index=i, id=md['episode_id'][i], subject=md['episode_subject'][i], start=start, end=end))
            prior_end = end
        labels = self.manifest.get('frame_split_labels')
        if labels:
            self.split_labels = dict(enumerate(labels))
        else:
            self.split_labels = {int(i): label for i, label in zip(md['splits_split_id'], md['splits_split'])}
        if set(self.split_labels.values()) != {'train', 'user', 'gesture', 'both'}:
            raise ValueError('Expected official train/user/gesture/both split table')

    @property
    def fields(self):
        return MappingProxyType(self._fields)

    def field(self, name):
        if name not in self._arrays:
            spec = self._fields[name]
            path = self.root / spec.get('filename', f'{name}.dat')
            dtype, shape = np.dtype(spec['dtype']), tuple(spec['shape'])
            expected = int(np.prod(shape)) * dtype.itemsize
            actual = path.stat().st_size
            if actual != expected:
                raise ValueError(f'{name} file size {actual} != manifest size {expected}')
            self._arrays[name] = np.memmap(path, dtype=dtype, mode='r', shape=shape)
        return self._arrays[name]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = {}
        return state


def union_intervals(intervals):
    """Merge half-open row intervals so overlapping training windows count once."""
    merged = []
    for start, end in sorted((int(a), int(b)) for a,b in intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _validate_window(store, window):
    episode = store.episodes[int(window['episode'])]
    start, end = int(window['start']), int(window['end'])
    if start < episode['start'] or end > episode['end'] or end <= start:
        raise ValueError('Window must lie wholly inside its episode')
    if int(window['center']) != start+(end-start)//2:
        raise ValueError('Window center must match the deterministic window interval')


def compute_normalization(store, windows, chunk_rows=262144):
    """Population moments over each distinct retained training row, per hand."""
    intervals = []
    for window in windows:
        if window['role'] == 'train':
            _validate_window(store, window)
            intervals.append((window['start'],window['end']))
    intervals = union_intervals(intervals)
    if not intervals:
        raise ValueError('No retained training samples for normalization')
    result = {}
    for hand in ('left','right'):
        source = store.field(f'emg_{hand}_filtered')
        count, mean, m2 = 0, np.zeros(8), np.zeros(8)
        for start,end in intervals:
            for offset in range(start,end,chunk_rows):
                values = np.asarray(source[offset:min(end,offset+chunk_rows)],dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite training EMG for {hand}')
                n = len(values)
                local_mean = values.mean(axis=0)
                delta = local_mean - mean
                m2 += ((values-local_mean)**2).sum(axis=0) + delta**2 * count*n/(count+n)
                mean += delta*n/(count+n)
                count += n
        std = np.sqrt(m2/count)
        std[std < 1e-8] = 1.0
        result[hand] = dict(mean=mean.tolist(), std=std.tolist(), count=count)
    return result


class LegacyDataset:
    """Each fixed interval appears twice, left then right; jitter is disabled."""

    def __init__(self, root, windows, norm=None):
        self.store = LegacyStore(root)
        self.windows = [dict(window) for window in windows]
        self.norm = norm
        for window in self.windows:
            _validate_window(self.store, window)

    def __len__(self):
        return 2*len(self.windows)

    def __getitem__(self, index):
        import torch  # Partitioning and metadata inspection need no torch runtime.
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = self.windows[index//2]
        hand_index = index%2
        hand = ('left','right')[hand_index]
        start,end = window['start'],window['end']
        store = self.store
        emg = np.array(store.field(f'emg_{hand}_filtered')[start:end], dtype=np.float32).T
        if self.norm is not None:
            stats = self.norm[hand]
            mean = np.asarray(stats['mean'],dtype=np.float32)[:,None]
            std = np.asarray(stats['std'],dtype=np.float32)[:,None]
            if mean.shape != (8,1) or std.shape != (8,1) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
                raise ValueError('Normalization requires finite 8-channel means and positive standard deviations')
            emg = (emg-mean)/std
        fingers = np.asarray(store.field(f'generated_joint_angles_{hand}')[start:end],dtype=np.float32).T
        wrist = np.stack([store.field(f'mocap_{hand}_wrist_{axis}')[start:end] for axis in ('pitch','yaw')])
        angles = np.concatenate([fingers,np.deg2rad(wrist)],axis=0).astype(np.float32)
        label_valid = np.array(store.field('generated_label_valid')[start:end,hand_index],dtype=bool)
        wrist_valid = np.array(store.field(f'mocap_{hand}_wrist_angles_valid')[start:end],dtype=bool)
        mask = np.concatenate([np.broadcast_to(label_valid,(20,end-start)),np.broadcast_to(wrist_valid,(2,end-start))]).copy()
        mask &= np.isfinite(angles)
        angles = np.nan_to_num(angles,nan=0,posinf=0,neginf=0)
        if not np.isfinite(emg).all():
            raise ValueError('Nonfinite EMG in dataset window')
        result = dict(window, hand=hand, subject=store.episodes[window['episode']]['subject'])
        result.update(emg=torch.from_numpy(np.ascontiguousarray(emg)), joint_angles=torch.from_numpy(angles),
                      label_valid_mask=torch.from_numpy(label_valid), angle_valid_mask=torch.from_numpy(mask))
        return result
