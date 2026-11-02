"""Complete official reference centers with the native pure-EMG point reader.

Input identity/coverage only: neither a model evaluation nor training admission.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .classic_emg import _activate, _approved_upstream, _file_hash, _hash
from .classic_emg_cohort import _safe, _StatGuard
from .classic_emg_data import _safe_path
from .data import LegacyStore
from .emg_initializer_data import _schema, _view
from .emgformer_literal import NORM_ASSET, _validated

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = _file_hash(__file__)
FORMAT = 'native_complete_official_center_reference_grid_v1'
OFFICIAL = {
    'egoemg/center_frame.py': '9ed981e3652d94a25d4607ff3e0eb3802b0f279077eccafe933773026aeb085a',
    'egoemg/datasets/egoemg_memmap_dataset.py': '9f8999ccc6a2d964713ff85886781ecc4d41ddb18f2986e0fbe5c6cc11236060',
}


def _sources(upstream):
    if _file_hash(__file__) != IMPORTED_SHA:
        raise ValueError('loaded reference input source changed')
    if any(_file_hash(upstream / name) != digest for name, digest in OFFICIAL.items()):
        raise ValueError('fixed official center reference source changed')
    names = ('egoemg_repro/center_reference_data.py', 'egoemg_repro/emg_initializer_data.py',
        'egoemg_repro/emgformer_literal.py', 'egoemg_repro/data.py',
        'egoemg_repro/classic_emg_cohort.py', 'egoemg_repro/classic_emg_data.py')
    return {**{name: ROOT / name for name in names},
        **{'upstream/' + name: upstream / name for name in OFFICIAL},
        'upstream/' + NORM_ASSET: upstream / NORM_ASSET}


def reference_windows(data, upstream, *, required_window_length=12000):
    """Return the complete sealed official reference grid, never a subset."""
    if type(required_window_length) is not int or required_window_length != 12000:
        raise ValueError('this native S/M/L reference requires the maximum trained window12000')
    data = _safe(data)
    upstream = _approved_upstream(upstream)
    paths = {'metadata:manifest': data / 'manifest.json', 'metadata:episodes': data / 'metadata.npz',
        **_sources(upstream)}
    guard = _StatGuard(paths.values())
    store = LegacyStore(data)
    for name in ('episode_index', 'frame_split_id'):
        spec = store.fields.get(name)
        if (not spec or tuple(spec['shape']) != (store.total_rows,)
                or np.dtype(spec['dtype']).kind not in 'iu'):
            raise ValueError('complete integer reference episode/split field required')
        path = _safe_path(data, spec.get('filename', name + '.dat'))
        paths['field:' + name] = path
        guard.add(path)
        store.field(name)  # Validate actual bytes and schema before upstream IO.
    source_hashes = {name: _file_hash(path) for name, path in paths.items()}
    guard.check()
    _activate(upstream)
    from egoemg import center_frame
    if _file_hash(center_frame.__file__) != OFFICIAL['egoemg/center_frame.py']:
        raise ValueError('imported official center collector differs')
    grid = center_frame.collect_val_centers(data, required_window_length)
    windows = []
    split = store.field('frame_split_id')
    episode_index = store.field('episode_index')
    for episode, values in grid.items():
        if type(episode) is not int or not 0 <= episode < len(store.episodes):
            raise ValueError('reference episode metadata differs')
        source = store.episodes[episode]
        for center, reference_start in values:
            start, end = center - 6000, center + 6000
            label = store.split_labels.get(int(split[center]))
            if (type(center) is not int or type(reference_start) is not int
                    or center != reference_start + 3895 or int(episode_index[center]) != episode
                    or start < source['start'] or end > source['end'] - 1
                    or label not in ('gesture', 'user', 'both')):
                raise ValueError('official reference center/boundary/split differs')
            windows.append(dict(episode=episode, start=start, end=end, center=center,
                reference_start=reference_start, role='test', split=label))
    if not windows or len({w['center'] for w in windows}) != len(windows):
        raise ValueError('complete nonempty unique official reference grid required')
    guard.check()
    value = dict(format=FORMAT, reference_window_length=7790, required_window_length=12000,
        windows=windows, source_sha256=source_hashes,
        counts=dict(windows=len(windows), hand_requests=2 * len(windows),
            by_split={s: 2 * sum(w['split'] == s for w in windows) for s in ('gesture', 'user', 'both')}),
        scientific_eligible=False, full_paper_complete=False)
    return dict(value, sha256=_hash(value))


class CenterReferenceDataset:
    """All official reference windows, left then right, without visual IO."""

    def __init__(self, data, upstream, config, windows):
        self.data = _safe(data)
        root, _, _ = _validated(upstream, config)
        paths = {'metadata:manifest': self.data / 'manifest.json',
            'metadata:episodes': self.data / 'metadata.npz', **_sources(root)}
        self._guard = _StatGuard(paths.values())
        self.store = LegacyStore(self.data)
        for name, path in _schema(self.store).items():
            paths['field:' + name] = path
            self._guard.add(path)
        if 'episode_index' not in self.store.fields:
            raise ValueError('complete reference episode source required')
        episode_path = _safe_path(self.data,
            self.store.fields['episode_index'].get('filename', 'episode_index.dat'))
        self._guard.add(episode_path)
        expected = reference_windows(self.data, root)
        if windows != expected:
            raise ValueError('caller reference grid differs from complete official source')
        self.grid = copy.deepcopy(expected)
        self.windows = self.grid['windows']
        self._grid_sha256 = expected['sha256']
        _activate(root)
        self.views = [_view(self.store, self.windows, hand, 'test', root, config)
            for hand in ('left', 'right')]
        self._guard.check()

    def __len__(self):
        return 2 * len(self.windows)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        if (_hash({k: v for k, v in self.grid.items() if k != 'sha256'}) != self._grid_sha256
                or self.grid['sha256'] != self._grid_sha256 or self.windows != self.grid['windows']):
            raise ValueError('complete reference grid changed')
        self._guard.check()
        hand_index, position = divmod(index, len(self.windows))
        hand = ('left', 'right')[hand_index]
        window = self.windows[position]
        sample = self.views[hand_index]._getitem_center_supervised_emg(
            window['episode'], window['start'], window['end'])
        for name, shape, dtype in (('emg', (8, 12000), np.float32),
                ('joint_angles', (22, 1), np.float32), ('label_valid_mask', (1,), np.bool_)):
            array = sample.get(name)
            if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
                raise ValueError('typed native same-center input differs: ' + name)
            if not bool(np.isfinite(array).all()):
                raise ValueError('nonfinite native same-center input: ' + name)
        if sample['target_hand'] != hand or 'vision_img' in sample or 'vision_valid_mask' in sample:
            raise ValueError('native pure-EMG hand or input modality differs')
        sample.update(episode=window['episode'], start=window['start'], end=window['end'],
            center=window['center'], hand=hand, split=window['split'], role='test',
            subject=str(self.store.episodes[window['episode']]['subject']), jitter_offset=0,
            reference_sample_index=index, reference_unique_sample_index=index,
            reference_repeat_index=0)
        self._guard.check()
        return sample
