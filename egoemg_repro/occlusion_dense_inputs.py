"""Source-native paired dense inputs; no full-content or trained-model permit.

Standalone EMG is read independently with its original eight-channel per-hand
normalization, on the exact fusion-selected window. No augmentation, jitter,
repeat, nearest-center substitution or extra full-window split filter is added.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .classic_emg import _activate, _file_hash, _hash
from .classic_emg_cohort import _safe, _StatGuard
from .data import LegacyStore
from .emg_initializer_literal import compose_emg_initializer, emg_initializer_normalization
from .fusion_literal_cohort import _numeric_paths, _bind_crop_generations, _transform
from .fusion_literal_data import _FusionPointView, fusion_normalization
from .occlusion_dense_data import native_dense_points, _typed_sha, _recipe
from .rn18_fusion_data import _Rn18PointView, rn18_fusion_normalization
from .vision_literal_data import _crop_path, _schema

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = _file_hash(__file__)
LOCAL = ('occlusion_dense_inputs', 'occlusion_dense_data', 'fusion_literal_data',
    'fusion_literal_cohort', 'vision_literal_data', 'rn18_fusion_data',
    'emg_initializer_literal', 'classic_emg', 'classic_emg_cohort', 'data')


class DensePredictionDataset:
    """Pure IO over all original source keys; use a separate inference admission."""
    def __init__(self, data, crops, *, upstream, config, recipe, expected_index=None):
        if _file_hash(__file__) != IMPORTED_SHA:
            raise ValueError('imported dense prediction input source changed')
        data, crops, upstream = _safe(data), _safe(crops), _safe(upstream)
        source = {name: ROOT/'egoemg_repro'/(name+'.py') for name in LOCAL}
        guard = _StatGuard([data/'manifest.json', data/'metadata.npz', *source.values()])
        self.index = native_dense_points(data, upstream=upstream, config=config,
            recipe=recipe, expected=expected_index)
        self.upstream, self.config, self.recipe = upstream, copy.deepcopy(config), recipe
        _, value, _ = _recipe(upstream, config, recipe)
        self.store = LegacyStore(data)
        self.frame_mapping_field = _schema(self.store)
        self.paths = _numeric_paths(self.store)
        for p in self.paths.values():
            guard.add(p)
        databases = {}
        for ep in sorted(set(self.index['points']['episode'].tolist())):
            database = _crop_path(crops, self.store.episodes[ep]['id'])
            payload = _safe(database/'data.mdb' if database.is_dir() else database)
            self.paths['crop:'+str(ep)] = payload
            databases[database] = payload
            guard.add(payload)
        _bind_crop_generations(databases, guard)
        self.emg_config = compose_emg_initializer(upstream)
        standalone_norm = emg_initializer_normalization(upstream)
        fusion_norm = (fusion_normalization(upstream) if recipe == 'vits_s'
            else rn18_fusion_normalization(upstream, config))
        for name in {standalone_norm['asset'], fusion_norm['asset'],
                'egoemg/datasets/egoemg_memmap_dataset.py', 'egoemg/datasets/layout_utils.py', 'egoemg/transforms.py'}:
            path = _safe(upstream/name.removeprefix('upstream://'))
            guard.add(path)
            source['upstream/'+name] = path
        _activate(upstream)
        from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
        from egoemg.transforms import Compose
        from hydra.utils import instantiate
        self.native = EgoEmgMemmapDataset
        self.fusion_views, self.standalone_views = [], []
        length = self.index['identity']['protocol']['window_length']
        for hand in ('left', 'right'):
            fusion_transforms = Compose([instantiate(t) for t in config.transforms.test])
            emg_transforms = Compose([instantiate(t) for t in self.emg_config.transforms.val])
            cls = _FusionPointView if recipe == 'vits_s' else _Rn18PointView
            fusion = cls(self.store, crops, hand, self.frame_mapping_field,
                self.native, fusion_norm, fusion_transforms)
            standalone = _Rn18PointView(self.store, crops, hand, self.frame_mapping_field,
                self.native, standalone_norm, emg_transforms)
            fusion.window_length = standalone.window_length = length
            self.fusion_views.append(fusion)
            self.standalone_views.append(standalone)
        guard.check()
        self.guard, self._sources = guard, source
        self.identity = dict(format='source_native_dense_paired_input_io_v1', recipe=recipe,
            index_identity_sha256=self.index['identity']['sha256'],
            window_length=length, dense_stride=self.index['identity']['protocol']['dense_stride'],
            unique_hand_points=len(self.index['points']['sequence']),
            windows_crossing_center_split=self.index['identity']['windows_crossing_center_split'],
            fusion_input_channels=value['module']['featurizer']['conv_blocks'][0]['in_channels'],
            standalone_input_channels=8, fusion_normalization=fusion_norm,
            standalone_normalization=standalone_norm,
            source_files_sha256={n: _file_hash(p) for n, p in source.items()},
            reader_methods=['EgoEmgMemmapDataset._getitem_center_supervised',
                'EgoEmgMemmapDataset._getitem_center_supervised_emg'],
            standalone_window_policy='exact selected fusion window; no independent 12000-point center filtering',
            crop_and_label_masks_retained=True, native_point_wrist_conversion=True,
            jitter=False, repeat=1, full_input_or_model_admitted=False,
            scientific_eligible=False, figure13_reproduced=False, full_paper_complete=False)
        self.identity['sha256'] = _hash(self.identity)
        self._seal = self.identity['sha256']
        self._index_seal = self.index['identity']['sha256']
        self._arrays = dict(self.index['identity']['arrays_sha256'])
        self._readers = self._reader_binding()
        self.check()

    def _reader_binding(self):
        views = []
        for view in (*self.fusion_views, *self.standalone_views):
            if view.store is not self.store or view.native is not self.native:
                raise ValueError('native paired reader provider/store differs')
            arrays = {}
            for name, a in view._frame_memmaps.items():
                physical = (self.frame_mapping_field if name == 'image_head_frame_index'
                    else name.removesuffix('_paper') if name.endswith('_filtered_paper') else name)
                if (not isinstance(a, np.memmap) or a is not self.store.field(physical)
                    or a.flags.writeable or a.mode != 'r' or a.offset != 0
                    or not a.flags.c_contiguous or _safe(a.filename) != self.paths['field:'+physical]):
                    raise ValueError('native paired readonly input map differs')
                arrays[name] = [a.dtype.str, list(a.shape), list(a.strides), physical]
            views.append(dict(kind=type(view).__module__+'.'+type(view).__name__,
                hand=view.target_hand, layout=view.emg_layout, preference=view.emg_field_preference,
                interpolate=view.channel_interpolate, norm_mode=view.norm_mode,
                mean=np.asarray(view._emg_mean).tolist(), std=np.asarray(view._emg_std).tolist(),
                vision_mean=view.vision_mean.tolist(), vision_std=view.vision_std.tolist(),
                patch=view.vision_patch_size, crops=str(view.crops),
                episodes=list(view._episode_id), window_length=view.window_length,
                positions=getattr(view, '_emg2pose_channel_indices', None),
                transform=_transform(view.transform), arrays=arrays))
        return _hash(views)

    def check(self):
        self.guard.check()
        if (self.identity.get('sha256') != self._seal
            or _hash({k: v for k, v in self.identity.items() if k != 'sha256'}) != self._seal
            or _hash({k: v for k, v in self.index['identity'].items() if k != 'sha256'}) != self._index_seal
            or self.index['identity'].get('sha256') != self._index_seal
            or set(self.index['points']) != set(self._arrays)
            or any(a.flags.writeable or _typed_sha(a) != self._arrays[n] for n, a in self.index['points'].items())
            or self._reader_binding() != self._readers):
            raise ValueError('native paired input identity, keys or reader semantics changed')

    def __len__(self):
        return len(self.index['points']['sequence'])

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        self.check()
        p = self.index['points']
        ep, start, end, center, hi = (int(p[n][index]) for n in ('episode', 'start', 'end', 'center', 'hand_index'))
        fusion = self.native._getitem_center_supervised(self.fusion_views[hi], ep, start, end, center)
        emg = self.native._getitem_center_supervised_emg(self.standalone_views[hi], ep, start, end)
        for packet, channels in ((fusion, self.identity['fusion_input_channels']), (emg, 8)):
            if (packet['emg'].dtype != np.float32 or packet['emg'].shape != (channels, end-start)
                or packet['joint_angles'].dtype != np.float32 or packet['joint_angles'].shape != (22, 1)
                or not np.isfinite(packet['emg']).all() or not np.isfinite(packet['joint_angles']).all()):
                raise ValueError('complete finite original paired EMG/center targets required; no repair')
        if (not np.array_equal(fusion['joint_angles'], emg['joint_angles'])
            or not np.array_equal(fusion['label_valid_mask'], emg['label_valid_mask'])):
            raise ValueError('standalone and fusion inputs disagree on the exact center target/mask')
        key = dict(sequence=index, split=('user', 'gesture', 'both')[int(p['group_index'][index])],
            hand=('left', 'right')[hi], ep_idx=ep, center_idx=center, window_start=start, window_end=end,
            native_index=int(p['native_index'][index]), group_index=int(p['group_index'][index]))
        self.check()
        return dict(key=key, fusion=fusion, standalone_emg=emg)
