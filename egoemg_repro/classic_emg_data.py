"""Explicit official-literal classic data over the frozen legacy memmaps.

The 7790-point cohort, pooled official normalization and generated-only mask are
a new protocol. This module neither cleans labels nor changes old runtime code.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import UPSTREAM_COMMIT
from . import classic_emg
from .data import LegacyDataset, LegacyStore, file_sha256
from .partitions import build_partitions


MASK_POLICY = 'official_literal_loader'
WINDOW_LENGTH = 7790


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _mask_policy(policy):
    if policy != MASK_POLICY:
        raise ValueError('classic mask policy must explicitly be official_literal_loader; legacy per-angle adaptation is not implemented')


def classic_normalization(upstream: Path) -> dict:
    """The pinned field-aware scalar actually selected by the official reader."""
    root = classic_emg._approved_upstream(upstream)
    asset = json.loads((root / 'assets/per_dataset_norm_stats.json').read_text())
    item = asset['egoemg__filtered']
    return dict(format='classic_fixed_normalization_v1', source_revision=UPSTREAM_COMMIT,
                asset='upstream://assets/per_dataset_norm_stats.json', asset_sha256=classic_emg.NORM_SHA256,
                dataset_name='egoemg', field='filtered', key='egoemg__filtered',
                mean=item['mean'], std=item['std'], epsilon=1e-6, mask_policy=MASK_POLICY,
                estimated_from_project_train_rows=False,
                order='filtered float32 T,C; raw zero-fill if16; pooled scalar; original resolved transforms; C,T',
                limitation='fixed public pooled asset; its sampling is not proven to match the independent retained training rows')


def _normalization(upstream, normalization, mask_policy):
    _mask_policy(mask_policy)
    expected = classic_normalization(upstream)
    if not isinstance(normalization, dict) or _hash(normalization) != _hash(expected):
        raise ValueError('normalization must exactly match the pinned official filtered scalar and mask policy')
    return expected


def _field_selection(raw):
    entry = raw['dataset']['train'][0]
    return dict(declared_modalities=entry['modalities'], declared_explicit_fields=entry.get('fields'),
                effective_filtered_fields=['emg_left_filtered', 'emg_right_filtered'],
                reason='pinned emg modality omits filtered fields; explicit target-hand field selection enables the original ordinary path')


def _expected_recipe(upstream, variant):
    if variant not in ('vemg2pose', 'neuropose'):
        raise ValueError('base classic config variant must be vemg2pose or neuropose')
    config = classic_emg.compose_classic(upstream, variant)
    root, contract, raw, _ = classic_emg._validated(upstream, config)
    effective_loss, _ = classic_emg._loss_evidence(root, raw['loss_weights'])
    early = next(c for c in raw['callbacks'] if c['_target_'].endswith('.EarlyStopping'))
    return dict(schema_version=1, protocol='classic-official-literal-legacy-memmap-v1', variant=variant,
                upstream_experiment=contract.experiment, seed=raw['seed'],
                window_length=7790, train_stride=780, eval_stride=7790, val_fraction=.12, jitter=False,
                batch_size=raw['batch_size'], num_workers=raw['num_workers'],
                max_epochs=raw['trainer']['max_epochs'], lr=raw['optimizer']['lr'],
                weight_decay=raw['optimizer']['weight_decay'], eta_min=raw['lr_scheduler']['scheduler']['eta_min'],
                gradient_clip=raw['trainer']['gradient_clip_val'], precision=raw['trainer']['precision'],
                monitor=raw['monitor_metric'], monitor_mode=raw['monitor_mode'],
                early_stopping_patience=early['patience'], augmentation='original_resolved_rotation',
                reader_field_selection=_field_selection(raw),
                layout=dict(declared=contract.declared_layout, effective=contract.effective_layout,
                            input_channels=contract.input_channels, channel_indices=list(contract.channel_indices),
                            channel_interpolate=contract.channel_interpolate, reason=contract.layout_reason),
                normalization=classic_normalization(root), mask_policy=MASK_POLICY,
                raw_loss_weights=raw['loss_weights'], effective_loss_weights=effective_loss)


def load_classic_config(path: Path, upstream: Path) -> dict:
    """Admit only a complete, exact base recipe; no implicit overrides or init."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('classic config must be an exact recipe object')
    expected = _expected_recipe(upstream, value.get('variant'))
    if _hash(value) != _hash(expected):
        raise ValueError('classic config differs from the approved exact base recipe')
    return value


def build_classic_partitions(store, *, upstream: Path, val_fraction=.12, mask_policy=MASK_POLICY) -> dict:
    """Retain old footprint guards, with a new grid and honest norm identity."""
    _mask_policy(mask_policy)
    norm = classic_normalization(upstream)
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)) or not math.isfinite(val_fraction):
        raise ValueError('val_fraction must be finite and in [0,1)')
    result = build_partitions(store, window_length=7790, train_stride=780, eval_stride=7790, val_fraction=val_fraction)
    provenance = result['provenance']
    del provenance['partition_sha256']
    provenance.update(protocol='classic-official-literal-legacy-memmap-v1', normalization=norm, mask_policy=mask_policy,
                      cohort='new 7790/780/7790 grid; distinct from the legacy 12000-point comparison cohort')
    provenance['partition_sha256'] = _hash(result)
    return result


def _partitions(root, partitions, upstream, mask_policy):
    try:
        expected = build_classic_partitions(LegacyStore(root), upstream=upstream,
            val_fraction=partitions['settings']['val_fraction'], mask_policy=mask_policy)
        if _hash(expected) != _hash(partitions):
            raise ValueError('classic partition differs from deterministic disjoint-role construction')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid classic partition') from exc


def _safe_path(root, filename):
    filename = Path(filename)
    if filename.is_absolute() or '..' in filename.parts:
        raise ValueError('unsafe classic field path')
    path = root
    for part in filename.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('classic field path must not contain a symlink')
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('unsafe classic field path')
    return path


def _schema(store):
    n = store.total_rows
    expected = {'frame_split_id': ((n,), None), 'generated_label_valid': ((n, 2), np.dtype(bool))}
    for hand in ('left', 'right'):
        expected[f'emg_{hand}_filtered'] = ((n, 8), np.dtype('float32'))
        expected[f'generated_joint_angles_{hand}'] = ((n, 20), np.dtype('float32'))
        for axis in ('pitch', 'yaw'):
            expected[f'mocap_{hand}_wrist_{axis}'] = ((n,), np.dtype('float32'))
        expected[f'mocap_{hand}_wrist_angles_valid'] = ((n,), np.dtype(bool))
    paths = {}
    for name, (shape, dtype) in expected.items():
        try:
            spec = store.fields[name]
            actual_dtype = np.dtype(spec['dtype'])
            if tuple(spec['shape']) != shape or (dtype is not None and actual_dtype != dtype):
                raise ValueError('classic field schema dtype/shape mismatch for ' + name)
            if name == 'frame_split_id' and actual_dtype.kind not in 'iu':
                raise ValueError('classic frame split dtype must be integer')
            path = _safe_path(store.root.resolve(), spec.get('filename', name + '.dat'))
            if path.stat().st_size != math.prod(shape) * actual_dtype.itemsize:
                raise ValueError('classic field schema size mismatch for ' + name)
            paths[name] = path
        except (KeyError, TypeError) as exc:
            raise ValueError('missing or invalid classic field schema for ' + name) from exc
    return paths


class ClassicDataset:
    """Readonly literal values; an explicit admission rejects nonfinite samples.

    LegacyDataset supplies only Store/window validation, never its cleaned sample.
    Original transforms are built internally from the exact approved Hydra config.
    """
    def __init__(self, root, windows, *, upstream, config, normalization, role, mask_policy):
        _mask_policy(mask_policy)  # Reject alternate protocols before source or data access.
        if role not in ('train', 'val', 'test'):
            raise ValueError('classic role must be train, val or test')
        self.normalization = _normalization(upstream, normalization, mask_policy)
        source, self.contract, _, _ = classic_emg._validated(upstream, config)
        self.base = LegacyDataset(root, windows, norm=None)
        self.store, self.windows = self.base.store, self.base.windows
        _schema(self.store)
        train_id = next(i for i, name in self.store.split_labels.items() if name == 'train')
        for w in self.windows:
            if w['role'] != role or w['end'] - w['start'] != WINDOW_LENGTH:
                raise ValueError('classic window role/length mismatch')
            split = self.store.split_labels[int(self.store.field('frame_split_id')[w['center']])]
            if w['split'] != split or (role in ('train', 'val') and split != 'train') or (role == 'test' and split == 'train'):
                raise ValueError('classic window center split/role mismatch')
            if role in ('train', 'val') and not np.all(self.store.field('frame_split_id')[w['start']:w['end']] == train_id):
                raise ValueError('classic train/val window must be wholly official train rows')
        self.role, self.mask_policy = role, mask_policy
        classic_emg._activate(source)
        from hydra.utils import instantiate
        from egoemg.transforms import Compose
        self.transform = Compose([instantiate(t) for t in config.transforms[role]])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        import torch
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        w = self.windows[index // 2]
        h = index % 2
        hand = ('left', 'right')[h]
        start, end = w['start'], w['end']
        emg = np.asarray(self.store.field(f'emg_{hand}_filtered')[start:end], dtype=np.float32)
        if self.contract.input_channels == 16:
            from egoemg.datasets.layout_utils import place_sparse_channels
            emg = place_sparse_channels(emg, 16, np.asarray(self.contract.channel_indices))
            channel_mask = np.zeros(16, dtype=bool)
            channel_mask[list(self.contract.channel_indices)] = True
        else:
            channel_mask = np.ones(8, dtype=bool)
        stats = self.normalization
        emg = (emg - stats['mean']) / (stats['std'] + stats['epsilon'])
        emg = self.transform({'emg': emg})
        if isinstance(emg, dict):
            emg = emg.get('emg', emg)
        if hasattr(emg, 'numpy'):
            emg = emg.numpy()
        fingers = np.asarray(self.store.field(f'generated_joint_angles_{hand}')[start:end]).T
        pitch = np.deg2rad(np.asarray(self.store.field(f'mocap_{hand}_wrist_pitch')[start:end], dtype=np.float32))
        yaw = np.deg2rad(np.asarray(self.store.field(f'mocap_{hand}_wrist_yaw')[start:end], dtype=np.float32))
        angles = np.concatenate([fingers, np.stack([pitch, yaw], axis=0)], axis=0)
        valid = np.asarray(self.store.field('generated_label_valid')[start:end, h])
        result = dict(w, hand=hand, subject=self.store.episodes[w['episode']]['subject'],
                      target_hand=hand, target_hand_index=h, dataset_name='egoemg', mask_policy=self.mask_policy)
        result.update(emg=torch.from_numpy(np.array(emg.T, copy=True, order='C')),
                      joint_angles=torch.from_numpy(np.array(angles, copy=True, order='C')),
                      label_valid_mask=torch.from_numpy(np.array(valid, copy=True)),
                      emg_channel_mask=torch.from_numpy(channel_mask))
        return result


def admit_classic_sample(sample, contract):
    """Explicit train/evaluation admission; no replacement or finite-mask repair."""
    import torch
    _mask_policy(sample.get('mask_policy'))
    if contract != classic_emg.classic_contract(contract.variant):
        raise ValueError('classic sample contract differs')
    for key, shape, dtype in (('emg', (contract.input_channels, 7790), torch.float32),
                              ('joint_angles', (22, 7790), torch.float32),
                              ('label_valid_mask', (7790,), torch.bool)):
        value = sample.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError('classic sample schema mismatch for ' + key)
        if dtype == torch.float32 and not torch.isfinite(value).all():
            raise ValueError('classic sample must be finite at explicit admission: ' + key)
    if 'angle_valid_mask' in sample or sample.get('dataset_name') != 'egoemg' or sample.get('hand') not in ('left', 'right'):
        raise ValueError('classic sample schema/protocol mismatch')


def build_classic_datasets(root, partitions, *, upstream, config, normalization, mask_policy):
    _normalization(upstream, normalization, mask_policy)
    classic_emg._validated(upstream, config)
    _partitions(root, partitions, upstream, mask_policy)
    return {role: ClassicDataset(root, [w for w in partitions['windows'] if w['role'] == role],
                upstream=upstream, config=config, normalization=normalization, role=role, mask_policy=mask_policy)
            for role in ('train', 'val', 'test')}


def _stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def classic_data_identity(root, partitions, *, upstream, config, normalization, mask_policy) -> dict:
    """Hash actual consumed fields, checked stable through the entire admission.

    Private roots and inode/stat stamps never enter the migratable content token.
    This is a data/config identity, not a completeness or training-success claim.
    """
    normalization = _normalization(upstream, normalization, mask_policy)
    source, contract, raw, _ = classic_emg._validated(upstream, config)
    root = Path(root).resolve()
    before = {name: _stamp(_safe_path(root, name)) for name in ('manifest.json', 'metadata.npz')}
    store = LegacyStore(root)
    paths = _schema(store)
    relative = {name: str(path.relative_to(root)) for name, path in paths.items()}
    paths.update({name: _safe_path(root, name) for name in before})
    relative.update({name: name for name in before})
    for name, path in paths.items():
        before.setdefault(name, _stamp(path))
    _partitions(root, partitions, source, mask_policy)
    records = {}
    for name, path in paths.items():
        if _stamp(_safe_path(root, relative[name])) != before[name]:
            raise ValueError('classic data changed before content hashing: ' + name)
        digest = file_sha256(path)
        if _stamp(_safe_path(root, relative[name])) != before[name]:
            raise ValueError('classic data changed during content hashing: ' + name)
        if name in store.fields:
            spec = store.fields[name]
            records[name] = dict(sha256=digest, dtype=np.dtype(spec['dtype']).str, shape=list(spec['shape']))
        else:
            records[name] = dict(sha256=digest)
    project = Path(__file__).resolve().parents[1]
    runtime = {name: file_sha256(project / name) for name in ('egoemg_repro/__init__.py', 'egoemg_repro/classic_emg.py',
        'egoemg_repro/classic_emg_data.py', 'egoemg_repro/data.py', 'egoemg_repro/partitions.py')}
    result = dict(format='classic_emg_literal_data_identity_v1', mask_policy=mask_policy, normalization=normalization,
        reader_field_selection=_field_selection(raw),
        partition_sha256=partitions['provenance']['partition_sha256'],
        fields={name: records[name] for name in store.fields if name in records},
        manifest_sha256=records['manifest.json']['sha256'], metadata_sha256=records['metadata.npz']['sha256'],
        total_rows=store.total_rows, episode_count=len(store.episodes),
        model_protocol=classic_emg.resolved_classic_identity(config, contract), runtime_files_sha256=runtime,
        finite_policy='preserve literal raw values; explicit admit_classic_sample rejects any nonfinite EMG or label',
        scope='legacy memmap content/config identity; no full-cohort completeness, finite-data scan, training or benchmark claim')
    for name in paths:
        if _stamp(_safe_path(root, relative[name])) != before[name]:
            raise ValueError('classic data changed before admission completed: ' + name)
    result['sha256'] = _hash(result)
    return result
