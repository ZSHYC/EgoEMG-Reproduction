"""Exact original EgoEMG S/M/L recipes; no corpus or training admission."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np

from . import UPSTREAM_COMMIT
from .classic_emg import _activate, _approved_upstream, _file_hash, _hash

SIZES = ('small', 'middle', 'large')
CONTEXT_KEY = '_literal_egoemg_emgformer'
NORM_ASSET = 'assets/per_dataset_norm_stats_repro_filtered_paper_alias.json'
NATIVE = ('egoemg/datamodule.py', 'egoemg/lightning.py', 'egoemg/train.py',
          'egoemg/_train_utils.py', 'egoemg/metrics.py', 'egoemg/transforms.py',
          'egoemg/datasets/egoemg_memmap_dataset.py',
          'egoemg/UmeTrack/dataset/generic_hand_model.json')


def _size(size):
    if type(size) is not str or size not in SIZES:
        raise ValueError('original EgoEMG size must be small, middle or large')
    return size


def _binding(root, size):
    return dict(format='original_egoemg_emgformer_config_v1', upstream=str(root), size=size,
                experiment='emgformer/egoemg_emgformer_'+size)


def _compose(root, size):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root/'config'), version_base='1.1'):
        cfg = compose(config_name='base', overrides=['experiment=emgformer/egoemg_emgformer_'+size])
    OmegaConf.resolve(cfg)
    return cfg


def compose_emgformer(upstream, size='small'):
    size = _size(size)
    root = _approved_upstream(upstream)
    cfg = _compose(root, size)
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, CONTEXT_KEY, _binding(root, size), force_add=True)
    return cfg


def _validated(upstream, config):
    from omegaconf import OmegaConf
    root = _approved_upstream(upstream)
    try:
        binding = OmegaConf.to_container(config[CONTEXT_KEY], resolve=True)
        size = _size(binding['size'])
        value = OmegaConf.to_container(config, resolve=True)
        del value[CONTEXT_KEY]
        if (_hash(binding) != _hash(_binding(root, size)) or
                _hash(value) != _hash(OmegaConf.to_container(_compose(root, size), resolve=True))):
            raise ValueError('EMGFormer config differs from exact original recipe/binding')
        return root, size, value
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid original EMGFormer configuration binding') from exc


def _normalization_source(root, value):
    declared = value['datamodule']['per_dataset_norm_stats_path']
    if not isinstance(declared, str) or not declared:
        raise ValueError('declared EMGFormer normalization source is missing')
    candidate = Path(declared)
    if not candidate.is_absolute():
        candidate = root/candidate
    if candidate.resolve() != (root/NORM_ASSET).resolve():
        raise ValueError('declared EMGFormer normalization source differs from pinned asset')
    return dict(declared_path=declared, relative_path_base='approved upstream root',
                asset=NORM_ASSET, asset_sha256=_file_hash(candidate))


def emgformer_protocol(upstream, config):
    root, size, value = _validated(upstream, config)
    effective = {}
    for name, weight in value['loss_weights'].items():
        key = {'fingertip_distance': 'landmark/fingertip', 'landmark_distance': 'landmark/all'}.get(name, name)
        effective[key] = effective.get(key, 0.) + float(weight)
    dm = value['datamodule']
    paths = sorted({*root.glob('config/**/*.yaml'), *(root/name for name in NATIVE), root/NORM_ASSET})
    result = dict(format='original_egoemg_emgformer_execution_contract_v1', size=size,
        upstream_revision=UPSTREAM_COMMIT, upstream_experiment='emgformer/egoemg_emgformer_'+size,
        module=value['module'], optimizer=value['optimizer'], lr_scheduler=value['lr_scheduler'],
        trainer=value['trainer'], callbacks=value['callbacks'], batch_size=value['batch_size'],
        raw_loss_weights=value['loss_weights'], effective_loss_weights=effective,
        seed=value['seed'], matmul_precision=value['matmul_precision'],
        transforms=value['transforms'], batch_augmentation=value['batch_augmentation'],
        declared_num_workers=value['num_workers'],
        window=dict(train_length=dm['window_length'], eval_length=dm['val_test_window_length'],
                    train_stride=dm['stride'], eval_stride=dm['val_test_stride']),
        declared_layout=value['egoemg_emg_layout'], requested_emg_field=value['egoemg_emg_field_preference'],
        normalization_source=_normalization_source(root, value),
        declared_corpus='unified EgoEMG/ShowEE/Incre; this contract has not admitted those arrays',
        native_loader_executed=False, dataset_admitted=False, scientific_eligible=False,
        production_ready=False, full_paper_complete=False,
        sources_sha256={str(p.relative_to(root)): _file_hash(p) for p in paths})
    result['sha256'] = _hash(result)
    return result


def emgformer_normalization(upstream, config):
    root, size, value = _validated(upstream, config)
    source = _normalization_source(root, value)
    _activate(root)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    hands = {}
    for hand in ('left', 'right'):
        view = SimpleNamespace(norm_mode='per-dataset', norm_stats_path=str(root/source['asset']),
            dataset_name=value['eval_dataset_name'], emg_field_preference=value['egoemg_emg_field_preference'],
            emg_layout=value['egoemg_emg_layout'], target_hand=hand)
        EgoEmgMemmapDataset._load_norm_stats(view)
        mean, std = np.asarray(view._emg_mean), np.asarray(view._emg_std)
        if (mean.shape != (8,) or std.shape != (8,) or mean.dtype != np.float32 or std.dtype != np.float32
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError('original per-hand float32 eight-channel normalization differs')
        hands[hand] = dict(mean=mean.tolist(), std=std.tolist(), dtype='float32')
    return dict(format='original_egoemg_emgformer_normalization_v1', size=size, asset=source['asset'],
                asset_sha256=source['asset_sha256'], declared_source=source, hands=hands,
                epsilon=value['datamodule']['norm_eps'], estimated_from_project_train_rows=False,
                dataset_admitted=False, scientific_eligible=False, production_ready=False, full_paper_complete=False)
