"""Exact original cotrain-S initializer recipe, never a training permit."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import UPSTREAM_COMMIT
from .classic_emg import _activate,_approved_upstream,_file_hash,_hash

CONTEXT_KEY='_literal_emg_initializer'
EXPERIMENT='emgformer/regression_egoemg_incre_cotrain_wl12000_small'
NORM_ASSET='assets/per_dataset_norm_stats_repro_filtered_paper_alias.json'
NATIVE_SOURCES=('egoemg/datamodule.py','egoemg/lightning.py','egoemg/train.py','egoemg/_train_utils.py',
    'egoemg/metrics.py','egoemg/transforms.py','egoemg/datasets/egoemg_memmap_dataset.py')


def _binding(root):
    return dict(format='original_emg_initializer_config_binding_v1',upstream=str(root),experiment=EXPERIMENT)


def _compose(root):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root/'config'),version_base='1.1'):
        cfg=compose(config_name='base',overrides=['experiment='+EXPERIMENT])
    OmegaConf.resolve(cfg);return cfg


def compose_emg_initializer(upstream):
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream);cfg=_compose(root)
    OmegaConf.update(cfg,CONTEXT_KEY,_binding(root),force_add=True);return cfg


def _validated(upstream,config):
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream)
    try:
        value=OmegaConf.to_container(config,resolve=True)
        if _hash(value.pop(CONTEXT_KEY))!=_hash(_binding(root)) or _hash(value)!=_hash(OmegaConf.to_container(_compose(root),resolve=True)):
            raise ValueError('EMG initializer config differs from exact original recipe')
        return root,value
    except (KeyError,TypeError,AttributeError) as exc:raise ValueError('invalid EMG initializer config binding') from exc


def emg_initializer_protocol(upstream,config):
    root,value=_validated(upstream,config);_activate(root)
    from egoemg.datamodule import make_data_module
    dm=make_data_module(config);effective={}
    for name,weight in value['loss_weights'].items():
        key={'fingertip_distance':'landmark/fingertip','landmark_distance':'landmark/all'}.get(name,name)
        effective[key]=effective.get(key,0.)+float(weight)
    paths=sorted({*root.glob('config/**/*.yaml'),*(root/name for name in NATIVE_SOURCES),root/NORM_ASSET})
    result=dict(format='original_emg_initializer_execution_contract_v1',upstream_revision=UPSTREAM_COMMIT,
        upstream_experiment=EXPERIMENT,module=value['module'],optimizer=value['optimizer'],lr_scheduler=value['lr_scheduler'],
        trainer=value['trainer'],callbacks=value['callbacks'],raw_loss_weights=value['loss_weights'],effective_loss_weights=effective,
        transforms=value['transforms'],batch_augmentation=value['batch_augmentation'],
        seed=value['seed'],matmul_precision=value['matmul_precision'],
        seed_policy='original setup_runtime before original native module factory',
        batch_size_by_role={role:dm.batch_size for role in ('train','val','test')},dataset_repeat=dm.dataset_repeat,
        window=dict(train_length=dm.window_length,eval_length=dm.val_test_window_length,train_stride=dm.stride,eval_stride=dm.val_test_stride),
        declared_training_corpus='EgoEMG both hands plus Incre right hand; data not admitted here',
        evaluation_adaptation='initializer val uses fixed future fusion validation intervals; no initializer test role',
        sources_sha256={str(path.relative_to(root)):_file_hash(path) for path in paths},
        scientific_eligible=False,production_ready=False,full_paper_complete=False)
    result['sha256']=_hash(result);return result


def emg_initializer_normalization(upstream):
    root=_approved_upstream(upstream);_activate(root)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    hands={}
    for hand in ('left','right'):
        view=SimpleNamespace(norm_mode='per-dataset',norm_stats_path=str(root/NORM_ASSET),dataset_name='egoemg',
            emg_field_preference='filtered_paper',emg_layout='target_hand',target_hand=hand)
        EgoEmgMemmapDataset._load_norm_stats(view)
        mean,std=np.asarray(view._emg_mean),np.asarray(view._emg_std)
        if mean.shape!=(8,) or std.shape!=(8,) or mean.dtype!=np.float32 or std.dtype!=np.float32 or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
            raise ValueError('original per-hand eight-channel normalization differs')
        hands[hand]=dict(mean=mean.tolist(),std=std.tolist(),dtype='float32')
    return dict(format='original_emg_initializer_per_hand_normalization_v1',asset=NORM_ASSET,asset_sha256=_file_hash(root/NORM_ASSET),
        namespace='egoemg',requested_field='filtered_paper',physical_legacy_field='filtered',hands=hands,epsilon=1e-6,
        native_loader='EgoEmgMemmapDataset._load_norm_stats',estimated_from_project_train_rows=False,
        scientific_eligible=False,production_ready=False,full_paper_complete=False)
