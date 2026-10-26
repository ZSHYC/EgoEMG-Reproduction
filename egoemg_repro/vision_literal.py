"""Pinned original vision configuration and executed-loader contract.

No model, datamodule, optimizer or CUDA object is constructed here. The source
declares a merged training corpus; this metadata does not admit available data
as that corpus or authorize a scientific result.
"""
from __future__ import annotations

from pathlib import Path

from . import UPSTREAM_COMMIT
from .classic_emg import _activate, _approved_upstream, _file_hash, _hash


VARIANTS=('vit_small','vit_base','vit_large')
CONTEXT_KEY='_literal_vision'
EXPERIMENT='fusion/vision_vit_small'
SEED_POLICY='original setup_runtime before native factory; configuration metadata alone does not construct a runtime'
SOURCE_FILES=('config/base.yaml','config/lineage/_base.yaml','config/lineage/fusion.yaml',
    'config/experiment/fusion/vision_vit_small.yaml','config/dataset/egoemg_unified_angle_regression.yaml',
    'config/datamodule/default.yaml','config/lr_scheduler/cosine.yaml','egoemg/datamodule.py',
    'egoemg/lightning.py','egoemg/train.py','egoemg/models/modules/vit_vision.py')


def _variant(value):
    if type(value) is not str or value not in VARIANTS:raise ValueError('unsupported original vision variant')
    return value


def _binding(root,variant):
    return dict(format='original_vision_config_binding_v1',upstream=str(root),variant=variant,seed_policy=SEED_POLICY)


def _compose(root,variant):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    overrides=['experiment='+EXPERIMENT]
    if variant!='vit_small':overrides.append('module.backbone_type='+variant)
    with initialize_config_dir(config_dir=str(root/'config'),version_base='1.1'):
        config=compose(config_name='base',overrides=overrides)
    OmegaConf.resolve(config)
    return config


def compose_vision(upstream,variant='vit_small'):
    """One exact author S recipe or an explicit B/L backbone derivation."""
    variant=_variant(variant);root=_approved_upstream(upstream);config=_compose(root,variant)
    from omegaconf import OmegaConf
    OmegaConf.update(config,CONTEXT_KEY,_binding(root,variant),force_add=True)
    return config


def _validated(upstream,config):
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream)
    try:
        binding=OmegaConf.to_container(config[CONTEXT_KEY],resolve=True);variant=_variant(binding['variant'])
        if _hash(binding)!=_hash(_binding(root,variant)):raise ValueError('vision config source binding differs')
        actual=OmegaConf.to_container(config,resolve=True);del actual[CONTEXT_KEY]
        expected=OmegaConf.to_container(_compose(root,variant),resolve=True)
        if _hash(actual)!=_hash(expected):raise ValueError('vision config differs from exact original recipe/explicit derivation')
        return root,variant,actual
    except (KeyError,TypeError,AttributeError) as exc:raise ValueError('invalid original vision config binding') from exc


def vision_protocol(upstream,config):
    """Safe recipe metadata, distinguishing YAML declarations from native loader execution."""
    root,variant,value=_validated(upstream,config)
    raw=value['loss_weights'];effective={}
    for name,weight in raw.items():
        key={'fingertip_distance':'landmark/fingertip','landmark_distance':'landmark/all'}.get(name,name)
        effective[key]=effective.get(key,0.)+float(weight)
    dataset=value['dataset'];dm=value['datamodule']
    result=dict(format='original_vision_execution_contract_v1',scientific_eligible=False,production_ready=False,
        full_paper_complete=False,upstream_revision=UPSTREAM_COMMIT,variant=variant,upstream_experiment=EXPERIMENT,
        recipe_origin='pinned_author_experiment' if variant=='vit_small' else 'explicit_small_recipe_derivation',
        module=value['module'],raw_loss_weights=raw,effective_loss_weights=effective,
        optimizer=value['optimizer'],lr_scheduler=value['lr_scheduler'],trainer=value['trainer'],callbacks=value['callbacks'],
        seed=value['seed'],seed_policy=SEED_POLICY,
        batch_size_by_role={role:value['batch_size'] for role in ('train','val','test')},
        unused_declared_val_batch_size=value['val_batch_size'],
        batch_size_semantics='native make_data_module passes batch_size only; all _make_loader roles use self.batch_size',
        repeat_by_role={role:dm['dataset_repeat'] for role in ('train','val','test')},
        repeat_semantics='native _build_dataset repeats every role after left/right concatenation',
        jitter_by_role={role:dataset[role][0]['jitter'] for role in ('train','val','test')},
        window=dict(train_length=dm['window_length'],eval_length=dm['val_test_window_length'],
            train_stride=dm['stride'],eval_stride=dm['val_test_stride']),
        declared_training_corpus='physically_merged_egoemg_showee_incre',
        declared_dataset_names={role:[entry['dataset_name'] for entry in dataset[role]] for role in ('train','val','test')},
        declared_validation_splits=list(dataset['val'][0]['allowed_splits']),
        data_admission='not performed; legacy-only cohort and independent train-only validation require explicit separate identity',
        mask_semantics='original generated center label mask for all22 angles; no legacy per-angle wrist cleanup',
        missing_crop_semantics='original point reader returns normalized black input and vision-invalid flag; label mask unchanged',
        sources_sha256={name:_file_hash(root/name) for name in SOURCE_FILES})
    result['sha256']=_hash(result)
    return result
