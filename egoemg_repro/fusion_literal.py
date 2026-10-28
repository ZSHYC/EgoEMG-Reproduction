"""Exact pinned sixteen-slot ViT-S fusion configuration and safe metadata.

Composition admits neither initializers nor data and constructs no model.
The original declared merged training corpus is not replaced by a legacy claim.
"""
from __future__ import annotations

import copy

from . import UPSTREAM_COMMIT
from .classic_emg import _activate, _approved_upstream, _file_hash, _hash


CONTEXT_KEY='_literal_fusion'
EXPERIMENT='fusion/fusion_vits_s_center_16ch_wl7790'
SEED_POLICY='original setup_runtime before native factory; native branch and residual initialization retained'
SOURCE_FILES=('config/base.yaml','config/lineage/_base.yaml','config/lineage/fusion.yaml',
    'config/experiment/'+EXPERIMENT+'.yaml','config/module/mid_fusion.yaml',
    'config/module/featurizer/tds_slim_16ch.yaml','config/module/decoder/transformer/preset/small.yaml',
    'config/module/head/mlp.yaml','config/transforms/emgformer_regression_aug_extended.yaml',
    'config/dataset/egoemg_unified_angle_regression.yaml','config/datamodule/default.yaml',
    'config/lr_scheduler/cosine.yaml','egoemg/datamodule.py','egoemg/lightning.py',
    'egoemg/train.py','egoemg/_train_utils.py','egoemg/models/modules/mid_fusion.py',
    'egoemg/datasets/egoemg_memmap_dataset.py')


def _binding(root):
    return dict(format='original_fusion_config_binding_v1',upstream=str(root),
                experiment=EXPERIMENT,seed_policy=SEED_POLICY)


def _compose(root):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root/'config'),version_base='1.1'):
        config=compose(config_name='base',overrides=['experiment='+EXPERIMENT])
    OmegaConf.resolve(config)
    return config


def compose_fusion(upstream):
    """Return the unedited author training recipe plus explicit source binding."""
    root=_approved_upstream(upstream)
    from omegaconf import OmegaConf
    config=_compose(root)
    OmegaConf.update(config,CONTEXT_KEY,_binding(root),force_add=True)
    return config


def _validated(upstream,config):
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream)
    try:
        binding=OmegaConf.to_container(config[CONTEXT_KEY],resolve=True)
        if _hash(binding)!=_hash(_binding(root)):raise ValueError('fusion config source binding differs')
        value=OmegaConf.to_container(config,resolve=True);del value[CONTEXT_KEY]
        expected=OmegaConf.to_container(_compose(root),resolve=True)
        if _hash(value)!=_hash(expected):raise ValueError('fusion config differs from exact original recipe')
        return root,value
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('invalid original fusion configuration binding') from exc


def fusion_protocol(upstream,config):
    """Safe executed-loader metadata, without source paths or data admission."""
    root,value=_validated(upstream,config);dm=value['datamodule'];dataset=value['dataset']
    module=copy.deepcopy(value['module'])
    module['vision_pretrained_checkpoint']='<explicit local vision branch checkpoint>'
    if module.get('mano_model_path') is not None:module['mano_model_path']='<declared MANO path>'
    raw=value['loss_weights'];effective={}
    for name,weight in raw.items():
        key={'fingertip_distance':'landmark/fingertip','landmark_distance':'landmark/all'}.get(name,name)
        effective[key]=effective.get(key,0.)+float(weight)
    result=dict(format='original_fusion_execution_contract_v1',scientific_eligible=False,
        production_ready=False,full_paper_complete=False,upstream_revision=UPSTREAM_COMMIT,
        upstream_experiment=EXPERIMENT,recipe_origin='pinned_author_training_recipe_under_released_evaluation',
        module=module,raw_loss_weights=raw,effective_loss_weights=effective,
        optimizer=value['optimizer'],lr_scheduler=value['lr_scheduler'],trainer=value['trainer'],
        callbacks=value['callbacks'],component_lr_scales=value['component_lr_scales'],
        seed=value['seed'],seed_policy=SEED_POLICY,matmul_precision=value['matmul_precision'],
        layout=dict(name=value['egoemg_emg_layout'],input_channels=16,source_channels=8,
            channel_indices=value['egoemg_emg2pose_channel_indices'],
            channel_interpolate=value['egoemg_channel_interpolate'],
            placement='native place_sparse_channels; zero-fill missing slots'),
        window=dict(train_length=dm['window_length'],eval_length=dm['val_test_window_length'],
            train_stride=dm['stride'],eval_stride=dm['val_test_stride']),
        batch_size_by_role={role:value['batch_size'] for role in ('train','val','test')},
        unused_declared_val_batch_size=value['val_batch_size'],
        batch_size_semantics='native make_data_module passes batch_size only; all roles use self.batch_size',
        repeat_by_role={role:dm['dataset_repeat'] for role in ('train','val','test')},
        repeat_semantics='native _build_dataset repeats every role after left/right concatenation',
        jitter_by_role={role:dataset[role][0]['jitter'] for role in ('train','val','test')},
        declared_training_corpus='physically_merged_egoemg_showee_incre',
        declared_dataset_names={role:[entry['dataset_name'] for entry in dataset[role]] for role in ('train','val','test')},
        declared_validation_splits=list(dataset['val'][0]['allowed_splits']),
        data_admission='not performed; independent legacy roles require a separate explicit identity',
        initialization_admission='not performed; runtime must verify both local branch files',
        emg_initialization_rule='native allowed-prefix exact-name/shape transfer; mismatches recorded and skipped',
        residual_initialization='native last Linear normal(std=1e-5), zero bias; not exact-zero weights',
        sources_sha256={name:_file_hash(root/name) for name in SOURCE_FILES})
    result['sha256']=_hash(result)
    return result
