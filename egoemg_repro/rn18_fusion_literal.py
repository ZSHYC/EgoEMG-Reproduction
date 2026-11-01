"""Pinned RN18+S author composition, without runtime or initializer admission."""
from __future__ import annotations

import copy

from . import UPSTREAM_COMMIT
from .classic_emg import _approved_upstream,_file_hash,_hash

CONTEXT_KEY='_literal_rn18_fusion'
EXPERIMENT='fusion/fusion_rn18_s_center_8ch'
NORM_ASSET='assets/per_dataset_norm_stats_repro_filtered_paper_alias.json'
SEED_POLICY='original setup_runtime before native factory; preserve original branch and residual initialization'
IMPORTED_SHA=_file_hash(__file__)
NATIVE=('egoemg/train.py','egoemg/_train_utils.py','egoemg/lightning.py','egoemg/datamodule.py',
    'egoemg/models/modules/mid_fusion.py','egoemg/models/modules/resnet_vision.py',
    'egoemg/models/modules/base.py','egoemg/models/modules/_pooling.py',
    'egoemg/models/featurizers/tds.py','egoemg/models/decoders/transformer.py',
    'egoemg/datasets/egoemg_memmap_dataset.py','egoemg/datasets/layout_utils.py',
    'egoemg/transforms.py','egoemg/transforms_batch.py','egoemg/metrics.py')


def _binding(root):
    return dict(format='original_rn18_fusion_config_binding_v1',upstream=str(root),
        experiment=EXPERIMENT,seed_policy=SEED_POLICY)


def _compose(root):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root/'config'),version_base='1.1'):
        cfg=compose(config_name='base',overrides=['experiment='+EXPERIMENT])
    OmegaConf.resolve(cfg);return cfg


def compose_rn18_fusion(upstream):
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream);cfg=_compose(root)
    OmegaConf.update(cfg,CONTEXT_KEY,_binding(root),force_add=True);return cfg


def _validated(upstream,config):
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream)
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 fusion source changed')
    try:
        if _hash(OmegaConf.to_container(config[CONTEXT_KEY],resolve=True))!=_hash(_binding(root)):
            raise ValueError('RN18 fusion source/family binding differs')
        actual=OmegaConf.to_container(config,resolve=True);del actual[CONTEXT_KEY]
        if _hash(actual)!=_hash(OmegaConf.to_container(_compose(root),resolve=True)):
            raise ValueError('RN18 fusion configuration differs from exact author recipe')
        return root,actual
    except (KeyError,TypeError,AttributeError)as exc:raise ValueError('invalid RN18 fusion configuration binding')from exc


def rn18_fusion_protocol(upstream,config):
    root,value=_validated(upstream,config);dm=value['datamodule'];roles=('train','val','test')
    effective={}
    for name,weight in value['loss_weights'].items():
        key={'fingertip_distance':'landmark/fingertip','landmark_distance':'landmark/all'}.get(name,name)
        effective[key]=effective.get(key,0.)+float(weight)
    module=copy.deepcopy(value['module'])
    module['vision_pretrained_checkpoint']='<explicit completed RN18 last checkpoint>'
    if module.get('mano_model_path')is not None:module['mano_model_path']='<declared unused MANO path>'
    eval_workers=dm.get('eval_num_workers');eval_workers=value['num_workers']if eval_workers is None else eval_workers
    paths={*root.glob('config/**/*.yaml'),*(root/name for name in NATIVE),root/NORM_ASSET}
    result=dict(format='original_rn18_small_fusion_execution_contract_v1',upstream_revision=UPSTREAM_COMMIT,
        upstream_experiment=EXPERIMENT,recipe_origin='pinned_author_experiment',module=module,
        raw_loss_weights=value['loss_weights'],effective_loss_weights=effective,
        optimizer=value['optimizer'],lr_scheduler=value['lr_scheduler'],trainer=value['trainer'],callbacks=value['callbacks'],
        component_lr_scales=value['component_lr_scales'],transforms=value['transforms'],batch_augmentation=value['batch_augmentation'],
        seed=value['seed'],seed_policy=SEED_POLICY,matmul_precision=value['matmul_precision'],
        layout=dict(name=value['egoemg_emg_layout'],input_channels=value['module']['featurizer']['conv_blocks'][0]['in_channels'],
            channel_indices=value['egoemg_emg2pose_channel_indices'],channel_interpolate=value['egoemg_channel_interpolate']),
        window=dict(train_length=dm['window_length'],eval_length=dm['val_test_window_length'],train_stride=dm['stride'],eval_stride=dm['val_test_stride']),
        batch_size_by_role={r:value['batch_size']for r in roles},unused_declared_val_batch_size=value['val_batch_size'],
        batch_size_semantics='native global batch_size for all roles; declared val_batch_size unused',
        workers_by_role=dict(train=value['num_workers'],val=eval_workers,test=eval_workers),
        loader={k:dm[k]for k in('prefetch_factor','persistent_workers','pin_memory')},
        repeat_by_role={r:dm['dataset_repeat']for r in roles},jitter_by_role={r:value['dataset'][r][0]['jitter']for r in roles},
        vision_patch_size=value['vision_patch_size'],vision_num_frames=value['vision_num_frames'],
        vision_frame_selection=value['vision_frame_selection'],center_target_only=value['center_target_only'],
        requested_emg_field=value['egoemg_emg_field_preference'],normalization_asset=NORM_ASSET,
        normalization_asset_sha256=_file_hash(root/NORM_ASSET),
        declared_dataset_names={r:[d['dataset_name']for d in value['dataset'][r]]for r in roles},
        declared_corpus='physically merged EgoEMG/ShowEE/Incre; not admitted by this contract',
        normalization_admission='not performed; declared namespace and legacy substitution require separate native reader proof',
        initialization_rules=dict(vision='completed RN18 LAST',emg='compatible completed EMG validation BEST',
            transfer='original allowed-prefix exact-name/shape transfer; mismatches recorded, no expansion',
            residual='native final Linear normal(std=1e-5), zero bias; not exact-zero weights'),
        freeze_comment_executable_conflict=dict(source_comment='Vision backbone frozen',
            execute_unfrozen=not module['freeze_vision_branch'],decision='follow executable resolved flags'),
        sources_sha256={str(p.relative_to(root)):_file_hash(p)for p in sorted(paths)},
        local_sources_sha256={__file__.split('/egoemg_repro/')[-1]:IMPORTED_SHA},
        native_model_constructed=False,dataset_admitted=False,initializers_admitted=False,
        scientific_eligible=False,production_ready=False,full_paper_complete=False)
    result['sha256']=_hash(result);return result
