"""Source-bound prospective WiLoR22-angle configuration, never an asset permit.

The pinned release has a module but no standalone vision experiment. This
explicit S-recipe/paper derivation preserves inherited code behavior, including
its constant learning rate. No model, dataset, tensor library or GPU is loaded.
"""
from __future__ import annotations

from pathlib import Path

from . import UPSTREAM_COMMIT
from .classic_emg import _activate,_approved_upstream,_file_hash,_hash

CONTEXT_KEY='_native_wilor'
EXPERIMENT='fusion/vision_vit_small'
MODULE='config/module/wilor_vit.yaml'
SOURCE_FILES=('config/base.yaml','config/lineage/_base.yaml','config/lineage/fusion.yaml',
    'config/experiment/fusion/vision_vit_small.yaml',MODULE,
    'config/dataset/egoemg_unified_angle_regression.yaml','config/datamodule/default.yaml',
    'config/lr_scheduler/cosine.yaml','egoemg/datamodule.py','egoemg/train.py',
    'egoemg/lightning.py','egoemg/models/modules/wilor_vit.py')
PAPER_SHA256='124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725'


def _binding(root):
    return dict(format='native_wilor_config_binding_v1',upstream=str(root),
        recipe_origin='explicit_small_recipe_paper_derivation')


def _compose(root):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root/'config'),version_base='1.1'):
        config=compose(config_name='base',overrides=['experiment='+EXPERIMENT])
    # Assignment replaces the complete mapping; DINO constructor kwargs cannot leak.
    config.module=OmegaConf.load(root/MODULE)
    OmegaConf.update(config,'module.pretrained_path',None,force_add=True)
    config.batch_size=18;config.optimizer.lr=1e-5;config.trainer.max_epochs=30
    OmegaConf.resolve(config)
    return config


def compose_wilor(upstream):
    """Compose one declared derivation; no arbitrary caller recipe overrides."""
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream);config=_compose(root)
    OmegaConf.update(config,CONTEXT_KEY,_binding(root),force_add=True)
    return config


def _validated(upstream,config):
    from omegaconf import OmegaConf
    root=_approved_upstream(upstream)
    try:
        binding=OmegaConf.to_container(config[CONTEXT_KEY],resolve=True)
        if _hash(binding)!=_hash(_binding(root)):raise ValueError('WiLoR source binding differs')
        actual=OmegaConf.to_container(config,resolve=True);del actual[CONTEXT_KEY]
        expected=OmegaConf.to_container(_compose(root),resolve=True)
        if _hash(actual)!=_hash(expected):raise ValueError('WiLoR configuration differs from explicit pinned derivation')
        return root,actual
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('bound native WiLoR configuration required') from exc


def wilor_protocol(upstream,config):
    """Return safe recipe metadata, distinguishing source inheritance from evidence."""
    root,value=_validated(upstream,config)
    raw=value['loss_weights'];effective={}
    for name,weight in raw.items():
        key={'fingertip_distance':'landmark/fingertip','landmark_distance':'landmark/all'}.get(name,name)
        effective[key]=effective.get(key,0.)+float(weight)
    dataset=value['dataset'];dm=value['datamodule']
    archived=sorted(str(p.relative_to(root)) for p in (root/'config/experiment').rglob('*wilor*.yaml'))
    result=dict(format='native_wilor_execution_contract_v1',scientific_eligible=False,
        production_ready=False,full_paper_complete=False,upstream_revision=UPSTREAM_COMMIT,
        recipe_origin='explicit_small_recipe_paper_derivation',base_experiment=EXPERIMENT,
        official_standalone_experiment_found=False,archived_fusion_experiments=archived,
        derivation=dict(module='replace complete mapping with original module/wilor_vit; no pretrained file admitted',
            batch_size='paper Table8 value18',lr='paper C.2/Table8 value1e-5',max_epochs='paper Table8 value30',
            other_fields='unchanged original vision-S inheritance; not an author standalone WiLoR recipe'),
        inherited_fields=['weight_decay','precision','gradient_clip_val','callbacks','component_lr_scales',
            'seed','data_window_stride_repeat_jitter','effective_loss_aliases','unused_val_batch_size'],
        module=value['module'],raw_loss_weights=raw,effective_loss_weights=effective,
        optimizer=value['optimizer'],lr_scheduler=value['lr_scheduler'],
        scheduler_behavior='constant_initial_lr_equals_eta_min',trainer=value['trainer'],callbacks=value['callbacks'],
        component_lr_scales=value['component_lr_scales'],seed=value['seed'],
        seed_policy='original setup_runtime before native factory; this contract does not construct a runtime',
        batch_size_by_role={r:value['batch_size'] for r in ('train','val','test')},
        unused_declared_val_batch_size=value['val_batch_size'],
        batch_size_semantics='native make_data_module passes batch_size only; every role uses self.batch_size',
        repeat_by_role={r:dm['dataset_repeat'] for r in ('train','val','test')},
        repeat_semantics='native _build_dataset repeats every role after left/right concatenation',
        jitter_by_role={r:dataset[r][0]['jitter'] for r in ('train','val','test')},
        window=dict(train_length=dm['window_length'],eval_length=dm['val_test_window_length'],
            train_stride=dm['stride'],eval_stride=dm['val_test_stride']),
        declared_training_corpus='physically_merged_egoemg_showee_incre',
        declared_dataset_names={r:[v['dataset_name'] for v in dataset[r]] for r in ('train','val','test')},
        declared_validation_splits=list(dataset['val'][0]['allowed_splits']),
        data_admission='none; legacy-only data and independent jitter-safe validation need separate content binding',
        mask_semantics='original generated center label mask for all22 angles; no per-angle legacy cleanup',
        missing_crop_semantics='original normalized black RGB with vision-invalid flag; label mask unchanged',
        pretrained_admission='required before any native fresh model; configuration is not an asset permit',
        model_expectations=dict(parameters=631631657,trainable_parameters=631579158,frozen_parameters=52499,
            backbone_feature_dimension=1280,head_activation='GELU',input_rgb=[3,256,256],
            original_height_crop=[3,192,256],prediction=[22,1],
            frozen_backbone_components=['pose_emb','shape_emb','cam_emb','decpose','decshape','deccam']),
        paper_reference=dict(filename='EgoEMG.pdf',sha256=PAPER_SHA256,table=8,page=22,
            reference_kind='fixed transcription of supplied PDF, not runtime PDF read',reported_backbone_dimension=1576,
            code_disagreement='retain code1280 feature and GELU head; not paper1576 or generic ReLU',
            precision_scope='paper BF16 sentence explicitly names generic vision; WiLoR BF16 is inherited code derivation'),
        distributed_scope='trainer metadata retains original six-device declaration; no single-rank equivalence or execution claim',
        sources_sha256={n:_file_hash(root/n) for n in SOURCE_FILES})
    result['sha256']=_hash(result)
    return result
