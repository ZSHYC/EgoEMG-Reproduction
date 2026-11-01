"""Exact RN18+S metadata contracts; no training/data/checkpoint permit."""
import hashlib
import importlib
import os
from pathlib import Path

import pytest

pytest.importorskip('hydra')
OmegaConf=pytest.importorskip('omegaconf').OmegaConf


def component():return importlib.import_module('egoemg_repro.rn18_fusion_literal')


def upstream():
    p=Path(os.environ.get('EGOEMG_TEST_UPSTREAM',Path(__file__).resolve().parents[1]/'upstream'))
    if not(p/'config/base.yaml').is_file():pytest.skip('pinned original upstream required')
    return p


def test_exact_rn18_small_recipe_is_distinct_from_vit_and_rn50_middle():
    m=component();root=upstream();cfg=m.compose_rn18_fusion(root);p=m.rn18_fusion_protocol(root,cfg)
    assert p['upstream_experiment']=='fusion/fusion_rn18_s_center_8ch'
    assert p['module']['_target_']=='egoemg.models.modules.mid_fusion.MidFusionPoseFormer'
    assert p['module']['vision_backbone_type']=='resnet18'and p['module']['vision_embed_dim']==512
    assert p['module']['featurizer']['conv_blocks'][0]['in_channels']==8
    assert p['module']['decoder']['num_layers']==3 and p['module']['decoder']['dropout']==.1
    assert p['module']['fusion_mode']=='center_supervised'and p['module']['out_channels']==22
    assert p['layout']==dict(name='target_hand',input_channels=8,channel_indices=None,channel_interpolate=False)
    assert p['window']==dict(train_length=12000,eval_length=12000,train_stride=1200,eval_stride=12000)
    assert p['center_target_only']is True and p['vision_patch_size']==256
    assert p['batch_size_by_role']==dict(train=200,val=200,test=200)
    assert p['workers_by_role']==dict(train=10,val=10,test=10)and p['unused_declared_val_batch_size']==100
    assert p['repeat_by_role']==dict(train=1,val=1,test=1)
    assert p['jitter_by_role']==dict(train=True,val=False,test=False)
    assert p['loader']==dict(prefetch_factor=2,persistent_workers=True,pin_memory=True)
    assert p['optimizer']==dict(_target_='torch.optim.Adam',lr=.0006,betas=[.9,.999],eps=1e-8,weight_decay=.0001,amsgrad=False)
    assert p['lr_scheduler']['scheduler']['T_max']==150 and p['lr_scheduler']['scheduler']['eta_min']==1e-5
    assert p['trainer']['max_epochs']==150 and p['trainer']['check_val_every_n_epoch']==1
    assert p['trainer']['devices']==[2,3,4,5]and p['trainer']['precision']=='bf16-mixed'and p['trainer']['gradient_clip_val']==1
    assert not any(x['_target_'].endswith('EarlyStopping')for x in p['callbacks'])
    assert p['callbacks'][0]['monitor']=='val_mae'and p['callbacks'][0]['mode']=='min'
    assert p['callbacks'][0]['save_top_k']==3 and p['callbacks'][0]['save_last']is True
    assert p['raw_loss_weights']['landmark/fingertip']==.01 and p['raw_loss_weights']['fingertip_distance']==.01
    assert p['effective_loss_weights']['landmark/fingertip']==.02
    assert p['module']['freeze_vision_branch']is False and p['module']['freeze_vision_head']is False
    assert p['module']['lock_vision_batch_norm']is False
    assert p['freeze_comment_executable_conflict']['execute_unfrozen']is True
    assert p['seed']==42 and p['matmul_precision']=='high'
    assert p['normalization_asset']=='assets/per_dataset_norm_stats_repro_filtered_paper_alias.json'
    assert p['normalization_asset_sha256']=='d4b73febf386ec7b2028639eed8bbd7363d13cbe3fe489149915bcef071ee96a'
    assert p['declared_dataset_names']=={r:['egoemg_unified','egoemg_unified']for r in('train','val','test')}
    assert all(p[k]is False for k in('scientific_eligible','production_ready','full_paper_complete','dataset_admitted','initializers_admitted','native_model_constructed'))
    source='config/experiment/fusion/fusion_rn18_s_center_8ch.yaml'
    assert p['sources_sha256'][source]==hashlib.sha256((root/source).read_bytes()).hexdigest()
    assert p['module']['vision_pretrained_checkpoint']=='<explicit completed RN18 last checkpoint>'


@pytest.mark.parametrize('path,value',[
    ('module.vision_backbone_type','resnet50'),('module.vision_embed_dim',2048),
    ('module.featurizer.conv_blocks.0.in_channels',16),('module.decoder.num_layers',6),
    ('module.fusion_mode','vision_only'),('module.freeze_vision_branch',True),('module.freeze_vision_head',True),
    ('module.lock_vision_batch_norm',True),('module.vision_pretrained_checkpoint','unadmitted.ckpt'),
    ('pretrained_emg_checkpoint','unadmitted.ckpt'),('batch_size',100),('val_batch_size',200),('num_workers',0),
    ('datamodule.window_length',7790),('datamodule.val_test_stride',7790),('datamodule.stride',400),
    ('datamodule.dataset_repeat',2),('datamodule.pin_memory',False),('datamodule.prefetch_factor',1),
    ('egoemg_emg_layout','emg2pose_interpolate16'),('center_target_only',False),
    ('optimizer._target_','torch.optim.AdamW'),('optimizer.lr',.0001),
    ('loss_weights.fingertip_distance',0.),('loss_weights.landmark/fingertip',0.),
    ('trainer.check_val_every_n_epoch',2),('trainer.max_epochs',100),('trainer.precision','32-true'),
    ('callbacks.0.monitor','val_loss'),('seed',43)])
def test_any_bound_config_drift_rejects_before_constructor(path,value):
    m=component();root=upstream();cfg=m.compose_rn18_fusion(root);OmegaConf.update(cfg,path,value,force_add=True)
    with pytest.raises(ValueError):m.rn18_fusion_protocol(root,cfg)


@pytest.mark.parametrize('damage',['unbound','other_family','foreign_source'])
def test_recipe_or_source_binding_cannot_be_forged(damage):
    m=component();root=upstream();cfg=m.compose_rn18_fusion(root)
    if damage=='unbound':
        raw=OmegaConf.to_container(cfg,resolve=True);del raw[m.CONTEXT_KEY];cfg=OmegaConf.create(raw)
    elif damage=='other_family':cfg[m.CONTEXT_KEY].experiment='fusion/fusion_resnet_emgfusion_center'
    else:cfg[m.CONTEXT_KEY].upstream='/unadmitted/source'
    with pytest.raises(ValueError):m._validated(root,cfg)


def test_metadata_cannot_instantiate_data_models_or_optimizer(monkeypatch):
    m=component();root=upstream()
    def deny(*a,**kw):raise AssertionError('RN18 fusion metadata cannot instantiate runtime objects')
    monkeypatch.setattr('hydra.utils.instantiate',deny)
    p=m.rn18_fusion_protocol(root,m.compose_rn18_fusion(root))
    assert not p['native_model_constructed']and not p['initializers_admitted']
