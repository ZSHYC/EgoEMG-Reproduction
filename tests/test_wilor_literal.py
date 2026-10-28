"""Prospective WiLoR recipe metadata; no model or real dataset is constructed."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.wilor_literal')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.wilor_literal':pytest.fail('native WiLoR configuration contract missing')
        raise


class RepeatWitnessDataset:
    def __init__(self,n,transform=None):self.n=n
    def __len__(self):return self.n
    def __getitem__(self,index):return index


def test_wilor_recipe_replaces_entire_module_and_exposes_exact_prospective_derivation():
    pytest.importorskip('hydra');m=api();cfg=m.compose_wilor(UPSTREAM);v=m.wilor_protocol(UPSTREAM,cfg)
    assert v['module']=={'_target_':'egoemg.models.modules.wilor_vit.WiLoRViTPose','out_channels':22,
        'head_hidden':512,'head_dropout':.1,'pretrained_path':None}
    assert v['recipe_origin']=='explicit_small_recipe_paper_derivation'
    assert v['base_experiment']=='fusion/vision_vit_small' and v['official_standalone_experiment_found'] is False
    assert v['batch_size_by_role']==dict(train=18,val=18,test=18) and v['unused_declared_val_batch_size']==256
    assert v['repeat_by_role']==dict(train=2,val=2,test=2)
    assert v['jitter_by_role']==dict(train=True,val=False,test=False)
    assert v['window']==dict(train_length=7790,eval_length=7790,train_stride=780,eval_stride=7790)
    assert v['optimizer']['lr']==1e-5 and v['optimizer']['weight_decay']==.01
    assert v['lr_scheduler']['scheduler']['T_max']==30 and v['lr_scheduler']['scheduler']['eta_min']==1e-5
    assert v['scheduler_behavior']=='constant_initial_lr_equals_eta_min'
    assert v['trainer']['max_epochs']==30 and v['trainer']['precision']=='bf16-mixed'
    assert v['trainer']['gradient_clip_val']==1 and v['trainer']['devices']==[0,1,2,3,4,5]
    assert v['component_lr_scales']=={'backbone':1.,'head':1.}
    assert v['raw_loss_weights']=={'mae':1.,'landmark/fingertip':.01,'fingertip_distance':.01}
    assert v['effective_loss_weights']=={'mae':1.,'landmark/fingertip':.02}
    checkpoint,_,early=v['callbacks']
    assert checkpoint['save_top_k']==3 and checkpoint['save_last'] is True and checkpoint['monitor']=='val_mae'
    assert early['patience']==30 and early['mode']=='min' and early['check_on_train_epoch_end'] is False
    assert v['model_expectations']==dict(parameters=631631657,trainable_parameters=631579158,frozen_parameters=52499,
        backbone_feature_dimension=1280,head_activation='GELU',input_rgb=[3,256,256],original_height_crop=[3,192,256],
        prediction=[22,1],frozen_backbone_components=['pose_emb','shape_emb','cam_emb','decpose','decshape','deccam'])
    assert v['paper_reference']['table']==8 and v['paper_reference']['page']==22
    assert v['paper_reference']['reported_backbone_dimension']==1576
    assert 'precision' in v['inherited_fields'] and 'weight_decay' in v['inherited_fields']
    assert v['pretrained_admission']=='required before any native fresh model; configuration is not an asset permit'
    assert v['scientific_eligible'] is False and v['full_paper_complete'] is False
    assert v['declared_training_corpus']=='physically_merged_egoemg_showee_incre'
    assert v['upstream_revision']=='f7cbd048b7092139852270491d3859cd472e0311'
    assert all(not Path(n).is_absolute() for n in v['sources_sha256'])


@pytest.mark.parametrize('damage',['batch','batch_type','repeat','val_batch_size','loss','precision','clip_type',
    'patience','floor','lr','epochs','out_channels','pretrained_path','module_type','dino_key','jitter','dataset',
    'binding','binding_extra'])
def test_wilor_rejects_changed_recipe_before_any_model_or_asset_admission(damage):
    pytest.importorskip('hydra');from omegaconf import OmegaConf
    m=api();cfg=m.compose_wilor(UPSTREAM)
    if damage=='batch':cfg.batch_size=1
    elif damage=='batch_type':cfg.batch_size=18.
    elif damage=='repeat':cfg.datamodule.dataset_repeat=1
    elif damage=='val_batch_size':cfg.val_batch_size=18
    elif damage=='loss':cfg.loss_weights.fingertip_distance=.02
    elif damage=='precision':cfg.trainer.precision='32-true'
    elif damage=='clip_type':cfg.trainer.gradient_clip_val=True
    elif damage=='patience':cfg.callbacks[2].patience=31
    elif damage=='floor':cfg.lr_scheduler.scheduler.eta_min=1e-6
    elif damage=='lr':cfg.optimizer.lr=1e-4
    elif damage=='epochs':cfg.trainer.max_epochs=150
    elif damage=='out_channels':cfg.module.out_channels=20
    elif damage=='pretrained_path':cfg.module.pretrained_path='/private/unverified.ckpt'
    elif damage=='module_type':cfg.module._target_='egoemg.models.modules.vit_vision.VisionViTPose'
    elif damage=='dino_key':OmegaConf.update(cfg,'module.backbone_type','vit_small',force_add=True)
    elif damage=='jitter':cfg.dataset.train[0].jitter=False
    elif damage=='dataset':cfg.dataset.train[0].dataset_name='egoemg'
    elif damage=='binding':cfg[m.CONTEXT_KEY].upstream='/private/other/source'
    elif damage=='binding_extra':OmegaConf.update(cfg,m.CONTEXT_KEY+'.qualified',True,force_add=True)
    with pytest.raises(ValueError):m.wilor_protocol(UPSTREAM,cfg)


@pytest.mark.parametrize('cfg',[None,True,{},'wilor'])
def test_wilor_requires_bound_configuration(cfg):
    pytest.importorskip('hydra')
    with pytest.raises(ValueError):api().wilor_protocol(UPSTREAM,cfg)


def test_public_wilor_metadata_excludes_private_environment_paths(monkeypatch):
    pytest.importorskip('hydra');m=api();monkeypatch.setenv('EGOEMG_ROOT','/private/participant-data')
    v=m.wilor_protocol(UPSTREAM,m.compose_wilor(UPSTREAM));raw=json.dumps(v)
    assert '/private/' not in raw and str(UPSTREAM) not in raw
    assert v['module']['pretrained_path'] is None


def test_native_wilor_data_declaration_uses_batch18_and_repeat2_for_every_role():
    pytest.importorskip('pytorch_lightning');m=api();m._activate(UPSTREAM)
    from egoemg.datamodule import make_data_module
    cfg=m.compose_wilor(UPSTREAM);v=m.wilor_protocol(UPSTREAM,cfg);dm=make_data_module(cfg)
    dm.dataset_conf={r:[{'_target_':'test_wilor_literal.RepeatWitnessDataset','n':3}] for r in ('train','val','test')}
    dm.norm_mode=None;dm.num_workers=dm.eval_num_workers=0
    for role in ('train','val','test'):
        ds=dm._build_dataset(role,None);assert len(ds)==6 and [ds[i] for i in range(len(ds))]==[0,1,2,0,1,2]
        loader=dm._make_loader(ds,shuffle=False)
        assert loader.batch_size==v['batch_size_by_role'][role]==18 and loader.drop_last is False


def test_wilor_contract_import_and_composition_do_not_import_torch():
    pytest.importorskip('hydra');api()
    code="""import json,sys
from egoemg_repro.wilor_literal import compose_wilor,wilor_protocol
x=wilor_protocol(sys.argv[1],compose_wilor(sys.argv[1]))
assert 'torch' not in sys.modules and not x['scientific_eligible']
print(json.dumps({'torch_imported':False,'batch':x['batch_size_by_role']['train']}))
"""
    p=subprocess.run([sys.executable,'-B','-c',code,str(UPSTREAM)],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)=={'torch_imported':False,'batch':18}
