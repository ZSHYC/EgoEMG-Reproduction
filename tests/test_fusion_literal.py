"""Exact original fusion metadata; no construction or scientific admission."""
import copy
import importlib
import json
import os
from pathlib import Path

import pytest


UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.fusion_literal')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.fusion_literal':pytest.fail('exact native fusion contract missing')
        raise


class RepeatWitnessDataset:
    def __init__(self,n,transform=None):self.n=n;self.transform=transform
    def __len__(self):return self.n
    def __getitem__(self,index):return index


def test_fusion_composition_is_exact_original_training_recipe_with_only_source_binding():
    pytest.importorskip('hydra');m=api()
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    actual=m.compose_fusion(UPSTREAM)
    with initialize_config_dir(config_dir=str(UPSTREAM/'config'),version_base='1.1'):
        original=compose(config_name='base',overrides=['experiment=fusion/fusion_vits_s_center_16ch_wl7790'])
    raw=OmegaConf.to_container(actual,resolve=True);binding=raw.pop(m.CONTEXT_KEY)
    assert raw==OmegaConf.to_container(original,resolve=True)
    assert binding['upstream']==str(UPSTREAM) and binding['experiment']==m.EXPERIMENT


def test_fusion_protocol_retains_actual_sixteen_slot_recipe_and_separate_completion_boundary():
    pytest.importorskip('hydra');m=api();p=m.fusion_protocol(UPSTREAM,m.compose_fusion(UPSTREAM))
    assert p['upstream_experiment']=='fusion/fusion_vits_s_center_16ch_wl7790'
    assert p['module']['_target_']=='egoemg.models.modules.mid_fusion.MidFusionPoseFormer'
    assert p['module']['vision_backbone_type']=='vit_small' and p['module']['fusion_mode']=='center_supervised'
    assert p['module']['featurizer']['conv_blocks'][0]['in_channels']==16
    assert p['module']['decoder']['num_layers']==3 and p['module']['decoder']['num_heads']==4
    assert p['layout']==dict(name='emg2pose_interpolate16',input_channels=16,
        source_channels=8,channel_indices=[10,12,0,1,2,4,5,6],channel_interpolate=False,
        placement='native place_sparse_channels; zero-fill missing slots')
    assert p['window']==dict(train_length=7790,eval_length=7790,train_stride=400,eval_stride=7790)
    assert p['raw_loss_weights']=={'mae':1.,'landmark/fingertip':.01,'fingertip_distance':.01,'delta_reg':0.}
    assert p['effective_loss_weights']=={'mae':1.,'landmark/fingertip':.02,'delta_reg':0.}
    assert p['batch_size_by_role']==dict(train=200,val=200,test=200)
    assert p['unused_declared_val_batch_size']==100
    assert p['repeat_by_role']==dict(train=2,val=2,test=2)
    assert p['jitter_by_role']==dict(train=True,val=False,test=False)
    assert p['trainer']['precision']=='bf16-mixed' and p['trainer']['max_epochs']==150
    assert p['trainer']['gradient_clip_val']==1 and p['matmul_precision']=='high'
    assert p['optimizer']['lr']==.0001 and p['optimizer']['weight_decay']==.01
    assert p['lr_scheduler']['scheduler']['T_max']==150 and p['lr_scheduler']['scheduler']['eta_min']==1e-5
    assert all(v==1 for v in p['component_lr_scales'].values())
    assert p['callbacks'][0]['monitor']=='val_mae' and p['callbacks'][0]['save_top_k']==3
    assert not any('EarlyStopping' in cb['_target_'] for cb in p['callbacks'])
    assert p['declared_training_corpus']=='physically_merged_egoemg_showee_incre'
    assert p['emg_initialization_rule']=='native allowed-prefix exact-name/shape transfer; mismatches recorded and skipped'
    assert p['residual_initialization']=='native last Linear normal(std=1e-5), zero bias; not exact-zero weights'
    assert p['scientific_eligible'] is p['production_ready'] is p['full_paper_complete'] is False


@pytest.mark.parametrize('damage',['window','stride','repeat','channels','interpolate','positions','loss',
    'batch_type','clip_type','norm','vision_initializer','emg_initializer','dropout','precision','binding'])
def test_edited_fusion_config_or_typed_binding_rejects_before_any_model(damage):
    pytest.importorskip('hydra');m=api();c=m.compose_fusion(UPSTREAM)
    if damage=='window':c.datamodule.window_length=12000
    elif damage=='stride':c.datamodule.stride=780
    elif damage=='repeat':c.datamodule.dataset_repeat=1
    elif damage=='channels':c.module.featurizer.conv_blocks[0].in_channels=8
    elif damage=='interpolate':c.egoemg_channel_interpolate=True
    elif damage=='positions':c.egoemg_emg2pose_channel_indices[0]=9
    elif damage=='loss':c.loss_weights.fingertip_distance=.02
    elif damage=='batch_type':c.batch_size=200.
    elif damage=='clip_type':c.trainer.gradient_clip_val=True
    elif damage=='norm':c.datamodule.norm_mode=None
    elif damage=='vision_initializer':c.module.vision_pretrained_checkpoint=None
    elif damage=='emg_initializer':c.pretrained_emg_checkpoint=None
    elif damage=='dropout':c.module.decoder.dropout=.1
    elif damage=='precision':c.trainer.precision='32-true'
    elif damage=='binding':c[m.CONTEXT_KEY].upstream='/unrelated/source'
    with pytest.raises(ValueError):m.fusion_protocol(UPSTREAM,c)


def test_safe_protocol_redacts_initializer_roots_without_modifying_bound_config(monkeypatch):
    pytest.importorskip('hydra');m=api();monkeypatch.setenv('EGOEMG_ROOT','/private/participant-data')
    from omegaconf import OmegaConf
    c=m.compose_fusion(UPSTREAM);before=copy.deepcopy(OmegaConf.to_container(c,resolve=True))
    p=m.fusion_protocol(UPSTREAM,c);text=json.dumps(p)
    assert '/private/' not in text and str(UPSTREAM) not in text
    assert all(not Path(name).is_absolute() for name in p['sources_sha256'])
    assert OmegaConf.to_container(c,resolve=True)==before
    assert '/private/' in c.module.vision_pretrained_checkpoint


def test_native_fusion_datamodule_actually_repeats_and_uses_batch200_for_every_role():
    pytest.importorskip('pytorch_lightning');m=api()
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    from egoemg.datamodule import make_data_module
    c=m.compose_fusion(UPSTREAM);p=m.fusion_protocol(UPSTREAM,c);dm=make_data_module(c)
    dm.dataset_conf={role:[{'_target_':'test_fusion_literal.RepeatWitnessDataset','n':3}] for role in ('train','val','test')}
    dm.norm_mode=None;dm.num_workers=dm.eval_num_workers=0
    for role in ('train','val','test'):
        ds=dm._build_dataset(role,None);assert len(ds)==6 and [ds[i] for i in range(len(ds))]==[0,1,2,0,1,2]
        loader=dm._make_loader(ds,shuffle=False)
        assert loader.batch_size==p['batch_size_by_role'][role]==200 and not loader.drop_last


def test_missing_upstream_does_not_construct_a_model(tmp_path):
    with pytest.raises(ValueError):api().compose_fusion(tmp_path/'absent')
