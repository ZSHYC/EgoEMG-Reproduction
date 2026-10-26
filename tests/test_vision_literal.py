"""Exact vision execution metadata; these checks do not train or qualify a GPU."""
import importlib
import json
import os
from pathlib import Path

import pytest


UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()


class RepeatWitnessDataset:
    """Controlled values at the actual native datamodule's dataset boundary."""
    def __init__(self,n,transform=None):self.n=n;self.transform=transform
    def __len__(self):return self.n
    def __getitem__(self,index):return index


def api():
    try:return importlib.import_module('egoemg_repro.vision_literal')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.vision_literal':pytest.fail('exact vision execution contract missing')
        raise


@pytest.mark.parametrize('variant',['vit_small','vit_base','vit_large'])
def test_vision_contract_keeps_actual_original_recipe_and_declares_backbone_derivation(variant):
    pytest.importorskip('hydra');m=api();config=m.compose_vision(UPSTREAM,variant)
    value=m.vision_protocol(UPSTREAM,config)
    assert value['variant']==variant and value['module']['backbone_type']==variant
    assert value['module']['pretrained'] is True and value['module']['out_channels']==22
    assert value['module']['head_hidden']==512 and value['module']['head_dropout']==.1
    assert value['raw_loss_weights']=={'mae':1.,'landmark/fingertip':.01,'fingertip_distance':.01}
    assert value['effective_loss_weights']=={'mae':1.,'landmark/fingertip':.02}
    assert value['batch_size_by_role']=={'train':360,'val':360,'test':360}
    assert value['unused_declared_val_batch_size']==256
    assert value['repeat_by_role']=={'train':2,'val':2,'test':2}
    assert value['jitter_by_role']=={'train':True,'val':False,'test':False}
    assert value['window']=={'train_length':7790,'eval_length':7790,'train_stride':780,'eval_stride':7790}
    assert value['trainer']['precision']=='bf16-mixed' and value['trainer']['gradient_clip_val']==1
    assert value['trainer']['max_epochs']==150 and value['trainer']['devices']==[0,1,2,3,4,5]
    assert value['optimizer']['lr']==.0001 and value['optimizer']['weight_decay']==.01
    assert value['lr_scheduler']['scheduler']['T_max']==150
    checkpoint,_,early=value['callbacks']
    assert checkpoint['monitor']=='val_mae' and checkpoint['mode']=='min' and checkpoint['save_top_k']==3
    assert checkpoint['save_last'] is True and early['patience']==30
    assert value['declared_training_corpus']=='physically_merged_egoemg_showee_incre'
    assert value['recipe_origin']==('pinned_author_experiment' if variant=='vit_small' else 'explicit_small_recipe_derivation')
    assert value['scientific_eligible'] is False and value['full_paper_complete'] is False


@pytest.mark.parametrize('damage',['batch','batch_type','repeat','validation_field','jitter','precision','clip_type','loss',
    'patience','pretrained','backbone','dataset_name','binding'])
def test_vision_contract_rejects_edited_config_before_any_factory(damage):
    pytest.importorskip('hydra');m=api();config=m.compose_vision(UPSTREAM)
    if damage=='batch':config.batch_size=1
    elif damage=='batch_type':config.batch_size=360.
    elif damage=='repeat':config.datamodule.dataset_repeat=1
    elif damage=='validation_field':config.val_batch_size=360
    elif damage=='jitter':config.dataset.train[0].jitter=False
    elif damage=='precision':config.trainer.precision='32-true'
    elif damage=='clip_type':config.trainer.gradient_clip_val=True
    elif damage=='loss':config.loss_weights.fingertip_distance=.02
    elif damage=='patience':config.callbacks[2].patience=31
    elif damage=='pretrained':config.module.pretrained=False
    elif damage=='backbone':config.module.backbone_type='resnet18'
    elif damage=='dataset_name':config.dataset.train[0].dataset_name='egoemg'
    elif damage=='binding':config[m.CONTEXT_KEY].upstream='/private/unrelated/source'
    with pytest.raises(ValueError):m.vision_protocol(UPSTREAM,config)


@pytest.mark.parametrize('variant',[None,True,{},'resnet18','vit_huge'])
def test_unsupported_vision_variant_is_rejected_before_source_access(tmp_path,variant):
    with pytest.raises(ValueError):api().compose_vision(tmp_path/'absent',variant)


def test_safe_vision_metadata_excludes_private_environment_paths_without_constructing_models(monkeypatch):
    pytest.importorskip('hydra');m=api();monkeypatch.setenv('EGOEMG_ROOT','/private/participant-data')
    value=m.vision_protocol(UPSTREAM,m.compose_vision(UPSTREAM))
    raw=json.dumps(value)
    assert '/private/' not in raw and str(UPSTREAM) not in raw
    assert all(not Path(name).is_absolute() for name in value['sources_sha256'])


def test_vision_repeat_and_batch_claims_match_actual_original_datamodule_for_every_role():
    """One source characterization: repeat/val_batch_size assumptions surprised us."""
    pytest.importorskip('pytorch_lightning');m=api()
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    from egoemg.datamodule import make_data_module
    config=m.compose_vision(UPSTREAM);protocol=m.vision_protocol(UPSTREAM,config)
    dm=make_data_module(config)
    dm.dataset_conf={role:[{'_target_':'test_vision_literal.RepeatWitnessDataset','n':3}] for role in ('train','val','test')}
    dm.norm_mode=None;dm.num_workers=dm.eval_num_workers=0
    for role in ('train','val','test'):
        ds=dm._build_dataset(role,None)
        assert len(ds)==3*protocol['repeat_by_role'][role]
        assert [ds[i] for i in range(len(ds))]==[0,1,2,0,1,2]
        loader=dm._make_loader(ds,shuffle=False)
        assert loader.batch_size==protocol['batch_size_by_role'][role]==360
        assert loader.drop_last is False
