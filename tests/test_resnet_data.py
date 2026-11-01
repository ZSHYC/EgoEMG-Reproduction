"""ResNet uses the admitted original point reader and its own native loaders."""
import copy
import dataclasses
import importlib
import random

import numpy as np
import pytest

from test_vision_literal_cohort import files
from test_vision_literal_data import UPSTREAM


def api():
    return importlib.import_module('egoemg_repro.resnet_data')


def admitted(files, variant='resnet18', **kwargs):
    from egoemg_repro.resnet_literal import compose_resnet
    return api().admit_resnet_cohort(*files[:2], upstream=UPSTREAM,
        config=compose_resnet(UPSTREAM, variant), require_full=False, **kwargs)


@pytest.mark.parametrize('variant,batch', [('resnet18',900),('resnet50',400),('resnet152',180)])
def test_reader_projection_preserves_samples_but_identifies_native_loader_differences(variant,batch):
    from omegaconf import OmegaConf
    from egoemg_repro.resnet_literal import compose_resnet
    companion, projection=api()._data_projection(UPSTREAM,compose_resnet(UPSTREAM,variant))
    assert projection['variant']==variant
    assert projection['companion_role']=='original vision-S data only; no ViT construction or training'
    assert projection['loader_only']['resnet']==dict(batch_size=batch,num_workers=8,eval_num_workers=8)
    assert projection['loader_only']['companion']==dict(batch_size=360,num_workers=10,eval_num_workers=10)
    actual=OmegaConf.to_container(compose_resnet(UPSTREAM,variant),resolve=True)
    expected=OmegaConf.to_container(companion,resolve=True)
    for field in ('dataset','datamodule','transforms','vision_patch_size','vision_frame_selection',
                  'skip_emg_loading','vision_num_frames','center_frame_eval'):
        assert field in projection['compared_fields'] and actual[field]==expected[field]
    assert projection['scientific_eligible'] is False


def test_full_content_admission_never_constructs_models_or_draws_rng(files,monkeypatch):
    import torch
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    import egoemg.train as native
    def forbidden(*a,**k):raise AssertionError('data admission must not construct models')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    c=admitted(files);v=c.identity
    assert v['counts']==dict(train=156,val=4,test=12)
    assert v['unique_counts']==dict(train=78,val=2,test=6)
    assert v['full_payload_content_hashed'] is True and v['registered_full_legacy'] is False
    assert len(v['common_data']['payload_files_sha256'])==12
    assert v['scientific_eligible'] is False and v['production_ready'] is False
    assert torch.equal(torch.get_rng_state(),before[0]) and random.getstate()==before[2]
    assert np.array_equal(np.random.get_state()[1],before[1][1])
    assert np.random.get_state()[2:]==before[1][2:]
    c.check()


def test_cpu_engineering_native_collation_retains_point_values_masks_and_global_rng(files):
    import torch
    from egoemg.datamodule import make_data_module
    c=admitted(files);loaders=api().native_resnet_loaders(c,scope='cpu_engineering')
    dm=make_data_module(c.config);dm.batch_size=2;dm.num_workers=dm.eval_num_workers=0
    for role,loader in loaders.items():
        assert loader.batch_size==2 and loader.num_workers==0 and loader.generator is None
        assert loader.drop_last is False and loader.pin_memory is False
        assert len(loader.dataset)==4 and loader.collate_fn is dm._collate_fn
        direct=dm._make_loader(torch.utils.data.Subset(c.datasets[role],range(4)),shuffle=role=='train')
        torch.manual_seed(71);np.random.seed(32);expected=next(iter(direct));rng=torch.get_rng_state().clone()
        torch.manual_seed(71);np.random.seed(32);actual=next(iter(loader))
        for field in ('vision_img','joint_angles','label_valid_mask','vision_sample_index',
                      'jitter_offset','vision_repeat_index'):
            assert torch.equal(actual[field],expected[field]),(role,field)
        assert torch.equal(torch.get_rng_state(),rng)
    missing=c.datasets['val'][len(c.datasets['val'].windows)]
    assert missing['vision_valid_mask'].tolist()==[False] and missing['label_valid_mask'].tolist()==[True]
    np.testing.assert_allclose(missing['vision_img'][:,0,0],[-.485/.229,-.456/.224,-.406/.225],atol=1e-6)


@pytest.mark.parametrize('kind',['target','crop','metadata','split'])
def test_changed_payload_rejects_before_native_loader_creation(files,monkeypatch,kind):
    from egoemg import datamodule
    c=admitted(files)
    p={'target':files[0]/'generated_joint_angles_left.dat','crop':files[1]/'episode_000000.data.mdb',
       'metadata':files[0]/'metadata.npz','split':files[0]/'frame_split_id.dat'}[kind]
    with p.open('r+b')as f:f.write(b'changed!')
    def forbidden(*a,**k):raise AssertionError('changed inputs must reject before loaders')
    monkeypatch.setattr(datamodule,'make_data_module',forbidden)
    with pytest.raises(ValueError):api().native_resnet_loaders(c,scope='cpu_engineering')


@pytest.mark.parametrize('damage',['repeat','role','mean','std','hand','native','memmap','mapping',
                                  'window','episode','crop_root','identity','config'])
def test_reader_index_mask_and_projection_mutations_reject(files,damage):
    c=admitted(files);ds=c.datasets['val'];view=ds.views[0]
    if damage=='repeat':ds.repeat=1
    elif damage=='role':ds.role='test'
    elif damage=='mean':view.vision_mean[0]=0
    elif damage=='std':view.vision_std[0]=1
    elif damage=='hand':view.target_hand='right'
    elif damage=='native':view.native=object
    elif damage=='memmap':view._frame_memmaps['generated_label_valid']=np.ones(9*7790,dtype=bool)
    elif damage=='mapping':ds.frame_mapping_field='invented'
    elif damage=='window':ds.windows[0]['center']+=1
    elif damage=='episode':view._episode_id[0]='wrong_episode'
    elif damage=='crop_root':view.crops=files[0]
    elif damage=='identity':c.identity['scientific_eligible']=True
    else:c.config.batch_size=256
    with pytest.raises(ValueError):c.check()


def test_forged_or_small_cohort_never_grants_a_full_or_gpu_admission(files):
    c=admitted(files)
    with pytest.raises(ValueError):dataclasses.replace(c).check()
    with pytest.raises(ValueError):c.check(require_full=True)
    for scope in ('full','cuda_engineering'):
        with pytest.raises(ValueError):api().native_resnet_loaders(c,scope=scope)
    expected=copy.deepcopy(c.identity);expected['counts']['test']=8612
    with pytest.raises(ValueError):admitted(files,expected=expected)


@pytest.mark.parametrize('flag',[None,1,'full'])
def test_full_requirement_rejects_nonbool_before_io(flag):
    with pytest.raises(ValueError):api().admit_resnet_cohort(None,None,upstream=None,config=None,require_full=flag)


@pytest.mark.parametrize('scope',[None,True,'bounded',''])
def test_unknown_scope_cannot_relax_cohort_admission(files,scope):
    c=admitted(files)
    with pytest.raises(ValueError):api().native_resnet_loaders(c,scope=scope)
