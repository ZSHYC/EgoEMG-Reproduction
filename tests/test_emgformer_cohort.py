"""Complete physical bytes and exact native full/point consumption boundaries."""
import copy
import hashlib
import importlib
from pathlib import Path
import random

import numpy as np
import pytest

from egoemg_repro.data import LegacyStore
from test_emgformer_data import UPSTREAM,W,cfg
from test_emg_initializer_data import source


def api():
    try:return importlib.import_module('egoemg_repro.emgformer_cohort')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emgformer_cohort':pytest.fail('complete native EMGFormer admission missing')
        raise


@pytest.fixture
def files(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    from egoemg_repro.emgformer_data import emgformer_data_contract
    source(tmp_path,optional_labels=True);config=cfg()
    return tmp_path,config,emgformer_data_contract(UPSTREAM,config)['normalization']


def admit(files,**kw):
    data,config,norm=files
    return api().admit_emgformer_cohort(data,upstream=UPSTREAM,config=config,normalization=norm,**kw)


def write_value(root,field,row,value):
    channels=8 if field.startswith('emg')else 20 if field.startswith('generated_joint')else None
    shape=(24*W,channels)if channels else(24*W,)
    a=np.memmap(root/(field+'.dat'),dtype='float32',mode='r+',shape=shape);a[row]=value;a.flush();del a


def test_full_bytes_exact_role_consumption_and_rng_files_unchanged(files):
    import torch
    before={p:hashlib.sha256(p.read_bytes()).hexdigest()for p in files[0].iterdir()if p.is_file()}
    rng=(torch.get_rng_state().clone(),np.random.get_state(),random.getstate())
    c=admit(files);v=c.identity
    assert v['format']=='original_egoemg_emgformer_full_content_admission_v1'
    assert v['counts']==dict(train=340,val=4,test=8)and v['unique_counts']==v['counts']
    assert v['payload_file_count']==16 and len(v['payload_files_sha256'])==16 and v['full_payload_content_hashed']
    want={'train':(215999,215999),'val':(24000,2),'test':(48000,4)}
    for role,(emg_rows,target_rows)in want.items():
        f=v['finite_footprints'][role]
        assert f['unique_possible_input_rows']==emg_rows and f['unique_target_rows']==target_rows
        assert f['consumed_emg_elements']==emg_rows*16 and f['consumed_target_elements']==target_rows*44
        assert f['consumed_emg_nonfinite_entries']==f['normalized_emg_nonfinite_entries']==f['consumed_target_nonfinite_entries']==0
    assert v['canonical_unified_namespace_available']is False and v['explicit_normalization_namespace']=='egoemg'
    assert v['metadata_matches_registered_legacy_release']is False
    assert all(v[k]is False for k in ('scientific_eligible','production_ready','full_paper_complete'))
    assert torch.equal(rng[0],torch.get_rng_state())and rng[2]==random.getstate()
    after=np.random.get_state();assert np.array_equal(rng[1][1],after[1])and rng[1][2:]==after[2:]
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest()for p in before}==before
    assert not torch.cuda.is_initialized();c.check()


@pytest.mark.parametrize('role,row',[('train',215998),('val',222000),('test',282000)])
@pytest.mark.parametrize('field',['emg_right_filtered','generated_joint_angles_left','mocap_right_wrist_yaw'])
def test_consumed_last_jitter_or_heldout_center_nan_rejects_even_masked(files,role,row,field):
    write_value(files[0],field,row,np.nan)
    a=np.memmap(files[0]/'generated_label_valid.dat',dtype=bool,mode='r+',shape=(24*W,2));a[row]=False;a.flush();del a
    with pytest.raises(ValueError,match='nonfinite'):admit(files)


@pytest.mark.parametrize('row',[216001,240001,287999])
def test_unconsumed_offcenter_heldout_targets_are_hashed_preserved_but_not_scanned_as_consumed(files,row):
    field='generated_joint_angles_left';write_value(files[0],field,row,np.nan)
    c=admit(files)
    assert c.identity['consumed_target_nonfinite_entries']==0
    assert c.identity['payload_files_sha256']['field:'+field]==hashlib.sha256((files[0]/(field+'.dat')).read_bytes()).hexdigest()
    assert np.isnan(c.datasets['test'].store.field(field)[row,0])


def test_finite_extreme_emg_is_preserved_with_pinned_native_normalization(files):
    write_value(files[0],'emg_left_filtered',0,np.finfo(np.float32).max)
    c=admit(files)
    assert c.identity['finite_footprints']['train']['normalized_emg_nonfinite_entries']==0
    assert c.datasets['train'].store.field('emg_left_filtered')[0,0]==np.finfo(np.float32).max


@pytest.mark.parametrize('damage',['forged_receipt','finite_payload','wrong_format'])
def test_strict_reopen_rehashes_actual_content_and_rejects_forgery(files,damage):
    from egoemg_repro.classic_emg import _hash
    c=admit(files);expected=copy.deepcopy(c.identity)
    if damage=='forged_receipt':expected['payload_files_sha256']['field:emg_right_filtered']='0'*64
    elif damage=='finite_payload':write_value(files[0],'emg_right_filtered',287999,1.2345)
    else:expected['format']='another_model'
    expected['sha256']=_hash({k:v for k,v in expected.items()if k!='sha256'})
    with pytest.raises(ValueError,match='receipt|identity|content'):admit(files,expected=expected)


@pytest.mark.parametrize('role,hand',[('train',0),('val',1),('test',0),('test',1)])
@pytest.mark.parametrize('damage',['center','norm_dtype','manifest','split_index','map_stride','transform'])
def test_every_role_and_hand_reader_computation_is_persistently_bound(files,role,hand,damage):
    c=admit(files);view=c.datasets[role].views[hand]
    if damage=='center':view._center_target_only=not view._center_target_only
    elif damage=='norm_dtype':view._emg_std=view._emg_std.astype(np.float64)
    elif damage=='manifest':view._manifest['fields']['generated_joint_angles_left']['dtype']='float64'
    elif damage=='split_index':view._window_split_id[0]=(int(view._window_split_id[0])+1)%4
    elif damage=='map_stride':next(iter(view._frame_memmaps.values())).strides=(0,)*next(iter(view._frame_memmaps.values())).ndim
    else:view.transform.transforms=[]
    with pytest.raises(ValueError,match='binding|changed|mapping'):c.check()


@pytest.mark.parametrize('damage',['config','identity','partition','role_map','reader_windows','reader_config','reader_store','reader_contract','file_mode','hardlink'])
def test_corpus_identity_files_and_reader_membership_remain_bound(files,damage):
    c=admit(files)
    if damage=='config':c.config.optimizer.lr=.1
    elif damage=='identity':c.identity['counts']['test']+=2
    elif damage=='partition':c.partitions['windows'][0]['center']+=1
    elif damage=='role_map':object.__setattr__(c,'datasets',{'train':c.datasets['train'],'val':c.datasets['val'],'test':c.datasets['val']})
    elif damage=='reader_windows':c.datasets['test'].windows[0]['center']+=1
    elif damage=='reader_config':c.datasets['val'].config.optimizer.lr=.1
    elif damage=='reader_store':c.datasets['test'].store.metadata['episode_id'][0]='changed'
    elif damage=='reader_contract':c.datasets['val'].contract['center_target_only_by_role']['val']=False
    elif damage=='file_mode':(files[0]/'emg_left_filtered.dat').chmod(0o400)
    else:(files[0]/'another-link.dat').hardlink_to(files[0]/'emg_right_filtered.dat')
    with pytest.raises(ValueError,match='identity|binding|changed|stamp|config|recipe|partition'):c.check()


@pytest.mark.parametrize('when',['before_add','after_add'])
def test_file_generations_registered_before_numeric_open(files,monkeypatch,when):
    m=api();target=files[0]/'emg_right_filtered.dat';replacement=files[0]/'replacement.dat'
    np.full((24*W,8),np.nan,np.float32).tofile(replacement);original=m._Guard.add;events=[]
    def add(guard,path):
        if Path(path)==target and not events:
            events.append(when)
            if when=='before_add':replacement.replace(target)
            value=original(guard,path)
            if when=='after_add':replacement.replace(target)
            return value
        return original(guard,path)
    monkeypatch.setattr(m._Guard,'add',add)
    with pytest.raises(ValueError,match='nonfinite|changed|stamp|binding'):admit(files)
    assert events==[when]


@pytest.mark.parametrize('chunk',[True,0,-1,262145,1.5])
def test_invalid_chunk_bound_rejects_before_io(tmp_path,chunk):
    with pytest.raises(ValueError,match='chunk'):api().admit_emgformer_cohort(tmp_path,upstream=UPSTREAM,config=None,normalization=None,chunk_rows=chunk)


def test_loader_factory_requires_intact_admission_and_preserves_native_policy(files,monkeypatch):
    import torch
    c=admit(files);m=api();loaders=m.native_emgformer_loaders(c)
    for role,loader in loaders.items():
        assert loader.batch_size==500 and loader.num_workers==6 and loader.prefetch_factor==1
        assert loader.persistent_workers and not loader.pin_memory and not loader.drop_last and loader.generator is None
        assert type(loader.sampler)is(torch.utils.data.RandomSampler if role=='train'else torch.utils.data.SequentialSampler)
    diagnostic=m.native_emgformer_loaders(c,cpu_engineering=True)
    batch=next(iter(diagnostic['val']))
    assert batch['emg'].shape==(4,8,W)and batch['joint_angles'].shape==(4,22,1)
    with (files[0]/'label_gesture_class.dat').open('r+b')as f:f.write(np.array([99],np.int32).tobytes())
    from egoemg import datamodule
    def forbidden(*a,**kw):raise AssertionError('changed cohort must reject before loader factory')
    monkeypatch.setattr(datamodule,'make_data_module',forbidden)
    with pytest.raises(ValueError,match='stamp|changed'):m.native_emgformer_loaders(c)
