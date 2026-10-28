"""Complete actual numeric bytes and every consumed trajectory row, no images."""
import copy
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.data import LegacyStore
from test_emg_initializer_literal import UPSTREAM,api as literal_api,config
from test_emg_initializer_data import W,source,api as data_api


def api():
    try:return importlib.import_module('egoemg_repro.emg_initializer_cohort')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emg_initializer_cohort':pytest.fail('complete EMG initializer full trajectory admission missing')
        raise


@pytest.fixture
def files(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    source(tmp_path,optional_labels=True)
    return tmp_path,config(),literal_api().emg_initializer_normalization(UPSTREAM)


def admit(files,**kw):
    data,cfg,norm=files
    return api().admit_emg_initializer_cohort(data,upstream=UPSTREAM,config=cfg,normalization=norm,**kw)


def test_all_actual_consumed_inputs_trajectory_rows_and_no_rng_or_file_changes(files):
    import random
    import torch
    original=(torch.get_rng_state().clone(),np.random.get_state(),random.getstate())
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files[0].iterdir() if p.is_file()}
    c=admit(files);v=c.identity
    assert v['format']=='original_emg_initializer_full_content_admission_v1'
    assert v['payload_file_count']==16 and v['counts']==dict(train=328,val=4)
    assert v['counts']==v['unique_counts'] and v['repeat_by_role']==dict(train=1,val=1)
    assert not any('crop' in key or 'image' in key for key in v['payload_files_sha256'])
    assert v['full_payload_content_hashed'] and v['full_trajectory_target_nonfinite_entries']==v['consumed_emg_nonfinite_entries']==0
    for role,ds in c.datasets.items():
        rows=set()
        for w in ds.windows:rows.update(range(w['footprint_start'],w['footprint_end']))
        finite=v['finite_footprints'][role]
        assert finite['unique_possible_rows']==len(rows)
        assert finite['full_trajectory_target_elements']==44*len(rows) and finite['consumed_emg_elements']==16*len(rows)
    assert v['metadata_matches_registered_legacy_release'] is False and v['normalization']==files[2]
    assert all(v[k] is False for k in ('scientific_eligible','production_ready','full_paper_complete'))
    assert torch.equal(original[0],torch.get_rng_state()) and original[2]==random.getstate()
    after=np.random.get_state();assert np.array_equal(original[1][1],after[1]) and original[1][2:]==after[2:]
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}==before
    assert not torch.cuda.is_initialized();c.check()


@pytest.mark.parametrize('field',['emg_left_filtered','emg_right_filtered','generated_joint_angles_right','mocap_left_wrist_yaw'])
def test_last_possible_noncenter_trajectory_nan_rejects_even_with_false_mask(files,field):
    p=data_api().build_initializer_partitions(LegacyStore(files[0]));w=next(w for w in p['windows'] if w['role']=='train')
    row=w['footprint_end']-1;n=24*W;channels=8 if field.startswith('emg') else 20 if field.startswith('generated') else None
    shape=(n,channels) if channels else (n,)
    a=np.memmap(files[0]/(field+'.dat'),dtype='float32',mode='r+',shape=shape);a[row]=np.nan;a.flush();del a
    mask=np.memmap(files[0]/'generated_label_valid.dat',dtype='bool',mode='r+',shape=(n,2));mask[row]=False;mask.flush();del mask
    with pytest.raises(ValueError,match='nonfinite'):admit(files)


def test_unconsumed_future_test_targets_are_not_cleaned_or_added_as_filters(files):
    p=data_api().build_initializer_partitions(LegacyStore(files[0]));covered=np.zeros(24*W,bool)
    for w in p['windows']:covered[w['footprint_start']:w['footprint_end']]=True
    row=int(np.flatnonzero(~covered)[-1]);a=np.memmap(files[0]/'generated_joint_angles_left.dat',dtype='float32',mode='r+',shape=(24*W,20));a[row,0]=np.nan;a.flush();del a
    assert admit(files).identity['full_trajectory_target_nonfinite_entries']==0
    assert np.isnan(np.fromfile(files[0]/'generated_joint_angles_left.dat',dtype='float32').reshape(24*W,20)[row,0])


@pytest.mark.parametrize('damage',['forged_receipt','finite_payload_change'])
def test_strict_reopen_rehashes_actual_bytes(files,damage):
    from egoemg_repro.classic_emg import _hash
    first=admit(files);expected=copy.deepcopy(first.identity)
    if damage=='forged_receipt':
        expected['payload_files_sha256']['field:emg_left_filtered']='0'*64
        expected['sha256']=_hash({k:v for k,v in expected.items() if k!='sha256'})
    else:
        with (files[0]/'emg_left_filtered.dat').open('r+b') as f:f.write(np.array([1.234],np.float32).tobytes())
    with pytest.raises(ValueError,match='receipt|identity|content'):admit(files,expected=expected)


@pytest.mark.parametrize('damage',['role','repeat','block_start','block_cumsum','episode_end','norm','transform','strides','offset','target_hand','config'])
def test_all_reader_indexes_layouts_transforms_and_source_configuration_are_bound(files,damage):
    c=admit(files);ds=c.datasets['train'];view=ds.views[0]
    if damage=='role':ds.role='val'
    elif damage=='repeat':ds.repeat=2
    elif damage=='block_start':view._block_start[0]+=1
    elif damage=='block_cumsum':view._block_cumsum[-1]-=1
    elif damage=='episode_end':view._episode_end_idx[0]-=1
    elif damage=='norm':view._emg_std[0]*=2
    elif damage=='transform':view.transform.transforms=[]
    elif damage=='strides':view._frame_memmaps['emg_left_filtered_paper'].strides=(0,4)
    elif damage=='offset':view._frame_memmaps['emg_left_filtered_paper'].offset+=4
    elif damage=='target_hand':view.target_hand='right'
    elif damage=='config':c.config.optimizer.lr=.1
    with pytest.raises(ValueError,match='binding|mapping|changed|config|recipe'):c.check()


@pytest.mark.parametrize('when',['before_add','after_add'])
def test_generations_registered_before_numeric_open(files,monkeypatch,when):
    m=api();target=files[0]/'emg_right_filtered.dat';replacement=files[0]/'replacement.dat'
    np.full((24*W,8),np.nan,np.float32).tofile(replacement);original=m._StatGuard.add;events=[]
    def add(guard,path):
        if Path(path)==target and not events:
            events.append(when)
            if when=='before_add':replacement.replace(target)
            result=original(guard,path)
            if when=='after_add':replacement.replace(target)
            return result
        return original(guard,path)
    monkeypatch.setattr(m._StatGuard,'add',add)
    with pytest.raises(ValueError,match='nonfinite|changed|stamp|binding'):admit(files)
    assert events==[when]


def test_payload_change_fails_before_native_loader_factory(files,monkeypatch):
    c=admit(files)
    from egoemg import datamodule
    with (files[0]/'label_gesture_active.dat').open('r+b') as f:f.write(b'\x01')
    def forbidden(*args,**kw):raise AssertionError('must reject before loader factory')
    monkeypatch.setattr(datamodule,'make_data_module',forbidden)
    with pytest.raises(ValueError,match='changed|stamp'):api().native_emg_initializer_loaders(c)


def test_native_260_loaders_use_original_collation_global_rng_and_repeat1(files):
    import torch
    c=admit(files);loaders=api().native_emg_initializer_loaders(c)
    from egoemg.datamodule import make_data_module
    original=make_data_module(files[1]);original.num_workers=original.eval_num_workers=0
    for role,loader in loaders.items():
        assert loader.batch_size==260 and loader.num_workers==0 and loader.generator is None and not loader.drop_last
        assert loader.collate_fn is original._collate_fn
        assert type(loader.sampler) is (torch.utils.data.RandomSampler if role=='train' else torch.utils.data.SequentialSampler)
    torch.manual_seed(73);want=next(iter(original._make_loader(c.datasets['val'],shuffle=False)));after=torch.get_rng_state()
    torch.manual_seed(73);got=next(iter(loaders['val']))
    for key in ('emg','joint_angles','label_valid_mask','initializer_sample_index'):assert torch.equal(got[key],want[key])
    assert got['emg'].shape==(4,8,W) and got['joint_angles'].shape==(4,22,W) and torch.equal(after,torch.get_rng_state())


@pytest.mark.parametrize('chunk',[True,0,-1,262145,1.5])
def test_strict_scan_chunk_bounds(files,chunk):
    with pytest.raises(ValueError,match='chunk'):admit(files,chunk_rows=chunk)
