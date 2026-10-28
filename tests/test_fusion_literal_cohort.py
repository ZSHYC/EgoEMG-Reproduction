"""Complete coupled bytes, all possible consumed rows, and original loaders."""
import copy
import hashlib
import importlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_fusion_literal_data import api as reader_api,config,joint_store,UPSTREAM,W
from test_vision_literal_data import crop_store,release_fixture_readers


def api():
    try:return importlib.import_module('egoemg_repro.fusion_literal_cohort')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.fusion_literal_cohort':pytest.fail('complete fusion input admission missing')
        raise


@pytest.fixture
def files(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    data=tmp_path/'data';joint_store(data,n=9*W);crops=tmp_path/'crops';crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*W,W,W,W]).astype('int8').tofile(data/'frame_split_id.dat')
    try:yield data,crops,config(),reader_api().fusion_normalization(UPSTREAM)
    finally:release_fixture_readers(tmp_path)


def admit(files,**kwargs):
    data,crops,cfg,norm=files
    return api().admit_fusion_cohort(data,crops,upstream=UPSTREAM,config=cfg,normalization=norm,**kwargs)


def test_complete_actual_inputs_finite_footprints_and_no_rng_model_or_data_change(files,monkeypatch):
    import torch
    from egoemg_repro import fusion_literal_runtime
    def forbidden(*args,**kwargs):raise AssertionError('admission must not construct a model')
    monkeypatch.setattr(fusion_literal_runtime,'build_fusion_runtime',forbidden)
    original=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files[0].iterdir() if p.is_file()}
    c=admit(files);value=c.identity
    assert value['format']=='original_fusion_full_content_admission_v1'
    assert value['payload_file_count']==14 and len(value['payload_files_sha256'])==14
    assert {'field:emg_left_filtered','field:emg_right_filtered'}<=value['payload_files_sha256'].keys()
    assert value['full_payload_content_hashed'] is True
    assert value['consumed_emg_nonfinite_entries']==value['possible_center_target_nonfinite_entries']==0
    assert value['metadata_matches_registered_legacy_release'] is False
    assert value['counts']['val']==4 and value['counts']['test']==12
    assert value['unique_counts']['val']==2 and value['unique_counts']['test']==6
    assert all(value[k] is False for k in ('scientific_eligible','production_ready','full_paper_complete'))
    assert value['normalization']==files[3]
    assert torch.equal(torch.get_rng_state(),original[0]) and random.getstate()==original[2]
    after=np.random.get_state();assert np.array_equal(after[1],original[1][1]) and after[2:]==original[1][2:]
    assert not torch.cuda.is_initialized()
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}==before
    assert not any('/home/' in str(v) or '/tmp/' in str(v) for v in value.values())
    c.check()


@pytest.mark.parametrize('kind',['left_emg','right_emg','crop','metadata','source','norm_asset'])
def test_guard_rejects_actual_data_source_and_norm_file_changes(files,kind,tmp_path):
    c=admit(files)
    if kind.endswith('_emg'):path=files[0]/('emg_'+kind.split('_')[0]+'_filtered.dat')
    elif kind=='crop':path=files[1]/'episode_000000.data.mdb'
    elif kind=='metadata':path=files[0]/'metadata.npz'
    else:
        # Do not mutate real shared source/upstream. Its registered private
        # stamp is replaced with an actual task-owned file to test same guard.
        original=next(p for p in c.guard.paths if (p.name=='fusion_literal_data.py' if kind=='source' else p.name=='per_dataset_norm_stats_unified.json'))
        path=tmp_path/(kind+'.copy');path.write_bytes(original.read_bytes())
        c.guard.add(path)
    with path.open('r+b') as f:f.seek(0);f.write(b'changed!')
    with pytest.raises(ValueError,match='stamp|changed'):c.check()


@pytest.mark.parametrize('damage',['resealed_receipt','finite_payload_edit'])
def test_expected_receipt_recomputes_actual_bytes(files,damage):
    from egoemg_repro.classic_emg import _hash
    first=admit(files);expected=copy.deepcopy(first.identity)
    if damage=='resealed_receipt':
        expected['payload_files_sha256']['field:emg_left_filtered']='0'*64
        expected['sha256']=_hash({k:v for k,v in expected.items() if k!='sha256'})
    else:
        values=np.memmap(files[0]/'emg_left_filtered.dat',dtype='float32',mode='r+',shape=(9*W,8))
        values[0,0]+=1.;values.flush();del values
    with pytest.raises(ValueError,match='identity|receipt|content'):admit(files,expected=expected)


@pytest.mark.parametrize('field,where',[('emg_left_filtered','first'),('emg_right_filtered','jitter_edge'),
    ('generated_joint_angles_left','center_edge'),('mocap_right_wrist_pitch','center_edge')])
def test_all_consumed_nan_locations_reject_even_with_invalid_label(files,field,where):
    c=admit(files);w=next(w for w in c.partitions['windows'] if w['role']=='train' and w['jitter_upper']>1)
    row=w['footprint_end']-1 if where=='jitter_edge' else w['center']+w['jitter_upper']-1 if where=='center_edge' else w['start']
    shape=(9*W,8 if field.startswith('emg') else 20) if field.startswith(('emg','generated')) else (9*W,)
    values=np.memmap(files[0]/(field+'.dat'),dtype='float32',mode='r+',shape=shape)
    values[row] = np.nan;values.flush();del values
    labels=np.memmap(files[0]/'generated_label_valid.dat',dtype='bool',mode='r+',shape=(9*W,2))
    labels[row]=False;labels.flush();del labels
    with pytest.raises(ValueError,match='nonfinite'):admit(files)


def test_unconsumed_label_endpoint_is_not_cleaned_or_added_as_filter(files):
    values=np.memmap(files[0]/'generated_joint_angles_left.dat',dtype='float32',mode='r+',shape=(9*W,20))
    values[0,0]=np.nan;values.flush();del values
    assert admit(files).identity['possible_center_target_nonfinite_entries']==0
    assert np.isnan(np.fromfile(files[0]/'generated_joint_angles_left.dat',dtype='float32')[0])


@pytest.mark.parametrize('chunk',[True,0,-1,262145,1.5])
def test_scan_chunk_bound_is_strict(files,chunk):
    with pytest.raises(ValueError,match='chunk'):admit(files,chunk_rows=chunk)


@pytest.mark.parametrize('damage',['role','repeat','normalization','protocol','native','view_norm','view_channels','view_hand','transform','array_mapping','view_crops','shadow_crop'])
def test_reader_bindings_cannot_change_after_full_admission(files,damage):
    c=admit(files);ds=c.datasets['train'];view=ds.views[0]
    if damage=='role':ds.role='val'
    elif damage=='repeat':ds.repeat=1
    elif damage=='normalization':ds.normalization['mean']=0
    elif damage=='protocol':ds.protocol['batch_size_by_role']['train']=1
    elif damage=='native':ds.native=object
    elif damage=='view_norm':view._emg_mean=0
    elif damage=='view_channels':view._emg2pose_channel_indices=list(range(8))
    elif damage=='view_hand':view.target_hand='right'
    elif damage=='transform':view.transform.transforms[1].channel_dim=0
    elif damage=='array_mapping':view._frame_memmaps['emg_left_filtered_paper']=view._frame_memmaps['generated_joint_angles_left']
    elif damage=='view_crops':view.crops=files[0]
    elif damage=='shadow_crop':
        shadow=files[1]/'episode_000000.lmdb';shadow.mkdir();shutil.copyfile(files[1]/'episode_000000.data.mdb',shadow/'data.mdb')
    with pytest.raises(ValueError,match='binding|changed|reader|source|identity'):c.check()


def test_original_native_batch200_all_roles_partial_batches_and_rng(files):
    import torch
    c=admit(files)
    from egoemg.datamodule import make_data_module
    direct=make_data_module(files[2]);direct.num_workers=direct.eval_num_workers=0
    loaders=api().native_fusion_loaders(c)
    for role,loader in loaders.items():
        assert loader.batch_size==200 and loader.num_workers==0 and loader.generator is None and not loader.drop_last
        assert type(loader.sampler) is (torch.utils.data.RandomSampler if role=='train' else torch.utils.data.SequentialSampler)
        assert loader.collate_fn is direct._collate_fn
        expected_loader=direct._make_loader(c.datasets[role],shuffle=role=='train')
        torch.manual_seed(73);np.random.seed(19);expected=list(expected_loader);after=torch.get_rng_state().clone()
        torch.manual_seed(73);np.random.seed(19);actual=list(loader)
        assert len(actual)==len(expected) and torch.equal(torch.get_rng_state(),after)
        assert actual[-1]['emg'].shape[0]==len(c.datasets[role])%200
        for got,want in zip(actual,expected):
            for key in ('emg','vision_img','joint_angles','label_valid_mask','vision_valid_mask','fusion_sample_index','fusion_repeat_index','jitter_offset'):
                assert torch.equal(got[key],want[key]),(role,key)
            assert got['dataset_name']==want['dataset_name'] and got['keystroke_labels']==[]
    assert c.identity['counts']['train']>200  # both full200 and final partial really exercised


def test_changed_inputs_reject_before_native_loader_factory(files,monkeypatch):
    c=admit(files)
    from egoemg import datamodule
    with (files[0]/'frame_split_id.dat').open('ab') as f:f.write(b'x')
    def forbidden(*args,**kwargs):raise AssertionError('input change must reject before factory')
    monkeypatch.setattr(datamodule,'make_data_module',forbidden)
    with pytest.raises(ValueError,match='stamp|changed'):api().native_fusion_loaders(c)
