"""Actual point files, complete content admission and unchanged native loaders."""
import copy
import importlib
import json
import random

import numpy as np
import pytest

from test_vision_literal_data import sample_store,crop_store,config,UPSTREAM,release_fixture_readers


def api():
    try:return importlib.import_module('egoemg_repro.vision_literal_cohort')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.vision_literal_cohort':pytest.fail('complete vision content/loader admission missing')
        raise


@pytest.fixture
def files(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    data=tmp_path/'data';crops=tmp_path/'crops';n=9*7790;sample_store(data,n=n);crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*7790,7790,7790,7790]).astype('int8').tofile(data/'frame_split_id.dat')
    try:yield data,crops,config()
    finally:release_fixture_readers(tmp_path)


def admitted(files,**kwargs):
    data,crops,cfg=files
    return api().admit_vision_cohort(data,crops,upstream=UPSTREAM,config=cfg,**kwargs)


def test_complete_admission_hashes_actual_point_payloads_without_model_or_rng_draw(files,monkeypatch):
    import torch
    from egoemg_repro import vision_literal_runtime as runtime
    def forbidden(*args,**kwargs):raise AssertionError('content admission must not construct a model')
    monkeypatch.setattr(runtime,'build_vision_runtime',forbidden)
    before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    c=admitted(files);value=c.identity
    assert value['format']=='original_vision_full_content_admission_v1'
    assert value['full_payload_content_hashed'] is True and value['possible_center_target_nonfinite_entries']==0
    assert value['counts']['val']==4 and value['counts']['test']==12
    assert value['unique_counts']['val']==2 and value['unique_counts']['test']==6
    assert value['metadata_matches_registered_legacy_release'] is False
    assert value['scientific_eligible'] is False and value['full_paper_complete'] is False
    assert value['payload_file_count']==12  # 11 point/partition/metadata files + one crop database
    assert len(value['payload_files_sha256'])==12
    assert torch.equal(torch.get_rng_state(),before[0])
    assert np.array_equal(np.random.get_state()[1],before[1][1]) and np.random.get_state()[2:]==before[1][2:]
    assert random.getstate()==before[2] and torch.cuda.is_initialized() is False
    c.guard.check()


def test_unacquired_unused_raw_and_filtered_emg_are_not_vision_inputs(files):
    data,crops,cfg=files;m=json.loads((data/'manifest.json').read_text())
    m['fields']['emg_left_raw']=dict(filename='missing-raw.dat',shape=[9*7790,8],dtype='float32')
    (data/'manifest.json').write_text(json.dumps(m))
    for hand in ('left','right'):(data/f'emg_{hand}_filtered.dat').unlink()
    c=admitted(files)
    assert c.identity['payload_file_count']==12
    assert all('emg_' not in name for name in c.identity['payload_files_sha256'])


@pytest.mark.parametrize('kind',['target','crop','metadata'])
def test_frozen_input_guard_detects_actual_payload_or_metadata_changes(files,kind):
    c=admitted(files);data,crops,cfg=files
    if kind=='target':path=data/'generated_joint_angles_left.dat'
    elif kind=='crop':path=crops/'episode_000000.data.mdb'
    else:path=data/'metadata.npz'
    with path.open('r+b') as f:f.seek(0);f.write(b'changed!')
    with pytest.raises(ValueError,match='changed|stamp'):c.guard.check()


def test_expected_content_receipt_is_recomputed_and_cannot_hide_changed_valid_values(files):
    from egoemg_repro.classic_emg import _hash
    first=admitted(files);expected=copy.deepcopy(first.identity)
    # Re-sealing an invented payload hash still must be compared with real bytes.
    key=next(iter(expected['payload_files_sha256']));expected['payload_files_sha256'][key]='0'*64
    expected['sha256']=_hash({k:v for k,v in expected.items() if k!='sha256'})
    with pytest.raises(ValueError,match='receipt|content|identity'):admitted(files,expected=expected)


def test_nonfinite_at_any_possible_jitter_center_rejects_without_cleanup(files):
    initial=admitted(files);data,_,_=files
    w=next(w for w in initial.partitions['windows'] if w['role']=='train' and w['jitter_upper']>1)
    values=np.memmap(data/'generated_joint_angles_left.dat',dtype='float32',mode='r+',shape=(9*7790,20))
    values[w['center']+w['jitter_upper']-1,3]=np.nan;values.flush();del values
    with pytest.raises(ValueError,match='nonfinite'):admitted(files)


def test_unread_endpoint_nan_does_not_add_an_invented_label_filter(files):
    data,_,_=files
    values=np.memmap(data/'generated_joint_angles_left.dat',dtype='float32',mode='r+',shape=(9*7790,20))
    values[0,3]=np.nan;values.flush();del values
    assert admitted(files).identity['possible_center_target_nonfinite_entries']==0


@pytest.mark.parametrize('chunk',[True,0,-1,262145,1.5])
def test_finite_scan_memory_bound_is_a_strict_positive_integer(files,chunk):
    with pytest.raises(ValueError,match='chunk'):admitted(files,chunk_rows=chunk)


def test_native_factory_loader_collator_global_rng_and_repeat_parity(files):
    import torch
    from egoemg.datamodule import make_data_module
    c=admitted(files);loaders=api().native_vision_loaders(c)
    direct=make_data_module(files[2]);direct.num_workers=direct.eval_num_workers=0
    assert set(loaders)=={'train','val','test'}
    for role,loader in loaders.items():
        assert loader.batch_size==360 and loader.num_workers==0 and loader.generator is None
        assert loader.drop_last is False and len(loader.dataset)==c.identity['counts'][role]
        assert type(loader.sampler) is (torch.utils.data.RandomSampler if role=='train' else torch.utils.data.SequentialSampler)
        assert loader.collate_fn is direct._collate_fn
        expected_loader=direct._make_loader(c.datasets[role],shuffle=role=='train')
        torch.manual_seed(73);np.random.seed(19);expected=next(iter(expected_loader));rng=torch.get_rng_state().clone()
        torch.manual_seed(73);np.random.seed(19);actual=next(iter(loader))
        for key in ('vision_img','joint_angles','label_valid_mask','vision_sample_index','jitter_offset','vision_repeat_index'):
            assert torch.equal(actual[key],expected[key]),(role,key)
        assert actual['dataset_name']==expected['dataset_name'] and actual['keystroke_labels']==[]
        assert torch.equal(torch.get_rng_state(),rng)


def test_loader_creation_rechecks_frozen_input_before_native_factory(files,monkeypatch):
    from egoemg import datamodule
    c=admitted(files);path=files[0]/'frame_split_id.dat'
    with path.open('ab') as f:f.write(b'changed')
    def forbidden(*args,**kwargs):raise AssertionError('changed source must reject before native loader factory')
    monkeypatch.setattr(datamodule,'make_data_module',forbidden)
    with pytest.raises(ValueError,match='changed|stamp'):api().native_vision_loaders(c)
