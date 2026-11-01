"""Complete actual RN18 fusion bytes/footprints; engineering grants no full fit."""
import copy
import hashlib
import importlib
import json
import random

import numpy as np
import pytest

from test_fusion_literal_data import joint_store,UPSTREAM
from test_rn18_fusion_data import config,W
from test_vision_literal_data import crop_store,release_fixture_readers


def api():return importlib.import_module('egoemg_repro.rn18_fusion_cohort')


@pytest.fixture
def files(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    data=tmp_path/'data';joint_store(data,n=9*W);crops=tmp_path/'crops';crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*W,W,W,W]).astype(np.int8).tofile(data/'frame_split_id.dat')
    try:yield data,crops,config()
    finally:release_fixture_readers(tmp_path)


def admit(files,**kwargs):
    data,crops,cfg=files
    return api().admit_rn18_fusion_cohort(data,crops,upstream=UPSTREAM,config=cfg,require_full=False,**kwargs)


def test_complete_input_hashes_numeric_footprints_and_rng_without_models(files,monkeypatch):
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    import torch,egoemg.train as native
    def deny(*a,**kw):raise AssertionError('cohort cannot construct learned model/optimizer')
    monkeypatch.setattr(native,'make_lightning_module',deny);monkeypatch.setattr(torch.optim.Adam,'__init__',deny)
    before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    c=admit(files);v=c.identity
    assert v['format']=='original_rn18_fusion_full_content_admission_v1'
    assert v['payload_file_count']==14 and len(v['payload_files_sha256'])==14
    assert v['full_payload_content_hashed']and v['counts']==dict(train=80,val=2,test=6)
    assert v['unique_counts']==v['counts']and v['repeat_by_role']==dict(train=1,val=1,test=1)
    assert v['consumed_emg_nonfinite_entries']==v['possible_center_target_nonfinite_entries']==0
    assert not v['registered_full_legacy']and not v['scientific_eligible']and not v['full_paper_complete']
    assert torch.equal(before[0],torch.get_rng_state())and before[2]==random.getstate()
    after=np.random.get_state();assert np.array_equal(before[1][1],after[1])and before[1][2:]==after[2:]
    assert not torch.cuda.is_initialized();c.check(require_full=False)
    with pytest.raises(ValueError):c.check(require_full=True)


def test_full_scope_rejects_small_engineering_input(files):
    data,crops,cfg=files
    with pytest.raises(ValueError):api().admit_rn18_fusion_cohort(data,crops,upstream=UPSTREAM,config=cfg)


@pytest.mark.parametrize('chunk',[0,True,262145])
def test_invalid_scan_budget_rejects_before_input_io(chunk):
    with pytest.raises(ValueError):api().admit_rn18_fusion_cohort(None,None,upstream=None,config=None,chunk_rows=chunk)


def test_expected_receipt_is_recomputed_and_not_a_certificate(files):
    c=admit(files);again=admit(files,expected=c.identity);assert again.identity==c.identity
    fake=copy.deepcopy(c.identity);fake['counts']['test']=999
    from egoemg_repro.classic_emg import _hash
    fake['sha256']=_hash({k:v for k,v in fake.items()if k!='sha256'})
    with pytest.raises(ValueError):admit(files,expected=fake)


@pytest.mark.parametrize('field',['emg_left_filtered','generated_joint_angles_right','mocap_left_wrist_pitch'])
def test_nonfinite_actual_possible_input_or_center_fails_without_repair(files,field):
    c=admit(files);store=c.datasets['train'].store;w=c.datasets['train'].windows[0]
    point=w['start']if field.startswith('emg_')else w['center']
    a=store.field(field);rw=np.memmap(a.filename,mode='r+',dtype=a.dtype,shape=a.shape)
    rw[point]=np.nan;rw.flush()
    with pytest.raises(ValueError,match='nonfinite'):admit(files)


@pytest.mark.parametrize('damage',['identity','config','source_generation','input_generation','same_array_stride','same_array_offset','role','transform','foreign_dataset'])
def test_cached_cohort_identity_source_and_typed_reader_changes_reject(files,damage,tmp_path):
    c=admit(files);ds=c.datasets['train']
    if damage=='identity':c.identity['counts']['test']=999
    elif damage=='config':c.config.batch_size=100
    elif damage=='source_generation':
        p=tmp_path/'owned-source.py';p.write_text('old\n');c.guard.add(p);p.write_text('changed source generation\n')
    elif damage=='input_generation':
        with(files[0]/'frame_split_id.dat').open('ab')as f:f.write(b'changed')
    elif damage=='same_array_stride':ds.views[0]._frame_memmaps['emg_left_filtered_paper'].strides=(0,4)
    elif damage=='same_array_offset':ds.views[0]._frame_memmaps['emg_left_filtered_paper'].offset+=4
    elif damage=='role':ds.role='test'
    elif damage=='transform':ds.views[0].transform.transforms[1].channel_dim=0
    else:object.__setattr__(c,'datasets',dict(c.datasets,train=c.datasets['val']))
    with pytest.raises(ValueError):c.check(require_full=False)


def test_forged_cohort_object_is_not_process_registered(files):
    from dataclasses import replace
    c=admit(files);fake=replace(c)
    with pytest.raises(ValueError):fake.check(require_full=False)


def test_actual_provider_content_hash_change_cannot_hide_behind_unchanged_stats(files,monkeypatch):
    from pathlib import Path
    m=api();c=admit(files);target=Path(m.__file__).with_name('rn18_fusion_data.py');original=m._file_hash
    def changed_hash(p):return'0'*64 if Path(p)==target else original(p)
    monkeypatch.setattr(m,'_file_hash',changed_hash)
    with pytest.raises(ValueError):c.check(require_full=False)


def test_native_cpu_loader_and_partial_batches_preserve_original_rng_and_collation(files):
    import torch
    c=admit(files);loaders=api().native_rn18_fusion_loaders(c,scope='cpu_engineering')
    from egoemg.datamodule import make_data_module
    from torch.utils.data import Subset
    direct=make_data_module(files[2]);direct.batch_size=2;direct.num_workers=direct.eval_num_workers=0;direct.pin_memory=False
    for role,loader in loaders.items():
        assert loader.batch_size==2 and loader.num_workers==0 and loader.generator is None and not loader.drop_last
        assert loader.collate_fn is direct._collate_fn
        reference=direct._make_loader(Subset(c.datasets[role],range(min(4,len(c.datasets[role])))),shuffle=role=='train',num_workers=0)
        torch.manual_seed(73);np.random.seed(19);want=list(reference);after=torch.get_rng_state().clone()
        torch.manual_seed(73);np.random.seed(19);got=list(loader)
        assert len(want)==len(got)and torch.equal(after,torch.get_rng_state())
        for a,b in zip(got,want):
            for key in('emg','vision_img','joint_angles','label_valid_mask','vision_valid_mask','rn18_fusion_sample_index','jitter_offset'):
                assert torch.equal(a[key],b[key]),key
            assert a['emg'].shape[1:]==(8,12000)and a['joint_angles'].shape[1:]==(22,1)
    assert not torch.cuda.is_initialized()
