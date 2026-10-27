"""Native population admission and FD ownership, without model inference."""
from __future__ import annotations

import csv
from dataclasses import fields
import gc
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning');pytest.importorskip('hydra')
from test_emg2pose_memory_runtime import corpus,memory,source,UPSTREAM
from test_emg2pose_memory_data import make_tar,sealed
from test_emg2pose_memory_dataset import equal
from egoemg_repro.emg2pose_memory_index import open_source


def api():return importlib.import_module('egoemg_repro.emg2pose_memory_evaluation_data')


def build(source,memory,**kwargs):
    return api().build_evaluation_data(UPSTREAM,source,memory.corpus.metadata_csv,'small',
                                     scope='cpu_engineering',**kwargs)


def descriptors():
    result={}
    for p in Path('/proc/self/fd').iterdir():
        try:result[int(p.name)]=os.readlink(p)
        except FileNotFoundError:pass
    return result


def activate_native():
    from egoemg_repro import emg2pose_runtime as native
    native._activate(UPSTREAM)


def test_full_policy_is_original_and_engineering_is_explicit():
    p=api().evaluation_data_policy()
    assert p['scope']=='full' and p['selection']=='official_literal'
    assert p['batch_size']==320 and p['workers']==12 and p['precision']=='32-true'
    assert p['window_length']==p['stride']==7790 and p['padding']==[0,0]
    assert p['jitter'] is False and p['skip_ik_failures'] is True
    assert p['factory_pin_memory'] is True and p['multiprocessing_context'] is None
    assert p['scope_description']=='metadata and dataset admission only; no model, inference or scientific qualification'
    assert api().evaluation_data_policy('cpu_engineering')['batch_size']==1
    assert api().evaluation_data_policy('cpu_engineering')['workers']==0


@pytest.mark.parametrize(('scope','selection'),[(None,'official_literal'),(True,'official_literal'),
    ('mini','official_literal'),('full',None),('full',True),('full','auto'),('full','corrected')])
def test_invalid_or_implicit_policy_is_refused(scope,selection):
    with pytest.raises(ValueError):api().evaluation_data_policy(scope,selection)


def test_full_tiny_and_unverified_source_refuse_before_rng_or_fd_changes(source,memory):
    before=torch.get_rng_state().clone();fds=descriptors()
    with pytest.raises(ValueError,match='full|25253|complete'):
        api().build_evaluation_data(UPSTREAM,source,memory.corpus.metadata_csv,'small',role='val')
    with pytest.raises(ValueError,match='VerifiedSource|source'):
        api().build_evaluation_data(UPSTREAM,source.fileno(),memory.corpus.metadata_csv,'small',
                                   role='val',scope='cpu_engineering')
    assert torch.equal(torch.get_rng_state(),before) and descriptors()==fds


@pytest.mark.parametrize('role',[None,True,'train','validation'])
def test_only_val_and_test_metadata_roles_are_admitted(source,memory,role):
    with pytest.raises(ValueError):build(source,memory,role=role)


def test_shared_recording_id_refuses_validation_containing_other_roles_before_construction(source,memory,monkeypatch):
    activate_native()
    from egoemg_repro.emg2pose_memory_dataset import SealedTarDataset
    def forbidden(*_args,**_kwargs):raise AssertionError('storage constructor must not run')
    monkeypatch.setattr(SealedTarDataset,'__post_init__',forbidden)
    before=torch.get_rng_state().clone();fds=descriptors()
    with pytest.raises(ValueError,match='non.val|validation.*role|val.*role'):
        build(source,memory,role='val')
    assert descriptors()==fds and torch.equal(torch.get_rng_state(),before)
    source.assert_current()


def test_literal_test_discloses_native_extra_roles_and_control_selects_exact_file(source,memory):
    m=api();before=torch.get_rng_state().clone()
    with build(source,memory,role='test') as literal:
        r=literal.population['by_condition']['user_stage']
        assert r['requested_rows']==1 and r['selected_rows']==3 and r['extra_rows']==2
        assert r['selected_by_role']==dict(train=1,val=1,test=1)
        assert r['windows']==3 and r['requested_selection_exact'] is False
        assert len(literal.context._emg2pose_datasets[0])==3
        assert literal.identity['inference_performed'] is False
        assert literal.identity['scientific_complete'] is False
        literal.check()
    with build(source,memory,role='test',selection='metadata_indices') as control:
        r=control.population['by_condition']['user_stage']
        assert r['selected_by_role']=={'test':1} and r['windows']==1
        assert r['requested_selection_exact'] is True and r['extra_rows']==r['omitted_rows']==0
        ds=control.context._emg2pose_datasets[0]
        assert list(ds.allowed_splits)==['test'] and list(ds.allowed_session_indices)==r['requested_row_indices']
    assert torch.equal(torch.get_rng_state(),before)


def test_metadata_admission_does_not_construct_models_or_read_signal_windows(source,memory,monkeypatch):
    activate_native()
    from egoemg_repro import emg2pose_memory_runtime as runtime
    from egoemg_repro.emg2pose_memory_dataset import SealedTarDataset
    def forbidden(*_a,**_k):raise AssertionError('no model or HDF5 signal window in metadata admission')
    monkeypatch.setattr(runtime,'build_runtime',forbidden)
    monkeypatch.setattr(SealedTarDataset,'_get_window',forbidden)
    with build(source,memory,role='val',selection='metadata_indices') as data:
        assert data.population['by_condition']['user']['selected_by_role']=={'val':1}
        assert data.context.batch_size==1 and data.context.num_workers==0
        assert all(loader._iterator is None for loader in data.context.dataloaders)


@pytest.mark.parametrize('selection',['official_literal','metadata_indices'])
def test_all_actual_windows_equal_original_reader_and_original_globals_unchanged(source,memory,selection):
    activate_native()
    from egoemg.test_analysis import EMG2PoseEvaluation
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    original=EMG2PoseEvaluation.get_dataloaders.__globals__['Emg2PoseDataset']
    with build(source,memory,role='test',selection=selection) as data:
        for ds in data.context._emg2pose_datasets:
            kwargs={f.name:getattr(ds,f.name) for f in fields(Emg2PoseDataset)}
            kwargs.update(memmap_dir=memory.corpus.data,root_dir=None)
            gold=Emg2PoseDataset(**kwargs)
            try:
                assert len(gold)==len(ds)>0
                for i in range(len(ds)):equal(ds[i],gold[i])
                data.check()  # normal lazy field caches are not semantic changes
            finally:
                for value in gold._memmaps.values():value._mmap.close()
    assert EMG2PoseEvaluation.get_dataloaders.__globals__['Emg2PoseDataset'] is original is Emg2PoseDataset


@pytest.mark.parametrize('damage',['profile','config','filter','blocks','transform','role','dataset_type','per_user'])
def test_semantic_changes_after_admission_are_refused(source,memory,damage):
    activate_native()
    from egoemg.transforms import ExtractStructuredFields
    with build(source,memory,role='test',selection='metadata_indices') as data:
        ds=data.context._emg2pose_datasets[0]
        if damage=='profile':data.context.batch_size=2
        elif damage=='config':data.config.seed+=1
        elif damage=='filter':ds.allowed_session_indices=None
        elif damage=='blocks':ds._block_start[0]+=1
        elif damage=='transform':ds.transform=ExtractStructuredFields(emg_field='changed')
        elif damage=='role':data.context.split='val'
        elif damage=='dataset_type':data.context.dataset_type='egoemg'
        else:data.context.per_user=False
        with pytest.raises(ValueError):data.check()


def test_loader_construction_failure_releases_new_descriptors_and_preserves_source(source,memory,monkeypatch):
    fds=descriptors()
    def fail(*_a,**_k):raise RuntimeError('loader construction failure')
    monkeypatch.setattr(torch.utils.data.DataLoader,'__init__',fail)
    with pytest.raises(RuntimeError,match='loader construction failure'):
        build(source,memory,role='test',selection='metadata_indices')
    assert descriptors()==fds;source.assert_current()


def test_npz_files_close_without_waiting_for_cycle_collection(source,memory):
    activate_native()
    from egoemg.test_analysis import EMG2PoseEvaluation
    gc.collect()  # settle unrelated fixture/import cycles before the observation
    enabled=gc.isenabled();gc.disable()
    try:
        before=descriptors()
        with build(source,memory,role='test',selection='metadata_indices') as data:
            assert data.population['by_condition']['user_stage']['windows']==1
        assert descriptors()==before
    finally:
        if enabled:gc.enable()


def test_close_owns_only_dataset_fds_is_idempotent_and_closed_source_is_refused(source,memory):
    data=build(source,memory,role='test',selection='metadata_indices')
    descriptor=data.context._emg2pose_datasets[0]._archive_descriptor
    borrowed=source.fileno();data.close();data.close()
    with pytest.raises(OSError):os.fstat(descriptor)
    os.fstat(borrowed);source.assert_current()
    with pytest.raises(ValueError,match='closed'):data.check()
    with pytest.raises(ValueError,match='closed'):
        source.close();build(source,memory,role='test')


def test_public_snapshots_do_not_mutate_context_identity(source,memory):
    with build(source,memory,role='test',selection='metadata_indices') as data:
        identity=data.identity;population=data.population
        identity['role']='val';population['by_condition']['user_stage']['windows']=999
        assert data.identity['role']=='test'
        assert data.population['by_condition']['user_stage']['windows']==1
        data.check()


@pytest.fixture(scope='module')
def val_memory(corpus,tmp_path_factory):
    from egoemg_repro import emg2pose_memory_index as index,emg2pose_memory_data as storage
    root=tmp_path_factory.mktemp('eval-pop-all-val')
    with corpus.metadata_csv.open() as stream:rows=list(csv.DictReader(stream))
    for i,row in enumerate(rows):
        row['split']='val';row['generalization']='user' if i==0 else 'user_stage'
        row['held_out_user']=True;row['held_out_stage']=row['generalization']=='user_stage'
        row['stage']='private-val-stage-'+str(i)
        path=root/(row['filename']+'.hdf5');shutil.copy2(corpus.root/path.name,path)
        with h5py.File(path,'r+') as f:
            for key in ('split','generalization','held_out_user','held_out_stage','stage'):
                f['emg2pose'].attrs[key]=row[key]
    csv_path=root/'metadata.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    raw=make_tar([('source/'+p.name,tarfile.REGTYPE,p.read_bytes()) for p in sorted(root.glob('*.hdf5'))])
    with sealed(raw) as fd:
        inventory=storage.inspect_archive(fd,expected_bytes=len(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),
            upstream=UPSTREAM,metadata_csv=csv_path,cohort='engineering')
        output=root/'index';receipt=index.prepare_index(fd,inventory,output,upstream=UPSTREAM,metadata_csv=csv_path)
        yield SimpleNamespace(fd=fd,index=output,receipt=receipt,corpus=SimpleNamespace(metadata_csv=csv_path))


def test_all_val_condition_overlap_and_explicit_control_are_measured(val_memory):
    memory=val_memory
    with open_source(memory.fd,memory.index,expected_completion_sha256=memory.receipt['completion_sha256'],
                     upstream=UPSTREAM,metadata_csv=memory.corpus.metadata_csv) as source:
        for selection,expected_windows,overlap in [('official_literal',[3,3],3),('metadata_indices',[1,2],0)]:
            with build(source,memory,role='val',selection=selection) as data:
                groups=data.population['by_condition']
                assert [groups[k]['windows'] for k in ('user','user_stage')]==expected_windows
                assert data.population['pairwise_row_overlap']=={'user|user_stage':overlap}
                assert len({groups[k]['window_plan_sha256'] for k in groups})==(1 if overlap else 2)
                assert sum(len(ds) for ds in data.context._emg2pose_datasets)==sum(expected_windows)
                data.check()
