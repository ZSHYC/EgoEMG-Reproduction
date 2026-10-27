"""Actual official windows, RNG and owned two-object multiprocessing storage."""
from copy import deepcopy
import importlib
import multiprocessing as mp
import os
from pathlib import Path

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('hydra')
pytest.importorskip('h5py')
import numpy as np

from test_emg2pose_memory_dataset import data as raw_data,reference,equal
from test_emg2pose_compact_data import packed,UPSTREAM


def api():return importlib.import_module('egoemg_repro.emg2pose_compact_dataset')


def _fixture_capacity():
    # Spawn starts before pytest's parent monkeypatch can be inherited. Install
    # this synthetic capacity ONLY inside the real unit-test child, before its
    # compact dataset is unpickled. Actual server qualification never uses it.
    from egoemg_repro import emg2pose_compact_archive as archive
    archive.available_memory_bytes=lambda:2*1024**4
    return True


class _FixtureCapacity:
    def __reduce__(self):return (_fixture_capacity,())


def _fixture_dataset(_capacity,dataset):return dataset


class _SpawnFixture:
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):return self.dataset[index]
    def __reduce__(self):return (_fixture_dataset,(_FixtureCapacity(),self.dataset))


@pytest.fixture
def data(raw_data,monkeypatch,tmp_path):
    from egoemg_repro import emg2pose_compact_data as scan,emg2pose_compact_index as index
    fd=raw_data['source'].fileno();raw=os.pread(fd,os.fstat(fd).st_size,0)
    metadata=tmp_path/'metadata.csv';location=tmp_path/'compact-index'
    with packed(monkeypatch,raw) as reader:
        inventory=scan.inspect_compact(reader,expected_receipt_sha256=reader.receipt['sha256'],
            metadata_csv=metadata,upstream=UPSTREAM,chunk_rows=3)
        receipt=index.prepare_index(reader,inventory,location,upstream=UPSTREAM,metadata_csv=metadata)
        with index.open_source(reader,location,expected_completion_sha256=receipt['completion_sha256'],
                upstream=UPSTREAM,metadata_csv=metadata) as source:
            yield dict(source=source,index=location,gold=raw_data['gold'],reader=reader)


def candidate(data,**options):
    source=data['source']
    return api().CompactTarDataset(memmap_dir=data['index'],compact_payload_fd=source.payload_fileno(),
        compact_index_fd=source.block_index_fileno(),compact_handle_identity=source.handle_identity,**options)


@pytest.mark.parametrize('skip',[False,True])
@pytest.mark.parametrize('structured',[False,True])
@pytest.mark.parametrize('norm',[None,'per-dataset'])
def test_all_original_windows_padding_filters_normalization_and_transforms_match(data,skip,structured,norm):
    from egoemg.transforms import ExtractStructuredFields
    options=dict(window_length=2,stride=1,padding=(2,1),skip_ik_failures=skip,
        transform=ExtractStructuredFields(),transform_input='structured' if structured else 'dict',
        norm_mode=norm,norm_stats_path=str(UPSTREAM/'assets/per_dataset_norm_stats.json'),
        dataset_name='emg2pose',channel_indices=[3,1,0],pad_joint_angles_to=22)
    old=reference(data,**options);new=candidate(data,**options)
    try:
        assert len(new)==len(old)>0
        for i in range(len(old)):equal(new[i],old[i])
        assert [x.tolist() for x in new.session_index_groups()]==[x.tolist() for x in old.session_index_groups()]
    finally:new.close()


def test_cross_session_global_field_slices_and_original_all_false_fallback(data):
    from egoemg_repro import emg2pose_memory_dataset as original
    before=(original._WindowStore.read,original.FdSlice,original._fd_guard)
    old=reference(data,window_length=2,skip_ik_failures=False,transform=None)
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None)
    try:
        for start,end in ((0,0),(0,2),(6,11),(0,32),(31,32)):equal(new._get_window(start,end),old._get_window(start,end))
        assert new._get_window(0,2)['valid_mask'].tolist()==[True,True]
        new._ensure_memmaps();old._ensure_memmaps()
        assert new._memmaps['valid_mask'][0:2].tolist()==[False,False]
        for name in ('emg','joint_angles','time','valid_mask'):
            assert new._memmaps[name][-1].tobytes()==old._memmaps[name][-1].tobytes()
            assert new._memmaps[name][6:11].tobytes()==old._memmaps[name][6:11].tobytes()
            assert not new._memmaps[name][6:11].flags.writeable
        assert (original._WindowStore.read,original.FdSlice,original._fd_guard)==before
    finally:new.close()


def test_original_jitter_and_cyclic_sampler_consume_identical_rng_and_windows(data):
    from egoemg.datamodule import SessionShuffleSampler
    options=dict(window_length=2,stride=3,jitter=True,skip_ik_failures=False,transform=None)
    old=reference(data,**options);new=candidate(data,**options)
    try:
        a=SessionShuffleSampler(old.session_index_groups(),batch_size=2,seed=31)
        b=SessionShuffleSampler(new.session_index_groups(),batch_size=2,seed=31)
        for epoch in (0,1,9):
            a.set_epoch(epoch);b.set_epoch(epoch);gold=list(a);actual=list(b)
            assert actual==gold and len(actual)==32 and len(set(actual))<len(actual)
            np.random.seed(149)
            for i in actual:
                before=np.random.get_state();expected=old[i];after=np.random.get_state()
                np.random.set_state(before);equal(new[i],expected);current=np.random.get_state()
                assert current[0]==after[0] and np.array_equal(current[1],after[1]) and current[2:]==after[2:]
    finally:new.close()


@pytest.mark.parametrize('filters',[{'allowed_splits':['val']},{'allowed_sides':['left']},
    {'allowed_sessions':['b-train']},{'allowed_session_indices':[1,3]}])
def test_original_filters_and_skip_fields_match(data,filters):
    options=dict(window_length=2,stride=1,skip_ik_failures=False,transform=None,skip_fields=['time'],**filters)
    old=reference(data,**options);new=candidate(data,**options)
    try:
        assert len(new)==len(old)>0
        for i in range(len(old)):equal(new[i],old[i])
    finally:new.close()


def test_dataset_owns_both_source_objects_and_releases_only_its_rights(data):
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None)
    old=reference(data,window_length=2,skip_ik_failures=False,transform=None)
    new[0];fds=(new._compact_reader.payload_fileno(),new._compact_reader.index_fileno())
    data['source'].close();data['reader'].close()
    try:equal(new[0],old[0])
    finally:new.close()
    new.close()
    for fd in fds:
        with pytest.raises(OSError):os.fstat(fd)
    with pytest.raises(ValueError):new[0]


@pytest.mark.parametrize('which',['payload','block_index'])
def test_wrong_actual_kernel_object_never_passes_parent_proof(data,which):
    from test_emg2pose_memory_data import sealed
    source=data['source'];options=dict(memmap_dir=data['index'],compact_payload_fd=source.payload_fileno(),
        compact_index_fd=source.block_index_fileno(),compact_handle_identity=source.handle_identity,
        window_length=2,transform=None)
    with sealed(b'unrelated') as other:
        options['compact_payload_fd' if which=='payload' else 'compact_index_fd']=other
        with pytest.raises(ValueError):api().CompactTarDataset(**options)
    assert source.pread(6,0)==b'corpus'


@pytest.mark.parametrize('damage',['raw_handle','packet','completion','empty_stamps','same_pair'])
def test_malformed_parent_proof_is_refused_before_owning_rights(data,damage):
    source=data['source'];proof=source.handle_identity
    options=dict(memmap_dir=data['index'],compact_payload_fd=source.payload_fileno(),
        compact_index_fd=source.block_index_fileno(),compact_handle_identity=proof,window_length=2,transform=None)
    if damage=='raw_handle':proof['format']='emg2pose_verified_source_handle_v1'
    elif damage=='packet':proof['compact_receipt']['raw_sha256']='0'*64
    elif damage=='completion':proof['index_completion_sha256']='0'*64
    elif damage=='empty_stamps':proof['binding_stamps']={}
    else:options['compact_index_fd']=source.payload_fileno()
    before=set(os.listdir('/proc/self/fd'))
    with pytest.raises(ValueError):api().CompactTarDataset(**options)
    assert set(os.listdir('/proc/self/fd'))==before
    assert source.pread(6,0)==b'corpus'


def test_mutated_metadata_is_rejected_even_after_a_window_is_cached(data):
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None);new[0]
    path=data['index']/'metadata.npz';path.write_bytes(path.read_bytes()+b'x')
    try:
        with pytest.raises(ValueError):new[0]
    finally:new.close()


@pytest.mark.parametrize('which',['payload','block_index'])
def test_external_close_and_numeric_fd_reuse_does_not_close_unrelated_object(data,which):
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None);new[0]
    fd=new._compact_reader.payload_fileno() if which=='payload' else new._compact_reader.index_fileno()
    os.close(fd);sentinel=os.open('/dev/null',os.O_RDONLY)
    assert sentinel==fd
    try:
        with pytest.raises(ValueError):new[0]
        new.close();assert os.fstat(sentinel).st_rdev==os.stat('/dev/null').st_rdev
    finally:os.close(sentinel);new.close()


def _consume_owned(dataset,queue,release):
    try:
        queue.put(('ready',None));assert release.wait(45)
        samples=[dataset[i] for i in range(len(dataset))];queue.put(('ok',samples))
    except BaseException as error:queue.put(('error',repr(error)))
    finally:dataset.close()


@pytest.mark.parametrize('context',['fork','spawn'])
def test_real_child_keeps_both_rights_after_parent_dataset_close(data,context):
    options=dict(window_length=2,stride=2,skip_ik_failures=False,transform=None)
    old=reference(data,**options);new=candidate(data,**options);new[0]
    ctx=mp.get_context(context);queue=ctx.Queue();release=ctx.Event()
    process=ctx.Process(target=_consume_owned,args=(_SpawnFixture(new) if context=='spawn' else new,queue,release))
    try:
        process.start();assert queue.get(timeout=45)[0]=='ready';new.close();release.set()
        status,samples=queue.get(timeout=45);process.join(timeout=45)
        assert process.exitcode==0 and status=='ok',samples
        assert len(samples)==len(old)
        for i,value in enumerate(samples):equal(value,old[i])
    finally:
        release.set()
        if process.is_alive():process.terminate();process.join(timeout=10)
        queue.close();queue.join_thread();new.close()


@pytest.mark.parametrize('context',['fork','spawn'])
def test_actual_two_worker_dataloader_batches_match_original(data,context):
    options=dict(window_length=4,stride=4,skip_ik_failures=False,transform=None)
    old=reference(data,**options);new=candidate(data,**options);new[0]
    try:
        gold=list(torch.utils.data.DataLoader(old,batch_size=2,num_workers=2,multiprocessing_context=context))
        actual=list(torch.utils.data.DataLoader(_SpawnFixture(new) if context=='spawn' else new,
            batch_size=2,num_workers=2,multiprocessing_context=context))
        assert len(actual)==len(gold)>0
        for a,b in zip(actual,gold):equal(a,b)
    finally:new.close()


@pytest.mark.parametrize('name',['normalization.json','emg2pose_memory_dataset.py'])
def test_renamed_changed_normalization_cannot_bypass_original_asset_binding(data,tmp_path,name):
    import json
    path=tmp_path/name;values=json.loads((UPSTREAM/'assets/per_dataset_norm_stats.json').read_text())
    values['emg2pose']['std']=1.;path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        new=candidate(data,window_length=2,transform=None,norm_mode='per-dataset',dataset_name='emg2pose',norm_stats_path=str(path))
        try:new[0]
        finally:new.close()


def test_partial_plain_pickle_failure_releases_already_exported_memory_right(data,monkeypatch):
    import multiprocessing.reduction as reduction
    from multiprocessing import resource_sharer
    import time
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None);new[0]
    real=reduction.DupFd;warm=real(new._compact_reader.payload_fileno());os.close(warm.detach())
    def rights():
        count=0
        for p in Path('/proc/self/fd').iterdir():
            try:count+=os.fstat(int(p.name)).st_ino==os.fstat(new._compact_reader.payload_fileno()).st_ino
            except OSError:pass
        return count
    deadline=time.monotonic()+2
    while resource_sharer._resource_sharer._cache and time.monotonic()<deadline:time.sleep(.01)
    before=rights();transfers=[]
    def fail_second(fd):
        if transfers:raise RuntimeError('actual second exported right failure')
        value=real(fd);transfers.append(value);return value
    monkeypatch.setattr(reduction,'DupFd',fail_second)
    try:
        with pytest.raises(RuntimeError):new.__getstate__()
        deadline=time.monotonic()+2
        while rights()!=before and time.monotonic()<deadline:time.sleep(.01)
        after=rights()
        if after!=before:os.close(transfers[0].detach())
        assert after==before,'failed serialization retained an owned immutable memory descriptor'
        new[0]
    finally:new.close()


@pytest.mark.parametrize('warm',[False,True])
def test_resource_pressure_refuses_even_a_previously_cached_window(data,monkeypatch,warm):
    from egoemg_repro import emg2pose_compact_archive as archive
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None)
    if warm:new[0]
    monkeypatch.setattr(archive,'available_memory_bytes',lambda:0)
    try:
        with pytest.raises(RuntimeError):new[0]
    finally:new.close()


def test_identical_copied_normalization_asset_is_bound_through_cached_reads(data,tmp_path):
    path=tmp_path/'copied-norm.json';path.write_bytes((UPSTREAM/'assets/per_dataset_norm_stats.json').read_bytes())
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None,norm_mode='per-dataset',
        dataset_name='emg2pose',norm_stats_path=str(path));new[0];path.write_bytes(path.read_bytes()+b' ')
    try:
        with pytest.raises(ValueError):new[0]
    finally:new.close()


@pytest.mark.parametrize('skip',[False,True])
def test_explicit_ik_mask_uses_original_window_and_raw_mask_rules(tmp_path,monkeypatch,skip):
    from test_emg2pose_memory_data import corpus_fixture
    from egoemg_repro import emg2pose_compact_data as scan,emg2pose_compact_index as index
    from egoemg_repro.emg2pose_runtime import _activate
    _paths,csv,_arrays,_old,_converted,raw=corpus_fixture(tmp_path,monkeypatch,explicit_ik=True)
    with packed(monkeypatch,raw) as reader:
        inv=scan.inspect_compact(reader,expected_receipt_sha256=reader.receipt['sha256'],metadata_csv=csv,
            upstream=UPSTREAM,chunk_rows=3)
        location=tmp_path/'compact-index';receipt=index.prepare_index(reader,inv,location,upstream=UPSTREAM,metadata_csv=csv)
        with index.open_source(reader,location,expected_completion_sha256=receipt['completion_sha256'],
                upstream=UPSTREAM,metadata_csv=csv) as source:
            _activate(UPSTREAM);case=dict(source=source,index=location,gold=tmp_path/'old-memmap')
            options=dict(window_length=2,stride=1,skip_ik_failures=skip,transform=None)
            old=reference(case,**options);new=candidate(case,**options)
            try:
                assert inv['valid_rows']==16
                new._ensure_memmaps();assert int(new._memmaps['valid_mask'][:].sum())==16
                assert len(new)==len(old)>0
                for i in range(len(old)):equal(new[i],old[i])
            finally:new.close()


def test_second_constructor_duplicate_failure_closes_first_without_touching_parent(data,monkeypatch):
    module=api();real=os.dup;owned=[]
    def fail_second(fd):
        if owned:raise OSError('second compact right refused')
        new_fd=real(fd);owned.append(new_fd);return new_fd
    monkeypatch.setattr(module.os,'dup',fail_second)
    with pytest.raises(ValueError):candidate(data,window_length=2,transform=None)
    assert len(owned)==1
    with pytest.raises(OSError):os.fstat(owned[0])
    assert os.fstat(data['source'].payload_fileno()).st_size>0
    assert os.fstat(data['source'].block_index_fileno()).st_size>0


@pytest.mark.parametrize('failure',['second_transfer','wrong_second_object'])
def test_failed_unpickle_closes_transferred_rights_and_preserves_parent(data,failure):
    from types import SimpleNamespace
    module=api();new=candidate(data,window_length=2,transform=None)
    first=os.dup(new._compact_reader.payload_fileno());owned=[first]
    def refused():raise RuntimeError('second transfer unavailable')
    if failure=='second_transfer':second=SimpleNamespace(detach=refused)
    else:
        wrong=os.open('/dev/null',os.O_RDONLY);owned.append(wrong);second=SimpleNamespace(detach=lambda:wrong)
    state=dict(new.__dict__);state.update(_compact_reader=None,_memmaps={},_window_store=None,
        _compact_transfer=(SimpleNamespace(detach=lambda:first),second))
    restored=module.CompactTarDataset.__new__(module.CompactTarDataset)
    try:
        with pytest.raises((ValueError,RuntimeError)):restored.__setstate__(state)
        for fd in owned:
            with pytest.raises(OSError):os.fstat(fd)
        new[0]
    finally:restored.close();new.close()
