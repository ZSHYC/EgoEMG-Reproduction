"""Storage-only subclass agrees with actual original windows and real IPC."""
from copy import deepcopy
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('hydra')
pytest.importorskip('h5py')
import h5py
import numpy as np

UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1]/'upstream'))


def api():
    return importlib.import_module('egoemg_repro.emg2pose_memory_dataset')


@pytest.fixture
def data(tmp_path, monkeypatch):
    from test_emg2pose_data import fixture
    from test_emg2pose_memory_data import sealed, make_tar
    from egoemg_repro import emg2pose_data as storage, emg2pose_memory_data as memory
    from egoemg_repro import emg2pose_memory_index as index
    from egoemg_repro.emg2pose_runtime import _activate
    import tarfile
    paths, metadata, _ = fixture(tmp_path)
    # Different session EMG values catch reading a correct row in a wrong file.
    for i,path in enumerate(paths):
        with h5py.File(path,'r+') as f:
            values=f['emg2pose/timeseries'][:]
            values['emg'] += i*1000
            f['emg2pose/timeseries'][:] = values
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=64*1024**3))
    original=storage.inspect_corpus(paths,metadata,cohort='engineering',upstream=UPSTREAM)
    receipt=tmp_path/'original.json';receipt.write_text(json.dumps(original))
    gold=tmp_path/'original-memmap'
    storage.convert_corpus(receipt,gold,upstream=UPSTREAM,chunk_rows=3)
    raw=make_tar([('corpus/'+p.name,tarfile.REGTYPE,p.read_bytes()) for p in reversed(paths)])
    with sealed(raw) as fd:
        import hashlib
        inv=memory.inspect_archive(fd,expected_bytes=len(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),
                                    metadata_csv=metadata,upstream=UPSTREAM,cohort='engineering',chunk_rows=3)
        location=tmp_path/'memory-index'
        receipt=index.prepare_index(fd,inv,location,upstream=UPSTREAM,metadata_csv=metadata)
        with index.open_source(fd,location,expected_completion_sha256=receipt['completion_sha256'],
                                upstream=UPSTREAM,metadata_csv=metadata) as source:
            _activate(UPSTREAM)
            yield dict(source=source,index=location,gold=gold)


def candidate(data, **options):
    return api().SealedTarDataset(memmap_dir=data['index'],archive_fd=data['source'].fileno(),
                                  archive_handle_identity=data['source'].handle_identity,**options)


def reference(data, **options):
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    return Emg2PoseDataset(memmap_dir=data['gold'],**options)


def equal(actual,expected):
    assert set(actual)==set(expected)
    for key,value in expected.items():
        if isinstance(value,np.ndarray):
            assert actual[key].dtype==value.dtype and actual[key].shape==value.shape,key
            assert actual[key].tobytes()==value.tobytes(),key
        elif torch.is_tensor(value):
            assert torch.equal(actual[key],value),key
        else:
            assert actual[key]==value,key


@pytest.mark.parametrize('skip', [False,True])
@pytest.mark.parametrize('structured', [False,True])
@pytest.mark.parametrize('norm', [None,'per-dataset'])
def test_all_original_windows_padding_filters_norm_and_transforms_match(data,skip,structured,norm):
    from egoemg.transforms import ExtractStructuredFields
    options=dict(window_length=2,stride=1,padding=(2,1),skip_ik_failures=skip,
                 transform=ExtractStructuredFields(),transform_input='structured' if structured else 'dict',
                 norm_mode=norm,norm_stats_path=str(UPSTREAM/'assets/per_dataset_norm_stats.json'),
                 dataset_name='emg2pose',channel_indices=[3,1,0],pad_joint_angles_to=22)
    old=reference(data,**options);new=candidate(data,**options)
    try:
        assert len(new)==len(old)>0
        for i in range(len(old)):
            equal(new[i],old[i])
        assert [a.tolist() for a in new.session_index_groups()]==[a.tolist() for a in old.session_index_groups()]
    finally:
        new.close()


def test_cross_session_and_empty_global_field_views_match_real_memmap(data):
    old=reference(data,window_length=2,skip_ik_failures=False,transform=None)
    new=candidate(data,window_length=2,skip_ik_failures=False,transform=None)
    try:
        for a,b in [(0,0),(6,11),(0,32),(31,32)]:
            equal(new._get_window(a,b),old._get_window(a,b))
        new._ensure_memmaps();old._ensure_memmaps()
        for name in ['emg','joint_angles','time','valid_mask']:
            assert new._memmaps[name][-1].tobytes()==old._memmaps[name][-1].tobytes()
            assert new._memmaps[name][6:11].tobytes()==old._memmaps[name][6:11].tobytes()
    finally:new.close()


def test_original_jitter_consumes_identical_numpy_rng_state(data):
    options=dict(window_length=2,stride=3,jitter=True,skip_ik_failures=False,transform=None)
    old=reference(data,**options);new=candidate(data,**options)
    try:
        np.random.seed(149)
        for i in range(len(old)):
            before=np.random.get_state()
            expected=old[i];after=np.random.get_state()
            np.random.set_state(before)
            equal(new[i],expected)
            current=np.random.get_state()
            assert current[0]==after[0] and np.array_equal(current[1],after[1]) and current[2:]==after[2:]
    finally:new.close()


@pytest.mark.parametrize('filters',[{'allowed_splits':['val']},{'allowed_sides':['left']},
                                    {'allowed_sessions':['b-train']},{'allowed_session_indices':[1,3]}])
def test_original_session_filters_and_skip_fields_match(data,filters):
    options=dict(window_length=2,stride=1,skip_ik_failures=False,transform=None,
                 skip_fields=['time'],**filters)
    old=reference(data,**options);new=candidate(data,**options)
    try:
        assert len(new)==len(old)>0
        for i in range(len(old)):equal(new[i],old[i])
    finally:new.close()


def test_dataset_owns_its_descriptor_and_rejects_close_or_wrong_object(data):
    options=dict(window_length=2,skip_ik_failures=False,transform=None)
    old=reference(data,**options);new=candidate(data,**options)
    data['source'].close()
    try:equal(new[0],old[0])
    finally:new.close()
    new.close()
    with pytest.raises(ValueError):new[0]


def test_sealed_handle_identity_cannot_be_rebound_by_numeric_fd(data):
    from test_emg2pose_memory_data import sealed
    proof=deepcopy(data['source'].handle_identity)
    with sealed(b'unrelated') as other,pytest.raises(ValueError):
        api().SealedTarDataset(memmap_dir=data['index'],archive_fd=other,
                              archive_handle_identity=proof,window_length=2,transform=None)


def _read_in_child(dataset,queue):
    try:queue.put(('ok',[dataset[i] for i in range(len(dataset))]))
    except BaseException as error:queue.put(('error',repr(error)))
    finally:dataset.close()


@pytest.mark.parametrize('context',['fork','spawn'])
def test_real_process_reads_same_source_after_parent_cache_warmup(data,context):
    options=dict(window_length=2,stride=2,skip_ik_failures=False,transform=None)
    old=reference(data,**options);new=candidate(data,**options)
    new[0]
    ctx=mp.get_context(context);queue=ctx.Queue();process=ctx.Process(target=_read_in_child,args=(new,queue))
    try:
        process.start();status,samples=queue.get(timeout=45);process.join(timeout=45)
        assert process.exitcode==0 and status=='ok',samples
        assert len(samples)==len(old)
        for i,value in enumerate(samples):equal(value,old[i])
        equal(new[0],old[0])
    finally:
        if process.is_alive():process.terminate();process.join(timeout=10)
        queue.close();queue.join_thread();new.close()


def test_actual_spawn_dataloader_batches_match_original(data):
    options=dict(window_length=4,stride=4,skip_ik_failures=False,transform=None)
    old=reference(data,**options);new=candidate(data,**options)
    try:
        gold=list(torch.utils.data.DataLoader(old,batch_size=2,num_workers=2,multiprocessing_context='spawn'))
        actual=list(torch.utils.data.DataLoader(new,batch_size=2,num_workers=2,multiprocessing_context='spawn'))
        assert len(actual)==len(gold)>0
        for a,b in zip(actual,gold):equal(a,b)
    finally:new.close()
