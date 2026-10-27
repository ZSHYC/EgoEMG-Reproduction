"""Separate sealed compact bytes: ownership, complete hashes and actual file IO."""
import copy
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile

import numpy as np
import pytest


CHUNK=1024**2
RECORD=struct.Struct('<QII32s32s')


def api():return importlib.import_module('egoemg_repro.emg2pose_compact_archive')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def seal(value):
    value={k:v for k,v in value.items() if k!='sha256'}
    return dict(value,sha256=sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()))
def capacity(monkeypatch):monkeypatch.setattr(api(),'available_memory_bytes',lambda:2*1024**4)
def memfd():
    from egoemg_repro.emg2pose_memory_archive import _create_memfd
    return _create_memfd()


def pack(monkeypatch,raw):
    capacity(monkeypatch)
    cuts=(raw[:151],raw[151:CHUNK+29],raw[CHUNK+29:])
    return api().pack_stream((p for p in cuts if p),expected_bytes=len(raw),
                            max_encoded_bytes=2*len(raw)+65536,expected_sha256=sha(raw))


def test_ragged_stream_has_independent_canonical_records_and_full_byte_identity(monkeypatch):
    raw=(bytes(range(256))*((2*CHUNK+17+255)//256))[:2*CHUNK+17]
    with pack(monkeypatch,raw) as reader:
        receipt=reader.receipt
        assert receipt['raw_bytes']==len(raw) and receipt['raw_sha256']==sha(raw)
        assert receipt['block_count']==3 and receipt['index_bytes']==240
        assert receipt['complete_bytes'] is True and receipt['full_corpus_complete'] is False
        assert receipt['source_qualified'] is False and receipt['scientific_complete'] is False
        assert receipt['full_paper_complete'] is False
        cursor=0
        for i,n in enumerate((CHUNK,CHUNK,17)):
            offset,encoded_n,raw_n,raw_sha,encoded_sha=RECORD.unpack(os.pread(reader.index_fileno(),80,i*80))
            assert offset==cursor and raw_n==n
            assert raw_sha.hex()==sha(raw[i*CHUNK:i*CHUNK+n])
            assert encoded_sha.hex()==sha(os.pread(reader.payload_fileno(),encoded_n,offset))
            cursor+=encoded_n
        assert cursor==receipt['payload_bytes']
        assert reader.pread(len(raw),0)==raw
        assert not hasattr(reader,'fileno')
        for fd in (reader.payload_fileno(),reader.index_fileno()):
            with pytest.raises(OSError):os.pwrite(fd,b'x',0)


@pytest.fixture
def stored(monkeypatch):
    raw=(bytes(range(251))*((3*CHUNK+17+250)//251))[:3*CHUNK+17]
    with pack(monkeypatch,raw) as reader:yield reader,raw


def test_clone_and_independent_slices_survive_original_close(stored):
    reader,raw=stored;a=reader.slice(CHUNK-19,80);b=reader.slice(0,91);clone=reader.clone()
    reader.close()
    try:
        assert a.read(27)==raw[CHUNK-19:CHUNK+8] and b.read()==raw[:91]
        assert a.tell()==27 and a.seek(-5,os.SEEK_END)==75
        target=bytearray(13);assert a.readinto(target)==5
        assert target[:5]==raw[CHUNK+56:CHUNK+61] and target[5:]==b'\0'*8
        assert a.read()==b'' and a.seek(100)==100 and a.read()==b''
        assert clone.pread(120,CHUNK-20)==raw[CHUNK-20:CHUNK+100]
        with pytest.raises(io.UnsupportedOperation):a.fileno()
        with pytest.raises(io.UnsupportedOperation):a.write(b'x')
    finally:a.close();b.close();clone.close()
    with pytest.raises(ValueError):a.read(1)


def test_reopen_independently_checks_every_block_and_receipt_and_borrows_fds(stored):
    reader,raw=stored;receipt=reader.receipt
    with api().open_compact(reader.payload_fileno(),reader.index_fileno(),receipt,
                            expected_receipt_sha256=receipt['sha256']) as reopened:
        assert reopened.pread(137,2*CHUNK-40)==raw[2*CHUNK-40:2*CHUNK+97]
    assert os.fstat(reader.payload_fileno()).st_size==receipt['payload_bytes']
    receipt['scientific_complete']=True
    assert reader.receipt['scientific_complete'] is False


def test_lru_is_bounded_and_evicted_ranges_still_recover(monkeypatch):
    raw=(bytes(range(251))*((10*CHUNK+17+250)//251))[:10*CHUNK+17]
    with pack(monkeypatch,raw) as reader:
        for i in range(11):assert reader.pread(1,i*CHUNK)==raw[i*CHUNK:i*CHUNK+1]
        assert reader.cache_bytes<=8*CHUNK
        assert reader.pread(79,CHUNK-20)==raw[CHUNK-20:CHUNK+59]
        assert reader.cache_bytes<=8*CHUNK


@pytest.mark.parametrize('size,offset',[(True,0),(-1,0),(8*CHUNK+1,0),(1,-1),(1,True),(1.0,0),(1,0.0)])
def test_invalid_or_unbounded_positioned_reads_refuse(stored,size,offset):
    with pytest.raises(ValueError):stored[0].pread(size,offset)


@pytest.mark.parametrize('offset,length',[(-1,1),(True,1),(0,False),(0,2**63),(2**63,1)])
def test_invalid_slice_extents_refuse(stored,offset,length):
    with pytest.raises(ValueError):stored[0].slice(offset,length)


@pytest.mark.parametrize('change',[
    {'format':'emg2pose_sealed_memory_archive_v1'}, {'codec':'xz6'}, {'raw_bytes':True},
    {'block_count':0}, {'index_bytes':81}, {'chunk_bytes':16*CHUNK},
    {'full_corpus_complete':True}, {'source_qualified':True}, {'scientific_complete':True},
    {'full_paper_complete':True}, {'complete_bytes':False}, {'extra':'forged'},
    {'raw_sha256':'0'*64}, {'payload_sha256':'0'*64}, {'index_sha256':'0'*64},
])
def test_resealed_receipt_changes_cannot_bypass_complete_format_or_hash_checks(stored,change):
    reader,_=stored;receipt=seal(dict(reader.receipt,**change))
    with pytest.raises(ValueError):
        api().open_compact(reader.payload_fileno(),reader.index_fileno(),receipt,
                           expected_receipt_sha256=receipt['sha256'])


def modified(stored,kind):
    reader,_=stored;receipt=reader.receipt
    payload=os.pread(reader.payload_fileno(),receipt['payload_bytes'],0)
    index=bytearray(os.pread(reader.index_fileno(),receipt['index_bytes'],0))
    if kind=='gap':struct.pack_into('<Q',index,0,1)
    elif kind=='reorder':index[:80],index[80:160]=index[80:160],index[:80]
    elif kind=='raw_length':struct.pack_into('<I',index,12,CHUNK-1)
    elif kind=='encoded_length':struct.pack_into('<I',index,8,CHUNK+65537)
    elif kind=='raw_digest':index[16:48]=b'\0'*32
    elif kind=='encoded_digest':index[48:80]=b'\0'*32
    elif kind=='payload':payload=b'x'+payload[1:]
    elif kind=='index_trailing':index+=b'x'
    elif kind=='payload_trailing':payload+=b'x'
    else:raise AssertionError(kind)
    receipt['payload_bytes']=len(payload);receipt['payload_sha256']=sha(payload)
    receipt['index_bytes']=len(index);receipt['index_sha256']=sha(index)
    fds=[]
    for body in (payload,index):
        fd=memfd();fds.append(fd);os.write(fd,body)
        fcntl.fcntl(fd,1033,15)
    return fds,seal(receipt)


@pytest.mark.parametrize('kind',['gap','reorder','raw_length','encoded_length','raw_digest','encoded_digest',
                                'payload','index_trailing','payload_trailing'])
def test_corrupt_canonical_storage_is_rejected_even_with_resealed_outer_hashes(stored,kind):
    fds,receipt=modified(stored,kind)
    try:
        with pytest.raises(ValueError):api().open_compact(*fds,receipt,expected_receipt_sha256=receipt['sha256'])
        for fd in fds:assert os.fstat(fd).st_size>0
    finally:
        for fd in fds:os.close(fd)


def test_unsealed_sources_and_wrong_frozen_receipt_are_rejected_without_closing_borrowed_fds(stored):
    reader,_=stored;receipt=reader.receipt;fd=memfd()
    try:
        os.write(fd,os.pread(reader.payload_fileno(),receipt['payload_bytes'],0))
        with pytest.raises(ValueError):api().open_compact(fd,reader.index_fileno(),receipt,expected_receipt_sha256=receipt['sha256'])
        assert os.fstat(fd).st_size==receipt['payload_bytes']
        with pytest.raises(ValueError):api().open_compact(reader.payload_fileno(),reader.index_fileno(),receipt,expected_receipt_sha256='0'*64)
    finally:os.close(fd)


@pytest.mark.parametrize('kind',['short','long','wrong_hash','oversized_chunk','wrong_chunk','budget','iterator','pressure'])
def test_failed_pack_releases_partial_storage_and_never_returns_a_ready_object(monkeypatch,kind):
    capacity(monkeypatch);module=api();before=set(Path('/proc/self/fd').iterdir());raw=b'a'*CHUNK
    options=dict(expected_bytes=len(raw),expected_sha256=sha(raw),max_encoded_bytes=2*CHUNK)
    def chunks():
        yield raw
        if kind=='iterator':raise RuntimeError('source iterator interrupted')
        if kind=='pressure':monkeypatch.setattr(module,'available_memory_bytes',lambda:0)
    values=chunks()
    if kind=='short':options['expected_bytes']+=1
    elif kind=='long':options['expected_bytes']-=1
    elif kind=='wrong_hash':options['expected_sha256']='0'*64
    elif kind=='oversized_chunk':values=[b'a'*(16*CHUNK+1)]
    elif kind=='wrong_chunk':values=[bytearray(raw)]
    elif kind=='budget':options['max_encoded_bytes']=1
    with pytest.raises((ValueError,RuntimeError)):module.pack_stream(values,**options)
    assert set(Path('/proc/self/fd').iterdir())==before


def test_initial_capacity_and_read_pressure_are_observed(monkeypatch,stored):
    reader,_=stored;monkeypatch.setattr(api(),'available_memory_bytes',lambda:0)
    with pytest.raises(RuntimeError):api().pack_stream([b'x'],expected_bytes=1,max_encoded_bytes=1000)
    with pytest.raises(RuntimeError):reader.pread(1,0)


def test_invalid_borrowed_fd_refuses_without_leaking_duplicate_rights(stored):
    reader,_=stored;receipt=reader.receipt;before=set(Path('/proc/self/fd').iterdir())
    with pytest.raises(ValueError):
        api().open_compact(reader.payload_fileno(),2**30-1,receipt,expected_receipt_sha256=receipt['sha256'])
    assert set(Path('/proc/self/fd').iterdir())==before


def test_reused_fd_identity_is_refused_and_close_preserves_unrelated_sentinel(stored):
    reader,_=stored;fd=reader.payload_fileno();size=reader.receipt['payload_bytes']
    sentinel=memfd()
    try:
        os.write(sentinel,b'x'*size);fcntl.fcntl(sentinel,1033,15);os.dup2(sentinel,fd)
        with pytest.raises(ValueError):reader.pread(1,0)
        reader.close()
        assert os.fstat(fd).st_ino==os.fstat(sentinel).st_ino
    finally:os.close(fd);os.close(sentinel)


def test_real_short_writes_are_completed_and_zero_write_failure_releases_storage(monkeypatch):
    module=api();capacity(monkeypatch);original=os.pwrite
    monkeypatch.setattr(module.os,'pwrite',lambda fd,data,offset:original(fd,data[:7],offset))
    with pack(monkeypatch,b'original bytes'*2000) as reader:assert reader.pread(28,0)==b'original bytes'*2
    before=set(Path('/proc/self/fd').iterdir())
    monkeypatch.setattr(module.os,'pwrite',lambda *_args:0)
    with pytest.raises(OSError):module.pack_stream([b'x'],expected_bytes=1,max_encoded_bytes=1000)
    assert set(Path('/proc/self/fd').iterdir())==before


def test_iterator_interrupt_releases_all_partial_storage(monkeypatch):
    capacity(monkeypatch)
    def chunks():
        yield b'x'*CHUNK
        raise KeyboardInterrupt('actual iterator interruption')
    before=set(Path('/proc/self/fd').iterdir())
    with pytest.raises(KeyboardInterrupt):api().pack_stream(chunks(),expected_bytes=CHUNK,max_encoded_bytes=2*CHUNK)
    assert set(Path('/proc/self/fd').iterdir())==before


def test_actual_kernel_pack_works_when_python_omits_memfd_wrapper_and_constants(monkeypatch):
    for name in ('memfd_create','MFD_ALLOW_SEALING','MFD_CLOEXEC'):monkeypatch.delattr(os,name,raising=False)
    with pack(monkeypatch,b'unchanged Linux kernel bytes') as reader:
        assert reader.pread(100,0)==b'unchanged Linux kernel bytes'


def test_actual_fresh_process_opens_both_inherited_descriptors(stored):
    reader,raw=stored;receipt=reader.receipt
    code='''import json,sys,os,hashlib
sys.path.insert(0,sys.argv[1])
from egoemg_repro import emg2pose_compact_archive as archive
archive.available_memory_bytes=lambda:2*1024**4
assert os.getpid()!=int(sys.argv[5])
with archive.open_compact(int(sys.argv[2]),int(sys.argv[3]),json.loads(sys.stdin.read()),expected_receipt_sha256=sys.argv[4]) as reader:
 print(hashlib.sha256(reader.pread(8*1024**2,0)).hexdigest())
'''
    fds=(reader.payload_fileno(),reader.index_fileno())
    result=subprocess.run([sys.executable,'-c',code,str(Path(api().__file__).parents[1]),*map(str,fds),receipt['sha256'],str(os.getpid())],
                          input=json.dumps(receipt),text=True,capture_output=True,pass_fds=fds,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()==sha(raw)


def test_actual_tar_and_hdf_fileobj_fields_equal_original_storage(monkeypatch):
    h5py=pytest.importorskip('h5py')
    arrays=np.zeros(20000,dtype=[('time','<f8'),('joint_angles','<f4',(20,)),('emg','<f4',(16,))])
    arrays['time']=np.arange(len(arrays),dtype=np.float64)/2000
    arrays['joint_angles']=np.arange(len(arrays)*20,dtype=np.float32).reshape(-1,20)/333
    arrays['emg']=np.sin(np.arange(len(arrays)*16,dtype=np.float32)).reshape(-1,16)
    body=io.BytesIO()
    with h5py.File(body,'w') as file:
        group=file.create_group('emg2pose');group.attrs['tag']='exact original metadata';group.create_dataset('timeseries',data=arrays)
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w') as tar:
        info=tarfile.TarInfo('source/original.hdf5');info.size=len(body.getvalue());tar.addfile(info,io.BytesIO(body.getvalue()))
    with pack(monkeypatch,archive.getvalue()) as reader:
        with reader.slice(0,len(archive.getvalue())) as view:
            with tarfile.open(fileobj=view,mode='r:') as tar:member=tar.getmember('source/original.hdf5')
        with reader.slice(member.offset_data,member.size) as view:
            with h5py.File(view,'r') as file:
                assert file['emg2pose'].attrs['tag']=='exact original metadata'
                for start,end in [(0,19),(6900,14690),(19000,20000)]:
                    actual=file['emg2pose/timeseries'][start:end]
                    assert actual.dtype==arrays.dtype and actual.tobytes()==arrays[start:end].tobytes()
