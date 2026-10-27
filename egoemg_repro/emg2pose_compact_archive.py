"""Separately sealed compact bytes with complete readback and bounded range IO.

Payload/index fds are never raw TAR descriptors. All logical original bytes are
preserved; this storage primitive grants no dataset or scientific qualification.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import struct
import threading

from . import emg2pose_compact_codec as codec
from .emg2pose_memory_archive import available_memory_bytes,RESERVE_BYTES,_GET_SEALS,_ADD_SEALS,_SEALS,_create_memfd


FORMAT='emg2pose_sealed_compact_bytes_v1'
RECORD=struct.Struct('<QII32s32s')
MAX_SOURCE_BYTES=462824048640
MAX_INPUT_BYTES=16*1024**2
MAX_READ_BYTES=8*1024**2
WORKSPACE_BYTES=256*1024**2
_SHA=re.compile('[0-9a-f]{64}')
_TOKEN=object()
_FLAGS=dict(complete_bytes=True,full_corpus_complete=False,source_qualified=False,
            scientific_complete=False,full_paper_complete=False)


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _observe(extra=0):
    if available_memory_bytes()<RESERVE_BYTES+WORKSPACE_BYTES+extra:
        raise RuntimeError('compact storage requires unchanged256GiB reserve and bounded workspace')


def _sha(value):return type(value) is str and _SHA.fullmatch(value) is not None


def _stamp(fd):
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or fcntl.fcntl(fd,_GET_SEALS)&_SEALS!=_SEALS:
            raise ValueError('fully sealed compact payload and index required')
        return info.st_dev,info.st_ino,info.st_size
    except OSError as error:
        raise ValueError('invalid sealed compact descriptor') from error


def _pread(fd,size,offset):
    body=os.pread(fd,size,offset)
    if len(body)!=size:raise ValueError('short sealed compact storage read')
    return body


def _pwrite(fd,body,offset):
    view=memoryview(body)
    while view:
        n=os.pwrite(fd,view,offset)
        if n<=0:raise OSError('short compact storage write')
        offset+=n;view=view[n:]


def _receipt(value,expected):
    fields={'format','codec','chunk_bytes','raw_bytes','raw_sha256','payload_bytes','payload_sha256',
            'index_bytes','index_sha256','block_count','seals','sha256',*_FLAGS}
    if type(value) is not dict or set(value)!=fields or not _sha(expected):
        raise ValueError('strict compact receipt and separately frozen seal required')
    value=deepcopy(value)
    if (value['format']!=FORMAT or value['codec']!=codec.CODEC or
            type(value['chunk_bytes']) is not int or value['chunk_bytes']!=codec.CHUNK_BYTES or
            value['seals']!=_SEALS or type(value['seals']) is not int or
            any(value[k] is not v for k,v in _FLAGS.items()) or
            any(not _sha(value[k]) for k in ('sha256','raw_sha256','payload_sha256','index_sha256'))):
        raise ValueError('compact format, codec, flags or hash identity differs')
    if any(type(value[k]) is not int or value[k]<=0 for k in ('raw_bytes','payload_bytes','index_bytes','block_count')):
        raise ValueError('compact byte lengths must be positive exact integers')
    count=(value['raw_bytes']+codec.CHUNK_BYTES-1)//codec.CHUNK_BYTES
    if (value['raw_bytes']>MAX_SOURCE_BYTES or value['block_count']!=count or
            value['index_bytes']!=count*RECORD.size or value['payload_bytes']>count*codec.MAX_ENCODED_BYTES):
        raise ValueError('compact receipt lengths or canonical block count differ')
    if value['sha256']!=expected or _hash({k:v for k,v in value.items() if k!='sha256'})!=expected:
        raise ValueError('compact receipt differs from separately frozen seal')
    return value


def _entry(index,i,receipt,cursor=None):
    body=_pread(index,RECORD.size,i*RECORD.size)
    offset,encoded_n,raw_n,raw_sha,encoded_sha=RECORD.unpack(body)
    expected_raw=min(codec.CHUNK_BYTES,receipt['raw_bytes']-i*codec.CHUNK_BYTES)
    if (raw_n!=expected_raw or not 0<encoded_n<=codec.MAX_ENCODED_BYTES or
            offset+encoded_n>receipt['payload_bytes'] or (cursor is not None and offset!=cursor)):
        raise ValueError('noncanonical compact block extent or length')
    return body,offset,encoded_n,dict(raw_bytes=raw_n,raw_sha256=raw_sha.hex(),encoded_sha256=encoded_sha.hex())


def _validate(payload,index,receipt):
    raw_sha=hashlib.sha256();payload_sha=hashlib.sha256();index_sha=hashlib.sha256();cursor=0
    for i in range(receipt['block_count']):
        _observe();record,offset,n,identity=_entry(index,i,receipt,cursor)
        encoded=_pread(payload,n,offset);raw=codec.decode_block(encoded,**identity)
        index_sha.update(record);payload_sha.update(encoded);raw_sha.update(raw);cursor+=n;_observe()
    if (cursor!=receipt['payload_bytes'] or raw_sha.hexdigest()!=receipt['raw_sha256'] or
            payload_sha.hexdigest()!=receipt['payload_sha256'] or index_sha.hexdigest()!=receipt['index_sha256']):
        raise ValueError('complete ordered compact raw/payload/index hash differs')


def open_compact(payload_fd,index_fd,receipt,*,expected_receipt_sha256):
    """Borrow and independently validate both complete sealed storage objects."""
    value=_receipt(receipt,expected_receipt_sha256)
    if any(type(fd) is not int or fd<0 for fd in (payload_fd,index_fd)):
        raise ValueError('compact descriptors must be nonnegative exact integers')
    _observe();fds=[]
    try:
        for fd in (payload_fd,index_fd):fds.append(os.dup(fd))
        stamps=tuple(_stamp(fd) for fd in fds)
        if (stamps[0][2],stamps[1][2])!=(value['payload_bytes'],value['index_bytes']):
            raise ValueError('sealed compact descriptor byte size differs')
        _validate(*fds,value)
        if tuple(_stamp(fd) for fd in fds)!=stamps:raise ValueError('compact storage changed during admission')
        _observe();reader=CompactReader(*fds,value,_token=_TOKEN);fds=[];return reader
    except OSError as error:
        raise ValueError('invalid or unreadable borrowed compact descriptors') from error
    finally:
        for fd in fds:os.close(fd)


def pack_stream(chunks,*,expected_bytes,max_encoded_bytes,expected_sha256=None):
    """Pack all declared bytes; close partial fds on every failed transaction."""
    if (type(expected_bytes) is not int or not 0<expected_bytes<=MAX_SOURCE_BYTES or
            type(max_encoded_bytes) is not int or max_encoded_bytes<=0 or
            (expected_sha256 is not None and not _sha(expected_sha256))):
        raise ValueError('explicit bounded compact source lengths/hash required')
    count=(expected_bytes+codec.CHUNK_BYTES-1)//codec.CHUNK_BYTES
    _observe(max_encoded_bytes+count*RECORD.size)
    fds=[]
    try:
        # Reuse the verified Linux operation, including its libc path when the
        # server Python omits os.memfd_create and the MFD constants. Kernel names
        # carry no format authority; the two inodes and compact receipt do.
        for _ in range(2):fds.append(_create_memfd())
        raw_sha=hashlib.sha256();payload_sha=hashlib.sha256();index_sha=hashlib.sha256()
        pending=bytearray();received=cursor=ordinal=0
        def commit(raw):
            nonlocal cursor,ordinal
            _observe();encoded=codec.encode_block(raw)
            if cursor+len(encoded)>max_encoded_bytes:raise RuntimeError('declared compact payload budget exhausted')
            raw_digest=hashlib.sha256(raw).digest();encoded_digest=hashlib.sha256(encoded).digest()
            record=RECORD.pack(cursor,len(encoded),len(raw),raw_digest,encoded_digest)
            _pwrite(fds[0],encoded,cursor);_pwrite(fds[1],record,ordinal*RECORD.size)
            cursor+=len(encoded);ordinal+=1;payload_sha.update(encoded);index_sha.update(record);_observe()
        for chunk in chunks:
            _observe()
            if type(chunk) is not bytes or not 0<len(chunk)<=MAX_INPUT_BYTES:
                raise ValueError('compact input chunks must be1..16MiB immutable bytes')
            if received+len(chunk)>expected_bytes:raise ValueError('compact source exceeds declared byte length')
            received+=len(chunk);raw_sha.update(chunk);view=memoryview(chunk)
            while view:
                n=min(codec.CHUNK_BYTES-len(pending),len(view));pending.extend(view[:n]);view=view[n:]
                if len(pending)==codec.CHUNK_BYTES:commit(bytes(pending));pending.clear()
        _observe()
        if received!=expected_bytes:raise ValueError('incomplete compact source byte length')
        if expected_sha256 is not None and raw_sha.hexdigest()!=expected_sha256:
            raise ValueError('complete original source SHA256 differs')
        if pending:commit(bytes(pending))
        if ordinal!=count:raise ValueError('incomplete canonical compact block coverage')
        value=dict(format=FORMAT,codec=codec.CODEC,chunk_bytes=codec.CHUNK_BYTES,raw_bytes=received,
            raw_sha256=raw_sha.hexdigest(),payload_bytes=cursor,payload_sha256=payload_sha.hexdigest(),
            index_bytes=ordinal*RECORD.size,index_sha256=index_sha.hexdigest(),block_count=ordinal,seals=_SEALS,**_FLAGS)
        value['sha256']=_hash(value)
        for fd in fds:fcntl.fcntl(fd,_ADD_SEALS,_SEALS)
        reader=open_compact(*fds,value,expected_receipt_sha256=value['sha256'])
        return reader
    finally:
        for fd in fds:os.close(fd)


class _Cache:
    def __init__(self):
        self.data=OrderedDict();self.pid=os.getpid();self.lock=threading.RLock();self.users=0
    def local(self):
        if self.pid!=os.getpid():
            self.data.clear();self.pid=os.getpid();self.lock=threading.RLock()
        return self.lock


class CompactReader:
    """Owned compact rights; logical pread and slices never expose a raw fd."""
    def __init__(self,payload,index,receipt,*,_token=None,_cache=None):
        if _token is not _TOKEN:raise ValueError('use verified compact pack/open admission')
        self._fds=(payload,index);self._stamps=tuple(_stamp(fd) for fd in self._fds)
        self._receipt=deepcopy(receipt);self._cache=_Cache() if _cache is None else _cache
        with self._cache.local():self._cache.users+=1
    def __enter__(self):self._current();return self
    def __exit__(self,*_args):self.close()
    @property
    def receipt(self):self._current();return deepcopy(self._receipt)
    @property
    def cache_bytes(self):
        self._current()
        with self._cache.local():return sum(map(len,self._cache.data.values()))
    def _current(self):
        if self._fds is None:raise ValueError('compact reader is closed')
        if tuple(_stamp(fd) for fd in self._fds)!=self._stamps:raise ValueError('compact descriptor identity changed')
    def payload_fileno(self):self._current();return self._fds[0]
    def index_fileno(self):self._current();return self._fds[1]
    def clone(self):
        self._current();fds=[]
        try:
            for fd in self._fds:fds.append(os.dup(fd))
            clone=CompactReader(*fds,self._receipt,_token=_TOKEN,_cache=self._cache);fds=[];return clone
        finally:
            for fd in fds:os.close(fd)
    def pread(self,size,offset):
        if type(size) is not int or not 0<=size<=MAX_READ_BYTES or type(offset) is not int or offset<0:
            raise ValueError('bounded integer compact read size/offset required')
        self._current();_observe();remaining=min(size,max(0,self._receipt['raw_bytes']-offset));parts=[]
        while remaining:
            i=offset//codec.CHUNK_BYTES
            with self._cache.local():
                raw=self._cache.data.pop(i,None)
                if raw is None:
                    _,start,n,identity=_entry(self._fds[1],i,self._receipt)
                    raw=codec.decode_block(_pread(self._fds[0],n,start),**identity)
                self._cache.data[i]=raw
                while len(self._cache.data)>8:self._cache.data.popitem(last=False)
            within=offset%codec.CHUNK_BYTES;n=min(remaining,len(raw)-within)
            parts.append(raw[within:within+n]);remaining-=n;offset+=n;self._current();_observe()
        return b''.join(parts)
    def slice(self,offset,length):return CompactSlice(self,offset,length)
    def close(self):
        if self._fds is None:return
        fds,self._fds=self._fds,None
        for fd,stamp in zip(fds,self._stamps):
            try:
                if _stamp(fd)==stamp:os.close(fd)
            except ValueError:pass
        with self._cache.local():
            self._cache.users-=1
            if not self._cache.users:self._cache.data.clear()
    def __del__(self):
        if getattr(self,'_fds',None) is not None:self.close()


class CompactSlice(io.RawIOBase):
    """Independent seek cursor and duplicate rights for TAR/HDF5 fileobj IO."""
    def __init__(self,reader,offset,length):
        super().__init__();self._reader=None;self._position=0
        if type(reader) is not CompactReader or any(type(x) is not int or x<0 for x in (offset,length)):
            raise ValueError('verified compact reader and integer slice bounds required')
        if offset+length>reader.receipt['raw_bytes']:raise ValueError('compact slice exceeds original byte extent')
        self._offset,self._length=offset,length;self._reader=reader.clone()
    def _open(self):
        if self.closed or self._reader is None:raise ValueError('compact slice is closed')
        self._reader._current()
    def readable(self):return True
    def seekable(self):return True
    def writable(self):return False
    def tell(self):self._open();return self._position
    def fileno(self):raise io.UnsupportedOperation('compact logical bytes have no raw TAR fd')
    def seek(self,offset,whence=os.SEEK_SET):
        self._open()
        if type(offset) is not int or type(whence) is not int or whence not in (os.SEEK_SET,os.SEEK_CUR,os.SEEK_END):
            raise ValueError('invalid compact slice seek')
        position={os.SEEK_SET:0,os.SEEK_CUR:self._position,os.SEEK_END:self._length}[whence]+offset
        if position<0:raise ValueError('negative compact slice seek')
        self._position=position;return position
    def readinto(self,buffer):
        self._open();view=memoryview(buffer).cast('B')
        if view.readonly:raise TypeError('compact readinto requires a writable buffer')
        count=min(len(view),max(0,self._length-self._position));done=0
        while done<count:
            body=self._reader.pread(min(MAX_READ_BYTES,count-done),self._offset+self._position)
            if not body:raise ValueError('short compact logical slice read')
            view[done:done+len(body)]=body;done+=len(body);self._position+=len(body)
        return done
    def read(self,size=-1):
        self._open()
        if size is None:size=-1
        if type(size) is not int:raise ValueError('compact read size must be integer')
        count=max(0,self._length-self._position)
        if size>=0:count=min(count,size)
        _observe(2*count);body=bytearray(count);self.readinto(body);_observe(count);return bytes(body)
    def write(self,_buffer):self._open();raise io.UnsupportedOperation('compact bytes are read-only')
    def close(self):
        if self._reader is not None:self._reader.close();self._reader=None
        super().close()
