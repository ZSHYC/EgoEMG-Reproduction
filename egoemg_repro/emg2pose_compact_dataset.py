"""Storage-only official windows over an admitted immutable compact FD pair.

The root runtime must admit an actual VerifiedCompactSource. This module is
its trusted-parent dataset/IPC boundary: attach the *same sealed kernel
objects* after checking that parent's frozen sidecars and concrete bindings.
It is not a public alternative to full source hash/decode admission.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from types import FunctionType

from omegaconf import OmegaConf
from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset

from . import emg2pose_data as storage,emg2pose_runtime as native
from . import emg2pose_compact_archive as archive,emg2pose_compact_index as index
from . import emg2pose_memory_dataset as original


_CODE=(Path(__file__),Path(original.__file__))
_IMPORTED={p:storage._sha256(p) for p in _CODE}


def _pair_guard(payload,block_index,proof):
    try:
        if proof['format']!='emg2pose_verified_compact_handle_v1':raise ValueError('compact parent binding required')
        if any(type(fd) is not int for fd in (payload,block_index)):raise ValueError('explicit compact descriptor pair required')
        actual=[]
        for fd,name in ((payload,'payload'),(block_index,'block_index')):
            claim=proof[name]
            if (set(claim)!={'device','inode','size','seals'} or any(type(x) is not int for x in claim.values()) or
                    claim['seals']!=archive._SEALS):raise ValueError('invalid immutable compact kernel binding')
            stamp=archive._stamp(fd)
            if stamp!=(claim['device'],claim['inode'],claim['size']):raise ValueError('compact kernel object differs from admitted parent')
            actual.append(stamp)
        if actual[0][:2]==actual[1][:2]:raise ValueError('distinct compact payload and block-index objects required')
        packet=proof['compact_receipt'];archive._receipt(packet,packet['sha256'])
        if (actual[0][2],actual[1][2])!=(packet['payload_bytes'],packet['index_bytes']):
            raise ValueError('compact kernel sizes differ from admitted packet')
    except (KeyError,TypeError,OSError) as error:raise ValueError('invalid verified compact parent binding') from error


def _reader_guard(reader,proof):
    if type(reader) is not archive.CompactReader:raise ValueError('exact inherited compact reader required')
    archive._observe();reader._current();_pair_guard(reader.payload_fileno(),reader.index_fileno(),proof)


def _attach(payload,block_index,proof):
    """Private friend: duplicate only objects already bound by the live parent."""
    _pair_guard(payload,block_index,proof);archive._observe();owned=[]
    try:
        for fd in (payload,block_index):owned.append(os.dup(fd))
        reader=archive.CompactReader(*owned,proof['compact_receipt'],_token=archive._TOKEN)
        owned=[];return reader
    finally:
        for fd in owned:os.close(fd)


def _slice(reader,offset,length):return reader.slice(offset,length)


class _WindowStore(original._WindowStore):
    # Retain the exact existing storage read algorithm and its IK/finite rules.
    # Only physical IO and its guard differ; no original globals are mutated.
    read=FunctionType(original._WindowStore.read.__code__,
        {**original._WindowStore.read.__globals__,'_fd_guard':_reader_guard,'FdSlice':_slice},
        name='read')


def _bound_index(root,proof,norm_stats_path):
    if not isinstance(proof,dict) or not isinstance(proof.get('binding_stamps'),dict):
        raise ValueError('verified compact index bindings required')
    root=storage._safe_path(root)
    for name in ('source-inventory.json','metadata.npz','manifest.json','completion.json'):
        if str(root/name) not in proof['binding_stamps']:raise ValueError('index differs from admitted compact parent')
    original._binding_guard(proof)
    completion=storage._json(root/'completion.json');manifest=storage._json(root/'manifest.json')
    inventory=storage._json(root/'source-inventory.json');packet=proof['compact_receipt']
    if (storage._sha256(root/'completion.json')!=proof['index_completion_sha256'] or
            completion.get('format')!=index.FORMAT or completion.get('complete') is not True or
            manifest.get('format')!=index.DATASET or inventory.get('compact_receipt')!=packet or
            (manifest['archive_bytes'],manifest['archive_sha256'])!=(packet['raw_bytes'],packet['raw_sha256'])):
        raise ValueError('completed compact index differs from admitted parent source')
    for name,key in (('manifest.json','manifest_sha256'),('metadata.npz','metadata_sha256'),
                     ('source-inventory.json','source_inventory_sha256')):
        if storage._sha256(root/name)!=completion[key]:raise ValueError('compact sidecar differs from frozen completion')
    official=Path(inspect.getfile(Emg2PoseDataset)).resolve();upstream=official.parents[2]
    storage._official(upstream)
    if str(official) not in proof['binding_stamps']:raise ValueError('imported official window reader differs from parent binding')
    normalization=[upstream/'assets/per_dataset_norm_stats.json']
    if norm_stats_path is not None:normalization.append(storage._safe_path(norm_stats_path,file=True))
    if any(storage._sha256(path)!=native.NORM_SHA256 for path in normalization):
        raise ValueError('original pinned normalization asset required')
    files=[*_CODE,*normalization]
    if any(storage._sha256(p)!=digest for p,digest in _IMPORTED.items()):raise ValueError('imported compact window producer changed')
    for path in files:
        proof['binding_stamps'][str(path)]=list(native._field_stamp(path))
    original._binding_guard(proof)
    return manifest


@dataclass
class CompactTarDataset(Emg2PoseDataset):
    """Inherit all official data semantics; own only compact physical storage."""
    compact_payload_fd:int|None=None
    compact_index_fd:int|None=None
    compact_handle_identity:dict|None=None

    def _load_memmaps(self):
        self._compact_reader=None
        value=self.compact_handle_identity
        proof=OmegaConf.to_container(value,resolve=True) if OmegaConf.is_config(value) else deepcopy(value)
        try:
            _pair_guard(self.compact_payload_fd,self.compact_index_fd,proof)
            manifest=_bound_index(self.memmap_dir,proof,self.norm_stats_path)
            reader=_attach(self.compact_payload_fd,self.compact_index_fd,proof)
        except (KeyError,TypeError,AttributeError,OSError) as error:raise ValueError('invalid compact dataset parent binding') from error
        self.compact_handle_identity=proof;self._manifest=manifest;self._memmap_specs={}
        self._compact_pid=os.getpid();self._window_store=None;self._compact_reader=reader

    def _ensure_memmaps(self):
        reader=getattr(self,'_compact_reader',None)
        if reader is None:raise ValueError('compact dataset is closed')
        _reader_guard(reader,self.compact_handle_identity);original._binding_guard(self.compact_handle_identity)
        if self._compact_pid!=os.getpid():
            self._memmaps={};self._window_store=None;self._compact_pid=os.getpid()
        if not self._memmaps:
            self._window_store=_WindowStore(reader,self._manifest,self.compact_handle_identity)
            self._memmaps={name:original._FieldView(self._window_store,name,spec) for name,spec in self._manifest['fields'].items()}

    def close(self):
        reader=getattr(self,'_compact_reader',None)
        if reader is not None:
            self._compact_reader=None;self._memmaps={};self._window_store=None;reader.close()

    def __del__(self):self.close()

    def __getstate__(self):
        self._ensure_memmaps()
        from multiprocessing.reduction import DupFd
        state=dict(self.__dict__)
        from multiprocessing.context import get_spawning_popen
        transfers=[]
        try:
            for fd in (self._compact_reader.payload_fileno(),self._compact_reader.index_fileno()):
                transfers.append(DupFd(fd))
        except BaseException:
            # Plain pickle's resource sharer owns a real duplicate immediately.
            # Spawn's Popen wrapper instead borrows our existing fd number.
            if get_spawning_popen() is None:
                for transfer in transfers:os.close(transfer.detach())
            raise
        state['_compact_transfer']=tuple(transfers)
        state['_compact_reader']=None;state['_memmaps']={};state['_window_store']=None
        state['compact_payload_fd']=state['compact_index_fd']=None
        return state

    def __setstate__(self,state):
        state=dict(state);transfers=state.pop('_compact_transfer');self.__dict__.update(state)
        self._compact_reader=None;owned=[]
        try:
            for transfer in transfers:owned.append(transfer.detach())
            if len(owned)!=2:raise ValueError('both compact descriptor transfers required')
            _pair_guard(*owned,self.compact_handle_identity);original._binding_guard(self.compact_handle_identity)
            self._compact_reader=_attach(*owned,self.compact_handle_identity)
            self.compact_payload_fd=self._compact_reader.payload_fileno()
            self.compact_index_fd=self._compact_reader.index_fileno();self._compact_pid=os.getpid()
        finally:
            for fd in owned:os.close(fd)
