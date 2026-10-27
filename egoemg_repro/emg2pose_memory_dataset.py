"""Storage-only subclass of the pinned official EMG2Pose windowed reader.

Activate the exact official source path before importing this module. The
top-level class is importable by real spawned workers; FD rights are transferred
explicitly. Windowing, transforms, masks, normalization and session groups are
the unchanged inherited methods.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
from egoemg.utils import get_ik_failures_mask

from . import emg2pose_data as storage
from .emg2pose_memory_archive import _GET_SEALS, _SEALS
from .emg2pose_memory_data import FdSlice


def _fd_guard(fd, proof):
    try:
        info = os.fstat(fd)
        if ((info.st_dev,info.st_ino,info.st_size) != (proof['device'],proof['inode'],proof['size']) or
                fcntl.fcntl(fd,_GET_SEALS) & _SEALS != _SEALS or proof['seals'] & _SEALS != _SEALS):
            raise ValueError('descriptor differs from verified sealed source')
    except (OSError,KeyError,TypeError) as error:
        raise ValueError('invalid verified source handle') from error


def _binding_guard(proof):
    for name, expected in proof['binding_stamps'].items():
        info = storage._safe_path(name,file=True).stat()
        current = (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        if current != tuple(expected):
            raise ValueError('verified dataset binding changed')


class _FieldView:
    def __init__(self, store, name, spec):
        self.store, self.name = store,name
        self.shape, self.dtype = tuple(spec['shape']),np.dtype(spec['dtype'])

    def __getitem__(self, key):
        if type(key) is int:
            index = key+self.shape[0] if key < 0 else key
            if index < 0 or index >= self.shape[0]:
                raise IndexError(key)
            return self.store.read(index,index+1)[self.name][0]
        if not isinstance(key,slice) or key.step not in (None,1):
            raise TypeError('source views support contiguous global row slices')
        start,end,_ = key.indices(self.shape[0])
        return self.store.read(start,max(start,end))[self.name]


class _WindowStore:
    def __init__(self, fd, manifest, proof):
        self.fd,self.manifest,self.proof = fd,manifest,proof
        self.sessions = manifest['sessions']
        self.ends = [s['end_idx'] for s in self.sessions]
        self.last = None

    def read(self, start, end):
        _fd_guard(self.fd,self.proof)
        if self.last is not None and self.last[:2] == (start,end):
            return self.last[2]
        pieces = {name:[] for name in ('emg','joint_angles','time','valid_mask')}
        position = start
        si = bisect_right(self.ends,position)
        while position < end:
            session = self.sessions[si]
            stop = min(end,session['end_idx'])
            a,b = position-session['start_idx'],stop-session['start_idx']
            with FdSlice(self.fd,session['member_offset'],session['bytes']) as raw:
                with h5py.File(raw,'r') as file:
                    group = file['emg2pose']
                    chunk = group['timeseries'][a:b]
                    no_fail = ~group['ik_failure_mask'][a:b] if session['explicit_ik_mask'] else get_ik_failures_mask(chunk['joint_angles'])
                    values = {name:np.ascontiguousarray(chunk[name]) for name in ('emg','joint_angles','time')}
                    values['valid_mask'] = np.asarray(no_fail & np.isfinite(chunk['joint_angles']).all(axis=1),dtype=bool)
                    for name,value in values.items():pieces[name].append(value)
            position,si = stop,si+1
        values = {}
        for name,parts in pieces.items():
            spec = self.manifest['fields'][name]
            if not parts:
                value = np.empty((0,*spec['shape'][1:]),dtype=np.dtype(spec['dtype']))
            else:
                value = parts[0] if len(parts)==1 else np.concatenate(parts,axis=0)
            value.setflags(write=False)
            values[name] = value
        self.last = (start,end,values)
        return values


@dataclass
class SealedTarDataset(Emg2PoseDataset):
    """Keep the original reader; provide verified sealed-TAR field views."""

    archive_fd: int | None = None
    archive_handle_identity: dict | None = None

    def _load_memmaps(self):
        self._archive_descriptor = None
        proof = self.archive_handle_identity
        proof = OmegaConf.to_container(proof,resolve=True) if OmegaConf.is_config(proof) else deepcopy(proof)
        if not isinstance(proof,dict) or not isinstance(proof.get('binding_stamps'),dict) or type(self.archive_fd) is not int:
            raise ValueError('VerifiedSource-derived descriptor and binding are required')
        _fd_guard(self.archive_fd,proof)
        _binding_guard(proof)
        root = storage._safe_path(self.memmap_dir)
        for name in ('source-inventory.json','metadata.npz','manifest.json','completion.json'):
            if str(root/name) not in proof['binding_stamps']:
                raise ValueError('index directory differs from verified source binding')
        completion = storage._json(root/'completion.json')
        manifest = storage._json(root/'manifest.json')
        if (storage._sha256(root/'completion.json') != proof['index_completion_sha256'] or
                completion.get('format') != 'emg2pose_memory_index_v1' or completion.get('complete') is not True or
                manifest.get('format') != 'emg2pose_sealed_tar_dataset_v1' or
                storage._sha256(root/'manifest.json') != completion['manifest_sha256'] or
                storage._sha256(root/'metadata.npz') != completion['metadata_sha256'] or
                manifest['archive_sha256'] != proof['archive_sha256'] or manifest['archive_bytes'] != proof['size']):
            raise ValueError('completed memory index differs from verified binding')
        _binding_guard(proof)
        self.archive_handle_identity = proof
        self._manifest = manifest
        self._memmap_specs = {}
        self._archive_pid = os.getpid()
        self._window_store = None
        self._archive_descriptor = os.dup(self.archive_fd)

    def _ensure_memmaps(self):
        if getattr(self,'_archive_descriptor',None) is None:
            raise ValueError('sealed dataset is closed')
        _fd_guard(self._archive_descriptor,self.archive_handle_identity)
        _binding_guard(self.archive_handle_identity)
        if self._archive_pid != os.getpid():
            # A fork inherits only immutable FD rights and cached numpy data;
            # discard the process-local views. No h5py handles stay open.
            self._memmaps = {}
            self._window_store = None
            self._archive_pid = os.getpid()
        if not self._memmaps:
            self._window_store = _WindowStore(self._archive_descriptor,self._manifest,self.archive_handle_identity)
            self._memmaps = {name:_FieldView(self._window_store,name,spec) for name,spec in self._manifest['fields'].items()}

    def close(self):
        if getattr(self,'_archive_descriptor',None) is not None:
            descriptor,self._archive_descriptor = self._archive_descriptor,None
            self._memmaps = {}
            self._window_store = None
            os.close(descriptor)

    def __del__(self):
        self.close()

    def __getstate__(self):
        self._ensure_memmaps()
        from multiprocessing.reduction import DupFd
        state = dict(self.__dict__)
        state['_fd_transfer'] = DupFd(self._archive_descriptor)
        state['_archive_descriptor'] = None
        state['_memmaps'] = {}
        state['_window_store'] = None
        state['archive_fd'] = None
        return state

    def __setstate__(self, state):
        state = dict(state)
        transfer = state.pop('_fd_transfer')
        self.__dict__.update(state)
        descriptor = transfer.detach()
        try:
            _fd_guard(descriptor,self.archive_handle_identity)
            _binding_guard(self.archive_handle_identity)
        except BaseException:
            os.close(descriptor)
            raise
        self._archive_descriptor = self.archive_fd = descriptor
        self._archive_pid = os.getpid()
