"""Private original-TAR/HDF5 inventory for a complete sealed memory source.

This format is deliberately distinct from completed direct-memmap data. Source
bytes remain in the caller-owned sealed archive; no extraction or source-sized
arrays are written. HDF5/NumPy are needed only for corpus inspection.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from types import SimpleNamespace

from .emg2pose_memory_archive import _GET_SEALS, _SEALS


_HASH = re.compile(r'[0-9a-f]{64}')
_READ_BYTES = 8*1024**2


class FdSlice(io.RawIOBase):
    """Read-only, independently positioned view bounded to one sealed extent.

    This owns a duplicate, never the caller's descriptor. pread avoids changing
    another HDF5 file object's cursor. Close HDF5 before closing this object.
    """

    def __init__(self, fd, offset, length):
        super().__init__()
        self._descriptor = None
        self._position = 0
        if any(type(x) is not int or x < 0 for x in (fd, offset, length)):
            raise ValueError('descriptor and slice extent must be nonnegative integers')
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or fcntl.fcntl(fd, _GET_SEALS) & _SEALS != _SEALS:
                raise ValueError('an immutable fully sealed source is required')
        except OSError as error:
            raise ValueError('an immutable fully sealed source is required') from error
        if offset+length > info.st_size:
            raise ValueError('slice extent exceeds sealed source bounds')
        self._offset, self._length = offset, length
        self._descriptor = os.dup(fd)

    def fileno(self):
        if self.closed or self._descriptor is None:
            raise ValueError('sealed slice is closed')
        return self._descriptor

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        self.fileno()
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        self.fileno()
        if type(offset) is not int or whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            raise ValueError('invalid slice seek')
        origin = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self._length}[whence]
        position = origin+offset
        if position < 0:
            raise ValueError('negative slice seek')
        self._position = position
        return position

    def read(self, size=-1):
        descriptor = self.fileno()
        if size is None:
            size = -1
        if type(size) is not int:
            raise ValueError('read size must be an integer')
        remaining = max(0, self._length-self._position)
        count = remaining if size < 0 else min(size, remaining)
        if not count:
            return b''
        body = os.pread(descriptor, count, self._offset+self._position)
        if len(body) != count:
            raise OSError('short immutable source read')
        self._position += count
        return body

    def readinto(self, buffer):
        view = memoryview(buffer).cast('B')
        if view.readonly:
            raise TypeError('readinto requires a writable buffer')
        body = self.read(len(view))
        view[:len(body)] = body
        return len(body)

    def write(self, buffer):
        self.fileno()
        raise io.UnsupportedOperation('sealed source slices are read-only')

    def close(self):
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            try:
                os.close(descriptor)
            finally:
                super().close()
        else:
            super().close()


def _digest(stream):
    digest = hashlib.sha256()
    for part in iter(lambda: stream.read(_READ_BYTES), b''):
        digest.update(part)
    return digest.hexdigest()


def _range_digest(fd, offset, length):
    with FdSlice(fd, offset, length) as source:
        return _digest(source)


def _member_name(member):
    name = member.name
    if member.isdir() and name.endswith('/'):
        name = name[:-1]
    if (not name or name.startswith('/') or '\\' in name or
            any(p in ('', '.', '..') for p in name.split('/')) or
            any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in name)):
        raise ValueError('unsafe or ambiguous TAR member name')
    return name


def index_archive(fd, *, expected_bytes, expected_sha256) -> dict:
    """Hash every original byte and enumerate safe regular members/directories.

    Each member's payload and complete header span are hashed. All bytes after
    the last physical extent must be the zero TAR trailer, including at least
    two records; an appended hidden archive is rejected rather than ignored.
    """
    if (type(expected_bytes) is not int or expected_bytes < 1024 or expected_bytes % 512 or
            not isinstance(expected_sha256, str) or not _HASH.fullmatch(expected_sha256)):
        raise ValueError('invalid complete TAR byte identity')
    if os.fstat(fd).st_size != expected_bytes:
        raise ValueError('sealed archive size differs from expected bytes')
    members, names, end = [], set(), 0
    with FdSlice(fd, 0, expected_bytes) as source:
        if _digest(source) != expected_sha256:
            raise ValueError('complete archive SHA256 differs')
        source.seek(0)
        try:
            with tarfile.open(fileobj=source, mode='r:') as archive:
                for member in archive:
                    name = _member_name(member)
                    if name in names:
                        raise ValueError('duplicate TAR member name')
                    names.add(name)
                    if (member.sparse is not None or member.type == tarfile.GNUTYPE_SPARSE or
                            any(k.startswith('GNU.sparse') for k in member.pax_headers)):
                        raise ValueError('sparse TAR members are unsupported')
                    if not (member.isdir() or member.isreg()) or (member.isdir() and member.size):
                        raise ValueError('unsupported TAR member type')
                    if (member.offset < end or member.offset_data < member.offset+512 or
                            member.offset_data % 512 or member.size < 0):
                        raise ValueError('invalid or overlapping TAR extent')
                    end = member.offset_data + ((member.size+511)//512)*512
                    if end+1024 > expected_bytes:
                        raise ValueError('truncated TAR member or trailer')
                    members.append({'name': name, 'kind': 'directory' if member.isdir() else 'file',
                                    'offset': member.offset, 'offset_data': member.offset_data,
                                    'bytes': member.size, 'pax_headers': dict(member.pax_headers),
                                    'sha256': _range_digest(source.fileno(), member.offset_data, member.size),
                                    'headers_sha256': _range_digest(source.fileno(), member.offset, member.offset_data-member.offset)})
        except (tarfile.TarError, EOFError, OSError) as error:
            raise ValueError('invalid or truncated original TAR') from error
        if expected_bytes-end < 1024:
            raise ValueError('missing complete TAR trailer')
        source.seek(end)
        for part in iter(lambda: source.read(_READ_BYTES), b''):
            if part.strip(b'\x00'):
                raise ValueError('nonzero or appended data after TAR members')
    return {'format': 'emg2pose_sealed_tar_index_v1', 'archive_bytes': expected_bytes,
            'archive_sha256': expected_sha256, 'trailer_bytes': expected_bytes-end, 'members': members}


def _acquisition(cohort, expected_bytes, expected_sha256, receipt):
    from . import emg2pose_archive as transport, emg2pose_data as storage
    if cohort == 'engineering':
        return None
    if not isinstance(receipt, dict) or not isinstance(receipt.get('source'), dict):
        raise ValueError('official cohort requires a complete memory acquisition receipt')
    receipt = deepcopy(receipt)
    source = receipt['source']
    url, byte_count = transport._ASSETS['mini' if cohort == 'mini' else 'full']
    if (receipt.get('format') != 'emg2pose_sealed_memory_archive_v1' or
            receipt.get('complete') is not True or receipt.get('full_corpus_complete') is not False or
            receipt.get('source_bytes_received') != expected_bytes or
            receipt.get('full_sha256') != expected_sha256 or receipt.get('seals') != _SEALS or
            source.get('url') != url or source.get('expected_bytes') != expected_bytes or
            expected_bytes != byte_count or not isinstance(source.get('etag'), str) or
            not re.fullmatch(r'"[^"\r\n]+"', source['etag']) or
            not isinstance(source.get('last_modified'), str) or not source['last_modified'].strip()):
        raise ValueError('official acquisition identity is incomplete or differs from the archive')
    if cohort == 'mini' and (expected_bytes != storage.MINI_BYTES or expected_sha256 != storage.MINI_SHA256):
        raise ValueError('official mini archive identity differs')
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {'receipt_sha256': digest, 'receipt': receipt}


def inspect_archive(fd, *, expected_bytes, expected_sha256, metadata_csv, upstream,
                    cohort='engineering', acquisition_receipt=None, chunk_rows=262144) -> dict:
    """Inventory all HDF5 fields/masks in original logical session order.

    Full/mini admission binds the exact official CSV and completed byte-cache
    receipt. The original pinned helpers define ordering, IK masks and blocks.
    The returned private receipt is not a completed memmap directory or a model
    result; the caller must retain ownership of the volatile source descriptor.
    """
    import h5py
    import numpy as np
    from . import emg2pose_data as storage

    if cohort not in {'engineering', 'mini', 'full'}:
        raise ValueError('cohort must be engineering, mini or full')
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= 262144:
        raise ValueError('chunk_rows must be an integer from1 to262144')
    upstream, official_identity, helpers = storage._official(upstream)
    metadata_csv = storage._safe_path(metadata_csv, file=True)
    csv_hash = storage._sha256(metadata_csv)
    metadata = storage._csv_rows(metadata_csv)
    if cohort != 'engineering' and (csv_hash != storage.CSV_SHA256 or len(metadata) != storage.FULL_FILE_COUNT):
        raise ValueError('official cohort requires the exact pinned full metadata CSV')
    acquisition = _acquisition(cohort, expected_bytes, expected_sha256, acquisition_receipt)
    source_index = index_archive(fd, expected_bytes=expected_bytes, expected_sha256=expected_sha256)
    files = [m for m in source_index['members'] if m['kind'] == 'file' and PurePosixPath(m['name']).suffix == '.hdf5']
    names = [PurePosixPath(m['name']).name for m in files]
    if not files or len(set(names)) != len(names):
        raise ValueError('empty HDF5 population or duplicate physical filename')
    if cohort == 'full' and {PurePosixPath(n).stem for n in names} != set(metadata):
        raise ValueError('full archive must contain the exact official filename population')
    if cohort == 'mini' and len(files) != 30:
        raise ValueError('official mini requires all30 HDF5 files')
    sessions = []
    fields = {'time': np.dtype('<f8'), 'emg': np.dtype(('<f4', (16,))),
              'joint_angles': np.dtype(('<f4', (20,)))}
    for member in sorted(files, key=lambda m: PurePosixPath(m['name']).name):
        try:
            with FdSlice(fd, member['offset_data'], member['bytes']) as raw:
                with h5py.File(raw, 'r') as file:
                    objects = storage._hdf5_objects(file)
                    group = file['emg2pose']
                    ts = group['timeseries']
                    if not isinstance(ts, h5py.Dataset) or ts.ndim != 1 or len(ts) <= 0:
                        raise ValueError('timeseries must be a nonempty one-dimensional dataset')
                    if not ts.dtype.fields or any(name not in ts.dtype.fields or ts.dtype.fields[name][0] != dtype for name, dtype in fields.items()):
                        raise ValueError('source dtype must contain time f8, EMG16 f4 and angles20 f4')
                    attrs = group.attrs
                    filename = storage._safe_name(attrs['filename'])
                    if PurePosixPath(member['name']).stem != filename or filename not in metadata:
                        raise ValueError('source member filename does not match metadata CSV')
                    for key, expected in metadata[filename].items():
                        actual = storage._bool(attrs[key]) if key.startswith('held_out_') else (float(attrs[key]) if key in {'start', 'end'} else storage._text(attrs[key]))
                        if actual != expected:
                            raise ValueError('source attributes conflict with metadata CSV: '+key)
                    if float(attrs['sample_rate']) != 2000 or float(attrs['num_channels']) != 16:
                        raise ValueError('source sample rate or channel metadata differs')
                    explicit_ik = 'ik_failure_mask' in group
                    if explicit_ik and (group['ik_failure_mask'].shape != (len(ts),) or group['ik_failure_mask'].dtype != np.dtype('bool')):
                        raise ValueError('explicit IK mask must have one bool per source row')
                    sessions.append(dict(metadata[filename], session_id=storage._text(attrs['session']),
                                         member_name=member['name'], member_offset=member['offset_data'],
                                         rows=len(ts), bytes=member['bytes'], sha256=member['sha256'],
                                         explicit_ik_mask=explicit_ik, sample_rate=float(attrs['sample_rate']),
                                         hdf5_objects=objects))
        except (OSError, KeyError, TypeError, UnicodeError) as error:
            raise ValueError('invalid or unreadable HDF5 member: '+member['name']) from error
    sessions = [vars(s) for s in helpers['order_sessions_by_split']([SimpleNamespace(**s) for s in sessions])]
    offset = 0
    for session in sessions:
        session['start_idx'], session['end_idx'] = offset, offset+session['rows']
        offset += session['rows']
    digests = {name: hashlib.sha256() for name in storage.FIELD_DTYPES}
    blocks, session_hashes, valid_rows = [], [], 0
    for si, session in enumerate(sessions):
        local = {name: hashlib.sha256() for name in digests}
        opened, local_blocks = None, []
        with FdSlice(fd, session['member_offset'], session['bytes']) as raw:
            with h5py.File(raw, 'r') as file:
                group = file['emg2pose']
                for start in range(0, session['rows'], chunk_rows):
                    end = min(session['rows'], start+chunk_rows)
                    chunk = group['timeseries'][start:end]
                    no_fail = ~group['ik_failure_mask'][start:end] if session['explicit_ik_mask'] else helpers['get_ik_failures_mask'](chunk['joint_angles'])
                    valid = np.asarray(no_fail & np.isfinite(chunk['joint_angles']).all(axis=1), dtype=bool)
                    values = {name: chunk[name] for name in ('emg', 'joint_angles', 'time')}
                    values['valid_mask'] = valid
                    for name, value in values.items():
                        payload = np.ascontiguousarray(value).tobytes()
                        digests[name].update(payload)
                        local[name].update(payload)
                    valid_rows += int(np.count_nonzero(valid))
                    opened = helpers['_update_blocks_from_valid_chunk'](valid, global_start=session['start_idx']+start,
                                                                       open_block_start=opened, blocks=local_blocks)
        if opened is not None:
            local_blocks.append((opened, session['end_idx']))
        blocks.extend([[si, int(a), int(b)] for a, b in local_blocks if b-a >= 1])
        session_hashes.append({'filename': session['filename'], 'source_sha256': session['sha256'],
                               'fields_sha256': {name: d.hexdigest() for name, d in local.items()}})
    shapes = {'emg': [offset, 16], 'joint_angles': [offset, 20], 'time': [offset], 'valid_mask': [offset]}
    field_specs = {name: {'dtype': dtype.str, 'shape': shapes[name], 'sha256': digests[name].hexdigest()}
                   for name, dtype in storage.FIELD_DTYPES.items()}
    if storage._sha256(metadata_csv) != csv_hash:
        raise ValueError('metadata CSV changed during source inspection')
    return {'format': 'emg2pose_sealed_tar_inventory_v1', 'complete': True, 'cohort': cohort,
            'engineering_only': cohort != 'full', 'full_corpus_complete': cohort == 'full',
            'inspected_file_set_complete': cohort in {'mini', 'full'}, 'source_index': source_index,
            'metadata_csv': {'path': str(metadata_csv), 'sha256': csv_hash, 'filename_count': len(metadata)},
            'upstream': {'path': str(upstream), **official_identity}, 'acquisition': acquisition,
            'reader_sha256': storage._sha256(Path(__file__)),
            'storage_helpers_sha256': storage._sha256(Path(storage.__file__)),
            'counts': {key: dict(Counter(s[key] for s in sessions)) for key in ('split', 'generalization', 'side')},
            'total_rows': offset, 'valid_rows': valid_rows, 'sessions': sessions, 'blocks': blocks,
            'fields': field_specs, 'session_field_hashes': session_hashes,
            'source_retention': 'all original TAR/HDF5 bytes remain in the sealed source while descriptor owners live',
            'mask_policy': 'raw official IK/finite mask; whole-window all-false fallback belongs only to unchanged official reader'}
