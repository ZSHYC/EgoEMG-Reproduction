"""Private compact native metadata and explicit ownership of verified TAR bytes.

The independently frozen completion digest binds this index transaction. Source
content identity is separate from cache-instance paths, descriptors and timings.
Opening a source hashes its actual immutable bytes once; no global fd registry
keeps an abandoned multi-hundred-GB cache alive.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from . import emg2pose_data as storage, emg2pose_memory_data as memory
from .emg2pose_memory_archive import _GET_SEALS, _SEALS


_TOKEN = object()
_HASH = re.compile(r'[0-9a-f]{64}')
_FILES = ('source-inventory.json', 'metadata.npz', 'manifest.json', 'completion.json')
_CODE = ('__init__.py', 'download.py', 'emg2pose_data.py', 'emg2pose_memory_archive.py',
         'emg2pose_memory_data.py', 'emg2pose_memory_index.py')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sources():
    root = Path(__file__).parent
    return {name: storage._sha256(root/name) for name in _CODE}


def _stamp(path):
    info = storage._safe_path(path, file=True).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _logical_inventory(value):
    value = deepcopy(value)
    value['metadata_csv'].pop('path', None)
    value['upstream'].pop('path', None)
    return value


def _arrays(values):
    if any(value.dtype.kind == 'O' for value in values.values()):
        raise ValueError('object metadata arrays are forbidden')
    return {name: {'dtype': value.dtype.str, 'shape': list(value.shape),
                   'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
            for name, value in sorted(values.items())}


def _sessions(inventory):
    keys = ('filename', 'session_id', 'start_idx', 'end_idx', 'rows', 'member_name',
            'member_offset', 'bytes', 'sha256', 'explicit_ik_mask')
    return [{key: session[key] for key in keys} for session in inventory['sessions']]


def _identity(inventory, arrays):
    return {'format': 'emg2pose_memory_data_identity_v1', 'cohort': inventory['cohort'],
            'full_corpus_complete': inventory['full_corpus_complete'],
            'engineering_only': inventory['engineering_only'], 'rows': inventory['total_rows'],
            'file_count': len(inventory['sessions']), 'input_channels': 16, 'angle_dimensions': 20,
            'counts': inventory['counts'], 'present_splits': sorted(inventory['counts']['split']),
            'archive_bytes': inventory['source_index']['archive_bytes'],
            'archive_sha256': inventory['source_index']['archive_sha256'],
            'metadata_csv_sha256': inventory['metadata_csv']['sha256'],
            'fields': inventory['fields'], 'metadata_arrays': arrays,
            'upstream_revision': inventory['upstream']['revision'],
            'official_helpers_sha256': inventory['upstream']['files_sha256'],
            'reader_sha256': inventory['reader_sha256'],
            'storage_helpers_sha256': inventory['storage_helpers_sha256']}


def prepare_index(fd: int, inventory: dict, output: Path, *, upstream: Path, metadata_csv: Path) -> dict:
    """Independently verify source inventory, atomically publish only sidecars."""
    output = storage._safe_path(output)
    stage = output.with_name(output.name+'.incomplete')
    if output.exists() or stage.exists():
        raise FileExistsError('existing index or incomplete transaction is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('index parent must already exist')
    if not isinstance(inventory, dict) or inventory.get('format') != 'emg2pose_sealed_tar_inventory_v1' or inventory.get('complete') is not True:
        raise ValueError('complete sealed-TAR source inventory is required')
    try:
        actual = memory.inspect_archive(fd, expected_bytes=inventory['source_index']['archive_bytes'],
                                        expected_sha256=inventory['source_index']['archive_sha256'],
                                        metadata_csv=metadata_csv, upstream=upstream, cohort=inventory['cohort'],
                                        acquisition_receipt=(inventory.get('acquisition') or {}).get('receipt'))
        if _logical_inventory(actual) != _logical_inventory(inventory):
            raise ValueError('inventory differs from actual sealed source')
    except (KeyError, TypeError) as error:
        raise ValueError('invalid source inventory') from error
    md = storage._sidecar(actual['sessions'], actual['blocks'])
    arrays = _arrays(md)
    identity = _identity(actual, arrays)
    code = _sources()
    required = sum(v.nbytes for v in md.values()) + len(json.dumps(actual, allow_nan=False).encode()) + 4*1024**2
    storage._space(output.parent, required)
    stage.mkdir(mode=0o700)
    try:
        storage._write_json(stage/'source-inventory.json', actual)
        np.savez(stage/'metadata.npz', **md)
        with np.load(stage/'metadata.npz', allow_pickle=False) as opened:
            if set(opened.files) != set(md) or any(opened[k].dtype != md[k].dtype or opened[k].tobytes() != md[k].tobytes() for k in md):
                raise ValueError('native metadata serialization differs')
        with (stage/'metadata.npz').open('rb') as stream:
            os.fsync(stream.fileno())
        manifest = {'format': 'emg2pose_sealed_tar_dataset_v1', 'total_rows': actual['total_rows'],
                    'sessions': _sessions(actual), 'fields': actual['fields'],
                    'archive_bytes': actual['source_index']['archive_bytes'],
                    'archive_sha256': actual['source_index']['archive_sha256'],
                    'source_inventory_sha256': storage._sha256(stage/'source-inventory.json'),
                    'metadata_sha256': storage._sha256(stage/'metadata.npz'),
                    'metadata_arrays': arrays}
        storage._write_json(stage/'manifest.json', manifest)
        completion = {'format': 'emg2pose_memory_index_v1', 'complete': True,
                      'manifest_sha256': storage._sha256(stage/'manifest.json'),
                      'source_inventory_sha256': manifest['source_inventory_sha256'],
                      'metadata_sha256': manifest['metadata_sha256'], 'producer_files_sha256': code,
                      'data_identity': identity}
        storage._space(stage)
        if code != _sources() or storage._sha256(metadata_csv) != actual['metadata_csv']['sha256']:
            raise ValueError('producer or source metadata changed during index preparation')
        storage._write_json(stage/'completion.json', completion)
        receipt = {'completion_sha256': storage._sha256(stage/'completion.json'), 'data_identity': deepcopy(identity)}
        storage._publish_directory(stage, output)
        return receipt
    except BaseException as error:
        if stage.exists():
            (stage/'completion.json').unlink(missing_ok=True)
            storage._write_json(stage/'status.json', {'complete': False, 'error_type': type(error).__name__,
                                                     'error': str(error), 'policy': 'owned incomplete index retained; no source removed'})
        raise


def _load_index(index, expected_completion_sha256, upstream, metadata_csv):
    if not isinstance(expected_completion_sha256, str) or not _HASH.fullmatch(expected_completion_sha256):
        raise ValueError('separately frozen completion SHA256 is required')
    index, metadata_csv = storage._safe_path(index), storage._safe_path(metadata_csv, file=True)
    if index.name.endswith('.incomplete') or set(p.name for p in index.iterdir()) != set(_FILES):
        raise ValueError('not an exact completed memory index')
    upstream, official, _ = storage._official(upstream)
    paths = [index/name for name in _FILES] + [metadata_csv]
    paths += [Path(__file__).parent/name for name in _CODE]
    paths += [upstream/name for name in storage.SOURCE_HASHES]
    stamps = {path: _stamp(path) for path in paths}
    if storage._sha256(index/'completion.json') != expected_completion_sha256:
        raise ValueError('completion differs from separately frozen receipt')
    completion = storage._json(index/'completion.json')
    manifest = storage._json(index/'manifest.json')
    inventory = storage._json(index/'source-inventory.json')
    if (completion.get('format') != 'emg2pose_memory_index_v1' or completion.get('complete') is not True or
            manifest.get('format') != 'emg2pose_sealed_tar_dataset_v1' or
            inventory.get('format') != 'emg2pose_sealed_tar_inventory_v1' or inventory.get('complete') is not True):
        raise ValueError('invalid memory index formats or completion')
    for name, key in [('manifest.json', 'manifest_sha256'), ('metadata.npz', 'metadata_sha256'),
                      ('source-inventory.json', 'source_inventory_sha256')]:
        if completion.get(key) != storage._sha256(index/name):
            raise ValueError('index transaction file digest differs')
    if completion.get('producer_files_sha256') != _sources():
        raise ValueError('memory index producer code differs')
    if (inventory['upstream']['revision'] != official['revision'] or
            inventory['upstream']['files_sha256'] != official['files_sha256'] or
            inventory.get('reader_sha256') != storage._sha256(Path(memory.__file__)) or
            inventory.get('storage_helpers_sha256') != storage._sha256(Path(storage.__file__))):
        raise ValueError('source/helper identity differs')
    cohort = inventory['cohort']
    if cohort not in {'engineering', 'mini', 'full'}:
        raise ValueError('unknown source cohort')
    for name, expected in [('full_corpus_complete', cohort == 'full'), ('engineering_only', cohort != 'full'),
                           ('inspected_file_set_complete', cohort != 'engineering')]:
        if inventory.get(name) is not expected:
            raise ValueError('source cohort claims disagree')
    csv_hash = storage._sha256(metadata_csv)
    rows = storage._csv_rows(metadata_csv)
    if csv_hash != inventory['metadata_csv']['sha256'] or len(rows) != inventory['metadata_csv']['filename_count']:
        raise ValueError('bound metadata CSV differs')
    if cohort != 'engineering' and (csv_hash != storage.CSV_SHA256 or len(rows) != storage.FULL_FILE_COUNT):
        raise ValueError('official cohort requires pinned full metadata')
    from .emg2pose_runtime import _source_records
    sessions, total_rows, names = _source_records(inventory, rows)
    if total_rows != inventory['total_rows'] or (cohort == 'full' and names != set(rows)) or (cohort == 'mini' and len(sessions) != 30):
        raise ValueError('source population/row count differs')
    archive = inventory['source_index']
    acquired = memory._acquisition(cohort, archive['archive_bytes'], archive['archive_sha256'],
                                   (inventory.get('acquisition') or {}).get('receipt'))
    if acquired != inventory.get('acquisition'):
        raise ValueError('source acquisition differs')
    shapes = {'emg': [total_rows,16], 'joint_angles': [total_rows,20], 'time': [total_rows], 'valid_mask': [total_rows]}
    if set(inventory['fields']) != set(shapes):
        raise ValueError('exactly four source fields are required')
    for key, shape in shapes.items():
        spec = inventory['fields'][key]
        if spec['shape'] != shape or np.dtype(spec['dtype']) != storage.FIELD_DTYPES[key] or not _HASH.fullmatch(spec['sha256']):
            raise ValueError('field schema differs from16ch/20D source')
    counts = {key: dict(Counter(s[key] for s in sessions)) for key in ('split','generalization','side')}
    if inventory['counts'] != counts:
        raise ValueError('source counts differ')
    prior = 0
    for block in inventory['blocks']:
        if not isinstance(block, list) or len(block) != 3 or any(type(x) is not int for x in block):
            raise ValueError('invalid source block')
        si,a,b = block
        if si < 0 or si >= len(sessions) or a < prior or b <= a or a < sessions[si]['start_idx'] or b > sessions[si]['end_idx']:
            raise ValueError('source block overlaps or crosses sessions')
        prior = b
    if sum(b-a for _,a,b in inventory['blocks']) != inventory['valid_rows']:
        raise ValueError('raw valid count differs from blocks')
    expected_arrays = storage._sidecar(sessions, inventory['blocks'])
    with np.load(index/'metadata.npz', allow_pickle=False) as opened:
        if set(opened.files) != set(expected_arrays) or any(opened[k].dtype != v.dtype or opened[k].tobytes() != v.tobytes() for k,v in expected_arrays.items()):
            raise ValueError('native metadata arrays differ from source')
    arrays = _arrays(expected_arrays)
    identity = _identity(inventory, arrays)
    expected_manifest = {'format': 'emg2pose_sealed_tar_dataset_v1', 'total_rows': total_rows,
                         'sessions': _sessions(inventory), 'fields': inventory['fields'],
                         'archive_bytes': archive['archive_bytes'], 'archive_sha256': archive['archive_sha256'],
                         'source_inventory_sha256': completion['source_inventory_sha256'],
                         'metadata_sha256': completion['metadata_sha256'], 'metadata_arrays': arrays}
    if manifest != expected_manifest or completion.get('data_identity') != identity:
        raise ValueError('index manifest or content identity differs')
    if any(_stamp(path) != stamp for path,stamp in stamps.items()):
        raise ValueError('index binding changed during admission')
    return index, manifest, identity, stamps


class VerifiedSource:
    """Own bytes verified by open_source, with an independently bound index."""

    def __init__(self, descriptor, index, manifest, identity, stamps, *, _token):
        self._descriptor = None
        if _token is not _TOKEN:
            raise TypeError('use open_source to construct a verified source')
        self.index = index
        self._manifest, self._identity, self._stamps = deepcopy(manifest), deepcopy(identity), stamps
        info = os.fstat(descriptor)
        self._handle = {'device': info.st_dev, 'inode': info.st_ino, 'size': info.st_size,
                        'seals': fcntl.fcntl(descriptor, _GET_SEALS), 'archive_sha256': identity['archive_sha256'],
                        'index_completion_sha256': storage._sha256(index/'completion.json'),
                        'binding_stamps': {str(path):list(value) for path,value in stamps.items()}}
        # Ownership transfers only after construction succeeds; until then the
        # factory remains responsible for closing the provisional descriptor.
        self._descriptor = descriptor

    def fileno(self):
        if self._descriptor is None:
            raise ValueError('verified source is closed')
        info = os.fstat(self._descriptor)
        if ((info.st_dev,info.st_ino,info.st_size) != (self._handle['device'],self._handle['inode'],self._handle['size']) or
                fcntl.fcntl(self._descriptor,_GET_SEALS) & _SEALS != _SEALS):
            raise ValueError('verified immutable handle changed')
        return self._descriptor

    def assert_current(self):
        self.fileno()
        if any(_stamp(path) != stamp for path,stamp in self._stamps.items()):
            raise ValueError('verified index binding changed')

    @property
    def data_identity(self):
        self.assert_current()
        return deepcopy(self._identity)

    @property
    def manifest(self):
        self.assert_current()
        return deepcopy(self._manifest)

    @property
    def handle_identity(self):
        self.assert_current()
        return deepcopy(self._handle)

    def close(self):
        if self._descriptor is not None:
            descriptor,self._descriptor = self._descriptor,None
            os.close(descriptor)

    def __enter__(self):
        self.assert_current()
        return self

    def __exit__(self,*_):
        self.close()

    def __del__(self):
        if getattr(self,'_descriptor',None) is not None:
            self.close()


def open_source(fd: int, index: Path, *, expected_completion_sha256: str,
                upstream: Path, metadata_csv: Path) -> VerifiedSource:
    """Verify actual immutable bytes once, retain a duplicate until explicit close."""
    try:
        index,manifest,identity,stamps = _load_index(index, expected_completion_sha256, upstream, metadata_csv)
        with memory.FdSlice(fd,0,manifest['archive_bytes']) as raw:
            if os.fstat(raw.fileno()).st_size != manifest['archive_bytes'] or memory._digest(raw) != manifest['archive_sha256']:
                raise ValueError('actual source bytes differ from the bound index')
            descriptor = os.dup(raw.fileno())
        try:
            source = VerifiedSource(descriptor,index,manifest,identity,stamps,_token=_TOKEN)
        except BaseException:
            os.close(descriptor)
            raise
        try:
            source.assert_current()
            return source
        except BaseException:
            # Invalidate the actual owner, rather than closing its numeric fd
            # behind its back (a deferred destructor could close a reused fd).
            source.close()
            raise
    except (OSError,KeyError,TypeError,EOFError) as error:
        raise ValueError('invalid completed memory index or source handle') from error
