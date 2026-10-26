"""Private, lossless EMG2Pose inventory and direct official-memmap conversion.

Receipts contain source paths and participant metadata: keep them with private
data. No dataset split, invalid label, input precision or source file is changed.
The complete original HDF5 files remain the persistent source for extra fields.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter, deque
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from types import SimpleNamespace
from typing import Sequence
import subprocess
import tarfile

import h5py
import numpy as np

from . import UPSTREAM_COMMIT
from .download import MIN_FREE_BYTES


CSV_SHA256 = '0e7ad04a79a0f162385d22ec8d6f43af9b8360978ca11ef9d601bbf9594b09e6'
FULL_FILE_COUNT = 25253
MINI_BYTES = 647198208
MINI_SHA256 = 'd7400e98508ccbb2139c2d78e552867b23501f637456546fd6680f3fe7fec50d'
SOURCE_HASHES = {
    'scripts/data/convert_hdf5_to_zarr.py': '15170f53d248de7e5f886e160322a6ae68fde41b07b83126032d58403ca7b441',
    'egoemg/utils.py': 'd8faf33e037b4decec3b7febcb2d3db1576dc83e74d36a8f4b4739a8aa5241d2',
    'egoemg/datasets/emg2pose_dataset.py': '82fbd93849b906543f6f0705b8ca220b40ff81e3ee26175527c2a2d3124f2cc6',
}
FIELD_DTYPES = {'emg': np.dtype('<f4'), 'joint_angles': np.dtype('<f4'),
                'time': np.dtype('<f8'), 'valid_mask': np.dtype('bool')}


def _hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256(path):
    with Path(path).open('rb') as stream:
        return _hash_stream(stream)


def _safe_path(value, *, file=False):
    path = Path(value).expanduser()
    if '..' in path.parts:
        raise ValueError('unsafe parent traversal in path')
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe symlink path')
    if file and not path.is_file():
        raise ValueError('required source file is missing')
    return path


def _json(path):
    try:
        value = json.loads(_safe_path(path, file=True).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid JSON receipt') from exc
    if not isinstance(value, dict):
        raise ValueError('receipt must be an object')
    return value


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _official(upstream):
    root = _safe_path(upstream)
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError('upstream revision must match the pinned official commit')
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], capture_output=True, text=True)
    if status.returncode or status.stdout.strip():
        raise ValueError('upstream tree must be clean, including conversion scripts')
    hashes = {}
    for name, expected in SOURCE_HASHES.items():
        hashes[name] = _sha256(root / name)
        if expected is not None and hashes[name] != expected:
            raise ValueError('upstream source content differs from pinned conversion code')
    namespace = {'np': np, 'deque': deque, 'Sequence': Sequence, 'SessionInfo': SimpleNamespace}
    selections = {
        'scripts/data/convert_hdf5_to_zarr.py': {'_normalize_split', 'order_sessions_by_split', '_update_blocks_from_valid_chunk'},
        'egoemg/utils.py': {'get_ik_failures_mask'},
    }
    for name, wanted in selections.items():
        tree = ast.parse((root / name).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        if len(nodes) != len(wanted):
            raise ValueError('upstream conversion helper is missing')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / name), 'exec'), namespace)
    return root, {'revision': UPSTREAM_COMMIT, 'files_sha256': hashes}, namespace


def _text(value):
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode('utf-8')
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('metadata text must be nonempty UTF-8')
    return value


def _bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value == 'True':
        return True
    if value == 'False':
        return False
    raise ValueError('metadata boolean must be bool or literal True/False')


def _safe_name(value):
    value = _text(value)
    if value in {'.', '..'} or '/' in value or '\\' in value:
        raise ValueError('unsafe filename identifier')
    return value


def _typed_attr(value, dtype):
    array = np.asarray(value)
    result = {'dtype': str(dtype), 'dtype_metadata': repr(dtype.metadata), 'shape': list(array.shape)}
    if array.dtype.kind == 'O':
        result['text_values'] = [_text(x) for x in array.reshape(-1)]
    elif array.dtype.kind == 'U':
        result['text_values'] = array.reshape(-1).tolist()
    else:
        result['bytes_base64'] = base64.b64encode(array.tobytes()).decode('ascii')
        result['numpy_dtype'] = array.dtype.str
    return result


def _hdf5_objects(file):
    objects, seen = {}, {}
    def visit(obj, name):
        address = h5py.h5o.get_info(obj.id).addr
        if address in seen:
            objects[name] = {'hardlink_alias_of': seen[address]}
            return
        seen[address] = name
        record = {'kind': 'dataset' if isinstance(obj, h5py.Dataset) else 'group',
                  'attrs': {key: _typed_attr(obj.attrs[key], obj.attrs.get_id(key).dtype) for key in obj.attrs}}
        objects[name] = record
        if isinstance(obj, h5py.Dataset):
            if obj.external or obj.is_virtual:
                raise ValueError('unsafe external or virtual HDF5 storage')
            record.update(shape=list(obj.shape), dtype=repr(obj.dtype.descr))
        else:
            for key in obj:
                if not isinstance(obj.get(key, getlink=True), h5py.HardLink):
                    raise ValueError('unsafe external/soft HDF5 link')
                visit(obj[key], name.rstrip('/') + '/' + key)
    visit(file, '/')
    return objects


def _csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    required = {'filename', 'session', 'user', 'stage', 'side', 'moving_hand', 'split',
                'generalization', 'held_out_user', 'held_out_stage', 'start', 'end'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('metadata CSV lacks required session columns')
    result = {}
    for row in rows:
        key = _safe_name(row['filename'])
        if key in result:
            raise ValueError('duplicate filename in metadata CSV')
        row = {k: row[k] for k in sorted(required)}
        for field in ('held_out_user', 'held_out_stage'):
            row[field] = _bool(row[field])
        if row['split'] not in {'train', 'val', 'test'} or row['side'] not in {'left', 'right'}:
            raise ValueError('unknown split or side in metadata CSV')
        flags = {'none': (False, False), 'user': (True, False), 'stage': (False, True), 'user_stage': (True, True)}
        if flags.get(row['generalization']) != (row['held_out_user'], row['held_out_stage']):
            raise ValueError('generalization and held-out booleans disagree')
        for field in ('start', 'end'):
            row[field] = float(row[field])
            if not np.isfinite(row[field]):
                raise ValueError('nonfinite session time in metadata CSV')
        result[key] = row
    return result


def _mini_acquisition(path, sessions):
    receipt_path = _safe_path(path, file=True)
    receipt = _json(receipt_path)
    if (receipt.get('complete') is not True or receipt.get('bytes') != MINI_BYTES or
            receipt.get('expected_bytes') != MINI_BYTES or receipt.get('sha256') != MINI_SHA256):
        raise ValueError('official mini acquisition identity is incomplete')
    archive = _safe_path(receipt.get('archive', ''), file=True)
    if archive.stat().st_size != MINI_BYTES or _sha256(archive) != MINI_SHA256:
        raise ValueError('official mini archive identity differs')
    claimed = {}
    for entry in receipt.get('files', []):
        member_name = PurePosixPath(entry['name'])
        if member_name.is_absolute() or '..' in member_name.parts or str(member_name) in claimed:
            raise ValueError('unsafe or duplicate acquisition member')
        claimed[str(member_name)] = entry
    actual, hdf5_members = {}, {}
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError('unsafe official mini archive member')
            if member.isdir():
                continue
            if not member.isfile() or member.name in actual:
                raise ValueError('invalid or duplicate archive member')
            digest = _hash_stream(tar.extractfile(member))
            actual[member.name] = {'name': member.name, 'bytes': member.size, 'sha256': digest, 'offset_data': member.offset_data}
            if name.suffix == '.hdf5':
                if name.name in hdf5_members:
                    raise ValueError('duplicate HDF5 basename in mini archive')
                hdf5_members[name.name] = actual[member.name]
    if actual != claimed or len(hdf5_members) != 30 or len(actual) != 31:
        raise ValueError('official mini acquisition member set differs')
    by_name = {Path(s['path']).name: s for s in sessions}
    if set(by_name) != set(hdf5_members):
        raise ValueError('inspected files are not the complete official mini set')
    for name, entry in hdf5_members.items():
        if (by_name[name]['bytes'], by_name[name]['sha256']) != (entry['bytes'], entry['sha256']):
            raise ValueError('mini source differs from the verified archive member')
    return {'path': str(receipt_path), 'sha256': _sha256(receipt_path), 'archive_sha256': MINI_SHA256}


def inspect_corpus(paths: Sequence[Path], metadata_csv: Path, *, cohort: str,
                   upstream: Path, acquisition_receipt: Path | None = None) -> dict:
    """Inventory engineering files or prove an exact official mini/full cohort."""
    if cohort not in {'engineering', 'mini', 'full'}:
        raise ValueError('cohort must be engineering, mini or full')
    official_root, official_identity, helpers = _official(upstream)
    metadata_csv = _safe_path(metadata_csv, file=True)
    csv_hash = _sha256(metadata_csv)
    if cohort != 'engineering' and csv_hash != CSV_SHA256:
        raise ValueError('official cohort requires the pinned full metadata CSV')
    rows = _csv_rows(metadata_csv)
    if cohort != 'engineering' and len(rows) != FULL_FILE_COUNT:
        raise ValueError('official metadata must contain exactly 25253 filenames')
    paths = [_safe_path(path, file=True) for path in paths]
    if not paths or len(set(paths)) != len(paths) or len({p.name for p in paths}) != len(paths):
        raise ValueError('empty corpus or duplicate source filename')
    sessions = []
    for path in sorted(paths):
        before = _sha256(path)
        try:
            with h5py.File(path, 'r') as file:
                objects = _hdf5_objects(file)
                group = file['emg2pose']
                ts = group['timeseries']
                if not isinstance(ts, h5py.Dataset) or ts.ndim != 1 or len(ts) <= 0:
                    raise ValueError('timeseries must be a nonempty one-dimensional dataset')
                fields = {'time': np.dtype('<f8'), 'emg': np.dtype(('<f4', (16,))),
                          'joint_angles': np.dtype(('<f4', (20,)))}
                if not ts.dtype.fields or any(name not in ts.dtype.fields or ts.dtype.fields[name][0] != dtype for name, dtype in fields.items()):
                    raise ValueError('source dtype must contain time f8, EMG16 f4 and angles20 f4')
                attrs = group.attrs
                filename = _safe_name(attrs['filename'])
                if path.suffix != '.hdf5' or path.stem != filename or filename not in rows:
                    raise ValueError('source filename does not match metadata CSV')
                for key, expected in rows[filename].items():
                    actual = _bool(attrs[key]) if key.startswith('held_out_') else (float(attrs[key]) if key in {'start', 'end'} else _text(attrs[key]))
                    if actual != expected:
                        raise ValueError('source attributes conflict with metadata CSV: ' + key)
                if float(attrs['sample_rate']) != 2000 or float(attrs['num_channels']) != 16:
                    raise ValueError('source sample rate or channel metadata differs')
                explicit_ik = 'ik_failure_mask' in group
                if explicit_ik and (group['ik_failure_mask'].shape != (len(ts),) or group['ik_failure_mask'].dtype != np.dtype('bool')):
                    raise ValueError('explicit IK mask must have one bool per source row')
                record = dict(rows[filename], session_id=_text(attrs['session']), path=str(path),
                              rows=len(ts), bytes=path.stat().st_size, sha256=before,
                              explicit_ik_mask=explicit_ik, sample_rate=float(attrs['sample_rate']),
                              hdf5_objects=objects)
                sessions.append(record)
        except (OSError, KeyError, TypeError, UnicodeError) as exc:
            raise ValueError('invalid or unreadable HDF5 source: ' + path.name) from exc
        if _sha256(path) != before:
            raise ValueError('source changed during inspection')
    if len({s['filename'] for s in sessions}) != len(sessions):
        raise ValueError('duplicate physical filename identifier')
    if cohort == 'full' and {s['filename'] for s in sessions} != set(rows):
        raise ValueError('full corpus must contain the exact official filename set')
    acquisition = None
    if cohort == 'mini':
        if acquisition_receipt is None:
            raise ValueError('official mini requires a verified acquisition receipt')
        acquisition = _mini_acquisition(acquisition_receipt, sessions)
    objects = [SimpleNamespace(**s) for s in sessions]
    objects = helpers['order_sessions_by_split'](objects)
    sessions = [vars(obj) for obj in objects]
    offset = 0
    for s in sessions:
        s['start_idx'], s['end_idx'] = offset, offset + s['rows']
        offset += s['rows']
    if _sha256(metadata_csv) != csv_hash:
        raise ValueError('metadata CSV changed during inspection')
    return {'format': 'emg2pose_source_inventory_v1', 'cohort': cohort,
            'engineering_only': cohort != 'full', 'full_corpus_complete': cohort == 'full',
            'inspected_file_set_complete': cohort in {'mini', 'full'},
            'metadata_csv': {'path': str(metadata_csv), 'sha256': csv_hash, 'filename_count': len(rows)},
            'upstream': {'path': str(official_root), **official_identity},
            'converter_sha256': _sha256(Path(__file__)), 'acquisition': acquisition,
            'counts': {key: dict(Counter(s[key] for s in sessions)) for key in ('split', 'generalization', 'side')},
            'total_rows': offset, 'sessions': sessions,
            'source_policy': 'original HDF5 retained; no fields, attributes, rows or source bits discarded'}


def _space(directory, required=0):
    if shutil.disk_usage(directory).free < MIN_FREE_BYTES + required:
        raise RuntimeError('insufficient storage for conversion and 15 GiB reserve')


def _bytes_array(values):
    values = [str(v).encode('utf-8') for v in values]
    return np.asarray(values, dtype=f'S{max([1] + [len(v) for v in values])}')


def _sidecar(sessions, blocks):
    md = {}
    for category, source_key, dtype in [('users', 'user', 'i4'), ('stages', 'stage', 'i4'), ('sides', 'side', 'i1'), ('splits', 'split', 'i4')]:
        values = sorted({s[source_key] for s in sessions})
        md[category + '_' + source_key] = _bytes_array(values)
        md['session_' + source_key + '_id'] = np.asarray([values.index(s[source_key]) for s in sessions], dtype=dtype)
    for name, source_key, dtype in [('start_idx', 'start_idx', 'i8'), ('end_idx', 'end_idx', 'i8'), ('length', 'rows', 'i8'),
                                    ('held_out_user', 'held_out_user', 'bool'), ('held_out_stage', 'held_out_stage', 'bool'),
                                    ('sample_rate', 'sample_rate', 'f4')]:
        md['session_' + name] = np.asarray([s[source_key] for s in sessions], dtype=dtype)
    for name in ('session_id', 'filename', 'generalization', 'moving_hand'):
        md['session_' + name] = _bytes_array([s[name] for s in sessions])
    for index, name, dtype in [(0, 'session_idx', 'i4'), (1, 'start', 'i8'), (2, 'end', 'i8')]:
        md['blocks_' + name] = np.asarray([b[index] for b in blocks], dtype=dtype)
    md['blocks_length'] = md['blocks_end'] - md['blocks_start']
    return md


def _publish_directory(stage, output):
    # Linux renameat2(RENAME_NOREPLACE) atomically refuses even a racing empty
    # destination directory; plain os.rename could silently replace that directory.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise RuntimeError('atomic no-replace directory publication is unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def convert_corpus(source_receipt: Path, output: Path, *, chunk_rows: int = 262144,
                   upstream: Path) -> dict:
    """Publish all fields and sidecars together, retaining failed private staging."""
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('chunk_rows must be a positive integer')
    output = _safe_path(output)
    stage = output.with_name(output.name + '.incomplete')
    if output.exists() or stage.exists():
        raise FileExistsError('existing output or incomplete conversion is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('output parent must already exist')
    stage.mkdir(mode=0o700)
    maps = {}
    try:
        source_receipt = _safe_path(source_receipt, file=True)
        receipt_hash = _sha256(source_receipt)
        recorded = _json(source_receipt)
        current = inspect_corpus([Path(s['path']) for s in recorded['sessions']], Path(recorded['metadata_csv']['path']),
                                 cohort=recorded['cohort'], upstream=upstream,
                                 acquisition_receipt=Path(recorded['acquisition']['path']) if recorded.get('acquisition') else None)
        if current != recorded:
            raise ValueError('source identity changed from the inspection receipt')
        _, identity, helpers = _official(upstream)
        n = current['total_rows']
        shapes = {'emg': (n, 16), 'joint_angles': (n, 20), 'time': (n,), 'valid_mask': (n,)}
        data_bytes = sum(int(np.prod(shapes[k])) * dtype.itemsize for k, dtype in FIELD_DTYPES.items())
        _space(stage, data_bytes + 2 * 1024**2 + len(current['sessions']) * 8192)
        _write_json(stage / 'source-receipt.json', current)
        for key, dtype in FIELD_DTYPES.items():
            maps[key] = np.memmap(stage / (key + '.dat'), mode='w+', dtype=dtype, shape=shapes[key])
        digests = {key: hashlib.sha256() for key in maps}
        blocks, session_hashes = [], []
        for si, session in enumerate(current['sessions']):
            path = _safe_path(session['path'], file=True)
            if _sha256(path) != session['sha256']:
                raise ValueError('source changed before conversion')
            local_hashes = {key: hashlib.sha256() for key in maps}
            opened, local_blocks = None, []
            with h5py.File(path, 'r') as file:
                g = file['emg2pose']
                for start in range(0, session['rows'], chunk_rows):
                    end = min(session['rows'], start + chunk_rows)
                    _space(stage, (end - start) * 153)
                    chunk = g['timeseries'][start:end]
                    no_fail = (~g['ik_failure_mask'][start:end]) if session['explicit_ik_mask'] else helpers['get_ik_failures_mask'](chunk['joint_angles'])
                    valid = np.asarray(no_fail & np.isfinite(chunk['joint_angles']).all(axis=1), dtype=bool)
                    a, b = session['start_idx'] + start, session['start_idx'] + end
                    values = {key: chunk[key] for key in ('emg', 'joint_angles', 'time')}
                    values['valid_mask'] = valid
                    for key, value in values.items():
                        maps[key][a:b] = value
                        payload = np.ascontiguousarray(value).tobytes()
                        if maps[key][a:b].tobytes() != payload:
                            raise ValueError('converted field differs from source bytes')
                        digests[key].update(payload)
                        local_hashes[key].update(payload)
                    opened = helpers['_update_blocks_from_valid_chunk'](valid, global_start=a, open_block_start=opened, blocks=local_blocks)
                    _space(stage)
            if opened is not None:
                local_blocks.append((opened, session['end_idx']))
            blocks.extend((si, a, b) for a, b in local_blocks if b - a >= 1)
            if _sha256(path) != session['sha256']:
                raise ValueError('source changed while converting')
            session_hashes.append({'filename': session['filename'], 'source_sha256': session['sha256'],
                                   'fields_sha256': {k: d.hexdigest() for k, d in local_hashes.items()}})
        for mm in maps.values():
            mm.flush()
        for mm in maps.values():
            mm._mmap.close()
        maps.clear()
        fields = {}
        for key, dtype in FIELD_DTYPES.items():
            path = stage / (key + '.dat')
            expected_bytes = int(np.prod(shapes[key])) * dtype.itemsize
            if path.stat().st_size != expected_bytes or _sha256(path) != digests[key].hexdigest():
                raise ValueError('field file size or final hash verification failed')
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            fields[key] = {'filename': path.name, 'dtype': dtype.str, 'shape': list(shapes[key]),
                           'bytes': expected_bytes, 'sha256': digests[key].hexdigest()}
        md = _sidecar(current['sessions'], blocks)
        _space(stage, sum(v.nbytes for v in md.values()) + 1024**2)
        np.savez(stage / 'metadata.npz', **md)
        with np.load(stage / 'metadata.npz', allow_pickle=False) as actual:
            if set(actual.files) != set(md) or any(actual[k].dtype != v.dtype or actual[k].tobytes() != v.tobytes() for k, v in md.items()):
                raise ValueError('metadata sidecar verification failed')
        with (stage / 'metadata.npz').open('rb') as stream:
            os.fsync(stream.fileno())
        manifest = {'format_version': 'emg2pose_direct_memmap_v1', 'total_rows': n, 'fields': fields,
                    'sessions': [{k: s[k] for k in ('session_id', 'filename', 'start_idx', 'end_idx', 'rows', 'split', 'generalization')} for s in current['sessions']],
                    'source_receipt_sha256': receipt_hash}
        _write_json(stage / 'manifest.json', manifest)
        result = {'format': 'emg2pose_conversion_v1', 'complete': True, 'engineering_only': current['engineering_only'],
                  'cohort': current['cohort'], 'full_corpus_complete': current['full_corpus_complete'],
                  'inspected_file_set_complete': current['inspected_file_set_complete'],
                  'total_rows': n, 'file_count': len(current['sessions']), 'valid_blocks': len(blocks),
                  'counts': current['counts'], 'field_bytes': data_bytes, 'fields': fields,
                  'metadata_sha256': _sha256(stage / 'metadata.npz'), 'manifest_sha256': _sha256(stage / 'manifest.json'),
                  'source_receipt_sha256': receipt_hash, 'converter_sha256': _sha256(Path(__file__)),
                  'upstream': identity, 'session_field_hashes': session_hashes,
                  'source_retention': 'original HDF5 files retained in place, including all additional fields and typed attrs',
                  'mask_policy': 'raw official IK/finite mask preserved; whole-window all-false fallback belongs only to the unchanged official reader'}
        if (_sha256(source_receipt) != receipt_hash or
                _sha256(current['metadata_csv']['path']) != current['metadata_csv']['sha256'] or
                any(_sha256(_safe_path(s['path'], file=True)) != s['sha256'] for s in current['sessions'])):
            raise ValueError('source or metadata identity changed before publication')
        _space(stage)
        _write_json(stage / 'completion.json', result)
        descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _publish_directory(stage, output)
        return result
    except BaseException as exc:
        if stage.exists():
            (stage / 'completion.json').unlink(missing_ok=True)
            status = stage / 'status.json'
            status.unlink(missing_ok=True)
            _write_json(status, {'complete': False, 'error_type': type(exc).__name__, 'error': str(exc),
                                 'policy': 'private incomplete evidence retained; no output published'})
        raise
    finally:
        for mm in maps.values():
            mm._mmap.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--chunk-rows', type=int, default=262144)
    args = parser.parse_args(argv)
    result = convert_corpus(args.source_receipt, args.output, chunk_rows=args.chunk_rows, upstream=args.upstream)
    print(json.dumps({k: result[k] for k in ('complete', 'cohort', 'file_count', 'total_rows', 'full_corpus_complete')}))


if __name__ == '__main__':
    main()
