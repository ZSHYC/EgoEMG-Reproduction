"""Whole frozen-cohort CPU admission, never a CUDA/formal training permit.

Every verification hashes the actual Task2 source/data identity and rescans all
unique role footprints. Runtime guards then use stat checks, not repeated hashes.
Raw values and generated supervision are never repaired or filtered.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import uuid
import weakref

import numpy as np

from . import classic_emg, classic_emg_data as data
from .data import LegacyStore, file_sha256, union_intervals

ROLES = ('train', 'val', 'test')
HANDS = ('left', 'right')
MAX_CHUNK_BYTES = 64*1024**2
RESERVE_BYTES = 15*1024**3
_FORMAT = 'classic_full_cohort_cpu_admission_v1'
_PROBE_SHA = '2ce5ec6e3cd197eabd0ea6c0217f1fcd9b43fb8882f2e042f95c02f1795c3a49'
_FROZEN = {
    'egoemg_repro/__init__.py': '6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96',
    'egoemg_repro/classic_emg.py': 'cc5a9b838f57fbd6536e61f9aac1cc6328f4b7e5e2da486f7d1abdba4785c1a5',
    'egoemg_repro/classic_emg_data.py': '5d1783045b094e2ac404ea8a3605122e93ef40c5995d4d18a964f7b0bff62e06',
    'egoemg_repro/data.py': 'de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e',
    'egoemg_repro/partitions.py': 'ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678',
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _safe(path, *, exists=True):
    raw = Path(path).expanduser()
    if '..' in raw.parts:
        raise ValueError('parent traversal is forbidden')
    absolute = raw if raw.is_absolute() else Path.cwd()/raw
    # Check lexical ancestors before resolve: resolve alone would hide a link.
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise ValueError('symlink root, ancestor or source is forbidden')
    return absolute.resolve(strict=exists)


def _stamp(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('source must remain a regular nonsymlink file')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _json(path):
    path = _safe(path)
    before = _stamp(path)
    value = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if _stamp(path) != before:
        raise ValueError('document changed during reading')
    return value


class _StatGuard:
    """Private process-local stamps; no path/inode belongs in content tokens."""
    def __init__(self, paths):
        self.paths, self.parents = {}, set()
        for path in paths:
            self.add(path)

    def add(self, path):
        path = _safe(path)
        if path not in self.paths:
            self.paths[path] = _stamp(path)
            self.parents.update(path.parents)

    def check(self):
        for parent in self.parents:
            if parent.is_symlink():
                raise ValueError('source ancestor changed to symlink')
        for path, before in self.paths.items():
            try:
                current = _stamp(path)
            except OSError as exc:
                raise ValueError('source changed during cohort use') from exc
            if current != before:
                raise ValueError('data/source stamp changed during cohort use')

    @contextmanager
    def read(self):
        self.check()
        try:
            yield
        finally:
            self.check()


def _source_paths(upstream):
    project = Path(__file__).resolve().parents[1]
    return sorted({*upstream.glob('egoemg/**/*.py'), *upstream.glob('config/**/*.yaml'),
                   *upstream.glob('egoemg/UmeTrack/**/*.json'), upstream/'assets/per_dataset_norm_stats.json',
                   *(project/name for name in _FROZEN), Path(__file__).resolve()})


def _prepare(root, partitions, upstream, config, finite_probe):
    root, upstream = _safe(root), _safe(upstream)
    documents = [Path(x) for x in (partitions, config, finite_probe) if isinstance(x, (str, os.PathLike))]
    guard = _StatGuard([root/'manifest.json', root/'metadata.npz', *_source_paths(upstream), *documents])
    store = LegacyStore(root)
    paths = data._schema(store)
    for path in paths.values():
        guard.add(path)
    guard.check()
    project = Path(__file__).resolve().parents[1]
    for name, digest in _FROZEN.items():
        if file_sha256(project/name) != digest:
            raise ValueError('frozen Task1/Task2 source differs')
    if isinstance(config, (str, os.PathLike)):
        recipe = data.load_classic_config(_safe(config), upstream)
        composed = classic_emg.compose_classic(upstream, recipe['variant'])
    else:
        _, contract, _, _ = classic_emg._validated(upstream, config)
        recipe = data._expected_recipe(upstream, contract.variant)
        composed = config
    if recipe['variant'] not in ('vemg2pose', 'neuropose'):
        raise ValueError('cohort admission requires an exact approved base recipe')
    partition = _json(partitions) if isinstance(partitions, (str, os.PathLike)) else json.loads(
        json.dumps(partitions, allow_nan=False))
    expected_settings = {k:recipe[k] for k in ('window_length','train_stride','eval_stride','val_fraction','jitter')}
    if partition.get('settings') != expected_settings:
        raise ValueError('partition settings differ from exact Task2 base recipe')
    norm = data.classic_normalization(upstream)
    identity = data.classic_data_identity(root, partition, upstream=upstream, config=composed,
                                         normalization=norm, mask_policy=data.MASK_POLICY)
    guard.check()
    spans, footprints = {}, {}
    for role in ROLES:
        windows = [w for w in partition['windows'] if w['role'] == role]
        if not windows:
            raise ValueError('full cohort requires every role')
        spans[role] = union_intervals((w['start'], w['end']) for w in windows)
        rows = sum(b-a for a,b in spans[role])
        if partition['coverage'][role]['unique_rows'] != rows:
            raise ValueError('partition role footprint differs')
        footprints[role] = dict(unique_rows=rows, windows=len(windows), hand_samples=2*len(windows),
                                interval_count=len(spans[role]), interval_sha256=_hash(spans[role]))
    return store, partition, recipe, norm, identity, spans, footprints, guard


def _cell():
    return dict(unique_rows=0, elements=0, nonfinite_elements=0, _digest=hashlib.sha256())


def _consume(cell, values):
    cell['unique_rows'] += len(values)
    cell['elements'] += values.size
    bad = int(np.count_nonzero(~np.isfinite(values))) if values.dtype.kind == 'f' else 0
    cell['nonfinite_elements'] += bad
    cell['_digest'].update(np.ascontiguousarray(values).tobytes())
    if bad:
        raise ValueError('nonfinite literal/fixed values forbid cohort admission')


def _fixed_emg(raw, recipe, norm):
    if recipe['layout']['input_channels'] == 16:
        values = np.zeros((len(raw),16), dtype=np.float32)
        values[:,recipe['layout']['channel_indices']] = raw
    else:
        values = raw
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return (values-norm['mean'])/(norm['std']+norm['epsilon'])


def _scan(store, spans, recipe, norm, guard, chunk_bytes):
    raw = {name:{role:_cell() for role in ROLES} for name in data._schema(store)}
    fixed = {hand:{role:{kind:_cell() for kind in ('emg','angles')} for role in ROLES} for hand in HANDS}
    supervision = {role:dict(left=0,right=0,generated_supervised_hand_rows=0) for role in ROLES}
    rows_per_chunk = max(1, chunk_bytes//(22*4))
    for role in ROLES:
        for start,end in spans[role]:
            for offset in range(start,end,rows_per_chunk):
                stop = min(end,offset+rows_per_chunk)
                with guard.read():
                    values = {name:np.asarray(store.field(name)[offset:stop]) for name in raw}
                    for name,array in values.items():
                        _consume(raw[name][role],array)
                    generated = values['generated_label_valid']
                    for h,hand in enumerate(HANDS):
                        count = int(np.count_nonzero(generated[:,h]))
                        supervision[role][hand] += count
                        supervision[role]['generated_supervised_hand_rows'] += count
                        emg = _fixed_emg(values[f'emg_{hand}_filtered'], recipe, norm)
                        _consume(fixed[hand][role]['emg'], emg)
                        del emg
                        angles = np.concatenate([values[f'generated_joint_angles_{hand}'],
                            np.deg2rad(values[f'mocap_{hand}_wrist_pitch'])[:,None],
                            np.deg2rad(values[f'mocap_{hand}_wrist_yaw'])[:,None]], axis=1)
                        _consume(fixed[hand][role]['angles'], angles)
                        del angles
        if supervision[role]['generated_supervised_hand_rows'] == 0:
            raise ValueError('every required role needs actual generated supervision')
    def finish(value):
        if isinstance(value,dict):
            if '_digest' in value:
                value['sha256'] = value.pop('_digest').hexdigest()
            for child in value.values():
                finish(child)
    finish(raw);finish(fixed)
    return raw,fixed,supervision


def _supplement(path, partition, raw):
    if path is None:
        return None
    record = _json(path)
    if (record.get('probe_sha256') != _PROBE_SHA or record.get('partition_canonical_sha256') !=
            partition['provenance']['partition_sha256'] or record.get('settings') != partition['settings'] or
            record.get('coverage') != partition['coverage'] or record.get('source_stat_unchanged') is not True):
        raise ValueError('supplemental finite probe is not the matching frozen evidence')
    floats = {name for name in raw if name.startswith(('emg_','generated_joint_angles_')) or
              name.endswith(('_wrist_pitch','_wrist_yaw'))}
    if set(record.get('field_counts',{})) != floats:
        raise ValueError('supplemental finite probe lacks full floating fields')
    for name in floats:
        if set(record['field_counts'][name]) != set(ROLES):
            raise ValueError('supplemental finite probe lacks all roles')
        for role in ROLES:
            actual=raw[name][role];claimed=record['field_counts'][name][role]
            if any(claimed.get(k)!=v for k,v in dict(unique_rows=actual['unique_rows'],
                consumed_elements=actual['elements'],nonfinite_elements=0,rows_with_nonfinite=0).items()):
                raise ValueError('supplemental finite probe counts differ from fresh scan')
    return dict(file_sha256=file_sha256(_safe(path)), script_sha256=_PROBE_SHA,
                authority='supplement only; identity and all values independently recomputed')


def _audit(root, partitions, *, upstream, config, finite_probe=None, chunk_bytes=MAX_CHUNK_BYTES):
    if type(chunk_bytes) is not int or not 88 <= chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError('chunk byte bound must be between 88 and 64 MiB')
    store,partition,recipe,norm,identity,spans,footprints,guard = _prepare(root,partitions,upstream,config,finite_probe)
    raw,fixed,supervision = _scan(store,spans,recipe,norm,guard,chunk_bytes)
    supplement = _supplement(finite_probe,partition,raw)
    guard.check()
    record = dict(format=_FORMAT,complete=True,cpu_cohort_admitted=True,production_ready=False,
        cuda_qualified=False,gpu_capacity_qualified=False,
        scope='all frozen partition windows of bound legacy source; CPU raw and fixed numeric admission only',
        data_identity=identity,recipe=recipe,footprints=footprints,raw_fields=raw,fixed_values=fixed,
        supervision=supervision,supplement=supplement,
        admission_source_sha256=file_sha256(Path(__file__)),chunk_byte_limit=MAX_CHUNK_BYTES,
        transform_scope='fixed float32 normalization and sparse zero-fill; rotation is permutation; random transforms require runtime per-batch admission',
        empty_label_policy='all frozen windows retained; individual empty-label windows are not dropped',
        remaining_requirements=['actual CUDA numeric qualification','actual 600/256 train and validation capacity',
                                'final formal source/runtime admission','runtime per-batch finite admission'])
    record['sha256'] = _hash(record)
    guard.check()
    return record,guard


def _verified(root, partitions, *, upstream, config, receipt, finite_probe=None, chunk_bytes=MAX_CHUNK_BYTES):
    receipt_guard = None
    if isinstance(receipt,(str,os.PathLike)):
        receipt_guard = _StatGuard([receipt])
        expected = _json(receipt)
    else:
        expected = json.loads(json.dumps(receipt,allow_nan=False))
    if not isinstance(expected,dict) or expected.get('format') != _FORMAT:
        raise ValueError('not a whole-cohort CPU admission receipt')
    actual,guard = _audit(root,partitions,upstream=upstream,config=config,finite_probe=finite_probe,chunk_bytes=chunk_bytes)
    if receipt_guard is not None:
        receipt_guard.check()
        guard.paths.update(receipt_guard.paths)
        guard.parents.update(receipt_guard.parents)
    if _hash(expected) != _hash(actual):
        raise ValueError('cohort receipt differs from recomputed identity/counts/admission')
    guard.check()  # comparisons may run arbitrary caller-observed IO hooks
    return actual,guard


def verify_cohort_receipt(root, partitions, *, upstream, config, receipt, finite_probe=None, chunk_bytes=MAX_CHUNK_BYTES):
    """Recompute actual hashes, full footprints, fixed values and supervision."""
    return _verified(root,partitions,upstream=upstream,config=config,receipt=receipt,
                     finite_probe=finite_probe,chunk_bytes=chunk_bytes)[0]


def cohort_stat_guard(root, partitions, *, upstream, config, receipt, finite_probe=None, chunk_bytes=MAX_CHUNK_BYTES):
    """Full verification once; return its SAME process-local guard for reads."""
    return _verified(root,partitions,upstream=upstream,config=config,receipt=receipt,
                     finite_probe=finite_probe,chunk_bytes=chunk_bytes)[1]


def _fsync_dir(path):
    fd = os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _Transaction:
    def __init__(self,root,budget,reserve):
        self.root,self.budget,self.reserve = _safe(root),budget,reserve
        self.owned = {}
        self.payloads = {}
        value=self.root.lstat();self.root_id=(value.st_dev,value.st_ino)
        self.dir_fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        opened=os.fstat(self.dir_fd)
        if (opened.st_dev,opened.st_ino)!=self.root_id:
            os.close(self.dir_fd)
            raise ValueError('owned output directory changed before capability acquisition')
        self._finalizer=weakref.finalize(self,os.close,self.dir_fd)

    def close(self):
        self._finalizer()

    def _directory_bound(self):
        _safe(self.root)
        value=self.root.lstat()
        opened=os.fstat(self.dir_fd)
        if ((value.st_dev,value.st_ino)!=self.root_id or
                (opened.st_dev,opened.st_ino)!=self.root_id):
            raise ValueError('owned output directory changed')

    def check(self,extra):
        self._directory_bound()
        usage=sum(p.lstat().st_size for p in self.root.iterdir() if p.is_file())
        if usage+extra>self.budget or shutil.disk_usage(self.root).free-extra<self.reserve:
            raise ValueError('cohort output capacity/reserve would be exceeded')

    def revoke(self,path):
        owned=self.owned.get(path)
        if not owned:
            return
        if path.parent!=self.root:
            raise ValueError('success artifact is outside the owned output directory')
        self._directory_bound()
        try:
            observed=path.lstat()
        except FileNotFoundError:
            return
        # Canonical-name checks fail closed on a rebind. The final ownership
        # check, unlink and fsync use only the ORIGINAL opened directory, so a
        # later rebind cannot redirect cleanup into a neighbor's namespace.
        self._directory_bound()
        if not stat.S_ISREG(observed.st_mode) or (observed.st_dev,observed.st_ino)!=owned:
            return
        try:
            value=os.stat(path.name,dir_fd=self.dir_fd,follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISREG(value.st_mode) and (value.st_dev,value.st_ino)==owned:
            os.unlink(path.name,dir_fd=self.dir_fd)
            os.fsync(self.dir_fd)

    def verify_published(self,path):
        self.check(0)
        before=_stamp(_safe(path))
        if (before[:2]!=self.owned.get(path) or file_sha256(path)!=self.payloads.get(path) or
                _stamp(path)!=before):
            raise ValueError('published cohort receipt changed after commit')


def _atomic_json(path,value,transaction):
    payload=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(payload)>4*1024**2:
        raise ValueError('cohort receipt exceeds small bounded output')
    transaction.check(len(payload))
    temporary=path.with_name(path.name+'.part-'+uuid.uuid4().hex)
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        for offset in range(0,len(payload),65536):
            part=payload[offset:offset+65536];transaction.check(len(part))
            if stream.write(part)!=len(part):
                raise OSError('short cohort receipt write')
        stream.flush();os.fsync(stream.fileno())
        value=os.fstat(stream.fileno());identity=(value.st_dev,value.st_ino)
    transaction.check(0)
    os.link(temporary,path,follow_symlinks=False)  # atomic NOREPLACE
    transaction.owned[path]=identity
    transaction.payloads[path]=hashlib.sha256(payload).hexdigest()
    temporary.unlink()
    _fsync_dir(path.parent)


def admit_classic_cohort(root, partitions, *, upstream, config, finite_probe=None, output,
                         chunk_bytes=MAX_CHUNK_BYTES,budget_bytes=8*1024**2,reserve_bytes=RESERVE_BYTES):
    """Publish one small success receipt LAST; it never authorizes production."""
    if type(chunk_bytes) is not int or not 88 <= chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError('chunk byte bound must be between 88 and 64 MiB')
    if type(budget_bytes) is not int or budget_bytes<=0 or type(reserve_bytes) is not int or reserve_bytes<RESERVE_BYTES:
        raise ValueError('invalid cohort capacity/reserve budget')
    root,upstream=_safe(root),_safe(upstream)
    destination=_safe(output,exists=False)
    if destination.exists():
        raise FileExistsError('cohort output already exists')
    if not destination.parent.is_dir():
        raise ValueError('cohort output parent must already exist')
    if shutil.disk_usage(destination.parent).free-budget_bytes<reserve_bytes:
        raise ValueError('cohort output capacity/reserve admission failed')
    destination.mkdir()
    transaction=_Transaction(destination,budget_bytes,reserve_bytes)
    try:
        record,guard=_audit(root,partitions,upstream=upstream,config=config,finite_probe=finite_probe,chunk_bytes=chunk_bytes)
        guard.check()
        _atomic_json(destination/'receipt.json',record,transaction)
        guard.check()
        transaction.verify_published(destination/'receipt.json')
        guard.check()  # the last operation also covers published-file verification
        return record
    except BaseException as exc:
        try:
            transaction.revoke(destination/'receipt.json')
        finally:
            try:
                _atomic_json(destination/'failure.json',dict(format=_FORMAT,complete=False,
                    cpu_cohort_admitted=False,production_ready=False,error_class=type(exc).__name__),transaction)
            except (OSError,ValueError):
                pass  # no success remains; never consume the protected reserve
        raise
    finally:
        transaction.close()
