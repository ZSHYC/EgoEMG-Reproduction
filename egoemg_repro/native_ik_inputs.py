"""Read-only full-pose admission and the original six logical IK shards.

No optimizer, geometry model, CUDA or label output is constructed here.
Engineering inputs never gain a full-release identity.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import weakref

import numpy as np

from .classic_emg_cohort import _safe
from .data import file_sha256

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'f7cbd048b7092139852270491d3859cd472e0311'
REFERENCE_SHA = '757eef807837c8c44a6be6cbe0b237473f30953a543477dfc13fc32543c96b6a'
SOURCE_SHA = {
    'scripts/ik/batch_ik_mesh.py': '794fdfcb82a692812de4ca7a15fd4e41c0c3f3717424dd917fb05e4f9508da2c',
    'egoemg/UmeTrack/dataset/generic_hand_model.json': '2eb8029a8e1cedc2cf5b40c14c0955e2004277024df3c6304ce0bb74106a7c31',
}
HANDS = ('right', 'left')
IMPORTED_SHA = file_sha256(__file__)
_ADMITTED = weakref.WeakKeyDictionary()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _nine(path):
    try:
        s = path.lstat()
    except OSError as exc:
        raise ValueError('IK input/source changed or missing') from exc
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('IK input/source must be a regular nonsymlink file')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
            s.st_mode, s.st_uid, s.st_gid, s.st_nlink)


class _Guard:
    def __init__(self, paths, absent):
        self.stamps = {_safe(p): _nine(_safe(p)) for p in paths}
        self.parents = {q for p in self.stamps for q in p.parents}
        self.absent = tuple(absent)
        self.check()

    def check(self):
        if any(p.is_symlink() for p in self.parents):
            raise ValueError('IK input/source ancestor changed')
        for p in self.absent:
            if p.exists() or p.is_symlink():
                raise ValueError('unexpected valid.npy changes original IK frame selection')
        for p, stamp in self.stamps.items():
            if _nine(p) != stamp:
                raise ValueError('IK input/source generation changed')


def _state(inputs):
    if type(inputs) is not NativeIKInputs or inputs not in _ADMITTED:
        raise ValueError('registered native IK input object required')
    return _ADMITTED[inputs]


@dataclass(frozen=True, eq=False)
class NativeIKInputs:
    """Process-local admission; public metadata and consumed rows are copies."""

    @property
    def identity(self):
        self.check()
        return copy.deepcopy(_state(self)['identity'])

    @property
    def shards(self):
        self.check()
        return _state(self)['shards']

    def check(self):
        s = _state(self)
        s['guard'].check()
        if _hash(s['identity']) != s['identity_hash']:
            raise ValueError('native IK admitted identity changed')
        for hand, a in s['arrays'].items():
            if (type(a) is not np.memmap or a.dtype != np.dtype('float32')
                or a.shape != (s['n'], 48) or a.strides != (192, 4)
                or a.mode != 'r' or a.flags.writeable or a.offset != 0
                or Path(a.filename) != s['paths'][hand]):
                raise ValueError('native IK read-only pose binding changed')
        s['guard'].check()

    def batches(self, hand, shard):
        self.check()
        if hand not in HANDS or type(shard) is not int or not 0 <= shard < 6:
            raise ValueError('explicit right/left hand and original shard0..5 required')
        s = _state(self)
        start, end = s['shards'][shard]
        for first in range(start, end, s['batch_size']):
            self.check()
            last = min(first + s['batch_size'], end)
            indices = np.arange(first, last, dtype=np.int64)
            raw = np.array(s['arrays'][hand][first:last], dtype=np.float32, copy=True, order='C')
            canonical = raw.copy()
            canonical[:, :3] = 0.0
            self.check()
            try:
                yield indices, raw, canonical
            finally:
                self.check()


def admit_native_ik_inputs(data, geometry, upstream, *, engineering=False,
                           batch_size=10096, chunk_rows=262144, reference=None):
    """Rehash and scan all poses; preserve default original batch membership.

    Six logical shards are independent of the number of physical GPUs used
    later. An optional valid.npy is rejected for this admitted released source,
    which has no such file; marker or label masks must not replace that policy.
    """
    if type(engineering) is not bool:
        raise ValueError('explicit engineering boolean required')
    if any(type(v) is not int or v < 1 for v in (batch_size, chunk_rows)):
        raise ValueError('positive integer IK batch/chunk size required')
    if chunk_rows > 262144:
        raise ValueError('IK numeric chunks must be bounded by262144 rows')
    if not engineering and batch_size != 10096:
        raise ValueError('full registered IK source requires original batch10096')
    data, geometry, upstream = [_safe(p) for p in (data, geometry, upstream)]
    if engineering and reference is not None:
        raise ValueError('engineering inputs cannot supply a full reference')
    ref_path = None
    ref = None
    if not engineering:
        ref_path = _safe(reference or ROOT / 'results/legacy-geometry-complete-numeric-admission-20261101.json')
        before = _nine(ref_path)
        if file_sha256(ref_path) != REFERENCE_SHA:
            raise ValueError('exact published complete geometry reference required')
        ref = json.loads(ref_path.read_text())
        if _nine(ref_path) != before:
            raise ValueError('complete geometry reference changed during reading')
    source_paths = {f'upstream/{name}': _safe(upstream / name) for name in SOURCE_SHA}
    for name, expected in SOURCE_SHA.items():
        if file_sha256(source_paths[f'upstream/{name}']) != expected:
            raise ValueError('original pinned IK source differs')
    if not engineering:
        if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM
            or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()):
            raise ValueError('full registered IK source requires clean pinned upstream')
        source_paths.update({'upstream/' + p.relative_to(upstream).as_posix(): _safe(p)
                             for p in upstream.glob('egoemg/**/*.py')})
    for name in ('native_ik_inputs.py', 'classic_emg_cohort.py', 'classic_emg.py',
                 'classic_emg_data.py', 'data.py', 'partitions.py', '__init__.py'):
        source_paths['egoemg_repro/' + name] = _safe(ROOT / 'egoemg_repro' / name)
    if file_sha256(__file__) != IMPORTED_SHA:
        raise ValueError('imported native IK input source changed')
    manifest_path, metadata_path = [_safe(data / n) for n in ('manifest.json', 'metadata.npz')]
    initial = _Guard([manifest_path, metadata_path, *source_paths.values(), *([ref_path] if ref_path else [])],
                     [data / 'valid.npy', geometry / 'valid.npy'])
    manifest = json.loads(manifest_path.read_text())
    n = manifest.get('total_rows')
    if type(n) is not int or n < 6:
        raise ValueError('at least6 integer rows required for original IK shards')
    if not engineering and n != 66161725:
        raise ValueError('full registered released IK row count differs')
    fields = manifest.get('fields')
    if type(fields) is not dict:
        raise ValueError('IK manifest fields required')
    paths = {}
    for hand in HANDS:
        name = f'generated_mano_{hand}_pose'
        expected = dict(filename=name + '.dat', dtype='float32', shape=[n, 48])
        field = fields.get(name)
        if (type(field) is not dict or field != expected
            or any(type(v) is not int for v in field.get('shape', []))):
            raise ValueError('exact original float32[N,48] IK pose schema required')
        paths[hand] = _safe(geometry / expected['filename'])
        if _nine(paths[hand])[2] != n * 48 * 4:
            raise ValueError('exact physical IK pose file size required')
    initial.check()
    guard = _Guard([*initial.stamps, *paths.values()], initial.absent)
    initial.check()
    arrays = {h: np.memmap(p, np.float32, mode='r', shape=(n, 48)) for h, p in paths.items()}
    hashes = {}
    named_paths = dict(source_paths, **{'data/manifest.json': manifest_path,
                                       'data/metadata.npz': metadata_path})
    named_paths.update({f'geometry/generated_mano_{h}_pose.dat': p for h, p in paths.items()})
    for name, p in named_paths.items():
        guard.check()
        hashes[name] = file_sha256(p)
        guard.check()
    initial.check()
    for name, expected in SOURCE_SHA.items():
        if hashes['upstream/' + name] != expected:
            raise ValueError('original pinned IK source changed before admission guard')
    if hashes['egoemg_repro/native_ik_inputs.py'] != IMPORTED_SHA:
        raise ValueError('imported native IK input source changed before admission guard')
    if ref is not None:
        if hashes['data/manifest.json'] != ref['manifest_sha256'] or hashes['data/metadata.npz'] != ref['metadata_sha256']:
            raise ValueError('full registered IK manifest/metadata differs')
        bulk = {v['field']: v for v in ref['bulk_files']}
        for hand in HANDS:
            name = f'generated_mano_{hand}_pose'
            if hashes['geometry/' + name + '.dat'] != bulk[name]['actual_full_sha256']:
                raise ValueError('full registered IK complete pose bytes differ')
    scan = {}
    for hand, a in arrays.items():
        zero_rows = canonical_zero_rows = 0
        for first in range(0, n, chunk_rows):
            guard.check()
            values = a[first:min(n, first + chunk_rows)]
            if not np.isfinite(values).all():
                raise ValueError('all raw IK pose elements must be finite')
            zero_rows += int(np.count_nonzero(~np.any(values != 0, axis=1)))
            canonical_zero_rows += int(np.count_nonzero(~np.any(values[:, 3:] != 0, axis=1)))
            guard.check()
        scan[hand] = dict(rows=n, elements=n * 48, nonfinite_elements=0,
                          zero_rows=zero_rows, canonical_zero_rows=canonical_zero_rows)
        if ref is not None and zero_rows != ref['full_numeric_scan'][f'generated_mano_{hand}_pose']['zero_rows']:
            raise ValueError('full registered IK numeric coverage differs')
    initial.check()
    if not engineering and subprocess.check_output(
            ['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('full registered IK upstream changed during admission')
    guard.check()
    width = n // 6
    shards = tuple((k * width, (k + 1) * width if k < 5 else n) for k in range(6))
    identity = dict(format='original_native_ik_complete_pose_inputs_v1',
        upstream_revision=UPSTREAM, engineering=engineering, full_available_release=not engineering,
        total_rows=n, samples_per_hand=n, hands_order=list(HANDS), logical_worker_partitions=6,
        shards=[list(v) for v in shards], batch_size=batch_size, subsample_step=1,
        batches_per_hand=[math.ceil((b - a) / batch_size) for a, b in shards],
        optional_global_valid_npy='absent in both admitted source roots',
        zero_poses_retained=True, marker_and_label_masks_used=False,
        canonicalization='copy original float32[48], zero only first3 values',
        native_policy=dict(right_mano_for_both_hands=True, flat_hand_mean=True, beta='native internal zero',
                           additional_left_pose_reflection=False, finger_angles=20, regenerated_wrist_angles=False),
        complete_geometry_reference_sha256=REFERENCE_SHA if ref is not None else None,
        files_sha256=hashes, scan=scan, scientific_eligible=False, full_frame_labels_regenerated=False,
        full_paper_complete=False)
    inputs = NativeIKInputs()
    _ADMITTED[inputs] = dict(guard=guard, paths=paths, arrays=arrays, n=n, shards=shards,
                             batch_size=batch_size, identity=identity, identity_hash=_hash(identity))
    inputs.check()
    return inputs
