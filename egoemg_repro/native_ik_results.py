"""Original IK batch packets, independent readback, and isolated label shards.

Inspection and assembly import no Torch. A structural packet check does not
prove optimizer execution: production requires the registered native runtime;
acceptance additionally requires fresh geometry and actual producer/reader exit.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import weakref

import numpy as np

from .classic_emg_cohort import _safe
from .data import file_sha256
from .emg2pose_memory_process import process_identity, same_process
from .native_ik_inputs import _state, _hash, _Guard, HANDS

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = file_sha256(__file__)
_BATCHES = weakref.WeakKeyDictionary()
_PACKETS = weakref.WeakKeyDictionary()
_ACTOR_KEYS = ('pid', 'start_ticks', 'uid', 'sid', 'pgrp')


def seal(value):
    return dict(value, sha256=_hash(value))


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def _read_json(path, guard):
    guard.check()
    value = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite IK JSON')))
    guard.check()
    if type(value) is not dict or value.get('sha256') != _hash({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('IK result seal differs')
    return value


def require_closed(actor):
    if (type(actor) is not dict or any(k not in actor for k in _ACTOR_KEYS)
        or any(type(actor[k]) is not int or actor[k] < 0 for k in ('pid', 'uid', 'sid', 'pgrp'))
        or actor['pid'] < 1 or type(actor['start_ticks']) is not str or not actor['start_ticks'].isdigit()):
        raise ValueError('actual original IK actor identity required')
    if same_process(actor):
        raise ValueError('original IK actor must actually exit before acceptance')


def _new_output(inputs, output):
    inputs.check()
    output = _safe(output, exists=False)
    s = _state(inputs)
    forbidden = {p.parent for p in s['paths'].values()}
    for p in s['guard'].stamps:
        if p.name == 'manifest.json': forbidden.add(p.parent)
        if p.as_posix().endswith('/scripts/ik/batch_ik_mesh.py'): forbidden.add(p.parents[2])
    if output.exists() or any(output.is_relative_to(p) for p in forbidden):
        raise ValueError('new isolated IK output required; source directories cannot be written')
    return output


def _array_sha(a):
    import hashlib
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def _batch_state(batch):
    if type(batch) is not IKBatch or batch not in _BATCHES:
        raise ValueError('registered original IK batch required')
    return _BATCHES[batch]


@dataclass(frozen=True, eq=False)
class IKBatch:
    def check(self):
        s = _batch_state(self); s['inputs'].check()
        if file_sha256(__file__) != IMPORTED_SHA:
            raise ValueError('imported IK results source changed')
        if _hash(s['identity']) != s['identity_hash'] or any(
            a.flags.writeable or _array_sha(a) != s['identity']['arrays_sha256'][n]
            for n, a in s['arrays'].items()):
            raise ValueError('original IK batch binding changed')

    @property
    def identity(self):
        self.check(); return copy.deepcopy(_BATCHES[self]['identity'])

    @property
    def arrays(self):
        self.check(); return {n: a.copy() for n, a in _BATCHES[self]['arrays'].items()}


def select_ik_batch(inputs, hand, shard, batch_index):
    """O(1) selection of exactly one original contiguous logical-worker batch."""
    inputs.check()
    s = _state(inputs)
    if hand not in HANDS or type(shard) is not int or not 0 <= shard < 6:
        raise ValueError('explicit right/left hand and original shard0..5 required')
    if (type(batch_index) is not int or batch_index < 0
        or batch_index >= s['identity']['batches_per_hand'][shard]):
        raise ValueError('original in-range integer IK batch index required')
    first = s['shards'][shard][0] + batch_index * s['batch_size']
    last = min(first + s['batch_size'], s['shards'][shard][1])
    raw = np.array(s['arrays'][hand][first:last], dtype=np.float32, copy=True, order='C')
    canonical = raw.copy(); canonical[:, :3] = 0.
    arrays = dict(indices=np.arange(first, last, dtype=np.int64), raw_poses=raw, canonical_poses=canonical)
    for a in arrays.values(): a.flags.writeable = False
    inputs.check()
    identity = dict(format='original_native_ik_selected_batch_v1', input_identity_sha256=_hash(inputs.identity),
        hand=hand, shard=shard, batch_index=batch_index, start=first, end=last, rows=last-first,
        logical_batch_size=s['batch_size'], engineering=s['identity']['engineering'],
        full_available_release=s['identity']['full_available_release'],
        original_default_batch_size=10096, arrays_sha256={n: _array_sha(a) for n, a in arrays.items()})
    batch = IKBatch()
    _BATCHES[batch] = dict(inputs=inputs, arrays=arrays, identity=identity, identity_hash=_hash(identity))
    batch.check(); return batch


def _geometry_source(batch):
    s = _state(_batch_state(batch)['inputs'])
    source = next(p for p in s['guard'].stamps if p.as_posix().endswith('/scripts/ik/batch_ik_mesh.py'))
    model = next(p for p in s['guard'].stamps if p.as_posix().endswith('/egoemg/UmeTrack/dataset/generic_hand_model.json'))
    tree = ast.parse(source.read_text())
    value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'MANO_TO_UMETRACK' for t in n.targets))
    mano = sorted(value)
    limits = np.asarray(json.loads(model.read_text())['joint_limits'][:20], np.float32)
    batch.check()
    return mano, [value[n] for n in mano], limits


def validate_ik_state(batch, arrays, record):
    """Typed/source/trace inspection only, never a native-execution certificate."""
    batch.check(); b = batch.identity['rows']
    spec = dict(indices=(b,), raw_poses=(b, 48), canonical_poses=(b, 48), angles=(b, 20),
        raw_angles=(b, 20), global_orient=(b, 3), trans_raw=(b, 3), residual=(b, 20, 3),
        predicted_landmarks=(b, 21, 3), aligned_mano_joints=(b, 21, 3))
    if type(arrays) is not dict or set(arrays) != set(spec):
        raise ValueError('complete exact original IK raw-state arrays required')
    for name, shape in spec.items():
        a = arrays[name]; dtype = np.dtype('int64' if name == 'indices' else 'float32')
        if (type(a) is not np.ndarray or a.shape != shape or a.dtype != dtype
            or not a.flags.c_contiguous or not np.isfinite(a).all()):
            raise ValueError('finite exact original IK typed state required: ' + name)
    for name, expected in batch.arrays.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError('original IK raw/canonical/index witness differs')
    mano, umetrack, limits = _geometry_source(batch)
    if not ((arrays['angles'] >= limits[:, 0]).all() and (arrays['angles'] <= limits[:, 1]).all()):
        raise ValueError('original IK finger angles outside original limits')
    expected_residual = arrays['predicted_landmarks'][:, umetrack] - arrays['aligned_mano_joints'][:, mano]
    if not np.array_equal(arrays['residual'], expected_residual):
        raise ValueError('original IK raw geometric residual binding differs')
    steps = record.get('actual_ik_optimizer_steps')
    closures = record.get('actual_ik_closure_evaluations')
    trace = record.get('ik_step_trace')
    if (type(steps) is not int or not 1 <= steps <= 200 or record.get('actual_ik_outer_steps') != steps
        or type(closures) is not int or closures < steps or type(trace) is not list or len(trace) != steps
        or record.get('actual_original_step_return_preserved') is not True
        or record.get('actual_rebuild_optimizer_steps') != 0):
        raise ValueError('original IK actual optimizer-step trace differs')
    previous = 0
    for i, row in enumerate(trace):
        count = row.get('closure_calls_so_far')
        loss = row.get('returned_pre_step_loss')
        if (row.get('outer_step') != i + 1 or type(count) is not int or not previous < count <= closures
            or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0):
            raise ValueError('original IK closure trace differs')
        previous = count
    objective = record.get('final_objective_mm2')
    if (previous != closures or type(objective) not in (int, float) or not math.isfinite(objective)
        or objective < 0 or record.get('native_final_loss') != objective
        or type(record.get('seconds')) not in (int, float) or not math.isfinite(record['seconds']) or record['seconds'] < 0):
        raise ValueError('original IK final objective/time differs')
    # Only a reduction-rounding check here. Fresh Torch geometry requires exact
    # native FP32 objective equality, not this model-free FP64 diagnostic bound.
    distance2 = np.square(arrays['residual'].astype(np.float64)).sum(-1)
    if not np.isclose(distance2.mean(), objective, rtol=2e-6, atol=1e-9):
        raise ValueError('original IK final objective inconsistent with saved residuals')
    distance = np.sqrt(distance2)
    batch.check()
    return dict(rows=b, raw_zero_pose_rows=int(np.count_nonzero(~np.any(arrays['raw_poses'] != 0, axis=1))),
        canonical_zero_pose_rows=int(np.count_nonzero(~np.any(arrays['canonical_poses'] != 0, axis=1))),
        mean_landmark_distance_mm=float(distance.mean()), sum_landmark_distance_mm=float(distance.sum()),
        sum_squared_landmark_distance_mm2=float(distance2.sum()), landmark_comparisons=int(distance.size),
        geometric_reconstruction_verified=False, native_optimizer_execution_verified=False)


def _runtime(runtime):
    from .native_ik_runtime import IKRuntime
    if type(runtime) is not IKRuntime:
        raise ValueError('registered original IK runtime required')
    runtime.check()
    return runtime.identity


def produce_ik_batch(batch, runtime, output, *, guard=lambda: None):
    identity = _runtime(runtime); batch.check(); guard()
    inputs = _batch_state(batch)['inputs']
    output = _new_output(inputs, output)
    def checked():
        batch.check(); runtime.check(); guard()
    output.mkdir(parents=True, mode=0o700)
    optimized = runtime.optimize(batch.arrays['canonical_poses'], guard=checked)
    arrays = {**batch.arrays, **{n: optimized[n] for n in ('angles', 'raw_angles', 'global_orient', 'trans_raw')},
              **optimized['arrays']}
    record = {k: v for k, v in optimized.items() if k not in ('angles', 'raw_angles', 'global_orient', 'trans_raw', 'arrays')}
    statistics = validate_ik_state(batch, arrays, record)
    if record['source_identity_sha256'] != identity['source_identity_sha256']:
        raise ValueError('original IK runtime/source identity differs')
    checked()
    with (output / 'state.npz').open('xb') as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    state_guard = _Guard([output / 'state.npz'], [])
    state_sha = file_sha256(output / 'state.npz'); state_guard.check(); checked()
    result = seal(dict(format='original_native_ik_raw_batch_v1', batch=batch.identity,
        runtime_identity=identity, actor=process_identity(), record=record, statistics=statistics,
        state_file='state.npz', state_file_sha256=state_sha, results_source_sha256=IMPORTED_SHA,
        pending_independent_geometry_verification=True, full_frame_labels_regenerated=False,
        original_default_batch10096_qualified=False, scientific_eligible=False, full_paper_complete=False))
    write_json(output / 'result.json', result)
    state_guard.check(); checked()
    return result


@dataclass(frozen=True, eq=False)
class IKPacket:
    def check(self, *, rehash=False):
        if type(self) is not IKPacket or self not in _PACKETS:
            raise ValueError('registered source-bound IK packet required')
        s = _PACKETS[self]; s['batch'].check(); s['guard'].check()
        if rehash:
            for path, sha in s['hashes'].items():
                if file_sha256(path) != sha:
                    raise ValueError('original IK packet content changed')
            s['guard'].check()

    @property
    def metadata(self):
        self.check(); return copy.deepcopy(_PACKETS[self]['metadata'])

    @property
    def arrays(self):
        self.check(rehash=True)
        s = _PACKETS[self]
        with np.load(s['directory'] / 'state.npz', allow_pickle=False) as saved:
            arrays = {n: np.ascontiguousarray(saved[n]) for n in saved.files}
        self.check(rehash=True)
        validate_ik_state(s['batch'], arrays, s['metadata']['record'])
        return arrays


def inspect_ik_batch(inputs, directory, *, require_producer_closed=True):
    inputs.check(); directory = _safe(directory)
    paths = [directory / 'result.json', directory / 'state.npz']
    if not all(p.is_file() for p in paths):
        raise ValueError('complete original IK batch result and raw state required')
    guard = _Guard(paths, [])  # Register BEFORE the first JSON/hash/NPZ read.
    metadata = _read_json(paths[0], guard)
    if (metadata.get('format') != 'original_native_ik_raw_batch_v1'
        or metadata.get('state_file') != 'state.npz' or metadata.get('results_source_sha256') != IMPORTED_SHA
        or metadata.get('pending_independent_geometry_verification') is not True
        or any(metadata.get(n) is not False for n in ('full_frame_labels_regenerated',
            'original_default_batch10096_qualified', 'scientific_eligible', 'full_paper_complete'))):
        raise ValueError('original pending IK batch/source policy differs')
    if require_producer_closed: require_closed(metadata.get('actor'))
    selected = metadata.get('batch', {})
    batch = select_ik_batch(inputs, selected.get('hand'), selected.get('shard'), selected.get('batch_index'))
    if selected != batch.identity:
        raise ValueError('original IK input identity or exact batch membership differs')
    hashes = {p: file_sha256(p) for p in paths}; guard.check()
    if hashes[paths[1]] != metadata.get('state_file_sha256'):
        raise ValueError('original IK raw-state digest differs')
    packet = IKPacket()
    _PACKETS[packet] = dict(batch=batch, guard=guard, hashes=hashes, directory=directory, metadata=metadata)
    arrays = packet.arrays
    if validate_ik_state(batch, arrays, metadata['record']) != metadata['statistics']:
        raise ValueError('original IK raw-state statistics differ')
    packet.check(rehash=True); return packet


def verify_ik_batch(inputs, runtime, directory, output, *, guard=lambda: None):
    identity = _runtime(runtime)
    packet = inspect_ik_batch(inputs, directory)
    metadata = packet.metadata
    if metadata['runtime_identity'] != identity:
        raise ValueError('fresh native IK geometry runtime/source/backend differs')
    output = _new_output(inputs, output)
    output.mkdir(parents=True, mode=0o700)
    arrays = packet.arrays
    import torch
    original_init, original_step = torch.optim.Optimizer.__init__, torch.optim.LBFGS.step
    def denied(*a, **kw): raise RuntimeError('fresh native IK geometry reader cannot optimize')
    torch.optim.Optimizer.__init__ = denied; torch.optim.LBFGS.step = denied
    try:
        packet.check(rehash=True); guard()
        rebuilt = runtime.rebuild(arrays['canonical_poses'], *(arrays[n] for n in ('angles', 'raw_angles', 'global_orient', 'trans_raw')))
        guard(); packet.check(rehash=True)
    finally:
        torch.optim.Optimizer.__init__ = original_init; torch.optim.LBFGS.step = original_step
    record = metadata['record']
    if (rebuilt['final_objective_mm2'] != record['native_final_loss']
        or rebuilt['model_state_sha256'] != record['model_state_sha256']
        or rebuilt['source_identity_sha256'] != record['source_identity_sha256']
        or rebuilt['actual_rebuild_optimizer_steps'] != 0):
        raise ValueError('fresh original IK geometry objective/source/state differs')
    for name, value in rebuilt['arrays'].items():
        if arrays[name].dtype != value.dtype or not np.array_equal(arrays[name], value):
            raise ValueError('fresh original IK geometric arrays differ: ' + name)
    result = seal(dict(format='original_native_ik_batch_geometry_verified_v1', actor=process_identity(),
        producer_actor=metadata['actor'], producer_result_sha256=_PACKETS[packet]['hashes'][_PACKETS[packet]['directory'] / 'result.json'],
        state_file_sha256=metadata['state_file_sha256'], batch=metadata['batch'], runtime_identity=identity,
        all_typed_geometric_arrays_exact=True, original_native_final_objective_exact=True,
        model_state_sha256=record['model_state_sha256'], actual_ik_optimizer_steps=0,
        optimizer_construction_and_lbfgs_step_denied=True, results_source_sha256=IMPORTED_SHA,
        scientific_eligible=False, full_paper_complete=False))
    write_json(output / 'verification.json', result)
    packet.check(rehash=True); inputs.check(); guard()
    return result


def _accepted(inputs, directory, verification):
    packet = inspect_ik_batch(inputs, directory)
    path = _safe(verification) / 'verification.json'
    receipt_guard = _Guard([path], [])
    receipt = _read_json(path, receipt_guard); require_closed(receipt.get('actor'))
    metadata = packet.metadata
    expected = dict(format='original_native_ik_batch_geometry_verified_v1', producer_actor=metadata['actor'],
        producer_result_sha256=_PACKETS[packet]['hashes'][_PACKETS[packet]['directory'] / 'result.json'],
        state_file_sha256=metadata['state_file_sha256'], batch=metadata['batch'],
        runtime_identity=metadata['runtime_identity'], all_typed_geometric_arrays_exact=True,
        original_native_final_objective_exact=True, model_state_sha256=metadata['record']['model_state_sha256'],
        actual_ik_optimizer_steps=0, optimizer_construction_and_lbfgs_step_denied=True,
        results_source_sha256=IMPORTED_SHA, scientific_eligible=False, full_paper_complete=False)
    if any(receipt.get(k) != v for k, v in expected.items()) or receipt['actor'] == metadata['actor']:
        raise ValueError('independent original IK verification binding differs')
    packet.check(rehash=True); receipt_guard.check()
    return packet, receipt_guard, path, file_sha256(path)


def finalize_ik_shard(inputs, hand, shard, batches, output, *, guard=lambda: None):
    """Stream one complete verified shard to a NEW float32[rows,20] label file.

    Each batch item is (producer_directory, fresh_verification_directory).
    This is a component result; it cannot by itself declare full-data success.
    """
    select_ik_batch(inputs, hand, shard, 0)
    expected = inputs.identity['batches_per_hand'][shard]
    if type(batches) not in (list, tuple) or len(batches) != expected:
        raise ValueError('complete original IK shard batch and verification list required')
    output = _new_output(inputs, output)
    # Check every completion/exit before creating a label file; re-open each
    # packet again during streaming, so no accepted array is a persistent cache.
    for index, item in enumerate(batches):
        if type(item) not in (list, tuple) or len(item) != 2:
            raise ValueError('original IK batch/verification pair required')
        packet, rg, _, _ = _accepted(inputs, *item)
        if packet.metadata['batch'] != select_ik_batch(inputs, hand, shard, index).identity:
            raise ValueError('complete original IK shard order or membership differs')
        rg.check(); guard()
    output.mkdir(parents=True, mode=0o700)
    first, last = inputs.shards[shard]
    label_path = output / 'finger_angles.dat'
    with label_path.open('xb') as stream:
        stream.truncate((last-first) * 20 * 4)
    labels = np.memmap(label_path, np.float32, mode='r+', shape=(last-first, 20))
    observed_guards = []
    records = []; totals = dict(rows=0, raw_zero_pose_rows=0, canonical_zero_pose_rows=0,
        sum_landmark_distance_mm=0., sum_squared_landmark_distance_mm2=0., landmark_comparisons=0,
        actual_ik_optimizer_steps=0, actual_ik_closure_evaluations=0)
    for index, item in enumerate(batches):
        packet, rg, path, receipt_sha = _accepted(inputs, *item)
        metadata = packet.metadata; b = metadata['batch']; arrays = packet.arrays
        if b != select_ik_batch(inputs, hand, shard, index).identity:
            raise ValueError('original IK shard membership changed during label copy')
        labels[b['start']-first:b['end']-first] = arrays['angles']
        labels.flush(); packet.check(rehash=True); rg.check(); guard()
        observed_guards.extend((_PACKETS[packet]['guard'], rg))
        for key in totals:
            source = metadata['record'] if key.startswith('actual_ik_') else metadata['statistics']
            totals[key] += source[key]
        records.append(dict(batch=b, result_sha256=_PACKETS[packet]['hashes'][_PACKETS[packet]['directory'] / 'result.json'],
            raw_state_sha256=metadata['state_file_sha256'], verification_sha256=receipt_sha,
            producer_actor=metadata['actor'], verifier_actor=json.loads(path.read_text())['actor']))
    del labels
    if totals['rows'] != last-first:
        raise ValueError('complete original IK shard output count differs')
    label_guard = _Guard([label_path], [])
    label_sha = file_sha256(label_path); label_guard.check(); inputs.check(); guard()
    result = seal(dict(format='original_native_ik_verified_label_shard_v1', actor=process_identity(),
        input_identity_sha256=_hash(inputs.identity), hand=hand, shard=shard, start=first, end=last,
        label_file='finger_angles.dat', label_dtype='float32', label_shape=[last-first, 20], label_sha256=label_sha,
        results_source_sha256=IMPORTED_SHA, batch_size=inputs.identity['batch_size'],
        engineering=inputs.identity['engineering'], batches=records, totals=totals,
        all_batch_geometric_readers_exited=True, source_labels_replaced=False,
        full_frame_labels_regenerated=False, scientific_eligible=False, full_paper_complete=False))
    for observed in observed_guards: observed.check()
    write_json(output / 'shard.json', result)
    label_guard.check(); inputs.check(); guard()
    return result


def assemble_ik_labels(inputs, shards, output, *, guard=lambda: None):
    """Join all twelve CLOSED shard finalizers; never modify released labels."""
    inputs.check()
    if type(shards) not in (list, tuple) or len(shards) != 12:
        raise ValueError('complete twelve label shards: six per hand required')
    output = _new_output(inputs, output)
    admitted = []
    for index, directory in enumerate(shards):
        directory = _safe(directory)
        path, label = directory/'shard.json', directory/'finger_angles.dat'
        frozen = _Guard([path, label], [])
        record = _read_json(path, frozen); require_closed(record.get('actor'))
        hand, shard = HANDS[index//6], index % 6
        start, end = inputs.shards[shard]
        if (record.get('format') != 'original_native_ik_verified_label_shard_v1'
            or record.get('input_identity_sha256') != _hash(inputs.identity)
            or record.get('hand') != hand or record.get('shard') != shard
            or record.get('start') != start or record.get('end') != end
            or record.get('label_file') != label.name or record.get('label_dtype') != 'float32'
            or record.get('label_shape') != [end-start, 20] or label.stat().st_size != (end-start)*80
            or record.get('results_source_sha256') != IMPORTED_SHA
            or record.get('engineering') != inputs.identity['engineering']
            or record.get('batch_size') != inputs.identity['batch_size']
            or record.get('all_batch_geometric_readers_exited') is not True
            or record.get('source_labels_replaced') is not False
            or len(record.get('batches', [])) != inputs.identity['batches_per_hand'][shard]):
            raise ValueError('complete original IK label shard/source binding differs')
        for j, batch in enumerate(record['batches']):
            if batch.get('batch') != select_ik_batch(inputs, hand, shard, j).identity:
                raise ValueError('original IK completed shard batch coverage differs')
            require_closed(batch.get('producer_actor')); require_closed(batch.get('verifier_actor'))
        if file_sha256(label) != record.get('label_sha256'):
            raise ValueError('original IK verified shard label bytes differ')
        frozen.check()
        admitted.append((directory, record, frozen, file_sha256(path)))
    output.mkdir(parents=True, mode=0o700)
    n = inputs.identity['total_rows']; hands = {}
    _, _, limits = _geometry_source(select_ik_batch(inputs, 'right', 0, 0))
    for h, hand in enumerate(HANDS):
        target = output/f'generated_umetrack_{hand}_finger_angles.dat'
        with target.open('xb') as stream:
            stream.truncate(n*80)
        labels = np.memmap(target, np.float32, mode='r+', shape=(n, 20))
        total_rows = 0
        for directory, record, frozen, _ in admitted[h*6:(h+1)*6]:
            frozen.check(); guard(); inputs.check()
            start, end = record['start'], record['end']
            source = np.memmap(directory/'finger_angles.dat', np.float32, mode='r', shape=(end-start, 20))
            for a in range(0, end-start, 262144):
                frozen.check(); guard()
                values = np.array(source[a:min(end-start, a+262144)], copy=True)
                if not np.isfinite(values).all() or not ((values >= limits[:, 0]).all() and (values <= limits[:, 1]).all()):
                    raise ValueError('finite original-limit full IK labels required')
                labels[start+a:start+a+len(values)] = values
                frozen.check()
                total_rows += len(values)
            del source
        labels.flush(); del labels
        if total_rows != n: raise ValueError('complete original IK hand label row count differs')
        hands[hand] = dict(filename=target.name, dtype='float32', shape=[n, 20],
                          file_sha256=file_sha256(target), rows=total_rows)
    for _, _, frozen, _ in admitted: frozen.check()
    inputs.check(); guard()
    result = seal(dict(format='original_native_ik_isolated_bilateral_labels_v1', actor=process_identity(),
        input_identity_sha256=_hash(inputs.identity), results_source_sha256=IMPORTED_SHA,
        hands=hands, rows_per_hand=n, total_rows=2*n, logical_shards_per_hand=6,
        batch_size=inputs.identity['batch_size'], engineering=inputs.identity['engineering'],
        shard_result_files_sha256=[sha for _, _, _, sha in admitted],
        actual_ik_optimizer_steps=sum(r['totals']['actual_ik_optimizer_steps'] for _, r, _, _ in admitted),
        actual_ik_closure_evaluations=sum(r['totals']['actual_ik_closure_evaluations'] for _, r, _, _ in admitted),
        all_shard_finalizers_exited=True, regenerated_finger_angles=20, regenerated_wrist_angles=False,
        source_labels_replaced=False, full_frame_labels_regenerated=False,
        scientific_eligible=False, full_paper_complete=False))
    write_json(output/'labels.json', result)
    for _, _, frozen, _ in admitted: frozen.check()
    inputs.check(); guard()
    return result
