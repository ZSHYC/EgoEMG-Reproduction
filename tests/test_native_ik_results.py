"""Full-shard membership and raw-state boundaries; no optimizer stand-in.

The structural carrier below is synthetic engineering data. It exercises typed
I/O checks only and never supplies evidence of an actual native optimization.
"""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from egoemg_repro.native_ik_inputs import admit_native_ik_inputs
from egoemg_repro import native_ik_results as results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    data, geometry, upstream = [tmp_path / n for n in ('data', 'geometry', 'upstream')]
    for p in (data, geometry, upstream):
        p.mkdir()
    for rel in ('scripts/ik/batch_ik_mesh.py', 'egoemg/UmeTrack/dataset/generic_hand_model.json'):
        p = upstream / rel; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'upstream' / rel, p)
    fields = {}
    for hand in ('right', 'left'):
        name = 'generated_mano_' + hand + '_pose'
        a = np.arange(53 * 48, dtype=np.float32).reshape(53, 48) / 100
        if hand == 'left': a += 1
        a[[0, 11, 52]] = 0
        a.tofile(geometry / (name + '.dat'))
        fields[name] = dict(filename=name + '.dat', dtype='float32', shape=[53, 48])
    (data / 'manifest.json').write_text(json.dumps(dict(total_rows=53, fields=fields)))
    np.savez(data / 'metadata.npz', total=np.array([53]))
    return admit_native_ik_inputs(data, geometry, upstream, engineering=True, batch_size=3)


def carrier(batch):
    arrays = batch.arrays
    b = len(arrays['indices'])
    limits = np.asarray(json.loads((ROOT / 'upstream/egoemg/UmeTrack/dataset/generic_hand_model.json').read_text())['joint_limits'][:20], np.float32)
    arrays.update(angles=np.tile((limits[:, 0] + limits[:, 1]) / 2, (b, 1)),
        raw_angles=np.zeros((b, 20), np.float32),
        global_orient=np.zeros((b, 3), np.float32), trans_raw=np.zeros((b, 3), np.float32),
        residual=np.zeros((b, 20, 3), np.float32),
        predicted_landmarks=np.zeros((b, 21, 3), np.float32),
        aligned_mano_joints=np.zeros((b, 21, 3), np.float32))
    record = dict(final_objective_mm2=0., native_final_loss=0.,
        model_state_sha256='1' * 64, actual_ik_outer_steps=1, actual_ik_optimizer_steps=1,
        actual_ik_closure_evaluations=1, actual_original_step_return_preserved=True,
        actual_rebuild_optimizer_steps=0, source_identity_sha256='2' * 64,
        seconds=.1, ik_step_trace=[dict(outer_step=1, closure_calls_so_far=1, returned_pre_step_loss=0.)])
    return arrays, record


def test_every_original_shard_and_remainder_have_exact_stream_membership(inputs):
    assert 'torch' not in sys.modules
    for hand in ('right', 'left'):
        rows = []
        for shard in range(6):
            expected = list(inputs.batches(hand, shard))
            for index, (ids, raw, canonical) in enumerate(expected):
                batch = results.select_ik_batch(inputs, hand, shard, index)
                actual = batch.arrays
                for key, wanted in zip(('indices', 'raw_poses', 'canonical_poses'), (ids, raw, canonical)):
                    np.testing.assert_array_equal(actual[key], wanted)
                assert batch.identity['rows'] == len(ids)
                assert batch.identity['batch_index'] == index
                assert batch.identity['engineering'] is True
                rows.extend(actual['indices'].tolist())
        assert rows == list(range(53))
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('hand,shard,batch', [('both', 0, 0), ('left', True, 0), ('left', 6, 0),
    ('right', 0, -1), ('right', 0, True), ('right', 0, 3), ('right', 5, 5)])
def test_arbitrary_membership_is_rejected(inputs, hand, shard, batch):
    with pytest.raises(ValueError, match='hand|shard|batch'):
        results.select_ik_batch(inputs, hand, shard, batch)


def test_selection_returns_copies_and_binds_live_source(inputs):
    batch = results.select_ik_batch(inputs, 'left', 0, 0)
    a = batch.arrays; identity = batch.identity
    a['raw_poses'][:] = 999; identity['hand'] = 'right'
    assert batch.identity['hand'] == 'left'
    assert not (batch.arrays['raw_poses'] == 999).any()
    from egoemg_repro.native_ik_inputs import _state
    path = _state(inputs)['paths']['left']
    with path.open('r+b') as f:
        f.seek(48 * 4); f.write(np.zeros(48, np.float32).tobytes())
    with pytest.raises(ValueError, match='changed'):
        batch.check()


def test_model_free_carrier_validation_is_explicitly_not_geometry_acceptance(inputs):
    batch = results.select_ik_batch(inputs, 'right', 0, 0)
    arrays, record = carrier(batch)
    stats = results.validate_ik_state(batch, arrays, record)
    assert stats['rows'] == 3
    assert stats['raw_zero_pose_rows'] == 1
    assert stats['mean_landmark_distance_mm'] == 0
    assert stats['geometric_reconstruction_verified'] is False
    assert stats['native_optimizer_execution_verified'] is False
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('fault', ['missing', 'extra', 'float64', 'shape', 'nan', 'raw', 'canonical',
    'indices', 'angles', 'residual', 'loss', 'trace_count', 'closure', 'steps', 'trace_loss'])
def test_typed_raw_state_and_actual_step_schema_are_strict(inputs, fault):
    batch = results.select_ik_batch(inputs, 'right', 0, 0)
    arrays, record = carrier(batch)
    if fault == 'missing': del arrays['raw_angles']
    elif fault == 'extra': arrays['unbound'] = np.zeros(3)
    elif fault == 'float64': arrays['angles'] = arrays['angles'].astype(np.float64)
    elif fault == 'shape': arrays['trans_raw'] = np.zeros((3, 4), np.float32)
    elif fault == 'nan': arrays['global_orient'][0, 0] = np.nan
    elif fault == 'raw': arrays['raw_poses'][0, 0] = 1
    elif fault == 'canonical': arrays['canonical_poses'][0, 1] = 1
    elif fault == 'indices': arrays['indices'] = arrays['indices'][::-1].copy()
    elif fault == 'angles': arrays['angles'][0, 0] = 100
    elif fault == 'residual': arrays['residual'][0, 0, 0] = 1
    elif fault == 'loss': record['native_final_loss'] = 2
    elif fault == 'trace_count': record['ik_step_trace'] = []
    elif fault == 'closure': record['ik_step_trace'][0]['closure_calls_so_far'] = 0
    elif fault == 'steps': record['actual_ik_optimizer_steps'] = True
    elif fault == 'trace_loss': record['ik_step_trace'][0]['returned_pre_step_loss'] = float('nan')
    with pytest.raises(ValueError):
        results.validate_ik_state(batch, arrays, record)


def test_no_raw_state_can_be_written_with_unregistered_runtime(inputs, tmp_path):
    batch = results.select_ik_batch(inputs, 'right', 0, 0)
    target = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='registered|runtime'):
        results.produce_ik_batch(batch, object(), target)
    assert not target.exists()


def test_incomplete_and_unverified_packets_cannot_materialize_labels(inputs, tmp_path):
    target = tmp_path / 'labels'
    with pytest.raises(ValueError, match='complete|batch|verification'):
        results.finalize_ik_shard(inputs, 'right', 0, [], target)
    assert not target.exists()


def test_packet_does_not_follow_symlink_or_accept_incomplete_output(inputs, tmp_path):
    target = tmp_path / 'partial'; target.mkdir()
    (target / 'result.json').write_text('{}')
    with pytest.raises(ValueError):
        results.inspect_ik_batch(inputs, target)
    linked = tmp_path / 'linked'; linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        results.inspect_ik_batch(inputs, linked)


def test_both_hands_need_all_six_verified_shards_before_assembly(inputs, tmp_path):
    output = tmp_path/'assembled'
    with pytest.raises(ValueError, match='twelve|six|complete'):
        results.assemble_ik_labels(inputs, [], output)
    assert not output.exists()


def structural_packet(inputs, directory):
    """Write a carrier for decoder tests; current actor is intentionally LIVE.

    It cannot pass normal closed-producer admission or native verification.
    """
    from egoemg_repro.emg2pose_memory_process import process_identity
    from egoemg_repro.data import file_sha256
    batch = results.select_ik_batch(inputs, 'right', 0, 0)
    arrays, record = carrier(batch)
    directory.mkdir()
    np.savez(directory/'state.npz', **arrays)
    value = results.seal(dict(format='original_native_ik_raw_batch_v1', batch=batch.identity,
        runtime_identity={'format':'structural_decoder_carrier_only'}, actor=process_identity(), record=record,
        statistics=results.validate_ik_state(batch, arrays, record), state_file='state.npz',
        state_file_sha256=file_sha256(directory/'state.npz'), results_source_sha256=results.IMPORTED_SHA,
        pending_independent_geometry_verification=True, full_frame_labels_regenerated=False,
        original_default_batch10096_qualified=False, scientific_eligible=False, full_paper_complete=False))
    results.write_json(directory/'result.json', value)
    return value


def test_live_producer_is_never_accepted_as_closed(inputs, tmp_path):
    directory = tmp_path/'packet'; structural_packet(inputs, directory)
    with pytest.raises(ValueError, match='exit'):
        results.inspect_ik_batch(inputs, directory)
    packet = results.inspect_ik_batch(inputs, directory, require_producer_closed=False)
    assert packet.metadata['statistics']['geometric_reconstruction_verified'] is False
    assert packet.metadata['statistics']['native_optimizer_execution_verified'] is False
    copied = packet.arrays; copied['raw_poses'][:] = 999
    assert not (packet.arrays['raw_poses'] == 999).any()


def test_same_size_packet_replacement_invalidates_consumption(inputs, tmp_path):
    directory = tmp_path/'packet'; structural_packet(inputs, directory)
    packet = results.inspect_ik_batch(inputs, directory, require_producer_closed=False)
    path = directory/'state.npz'; raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match='changed'):
        _ = packet.arrays


def test_first_hash_race_is_guarded_before_any_payload_read(inputs, tmp_path, monkeypatch):
    directory = tmp_path/'packet'; structural_packet(inputs, directory)
    original = results.file_sha256
    changed = False
    def hashed(path):
        nonlocal changed
        digest = original(path)
        if Path(path) == directory/'state.npz' and not changed:
            changed = True
            raw = Path(path).read_bytes()
            Path(path).write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        return digest
    monkeypatch.setattr(results, 'file_sha256', hashed)
    with pytest.raises(ValueError, match='changed'):
        results.inspect_ik_batch(inputs, directory, require_producer_closed=False)
    assert changed


def test_resealed_wrong_hand_cannot_rebind_raw_source_witness(inputs, tmp_path):
    directory = tmp_path/'packet'; value = structural_packet(inputs, directory)
    path = directory/'result.json'; value.pop('sha256'); value['batch']['hand'] = 'left'
    path.unlink(); results.write_json(path, results.seal(value))
    with pytest.raises(ValueError, match='membership|witness|identity'):
        results.inspect_ik_batch(inputs, directory, require_producer_closed=False)


def test_unregistered_batch_cannot_consume_source_or_supply_state():
    with pytest.raises(ValueError, match='registered'):
        results.IKBatch().check()


def test_label_outputs_cannot_be_created_inside_source_roots(inputs):
    from egoemg_repro.native_ik_inputs import _state
    source_root = _state(inputs)['paths']['right'].parent
    with pytest.raises(ValueError, match='isolated|source'):
        results._new_output(inputs, source_root/'new-labels')
    assert not (source_root/'new-labels').exists()
