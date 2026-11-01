"""Regressions for the single final review. Structural fixtures prove no IK.

These file carriers simulate already-exited actors for boundary tests only.
The separately retained real24-frame run is the native execution evidence.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from egoemg_repro import native_ik_results as results
from egoemg_repro.data import file_sha256
from egoemg_repro.native_ik_inputs import _hash
from scripts.paper import run_native_mano_ik as workflow
from test_native_ik_results import inputs, carrier


@pytest.fixture(scope='module')
def closed_actors():
    program = "import os,json;from pathlib import Path;p=os.getpid();f=Path('/proc',str(p),'stat').read_text().rsplit(')',1)[1].split();print(json.dumps(dict(pid=p,start_ticks=f[19],uid=os.getuid(),sid=int(f[3]),pgrp=int(f[2]),state=f[0])))"
    actors = [json.loads(subprocess.check_output([sys.executable, '-c', program], text=True)) for _ in range(9)]
    for actor in actors: results.require_closed(actor)
    return actors


def qualification(closed_actors):
    owner = closed_actors[0]
    roles = [f'qual-{h}-{r}' for h in ('right', 'left') for r in ('produce', 'verify')]
    expected = {role: closed_actors[1+i*2] for i, role in enumerate(roles)}
    stages = [dict(role=role, worker=expected[role], guardian=closed_actors[2+i*2], owner=owner,
                   status='success', worker_exit_code=0, all_owned_consumers_exited=True) for i, role in enumerate(roles)]
    return owner, expected, stages


def test_qualified_backend_must_match_before_any_formal_optimization():
    original = dict(torch='2.2.2+cu118', cuda_matmul_allow_tf32=False, precision='float32')
    config = dict(action='workflow', request=dict(engineering=False), qualified_runtime_identity=original)
    changed = dict(original, torch='2.2.2+cu121')
    with pytest.raises(ValueError, match='qualified|backend|runtime'):
        workflow.check_qualified_runtime(changed, config)
    workflow.check_qualified_runtime(original, config)


@pytest.mark.parametrize('fault', ['duplicates', 'worker', 'owner', 'role'])
def test_four_qualification_stages_bind_roles_owner_and_packet_actors(closed_actors, fault):
    owner, expected, stages = qualification(closed_actors)
    if fault == 'duplicates': stages = [copy.deepcopy(stages[0])] * 4
    elif fault == 'worker': stages[1]['worker'] = stages[0]['worker']
    elif fault == 'owner': stages[0]['owner'] = stages[1]['worker']
    else: stages[0]['role'] = 'unrelated'
    with pytest.raises(ValueError, match='role|actor|owner|duplicate|worker'):
        workflow.validate_qualification_stages(stages, owner, expected)


def test_actual_distinct_closed_qualification_stage_identities_are_accepted(closed_actors):
    owner, expected, stages = qualification(closed_actors)
    workflow.validate_qualification_stages(stages, owner, expected)


@pytest.fixture
def structural_shards(inputs, tmp_path, closed_actors):
    """Real-file schema carriers only; no native optimization is claimed."""
    shards = []
    for hand in ('right', 'left'):
        for shard in range(6):
            root = tmp_path/f'{hand}-{shard}'; root.mkdir()
            pairs = []
            for index in range(inputs.identity['batches_per_hand'][shard]):
                batch = results.select_ik_batch(inputs, hand, shard, index)
                arrays, record = carrier(batch)
                producer = root/f'produce-{index}'; producer.mkdir()
                verifier = root/f'verify-{index}'; verifier.mkdir()
                np.savez(producer/'state.npz', **arrays)
                value = results.seal(dict(format='original_native_ik_raw_batch_v1', batch=batch.identity,
                    runtime_identity={'format':'structural_decoder_carrier_only'}, actor=closed_actors[0], record=record,
                    statistics=results.validate_ik_state(batch, arrays, record), state_file='state.npz',
                    state_file_sha256=file_sha256(producer/'state.npz'), results_source_sha256=results.IMPORTED_SHA,
                    pending_independent_geometry_verification=True, full_frame_labels_regenerated=False,
                    original_default_batch10096_qualified=False, scientific_eligible=False, full_paper_complete=False))
                results.write_json(producer/'result.json', value)
                results.write_json(verifier/'verification.json', results.seal(dict(format='original_native_ik_batch_geometry_verified_v1',
                    actor=closed_actors[1], producer_actor=value['actor'], producer_result_sha256=file_sha256(producer/'result.json'),
                    state_file_sha256=value['state_file_sha256'], batch=value['batch'], runtime_identity=value['runtime_identity'],
                    all_typed_geometric_arrays_exact=True, original_native_final_objective_exact=True,
                    model_state_sha256=record['model_state_sha256'], actual_ik_optimizer_steps=0,
                    optimizer_construction_and_lbfgs_step_denied=True, results_source_sha256=results.IMPORTED_SHA,
                    scientific_eligible=False, full_paper_complete=False)))
                pairs.append((producer, verifier))
            output = root/'labels'
            finalized = results.finalize_ik_shard(inputs, hand, shard, pairs, output)
            # Simulate finalizer exit in this schema-only fixture. These are
            # actual closed OS identities, not native-execution evidence.
            finalized.pop('sha256'); finalized['actor'] = closed_actors[2]
            for row, pair in zip(finalized['batches'], pairs):
                row.update(producer_directory=str(pair[0]), verification_directory=str(pair[1]))
            (output/'shard.json').unlink(); results.write_json(output/'shard.json', results.seal(finalized))
            shards.append(output)
    return shards


def test_raw_packet_drift_after_finalization_rejects_bilateral_assembly(inputs, structural_shards, tmp_path):
    (structural_shards[0].parent/'produce-0/state.npz').write_bytes(b'truncated after shard finalization')
    with pytest.raises(ValueError, match='changed|digest|state|packet'):
        results.assemble_ik_labels(inputs, structural_shards, tmp_path/'assembled')


def test_verifier_drift_after_finalization_rejects_bilateral_assembly(inputs, structural_shards, tmp_path):
    (structural_shards[0].parent/'verify-0/verification.json').write_text('{}')
    with pytest.raises(ValueError, match='seal|verification|result'):
        results.assemble_ik_labels(inputs, structural_shards, tmp_path/'assembled')


def test_first_bilateral_target_mutation_during_second_target_is_rejected(inputs, structural_shards, tmp_path, monkeypatch):
    original = results.file_sha256; changed = False
    output = tmp_path/'assembled'
    def hashing(path):
        nonlocal changed
        digest = original(path)
        if Path(path) == output/'generated_umetrack_left_finger_angles.dat' and not changed:
            changed = True
            right = output/'generated_umetrack_right_finger_angles.dat'
            values = np.fromfile(right, np.float32); values[0] += .001; values.tofile(right)
        return digest
    monkeypatch.setattr(results, 'file_sha256', hashing)
    with pytest.raises(ValueError, match='changed|label|target'):
        results.assemble_ik_labels(inputs, structural_shards, output)
    assert changed


def test_qualification_payload_guards_outlive_admission(tmp_path):
    packet = tmp_path/'state.npz'; packet.write_bytes(b'first actual bytes')
    live = results.IKArtifactBindings(); live.add_file(packet)
    packet.write_bytes(b'other actual bytes')
    with pytest.raises(ValueError, match='changed'):
        live.check()


def test_postassembler_acceptance_reopens_final_bytes_and_retains_raw_graph(inputs, structural_shards, tmp_path, closed_actors):
    output = tmp_path/'assembled'
    value = results.assemble_ik_labels(inputs, structural_shards, output)
    with pytest.raises(ValueError, match='exit'):
        results.inspect_ik_labels(inputs, output)
    value.pop('sha256'); value['actor'] = closed_actors[2]
    (output/'labels.json').unlink(); results.write_json(output/'labels.json', results.seal(value))
    metadata, live = results.inspect_ik_labels(inputs, output)
    assert metadata['rows_per_hand'] == 53
    assert metadata['full_frame_labels_regenerated'] is False
    assert structural_shards[0].parent/'produce-0/state.npz' in live.hashes
    right = output/'generated_umetrack_right_finger_angles.dat'
    values = np.fromfile(right, np.float32); values[0] += .001; values.tofile(right)
    with pytest.raises(ValueError, match='changed'):
        live.check()
    with pytest.raises(ValueError, match='digest'):
        results.inspect_ik_labels(inputs, output)
