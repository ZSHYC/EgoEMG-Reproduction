"""New full-run integration; engineering packets never certify full training."""
import copy
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', ROOT/'upstream')).resolve()


def api():
    try: return importlib.import_module('egoemg_repro.rn18_occlusion_run')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.rn18_occlusion_run': pytest.fail('complete RN18 dense inference/replay integration missing')
        raise


def arguments():
    return dict(evidence=ROOT/'results/native-dense-occlusion', upstream=UPSTREAM,
        index_root='/data/shihao/egoemg-native-dense-occlusion-complete-indices-20261101',
        paired_root='/data/shihao/egoemg-native-dense-paired-inputs-v2-20261101',
        geometry_root='/data/shihao/egoemg-native-dense-geometry-rn18_s-full-20261101')


@pytest.fixture(scope='module')
def actual_evidence():
    m = api()
    if not Path(arguments()['paired_root']).is_dir():
        if os.environ.get('EGOEMG_REQUIRE_DENSE_RUN_TEST') == '1': pytest.fail('actual completed dense source roots required')
        pytest.skip('actual full source journals required; no fake complete-source positive')
    return m.load_dense_evidence(**arguments())


@pytest.fixture(scope='module')
def actual_inputs(actual_evidence):
    import torch
    original = torch.nn.Module.__init__
    def denied(*a, **kw): raise AssertionError('input admission constructed learned model')
    torch.nn.Module.__init__ = denied
    try:
        return api().admit_dense_run_inputs(actual_evidence,
            data='/home/shihao/projects/EgoEMG-Reproduction/data/legacy',
            crops='/home/shihao/projects/EgoEMG-Reproduction/data/crops')
    finally: torch.nn.Module.__init__ = original


def test_real_evidence_closes_exact_full_keys_gt_and_completed_producer_footprints(actual_evidence):
    p = actual_evidence; p.check()
    assert len(p.keys()) == len(p.geometry()) == 11114
    assert p.identity['counts'] == dict(points=11114, zero_geometry=327, crop_missing=440, label_invalid=327)
    assert p.identity['batch_sizes'][-1] == 40 and len(p.identity['batch_sizes']) == 176
    assert p.identity['actual_completed_producer_footprints_bound']
    assert not p.identity['trained_model_or_inference_permit']


def test_resealed_proofs_or_forged_evidence_cannot_bypass_source_checks(tmp_path, actual_evidence):
    m = api(); directory = tmp_path/'proofs'; directory.mkdir()
    for name in m.PUBLIC_PROOFS:
        (directory/name).parent.mkdir(parents=True,exist_ok=True)
        (directory/name).write_bytes((arguments()['evidence']/name).read_bytes())
    target = directory/'paired-full-io-complete-20261101.json'
    value = json.loads(target.read_text()); value['actual_total_paired_hand_requests'] = 1
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError): m.load_dense_evidence(**dict(arguments(), evidence=directory))
    forged = copy.copy(actual_evidence)
    with pytest.raises(ValueError, match='registered'): forged.check()


def test_real_full_input_admission_matches_complete_io_without_models(actual_inputs):
    p = actual_inputs; p.check()
    assert p.identity['actual_complete_payload_files'] == 53
    assert p.identity['actual_complete_payload_bytes'] == 61377230438
    assert p.identity['complete_pre_forward_payload_hashes_verified']
    assert not p.identity['trained_model_or_inference_permit']
    invalid=next(i for i,r in enumerate(p.evidence.geometry())if r['status']=='invalid_zero_pose')
    row = p[invalid]
    assert row['key']['sequence'] == invalid and not row['fusion']['label_valid_mask'].any()
    assert p.verified_samples == 1
    clone = copy.copy(p)
    with pytest.raises(ValueError, match='registered'): clone.check()


def key(i):
    return dict(sequence=i, split='user', hand='left', ep_idx=0, center_idx=6000+i,
        window_start=i, window_end=12000+i, group_index=0, native_index=i)


def packet():
    import torch
    target = np.arange(66, dtype=np.float32).reshape(3, 22)/17
    full = np.repeat(np.arange(230, dtype=np.float32)[None, None, :], 3, axis=0)
    full = np.repeat(full, 22, axis=1)
    a = dict(fusion_prediction=target+.3, vision_prediction=target-.7,
        emg_prediction=full[:, :, 115].copy(), emg_full_prediction=full,
        targets=target, label_valid_mask=np.array([[False], [True], [False]]),
        crop_valid_mask=np.array([[False], [True], [False]]))
    for role in ('fusion', 'vision', 'emg'):
        a['mae_'+role+'_rad'] = np.array([float((torch.from_numpy(a[role+'_prediction'][i])
            - torch.from_numpy(target[i])).abs().mean()) for i in range(3)], np.float64)
    return a


def test_raw_packet_preserves_all_predictions_trajectory_center_flags_and_source_errors(tmp_path):
    m = api(); a = packet(); keys = [key(i) for i in range(3)]
    record = m.write_dense_batch(tmp_path, 0, keys=keys, arrays=a, inputs_sha256={'fusion:emg': 'a'*64})
    b = m.read_dense_batch(tmp_path, record, expected_keys=keys)
    assert set(a) == set(b)
    for name in a: assert a[name].dtype == b[name].dtype and np.array_equal(a[name], b[name])
    assert b['label_valid_mask'].shape == (3, 1) and not b['label_valid_mask'][0, 0]
    assert len(b['mae_vision_rad']) == 3
    assert not (tmp_path/'result.json').exists()  # A raw batch cannot finish a full pass.


@pytest.mark.parametrize('damage', ['nonfinite', 'dtype', 'time_center', 'error', 'mask_shape', 'missing_crop_mask', 'duplicate', 'mixed_hand'])
def test_corrupt_raw_semantics_reject_before_export(tmp_path, damage):
    m = api(); a = packet(); keys = [key(i) for i in range(3)]
    if damage == 'nonfinite': a['vision_prediction'][0, 0] = np.nan
    elif damage == 'dtype': a['targets'] = a['targets'].astype(np.float64)
    elif damage == 'time_center': a['emg_prediction'][0, 0] += 1
    elif damage == 'error': a['mae_vision_rad'][0] += .01
    elif damage == 'mask_shape': a['label_valid_mask'] = a['label_valid_mask'][:, :, None]
    elif damage == 'missing_crop_mask': a.pop('crop_valid_mask')
    elif damage == 'duplicate': keys[1] = dict(keys[0], sequence=1)
    elif damage == 'mixed_hand': keys[1]['hand'] = 'right'
    with pytest.raises(ValueError):
        m.write_dense_batch(tmp_path, 0, keys=keys, arrays=a, inputs_sha256={'fusion:emg': 'a'*64})
    assert not list(tmp_path.glob('*.npz'))


def test_changed_or_truncated_packet_and_unsafe_paths_never_replay(tmp_path):
    m = api(); keys = [key(i) for i in range(3)]
    record = m.write_dense_batch(tmp_path, 0, keys=keys, arrays=packet(), inputs_sha256={'fusion:emg': 'a'*64})
    path = tmp_path/record['file']; path.write_bytes(path.read_bytes()[:-11])
    with pytest.raises(ValueError): m.read_dense_batch(tmp_path, record, expected_keys=keys)
    forged = copy.deepcopy(record); forged['file'] = '../outside.npz'
    with pytest.raises(ValueError): m.read_dense_batch(tmp_path, forged, expected_keys=keys)


def test_consistent_but_different_replay_predictions_are_rejected(tmp_path):
    import torch
    m = api(); keys = [key(i) for i in range(3)]
    first = tmp_path/'first'; second = tmp_path/'second'; first.mkdir(); second.mkdir()
    a = packet(); r = m.write_dense_batch(first, 0, keys=keys, arrays=a, inputs_sha256={'fusion:emg': 'a'*64})
    a['fusion_prediction'][0, 0] += .2
    a['mae_fusion_rad'][0] = float((torch.from_numpy(a['fusion_prediction'][0])-torch.from_numpy(a['targets'][0])).abs().mean())
    s = m.write_dense_batch(second, 0, keys=keys, arrays=a, inputs_sha256={'fusion:emg': 'a'*64})
    with pytest.raises(ValueError, match='replay'):
        m.compare_dense_batches(first, r, second, s, expected_keys=keys)


def test_partial_or_forged_completed_pass_is_rejected(tmp_path, actual_evidence):
    m = api()
    (tmp_path/'identity.json').write_text(json.dumps(dict(scope='complete', source_identity_sha256=actual_evidence.identity['sha256'])))
    (tmp_path/'result.json').write_text(json.dumps(dict(scientific_eligible=True, complete=True, points=11114)))
    with pytest.raises(ValueError): m.read_dense_pass(tmp_path, evidence=actual_evidence)


def test_complete_run_cannot_start_without_registered_inputs_and_real_qualification(tmp_path, actual_inputs):
    from types import SimpleNamespace
    m = api()
    with pytest.raises(ValueError):
        m.run_dense_pass(SimpleNamespace(check=lambda: None), tmp_path, tmp_path/'fake', scope='complete')
    with pytest.raises(ValueError, match='qualification'):
        m.run_dense_pass(actual_inputs, tmp_path, tmp_path/'unqualified', scope='complete')
    assert not (tmp_path/'unqualified').exists()


def test_resealed_qualification_markers_without_actual_raw_passes_are_rejected(tmp_path, actual_evidence):
    import hashlib
    m = api(); first = tmp_path/'first'; second = tmp_path/'second'; first.mkdir(); second.mkdir()
    for p in (first, second): (p/'result.json').write_text('{}')
    value = m._seal(dict(format='actual_dense_cpu_qualification_acceptance_v1',
        evidence_identity_sha256=actual_evidence.identity['sha256'], model_identity_sha256=m.MODEL_ID,
        source_files_sha256=actual_evidence.identity['source_files_sha256'],
        first=str(first), second=str(second),
        first_result_file_sha256=hashlib.sha256(b'{}').hexdigest(),
        second_result_file_sha256=hashlib.sha256(b'{}').hexdigest(),
        actual_batch_size=64, actual_workers=4, actual_points_each=618, actual_batches_each=12,
        separate_fresh_replay_exact=True, all_workers_exited=True))
    path = tmp_path/'qualification.json'; path.write_text(json.dumps(value))
    with pytest.raises(ValueError): m._qualification(path, actual_evidence)


def test_cli_import_is_guarded_and_does_not_start_work(monkeypatch):
    api()
    def denied(*a, **kw): raise AssertionError('import launched dense prediction')
    monkeypatch.setattr(api(), 'run_dense_pass', denied)
    path = ROOT/'scripts/paper/run_rn18_dense_occlusion.py'
    assert path.is_file(), 'guarded dense inference/report CLI missing'
    runpy.run_path(str(path), run_name='spawn_probe')


def test_source_four_worker_loader_closes_its_children_after_late_consumer_failure():
    m = api(); handles = []
    class TinyPackets:
        def __len__(self): return 65
        def __getitem__(self, i):
            fusion = dict(emg=np.zeros((8, 12000), np.float32),
                joint_angles=np.zeros((22, 1), np.float32), label_valid_mask=np.array([False]),
                vision_img=np.zeros((1, 3, 4, 4), np.float32), vision_valid_mask=np.array([False]),
                _ep_idx=0, _center_idx=6000+i, target_hand='left')
            standalone = {n: v for n, v in fusion.items() if n in ('joint_angles', 'label_valid_mask')}
            standalone['emg'] = np.zeros((8, 12000), np.float32)
            return dict(key=key(i), fusion=fusion, standalone_emg=standalone, verified_input_sha256='a'*64)
    with pytest.raises(RuntimeError, match='late consumer failure'):
        with m._managed_loader(TinyPackets(), list(range(65)), upstream=UPSTREAM) as (iterator, workers, pin):
            handles.extend(workers)
            batch = next(iterator)
            assert batch['fusion']['emg'].shape == (64, 8, 12000)
            assert len(workers) == 4
            raise RuntimeError('late consumer failure')
    assert len(handles) == 4
    for handle in handles: m._actor_closed(handle)


def comparison_packets(tmp_path):
    """Engineering arrays and real exited OS actors; no full-source positive."""
    from types import SimpleNamespace
    m = api(); keys = [key(i) for i in range(3)]; identity = m._seal(dict(batch_ranges=[[0, 3]]))
    code = 'import json;from egoemg_repro.emg2pose_memory_process import process_identity;print(json.dumps(process_identity()))'
    actors = [json.loads(subprocess.check_output([sys.executable, '-B', '-c', code], text=True)) for _ in range(2)]
    results = []
    for i, actor in enumerate(actors):
        root = tmp_path/('pass'+str(i)); root.mkdir(); (root/'batches').mkdir()
        record = m.write_dense_batch(root/'batches', 0, keys=keys, arrays=packet(), inputs_sha256={'fusion:emg': 'a'*64})
        result = dict(actor=actor, model_state_sha256_after={'engineering': 'a'*64},
            fresh_model_forward_replay_verified=bool(i), replay_of_result_file_sha256=None,
            replay_of_identity_sha256=identity['sha256'] if i else None)
        if i: result['replay_of_result_file_sha256'] = m._sha(Path(results[0]['directory'])/'result.json')
        (root/'result.json').write_text(json.dumps(result))
        results.append(dict(directory=str(root), identity=identity, result=result, records=[record]))
    return results[0], results[1], SimpleNamespace(keys=lambda: keys)


@pytest.mark.parametrize('damage', ['same_directory', 'copied_actor', 'wrong_parent_digest'])
def test_a_pass_cannot_be_its_own_fresh_replay_or_bind_another_parent(tmp_path, damage):
    m = api(); first, second, evidence = comparison_packets(tmp_path)
    if damage == 'same_directory': second = first
    elif damage == 'copied_actor': second['result']['actor'] = first['result']['actor']
    else: second['result']['replay_of_result_file_sha256'] = '0'*64
    with pytest.raises(ValueError, match='replay'):
        m._compare_passes(first, second, evidence)


def test_distinct_closed_actors_with_the_exact_preceding_result_can_compare(tmp_path):
    m = api(); first, second, evidence = comparison_packets(tmp_path)
    m._compare_passes(first, second, evidence)


def test_report_stays_pending_before_independent_verification(tmp_path, monkeypatch):
    """Report-control logic only; source/model gates and math are substituted.

    The placeholders are explicitly not a positive full population experiment.
    """
    from types import SimpleNamespace
    m = api(); keys = [key(0)]
    prediction = dict(keys[0], mae_vision_rad=.1, mae_emg_rad=.2, mae_fusion_rad=.09)
    geometry = [dict(occlusion_score=None, visible_ratio=None, status='invalid_zero_pose')]
    evidence = SimpleNamespace(identity=dict(sha256='0'*64, completed_producer_footprints={}),
        upstream=UPSTREAM, keys=lambda: keys, geometry=lambda: geometry, check=lambda: None)
    fake_pass = dict(identity=dict(scope='complete', hardware=dict(device='cpu')),
        result=dict(fresh_model_forward_replay_verified=True), predictions=[prediction])
    monkeypatch.setattr(m, 'read_dense_pass', lambda *a, **kw: fake_pass)
    monkeypatch.setattr(m, '_compare_passes', lambda *a, **kw: None)
    monkeypatch.setattr(m, 'join_dense_predictions', lambda *a, **kw: dict(csv_rows=[dict(mae_vision_deg=5.73)]))
    monkeypatch.setattr(m, 'analyze_source_csv', lambda *a, **kw: dict(sha256='a'*64,
        filtered_rows=0, binned_rows=0, bins=[], spearman=dict(rho=None, pvalue=None)))
    original = subprocess.run
    def substitute_plot(command, *a, **kw):
        if len(command) > 2 and str(command[2]).endswith('plot_occlusion_fusion.py'):
            Path(command[command.index('--output')+1]).write_bytes(b'engineering placeholder; not a scientific image')
            return subprocess.CompletedProcess(command, 0, stdout='engineering plot placeholder', stderr='')
        return original(command, *a, **kw)
    monkeypatch.setattr(subprocess, 'run', substitute_plot)
    out = tmp_path/'report'; result = m.build_dense_report('placeholder-a', 'placeholder-b', evidence=evidence, output=out)
    assert result['available_legacy_figure13_pipeline_complete'] is False
    assert result['scientific_eligible'] is False
    assert result['verification_status'] == 'pending independent reconstruction'
    assert not (out/'completion.json').exists()


def test_report_cannot_promote_while_verification_stage_worker_is_alive(tmp_path, actual_evidence):
    from egoemg_repro.emg2pose_memory_process import process_identity
    m = api(); actor = process_identity()
    stage = dict(status='success', worker_exit_code=0, all_owned_consumers_exited=True,
        worker=actor, guardian=actor, owner=actor)
    with pytest.raises(ValueError):
        m._promote_dense_report(tmp_path/'report', tmp_path/'rebuilt', evidence=actual_evidence, owned_stage=stage)
    assert not (tmp_path/'report/completion.json').exists()


def test_actual_RN18_full_and_qualification_groups_retain_all_source_tails(actual_evidence):
    m=api();full=m._scope_batches(actual_evidence,'complete');qual=m._scope_batches(actual_evidence,'qualification')
    assert len(full)==176 and sum(b-a for a,b in full)==11114
    assert len(qual)==12 and [b-a for a,b in qual]==[64,24,64,24,64,53,64,53,64,40,64,40]
    assert sum(b-a for a,b in qual)==618


def test_evidence_guard_replacement_is_rejected_before_calling_it(actual_evidence,monkeypatch):
    class Poison:
        def check(self):raise AssertionError('unregistered guard executed')
    with monkeypatch.context()as patch:
        patch.setattr(actual_evidence,'_guard',Poison())
        with pytest.raises(ValueError):actual_evidence.check()
    actual_evidence.check()


def test_RN18_consumed_file_scope_retains_all_keys_and_explains_shared_only_crop(actual_evidence,actual_inputs):
    p=actual_evidence;p.check()
    assert len(actual_inputs)==11114
    assert len(p.identity['complete_payload_sha256'])==53
    assert len(p.identity['shared_IO_payload_sha256'])==54
    assert set(p.identity['shared_unconsumed_payload_sha256'])=={'crop:000004'}
    assert all(k['ep_idx']!=4 for k in p.keys())
    assert actual_inputs.identity['actual_complete_payload_files']==53


def test_changed_trained_loader_source_cannot_reuse_the_old_model_identity(monkeypatch):
    m=api();original=m._sha;target=(ROOT/'egoemg_repro/rn18_occlusion_models.py').resolve()
    def changed(path):
        return '0'*64 if Path(path).resolve()==target else original(path)
    monkeypatch.setattr(m,'_sha',changed)
    with pytest.raises(ValueError):m.load_dense_evidence(**arguments())


@pytest.mark.parametrize('shortage',['disk','available_memory','own_RSS'])
def test_runtime_resource_shortage_refuses_work(shortage,tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    m=api();gib=1024**3;original=Path.read_text
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=(224 if shortage=='disk'else 512)*gib))
    def read(path,*a,**kw):
        if str(path)=='/proc/meminfo':return 'MemAvailable: '+str((127 if shortage=='available_memory'else 512)*gib//1024)+' kB\n'
        if str(path)=='/proc/self/status':return 'VmRSS: '+str((129 if shortage=='own_RSS'else 1)*gib//1024)+' kB\n'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(ValueError):m._runtime_budget(tmp_path)
