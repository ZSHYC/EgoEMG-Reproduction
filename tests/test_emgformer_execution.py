"""Exercise the real operational gates without Torch, data, or GPU execution."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
CANDIDATE = '24290047e6eaab7e845f009f645c58c9b1d18e38'
BASE = '9f22cfa8d973af8b4ee20a75bd8ad5e53ace0a29'
COMMIT = 'a' * 40
UUIDS = ['GPU-8b2cacdf-4470-21f2-01c5-e15475dbdfae',
         'GPU-20575323-8e23-b0b8-74b4-b987d870b0e3']
SOURCES = {
    'private/runtime-worker.py': 'scripts/run_native_emgformer_worker.py',
    'private/runtime-probe.py': 'scripts/run_native_emgformer_controller.py',
    'private/recorder.py': 'tmp/emg2pose-memory-recovery-exit-recorder.py',
    'egoemg_repro/emgformer_resource.py': 'egoemg_repro/emgformer_resource.py',
    'egoemg_repro/emgformer_execution.py': 'egoemg_repro/emgformer_execution.py',
    'egoemg_repro/emg2pose_memory_process.py': 'egoemg_repro/emg2pose_memory_process.py',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def seal(root):
    files = {str(p.relative_to(root)): digest(p) for p in root.rglob('*')
             if p.is_file() and p.name != 'stage-receipt.json'}
    body = dict(source_commit=COMMIT, files_sha256=files, full_paper_complete=False)
    bundle = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    put(root/'private/stage-receipt.json', dict(body, bundle_sha256=bundle, root=str(root)))


@pytest.fixture
def capsule(tmp_path, monkeypatch):
    root = tmp_path/'capsule'
    for name, source in SOURCES.items():
        p = root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        original = REPO/source
        p.write_bytes(original.read_bytes() if original.exists() else b'not implemented yet\n')
    sources = {n: digest(root/n) for n in SOURCES}
    put(root/'private/review-initial.json', dict(format='native_emgformer_operational_initial_review_v1',
        reviewer='/root/native_emgformer_gpu_workflow_review', base_commit=BASE,
        candidate_commit=CANDIDATE, critical=0, important=4, finding_ids=['I1','I2','I3','I4']))
    put(root/'private/review-resolution.json', dict(format='native_emgformer_operational_review_resolution_v1',
        reviewed_candidate_commit=CANDIDATE, resolved_source_commit=COMMIT,
        resolved_important_ids=['I1','I2','I3','I4'], critical=0, important=0,
        regression_tests_passed=20, regression_log_sha256='b'*64, sources_sha256=sources,
        second_review_requested=False))
    put(root/'private/final-review.json', dict(format='native_emgformer_operational_final_review_v1',
        reviewed_candidate_commit=CANDIDATE, resolved_source_commit=COMMIT,
        critical=0, important=0, second_review_requested=False, sources_sha256=sources,
        initial_review_sha256=digest(root/'private/review-initial.json'),
        resolution_sha256=digest(root/'private/review-resolution.json')))
    put(root/'private/request.json', dict(candidate_uuids=UUIDS, size='small',
        execution_mode='full_after_qualification'))
    put(root/'private/registered-cohort.json', dict(sha256='c'*64))
    seal(root)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', ['runtime-worker.py', 'qual-prefix'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUIDS[0])
    put(root/'private/gpu-binding.json', dict(uuid=UUIDS[0]))
    return root


def prefix(root, role):
    """Run the actual script up to its first external execution boundary."""
    path = REPO/'scripts'/('run_native_emgformer_'+role+'.py')
    tree = ast.parse(path.read_text(), filename=str(path))
    end = next(i for i, n in enumerate(tree.body) if
        (role == 'worker' and isinstance(n, ast.Import) and n.names[0].name == 'torch') or
        (role == 'controller' and isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
         and isinstance(n.value.func, ast.Name) and n.value.func.id == 'guard'))
    namespace = dict(__file__=str(root/('private/runtime-worker.py' if role == 'worker' else 'private/runtime-probe.py')))
    exec(compile(ast.Module(body=tree.body[:end], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('role', ['worker', 'controller'])
def test_vacuous_review_cannot_authorize_gpu(capsule, role):
    put(capsule/'private/final-review.json', dict(critical=0, important=0,
        second_review_requested=False, sources_sha256={}))
    seal(capsule)
    with pytest.raises((ValueError, AssertionError), match='review'):
        prefix(capsule, role)


def test_controller_detects_receipt_metadata_mutation(capsule, monkeypatch):
    ns = prefix(capsule, 'controller')
    ns['shutil'] = SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=1000*1024**3))
    real_path = ns['Path']
    ns['Path'] = lambda p: SimpleNamespace(read_text=lambda: 'MemAvailable: 999999999 kB') if p == '/proc/meminfo' else real_path(p)
    path = capsule/'private/stage-receipt.json'
    value = json.loads(path.read_text()); value['source_commit'] = 'c'*40
    value['bundle_sha256'] = 'd'*64; put(path, value)
    with pytest.raises((ValueError, AssertionError), match='(immutable|stage|receipt)'):
        ns['guard']()


@pytest.mark.parametrize('field,value', [('bundle_sha256','d'*64), ('root','/unrelated/root')])
@pytest.mark.parametrize('role', ['worker', 'controller'])
def test_stage_claims_must_match_canonical_bytes_and_root(capsule, field, value, role):
    path = capsule/'private/stage-receipt.json'
    v = json.loads(path.read_text()); v[field] = value; put(path, v)
    with pytest.raises((ValueError, AssertionError), match='(bundle|root|stage)'):
        prefix(capsule, role)


@pytest.mark.parametrize('role', ['worker', 'controller'])
@pytest.mark.parametrize('change', ['missing-source','bool-count','stale-candidate','unbound-resolution'])
def test_review_requires_exact_sources_typed_counts_and_resolution(capsule, role, change):
    path = capsule/'private/final-review.json'; v = json.loads(path.read_text())
    if change == 'missing-source': v['sources_sha256'].pop('private/runtime-worker.py')
    elif change == 'bool-count': v['critical'] = False
    elif change == 'stale-candidate': v['reviewed_candidate_commit'] = 'e'*40
    else: v['resolution_sha256'] = 'f'*64
    put(path, v); seal(capsule)
    with pytest.raises((ValueError, AssertionError), match='review'):
        prefix(capsule, role)


def test_gpu_worker_requires_completed_current_preflight_before_torch(capsule):
    with pytest.raises((ValueError, AssertionError), match='(preflight|authorization)'):
        prefix(capsule, 'worker')


def real_guard(root, snapshots, *, alive):
    tree = ast.parse((REPO/'scripts/run_native_emgformer_controller.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'guard')
    owner = dict(pid=23456, start_ticks='1234')
    put(root/'private/owned-qual-prefix/started.json', dict(worker=owner))
    calls = []
    def inventory(force=False):
        calls.append(force)
        return snapshots[1 if force else 0]
    def path(value):
        if value == '/proc/meminfo':
            return SimpleNamespace(read_text=lambda: 'MemAvailable: 999999999 kB')
        return Path(value)
    ns = dict(sys=sys, stamps={}, stamp=lambda _: None, Path=path,
        shutil=SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=1000*1024**3)),
        ROOT=root, mode='qual-prefix', phase_started=time.monotonic(), time=time,
        selected=dict(uuid=UUIDS[0]), inventory=inventory,
        read=lambda p: json.loads(p.read_text()), same_process=lambda i: alive,
        gpu_owners={}, observations=[], datetime=__import__('datetime'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-controller-guard', 'exec'), ns)
    return ns['guard'], calls


def snapshot(pid):
    return dict(gpus=[dict(uuid=UUIDS[0],free_mib=70000)], observed_utc='2026-10-29T00:00:00Z',
        applications=[] if pid is None else [dict(uuid=UUIDS[0],pid=pid)])


def test_reaped_owned_worker_refreshes_cached_nvml_instead_of_false_foreign(tmp_path):
    guard, _ = real_guard(tmp_path, [snapshot(23456), snapshot(None)], alive=False)
    guard()


@pytest.mark.parametrize('pid', [23456,34567])
def test_refresh_after_owner_exit_still_rejects_reused_or_foreign_pid(tmp_path, pid):
    guard, _ = real_guard(tmp_path, [snapshot(23456), snapshot(pid)], alive=False)
    with pytest.raises(AssertionError, match='foreign'):
        guard()


def test_live_owned_worker_is_still_observed(tmp_path):
    guard, _ = real_guard(tmp_path, [snapshot(23456), snapshot(23456)], alive=True)
    guard()


def test_owner_exit_between_inventory_and_application_check_forces_refresh(tmp_path):
    guard, _ = real_guard(tmp_path, [snapshot(23456), snapshot(None)], alive=True)
    observations = iter([True, False, False])
    guard.__globals__['same_process'] = lambda _: next(observations, False)
    guard()


@pytest.mark.parametrize('failure', [None, 'missing-completion', 'nonzero-exit', 'stale-source',
    'stale-stage', 'partial-input', 'wrong-phase', 'wrong-owner', 'missing-authorization', 'stale-authorization'])
def test_actual_owned_worker_admission_before_torch(capsule, failure):
    """Use real guardian/worker PIDs; only the admission boundary is executed."""
    import subprocess
    from egoemg_repro.emg2pose_memory_process import process_identity, run_owned_stage
    root = capsule
    controller = process_identity()
    dead = dict(pid=2147483646,start_ticks='1',uid=os.getuid(),sid=2147483646,pgrp=2147483646,state='S')
    guardian = dict(dead,pid=2147483645,sid=2147483645,pgrp=2147483645)
    registration = dict(owner=controller,worker=dead,guardian=guardian,descriptor_bindings={})
    put(root/'private/owned-preflight/started.json', registration)
    completion = dict(registration,status='success',worker_exit_code=0,all_owned_consumers_exited=True)
    if failure == 'nonzero-exit': completion['worker_exit_code'] = 1
    put(root/'private/owned-preflight/stage-result.json', completion)
    stage = json.loads((root/'private/stage-receipt.json').read_text())
    stage_sha = digest(root/'private/stage-receipt.json')
    phase = dict(phase='preflight',source_commit=COMMIT,bundle_sha256=stage['bundle_sha256'],
        stage_receipt_sha256=stage_sha,controller=controller,actual_owner={k:dead[k]for k in('pid','start_ticks')},
        actual_model_constructions=0,actual_optimizer_constructions=0,actual_optimizer_updates=0,
        actual_cuda_initialized=False,actual_gpu_forward_observations={},
        result=dict(identity=dict(sha256='c'*64),physical_input_bytes=16540457757,
                    input_paths={str(i):str(root/('input-'+str(i)))for i in range(16)},immutable_external_file_stamps={}))
    if failure == 'stale-source': phase['source_commit'] = 'f'*40
    if failure == 'stale-stage': phase['stage_receipt_sha256'] = 'f'*64
    if failure == 'partial-input': phase['result']['input_paths'].pop('0')
    put(root/'private/phase-preflight.json', phase)
    auth = dict(phase='qual-prefix',controller=controller,stage_receipt_sha256=stage_sha,
        review_sha256=digest(root/'private/final-review.json'),preflight=dict(
            phase_sha256=digest(root/'private/phase-preflight.json'),
            owned_completion_sha256=digest(root/'private/owned-preflight/stage-result.json')))
    if failure == 'wrong-phase': auth['phase'] = 'full-fit'
    if failure == 'wrong-owner': auth['controller'] = dead
    put(root/'private/authorization-qual-prefix.json', auth)
    env = dict(os.environ,EGOEMG_STAGE_RECEIPT_SHA256=stage_sha,
        EGOEMG_PHASE_AUTHORIZATION_SHA256=digest(root/'private/authorization-qual-prefix.json'),
        CUDA_VISIBLE_DEVICES=UUIDS[0],PYTHONPATH=str(REPO))
    if failure == 'missing-completion': (root/'private/owned-preflight/stage-result.json').unlink()
    if failure == 'missing-authorization': (root/'private/authorization-qual-prefix.json').unlink()
    if failure == 'stale-authorization': env['EGOEMG_PHASE_AUTHORIZATION_SHA256'] = 'f'*64
    # Compile the actual worker prefix as in the smaller probes; no Torch import.
    script = root/'boundary.py'
    script.write_text('import ast\nfrom pathlib import Path\n'
        + 'p=Path('+repr(str(REPO/'scripts/run_native_emgformer_worker.py'))+')\n'
        + 't=ast.parse(p.read_text())\n'
        + "i=next(i for i,n in enumerate(t.body)if isinstance(n,ast.Import)and n.names[0].name=='torch')\n"
        + 'ns=dict(__file__='+repr(str(root/'private/runtime-worker.py'))+')\n'
        + 'exec(compile(ast.Module(body=t.body[:i],type_ignores=[]),str(p),"exec"),ns)\n'
        + 'Path('+repr(str(root/'admission-reached'))+').write_text("before Torch")\n')
    if failure is None:
        result = run_owned_stage([sys.executable,str(script),'qual-prefix'],root/'private/owned-qual-prefix',
            descriptors={},guard=lambda:None,environment=env,poll_seconds=.01)
        assert result['worker_exit_code'] == 0
        assert (root/'admission-reached').read_text() == 'before Torch'
    else:
        from egoemg_repro.emg2pose_memory_process import OwnedStageError
        with pytest.raises(OwnedStageError):
            run_owned_stage([sys.executable,str(script),'qual-prefix'],root/'private/owned-qual-prefix',
                descriptors={},guard=lambda:None,environment=env,poll_seconds=.01)
        assert not (root/'admission-reached').exists()
    terminal=json.loads((root/'private/owned-qual-prefix/stage-result.json').read_text())
    assert terminal['all_owned_consumers_exited'] is True
