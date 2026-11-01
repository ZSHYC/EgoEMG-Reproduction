"""Actual retained CPU fit lifecycle; never a synthetic full-benchmark positive."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIT = Path('/data/shihao/egoemg-native-emgformer-train-review-fixed-cpu-4c406a602e16-20261029/private/fit')
DATA = FIT.parent/'synthetic-input'
UPSTREAM = Path('/home/shihao/projects/EgoEMG-Reproduction/upstream')
FULL_FIT = Path('/data/shihao/egoemg-native-emgformer-reviewed-full-75c45c3fb6e8-20261101/private/full-fit')


def api():
    try: return importlib.import_module('egoemg_repro.emgformer_trajectory_eval')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.emgformer_trajectory_eval':
            pytest.fail('full native trajectory pass/profile/report lifecycle missing')
        raise


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_bytes():
    paths = [*DATA.iterdir(), FIT/'identity.json', FIT/'fit-result.json', FIT/'epochs.jsonl', *list((FIT/'checkpoints').glob('*.ckpt'))]
    return {str(p): (sha(p), p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths if p.is_file()}


@pytest.fixture(scope='module')
def chain(tmp_path_factory):
    m = api()
    for name in ('run_trajectory_pass', 'read_trajectory_pass', 'read_trajectory_profile',
                 'freeze_trajectory_validation', 'publish_trajectory_report', 'verify_trajectory_report',
                 'promote_trajectory_report', 'run_trajectory_workflow'):
        assert hasattr(m, name), 'new lifecycle API missing: '+name
    if not FIT.is_dir():
        if os.environ.get('EGOEMG_REQUIRE_TRAJECTORY_EVAL') == '1': pytest.fail('required retained actual engineering fit unavailable')
        pytest.skip('retained server engineering fit required')
    before = original_bytes(); root = tmp_path_factory.mktemp('actual-trajectory-chain')/'flow'
    script = ROOT/'scripts/paper/run_emgformer_trajectory.py'
    assert script.is_file(), 'guarded workflow CLI missing'
    command = [sys.executable, '-B', str(script), 'workflow', '--fit', str(FIT), '--data', str(DATA),
        '--upstream', str(UPSTREAM), '--engineering', '--output', str(root)]
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=850)
    root.parent.joinpath('actual-workflow.log').write_text(p.stdout+p.stderr)
    assert p.returncode == 0, (p.stdout+p.stderr)[-5000:]
    value = json.loads((root/'workflow-result.json').read_text())
    assert value['all_owned_consumers_exited'] and value['parent_torch_unimported']
    assert len(value['actual_owned_stages']) == 7
    yield root
    assert original_bytes() == before, 'original actual fit/input bytes or generations changed'


def run_owned(root, name, arguments):
    from egoemg_repro.emg2pose_memory_process import run_owned_stage
    stage = root/('owned-'+name)
    result = run_owned_stage([sys.executable, '-B', str(ROOT/'scripts/paper/run_emgformer_trajectory.py'), *arguments],
        stage, descriptors={}, guard=lambda: api()._sources(),
        environment=dict(os.environ, EGOEMG_TRAJECTORY_OWNED_STAGE=str(stage)), poll_seconds=.25)
    assert result['status'] == 'success' and result['worker_exit_code'] == 0 and result['all_owned_consumers_exited'], (stage/'worker.log').read_text()[-3000:]
    return result


def test_guarded_cli_import_has_no_work_or_torch_construction():
    m = api(); script = ROOT/'scripts/paper/run_emgformer_trajectory.py'
    assert script.is_file()
    code = "import runpy,sys;runpy.run_path(sys.argv[1],run_name='__mp_main__');assert 'torch' not in sys.modules"
    p = subprocess.run([sys.executable, '-B', '-c', code, str(script)], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[-2000:]


@pytest.mark.parametrize('role,kind,required', [('train', 'native', False), ('val', 'unknown', False), ('val', 'native', 1)])
def test_invalid_scope_or_unregistered_inputs_reject_before_models(role, kind, required, monkeypatch, tmp_path):
    m = api(); import torch
    def deny(*a, **kw): raise AssertionError('invalid request constructed model')
    monkeypatch.setattr(torch.nn.Module, '__init__', deny)
    with pytest.raises(ValueError): m.run_trajectory_pass(object(), FIT, tmp_path/'out', role=role, kind=kind, require_full=required)


def test_full_actual_cpu_role_chain_and_closed_independent_model_replay(chain):
    m = api()
    for role, samples in [('val', 4), ('test', 8)]:
        a = m.read_trajectory_pass(chain/(role+'-native')); b = m.read_trajectory_pass(chain/(role+'-replay'))
        assert a['samples'] == b['samples'] == samples and a['batch_size'] == b['batch_size'] == 2
        assert a['packet_content_sha256'] == b['packet_content_sha256'] and a['statistics'] == b['statistics']
        assert a['actor']['pid'] != b['actor']['pid'] and a['actual_trained_model_constructions'] == b['actual_trained_model_constructions'] == 1
        assert a['actual_model_forwards'] == b['actual_model_forwards'] == samples//2
        assert a['model_before_sha256'] == a['model_after_sha256'] == b['model_after_sha256']
        assert a['actual_optimizer_updates'] == b['actual_optimizer_updates'] == 0
        assert a['formal_evaluation'] is b['formal_evaluation'] is False
        assert a['native_metrics'] and b['native_metrics'] == {}
        assert len(a['statistics']['trajectory']['by_joint_group']) == 10
    profile = m.read_trajectory_profile(chain/'profile')
    test = m.read_trajectory_pass(chain/'test-native')
    assert profile['finished_ns'] <= test['started_ns'] and profile['prior_point_test_exposure_disclosed']
    value = m.verify_trajectory_report(chain/'test-replay', chain/'report')
    assert value['trajectory_pipeline_complete'] and value['scientific_eligible'] is False
    assert value['test']['unique_samples'] == 8 and value['training']['scope'] == 'cpu_engineering'
    assert (chain/'report/completion.json').is_file()


@pytest.fixture(scope='module')
def negative_native(chain):
    root = chain.parent/'negative-pass'; root.mkdir()
    run_owned(root, 'native', ['pass', '--fit', str(FIT), '--data', str(DATA), '--upstream', str(UPSTREAM),
        '--role', 'val', '--kind', 'native', '--engineering', '--output', str(root/'val-native')])
    return root/'val-native'


@pytest.mark.parametrize('damage', ['statistics', 'model_state', 'formal_flag', 'private_science_flag'])
def test_resealed_pass_fields_cannot_replace_actual_raw_fit_or_scope(negative_native, damage):
    m = api(); path = negative_native/'pass.json'; before = path.read_bytes(); value = json.loads(before)
    if damage == 'statistics': value['statistics']['trajectory']['overall']['mae_deg'] += 1
    elif damage == 'model_state': value['model_after_sha256'] = '0'*64
    elif damage == 'formal_flag': value['formal_evaluation'] = True
    else: value['scientific_eligible'] = True
    value = m._seal({k: v for k, v in value.items() if k != 'sha256'})
    try:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='statistics|state|formal|scope|private'):
            m.read_trajectory_pass(negative_native)
    finally: path.write_bytes(before)


@pytest.mark.parametrize('damage', ['same_actor', 'same_directory', 'wrong_parent'])
def test_fresh_pair_requires_different_closed_actors_paths_and_exact_parent(chain, damage):
    m = api(); a = m.read_trajectory_pass(chain/'val-native'); b = copy.deepcopy(m.read_trajectory_pass(chain/'val-replay'))
    if damage == 'same_actor': b['actor'] = a['actor']
    elif damage == 'same_directory': b['directory'] = a['directory']
    else: b['reference_file_sha256'] = '0'*64
    with pytest.raises(ValueError, match='fresh|replay|reference|actor'):
        m._ensure_fresh(a, b)


def test_report_is_pending_until_closed_independent_verifier(chain):
    m = api(); root = chain.parent/'pending-report'; root.mkdir()
    run_owned(root, 'report', ['report', '--test-replay', str(chain/'test-replay'), '--output', str(root/'report')])
    value = json.loads((root/'report/metrics.json').read_text())
    assert value['trajectory_pipeline_complete'] is False and value['scientific_eligible'] is False
    assert not (root/'report/completion.json').exists()
    from egoemg_repro.emg2pose_memory_process import process_identity
    actor = process_identity()
    fake = dict(status='success', worker_exit_code=0, all_owned_consumers_exited=True, worker=actor, guardian=actor, owner=actor)
    with pytest.raises(ValueError): m.promote_trajectory_report(root/'report', root/'missing-verification.json', owned_stage=fake)


def test_pretest_profile_and_formal_fit_gates_precede_model_construction(chain, monkeypatch, tmp_path):
    m = api(); import torch
    from egoemg_repro.emgformer_literal import compose_emgformer
    from egoemg_repro.emgformer_trajectory_data import admit_trajectory_inputs
    identity = json.loads((FIT/'identity.json').read_text())
    inputs = admit_trajectory_inputs(DATA, upstream=UPSTREAM, config=compose_emgformer(UPSTREAM, 'small'),
        normalization=identity['cohort']['normalization'], expected_point_identity=identity['cohort'])
    def deny(*a, **kw): raise AssertionError('pretest/fit gate constructed model')
    monkeypatch.setattr(torch.nn.Module, '__init__', deny)
    with pytest.raises(ValueError, match='profile|validation'):
        m.run_trajectory_pass(inputs, FIT, tmp_path/'premature', role='test', kind='native', require_full=False)
    with pytest.raises(ValueError, match='full|engineering'):
        m.run_trajectory_pass(inputs, FIT, tmp_path/'not-full', role='val', kind='native', require_full=True)
    with pytest.raises(ValueError, match='fresh|output|reference|directory'):
        m.run_trajectory_pass(inputs, FIT, chain/'val-native', role='val', kind='replay',
            reference=chain/'val-native', require_full=False)


def test_completed_public_report_still_rebuilds_without_models(chain, monkeypatch):
    m = api(); import torch
    def deny(*a, **kw): raise AssertionError('report verification constructed model or optimizer')
    monkeypatch.setattr(torch.nn.Module, '__init__', deny); monkeypatch.setattr(torch.optim.Optimizer, '__init__', deny)
    result = m.verify_trajectory_report(chain/'test-replay', chain/'report')
    assert result['trajectory_pipeline_complete'] and result['full_paper_complete'] is False


@pytest.mark.parametrize('damage', ['missing', 'file_bytes', 'resealed_source_subject'])
def test_actual_packet_and_original_source_are_required(negative_native, damage):
    m = api(); raw = negative_native/'raw/000000.pt'; journal = negative_native/'packets.jsonl'; path = negative_native/'pass.json'
    before = {p: p.read_bytes() for p in (raw, journal, path)}
    try:
        if damage == 'missing': raw.unlink()
        elif damage == 'file_bytes': raw.write_bytes(before[raw]+b'changed')
        else:
            records = [json.loads(line) for line in before[journal].splitlines()]
            records[0]['metadata']['user'][0] = 'substituted-subject'
            records[0] = m._seal({k: v for k, v in records[0].items() if k != 'sha256'})
            journal.write_text(''.join(json.dumps(r)+'\n' for r in records))
            value = json.loads(before[path]); value['packet_journal_sha256'] = sha(journal)
            value['packet_content_sha256'] = m._hash([m._hash(m._content(r)) for r in records])
            path.write_text(json.dumps(m._seal({k: v for k, v in value.items() if k != 'sha256'})))
        with pytest.raises(ValueError, match='trajectory|raw|packet|source|artifact'):
            m.read_trajectory_pass(negative_native)
    finally:
        for p, value in before.items(): p.write_bytes(value)


def test_six_original_loader_workers_close_after_consumer_failure(tmp_path):
    """Actual six workers, engineering labels only; no full-producer positive."""
    if not FIT.is_dir():
        if os.environ.get('EGOEMG_REQUIRE_TRAJECTORY_EVAL') == '1': pytest.fail('required actual engineering inputs unavailable')
        pytest.skip('retained server engineering inputs required')
    script = tmp_path/'six-workers.py'; result_file = tmp_path/'closed.json'
    script.write_text('''import json,sys
from pathlib import Path
import torch
from egoemg_repro.emgformer_trajectory_eval import _close_loader
from egoemg_repro.emgformer_literal import compose_emgformer
from egoemg_repro.emgformer_trajectory_data import admit_trajectory_inputs,trajectory_loader
from egoemg_repro.emg2pose_memory_process import process_identity
def main():
    torch.set_num_threads(2)
    fit,data,upstream,output=map(Path,sys.argv[1:])
    identity=json.loads((fit/'identity.json').read_text())
    inputs=admit_trajectory_inputs(data,upstream=upstream,config=compose_emgformer(upstream,'small'),normalization=identity['cohort']['normalization'],expected_point_identity=identity['cohort'])
    loader=trajectory_loader(inputs,'val',cpu_engineering=False)
    assert loader.num_workers==6 and loader.batch_size==500
    try:
        iterator=iter(loader); batch=next(iterator)
        assert len(batch['emg'])==4
        raise RuntimeError('injected consumer failure')
    except RuntimeError as error:
        assert str(error)=='injected consumer failure'
    finally:
        workers=_close_loader(loader)
    assert len(workers)==6 and all(process_identity(h['pid']) is None for h in workers)
    # Lightning's native teardown clears DataLoader._iterator. Keep the actual
    # pre-teardown Process objects/identities; no learned inference is performed.
    import pytorch_lightning as pl
    loader=trajectory_loader(inputs,'val',cpu_engineering=False);observed=[]
    class Consumer(pl.LightningModule):
        def validation_step(self,batch,index):
            iterator=loader._iterator
            observed.append((iterator,[(p,process_identity(p.pid))for p in iterator._workers]))
    trainer=pl.Trainer(accelerator='cpu',devices=1,precision='32-true',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
    trainer.validate(Consumer(),dataloaders=loader,verbose=False)
    assert loader._iterator is None and len(observed)==1 and len(observed[0][1])==6
    after_trainer=_close_loader(loader,snapshot=observed[0])
    assert len(after_trainer)==6 and all(process_identity(h['pid'])is None for h in after_trainer)
    output.write_text(json.dumps(dict(actual_workers=workers,after_trainer_workers=after_trainer,native_trainer_reset_iterator_observed=True,all_exited=True,scientific_eligible=False)))
if __name__=='__main__':main()
''')
    from egoemg_repro.emg2pose_memory_process import run_owned_stage
    deadline = time.monotonic()+180
    def guard():
        api()._sources()
        if time.monotonic() > deadline: raise TimeoutError('six-worker failure cleanup timeout')
    owned = run_owned_stage([sys.executable, '-B', str(script), str(FIT), str(DATA), str(UPSTREAM), str(result_file)],
        tmp_path/'owned-loader', descriptors={}, guard=guard)
    assert owned['all_owned_consumers_exited'] and owned['worker_exit_code'] == 0
    value = json.loads(result_file.read_text())
    assert value['all_exited'] and len(value['actual_workers']) == 6 and value['scientific_eligible'] is False
    assert value['native_trainer_reset_iterator_observed'] and len(value['after_trainer_workers']) == 6


def test_real_full_fit_cohort_receipt_generation_remains_bound(monkeypatch):
    """Read the genuine completed fit; simulate stat loss without changing it."""
    m = api()
    if not FULL_FIT.is_dir():
        if os.environ.get('EGOEMG_REQUIRE_TRAJECTORY_EVAL') == '1': pytest.fail('required genuine native200 fit unavailable')
        pytest.skip('genuine completed server fit required for receipt boundary')
    import torch
    def deny(*a, **kw): raise AssertionError('receipt-only admission constructed model or optimizer')
    monkeypatch.setattr(torch.nn.Module, '__init__', deny); monkeypatch.setattr(torch.optim.Optimizer, '__init__', deny)
    tx = m._Read(); result, identity, _ = m._fit(FULL_FIT, tx)
    assert result['completed_epochs'] == 200 and result['optimizer_executions'] == 25000
    binding = identity['cohort_receipt']; assert binding is not None
    receipt = Path(binding['path']); before = receipt.read_bytes(); nine = m._nine
    def changed(path):
        stamp = nine(path)
        if Path(path) == receipt: stamp[-1] += 1
        return stamp
    monkeypatch.setattr(m, '_nine', changed)
    with pytest.raises(ValueError, match='generation|source|artifact'):
        tx.check()
    assert receipt.read_bytes() == before
