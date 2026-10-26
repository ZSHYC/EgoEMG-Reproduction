"""Bounded real Lightning tests for the independent classic protocol."""
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT = Path(__file__).parents[1]
UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', PROJECT / 'upstream')).resolve()
W = 7790


def rt():
    return importlib.import_module('egoemg_repro.classic_emg_train')


def tensor_runtime():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pytorch_lightning')
    pytest.importorskip('hydra')
    torch.set_num_threads(2)
    return torch


def corpus(root, *, empty=False):
    """Engineering fixture: one retained train, one val, one held-out window."""
    root.mkdir()
    n = W * 3
    arrays = dict(frame_split_id=np.repeat(np.array([0, 0, 1], dtype=np.int8), W),
                  generated_label_valid=np.full((n, 2), not empty, dtype=bool))
    for h, hand in enumerate(('left', 'right')):
        arrays['emg_' + hand + '_filtered'] = np.random.RandomState(10+h).normal(0, .05, (n, 8)).astype('f4')
        arrays['generated_joint_angles_' + hand] = np.full((n, 20), .03 + h * .01, dtype='f4')
        for axis in ('pitch', 'yaw'):
            arrays[f'mocap_{hand}_wrist_{axis}'] = np.full(n, 2 + h, dtype='f4')
        arrays[f'mocap_{hand}_wrist_angles_valid'] = np.zeros(n, dtype=bool)
    fields = {}
    for name, value in arrays.items():
        value.tofile(root / (name + '.dat'))
        fields[name] = dict(filename=name+'.dat', dtype=str(value.dtype), shape=list(value.shape))
    (root / 'manifest.json').write_text(json.dumps(dict(format_version='egoemg_v2_memmap', total_rows=n,
        fields=fields, episode_fields={}, frame_split_labels=['train', 'user', 'gesture', 'both'])))
    np.savez(root / 'metadata.npz', episode_id=np.array(['engineering'], dtype='S16'),
        episode_subject=np.array(['fixture'], dtype='S8'), episode_subject_id=np.array([0]),
        episode_source_parquet=np.array(['fixture.parquet'], dtype='S16'), episode_zed_video_path=np.array([''], dtype='S1'),
        episode_head_video_path=np.array([''], dtype='S1'), episode_start_idx=np.array([0]), episode_end_idx=np.array([n]),
        episode_length=np.array([n]), episode_beta_idx=np.array([0]), subjects_subject=np.array(['fixture'], dtype='S8'),
        splits_split=np.array(['train', 'user', 'gesture', 'both'], dtype='S8'), splits_split_id=np.arange(4))
    from egoemg_repro.classic_emg_data import build_classic_partitions
    from egoemg_repro.data import LegacyStore
    partitions = build_classic_partitions(LegacyStore(root), upstream=UPSTREAM)
    assert partitions['coverage']['train']['hand_samples'] == 2
    assert partitions['coverage']['val']['hand_samples'] == 2
    return partitions


@pytest.fixture(autouse=True)
def deterministic_capacity(monkeypatch):
    # Fixture artifacts are tiny; production reserve must not depend on CI host free space.
    module = rt()
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 1024**3))


def run(tmp_path, variant='neuropose', *, output='run', epochs=1, stop=None, resume=None, empty=False,
        max_steps=None, val_batches=None, data=None, partitions=None):
    tensor_runtime()
    if data is None:
        data = tmp_path / 'data'
        partitions = corpus(data, empty=empty)
    return rt().run_classic(data, upstream=UPSTREAM, config=PROJECT/'configs'/f'{variant}.json',
        partitions=partitions, output=tmp_path/output, engineering=True, device='cpu', batch_size=1,
        num_workers=0, engineering_epochs=epochs, stop_after_epoch=stop, max_steps=max_steps,
        limit_val_batches=val_batches, resume=resume)


@pytest.mark.parametrize('bad', ['device', 'batch', 'workers', 'epochs', 'steps', 'val'])
def test_engineering_policy_rejects_invalid_bounds(bad):
    options = dict(engineering=True, device='cpu', batch_size=1, num_workers=0, engineering_epochs=2,
                   max_steps=None, limit_val_batches=None)
    options[dict(device='device', batch='batch_size', workers='num_workers', epochs='engineering_epochs',
                 steps='max_steps', val='limit_val_batches')[bad]] = dict(device='cuda:0', batch=0,
                 workers=-1, epochs=101, steps=0, val=0)[bad]
    with pytest.raises(ValueError):
        rt().runtime_policy(**options)


def test_formal_run_is_gated_before_source_or_data_access(tmp_path):
    with pytest.raises(ValueError, match='CUDA|production|formal'):
        rt().run_classic(tmp_path/'missing', upstream=tmp_path/'missing', config=tmp_path/'missing',
            partitions={}, output=tmp_path/'run', engineering=False)
    assert not (tmp_path/'run').exists()


def test_capacity_preflight_preserves_reserve_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(rt().shutil, 'disk_usage', lambda p: SimpleNamespace(free=1024**3))
    with pytest.raises(OSError, match='15|space|reserve'):
        rt().run_classic(tmp_path/'missing', upstream=UPSTREAM, config=PROJECT/'configs/neuropose.json',
            partitions={}, output=tmp_path/'run', engineering=True)
    assert not (tmp_path/'run').exists()


def test_dynamic_capacity_failure_never_publishes_partial_artifact(tmp_path, monkeypatch):
    calls = []
    def capacity(path):
        calls.append(True)
        return SimpleNamespace(free=(100 if len(calls) < 3 else 1)*1024**3)
    monkeypatch.setattr(rt().shutil, 'disk_usage', capacity)
    with pytest.raises(OSError, match='reserve'):
        rt()._atomic(tmp_path/'checkpoint', lambda stream:stream.write(b'actual bytes'), 4096)
    assert not (tmp_path/'checkpoint').exists() and list(tmp_path.iterdir()) == []


def test_csv_logger_uses_atomic_capacity_guard_for_hparams_and_metrics(tmp_path, monkeypatch):
    tensor_runtime()
    logger = rt().ClassicCSVLogger(str(tmp_path), name='lightning', version='engineering')
    logger.log_hyperparams({'seed':42, 'purpose':'engineering'})
    logger.log_metrics({'val_loss':.5}, step=0)
    logger.save()
    log = tmp_path/'lightning/engineering/metrics.csv'
    before = log.read_bytes()
    monkeypatch.setattr(rt().shutil, 'disk_usage', lambda p:SimpleNamespace(free=1024**3))
    logger.log_metrics({'val_loss':.4}, step=1)
    with pytest.raises(OSError, match='reserve'):
        logger.save()
    assert log.read_bytes() == before


def test_actual_lightning_validation_uses_uneven_batch_weighting(tmp_path):
    torch = tensor_runtime()
    data = tmp_path/'data'; partitions = corpus(data)
    # Explicit engineering fixture: left has one generated-valid time, right all.
    labels = np.memmap(data/'generated_label_valid.dat', dtype=bool, mode='r+', shape=(3*W,2))
    labels[W:2*W,0] = False; labels[W,0] = True; labels.flush(); del labels
    prepared = rt().prepare_classic_run(data, upstream=UPSTREAM, config=PROJECT/'configs/neuropose.json',
        partitions=partitions, policy=rt().runtime_policy(engineering=True))
    from egoemg_repro.classic_emg_data import build_classic_datasets
    datasets = build_classic_datasets(data, partitions, upstream=UPSTREAM, config=prepared.cfg,
        normalization=prepared.recipe['normalization'], mask_policy='official_literal_loader')
    class ThreeEngineeringSamples:
        def __len__(self): return 3
        def __getitem__(self, index):
            item = datasets['val'][index%2]; item['classic_sample_index'] = index
            return item
    observer = rt().ClassicProtocolCallback(prepared.identity, {}, 0, 3)
    module = rt().build_classic_lightning(UPSTREAM, prepared.cfg, prepared.identity)
    import pytorch_lightning as pl
    trainer = pl.Trainer(accelerator='cpu', devices=1, precision='32-true', logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, callbacks=[observer])
    actual = trainer.validate(module, dataloaders=torch.utils.data.DataLoader(ThreeEngineeringSamples(), batch_size=2), verbose=False)
    event = observer.validation_events[-1]
    assert event['batch_sizes'] == [2,1] and event['full']
    assert event['valid_counts'] == [W+1,1]
    records = observer.val_losses
    expected = (2*records[0][1]+records[1][1])/3
    wrong = ((W+1)*records[0][1]+records[1][1])/(W+2)
    assert actual[0]['val_loss'] == pytest.approx(expected, rel=2e-6)
    assert abs(expected-wrong) > 1e-7


@pytest.mark.parametrize('variant', ['vemg2pose', 'neuropose'])
def test_real_official_module_keeps_tuple_22d_and_generated_mask(tmp_path, variant):
    torch = tensor_runtime()
    from egoemg_repro.classic_emg import compose_classic
    from egoemg_repro.classic_emg_data import build_classic_datasets, classic_normalization
    data = tmp_path/'data'; partitions = corpus(data)
    prepared = rt().prepare_classic_run(data, upstream=UPSTREAM, config=PROJECT/'configs'/f'{variant}.json',
        partitions=partitions, policy=rt().runtime_policy(engineering=True))
    cfg = compose_classic(UPSTREAM, variant)
    module = rt().build_classic_lightning(UPSTREAM, cfg, prepared.identity)
    assert type(module).__name__ == 'EmgPredictionModule' and module.automatic_optimization
    assert module.loss_weights == {'mae': 1, 'landmark/fingertip': .01, 'fingertip_distance': .01}
    datasets = build_classic_datasets(data, partitions, upstream=UPSTREAM, config=cfg,
        normalization=classic_normalization(UPSTREAM), mask_policy='official_literal_loader')
    batch = torch.utils.data.default_collate([datasets['val'][0]])
    original = module.model
    with torch.no_grad():
        pred, target, mask = original.eval()(batch)
    left = 1790 if variant == 'vemg2pose' else 0
    assert module.model is original
    assert pred.shape == target.shape == (1, 22, W-left)
    assert torch.equal(target, batch['joint_angles'][..., left:])
    assert mask.ndim == 2 and torch.equal(mask, batch['label_valid_mask'][..., left:])
    for damage in ('channels', 'nan', 'angle_mask'):
        bad = copy.deepcopy(batch)
        if damage == 'channels': bad['joint_angles'] = bad['joint_angles'][:, :20]
        elif damage == 'nan': bad['emg'][0, 0, 0] = float('nan')
        else: bad['angle_valid_mask'] = torch.ones_like(bad['joint_angles'], dtype=torch.bool)
        with pytest.raises(ValueError, match='schema|finite|mask'):
            original(bad)
    real_forward = original.forward
    good = (pred,target,mask)
    for bad in (pred, good[:2], (pred[:,:20],target,mask),
                (pred,target[...,1:],mask), (pred,target,mask[:,None,:]),
                (pred,target+1,mask), (pred,target,~mask)):
        original.forward = lambda *args, result=bad, **kw:result
        with pytest.raises(ValueError, match='tuple|schema|context|mask'):
            original(batch)
    original.forward = real_forward


def test_real_lightning_cpu_step_full_validation_and_complete_checkpoint(tmp_path):
    tensor_runtime()
    result = run(tmp_path)
    assert result['scientific_eligible'] is False and result['scope'] == 'engineering'
    assert result['optimizer_updates'] == result['global_step'] == 2
    assert result['completed_epochs'] == 1 and result['clip_value'] == 0
    assert all(e['gradient_finite'] and e['gradient_max'] > 0 and e['parameters_changed'] for e in result['updates'])
    event = result['validation_events'][-1]
    assert event['samples'] == 2 and event['full'] and event['selection_scope'] == 'engineering'
    assert event['val_loss'] == pytest.approx(event['batch_weighted_loss'], rel=2e-6)
    checkpoint = rt().read_classic_checkpoint(tmp_path/'run/checkpoints/last.ckpt', result['identity'])
    assert checkpoint['optimizer_states'] and checkpoint['lr_schedulers'] and checkpoint['loops']
    assert checkpoint['classic']['selected_best']['checkpoint']['classic']['role'] == 'best'
    assert checkpoint['classic']['rng']['torch'].numel() > 0
    assert checkpoint['lr_schedulers'][0]['T_max'] == 100
    assert checkpoint['lr_schedulers'][0]['last_epoch'] == 1
    assert not any(name.endswith('test') for name in result)


def test_own_json_identity_roundtrip_admits_checkpoint_in_new_process(tmp_path, strict_resume_source):
    tensor_runtime()
    base, data, partitions, result = strict_resume_source
    checkpoint = base/'run/checkpoints/last.ckpt'
    identity_path = base/'run/identity.json'
    loaded_identity = json.loads(identity_path.read_text())
    script = '''import json, sys
from pathlib import Path
from egoemg_repro.classic_emg_train import read_classic_checkpoint
identity=json.loads(Path(sys.argv[2]).read_text())
checkpoint=read_classic_checkpoint(Path(sys.argv[1]), identity)
assert checkpoint['classic']['identity'] == identity
assert isinstance(checkpoint['classic']['rng']['numpy'], tuple)
assert isinstance(checkpoint['classic']['rng']['python'], tuple)
print('own JSON identity / full typed state offline admission PASS')
'''
    child = subprocess.run([sys.executable,'-B','-c',script,str(checkpoint),str(identity_path)],
                           capture_output=True,text=True,timeout=60)
    assert child.returncode == 0, child.stdout+child.stderr
    assert 'offline admission PASS' in child.stdout
    assert loaded_identity == result['identity']
    rt().read_classic_checkpoint(checkpoint, loaded_identity)


def test_empty_label_batch_uses_real_zero_loss_automatic_optimizer(tmp_path):
    tensor_runtime()
    result = run(tmp_path, empty=True)
    assert result['optimizer_updates'] == 2
    assert all(e['loss'] == 0 and e['parameters_changed'] for e in result['updates'])
    assert result['validation_events'][-1]['val_loss'] == 0


def test_partial_validation_and_step_limit_never_formal_best(tmp_path):
    tensor_runtime()
    result = run(tmp_path, val_batches=1)
    assert result['scientific_eligible'] is False
    assert result['optimizer_updates'] == 2
    assert not result['validation_events'][-1]['full']
    assert result['selected_best'] is None
    with pytest.raises(ValueError, match='epoch|partial'):
        rt().read_classic_checkpoint(tmp_path/'run/checkpoints/last.ckpt', result['identity'])


def test_step_limit_before_validation_has_no_invented_evaluation(tmp_path):
    tensor_runtime()
    result = run(tmp_path, max_steps=1)
    assert result['scientific_eligible'] is False and result['optimizer_updates'] == 1
    assert result['validation_events'] == [] and result['selected_best'] is None
    with pytest.raises(ValueError, match='epoch|partial'):
        rt().read_classic_checkpoint(tmp_path/'run/checkpoints/last.ckpt', result['identity'])


def test_original_early_stopping_uses_loss_and_restores_wait_count():
    tensor_runtime()
    early = rt().classic_early_stopping()
    import torch
    trainer = SimpleNamespace(current_epoch=0, should_stop=False, strategy=SimpleNamespace(reduce_boolean_decision=lambda x, **kw:x))
    stop, _ = early._evaluate_stopping_criteria(torch.tensor(2.0))
    assert not stop
    for i in range(49):
        stop, _ = early._evaluate_stopping_criteria(torch.tensor(2.0))
        assert not stop
    restored = rt().classic_early_stopping(); restored.load_state_dict(early.state_dict())
    assert restored.wait_count == 49 and restored.best_score == 2
    stop, _ = restored._evaluate_stopping_criteria(torch.tensor(2.0))
    assert stop and restored.wait_count == 50
    assert restored.monitor == 'val_loss' and restored.mode == 'min' and restored.patience == 50
    untouched = rt().classic_early_stopping()
    from pytorch_lightning.trainer.states import TrainerFn
    untouched.on_validation_end(SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING), sanity_checking=True), None)
    assert untouched.wait_count == 0 and torch.isinf(untouched.best_score)
    stop, reason = untouched._evaluate_stopping_criteria(torch.tensor(float('nan')))
    assert stop and 'not finite' in reason


def test_opposite_mae_ranking_uses_batch_weighted_official_loss():
    # Candidate A is better on MAE but worse on total official double-key FK loss.
    a = [(1, .1 + .02 * 100, 100), (2, 1 + .02 * 100, 1)]
    b = [(1, .2 + .02 * 1, 1), (2, 1.1 + .02 * 1, 100)]
    assert rt().batch_weighted_loss(a) > rt().batch_weighted_loss(b)
    assert rt().batch_weighted_loss(a) == pytest.approx((2.1+2*3)/3)
    assert rt().batch_weighted_loss(a) != pytest.approx((100*2.1+3)/101)


@pytest.mark.parametrize('variant', ['vemg2pose', 'neuropose'])
def test_real_epoch_resume_matches_uninterrupted_optimizer_scheduler_rng_and_best(tmp_path, variant):
    torch = tensor_runtime()
    data = tmp_path/'data'; partitions = corpus(data)
    full = run(tmp_path, variant, output='full', epochs=2, data=data, partitions=partitions)
    pause = run(tmp_path, variant, output='pause', epochs=2, stop=1, data=data, partitions=partitions)
    offline = tmp_path/'offline-last.ckpt'
    shutil.copy2(tmp_path/'pause/checkpoints/last.ckpt', offline)
    (tmp_path/'pause').rename(tmp_path/'old-unavailable')
    resumed = run(tmp_path, variant, output='resumed', epochs=2, data=data, partitions=partitions, resume=offline)
    assert full['identity'] == pause['identity'] == resumed['identity']
    assert resumed['completed_epochs'] == 2 and resumed['optimizer_updates'] == full['optimizer_updates'] == 4
    assert resumed['updates'] == full['updates']
    assert resumed['validation_events'] == full['validation_events']
    a = rt().read_classic_checkpoint(tmp_path/'full/checkpoints/last.ckpt', full['identity'])
    b = rt().read_classic_checkpoint(tmp_path/'resumed/checkpoints/last.ckpt', resumed['identity'])
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'loops'):
        assert rt().state_digest(a[key]) == rt().state_digest(b[key]), key
    assert rt().state_digest(a['classic']['rng']) == rt().state_digest(b['classic']['rng'])
    assert a['classic']['selected_best']['checkpoint']['epoch'] == b['classic']['selected_best']['checkpoint']['epoch']
    early_a = next(v for k,v in a['callbacks'].items() if k.startswith('EarlyStopping'))
    early_b = next(v for k,v in b['callbacks'].items() if k.startswith('EarlyStopping'))
    assert rt().state_digest(early_a) == rt().state_digest(early_b)
    # Strict offline reconstruction comes from the carried snapshot, no old run path.
    assert (tmp_path/'resumed/checkpoints/recovered-best.ckpt').is_file() or b['epoch'] == b['classic']['selected_best']['checkpoint']['epoch']


@pytest.fixture(scope='module')
def strict_resume_source(tmp_path_factory):
    tensor_runtime()
    base = tmp_path_factory.mktemp('classic-task3-strict-source')
    data = base/'data'; partitions = corpus(data)
    # Explicit test-only deterministic capacity injection; no production switch.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rt().shutil, 'disk_usage', lambda p:SimpleNamespace(free=100*1024**3))
        result = run(base, epochs=2, stop=1, data=data, partitions=partitions)
    return base, data, partitions, result


@pytest.mark.parametrize('damage', ['identity', 'model_only', 'role', 'dtype', 'shape', 'key', 'optimizer', 'callbacks', 'truncated'])
def test_strict_resume_rejects_changed_identity_or_incomplete_state(tmp_path, damage, strict_resume_source):
    torch = tensor_runtime()
    base, data, partitions, result = strict_resume_source
    source = base/'run/checkpoints/last.ckpt'
    value = torch.load(source, map_location='cpu', weights_only=False)
    broken = tmp_path/'broken.ckpt'
    if damage == 'identity':
        value['classic']['identity']['recipe']['normalization']['std'] = 18.49
    elif damage == 'model_only': value = {'state_dict': value['state_dict']}
    elif damage == 'role': value = value['classic']['selected_best']['checkpoint']
    elif damage in ('dtype','shape','key'):
        name = next(iter(value['state_dict']))
        if damage == 'dtype': value['state_dict'][name] = value['state_dict'][name].double()
        elif damage == 'shape': value['state_dict'][name] = value['state_dict'][name].reshape(-1)[:1]
        else: del value['state_dict'][name]
    elif damage == 'optimizer': value['optimizer_states'][0]['state'] = {}
    elif damage == 'callbacks':
        name = next(k for k in value['callbacks'] if k.startswith('EarlyStopping'))
        del value['callbacks'][name]
    if damage != 'model_only':
        value['classic_seal'] = rt()._sealed(value)
    torch.save(value, broken)
    if damage == 'truncated': broken.write_bytes(broken.read_bytes()[:100])
    with pytest.raises(ValueError, match='identity|role|state|dtype|shape|key|optimizer|callback|truncated|unreadable'):
        run(tmp_path, output='resumed', epochs=2, data=data, partitions=partitions, resume=broken)


def test_changed_data_rejected_before_factory_on_resume(tmp_path, monkeypatch):
    tensor_runtime()
    data = tmp_path/'data'; partitions = corpus(data)
    run(tmp_path, epochs=2, stop=1, data=data, partitions=partitions)
    with (data/'emg_left_filtered.dat').open('r+b') as stream:
        value = stream.read(1); stream.seek(0); stream.write(bytes([value[0] ^ 1]))
    calls = []
    monkeypatch.setattr(rt(), 'build_classic_lightning', lambda *a, **kw:calls.append(True))
    with pytest.raises(ValueError, match='identity'):
        run(tmp_path, output='resumed', epochs=2, data=data, partitions=partitions,
            resume=tmp_path/'run/checkpoints/last.ckpt')
    assert calls == [] and not (tmp_path/'resumed').exists()


def test_failed_final_status_cannot_be_mistaken_for_finished_run(tmp_path, monkeypatch):
    tensor_runtime()
    real_write = rt()._write_json
    injected = []
    def fail_once(path, value):
        if Path(path).name == 'status.json' and value.get('state') == 'engineering_finished' and not injected:
            injected.append(True)
            raise OSError('injected ENOSPC final status')
        return real_write(path, value)
    monkeypatch.setattr(rt(), '_write_json', fail_once)
    with pytest.raises(OSError, match='ENOSPC'):
        run(tmp_path)
    assert injected and json.loads((tmp_path/'run/status.json').read_text())['state'] == 'failed'
    assert json.loads((tmp_path/'run/result.json').read_text())['scientific_eligible'] is False
    assert not (tmp_path/'run/completion.json').exists()


def test_post_replace_fsync_and_low_reserve_cannot_leave_finished_marker(tmp_path, monkeypatch):
    tensor_runtime()
    real_sync = rt()._fsync_dir
    injected = []
    marker = tmp_path/'run/status.json'
    def fail_after_publication(path):
        if (Path(path) == tmp_path/'run' and marker.exists() and not injected and
                json.loads(marker.read_text()).get('state') == 'engineering_finished'):
            injected.append(True)
            monkeypatch.setattr(rt().shutil,'disk_usage',lambda p:SimpleNamespace(free=1024**3))
            raise OSError('injected ENOSPC directory fsync after publication')
        return real_sync(path)
    monkeypatch.setattr(rt(), '_fsync_dir', fail_after_publication)
    unrelated = tmp_path/'old-valid-artifact'; unrelated.write_bytes(b'keep original')
    with pytest.raises(OSError, match='fsync'):
        run(tmp_path)
    assert injected
    assert not marker.exists() or json.loads(marker.read_text()).get('state') != 'engineering_finished'
    assert (tmp_path/'run/checkpoints/last.ckpt').is_file()
    assert (tmp_path/'run/result.json').is_file()
    assert unrelated.read_bytes() == b'keep original'


def test_nonfinite_loss_and_gradient_guards_precede_optimizer_update():
    torch = tensor_runtime()
    observer = rt().ClassicProtocolCallback({'policy':{'precision':'32-true'}}, {}, 0, 0)
    with pytest.raises(ValueError, match='loss.*finite'):
        observer.on_before_backward(None, None, torch.tensor(float('nan'), requires_grad=True))
    layer = torch.nn.Linear(1,1)
    layer.weight.grad = torch.full_like(layer.weight, float('inf'))
    before = layer.weight.detach().clone()
    with pytest.raises(ValueError, match='gradient.*finite'):
        observer.on_before_optimizer_step(None, layer, torch.optim.AdamW(layer.parameters()))
    assert torch.equal(before, layer.weight)
