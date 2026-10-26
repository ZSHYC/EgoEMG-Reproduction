"""Real HDF5, unchanged official network/loss and Lightning engineering checks."""
import csv
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from egoemg_repro import emg2pose_data, emg2pose_runtime


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream'))


def check():
    return importlib.import_module('egoemg_repro.emg2pose_check')


def framework():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pytorch_lightning')
    pytest.importorskip('hydra')
    return torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module', autouse=True)
def sufficient_test_storage():
    # Tiny non-capacity cases must not depend on the CI host's free disk.
    # Capacity cases override this inside their function, after module setup.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(emg2pose_data.shutil, 'disk_usage', lambda path: SimpleNamespace(
            total=128 * 1024**3, used=64 * 1024**3, free=64 * 1024**3))
        yield


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp('private-engineering-val')
    count = 7790 * 3
    values = np.zeros(count, dtype=[('time', '<f8'), ('joint_angles', '<f4', (20,)), ('emg', '<f4', (16,))])
    values['time'] = np.arange(count) / 2000
    values['joint_angles'] = np.linspace(.15, .85, 20, dtype=np.float32)
    values['emg'] = np.sin(np.arange(count * 16, dtype=np.float32) / 83).reshape(count, 16)
    row = dict(filename='val-only', session='original-val-session', user='private-engineering-user',
               stage='private-engineering-stage', side='left', moving_hand='both', split='val',
               generalization='user', held_out_user=True, held_out_stage=False, start=0, end=count / 2000)
    path = root / 'val-only.hdf5'
    with h5py.File(path, 'w') as file:
        group = file.create_group('emg2pose')
        group.create_dataset('timeseries', data=values)
        for key, value in row.items():
            group.attrs[key] = value
        group.attrs['sample_rate'] = 2000.0
        group.attrs['num_channels'] = 16
    csv_path = root / 'metadata.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    receipt = root / 'original-inventory.json'
    receipt.write_text(json.dumps(emg2pose_data.inspect_corpus([path], csv_path, cohort='engineering', upstream=UPSTREAM)))
    output = root / 'complete-data'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(emg2pose_data.shutil, 'disk_usage', lambda path: SimpleNamespace(
            total=128 * 1024**3, used=64 * 1024**3, free=64 * 1024**3))
        emg2pose_data.convert_corpus(receipt, output, upstream=UPSTREAM)
    return SimpleNamespace(root=root, data=output, source_receipt=receipt, metadata_csv=csv_path)


@pytest.fixture(scope='module')
def completed(corpus):
    framework()
    output = corpus.root / 'engineering-result'
    result = check().run_engineering_check(UPSTREAM, corpus.data, output, size='small', device='cpu',
                                         source_split='val', source_receipt=corpus.source_receipt)
    return SimpleNamespace(result=result, output=output, corpus=corpus)


def test_val_only_mini_never_becomes_training_cohort(completed):
    result = completed.result
    assert result['engineering_only'] is True and result['completed_training'] is False
    assert result['independent_validation'] is False and result['selected_best'] is False
    assert result['source_split'] == result['optimization']['source_split'] == 'val'
    assert result['optimization']['transforms'] == 'unchanged official val transforms; no training augmentation'
    assert result['optimization']['batch_size'] == 1
    assert result['runtime_identity']['data']['present_splits'] == ['val']
    assert result['runtime_identity']['data']['full_corpus_complete'] is False
    with np.load(completed.corpus.data / 'metadata.npz', allow_pickle=False) as md:
        assert md['splits_split'].tolist() == [b'val']
    assert json.loads((completed.output / 'completion.json').read_text())['complete'] is True


def test_actual_update_and_strict_reload_preserve_predictions(completed):
    torch = framework()
    result = completed.result
    assert result['actual_updates'] == 2
    assert result['optimization']['framework'] == 'pytorch_lightning.Trainer.fit'
    assert result['optimization']['optimizer'] == 'torch.optim.adamw.AdamW'
    assert result['optimization']['scheduler'] == 'torch.optim.lr_scheduler.CosineAnnealingLR'
    assert result['optimization']['scheduler_interval'] == 'epoch'
    assert result['optimization']['loss_weights'] == {'mae': 1, 'landmark/fingertip': .01}
    assert {'AngleMAE', 'LandmarkDistances'} <= set(result['optimization']['metric_classes'])
    assert result['optimization']['precision'] == '32-true'
    assert [v['global_step'] for v in result['updates']] == [1, 2]
    for step in result['updates']:
        assert np.isfinite(step['loss']) and np.isfinite(step['gradient_norm_before_clip'])
        assert step['gradient_norm_before_clip'] > 0 and step['gradient_finite']
        assert step['gradient_norm_after_clip'] <= 1.00001
        assert step['changed_parameter_tensors'] > 0
        assert step['before_parameters_sha256'] != step['after_parameters_sha256']
        assert math_isclose(step['loss'], step['official_loss_terms']['train_mae'] + .01 * step['official_loss_terms']['train_landmark/fingertip'])
    cfg = emg2pose_runtime.compose_config(UPSTREAM, 'small', completed.corpus.data,
                                        source_receipt=completed.corpus.source_receipt)
    restored = check().reload_engineering_checkpoint(UPSTREAM, cfg, completed.output / 'engineering.ckpt',
                                                     source_receipt=completed.corpus.source_receipt)
    assert restored.rng_restored is True
    assert random.getstate() == restored.checkpoint['emg2pose_engineering']['rng']['python']
    actual_numpy = np.random.get_state()
    saved_numpy = restored.checkpoint['emg2pose_engineering']['rng']['numpy']
    assert actual_numpy[0] == saved_numpy[0] and np.array_equal(actual_numpy[1], saved_numpy[1])
    assert actual_numpy[2:] == saved_numpy[2:]
    assert torch.equal(torch.get_rng_state(), restored.checkpoint['emg2pose_engineering']['rng']['torch'])
    assert restored.checkpoint['global_step'] == 2
    assert all(int(state['step']) == 2 for state in restored.optimizer.state.values())
    assert restored.scheduler.state_dict() == restored.checkpoint['lr_schedulers'][0]
    original, _ = emg2pose_runtime.build_runtime(UPSTREAM, cfg)
    assert any(not torch.equal(p, original.state_dict()[name]) for name, p in restored.module.state_dict().items())


def test_full_engineering_evaluation_is_not_bounded(completed):
    first, second = completed.result['evaluations']
    assert first == second
    assert first['windows'] == 3
    assert first['prediction_elements'] == first['target_elements'] == 3 * 20 * 7280
    assert first['mask_elements'] == first['valid_frames'] == 3 * 7280
    assert len(first['prediction_sha256']) == len(first['target_sha256']) == len(first['mask_sha256']) == 64
    assert first['all_windows_consumed'] is True
    assert completed.result['strict_reload_equal'] is True
    assert completed.result['optimization']['original_config_unchanged'] is True
    torch = framework()
    cfg = emg2pose_runtime.compose_config(UPSTREAM, 'small', completed.corpus.data,
                                        source_receipt=completed.corpus.source_receipt)
    restored = check().reload_engineering_checkpoint(UPSTREAM, cfg, completed.output / 'engineering.ckpt',
                                                     source_receipt=completed.corpus.source_receipt)
    restored.module.eval()
    restored.datamodule.setup('validate')
    prediction_hash, target_hash, mask_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    with torch.inference_mode():
        for index in range(3):
            batch = restored.datamodule._collate_fn([restored.datamodule.val_dataset[index]])
            prediction, target, mask = restored.module(batch)
            prediction_hash.update(prediction.numpy().tobytes())
            target_hash.update(target.numpy().tobytes())
            mask_hash.update(mask.numpy().tobytes())
    assert first['prediction_sha256'] == prediction_hash.hexdigest()
    assert first['target_sha256'] == target_hash.hexdigest()
    assert first['mask_sha256'] == mask_hash.hexdigest()
    with pytest.raises(TypeError):
        check().run_engineering_check(UPSTREAM, completed.corpus.data, completed.output, size='small',
                                     device='cpu', source_split='val', source_receipt=completed.corpus.source_receipt,
                                     max_batches=1)


@pytest.mark.parametrize('damage', ['missing_model_key', 'wrong_head_shape', 'wrong_tensor_dtype',
                                   'missing_optimizer', 'wrong_optimizer_moment', 'missing_scheduler',
                                   'wrong_scheduler', 'truncated_scheduler_lr', 'missing_fit_loop', 'wrong_optimizer_step_dtype',
                                   'missing_rng', 'wrong_rng', 'wrong_size',
                                   'wrong_config', 'wrong_data', 'wrong_runtime', 'truncated_bytes'])
def test_truncated_state_or_wrong_identity_fails_strictly(completed, tmp_path, damage):
    torch = framework()
    cfg = emg2pose_runtime.compose_config(UPSTREAM, 'small', completed.corpus.data,
                                        source_receipt=completed.corpus.source_receipt)
    checkpoint = tmp_path / 'engineering.ckpt'
    shutil.copyfile(completed.output / 'engineering.ckpt', checkpoint)
    receipt = json.loads((completed.output / 'checkpoint-receipt.json').read_text())
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if damage == 'missing_model_key':
        del state['state_dict'][next(iter(state['state_dict']))]
    elif damage in {'wrong_head_shape', 'wrong_tensor_dtype'}:
        name = next(k for k, v in state['state_dict'].items() if v.is_floating_point() and v.ndim > 1)
        value = state['state_dict'][name]
        state['state_dict'][name] = value[..., :1] if damage == 'wrong_head_shape' else value.double()
    elif damage == 'missing_optimizer':
        state['optimizer_states'] = []
    elif damage == 'wrong_optimizer_moment':
        entry = next(iter(state['optimizer_states'][0]['state'].values()))
        entry['exp_avg'] = entry['exp_avg'].flatten()[:1]
    elif damage == 'wrong_optimizer_step_dtype':
        entry = next(iter(state['optimizer_states'][0]['state'].values()))
        entry['step'] = entry['step'].double()
    elif damage == 'missing_scheduler':
        state['lr_schedulers'] = []
    elif damage == 'wrong_scheduler':
        state['lr_schedulers'][0]['T_max'] = 1
    elif damage == 'truncated_scheduler_lr':
        state['lr_schedulers'][0]['_last_lr'] = []
    elif damage == 'missing_fit_loop':
        del state['loops']['fit_loop']
    elif damage == 'missing_rng':
        del state['emg2pose_engineering']['rng']
    elif damage == 'wrong_rng':
        state['emg2pose_engineering']['rng']['torch'] = torch.zeros(1, dtype=torch.float32)
    elif damage == 'truncated_bytes':
        pass
    else:
        identity = state['emg2pose_engineering']['runtime_identity']
        if damage == 'wrong_size':
            identity['size'] = 'large'
        elif damage == 'wrong_config':
            identity['config_sha256'] = '0' * 64
        elif damage == 'wrong_data':
            identity['data']['metadata_sha256'] = '0' * 64
        else:
            identity['runtime_files_sha256']['egoemg_repro/emg2pose_runtime.py'] = '0' * 64
    if damage == 'truncated_bytes':
        checkpoint.write_bytes(checkpoint.read_bytes()[:200])
    else:
        torch.save(state, checkpoint)
    # Bind changed bytes deliberately: rejection must also validate structure
    # and live runtime identity, rather than stopping at a stale file digest.
    receipt['checkpoint_sha256'] = digest(checkpoint)
    receipt['checkpoint_bytes'] = checkpoint.stat().st_size
    (tmp_path / 'checkpoint-receipt.json').write_text(json.dumps(receipt))
    with pytest.raises((ValueError, RuntimeError)):
        check().reload_engineering_checkpoint(UPSTREAM, cfg, checkpoint, source_receipt=completed.corpus.source_receipt)


@pytest.mark.parametrize(('source_split', 'updates'), [('train', 2), ('test', 2), ('val', 1), ('val', 3)])
def test_scope_cannot_relabel_val_or_expand_updates(corpus, tmp_path, source_split, updates):
    with pytest.raises(ValueError):
        check().run_engineering_check(UPSTREAM, corpus.data, tmp_path / 'result', size='small', device='cpu',
                                     source_split=source_split, updates=updates, source_receipt=corpus.source_receipt)
    assert not (tmp_path / 'result').exists()


def test_existing_output_is_not_overwritten(completed):
    before = digest(completed.output / 'completion.json')
    with pytest.raises(FileExistsError):
        check().run_engineering_check(UPSTREAM, completed.corpus.data, completed.output, size='small', device='cpu',
                                     source_split='val', source_receipt=completed.corpus.source_receipt)
    assert digest(completed.output / 'completion.json') == before


def test_low_storage_refuses_run_before_creating_output(corpus, tmp_path, monkeypatch):
    framework()
    monkeypatch.setattr(emg2pose_data.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024**3))
    output = tmp_path / 'low-storage-result'
    with pytest.raises(RuntimeError, match='storage|15 GiB|reserve'):
        check().run_engineering_check(UPSTREAM, corpus.data, output, size='small', device='cpu',
                                     source_split='val', source_receipt=corpus.source_receipt)
    assert not output.exists()


def test_dynamic_storage_drop_prevents_checkpoint_and_completion(corpus, tmp_path, monkeypatch):
    framework()
    from pytorch_lightning import Trainer
    free = [64 * 1024**3]
    monkeypatch.setattr(emg2pose_data.shutil, 'disk_usage', lambda path: SimpleNamespace(free=free[0]))
    original = Trainer.save_checkpoint
    def drop_then_save(self, *args, **kwargs):
        free[0] = emg2pose_data.MIN_FREE_BYTES + 1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Trainer, 'save_checkpoint', drop_then_save)
    output = tmp_path / 'dynamic-storage-result'
    with pytest.raises(RuntimeError, match='storage|15 GiB|reserve'):
        check().run_engineering_check(UPSTREAM, corpus.data, output, size='small', device='cpu',
                                     source_split='val', source_receipt=corpus.source_receipt)
    assert not (output / 'engineering.ckpt').exists()
    assert not (output / 'completion.json').exists()
    assert json.loads((output / 'status.json').read_text())['complete'] is False


def test_final_status_failure_cannot_leave_success_marker(corpus, tmp_path, monkeypatch):
    framework()
    rt = check()
    original = rt._write_json
    def fail_final_status(path, value):
        if path.name == 'status.json' and value.get('stage') == 'complete':
            raise OSError(errno.ENOSPC, 'injected final status ENOSPC')
        return original(path, value)
    monkeypatch.setattr(rt, '_write_json', fail_final_status)
    output = tmp_path / 'final-status-failure'
    with pytest.raises(OSError, match='ENOSPC|No space'):
        rt.run_engineering_check(UPSTREAM, corpus.data, output, size='small', device='cpu',
                                 source_split='val', source_receipt=corpus.source_receipt)
    assert not (output / 'completion.json').exists()
    status = json.loads((output / 'status.json').read_text())
    assert status['stage'] == 'failed' and status['complete'] is False


def test_json_fsync_failure_does_not_replace_existing_bytes(tmp_path, monkeypatch):
    rt = check()
    path = tmp_path / 'existing.json'
    path.write_text('old durable bytes')
    def fail_fsync(descriptor):
        raise OSError(errno.ENOSPC, 'injected file fsync ENOSPC')
    monkeypatch.setattr(rt.os, 'fsync', fail_fsync)
    with pytest.raises(OSError, match='ENOSPC|No space'):
        rt._write_json(path, {'new': True})
    assert path.read_text() == 'old durable bytes'
    assert list(tmp_path.iterdir()) == [path]


def test_json_output_preserves_reserve_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(emg2pose_data.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024**3))
    path = tmp_path / 'not-created.json'
    with pytest.raises(RuntimeError, match='storage|15 GiB|reserve'):
        check()._write_json(path, {'new': True})
    assert not path.exists()


def test_completion_is_the_last_output_commit(corpus, tmp_path, monkeypatch):
    framework()
    rt = check()
    original = rt._write_json
    written = []
    def observe(path, value):
        result = original(path, value)
        written.append(path.name)
        return result
    monkeypatch.setattr(rt, '_write_json', observe)
    output = tmp_path / 'commit-order'
    result = rt.run_engineering_check(UPSTREAM, corpus.data, output, size='small', device='cpu',
                                      source_split='val', source_receipt=corpus.source_receipt)
    assert written[-1] == 'completion.json'
    actual_bytes = (output / 'engineering.ckpt').stat().st_size
    assert result['storage']['checkpoint_payload_budget_bytes'] >= actual_bytes
    assert result['storage']['reserve_bytes'] == 15 * 1024**3


def math_isclose(a, b):
    return bool(np.isclose(a, b, rtol=1e-6, atol=1e-7))


def test_cli_requires_explicit_source_receipt_and_split():
    result = subprocess.run([sys.executable, '-m', 'egoemg_repro.emg2pose_check', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--source-split' in result.stdout and '--source-receipt' in result.stdout
    result = subprocess.run([sys.executable, '-m', 'egoemg_repro.emg2pose_check', '--max-batches', '1'], capture_output=True, text=True)
    assert result.returncode == 2
