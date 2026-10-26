"""Two real Lightning updates and exhaustive engineering-only reload checks.

The original val split and val transforms are deliberately retained. This is
neither independent validation nor paper performance or completed training.
All artifacts contain private provenance and belong in a private directory.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from . import emg2pose_data as storage
from . import emg2pose_runtime as runtime


FORMAT = 'emg2pose_two_update_engineering_v1'
IO_OVERHEAD_BYTES = 16 * 1024**2


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path, write, budget, *, replace=True):
    """Bound temporary bytes, retain15GiB throughout, sync before publication."""
    path = storage._safe_path(path)
    storage._space(path.parent, budget)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.incomplete', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb', buffering=0) as stream:
            class Writer:
                written = 0
                def write(self, value):
                    count = len(value)
                    if self.written + count > budget:
                        raise RuntimeError('engineering serialization exceeded its conservative payload budget')
                    storage._space(path.parent, count)
                    actual = stream.write(value)
                    if actual != count:
                        raise OSError('incomplete engineering output write')
                    self.written += actual
                    storage._space(path.parent)
                    return actual
                def flush(self):
                    stream.flush()
                def tell(self):
                    return stream.tell()
            writer = Writer()
            write(writer)
            stream.flush()
            os.fsync(stream.fileno())
        storage._space(path.parent)
        storage._safe_path(path)
        if replace:
            os.replace(temporary, path)
        else:
            # No prior checkpoint is replaced, even if a racing file appears.
            os.link(temporary, path)
            temporary.unlink()
        _sync_directory(path.parent)
        return writer.written
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path, value):
    payload = value.encode('utf-8')
    return _atomic_write(path, lambda stream: stream.write(payload), len(payload) + 4096)


def _write_json(path, value):
    return _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _tensor_payload_bytes(value, seen=None):
    import torch
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_tensor_payload_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_payload_bytes(v, seen) for v in value)
    return 0


def _checkpoint_io():
    import torch
    from pytorch_lightning.plugins.io import CheckpointIO

    class EngineeringIO(CheckpointIO):
        evidence = None
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            if storage_options is not None:
                raise ValueError('engineering CheckpointIO accepts only local storage')
            tensor_bytes = _tensor_payload_bytes(checkpoint)
            budget = 2 * tensor_bytes + IO_OVERHEAD_BYTES
            written = _atomic_write(Path(path), lambda stream: torch.save(checkpoint, stream), budget, replace=False)
            self.evidence = {'reserve_bytes': storage.MIN_FREE_BYTES, 'checkpoint_tensor_bytes': tensor_bytes,
                             'checkpoint_payload_budget_bytes': budget, 'checkpoint_actual_bytes': written,
                             'policy': 'single same-directory temporary checkpoint; per-write reserve; file+directory fsync; atomic no-replace publication'}
        def load_checkpoint(self, path, map_location=None):
            raise RuntimeError('use strict reload_engineering_checkpoint; Trainer resume is outside Task4')
        def remove_checkpoint(self, path):
            raise RuntimeError('engineering checkpoints are never automatically removed')

    return EngineeringIO()


def _source_sha():
    return storage._sha256(Path(__file__))


def _tensor_bytes(value):
    import torch
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _parameters_sha(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _rng_state(device):
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def _restore_rng(state, device):
    import torch
    if not isinstance(state, Mapping) or set(state) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('incomplete engineering RNG state')
    try:
        random.Random().setstate(state['python'])
        np.random.RandomState().set_state(state['numpy'])
        value = state['torch']
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.shape != torch.get_rng_state().shape:
            raise ValueError('invalid Torch RNG state')
        torch.Generator(device='cpu').set_state(value.cpu())
        if device.type == 'cuda':
            cuda = state['cuda']
            if not isinstance(cuda, torch.Tensor) or cuda.dtype != torch.uint8:
                raise ValueError('invalid CUDA RNG state')
            torch.Generator(device=device).set_state(cuda.cpu())
        elif state['cuda'] is not None:
            raise ValueError('CPU engineering state must not claim CUDA RNG')
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(value.cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda'].cpu(), device)
    except (TypeError, RuntimeError, KeyError) as exc:
        raise ValueError('invalid engineering RNG state') from exc


def _finite_batch(batch):
    import torch
    for name in ('emg', 'joint_angles'):
        if not torch.isfinite(batch[name]).all():
            raise ValueError('nonfinite engineering runtime input: ' + name)


def _move(batch, device):
    import torch
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _loader(dataset, dm):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, collate_fn=dm._collate_fn)


def _evaluate(module, dataset, dm, device, progress):
    """Stream every selected window; never retain predictions or limit batches."""
    import torch
    digests = {key: hashlib.sha256() for key in ('input', 'prediction', 'target', 'mask', 'window_order', 'metrics')}
    counts = dict(windows=0, prediction_elements=0, target_elements=0, mask_elements=0, valid_frames=0)
    sums = {}
    module.eval()
    with torch.inference_mode():
        for index, batch in enumerate(_loader(dataset, dm)):
            _finite_batch(batch)
            batch = _move(batch, device)
            prediction, target, mask = module(batch)
            if (prediction.shape != target.shape or tuple(prediction.shape) != (1, 20, 7280) or
                    tuple(mask.shape) != (1, 7280) or not torch.isfinite(prediction).all()):
                raise ValueError('invalid actual engineering prediction shape or finite state')
            for key, value in [('input', batch['emg']), ('prediction', prediction), ('target', target), ('mask', mask)]:
                digests[key].update(_tensor_bytes(value))
            order = [int(batch[key][0]) for key in ('session_idx', 'window_start_idx', 'window_end_idx')]
            digests['window_order'].update(json.dumps(order, separators=(',', ':')).encode() + b'\n')
            # These are the actual official metric objects, not a replacement
            # implementation or the scientific per-user aggregation in Task5.
            metrics = {}
            for metric in module.regression_metrics:
                metrics.update(metric(prediction, target, mask.bool(), 'engineering'))
            values = {key: float(value.detach().cpu()) for key, value in metrics.items()}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('nonfinite official engineering metric')
            digests['metrics'].update(json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + value
            counts['windows'] += 1
            counts['prediction_elements'] += prediction.numel()
            counts['target_elements'] += target.numel()
            counts['mask_elements'] += mask.numel()
            counts['valid_frames'] += int(mask.bool().sum())
            if (index + 1) % 25 == 0:
                progress(index + 1, len(dataset))
    if counts['windows'] != len(dataset) or not counts['windows']:
        raise RuntimeError('engineering evaluation did not consume every available window')
    progress(counts['windows'], len(dataset))
    return {**counts, **{key + '_sha256': digest.hexdigest() for key, digest in digests.items()},
            'all_windows_consumed': True,
            'metric_window_means_private': {key: value / counts['windows'] for key, value in sums.items()},
            'metric_aggregation': 'private unweighted window means only; not paper per-user performance'}


def _audit_callback(identity, check_sha, source_split, device, status):
    import torch
    from pytorch_lightning import Callback

    class Audit(Callback):
        def __init__(self):
            self.records = []

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            _finite_batch(batch)
            self.before = {name: value.detach().cpu().clone() for name, value in module.named_parameters()}
            self.before_sha = _parameters_sha(module)
            self.loss = None
            self.gradient_norm = None

        def on_before_backward(self, trainer, module, loss):
            if not torch.isfinite(loss) or loss.ndim != 0:
                raise ValueError('engineering loss is nonfinite or not scalar')
            self.loss = float(loss.detach().cpu())
            self.loss_terms = {key: float(trainer.callback_metrics[key].detach().cpu())
                               for key in ('train_mae', 'train_landmark/fingertip')}
            if not all(math.isfinite(v) for v in self.loss_terms.values()):
                raise ValueError('official weighted loss metrics are nonfinite')

        def on_after_backward(self, trainer, module):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError('engineering gradients are absent or nonfinite')
            self.gradient_norm = float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in gradients])).cpu())
            if not math.isfinite(self.gradient_norm) or self.gradient_norm <= 0:
                raise ValueError('engineering gradients have no finite nonzero norm')

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            changed = sum(not torch.equal(self.before[name], value.detach().cpu()) for name, value in module.named_parameters())
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            post_clip = float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in gradients])).cpu())
            if (self.loss is None or self.gradient_norm is None or changed <= 0 or
                    not math.isfinite(post_clip) or post_clip > 1.00001):
                raise RuntimeError('actual Lightning update or gradient clip was not observed')
            self.records.append({'global_step': trainer.global_step, 'loss': self.loss, 'gradient_finite': True,
                                 'official_loss_terms': self.loss_terms,
                                 'gradient_norm_before_clip': self.gradient_norm, 'gradient_norm_after_clip': post_clip,
                                 'changed_parameter_tensors': changed, 'before_parameters_sha256': self.before_sha,
                                 'after_parameters_sha256': _parameters_sha(module)})
            self.before = None
            status(stage='optimization', actual_updates=len(self.records))

        def on_save_checkpoint(self, trainer, module, checkpoint):
            checkpoint['emg2pose_engineering'] = {
                'format': FORMAT, 'engineering_only': True, 'completed_training': False,
                'runtime_identity': identity, 'check_source_sha256': check_sha,
                'source_split': source_split, 'actual_updates': trainer.global_step,
                'device': str(device), 'rng': _rng_state(device), 'updates': self.records,
                'optimizer_state_count': len(checkpoint['optimizer_states'][0]['state'])}

    return Audit()


def _restore_optimizer(module, checkpoint):
    import torch
    configured = module.configure_optimizers()
    optimizer, scheduler_spec = configured['optimizer'], configured['lr_scheduler']
    scheduler = scheduler_spec['scheduler']
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError('engineering runtime requires actual official AdamW/cosine')
    states, schedules = checkpoint['optimizer_states'], checkpoint['lr_schedulers']
    if not isinstance(states, list) or len(states) != 1 or not isinstance(schedules, list) or len(schedules) != 1:
        raise ValueError('incomplete optimizer/scheduler state')
    state, schedule = states[0], schedules[0]
    if set(state) != {'state', 'param_groups'} or len(state['param_groups']) != len(optimizer.param_groups):
        raise ValueError('optimizer structure differs')
    ids = {}
    for saved, actual in zip(state['param_groups'], optimizer.param_groups):
        if len(saved['params']) != len(actual['params']):
            raise ValueError('optimizer parameter groups differ')
        for key in ('betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused'):
            if saved.get(key) != actual.get(key):
                raise ValueError('optimizer recipe differs')
        for index, parameter in zip(saved['params'], actual['params']):
            if type(index) is not int or index in ids:
                raise ValueError('optimizer parameter IDs differ')
            ids[index] = parameter
    if (not state['state'] or len(state['state']) != checkpoint['emg2pose_engineering']['optimizer_state_count'] or
            not set(state['state']).issubset(ids)):
        raise ValueError('optimizer initialized parameter state differs')
    for index, value in state['state'].items():
        if set(value) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('incomplete AdamW moment state')
        parameter = ids[index]
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = value[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or
                    tensor.layout != parameter.layout or not torch.isfinite(tensor).all()):
                raise ValueError('AdamW moment shape/dtype/finite state differs')
        step = value['step']
        if (not isinstance(step, torch.Tensor) or step.ndim != 0 or step.dtype != torch.float32 or
                step.layout != torch.strided or not torch.isfinite(step) or float(step) != 2):
            raise ValueError('AdamW state does not contain exactly two updates')
    fresh = scheduler.state_dict()
    if (set(schedule) != set(fresh) or any(schedule[k] != fresh[k] for k in ('T_max', 'eta_min', 'base_lrs')) or
            type(schedule['last_epoch']) is not int or not 0 <= schedule['last_epoch'] <= 2 or
            schedule['_step_count'] != schedule['last_epoch'] + 1 or
            not isinstance(schedule['_last_lr'], list) or len(schedule['_last_lr']) != len(state['param_groups'])):
        raise ValueError('epoch cosine scheduler state differs')
    for lr, group, base in zip(schedule['_last_lr'], state['param_groups'], schedule['base_lrs']):
        expected = schedule['eta_min'] + (base - schedule['eta_min']) * (1 + math.cos(math.pi * schedule['last_epoch'] / schedule['T_max'])) / 2
        if not math.isfinite(lr) or lr != group['lr'] or not math.isclose(lr, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('optimizer/scheduler learning rates differ')
    optimizer.load_state_dict(state)
    scheduler.load_state_dict(schedule)
    return optimizer, scheduler


def reload_engineering_checkpoint(upstream: Path, config, checkpoint: Path, *, source_receipt: Path):
    """Strictly restore a trusted local full engineering checkpoint bundle."""
    import torch
    path = storage._safe_path(checkpoint, file=True)
    receipt = storage._json(path.parent / 'checkpoint-receipt.json')
    before = runtime._field_stamp(path)
    if (receipt.get('format') != FORMAT or receipt.get('checkpoint_filename') != path.name or
            receipt.get('checkpoint_bytes') != before[2] or storage._sha256(path) != receipt.get('checkpoint_sha256')):
        raise ValueError('engineering checkpoint receipt differs')
    try:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if runtime._field_stamp(path) != before:
            raise ValueError('checkpoint changed while loading')
        metadata = saved['emg2pose_engineering']
        identity = runtime.runtime_identity(upstream, config, source_receipt)
        if (metadata.get('format') != FORMAT or metadata.get('engineering_only') is not True or
                metadata.get('completed_training') is not False or metadata.get('runtime_identity') != identity or
                metadata.get('check_source_sha256') != _source_sha() or metadata.get('source_split') != 'val' or
                metadata.get('actual_updates') != 2 or saved['global_step'] != 2 or len(metadata['updates']) != 2 or
                not isinstance(saved['loops'], dict) or not saved['loops']):
            raise ValueError('engineering checkpoint scope/runtime/config/data identity differs')
        loops = saved['loops']
        fit_keys = {'state_dict', 'epoch_loop.state_dict', 'epoch_loop.batch_progress', 'epoch_loop.scheduler_progress',
                    'epoch_loop.automatic_optimization.state_dict', 'epoch_loop.automatic_optimization.optim_progress',
                    'epoch_loop.manual_optimization.state_dict', 'epoch_loop.manual_optimization.optim_step_progress',
                    'epoch_loop.val_loop.state_dict', 'epoch_loop.val_loop.batch_progress', 'epoch_progress'}
        if (set(loops) != {'fit_loop', 'validate_loop', 'test_loop', 'predict_loop'} or
                not isinstance(loops['fit_loop'], dict) or set(loops['fit_loop']) != fit_keys or
                loops['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] != 2 or
                loops['fit_loop']['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed'] != 2):
            raise ValueError('incomplete Lightning engineering loop state')
        module, dm = runtime.reload_runtime(upstream, config, saved['state_dict'], identity=identity)
        device = torch.device(metadata['device'])
        module.to(device)
        optimizer, scheduler = _restore_optimizer(module, saved)
        _restore_rng(metadata['rng'], device)
        if runtime._field_stamp(path) != before:
            raise ValueError('checkpoint changed during strict restoration')
        return SimpleNamespace(module=module, datamodule=dm, optimizer=optimizer, scheduler=scheduler,
                               checkpoint=saved, rng_restored=True)
    except (KeyError, TypeError, EOFError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError('invalid strict engineering checkpoint: ' + str(exc)) from exc


def run_engineering_check(upstream: Path, data: Path, output: Path, *, size: str, device: str,
                          source_split: str, source_receipt: Path, metadata_csv: Path | None = None,
                          updates: int = 2) -> dict:
    """Exactly two real updates, then two complete selected-val evaluations."""
    if source_split != 'val' or type(updates) is not int or updates != 2:
        raise ValueError('Task4 engineering requires original source_split=val and exactly two updates')
    if device != 'cpu' and not re.fullmatch(r'cuda:\d+', device):
        raise ValueError('device must be cpu or an explicit local cuda:index')
    output = storage._safe_path(output)
    if output.exists():
        raise FileExistsError('engineering output already exists')
    if not output.parent.is_dir():
        raise ValueError('engineering output parent must already exist')
    storage._space(output.parent, IO_OVERHEAD_BYTES)
    cfg = runtime.compose_config(upstream, size, data, source_receipt=source_receipt, metadata_csv=metadata_csv)
    identity = runtime.runtime_identity(upstream, cfg, source_receipt)
    if not identity['data']['engineering_only'] or source_split not in identity['data']['present_splits']:
        raise ValueError('Task4 accepts explicit engineering/mini val data only')
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    target_device = torch.device(device)
    if target_device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('requested local device cannot execute bf16-mixed')
    precision = '32-true' if device == 'cpu' else 'bf16-mixed'
    check_sha, config_before = _source_sha(), OmegaConf.to_yaml(cfg, resolve=True)
    module, dm = runtime.build_runtime(upstream, cfg)
    dm.setup('validate')
    dataset = dm.val_dataset
    if dataset is None or not len(dataset):
        raise ValueError('no original val windows available')
    # A full model + two AdamW moments, doubled for serialization overhead;
    # actual checkpoint contents receive a fresh exact-tensor budget at save.
    preflight_budget = 6 * _tensor_payload_bytes(module.state_dict()) + IO_OVERHEAD_BYTES
    storage._space(output.parent, preflight_budget)
    output.mkdir(mode=0o700)
    owned = output.stat()
    started = {'pid': os.getpid(), 'started_unix_ns': time.time_ns(), 'complete': False,
               'engineering_only': True, 'size': size, 'source_split': source_split, 'check_source_sha256': check_sha}
    proc = Path('/proc/self/stat')
    if proc.exists():
        started['process_start_ticks'] = proc.read_text().rsplit(')', 1)[1].split()[19]
    def status(**values):
        _write_json(output / 'status.json', {**started, **values})
    try:
        status(stage='constructed', available_windows=len(dataset))
        _write_text(output / 'resolved-config.yaml', config_before)
        _write_json(output / 'provenance.json', {'runtime_identity': identity, **started,
                                               'scope': 'private engineering only; original val optimized and reused for reload checks'})
        audit = _audit_callback(identity, check_sha, source_split, target_device, status)
        checkpoint_io = _checkpoint_io()
        trainer = Trainer(accelerator='cpu' if device == 'cpu' else 'gpu',
                          devices=1 if device == 'cpu' else [target_device.index],
                          precision=precision, max_steps=2, max_epochs=int(cfg.trainer.max_epochs),
                          gradient_clip_val=float(cfg.trainer.gradient_clip_val), gradient_clip_algorithm='norm',
                          callbacks=[audit], plugins=[checkpoint_io], logger=False, enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
                          deterministic=True, default_root_dir=str(output), log_every_n_steps=1)
        trainer.fit(module, train_dataloaders=_loader(dataset, dm))
        if trainer.global_step != 2 or len(audit.records) != 2:
            raise RuntimeError('Lightning did not execute exactly two actual updates')
        checkpoint = output / 'engineering.ckpt'
        trainer.save_checkpoint(checkpoint, weights_only=False)
        _write_json(output / 'checkpoint-receipt.json', {'format': FORMAT, 'checkpoint_filename': checkpoint.name,
                    'checkpoint_sha256': storage._sha256(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size})
        first = _evaluate(module, dataset, dm, target_device,
                          lambda count, total: status(stage='evaluation_before_reload', windows=count, total=total))
        restored = reload_engineering_checkpoint(upstream, cfg, checkpoint, source_receipt=source_receipt)
        restored.datamodule.setup('validate')
        second = _evaluate(restored.module, restored.datamodule.val_dataset, restored.datamodule, target_device,
                           lambda count, total: status(stage='evaluation_after_reload', windows=count, total=total))
        if first != second:
            raise RuntimeError('complete engineering evaluation differs after strict reload')
        if (OmegaConf.to_yaml(cfg, resolve=True) != config_before or _source_sha() != check_sha or
                runtime.runtime_identity(upstream, cfg, source_receipt) != identity):
            raise RuntimeError('source/config/data identity changed during engineering check')
        optimization = {'source_split': source_split, 'batch_size': 1, 'num_workers': 0, 'shuffle': False,
                        'transforms': 'unchanged official val transforms; no training augmentation',
                        'framework': 'pytorch_lightning.Trainer.fit', 'precision': precision, 'device': device,
                        'optimizer': type(trainer.optimizers[0]).__module__ + '.' + type(trainer.optimizers[0]).__name__,
                        'scheduler': type(trainer.lr_scheduler_configs[0].scheduler).__module__ + '.' + type(trainer.lr_scheduler_configs[0].scheduler).__name__,
                        'scheduler_interval': trainer.lr_scheduler_configs[0].interval,
                        'scheduler_state': trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                        'loss_weights': dict(module.loss_weights), 'metric_classes': [type(v).__name__ for v in module.regression_metrics],
                        'original_config_unchanged': True, 'official_forward_loss_unchanged': True,
                        'formal_recipe_differences': ['two steps only', 'batch1', 'val source with val transforms', 'zero loader workers',
                                                      'no validation loop or model selection', 'CPU float32 instead of bf16-mixed'] if device == 'cpu' else
                                                     ['two steps only', 'batch1', 'val source with val transforms', 'zero loader workers', 'no validation loop or model selection']}
        result = {'format': FORMAT, 'engineering_only': True, 'completed_training': False,
                  'independent_validation': False, 'selected_best': False, 'source_split': source_split,
                  'actual_updates': 2, 'updates': audit.records, 'optimization': optimization,
                  'runtime_identity': identity, 'check_source_sha256': check_sha,
                  'storage': {'preflight_payload_budget_bytes': preflight_budget, **checkpoint_io.evidence},
                  'checkpoint_sha256': storage._sha256(checkpoint), 'evaluations': [first, second],
                  'strict_reload_equal': True, 'rng_restored': restored.rng_restored,
                  'scope': 'same-val engineering optimization/reload only; not independent validation or paper result'}
        _write_json(output / 'engineering-result.json', result)
        status(stage='complete', complete=True, actual_updates=2, windows=first['windows'])
        # This durable marker is the last successful output commit. Any
        # exception, including its own fsync, removes this run's marker below.
        _write_json(output / 'completion.json', {'format': FORMAT, 'complete': True, 'engineering_only': True,
                    'result_sha256': storage._sha256(output / 'engineering-result.json'),
                    'checkpoint_sha256': result['checkpoint_sha256'], 'check_source_sha256': check_sha})
        return result
    except BaseException as exc:
        current = output.lstat()
        if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino) and not output.is_symlink():
            (output / 'completion.json').unlink(missing_ok=True)
            status_path = output / 'status.json'
            try:
                if status_path.exists() and storage._json(status_path).get('complete') is True:
                    status_path.unlink()
                _sync_directory(output)
                status(stage='failed', error_type=type(exc).__name__, error=str(exc))
            except (OSError, RuntimeError, ValueError):
                # A genuinely full device may prevent even a failure write.
                # Existing partial provenance remains; no success marker does.
                pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('upstream', 'data', 'output', 'source-receipt'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--metadata-csv', type=Path)
    parser.add_argument('--size', choices=tuple(runtime.SIZES), required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--source-split', choices=['val'], required=True)
    parser.add_argument('--updates', type=int, choices=[2], default=2)
    args = parser.parse_args(argv)
    result = run_engineering_check(args.upstream, args.data, args.output, size=args.size, device=args.device,
                                  source_split=args.source_split, source_receipt=args.source_receipt,
                                  metadata_csv=args.metadata_csv, updates=args.updates)
    print(json.dumps({'engineering_only': True, 'actual_updates': result['actual_updates'],
                      'evaluated_windows': result['evaluations'][0]['windows'], 'strict_reload_equal': result['strict_reload_equal']}))


if __name__ == '__main__':
    main()
