"""Independent, bounded classic Lightning training and strict epoch restore.

The official module, model tuple, 2-D mask, metrics, loss and automatic optimizer
are unchanged. CPU runs are engineering evidence. Formal CUDA production remains
gated on a separate actual FP16/GradScaler qualification and Task4 reporting.
Checkpoints are trusted local pickle artifacts, not an untrusted upload format.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io as text_io
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from . import classic_emg as model_runtime
from . import classic_emg_data as data_runtime
from .data import file_sha256

try:  # Metadata/capacity tests also run in the intentionally Torch-free tier.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.plugins.io import CheckpointIO
    from pytorch_lightning.loggers import CSVLogger
except ImportError:
    pl = None
    Callback = CheckpointIO = CSVLogger = object


FORMAT = 'classic_literal_lightning_epoch_v1'
RESERVE_BYTES = 15 * 1024**3
IO_OVERHEAD = 32 * 1024**2
LOG_LIMIT_BYTES = 16 * 1024**2
IMPORTED_SOURCE_SHA256 = file_sha256(Path(__file__))
FRAMEWORK_FILES = ('loops/fit_loop.py','loops/training_epoch_loop.py','loops/fetchers.py',
    'callbacks/model_checkpoint.py','callbacks/early_stopping.py','plugins/precision/amp.py',
    'trainer/connectors/checkpoint_connector.py')


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _runtime():
    if pl is None:
        raise ImportError('classic training requires the declared pytorch-lightning runtime')
    import torch
    return torch


def runtime_policy(*, engineering, device='cpu', batch_size=1, num_workers=0, engineering_epochs=1,
                   max_steps=None, limit_val_batches=None):
    """Exact engineering policy, separate from the immutable official recipe."""
    if engineering is not True:
        raise ValueError('formal CUDA production is gated pending real FP16 qualification and Task4')
    if device != 'cpu':
        raise ValueError('this qualified engineering policy uses CPU32; CUDA FP16 is unqualified')
    for key, value, lo, hi in (('batch_size', batch_size, 1, 600), ('num_workers', num_workers, 0, 0),
                               ('engineering_epochs', engineering_epochs, 1, 100)):
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError('invalid engineering ' + key)
    for key, value in (('max_steps', max_steps), ('limit_val_batches', limit_val_batches)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError('invalid engineering ' + key)
    return dict(scope='engineering', scientific_eligible=False, device=device, precision='32-true',
                batch_size=batch_size, num_workers=num_workers, max_epochs=engineering_epochs,
                max_steps=max_steps, limit_val_batches=limit_val_batches,
                world_size=1, global_batch_size=batch_size, clip_value=0,
                loader_rng='separate seeded loader and train-sampler generators, checkpointed at epoch boundary',
                restart='epoch only; workers0 bitwise evidence required; no multiworker or mid-epoch promise',
                logger='actual Lightning CSVLogger replacing unavailable TensorBoard logging only',
                formal_precision='16-mixed GradScaler not yet qualified; no implicit CPU fallback')


def _safe(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('classic output/checkpoint path must not contain symlinks')
    return path


def _space(path, budget=0):
    path = _safe(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < RESERVE_BYTES + budget:
        raise OSError('classic output requires 15GiB reserve plus temporary payload space')


def _fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(path, write, budget, *, replace=True):
    path = _safe(path)
    _space(path.parent, budget)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name+'.', suffix='.incomplete')
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb', buffering=0) as stream:
            class Writer:
                written = 0
                def write(self, payload):
                    if self.written + len(payload) > budget:
                        raise OSError('classic serialization exceeded conservative payload budget')
                    _space(path.parent, len(payload))
                    n = stream.write(payload)
                    if n != len(payload):
                        raise OSError('short classic artifact write')
                    self.written += n
                    _space(path.parent)
                    return n
                def flush(self):
                    stream.flush()
                def tell(self):
                    return stream.tell()
            writer = Writer()
            write(writer)
            stream.flush()
            os.fsync(stream.fileno())
        _space(path.parent)
        _safe(path)
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    _atomic(path, lambda stream: stream.write(payload), len(payload)+4096)


class ClassicCSVLogger(CSVLogger):
    """Real CSVLogger accumulation; bounded atomic local serialization only."""
    def save(self):
        _runtime()
        experiment = self.experiment
        directory = _safe(experiment.log_dir)
        _space(directory, LOG_LIMIT_BYTES)
        directory.mkdir(parents=True, exist_ok=True)
        from omegaconf import OmegaConf
        hparams = OmegaConf.to_container(OmegaConf.create(experiment.hparams), resolve=True)
        payload = (json.dumps(hparams, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
        if len(payload) > LOG_LIMIT_BYTES:
            raise OSError('classic hyperparameter log exceeds 16MiB bound')
        _atomic(directory/'hparams.json', lambda stream:stream.write(payload), len(payload)+4096)
        if not experiment.metrics:
            return
        path = _safe(experiment.metrics_file_path)
        rows = []
        if path.exists():
            if path.stat().st_size > LOG_LIMIT_BYTES:
                raise OSError('classic CSV log exceeds 16MiB bound')
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
        rows.extend(experiment.metrics)
        keys = sorted({key for row in rows for key in row})
        stream = text_io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
        payload = stream.getvalue().encode()
        if len(payload) > LOG_LIMIT_BYTES:
            raise OSError('classic CSV log exceeds 16MiB bound')
        _atomic(path, lambda output:output.write(payload), len(payload)+4096)
        experiment.metrics = []
        experiment.metrics_keys = keys


def _payload_bytes(value, seen=None):
    torch = _runtime()
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_payload_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_bytes(v, seen) for v in value)
    return 0


def state_digest(value):
    """Typed tensor/state content digest, independent of pickle storage IDs."""
    torch = _runtime()
    from omegaconf import OmegaConf
    digest = hashlib.sha256()
    def visit(item):
        if OmegaConf.is_config(item):
            visit(OmegaConf.to_container(item, resolve=True)); return
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(('tensor:'+str(tensor.dtype)+':'+str(tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()); return
        if isinstance(item, np.ndarray):
            digest.update(('numpy:'+item.dtype.str+':'+str(item.shape)).encode()); digest.update(item.tobytes()); return
        if isinstance(item, dict):
            digest.update(b'dict{')
            for key in sorted(item, key=lambda k:(type(k).__name__, str(k))):
                visit(key); visit(item[key])
            digest.update(b'}'); return
        if isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode()+b'[')
            for child in item: visit(child)
            digest.update(b']'); return
        if isinstance(item, type):
            digest.update(('type:'+item.__module__+'.'+item.__qualname__).encode()); return
        if isinstance(item, (str, bytes, bool, int, float, type(None), np.generic)):
            digest.update((type(item).__name__+':'+repr(item)+';').encode()); return
        raise ValueError('unsupported checkpoint state type: '+type(item).__name__)
    visit(value)
    return digest.hexdigest()


def _rng(generators):
    torch = _runtime()
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                generators={name:g.get_state() for name,g in generators.items()})


def _restore_rng(state, generators):
    torch = _runtime()
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    if set(state['generators']) != set(generators):
        raise ValueError('checkpoint loader RNG streams differ')
    for name, generator in generators.items():
        generator.set_state(state['generators'][name])


def batch_weighted_loss(records):
    """Audit actual logged losses; third valid-count field is deliberately unused."""
    denominator = sum(record[0] for record in records)
    if not denominator:
        raise ValueError('validation has no samples')
    return sum(record[0]*record[1] for record in records)/denominator


@dataclass
class PreparedClassic:
    data: Path
    upstream: Path
    cfg: object
    recipe: dict
    partitions: dict
    identity: dict
    stamps: dict


def _stamp(path):
    item = _safe(path).stat()
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)


def _check_sources(identity):
    root = Path(__file__).resolve().parents[1]
    if file_sha256(Path(__file__)) != IMPORTED_SOURCE_SHA256:
        raise ValueError('classic runtime source changed after module import')
    for name, digest in identity['runtime_files_sha256'].items():
        if file_sha256(root/name) != digest:
            raise ValueError('classic runtime source identity changed: '+name)
    if pl is not None:
        framework = Path(pl.__file__).resolve().parent
        if identity['framework_files_sha256'] != {name:file_sha256(framework/name) for name in FRAMEWORK_FILES}:
            raise ValueError('classic actual Lightning implementation identity changed')


def prepare_classic_run(data, *, upstream, config, partitions, policy):
    _runtime()
    recipe = data_runtime.load_classic_config(config, upstream)
    canonical_policy = runtime_policy(engineering=True, device=policy['device'], batch_size=policy['batch_size'],
        num_workers=policy['num_workers'], engineering_epochs=policy['max_epochs'], max_steps=policy['max_steps'],
        limit_val_batches=policy['limit_val_batches'])
    if policy != canonical_policy:
        raise ValueError('runtime policy differs from approved engineering policy')
    if partitions.get('settings', {}).get('val_fraction') != recipe['val_fraction']:
        raise ValueError('partition fraction differs from exact classic recipe')
    cfg = model_runtime.compose_classic(upstream, recipe['variant'])
    data = _safe(data).resolve()
    # Capture before hashing, and check again before construction/first read.
    from .data import LegacyStore
    store = LegacyStore(data)
    paths = data_runtime._schema(store)
    paths.update({name:data_runtime._safe_path(data, name) for name in ('manifest.json', 'metadata.npz')})
    stamps = {str(path.relative_to(data)):_stamp(path) for path in paths.values()}
    data_identity = data_runtime.classic_data_identity(data, partitions, upstream=upstream, config=cfg,
        normalization=recipe['normalization'], mask_policy=recipe['mask_policy'])
    sources = dict(data_identity['runtime_files_sha256'])
    sources['egoemg_repro/classic_emg_train.py'] = file_sha256(Path(__file__))
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'hydra-core', 'omegaconf', 'pytorch-lightning',
                 'torchmetrics', 'lightning-utilities', 'pandas', 'joblib'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    identity = dict(format=FORMAT, recipe=recipe, model=data_identity['model_protocol'], data=data_identity,
        cohort_counts={role:partitions['coverage'][role]['hand_samples'] for role in ('train','val','test')},
        policy=policy, runtime_files_sha256=sources, versions=versions,
        framework_files_sha256={name:file_sha256(Path(pl.__file__).resolve().parent/name) for name in FRAMEWORK_FILES},
        declared_devices=cfg.trainer.devices, declared_per_rank_batch=recipe['batch_size'],
        batchnorm='single CPU process; not equivalent to original six-rank batch statistics',
        scope='engineering only; no complete-cohort or benchmark claim')
    identity['declared_devices'] = list(identity['declared_devices'])
    # Identities are persisted as JSON. Canonicalize only this metadata, not
    # checkpoint tensors, optimizer tuples, or Python/NumPy RNG tuples.
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    identity['sha256'] = _json_hash(identity)
    prepared = PreparedClassic(data, Path(upstream).resolve(), cfg, recipe, copy.deepcopy(partitions), identity, stamps)
    _check_prepared(prepared)
    return prepared


def _check_prepared(prepared):
    _check_identity(prepared.identity)
    for relative, expected in prepared.stamps.items():
        if _stamp(data_runtime._safe_path(prepared.data, relative)) != expected:
            raise ValueError('classic data changed after content admission: '+relative)


def _check_identity(identity):
    try:
        value = {key:value for key,value in identity.items() if key != 'sha256'}
        if identity['format'] != FORMAT or identity['sha256'] != _json_hash(value):
            raise ValueError('classic identity digest differs')
        _check_sources(identity)
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid classic training identity') from exc


def _admit_batch(batch, contract):
    torch = _runtime()
    n = batch['emg'].shape[0]
    for index in range(n):
        sample = {key:(value[index] if torch.is_tensor(value) or isinstance(value, (list, tuple)) else value)
                  for key,value in batch.items()}
        data_runtime.admit_classic_sample(sample, contract)


def build_classic_lightning(upstream, cfg, identity):
    """Actual factory, seeded before construction, with observing-only guards."""
    torch = _runtime()
    _check_identity(identity)
    source, contract, _, _ = model_runtime._validated(upstream, cfg)
    expected = model_runtime.resolved_classic_identity(cfg, contract)
    if _json_hash(expected) != _json_hash(identity['model']) or contract.variant != identity['recipe']['variant']:
        raise ValueError('model/config identity differs before factory')
    model_runtime._activate(source)
    from egoemg._train_utils import setup_runtime
    from egoemg.train import make_lightning_module
    setup_runtime(cfg)
    module = make_lightning_module(cfg)
    model = module.model
    if (sum(p.numel() for p in model.parameters()) != contract.parameters or
        (model.left_context,model.right_context) != (contract.left_context,contract.right_context) or
        model.out_channels != 20 or model.provide_initial_pos is not False):
        raise ValueError('actual classic factory contract differs')
    def before(_model, arguments):
        _admit_batch(arguments[0], contract)
    def after(_model, arguments, output):
        batch = arguments[0]
        length = 7790-contract.left_context-contract.right_context
        if not isinstance(output, tuple) or len(output) != 3:
            raise ValueError('official classic tuple schema differs')
        pred, target, mask = output
        span = slice(contract.left_context, 7790-contract.right_context)
        if (pred.shape != (batch['emg'].shape[0],22,length) or pred.dtype != torch.float32 or
            target.shape != pred.shape or mask.shape != (pred.shape[0],length) or mask.dtype != torch.bool or
            not torch.equal(target,batch['joint_angles'][...,span]) or
            not torch.equal(mask,batch['label_valid_mask'][...,span])):
            raise ValueError('official classic tuple/context/target/mask schema differs')
        if not torch.isfinite(pred).all():
            raise ValueError('classic prediction must be finite')
    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    return module


def classic_early_stopping():
    _runtime()
    from pytorch_lightning.callbacks import EarlyStopping
    return EarlyStopping(monitor='val_loss', mode='min', patience=50, min_delta=0,
                         check_finite=True, check_on_train_epoch_end=False, verbose=False)


def _optimizer_step(optimizer):
    values = [int(value['step'].item() if hasattr(value['step'], 'item') else value['step'])
              for value in optimizer.state.values() if 'step' in value]
    return max(values, default=0)


class ClassicProtocolCallback(Callback):
    """Observe the untouched automatic optimization and full validation events."""
    def __init__(self, identity, generators, train_samples, val_samples, stop_after_epoch=None):
        _runtime()
        self.identity, self.generators = identity, generators
        self.train_samples, self.val_samples = train_samples, val_samples
        self.stop_after_epoch = stop_after_epoch
        self.updates, self.validation_events = [], []
        self.completed_epochs = 0
        self.pending_rng = None
        self.epoch_samples = 0
        self.full_epoch = False

    def state_dict(self):
        return dict(updates=self.updates, validation_events=self.validation_events, completed_epochs=self.completed_epochs)

    def load_state_dict(self, state):
        self.updates = state['updates']; self.validation_events = state['validation_events']
        self.completed_epochs = state['completed_epochs']

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.pending_rng = checkpoint['classic']['rng']

    def on_train_epoch_start(self, trainer, pl_module):
        if self.pending_rng is not None:
            _restore_rng(self.pending_rng, self.generators)
            self.pending_rng = None
        self.epoch_samples = 0
        self.full_epoch = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        _admit_batch(batch, model_runtime.classic_contract(self.identity['recipe']['variant']))
        self.before_step = _optimizer_step(trainer.optimizers[0])
        self.before_parameters = state_digest(dict(pl_module.named_parameters()))
        self.epoch_samples += len(batch['emg'])
        self.scale_before = trainer.precision_plugin.scaler.get_scale() if getattr(trainer.precision_plugin,'scaler',None) else None

    def on_before_backward(self, trainer, pl_module, loss):
        if not _runtime().isfinite(loss):
            raise ValueError('classic loss must be finite before backward')

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        torch = _runtime()
        gradients = [parameter.grad for parameter in pl_module.parameters() if parameter.grad is not None]
        self.gradient_finite = all(bool(torch.isfinite(value).all()) for value in gradients)
        self.gradient_max = max((float(value.detach().abs().max()) for value in gradients), default=0.)
        if self.identity['policy']['precision'] == '32-true' and not self.gradient_finite:
            raise ValueError('classic CPU gradient must be finite')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        after = _optimizer_step(trainer.optimizers[0])
        scaler = getattr(trainer.precision_plugin, 'scaler', None)
        self.updates.append(dict(epoch=trainer.current_epoch, batch=batch_idx, global_step=trainer.global_step,
            optimizer_step_before=self.before_step, optimizer_step_after=after, optimizer_updated=after>self.before_step,
            parameters_changed=state_digest(dict(pl_module.named_parameters())) != self.before_parameters,
            gradient_finite=self.gradient_finite, gradient_max=self.gradient_max,
            scale_before=self.scale_before, scale_after=scaler.get_scale() if scaler else None,
            input_sha256=state_digest({key:batch[key] for key in ('emg','joint_angles','label_valid_mask','classic_sample_index')}),
            loss=float(outputs['loss'].detach())))

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_seen, self.val_losses = [], []

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        _admit_batch(batch, model_runtime.classic_contract(self.identity['recipe']['variant']))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_seen.extend(batch['classic_sample_index'].tolist())
        loss = float(outputs.detach())
        if not math.isfinite(loss):
            raise ValueError('classic validation loss must be finite')
        self.val_losses.append((len(batch['emg']), loss, int(batch['label_valid_mask'].sum())))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        full = sorted(self.val_seen) == list(range(self.val_samples))
        weighted = batch_weighted_loss(self.val_losses)
        actual = float(trainer.callback_metrics['val_loss'])
        if not math.isfinite(actual) or not math.isclose(weighted, actual, rel_tol=2e-6, abs_tol=1e-7):
            raise ValueError('actual Lightning val_loss differs from finite batch-size weighted loss')
        self.validation_events.append(dict(epoch=trainer.current_epoch, global_step=trainer.global_step,
            samples=len(self.val_seen), expected_samples=self.val_samples, full=full,
            sample_index_sha256=_json_hash(self.val_seen), val_loss=actual, batch_weighted_loss=weighted,
            batch_sizes=[value[0] for value in self.val_losses], valid_counts=[value[2] for value in self.val_losses],
            selection_scope='engineering' if full else 'ineligible-partial', scientific_eligible=False))

    def on_train_epoch_end(self, trainer, pl_module):
        self.full_epoch = self.epoch_samples == self.train_samples and bool(self.validation_events) and self.validation_events[-1]['full']
        if self.full_epoch:
            self.completed_epochs = trainer.current_epoch+1
        if self.stop_after_epoch is not None and trainer.current_epoch+1 >= self.stop_after_epoch:
            trainer.should_stop = True

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['classic'] = dict(format=FORMAT, identity=self.identity, rng=_rng(self.generators),
            epoch_boundary=self.full_epoch, completed_epochs=self.completed_epochs,
            scope='engineering', scientific_eligible=False,
            precision_plugin=type(trainer.precision_plugin).__qualname__,
            optimizer_schema=_optimizer_schema(checkpoint['optimizer_states']),
            optimizer_updates=sum(event['optimizer_updated'] for event in self.updates),
            selected_best=None)


def _sealed(checkpoint):
    return state_digest({key:value for key,value in checkpoint.items() if key != 'classic_seal'})


def _optimizer_schema(states):
    torch = _runtime()
    if not isinstance(states, list) or len(states) != 1:
        raise ValueError('classic checkpoint requires one original optimizer state')
    optimizer = states[0]
    if not optimizer.get('state') or len(optimizer.get('param_groups', [])) != 1:
        raise ValueError('classic checkpoint optimizer state is missing or reset')
    schema = {}
    for index, values in optimizer['state'].items():
        if set(values) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('classic checkpoint optimizer AdamW state fields differ')
        schema[str(index)] = {key:dict(shape=list(value.shape),dtype=str(value.dtype)) if torch.is_tensor(value)
                              else dict(type=type(value).__name__) for key,value in values.items()}
    return dict(parameters=optimizer['param_groups'][0]['params'], states=schema)


def _validate_epoch_state(checkpoint, identity):
    """Do not let PL's backwards-compatible missing-key/zip behavior reset state."""
    meta, callbacks = checkpoint['classic'], checkpoint['callbacks']
    epochs = meta['completed_epochs']
    expected_keys = {'state_dict','epoch_loop.state_dict','epoch_loop.batch_progress','epoch_loop.scheduler_progress',
        'epoch_loop.automatic_optimization.state_dict','epoch_loop.automatic_optimization.optim_progress',
        'epoch_loop.manual_optimization.state_dict','epoch_loop.manual_optimization.optim_step_progress',
        'epoch_loop.val_loop.state_dict','epoch_loop.val_loop.batch_progress','epoch_progress'}
    loop = checkpoint['loops']['fit_loop']
    if set(loop) != expected_keys:
        raise ValueError('classic checkpoint loop state keys differ')
    for category in ('total','current'):
        progress = loop['epoch_progress'][category]
        if progress != dict(ready=epochs,started=epochs,processed=epochs,completed=epochs-1):
            raise ValueError('classic checkpoint is not a genuine original epoch boundary')
    for key, count in (('epoch_loop.batch_progress', identity['cohort_counts']['train']),
                       ('epoch_loop.val_loop.batch_progress', identity['cohort_counts']['val'])):
        batches = math.ceil(count/identity['policy']['batch_size'])
        if loop[key]['current'] != dict(ready=batches,started=batches,processed=batches,completed=batches) or not loop[key]['is_last_batch']:
            raise ValueError('classic checkpoint has partial batch/validation loop state')
    if checkpoint['epoch'] != epochs-1:
        raise ValueError('classic checkpoint epoch and loop state disagree')
    if _optimizer_schema(checkpoint['optimizer_states']) != meta['optimizer_schema']:
        raise ValueError('classic checkpoint optimizer state schema differs')
    schedulers = checkpoint['lr_schedulers']
    if (len(schedulers) != 1 or schedulers[0].get('T_max') != 100 or
        schedulers[0].get('eta_min') != identity['recipe']['eta_min'] or
        schedulers[0].get('last_epoch') != epochs or schedulers[0].get('_step_count') != epochs+1):
        raise ValueError('classic checkpoint original epoch cosine scheduler state differs')
    early = [state for name,state in callbacks.items() if name.startswith('EarlyStopping{')]
    monitors = [state for name,state in callbacks.items() if name.startswith('ModelCheckpoint{')]
    if len(early) != 1 or len(monitors) != 1 or 'ClassicProtocolCallback' not in callbacks:
        raise ValueError('classic checkpoint required callback state missing')
    if (set(early[0]) != {'wait_count','stopped_epoch','best_score','patience'} or
        early[0]['patience'] != 50 or not 0 <= early[0]['wait_count'] <= 50):
        raise ValueError('classic checkpoint EarlyStopping callback state differs')
    monitor = monitors[0]
    if set(monitor) != {'monitor','best_model_score','best_model_path','current_score','dirpath','best_k_models',
                        'kth_best_model_path','kth_value','last_model_path'} or monitor['monitor'] != 'val_loss':
        raise ValueError('classic checkpoint ModelCheckpoint callback state differs')
    parent = Path(monitor['dirpath'])
    best = Path(monitor['best_model_path'])
    if (not parent.is_absolute() or best.parent != parent or best.name in ('last.ckpt','.', '..') or
        set(monitor['best_k_models']) != {str(best)} or
        float(monitor['best_k_models'][str(best)]) != float(monitor['best_model_score'])):
        raise ValueError('classic checkpoint selected-best callback paths/scores differ')
    for key in ('kth_best_model_path','last_model_path'):
        path = monitor[key]
        if path and (Path(path).parent != parent or '..' in Path(path).parts):
            raise ValueError('classic checkpoint callback artifact path escapes its original run')
    protocol = callbacks['ClassicProtocolCallback']
    if protocol['completed_epochs'] != epochs or len(protocol['validation_events']) != epochs:
        raise ValueError('classic checkpoint protocol callback epoch state differs')
    full = [event for event in protocol['validation_events'] if event['full']]
    if len(full) != epochs or float(monitor['best_model_score']) != min(event['val_loss'] for event in full):
        raise ValueError('classic checkpoint best is not complete-validation minimum loss')
    if len(protocol['updates']) != checkpoint['global_step'] or meta['optimizer_updates'] != sum(event['optimizer_updated'] for event in protocol['updates']):
        raise ValueError('classic checkpoint update evidence/state differs')


def _validate_checkpoint(checkpoint, identity, *, role='last', boundary=True):
    torch = _runtime()
    _check_identity(identity)
    try:
        meta = checkpoint['classic']
        if (checkpoint['classic_seal'] != _sealed(checkpoint) or meta['identity'] != identity or
            meta['format'] != FORMAT or meta['role'] != role):
            raise ValueError('classic checkpoint identity/role/integrity differs')
        for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'loops', 'callbacks'):
            if not checkpoint[key]:
                raise ValueError('classic checkpoint missing full '+key)
        if boundary and not meta['epoch_boundary']:
            raise ValueError('classic resume requires a full epoch boundary, not partial training/validation')
        if boundary:
            _validate_epoch_state(checkpoint, identity)
        if not meta['rng']['generators'] or meta['rng']['torch'].dtype != torch.uint8:
            raise ValueError('classic checkpoint RNG state missing')
        if meta['precision_plugin'] == 'MixedPrecision' and not checkpoint.get('MixedPrecision'):
            raise ValueError('classic checkpoint missing actual GradScaler state')
        for name, value in checkpoint['state_dict'].items():
            if not torch.is_tensor(value) or (value.is_floating_point() and not torch.isfinite(value).all()):
                raise ValueError('classic checkpoint model tensor is nonfinite or invalid: '+name)
        selected = meta['selected_best']
        if role == 'last' and meta['epoch_boundary']:
            if not selected or selected['state_sha256'] != _sealed(selected['checkpoint']):
                raise ValueError('classic last checkpoint lacks bound selected-best snapshot')
            _validate_checkpoint(selected['checkpoint'], identity, role='best', boundary=True)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('classic checkpoint missing or truncated complete state') from exc
    return checkpoint


def read_classic_checkpoint(path, identity, *, role='last'):
    torch = _runtime()
    path = _safe(path)
    before = _stamp(path)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:
        raise ValueError('classic checkpoint is truncated or unreadable') from exc
    _validate_checkpoint(checkpoint, identity, role=role)
    if _stamp(path) != before:
        raise ValueError('classic checkpoint changed during admission')
    return checkpoint


class ClassicCheckpointIO(CheckpointIO):
    """Reserve-aware atomic IO plus offline selected-best carrying epoch last."""
    def __init__(self, directory, identity, *, resume=None, admitted=None):
        _runtime()
        self.directory, self.identity = _safe(directory), identity
        self.resume, self.admitted = Path(resume).absolute() if resume else None, admitted

    def _owned(self, path):
        path = _safe(path)
        if path.parent != self.directory:
            raise ValueError('classic checkpoint path is outside this owned run')
        return path

    def _save(self, checkpoint, path):
        torch = _runtime()
        checkpoint['classic_seal'] = _sealed(checkpoint)
        budget = 2*_payload_bytes(checkpoint)+IO_OVERHEAD
        _atomic(path, lambda stream:torch.save(checkpoint, stream), budget)

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise ValueError('remote classic checkpoint storage is unsupported')
        path = self._owned(path)
        meta = checkpoint['classic']
        if meta['identity'] != self.identity:
            raise ValueError('checkpoint identity differs at publication')
        meta['role'] = 'last' if path.name == 'last.ckpt' else 'best'
        if meta['role'] == 'last':
            candidates = [state.get('best_model_path') for state in checkpoint['callbacks'].values()
                          if isinstance(state, dict) and 'best_model_path' in state]
            candidate = next((value for value in candidates if value), None)
            if candidate:
                best = read_classic_checkpoint(self._owned(candidate), self.identity, role='best')
                meta['selected_best'] = dict(state_sha256=_sealed(best), checkpoint=best)
        self._save(checkpoint, path)

    def load_checkpoint(self, path, map_location=None):
        if self.resume is None or Path(path).absolute() != self.resume or self.admitted is None:
            return read_classic_checkpoint(path, self.identity)
        # Re-read the admitted artifact, rather than trusting a possibly changed file.
        original = read_classic_checkpoint(path, self.identity)
        if original['classic_seal'] != self.admitted['classic_seal']:
            raise ValueError('resume artifact changed after pre-factory admission')
        checkpoint = copy.deepcopy(original)
        best_path = self.directory/'recovered-best.ckpt'
        best = copy.deepcopy(checkpoint['classic']['selected_best']['checkpoint'])
        self._save(best, best_path)
        for state in checkpoint['callbacks'].values():
            if isinstance(state, dict) and 'best_model_path' in state:
                state['dirpath'] = str(self.directory)
                state['best_model_path'] = str(best_path)
                state['best_k_models'] = {str(best_path):state['best_model_score']}
                state['kth_best_model_path'] = str(best_path)
                state['last_model_path'] = str(self.directory/'last.ckpt')
        return checkpoint

    def remove_checkpoint(self, path):
        self._owned(path).unlink(missing_ok=True)
        _fsync_dir(self.directory)


def _strict_state(module, checkpoint):
    expected, actual = module.state_dict(), checkpoint['state_dict']
    if set(actual) != set(expected):
        raise ValueError('classic checkpoint model state keys differ')
    for name, value in actual.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError('classic checkpoint tensor shape/dtype differs: '+name)
    # Lightning will load the same complete state strictly, after setup.


class _AdmittedDataset:
    def __init__(self, dataset, prepared=None):
        self.dataset, self.prepared = dataset, prepared
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        if self.prepared is not None:
            _check_prepared(self.prepared)
        item = self.dataset[index]
        data_runtime.admit_classic_sample(item, self.dataset.contract)
        if self.prepared is not None:
            _check_prepared(self.prepared)
        item['classic_sample_index'] = index
        return item


def run_classic(data, *, upstream, config, partitions, output, engineering, device='cpu', batch_size=1,
                num_workers=0, engineering_epochs=1, stop_after_epoch=None, max_steps=None,
                limit_val_batches=None, resume=None):
    """Run genuine CPU automatic optimization; no scientific completion/test claim."""
    policy = runtime_policy(engineering=engineering, device=device, batch_size=batch_size, num_workers=num_workers,
        engineering_epochs=engineering_epochs, max_steps=max_steps, limit_val_batches=limit_val_batches)
    output = _safe(output)
    _space(output, IO_OVERHEAD)
    if output.exists():
        raise FileExistsError('classic output already exists; runs never overwrite or restart implicitly')
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= engineering_epochs):
        raise ValueError('stop_after_epoch must be an explicit valid epoch boundary')
    prepared = prepare_classic_run(data, upstream=upstream, config=config, partitions=partitions, policy=policy)
    admitted = read_classic_checkpoint(resume, prepared.identity) if resume else None
    if admitted is not None and admitted['classic']['completed_epochs'] >= engineering_epochs:
        raise ValueError('resume checkpoint already reached configured epoch budget')
    _check_prepared(prepared)  # Identity gates precede factory and dataset reads.
    torch = _runtime()
    module = build_classic_lightning(prepared.upstream, prepared.cfg, prepared.identity)
    if admitted is not None:
        _strict_state(module, admitted)
    _check_prepared(prepared)
    datasets = data_runtime.build_classic_datasets(prepared.data, prepared.partitions, upstream=prepared.upstream,
        config=prepared.cfg, normalization=prepared.recipe['normalization'], mask_policy=prepared.recipe['mask_policy'])
    if not len(datasets['train']) or not len(datasets['val']):
        raise ValueError('classic training requires nonempty disjoint training and validation')
    # EmgPredictionModule intentionally calls model.forward() directly. Its
    # source stays unchanged; the observing __call__ hooks above are therefore
    # an explicit pre-optimization tuple witness, not a claimed per-step hook.
    witness = torch.utils.data.default_collate([_AdmittedDataset(datasets['val'], prepared)[0]])
    was_training = module.model.training
    with torch.no_grad():
        module.model.eval()(witness)
    module.model.train(was_training)
    generators = {name:torch.Generator().manual_seed(prepared.recipe['seed']+offset)
                  for offset,name in enumerate(('train_loader','train_sampler','val_loader'))}
    sampler = torch.utils.data.RandomSampler(datasets['train'], generator=generators['train_sampler'])
    train_loader = torch.utils.data.DataLoader(_AdmittedDataset(datasets['train'], prepared), batch_size=batch_size, sampler=sampler,
        generator=generators['train_loader'], num_workers=0)
    val_loader = torch.utils.data.DataLoader(_AdmittedDataset(datasets['val'], prepared), batch_size=batch_size, shuffle=False,
        generator=generators['val_loader'], num_workers=0)
    observer = ClassicProtocolCallback(prepared.identity, generators, len(datasets['train']), len(datasets['val']), stop_after_epoch)
    from hydra.utils import instantiate
    # Instantiate the actual approved callback configs; only local artifact path,
    # and partial-validation eligibility, are explicit runtime adaptations.
    callbacks = [observer]
    checkpoint_callback = None
    partial = limit_val_batches is not None and limit_val_batches < len(val_loader)
    for entry in prepared.cfg.callbacks:
        target = str(entry['_target_'])
        if target.endswith('.ModelCheckpoint'):
            callback = instantiate(entry, dirpath=str(output/'checkpoints'), save_on_train_epoch_end=True,
                                   save_top_k=0 if partial else 1)
            checkpoint_callback = callback
        elif target.endswith('.EarlyStopping') and partial:
            continue
        else:
            callback = instantiate(entry)
        callbacks.append(callback)
    io = ClassicCheckpointIO(output/'checkpoints', prepared.identity, resume=resume, admitted=admitted)
    # From this point the run owns its new directory. Failures preserve diagnostics.
    output.mkdir(parents=True, exist_ok=False)
    (output/'checkpoints').mkdir()
    try:
        _write_json(output/'identity.json', prepared.identity)
        _write_json(output/'status.json', dict(state='running', scientific_eligible=False))
        logger = ClassicCSVLogger(str(output), name='lightning', version='engineering')
        trainer = pl.Trainer(accelerator='cpu', devices=1, precision='32-true', max_epochs=engineering_epochs,
            max_steps=max_steps if max_steps is not None else -1, gradient_clip_val=0, callbacks=callbacks,
            logger=logger, plugins=[io], num_sanity_val_steps=0, enable_progress_bar=False,
            enable_model_summary=False, log_every_n_steps=1, check_val_every_n_epoch=1,
            limit_val_batches=limit_val_batches if limit_val_batches is not None else 1.0,
            default_root_dir=str(output), deterministic=True)
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    ckpt_path=str(Path(resume).absolute()) if resume else None)
        _check_prepared(prepared)
        result = dict(format=FORMAT, scope='engineering', scientific_eligible=False, identity=prepared.identity,
            global_step=trainer.global_step, optimizer_updates=sum(event['optimizer_updated'] for event in observer.updates),
            updates=observer.updates, completed_epochs=observer.completed_epochs, configured_max_epochs=engineering_epochs,
            official_max_epochs=100, clip_value=0, precision_actual='32-true',
            validation_events=observer.validation_events,
            selected_best=dict(path=Path(checkpoint_callback.best_model_path).name,
                val_loss=float(checkpoint_callback.best_model_score), selection_scope='engineering')
                if checkpoint_callback.best_model_path else None,
            stop_reason='engineering-max-steps' if max_steps is not None and trainer.global_step>=max_steps else
                ('engineering-epoch-pause' if stop_after_epoch is not None and observer.completed_epochs<engineering_epochs
                 else 'engineering-epoch-budget'),
            rng_sha256=state_digest(_rng(generators)),
            limitations=['CPU32 does not qualify CUDA16-mixed/overflow', 'no scientific full cohort, formal best or frozen test',
                         'epoch restore only; workers0 streams, no persistent-worker continuation claim'])
        _write_json(output/'result.json', result)
        _write_json(output/'status.json', dict(state='engineering_finished', scientific_eligible=False,
            result_sha256=file_sha256(output/'result.json')))
        return result
    except BaseException as exc:
        # Publication can succeed before directory fsync fails. Invalidate our
        # owned success marker before any reserve-constrained failure write.
        # Leave valid checkpoints, prior source artifacts and result bytes intact.
        try:
            (output/'status.json').unlink(missing_ok=True)
            _fsync_dir(output)
        except OSError:
            pass
        try:
            _write_json(output/'status.json', dict(state='failed', scientific_eligible=False,
                error_type=type(exc).__name__, error=str(exc)))
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','upstream','config','partitions','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--engineering-epochs', type=int, default=1)
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--limit-val-batches', type=int)
    parser.add_argument('--resume', type=Path)
    args = vars(parser.parse_args(argv))
    args['partitions'] = json.loads(args['partitions'].read_text())
    result = run_classic(**args)
    print(json.dumps(dict(scope=result['scope'], scientific_eligible=False,
                         completed_epochs=result['completed_epochs'], optimizer_updates=result['optimizer_updates'])))


if __name__ == '__main__':
    main()
