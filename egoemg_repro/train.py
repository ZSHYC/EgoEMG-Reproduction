"""Transparent legacy-EgoEMG training around the unchanged official module.

No upstream Lightning datamodule is used: its default validation policy is not
an independent legacy-data holdout. All selection here uses role=val only.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import copy
import json
import math
from pathlib import Path
import random
import shutil
import os
import subprocess
import sys
import time
import uuid

import numpy as np

from .metrics import AngleAccumulator, PairedCenterAccumulator
from . import UPSTREAM_COMMIT
from .download import MIN_FREE_BYTES


def load_config(path):
    with open(path) as stream:
        return json.load(stream)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_identity(root):
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted({*root.glob('egoemg/**/*.py'), *root.glob('config/**/*.yaml'),
                    *root.glob('egoemg/UmeTrack/**/*.json')})
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip() or None
    return {'sha256': digest.hexdigest(), 'revision': revision, 'file_count': len(files),
            'scope': 'official Python, Hydra YAML and UmeTrack JSON contents'}


def validate_upstream(root):
    """Reject unknown bases and source edits before loading data or model modules."""
    root = Path(root).resolve()
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError(f'upstream revision must be the approved {UPSTREAM_COMMIT}')
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
                             '--untracked-files=all', '--', 'egoemg', 'config'],
                            capture_output=True, text=True)
    if status.returncode or status.stdout.strip():
        raise ValueError('upstream code/config must be clean; source patches are not supported')
    identity = source_identity(root)
    identity['source_policy'] = 'required approved revision; clean egoemg/ and config/ trees'
    return identity


def reproduction_identity(root=None, *, model_kind='emg', config_path=None, recipe_bytes=None):
    """Hash executed runtime sources and the source default recipe, including dirty edits."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    names = ['egoemg_repro/' + name + '.py' for name in
             ('__init__', 'train', 'data', 'partitions', 'metrics', 'download')]
    if model_kind in ('vision', 'fusion'):
        names += ['egoemg_repro/visual.py', 'egoemg_repro/crops.py']
    recipe = Path(config_path) if config_path is not None else root / 'configs/emgformer_s.json'
    digest = hashlib.sha256()
    runtime_files = {}
    for name in names:
        contents = (root / name).read_bytes()
        digest.update(name.encode())
        digest.update(contents)
        runtime_files[name] = hashlib.sha256(contents).hexdigest()
    recipe_bytes = recipe.read_bytes() if recipe_bytes is None else recipe_bytes
    recipe_sha = hashlib.sha256(recipe_bytes).hexdigest()
    digest.update(b'selected_recipe')
    digest.update(recipe_bytes)
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip() or None
    return {'sha256': digest.hexdigest(), 'revision': revision, 'files': names,
            'selected_recipe_sha256': recipe_sha, 'model_kind': model_kind,
            'runtime_files_sha256': runtime_files,
            'scope': 'exact selected-model runtime Python and selected-recipe bytes; resolved config is separately hashed'}


def assert_compatible_provenance(recorded, current):
    for key, expected in current.items():
        actual = recorded.get(key)
        if key == 'reproduction':
            # A docs-only commit may advance while executed bytes remain identical.
            compatible = actual is not None and actual.get('sha256') == expected.get('sha256')
        else:
            compatible = actual == expected
        if not compatible:
            raise ValueError(f'checkpoint {key} does not match current inputs')


def data_identity(root):
    """Bind metadata and verified-transfer digests, without re-reading all payloads.

    The acquisition ledger records local verification, not publisher authenticity.
    Without a ledger, the scope explicitly remains metadata plus array sizes.
    """
    root = Path(root)
    manifests = {}
    for path in sorted({*root.glob('*.json'), *root.glob('*.npz')}):
        if path.name != 'acquisition_ledger.json':
            manifests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    arrays = {p.name: p.stat().st_size for p in sorted(root.iterdir())
              if p.is_file() and p.suffix in ('.dat', '.npy', '.bin', '.mmap')}
    receipt_path = root / 'acquisition_ledger.json'
    payloads = {}
    if receipt_path.exists():
        for name, record in load_config(receipt_path).get('files', {}).items():
            path = root / name
            if path.name != name or not path.is_file() or path.stat().st_size != record['bytes']:
                raise ValueError(f'acquisition receipt file identity mismatch: {name}')
            payloads[name] = {key: record[key] for key in ('bytes', 'sha256', 'checksum_basis') if key in record}
    identity = {'metadata_sha256': manifests, 'array_bytes': arrays, 'verified_transfer_records': payloads,
                'scope': 'metadata contents, array sizes and acquisition-ledger hashes; arrays are not rehashed here',
                'arrays_with_transfer_sha256': sorted(set(arrays) & set(payloads)),
                'arrays_without_transfer_sha256': sorted(set(arrays) - set(payloads))}
    identity['sha256'] = canonical_hash(identity)
    return identity


def visual_data_identity(data, crops, manifest=None, *, allow_missing_episodes=False):
    """Bind the actual row/video mapping plus acquired or generated crop receipts."""
    from .data import LegacyStore, file_sha256
    store, crops = LegacyStore(data), Path(crops)
    mapping = next((name for name in ('image_webcam_frame_index', 'image_head_frame_index')
                    if name in store.fields), None)
    if mapping is None:
        raise ValueError('visual alignment field is unavailable')
    store.field(mapping)  # Enforce manifest dtype/shape/byte size before hashing.
    specification = dict(store.fields[mapping])
    alignment = {'field': mapping, 'specification': specification,
                 'sha256': file_sha256(store.root / specification.get('filename', mapping + '.dat'))}
    receipt_path, cache_path = crops / 'acquisition_ledger.json', crops / 'center_cache_manifest.json'
    cache = None
    if cache_path.exists():
        cache = load_config(cache_path)
        records = cache.get('payload_files', {})
        if cache.get('manifest_sha256') != file_sha256(store.manifest_path) or cache.get('metadata_sha256') != file_sha256(store.metadata_path):
            raise ValueError('crop cache metadata identity differs from the legacy data')
        if manifest is not None:
            windows_hash = hashlib.sha256(json.dumps(manifest['windows'], sort_keys=True).encode()).hexdigest()
            if cache.get('windows_sha256') != windows_hash:
                raise ValueError('crop cache does not cover the identical partition windows')
        receipt_kind = 'generated_fixed_center_cache'
    elif receipt_path.exists():
        records = load_config(receipt_path).get('files', {})
        receipt_kind = 'verified_acquisition_ledger'
    else:
        raise ValueError('visual crops require an acquisition or generated-cache payload receipt')
    payloads = {}
    for name, record in sorted(records.items()):
        if not name.endswith(('.lmdb', '.data.mdb')):
            continue
        if Path(name).name != name:
            raise ValueError('crop receipt filenames must be single path components')
        path = crops / name
        if path.is_dir():
            path = path / 'data.mdb'
        if not path.is_file() or path.stat().st_size != record.get('bytes'):
            raise ValueError(f'crop receipt size/file mismatch: {name}')
        digest = record.get('sha256', '')
        if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
            raise ValueError(f'crop receipt requires SHA256: {name}')
        payloads[name] = {key: record[key] for key in ('bytes', 'sha256', 'checksum_basis') if key in record}
    if not payloads:
        raise ValueError('crop receipt contains no verified LMDB payloads')
    missing_episodes = []
    if manifest is not None:
        needed = {int(window['episode']) for window in manifest['windows']}
        for index in sorted(needed):
            episode_id = store.episodes[index]['id']
            if not any(episode_id + suffix in payloads for suffix in ('.lmdb', '.data.mdb')):
                missing_episodes.append(episode_id)
    if missing_episodes and not allow_missing_episodes:
        raise ValueError('crop receipt lacks episode payloads required for the complete partitions')
    identity = {'alignment': alignment, 'receipt_kind': receipt_kind, 'payloads': payloads,
                'missing_episode_payloads': missing_episodes,
                'scope': 'actual alignment field SHA256 plus verified crop receipt bytes/hashes; LMDBs are not rehashed here'}
    if cache is not None:
        identity['center_cache_manifest_sha256'] = file_sha256(cache_path)
    identity['sha256'] = canonical_hash(identity)
    return identity


def validate_initializer(checkpoint, expected_kind, provenance):
    """Require clean branch artifacts selected by comparable, complete validation."""
    best = selected_best_checkpoint(checkpoint, bounded=False)
    if checkpoint['provenance'].get('model_kind', 'emg') != expected_kind:
        raise ValueError('initializer branch kind differs from the requested clean branch')
    shared = {key: provenance[key] for key in ('source', 'data', 'partition_sha256', 'normalization_sha256')}
    if expected_kind == 'vision':
        shared['visual_data'] = provenance['visual_data']
    assert_compatible_provenance(checkpoint['provenance'], shared)
    recorded_files = checkpoint['provenance'].get('reproduction', {}).get('runtime_files_sha256', {})
    current_files = provenance.get('reproduction', {}).get('runtime_files_sha256', {})
    if not recorded_files or any(current_files.get(name) != digest for name, digest in recorded_files.items()):
        raise ValueError('initializer reproduction runtime source differs from current shared code')
    if checkpoint['global_step'] != best['global_step']:
        raise ValueError('initializer must contain selected best weights; use the branch best.pt')
    if checkpoint['model'].keys() != best['model'].keys():
        raise ValueError('initializer model does not match selected best weights')
    for name, value in checkpoint['model'].items():
        other = best['model'][name]
        if hasattr(value, 'detach'):
            import torch
            equal = torch.equal(value.cpu(), other.cpu())
        else:
            equal = np.array_equal(value, other)
        if not equal:
            raise ValueError('initializer model does not match selected best weights')
    return {'model_kind': expected_kind, 'selected_step': best['global_step'],
            'best_val_mae_deg': best['best_val_mae_deg'], 'selection': 'complete independent validation'}


def initialization_identity(config, provenance, *, vision_checkpoint=None, emg_checkpoint=None, loaded=None):
    from .data import file_sha256
    import torch
    kind = model_kind(config)
    paths = {'vision': vision_checkpoint, 'emg': emg_checkpoint}
    if kind == 'emg' and any(paths.values()):
        raise ValueError('EMG training does not accept visual branch initializers')
    if kind == 'vision' and emg_checkpoint is not None:
        raise ValueError('vision-only training does not accept an EMG initializer')
    if loaded is not None:
        identity = loaded['provenance']['initialization']
        for branch, path in paths.items():
            if path is not None and file_sha256(path) != identity.get(branch, {}).get('sha256'):
                raise ValueError('provided initializer differs from checkpoint initialization provenance')
        return identity
    if kind == 'fusion' and not all(paths.values()):
        raise ValueError('clean fusion requires both --vision-checkpoint and --emg-checkpoint')
    identity = {'policy': 'clean branches then official residual-zero initialization' if kind == 'fusion' else
                'ImageNet1K_V1 backbone' if kind == 'vision' and config.get('vision_pretrained', True) else 'random',
                'seed_before_model': config['seed']}
    for branch, path in paths.items():
        if path is not None:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            identity[branch] = dict(validate_initializer(checkpoint, branch, provenance), sha256=file_sha256(path))
    return identity


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(upstream, config):
    """Compose the released experiment and instantiate only its official module."""
    validate_upstream(upstream)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    upstream = Path(upstream).resolve()
    sys.path.insert(0, str(upstream))
    seed_everything(config['seed'])  # Must precede every model initialization.
    with initialize_config_dir(config_dir=str(upstream / 'config'), version_base=None):
        composed = compose(config_name='base', overrides=[
            'experiment=' + config['upstream_experiment']])
    model = instantiate(composed.module)
    if model.out_channels != 22 or model.left_context != 510 or model.right_context != 0:
        raise RuntimeError('upstream module does not match the approved 22-angle / 510-context architecture')
    model.reproduction_hydra_config = OmegaConf.to_container(composed.module, resolve=True)
    return model


def align_batch(model, batch):
    """Crop TDS context, then use upstream linear prediction interpolation."""
    predictions = model(batch)
    if not hasattr(predictions, 'shape'):
        raise TypeError('expected released dense trajectory module output')
    start = model.left_context
    stop = None if not model.right_context else -model.right_context
    target = batch['joint_angles'][..., start:stop]
    valid = batch['angle_valid_mask'][..., start:stop].bool()
    # angle_valid_mask is authoritative: wrist validity is independent of fingers.
    predictions = model.align_predictions(predictions, target.shape[-1])
    return predictions, target, valid


class OfficialLoss:
    """Masked radians L1 + 0.01 official generic-UmeTrack fingertip distance."""
    def __init__(self, fingertip_weight=0.01):
        from egoemg.metrics import LandmarkDistances
        self.landmarks = LandmarkDistances()
        self.fingertip_weight = fingertip_weight

    def __call__(self, prediction, target, valid):
        import torch
        if not valid.any():
            zero = prediction.sum() * 0.0
            return zero, {'angle_l1_rad': 0.0, 'fingertip_distance': 0.0, 'loss': float(zero.detach()),
                          'valid_angle_count': 0}
        angle = (prediction[valid] - target[valid]).abs().mean()
        # The official FK itself appends two null wrist angles; input MUST be 20.
        # Invalid targets are zero-filled before FK; validity still controls loss.
        finger_valid = valid[:, :20]
        safe_target = torch.where(finger_valid, target[:, :20], torch.zeros_like(target[:, :20]))
        fingertip = self.landmarks(prediction[:, :20], safe_target, finger_valid,
                                  'loss')['loss_landmark/fingertip']
        total = angle + self.fingertip_weight * fingertip
        return total, {'angle_l1_rad': float(angle.detach()),
                       'fingertip_distance': float(fingertip.detach()), 'loss': float(total.detach()),
                       'valid_angle_count': int(valid.sum())}


def make_optimizer(model, config):
    import torch
    optimizer_class = config.get('optimizer_class', 'AdamW')
    if optimizer_class not in ('Adam', 'AdamW'):
        raise ValueError('optimizer_class must be Adam or AdamW')
    optimizer = getattr(torch.optim, optimizer_class)(model.parameters(), lr=config['lr'],
                                                     weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['eta_min'])
    return optimizer, scheduler


def model_kind(config):
    kind = config.get('model_kind', 'emg')
    if kind not in ('emg', 'vision', 'fusion'):
        raise ValueError('model_kind must be emg, vision or fusion')
    return kind


def build_training_model(upstream, config, *, vision_checkpoint=None, emg_checkpoint=None, resume_state=None):
    if model_kind(config) == 'emg':
        return build_model(upstream, config)
    validate_upstream(upstream)
    from .visual import build_visual_model
    return build_visual_model(upstream, config, vision_checkpoint=vision_checkpoint,
                              emg_checkpoint=emg_checkpoint, resume_state=resume_state)


def aligned_training_batch(model, batch, config):
    if model_kind(config) == 'emg':
        return align_batch(model, batch)
    from .visual import align_visual_batch
    return align_visual_batch(model, batch)


def build_datasets(data, manifest, norm, config, *, crops=None):
    from .data import LegacyDataset
    kind = model_kind(config)
    if kind != 'emg' and crops is None:
        raise ValueError('--crops is required for visual/fusion data')
    datasets = {}
    for role in ('train', 'val', 'test'):
        windows = [window for window in manifest['windows'] if window['role'] == role]
        if kind == 'emg':
            datasets[role] = LegacyDataset(data, windows, norm=norm)
        else:
            from .visual import VisualDataset
            datasets[role] = VisualDataset(data, windows, crops, norm=norm, kind=kind,
                                           emg_window_length=config.get('emg_window_length', config['window_length']),
                                           emg_layout=config.get('emg_layout', 'target_hand'))
    return datasets


def selection_score(report, config):
    sampling = 'trajectory' if model_kind(config) == 'emg' else 'center'
    return report['sampling'][sampling]['overall']['mae_deg']


def augment_training_batch(batch, config):
    """Apply the released transform to each (time,channel) sample after normalization."""
    import torch
    if config.get('augmentation') == 'official_rotation':
        from egoemg.transforms import RotationAugmentation
        rotation = RotationAugmentation()
        batch = dict(batch)
        batch['emg'] = torch.stack([rotation(sample.T).T for sample in batch['emg']])
    elif config.get('augmentation') != 'none':
        raise ValueError('unknown augmentation policy')
    return batch


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v
            for k, v in batch.items()}


def precision_context(device, config):
    import torch
    if str(device).startswith('cuda') and config['precision'] == 'bfloat16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def evaluate(model, loader, device, config, max_batches=None):
    """Deterministic inference with explicit trajectory/center and crop populations."""
    import torch
    visual = model_kind(config) != 'emg'
    meters = {mode: AngleAccumulator(mode) for mode in (('center',) if visual else ('trajectory', 'center'))}
    paired = PairedCenterAccumulator() if visual else None
    model.eval()
    batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            with precision_context(device, config):
                prediction, target, valid = aligned_training_batch(model, batch, config)
            prediction, target, valid = (x.float().cpu().numpy() if x.dtype != torch.bool
                                         else x.cpu().numpy() for x in (prediction, target, valid))
            center = 0 if visual else batch['emg'].shape[-1] // 2 - model.left_context
            for index in range(len(prediction)):
                metadata = {'hand': batch['hand'][index], 'split': batch['split'][index],
                            'user': str(batch['subject'][index].item() if hasattr(batch['subject'][index], 'item')
                                        else batch['subject'][index])}
                for meter in meters.values():
                    meter.update(prediction[index], target[index], valid[index],
                                 center_index=center, **metadata)
                if paired is not None:
                    crop_valid = bool(batch['vision_valid_mask'][index].bool().any().item())
                    paired.update(prediction[index], target[index], valid[index],
                                  vision_valid=crop_valid, **metadata)
            batches += 1
    report = {'sampling': {mode: meter.report() for mode, meter in meters.items()},
              'evaluated_batches': batches, 'bounded': max_batches is not None,
              'trajectory_note': 'unavailable: fixed-center targets only' if visual else
                  'aligned samples in declared windows; overlaps, if any, count repeatedly'}
    if paired is not None:
        report['paired_population'] = paired.report()
        report['selection_population'] = 'all requested centers, including missing-crop black images'
    return report


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def append_event(output, event):
    with (Path(output) / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def rng_state(device):
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None}


def restore_rng(state, device):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and str(device).startswith('cuda'):
        torch.cuda.set_rng_state(state['cuda'].cpu(), device=device)


def validation_best(checkpoint, *, bounded):
    if bool(checkpoint.get('validation_bounded', False)) != bounded:
        return float('inf')  # Incomparable validation subsets cannot select a best model.
    value = checkpoint['best_val_mae_deg']
    return value if value is not None else float('inf')


def _cpu_snapshot(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def checkpoint_with_best(payload, best_checkpoint=None):
    """Embed a complete best training state; no sibling file is authoritative."""
    checkpoint = dict(payload)
    checkpoint['checkpoint_schema'] = 2
    checkpoint['best'] = _cpu_snapshot(payload if best_checkpoint is None else best_checkpoint)
    selected_best_checkpoint(checkpoint)
    return checkpoint


def selected_best_checkpoint(checkpoint, provenance=None, *, bounded=None):
    if checkpoint.get('checkpoint_schema') != 2 or 'best' not in checkpoint:
        raise ValueError('checkpoint lacks self-contained best state; legacy checkpoint migration is not supported')
    best = checkpoint['best']
    required = ('model', 'optimizer', 'scheduler', 'rng', 'epoch', 'next_batch', 'global_step',
                'best_val_mae_deg', 'validation_bounded', 'provenance', 'run_id')
    if any(key not in best for key in required):
        raise ValueError('checkpoint best training state is incomplete')
    score = checkpoint['best_val_mae_deg']
    if score is None or not math.isfinite(score) or score != best['best_val_mae_deg']:
        raise ValueError('checkpoint best score does not match embedded best state')
    if best['global_step'] > checkpoint['global_step']:
        raise ValueError('best checkpoint cannot follow the current step')
    if best['run_id'] != checkpoint.get('run_id'):
        raise ValueError('embedded best belongs to another run')
    if best['validation_bounded'] != checkpoint['validation_bounded']:
        raise ValueError('embedded best uses an incompatible validation policy')
    if bounded is not None and checkpoint['validation_bounded'] != bounded:
        raise ValueError('checkpoint best uses an incompatible validation policy')
    assert_compatible_provenance(best['provenance'], checkpoint['provenance'])
    if provenance is not None:
        assert_compatible_provenance(checkpoint['provenance'], provenance)
        assert_compatible_provenance(best['provenance'], provenance)
    return best


def _estimated_checkpoint_bytes(value):
    """Conservative payload+serialization allowance; do not credit old files as free."""
    def size(item):
        if hasattr(item, 'nelement') and hasattr(item, 'element_size'):
            return item.nelement() * item.element_size() + 1024
        if isinstance(item, np.ndarray):
            return item.nbytes + 1024
        if isinstance(item, dict):
            return sum(size(key) + size(val) for key, val in item.items()) + 1024
        if isinstance(item, (tuple, list)):
            return sum(size(val) for val in item) + 1024
        if isinstance(item, (bytes, str)):
            return len(item) * 4 + 64
        return 64
    return 2 * size(value) + 1024**2


def save_checkpoint(path, checkpoint, *, serializer=None):
    """Atomically publish while retaining the shared 15 GiB free-space reserve."""
    path = Path(path)
    temporary = Path(str(path) + '.tmp')
    required = _estimated_checkpoint_bytes(checkpoint)
    # free already excludes every existing last/best/temporary file. Replacing a
    # checkpoint first needs all new bytes alongside its still-live old version.
    if shutil.disk_usage(path.parent).free < MIN_FREE_BYTES + required:
        raise RuntimeError(f'insufficient storage for checkpoint temporary file plus {MIN_FREE_BYTES} byte reserve')
    if serializer is None:
        import torch
        serializer = torch.save
    try:
        serializer(checkpoint, temporary)
        if shutil.disk_usage(path.parent).free < MIN_FREE_BYTES:
            raise RuntimeError('serialized checkpoint would violate the storage reserve')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_checkpoints(output, checkpoint, *, improved, serializer=None):
    output = Path(output)
    best = selected_best_checkpoint(checkpoint)
    save_checkpoint(output / 'last.pt', checkpoint, serializer=serializer)
    # last.pt is the single authoritative transaction and already includes best.
    # best.pt is a recoverable convenience projection, including optimizer/RNG.
    if improved or not (output / 'best.pt').exists():
        save_checkpoint(output / 'best.pt', checkpoint_with_best(best, best), serializer=serializer)


def prepare_output(output, resume_checkpoint=None, *, loader=None):
    """Permit only a fresh directory or artifacts verifiably belonging to this run."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not any(output.iterdir()):
        return resume_checkpoint['run_id'] if resume_checkpoint else str(uuid.uuid4())
    if resume_checkpoint is None:
        raise ValueError('fresh output directory contains existing artifacts; use --resume or an empty output')
    selected_best_checkpoint(resume_checkpoint)
    run_path, provenance_path = output / 'run.json', output / 'provenance.json'
    if not run_path.exists() or load_config(run_path).get('run_id') != resume_checkpoint['run_id']:
        raise ValueError('output artifacts belong to an unknown or unrelated run')
    if not provenance_path.exists():
        raise ValueError('output lacks run provenance')
    assert_compatible_provenance(load_config(provenance_path), resume_checkpoint['provenance'])
    if loader is None:
        import torch
        loader = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    for name in ('last.pt', 'best.pt'):
        path = output / name
        if path.exists():
            existing = loader(path)
            if existing.get('run_id') != resume_checkpoint['run_id']:
                raise ValueError(f'{name} belongs to an unrelated run')
            selected_best_checkpoint(existing, resume_checkpoint['provenance'])
    return resume_checkpoint['run_id']


def make_checkpoint(model, optimizer, scheduler, *, epoch, next_batch, step, best,
                    provenance, run_id, validation_bounded=False, best_checkpoint=None):
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'epoch': epoch, 'next_batch': next_batch,
               'global_step': step, 'best_val_mae_deg': best,
               'rng': rng_state(next(model.parameters()).device), 'provenance': provenance,
               'validation_bounded': validation_bounded, 'run_id': run_id}
    return checkpoint_with_best(payload, best_checkpoint)


def resolve_configuration(args, loaded=None):
    """Keep original recipe identity separate from effective, override-resolved values."""
    supplied = load_config(args.config)
    if loaded is None:
        recipe_bytes = Path(args.config).read_bytes()
        config = copy.deepcopy(supplied)
    else:
        recipe = loaded['provenance'].get('recipe')
        if not recipe or 'source_utf8' not in recipe or 'effective_config' not in recipe:
            raise ValueError('checkpoint lacks original/effective config artifacts; explicit migration is required')
        recipe_bytes = recipe['source_utf8'].encode('utf-8')
        expected_source = loaded['provenance']['reproduction']['selected_recipe_sha256']
        if hashlib.sha256(recipe_bytes).hexdigest() != expected_source:
            raise ValueError('checkpoint original recipe content does not match its identity')
        original, effective = json.loads(recipe_bytes), recipe['effective_config']
        if canonical_hash(effective) != loaded['provenance']['config_sha256']:
            raise ValueError('checkpoint effective config content does not match its identity')
        if canonical_hash(supplied) not in (canonical_hash(original), canonical_hash(effective)):
            raise ValueError('supplied config differs from original recipe and effective run config')
        # Original-recipe callers recover initial CLI overrides automatically.
        config = copy.deepcopy(effective)
    for name in ('epochs', 'batch_size', 'num_workers'):
        override = getattr(args, name)
        if override is not None:
            config[name] = override
    if model_kind(config) != 'emg' and 'augmentation' not in config:
        config['augmentation'] = 'none'
    if loaded is not None and canonical_hash(config) != loaded['provenance']['config_sha256']:
        raise ValueError('resume/evaluation config overrides differ from the effective run config')
    return config, recipe_bytes


def run(args):
    upstream_source = validate_upstream(args.upstream)
    import torch
    from torch.utils.data import DataLoader
    from .data import LegacyStore, compute_normalization
    from .partitions import build_partitions
    if args.resume and args.evaluate:
        raise ValueError('resume and evaluate are mutually exclusive')
    loaded = None
    if args.resume or args.evaluate:
        loaded = torch.load(args.resume or args.evaluate, map_location='cpu', weights_only=False)
        selected_best_checkpoint(loaded)
    config, recipe_bytes = resolve_configuration(args, loaded)
    if config['batch_size'] < 1 or config['epochs'] < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError('epochs, batch size and max steps must be positive')
    kind = model_kind(config)
    output = Path(args.output)
    reproduction_source = reproduction_identity(model_kind=kind, recipe_bytes=recipe_bytes)
    if kind != 'emg' and not getattr(args, 'crops', None):
        raise ValueError('--crops is required for visual/fusion runs')
    bounded = args.max_steps is not None
    if loaded is not None:
        assert_compatible_provenance(loaded['provenance'], {
            'source': upstream_source, 'reproduction': reproduction_source,
            'config_sha256': canonical_hash(config)})
    run_id = prepare_output(output, loaded)
    if args.evaluate:
        # Evaluation reuses exactly the training partition and normalization.
        manifest = load_config(output / 'partitions.json')
        norm = load_config(output / 'normalization.json')
    else:
        store = LegacyStore(args.data)
        manifest = build_partitions(store, **{k: config[k] for k in
            ('window_length', 'train_stride', 'eval_stride', 'val_fraction')})
        norm = compute_normalization(store, [w for w in manifest['windows'] if w['role'] == 'train'])
    provenance = {'source': upstream_source, 'reproduction': reproduction_source, 'data': data_identity(args.data),
                  'model_kind': kind, 'recipe': {'source_utf8': recipe_bytes.decode('utf-8'), 'effective_config': config},
                  'partition_sha256': canonical_hash(manifest), 'config_sha256': canonical_hash(config),
                  'normalization_sha256': canonical_hash(norm)}
    if kind != 'emg':
        provenance['visual_data'] = visual_data_identity(args.data, args.crops, manifest,
                                                        allow_missing_episodes=bounded)
    provenance['initialization'] = initialization_identity(config, provenance,
        vision_checkpoint=getattr(args, 'vision_checkpoint', None),
        emg_checkpoint=getattr(args, 'emg_checkpoint', None), loaded=loaded)
    if loaded is not None:
        selected_best_checkpoint(loaded, provenance)
    seed_everything(config['seed'])
    torch.set_float32_matmul_precision(config.get('matmul_precision', 'high'))
    model = build_training_model(args.upstream, config,
        vision_checkpoint=getattr(args, 'vision_checkpoint', None),
        emg_checkpoint=getattr(args, 'emg_checkpoint', None),
        resume_state=loaded['model'] if loaded is not None else None).to(args.device)
    if loaded is None and hasattr(model, 'reproduction_initialization'):
        provenance['initialization']['official_builder'] = model.reproduction_initialization
    optimizer, scheduler = make_optimizer(model, config)
    loss_fn = OfficialLoss(config['fingertip_weight'])
    if loaded is not None:
        selected_best_checkpoint(loaded, provenance)
        model.load_state_dict(loaded['model'])
    if not args.evaluate:
        write_json(output / 'run.json', {'run_id': run_id})
        (output / 'recipe_source.json').write_bytes(recipe_bytes)
        write_json(output / 'config.json', config)
        write_json(output / 'partitions.json', manifest)
        write_json(output / 'normalization.json', norm)
        write_json(output / 'provenance.json', provenance)
        write_json(output / 'official_module.json', model.reproduction_hydra_config)
        write_json(output / 'environment.json', {
            'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'device': args.device, 'gpu_name': torch.cuda.get_device_name(args.device)
                if str(args.device).startswith('cuda') else None,
            'initialization': provenance['initialization'],
            'missing_crop_policy': 'official black image and original valid labels retained in train/eval; all-label selection' if kind != 'emg' else None,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'precision': config['precision'] if str(args.device).startswith('cuda') else 'float32'})
    datasets = build_datasets(args.data, manifest, norm, config, crops=getattr(args, 'crops', None))
    if not all(len(datasets[role]) for role in ('train', 'val', 'test')):
        raise ValueError('all train / independent validation / official test partitions must be nonempty')

    def loader(role, epoch=0):
        generator = torch.Generator().manual_seed(config['seed'] + epoch)
        return DataLoader(datasets[role], batch_size=config['batch_size'], shuffle=role == 'train',
                          num_workers=config['num_workers'], pin_memory=str(args.device).startswith('cuda'),
                          generator=generator, drop_last=False)

    if args.evaluate:
        report = evaluate(model, loader('test'), args.device, config,
                          config['bounded_eval_batches'] if bounded else None)
        report['checkpoint_step'] = loaded['global_step']
        report['partition_sha256'] = provenance['partition_sha256']
        write_json(output / 'evaluation.json', report)
        return report
    epoch, next_batch, step, best = 0, 0, 0, float('inf')
    best_checkpoint = None
    if args.resume:
        optimizer.load_state_dict(loaded['optimizer'])
        scheduler.load_state_dict(loaded['scheduler'])
        epoch, next_batch, step = loaded['epoch'], loaded['next_batch'], loaded['global_step']
        best = validation_best(loaded, bounded=bounded)
        if math.isfinite(best):
            best_checkpoint = selected_best_checkpoint(loaded, provenance)
        # Recover both artifacts from authoritative embedded state, including relocation.
        publish_checkpoints(output, loaded, improved=True)
        restore_rng(loaded['rng'], args.device)
    append_event(output, {'event': 'start', 'config': config, 'bounded_integration': bounded,
                         'official_batch_size': config['official_batch_size'], 'actual_batch_size': config['batch_size'],
                         'batch_adjustment': config['batch_size_note'],
                         'train_samples': len(datasets['train']), 'validation_samples': len(datasets['val']),
                         'test_samples': len(datasets['test']), 'step': step})
    for current_epoch in range(epoch, config['epochs']):
        model.train()
        start_time = time.monotonic()
        train_loader = loader('train', current_epoch)
        epoch_completed = True
        for batch_index, batch in enumerate(train_loader):
            if current_epoch == epoch and batch_index < next_batch:
                continue
            batch = move_batch(augment_training_batch(batch, config), args.device)
            optimizer.zero_grad(set_to_none=True)
            with precision_context(args.device, config):
                prediction, target, valid = aligned_training_batch(model, batch, config)
                loss, terms = loss_fn(prediction, target, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            if terms['valid_angle_count'] == 0:
                next_batch = batch_index + 1
                append_event(output, {'event': 'skipped_batch', 'reason': 'no_valid_angle_labels',
                                     'epoch': current_epoch, 'batch': batch_index, 'step': step,
                                     'optimizer_updated': False})
                continue
            loss.backward()
            norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'],
                                                        error_if_nonfinite=True)
            optimizer.step()
            step += 1
            append_event(output, {'event': 'train_step', 'epoch': current_epoch, 'batch': batch_index,
                                 'step': step, **terms, 'gradient_norm': float(norm_value),
                                 'lr': optimizer.param_groups[0]['lr'],
                                 'elapsed_epoch_seconds': time.monotonic() - start_time,
                                 'cuda_peak_bytes': torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else 0})
            next_batch = batch_index + 1
            if bounded and step >= args.max_steps:
                epoch_completed = next_batch == len(train_loader)
                break
        if epoch_completed:
            scheduler.step()
            checkpoint_epoch, checkpoint_batch = current_epoch + 1, 0
        else:
            checkpoint_epoch, checkpoint_batch = current_epoch, next_batch
        validation = evaluate(model, loader('val'), args.device, config,
                              config['bounded_eval_batches'] if bounded else None)
        score = selection_score(validation, config)
        if score is None:
            raise ValueError('validation has no valid angle observations')
        improved = score < best
        if improved:
            best = score
            best_checkpoint = None
        state = make_checkpoint(model, optimizer, scheduler,
                                epoch=checkpoint_epoch, next_batch=checkpoint_batch, step=step, best=best,
                                provenance=provenance, run_id=run_id, validation_bounded=bounded,
                                best_checkpoint=best_checkpoint)
        publish_checkpoints(output, state, improved=improved)
        best_checkpoint = selected_best_checkpoint(state, provenance, bounded=bounded)
        append_event(output, {'event': 'validation', 'epoch': current_epoch, 'step': step,
                             'improved': improved, 'best_val_mae_deg': best, 'metrics': validation})
        next_batch = 0
        if bounded and step >= args.max_steps:
            result = {'status': 'bounded_integration_complete', 'global_step': step,
                      'test_evaluated': False, 'validation': validation}
            write_json(output / 'summary.json', result)
            return result
    if bounded:
        result = {'status': 'bounded_integration_exhausted', 'global_step': step,
                  'requested_max_steps': args.max_steps, 'test_evaluated': False,
                  'best_val_mae_deg': best if math.isfinite(best) else None,
                  'reason': 'configured epochs exhausted before the optimizer-update budget'}
        write_json(output / 'summary.json', result)
        return result
    # Official test is evaluated once, after training, at the validation-selected checkpoint.
    authoritative = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    best_state = selected_best_checkpoint(authoritative, provenance, bounded=False)
    if best_state['best_val_mae_deg'] != best:
        raise ValueError('final best score does not match the selected checkpoint')
    model.load_state_dict(best_state['model'])
    test = evaluate(model, loader('test'), args.device, config)
    result = {'status': 'training_complete', 'global_step': step,
              'selected_checkpoint_step': best_state['global_step'], 'test_evaluated': True,
              'best_val_mae_deg': best, 'test': test, 'provenance': provenance}
    write_json(output / 'summary.json', result)
    append_event(output, {'event': 'final_test', **result})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'upstream', 'output', 'config'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--crops')
    parser.add_argument('--vision-checkpoint')
    parser.add_argument('--emg-checkpoint')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--evaluate', metavar='CHECKPOINT')
    parser.add_argument('--device', default='cuda:0')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
