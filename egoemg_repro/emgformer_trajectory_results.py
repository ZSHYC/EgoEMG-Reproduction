"""Native same-forward alignment, lossless raw packets and trajectory reductions.

Raw utilities also accept explicit engineering tensors. They never certify a
trained producer, complete population, fit selection or scientific experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .classic_emg import _activate, _file_hash, _hash
from .classic_emg_cohort import _safe, _StatGuard
from .classic_emg_train import state_digest
from .classic_emg_report import JOINT_NAMES, FINGERS, PD_GROUPS
from .metrics import AngleAccumulator
from .paper_analysis import JOINT_GROUPS

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = _file_hash(__file__)
W = 12000
OFFICIAL = {
    'egoemg/lightning.py': 'f32b97dab2bb94aff35d0513d964b15087a859caaf11f36030f79074edae1899',
    'egoemg/models/modules/base.py': 'b19dbbfe77db813f20553ded787bb7df1825cd48e529a2312e8c1caa3c83a54b',
}
ARRAYS = {'raw_prediction', 'prediction', 'target', 'mask', 'source_target', 'source_mask'}
META = {'index', 'unique', 'repeat', 'start', 'end', 'center', 'episode', 'hand', 'split', 'user'}


def _sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def _seal(value):
    return dict(value, sha256=_hash(value))


def _sealed(value):
    if not isinstance(value, dict) or value.get('sha256') != _hash({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('exact raw trajectory record seal required')
    return value


def _guard(upstream):
    root = _safe(upstream)
    local = [Path(__file__), ROOT/'egoemg_repro/metrics.py', ROOT/'egoemg_repro/classic_emg_report.py',
        ROOT/'egoemg_repro/classic_emg_train.py', ROOT/'egoemg_repro/paper_analysis.py']
    guard = _StatGuard([*local, *(root/n for n in OFFICIAL)])
    if _file_hash(__file__) != IMPORTED_SHA or any(_sha(root/n) != h for n, h in OFFICIAL.items()):
        raise ValueError('original trajectory result/forward source changed')
    _activate(root); guard.check()
    return guard


def _alignment(left, right, raw_steps):
    if any(type(x) is not int or x < 0 for x in (left, right)) or left+right >= W or type(raw_steps) is not int or raw_steps < 1:
        raise ValueError('actual original context and prediction length required')
    time = W-left-right
    return dict(window_length=W, left_context=left, right_context=right,
        raw_prediction_time_steps=raw_steps, aligned_prediction_time_steps=time,
        aligned_center_index=time//2, source_aligned_center_index=left+time//2,
        original_point_center_index=W//2)


def _validate_arrays(arrays, alignment):
    import torch
    if not isinstance(arrays, dict) or set(arrays) != ARRAYS:
        raise ValueError('all six original trajectory tensors required')
    a = alignment
    if not isinstance(a, dict) or a != _alignment(a.get('left_context'), a.get('right_context'), a.get('raw_prediction_time_steps')):
        raise ValueError('exact original trajectory alignment metadata required')
    p, raw, t, mask = (arrays[n] for n in ('prediction', 'raw_prediction', 'target', 'mask'))
    source, valid = arrays['source_target'], arrays['source_mask']
    if any(not torch.is_tensor(x) or x.device.type != 'cpu' for x in arrays.values()):
        raise ValueError('actual CPU raw trajectory tensors required')
    n, time = len(p), a['aligned_prediction_time_steps']
    if (not 1 <= n <= 500 or p.shape != (n, 22, time) or t.shape != p.shape
        or raw.shape != (n, 22, a['raw_prediction_time_steps']) or source.shape != (n, 22, W)
        or mask.shape != (n, time) or valid.shape != (n, W)
        or p.dtype not in (torch.float32, torch.bfloat16) or raw.dtype != p.dtype
        or t.dtype != torch.float32 or source.dtype != torch.float32
        or mask.dtype != torch.bool or valid.dtype != torch.bool
        or any(not bool(torch.isfinite(x).all()) for x in (p, raw, t, source))):
        raise ValueError('typed finite full native trajectory schema differs')
    stop = None if a['right_context'] == 0 else -a['right_context']
    if (not torch.equal(t, source[..., a['left_context']:stop])
        or not torch.equal(mask, valid[..., a['left_context']:stop])):
        raise ValueError('raw original context-cropped targets/masks differ')
    return n


def capture_native_trajectory(module, batch, *, upstream):
    """Observe exactly one original full forward, including its same head output."""
    import torch
    guard = _guard(upstream)
    model = module.model
    if (module.training or model.training or getattr(model, 'center_supervised', False)
        or getattr(model, 'supervise_at_prediction_rate', False) or module.ignore_head_tail_dims != 0
        or batch['joint_angles'].ndim != 3 or batch['joint_angles'].shape[1:] != (22, W)
        or batch['joint_angles'].dtype != torch.float32
        or batch['label_valid_mask'].shape != (len(batch['emg']), W)
        or batch['label_valid_mask'].dtype != torch.bool):
        raise ValueError('original eval-mode full-target forward required')
    from egoemg.models.modules.base import BaseModule
    if (getattr(model.align_predictions, '__func__', None) is not BaseModule.align_predictions
        or getattr(model.align_mask, '__func__', None) is not BaseModule.align_mask):
        raise ValueError('original native alignment callables required')
    observed = []
    def head_hook(owner, arguments, output): observed.append(output.detach().clone())
    hook = model.head.register_forward_hook(head_hook)
    try:
        with torch.inference_mode():
            out = module.forward(batch)
            if not isinstance(out, tuple) or len(out) != 3 or len(observed) != 1:
                raise ValueError('exact one-head native full forward tuple required')
            pred, target, mask = out; raw = observed[0]
            alignment = _alignment(model.left_context, model.right_context, raw.shape[-1])
            stop = None if model.right_context == 0 else -model.right_context
            wanted = batch['joint_angles'][..., model.left_context:stop]
            wanted_mask = batch['label_valid_mask'][..., model.left_context:stop]
            aligned = model.align_predictions(raw, wanted.shape[-1])
            wanted_mask = model.align_mask(wanted_mask, wanted.shape[-1])
            if (pred.dtype != aligned.dtype or not torch.equal(pred, aligned)
                or target.dtype != wanted.dtype or not torch.equal(target, wanted)
                or mask.dtype != wanted_mask.dtype or not torch.equal(mask, wanted_mask)):
                raise ValueError('same-forward original full alignment/target/mask differs')
            def cpu(x): return x.detach().cpu().clone().contiguous()
            arrays = dict(raw_prediction=cpu(raw), prediction=cpu(pred), target=cpu(target), mask=cpu(mask),
                source_target=cpu(batch['joint_angles']), source_mask=cpu(batch['label_valid_mask']))
            _validate_arrays(arrays, alignment); guard.check()
            return dict(arrays=arrays, alignment=alignment, actual_model_forwards=1,
                same_forward_original_alignment_exact=True, input_sha256=state_digest(batch),
                scientific_eligible=False, full_paper_complete=False)
    finally:
        hook.remove()


def _metadata(metadata, n, input_sha256, sample_input_sha256):
    if not isinstance(metadata, dict) or set(metadata) != META or any(type(x) is not list or len(x) != n for x in metadata.values()):
        raise ValueError('complete original trajectory source metadata required')
    for name in META-{'hand', 'split', 'user'}:
        if any(type(x) is not int or x < 0 for x in metadata[name]):
            raise ValueError('integer trajectory source coordinates required')
    if (any(x not in ('left', 'right') for x in metadata['hand'])
        or any(x not in ('train', 'user', 'gesture', 'both') for x in metadata['split'])
        or any(type(x) is not str or not x for x in metadata['user'])
        or any(b-a != W or c != a+W//2 for a, b, c in zip(metadata['start'], metadata['end'], metadata['center']))):
        raise ValueError('native trajectory hand/window/subject metadata differs')
    if (not isinstance(input_sha256, str) or re.fullmatch('[a-f0-9]{64}', input_sha256) is None
        or type(sample_input_sha256) is not list or len(sample_input_sha256) != n
        or any(not isinstance(x, str) or re.fullmatch('[a-f0-9]{64}', x) is None for x in sample_input_sha256)):
        raise ValueError('actual typed trajectory input digests required')


def write_trajectory_packet(directory, ordinal, *, arrays, alignment, metadata, input_sha256,
                            sample_input_sha256, upstream):
    import torch
    guard = _guard(upstream); n = _validate_arrays(arrays, alignment)
    _metadata(metadata, n, input_sha256, sample_input_sha256)
    if type(ordinal) is not int or ordinal < 0: raise ValueError('integer raw trajectory ordinal required')
    root = _safe(directory); path = root/f'{ordinal:06d}.pt'
    with path.open('xb') as f: torch.save(arrays, f); f.flush(); os.fsync(f.fileno())
    guard.check()
    return _seal(dict(format='raw_native_full_trajectory_batch_v1', ordinal=ordinal, file=path.name,
        points=n, alignment=alignment, metadata=metadata, input_sha256=input_sha256,
        sample_input_sha256=sample_input_sha256, file_sha256=_sha(path),
        tensors_sha256={name: state_digest(value) for name, value in arrays.items()},
        trained_producer_or_full_population_permit=False))


def read_trajectory_packet(directory, record, *, upstream):
    import torch
    guard = _guard(upstream); record = _sealed(record)
    try:
        if (record['format'] != 'raw_native_full_trajectory_batch_v1' or type(record['ordinal']) is not int
            or record['ordinal'] < 0 or record['file'] != f"{record['ordinal']:06d}.pt"
            or record['trained_producer_or_full_population_permit'] is not False):
            raise ValueError('exact original raw trajectory batch record required')
        path = _safe(Path(directory)/record['file']); guard.add(path)
        if _sha(path) != record['file_sha256']: raise ValueError('raw trajectory file bytes changed')
        arrays = torch.load(path, map_location='cpu', weights_only=True)
        n = _validate_arrays(arrays, record['alignment'])
        _metadata(record['metadata'], n, record['input_sha256'], record['sample_input_sha256'])
        if n != record['points'] or {k: state_digest(v) for k, v in arrays.items()} != record['tensors_sha256']:
            raise ValueError('raw typed trajectory tensors changed')
        guard.check()
        return dict(arrays=arrays, alignment=record['alignment'], metadata=record['metadata'],
            input_sha256=record['input_sha256'], sample_input_sha256=record['sample_input_sha256'])
    except (OSError, KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('complete original raw trajectory packet unavailable') from exc


def trajectory_statistics(packets, *, samples, batch_size, role, upstream):
    """Stream a complete role; retain native batch and separate pooled metrics."""
    import torch
    guard = _guard(upstream)
    if (type(samples) is not int or samples <= 0 or samples%2 or role not in ('val', 'test')
        or type(batch_size) is not int or not 1 <= batch_size <= 500):
        raise ValueError('complete trajectory role and original batch size required')
    accumulators = {name: AngleAccumulator('trajectory' if name == 'trajectory' else 'center')
        for name in ('trajectory', 'aligned_center')}
    sums = {name: np.zeros(22, np.float64) for name in accumulators}
    counts = {name: np.zeros(22, np.int64) for name in accumulators}
    split_users = {name: {} for name in accumulators}; paired = {}; position = 0; sizes = []; alignment = None
    weighted = torch.tensor(0., dtype=torch.float32)
    for packet in packets:
        guard.check(); arrays = packet['arrays']; a = packet['alignment']; metadata = packet['metadata']
        n = _validate_arrays(arrays, a); _metadata(metadata, n, packet['input_sha256'], packet['sample_input_sha256'])
        if (position >= samples or n != min(batch_size, samples-position)
            or metadata['index'] != list(range(position, position+n)) or metadata['unique'] != metadata['index']
            or any(metadata['repeat']) or (alignment is not None and alignment != a)):
            raise ValueError('complete exact ordered trajectory batches required')
        alignment = a; sizes.append(n); pred, target, mask = arrays['prediction'].float(), arrays['target'], arrays['mask']
        expanded = mask[:, None].expand_as(target)
        mean = (pred-target).abs()[expanded].mean() if bool(expanded.any()) else pred.sum()*0.
        if not bool(torch.isfinite(mean)): raise ValueError('finite original batch MAE required')
        weighted += mean*n
        for j, index in enumerate(metadata['index']):
            hand, split, user = (metadata[k][j] for k in ('hand', 'split', 'user'))
            if hand != ('left' if index < samples//2 else 'right') or split not in (('train',) if role == 'val' else ('user', 'gesture', 'both')):
                raise ValueError('original trajectory hand/role order differs')
            key = index%(samples//2)
            value = tuple(metadata[k][j] for k in ('start', 'end', 'center', 'episode', 'split', 'user'))
            if index < samples//2:
                if any(v[2] == value[2] for v in paired.values()): raise ValueError('duplicate original trajectory window')
                paired[key] = value
            elif paired.get(key) != value: raise ValueError('paired trajectory source window differs')
            p, t, valid = pred[j].numpy(), target[j].numpy(), mask[j].numpy()
            for name, acc in accumulators.items():
                acc.update(p, t, valid, hand=hand, split=split, user=user, center_index=a['aligned_center_index'])
                pp, tt, mm = p, t, valid
                if name == 'aligned_center':
                    center = a['aligned_center_index']; pp, tt, mm = p[:, center:center+1], t[:, center:center+1], valid[center:center+1]
                error = np.abs(pp.astype(np.float64)-tt.astype(np.float64))[:, mm]
                sums[name] += error.sum(axis=1); counts[name] += int(mm.sum())
                cell = split_users[name].setdefault(split, {}).setdefault(user, [0., 0])
                cell[0] += float(error.sum()); cell[1] += int(error.size)
        position += n
    if position != samples: raise ValueError('complete trajectory population was not consumed')
    reports = {}
    for name, acc in accumulators.items():
        report = acc.report(); ss, cc = sums[name], counts[name]
        def record(ids): return AngleAccumulator._record([float(ss[list(ids)].sum()), int(cc[list(ids)].sum())])
        report['by_joint'] = [dict(index=i, name=JOINT_NAMES[i], **record([i])) for i in range(22)]
        report['by_finger'] = {k: record(ids) for k, ids in FINGERS.items()}
        report['by_pd_group'] = {k: record(ids) for k, ids in PD_GROUPS.items()}
        report['by_joint_group'] = {k: dict(indices=list(ids), **record(ids)) for k, ids in JOINT_GROUPS.items()}
        report['by_split_user_summary'] = {}
        for split, users in sorted(split_users[name].items()):
            means = [e/c*180/math.pi for e, c in users.values() if c]
            report['by_split_user_summary'][split] = dict(user_count=len(means),
                mean_mae_deg=float(np.mean(means)) if means else None,
                std_mae_deg=float(np.std(means)) if means else None,
                aggregation='unweighted user means; ddof=0')
        if (int(cc.sum()) != report['overall']['valid_angle_count']
            or not math.isclose(float(ss.sum()), report['overall']['absolute_error_sum_rad'], rel_tol=2e-12, abs_tol=1e-9)):
            raise ValueError('trajectory joint and pooled statistics do not conserve')
        reports[name] = report
    guard.check()
    return dict(format='original_native_full_trajectory_statistics_v1', role=role,
        consumed_samples=samples, unique_samples=samples, duplicate_consumptions=0,
        batch_sizes=sizes, alignment=alignment, native_batch_weighted_mae_rad=float(weighted/samples),
        **reports)
