"""Literal paired center dispatch and raw arrays, without a full-run permit.

The complete runner must separately bind full inputs, all original keys and GT,
then replay every saved batch. Adapter engineering doubles are not producers.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from .classic_emg import _file_hash
from .classic_emg_cohort import _StatGuard
from .occlusion_dense_analysis import source_analysis_contract
from .occlusion_dense_predictions import KEY_FIELDS, _key

IMPORTED_SHA = _file_hash(__file__)


def _source_guard():
    guard = _StatGuard([Path(__file__)])
    if _file_hash(__file__) != IMPORTED_SHA:
        raise ValueError('imported dense inference adapter changed')
    return guard


def source_collate_fusion(samples, *, upstream):
    """Execute the unchanged function from the pinned evaluation script.

    It is defined in scripts/eval/evaluate_occlusion_fusion.py, rather than
    egoemg.datamodule. Compile that function only, without geometry side effects.
    Its deliberate field selection (including omitted channel masks) is kept.
    """
    import torch
    guard = _source_guard()
    source_analysis_contract(upstream)
    path = Path(upstream)/'scripts/eval/evaluate_occlusion_fusion.py'
    guard.add(path)
    tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_collate_fusion')
    namespace = dict(np=np, torch=torch)
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec', dont_inherit=True)
    exec(code, namespace)
    value = namespace['_collate_fusion'](samples)
    guard.check()
    return value


def native_batch_ranges(keys):
    """Source B64 with a fresh tail for every split/hand group; no filtering."""
    guard = _source_guard()
    keys = list(keys)
    if not keys:
        raise ValueError('nonempty ordered native keys required')
    seen, groups, begin, previous = set(), [], 0, -1
    for i, key in enumerate(keys):
        _key(key, KEY_FIELDS)
        group = key.get('group_index')
        if (key['sequence'] != i or type(group) is not int or group not in (0, 1, 2)
            or key['split'] != ('user', 'gesture', 'both')[group]
            or key['hand'] not in ('left', 'right')
            or key['window_end']-key['window_start'] not in (7790, 12000)
            or key['center_idx'] != (key['window_start']+key['window_end'])//2):
            raise ValueError('exact ordered native key/center/group required')
        point = (key['hand'], key['ep_idx'], key['center_idx'])
        rank = 2*group + (key['hand'] == 'right')
        if point in seen or rank < previous:
            raise ValueError('duplicate point or reopened native split/hand group')
        seen.add(point)
        if rank != previous:
            if i:
                groups.append((begin, i))
            begin, previous = i, rank
    groups.append((begin, len(keys)))
    guard.check()
    return [(i, min(i+64, stop)) for start, stop in groups for i in range(start, stop, 64)]


def native_forward_arrays(models, fusion_batch, standalone_batch):
    """Source-native forwards, midpoint and literal unmasked22-angle errors.

    This computational adapter accepts engineering modules. Only the wrapper
    below admits real trained models; neither API admits complete input scope.
    """
    import torch
    guard = _source_guard()
    if (not isinstance(models, dict) or set(models) != {'fusion', 'vision', 'emg'}
        or any(not isinstance(m, torch.nn.Module) or any(x.training for x in m.modules())
            or any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in m.parameters())
            for m in models.values())):
        raise ValueError('three actual eval-mode CPU FP32 branches required; no zero fallback')
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise ValueError('source FP32 forward without autocast required')
    a, b = fusion_batch, standalone_batch
    try:
        emg, solo = a['emg'], b['emg']
        targets, mask = a['joint_angles'], a['label_valid_mask']
        size = emg.shape[0]
        if (emg.ndim != 3 or size < 1 or emg.shape[-1] not in (7790, 12000)
            or emg.shape[1] != (16 if emg.shape[-1] == 7790 else 8)
            or tuple(solo.shape) != (size, 8, emg.shape[-1])
            or tuple(targets.shape) != (size, 22, 1)
            or mask.dtype != torch.bool or tuple(mask.shape) != (size, 1)
            or not torch.equal(targets, b['joint_angles'])
            or not torch.equal(mask, b['label_valid_mask'])
            or b['joint_angles'].dtype != torch.float32 or b['label_valid_mask'].dtype != torch.bool
            or a['vision_img'].shape[0] != size or a['vision_img'].ndim not in (4, 5)
            or a['vision_valid_mask'].shape[0] != size or a['vision_valid_mask'].dtype != torch.bool):
            raise ValueError('same exact window, separate channel layouts and equal center labels/masks required')
        for tensor in (emg, solo, targets, a['vision_img']):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.device.type != 'cpu' or not torch.isfinite(tensor).all()):
                raise ValueError('complete finite source CPU FP32 inputs required; no casting/repair')
    except (KeyError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError('complete original paired native batch required') from exc
    with torch.no_grad():
        out = models['fusion'].forward(a)
        if not isinstance(out, tuple) or len(out) != 3:
            raise ValueError('native fusion center prediction/target/mask tuple required')
        fusion, actual_target, actual_mask = out
        if (not torch.equal(actual_target, targets) or actual_target.dtype != targets.dtype
            or not torch.equal(actual_mask, mask) or actual_mask.dtype != mask.dtype):
            raise ValueError('native fusion target or label mask changed')
        vision = models['vision'](a)
        if isinstance(vision, tuple):
            vision = vision[0]
        # Labels must not cause the native EMG forward to interpolate its entire
        # trajectory to the single target sample. Preserve the raw timeline.
        full_emg = models['emg']({'emg': solo})
        if isinstance(full_emg, tuple):
            full_emg = full_emg[0]
        if (tuple(fusion.shape) != (size, 22, 1) or tuple(vision.shape) != (size, 22, 1)
            or full_emg.ndim != 3 or tuple(full_emg.shape[:2]) != (size, 22) or full_emg.shape[-1] < 1):
            raise ValueError('original22-angle center/full-EMG output shapes required')
        for tensor in (fusion, vision, full_emg):
            if tensor.dtype != torch.float32 or tensor.device.type != 'cpu' or not torch.isfinite(tensor).all():
                raise ValueError('finite original FP32 predictions required')
        center = full_emg.shape[-1]//2
        tensors = dict(fusion_prediction=fusion.squeeze(-1), vision_prediction=vision.squeeze(-1),
            emg_prediction=full_emg[:, :, center], emg_full_prediction=full_emg,
            targets=targets.squeeze(-1), label_valid_mask=mask)
        errors = {'mae_'+role+'_rad': np.array([float((tensors[role+'_prediction'][i]
            - targets[i, :, 0]).abs().mean()) for i in range(size)], dtype=np.float64)
            for role in ('fusion', 'vision', 'emg')}
        arrays = {n: x.detach().cpu().numpy().copy() for n, x in tensors.items()}
    guard.check()
    return dict(arrays=arrays, errors=errors, emg_center_index=center,
        batch_size=size, full_inference_permit=False)


def predict_dense_batch(trained, fusion_batch, standalone_batch):
    """Require registered actual completed native model sources for a batch."""
    from .occlusion_dense_models import DenseTrainedModels
    if not isinstance(trained, DenseTrainedModels):
        raise ValueError('registered real completed trained dense models required')
    trained.check()
    value = native_forward_arrays(trained.models, fusion_batch, standalone_batch)
    trained.check()
    return value
