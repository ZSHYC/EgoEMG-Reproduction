"""Deterministic input interventions, never a trained/full inference permit.

These complementary experiments keep the original frozen model unchanged.
Normalized zeros mean the author's force_zero_emg input, not raw voltage zero.
The circular time shift preserves same-window signal and introduces a seam;
it is not an independent EMG sample or complete removal of pose information.
"""
from __future__ import annotations

import copy
import math
from numbers import Real


def apply_input_condition(batch, *, emg, vision, vision_mean, vision_std):
    """Copy one native single-image CPU FP32 batch and vary only its inputs.

    Image constants are the actual native view's RGB-byte normalization values.
    A caller must separately bind that view, original full inputs and completed
    models. Target/label validity never follows the induced missing-vision mask.
    """
    import torch

    if (emg not in ('aligned', 'normalized_zero', 'half_window_roll')
        or vision not in ('native', 'missing')):
        raise ValueError('explicit fixed EMG and vision input conditions required')
    required = ('emg', 'vision_img', 'vision_valid_mask', 'joint_angles', 'label_valid_mask')
    if (not isinstance(batch, dict) or any(n not in batch for n in required)
        or 'vision_features' in batch):
        raise ValueError('native image batch required; cached features would bypass image intervention')
    if any(not isinstance(batch[n], torch.Tensor) for n in required):
        raise ValueError('complete native tensor inputs required, without implicit conversion')
    signal, image, target = (batch[n] for n in ('emg', 'vision_img', 'joint_angles'))
    if (signal.ndim != 3 or signal.shape[0] < 1 or signal.shape[-1] not in (7790, 12000)
        or signal.shape[1] != (16 if signal.shape[-1] == 7790 else 8)):
        raise ValueError('original7790/16-channel or12000/8-channel input required')
    size, _, length = signal.shape
    if (image.ndim != 4 or image.shape[:2] != (size, 3)
        or min(image.shape[-2:]) < 1 or target.shape != (size, 22, 1)):
        raise ValueError('one original RGB image and22-angle center target per request required')
    for tensor in (signal, image, target):
        if (tensor.device.type != 'cpu' or tensor.dtype != torch.float32
            or tensor.requires_grad or not torch.isfinite(tensor).all()):
            raise ValueError('finite nongradient native CPU float32 inputs required; no repair')
    for name in ('vision_valid_mask', 'label_valid_mask'):
        tensor = batch[name]
        if tensor.dtype != torch.bool or tensor.device.type != 'cpu' or tensor.shape != (size, 1):
            raise ValueError('original boolean per-request vision and label masks required')
    for vector in (vision_mean, vision_std):
        if (not isinstance(vector, (list, tuple)) or len(vector) != 3
            or any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) for v in vector)):
            raise ValueError('three explicit finite native RGB normalization constants required')
    mean = torch.tensor(vision_mean, dtype=torch.float32)
    std = torch.tensor(vision_std, dtype=torch.float32)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('finite float32 means and positive float32 standard deviations required')

    with torch.no_grad():
        output = {name: value.detach().clone() if isinstance(value, torch.Tensor)
            else copy.deepcopy(value) for name, value in batch.items()}
        if emg == 'normalized_zero':
            # The native forward replaces the entire already-normalized tensor,
            # including the16-channel layout's normalized structural padding.
            output['emg'] = torch.zeros_like(signal)
        elif emg == 'half_window_roll':
            output['emg'] = torch.roll(signal, shifts=length//2, dims=-1)
        if vision == 'missing':
            black = (torch.zeros(3, dtype=torch.float32)-mean)/std
            if not torch.isfinite(black).all():
                raise ValueError('native normalized black image must remain finite')
            output['vision_img'] = black.reshape(1, 3, 1, 1).expand_as(image).clone()
            output['vision_valid_mask'] = torch.zeros_like(batch['vision_valid_mask'])

    condition = dict(format='fixed_native_input_condition_v1', emg=emg, vision=vision,
        emg_window_length=length, emg_channels=signal.shape[1],
        emg_roll_samples=length//2 if emg == 'half_window_roll' else 0,
        emg_zero_semantics='all normalized channels, like native force_zero_emg',
        missing_vision_semantics='native normalized RGB-byte0 and false vision mask; original label mask',
        vision_mean=mean.tolist(), vision_std=std.tolist(), deterministic=True,
        model_flags_changed=False, original_batch_mutated=False)
    return dict(batch=output, condition=condition, full_inference_permit=False)
