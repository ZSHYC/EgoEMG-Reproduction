"""Frozen native input-control dispatch; complete evaluation is a separate gate."""
from __future__ import annotations

import numpy as np

from .modality_controls import apply_input_condition


def _check_models(models):
    import torch
    if (not isinstance(models, dict) or set(models) != {'fusion', 'vision', 'emg'}
        or any(not isinstance(m, torch.nn.Module) for m in models.values())):
        raise ValueError('complete actual three-branch model dictionary required')
    for module in models.values():
        if any(m.training for m in module.modules()):
            raise ValueError('all native model modules must remain in eval mode')
        for tensor in (*module.parameters(), *module.buffers()):
            if (tensor.device.type != 'cpu'
                or tensor.is_floating_point() and tensor.dtype != torch.float32):
                raise ValueError('native CPU FP32 model parameters and floating buffers required')
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise ValueError('native FP32 control forwarding forbids autocast')


def native_control_arrays(models, batch, *, emg, vision, vision_mean, vision_std):
    """Apply an input condition and call native fusion/vision center forwards.

    This numerical boundary accepts engineering modules. It never grants trained
    provenance, complete source input admission or a scientific full-run permit.
    """
    import torch
    from .vision_literal_runtime import vision_state_sha256

    _check_models(models)
    intervention = apply_input_condition(batch, emg=emg, vision=vision,
        vision_mean=vision_mean, vision_std=vision_std)
    current = intervention['batch']; size = current['emg'].shape[0]
    tensor_items = lambda packet: {n:x for n,x in packet.items() if isinstance(x, torch.Tensor)}
    original_sha = vision_state_sha256(tensor_items(batch))
    input_sha = vision_state_sha256(tensor_items(current))
    state_before = {n:vision_state_sha256(m.state_dict()) for n,m in models.items()}
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        fused = models['fusion'].forward(current)
        if not isinstance(fused, tuple) or len(fused) != 3:
            raise ValueError('native fusion prediction, target and label-mask tuple required')
        fusion, target, mask = fused
        if (not isinstance(target, torch.Tensor) or not isinstance(mask, torch.Tensor)
            or target.dtype != batch['joint_angles'].dtype or mask.dtype != torch.bool
            or not torch.equal(target, batch['joint_angles']) or not torch.equal(mask, batch['label_valid_mask'])):
            raise ValueError('native control forward changed original targets or label validity')
        visual = models['vision'](current)
        if isinstance(visual, tuple):
            if (len(visual) != 3 or not torch.equal(visual[1], batch['joint_angles'])
                or visual[1].dtype != target.dtype or not torch.equal(visual[2], batch['label_valid_mask'])
                or visual[2].dtype != mask.dtype):
                raise ValueError('native vision forward changed target or label mask')
            visual = visual[0]
        for output in (fusion, visual):
            if (not isinstance(output, torch.Tensor) or output.shape != (size, 22, 1)
                or output.dtype != torch.float32 or output.device.type != 'cpu'
                or not torch.isfinite(output).all()):
                raise ValueError('finite unmodified native float32 center predictions required; no output repair')
        arrays = dict(fusion_prediction=fusion.squeeze(-1).numpy().copy(),
            vision_prediction=visual.squeeze(-1).numpy().copy(), targets=target.squeeze(-1).numpy().copy(),
            label_valid_mask=mask.numpy().copy(),
            natural_crop_valid_mask=batch['vision_valid_mask'].numpy().copy(),
            condition_vision_valid_mask=current['vision_valid_mask'].numpy().copy())
        for role, prediction in (('fusion', fusion), ('vision', visual)):
            arrays['mae_'+role+'_rad'] = np.array([float((prediction[i]-target[i]).abs().mean())
                for i in range(size)], dtype=np.float64)
    _check_models(models)
    state_after = {n:vision_state_sha256(m.state_dict()) for n,m in models.items()}
    if state_before != state_after:
        raise ValueError('native control forwarding changed complete parameter or buffer state')
    if not torch.equal(rng, torch.get_rng_state()):
        raise ValueError('eval-mode control forwarding changed Torch RNG state')
    if (input_sha != vision_state_sha256(tensor_items(current))
        or original_sha != vision_state_sha256(tensor_items(batch))):
        raise ValueError('native control forward mutated original or condition input tensors')
    return dict(arrays=arrays, condition=intervention['condition'],
        all_model_states_before_after_exact=True, model_state_sha256=state_before,
        torch_rng_unchanged=True, original_batch_unchanged=True,
        actual_optimizer_updates=0, full_inference_permit=False)


def predict_control_batch(trained, batch, **condition):
    """Admit only the existing genuinely registered completed producer objects."""
    # Reject arbitrary namespaces before importing the heavier native providers.
    if type(trained).__name__ not in ('DenseTrainedModels', 'RN18DenseTrainedModels'):
        raise ValueError('registered actual completed dense producers required')
    from .occlusion_dense_models import DenseTrainedModels
    from .rn18_occlusion_models import RN18DenseTrainedModels
    if not isinstance(trained, (DenseTrainedModels, RN18DenseTrainedModels)):
        raise ValueError('registered actual completed dense producers required')
    trained.check()
    result = native_control_arrays(trained.models, batch, **condition)
    trained.check()
    result['trained_model_identity_sha256'] = trained.identity['sha256']
    return result
