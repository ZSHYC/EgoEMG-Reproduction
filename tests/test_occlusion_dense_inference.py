"""Adapter engineering cases; these doubles never establish trained provenance."""
import copy
import importlib

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('egoemg_repro.occlusion_dense_inference')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.occlusion_dense_inference':
            pytest.fail('native paired dense forward adapter missing')
        raise


def packets():
    import torch
    a = dict(emg=torch.ones(3, 16, 7790),
        joint_angles=torch.arange(66, dtype=torch.float32).reshape(3, 22, 1) / 17,
        label_valid_mask=torch.tensor([False, True, False]).reshape(3, 1),
        vision_img=torch.zeros(3, 3, 224, 224),
        vision_valid_mask=torch.tensor([[False], [True], [False]]))
    b = {k: v.clone() for k, v in a.items() if k in ('joint_angles', 'label_valid_mask')}
    b['emg'] = torch.full((3, 8, 7790), 2.)
    return a, b


def models():
    import torch
    class Fusion(torch.nn.Module):
        def forward(self, batch):
            assert not torch.is_grad_enabled() and not torch.is_autocast_enabled()
            return batch['joint_angles'] + .3, batch['joint_angles'], batch['label_valid_mask']
    class Vision(torch.nn.Module):
        def forward(self, batch):
            return batch['joint_angles'] - .7, batch['joint_angles'], batch['label_valid_mask']
    class Emg(torch.nn.Module):
        def forward(self, batch):
            # Passing labels would silently invoke original label-length alignment.
            assert set(batch) == {'emg'} and batch['emg'].shape == (3, 8, 7790)
            assert float(batch['emg'].mean()) == 2.
            return torch.arange(6, dtype=torch.float32).reshape(1, 1, 6).expand(3, 22, 6)
    return {k: m.eval() for k, m in dict(fusion=Fusion(), vision=Vision(), emg=Emg()).items()}


def test_native_three_branch_center_and_literal_unmasked_row_errors():
    m = api(); a, b = packets()
    out = m.native_forward_arrays(models(), a, b)
    x = out['arrays']; target = a['joint_angles'].numpy().squeeze(-1)
    assert x['fusion_prediction'].shape == x['vision_prediction'].shape == (3, 22)
    assert x['emg_full_prediction'].shape == (3, 22, 6)
    np.testing.assert_array_equal(x['emg_prediction'], x['emg_full_prediction'][:, :, 3])
    np.testing.assert_array_equal(x['targets'], target)
    np.testing.assert_array_equal(x['label_valid_mask'], a['label_valid_mask'].numpy())
    for role in ('fusion', 'vision', 'emg'):
        expected = np.array([float((__import__('torch').from_numpy(x[role+'_prediction'][i])
            - a['joint_angles'][i, :, 0]).abs().mean()) for i in range(3)])
        np.testing.assert_array_equal(out['errors']['mae_'+role+'_rad'], expected)
    assert len(out['errors']['mae_fusion_rad']) == 3  # False masks/crops are retained.
    assert all(v.dtype == np.float32 for k, v in x.items() if k != 'label_valid_mask')
    assert not out['full_inference_permit']


def test_native_single_time_mask_keeps_original_two_dimensional_shape():
    m = api(); a, b = packets()
    assert a['label_valid_mask'].shape == (3, 1)
    out = m.native_forward_arrays(models(), a, b)
    assert out['arrays']['label_valid_mask'].shape == (3, 1)
    assert out['arrays']['label_valid_mask'].tolist() == [[False], [True], [False]]


@pytest.mark.parametrize('damage', ['channels', 'window', 'target', 'mask', 'mask_shape', 'nonfinite', 'dtype'])
def test_invalid_paired_batch_refused_before_any_model_call(damage):
    import torch
    m = api(); a, b = packets()
    if damage == 'channels': b['emg'] = b['emg'][:, :7]
    elif damage == 'window': b['emg'] = b['emg'][:, :, :-1]
    elif damage == 'target': b['joint_angles'][0, 0, 0] += 1
    elif damage == 'mask': b['label_valid_mask'][0] = True
    elif damage == 'mask_shape':
        a['label_valid_mask'] = a['label_valid_mask'].reshape(3, 1, 1)
        b['label_valid_mask'] = b['label_valid_mask'].reshape(3, 1, 1)
    elif damage == 'nonfinite': a['emg'][0, 0, 0] = float('nan')
    elif damage == 'dtype': a['emg'] = a['emg'].double()
    class Forbidden(torch.nn.Module):
        def forward(self, *args): raise AssertionError('bad batch reached forward')
    with pytest.raises(ValueError):
        m.native_forward_arrays({k: Forbidden().eval() for k in ('fusion', 'vision', 'emg')}, a, b)


def test_missing_branch_and_training_mode_are_not_repaired():
    m = api(); a, b = packets(); p = models(); p['vision'] = None
    with pytest.raises(ValueError): m.native_forward_arrays(p, a, b)
    p = models(); p['emg'].train()
    with pytest.raises(ValueError): m.native_forward_arrays(p, a, b)


def test_cpu_autocast_cannot_change_the_source_fp32_protocol():
    import torch
    m = api(); a, b = packets()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='autocast'):
            m.native_forward_arrays(models(), a, b)


def test_collate_equals_the_function_in_the_whole_original_evaluation_module(monkeypatch):
    import importlib.util
    import os
    from pathlib import Path
    import torch
    from egoemg_repro.classic_emg import _activate, _approved_upstream
    m = api()
    root = _approved_upstream(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).resolve().parents[1]/'upstream'))
    _activate(root)
    path = root/'scripts/eval/evaluate_occlusion_fusion.py'
    spec = importlib.util.spec_from_file_location('independent_original_dense_collate', path)
    original = importlib.util.module_from_spec(spec)
    def forbidden(*a, **kw): raise AssertionError('collate constructed a learned model')
    monkeypatch.setattr(torch.nn.Module, '__init__', forbidden)
    spec.loader.exec_module(original)
    a, _ = packets()
    samples = [{n: x[i].numpy() for n, x in a.items()} for i in range(3)]
    for i, s in enumerate(samples):
        s.update(_ep_idx=2, _center_idx=3895+i, target_hand='left',
            emg_channel_mask=np.ones(16, bool), vision_frame_indices=np.array([11], np.int64))
    expected = original._collate_fusion(samples)
    actual = m.source_collate_fusion(samples, upstream=root)
    assert set(actual) == set(expected) and 'emg_channel_mask' not in actual
    for n, x in expected.items():
        if isinstance(x, torch.Tensor):
            assert actual[n].dtype == x.dtype and torch.equal(actual[n], x)
        else: assert actual[n] == x


def test_full_forward_requires_registered_real_trained_admission():
    from types import SimpleNamespace
    m = api(); a, b = packets()
    with pytest.raises(ValueError):
        m.predict_dense_batch(SimpleNamespace(models=models(), check=lambda: None), a, b)


def key(i, group=0, hand='left'):
    return dict(sequence=i, split=('user', 'gesture', 'both')[group], hand=hand,
        ep_idx=0, center_idx=3895+i, window_start=i, window_end=i+7790,
        native_index=i, group_index=group)


def test_source_group_tails_are_retained_and_never_mixed():
    m = api(); keys = [key(i) for i in range(65)] + [key(65, hand='right'), key(66, 1)]
    batches = m.native_batch_ranges(keys)
    assert batches == [(0, 64), (64, 65), (65, 66), (66, 67)]


@pytest.mark.parametrize('damage', ['order', 'duplicate', 'reopened_group', 'split'])
def test_non_native_order_or_identity_rejected(damage):
    m = api(); keys = [key(i) for i in range(3)]
    if damage == 'order': keys.reverse()
    elif damage == 'duplicate': keys[1] = copy.deepcopy(keys[0])
    elif damage == 'reopened_group': keys[1] = key(1, hand='right')
    elif damage == 'split': keys[1]['split'] = 'gesture'
    with pytest.raises(ValueError): m.native_batch_ranges(keys)
