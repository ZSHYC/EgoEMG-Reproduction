"""Input-control boundaries; these tensor fixtures are not trained producers."""
import copy
import importlib
import pickle
import random

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="native tensor/runtime checks require the optional Torch tier")


def api():
    try:
        return importlib.import_module('egoemg_repro.modality_controls')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.modality_controls':
            pytest.fail('fixed-checkpoint input intervention implementation missing')
        raise


def packet(length=7790):
    channels = 16 if length == 7790 else 8
    return dict(
        emg=torch.arange(2*channels*length, dtype=torch.float32).reshape(2, channels, length),
        vision_img=torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5),
        vision_valid_mask=torch.tensor([[False], [True]]),
        joint_angles=torch.arange(44, dtype=torch.float32).reshape(2, 22, 1),
        label_valid_mask=torch.tensor([[True], [False]]),
        vision_frame_indices=torch.tensor([[17], [31]], dtype=torch.int64),
        target_hand=['left', 'right'],
    )


def apply(batch, emg='aligned', vision='native', mean=None, std=None):
    return api().apply_input_condition(batch, emg=emg, vision=vision,
        vision_mean=[127.5, 63.75, 255.] if mean is None else mean,
        vision_std=[63.75, 127.5, 51.] if std is None else std)


def test_aligned_condition_is_exact_independent_copy_without_rng_or_mask_changes():
    batch = packet()
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    result = apply(batch)
    out = result['batch']
    assert out is not batch and set(out) == set(batch)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(out[key], value)
            assert out[key].dtype == value.dtype and out[key].data_ptr() != value.data_ptr()
        else:
            assert out[key] == value and out[key] is not value
    out['joint_angles'][0, 0, 0] = -1000
    out['target_hand'][0] = 'changed'
    assert batch['joint_angles'][0, 0, 0] == 0 and batch['target_hand'][0] == 'left'
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    assert result['condition']['emg'] == 'aligned'
    assert result['condition']['vision'] == 'native'
    assert result['condition']['emg_roll_samples'] == 0
    assert result['full_inference_permit'] is False


@pytest.mark.parametrize('length', [7790, 12000])
def test_normalized_zero_clears_all_channels_like_native_force_zero_emg(length):
    batch = packet(length)
    # Native sparse16 padding is normalized too. This control clears it rather
    # than pretending that only the eight physical electrodes were replaced.
    batch['emg'][:, -1, :] = -.125
    before = batch['emg'].clone()
    out = apply(batch, emg='normalized_zero')['batch']
    assert torch.count_nonzero(out['emg']) == 0
    assert torch.equal(batch['emg'], before)
    for key in ('vision_img', 'vision_valid_mask', 'joint_angles', 'label_valid_mask'):
        assert torch.equal(out[key], batch[key])


@pytest.mark.parametrize('length', [7790, 12000])
def test_half_window_roll_uses_time_only_and_never_crosses_sample_or_channel(length):
    batch = packet(length)
    result = apply(batch, emg='half_window_roll')
    out = result['batch']; mid = length//2
    # Explicit index oracle, independently of torch.roll or production helper.
    for b, c in ((0, 0), (0, 3), (1, 0), (1, batch['emg'].shape[1]-1)):
        assert out['emg'][b, c, 0] == batch['emg'][b, c, mid]
        assert out['emg'][b, c, mid-1] == batch['emg'][b, c, length-1]
        assert out['emg'][b, c, mid] == batch['emg'][b, c, 0]
        assert out['emg'][b, c, length-1] == batch['emg'][b, c, mid-1]
    torch.testing.assert_close(out['emg'].sort(dim=-1).values,
        batch['emg'].sort(dim=-1).values, rtol=0, atol=0)
    assert result['condition']['emg_roll_samples'] == mid
    assert torch.equal(out['joint_angles'], batch['joint_angles'])


@pytest.mark.parametrize('emg', ['aligned', 'normalized_zero', 'half_window_roll'])
def test_missing_vision_is_native_normalized_black_and_keeps_label_validity(emg):
    batch = packet(); before = copy.deepcopy(batch)
    out = apply(batch, emg=emg, vision='missing')['batch']
    expected = torch.tensor([-2., -.5, -5.]).reshape(1, 3, 1, 1).expand(2, 3, 4, 5)
    assert torch.equal(out['vision_img'], expected)
    assert not out['vision_valid_mask'].any()
    assert torch.equal(out['label_valid_mask'], torch.tensor([[True], [False]]))
    assert torch.equal(out['joint_angles'], before['joint_angles'])
    assert torch.equal(batch['vision_img'], before['vision_img'])
    assert torch.equal(batch['vision_valid_mask'], before['vision_valid_mask'])


@pytest.mark.parametrize('damage', [
    'unknown_emg', 'unknown_vision', 'bad_length', 'wrong_channels', 'wrong_dtype',
    'nonfinite_emg', 'nonfinite_image', 'wrong_target_shape', 'wrong_label_dtype',
    'wrong_vision_mask_shape', 'bad_std', 'nonfinite_mean', 'cached_features',
    'requires_gradient',
])
def test_invalid_or_ambiguous_inputs_are_rejected_without_repair(damage):
    batch = packet(); options = {}
    if damage == 'unknown_emg': options['emg'] = 'random_guess'
    elif damage == 'unknown_vision': options['vision'] = 'occlusion_score_one'
    elif damage == 'bad_length': batch['emg'] = batch['emg'][..., :-1]
    elif damage == 'wrong_channels': batch['emg'] = batch['emg'][:, :8]
    elif damage == 'wrong_dtype': batch['emg'] = batch['emg'].to(torch.bfloat16)
    elif damage == 'nonfinite_emg': batch['emg'][0, 0, 0] = float('nan')
    elif damage == 'nonfinite_image': batch['vision_img'][0, 0, 0, 0] = float('inf')
    elif damage == 'wrong_target_shape': batch['joint_angles'] = batch['joint_angles'].squeeze(-1)
    elif damage == 'wrong_label_dtype': batch['label_valid_mask'] = batch['label_valid_mask'].float()
    elif damage == 'wrong_vision_mask_shape': batch['vision_valid_mask'] = torch.ones(2, 2, dtype=torch.bool)
    elif damage == 'bad_std': options['std'] = [63.75, 0., 51.]
    elif damage == 'nonfinite_mean': options['mean'] = [127.5, float('nan'), 255.]
    elif damage == 'cached_features': batch['vision_features'] = torch.ones(2, 384)
    elif damage == 'requires_gradient': batch['emg'].requires_grad_()
    with pytest.raises(ValueError):
        apply(batch, **options)
