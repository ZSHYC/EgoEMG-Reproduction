"""Native output boundary tests plus explicit real-producer engineering cases."""
import copy
import importlib
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
import torch

CONDITIONS = list(itertools.product(('aligned', 'normalized_zero', 'half_window_roll'), ('native', 'missing')))
KW = dict(vision_mean=[127.5, 63.75, 255.], vision_std=[63.75, 127.5, 51.])


def api():
    try:
        return importlib.import_module('egoemg_repro.modality_control_inference')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.modality_control_inference':
            pytest.fail('registered native control inference adapter missing')
        raise


def packet():
    return dict(emg=torch.arange(249280, dtype=torch.float32).reshape(2, 16, 7790),
        vision_img=torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5),
        vision_valid_mask=torch.tensor([[False], [True]]),
        joint_angles=torch.arange(44, dtype=torch.float32).reshape(2, 22, 1),
        label_valid_mask=torch.tensor([[True], [False]]))


def engineering_models(damage=None):
    class Fusion(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.scale = torch.nn.Parameter(torch.ones(()))
            self.register_buffer('seen', torch.zeros(()))

        def forward(self, b):
            assert not torch.is_grad_enabled()
            value = (b['emg'][:, 0, 0] + 5*b['vision_img'][:, 0, 0, 0]*b['vision_valid_mask'][:, 0])*self.scale
            prediction = value[:, None, None].expand(-1, 22, 1)
            if damage == 'nan_output': prediction = prediction*float('nan')
            if damage == 'half_output': prediction = prediction.half()
            if damage == 'wrong_output_shape': prediction = prediction.squeeze(-1)
            if damage == 'state_mutation': self.seen.add_(1)
            if damage == 'input_mutation': b['emg'].add_(1)
            if damage == 'rng_mutation': torch.rand(())
            target = b['joint_angles']+1 if damage == 'wrong_target' else b['joint_angles']
            mask = ~b['label_valid_mask'] if damage == 'wrong_mask' else b['label_valid_mask']
            return prediction, target, mask

    class Vision(torch.nn.Module):
        def forward(self, b):
            assert not torch.is_grad_enabled()
            return (b['vision_img'][:, 0, 0, 0]+2)[:, None, None].expand(-1, 22, 1)

    return dict(fusion=Fusion().eval(), vision=Vision().eval(), emg=torch.nn.Identity().eval())


@pytest.mark.parametrize('emg,vision', CONDITIONS)
def test_six_conditions_use_same_native_center_targets_masks_and_raw_errors(emg, vision):
    batch = packet(); result = api().native_control_arrays(engineering_models(), batch, emg=emg, vision=vision, **KW)
    x = result['arrays']
    # Literal independent time-index/image oracle; not the helper under test.
    expected_fusion = {
        ('aligned', 'native'): [0., 124940.], ('aligned', 'missing'): [0., 124640.],
        ('normalized_zero', 'native'): [0., 300.], ('normalized_zero', 'missing'): [0., 0.],
        ('half_window_roll', 'native'): [3895., 128835.], ('half_window_roll', 'missing'): [3895., 128535.],
    }[(emg, vision)]
    np.testing.assert_array_equal(x['fusion_prediction'], np.array(expected_fusion, np.float32)[:, None].repeat(22, 1))
    expected_vision = [2., 62.] if vision == 'native' else [0., 0.]
    np.testing.assert_array_equal(x['vision_prediction'], np.array(expected_vision, np.float32)[:, None].repeat(22, 1))
    np.testing.assert_array_equal(x['targets'], batch['joint_angles'].numpy().squeeze(-1))
    np.testing.assert_array_equal(x['label_valid_mask'], [[True], [False]])
    np.testing.assert_array_equal(x['natural_crop_valid_mask'], [[False], [True]])
    np.testing.assert_array_equal(x['condition_vision_valid_mask'], [[False], [True]] if vision == 'native' else [[False], [False]])
    for role in ('vision', 'fusion'):
        expected = [float((torch.from_numpy(x[role+'_prediction'][i])-batch['joint_angles'][i, :, 0]).abs().mean())for i in range(2)]
        np.testing.assert_array_equal(x['mae_'+role+'_rad'], np.array(expected, np.float64))
    assert result['all_model_states_before_after_exact'] and result['torch_rng_unchanged']
    assert result['original_batch_unchanged'] and not result['full_inference_permit']


@pytest.mark.parametrize('damage', ['wrong_target', 'wrong_mask', 'half_output', 'nan_output',
    'wrong_output_shape', 'state_mutation', 'input_mutation', 'rng_mutation'])
def test_changed_native_outputs_inputs_rng_or_buffers_fail_instead_of_being_repaired(damage):
    batch = packet(); before = copy.deepcopy(batch)
    with pytest.raises(ValueError):
        api().native_control_arrays(engineering_models(damage), batch, emg='aligned', vision='native', **KW)
    for key, value in before.items():
        assert torch.equal(batch[key], value)


@pytest.mark.parametrize('damage', ['train_mode', 'half_model', 'autocast', 'missing_role'])
def test_non_native_evaluation_precision_or_roles_are_rejected(damage):
    models = engineering_models()
    if damage == 'train_mode': models['fusion'].train()
    elif damage == 'half_model': models['fusion'].half()
    elif damage == 'missing_role': del models['emg']
    with pytest.raises(ValueError):
        if damage == 'autocast':
            with torch.autocast('cpu', dtype=torch.bfloat16):
                api().native_control_arrays(models, packet(), emg='aligned', vision='native', **KW)
        else:
            api().native_control_arrays(models, packet(), emg='aligned', vision='native', **KW)


def test_unregistered_namespace_cannot_grant_trained_control_prediction():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='registered'):
        api().predict_control_batch(SimpleNamespace(models=engineering_models(), check=lambda:None),
            packet(), emg='aligned', vision='native', **KW)


@pytest.fixture(scope='module', params=['vits_s', 'rn18_s'])
def real_producer(request):
    api()
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    fit = os.environ.get('EGOEMG_CONTROL_'+('VITS' if request.param=='vits_s' else 'RN18')+'_FIT')
    if not upstream or not fit or not Path(fit).is_dir():
        if os.environ.get('EGOEMG_REQUIRE_CONTROL_REAL') == '1':
            pytest.fail('both actual completed control producer families are required')
        pytest.skip('actual complete frozen producer sources unavailable here')
    if request.param == 'vits_s':
        from egoemg_repro.occlusion_dense_models import load_dense_trained_models
        from egoemg_repro.fusion_literal import compose_fusion as compose
        loader = load_dense_trained_models
    else:
        from egoemg_repro.rn18_occlusion_models import load_rn18_dense_trained_models
        from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion as compose
        loader = load_rn18_dense_trained_models
    from egoemg_repro.occlusion_dense_inputs import DensePredictionDataset
    original = torch.optim.Optimizer.__init__
    def forbidden(*args, **kwargs): raise AssertionError('control inference constructed optimizer')
    torch.optim.Optimizer.__init__ = forbidden
    try:
        trained = loader(fit, upstream=upstream, recipe=request.param)
        ds = DensePredictionDataset(trained.data_locations['data'], trained.data_locations['crops'],
            upstream=upstream, config=compose(upstream), recipe=request.param)
        yield trained, ds, upstream
    finally:
        torch.optim.Optimizer.__init__ = original


def test_real_registered_bilateral_six_conditions_and_native_aligned_agreement(real_producer):
    from egoemg_repro.occlusion_dense_inference import source_collate_fusion
    from egoemg_repro.vision_literal_runtime import vision_state_sha256
    trained, ds, upstream = real_producer
    before = {k:vision_state_sha256(m.state_dict())for k,m in trained.models.items()}
    right = next(i for i,h in enumerate(ds.index['points']['hand_index'])if int(h)==1)
    for index in (0, right):
        item = ds[index]; batch = source_collate_fusion([item['fusion']], upstream=upstream)
        native_view = ds.fusion_views[int(ds.index['points']['hand_index'][index])]
        norm = dict(vision_mean=native_view.vision_mean.tolist(), vision_std=native_view.vision_std.tolist())
        with torch.no_grad():
            reference = trained.models['fusion'].forward(batch)[0].squeeze(-1).numpy().copy()
        for emg, vision in CONDITIONS:
            out = api().predict_control_batch(trained, batch, emg=emg, vision=vision, **norm)
            assert not out['full_inference_permit'] and out['trained_model_identity_sha256']==trained.identity['sha256']
            assert all(np.isfinite(x).all()for x in out['arrays'].values())
            if (emg,vision)==('aligned','native'):
                np.testing.assert_array_equal(out['arrays']['fusion_prediction'], reference)
        ds.check()
    assert before=={k:vision_state_sha256(m.state_dict())for k,m in trained.models.items()}
    assert not torch.cuda.is_initialized()


def test_real_shallow_copy_cannot_bypass_control_registration(real_producer):
    trained, ds, upstream = real_producer
    with pytest.raises(ValueError, match='registered'):
        api().predict_control_batch(copy.copy(trained), {}, emg='aligned', vision='native', **KW)
