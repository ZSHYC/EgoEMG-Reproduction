import json
import os
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.visual import (VisualDataset, align_visual_batch, build_visual_model,
                               pretrained_weight_identity, sparse16)
from test_crops import add_video_indices, make_crops
from test_data import make_store, window


def test_sparse_layout_keeps_zero_slots_and_channel_order():
    emg = np.arange(24, dtype=np.float32).reshape(8, 3)
    mapped = sparse16(emg)
    assert mapped.shape == (16, 3)
    np.testing.assert_array_equal(mapped[[10, 12, 0, 1, 2, 4, 5, 6]], emg)
    np.testing.assert_array_equal(mapped[[3, 7, 8, 9, 11, 13, 14, 15]], 0)


def test_dataset_uses_center_labels_and_normalizes_already_flipped_jpeg(tmp_path):
    torch = pytest.importorskip('torch')
    data, crops = tmp_path / 'data', tmp_path / 'crops'
    data.mkdir()
    make_store(data)
    add_video_indices(data, np.arange(47) + 15)
    make_crops(crops)
    dataset = VisualDataset(data, [window()], crops)
    left, right = dataset[0], dataset[1]
    assert left['joint_angles'].shape == (22, 1)
    assert not left['angle_valid_mask'][20, 0]
    assert left['angle_valid_mask'][0, 0]
    assert left['vision_valid_mask'].tolist() == [True]
    assert right['vision_valid_mask'].tolist() == [False]
    assert left['vision_frame_indices'].tolist() == [17]
    assert left['vision_img'][0, 30, 30] > 2
    assert left['vision_img'][2, 30, 200] > 2
    np.testing.assert_allclose(right['vision_img'][:, 0, 0], [-.485/.229, -.456/.224, -.406/.225], atol=1e-6)
    assert left['center'] == right['center'] == 2


def test_fusion_shorter_emg_is_centered_inside_manifest_footprint(tmp_path):
    pytest.importorskip('torch')
    data, crops = tmp_path / 'data', tmp_path / 'crops'
    data.mkdir()
    make_store(data, lengths=(12000,))
    add_video_indices(data, np.full(12000, 17))
    make_crops(crops)
    sample = VisualDataset(data, [window(0, 12000)], crops, kind='fusion')[0]
    assert sample['emg'].shape == (8, 7790)
    assert sample['emg'][0, 0] == 2105
    assert sample['emg'][0, -1] == 9894
    assert sample['start'] == 0 and sample['end'] == 12000 and sample['center'] == 6000
    with pytest.raises(ValueError, match='footprint'):
        VisualDataset(data, [window(0, 4000)], crops, kind='fusion')


def test_official_vision_center_alignment_keeps_wrist_mask(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('hydra')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for fixed official-model integration')
    config = json.loads(Path('configs/resnet18.json').read_text())
    config['vision_pretrained'] = False
    model = build_visual_model(upstream, config)
    model.eval()
    batch = dict(vision_img=torch.zeros(2, 3, 32, 32), joint_angles=torch.ones(2, 22, 1),
                 label_valid_mask=torch.ones(2, 1, dtype=torch.bool), angle_valid_mask=torch.ones(2,22,1,dtype=torch.bool))
    batch['angle_valid_mask'][0,20] = False
    pred, target, valid = align_visual_batch(model, batch)
    assert pred.shape == target.shape == valid.shape == (2,22,1)
    assert not valid[0,20,0]
    pred[valid].sum().backward()
    assert model.head[-1].weight.grad is not None


def test_clean_fusion_loads_exact_branches_and_starts_at_vision(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('hydra')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for fixed official-model integration')
    from egoemg_repro.train import build_model
    torch.set_num_threads(2)
    vision_config = json.loads(Path('configs/resnet18.json').read_text())
    vision_config['vision_pretrained'] = False
    vision = build_visual_model(upstream, vision_config)
    emg = build_model(upstream, json.loads(Path('configs/emgformer_s.json').read_text()))
    vision_path, emg_path = tmp_path / 'vision.pt', tmp_path / 'emg.pt'
    torch.save({'model': vision.state_dict()}, vision_path)
    torch.save({'model': emg.state_dict()}, emg_path)
    config = json.loads(Path('configs/fusion_rn18_s.json').read_text())
    model = build_visual_model(upstream, config, vision_path, emg_path)
    vision.eval()
    model.eval()
    batch = dict(vision_img=torch.zeros(2,3,32,32), emg=torch.randn(2,8,config['emg_window_length']),
                 joint_angles=torch.zeros(2,22,1), label_valid_mask=torch.ones(2,1,dtype=torch.bool),
                 angle_valid_mask=torch.ones(2,22,1,dtype=torch.bool), vision_valid_mask=torch.ones(2,1,dtype=torch.bool))
    prediction, _, valid = align_visual_batch(model, batch)
    with torch.no_grad():
        expected = vision(batch)[0]
    torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-6)
    prediction[valid].sum().backward()
    assert model.head.net[-1].weight.grad is not None
    assert model.reproduction_initialization['residual_final'] == 'exact_zero_after_official_initialization'
    broken = emg.state_dict()
    broken.pop(next(iter(broken)))
    torch.save({'model': broken}, emg_path)
    with pytest.raises((ValueError, RuntimeError), match='missing|Missing'):
        build_visual_model(upstream, config, vision_path, emg_path)
    vision_path.unlink()
    emg_path.unlink()
    with torch.no_grad():
        model.head.net[-1].bias.fill_(0.3)
        model.head.net[-1].weight.fill_(0.01)
        expected_resumed = align_visual_batch(model, batch)[0]
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    resumed = build_visual_model(upstream, config, resume_state=model.state_dict()).eval()
    with torch.no_grad():
        actual_resumed = align_visual_batch(resumed, batch)[0]
    torch.testing.assert_close(actual_resumed, expected_resumed, atol=0, rtol=0)
    assert resumed.reproduction_initialization['restored_self_contained'] is True


def test_requested_imagenet_initialization_rejects_silent_offline_randomization(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('hydra')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for fixed official-model integration')
    config = json.loads(Path('configs/resnet18.json').read_text())
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    with pytest.raises(ValueError, match='ImageNet.*EGOEMG_NO_PRETRAINED_DOWNLOAD'):
        build_visual_model(upstream, config)


def test_vision_self_contained_resume_restores_weights_despite_offline_environment(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('hydra')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for fixed official-model integration')
    config = json.loads(Path('configs/resnet18.json').read_text())
    fresh_config = dict(config, vision_pretrained=False)
    model = build_visual_model(upstream, fresh_config).eval()
    with torch.no_grad():
        model.head[-1].bias.fill_(0.7)
    batch = dict(vision_img=torch.zeros(2,3,32,32))
    expected = model(batch).detach()
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    resumed = build_visual_model(upstream, config, resume_state=model.state_dict()).eval()
    torch.testing.assert_close(resumed(batch), expected, atol=0, rtol=0)
    assert resumed.reproduction_initialization['vision_backbone'] == 'self_contained_checkpoint'


def test_mocked_pretrained_identity_records_actual_backbone_url_and_version(monkeypatch):
    torch = pytest.importorskip('torch')
    torchvision = pytest.importorskip('torchvision')
    pytest.importorskip('hydra')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for mocked official pretrained construction')
    monkeypatch.delenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', raising=False)
    original_resnet18 = torchvision.models.resnet18
    fill = [0.]
    def offline_pretrained(*args, **kwargs):
        model = original_resnet18(weights=None)
        with torch.no_grad():
            model.conv1.weight.fill_(fill[0])
        return model
    monkeypatch.setattr(torchvision.models, 'resnet18', offline_pretrained)
    config = json.loads(Path('configs/resnet18.json').read_text())
    first = build_visual_model(upstream, config)
    identity = first.reproduction_initialization['pretrained_weights']
    assert identity['enum'] == 'torchvision.models.ResNet18_Weights.IMAGENET1K_V1'
    assert identity['source_url'] == torchvision.models.ResNet18_Weights.IMAGENET1K_V1.url
    assert identity['torchvision_version'] == torchvision.__version__
    assert len(identity['backbone_tensor_sha256']) == 64
    fill[0] = 1.
    second = build_visual_model(upstream, config)
    assert second.reproduction_initialization['pretrained_weights']['backbone_tensor_sha256'] != identity['backbone_tensor_sha256']
    from egoemg_repro.train import initialization_identity
    original_identity = {'policy': 'ImageNet1K_V1 backbone', 'official_builder': first.reproduction_initialization}
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    restored = build_visual_model(upstream, config, resume_state=first.state_dict())
    assert restored.reproduction_initialization['restored_self_contained']
    retained = initialization_identity(config, {}, loaded={'provenance': {'initialization': original_identity}})
    assert retained['official_builder']['pretrained_weights'] == identity


@pytest.mark.parametrize('backbone,parameters,enum,url', [
    ('resnet50', 24568406, 'ResNet50_Weights',
     'https://download.pytorch.org/models/resnet50-0676ba61.pth'),
    ('resnet152', 59204182, 'ResNet152_Weights',
     'https://download.pytorch.org/models/resnet152-394f9c45.pth'),
])
def test_larger_official_resnets_predict_backpropagate_and_strictly_resume(
        monkeypatch, backbone, parameters, enum, url):
    torch = pytest.importorskip('torch')
    pytest.importorskip('hydra')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for fixed official-model integration')
    torch.set_num_threads(2)
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    config = dict(json.loads(Path('configs/resnet18.json').read_text()),
                  vision_backbone_type=backbone, vision_pretrained=False)
    model = build_visual_model(upstream, config).eval()
    assert model.backbone_type == backbone
    assert model.reproduction_hydra_config['backbone_type'] == backbone
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters
    assert (model.left_context, model.right_context) == (0, 0)
    assert model.reproduction_initialization['vision_backbone'] == 'random'
    assert 'pretrained_weights' not in model.reproduction_initialization
    batch = dict(vision_img=torch.randn(2, 3, 32, 32),
                 joint_angles=torch.ones(2, 22, 1),
                 label_valid_mask=torch.ones(2, 1, dtype=torch.bool),
                 angle_valid_mask=torch.ones(2, 22, 1, dtype=torch.bool))
    batch['angle_valid_mask'][0, 20] = False
    prediction, target, valid = align_visual_batch(model, batch)
    assert prediction.shape == target.shape == valid.shape == (2, 22, 1)
    assert not valid[0, 20, 0]
    (prediction[valid] - target[valid]).abs().mean().backward()
    assert model.head[-1].weight.grad is not None
    assert model.backbone[0].weight.grad is not None
    assert torch.isfinite(model.backbone[0].weight.grad).all()
    identity = pretrained_weight_identity(model)
    assert identity['enum'] == 'torchvision.models.' + enum + '.IMAGENET1K_V1'
    assert identity['source_url'] == url
    assert len(identity['backbone_tensor_sha256']) == 64
    with torch.no_grad():
        model.backbone[0].weight.add_(.01)
        expected = align_visual_batch(model, batch)[0]
    assert pretrained_weight_identity(model)['backbone_tensor_sha256'] != identity['backbone_tensor_sha256']
    # A resumed run can retain its original pretrained request without fetching
    # weights again; architecture selection still comes from the recipe.
    resumed = build_visual_model(upstream, dict(config, vision_pretrained=True),
                                 resume_state=model.state_dict()).eval()
    with torch.no_grad():
        actual = align_visual_batch(resumed, batch)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert resumed.reproduction_initialization['restored_self_contained'] is True
    del resumed
    broken = dict(model.state_dict())
    broken['head.0.weight'] = broken['head.0.weight'][:, :1]
    with pytest.raises(RuntimeError, match='size mismatch'):
        build_visual_model(upstream, config, resume_state=broken)


@pytest.mark.parametrize('backbone', ['resnet34', 'vit_small', '', None, 50, ['resnet50']])
def test_backbone_override_rejects_invalid_types_before_hydra_composition(tmp_path, monkeypatch, backbone):
    pytest.importorskip('torch')
    hydra = pytest.importorskip('hydra')
    def forbidden_compose(*args, **kwargs):
        pytest.fail('invalid backbone override reached Hydra composition')
    monkeypatch.setattr(hydra, 'compose', forbidden_compose)
    config = dict(json.loads(Path('configs/resnet18.json').read_text()),
                  vision_backbone_type=backbone, vision_pretrained=False)
    with pytest.raises(ValueError, match='vision_backbone_type'):
        build_visual_model(tmp_path, config)


def test_backbone_override_rejects_fusion_before_clean_initializer_loading(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('hydra')
    config = dict(json.loads(Path('configs/fusion_rn18_s.json').read_text()),
                  vision_backbone_type='resnet50', vision_pretrained=False)
    with pytest.raises(ValueError, match='vision_backbone_type.*vision'):
        build_visual_model(tmp_path, config)


def test_backbone_override_rejects_other_vision_family_before_instantiation(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('hydra')
    config = dict(json.loads(Path('configs/resnet18.json').read_text()),
                  upstream_experiment='fusion/vision_vit_small',
                  vision_backbone_type='resnet50', vision_pretrained=False)
    with pytest.raises(ValueError, match='vision_backbone_type.*ResNet'):
        build_visual_model(tmp_path, config)


@pytest.mark.parametrize('name,batch_size', [('resnet50', 400), ('resnet152', 180)])
def test_larger_resnet_recipes_keep_shared_legacy_protocol_and_explicit_architecture(name, batch_size):
    config = json.loads(Path('configs', name + '.json').read_text())
    assert config['vision_backbone_type'] == name
    assert config['upstream_experiment'] == 'fusion/vision_resnet18'
    assert config['vision_pretrained'] is True
    assert config['batch_size'] == config['official_batch_size'] == batch_size
    assert {key: config[key] for key in (
        'seed', 'model_kind', 'window_length', 'train_stride', 'eval_stride', 'val_fraction',
        'epochs', 'optimizer_class', 'lr', 'weight_decay', 'eta_min', 'precision', 'gradient_clip',
        'fingertip_weight', 'augmentation', 'evaluation')} == {
        'seed': 42, 'model_kind': 'vision', 'window_length': 12000,
        'train_stride': 1200, 'eval_stride': 12000, 'val_fraction': .12,
        'epochs': 150, 'optimizer_class': 'AdamW', 'lr': .0001, 'weight_decay': .01,
        'eta_min': .00001, 'precision': 'bfloat16', 'gradient_clip': 1.,
        'fingertip_weight': .01, 'augmentation': 'none',
        'evaluation': 'fixed_center_all_labels_with_separate_crop_valid_population'}
