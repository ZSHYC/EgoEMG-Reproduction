import json
import os
from pathlib import Path
import subprocess
import sys

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


@pytest.mark.parametrize('backbone,batch', [('vit_small', 360), ('vit_base', 160), ('vit_large', 50)])
def test_dinov2_recipes_preserve_legacy_queue_and_disclose_derived_protocol(backbone, batch):
    path = Path('configs', backbone + '.json')
    assert path.is_file(), 'DINOv2 standalone recipe is missing'
    config = json.loads(path.read_text())
    assert config['upstream_experiment'] == 'fusion/vision_vit_small'
    assert config['vision_backbone_type'] == backbone
    assert config['vision_pretrained'] is True
    assert config['batch_size'] == config['official_batch_size'] == batch
    assert config['epochs'] == 150
    assert config['window_length'] == config['eval_stride'] == 12000
    assert config['train_stride'] == 1200 and config['val_fraction'] == .12
    assert config['optimizer_class'] == 'AdamW'
    assert (config['lr'], config['weight_decay'], config['eta_min']) == (.0001, .01, .00001)
    assert config['augmentation'] == 'none'
    assert '7790' in config['recipe_note'] and 'repeat2' in config['recipe_note']
    if backbone != 'vit_small':
        assert 'unpublished' in config['recipe_note'] and 'derived' in config['recipe_note']


def test_official_dinov2_small_constructs_at_256_without_pretrained_transport(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('timm')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for official DINOv2 integration')
    torch.set_num_threads(2)
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    config = dict(json.loads(Path('configs/resnet18.json').read_text()),
                  upstream_experiment='fusion/vision_vit_small', vision_backbone_type='vit_small',
                  vision_pretrained=False)
    model = build_visual_model(upstream, config)
    from egoemg.models.modules.vit_vision import VisionViTPose
    assert isinstance(model, VisionViTPose)
    assert model.backbone.patch_embed.img_size == (256, 256)
    assert model.head[0].in_features == 384
    assert sum(p.numel() for p in model.parameters()) == 21863318
    assert model.reproduction_initialization['vision_backbone'] == 'random'
    assert 'pretrained_weights' not in model.reproduction_initialization
    with pytest.raises(ValueError, match='pretrained.*source|source.*pretrained'):
        pretrained_weight_identity(model)


@pytest.mark.parametrize('backbone', ['vit_huge', 'vit_unknown', None, 7, ['vit_small'], 'resnet18'])
def test_invalid_vit_override_is_rejected_before_composition(tmp_path, monkeypatch, backbone):
    hydra = pytest.importorskip('hydra')
    def forbidden(*args, **kwargs):
        pytest.fail('invalid ViT override reached Hydra/network construction')
    monkeypatch.setattr(hydra, 'compose', forbidden)
    config = dict(json.loads(Path('configs/resnet18.json').read_text()),
                  upstream_experiment='fusion/vision_vit_small', vision_backbone_type=backbone)
    with pytest.raises(ValueError, match='vision_backbone_type'):
        build_visual_model(tmp_path, config)


def test_dinov2_pretrained_identity_observes_real_hf_route_and_loaded_tensors(monkeypatch):
    torch = pytest.importorskip('torch')
    timm = pytest.importorskip('timm')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for real timm loading with external transport fixture')
    import timm.models._builder as builder
    torch.set_num_threads(2)
    monkeypatch.delenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', raising=False)
    # Only the external transport is replaced. timm's source resolver, checkpoint
    # filtering/strict load and the official pose/backbone remain real.
    seed = timm.create_model('vit_small_patch14_dinov2', pretrained=False, num_classes=0, img_size=256)
    state = seed.state_dict()
    with torch.no_grad():
        state['patch_embed.proj.weight'].fill_(.25)
    def local_hf_transport(model_id, *args, **kwargs):
        assert model_id == 'timm/vit_small_patch14_dinov2.lvd142m'
        return dict(state)
    def no_url_fallback(*args, **kwargs):
        pytest.fail('HF route unexpectedly used the Meta URL')
    monkeypatch.setattr(builder, 'load_state_dict_from_hf', local_hf_transport)
    monkeypatch.setattr(builder, 'load_state_dict_from_url', no_url_fallback)
    config = dict(json.loads(Path('configs/vit_small.json').read_text()))
    model = build_visual_model(upstream, config)
    assert torch.all(model.backbone.patch_embed.proj.weight == .25)
    identity = model.reproduction_initialization['pretrained_weights']
    assert model.reproduction_initialization['vision_backbone'] == 'DINOv2'
    assert identity['family'] == 'DINOv2'
    assert identity['timm_model'] == 'vit_small_patch14_dinov2'
    assert identity['timm_version'] == timm.__version__
    assert identity['source_type'] == 'hf-hub'
    assert identity['source_location'] == 'timm/vit_small_patch14_dinov2.lvd142m'
    assert identity['resolved_pretrained_cfg'] == model.backbone.pretrained_cfg
    assert len(identity['backbone_tensor_sha256']) == 64
    assert identity['digest_format'] == 'length-prefixed canonical JSON header + raw tensor bytes, v1'
    with torch.no_grad():
        model.backbone.patch_embed.proj.weight.add_(.125)
    assert pretrained_weight_identity(model)['backbone_tensor_sha256'] != identity['backbone_tensor_sha256']
    assert builder.load_state_dict_from_hf is local_hf_transport
    assert builder.load_state_dict_from_url is no_url_fallback


def test_dinov2_failed_hf_transport_does_not_fall_back_or_randomize(monkeypatch):
    pytest.importorskip('timm')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for timm transport failure integration')
    import timm.models._builder as builder
    monkeypatch.delenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', raising=False)
    def failed_hf(*args, **kwargs):
        raise OSError('fixture HF transport unavailable')
    def forbidden_url(*args, **kwargs):
        pytest.fail('timm does not automatically fall back from HF to URL')
    monkeypatch.setattr(builder, 'load_state_dict_from_hf', failed_hf)
    monkeypatch.setattr(builder, 'load_state_dict_from_url', forbidden_url)
    with pytest.raises(Exception, match='fixture HF transport unavailable'):
        build_visual_model(upstream, json.loads(Path('configs/vit_small.json').read_text()))
    assert builder.load_state_dict_from_hf is failed_hf
    assert builder.load_state_dict_from_url is forbidden_url


def test_dinov2_requested_pretraining_rejects_offline_before_instantiation(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('hydra')
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    with pytest.raises(ValueError, match='DINOv2.*EGOEMG_NO_PRETRAINED_DOWNLOAD'):
        build_visual_model(tmp_path, json.loads(Path('configs/vit_small.json').read_text()))


def test_dinov2_small_real_256px_backward_optimizer_and_strict_offline_resume(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('timm')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for real DINOv2 CPU optimization and resume')
    import timm.models._builder as builder
    from egoemg_repro.train import make_optimizer
    def forbidden_transport(*args, **kwargs):
        pytest.fail('self-contained or random initialization attempted a weight download')
    monkeypatch.setattr(builder, 'load_state_dict_from_hf', forbidden_transport)
    monkeypatch.setattr(builder, 'load_state_dict_from_url', forbidden_transport)
    monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD', '1')
    torch.set_num_threads(2)
    config = dict(json.loads(Path('configs/vit_small.json').read_text()), vision_pretrained=False)
    model = build_visual_model(upstream, config)
    optimizer, scheduler = make_optimizer(model, config)
    batch = dict(vision_img=torch.randn(1, 3, 256, 256), joint_angles=torch.ones(1, 22, 1),
                 label_valid_mask=torch.ones(1, 1, dtype=torch.bool),
                 angle_valid_mask=torch.ones(1, 22, 1, dtype=torch.bool))
    batch['angle_valid_mask'][0, 20] = False
    prediction, target, mask = align_visual_batch(model, batch)
    assert prediction.shape == target.shape == mask.shape == (1, 22, 1)
    assert not mask[0, 20, 0]
    (prediction[mask] - target[mask]).abs().mean().backward()
    assert model.backbone.patch_embed.proj.weight.grad is not None
    assert torch.isfinite(model.backbone.patch_embed.proj.weight.grad).all()
    assert model.head[-1].weight.grad is not None
    before = model.backbone.patch_embed.proj.weight.detach().clone()
    optimizer.step()
    scheduler.step()
    assert not torch.equal(before, model.backbone.patch_embed.proj.weight)
    model.eval()
    with torch.no_grad():
        expected = align_visual_batch(model, batch)[0]
    resumed = build_visual_model(upstream, dict(config, vision_pretrained=True), resume_state=model.state_dict()).eval()
    with torch.no_grad():
        actual = align_visual_batch(resumed, batch)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    restored_optimizer, restored_scheduler = make_optimizer(resumed, config)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())
    assert restored_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert resumed.reproduction_initialization['vision_backbone'] == 'self_contained_checkpoint'
    assert 'pretrained_weights' not in resumed.reproduction_initialization
    del resumed, restored_optimizer
    broken = dict(model.state_dict())
    broken['head.0.weight'] = broken['head.0.weight'][:, :1]
    with pytest.raises(RuntimeError, match='size mismatch'):
        build_visual_model(upstream, config, resume_state=broken)


@pytest.mark.parametrize('backbone,parameters,dim,depth', [
    ('vit_base', 86182166, 768, 12), ('vit_large', 303833622, 1024, 24)])
def test_dinov2_larger_real_cpu_constructor_and_parameter_schema(backbone, parameters, dim, depth):
    pytest.importorskip('timm')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for separate-process B/L real constructors')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    required = 2*parameters*4 + 512*1024**2
    if available < required:
        pytest.skip(f'{backbone} real CPU construction not verified: MemAvailable={available} < reserve={required}')
    script = '''
import json,sys,torch
from pathlib import Path
from egoemg_repro.visual import build_visual_model
torch.set_num_threads(2)
config=json.loads(Path('configs',sys.argv[2]+'.json').read_text())
config['vision_pretrained']=False
model=build_visual_model(sys.argv[1],config)
print(json.dumps({'class':type(model).__name__,'parameters':sum(p.numel() for p in model.parameters()),
 'dim':model.head[0].in_features,'depth':len(model.backbone.blocks),
 'image_size':list(model.backbone.patch_embed.img_size),'output_dim':model.head[-1].out_features,
 'qkv':list(model.state_dict()['backbone.blocks.0.attn.qkv.weight'].shape),
 'initialization':model.reproduction_initialization['vision_backbone']}))
'''
    result = subprocess.run([sys.executable, '-c', script, upstream, backbone], text=True,
                            capture_output=True, check=True, timeout=180,
                            env=dict(os.environ, EGOEMG_NO_PRETRAINED_DOWNLOAD='1'))
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    assert actual == {'class': 'VisionViTPose', 'parameters': parameters, 'dim': dim, 'depth': depth,
                      'image_size': [256, 256], 'output_dim': 22, 'qkv': [3*dim, dim], 'initialization': 'random'}


def test_dinov2_real_timm_resolver_can_record_selected_url_transport(tmp_path):
    pytest.importorskip('timm')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for real timm URL route with external transport fixture')
    # A subprocess reads timm's real TIMM_USE_OLD_CACHE setting at import time.
    # Only the weight transport is replaced; source resolution is unchanged.
    script = '''
import json,sys,torch,timm
from pathlib import Path
import timm.models._builder as builder
from timm.models._hub import get_cache_dir
from egoemg_repro.visual import build_visual_model
torch.set_num_threads(2)
cfg=timm.get_pretrained_cfg('vit_small_patch14_dinov2').to_dict()
cache=Path(get_cache_dir())/cfg['url'].rsplit('/',1)[-1]
cache.write_bytes(b'transport fixture presence marker')
seed=timm.create_model('vit_small_patch14_dinov2',pretrained=False,num_classes=0,img_size=256)
state=seed.state_dict()
state['patch_embed.proj.weight'].fill_(.5)
def transport(url,*args,**kwargs):
 assert url==cfg['url']
 return dict(state)
def forbidden_hf(*args,**kwargs):
 raise AssertionError('real resolver selected URL; HF transport must not run')
builder.load_state_dict_from_url=transport
builder.load_state_dict_from_hf=forbidden_hf
config=json.loads(Path('configs/vit_small.json').read_text())
model=build_visual_model(sys.argv[1],config)
assert torch.all(model.backbone.patch_embed.proj.weight==.5)
assert builder.load_state_dict_from_url is transport
print(json.dumps(model.reproduction_initialization['pretrained_weights']))
'''
    env = dict(os.environ, TIMM_USE_OLD_CACHE='1', TORCH_HOME=str(tmp_path/'torch-cache'))
    env.pop('EGOEMG_NO_PRETRAINED_DOWNLOAD', None)
    result = subprocess.run([sys.executable, '-c', script, upstream], text=True, capture_output=True,
                            env=env, check=True, timeout=90)
    identity = json.loads(result.stdout.strip().splitlines()[-1])
    assert identity['source_type'] == 'url'
    assert identity['source_location'] == 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
    assert identity['resolved_pretrained_cfg']['hf_hub_id'] == 'timm/vit_small_patch14_dinov2.lvd142m'
