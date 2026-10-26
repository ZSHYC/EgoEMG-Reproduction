"""Pinned classic model contracts; no loader, mask adapter or training run."""
import ast
from dataclasses import FrozenInstanceError
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream')).resolve()
VARIANTS = ('vemg2pose', 'vemg2pose_with_aug', 'neuropose', 'neuropose_with_aug')


def runtime():
    return importlib.import_module('egoemg_repro.classic_emg')


def config(variant='vemg2pose'):
    pytest.importorskip('hydra')
    return runtime().compose_classic(UPSTREAM, variant)


def framework():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pytorch_lightning')
    pytest.importorskip('hydra')
    torch.set_num_threads(2)
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    return torch


@pytest.mark.parametrize('variant', VARIANTS)
def test_frozen_contract_retains_original_metadata_and_explicit_layout(variant):
    contract = runtime().classic_contract(variant)
    vemg = variant.startswith('vemg')
    assert contract.variant == variant
    expected_experiment = 'emg2pose/regression_' + variant.removesuffix('_with_aug') + '_egoemg'
    if variant.endswith('_with_aug'):
        expected_experiment += '_with_aug'
    assert contract.experiment == expected_experiment
    assert contract.input_channels == (8 if vemg else 16)
    assert contract.output_channels == 22 and contract.metadata_out_channels == 20
    assert (contract.left_context, contract.right_context) == ((1790, 0) if vemg else (0, 0))
    assert contract.parameters == (5963948 if vemg else 6354969)
    assert contract.declared_layout == 'emg2pose_interpolate16'
    assert contract.effective_layout == ('target_hand' if vemg else 'emg2pose_interpolate16')
    assert contract.channel_indices == (10, 12, 0, 1, 2, 4, 5, 6)
    assert contract.channel_interpolate is False
    assert contract.mask_contract == 'original official tuple with two-dimensional label mask; loader protocol not selected'
    assert ('8' in contract.layout_reason) if vemg else contract.layout_reason == 'official 16-slot layout retained'
    with pytest.raises(FrozenInstanceError):
        contract.input_channels = 99


@pytest.mark.parametrize('variant', ['unknown', 'emg2pose/regression_vemg2pose_egoemg', '../vemg2pose', 'resnet18'])
def test_unknown_variants_are_rejected_before_source_or_factory(variant):
    with pytest.raises(ValueError, match='variant'):
        runtime().compose_classic(Path('/does-not-exist'), variant)


@pytest.mark.parametrize('damage', ['wrong_pin', 'dirty_source', 'untracked_source'])
def test_unapproved_upstream_is_rejected_before_hydra(tmp_path, monkeypatch, damage):
    pytest.importorskip('hydra')
    rt = runtime()
    copied = tmp_path / 'upstream'
    subprocess.run(['git', 'clone', '--quiet', '--shared', str(UPSTREAM), str(copied)], check=True)
    if damage == 'wrong_pin':
        (copied / '.git/HEAD').write_text('0' * 40 + '\n')
    elif damage == 'dirty_source':
        with (copied / 'egoemg/models/modules/pose.py').open('a') as stream:
            stream.write('\n# changed fixture\n')
    else:
        (copied / 'egoemg/unapproved.py').write_text('# untracked fixture\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('unapproved source reached Hydra')
    monkeypatch.setattr('hydra.compose', forbidden)
    with pytest.raises(ValueError, match='pin|revision|clean'):
        rt.compose_classic(copied, 'vemg2pose')


@pytest.mark.parametrize('variant', VARIANTS)
def test_exact_recipe_keeps_raw_loss_keys_and_records_only_layout_decision(variant):
    cfg = config(variant)
    rt = runtime()
    contract = rt.classic_contract(variant)
    assert cfg.egoemg_emg_layout == contract.effective_layout
    assert cfg.module._target_.endswith('VEMG2PoseWithInitialState' if variant.startswith('vemg') else 'PoseModule')
    assert dict(cfg.loss_weights) == {'mae': 1, 'landmark/fingertip': .01, 'fingertip_distance': .01}
    assert cfg.trainer.max_epochs == 100 and cfg.trainer.precision == '16-mixed'
    assert cfg.trainer.gradient_clip_val == 0 and cfg.batch_size == (600 if variant.startswith('vemg') else 256)
    assert cfg.monitor_metric == 'val_loss' and cfg.monitor_mode == 'min'
    assert cfg.datamodule.window_length == 7790 and cfg.datamodule.stride == 780
    assert cfg.datamodule.val_test_stride == 7790
    assert cfg.checkpoint is None and cfg.pretrained_checkpoint is None and cfg.resume_ckpt is None
    for collection in (cfg.dataset, cfg.datamodule.dataset_conf):
        for role in ('train', 'val'):
            for entry in collection[role]:
                assert entry.emg_layout == contract.effective_layout
                assert entry.channel_interpolate is False
                assert entry.norm_stats_path == str(UPSTREAM / 'assets/per_dataset_norm_stats.json')
    assert cfg.dataset.test == []
    transforms = [str(t._target_).rsplit('.', 1)[-1] for t in cfg.transforms.train]
    expected = ['ExtractToTensor', 'RotationAugmentation'] if not variant.endswith('_with_aug') else [
        'ExtractField', 'RotationAugmentation', 'ToFloatTensor', 'RandomChannelMask',
        'RandomTimeMask', 'RandomFrequencyMask', 'RandomGaussianNoise']
    assert transforms == expected
    identity = rt.resolved_classic_identity(cfg, contract)
    assert identity['raw_loss_weights'] == dict(cfg.loss_weights)
    assert identity['effective_loss_weights'] == {'mae': 1.0, 'landmark/fingertip': .02}
    assert identity['contract']['declared_layout'] == 'emg2pose_interpolate16'
    assert identity['contract']['effective_layout'] == contract.effective_layout
    if variant.startswith('vemg'):
        assert identity['raw_config_sha256'] != identity['effective_config_sha256']
    else:
        # Normalization location is content-tokenized, so retaining the
        # official NeuroPose layout produces no semantic config difference.
        assert identity['raw_config_sha256'] == identity['effective_config_sha256']
    assert identity['official_source']['revision'] == 'f7cbd048b7092139852270491d3859cd472e0311'
    assert identity['runtime_files_sha256']['egoemg_repro/classic_emg.py'] == hashlib.sha256(Path(rt.__file__).read_bytes()).hexdigest()
    assert identity['official_source']['critical_files_sha256']['egoemg/lightning.py'] == hashlib.sha256((UPSTREAM / 'egoemg/lightning.py').read_bytes()).hexdigest()
    assert str(UPSTREAM) not in json.dumps(identity)


def test_effective_weights_match_execution_of_actual_official_reducer():
    cfg = config()
    rt = runtime()
    identity = rt.resolved_classic_identity(cfg, rt.classic_contract('vemg2pose'))
    tree = ast.parse((UPSTREAM / 'egoemg/lightning.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EmgPredictionModule')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_step')
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == '_LOSS_KEY_COMPAT' for t in n.targets))
    scope = dict(self=SimpleNamespace(loss_weights=dict(cfg.loss_weights)), stage='audit',
                 metrics={'audit_mae': 2., 'audit_landmark/fingertip': 3.})
    exec(compile(ast.Module(body=method.body[start:start + 3], type_ignores=[]), '<original-reducer>', 'exec'), scope)
    expected = identity['effective_loss_weights']['mae'] * 2 + identity['effective_loss_weights']['landmark/fingertip'] * 3
    assert scope['loss'] == pytest.approx(2.06)
    assert expected == pytest.approx(scope['loss'])


@pytest.mark.parametrize('damage', ['module', 'head', 'loss', 'initializer', 'resume', 'seed', 'layout', 'extra'])
def test_unknown_overrides_fail_before_real_factory(monkeypatch, damage):
    framework()
    from omegaconf import OmegaConf
    import egoemg.train as official
    rt = runtime()
    cfg = config()
    if damage == 'module':
        cfg.module.decoder.in_channels = 84
    elif damage == 'head':
        cfg.module.head._target_ = 'torch.nn.Linear'
    elif damage == 'loss':
        del cfg.loss_weights['fingertip_distance']
    elif damage == 'initializer':
        cfg.pretrained_checkpoint = '/private/unapproved.pt'
    elif damage == 'resume':
        cfg.resume_ckpt = '/private/unapproved.pt'
    elif damage == 'seed':
        cfg.seed = 7
    elif damage == 'layout':
        cfg.egoemg_emg_layout = 'emg2pose_interpolate16'
    else:
        OmegaConf.update(cfg, 'unapproved_override', True, force_add=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('unapproved config reached real factory')
    monkeypatch.setattr(official, 'make_lightning_module', forbidden)
    with pytest.raises(ValueError, match='config|recipe'):
        rt.build_classic_model(UPSTREAM, cfg)


@pytest.mark.parametrize(('variant', 'channels', 'wrong', 'length'), [('vemg2pose', 8, 16, 6000), ('neuropose', 16, 8, 7790)])
def test_real_official_forward_preserves_tuple_mask_and_wrong_layout_fails(variant, channels, wrong, length, monkeypatch):
    torch = framework()
    import egoemg.train as official
    cfg = config(variant)
    calls = []
    actual_factory = official.make_lightning_module
    def observed_factory(config):
        calls.append(torch.initial_seed())
        return actual_factory(config)
    monkeypatch.setattr(official, 'make_lightning_module', observed_factory)
    def offline(*args, **kwargs):
        raise AssertionError('classic construction or forward attempted network')
    monkeypatch.setattr('socket.socket.connect', offline)
    model = runtime().build_classic_model(UPSTREAM, cfg).eval()
    assert calls == [42]
    assert type(model).__module__ == 'egoemg.models.modules.pose'
    assert model.out_channels == 20 and sum(p.numel() for p in model.parameters()) == (5963948 if channels == 8 else 6354969)
    batch = dict(emg=torch.randn(1, channels, 7790), joint_angles=torch.randn(1, 22, 7790),
                 label_valid_mask=torch.ones(1, 7790, dtype=torch.bool),
                 angle_valid_mask=torch.zeros(1, 22, 7790, dtype=torch.bool))
    batch['label_valid_mask'][..., -1] = False
    with torch.no_grad():
        output = model(batch)
    assert isinstance(output, tuple) and len(output) == 3
    prediction, target, mask = output
    assert tuple(prediction.shape) == tuple(target.shape) == (1, 22, length)
    assert tuple(mask.shape) == (1, length) and mask.dtype == torch.bool
    assert torch.isfinite(prediction).all()
    assert torch.equal(target, batch['joint_angles'][..., model.left_context:])
    assert torch.equal(mask, batch['label_valid_mask'][..., model.left_context:])
    assert mask.any()  # Deliberately conflicting angle_valid_mask was not substituted.
    if variant == 'neuropose':
        assert not isinstance(model.featurizer.network[0].network[-1].kernel_size, __import__('omegaconf').ListConfig)
    batch['emg'] = torch.randn(1, wrong, 7790)
    with pytest.raises(RuntimeError, match='channels|Output size'):
        with torch.no_grad():
            model(batch)


def test_seed_setup_precedes_actual_factory_and_repeated_initialization_is_equal(monkeypatch):
    torch = framework()
    import egoemg.train as official
    import egoemg._train_utils as utils
    rt = runtime()
    cfg = config('vemg2pose')
    events = []
    setup, factory = utils.setup_runtime, official.make_lightning_module
    def seeded(config):
        events.append('seed')
        return setup(config)
    def constructed(config):
        events.append('factory')
        return factory(config)
    monkeypatch.setattr(utils, 'setup_runtime', seeded)
    monkeypatch.setattr(official, 'make_lightning_module', constructed)
    torch.manual_seed(19)
    first = rt.build_classic_model(UPSTREAM, cfg)
    expected = {key: value.clone() for key, value in first.state_dict().items()}
    del first
    torch.manual_seed(987)
    second = rt.build_classic_model(UPSTREAM, cfg)
    assert events == ['seed', 'factory', 'seed', 'factory']
    assert all(torch.equal(value, expected[key]) for key, value in second.state_dict().items())
    identity = rt.resolved_classic_identity(cfg, rt.classic_contract('vemg2pose'))
    assert 'before' in identity['seed_policy'] and 'correction' in identity['seed_policy']
