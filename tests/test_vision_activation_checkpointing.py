"""Opt-in native recomputation without changing the DINO model or batches."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from test_vision_literal_data import UPSTREAM, config
from test_vision_literal_train import native_runs, PrivateRun, ROOT


def api():
    return importlib.import_module('egoemg_repro.vision_literal_train')


def test_checkpoint_switch_is_explicit_bound_and_preserves_original_budget():
    pytest.importorskip('hydra'); pytest.importorskip('timm')
    m = api()
    ordinary = m.vision_fit_policy(UPSTREAM, config(), scope='full')
    enabled = m.vision_fit_policy(UPSTREAM, config(), scope='full', activation_checkpointing=True)
    assert ordinary['activation_checkpointing'] is False
    assert ordinary['activation_checkpoint_use_reentrant'] is None
    assert enabled['activation_checkpointing'] is True
    assert type(enabled['activation_checkpoint_use_reentrant']) is bool
    assert enabled['sha256'] != ordinary['sha256']
    assert enabled['batch_size_by_role'] == dict(train=360, val=360, test=360)
    assert enabled['precision'] == 'bf16-mixed' and enabled['gradient_clip_val'] == 1
    assert enabled['max_epochs'] == 150 and enabled['patience'] == 30
    assert {'torch.utils.checkpoint', 'timm.models._manipulate', 'timm.layers.config'} <= set(m._loop_sources())


@pytest.mark.parametrize('value', [None, 0, 1, 'true', {}, []])
def test_truthy_or_implicit_checkpoint_switch_rejects(value):
    pytest.importorskip('hydra')
    with pytest.raises(ValueError, match='boolean'):
        api().vision_fit_policy(UPSTREAM, config(), activation_checkpointing=value)


def test_real_native_dino_recomputation_preserves_parameters_predictions_gradients_and_rng():
    torch = pytest.importorskip('torch'); pytest.importorskip('timm'); pytest.importorskip('hydra')
    from egoemg_repro.vision_literal import compose_vision
    from egoemg_repro.classic_emg import _activate
    from egoemg_repro.vision_literal_runtime import vision_state_sha256
    compose_vision(UPSTREAM, 'vit_small')
    _activate(UPSTREAM)
    from egoemg.models.modules.vit_vision import VisionViTPose
    torch.manual_seed(219)
    model = VisionViTPose(pretrained=False, backbone_type='vit_small').train()
    prepared = SimpleNamespace(module=SimpleNamespace(model=model))
    parameter_ids = [id(p) for p in model.parameters()]
    digest = vision_state_sha256(model.state_dict())
    image = torch.randn(2, 3, 256, 256)
    rng_before = torch.get_rng_state().clone()
    plain = model({'vision_img': image}); plain.square().mean().backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    rng_after = torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    api()._configure_activation_checkpointing(prepared, True)
    assert model.backbone.grad_checkpointing is True
    assert [id(p) for p in model.parameters()] == parameter_ids
    assert vision_state_sha256(model.state_dict()) == digest
    torch.set_rng_state(rng_before)
    checkpointed = model({'vision_img': image}); checkpointed.square().mean().backward()
    assert torch.equal(checkpointed, plain)
    assert all(torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters())
    assert torch.equal(torch.get_rng_state(), rng_after)
    policy = api().vision_fit_policy(UPSTREAM, config(), activation_checkpointing=True)
    api()._check_activation_checkpointing(prepared, policy)
    api()._configure_activation_checkpointing(prepared, False)
    assert model.backbone.grad_checkpointing is False
    with pytest.raises(ValueError, match='checkpoint'):
        api()._check_activation_checkpointing(prepared, policy)


@pytest.fixture(scope='module')
def checkpointed_native_runs(native_runs):
    root, env, base, _ = native_runs
    output = root / 'checkpointed'
    for name, extra in [('checkpointed-prefix', ['--stop-after-epoch', '1']), ('checkpointed-resume', ['--resume'])]:
        result = subprocess.run([*base, '--output', str(output), '--activation-checkpointing', *extra],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
        (root / (name + '.stdout')).write_text(result.stdout)
        (root / (name + '.stderr')).write_text(result.stderr)
        assert result.returncode == 0, result.stderr[-3500:]
    return PrivateRun(root, env, base, output)


def test_native_checkpointed_fit_restores_all_state_and_full_small_validation(checkpointed_native_runs):
    _, _, _, output = checkpointed_native_runs
    m = api(); result = m.read_vision_fit(output, require_full=False)
    identity = json.loads((output / 'identity.json').read_text())
    assert identity['policy']['activation_checkpointing'] is True
    assert result['completed_epochs'] == 2 and result['optimizer_executions'] == 4
    assert all(result['resume_start_exact'].values())
    assert result['train_samples_per_epoch'] == result['val_samples_per_epoch'] == [4, 4]
    assert result['scientific_eligible'] is False and result['full_training_terminated'] is False


def test_opposite_switch_resume_rejects_without_overwriting_finished_artifacts(checkpointed_native_runs):
    _, env, base, output = checkpointed_native_runs
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (output / 'identity.json', output / 'fit-result.json', output / 'checkpoints/last.ckpt')}
    result = subprocess.run([*base, '--output', str(output), '--resume'], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and 'owned resume source/config/cohort differs' in result.stderr
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in protected.items())
