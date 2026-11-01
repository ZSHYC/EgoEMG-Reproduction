"""Real completed producer reloads; no synthetic full-training positives."""
import copy
import importlib
import json
import os
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.occlusion_dense_models')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.occlusion_dense_models':pytest.fail('trained dense triplet admission missing')
        raise


@pytest.fixture(scope='module')
def actual_models():
    m=api();path=os.environ.get('EGOEMG_DENSE_TRAINED_FIT')
    if not path or not Path(path).is_dir():
        if os.environ.get('EGOEMG_REQUIRE_DENSE_TRAINED_TEST')=='1':pytest.fail('trusted complete fusion and linked producer runs required')
        pytest.skip('actual completed fusion/vision/EMG runs required; no fake full producer positive')
    import torch
    torch.set_num_threads(2)
    original=torch.optim.Optimizer.__init__
    def forbidden(*a,**kw):raise AssertionError('prediction-only reload constructed optimizer')
    torch.optim.Optimizer.__init__=forbidden
    try:p=m.load_dense_trained_models(path,upstream=UPSTREAM,recipe='vits_s')
    finally:torch.optim.Optimizer.__init__=original
    return p


def test_real_completed_triple_matches_all_original_source_tensors_and_selections(actual_models):
    import torch
    from egoemg_repro.vision_literal_runtime import vision_state_sha256
    p=actual_models;p.check();identity=p.identity
    assert identity['trained_full_producers_verified']and not identity['dense_input_or_inference_permit']
    assert identity['selection_policy']==dict(fusion='completed full fusion validation best',
        vision='completed full vision LAST',emg='completed full EMG validation-best export')
    assert set(p.models)=={'fusion','vision','emg'}
    for role,module in p.models.items():
        state=torch.load(p.checkpoint_paths[role],map_location='cpu',weights_only=False)['state_dict']
        assert vision_state_sha256(state)==identity['complete_source_state_sha256'][role]
        if role!='fusion':state={k[6:]if k.startswith('model.')else k:v for k,v in state.items()}
        assert vision_state_sha256(state)==identity['model_state_sha256'][role]
        assert set(state)==set(module.state_dict())
        assert all(torch.equal(t,module.state_dict()[n])for n,t in state.items())
        assert not any(m.training for m in module.modules())
        assert all(x.dtype==torch.float32 and x.device.type=='cpu'for x in module.parameters())
    assert type(p.models['fusion'].model).__name__=='MidFusionPoseFormer'
    assert type(p.models['vision']).__name__=='VisionViTPose'
    assert type(p.models['emg']).__name__=='Emg2PoseFormer'
    assert not torch.cuda.is_initialized()and not identity['figure13_reproduced']


def test_resealed_or_copied_model_admission_is_not_registered(actual_models):
    p=actual_models;q=copy.copy(p)
    with pytest.raises(ValueError,match='registered'):q.check()
    original=copy.deepcopy(p.identity)
    try:
        from egoemg_repro.classic_emg import _hash
        p.identity['trained_full_producers_verified']=False
        p.identity['sha256']=_hash({k:v for k,v in p.identity.items()if k!='sha256'})
        with pytest.raises(ValueError):p.check()
    finally:p.identity.clear();p.identity.update(original)
    p.check()


@pytest.mark.parametrize('damage',['swapped_roles','train_mode','changed_tensor','patched_module_forward','stale_center_patch'])
def test_mutated_role_state_or_non_native_forward_rejected(actual_models,monkeypatch,damage):
    import torch
    p=actual_models;p.check()
    if damage=='swapped_roles':
        old=p.models['vision']
        try:
            p.models['vision']=p.models['emg']
            with pytest.raises(ValueError):p.check()
        finally:p.models['vision']=old
    elif damage=='train_mode':
        module=p.models['emg']
        try:
            module.train()
            with pytest.raises(ValueError):p.check()
        finally:module.eval()
    elif damage=='changed_tensor':
        parameter=next(p.models['fusion'].parameters());before=parameter.detach().clone()
        try:
            with torch.no_grad():parameter.add_(1.)
            with pytest.raises(ValueError):p.check()
        finally:
            with torch.no_grad():parameter.copy_(before)
    else:
        module=p.models['fusion']
        obj,name=(module,'forward')if damage=='patched_module_forward'else(module.model,'_forward_center_supervised')
        with monkeypatch.context()as local:
            local.setattr(obj,name,lambda *a,**kw:None)
            with pytest.raises(ValueError):p.check()
    p.check()


def test_wrong_family_and_missing_completed_fit_reject_before_model_construction(tmp_path,monkeypatch):
    import torch
    m=api()
    def forbidden(*a,**kw):raise AssertionError('bad fit constructed a model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    with pytest.raises(ValueError):m.load_dense_trained_models(tmp_path,upstream=UPSTREAM,recipe='rn18_s')
    with pytest.raises(ValueError):m.load_dense_trained_models(tmp_path,upstream=UPSTREAM,recipe='vits_s')


def test_restoring_the_exact_original_bound_forward_keeps_native_behavior(actual_models):
    p=actual_models;module=p.models['fusion'];original=module.forward
    try:
        module.forward=original
        p.check()
    finally:
        if 'forward'in vars(module):delattr(module,'forward')
    p.check()


def test_a_native_forward_bound_to_another_instance_is_not_the_admitted_model(actual_models):
    p=actual_models;module=p.models['fusion'];other=copy.copy(module)
    try:
        module.forward=other.forward
        with pytest.raises(ValueError):p.check()
    finally:
        if 'forward'in vars(module):delattr(module,'forward')
    p.check()
