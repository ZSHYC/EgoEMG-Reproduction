"""RN18 dense triplet integration, using genuine completed producer weights."""
import copy
import hashlib
import importlib
import os
from pathlib import Path
import pytest

UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).resolve().parents[1]/'upstream')).resolve()
FIT=os.environ.get('EGOEMG_RN18_DENSE_TRAINED_FIT')


def api():
    try:return importlib.import_module('egoemg_repro.rn18_occlusion_models')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_occlusion_models':pytest.fail('RN18 trained dense triplet integration missing')
        raise


@pytest.fixture(scope='module')
def actual_models():
    m=api()
    if not FIT or not Path(FIT).is_dir():
        if os.environ.get('EGOEMG_REQUIRE_RN18_DENSE_TRAINED_TEST')=='1':pytest.fail('actual completed RN18 fusion and linked full producers required')
        pytest.skip('actual completed RN18 producers required; no synthetic full positive')
    import torch
    torch.set_num_threads(2)
    original=torch.optim.Optimizer.__init__
    def forbidden(*a,**k):raise AssertionError('inference reload constructed optimizer')
    torch.optim.Optimizer.__init__=forbidden
    try:value=m.load_rn18_dense_trained_models(FIT,upstream=UPSTREAM,recipe='rn18_s')
    finally:torch.optim.Optimizer.__init__=original
    return value


def test_real_complete_RN18_triplet_retains_selected_full_states_and_native_classes(actual_models):
    import torch
    from egoemg_repro.vision_literal_runtime import vision_state_sha256
    p=actual_models;p.check();v=p.identity
    expected={'fusion':'b37f0f3c36ce48ecdcba43428db5529195bc08abfe7c7f5c06d5d4842b91dcd2',
        'vision':'10be7acdcf011dde9bc9a8b8af5d93654c048a8bc2ef2335e8f9eaa5e2970e8a',
        'emg':'4d53b5c6c548a16e29f466cd1917c2be62246adeb9cf0f730febcde3f71a5922'}
    assert v['trained_full_producers_verified']and not v['dense_input_or_inference_permit']
    assert v['recipe']=='rn18_s'and set(p.models)==set(expected)
    assert v['selected_completed_epochs']=={'fusion':145,'vision':150,'emg':74}
    assert v['complete_tensor_counts']['vision']==124 and v['complete_tensor_counts']['emg']==110
    assert sum(t.numel()for t in p.models['fusion'].parameters())==15452413
    for role,module in p.models.items():
        with Path(p.checkpoint_paths[role]).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected[role]
        state=torch.load(str(p.checkpoint_paths[role]),map_location='cpu',weights_only=False,mmap=True)['state_dict']
        if role!='fusion':state={n[6:]if n.startswith('model.')else n:t for n,t in state.items()}
        assert set(state)==set(module.state_dict())
        assert all(torch.equal(t,module.state_dict()[n])for n,t in state.items())
        assert vision_state_sha256(state)==v['model_state_sha256'][role]
        assert all(t.dtype==torch.float32 and t.device.type=='cpu'for t in module.parameters())
        assert not any(x.training for x in module.modules())
    assert type(p.models['fusion'].model).__name__=='MidFusionPoseFormer'
    assert type(p.models['vision']).__name__=='ResNetVisionPose'
    assert type(p.models['emg']).__name__=='Emg2PoseFormer'
    assert not torch.cuda.is_initialized()and not v['full_paper_complete']


def test_standalone_RN18_and_EMG_do_not_alias_fused_parameters_or_buffers(actual_models):
    p=actual_models
    storage={role:{t.untyped_storage().data_ptr()for t in model.state_dict().values()if t.numel()}for role,model in p.models.items()}
    assert not(storage['fusion']&storage['vision']or storage['fusion']&storage['emg']or storage['vision']&storage['emg'])
    p.check()


@pytest.mark.parametrize('damage',['roles','train_mode','parameter','BN_buffer','half_precision','patched_vision','patched_center','foreign_bound_forward','data_locations','checkpoint_paths'])
def test_changed_native_state_dispatch_or_input_source_is_rejected(actual_models,monkeypatch,damage):
    import torch
    p=actual_models;p.check()
    with monkeypatch.context()as patch:
        restore=None
        if damage=='roles':patch.setitem(p.models,'vision',p.models['emg'])
        elif damage=='train_mode':
            p.models['vision'].train();restore=lambda:p.models['vision'].eval()
        elif damage in ('parameter','BN_buffer'):
            tensor=next(p.models['emg'].parameters())if damage=='parameter'else next(t for n,t in p.models['vision'].named_buffers()if n.endswith('running_mean'))
            before=tensor.detach().clone()
            with torch.no_grad():tensor.add_(1.)
            def restore():
                with torch.no_grad():tensor.copy_(before)
        elif damage=='half_precision':
            module=p.models['emg'];before={n:t.clone()for n,t in module.state_dict().items()};module.half()
            def restore():module.float();module.load_state_dict(before,strict=True)
        elif damage=='patched_vision':patch.setattr(p.models['vision'],'forward',lambda *a,**k:None)
        elif damage=='patched_center':patch.setattr(p.models['fusion'].model,'_forward_center_supervised',lambda *a,**k:None)
        elif damage=='foreign_bound_forward':patch.setattr(p.models['fusion'],'forward',copy.copy(p.models['fusion']).forward)
        elif damage=='data_locations':patch.setitem(p.data_locations,'data','/unadmitted/other-data')
        elif damage=='checkpoint_paths':patch.setitem(p.checkpoint_paths,'emg',p.checkpoint_paths['vision'])
        try:
            with pytest.raises(ValueError):p.check()
        finally:
            if restore is not None:restore()
    p.check()


def test_resealed_or_copied_model_object_cannot_grant_trained_admission(actual_models):
    p=actual_models
    with pytest.raises(ValueError,match='registered'):copy.copy(p).check()
    old=copy.deepcopy(p.identity)
    try:
        from egoemg_repro.classic_emg import _hash
        p.identity['trained_full_producers_verified']=False;p.identity['sha256']=_hash({k:v for k,v in p.identity.items()if k!='sha256'})
        with pytest.raises(ValueError):p.check()
    finally:p.identity.clear();p.identity.update(old)
    p.check()


@pytest.mark.parametrize('family',['vits_s','unknown'])
def test_wrong_family_is_refused_before_any_model_constructor(tmp_path,monkeypatch,family):
    import torch
    m=api()
    def forbidden(*a,**k):raise AssertionError('wrong source constructed a model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    with pytest.raises(ValueError):m.load_rn18_dense_trained_models(tmp_path,upstream=UPSTREAM,recipe=family)


def test_unaccepted_fit_is_refused_before_any_model_constructor(tmp_path,monkeypatch):
    import torch
    m=api()
    def forbidden(*a,**k):raise AssertionError('unaccepted source constructed a model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    with pytest.raises(ValueError):m.load_rn18_dense_trained_models(tmp_path,upstream=UPSTREAM,recipe='rn18_s')


def test_resealed_completion_document_is_not_an_inference_source(actual_models,tmp_path,monkeypatch):
    import json,torch
    m=api();original=Path(m.ACCEPTED_PROOF);d=json.loads(original.read_text());d['all6_original_actors_physically_closed']=False
    other=tmp_path/'completion.json';other.write_text(json.dumps(d))
    monkeypatch.setattr(m,'ACCEPTED_PROOF',other)
    def forbidden(*a,**k):raise AssertionError('forged proof constructed a model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    with pytest.raises(ValueError):m.load_rn18_dense_trained_models(FIT,upstream=UPSTREAM,recipe='rn18_s')


@pytest.fixture(scope='module')
def actual_rn18_batches(actual_models):
    from egoemg_repro.occlusion_dense_inputs import DensePredictionDataset
    from egoemg_repro.occlusion_dense_inference import source_collate_fusion
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    ds=DensePredictionDataset(actual_models.data_locations['data'],actual_models.data_locations['crops'],
        upstream=UPSTREAM,config=compose_rn18_fusion(UPSTREAM),recipe='rn18_s')
    right=next(i for i,h in enumerate(ds.index['points']['hand_index'])if int(h)==1)
    pairs=[]
    for i in (0,right):
        item=ds[i]
        pairs.append((source_collate_fusion([item['fusion']],upstream=UPSTREAM),
            source_collate_fusion([item['standalone_emg']],upstream=UPSTREAM)))
    ds.check();return pairs


def test_rn18_batch_adapter_forwards_real_bilateral_native_inputs_without_training(actual_models,actual_rn18_batches):
    import numpy as np
    p=actual_models
    for fusion,emg in actual_rn18_batches:
        assert fusion['emg'].shape==(1,8,12000)and emg['emg'].shape==(1,8,12000)
        result=api().predict_rn18_dense_batch(p,fusion,emg)
        assert result['batch_size']==1 and not result['full_inference_permit']
        assert result['arrays']['fusion_prediction'].shape==(1,22)
        assert result['arrays']['vision_prediction'].shape==(1,22)
        assert result['arrays']['emg_prediction'].shape==(1,22)
        assert result['arrays']['emg_full_prediction'].shape[:2]==(1,22)
        assert result['emg_center_index']==result['arrays']['emg_full_prediction'].shape[-1]//2
        assert all(np.isfinite(v).all()for v in result['arrays'].values())
    p.check()


def test_rn18_batch_adapter_rejects_unregistered_model_before_inputs(actual_models):
    with pytest.raises(ValueError,match='registered'):api().predict_rn18_dense_batch(copy.copy(actual_models),{}, {})


@pytest.mark.parametrize('damage',['channels','targets'])
def test_rn18_batch_adapter_rejects_mismatched_native_inputs_before_forward(actual_models,actual_rn18_batches,monkeypatch,damage):
    import torch
    a,b=actual_rn18_batches[0];a=dict(a);b=dict(b)
    if damage=='channels':a['emg']=a['emg'].repeat(1,2,1)
    else:b['joint_angles']=b['joint_angles']+1.
    def forbidden(*a,**k):raise AssertionError('invalid pair reached a learned forward')
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    with pytest.raises(ValueError):api().predict_rn18_dense_batch(actual_models,a,b)


@pytest.mark.parametrize('field',['_guard','_fusion'])
def test_changed_source_guard_or_prepared_runtime_cannot_bypass_registration(actual_models,monkeypatch,field):
    from types import SimpleNamespace
    p=actual_models;p.check()
    with monkeypatch.context()as patch:
        patch.setattr(p,field,SimpleNamespace(check=lambda:None))
        with pytest.raises(ValueError):p.check()
    p.check()
