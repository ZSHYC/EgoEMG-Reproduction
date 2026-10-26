"""Real native DINO-S pretrained/model/loss/optimizer checks; no GPU/full training qualification."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT=Path(__file__).parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.vision_literal_runtime')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.vision_literal_runtime':pytest.fail('native pretrained vision runtime missing')
        raise


@pytest.fixture(scope='module')
def native_packet(tmp_path_factory):
    pytest.importorskip('pytorch_lightning');pytest.importorskip('timm');pytest.importorskip('huggingface_hub')
    cache=Path(os.environ.get('EGOEMG_VISION_TEST_HF_CACHE',ROOT/'tmp/vision-literal-hf/hub')).resolve()
    asset=cache/'models--timm--vit_small_patch14_dinov2.lvd142m/snapshots/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors'
    if not asset.is_file():
        if os.environ.get('EGOEMG_REQUIRE_VISION_PRETRAINED_TEST')=='1':pytest.fail('required actual DINO-S cache unavailable')
        pytest.skip('actual DINO-S source asset required; no synthetic pretrained positive')
    output=tmp_path_factory.mktemp('genuine-native-vision')/'initial.pt'
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_CACHE=str(cache),HF_HOME=str(cache.parent),
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('HF_TOKEN',None);env.pop('HUGGING_FACE_HUB_TOKEN',None)
    code='''import json,sys,torch,requests
from pathlib import Path
from egoemg_repro.classic_emg import _activate
from egoemg_repro.vision_literal import compose_vision
from egoemg_repro.vision_literal_runtime import build_vision_runtime,vision_optimizer_identity
torch.set_num_threads(2);_activate(Path(sys.argv[1]))
import egoemg.train as native
factory=native.make_lightning_module;seeds=[];requests_seen=[]
def observed(cfg):seeds.append(torch.initial_seed());return factory(cfg)
def forbidden(*args,**kwargs):requests_seen.append(True);raise AssertionError('native construction must be completely offline')
native.make_lightning_module=observed;requests.adapters.HTTPAdapter.send=forbidden
torch.manual_seed(7)
p=build_vision_runtime(Path(sys.argv[1]),compose_vision(Path(sys.argv[1])))
c=p.module.configure_optimizers();optim=vision_optimizer_identity(p,c['optimizer'],c['lr_scheduler']['scheduler'])
p.module.eval()
batch=dict(vision_img=torch.arange(3*256*256,dtype=torch.float32).reshape(1,3,256,256)/(3*256*256),
    joint_angles=torch.full((1,22,1),.2),label_valid_mask=torch.ones(1,1,dtype=torch.bool),
    vision_valid_mask=torch.ones(1,1,dtype=torch.bool),emg=torch.empty(1,16,0),dataset_name=['egoemg'])
with torch.no_grad():prediction=p.module.forward(batch)[0]
assert seeds==[42] and not requests_seen and torch.cuda.is_initialized() is False
torch.save(dict(identity=p.identity,state_dict=p.module.state_dict(),batch=batch,prediction=prediction,
    optimizer_identity=optim,factory_seeds=seeds,network_calls=len(requests_seen),cuda_initialized=False),sys.argv[2])
print(json.dumps(dict(parameters=sum(x.numel() for x in p.module.model.parameters()),mode=p.mode,seed=seeds[0])))
'''
    p=subprocess.run([sys.executable,'-B','-c',code,str(UPSTREAM),str(output)],cwd=ROOT,env=env,
        capture_output=True,text=True,timeout=180)
    assert p.returncode==0,p.stderr[-5000:]
    return output


def packet(path):
    torch=pytest.importorskip('torch')
    return torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)


def restored(path):
    from egoemg_repro.vision_literal import compose_vision
    data=packet(path);p=api().build_vision_runtime(UPSTREAM,compose_vision(UPSTREAM),
        state=data['state_dict'],expected_initialization=data['identity'])
    return p,data


def test_real_native_factory_retains_all_pretrained_tensors_and_original_component_groups(native_packet):
    data=packet(native_packet);identity=data['identity'];init=identity['initialization']
    assert init['fresh_pretrained_verified'] is True and init['retained_tensor_count']==174
    assert init['asset']['sha256']=='04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
    assert init['asset']['bytes']==88240510 and init['pretrained_identity']['source_type']=='hf-hub'
    assert data['factory_seeds']==[42] and data['network_calls']==0 and data['cuda_initialized'] is False
    assert data['prediction'].shape==(1,22,1)
    opt=data['optimizer_identity'];assert [g['name'] for g in opt['groups']]==['head','backbone']
    assert [len(g['parameter_names']) for g in opt['groups']]==[4,174]
    assert all(g['lr']==.0001 and g['weight_decay']==.01 for g in opt['groups'])
    assert opt['scheduler']['T_max']==150 and opt['scheduler']['eta_min']==1e-5
    assert identity['scientific_eligible'] is False and identity['full_paper_complete'] is False


def test_strict_native_restore_needs_no_pretrained_transport_and_reproduces_actual_prediction(native_packet,monkeypatch):
    import timm.models._builder as builder
    def forbidden(*args,**kwargs):raise AssertionError('strict full-state restore must not request pretrained weights')
    monkeypatch.setattr(builder,'load_state_dict_from_hf',forbidden);monkeypatch.setattr(builder,'load_state_dict_from_url',forbidden)
    p,data=restored(native_packet);p.module.eval()
    torch=pytest.importorskip('torch');torch.set_num_threads(2)
    assert p.mode=='strict_self_contained_restore' and p.factory_pretrained is False
    assert p.identity==data['identity']
    for name,value in p.module.state_dict().items():assert torch.equal(value,data['state_dict'][name]),name
    with torch.no_grad():actual=p.module.forward(data['batch'])[0]
    assert torch.equal(actual,data['prediction'])
    p.check()


@pytest.mark.parametrize('damage',['state_type','missing_identity','identity_without_state','identity_sha','identity_source',
    'identity_variant','state_key','state_shape','state_dtype','state_nonfinite','config'])
def test_native_restore_rejects_wrong_source_config_or_typed_state_before_factory(native_packet,monkeypatch,damage):
    torch=pytest.importorskip('torch');m=api()
    from egoemg_repro.vision_literal import compose_vision
    from egoemg_repro.classic_emg import _hash
    data=packet(native_packet);state=dict(data['state_dict']);identity=copy.deepcopy(data['identity']);cfg=compose_vision(UPSTREAM)
    key=next(iter(state))
    if damage=='state_type':state=True
    elif damage=='missing_identity':identity=None
    elif damage=='identity_without_state':state=None
    elif damage=='identity_sha':identity['sha256']='0'*64
    elif damage=='identity_source':
        identity['sources_sha256']['egoemg_repro/vision_literal_runtime.py']='0'*64
        identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    elif damage=='identity_variant':
        identity['variant']='vit_base';identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    elif damage=='state_key':state.pop(key)
    elif damage=='state_shape':state[key]=state[key].reshape(-1)
    elif damage=='state_dtype':state[key]=state[key].to(torch.float16)
    elif damage=='state_nonfinite':state[key]=state[key].clone();state[key].reshape(-1)[0]=float('nan')
    elif damage=='config':cfg.loss_weights.fingertip_distance=.02
    import egoemg.train as native
    def forbidden(*args,**kwargs):raise AssertionError('invalid inputs must reject before native factory')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    with pytest.raises(ValueError):m.build_vision_runtime(UPSTREAM,cfg,state=state,expected_initialization=identity)


def test_original_loss_uses_generated_mask_even_when_crop_flag_is_false(native_packet,monkeypatch):
    torch=pytest.importorskip('torch');p,data=restored(native_packet);module=p.module;module.eval();metrics={}
    monkeypatch.setattr(module,'log_dict',lambda values,**kwargs:metrics.update(values))
    monkeypatch.setattr(module,'log',lambda *args,**kwargs:None)
    batch=data['batch']
    with torch.no_grad():
        present=module._step(dict(batch),'train')
        missing=module._step(dict(batch,vision_valid_mask=torch.zeros(1,1,dtype=torch.bool)),'train')
    assert torch.equal(present,missing) and float(missing)>0
    expected=metrics['train_mae']+metrics['train_landmark/fingertip']*.01+metrics['train_landmark/fingertip']*.01
    assert torch.equal(missing,expected)


@pytest.mark.parametrize('damage',['lr','name','duplicate','scheduler','optimizer_type'])
def test_optimizer_admission_keeps_actual_native_groups_and_cosine_recipe(native_packet,damage):
    torch=pytest.importorskip('torch');m=api();p,_=restored(native_packet);configured=p.module.configure_optimizers()
    opt=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
    if damage=='lr':opt.param_groups[0]['lr']=.001
    elif damage=='name':opt.param_groups[0]['name']='backbone'
    elif damage=='duplicate':opt.param_groups[1]['params'][0]=opt.param_groups[0]['params'][0]
    elif damage=='scheduler':scheduler.T_max=100
    elif damage=='optimizer_type':opt=torch.optim.SGD(p.module.parameters(),lr=.0001)
    with pytest.raises(ValueError):m.vision_optimizer_identity(p,opt,scheduler)


def test_fresh_native_factory_requires_prefetched_offline_asset_before_constructing(monkeypatch,tmp_path):
    pytest.importorskip('pytorch_lightning');m=api()
    from egoemg_repro.vision_literal import compose_vision
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    import egoemg.train as native
    def forbidden(*args,**kwargs):raise AssertionError('missing offline asset must reject before factory')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    monkeypatch.setenv('HF_HUB_CACHE',str(tmp_path/'empty-cache'))
    with pytest.raises(ValueError):m.build_vision_runtime(UPSTREAM,compose_vision(UPSTREAM))
