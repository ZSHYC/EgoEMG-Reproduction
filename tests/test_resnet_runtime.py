"""Genuine original ResNet construction, objective and complete offline reload."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning')

from test_vision_literal_data import UPSTREAM


def api():return importlib.import_module('egoemg_repro.resnet_runtime')


def config(variant='resnet18'):
    from egoemg_repro.resnet_literal import compose_resnet
    return compose_resnet(UPSTREAM,variant)


def batch():
    import torch
    return dict(vision_img=torch.arange(3*256*256,dtype=torch.float32).reshape(1,3,256,256)/(3*256*256),
        joint_angles=torch.full((1,22,1),.2),label_valid_mask=torch.ones(1,1,dtype=torch.bool),
        vision_valid_mask=torch.zeros(1,1,dtype=torch.bool),emg=torch.empty(1,16,0),dataset_name=['egoemg'])


def genuine(variant='resnet18'):
    import torch
    pytest.importorskip('pytorch_lightning')
    cache=Path(torch.hub.get_dir())/'checkpoints'
    if not list(cache.glob(variant+'-*.pth')):
        if os.environ.get('EGOEMG_REQUIRE_RESNET_PRETRAINED_TEST')=='1':pytest.fail('required real ImageNet asset missing')
        pytest.skip('real prefetched ImageNet asset required; no synthetic positive')
    torch.set_num_threads(2)
    return api().build_resnet_runtime(UPSTREAM,config(variant))


@pytest.mark.parametrize('variant,parameters,retained,counters',[
    ('resnet18',11450454,100,20),('resnet50',24568406,265,53),('resnet152',59204182,775,155)])
def test_original_backbone_seed_imagenet_objective_and_complete_independent_reload(
        tmp_path,monkeypatch,variant,parameters,retained,counters):
    m=api()
    import torch
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    import egoemg.train as native
    factory=native.make_lightning_module;seeds=[]
    def observed(cfg):seeds.append(torch.initial_seed());return factory(cfg)
    def forbidden(*a,**k):raise AssertionError('prefetched native construction cannot download')
    monkeypatch.setattr(native,'make_lightning_module',observed)
    monkeypatch.setattr(torch.hub,'download_url_to_file',forbidden)
    torch.manual_seed(7);p=genuine(variant);module=p.module;module.eval()
    assert seeds==[42] and type(module).__module__=='egoemg.lightning'
    assert type(module.model).__name__=='ResNetVisionPose' and module.model.backbone_type==variant
    assert sum(x.numel()for x in module.parameters())==parameters
    assert all(x.requires_grad for x in module.parameters())
    assert p.identity['initialization']['retained_tensor_count']==retained
    assert len(p.identity['initialization']['native_default_batch_counter_names'])==counters
    assert p.identity['initialization']['native_backbone_tensor_count']==retained+counters
    for name in p.identity['initialization']['native_default_batch_counter_names']:
        assert module.model.backbone.state_dict()[name].item()==0
    assert p.identity['initialization']['fresh_pretrained_verified'] is True
    assert p.identity['initialization']['pretrained_identity']['enum'].endswith('_Weights.IMAGENET1K_V1')
    assert p.identity['scientific_eligible'] is False and not torch.cuda.is_initialized()
    configured=module.configure_optimizers()
    opt=m.resnet_optimizer_identity(p,configured['optimizer'],configured['lr_scheduler']['scheduler'])
    assert [g['name']for g in opt['groups']]==['head','backbone']
    assert all(g['lr']==.0001 and g['weight_decay']==.01 for g in opt['groups'])
    assert opt['scheduler']['T_max']==150 and opt['scheduler']['eta_min']==1e-5
    metrics={}
    monkeypatch.setattr(module,'log_dict',lambda values,**kw:metrics.update(values))
    monkeypatch.setattr(module,'log',lambda *a,**kw:None)
    sample=batch();loss=module._step(sample,'train')
    expected=metrics['train_mae']+metrics['train_landmark/fingertip']*.01+metrics['train_landmark/fingertip']*.01
    last=module.model.head[-1].weight
    actual_grad=torch.autograd.grad(loss,last,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected,last,retain_graph=True)[0]
    legacy_grad=torch.autograd.grad(metrics['train_mae']+metrics['train_landmark/fingertip']*.01,last)[0]
    assert torch.equal(loss,expected) and torch.equal(actual_grad,expected_grad)
    assert torch.isfinite(actual_grad).all() and not torch.equal(actual_grad,legacy_grad)
    with torch.no_grad():
        # A later state may contain trained BN buffers. Complete reload must
        # retain them, rather than reinstalling the fresh ImageNet/defaults.
        for name,buffer in module.named_buffers():
            if name.endswith('num_batches_tracked'):buffer.fill_(17)
        module.model.backbone[1].running_mean.add_(.125)
        prediction=module.forward(sample)[0]
    packet=tmp_path/'state.pt'
    torch.save(dict(identity=p.identity,state=module.state_dict(),batch=sample,prediction=prediction),packet)
    code='''import sys,torch,json,os
from pathlib import Path
from egoemg_repro.resnet_literal import compose_resnet
from egoemg_repro.resnet_runtime import build_resnet_runtime
def forbidden(*a,**k):raise AssertionError('reload cannot download or require original asset')
torch.hub.download_url_to_file=forbidden;torch.set_num_threads(2)
x=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
p=build_resnet_runtime(Path(sys.argv[1]),compose_resnet(Path(sys.argv[1]),sys.argv[3]),state=x['state'],expected_initialization=x['identity'])
assert p.mode=='strict_self_contained_restore' and p.factory_pretrained is False
for n,t in p.module.state_dict().items():assert torch.equal(t,x['state'][n]),n
p.module.eval()
with torch.no_grad():assert torch.equal(p.module.forward(x['batch'])[0],x['prediction'])
assert not torch.cuda.is_initialized();p.check()
s=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps(dict(complete_tensor_and_prediction_reopen_passed=True,variant=sys.argv[3],
 process=dict(pid=os.getpid(),start_ticks=s[19],uid=os.getuid()),loaded_state_sha256=p.loaded_state_sha256)))
'''
    env=dict(os.environ,TORCH_HOME=str(tmp_path/'empty-cache'),EGOEMG_NO_PRETRAINED_DOWNLOAD='1',CUDA_VISIBLE_DEVICES='')
    run=subprocess.run([sys.executable,'-B','-c',code,str(UPSTREAM),str(packet),variant],
        capture_output=True,text=True,env=env,timeout=120)
    assert run.returncode==0,run.stderr[-3500:]
    child=json.loads(run.stdout.splitlines()[-1])
    assert child['complete_tensor_and_prediction_reopen_passed'] is True
    assert child['process']['pid']!=os.getpid() and child['variant']==variant
    from egoemg_repro.vision_literal_runtime import vision_state_sha256
    assert child['loaded_state_sha256']==vision_state_sha256(module.state_dict())
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    evidence=dict(format='genuine_resnet_cpu_runtime_integration_v1',variant=variant,
        producer_process=dict(pid=os.getpid(),start_ticks=stat[19],uid=os.getuid()),reopen=child,
        construction_identity=p.identity,parameter_count=parameters,retained_imagenet_tensors=retained,
        native_default_bn_counters=counters,original_loss_coefficient=.02,
        original_loss_and_gradients_exact=True,legacy_001_gradient_differs=True,
        gradient_sha256=vision_state_sha256(dict(head_weight_gradient=actual_grad)),
        prediction_sha256=vision_state_sha256(dict(prediction=prediction)),
        saved_bn_counters=17,saved_bn_running_mean_offset=.125,optimizer_updates=0,
        cuda_initialized=False,scientific_eligible=False,full_paper_complete=False)
    (tmp_path/'runtime-evidence.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    p.check()


_PACKET=None


def saved():
    global _PACKET
    if _PACKET is None:
        p=genuine()
        _PACKET=dict(identity=copy.deepcopy(p.identity),state={k:v.detach().clone()for k,v in p.module.state_dict().items()})
    return _PACKET


@pytest.mark.parametrize('damage',['missing_identity','identity_without_state','state_type','identity_hash',
    'source','variant','missing_bn','shape','dtype','nonfinite','pretraining','config'])
def test_bad_restore_rejects_before_original_factory_or_pretrained_fallback(monkeypatch,damage):
    m=api()
    import torch
    from egoemg_repro.classic_emg import _hash
    x=saved();identity=copy.deepcopy(x['identity']);state=dict(x['state']);cfg=config()
    key='model.backbone.1.running_mean'
    if damage=='missing_identity':identity=None
    elif damage=='identity_without_state':state=None
    elif damage=='state_type':state=True
    elif damage=='identity_hash':identity['sha256']='0'*64
    elif damage=='source':identity['sources_sha256']['egoemg_repro/resnet_runtime.py']='0'*64
    elif damage=='variant':identity['variant']='resnet50'
    elif damage=='missing_bn':state.pop(key)
    elif damage=='shape':state[key]=state[key].reshape(1,-1)
    elif damage=='dtype':state[key]=state[key].double()
    elif damage=='nonfinite':state[key]=torch.full_like(state[key],float('nan'))
    elif damage=='pretraining':identity['initialization']['fresh_pretrained_verified']=False
    else:cfg.module.pretrained=False
    if damage in ('source','variant','pretraining'):
        identity['sha256']=_hash({k:v for k,v in identity.items()if k!='sha256'})
    import egoemg.train as native
    def forbidden(*a,**k):raise AssertionError('invalid restore must fail before factory/network')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    monkeypatch.setattr(torch.hub,'download_url_to_file',forbidden)
    with pytest.raises(ValueError):m.build_resnet_runtime(UPSTREAM,cfg,state=state,expected_initialization=identity)


@pytest.mark.parametrize('damage',['disabled','missing','corrupt'])
def test_fresh_pretraining_must_be_explicit_and_content_validated_before_factory(monkeypatch,tmp_path,damage):
    m=api()
    import torch
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    import egoemg.train as native
    def forbidden(*a,**k):raise AssertionError('invalid asset must fail before factory/network')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    if damage=='disabled':monkeypatch.setenv('EGOEMG_NO_PRETRAINED_DOWNLOAD','1')
    else:
        monkeypatch.setattr(torch.hub,'get_dir',lambda:str(tmp_path))
        if damage=='corrupt':
            (tmp_path/'checkpoints').mkdir();(tmp_path/'checkpoints/resnet18-f37072fd.pth').write_bytes(b'invalid asset')
    with pytest.raises(ValueError):m.build_resnet_runtime(UPSTREAM,config())
