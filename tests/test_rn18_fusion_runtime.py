"""RN18+S native adapter checks with explicitly untrained engineering branches."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.rn18_fusion_runtime')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_fusion_runtime':pytest.fail('RN18 fusion runtime missing')
        raise


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    torch=pytest.importorskip('torch')
    return torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)


@pytest.fixture(scope='module')
def branches(tmp_path_factory):
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');pytest.importorskip('torchvision')
    torch.set_num_threads(2)
    from egoemg_repro.classic_emg import _activate
    from egoemg_repro.emg_initializer_literal import compose_emg_initializer
    _activate(UPSTREAM)
    from egoemg.models.modules.resnet_vision import ResNetVisionPose
    from egoemg._train_utils import setup_runtime
    import egoemg.train as native
    torch.manual_seed(17)
    vision=ResNetVisionPose(pretrained=False)
    for module in vision.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            module.running_mean.fill_(.17);module.running_var.fill_(1.3);module.num_batches_tracked.fill_(7)
    config=compose_emg_initializer(UPSTREAM);setup_runtime(config);emg=native.make_lightning_module(config)
    root=tmp_path_factory.mktemp('rn18-fusion-untrained-engineering')
    vp=root/'vision.pt';ep=root/'emg.pt'
    torch.save({'state_dict':{'model.'+k:v for k,v in vision.state_dict().items()}},vp)
    torch.save({'state_dict':emg.state_dict()},ep)
    return dict(vision_checkpoint=vp,vision_sha256=sha(vp),emg_checkpoint=ep,emg_sha256=sha(ep))


def prepared(branches):
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    m=api();cfg=compose_rn18_fusion(UPSTREAM)
    init=m.engineering_rn18_fusion_initializers(UPSTREAM,cfg,**branches)
    return m.build_rn18_fusion_runtime(UPSTREAM,cfg,initializers=init),init


def test_engineering_initializer_explicit_full_requirement_cannot_reach_a_native_factory(branches):
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    init=api().engineering_rn18_fusion_initializers(UPSTREAM,compose_rn18_fusion(UPSTREAM),**branches)
    init.check(require_full=False)
    with pytest.raises(ValueError,match='engineering|full'):init.check(require_full=True)


def test_saved_construction_binds_the_specific_registered_initializer_identity(branches):
    p,init=prepared(branches)
    assert p.identity['initializers_identity_sha256']==init.identity['sha256']
    assert p.identity['trained_initializers_identity']is None
    assert not p.identity['initialization']['initializer_training_provenance_verified']


def batch():
    torch=pytest.importorskip('torch')
    return dict(emg=torch.linspace(-.2,.2,2*8*12000).reshape(2,8,12000),
        vision_img=torch.linspace(-1.,1.,2*3*256*256).reshape(2,3,256,256),
        joint_angles=torch.full((2,22,1),.2),label_valid_mask=torch.ones(2,1,dtype=torch.bool),
        vision_valid_mask=torch.ones(2,1,dtype=torch.bool),dataset_name=['egoemg','egoemg'])


def test_original_rn18_and_complete_emg_transfer_include_all_batchnorm_buffers(branches):
    torch=pytest.importorskip('torch');p,init=prepared(branches)
    assert type(p.module).__module__=='egoemg.lightning'
    assert type(p.module.model).__name__=='MidFusionPoseFormer'
    assert p.module.model.vision_backbone_type=='resnet18'
    assert p.module.model.featurizer.layers[0].conv[0].in_channels==8
    source=load(branches['vision_checkpoint'])['state_dict']
    for name,tensor in p.module.model.vision_backbone.state_dict().items():
        assert torch.equal(tensor,source['model.backbone.'+name])
    for name,tensor in p.module.model.head_vision.state_dict().items():
        assert torch.equal(tensor,source['model.head.'+name])
    counters=[x for n,x in p.module.model.vision_backbone.state_dict().items()if n.endswith('num_batches_tracked')]
    assert len(counters)==20 and all(x.item()==7 for x in counters)
    emg=load(branches['emg_checkpoint'])['state_dict'];transfer=p.identity['initialization']['emg_transfer']
    assert len(emg)==110 and len(transfer['matched'])==106
    assert {x['target_key']for x in transfer['skipped']}=={'head.net.0.weight','head.net.0.bias','head.net.3.weight','head.net.3.bias'}
    assert len(transfer['skipped'])==4 and all(x['reason']=='not_backbone_prefix'for x in transfer['skipped'])
    for item in transfer['matched']:
        assert torch.equal(p.module.model.state_dict()[item['target_key']],emg[item['source_key']])
    assert init.identity['scope']=='engineering'and not init.identity['trained_branches_admitted']
    assert p.identity['scientific_eligible']is False and p.identity['full_paper_complete']is False
    p.check()


def test_unfrozen_native_center_forward_loss_backward_and_adam_include_temporal_attention(branches,monkeypatch):
    torch=pytest.importorskip('torch');p,_=prepared(branches);model=p.module.model;model.train()
    assert all(x.requires_grad for x in p.module.parameters())
    assert all(x.training for x in model.vision_backbone.modules()if isinstance(x,torch.nn.BatchNorm2d))
    assert model.freeze_vision_branch is False and model.freeze_vision_head is False
    last=[x for x in model.head.modules()if isinstance(x,torch.nn.Linear)][-1]
    assert 0<float(last.weight.std())<3e-5 and torch.count_nonzero(last.bias)==0
    metrics={};monkeypatch.setattr(p.module,'log_dict',lambda v,**kw:metrics.update(v));monkeypatch.setattr(p.module,'log',lambda *a,**kw:None)
    b=batch();pred,target,mask=p.module(b)
    assert pred.shape==target.shape==(2,22,1)and torch.isfinite(pred).all()
    loss=p.module._step(b,'train');assert torch.isfinite(loss)
    assert torch.equal(loss,metrics['train_mae']+.01*metrics['train_landmark/fingertip']+.01*metrics['train_landmark/fingertip'])
    loss.backward()
    configured=p.module.configure_optimizers();opt=configured['optimizer'];sched=configured['lr_scheduler']['scheduler']
    assert type(opt)is torch.optim.Adam and sched.T_max==150 and sched.eta_min==1e-5
    names={id(v):n for n,v in p.module.named_parameters()};params=[x for g in opt.param_groups for x in g['params']]
    assert len(params)==len({id(x)for x in params})==len(names)
    assert {id(x)for x in params}==set(names)
    assert all(g['lr']==.0006 and g['weight_decay']==.0001 for g in opt.param_groups)
    unassigned=next(g for g in opt.param_groups if g['name']=='_unassigned')
    assert unassigned['params']and all(names[id(x)].startswith('model.temporal_attn.')for x in unassigned['params'])
    assert all(torch.isfinite(x.grad).all()for x in params if x.grad is not None)
    before=last.weight.detach().clone();opt.step();assert not torch.equal(before,last.weight)
    p.check();assert not torch.cuda.is_initialized()


def test_complete_restore_is_self_contained_in_fresh_process_empty_asset_cache(branches,tmp_path):
    torch=pytest.importorskip('torch');p,_=prepared(branches);p.module.eval();b=batch()
    with torch.no_grad():prediction=p.module(b)[0]
    packet=tmp_path/'self-contained.pt';torch.save(dict(identity=p.identity,state_dict=p.module.state_dict(),batch=b,prediction=prediction),packet)
    code='''import json,socket,sys,torch
from pathlib import Path
from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
from egoemg_repro.rn18_fusion_runtime import build_rn18_fusion_runtime
torch.set_num_threads(2)
def denied(*a,**kw):raise AssertionError('self-contained RN18 restore must never use network')
socket.socket.connect=denied
d=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True)
p=build_rn18_fusion_runtime(Path(sys.argv[2]),compose_rn18_fusion(Path(sys.argv[2])),state=d['state_dict'],expected_initialization=d['identity'])
p.module.eval()
with torch.no_grad():actual=p.module(d['batch'])[0]
assert torch.equal(actual,d['prediction'])and p.identity==d['identity']and not torch.cuda.is_initialized()
assert p.mode=='strict_self_contained_restore'
print(json.dumps(dict(fresh_process_prediction_exact=True,cuda_initialized=False)))
'''
    env=dict(os.environ,TORCH_HOME=str(tmp_path/'empty-torch'),HF_HOME=str(tmp_path/'empty-hf'),CUDA_VISIBLE_DEVICES='')
    result=subprocess.run([sys.executable,'-B','-c',code,str(packet),str(UPSTREAM)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-3500:]
    assert json.loads(result.stdout.splitlines()[-1])['fresh_process_prediction_exact']


@pytest.mark.parametrize('damage',['missing_bn','vision_head','emg_partial','emg_shape','nonfinite'])
def test_incomplete_native_branch_cannot_succeed(branches,tmp_path,damage):
    torch=pytest.importorskip('torch');args=dict(branches);role='vision'if damage in('missing_bn','vision_head')else'emg'
    state=dict(load(args[role+'_checkpoint'])['state_dict'])
    if damage=='missing_bn':state.pop('model.backbone.1.num_batches_tracked')
    elif damage=='vision_head':state.pop('model.head.3.bias')
    elif damage=='emg_partial':state.pop(next(iter(state)))
    elif damage=='emg_shape':
        k=next(iter(state));state[k]=state[k].reshape(-1)
    else:
        k=next(iter(state));state[k]=state[k].clone();state[k].view(-1)[0]=float('nan')
    path=tmp_path/'bad.pt';torch.save({'state_dict':state},path);args[role+'_checkpoint']=path;args[role+'_sha256']=sha(path)
    with pytest.raises(ValueError):prepared(args)


@pytest.mark.parametrize('damage',['key','dtype','nonfinite','identity','source','emg_transfer','config','mixed'])
def test_damaged_self_contained_restore_rejects_before_native_factory(branches,monkeypatch,damage):
    torch=pytest.importorskip('torch');p,init=prepared(branches)
    from egoemg_repro.classic_emg import _hash
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    state=dict(p.module.state_dict());identity=copy.deepcopy(p.identity);cfg=compose_rn18_fusion(UPSTREAM);key=next(iter(state))
    if damage=='key':state.pop(key)
    elif damage=='dtype':state[key]=state[key].half()
    elif damage=='nonfinite':state[key]=state[key].clone();state[key].view(-1)[0]=float('nan')
    elif damage=='identity':identity['sha256']='0'*64
    elif damage=='source':identity['sources_sha256']['egoemg_repro/rn18_fusion_runtime.py']='0'*64
    elif damage=='emg_transfer':identity['initialization']['emg_transfer']['matched'].pop()
    elif damage=='config':cfg.module.freeze_vision_branch=True
    if damage in('source','emg_transfer'):identity['sha256']=_hash({k:v for k,v in identity.items()if k!='sha256'})
    import egoemg.train as native
    def denied(*a,**kw):raise AssertionError('invalid restore reached native factory')
    monkeypatch.setattr(native,'make_lightning_module',denied)
    kw=dict(state=state,expected_initialization=identity)
    if damage=='mixed':kw['initializers']=init
    with pytest.raises(ValueError):api().build_rn18_fusion_runtime(UPSTREAM,cfg,**kw)


def test_engineering_registration_and_loaded_asset_bytes_cannot_be_resealed(branches,tmp_path):
    pytest.importorskip('torch');m=api()
    from egoemg_repro.classic_emg import _hash
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    cfg=compose_rn18_fusion(UPSTREAM);init=m.engineering_rn18_fusion_initializers(UPSTREAM,cfg,**branches)
    fake=copy.copy(init)
    with pytest.raises(ValueError):m.build_rn18_fusion_runtime(UPSTREAM,cfg,initializers=fake)
    init.identity['trained_branches_admitted']=True;init.identity['sha256']=_hash({k:v for k,v in init.identity.items()if k!='sha256'})
    with pytest.raises(ValueError):m.build_rn18_fusion_runtime(UPSTREAM,cfg,initializers=init)
    args=dict(branches);vp=tmp_path/'owned-vision.pt';vp.write_bytes(Path(args['vision_checkpoint']).read_bytes());args['vision_checkpoint']=vp
    p,_=prepared(args)
    with vp.open('ab')as f:f.write(b'changed')
    with pytest.raises(ValueError):p.check()
