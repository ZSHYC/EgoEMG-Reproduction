"""Actual original EMG factory, trajectory loss, AdamW and strict tensor reload."""
import copy
import importlib
from types import SimpleNamespace

import pytest

from test_emg_initializer_literal import UPSTREAM,config


def api():
    try:return importlib.import_module('egoemg_repro.emg_initializer_runtime')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emg_initializer_runtime':pytest.fail('original EMG initializer runtime missing')
        raise


@pytest.fixture(scope='module')
def packet():
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');pytest.importorskip('timm')
    torch.set_num_threads(2)
    p=api().build_emg_initializer_runtime(UPSTREAM,config())
    batch=dict(emg=torch.sin(torch.arange(8*12000,dtype=torch.float32).reshape(1,8,12000)*.001),
        joint_angles=torch.full((1,22,12000),.2),label_valid_mask=torch.ones(1,12000,dtype=torch.bool),dataset_name=['egoemg'])
    p.module.eval()
    with torch.no_grad():prediction=p.module(batch)[0]
    return dict(identity=copy.deepcopy(p.identity),state={k:v.clone() for k,v in p.module.state_dict().items()},batch=batch,prediction=prediction)


def restore(packet):
    return api().build_emg_initializer_runtime(UPSTREAM,config(),state=packet['state'],expected_initialization=packet['identity'])


def test_setup_seeds_before_original_factory_and_exact_random_native_state(packet,monkeypatch):
    import torch
    import egoemg.train as native
    factory=native.make_lightning_module;seeds=[]
    def observed(cfg):seeds.append(torch.initial_seed());return factory(cfg)
    monkeypatch.setattr(native,'make_lightning_module',observed);torch.manual_seed(7)
    p=api().build_emg_initializer_runtime(UPSTREAM,config())
    assert seeds==[42] and p.mode=='fresh_native_random'
    assert type(p.module).__module__=='egoemg.lightning' and type(p.module.model).__name__=='Emg2PoseFormer'
    for name,tensor in p.module.state_dict().items():assert torch.equal(tensor,packet['state'][name]),name
    v=p.identity
    assert len(v['model_schema'])==len(packet['state'])==110
    assert v['initialization']==dict(source='original factory random initialization',seed=42,pretrained=False)
    assert v['parameter_count']==sum(x.numel() for x in p.module.parameters())
    assert v['effective_loss_weights']=={'mae':1.,'landmark/fingertip':.01}
    assert v['batch_augmentation']==p.module.hparams.batch_augmentation
    assert all(v[k] is False for k in ('scientific_eligible','production_ready','full_paper_complete'))
    assert not torch.cuda.is_initialized();p.check()


def test_original_trajectory_forward_loss_backward_and_adam_update(packet,monkeypatch):
    import torch
    p=restore(packet);module=p.module;module.eval();metrics={}
    monkeypatch.setattr(module,'log_dict',lambda values,**kw:metrics.update(values))
    monkeypatch.setattr(module,'log',lambda *args,**kw:None)
    prediction,target,mask=module(packet['batch'])
    assert prediction.shape==target.shape==(1,22,11490) and mask.shape==(1,11490)
    assert torch.equal(prediction,packet['prediction'])
    loss=module._step(packet['batch'],'train')
    expected=metrics['train_mae']+.01*metrics['train_landmark/fingertip']
    assert torch.equal(loss,expected) and torch.isfinite(loss)
    cfg=module.configure_optimizers();opt=cfg['optimizer'];scheduler=cfg['lr_scheduler']['scheduler']
    identity=api().emg_initializer_optimizer_identity(p,opt,scheduler)
    names=[n for n,x in module.named_parameters() if x.requires_grad]
    assert len(identity['groups'])==1 and identity['groups'][0]['parameter_names']==names
    assert identity['scheduler']==dict(T_max=150,eta_min=5e-6,last_epoch=0,base_lrs=[.0005])
    before=api().emg_initializer_state_sha256(module.state_dict());loss.backward()
    gradients=[x.grad for x in module.parameters() if x.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    opt.step();scheduler.step()
    assert api().emg_initializer_state_sha256(module.state_dict())!=before
    assert api().emg_initializer_optimizer_identity(p,opt,scheduler)['scheduler']['last_epoch']==1


def test_native_batch_augmentation_applies_only_in_training(packet):
    import torch
    module=restore(packet).module
    module._trainer=SimpleNamespace(training=False)
    val=dict(packet['batch'],emg=packet['batch']['emg'].clone())
    assert module.on_after_batch_transfer(val,0) is val
    assert torch.equal(val['emg'],packet['batch']['emg'])
    module._trainer.training=True;torch.manual_seed(42)
    train=module.on_after_batch_transfer(dict(packet['batch'],emg=packet['batch']['emg'].clone()),0)
    assert train['emg'].shape==(1,8,12000) and torch.isfinite(train['emg']).all()
    assert not torch.equal(train['emg'],packet['batch']['emg'])
    module._trainer=None


@pytest.mark.parametrize('damage',['missing_identity','identity_without_state','seal','sources','versions','config','key','shape','dtype','nan'])
def test_strict_restore_rejects_wrong_config_lineage_and_complete_schema_before_factory(packet,monkeypatch,damage):
    import torch
    import egoemg.train as native
    from egoemg_repro.classic_emg import _hash
    state=dict(packet['state']);identity=copy.deepcopy(packet['identity']);cfg=config();key=next(iter(state))
    if damage=='missing_identity':identity=None
    elif damage=='identity_without_state':state=None
    elif damage=='seal':identity['sha256']='0'*64
    elif damage=='sources':identity['sources_sha256']['egoemg_repro/emg_initializer_runtime.py']='0'*64
    elif damage=='versions':identity['runtime_versions']['python']='0.0.0'
    elif damage=='config':cfg.optimizer.lr=.001
    elif damage=='key':state.pop(key)
    elif damage=='shape':state[key]=state[key].reshape(-1)
    elif damage=='dtype':state[key]=state[key].half()
    elif damage=='nan':state[key]=state[key].clone();state[key].reshape(-1)[0]=float('nan')
    if damage in ('sources','versions'):identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    def forbidden(*a,**kw):raise AssertionError('invalid restore must reject before native construction')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    with pytest.raises(ValueError):api().build_emg_initializer_runtime(UPSTREAM,cfg,state=state,expected_initialization=identity)


@pytest.mark.parametrize('damage',['lr','weight_decay','duplicate','missing','scheduler','eta','optimizer_type'])
def test_optimizer_requires_every_actual_native_parameter_once_and_original_cosine(packet,damage):
    import torch
    p=restore(packet);c=p.module.configure_optimizers();opt=c['optimizer'];s=c['lr_scheduler']['scheduler']
    if damage=='lr':opt.param_groups[0]['lr']=.001
    elif damage=='weight_decay':opt.param_groups[0]['weight_decay']=0
    elif damage=='duplicate':opt.param_groups[0]['params'].append(opt.param_groups[0]['params'][0])
    elif damage=='missing':opt.param_groups[0]['params'].pop()
    elif damage=='scheduler':s.T_max=2
    elif damage=='eta':s.eta_min=0
    elif damage=='optimizer_type':opt=torch.optim.SGD(p.module.parameters(),lr=.0005)
    with pytest.raises(ValueError):api().emg_initializer_optimizer_identity(p,opt,s)
