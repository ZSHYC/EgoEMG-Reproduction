"""Source-first S/M/L factories with the real0.02 objective and optimizer."""
import copy
import importlib
from pathlib import Path
import pytest
from test_emgformer_literal import UPSTREAM,api as literals


def api():return importlib.import_module('egoemg_repro.emgformer_runtime')


@pytest.fixture(scope='module',params=['small','middle','large'])
def packet(request):
    torch=pytest.importorskip('torch')
    torch.set_num_threads(2)
    size=request.param;cfg=literals().compose_emgformer(UPSTREAM,size)
    p=api().build_emgformer_runtime(UPSTREAM,cfg)
    batch=dict(emg=torch.sin(torch.arange(8*12000,dtype=torch.float32).reshape(1,8,12000)*.001),
        joint_angles=torch.full((1,22,12000),.2),label_valid_mask=torch.ones(1,12000,dtype=torch.bool),dataset_name=['egoemg'])
    p.module.eval()
    with torch.no_grad():prediction=p.module(batch)[0]
    return dict(size=size,config=cfg,identity=copy.deepcopy(p.identity),
        state={k:v.clone() for k,v in p.module.state_dict().items()},batch=batch,prediction=prediction)


def restore(packet):
    return api().build_emgformer_runtime(UPSTREAM,packet['config'],state=packet['state'],expected_initialization=packet['identity'])


def test_seeded_native_size_and_strict_reload_prediction(packet,monkeypatch):
    import torch
    import egoemg.train as native
    seen=[];factory=native.make_lightning_module
    def observed(cfg):seen.append(torch.initial_seed());return factory(cfg)
    monkeypatch.setattr(native,'make_lightning_module',observed)
    torch.manual_seed(7);p=api().build_emgformer_runtime(UPSTREAM,packet['config'])
    assert seen==[42] and p.mode=='fresh_native_random'
    assert type(p.module).__module__=='egoemg.lightning' and type(p.module.model).__name__=='Emg2PoseFormer'
    counts={'small':3460134,'middle':6617382,'large':16320806}
    assert p.identity['parameter_count']==sum(v.numel() for v in p.module.parameters())==counts[packet['size']]
    assert p.identity['effective_loss_weights']=={'mae':1.,'landmark/fingertip':.02}
    assert p.identity['scientific_eligible'] is p.identity['production_ready'] is p.identity['full_paper_complete'] is False
    assert api().emgformer_state_sha256(p.module.state_dict())==api().emgformer_state_sha256(packet['state'])
    restored=restore(packet);restored.module.eval()
    with torch.no_grad():pred,target,mask=restored.module(packet['batch'])
    assert pred.shape==target.shape==(1,22,11490) and mask.shape==(1,11490)
    assert torch.equal(pred,packet['prediction']) and not torch.cuda.is_initialized()
    restored.check()


def test_real_train_mode_loss_has_both_alias_terms_and_updates_all_parameters(packet,monkeypatch):
    import torch
    p=restore(packet);module=p.module;module.train();metrics={}
    monkeypatch.setattr(module,'log_dict',lambda values,**kw:metrics.update(values))
    monkeypatch.setattr(module,'log',lambda *args,**kw:None)
    loss=module._step(packet['batch'],'train')
    expected=metrics['train_mae']+.01*metrics['train_landmark/fingertip']+.01*metrics['train_landmark/fingertip']
    assert torch.equal(loss,expected) and torch.isfinite(loss)
    c=module.configure_optimizers();opt=c['optimizer'];scheduler=c['lr_scheduler']['scheduler']
    identity=api().emgformer_optimizer_identity(p,opt,scheduler)
    assert identity['scheduler']==dict(T_max=200,eta_min=1e-5,last_epoch=0,base_lrs=[.00005])
    assert identity['groups'][0]['parameter_names']==[n for n,x in module.named_parameters() if x.requires_grad]
    before=api().emgformer_state_sha256(module.state_dict());loss.backward()
    gradients=[x.grad for x in module.parameters() if x.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    opt.step();scheduler.step()
    assert api().emgformer_state_sha256(module.state_dict())!=before
    assert api().emgformer_optimizer_identity(p,opt,scheduler)['scheduler']['last_epoch']==1


@pytest.mark.parametrize('damage',['missing_identity','seal','size','sources','key','dtype','nan','config','objective','augmentation'])
def test_invalid_complete_state_or_lineage_rejects_before_original_factory(packet,monkeypatch,damage):
    import torch
    import egoemg.train as native
    from egoemg_repro.classic_emg import _hash
    from omegaconf import OmegaConf
    state=dict(packet['state']);identity=copy.deepcopy(packet['identity']);cfg=copy.deepcopy(packet['config']);key=next(iter(state))
    if damage=='missing_identity':identity=None
    elif damage=='seal':identity['sha256']='0'*64
    elif damage=='size':identity['size']='large' if packet['size']!='large' else 'small'
    elif damage=='sources':identity['sources_sha256']['bad']='0'*64
    elif damage=='key':state.pop(key)
    elif damage=='dtype':state[key]=state[key].half()
    elif damage=='nan':state[key]=state[key].clone();state[key].reshape(-1)[0]=float('nan')
    elif damage=='config':OmegaConf.update(cfg,'optimizer.lr',.0005)
    elif damage=='objective':identity['effective_loss_weights']['landmark/fingertip']=.01
    elif damage=='augmentation':identity['batch_augmentation']={'invented':True}
    if damage in ('size','sources','objective','augmentation'):identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    def forbidden(*a,**kw):raise AssertionError('invalid restore reached native factory')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    with pytest.raises(ValueError):api().build_emgformer_runtime(UPSTREAM,cfg,state=state,expected_initialization=identity)


@pytest.mark.parametrize('damage',['missing','duplicate','lr','scheduler','eta'])
def test_wrong_parameter_coverage_or_original_scheduler_is_rejected(packet,damage):
    p=restore(packet);c=p.module.configure_optimizers();opt=c['optimizer'];scheduler=c['lr_scheduler']['scheduler']
    if damage=='missing':opt.param_groups[0]['params'].pop()
    elif damage=='duplicate':opt.param_groups[0]['params'].append(opt.param_groups[0]['params'][0])
    elif damage=='lr':opt.param_groups[0]['lr']=.001
    elif damage=='scheduler':scheduler.T_max=150
    elif damage=='eta':scheduler.eta_min=5e-6
    with pytest.raises(ValueError):api().emgformer_optimizer_identity(p,opt,scheduler)


@pytest.mark.parametrize('field',['effective_loss_weights','production_ready'])
def test_prepared_check_rejects_resealed_false_objective_or_training_permit(packet,field):
    from egoemg_repro.classic_emg import _hash
    p=restore(packet)
    if field=='effective_loss_weights':p.identity[field]['landmark/fingertip']=.01
    else:p.identity[field]=True
    p.identity['sha256']=_hash({k:v for k,v in p.identity.items() if k!='sha256'})
    with pytest.raises(ValueError):p.check()


def test_prepared_check_requires_actual_native_loss_weights(packet):
    cfg=copy.deepcopy(packet['config'])
    p=api().build_emgformer_runtime(UPSTREAM,cfg,state=packet['state'],expected_initialization=packet['identity'])
    p.module.loss_weights['fingertip_distance']=0.
    with pytest.raises(ValueError):p.check()


@pytest.mark.parametrize('flag,value',[('maximize',True),('foreach',True),('capturable',True),
    ('differentiable',True),('fused',True)])
def test_optimizer_execution_defaults_must_remain_native(packet,flag,value):
    p=restore(packet);c=p.module.configure_optimizers();opt=c['optimizer'];scheduler=c['lr_scheduler']['scheduler']
    opt.param_groups[0][flag]=value
    with pytest.raises(ValueError):api().emgformer_optimizer_identity(p,opt,scheduler)


def test_freezing_and_removing_native_parameter_cannot_shrink_expected_coverage(packet):
    p=restore(packet);c=p.module.configure_optimizers();opt=c['optimizer'];scheduler=c['lr_scheduler']['scheduler']
    opt.param_groups[0]['params'][0].requires_grad_(False)
    opt.param_groups[0]['params'].pop(0)
    with pytest.raises(ValueError):api().emgformer_optimizer_identity(p,opt,scheduler)
