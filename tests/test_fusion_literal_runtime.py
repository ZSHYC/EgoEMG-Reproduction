"""Real original fusion CPU construction; engineering inputs, no trained accuracy."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_vision_literal_runtime import native_packet


ROOT=Path(__file__).parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.fusion_literal_runtime')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.fusion_literal_runtime':pytest.fail('native fusion construction missing')
        raise


def load(path):
    torch=pytest.importorskip('torch')
    return torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


@pytest.fixture(scope='module')
def branch_files(native_packet,tmp_path_factory):
    torch=pytest.importorskip('torch')
    from egoemg_repro.classic_emg import _activate
    from hydra import compose,initialize_config_dir
    _activate(UPSTREAM)
    from egoemg._train_utils import setup_runtime
    import egoemg.train as native
    root=tmp_path_factory.mktemp('fusion-native-engineering-branches')
    data=load(native_packet)
    vision=root/'vision.pt';torch.save({'state_dict':data['state_dict']},vision)
    with initialize_config_dir(config_dir=str(UPSTREAM/'config'),version_base='1.1'):
        c=compose(config_name='base',overrides=['experiment=emgformer/egoemg_emgformer_small'])
    setup_runtime(c);emg_module=native.make_lightning_module(c)
    emg=root/'emg.pt';torch.save({'state_dict':emg_module.state_dict()},emg)
    return dict(vision_checkpoint=vision,vision_sha256=sha(vision),emg_checkpoint=emg,emg_sha256=sha(emg))


@pytest.fixture(scope='module')
def fusion_packet(branch_files,tmp_path_factory):
    torch=pytest.importorskip('torch');torch.set_num_threads(2)
    from egoemg_repro.fusion_literal import compose_fusion
    p=api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),**branch_files)
    p.module.eval()
    batch=dict(emg=torch.linspace(-.2,.2,16*7790).reshape(1,16,7790),
        vision_img=torch.linspace(-1.,1.,3*256*256).reshape(1,3,256,256),
        joint_angles=torch.full((1,22,1),.2),label_valid_mask=torch.ones(1,1,dtype=torch.bool),
        vision_valid_mask=torch.ones(1,1,dtype=torch.bool),dataset_name=['egoemg'])
    with torch.no_grad():prediction=p.module(batch)[0]
    path=tmp_path_factory.mktemp('fusion-native-engineering-state')/'full.pt'
    torch.save(dict(identity=p.identity,state_dict=p.module.state_dict(),batch=batch,prediction=prediction),path)
    return path


def restored(path):
    torch=pytest.importorskip('torch')
    from egoemg_repro.fusion_literal import compose_fusion
    data=load(path)
    return api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),state=data['state_dict'],
        expected_initialization=data['identity']),data


def test_native_fresh_transfer_is_complete_for_vision_and_exact_partial_for_emg(fusion_packet,branch_files):
    torch=pytest.importorskip('torch');p,data=restored(fusion_packet);identity=data['identity']
    assert type(p.module).__module__=='egoemg.lightning' and type(p.module.model).__name__=='MidFusionPoseFormer'
    assert p.module.model.featurizer.layers[0].conv[0].in_channels==16
    assert data['prediction'].shape==(1,22,1) and torch.isfinite(data['prediction']).all()
    source=load(branch_files['vision_checkpoint'])['state_dict']
    for k,t in p.module.model.vision_backbone.state_dict().items():assert torch.equal(t,source['model.backbone.'+k])
    for k,t in p.module.model.head_vision.state_dict().items():assert torch.equal(t,source['model.head.'+k])
    transfer=identity['initialization']['emg_transfer']
    assert transfer['matched'] and any(x['target_key'].startswith('decoder.') for x in transfer['matched'])
    skip=next(x for x in transfer['skipped'] if x['source_key']=='model.featurizer.layers.0.conv.0.weight')
    assert skip['reason']=='shape_mismatch' and skip['source_shape'][1]==8 and skip['target_shape'][1]==16
    emg=load(branch_files['emg_checkpoint'])['state_dict']
    for item in transfer['matched']:
        actual=p.module.model.state_dict()[item['target_key']]
        assert torch.equal(actual,emg[item['source_key']].to(actual.dtype))
    residual=p.module.model.head.net[-1]
    assert torch.count_nonzero(residual.weight)>0 and torch.count_nonzero(residual.bias)==0
    assert 5e-6<float(residual.weight.std())<2e-5
    assert identity['initialization']['initializer_training_provenance_verified'] is False
    assert identity['scientific_eligible'] is identity['production_ready'] is identity['full_paper_complete'] is False


def test_real_original_loss_backward_and_every_optimizer_parameter_are_retained(fusion_packet,monkeypatch):
    torch=pytest.importorskip('torch');p,data=restored(fusion_packet);module=p.module;module.train();metrics={}
    monkeypatch.setattr(module,'log_dict',lambda values,**kwargs:metrics.update(values))
    monkeypatch.setattr(module,'log',lambda *args,**kwargs:None)
    loss=module._step(data['batch'],'train')
    assert torch.equal(loss,metrics['train_mae']+metrics['train_landmark/fingertip']*.01+metrics['train_landmark/fingertip']*.01)
    assert torch.isfinite(loss) and loss.requires_grad;loss.backward()
    configured=module.configure_optimizers();optimizer=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
    assert type(optimizer) is torch.optim.AdamW and type(scheduler) is torch.optim.lr_scheduler.CosineAnnealingLR
    assert scheduler.T_max==150 and scheduler.eta_min==1e-5
    groups=optimizer.param_groups;parameters=[p for g in groups for p in g['params']]
    assert len({id(x) for x in parameters})==len(parameters)
    assert {id(x) for x in parameters}=={id(x) for x in module.parameters() if x.requires_grad}
    assert all(g['lr']==.0001 and g['weight_decay']==.01 for g in groups)
    named={id(v):k for k,v in module.named_parameters()}
    unassigned=next(g for g in groups if g['name']=='_unassigned')
    assert all(named[id(x)].startswith('model.temporal_attn.') for x in unassigned['params'])
    assert all(torch.isfinite(x.grad).all() for x in parameters if x.grad is not None)
    optimizer.step();p.check();assert not torch.cuda.is_initialized()


def test_fresh_native_factory_is_seeded_and_offline_with_provided_branches(branch_files,monkeypatch):
    pytest.importorskip('torch');import torch,requests
    from egoemg_repro.fusion_literal import compose_fusion
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM);import egoemg.train as native
    seen=[];factory=native.make_lightning_module
    def observed(config):seen.append(torch.initial_seed());return factory(config)
    def forbidden(*args,**kwargs):raise AssertionError('native local branch construction must not use network')
    monkeypatch.setattr(native,'make_lightning_module',observed)
    monkeypatch.setattr(requests.adapters.HTTPAdapter,'send',forbidden)
    p=api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),**branch_files)
    assert seen==[42] and p.mode=='fresh_local_branch_initialization' and not torch.cuda.is_initialized()


def test_native_branch_dtype_conversion_is_recorded_without_changing_loader(branch_files,tmp_path):
    torch=pytest.importorskip('torch');from egoemg_repro.fusion_literal import compose_fusion
    args=dict(branch_files)
    for role in ('vision','emg'):
        source=load(args[role+'_checkpoint'])['state_dict']
        state={k:v.to(torch.float16) if v.is_floating_point() else v for k,v in source.items()}
        path=tmp_path/(role+'-half.pt');torch.save({'state_dict':state},path)
        args[role+'_checkpoint']=path;args[role+'_sha256']=sha(path)
    p=api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),**args)
    init=p.identity['initialization'];v=init['vision_transfer']
    assert v['source_backbone_state_sha256']!=v['loaded_backbone_state_sha256']
    assert all(x['source_dtype']=='torch.float16' and x['target_dtype']=='torch.float32'
        for x in init['emg_transfer']['matched'])
    assert all(x.dtype==torch.float32 for x in p.module.model.vision_backbone.parameters())
    p.check()


@pytest.mark.parametrize('damage',['vision_hash','emg_hash','vision_missing','emg_missing','mixed_mode','no_initializers'])
def test_invalid_initializers_reject_before_native_factory(branch_files,fusion_packet,tmp_path,monkeypatch,damage):
    pytest.importorskip('torch')
    from egoemg_repro.fusion_literal import compose_fusion
    import egoemg.train as native
    args=dict(branch_files)
    if damage=='vision_hash':args['vision_sha256']='0'*64
    elif damage=='emg_hash':args['emg_sha256']='0'*64
    elif damage=='vision_missing':args['vision_checkpoint']=tmp_path/'absent'
    elif damage=='emg_missing':args['emg_checkpoint']=tmp_path/'absent'
    elif damage=='mixed_mode':
        import torch
        data=load(fusion_packet);args.update(state=data['state_dict'],expected_initialization=data['identity'])
    elif damage=='no_initializers':args={}
    def forbidden(*args,**kwargs):raise AssertionError('bad inputs must fail before native factory')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    with pytest.raises(ValueError):api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),**args)


@pytest.mark.parametrize('damage',['vision_incomplete','vision_nonfinite','emg_empty'])
def test_native_partial_vision_or_invalid_emg_never_qualifies(branch_files,tmp_path,damage):
    torch=pytest.importorskip('torch');from egoemg_repro.fusion_literal import compose_fusion
    args=dict(branch_files);role='vision' if damage.startswith('vision') else 'emg'
    state=load(args[role+'_checkpoint'])['state_dict']
    if damage=='vision_incomplete':state=dict(state);state.pop('model.backbone.patch_embed.proj.weight')
    elif damage=='vision_nonfinite':
        state=dict(state);key='model.head.0.weight';state[key]=state[key].clone();state[key][0,0]=float('nan')
    else:state={k:v for k,v in state.items() if not k.startswith('model.decoder.')}
    path=tmp_path/'damaged.pt';torch.save({'state_dict':state},path);args[role+'_checkpoint']=path;args[role+'_sha256']=sha(path)
    with pytest.raises(ValueError):api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),**args)


@pytest.mark.parametrize('damage',['identity','source','initializer_identity','empty_transfer','key','shape','dtype','nonfinite','config'])
def test_strict_restore_rejects_identity_or_complete_state_damage_before_factory(fusion_packet,monkeypatch,damage):
    torch=pytest.importorskip('torch');from egoemg_repro.fusion_literal import compose_fusion
    from egoemg_repro.classic_emg import _hash
    data=load(fusion_packet);state=dict(data['state_dict']);identity=copy.deepcopy(data['identity']);cfg=compose_fusion(UPSTREAM);key=next(iter(state))
    if damage=='identity':identity['sha256']='0'*64
    elif damage=='source':identity['sources_sha256']['egoemg_repro/fusion_literal_runtime.py']='0'*64;identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    elif damage=='initializer_identity':identity.pop('initialization');identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    elif damage=='empty_transfer':identity['initialization']['emg_transfer']['matched']=[];identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    elif damage=='key':state.pop(key)
    elif damage=='shape':state[key]=state[key].reshape(-1)
    elif damage=='dtype':state[key]=state[key].to(torch.float16)
    elif damage=='nonfinite':state[key]=state[key].clone();state[key].reshape(-1)[0]=float('nan')
    else:cfg.module.decoder.dropout=.1
    import egoemg.train as native
    def forbidden(*args,**kwargs):raise AssertionError('damaged restore must fail before native factory')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    with pytest.raises(ValueError):api().build_fusion_runtime(UPSTREAM,cfg,state=state,expected_initialization=identity)


def test_self_contained_restore_in_fresh_process_has_no_initializer_paths_or_network(fusion_packet):
    pytest.importorskip('torch')
    code='''import sys,torch,requests,json
from pathlib import Path
from egoemg_repro.fusion_literal import compose_fusion
from egoemg_repro.fusion_literal_runtime import build_fusion_runtime
torch.set_num_threads(2)
def forbidden(*args,**kwargs):raise AssertionError('self-contained restore must not download')
requests.adapters.HTTPAdapter.send=forbidden
d=torch.load(sys.argv[2],map_location='cpu',weights_only=False,mmap=True)
p=build_fusion_runtime(Path(sys.argv[1]),compose_fusion(Path(sys.argv[1])),state=d['state_dict'],expected_initialization=d['identity'])
p.module.eval()
with torch.no_grad():actual=p.module(d['batch'])[0]
assert torch.equal(actual,d['prediction']) and not torch.cuda.is_initialized()
print(json.dumps({'fresh_process_prediction_exact':True,'cuda_initialized':False,'mode':p.mode}))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-c',code,str(UPSTREAM),str(fusion_packet)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stderr[-4000:]
    assert '"fresh_process_prediction_exact": true' in result.stdout


def test_guard_checks_initializer_stability_but_not_learned_parameter_bytes(branch_files,tmp_path):
    torch=pytest.importorskip('torch');from egoemg_repro.fusion_literal import compose_fusion
    args=dict(branch_files);path=tmp_path/'owned-vision.pt';shutil.copyfile(args['vision_checkpoint'],path);args['vision_checkpoint']=path
    p=api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),**args)
    with torch.no_grad():next(p.module.parameters()).add_(.001)
    p.check()
    with path.open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError):p.check()


@pytest.mark.parametrize('damage',['stripped_audit','source_key','source_dtype','native_prefix','source_overlap',
    'skip_reason','skip_shape','missing_match','missing_skip','missing_inventory','normalized_duplicate'])
def test_resealed_transfer_receipt_requires_complete_native_source_inventory(fusion_packet,branch_files,monkeypatch,damage):
    pytest.importorskip('torch')
    from egoemg_repro.fusion_literal import compose_fusion
    from egoemg_repro.classic_emg import _hash
    from egoemg_repro.vision_literal_runtime import _schema,vision_state_sha256
    data=load(fusion_packet);identity=copy.deepcopy(data['identity']);init=identity['initialization'];transfer=init['emg_transfer']
    if damage=='stripped_audit':
        for item in transfer['matched']:item.pop('source_key');item.pop('source_dtype')
        transfer['skipped']=[{'reason':x['reason']} for x in transfer['skipped']]
    elif damage=='source_key':transfer['matched'][0]['source_key']='model.unrecorded_tensor'
    elif damage=='source_dtype':transfer['matched'][0].pop('source_dtype')
    elif damage=='native_prefix':
        item=next(x for x in transfer['skipped'] if x['reason']=='not_backbone_prefix' and x['source_shape']==x['target_shape'])
        transfer['skipped'].remove(item);item.pop('reason');item.pop('target_shape')
        target='model.'+item['target_key'];item['target_dtype']=identity['model_schema'][target]['dtype']
        item['loaded_sha256']=vision_state_sha256({item['target_key']:data['state_dict'][target]});transfer['matched'].append(item)
    elif damage=='source_overlap':transfer['skipped'].append(dict(transfer['matched'][0],reason='missing_target'))
    elif damage=='skip_reason':transfer['skipped'][0]['reason']='missing_target'
    elif damage=='skip_shape':transfer['skipped'][0]['target_shape']=[1]
    elif damage=='missing_match':transfer['matched'].pop(0)
    elif damage=='missing_skip':transfer['skipped'].pop(0)
    elif damage=='missing_inventory':init.pop('emg_source_schema',None)
    elif damage=='normalized_duplicate':
        inventory=init.setdefault('emg_source_schema',_schema(load(branch_files['emg_checkpoint'])['state_dict']))
        key=next(iter(inventory));inventory[key.removeprefix('model.')]=copy.deepcopy(inventory[key])
    identity['sha256']=_hash({k:v for k,v in identity.items() if k!='sha256'})
    import egoemg.train as native
    def forbidden(*args,**kwargs):raise AssertionError('malformed complete audit must reject before factory')
    monkeypatch.setattr(native,'make_lightning_module',forbidden)
    with pytest.raises(ValueError):api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),
        state=data['state_dict'],expected_initialization=identity)


@pytest.mark.parametrize('when',['after_setup','inside_factory'])
@pytest.mark.parametrize('mutation',['remove','replace'])
def test_initializer_disappearance_or_replacement_never_uses_pretrained_fallback(branch_files,tmp_path,monkeypatch,when,mutation):
    pytest.importorskip('torch');import timm,requests
    from egoemg_repro.fusion_literal import compose_fusion
    import egoemg.train as native
    import egoemg._train_utils as native_utils
    args=dict(branch_files);path=tmp_path/'owned-vision.pt';replacement=tmp_path/'replacement.pt'
    shutil.copyfile(args['vision_checkpoint'],path);shutil.copyfile(path,replacement);args['vision_checkpoint']=path
    original_setup=native_utils.setup_runtime;original_factory=native.make_lightning_module;original_create=timm.create_model
    factories=[];pretrained=[]
    def mutate():
        if mutation=='remove':path.unlink()
        else:replacement.replace(path)
    def setup(config):
        original_setup(config)
        if when=='after_setup':mutate()
    def factory(config):
        factories.append(True)
        if when=='inside_factory':mutate()
        return original_factory(config)
    def create(*args,**kwargs):
        pretrained.append(kwargs.get('pretrained'))
        assert kwargs.get('pretrained') is False,'admitted local initializer must never trigger pretrained fallback'
        return original_create(*args,**kwargs)
    def no_network(*args,**kwargs):raise AssertionError('construction race must not cause network requests')
    monkeypatch.delenv('EGOEMG_NO_PRETRAINED_DOWNLOAD',raising=False)
    monkeypatch.setattr(native_utils,'setup_runtime',setup);monkeypatch.setattr(native,'make_lightning_module',factory)
    monkeypatch.setattr(timm,'create_model',create);monkeypatch.setattr(requests.adapters.HTTPAdapter,'send',no_network)
    with pytest.raises(ValueError):api().build_fusion_runtime(UPSTREAM,compose_fusion(UPSTREAM),**args)
    if when=='after_setup':assert not factories and not pretrained
    else:assert factories==[True] and pretrained==[False]
    assert 'EGOEMG_NO_PRETRAINED_DOWNLOAD' not in os.environ
