"""Guarded asset and tensor/group integrity; tiny utilities never qualify a model/GPU."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).parents[1]
UPSTREAM=ROOT/'upstream'


def api():
    try:return importlib.import_module('egoemg_repro.wilor_literal_runtime')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.wilor_literal_runtime':pytest.fail('genuine native WiLoR runtime missing')
        raise


def test_wilor_catalog_pins_actual_source_and_private_asset_content_without_private_paths():
    c=api()._catalog()
    assert c['format']=='native_wilor_fixed_assets_v1' and len(c['source_files_sha256'])==26
    assert c['source_revision']=='fcb911312a38fa8badd30d9656a167485d61b8f9'
    assert c['checkpoint']==dict(revision='99fe3d7acff8104ecca1055df7467709506c2fa6',bytes=2564989533,
        sha256='3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2')
    assert c['mean']==dict(filename='mano_data/mano_mean_params.npz',bytes=1178,
        sha256='efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988')
    assert c['source_files_sha256']['pretrained_models/model_config.yaml']=='64e75699eb7f1234ee16ba1bbb983db35e03f584fe4bd524559432b1efec3d2c'
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in c['source_files_sha256'])


@pytest.mark.parametrize('runtime,checkpoint,restore',[(True,None,False),(None,None,False),({},None,False),
    ('missing',True,False),('missing',None,False),('missing',None,1),('missing','checkpoint',True)])
def test_asset_argument_errors_reject_before_io_or_pickle(runtime,checkpoint,restore):
    with pytest.raises(ValueError):api().admit_wilor_assets(runtime,checkpoint,restore=restore)


@pytest.mark.parametrize('damage',['partial','wrong_bytes','wrong_sha','symlink','directory','bool_bytes'])
def test_actual_asset_reader_rejects_size_hash_and_namespace_damage(tmp_path,damage):
    m=api();p=tmp_path/'asset';p.write_bytes(b'abcdef')
    expected=dict(bytes=6,sha256=hashlib.sha256(b'abcdef').hexdigest())
    if damage=='partial':p.write_bytes(b'abc')
    elif damage=='wrong_bytes':expected['bytes']=7
    elif damage=='wrong_sha':expected['sha256']='0'*64
    elif damage=='symlink':q=tmp_path/'linked';q.symlink_to(p);p=q
    elif damage=='directory':p=tmp_path
    elif damage=='bool_bytes':expected['bytes']=True
    with pytest.raises((ValueError,OSError)):
        m._read_asset(p,expected,m._StatGuard([]))


def test_actual_asset_reader_registers_input_before_read_and_rejects_late_mutation(tmp_path,monkeypatch):
    m=api();p=tmp_path/'asset';p.write_bytes(b'abcdef');guard=m._StatGuard([]);original=m._stream_sha
    def changed(path,g):
        assert Path(path) in g.paths
        value=original(path,g);p.write_bytes(b'abcdeg');return value
    monkeypatch.setattr(m,'_stream_sha',changed)
    with pytest.raises(ValueError):m._read_asset(p,dict(bytes=6,sha256=hashlib.sha256(b'abcdef').hexdigest()),guard)


def test_forged_asset_instance_is_not_a_producer_owned_admission():
    m=api();fake=object.__new__(m.WilorAssets)
    with pytest.raises(ValueError):fake.check()


@pytest.mark.parametrize('assets',[None,True,{},object()])
def test_native_factory_refuses_non_admission_before_source_or_model(assets):
    with pytest.raises(ValueError):api().build_wilor_runtime(ROOT/'absent',{},assets=assets)


@pytest.mark.parametrize('damage',['empty','no_prefix','not_mapping','not_tensor','nonfinite'])
def test_pretrained_prefix_extraction_rejects_silent_random_fallback_and_invalid_tensors(damage):
    torch=pytest.importorskip('torch');m=api();x={'state_dict':{'backbone.weight':torch.ones(2)}}
    if damage=='empty':x={'state_dict':{}}
    elif damage=='no_prefix':x={'state_dict':{'head.weight':torch.ones(2)}}
    elif damage=='not_mapping':x=True
    elif damage=='not_tensor':x={'state_dict':{'backbone.weight':'not a tensor'}}
    elif damage=='nonfinite':x={'state_dict':{'backbone.weight':torch.full((2,),float('nan'))}}
    with pytest.raises(ValueError):m._extract_backbone(x)


def test_pretrained_prefix_extraction_keeps_all_actual_backbone_tensors_and_ignores_mano_head():
    torch=pytest.importorskip('torch');m=api();t=torch.arange(3.)
    x=m._extract_backbone({'state_dict':{'backbone.weight':t,'mano_head.extra':torch.ones(1)}})
    assert list(x)==['weight'] and torch.equal(x['weight'],t)


@pytest.mark.parametrize('damage',['missing','extra','shape','dtype','nonfinite','wrong_value','invalid_layout'])
def test_exact_tensor_gate_rejects_permissive_cast_or_incomplete_load(damage):
    torch=pytest.importorskip('torch');m=api();expected={'weight':torch.arange(3.)};actual={'weight':expected['weight'].clone()}
    if damage=='missing':actual={}
    elif damage=='extra':actual['extra']=torch.ones(1)
    elif damage=='shape':actual['weight']=actual['weight'].reshape(1,3)
    elif damage=='dtype':actual['weight']=actual['weight'].half()
    elif damage=='nonfinite':actual['weight'][1]=float('nan')
    elif damage=='wrong_value':actual['weight'][1]+=1
    elif damage=='invalid_layout':actual['weight']=actual['weight'].to_sparse()
    with pytest.raises(ValueError):m._require_exact_state(actual,expected)


def test_exact_tensor_gate_digests_actual_full_bytes_without_changing_state():
    torch=pytest.importorskip('torch');m=api();x={'weight':torch.arange(3.)};before=x['weight'].clone()
    value=m._require_exact_state(x,{'weight':before.clone()})
    assert isinstance(value,str) and len(value)==64 and torch.equal(x['weight'],before)


@pytest.mark.parametrize('damage',['file','namespace','missing_origin'])
def test_imported_wilor_from_another_root_is_rejected_without_replacing_modules(tmp_path,monkeypatch,damage):
    m=api();runtime=tmp_path/'runtime';(runtime/'wilor').mkdir(parents=True)
    wrong=SimpleNamespace(__file__=str(tmp_path/'other/wilor.py'))
    if damage=='namespace':wrong=SimpleNamespace(__file__=None,__path__=[str(tmp_path/'other/wilor')])
    elif damage=='missing_origin':wrong=SimpleNamespace(__file__=None)
    monkeypatch.setitem(sys.modules,'wilor',wrong)
    with pytest.raises(ValueError):m._check_import_roots(runtime,UPSTREAM)
    assert sys.modules['wilor'] is wrong


def test_imported_namespace_paths_must_all_belong_to_admitted_root(tmp_path,monkeypatch):
    m=api();runtime=tmp_path/'runtime';(runtime/'wilor').mkdir(parents=True)
    # This case admits a fresh namespace, without a previously captured wrapper.
    monkeypatch.delitem(sys.modules,'egoemg.models.modules.wilor_vit',raising=False)
    package=SimpleNamespace(__file__=None,__path__=[str(runtime/'wilor')]);monkeypatch.setitem(sys.modules,'wilor',package)
    m._check_import_roots(runtime,UPSTREAM)
    package.__path__.append(str(tmp_path/'different/wilor'))
    with pytest.raises(ValueError):m._check_import_roots(runtime,UPSTREAM)


def tiny_native_groups():
    torch=pytest.importorskip('torch');outer=torch.nn.Module();outer.model=torch.nn.Module()
    outer.model.backbone=torch.nn.Linear(2,2);outer.model.head=torch.nn.Linear(2,3)
    groups=[dict(name=n,params=list(getattr(outer.model,n).parameters()),lr=1e-5) for n in ('head','backbone')]
    opt=torch.optim.AdamW(groups,lr=1e-5,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=1e-5)
    return torch,outer,opt,sched


@pytest.mark.parametrize('damage',['lr','floor','tmax','group_name','params_order','duplicate','missing_group','betas',
    'eps','weight_decay','initial_lr','last_epoch_bool','stored_lr','wrong_optimizer_type','foreach','frozen_member'])
def test_optimizer_membership_and_constant_lr_contract_refuses_damage(damage):
    torch,module,opt,sched=tiny_native_groups();m=api()
    if damage=='lr':opt.param_groups[0]['lr']=1e-4
    elif damage=='floor':sched.eta_min=1e-6
    elif damage=='tmax':sched.T_max=150
    elif damage=='group_name':opt.param_groups[0]['name']='backbone'
    elif damage=='params_order':opt.param_groups[0]['params'].reverse()
    elif damage=='duplicate':opt.param_groups[1]['params'][0]=opt.param_groups[0]['params'][0]
    elif damage=='missing_group':opt.param_groups.pop()
    elif damage=='betas':opt.param_groups[0]['betas']=(.8,.999)
    elif damage=='eps':opt.param_groups[0]['eps']=1e-7
    elif damage=='weight_decay':opt.param_groups[0]['weight_decay']=.001
    elif damage=='initial_lr':opt.param_groups[0]['initial_lr']=1e-4
    elif damage=='last_epoch_bool':sched.last_epoch=True
    elif damage=='stored_lr':sched._last_lr[1]=1e-4
    elif damage=='wrong_optimizer_type':opt=torch.optim.SGD(module.parameters(),lr=1e-5)
    elif damage=='foreach':opt.param_groups[0]['foreach']=True
    elif damage=='frozen_member':module.model.backbone.weight.requires_grad_(False)
    with pytest.raises(ValueError):m._optimizer_contract(module,opt,sched)


def test_actual_tiny_two_group_scheduler_stays_constant_all_thirty_steps():
    torch,module,opt,sched=tiny_native_groups();m=api()
    for epoch in range(31):
        value=m._optimizer_contract(module,opt,sched)
        assert [g['name'] for g in value['groups']]==['head','backbone']
        assert [g['lr'] for g in value['groups']]==[1e-5,1e-5]
        assert value['scheduler']['T_max']==30 and value['scheduler']['last_epoch']==epoch
        if epoch<30:
            for p in module.parameters():p.grad=torch.zeros_like(p)
            opt.step();sched.step()


def test_asset_module_import_and_argument_rejection_never_import_torch():
    api()
    code="""import sys
from egoemg_repro.wilor_literal_runtime import admit_wilor_assets
try:admit_wilor_assets(True)
except ValueError:pass
else:raise AssertionError('invalid admission accepted')
assert 'torch' not in sys.modules
"""
    x=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert x.returncode==0,x.stderr


def test_imported_wrapper_with_missing_captured_runtime_rejects_as_source_binding_error(tmp_path,monkeypatch):
    m=api();runtime=tmp_path/'runtime';(runtime/'wilor').mkdir(parents=True)
    prior=SimpleNamespace(__file__=str(UPSTREAM/'egoemg/models/modules/wilor_vit.py'),WILOR_PATH=tmp_path/'wrong-runtime')
    monkeypatch.setitem(sys.modules,'egoemg.models.modules.wilor_vit',prior)
    with pytest.raises(ValueError):m._check_import_roots(runtime,UPSTREAM)


@pytest.mark.parametrize('damage',['scientific_flag','paper_flag','extra_permit','scope','seed_type','count_type',
    'schema','initial_sha','asset','runtime'])
def test_restore_metadata_cannot_promote_construction_or_change_bound_lineage(damage):
    # Tiny typed-state verifier fixture only; no asset registry/native factory or permit.
    torch=pytest.importorskip('torch');m=api();catalog=m._catalog();state={'weight':torch.ones(2)}
    protocol={'sha256':'a'*64,'model_expectations':{}};sources={'source':'b'*64};versions={'fixture':'unit-only'}
    identity=dict(format='native_wilor_module_construction_v1',scope='genuine native construction only; not a training or CUDA permit',
        scientific_eligible=False,full_paper_complete=False,protocol_sha256=protocol['sha256'],sources_sha256=sources,
        runtime_versions=versions,model_schema=m._schema(state),parameter_names=['weight'],model_contract={},
        initial_state_sha256='c'*64,initialization=dict(fresh_pretrained_verified=True,retained_tensor_count=404,
            backbone_sha256='d'*64,asset=catalog['checkpoint'],source_revision=catalog['source_revision'],factory_seed=42))
    if damage=='scientific_flag':identity['scientific_eligible']=True
    elif damage=='paper_flag':identity['full_paper_complete']=True
    elif damage=='extra_permit':identity['cuda_qualified']=True
    elif damage=='scope':identity['scope']='full scientific experiment'
    elif damage=='seed_type':identity['initialization']['factory_seed']=True
    elif damage=='count_type':identity['initialization']['retained_tensor_count']=404.
    elif damage=='schema':identity['model_schema']['weight']['dtype']='torch.float16'
    elif damage=='initial_sha':identity['initial_state_sha256']=False
    elif damage=='asset':identity['initialization']['asset']['sha256']='e'*64
    elif damage=='runtime':identity['runtime_versions']={'fixture':'different'}
    identity['sha256']=m._hash(identity)
    with pytest.raises(ValueError):m._restore_identity(identity,protocol,sources,versions,state)


def tiny_original_head():
    torch=pytest.importorskip('torch')
    return SimpleNamespace(head=torch.nn.Sequential(torch.nn.Linear(1280,512),torch.nn.GELU(),
        torch.nn.Dropout(.1),torch.nn.Linear(512,22)),avgpool=torch.nn.AdaptiveAvgPool2d((1,1)),
        out_channels=22,left_context=0,right_context=0)


@pytest.mark.parametrize('damage',['relu','dropout','inplace','pool','width','angles','bias','context','half'])
def test_original_wilor_head_contract_rejects_postconstruction_architecture_or_dtype_mutation(damage):
    torch=pytest.importorskip('torch');m=api();model=tiny_original_head()
    if damage=='relu':model.head[1]=torch.nn.ReLU()
    elif damage=='dropout':model.head[2].p=.2
    elif damage=='inplace':model.head[2].inplace=True
    elif damage=='pool':model.avgpool=torch.nn.AdaptiveAvgPool2d((2,1))
    elif damage=='width':model.head[0]=torch.nn.Linear(1280,513)
    elif damage=='angles':model.head[3]=torch.nn.Linear(512,20)
    elif damage=='bias':model.head[0].bias=None
    elif damage=='context':model.left_context=1
    elif damage=='half':model.head.half()
    with pytest.raises(ValueError):m._head_contract(model)


def test_original_real_667158_parameter_head_and_pool_are_kept_exact():
    pytest.importorskip('torch');m=api();model=tiny_original_head()
    assert sum(p.numel() for p in model.head.parameters())==667158
    assert m._head_contract(model) is None


def test_live_source_guard_keeps_external_input_generation_even_if_bytes_are_restored(tmp_path):
    pytest.importorskip('pytorch_lightning');pytest.importorskip('hydra');m=api()
    from egoemg_repro.wilor_literal import compose_wilor,wilor_protocol
    r=tmp_path/'external';r.mkdir();p=r/'README.md';p.write_bytes(b'original')
    assets=SimpleNamespace(runtime=r,identity={'source_files':{'README.md':{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}})
    guard=m._source_guard(UPSTREAM,wilor_protocol(UPSTREAM,compose_wilor(UPSTREAM)),assets)
    p.write_bytes(b'changed!');p.write_bytes(b'original')
    with pytest.raises(ValueError):guard.check()
