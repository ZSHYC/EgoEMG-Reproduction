"""WiLoR boundaries use ordinary original point files, never a CUDA surrogate."""
import copy
import importlib
import random

import numpy as np
import pytest

from test_vision_literal_cohort import files
from test_vision_literal_data import UPSTREAM


def api():
    try:return importlib.import_module('egoemg_repro.wilor_native_train')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.wilor_native_train':pytest.fail('native WiLoR data/fit boundary missing')
        raise


@pytest.fixture
def inputs(files):
    from egoemg_repro.wilor_literal import compose_wilor
    return (*files[:2],compose_wilor(UPSTREAM))


def admitted(inputs,**kwargs):
    data,crops,cfg=inputs
    return api().admit_wilor_cohort(data,crops,upstream=UPSTREAM,config=cfg,require_full=False,**kwargs)


def test_projection_compares_the_complete_original_data_configuration_without_model_permits(inputs):
    from omegaconf import OmegaConf
    from egoemg_repro.vision_literal import vision_protocol
    from egoemg_repro.wilor_literal import wilor_protocol
    m=api();cfg=inputs[2];companion,value=m._data_projection(UPSTREAM,cfg)
    assert value['format']=='native_wilor_data_projection_v1'
    assert value['wilor_protocol_sha256']==wilor_protocol(UPSTREAM,cfg)['sha256']
    assert value['companion_protocol_sha256']==vision_protocol(UPSTREAM,companion)['sha256']
    assert value['companion_role']=='original vision-S data only; no DINO model/training/CUDA permit'
    for name in value['compared_fields']:
        assert OmegaConf.to_container(cfg,resolve=True)[name]==OmegaConf.to_container(companion,resolve=True)[name]
    assert {'dataset','datamodule','transforms','skip_emg_loading','vision_patch_size','vision_frame_selection'}<=set(value['compared_fields'])
    assert value['scientific_eligible'] is False and value['full_paper_complete'] is False


def test_admission_hashes_real_inputs_and_keeps_rng_and_old_model_factory_unused(inputs,monkeypatch):
    import torch
    from egoemg_repro import wilor_literal_runtime,vision_literal_runtime
    def forbidden(*a,**k):raise AssertionError('data admission cannot construct either model')
    monkeypatch.setattr(wilor_literal_runtime,'build_wilor_runtime',forbidden)
    monkeypatch.setattr(vision_literal_runtime,'build_vision_runtime',forbidden)
    before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    c=admitted(inputs);v=c.identity
    assert v['format']=='native_wilor_data_admission_v1'
    assert v['common_data']['format']=='original_vision_full_content_admission_v1'
    assert v['counts']==dict(train=len(c.datasets['train']),val=4,test=12)
    assert v['unique_counts']=={r:n//2 for r,n in v['counts'].items()}
    assert v['full_payload_content_hashed'] is True and v['registered_full_legacy'] is False
    assert len(v['common_data']['payload_files_sha256'])==12
    assert v['scientific_eligible'] is False and v['full_paper_complete'] is False
    assert torch.equal(torch.get_rng_state(),before[0]) and random.getstate()==before[2]
    assert np.array_equal(np.random.get_state()[1],before[1][1]) and np.random.get_state()[2:]==before[1][2:]
    assert not torch.cuda.is_initialized();c.check()


def test_actual_native_batch18_loader_all_roles_has_identical_original_values_and_rng(inputs):
    import torch
    from egoemg.datamodule import make_data_module
    c=admitted(inputs);loaders=api().native_wilor_loaders(c,scope='cpu_engineering')
    assert set(loaders)=={'train','val','test'}
    dm=make_data_module(inputs[2]);dm.num_workers=dm.eval_num_workers=0;dm.batch_size=1
    for role,loader in loaders.items():
        assert loader.batch_size==1 and loader.num_workers==0 and loader.generator is None and not loader.drop_last
        assert len(loader.dataset)==2
        direct=dm._make_loader(torch.utils.data.Subset(c.datasets[role],range(2)),shuffle=role=='train')
        torch.manual_seed(83);np.random.seed(51);expected=next(iter(direct));rng=torch.get_rng_state().clone()
        torch.manual_seed(83);np.random.seed(51);actual=next(iter(loader))
        for name in ('vision_img','joint_angles','label_valid_mask','vision_sample_index','jitter_offset','vision_repeat_index'):
            assert torch.equal(actual[name],expected[name]),(role,name)
        assert actual['dataset_name']==expected['dataset_name'] and torch.equal(torch.get_rng_state(),rng)
    assert api().wilor_policy(scope='cuda_engineering')['batch_size_by_role']==dict(train=18,val=18,test=18)


def test_missing_crop_keeps_normalized_black_and_native_generated_label_mask(inputs):
    c=admitted(inputs);ds=c.datasets['val'];n=len(ds.windows)
    valid=ds[0];missing=ds[n]
    assert valid['vision_valid_mask'].tolist()==[True] and missing['vision_valid_mask'].tolist()==[False]
    assert missing['label_valid_mask'].tolist()==[True] and missing['joint_angles'].shape==(22,1)
    assert 'angle_valid_mask' not in missing and missing['emg'].shape==(16,0)
    np.testing.assert_allclose(missing['vision_img'][:,0,0],[-.485/.229,-.456/.224,-.406/.225],atol=1e-6)


@pytest.mark.parametrize('kind',['target','crop','metadata','partition'])
def test_changed_real_payload_rejects_before_original_loader_factory(inputs,kind,monkeypatch):
    from egoemg import datamodule
    c=admitted(inputs);data,crops,_=inputs
    p={'target':data/'generated_joint_angles_left.dat','crop':crops/'episode_000000.data.mdb',
       'metadata':data/'metadata.npz','partition':data/'frame_split_id.dat'}[kind]
    with p.open('r+b') as f:f.write(b'changed!')
    def forbidden(*a,**k):raise AssertionError('changed input must reject before loader construction')
    monkeypatch.setattr(datamodule,'make_data_module',forbidden)
    with pytest.raises(ValueError,match='changed|stamp'):api().native_wilor_loaders(c,scope='cpu_engineering')


def test_resealed_wrong_expected_payload_is_recomputed(inputs):
    from egoemg_repro.classic_emg import _hash
    c=admitted(inputs);v=copy.deepcopy(c.identity);v['counts']['val']=100
    v['sha256']=_hash({k:x for k,x in v.items() if k!='sha256'})
    with pytest.raises(ValueError,match='receipt|identity|content'):admitted(inputs,expected=v)


@pytest.mark.parametrize('kind',['repeat','role','mean','std','projection','seal','sources'])
def test_outer_binding_and_actual_reader_fields_cannot_be_resealed_or_mutated(inputs,kind):
    from egoemg_repro.classic_emg import _hash
    c=admitted(inputs)
    if kind=='repeat':c.datasets['val'].repeat=1
    elif kind=='role':c.datasets['val'].role='test'
    elif kind=='mean':c.datasets['val'].views[0].vision_mean[0]=0
    elif kind=='std':c.datasets['val'].views[0].vision_std[0]=1
    elif kind=='projection':c.identity['projection']['companion_role']='full CUDA qualification'
    elif kind=='seal':c.identity['scientific_eligible']=True
    else:c.identity['sources_sha256']['egoemg_repro/wilor_native_train.py']='0'*64
    if kind in ('projection','seal','sources'):c.identity['sha256']=_hash({k:v for k,v in c.identity.items() if k!='sha256'})
    with pytest.raises(ValueError):c.check()


@pytest.mark.parametrize('role',['train','val'])
def test_nonfinite_at_possible_jitter_or_validation_center_is_not_cleaned(inputs,role):
    c=admitted(inputs);ds=c.datasets[role];w=next(w for w in ds.windows if role!='train' or w['jitter_upper']>1)
    i=w['center']+max(0,w['jitter_upper']-1)
    p=inputs[0]/'generated_joint_angles_left.dat'
    a=np.memmap(p,dtype='float32',mode='r+',shape=(9*7790,20));a[i,3]=np.nan;a.flush();del a
    with pytest.raises(ValueError,match='nonfinite'):admitted(inputs)


def test_small_real_data_and_constructed_copy_never_qualify_as_full(inputs):
    import dataclasses
    c=admitted(inputs);m=api()
    with pytest.raises(ValueError,match='full|registered'):c.check(require_full=True)
    with pytest.raises(ValueError,match='full|registered'):m.admit_wilor_cohort(*inputs[:2],upstream=UPSTREAM,config=inputs[2])
    forged=dataclasses.replace(c)
    with pytest.raises(ValueError,match='admitted|registered'):forged.check()
    for scope in ('full','cuda_engineering'):
        with pytest.raises(ValueError,match='full|registered'):m.native_wilor_loaders(c,scope=scope)


@pytest.mark.parametrize('flag',[1,None,'full'])
def test_data_full_requirement_is_strict_bool_before_any_io(flag):
    with pytest.raises(ValueError,match='bool'):api().admit_wilor_cohort(None,None,upstream=None,config=None,require_full=flag)


@pytest.mark.parametrize('scope',['full','cuda_engineering','cpu_engineering'])
def test_exact_wilor_policy_separates_scopes_and_preserves_complete_native_budget(scope):
    p=api().wilor_policy(scope=scope);cpu=scope=='cpu_engineering';full=scope=='full'
    assert p['scope']==scope and p['max_epochs']==(30 if full else 2)
    assert p['precision']==('32-true' if cpu else 'bf16-mixed')
    assert p['batch_size_by_role']=={r:1 if cpu else 18 for r in ('train','val','test')}
    assert p['engineering_train_samples']==(None if full else 2 if cpu else 36)
    assert p['monitor']=='val_mae' and p['monitor_mode']=='min' and p['patience']==30
    assert p['save_top_k']==3 and p['save_last'] is True and p['gradient_clip_val']==1
    assert p['scheduler_T_max']==30 and p['scheduler_eta_min']==1e-5
    assert p['checkpoint_continuation_supported'] is False and p['scientific_eligible'] is False
    assert p['full_counts']==dict(train=202020,val=3152,test=8612)


@pytest.mark.parametrize('scope',[True,None,'wilor','gpu','Full'])
def test_invalid_scope_rejects_without_importing_model(scope):
    with pytest.raises(ValueError,match='scope'):api().wilor_policy(scope=scope)


@pytest.mark.parametrize('qualification',[None,True,{'passed':True}])
def test_full_fit_requires_raw_new_cuda_qualification_before_data_model_or_output(tmp_path,qualification):
    with pytest.raises(ValueError,match='qualification|artifact'):
        api().fit_wilor(tmp_path/'missing-data',tmp_path/'missing-crops',upstream=tmp_path/'missing',
            config=None,assets=None,output=tmp_path/'out',qualification=qualification)
    assert not (tmp_path/'out').exists()


def test_json_passed_summary_cannot_become_a_full_cuda_qualification(tmp_path):
    p=tmp_path/'summary.json';p.write_text('{"passed":true}')
    with pytest.raises(ValueError,match='qualification|artifact'):
        api().fit_wilor(None,None,upstream=None,config=None,assets=None,output=tmp_path/'out',qualification=p)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('value',[True,{},'old-qualification'])
def test_cpu_fit_never_inherits_a_cuda_or_old_model_permit(tmp_path,value):
    with pytest.raises(ValueError,match='CPU|qualification'):
        api().fit_wilor(None,None,upstream=None,config=None,assets=None,output=tmp_path/'out',
            scope='cpu_engineering',qualification=value)
    assert not (tmp_path/'out').exists()


def test_existing_owned_fit_is_rejected_before_data_or_model_access(tmp_path):
    output=tmp_path/'owned';output.mkdir();marker=output/'preserved';marker.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        api().fit_wilor(None,None,upstream=None,config=None,assets=None,output=output,scope='cpu_engineering')
    assert marker.read_bytes()==b'preserve'


@pytest.mark.parametrize('extra',[{'resume':True},{'batch_size':1},{'precision':'32-true'},{'max_epochs':1},{'stop_after_epoch':1}])
def test_caller_cannot_replace_the_recipe_or_enable_continuation(tmp_path,extra):
    with pytest.raises(TypeError):
        api().fit_wilor(None,None,upstream=None,config=None,assets=None,output=tmp_path/'out',**extra)


@pytest.mark.parametrize('reader',['read_wilor_fit','reopen_wilor_best'])
@pytest.mark.parametrize('flag',[1,None,'full'])
def test_fit_read_requirement_is_strict_bool_before_io(reader,flag):
    with pytest.raises(ValueError,match='bool'):getattr(api(),reader)(None,require_full=flag)


def budget_records(policy,n):
    train,val=(202020,3152) if policy['scope']=='full' else ((2,2) if policy['scope']=='cpu_engineering' else (36,3152))
    batches=(train+policy['batch_size_by_role']['train']-1)//policy['batch_size_by_role']['train']
    rows=[dict(epoch=i,optimizer_executions=(i+1)*batches,global_step=(i+1)*batches,
        train_samples=train,train_batches=batches,train_loss=.5,
        validation=dict(samples=val,val_mae=.25)) for i in range(n)]
    return train,val,rows


@pytest.mark.parametrize('scope',['full','cpu_engineering','cuda_engineering'])
def test_budget_predicate_requires_the_entire_original_or_engineering_budget(scope):
    m=api();p=m.wilor_policy(scope=scope);train,val,rows=budget_records(p,p['max_epochs'])
    assert m.wilor_stage_complete(p,rows,train_samples=train,val_samples=val,interrupted=False) is True
    assert m.wilor_stage_complete(p,rows[:-1],train_samples=train,val_samples=val,interrupted=False) is False
    assert m.wilor_stage_complete(p,rows,train_samples=train,val_samples=val,interrupted=True) is False


@pytest.mark.parametrize('damage',['epoch','counter','global_step','train','validation','batches','train_nonfinite','val_nonfinite',
    'bool_counter','policy_budget','policy_precision','policy_batch','extra_epoch'])
def test_counter_coverage_nonfinite_or_resealed_policy_changes_cannot_complete(damage):
    from egoemg_repro.classic_emg import _hash
    m=api();p=m.wilor_policy();train,val,rows=budget_records(p,30)
    if damage=='epoch':rows[-1]['epoch']=True
    elif damage=='counter':rows[-1]['optimizer_executions']-=1
    elif damage=='global_step':rows[-1]['global_step']+=1
    elif damage=='train':rows[-1]['train_samples']-=1
    elif damage=='validation':rows[-1]['validation']['samples']-=1
    elif damage=='batches':rows[-1]['train_batches']-=1
    elif damage=='train_nonfinite':rows[-1]['train_loss']=float('inf')
    elif damage=='val_nonfinite':rows[-1]['validation']['val_mae']=float('nan')
    elif damage=='bool_counter':rows[-1]['optimizer_executions']=True
    elif damage=='policy_budget':p['max_epochs']=29
    elif damage=='policy_precision':p['precision']='32-true'
    elif damage=='policy_batch':p['batch_size_by_role']['train']=1
    else:rows.append(copy.deepcopy(rows[-1]))
    p['sha256']=_hash({k:v for k,v in p.items() if k!='sha256'})
    assert m.wilor_stage_complete(p,rows,train_samples=train,val_samples=val,interrupted=False) is False


def test_cpu_rng_snapshot_does_not_query_visible_cuda(monkeypatch):
    torch=pytest.importorskip('torch')
    def forbidden(*a,**k):raise AssertionError('CPU scope cannot query CUDA')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden);monkeypatch.setattr(torch.cuda,'get_rng_state_all',forbidden)
    value=api()._rng_packet(cpu=True)
    assert value['cuda']==[] and value['generators']=={} and value['torch'].dtype==torch.uint8
    api()._validate_rng(value,cpu=True)


@pytest.mark.parametrize('damage',['missing','empty_cpu','bad_cpu_dtype','fake_cuda','python','numpy','extra_generator'])
def test_corrupted_global_native_rng_is_rejected_without_restoring_it(damage):
    torch=pytest.importorskip('torch');m=api();v=m._rng_packet(cpu=True);before=torch.get_rng_state().clone()
    if damage=='missing':del v['numpy']
    elif damage=='empty_cpu':v['torch']=torch.empty(0,dtype=torch.uint8)
    elif damage=='bad_cpu_dtype':v['torch']=v['torch'].to(torch.int64)
    elif damage=='fake_cuda':v['cuda']=[v['torch']]
    elif damage=='python':v['python']=(1,2,3)
    elif damage=='numpy':v['numpy']=('bad',)
    else:v['generators']={'invented':v['torch']}
    with pytest.raises(ValueError,match='RNG|rng'):m._validate_rng(v,cpu=True)
    assert torch.equal(torch.get_rng_state(),before)


def test_fit_batch_checks_actual_native_indices_center_mask_shapes_and_full_input_finiteness(inputs):
    import torch
    m=api();c=admitted(inputs);p=m.wilor_policy(scope='cpu_engineering')
    batches=list(m.native_wilor_loaders(c,scope='cpu_engineering')['val']);seen=set()
    assert m._batch(c,p,batches[0],'val',seen,2)==1 and seen=={0}
    assert m._batch(c,p,batches[1],'val',seen,2)==1 and seen=={0,1}
    for damage in ('index_dtype','out_of_range','wrong_order','image_shape','image_nonfinite','target_nonfinite',
        'mask_dtype','vision_mask_dtype','emg_shape','center','hand','repeat','unique','window','episode','dataset'):
        b=copy.deepcopy(batches[0])
        if damage=='index_dtype':b['vision_sample_index']=b['vision_sample_index'].float()
        elif damage=='out_of_range':b['vision_sample_index'][0]=2
        elif damage=='wrong_order':b=copy.deepcopy(batches[1])
        elif damage=='image_shape':b['vision_img']=b['vision_img'][:,:,:128,:]
        elif damage=='image_nonfinite':b['vision_img'][0,0,0,0]=float('nan')
        elif damage=='target_nonfinite':b['joint_angles'][0,0,0]=float('inf')
        elif damage=='mask_dtype':b['label_valid_mask']=b['label_valid_mask'].float()
        elif damage=='vision_mask_dtype':b['vision_valid_mask']=b['vision_valid_mask'].float()
        elif damage=='emg_shape':b['emg']=torch.zeros(1,8,0)
        elif damage=='center':b['center'][0]+=1
        elif damage=='hand':b['target_hand_index'][0]=1
        elif damage=='repeat':b['vision_repeat_index'][0]=1
        elif damage=='unique':b['vision_unique_sample_index'][0]=100
        elif damage=='window':b['vision_window_index'][0]=100
        elif damage=='episode':b['episode'][0]=100
        else:b['dataset_name']=['other']
        with pytest.raises(ValueError):m._batch(c,p,b,'val',set(),2)
    with pytest.raises(ValueError):m._batch(c,p,batches[0],'val',{0},2)


@pytest.mark.parametrize('checkpoint',[None,True,{}, {'native_wilor':{}}, {'state_dict':{}}])
def test_incomplete_native_checkpoint_never_falls_back_to_a_summary(checkpoint):
    with pytest.raises(ValueError,match='checkpoint|native|complete'):
        api()._validate_checkpoint(checkpoint,{},[])


def test_fit_source_generation_is_captured_before_content_hashing(tmp_path,monkeypatch):
    pytest.importorskip('pytorch_lightning');pytest.importorskip('timm')
    m=api();paths=list(m._sources());original_sources=m._sources
    # Use ordinary small source files; no positive model/data/CUDA identity.
    for name in paths:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original source\n')
    monkeypatch.setattr(m,'ROOT',tmp_path)
    def changed_and_restored_during_hash():
        value=original_sources();p=tmp_path/'egoemg_repro/wilor_native_train.py';raw=p.read_bytes()
        p.write_bytes(raw+b'# transient change\n');p.write_bytes(raw)
        return value
    monkeypatch.setattr(m,'_sources',changed_and_restored_during_hash)
    with pytest.raises(ValueError,match='changed|stamp'):m._capture_fit_sources()
