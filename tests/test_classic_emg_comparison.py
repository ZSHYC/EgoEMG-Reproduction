"""Prospective bounded comparisons; synthetic CPU fixtures never qualify CUDA."""
import copy

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning')
from egoemg_repro import classic_emg_formal as f


def runtime():
    return dict(torch_build='2.2.2+cu118',runtime_profile=f.cuda.BACKEND_PROFILE,
        versions={'pytorch-lightning':'2.5.0.post0'},
        torch_files_sha256={'utils/data/dataloader.py':
            'f9e77a4b0a0827cd318cff46c8e8f65eb718c80c8b6a5a7a4db386417610a4dc'})


def prediction_pair(x,y):
    record=dict(input_sha256='a'*64,target_sha256='b'*64,mask_sha256='c'*64,
        prediction=torch.tensor(x,dtype=torch.float32),model_sha256='d'*64,
        optimizer_sha256='e'*64,scaler_sha256='f'*64)
    a=dict(before=copy.deepcopy(record),after=copy.deepcopy(record),
        execution=dict(executions=1,step_before=2,step_after=3))
    b=copy.deepcopy(a);b['after']['prediction']=torch.tensor(y,dtype=torch.float32)
    return a,b


def test_fixed_eval_copy_rule_accepts_inclusive_boundary_and_measures_inequality():
    eps=2**-10;a,b=prediction_pair([1.],[1.-2*eps])
    result=f._compare_witness(a,b,profile='canonical_fp16_eval_copy_v1',initial=True)
    assert result['after']['measured_equal'] is False
    assert result['after']['prediction_max_abs_difference']==2*eps
    assert result['after']['max_threshold_ratio']==1.
    assert result['after']['model_equal'] is True


@pytest.mark.parametrize('x,y', [([0.,-1e-5,1000.],[0.,1e-5,1000.5]),
    ([1.,-.5],[1.,-.5]),([-1.,1.],[-1.+2**-9,1.-2**-9])])
def test_fixed_rule_is_symmetric_and_elementwise_at_zero_sign_and_large_values(x,y):
    a,b=prediction_pair(x,y)
    ab=f._compare_witness(a,b,profile='canonical_fp16_eval_copy_v1')
    ba=f._compare_witness(b,a,profile='canonical_fp16_eval_copy_v1')
    assert ab['after']['prediction_max_abs_difference']==ba['after']['prediction_max_abs_difference']
    assert ab['after']['max_threshold_ratio']==ba['after']['max_threshold_ratio']


@pytest.mark.parametrize('damage',['over','nan','inf','dtype','shape','scaler','input','initial_model','profile'])
def test_rule_never_hides_single_bad_element_or_strict_initial_state(damage):
    a,b=prediction_pair([1.,1.],[1.,1.]);profile='canonical_fp16_eval_copy_v1'
    if damage=='over':
        b['after']['prediction'][1]=torch.nextafter(torch.tensor(1.-2**-9),torch.tensor(-float('inf')))
    elif damage in ('nan','inf'):b['after']['prediction'][1]=float(damage)
    elif damage=='dtype':b['after']['prediction']=b['after']['prediction'].half()
    elif damage=='shape':b['after']['prediction']=b['after']['prediction'].reshape(1,2)
    elif damage=='scaler':b['after']['scaler_sha256']='0'*64
    elif damage=='input':b['after']['input_sha256']='0'*64
    elif damage=='initial_model':b['before']['model_sha256']='0'*64
    else:profile='caller-chosen-tolerance'
    with pytest.raises(ValueError):f._compare_witness(a,b,profile=profile,initial=True)


def test_cross_run_model_hashes_are_measurements_but_exact_default_still_rejects():
    a,b=prediction_pair([1.],[1.]);b['after']['model_sha256']='0'*64
    b['after']['optimizer_sha256']='1'*64
    result=f._compare_witness(a,b,profile='canonical_fp16_eval_copy_v1',initial=True)
    assert result['after']['model_equal'] is False and result['after']['optimizer_equal'] is False
    with pytest.raises(ValueError):f._compare_witness(a,b)


def batch(offset):
    return dict(emg=torch.full((2,8,3),float(offset)),joint_angles=torch.ones(2,22,3),
        label_valid_mask=torch.ones(2,3,dtype=torch.bool),classic_sample_index=torch.arange(2)+offset)


def stream(epochs):
    value=f.EngineeringInputs()
    for epoch in epochs:
        for role in ('train','val'):
            for index in range(2):value.record(role,epoch,index,batch(2*index))
    return value.value()


def test_every_consumed_engineering_batch_compares_exactly():
    a,p,b=stream([0,1]),stream([0]),stream([1])
    result=f._compare_engineering_inputs(a,p,b,counts={'train':4,'val':4},batch_size=2)
    assert result['train_batches']==4 and result['validation_batches']==4
    assert result['all_consumed_inputs_equal'] is True


@pytest.mark.parametrize('role',['train','val'])
@pytest.mark.parametrize('field',['emg','joint_angles','label_valid_mask','classic_sample_index'])
def test_later_batch_change_rejects_even_with_unchanged_coverage_counters_and_rng(role,field):
    a,p,b=stream([0,1]),stream([0]),f.EngineeringInputs()
    for r in ('train','val'):
        for index in range(2):
            v=batch(2*index)
            if r==role and index==1:
                if field=='label_valid_mask':v[field][0,0]=False
                elif field=='classic_sample_index':v[field]=v[field].flip(0)
                else:v[field].reshape(-1)[-1]+=1
            b.record(r,1,index,v)
    with pytest.raises(ValueError,match='input|batch|stream'):
        f._compare_engineering_inputs(a,p,b.value(),counts={'train':4,'val':4},batch_size=2)


@pytest.mark.parametrize('damage',['missing','duplicate','boundary','epoch'])
def test_input_receipt_requires_all_expected_roles_epochs_and_batch_boundaries(damage):
    a,p,b=stream([0,1]),stream([0]),stream([1])
    if damage=='missing':b['records'].pop()
    elif damage=='duplicate':b['records'].append(copy.deepcopy(b['records'][-1]))
    elif damage=='boundary':b['records'][-1]['samples']=1
    else:b['records'][-1]['epoch']=0
    with pytest.raises(ValueError):f._compare_engineering_inputs(a,p,b,counts={'train':4,'val':4},batch_size=2)


def rng_pair():
    generators={name:torch.Generator().manual_seed(i) for i,name in enumerate(('train_loader','train_sampler','val_loader'))}
    a=f.storage._rng(generators);a['cuda']=[torch.Generator().get_state()]
    b=copy.deepcopy(a);g=torch.Generator();g.set_state(a['generators']['train_loader'])
    torch.empty((),dtype=torch.int64).random_(generator=g);b['generators']['train_loader']=g.get_state()
    return a,b


def test_only_characterized_one_draw_loader_offset_is_accepted_without_mutation():
    a,b=rng_pair();before=f.storage.state_digest((a,b))
    result=f._compare_final_rng(a,b,runtime=runtime(),policy=f.formal_policy(variant='vemg2pose',engineering=True),
        profile=f.engineering_comparison_profile(runtime()))
    assert result['resumed_train_loader_additional_int64_draws']==1
    assert result['measured_full_rng_equal'] is False and result['other_streams_equal'] is True
    assert f.storage.state_digest((a,b))==before


@pytest.mark.parametrize('damage',['two_draws','reverse','other_stream','workers','rank','hash','runtime'])
def test_loader_exception_is_not_an_arbitrary_rng_or_framework_tolerance(damage):
    a,b=rng_pair();r=runtime();policy=f.formal_policy(variant='vemg2pose',engineering=True)
    profile=f.engineering_comparison_profile(r)
    if damage=='two_draws':
        g=torch.Generator();g.set_state(b['generators']['train_loader'])
        torch.empty((),dtype=torch.int64).random_(generator=g);b['generators']['train_loader']=g.get_state()
    elif damage=='reverse':a,b=b,a
    elif damage=='other_stream':b['generators']['val_loader']=torch.Generator().manual_seed(999).get_state()
    elif damage=='workers':policy['num_workers']=1
    elif damage=='rank':policy['world_size']=2
    elif damage=='hash':profile['loader_sources_sha256']['torch/utils/data/dataloader.py']='0'*64
    else:r['extra_runtime_change']=True
    with pytest.raises(ValueError):f._compare_final_rng(a,b,runtime=r,policy=policy,profile=profile)


def final_states():
    return dict(state_dict={'weight':torch.tensor([1.]),'num_batches_tracked':torch.tensor(2)},
        optimizer_states=[dict(param_groups=[{'lr':.001,'params':[0]}],state={0:{'step':torch.tensor(4.),
            'exp_avg':torch.tensor([.1]),'exp_avg_sq':torch.tensor([.01])}})],
        lr_schedulers=[{'T_max':100,'last_epoch':2}],MixedPrecision={'scale':65536.,'_growth_tracker':4})


def test_final_float_drift_is_measured_without_claiming_trajectory_equivalence():
    a=final_states();b=copy.deepcopy(a);b['state_dict']['weight']+=.01
    b['optimizer_states'][0]['state'][0]['exp_avg']+=.001
    result=f._compare_final_states(a,b)
    assert result['measured_final_states_equal'] is False
    assert result['model']['max_abs_difference']>0 and result['adamw_moments']['max_abs_difference']>0
    assert result['optimizer_steps_equal'] is True and result['scheduler_equal'] is True


@pytest.mark.parametrize('damage',['counter','step','group','scheduler','scaler','nan','dtype','shape'])
def test_final_drift_does_not_relax_counters_groups_or_state_schema(damage):
    a=final_states();b=copy.deepcopy(a)
    if damage=='counter':b['state_dict']['num_batches_tracked']+=1
    elif damage=='step':b['optimizer_states'][0]['state'][0]['step']+=1
    elif damage=='group':b['optimizer_states'][0]['param_groups'][0]['lr']+=.001
    elif damage=='scheduler':b['lr_schedulers'][0]['last_epoch']+=1
    elif damage=='scaler':b['MixedPrecision']['scale']/=2
    elif damage=='nan':b['optimizer_states'][0]['state'][0]['exp_avg'][0]=float('nan')
    elif damage=='dtype':b['state_dict']['weight']=b['state_dict']['weight'].half()
    else:b['state_dict']['weight']=b['state_dict']['weight'].reshape(1,1)
    with pytest.raises(ValueError):f._compare_final_states(a,b)


@pytest.mark.parametrize('engineering',[True,False])
def test_real_protocol_records_all_batches_only_in_engineering_scope(tmp_path,engineering):
    from types import SimpleNamespace as NS
    prepared=NS(identity={'counts':{'train':4,'val':4}},module=NS(model=torch.nn.Linear(1,1)),
        policy={'scope':'formal_engineering_acceptance' if engineering else 'formal_training',
            'variant':'neuropose','batch_size':2},guard=NS(check=lambda:None))
    observer=f.FormalProtocol(prepared,{},None,tmp_path);observer.counter=NS(begin=lambda:None)
    observer._witness=lambda *args:{}
    trainer=NS(current_epoch=0,precision_plugin=NS(scaler=NS(get_scale=lambda:65536.)))
    observer.on_train_epoch_start(trainer,None)
    for role in ('train','val'):
        for i in range(2):
            value=dict(emg=torch.zeros(2,16,7790),joint_angles=torch.ones(2,22,7790),
                label_valid_mask=torch.ones(2,7790,dtype=torch.bool),classic_sample_index=torch.arange(2)+2*i,
                mask_policy=['official_literal_loader']*2,dataset_name=['egoemg']*2,hand=['left','right'])
            if role=='train':observer.on_train_batch_start(trainer,None,value,i)
            else:observer.on_validation_batch_start(trainer,None,value,i)
    if engineering:
        records=observer.engineering_inputs.value()['records']
        assert [(r['role'],r['batch']) for r in records]==[('train',0),('train',1),('val',0),('val',1)]
        assert records[-1]['fields_sha256']['emg']==f.storage.state_digest(value['emg'])
    else:assert observer.engineering_inputs is None


def test_consumed_input_file_is_bounded_and_bound_to_actual_result_hash(tmp_path):
    import json,hashlib
    value=stream([0]);path=tmp_path/'engineering-inputs.json';path.write_text(json.dumps(value))
    manifest={'policy':{'batch_size':2,'stop_after_epoch':1},'state_identity':{'counts':{'train':4,'val':4}}}
    result={'resume_start':None,'engineering_inputs_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert f._engineering_inputs_file(tmp_path,result,manifest)==value
    changed=copy.deepcopy(value);changed['records'][-1]['fields_sha256']['emg']='0'*64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):f._engineering_inputs_file(tmp_path,result,manifest)


def test_changed_actual_loader_source_requires_new_characterization(monkeypatch):
    sources=f._native_loader_sources();sources['torch/utils/data/sampler.py']='0'*64
    monkeypatch.setattr(f,'_native_loader_sources',lambda:sources)
    with pytest.raises(ValueError):f.engineering_comparison_profile(runtime())
