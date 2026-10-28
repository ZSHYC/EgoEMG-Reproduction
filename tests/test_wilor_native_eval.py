"""Selected WiLoR native evaluation/replay boundaries; no fake CUDA acceptance."""
import copy
import hashlib
import importlib
import json

import pytest


def api():
    try:return importlib.import_module('egoemg_repro.wilor_native_eval')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.wilor_native_eval':pytest.fail('native WiLoR evaluation/replay boundary missing')
        raise


@pytest.mark.parametrize('scope,role,n,expected',[
    ('cpu_engineering','val',4,[0,1,2,3]),('cpu_engineering','test',12,[0,3,6,9]),
    ('cuda_engineering','val',3152,None),('cuda_engineering','test',8612,None),
    ('full','val',3152,None),('full','test',8612,None)])
def test_evaluation_layout_preserves_original_source_indices_and_both_native_repetitions(scope,role,n,expected):
    v=api()._evaluation_layout(scope,role,n)
    assert v['source_samples']==n and v['scope']==scope and v['role']==role
    assert v['source_indices']==(list(range(n)) if expected is None else expected)
    assert v['samples']==(4 if scope=='cpu_engineering' else n)
    assert v['batch_size']==(1 if scope=='cpu_engineering' else 18)
    assert v['scientific_eligible'] is False and v['full_paper_complete'] is False
    if scope!='cpu_engineering':assert v['samples']%v['batch_size']==(2 if role=='val' else 8)


@pytest.mark.parametrize('scope,role,n',[(True,'val',4),('cpu_engineering','train',4),('bad','val',4),
    ('cpu_engineering','val',True),('cpu_engineering','val',0),('cpu_engineering','val',2),
    ('cpu_engineering','val',5),('full','val',4),('cuda_engineering','test',12)])
def test_unknown_partial_or_noninteger_evaluation_population_is_rejected(scope,role,n):
    with pytest.raises(ValueError):api()._evaluation_layout(scope,role,n)


def parts(*,samples=4,batch_size=4,source_samples=None,role='test'):
    torch=pytest.importorskip('torch');source_samples=source_samples or samples
    source=list(range(samples)) if source_samples==samples else [0,source_samples//4,source_samples//2,3*source_samples//4]
    result=[]
    for start in range(0,samples,batch_size):
        n=min(batch_size,samples-start);indices=torch.arange(start,start+n);src=torch.tensor(source[start:start+n]);unique=src%(source_samples//2)
        hashes=[hashlib.sha256(('input-'+str(int(i))).encode()).hexdigest() for i in unique]
        result.append(dict(pred=torch.ones(n,22,1),target=torch.zeros(n,22,1),mask=torch.ones(n,1,dtype=torch.bool),
            index=indices,unique=indices%(samples//2),repeat=indices//(samples//2),
            source_index=src,source_unique=unique,source_repeat=src//(source_samples//2),source_window=src%(source_samples//4),
            vision=torch.tensor([[int(i)%(source_samples//2)<source_samples//4] for i in src],dtype=torch.bool),
            hand=['left' if int(i)%(source_samples//2)<source_samples//4 else 'right' for i in src],
            split=['train' if role=='val' else 'user']*n,user=['private-person']*n,center=torch.tensor([3895]*n),
            episode=torch.zeros(n,dtype=torch.int64),input_sha256='a'*64,sample_input_sha256=hashes))
    return result,source


@pytest.mark.parametrize('source_samples',[4,12])
def test_original_generated_mask_missing_crop_and_source_index_mapping_are_preserved(source_samples):
    m=api();p,indices=parts(source_samples=source_samples)
    v=m.wilor_statistics(p,samples=4,batch_size=4,role='test',source_samples=source_samples,source_indices=indices)
    assert v['format']=='native_wilor_center_statistics_v1'
    assert v['consumed_samples']==4 and v['unique_samples']==2 and v['duplicate_consumptions']==2
    assert v['source_population_samples']==source_samples and v['consumed']['samples']==dict(all=4,vision_valid=2,missing_crop=2)
    assert v['consumed']['all']['overall']['valid_angle_count']==88 and v['unique_first_repeat']['all']['overall']['valid_angle_count']==44
    assert len(v['consumed']['all']['by_joint'])==22
    assert sum(j['valid_angle_count'] for j in v['consumed']['all']['by_joint'])==88
    assert v['native_batch_weighted_mae_rad']==1. and v['scientific_eligible'] is False
    assert 'private-person' not in json.dumps(v) and list(v['consumed']['all']['by_user'])==['user_000']


def test_native_batch_mean_and_pooled_valid_angle_denominators_are_distinct_for_paired_repeats():
    torch=pytest.importorskip('torch');m=api();p,indices=parts(samples=8,batch_size=3)
    errors=torch.tensor([1.,100.,3.,9.]*2);valid=torch.tensor([True,False,True,True]*2)
    for part in p:
        i=part['index'];part['pred']=errors[i].reshape(-1,1,1).expand(-1,22,1).clone();part['mask']=valid[i].reshape(-1,1)
    v=m.wilor_statistics(p,samples=8,batch_size=3,role='test',source_samples=8,source_indices=indices)
    assert v['batch_sizes']==[3,3,2] and v['native_batch_weighted_mae_rad']==4.125
    assert v['consumed']['all']['overall']['absolute_error_sum_rad']==572.
    assert v['consumed']['all']['overall']['valid_angle_count']==132


@pytest.mark.parametrize('damage',['source_index','source_unique','source_repeat','source_window','index_dtype','pred_dtype',
    'target_nonfinite','hand','repeat_target','repeat_mask','repeat_vision','repeat_user','repeat_input','input_digest','batch_digest','duplicate_index'])
def test_resealed_source_metadata_wrong_repeat_pair_or_nonfinite_values_cannot_be_aggregated(damage):
    torch=pytest.importorskip('torch');m=api();p,indices=parts(source_samples=12);v=p[0]
    if damage=='source_index':v['source_index'][1]=1
    elif damage=='source_unique':v['source_unique'][1]=1
    elif damage=='source_repeat':v['source_repeat'][3]=0
    elif damage=='source_window':v['source_window'][1]=1
    elif damage=='index_dtype':v['source_index']=v['source_index'].float()
    elif damage=='pred_dtype':v['pred']=v['pred'].half()
    elif damage=='target_nonfinite':v['target'][0,0,0]=float('nan')
    elif damage=='hand':v['hand'][1]='left'
    elif damage=='repeat_target':v['target'][2,0,0]=1.
    elif damage=='repeat_mask':v['mask'][2,0]=False
    elif damage=='repeat_vision':v['vision'][2,0]=False
    elif damage=='repeat_user':v['user'][2]='other-person'
    elif damage=='repeat_input':v['sample_input_sha256'][2]='b'*64
    elif damage=='input_digest':v['sample_input_sha256'][0]='x'*64
    elif damage=='batch_digest':v['input_sha256']='x'*64
    else:v['index'][3]=2
    with pytest.raises(ValueError):m.wilor_statistics(p,samples=4,batch_size=4,role='test',source_samples=12,source_indices=indices)


@pytest.mark.parametrize('role,kind,reference,profile',[('train','native',None,None),('test','bad',None,None),
    ('test','native',None,None),('val','replay',None,None),('val','native','unexpected',None),('val','native',None,'unexpected')])
def test_invalid_pass_or_missing_pretest_freeze_is_rejected_before_fit_model_or_output(tmp_path,role,kind,reference,profile):
    with pytest.raises(ValueError):api().run_wilor_pass(None,tmp_path/'out',role=role,kind=kind,reference=reference,profile=profile,require_full=False)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('flag',[1,None,'full'])
def test_prediction_full_requirement_is_strict_boolean_before_any_io(tmp_path,flag):
    with pytest.raises(ValueError,match='bool'):api().run_wilor_pass(None,tmp_path/'out',role='val',kind='native',require_full=flag)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('value',[True,None,{'passed':True},{'format':'original_vision_cuda_qualification_v1','passed':True}])
def test_summary_or_old_dino_boolean_is_never_a_raw_wilor_cuda_qualification(value):
    with pytest.raises(ValueError,match='qualification|artifact'):api().read_wilor_qualification(value)


def cpu_header():
    device=dict(device_type='cpu',input_device='cpu',module_device='cpu',prediction_device='cpu',cuda_initialized=False)
    return dict(scope='cpu_engineering',kind='native',process=dict(pid=123456,start_ticks='12345',uid=1000),
        started_ns=100,finished_ns=200,statistics=dict(batch_sizes=[1,1,1,1]),
        forward_devices=[copy.deepcopy(device) for _ in range(4)],post_pass_module_device='cpu',
        execution_device=dict(accelerator='cpu',visible_device='',cuda_initialized=False))


@pytest.mark.parametrize('damage',['pid_bool','pid_missing','ticks','uid','start_bool','finish_bool','reverse_time',
    'before_fit','forward_count','forward_input','forward_prediction','forward_cuda','teardown','claimed_gpu'])
def test_execution_headers_reject_malformed_owner_time_or_inference_device_receipts(damage):
    v=cpu_header()
    if damage=='pid_bool':v['process']['pid']=True
    elif damage=='pid_missing':del v['process']['pid']
    elif damage=='ticks':v['process']['start_ticks']='unknown'
    elif damage=='uid':v['process']['uid']=-1
    elif damage=='start_bool':v['started_ns']=True
    elif damage=='finish_bool':v['finished_ns']=True
    elif damage=='reverse_time':v['finished_ns']=99
    elif damage=='before_fit':v['started_ns']=49
    elif damage=='forward_count':v['forward_devices'].pop()
    elif damage=='forward_input':v['forward_devices'][0]['input_device']='cuda:0'
    elif damage=='forward_prediction':v['forward_devices'][0]['prediction_device']='cuda:0'
    elif damage=='forward_cuda':v['forward_devices'][0]['cuda_initialized']=True
    elif damage=='teardown':v['post_pass_module_device']='cuda:0'
    else:v['execution_device']['accelerator']='gpu'
    with pytest.raises(ValueError):api()._execution_header(v,50)


@pytest.mark.parametrize('flag',[None,1,'full'])
@pytest.mark.parametrize('function',['publish_wilor_report','verify_wilor_report'])
def test_report_full_requirement_is_boolean_before_loading_any_raw_evidence(tmp_path,flag,function):
    with pytest.raises(ValueError,match='bool'):
        if function=='publish_wilor_report':getattr(api(),function)(None,None,None,tmp_path/'out',require_full=flag)
        else:getattr(api(),function)(None,tmp_path/'out',require_full=flag)
    assert not (tmp_path/'out').exists()


def test_report_existing_directory_is_preserved_before_raw_fit_is_opened(tmp_path):
    p=tmp_path/'report';p.mkdir();(p/'retained').write_bytes(b'existing')
    with pytest.raises(FileExistsError):api().publish_wilor_report(None,None,None,p,require_full=False)
    assert (p/'retained').read_bytes()==b'existing' and list(p.iterdir())==[p/'retained']


def curve_fixture(tmp_path):
    pytest.importorskip('torch')
    from egoemg_repro.vision_literal_train import _epoch_bytes
    rows=[dict(epoch=i,train_loss=float(3-i),validation=dict(val_mae=float(2-i)),optimizer_executions=2*(i+1)) for i in range(2)]
    path=tmp_path/'epochs.jsonl';path.write_bytes(_epoch_bytes(rows,'a'*64))
    result=dict(scope='cpu_engineering',identity_sha256='a'*64,sha256='b'*64,completed_epochs=2,optimizer_executions=4,
        journal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selected_best=dict(epoch=1,score=1.,model_sha256='c'*64))
    return rows,result,path


def test_training_curve_reads_actual_hash_chained_journal_and_marks_native_mae_best(tmp_path):
    rows,result,_=curve_fixture(tmp_path);v=api()._curve_data(tmp_path,result)
    assert v['epochs']==rows and v['selected_best']==result['selected_best']
    assert v['selected_epoch_zero_based']==1 and v['journal_sha256']==result['journal_sha256']
    assert 'composite' in v['training_unit'] and 'val_mae' in v['validation_unit'] and 'radians' in v['validation_unit']
    assert v['scope']=='cpu_engineering' and v['scientific_eligible'] is False


@pytest.mark.parametrize('damage',['journal','identity','best_epoch','best_score','epochs','updates'])
def test_training_curve_rejects_changed_raw_journal_or_resealed_wrong_selection(tmp_path,damage):
    _,v,path=curve_fixture(tmp_path)
    if damage=='journal':path.write_bytes(path.read_bytes()+b'changed\n')
    elif damage=='identity':v['identity_sha256']='d'*64
    elif damage=='best_epoch':v['selected_best']['epoch']=0
    elif damage=='best_score':v['selected_best']['score']=.5
    elif damage=='epochs':v['completed_epochs']=1
    else:v['optimizer_executions']=3
    with pytest.raises(ValueError):api()._curve_data(tmp_path,v)


def test_curve_images_are_exact_regenerable_and_engineering_watermarked(tmp_path):
    pytest.importorskip('matplotlib');_,result,_=curve_fixture(tmp_path)
    curves=api()._curve_data(tmp_path,result);first=api()._curve_images(curves);second=api()._curve_images(curves)
    assert first==second and set(first)=={'training-curves.png','training-curves.svg'}
    assert first['training-curves.png'].startswith(b'\x89PNG') and b'ENGINEERING ONLY' in first['training-curves.svg']
    assert b'Native validation MAE (radians)' in first['training-curves.svg'] and b'Selected best' in first['training-curves.svg']


@pytest.fixture(scope='module')
def retained_cpu_chain():
    import os
    from pathlib import Path
    source=os.environ.get('EGOEMG_WILOR_RETAINED_PASSES')
    if not source:pytest.skip('genuine retained 631M native CPU chain required; never synthesize a model/permit')
    root=Path(source);m=api();context=m._Evidence()
    values={name:m._read_pass(root/name,context) for name in ('val-native','val-replay','test-native','test-replay')}
    profile=m._read_profile(root/'profile',context);context.guard.check()
    assert values['test-native']['scope']=='cpu_engineering'
    return root,context,values,profile


def test_genuine_retained_native_cpu_chain_has_both_hand_repeats_exact_independent_replay(retained_cpu_chain):
    root,context,values,profile=retained_cpu_chain
    for role in ('val','test'):
        left,right=values[role+'-native'],values[role+'-replay']
        assert left['layout']['source_indices']==([0,1,2,3] if role=='val' else [0,3,6,9])
        assert left['statistics']==right['statistics'] and left['prediction_sha256']==right['prediction_sha256']
        assert left['process']!=right['process'] and right['started_ns']>left['finished_ns']
        assert left['statistics']['consumed_samples']==4 and left['statistics']['unique_samples']==2
        assert left['statistics']['consumed']['samples']==dict(all=4,vision_valid=2,missing_crop=2)
        assert left['statistics']['consumed']['all']['overall']['valid_angle_count']==88
        assert left['statistics']['unique_first_repeat']['all']['overall']['valid_angle_count']==44
        assert len(left['statistics']['consumed']['all']['by_joint'])==22
        assert left['model_before_sha256']==left['model_after_sha256']==right['model_after_sha256']
        assert left['statistics']['native_batch_weighted_mae_rad']==pytest.approx(left['native_metrics'][role+'_mae'],abs=2e-6,rel=2e-6)
        for v in (left,right):api()._execution_header(v,context.fits[api()._path(v['fit_path'])][0]['finished_ns'])
    assert values['test-native']['profile_sha256']==profile['sha256']
    assert values['test-native']['started_ns']>profile['finished_ns']
    context.guard.check()


def test_genuine_retained_report_reopens_actual_curves_and_never_discloses_private_paths(retained_cpu_chain):
    root,context,values,profile=retained_cpu_chain
    report=api().verify_wilor_report(root/'test-replay',root/'report',require_full=False)
    assert report['scientific_eligible'] is False and report['full_paper_complete'] is False
    assert report['completed_epochs']==2 and report['optimizer_executions']==4
    assert report['statistics']==values['test-replay']['statistics']
    assert report['curve_data']['selected_best']==values['test-replay']['selected_best']
    raw=(root/'report/metrics.json').read_text()
    assert str(root) not in raw and '/data/' not in raw and 'synthetic' not in raw
    assert (root/'report/completion.json').is_file()
    with pytest.raises(FileExistsError):api().publish_wilor_report(None,None,None,root/'report',require_full=False)
    context.guard.check()
