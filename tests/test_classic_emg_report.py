"""Hand-calculated report arithmetic, not inference or scientific completion."""
import importlib
import math
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

np=pytest.importorskip('numpy')


def api():
    try:return importlib.import_module('egoemg_repro.classic_emg_report')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.classic_emg_report':pytest.fail('classic report arithmetic not implemented')
        raise


@pytest.mark.parametrize('variant,length,center',[('vemg2pose',6000,2105),('neuropose',7790,3895)])
def test_original_window_center_and_full_trajectory_are_distinct(variant,length,center):
    r=api();p=np.zeros((22,length),dtype=np.float32);t=np.zeros_like(p);mask=np.ones(length,dtype=bool)
    p[0,center]=2
    if variant=='vemg2pose':p[0,3000]=100
    report=r.ClassicAngleReport(variant);report.update(p,t,mask,hand='left',split='user',user='private-uA')
    value=report.report();c=value['center']['overall'];a=value['trajectory']['overall']
    assert c['absolute_error_sum_rad']==2 and c['valid_angle_count']==22
    assert c['mae_deg']==pytest.approx(2/22*180/math.pi)
    assert a['absolute_error_sum_rad']==(102 if variant=='vemg2pose' else 2)
    assert a['valid_angle_count']==22*length
    assert value['alignment']['output_center']==center
    assert value['scientific_eligible'] is False


def test_hand_calculated_generated_mask_weighting_and_all_partitions_reconcile():
    r=api();s=r._AngleStatistics(center_index=1)
    items=[('left','user','uA',[1,1,0],[(0,0,1),(0,20,2),(1,1,3),(1,21,4)]),
        ('right','gesture','uB',[0,1,0],[(1,0,5),(1,20,8)]),
        ('left','both','uA',[1,0,1],[(0,2,7),(0,21,8),(2,3,9),(2,20,10)])]
    for hand,split,user,mask,entries in items:
        p=np.zeros((22,3),dtype=np.float32)
        for time,joint,error in entries:p[joint,time]=error
        s.update(p,np.zeros_like(p),np.array(mask,dtype=bool),hand=hand,split=split,user=user)
    v=s.report();a=v['trajectory'];c=v['center']
    assert a['overall']['absolute_error_sum_rad']==57 and a['overall']['valid_angle_count']==110
    assert a['invalid_angle_count']==88 and c['invalid_angle_count']==22
    assert c['overall']['absolute_error_sum_rad']==20 and c['overall']['valid_angle_count']==44
    assert c['by_split']['both']['mae_deg'] is None
    assert a['by_hand']['left']['absolute_error_sum_rad']==44 and a['by_hand']['left']['valid_angle_count']==88
    assert a['user_summary']['mean_mae_deg']==pytest.approx(6/11*180/math.pi)
    assert a['user_summary']['std_mae_deg']==pytest.approx(1/22*180/math.pi)
    assert [j['absolute_error_sum_rad'] for j in a['by_joint']]==[6,3,7,9]+[0]*16+[20,12]
    assert [j['valid_angle_count'] for j in a['by_joint']]==[5]*22
    assert a['by_finger']['thumb']['absolute_error_sum_rad']==25
    assert [a['by_pd_group'][g]['absolute_error_sum_rad'] for g in ['proximal','mid','distal']]==[9,7,9]
    assert a['by_component']['fingers']['absolute_error_sum_rad']==25
    assert a['by_component']['wrist']['absolute_error_sum_rad']==32
    assert a['overall']['mae_deg']!=pytest.approx((10/44+13/22+34/44)/3*180/math.pi)


@pytest.mark.parametrize('damage',['nonfinite_invalid','angle_mask','wrong_width','wrong_center','unknown_variant'])
def test_no_legacy_center_per_angle_mask_or_nonfinite_cleanup_is_allowed(damage):
    r=api();p=np.zeros((22,6000),np.float32);t=p.copy();mask=np.ones(6000,dtype=bool)
    if damage=='nonfinite_invalid':mask[0]=False;p[0,0]=np.nan
    elif damage=='angle_mask':mask=np.ones((22,6000),dtype=bool)
    elif damage=='wrong_width':p=p[:,:5490];t=t[:,:5490];mask=mask[:5490]
    with pytest.raises((ValueError,TypeError)):
        report=r.ClassicAngleReport('legacy' if damage=='unknown_variant' else 'vemg2pose',
            **({'center_index':3000} if damage=='wrong_center' else {}))
        report.update(p,t,mask,hand='left',split='user',user='u')


@pytest.mark.parametrize('variant,left',[('vemg2pose',1790),('neuropose',0)])
def test_actual_original_tuple_must_equal_its_exact_input_target_and_generated_mask(variant,left):
    r=api();original=np.zeros((2,22,7790),np.float32);original[:,0,3895]=2
    mask=np.ones((2,7790),dtype=bool);target=original[...,left:];generated=mask[...,left:]
    prediction=np.zeros_like(target,dtype=np.float16)
    result=r.validate_original_tuple((prediction,target,generated),variant=variant,
        original_target=original,original_mask=mask)
    assert result['output_center']==3895-left
    damaged=target.copy();damaged[1,0,-1]=1
    with pytest.raises(ValueError):r.validate_original_tuple((prediction,damaged,generated),variant=variant,
        original_target=original,original_mask=mask)


def ledger_windows(role='test'):
    return [dict(role=role,episode=i,start=i*7790,end=(i+1)*7790,
        split=('user' if i==0 else 'gesture') if role=='test' else 'train') for i in range(2)]


def ledger_batch(indices,*,role='test'):
    torch=pytest.importorskip('torch');n=len(indices);windows=ledger_windows(role)
    return dict(emg=torch.zeros(n,16,7790),joint_angles=torch.ones(n,22,7790),
        label_valid_mask=torch.ones(n,7790,dtype=torch.bool),classic_sample_index=torch.tensor(indices),
        mask_policy=['official_literal_loader']*n,dataset_name=['egoemg']*n,
        hand=[('left','right')[i%2] for i in indices],subject=[['private-A','private-B'][i//2] for i in indices],
        role=[role]*n,split=[windows[i//2]['split'] for i in indices],
        episode=torch.tensor([i//2 for i in indices]),start=torch.tensor([windows[i//2]['start'] for i in indices]),
        end=torch.tensor([windows[i//2]['end'] for i in indices]))


def test_report_input_fingerprints_match_native_encoding_without_inventing_a_loss():
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    from egoemg_repro import classic_emg_evaluate as native
    r=api();windows=ledger_windows();expected={0:'private-A',1:'private-B'}
    report=r.ReportInputLedger(windows,variant='neuropose',batch_size=3,role='test',subjects=expected)
    original=native.TestInputLedger(windows,variant='neuropose',batch_size=3)
    for indices in ([0,1,2],[3]):
        batch=ledger_batch(indices);a=report.prepare(batch);b=original.prepare(batch)
        for key in b:assert a[key]==b[key]
        output=(batch['joint_angles'].half(),batch['joint_angles'],batch['label_valid_mask'])
        report.commit(batch,a,output);original.commit(batch,b,loss=2.)
    value=report.finish();baseline=original.finish()
    for key in ['samples','batch_sizes','coverage_sha256','target_sha256','mask_sha256','sample_sha256','fingerprint_encoding','fingerprint_scope']:
        assert value[key]==baseline[key]
    assert 'batch_weighted_loss' not in value and 'test_loss' not in value
    assert len(value['predictions'])==2 and value['predictions'][-1]['shape']==[1,22,7790]


def test_validation_ledger_is_explicitly_separate_from_test_and_covers_every_hand():
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api()
    ledger=r.ReportInputLedger(ledger_windows('val'),variant='neuropose',batch_size=3,role='val',subjects={0:'private-A',1:'private-B'})
    for ids in ([0,1,2],[3]):
        b=ledger_batch(ids,role='val');receipt=ledger.prepare(b)
        ledger.commit(b,receipt,(b['joint_angles'].half(),b['joint_angles'],b['label_valid_mask']))
    assert ledger.finish()['samples']==4
    with pytest.raises(ValueError):r.ReportInputLedger(ledger_windows('val'),variant='neuropose',batch_size=3,role='test',subjects={0:'private-A',1:'private-B'})


@pytest.mark.parametrize('damage',['subject','split','role','later_emg','later_target','later_mask','prediction_dtype','duplicate','missing'])
def test_report_ledger_rejects_metadata_mutation_and_unconsumed_or_changed_inputs(damage):
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api()
    ledger=r.ReportInputLedger(ledger_windows(),variant='neuropose',batch_size=3,role='test',subjects={0:'private-A',1:'private-B'})
    b=ledger_batch([0,1,2])
    if damage in ['subject','split','role']:
        b[damage][1]='changed'
        with pytest.raises(ValueError):ledger.prepare(b)
        return
    receipt=ledger.prepare(b)
    if damage=='later_emg':b['emg'][0,0,0]=1
    elif damage=='later_target':b['joint_angles'][0,0,0]=2
    elif damage=='later_mask':b['label_valid_mask'][0,0]=False
    out=(b['joint_angles'].half(),b['joint_angles'],b['label_valid_mask'])
    if damage=='prediction_dtype':out=(out[0].double(),out[1],out[2])
    with pytest.raises(ValueError):
        ledger.commit(b,receipt,out)
        if damage=='duplicate':ledger.commit(b,receipt,out)
        else:ledger.finish()


def test_prediction_replay_requires_exact_typed_batches_and_never_uses_engineering_tolerance():
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api()
    outputs=[]
    for changed in [False,True]:
        ledger=r.ReportInputLedger(ledger_windows(),variant='neuropose',batch_size=3,role='test',subjects={0:'private-A',1:'private-B'})
        for ids in ([0,1,2],[3]):
            b=ledger_batch(ids);receipt=ledger.prepare(b);prediction=b['joint_angles'].half()
            if changed and ids==[3]:prediction[0,0,-1]=torch.nextafter(prediction[0,0,-1],torch.tensor(2.,dtype=torch.float16))
            ledger.commit(b,receipt,(prediction,b['joint_angles'],b['label_valid_mask']))
        outputs.append(ledger.finish())
    assert r.compare_prediction_replay(outputs[0],outputs[0])['prediction_hashes_equal'] is True
    with pytest.raises(ValueError):r.compare_prediction_replay(*outputs)


@pytest.mark.parametrize('variant,batch,last',[('vemg2pose',600,376),('neuropose',256,40)])
def test_validation_prediction_policy_preserves_actual_full_cohort_and_original_batch(variant,batch,last):
    p=api().report_pass_policy(variant=variant,role='val')
    assert p['samples']==1576 and p['batch_size']==batch and p['final_batch_size']==last
    assert p['num_workers']==0 and p['shuffle'] is False and p['drop_last'] is False
    assert p['precision']=='16-mixed' and p['prediction_agreement']=='exact_typed_prediction_replay_v1'
    assert p['scientific_eligible'] is False


@pytest.mark.parametrize('device',['cpu','cuda:1','bf16'])
def test_validation_calibration_rejects_wrong_device_before_reading_any_run(tmp_path,device):
    with pytest.raises(ValueError):api().run_validation_prediction(tmp_path/'missing',output=tmp_path/'output',device=device)
    assert not (tmp_path/'output').exists()


def test_validation_calibration_cannot_use_engineering_or_incomplete_training(tmp_path,monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api();e=r._evaluation()
    run=tmp_path/'run';run.mkdir();(run/'identity.json').write_text('{"scope":"formal_engineering_acceptance"}')
    (run/'status.json').write_text('{"state":"engineering_finished"}');(run/'result.json').write_text('{"passed":true}')
    def forbidden(*args,**kwargs):raise AssertionError('factory reached before actual formal termination')
    monkeypatch.setattr(e.cuda,'_factory',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)):r.run_validation_prediction(run,output=tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('damage',[None,'buffer','gradient','scaler'])
def test_generic_cpu_prediction_consumption_preserves_original_model_and_scaler(damage):
    """Generic helper fixture only; never reaches the CUDA artifact writer."""
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api();e=r._evaluation()
    from types import SimpleNamespace as NS
    from contextlib import nullcontext
    scaler={'scale':65536.}
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.register_buffer('counter',torch.tensor(0))
        def forward(self,batch):
            if damage=='buffer':self.counter+=1
            if damage=='gradient':self.weight.grad=torch.ones_like(self.weight)
            if damage=='scaler':scaler['scale']/=2
            return batch['joint_angles']*self.weight,batch['joint_angles'],batch['label_valid_mask']
    module=Module();before=e.storage.state_digest(module.state_dict())
    prepared=NS(module=module,precision=NS(test_step_context=nullcontext,state_dict=lambda:scaler.copy()),
        selection={'selected_model_sha256':before},selected_checkpoint={'MixedPrecision':scaler.copy()})
    ledger=r.ReportInputLedger(ledger_windows('val'),variant='neuropose',batch_size=3,role='val',subjects={0:'private-A',1:'private-B'})
    batches=[ledger_batch(ids,role='val') for ids in ([0,1,2],[3])]
    if damage:
        with pytest.raises(ValueError):r._consume_prediction_batches(prepared,batches,ledger,device='cpu')
    else:
        out=r._consume_prediction_batches(prepared,batches,ledger,device='cpu')
        assert out['coverage']['samples']==4 and out['statistics'] is None
        assert out['model_before_sha256']==out['model_after_sha256']==before
        assert all(p.grad is None for p in module.parameters())


def completed_ledger():
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api()
    ledger=r.ReportInputLedger(ledger_windows(),variant='neuropose',batch_size=3,role='test',subjects={0:'private-A',1:'private-B'})
    for ids in ([0,1,2],[3]):
        b=ledger_batch(ids);receipt=ledger.prepare(b)
        ledger.commit(b,receipt,(b['joint_angles'].half(),b['joint_angles'],b['label_valid_mask']))
    return ledger.finish()


@pytest.mark.parametrize('damage',[None,'missing','dtype','shape','order','chain','count','grouping','role'])
def test_saved_prediction_receipts_recompute_every_typed_batch_chain(damage):
    value=completed_ledger();r=api();policy=dict(role='test',variant='neuropose',samples=4,batch_size=3)
    if damage=='missing':value['predictions'].pop()
    elif damage=='dtype':value['predictions'][-1]['dtype']='torch.float64'
    elif damage=='shape':value['predictions'][-1]['shape'][1]=20
    elif damage=='order':value['predictions'].reverse()
    elif damage=='chain':value['predictions'][-1]['metadata_sha256']='0'*64
    elif damage=='count':value['samples']=3
    elif damage=='grouping':value['batch_sizes']=[2,2]
    elif damage=='role':value['role']='val'
    if damage:
        with pytest.raises(ValueError):r._validate_saved_coverage(value,policy)
    else:r._validate_saved_coverage(value,policy)


@pytest.mark.parametrize('artifact',[{'passed':True},{'observed_exact_replay':True},True,None])
def test_profile_gate_requires_raw_artifact_paths_not_passed_dictionaries(artifact):
    with pytest.raises(ValueError):api().verify_validation_profile(artifact)


def test_fake_profile_file_does_not_reach_any_factory(tmp_path,monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api();e=r._evaluation()
    p=tmp_path/'profile.json';p.write_text('{"format":"classic_validation_exact_replay_profile_v1","observed_exact_replay":true,"artifacts":[]}')
    def forbidden(*args,**kwargs):raise AssertionError('profile verification must not construct a model')
    monkeypatch.setattr(e.cuda,'_factory',forbidden)
    with pytest.raises(ValueError):r.verify_validation_profile(p)


def raw_prediction_pair(tmp_path,monkeypatch,*,role='val'):
    """JSON reader mechanics only: termination authority is stubbed, never GPU qualification."""
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api();e=r._evaluation()
    run=tmp_path/'trained';run.mkdir();(run/'result.json').write_text('{}')
    selection={'selected_model_sha256':'1'*64}
    manifest=dict(sha256='2'*64,policy={'variant':'vemg2pose'},
        state_identity=dict(bindings={'fixture':'not a real qualification'},runtime={'fixture':'no CUDA'}))
    best={'MixedPrecision':{'scale':65536.}}
    def context(path):
        assert Path(path)==run
        return dict(run=run,manifest=manifest,selection=selection,best=best,
            guard=e.cohort._StatGuard([run/'result.json']))
    monkeypatch.setattr(r,'_terminated_context',context)
    def forbidden(*args,**kwargs):raise AssertionError('raw/profile reader must not construct a model')
    monkeypatch.setattr(e.cuda,'_factory',forbidden)
    identity=r._identity_record(run,manifest,selection,role)
    policy=identity['policy'];n=policy['samples'];batch=policy['batch_size']
    coverage=completed_ledger();coverage.update(role=role,samples=n,
        batch_sizes=[min(batch,n-i) for i in range(0,n,batch)],predictions=[])
    digest=hashlib.sha256()
    for i in range(n):digest.update(i.to_bytes(8,'big'))
    coverage['coverage_sha256']=digest.hexdigest()
    chains={name:hashlib.sha256() for name in r.ReportInputLedger.CHAINS}
    for sequence,size in enumerate(coverage['batch_sizes']):
        record=dict(sequence=sequence,batch_size=size,dtype='torch.float16',shape=[size,22,6000])
        for name,chain in chains.items():
            content=e.artifacts.json_hash([name,sequence]);record[name+'_sha256']=content
            chain.update(bytes.fromhex(e.artifacts.json_hash(dict(sequence=sequence,batch_size=size,digest=content))))
        coverage['predictions'].append(record)
    coverage.update({name+'_sha256':chain.hexdigest() for name,chain in chains.items()})
    paths=[]
    for i,kind in enumerate(('reference','replay')):
        directory=tmp_path/kind;directory.mkdir();paths.append(directory)
        result=dict(coverage=copy.deepcopy(coverage),statistics=None,
            model_before_sha256=selection['selected_model_sha256'],model_after_sha256=selection['selected_model_sha256'],
            scaler_before_sha256=e.storage.state_digest(best['MixedPrecision']),
            scaler_after_sha256=e.storage.state_digest(best['MixedPrecision']),format=r.PASS_FORMAT,
            scope=policy['scope'],scientific_eligible=False,production_ready=False,identity_sha256=identity['sha256'],
            process=dict(pid=100+i,start_ticks=1000+i,phase='classic-'+role+'-'+kind),pass_kind=kind,
            reference=None if i==0 else dict(directory=str(paths[0]),result_sha256=e.file_sha256(paths[0]/'result.json')))
        (directory/'identity.json').write_text(json.dumps(identity))
        (directory/'result.json').write_text(json.dumps(result))
        (directory/'status.json').write_text(json.dumps(dict(state='prediction_pass_finished',scope=r.PASS_FORMAT,
            scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(directory/'result.json'))))
    return r,e,paths


@pytest.mark.parametrize('damage',[None,'state','scaler','process','status_flag_type','prediction','selection','statistics','same_process','partial'])
def test_raw_prediction_reader_reopens_complete_artifacts_and_rejects_resealed_corruption(tmp_path,monkeypatch,damage):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);directory=paths[1]
    result=json.loads((directory/'result.json').read_text());status=json.loads((directory/'status.json').read_text())
    if damage=='state':result['model_after_sha256']='3'*64
    elif damage=='scaler':result['scaler_after_sha256']='3'*64
    elif damage=='process':result['process']['start_ticks']=True
    elif damage=='status_flag_type':status['scientific_eligible']=0
    elif damage=='prediction':result['coverage']['predictions'][-1]['prediction_sha256']='3'*64
    elif damage=='selection':
        identity=json.loads((directory/'identity.json').read_text());identity['selection']['selected_model_sha256']='3'*64
        identity['sha256']=e.artifacts.json_hash({k:v for k,v in identity.items() if k!='sha256'})
        (directory/'identity.json').write_text(json.dumps(identity));result['identity_sha256']=identity['sha256']
    elif damage=='statistics':result['statistics']={'passed':True}
    elif damage=='same_process':result['process'].update(pid=100,start_ticks=1000)
    elif damage=='partial':result['coverage']['predictions'].pop()
    (directory/'result.json').write_text(json.dumps(result));status['result_sha256']=e.file_sha256(directory/'result.json')
    (directory/'status.json').write_text(json.dumps(status))
    if damage:
        with pytest.raises(ValueError):r._read_prediction_pass(directory,role='val')
    else:
        identity,value,guard=r._read_prediction_pass(directory,role='val');guard.check()
        assert identity['policy']['samples']==1576 and value['pass_kind']=='replay'
        assert paths[0]/'result.json' in guard.paths and paths[1]/'result.json' in guard.paths


def test_validation_profile_is_rebuilt_from_raw_passes_and_detects_later_mutation(tmp_path,monkeypatch):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch)
    profile,guard=r._build_validation_profile(*paths);guard.check()
    assert profile['observed_exact_replay'] is True and profile['universal_determinism_proven'] is False
    assert profile['scientific_eligible'] is False and profile['test_policy']['samples']==4306
    target=tmp_path/'profile.json';target.write_text(json.dumps(profile))
    got,held=r.verify_validation_profile(target);assert got==profile
    (paths[0]/'result.json').write_text('{}')
    with pytest.raises(ValueError):held.check()
    with pytest.raises(ValueError):r.verify_validation_profile(target)


@pytest.mark.parametrize('damage',[None,'existing','reversed','same','caller_flag'])
def test_freeze_validation_profile_is_a_new_raw_bound_file_not_an_acceptance_flag(tmp_path,monkeypatch,damage):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);target=tmp_path/'profile.json'
    if damage=='existing':target.write_text('foreign bytes')
    if damage=='reversed':paths.reverse()
    if damage=='same':paths[1]=paths[0]
    if damage=='caller_flag':paths[1]={'passed':True}
    if damage:
        with pytest.raises((ValueError,FileExistsError)):r.freeze_validation_profile(*paths,output=target)
        if damage=='existing':assert target.read_text()=='foreign bytes'
        else:assert not target.exists()
    else:
        result=r.freeze_validation_profile(*paths,output=target)
        assert r.verify_validation_profile(target)[0]==result


@pytest.mark.parametrize('damage',['close_error','late_source_change','parent_replaced'])
def test_profile_freeze_revokes_only_its_own_file_after_late_failure(tmp_path,monkeypatch,damage):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);parent=tmp_path/'profile-dir';parent.mkdir()
    target=parent/'profile.json';created={};acquire=e.artifacts._Directory.__init__;close=e.artifacts._Directory.close
    def remember(directory,path,**kwargs):
        acquire(directory,path,**kwargs)
        if directory.path==parent and 'directory' not in created:created['directory']=directory
    def fail(directory):
        is_owned=directory is created.get('directory') and not created.get('closed')
        result=close(directory)
        if is_owned:
            created['closed']=True
            if damage=='close_error':raise OSError('injected owned close failure')
            if damage=='late_source_change':(tmp_path/'trained'/'result.json').write_text('{"changed":true}')
            if damage=='parent_replaced':
                parent.rename(tmp_path/'moved');parent.mkdir();target.write_text('foreign profile')
        return result
    monkeypatch.setattr(e.artifacts._Directory,'__init__',remember);monkeypatch.setattr(e.artifacts._Directory,'close',fail)
    with pytest.raises((ValueError,OSError)):r.freeze_validation_profile(*paths,output=target)
    if damage=='parent_replaced':
        assert target.read_text()=='foreign profile' and not (tmp_path/'moved'/'profile.json').exists()
    else:assert not target.exists()
    assert (paths[0]/'result.json').exists() and (paths[1]/'result.json').exists()


def test_profile_cannot_write_inside_immutable_training_or_validation_sidecars(tmp_path,monkeypatch):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch)
    for parent in (tmp_path/'trained',*paths):
        with pytest.raises(ValueError):r.freeze_validation_profile(*paths,output=parent/'profile.json')
        assert not (parent/'profile.json').exists()


def test_profile_cli_help_does_not_import_model_or_initialize_cuda(monkeypatch,capsys):
    r=api()
    def forbidden():raise AssertionError('help must not initialize the evaluation runtime')
    monkeypatch.setattr(r,'_evaluation',forbidden)
    with pytest.raises(SystemExit) as exc:r.main(['--help'])
    assert exc.value.code==0 and 'freeze-validation' in capsys.readouterr().out


@pytest.mark.parametrize('command',['validation-pass','freeze-validation','verify-validation'])
def test_profile_cli_dispatch_keeps_validation_and_test_separate(monkeypatch,capsys,command):
    r=api();calls=[]
    def validation(run,**kwargs):calls.append((run,kwargs));return {'kind':'validation'}
    def freeze(reference,replay,**kwargs):calls.append((reference,replay,kwargs));return {'kind':'freeze'}
    def verify(path):calls.append(path);return ({'format':r.PROFILE_FORMAT,'sha256':'1'*64},None)
    monkeypatch.setattr(r,'run_validation_prediction',validation)
    monkeypatch.setattr(r,'freeze_validation_profile',freeze);monkeypatch.setattr(r,'verify_validation_profile',verify)
    arguments={'validation-pass':['--run','trained','--output','sidecar','--reference','first'],
        'freeze-validation':['--reference','first','--replay','second','--output','profile.json'],
        'verify-validation':['--profile','profile.json']}
    r.main([command,*arguments[command]]);value=json.loads(capsys.readouterr().out)
    assert len(calls)==1
    if command=='validation-pass':
        assert calls[0]==(Path('trained'),dict(output=Path('sidecar'),reference=Path('first'),device='cuda:0'))
        assert value['kind']=='validation'
    elif command=='freeze-validation':assert value['kind']=='freeze'
    else:assert value=={'format':r.PROFILE_FORMAT,'sha256':'1'*64,'verified':True}


def test_review_late_profile_writer_cleanup_failure_revokes_published_file(tmp_path,monkeypatch):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);target=tmp_path/'profile.json';unlink=r.os.unlink;injected=[]
    def late_unlink(name,*args,**kwargs):
        result=unlink(name,*args,**kwargs)
        if str(name).startswith('.profile.json.') and str(name).endswith('.incomplete') and not injected:
            injected.append(True);raise OSError('injected post-publication temporary-link cleanup failure')
        return result
    monkeypatch.setattr(r.os,'unlink',late_unlink)
    with pytest.raises(OSError):r.freeze_validation_profile(*paths,output=target)
    assert injected and not target.exists()


def test_review_profile_public_name_replacement_cannot_be_adopted_or_deleted(tmp_path,monkeypatch):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);target=tmp_path/'profile.json';moved=tmp_path/'moved-profile.json'
    owner=r if hasattr(r,'_publish_profile_file') else e.artifacts
    name='_publish_profile_file' if owner is r else '_write_json';write=getattr(owner,name)
    def replace(directory,filename,*args,**kwargs):
        result=write(directory,filename,*args,**kwargs)
        if filename==target.name:
            target.rename(moved);target.write_text('foreign profile bytes')
        return result
    monkeypatch.setattr(owner,name,replace)
    with pytest.raises(ValueError):r.freeze_validation_profile(*paths,output=target)
    assert target.read_text()=='foreign profile bytes'
    with pytest.raises(ValueError):r.verify_validation_profile(moved)


@pytest.mark.parametrize('after_close',[False,True])
def test_review_profile_final_rescue_close_failure_retains_revocation_capability(tmp_path,monkeypatch,after_close):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);parent=tmp_path/'profiles';parent.mkdir();target=parent/'profile.json'
    acquired=[];injected=[];acquire=e.artifacts._Directory.__init__;close=e.artifacts._Directory.close
    def remember(directory,path,**kwargs):
        acquire(directory,path,**kwargs)
        if directory.path==parent:acquired.append(directory)
    def fail(directory):
        if len(acquired)>=2 and directory is acquired[1] and not injected:
            injected.append(True)
            if after_close:close(directory)
            raise OSError('injected final rescue close failure')
        return close(directory)
    monkeypatch.setattr(e.artifacts._Directory,'__init__',remember);monkeypatch.setattr(e.artifacts._Directory,'close',fail)
    with pytest.raises(OSError):r.freeze_validation_profile(*paths,output=target)
    assert injected and not target.exists()


@pytest.mark.parametrize('failure',[None,'before_rescue_close','after_rescue_close'])
def test_review_prediction_publication_filesystem_mechanics_revoke_on_rescue_close(tmp_path,monkeypatch,failure):
    """File transaction only; no CUDA preparation/inference/qualification is mocked."""
    from types import SimpleNamespace as NS
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);output=tmp_path/'new-reference'
    identity=json.loads((paths[0]/'identity.json').read_text());result=json.loads((paths[0]/'result.json').read_text())
    owned=e.artifacts._Directory(output,create=True);rescue=e.artifacts._Directory(output,create=False)
    backup=None;close=e.artifacts._Directory.close;injected=[];publications={}
    e.artifacts._write_json(owned,'identity.json',identity)
    e.artifacts._write_json(owned,'progress.json',{'state':'running'})
    def fail(directory):
        if directory is rescue and failure and not injected:
            injected.append(True)
            if failure=='after_rescue_close':close(directory)
            raise OSError('injected prediction final rescue close failure')
        return close(directory)
    monkeypatch.setattr(e.artifacts._Directory,'close',fail)
    prepared=NS(guard=e.cohort._StatGuard([tmp_path/'trained'/'result.json']))
    try:
        backup=r._retain_directory(rescue)
        if failure:
            with pytest.raises(OSError):r._finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup,publications)
            assert injected and json.loads((output/'status.json').read_text())['state']=='failed'
            with pytest.raises((ValueError,FileNotFoundError)):r._read_prediction_pass(output,role='val')
        else:
            assert r._finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup,publications)==result
            assert r._read_prediction_pass(output,role='val')[1]==result
    finally:
        for directory in (owned,rescue,backup):
            if directory is not None:close(directory)
        r._discard_publications(publications)


def native_coverage_fixture(variant='vemg2pose'):
    r=api();batch=600 if variant=='vemg2pose' else 256;n=4306;digest=hashlib.sha256()
    for i in range(n):digest.update(i.to_bytes(8,'big'))
    return dict(samples=n,batch_sizes=[min(batch,n-i) for i in range(0,n,batch)],coverage_sha256=digest.hexdigest(),
        batch_weighted_loss=1.25,test_loss=1.25,target_sha256='1'*64,mask_sha256='2'*64,sample_sha256='3'*64,
        fingerprint_encoding='ordered chain of typed canonical-batch target/mask and frozen-sample receipts v1',
        fingerprint_scope='consumed original input target span and generated 2D mask; no train-step prediction-hook claim')


@pytest.mark.parametrize('variant',['vemg2pose','neuropose'])
@pytest.mark.parametrize('damage',[None,'count','grouping','order','loss','nonfinite','loss_type','hash','scope','extra'])
def test_original_native_test_coverage_requires_complete_grouping_and_finite_native_loss(variant,damage):
    value=native_coverage_fixture(variant);r=api()
    if damage=='count':value['samples']-=1
    elif damage=='grouping':value['batch_sizes'][-1]-=1
    elif damage=='order':value['coverage_sha256']='0'*64
    elif damage=='loss':value['test_loss']=2.
    elif damage=='nonfinite':value['batch_weighted_loss']=float('nan')
    elif damage=='loss_type':value['test_loss']=True
    elif damage=='hash':value['sample_sha256']='bad'
    elif damage=='scope':value['fingerprint_scope']='prediction hook'
    elif damage=='extra':value['prediction_sha256']='1'*64
    if damage:
        with pytest.raises(ValueError):r._validate_native_coverage(value,variant=variant)
    else:r._validate_native_coverage(value,variant=variant)


def test_actual_process_identity_is_read_from_proc_without_constructing_any_model():
    value=api()._pid_identity(os.getpid(),phase='unit-file-mechanics')
    assert value['pid']==os.getpid() and type(value['start_ticks']) is int and value['start_ticks']>0
    assert value['phase']=='unit-file-mechanics'


@pytest.mark.parametrize('pid',[True,0,-1,'123'])
def test_invalid_process_values_are_not_real_process_receipts(pid):
    with pytest.raises(ValueError):api()._pid_identity(pid,phase='unit-file-mechanics')


@pytest.mark.parametrize('payload',[b'',b'G',b'X'+bytes(32),b'G'+bytes(31),b'G'+bytes(33)])
def test_native_go_pipe_rejects_eof_partial_wrong_or_trailing_payload(payload):
    read,write=os.pipe();os.write(write,payload);os.close(write)
    try:
        with pytest.raises(ValueError):api()._receive_launch_go(read)
    finally:
        try:os.close(read)
        except OSError:pass


def test_native_go_pipe_rejects_regular_file_and_never_treats_it_as_parent_release(tmp_path):
    path=tmp_path/'fake-go';path.write_bytes(b'G'+bytes(32));fd=os.open(path,os.O_RDONLY)
    try:
        with pytest.raises(ValueError):api()._receive_launch_go(fd)
    finally:
        try:os.close(fd)
        except OSError:pass


def test_real_child_waits_for_parent_pipe_release_before_any_runtime_import(tmp_path):
    """Real OS process/pipe mechanics; no native test or GPU qualification."""
    r=api();assert callable(getattr(r,'_receive_launch_go',None))
    marker=tmp_path/'released';read,write=os.pipe();code='''
import json,sys
from pathlib import Path
from egoemg_repro.classic_emg_report import _receive_launch_go,_pid_identity
print('ready',flush=True)
digest,channel=_receive_launch_go(int(sys.argv[1]))
Path(sys.argv[2]).write_text('released')
print(json.dumps(dict(digest=digest,channel=channel,process=_pid_identity(__import__('os').getpid(),phase='pipe-fixture'),torch_imported='torch' in sys.modules)),flush=True)
'''
    process=None
    try:
        process=subprocess.Popen([sys.executable,'-B','-c',code,str(read),str(marker)],pass_fds=(read,),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        os.close(read);read=None
        assert process.stdout.readline().strip()=='ready'
        assert process.poll() is None and not marker.exists()
        channel=os.fstat(write);digest='1'*64;os.write(write,b'G'+bytes.fromhex(digest));os.close(write);write=None
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode==0,stderr
        value=json.loads(stdout);assert value['digest']==digest and value['torch_imported'] is False
        assert value['channel']['inode']==channel.st_ino and value['process']['pid']==process.pid and marker.exists()
    finally:
        for fd in (read,write):
            if fd is not None:os.close(fd)
        if process is not None and process.poll() is None:process.kill();process.wait(timeout=10)


@pytest.mark.parametrize('profile',[{'passed':True},True,None])
def test_native_launch_cannot_spawn_from_caller_pass_flags(tmp_path,monkeypatch,profile):
    r=api()
    def forbidden(*args,**kwargs):raise AssertionError('spawn reached before actual calibration/termination')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    with pytest.raises(ValueError):r.run_native_test(tmp_path/'run',validation_profile=profile,
        output=tmp_path/'native',launch_output=tmp_path/'launch')
    assert not (tmp_path/'native').exists() and not (tmp_path/'launch').exists()


def native_sidecar_fixture(tmp_path,monkeypatch):
    """Raw-reader fixture: termination and checkpoint-role authority are stubbed, never CUDA."""
    torch=pytest.importorskip('torch');r,e,paths=raw_prediction_pair(tmp_path,monkeypatch)
    run=tmp_path/'trained';context=r._terminated_context(run);checkpoint={'state_dict':{'fixture':torch.tensor([1.])},
        'MixedPrecision':{'scale':65536.},'epoch':0}
    (run/'checkpoints').mkdir();(run/'checkpoints'/'last.ckpt').write_bytes(b'fixture last not a formal checkpoint')
    context['selection'].update(selected_state_sha256=e.artifacts._seal(checkpoint),
        selected_model_sha256=e.storage.state_digest(checkpoint['state_dict']),selected_event_sha256='4'*64,
        last_checkpoint_sha256=e.file_sha256(run/'checkpoints'/'last.ckpt'))
    context['manifest']['cohort_sha256']='5'*64
    context['manifest']['state_identity']['bindings'].update(cohort_receipt_sha256='6'*64,data_sha256='7'*64,partition_sha256='8'*64)
    context.update(best=checkpoint,original_best=dict(path=str(run/'checkpoints'/'best.ckpt'),present=False,file_sha256=None))
    def read_context(path):
        assert Path(path)==run
        return dict(context,guard=e.cohort._StatGuard([run/'result.json',run/'checkpoints'/'last.ckpt']))
    monkeypatch.setattr(r,'_terminated_context',read_context)
    directory=tmp_path/'native';directory.mkdir();selected=directory/'selected-best.ckpt';selected.write_bytes(b'fixture selected snapshot')
    def read_role(path,identity,role):
        assert Path(path)==selected and role=='best'
        return copy.deepcopy(checkpoint)
    monkeypatch.setattr(e.artifacts,'_read',read_role)
    identity=r._native_identity_record(context)
    reference=dict(format=e.FORMAT,scope='immutable_selected_best_reference',scientific_eligible=False,
        training_manifest_sha256=context['manifest']['sha256'],
        **{key:context['selection'][key] for key in ('selected_state_sha256','selected_model_sha256','selected_event_sha256')},
        checkpoint_name=selected.name,checkpoint_sha256=e.file_sha256(selected),original_best=context['original_best'])
    result=dict(format=e.FORMAT,scope=e.SCOPE,scientific_eligible=False,production_ready=False,
        identity_sha256=identity['sha256'],selection=reference,metrics={'test_loss':1.25},coverage=native_coverage_fixture(),
        original_test_step=True,optimizer_executions=0,model_state_unchanged=True,scaler_state_unchanged=True,
        prediction_pass='not executed; subsequent report pass remains separate',
        process=dict(pid=100,start_ticks=1000,phase='selected-best-test'))
    for name,value in [('identity',identity),('selected-best',reference),('result',result)]:
        (directory/(name+'.json')).write_text(json.dumps(value))
    status=dict(state='selected_best_test_finished',scope=e.SCOPE,scientific_eligible=False,production_ready=False,
        result_sha256=e.file_sha256(directory/'result.json'))
    (directory/'status.json').write_text(json.dumps(status))
    return r,e,run,directory,context,checkpoint


@pytest.mark.parametrize('damage',[None,'missing_snapshot','wrong_snapshot','last_snapshot','snapshot_state','snapshot_scaler',
    'selection','identity','counter','fake_bool','coverage','metric','status','process'])
def test_native_reader_requires_actual_selected_snapshot_and_resealed_complete_native_records(tmp_path,monkeypatch,damage):
    r,e,run,directory,context,checkpoint=native_sidecar_fixture(tmp_path,monkeypatch)
    result=json.loads((directory/'result.json').read_text());status=json.loads((directory/'status.json').read_text())
    if damage=='missing_snapshot':(directory/'selected-best.ckpt').unlink()
    elif damage=='wrong_snapshot':(directory/'selected-best.ckpt').write_bytes(b'changed')
    elif damage=='last_snapshot':
        result['selection']['checkpoint_name']='last.ckpt';(directory/'selected-best.json').write_text(json.dumps(result['selection']))
    elif damage=='snapshot_state':checkpoint['state_dict']['fixture']+=1
    elif damage=='snapshot_scaler':checkpoint['MixedPrecision']['scale']=32768.
    elif damage=='selection':
        result['selection']['selected_event_sha256']='9'*64;(directory/'selected-best.json').write_text(json.dumps(result['selection']))
    elif damage=='identity':
        identity=json.loads((directory/'identity.json').read_text());identity['policy']['batch_size']=1
        identity['sha256']=e.artifacts.json_hash({k:v for k,v in identity.items() if k!='sha256'})
        (directory/'identity.json').write_text(json.dumps(identity));result['identity_sha256']=identity['sha256']
    elif damage=='counter':result['optimizer_executions']=1
    elif damage=='fake_bool':result['model_state_unchanged']=1
    elif damage=='coverage':result['coverage']['samples']-=1
    elif damage=='metric':result['metrics']['test_loss']=float('inf')
    elif damage=='status':status['scientific_eligible']=0
    elif damage=='process':result['process']['phase']='unbound-test'
    (directory/'result.json').write_text(json.dumps(result));status['result_sha256']=e.file_sha256(directory/'result.json')
    (directory/'status.json').write_text(json.dumps(status))
    if damage:
        with pytest.raises((ValueError,FileNotFoundError)):r._read_native_test(run,directory)
    else:
        identity,value,guard=r._read_native_test(run,directory);guard.check()
        assert value==result and identity['policy']['test_samples']==4306
        assert directory/'selected-best.ckpt' in guard.paths


@pytest.mark.parametrize('artifact',[{'passed':True},True,None])
def test_native_reader_cannot_use_success_flags_as_immutable_native_sidecars(tmp_path,artifact):
    with pytest.raises(ValueError):api()._read_native_test(tmp_path/'run',artifact)


def native_launch_fixture(tmp_path,monkeypatch):
    """Binding schema fixture with stubbed training authority, not actual pre-test/GPU qualification."""
    r,e,run,native,context,checkpoint=native_sidecar_fixture(tmp_path,monkeypatch)
    identity=r._identity_record(run,context['manifest'],context['selection'],'val')
    first=tmp_path/'reference';second=tmp_path/'replay'
    for directory in (first,second):
        result=json.loads((directory/'result.json').read_text())
        result.update(identity_sha256=identity['sha256'],
            model_before_sha256=context['selection']['selected_model_sha256'],model_after_sha256=context['selection']['selected_model_sha256'],
            scaler_before_sha256=e.storage.state_digest(checkpoint['MixedPrecision']),scaler_after_sha256=e.storage.state_digest(checkpoint['MixedPrecision']))
        if directory==second:result['reference']['result_sha256']=e.file_sha256(first/'result.json')
        (directory/'identity.json').write_text(json.dumps(identity));(directory/'result.json').write_text(json.dumps(result))
        status=json.loads((directory/'status.json').read_text());status['result_sha256']=e.file_sha256(directory/'result.json')
        (directory/'status.json').write_text(json.dumps(status))
    profile,_=r._build_validation_profile(first,second);profile_path=tmp_path/'validation-profile.json'
    profile_path.write_text(json.dumps(profile));launch=tmp_path/'launch';launch.mkdir();manifest_path=launch/'launch.json'
    read,write=os.pipe()
    try:channel=r._pipe_identity(read)
    finally:os.close(read);os.close(write)
    parent=r._pid_identity(os.getppid(),phase='classic-native-test-parent')
    child=r._pid_identity(os.getpid(),phase='selected-best-test');directory_stat=launch.stat()
    command=[sys.executable,'-B','-m','egoemg_repro.classic_emg_report','_native-child','--launch',str(manifest_path),'--go-fd',str(channel['read_fd'])]
    value=dict(format=r.NATIVE_LAUNCH_FORMAT,scope='pre_test_process_profile_binding',scientific_eligible=False,production_ready=False,
        run=str(run),native_output=str(native),validation_profile=dict(path=str(profile_path),file_sha256=e.file_sha256(profile_path),sha256=profile['sha256']),
        identity_sha256=identity['sha256'],selection=context['selection'],sources=identity['sources'],runtime=identity['runtime'],
        test_policy=profile['test_policy'],directory_identity=[directory_stat.st_dev,directory_stat.st_ino],
        parent=parent,child=child,pipe=channel,command=command)
    value['sha256']=e.artifacts.json_hash(value);manifest_path.write_text(json.dumps(value))
    return r,e,run,native,manifest_path,value,channel


@pytest.mark.parametrize('damage',[None,'profile_hash','manifest_hash','identity','selection','sources','runtime','batch','directory',
    'same_process','parent_phase','child_phase','pipe','command','nested','science_flag'])
def test_pretest_manifest_reopens_profile_context_and_rejects_resealed_binding_corruption(tmp_path,monkeypatch,damage):
    r,e,run,native,path,value,channel=native_launch_fixture(tmp_path,monkeypatch)
    if damage=='profile_hash':value['validation_profile']['file_sha256']='0'*64
    elif damage=='identity':value['identity_sha256']='0'*64
    elif damage=='selection':value['selection']=dict(value['selection'],selected_event_sha256='0'*64)
    elif damage=='sources':value['sources']=dict(value['sources'],extra='0'*64)
    elif damage=='runtime':value['runtime']={'changed':True}
    elif damage=='batch':value['test_policy']=dict(value['test_policy'],batch_size=1)
    elif damage=='directory':value['directory_identity'][1]+=1
    elif damage=='same_process':value['child'].update(pid=value['parent']['pid'],start_ticks=value['parent']['start_ticks'])
    elif damage=='parent_phase':value['parent']['phase']='unbound'
    elif damage=='child_phase':value['child']['phase']='unbound'
    elif damage=='pipe':value['pipe']['protocol']='ordinary file'
    elif damage=='command':value['command'][4]='unqualified-entry'
    elif damage=='nested':value['native_output']=str(run/'test')
    elif damage=='science_flag':value['scientific_eligible']=0
    value['sha256']=e.artifacts.json_hash({k:v for k,v in value.items() if k!='sha256'})
    if damage=='manifest_hash':value['sha256']='0'*64
    path.write_text(json.dumps(value))
    if damage:
        with pytest.raises(ValueError):r._read_launch_manifest(path)
    else:
        got,profile,context,guard=r._read_launch_manifest(path);guard.check()
        assert got==value and profile['observed_exact_replay'] is True and context['run']==run
        assert path in guard.paths and tmp_path/'validation-profile.json' in guard.paths


@pytest.mark.parametrize('damage',[None,'go_hash','pipe','self','parent','argv'])
def test_child_binding_admission_checks_live_self_parent_argv_and_actual_pipe_before_any_native_call(tmp_path,monkeypatch,damage):
    """Metadata-only admission fixture; argv/training authorities stubbed, no CUDA or native forward."""
    r,e,run,native,path,value,channel=native_launch_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'orig_argv',value['command'].copy())
    go=value['sha256'];received=channel.copy()
    if damage=='go_hash':go='0'*64
    elif damage=='pipe':received['inode']+=1
    elif damage=='self':value['child']['start_ticks']+=1
    elif damage=='parent':value['parent']['start_ticks']+=1
    elif damage=='argv':monkeypatch.setattr(sys,'orig_argv',['python','other-entry'])
    if damage in ('self','parent'):
        value['sha256']=e.artifacts.json_hash({k:v for k,v in value.items() if k!='sha256'});go=value['sha256'];path.write_text(json.dumps(value))
    if damage:
        with pytest.raises(ValueError):r._admit_native_child(path,go,received)
    else:
        manifest,admission,guard=r._admit_native_child(path,go,received);guard.check()
        assert manifest==value and admission['process']['pid']==os.getpid()
        assert admission['manifest_sha256']==go and admission['scientific_eligible'] is False


def completed_native_launch_fixture(tmp_path,monkeypatch):
    """Historical receipt reader mechanics only; no native forward or CUDA qualification."""
    r,e,run,native,path,value,channel=native_launch_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'orig_argv',value['command'].copy())
    _,admission,_=r._admit_native_child(path,value['sha256'],channel)
    (path.parent/'child-admission.json').write_text(json.dumps(admission))
    native_result=json.loads((native/'result.json').read_text());native_result['process']=value['child']
    (native/'result.json').write_text(json.dumps(native_result))
    native_status=json.loads((native/'status.json').read_text());native_status['result_sha256']=e.file_sha256(native/'result.json')
    (native/'status.json').write_text(json.dumps(native_status))
    identity=json.loads((native/'identity.json').read_text())
    result=dict(format=r.NATIVE_LAUNCH_FORMAT,scope='profile_bound_original_native_test_finished',scientific_eligible=False,production_ready=False,
        manifest_sha256=value['sha256'],manifest_file_sha256=e.file_sha256(path),
        admission_file_sha256=e.file_sha256(path.parent/'child-admission.json'),native_identity_sha256=identity['sha256'],
        native_result_sha256=e.file_sha256(native/'result.json'),native_selected_checkpoint_sha256=e.file_sha256(native/'selected-best.ckpt'),
        parent=value['parent'],child=value['child'],child_exit_code=0,test_prediction_pass_executed=False)
    (path.parent/'result.json').write_text(json.dumps(result))
    (path.parent/'status.json').write_text(json.dumps(dict(state='profile_bound_native_test_finished',scope=r.NATIVE_LAUNCH_FORMAT,
        scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(path.parent/'result.json'))))
    return r,e,path,native,value,admission,result


@pytest.mark.parametrize('damage',[None,'missing_admission','admission','native_process','native_hash','snapshot_hash','exit_type','scope','status','parent'])
def test_finished_native_launch_reopens_admission_native_snapshot_and_exact_child_result(tmp_path,monkeypatch,damage):
    r,e,path,native,value,admission,result=completed_native_launch_fixture(tmp_path,monkeypatch)
    if damage=='missing_admission':(path.parent/'child-admission.json').unlink()
    elif damage=='admission':
        admission['process']['start_ticks']+=1
        admission['sha256']=e.artifacts.json_hash({k:v for k,v in admission.items() if k!='sha256'})
        (path.parent/'child-admission.json').write_text(json.dumps(admission));result['admission_file_sha256']=e.file_sha256(path.parent/'child-admission.json')
    elif damage=='native_process':
        record=json.loads((native/'result.json').read_text());record['process']['start_ticks']+=1
        (native/'result.json').write_text(json.dumps(record));status=json.loads((native/'status.json').read_text())
        status['result_sha256']=e.file_sha256(native/'result.json');(native/'status.json').write_text(json.dumps(status))
        result['native_result_sha256']=e.file_sha256(native/'result.json')
    elif damage=='native_hash':result['native_result_sha256']='0'*64
    elif damage=='snapshot_hash':result['native_selected_checkpoint_sha256']='0'*64
    elif damage=='exit_type':result['child_exit_code']=False
    elif damage=='scope':result['scientific_eligible']=0
    elif damage=='parent':result['parent']=dict(result['parent'],start_ticks=result['parent']['start_ticks']+1)
    (path.parent/'result.json').write_text(json.dumps(result));status=json.loads((path.parent/'status.json').read_text())
    status['result_sha256']=e.file_sha256(path.parent/'result.json')
    if damage=='status':status['state']='failed'
    (path.parent/'status.json').write_text(json.dumps(status))
    if damage:
        with pytest.raises((ValueError,FileNotFoundError)):r.verify_native_test_launch(path.parent)
    else:
        manifest,identity,record,guard=r.verify_native_test_launch(path.parent);guard.check()
        assert manifest==value and record['process']==value['child']
        assert native/'selected-best.ckpt' in guard.paths and path.parent/'child-admission.json' in guard.paths


@pytest.mark.parametrize('device',['cpu','cuda:1','bf16'])
def test_native_launch_rejects_wrong_device_before_any_directory_or_spawn(tmp_path,monkeypatch,device):
    r=api()
    def forbidden(*args,**kwargs):raise AssertionError('wrong-device launch reached a process')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    with pytest.raises(ValueError):r.run_native_test(tmp_path/'missing',validation_profile=tmp_path/'missing-profile',
        output=tmp_path/'native',launch_output=tmp_path/'launch',device=device)
    assert not (tmp_path/'native').exists() and not (tmp_path/'launch').exists()


@pytest.mark.parametrize('fail_before_go',[False,True])
def test_real_parent_publishes_actual_child_binding_before_go_and_revokes_failed_probe(tmp_path,monkeypatch,fail_before_go):
    """Real process/pipe failure probe only: worker exits17 and never calls native/CUDA."""
    r,e,run,native,old_launch,value,channel=native_launch_fixture(tmp_path,monkeypatch)
    native.rename(tmp_path/'saved-native');launch=tmp_path/'new-launch';marker=tmp_path/'probe-observed.json'
    profile=tmp_path/'validation-profile.json';processes=[];released=[]
    spawn=subprocess.Popen;write=e.artifacts._write_json;os_write=os.write;read_manifest=r._read_launch_manifest
    code='''
import hashlib,json,os,sys
from pathlib import Path
from egoemg_repro.classic_emg_report import _receive_launch_go,_pid_identity
go,channel=_receive_launch_go(int(sys.argv[2]))
manifest=json.loads(Path(sys.argv[1]).read_text())
actual=hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!='sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert actual==go==manifest['sha256']
assert manifest['child']==_pid_identity(os.getpid(),phase='selected-best-test')
assert manifest['pipe']==channel
Path(sys.argv[3]).write_text(json.dumps(dict(go=go,child=manifest['child'],profile=manifest['validation_profile'],torch_imported='torch' in sys.modules)))
sys.exit(17)
'''
    monkeypatch.setattr(r,'_native_command',lambda path,fd:[sys.executable,'-B','-c',code,str(path),str(fd),str(marker)])
    def capture(*args,**kwargs):
        process=spawn(*args,**kwargs);processes.append(process);return process
    def observe(fd,raw):
        if len(raw)==33 and raw[:1]==b'G':released.append(raw)
        return os_write(fd,raw)
    def manifest_after_go(path,*args,**kwargs):
        assert released,'Full-cohort verification belongs before spawn or after GO, not in the bounded child wait'
        return read_manifest(path,*args,**kwargs)
    def fail_write(directory,name,*args,**kwargs):
        if fail_before_go and directory.path==launch and name=='launch.json':raise OSError('injected before-GO publication failure')
        return write(directory,name,*args,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',capture);monkeypatch.setattr(r.os,'write',observe)
    monkeypatch.setattr(r,'_read_launch_manifest',manifest_after_go);monkeypatch.setattr(e.artifacts,'_write_json',fail_write)
    try:
        with pytest.raises(OSError if fail_before_go else RuntimeError):
            r.run_native_test(run,validation_profile=profile,output=native,launch_output=launch)
        assert len(processes)==1 and processes[0].poll() is not None
        assert json.loads((launch/'status.json').read_text())['state']=='failed' and not native.exists()
        if fail_before_go:assert not marker.exists() and not released
        else:
            observed=json.loads(marker.read_text());assert observed['torch_imported'] is False
            assert observed['child']['pid']==processes[0].pid and len(released)==1
            assert observed['profile']['file_sha256']==e.file_sha256(profile)
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait(timeout=10)


@pytest.mark.parametrize('command',['native-test','verify-native-test','_native-child'])
def test_native_cli_dispatches_only_the_explicit_bound_entry(monkeypatch,capsys,command):
    r=api();calls=[]
    def native(run,**kwargs):calls.append((run,kwargs));return {'kind':'native'}
    def verify(directory):calls.append(directory);return ({'sha256':'1'*64},None,None,None)
    def child(path,fd):calls.append((path,fd));return {'kind':'child'}
    monkeypatch.setattr(r,'run_native_test',native);monkeypatch.setattr(r,'verify_native_test_launch',verify)
    monkeypatch.setattr(r,'_native_test_child',child)
    arguments={'native-test':['--run','trained','--validation-profile','profile.json','--output','native','--launch-output','launch'],
        'verify-native-test':['--launch-output','launch'],'_native-child':['--launch','launch/launch.json','--go-fd','5']}
    r.main([command,*arguments[command]]);value=json.loads(capsys.readouterr().out)
    assert len(calls)==1
    if command=='native-test':
        assert calls[0]==(Path('trained'),dict(validation_profile=Path('profile.json'),output=Path('native'),launch_output=Path('launch'),device='cuda:0'))
        assert value['kind']=='native'
    elif command=='verify-native-test':assert value=={'format':r.NATIVE_LAUNCH_FORMAT,'sha256':'1'*64,'verified':True}
    else:assert value['kind']=='child' and calls[0]==(Path('launch/launch.json'),5)


@pytest.mark.parametrize('failure',[None,'before_rescue_close','after_rescue_close','late_source','renamed_directory'])
def test_native_parent_publication_file_mechanics_keep_revocation_through_final_closes(tmp_path,monkeypatch,failure):
    """Filesystem finalization with stubbed raw authorities; never native or CUDA qualification."""
    r,e,path,native,value,admission,result=completed_native_launch_fixture(tmp_path,monkeypatch)
    (path.parent/'result.json').unlink();(path.parent/'status.json').unlink()
    (path.parent/'progress.json').write_text('{"state":"running"}')
    owned=e.artifacts._Directory(path.parent,create=False);rescue=e.artifacts._Directory(path.parent,create=False)
    backup=r._retain_directory(rescue);close=e.artifacts._Directory.close;injected=[];publications={}
    identity=json.loads((native/'identity.json').read_text());guard=e.cohort._StatGuard([tmp_path/'trained'/'result.json'])
    def fail(directory):
        if directory is rescue and failure and not injected:
            injected.append(True)
            if failure=='before_rescue_close':raise OSError('injected native parent rescue close failure')
            close(directory)
            if failure=='after_rescue_close':raise OSError('injected native parent rescue close after actual release')
            if failure=='late_source':(tmp_path/'trained'/'result.json').write_text('{"changed":true}')
            if failure=='renamed_directory':
                path.parent.rename(tmp_path/'moved-launch');path.parent.mkdir();(path.parent/'status.json').write_text('foreign status')
            return
        return close(directory)
    monkeypatch.setattr(e.artifacts._Directory,'close',fail)
    try:
        if failure:
            with pytest.raises((OSError,ValueError)):r._finish_native_publication(path,value,identity,guard,owned,rescue,backup,publications)
            assert injected
            if failure=='renamed_directory':
                assert (path.parent/'status.json').read_text()=='foreign status'
                assert not (tmp_path/'moved-launch'/'status.json').exists()
            else:assert json.loads((path.parent/'status.json').read_text())['state']=='failed'
        else:
            assert r._finish_native_publication(path,value,identity,guard,owned,rescue,backup,publications)==result
            assert r.verify_native_test_launch(path.parent)[2]['process']==value['child']
    finally:
        for directory in (owned,rescue,backup):close(directory)
        r._discard_publications(publications)


@pytest.mark.parametrize('kind',['native','prediction'])
@pytest.mark.parametrize('replaced',['status.json','result.json'])
def test_review_failed_publication_preserves_foreign_file_and_invalidates_moved_owned_inode(tmp_path,monkeypatch,kind,replaced):
    """Real filesystem rollback; fixture authorities do not qualify CUDA or a scientific result."""
    from types import SimpleNamespace as NS
    if kind=='native':
        r,e,path,native,value,admission,result=completed_native_launch_fixture(tmp_path,monkeypatch)
        output=path.parent;identity=json.loads((native/'identity.json').read_text())
        (output/'result.json').unlink();(output/'status.json').unlink()
        (output/'progress.json').write_text('{"state":"running"}')
    else:
        r,e,paths=raw_prediction_pair(tmp_path,monkeypatch);output=tmp_path/'new-reference'
        output.mkdir();identity=json.loads((paths[0]/'identity.json').read_text())
        result=json.loads((paths[0]/'result.json').read_text())
        (output/'identity.json').write_text(json.dumps(identity));(output/'progress.json').write_text('{"state":"running"}')
    owned=e.artifacts._Directory(output,create=False);rescue=e.artifacts._Directory(output,create=False)
    backup=r._retain_directory(rescue);close=e.artifacts._Directory.close;injected=[];publications={}
    guard=e.cohort._StatGuard([tmp_path/'trained'/'result.json']);moved=output/('moved-'+replaced)
    def substitute(directory):
        close(directory)
        if directory is rescue and not injected:
            injected.append(True);(output/replaced).rename(moved);(output/replaced).write_text('foreign replacement bytes')
    monkeypatch.setattr(e.artifacts._Directory,'close',substitute)
    try:
        with pytest.raises(ValueError):
            if kind=='native':r._finish_native_publication(path,value,identity,guard,owned,rescue,backup,publications)
            else:r._finish_prediction_publication(NS(guard=guard),output,identity,result,owned,rescue,backup,publications)
        assert injected
        assert (output/replaced).read_text()=='foreign replacement bytes'
        assert moved.read_bytes()==b''
        with pytest.raises((ValueError,FileNotFoundError)):
            if kind=='native':r.verify_native_test_launch(output)
            else:r._read_prediction_pass(output,role='val')
    finally:
        for directory in (owned,rescue,backup):close(directory)
        r._discard_publications(publications)


@pytest.mark.parametrize('damage',[None,'target','mask','sample','grouping','count','variant'])
def test_test_prediction_must_match_the_complete_consumed_native_inputs(tmp_path,monkeypatch,damage):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch,role='test')
    coverage=json.loads((paths[0]/'result.json').read_text())['coverage']
    keys=('samples','batch_sizes','coverage_sha256','target_sha256','mask_sha256','sample_sha256','fingerprint_encoding','fingerprint_scope')
    native={k:copy.deepcopy(coverage[k]) for k in keys};native.update(batch_weighted_loss=1.25,test_loss=1.25)
    if damage in ('target','mask','sample'):native[damage+'_sha256']='0'*64
    elif damage=='grouping':native['batch_sizes']=[1]*4306
    elif damage=='count':native['samples']=4305
    assert callable(getattr(r,'compare_native_prediction',None)),'native-to-prediction full input comparison missing'
    if damage:
        with pytest.raises(ValueError):r.compare_native_prediction(native,coverage,variant='neuropose' if damage=='variant' else 'vemg2pose')
    else:
        value=r.compare_native_prediction(native,coverage,variant='vemg2pose')
        assert value['samples']==4306 and value['native_target_mask_sample_equal'] is True
        assert value['native_prediction_hash_available'] is False


@pytest.mark.parametrize('kind',['reference','replay'])
def test_test_prediction_raw_reader_rejects_ungated_validation_shaped_artifacts(tmp_path,monkeypatch,kind):
    r,e,paths=raw_prediction_pair(tmp_path,monkeypatch,role='test')
    directory=paths[0 if kind=='reference' else 1]
    with pytest.raises(ValueError):r._read_prediction_pass(directory,role='test',expected_kind=kind)


@pytest.mark.parametrize('profile',[{'passed':True},None,True])
def test_test_prediction_requires_real_completed_native_bundle_before_model_preparation(tmp_path,monkeypatch,profile):
    r=api();assert callable(getattr(r,'run_test_prediction',None)),'gated test prediction entry missing'
    def forbidden(*args,**kwargs):raise AssertionError('no evaluation runtime before typed native bundle path')
    monkeypatch.setattr(r,'_evaluation',forbidden)
    with pytest.raises(ValueError):r.run_test_prediction(tmp_path/'run',output=tmp_path/'prediction',evaluation_profile=profile)


def test_independent_recompute_requires_an_explicit_completed_prediction_reference(tmp_path):
    r=api();assert callable(getattr(r,'recompute_selected_best',None)),'independent test prediction recomputation entry missing'
    with pytest.raises((TypeError,ValueError)):
        r.recompute_selected_best(tmp_path/'run',device='cuda:0',output=tmp_path/'recompute',evaluation_profile=tmp_path/'launch')


def original_test_statistics_fixture(variant='vemg2pose'):
    """Coherent synthetic statistics schema, never inference or actual cohort evidence."""
    length,center=(6000,2105) if variant=='vemg2pose' else (7790,3895)
    r=api();report=r.ClassicAngleReport(variant);p=np.ones((22,length),np.float16)
    target=np.zeros((22,length),np.float32);mask=np.zeros(length,bool);mask[center]=True
    for hand in ('left','right'):report.update(p,target,mask,hand=hand,split='user',user='fixture-private-user')
    value=report.report();value['samples']=4306;scale=2153
    for mode in ('trajectory','center'):
        m=value[mode];m['invalid_angle_count']*=scale
        records=[m['overall'],*m['by_joint']]
        for key in ('by_hand','by_split','by_user','by_component','by_finger','by_pd_group'):records.extend(m[key].values())
        records.extend(record for splits in m['by_hand_split'].values() for record in splits.values())
        for record in records:
            record['absolute_error_sum_rad']*=scale;record['valid_angle_count']*=scale
        empty={'absolute_error_sum_rad':0.,'valid_angle_count':0,'mae_deg':None}
        for split in ('gesture','both'):
            m['by_split'][split]=empty.copy()
            for hand in ('left','right'):m['by_hand_split'][hand][split]=empty.copy()
    return value


@pytest.mark.parametrize('damage',[None,'samples','alignment','alignment_bool','sum','count_bool','mean','nonfinite','invalid_count',
    'hand_split','component','joint','finger','pd','user_dispersion','missing_statistics'])
def test_saved_test_statistics_require_original_alignment_typed_sums_counts_and_conservation(damage):
    r=api();value=original_test_statistics_fixture()
    if damage=='samples':value['samples']=4305
    elif damage=='alignment':value['alignment']['output_center']=3000
    elif damage=='alignment_bool':value['alignment']['right_context']=False
    elif damage=='sum':value['center']['overall']['absolute_error_sum_rad']+=1.
    elif damage=='count_bool':value['center']['overall']['valid_angle_count']=True
    elif damage=='mean':value['center']['overall']['mae_deg']+=1.
    elif damage=='nonfinite':value['trajectory']['by_user']['fixture-private-user']['absolute_error_sum_rad']=float('nan')
    elif damage=='invalid_count':value['trajectory']['invalid_angle_count']-=1
    elif damage=='hand_split':value['center']['by_hand_split']['left']['user']['valid_angle_count']-=22
    elif damage=='component':value['center']['by_component']['wrist']['valid_angle_count']-=2
    elif damage=='joint':value['center']['by_joint'][0]['name']='wrist_yaw'
    elif damage=='finger':value['trajectory']['by_finger']['thumb']['absolute_error_sum_rad']+=1.
    elif damage=='pd':value['trajectory']['by_pd_group']['mid']['valid_angle_count']-=1
    elif damage=='user_dispersion':value['center']['user_summary']['std_mae_deg']=1.
    elif damage=='missing_statistics':value=None
    assert callable(getattr(r,'validate_test_statistics',None)),'complete original report statistics reader missing'
    if damage:
        with pytest.raises(ValueError):r.validate_test_statistics(value,variant='vemg2pose')
    else:
        r.validate_test_statistics(value,variant='vemg2pose')
        assert value['center']['overall']['valid_angle_count']==94732
        assert value['trajectory']['overall']['absolute_error_sum_rad']==94732.


def raw_test_prediction_pair(tmp_path,monkeypatch):
    """Raw test-report reader fixture; no true termination/checkpoint/CUDA authority."""
    r,e,path,native,launch,admission,parent=completed_native_launch_fixture(tmp_path,monkeypatch)
    run=tmp_path/'trained';context=r._terminated_context(run)
    identity=r._identity_record(run,context['manifest'],context['selection'],'test');policy=identity['policy']
    coverage=completed_ledger();n=4306;batch=600;coverage.update(role='test',samples=n,
        batch_sizes=[min(batch,n-i) for i in range(0,n,batch)],predictions=[])
    order=hashlib.sha256()
    for i in range(n):order.update(i.to_bytes(8,'big'))
    coverage['coverage_sha256']=order.hexdigest();chains={name:hashlib.sha256() for name in r.ReportInputLedger.CHAINS}
    for sequence,size in enumerate(coverage['batch_sizes']):
        record=dict(sequence=sequence,batch_size=size,dtype='torch.float16',shape=[size,22,6000])
        for name,chain in chains.items():
            digest=e.artifacts.json_hash([name,'test',sequence]);record[name+'_sha256']=digest
            chain.update(bytes.fromhex(e.artifacts.json_hash(dict(sequence=sequence,batch_size=size,digest=digest))))
        coverage['predictions'].append(record)
    coverage.update({name+'_sha256':chain.hexdigest() for name,chain in chains.items()})
    native_result=json.loads((native/'result.json').read_text())
    for key in ('samples','batch_sizes','coverage_sha256','target_sha256','mask_sha256','sample_sha256','fingerprint_encoding','fingerprint_scope'):
        native_result['coverage'][key]=copy.deepcopy(coverage[key])
    (native/'result.json').write_text(json.dumps(native_result));status=json.loads((native/'status.json').read_text())
    status['result_sha256']=e.file_sha256(native/'result.json');(native/'status.json').write_text(json.dumps(status))
    parent['native_result_sha256']=e.file_sha256(native/'result.json');(path.parent/'result.json').write_text(json.dumps(parent))
    status=json.loads((path.parent/'status.json').read_text());status['result_sha256']=e.file_sha256(path.parent/'result.json')
    (path.parent/'status.json').write_text(json.dumps(status))
    binding=dict(directory=str(path.parent),manifest_sha256=launch['sha256'],result_sha256=e.file_sha256(path.parent/'result.json'),
        native_result_sha256=e.file_sha256(native/'result.json'));paths=[]
    for i,kind in enumerate(('reference','replay')):
        directory=tmp_path/('test-'+kind);directory.mkdir();paths.append(directory)
        result=dict(coverage=copy.deepcopy(coverage),statistics=original_test_statistics_fixture(),
            model_before_sha256=context['selection']['selected_model_sha256'],model_after_sha256=context['selection']['selected_model_sha256'],
            scaler_before_sha256=e.storage.state_digest(context['best']['MixedPrecision']),scaler_after_sha256=e.storage.state_digest(context['best']['MixedPrecision']),
            format=r.PASS_FORMAT,scope=policy['scope'],scientific_eligible=False,production_ready=False,identity_sha256=identity['sha256'],
            process=dict(pid=10001+i,start_ticks=8001+i,phase='classic-test-'+kind),pass_kind=kind,evaluation_profile=copy.deepcopy(binding),
            reference=None if i==0 else dict(directory=str(paths[0]),result_sha256=e.file_sha256(paths[0]/'result.json')))
        (directory/'identity.json').write_text(json.dumps(identity));(directory/'result.json').write_text(json.dumps(result))
        (directory/'status.json').write_text(json.dumps(dict(state='prediction_pass_finished',scope=r.PASS_FORMAT,
            scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(directory/'result.json'))))
    return r,e,paths,path,native,launch


@pytest.mark.parametrize('damage',[None,'binding_hash','missing_binding','input_target','input_mask','input_sample',
    'prediction','metadata','dtype','statistics','no_statistics','same_process','native_process','profile_file','native_snapshot'])
def test_test_prediction_reader_reopens_native_and_reference_and_rejects_resealed_changes(tmp_path,monkeypatch,damage):
    r,e,paths,path,native,launch=raw_test_prediction_pair(tmp_path,monkeypatch)
    directory=paths[1];result=json.loads((directory/'result.json').read_text());coverage=result['coverage']
    if damage=='binding_hash':result['evaluation_profile']['manifest_sha256']='f'*64
    elif damage=='missing_binding':result.pop('evaluation_profile')
    elif damage in ('input_target','input_mask','input_sample','prediction','metadata'):
        key=damage.removeprefix('input_');coverage['predictions'][0][key+'_sha256']='f'*64
        chain=hashlib.sha256()
        for record in coverage['predictions']:
            chain.update(bytes.fromhex(e.artifacts.json_hash(dict(sequence=record['sequence'],batch_size=record['batch_size'],digest=record[key+'_sha256']))))
        coverage[key+'_sha256']=chain.hexdigest()
    elif damage=='dtype':coverage['predictions'][0]['dtype']='torch.float32'
    elif damage=='statistics':result['statistics']['center']['overall']['valid_angle_count']-=22
    elif damage=='no_statistics':result['statistics']=None
    elif damage=='same_process':result['process'].update(pid=10001,start_ticks=8001)
    elif damage=='native_process':result['process'].update(pid=launch['child']['pid'],start_ticks=launch['child']['start_ticks'])
    elif damage=='profile_file':(tmp_path/'validation-profile.json').write_text('{}')
    elif damage=='native_snapshot':(native/'selected-best.ckpt').write_bytes(b'changed snapshot')
    (directory/'result.json').write_text(json.dumps(result));status=json.loads((directory/'status.json').read_text())
    status['result_sha256']=e.file_sha256(directory/'result.json');(directory/'status.json').write_text(json.dumps(status))
    if damage:
        with pytest.raises(ValueError):r._read_prediction_pass(directory,role='test',expected_kind='replay')
    else:
        identity,got,guard=r._read_prediction_pass(directory,role='test',expected_kind='replay');guard.check()
        assert got['statistics']['center']['overall']['valid_angle_count']==94732
        assert paths[0]/'result.json' in guard.paths and path.parent/'launch.json' in guard.paths
        assert native/'selected-best.ckpt' in guard.paths


def test_test_prediction_cli_cannot_use_a_validation_profile_as_completed_native_admission(tmp_path,monkeypatch):
    r,e,path,native,launch,admission,parent=completed_native_launch_fixture(tmp_path,monkeypatch)
    output=tmp_path/'not-created'
    with pytest.raises((ValueError,FileNotFoundError,NotADirectoryError)):
        r.main(['test-pass','--run',str(tmp_path/'trained'),'--evaluation-profile',str(tmp_path/'validation-profile.json'),
            '--output',str(output)])
    assert not output.exists()


def test_verify_test_prediction_cli_reopens_raw_bound_evidence_and_keeps_engineering_flags(tmp_path,monkeypatch,capsys):
    r,e,paths,path,native,launch=raw_test_prediction_pair(tmp_path,monkeypatch)
    r.main(['verify-test-pass','--output',str(paths[1])]);value=json.loads(capsys.readouterr().out)
    assert value==dict(format=r.PASS_FORMAT,verified=True,role='test',pass_kind='replay',samples=4306,scientific_eligible=False)


def status_journal_fixture(tmp_path):
    """Actual bounded journal IO with synthetic metadata; never optimizer or CUDA evidence."""
    pytest.importorskip('pytorch_lightning');r=api();e=r._evaluation();run=tmp_path/'run';run.mkdir()
    manifest=dict(format=e.formal.FORMAT,scope='formal_engineering_acceptance',scientific_eligible=False,
        state_identity=dict(sha256='a'*64,counts=dict(train=1200,val=1576,test=4306)),
        policy=e.formal.formal_policy(variant='vemg2pose',engineering=True),sources=e.formal._source_snapshot())
    manifest['sha256']=e.artifacts.json_hash(manifest)
    (run/'identity.json').write_text(json.dumps(manifest));(run/'status.json').write_text('{"state":"engineering_finished"}')
    with e.artifacts.FormalJournal(run/'journal',identity_sha256='a'*64,train_samples=1200,batch_size=600) as journal:
        journal.append_attempt(dict(epoch=0,batch=0,pl_step=1,optimizer_step_before=0,optimizer_step_after=1,
            optimizer_executed=True,scale_before=65536.,scale_after=65536.,gradient_finite=True,gradient_abs_max=.125,loss=1.))
        journal.append_attempt(dict(epoch=0,batch=1,pl_step=2,optimizer_step_before=1,optimizer_step_after=1,
            optimizer_executed=False,scale_before=65536.,scale_after=32768.,gradient_finite=False,gradient_abs_max=None,loss=1.))
        digest=hashlib.sha256()
        for i in range(1576):digest.update(i.to_bytes(8,'big'))
        journal.append_epoch(dict(epoch=0,full=True,samples=1576,expected_samples=1576,val_loss=1.25,
            batch_weighted_loss=1.25,coverage_sha256=digest.hexdigest()))
    return r,e,run,manifest


@pytest.mark.parametrize('damage',[None,'partial_tail','chain','header','gap','source','validation_count'])
def test_readonly_classic_status_replays_bounded_actual_journal_without_model_or_liveness_claim(tmp_path,monkeypatch,damage):
    r,e,run,manifest=status_journal_fixture(tmp_path);segment=run/'journal'/'events-00000.jsonl'
    if damage=='partial_tail':
        with segment.open('ab') as f:f.write(b'{"uncommitted":')
    elif damage=='chain':segment.write_bytes(segment.read_bytes().replace(b'"loss":1.0',b'"loss":2.0',1))
    elif damage=='header':(run/'journal'/'journal.json').write_text('{}')
    elif damage=='gap':segment.rename(run/'journal'/'events-00001.jsonl')
    elif damage=='source':
        manifest['sources']['egoemg_repro/classic_emg_formal.py']='0'*64
        manifest['sha256']=e.artifacts.json_hash({k:v for k,v in manifest.items() if k!='sha256'})
        (run/'identity.json').write_text(json.dumps(manifest))
    elif damage=='validation_count':
        manifest['state_identity']['counts']['val']=1575
        manifest['sha256']=e.artifacts.json_hash({k:v for k,v in manifest.items() if k!='sha256'})
        (run/'identity.json').write_text(json.dumps(manifest))
    def forbidden(*args,**kwargs):raise AssertionError('readonly status must not construct a model, probe CUDA or evaluate')
    monkeypatch.setattr(e.cuda,'_factory',forbidden);monkeypatch.setattr(e,'prepare_selected_best_evaluation',forbidden)
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',forbidden);monkeypatch.setattr(torch.cuda,'init',forbidden)
    before={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    assert callable(getattr(r,'build_classic_status',None)),'read-only journal status API missing'
    if damage not in (None,'partial_tail'):
        with pytest.raises(ValueError):r.build_classic_status(run)
    else:
        value=r.build_classic_status(run)
        assert value['attempts']==2 and value['optimizer_executions']==1 and value['overflow_skips']==1
        assert value['completed_validation_epochs']==1 and value['last_validation_loss']==1.25
        assert value['partial_tail_ignored'] is (damage=='partial_tail')
        assert value['reported_state']=='engineering_finished'
        assert value['scientific_eligible'] is False and value['training_completion_verified'] is False
        assert value['process_liveness_verified'] is False
        assert str(run) not in json.dumps(value)
    after={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    assert before==after


def public_record_fixture():
    """Formatter input only, not admitted training/GPU evidence."""
    r=api();selection=dict(stop_reason='early-stopping',completed_epochs=52,best_epoch=1,best_val_loss=1.1,
        selected_state_sha256='1'*64,selected_model_sha256='2'*64,selected_event_sha256='3'*64,last_checkpoint_sha256='4'*64)
    recipe=dict(variant='vemg2pose',upstream_experiment='config/experiment/example.yaml',window_length=7790,
        train_stride=780,eval_stride=7790,val_fraction=.12,seed=42,jitter=False,batch_size=600,num_workers=12,
        max_epochs=100,lr=.0001,weight_decay=.01,eta_min=0.,gradient_clip=0,precision='16-mixed',
        monitor='val_loss',monitor_mode='min',early_stopping_patience=50,augmentation='original_resolved_rotation',
        layout=dict(declared='declared16',effective='target8',input_channels=8,channel_indices=list(range(8)),channel_interpolate=False,reason='fixture'),
        normalization=dict(mean=0.,std=1.,epsilon=1e-8),mask_policy='official_literal_loader',
        raw_loss_weights={'mae':1.,'landmark/fingertip':.01,'fingertip_distance':.01},
        effective_loss_weights={'mae':1.,'landmark/fingertip':.02})
    runtime=dict(runtime_profile='canonical_backend_nondeterministic_v1',versions={'torch':'2.2.2'},python='3.11.12',
        torch_build='2.2.2+cu118',cuda_build='11.8',device_name='fixture device',matmul_precision='high',
        precision_plugin='MixedPrecision',scaler_enabled=True,deterministic=False,cudnn_deterministic=False,cudnn_benchmark=False,
        torch_config='/private/build/path',visible_uuid='PRIVATE-MACHINE-ID')
    identity=dict(run='/private/training/path',sha256='5'*64,training_manifest_sha256='6'*64,training_result_sha256='7'*64,
        selection=selection,policy=r.report_pass_policy(variant='vemg2pose',role='test'),runtime=runtime,
        sources={'egoemg_repro/classic_emg_report.py':'8'*64},cohort_bindings=dict(data_sha256='9'*64,partition_sha256='a'*64,
            normalization_sha256='b'*64,cohort_receipt_sha256='c'*64,numeric_qualification_artifact_sha256='d'*64,
            private_hint_sha256='/private/binding-hint'))
    manifest=dict(state_identity=dict(recipe=recipe,effective=dict(precision='16-mixed',batch_size=600,global_batch_size=600,
        world_size=1,rank=0,num_workers=0,train_shuffle=True,seed=42,log_every_n_steps=50,max_epochs=100,clip_value=0),
        counts=dict(train=101714,val=1576,test=4306)),adaptations=dict(declared_devices=[0,1,2,3,4,5],effective_devices=[0],
            world_size=1,declared_per_rank_batch=600,effective_per_rank_batch=600,effective_global_batch=600,
            declared_num_workers=12,effective_num_workers=0,private_path='/private/secret'))
    training=dict(completed_epochs=52,attempts=8840,optimizer_executions=8838,natural_overflows=2,stop_reason='early-stopping',
        interrupted=False,selection_scope='terminated-full-validation-loss',
        runtime_profile='canonical_backend_nondeterministic_v1',trajectory_agreement_qualified=False,
        bitwise_trajectory_guaranteed=False,process={'private':'must not copy'})
    prediction=dict(statistics=original_test_statistics_fixture(),coverage={'samples':4306,'batch_sizes':[600]*7+[106],
        **{name+'_sha256':'e'*64 for name in r.ReportInputLedger.CHAINS},'coverage_sha256':'f'*64},
        reference={'directory':'/private/reference','result_sha256':'0'*64})
    native={'metrics':{'test_loss':1.25,'test_mae':1.,'test_landmark/all':2.}}
    binding={'directory':'/private/native-launch','manifest_sha256':'1'*64,'result_sha256':'2'*64,'native_result_sha256':'3'*64}
    return dict(identity=identity,prediction=prediction,manifest=manifest,training=training,native=native,
        subjects=['fixture-private-user','private-unused-user'],data_summary=dict(source_subject_count=2,test_subject_count=1,
            train_test_subject_overlap=dict(user=1,gesture=0,both=0)),binding=binding,recompute_result_sha256='4'*64)


def test_public_completed_formatter_keeps_counts_units_selection_and_excludes_private_identity():
    r=api();assert callable(getattr(r,'_public_completed_record',None)),'sanitized classic result formatter missing'
    value=r._public_completed_record(**public_record_fixture())
    assert value['statistics']['center']['overall']['valid_angle_count']==94732
    assert value['statistics']['center']['overall']['mae_deg']==pytest.approx(180/math.pi)
    assert set(value['statistics']['center']['by_user'])=={'participant_001'}
    assert value['selection']['best_epoch']==1 and value['training']['optimizer_executions']==8838
    assert value['protocol']['recipe']['effective_loss_weights']['landmark/fingertip']==.02
    assert value['protocol']['effective']['world_size']==1 and value['protocol']['adaptations']['declared_devices']==[0,1,2,3,4,5]
    assert value['native_metrics']['values']['test_mae']==1. and value['native_metrics']['values']['test_landmark/all']==2.
    assert value['full_paper_complete'] is False and value['completion_requires_owned_capsule'] is True
    assert value['evidence']['recompute_result_sha256']=='4'*64
    assert value['data']['train_test_subject_overlap']['user']==1
    raw=json.dumps(value)
    for private in ('/private/','fixture-private-user','private-unused-user','PRIVATE-MACHINE-ID','torch_config','visible_uuid'):
        assert private not in raw
    assert callable(getattr(r,'render_classic_completed',None)),'public metrics Markdown renderer missing'
    markdown=r.render_classic_completed(value)
    assert '94732' in markdown and 'THUMB_CMC_FE' in markdown and value['sha256'] in markdown
    assert '/private/' not in markdown and 'fixture-private-user' not in markdown


@pytest.mark.parametrize('variant,center,length',[('vemg2pose',2105,6000),('neuropose',3895,7790)])
def test_completed_markdown_carries_the_same_unrounded_public_evidence_for_both_original_alignments(variant,center,length):
    """Omitting any protocol/statistics/evidence field from Markdown must fail this contract."""
    r=api();packet=public_record_fixture();batch=600 if variant=='vemg2pose' else 256
    packet['identity']['policy']=r.report_pass_policy(variant=variant,role='test')
    state=packet['manifest']['state_identity'];state['recipe'].update(variant=variant,batch_size=batch)
    state['effective'].update(batch_size=batch,global_batch_size=batch)
    packet['manifest']['adaptations'].update(declared_per_rank_batch=batch,effective_per_rank_batch=batch,effective_global_batch=batch)
    packet['prediction']['statistics']=original_test_statistics_fixture(variant)
    packet['prediction']['coverage']['batch_sizes']=[min(batch,4306-i) for i in range(0,4306,batch)]
    report=r._public_completed_record(**packet);markdown=r.render_classic_completed(report)
    assert '```json\n' in markdown,'Markdown must carry the complete public aggregate evidence'
    evidence=json.loads(markdown.split('```json\n',1)[1].split('\n```',1)[0])
    assert evidence==report
    assert evidence['statistics']['alignment']['output_center']==center
    assert evidence['statistics']['center']['overall']['valid_angle_count']==94732
    assert evidence['statistics']['trajectory']['invalid_angle_count']==4306*22*(length-1)
    assert evidence['statistics']['center']['by_hand_split']['left']['user']['valid_angle_count']==47366
    assert evidence['statistics']['center']['user_summary']['std_mae_deg']==0.
    assert evidence['training']['stop_reason']=='early-stopping'
    assert evidence['protocol']['recipe']['effective_loss_weights']['landmark/fingertip']==.02
    assert evidence['data']['train_test_subject_overlap']=={'user':1,'gesture':0,'both':0}
    assert evidence['sources_sha256']=={'egoemg_repro/classic_emg_report.py':'8'*64}
    assert evidence['native_metrics']['angular_units']=='radians'
    assert '/private/' not in markdown and 'fixture-private-user' not in markdown


@pytest.mark.parametrize('damage',['unknown_native','unknown_subject','forged_statistics'])
def test_public_completed_formatter_does_not_silently_copy_unrecognized_private_or_statistical_fields(damage):
    r=api();assert callable(getattr(r,'_public_completed_record',None)),'sanitized classic result formatter missing'
    packet=public_record_fixture()
    if damage=='unknown_native':packet['native']['metrics']['test_/private/identity']=1.
    elif damage=='unknown_subject':packet['subjects']=['unrelated-private-user']
    elif damage=='forged_statistics':packet['prediction']['statistics']['center']['by_joint'][0]['valid_angle_count']=False
    with pytest.raises(ValueError):r._public_completed_record(**packet)


@pytest.mark.parametrize('artifact',[{'passed':True},None,True])
def test_completed_builder_cannot_accept_flags_instead_of_real_recompute_artifacts(tmp_path,monkeypatch,artifact):
    r=api();assert callable(getattr(r,'build_classic_completed',None)),'raw-bound completed report builder missing'
    def forbidden(*args,**kwargs):raise AssertionError('no runtime before real artifact paths')
    monkeypatch.setattr(r,'_evaluation',forbidden)
    with pytest.raises(ValueError):r.build_classic_completed(tmp_path/'run',artifact)


@pytest.mark.parametrize('failure',[None,'first_close','last_close','source_change','foreign_status','foreign_report','whole_directory',
    'foreign_capsule_after_publish','capsule_cleanup'])
def test_completion_capsule_is_last_and_rollback_preserves_foreign_files(tmp_path,monkeypatch,failure):
    """Real filesystem writer with stubbed evidence authority; never GPU/scientific qualification."""
    pytest.importorskip('pytorch_lightning');r=api()
    assert callable(getattr(r,'_public_completed_record',None)),'sanitized classic result formatter missing'
    assert callable(getattr(r,'_finish_completed_publication',None)),'owned completed report publication missing'
    e=r._evaluation();report=r._public_completed_record(**public_record_fixture());output=tmp_path/'public'
    owned=e.artifacts._Directory(output,create=True);rescue=e.artifacts._Directory(output,create=False)
    backup=r._retain_directory(rescue);publications={};close=e.artifacts._Directory.close;injected=[];order=[]
    source=tmp_path/'immutable-source';source.write_text('unchanged')
    def evidence(run,recompute):return report,e.cohort._StatGuard([source]),[]
    monkeypatch.setattr(r,'_completed_evidence',evidence)
    original=getattr(r,'_publish_profile_file');write=getattr(r,'_publish_bytes_file',None)
    assert callable(write),'shared creation-owned byte writer missing'
    def publish(directory,name,*args,**kwargs):
        if name=='completion.json':
            assert order==['owned_closed','rescue_closed']
            assert (output/'metrics.json').exists() and (output/'metrics.md').exists()
        value=original(directory,name,*args,**kwargs)
        if name=='completion.json' and failure=='foreign_capsule_after_publish':
            injected.append(True);(output/name).rename(output/('moved-'+name));(output/name).write_text('foreign bytes')
        return value
    monkeypatch.setattr(r,'_publish_profile_file',publish)
    unlink=r.os.unlink
    def cleanup(name,*args,**kwargs):
        value=unlink(name,*args,**kwargs)
        if failure=='capsule_cleanup' and str(name).startswith('.completion.json.') and str(name).endswith('.incomplete') and not injected:
            injected.append(True);raise OSError('post-link capsule temporary cleanup')
        return value
    monkeypatch.setattr(r.os,'unlink',cleanup)
    def fail(directory):
        if directory is owned:
            close(directory);order.append('owned_closed')
            if failure=='first_close' and not injected:injected.append(True);raise OSError('first semantic close')
            return
        if directory is rescue and not injected:
            close(directory);order.append('rescue_closed')
            if failure and failure not in ('foreign_capsule_after_publish','capsule_cleanup'):
                injected.append(True)
                if failure=='last_close':raise OSError('last semantic close')
                if failure=='source_change':source.write_text('changed')
                if failure in ('foreign_status','foreign_report'):
                    name='completion.json' if failure=='foreign_status' else 'metrics.json'
                    if (output/name).exists():(output/name).rename(output/('moved-'+name))
                    (output/name).write_text('foreign bytes')
                if failure=='whole_directory':output.rename(tmp_path/'moved-output');output.mkdir();(output/'completion.json').write_text('foreign bytes')
            return
        return close(directory)
    monkeypatch.setattr(e.artifacts._Directory,'close',fail)
    try:
        if failure:
            with pytest.raises((OSError,ValueError)):r._finish_completed_publication(tmp_path/'run',tmp_path/'recompute',output,report,
                e.cohort._StatGuard([source]),owned,rescue,backup,publications)
            assert injected
            if failure in ('foreign_status','foreign_report','whole_directory','foreign_capsule_after_publish'):
                name='metrics.json' if failure=='foreign_report' else 'completion.json'
                assert (output/name).read_text()=='foreign bytes'
                moved=output/('moved-'+name)
                if moved.exists():assert moved.read_bytes()==b''
            else:assert not (output/'completion.json').exists()
        else:
            r._finish_completed_publication(tmp_path/'run',tmp_path/'recompute',output,report,
                e.cohort._StatGuard([source]),owned,rescue,backup,publications)
            assert order==['owned_closed','rescue_closed']
            got,guard=r.verify_classic_completed(tmp_path/'run',tmp_path/'recompute',output);guard.check()
            assert got==report
            capsule=json.loads((output/'completion.json').read_text())
            assert capsule['metrics_json_sha256']==e.file_sha256(output/'metrics.json')
            assert capsule['metrics_md_sha256']==e.file_sha256(output/'metrics.md')
    finally:
        for directory in (owned,rescue,backup):close(directory)
        r._discard_publications(publications)


def test_status_cli_reads_actual_journal_without_evaluation(tmp_path,monkeypatch,capsys):
    r,e,run,manifest=status_journal_fixture(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('status CLI must not prepare a selected model')
    monkeypatch.setattr(e,'prepare_selected_best_evaluation',forbidden)
    r.main(['status','--run',str(run)]);value=json.loads(capsys.readouterr().out)
    assert value['optimizer_executions']==1 and value['overflow_skips']==1 and value['scientific_eligible'] is False


def test_publish_cli_cannot_create_completion_without_raw_recompute_chain(tmp_path,monkeypatch):
    pytest.importorskip('pytorch_lightning');r=api();e=r._evaluation();output=tmp_path/'not-created'
    def forbidden(*args,**kwargs):raise AssertionError('public writer cannot construct models or probe CUDA')
    monkeypatch.setattr(e.cuda,'_factory',forbidden)
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)):
        r.main(['publish-completed','--run',str(tmp_path/'missing-run'),'--recompute',str(tmp_path/'missing-recompute'),
            '--output',str(output)])
    assert not output.exists()


@pytest.mark.parametrize('damage',[None,'metadata_subject','attempt_count','execution_count','overflow_count',
    'counter_private','counter_float','epoch_float','runtime_private','bitwise_claim','trajectory_claim',
    'interrupted_int','selection_scope','devices_private','devices_float','declared_workers_private',
    'effective_batch','effective_devices','declared_batch'])
def test_completed_builder_composes_raw_readers_with_real_metadata_roster_without_model_construction(tmp_path,monkeypatch,damage):
    """End-to-end metadata composition; termination/checkpoint authorities remain explicitly stubbed."""
    base=native_sidecar_fixture;packet=public_record_fixture();data=tmp_path/'source-data'
    def extended(path,patch):
        r,e,run,native,context,checkpoint=base(path,patch)
        context['selection'].update({k:packet['identity']['selection'][k] for k in ('stop_reason','completed_epochs','best_epoch','best_val_loss')})
        state=context['manifest']['state_identity'];state.update(copy.deepcopy(packet['manifest']['state_identity']))
        state['runtime']=copy.deepcopy(packet['identity']['runtime'])
        state['bindings'].update(normalization_sha256='b'*64,numeric_qualification_artifact_sha256='d'*64)
        context['manifest'].update(adaptations=copy.deepcopy(packet['manifest']['adaptations']),provenance={'data':str(data)})
        context['partitions']={'windows':[dict(episode=0,role='train',split='train'),dict(episode=0,role='test',split='user')]}
        # The terminal authority is a fixture; its admitted checkpoint prefix stays
        # unchanged while result metadata is changed BEFORE downstream calibration.
        context['last']={'formal':{'completed_epochs':52,'journal_prefix':dict(epochs=52,attempts=8840,executions=8838,skips=2)}}
        training=copy.deepcopy(packet['training']);adaptations=context['manifest']['adaptations']
        forged_training={'attempt_count':('attempts',8841),'execution_count':('optimizer_executions',8837),
            'overflow_count':('natural_overflows',3),'counter_private':('optimizer_executions',{'private_path':'/private/counter'}),
            'counter_float':('optimizer_executions',8838.),'epoch_float':('completed_epochs',52.),
            'runtime_private':('runtime_profile',{'private_path':'/private/runtime'}),'bitwise_claim':('bitwise_trajectory_guaranteed',True),
            'trajectory_claim':('trajectory_agreement_qualified',True),'interrupted_int':('interrupted',0),
            'selection_scope':('selection_scope','engineering')}
        forged_adaptations={'devices_private':('declared_devices',[{'private_path':'/private/device'}]),
            'devices_float':('declared_devices',[0.,1,2,3,4,5]),
            'declared_workers_private':('declared_num_workers',{'private_path':'/private/workers'}),
            'effective_batch':('effective_per_rank_batch',1),'effective_devices':('effective_devices',[1]),
            'declared_batch':('declared_per_rank_batch',1)}
        if damage in forged_training:
            key,value=forged_training[damage];training[key]=value
        if damage in forged_adaptations:
            key,value=forged_adaptations[damage];adaptations[key]=value
        (run/'result.json').write_text(json.dumps(training))
        identity=r._native_identity_record(context);(native/'identity.json').write_text(json.dumps(identity))
        result=json.loads((native/'result.json').read_text());result['identity_sha256']=identity['sha256']
        (native/'result.json').write_text(json.dumps(result));status=json.loads((native/'status.json').read_text())
        status['result_sha256']=e.file_sha256(native/'result.json');(native/'status.json').write_text(json.dumps(status))
        return r,e,run,native,context,checkpoint
    monkeypatch.setattr(sys.modules[__name__],'native_sidecar_fixture',extended)
    data.mkdir();(data/'manifest.json').write_text(json.dumps(dict(total_rows=15580,
        fields={'frame_split_id':{'shape':[15580]}},frame_split_labels=['train','user','gesture','both'])))
    subjects=['fixture-private-user','private-unused-user']
    if damage=='metadata_subject':subjects[0]='different-private-subject'
    np.savez(data/'metadata.npz',episode_id=np.array(['private-episode-0','private-episode-1']),
        episode_subject=np.array(subjects),episode_start_idx=np.array([0,7790]),episode_end_idx=np.array([7790,15580]))
    r,e,paths,path,native,launch=raw_test_prediction_pair(tmp_path,monkeypatch)
    def forbidden(*args,**kwargs):raise AssertionError('completed builder must not construct a model or probe CUDA')
    monkeypatch.setattr(e,'prepare_selected_best_evaluation',forbidden)
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    if damage:
        with pytest.raises(ValueError):r.build_classic_completed(tmp_path/'trained',paths[1])
    else:
        value,guard,immutable=r._completed_evidence(tmp_path/'trained',paths[1]);guard.check()
        assert value['data']['source_subject_count']==2 and value['data']['test_subject_count']==1
        assert value['data']['train_test_subject_overlap']=={'user':1,'gesture':0,'both':0}
        assert value['statistics']['center']['overall']['valid_angle_count']==94732
        assert set(value['statistics']['center']['by_user'])=={'participant_001'}
        assert data/'metadata.npz' in guard.paths and paths[0] in immutable and paths[1] in immutable
        assert str(tmp_path) not in json.dumps(value)
