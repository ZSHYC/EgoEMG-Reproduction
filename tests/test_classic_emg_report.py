"""Hand-calculated report arithmetic, not inference or scientific completion."""
import importlib
import math
import copy
import hashlib
import json
from pathlib import Path

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


def raw_prediction_pair(tmp_path,monkeypatch):
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
    identity=r._identity_record(run,manifest,selection,'val')
    policy=identity['policy'];n=policy['samples'];batch=policy['batch_size']
    coverage=completed_ledger();coverage.update(role='val',samples=n,
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
            process=dict(pid=100+i,start_ticks=1000+i,phase='classic-val-'+kind),pass_kind=kind,
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
    backup=None;close=e.artifacts._Directory.close;injected=[]
    e.artifacts._write_json(owned,'identity.json',identity)
    e.artifacts._write_json(owned,'status.json',{'state':'running'})
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
            with pytest.raises(OSError):r._finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup)
            assert injected and json.loads((output/'status.json').read_text())['state']=='failed'
            with pytest.raises(ValueError):r._read_prediction_pass(output,role='val')
        else:
            assert r._finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup)==result
            assert r._read_prediction_pass(output,role='val')[1]==result
    finally:
        for directory in (owned,rescue,backup):
            if directory is not None:close(directory)
