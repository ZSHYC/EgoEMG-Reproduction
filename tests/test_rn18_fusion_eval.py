"""Repeat-one native RN18 statistics: all requested centers and both aggregations."""
import copy
import importlib

import pytest

from test_rn18_fusion_train import native_runs,branches


def api():
    try:return importlib.import_module('egoemg_repro.rn18_fusion_eval')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_fusion_eval':pytest.fail('RN18 full paired evaluator missing')
        raise


def parts():
    torch=pytest.importorskip('torch')
    from egoemg_repro.classic_emg import _hash
    result=[]
    for first,errors,valid in [(0,[1.,3.,99.,99.],[True,True,False,False]),(4,[9.,7.],[True,True])]:
        n=len(errors);ids=torch.arange(first,first+n);fingerprints=[_hash(dict(input=int(i)))for i in ids]
        result.append(dict(pred=torch.tensor(errors,dtype=torch.float32)[:,None,None].expand(n,22,1).clone(),
            target=torch.zeros(n,22,1),mask=torch.tensor(valid)[:,None],vision=torch.tensor([i%2==0 for i in ids])[:,None],
            index=ids,unique=ids.clone(),repeat=torch.zeros(n,dtype=torch.int64),center=ids+6000,episode=torch.zeros(n,dtype=torch.int64),
            window=ids%3,hand=['left'if i<3 else'right'for i in ids],split=['user','gesture','both','user','gesture','both'][first:first+n],
            user=['one']*n,sample_input_sha256=fingerprints,input_sha256=_hash(fingerprints)))
    return result


def test_repeat_one_keeps_six_centers_and_native_weighted_differs_from_angle_pool():
    pytest.importorskip('pytorch_lightning');m=api();r=m.rn18_fusion_statistics(parts(),samples=6,batch_size=4,role='test')
    assert r['consumed_samples']==r['unique_samples']==6 and r['duplicate_consumptions']==0
    assert r['batch_sizes']==[4,2]and r['native_batch_weighted_mae_rad']==4.
    a=r['consumed']['all'];assert a['overall']['valid_angle_count']==88
    assert a['overall']['absolute_error_sum_rad']==440.
    assert r['consumed']['samples']==dict(all=6,vision_valid=3,missing_crop=3)
    assert r['unique_first_repeat']==r['consumed']
    assert len(a['by_joint'])==22 and a['by_split_user_summary']['user']['user_count']==1


def test_validation_two_centers_is_complete_without_inventing_repeat_two():
    pytest.importorskip('pytorch_lightning');p=parts()[-1];p=copy.deepcopy(p)
    torch=pytest.importorskip('torch');p['index']=p['unique']=torch.arange(2);p['repeat']=torch.zeros(2,dtype=torch.int64);p['split']=['train']*2;p['hand']=['left','right']
    r=api().rn18_fusion_statistics([p],samples=2,batch_size=2,role='val')
    assert r['unique_samples']==2 and r['native_batch_weighted_mae_rad']==8.


@pytest.mark.parametrize('damage',['pose','native_metric'])
def test_saved_selected_validation_disagreement_cannot_be_recorded_as_accepted(damage):
    pytest.importorskip('pytorch_lightning');m=api();p=parts()[-1]
    record=dict(epoch=0,validation=dict(samples=2,pose_sha256=m._pose_sha([p]),val_mae=8.))
    statistics=dict(consumed_samples=2,native_batch_weighted_mae_rad=8.)
    if damage=='pose':p['pred'][0,0,0]+=.5
    else:statistics['native_batch_weighted_mae_rad']=9.
    with pytest.raises(ValueError,match='saved|selected|validation'):
        m._saved_validation('val',[p],statistics,record)


@pytest.mark.parametrize('damage',['duplicate','repeat','mask','fingerprint','batch_digest','partial'])
def test_resealed_metadata_cannot_repair_missing_or_duplicate_population(damage):
    pytest.importorskip('pytorch_lightning');m=api();p=parts();torch=pytest.importorskip('torch')
    if damage=='duplicate':p[-1]['unique'][0]=0
    elif damage=='repeat':p[-1]['repeat'][0]=1
    elif damage=='mask':p[-1]['mask']=p[-1]['mask'].float()
    elif damage=='fingerprint':p[-1]['sample_input_sha256'][0]='bad'
    elif damage=='batch_digest':p[-1]['input_sha256']='0'*64
    else:p.pop()
    with pytest.raises(ValueError):m.rn18_fusion_statistics(p,samples=6,batch_size=4,role='test')


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory,request):
    import hashlib,json,os,subprocess,sys
    from pathlib import Path
    pytest.importorskip('pytorch_lightning');api()
    from test_rn18_fusion_data import UPSTREAM
    fitroot=Path(os.environ['EGOEMG_RN18_FIT_FIXTURE'])if os.environ.get('EGOEMG_RN18_FIT_FIXTURE')else request.getfixturevalue('native_runs')[0]
    existing=os.environ.get('EGOEMG_RN18_EVAL_RETAINED')
    root=Path(existing)if existing else tmp_path_factory.mktemp('rn18-native-final-chain')
    source=Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    executable=[sys.executable,'-u','-B','-m','egoemg_repro.rn18_fusion_eval']
    common=['--data',str(fitroot/'data'),'--crops',str(fitroot/'crops'),'--upstream',str(UPSTREAM),'--fit',str(fitroot/'fit'),'--engineering']
    stages=[('val-native',['pass',*common,'--role','val','--kind','native']),
        ('val-replay',['pass',*common,'--role','val','--kind','replay','--reference',str(root/'val-native')]),
        ('profile',['freeze','--reference',str(root/'val-native'),'--replay',str(root/'val-replay')]),
        ('test-native',['pass',*common,'--role','test','--kind','native','--profile',str(root/'profile')]),
        ('test-replay',['pass',*common,'--role','test','--kind','replay','--profile',str(root/'profile'),'--reference',str(root/'test-native')]),
        ('public',['report','--fit',str(fitroot/'fit'),'--evaluation',str(root/'test-native'),'--replay',str(root/'test-replay'),'--engineering'])]
    for label,args in stages:
        if (root/label).exists():
            if label=='profile':api().read_rn18_fusion_profile(root/label)
            elif label=='public':api().verify_rn18_fusion_report(root/'test-replay',root/label,require_full=False)
            else:api().read_rn18_fusion_pass(root/label)
            continue
        child=subprocess.run([*executable,*args,'--output',str(root/label)],cwd=source,env=env,capture_output=True,text=True,timeout=120)
        (root/(label+'.log')).write_text(child.stdout+child.stderr);assert child.returncode==0,child.stderr[-5000:]
    child=subprocess.run([*executable,'verify','--recompute',str(root/'test-replay'),'--output',str(root/'public'),'--engineering'],cwd=source,env=env,capture_output=True,text=True,timeout=120)
    (root/'verify.log').write_text(child.stdout+child.stderr);assert child.returncode==0,child.stderr[-5000:]
    return root,fitroot,env,executable


def test_actual_native_replay_freeze_and_report_keep_all_requests(evaluated):
    import json,torch
    root,_,_,_=evaluated;m=api();p=m.read_rn18_fusion_pass(root/'test-replay');profile=m.read_rn18_fusion_profile(root/'profile')
    assert p['samples']==p['statistics']['unique_samples']==6 and p['statistics']['duplicate_consumptions']==0
    assert p['loader_workers']['count']==0 and p['loader_workers']['all_closed']
    assert p['started_ns']>profile['finished_ns']and len(profile['saved_fit_validation']['fit_pose_sha256'])==64
    assert not p['scientific_eligible']and not p['training_full_terminated']
    report=m.verify_rn18_fusion_report(root/'test-replay',root/'public',require_full=False)
    assert report['variant']=='fusion_rn18_s_center_8ch'and not report['scientific_eligible']
    payload=torch.load(root/'test-replay/predictions.pt',map_location='cpu',weights_only=True)
    assert torch.equal(torch.cat([x['index']for x in payload['parts']]),torch.arange(6))
    assert torch.cat([x['repeat']for x in payload['parts']]).sum()==0
    assert all((root/'public'/x).is_file()for x in ('metrics.json','metrics.md','curve-data.json','training-curves.png','training-curves.svg','completion.json'))
    with pytest.raises(ValueError):m.build_rn18_fusion_report(root/'test-replay')


@pytest.mark.parametrize('entry',['freeze','profile_read'])
def test_agreeing_resealed_standalone_predictions_cannot_bypass_saved_fit_validation(evaluated,tmp_path,entry):
    """Both standalone passes can agree while the selected fit stays immutable."""
    import json,torch
    from egoemg_repro.classic_emg import _hash,_file_hash
    root,_,_,_=evaluated;m=api()
    paths=[root/'val-native'/n for n in('predictions.pt','native-metrics.json','pass.json')]
    paths += [root/'val-replay'/n for n in('predictions.pt','pass.json')]
    paths += [root/'profile/profile.json']
    before={p:p.read_bytes()for p in paths}
    try:
        native=None
        for kind in('native','replay'):
            directory=root/('val-'+kind);file=directory/'predictions.pt'
            payload=torch.load(file,map_location='cpu',weights_only=True)
            for part in payload['parts']:part['pred']=part['pred']+1.
            torch.save(payload,file)
            value=json.loads((directory/'pass.json').read_text())
            statistics=m.rn18_fusion_statistics(payload['parts'],samples=value['samples'],batch_size=value['batch_size'],role='val')
            value.update(statistics=statistics,payload_sha256=_file_hash(file),prediction_sha256=m._payload_sha(payload['parts']))
            comparison=value['saved_fit_validation']
            comparison.update(evaluation_pose_sha256=m._pose_sha(payload['parts']),pose_exact_match=False,
                native_mae_difference_rad=statistics['native_batch_weighted_mae_rad']-comparison['fit_native_mae_rad'])
            if kind=='native':
                value['native_metrics']['val_mae']=statistics['native_batch_weighted_mae_rad']
                (directory/'native-metrics.json').write_text(json.dumps(value['native_metrics']))
            else:value['reference_sha256']=native['sha256']
            value['sha256']=_hash({k:v for k,v in value.items()if k!='sha256'})
            (directory/'pass.json').write_text(json.dumps(value))
            if kind=='native':native=value
            else:replay=value
        if entry=='freeze':
            output=tmp_path/'must-not-freeze'
            with pytest.raises(ValueError,match='saved|selected|validation'):
                m.freeze_rn18_fusion_validation(root/'val-native',root/'val-replay',output)
            assert not output.exists()
        else:
            profile=json.loads((root/'profile/profile.json').read_text())
            profile.update(reference_sha256=native['sha256'],replay_sha256=replay['sha256'],saved_fit_validation=native['saved_fit_validation'])
            profile['sha256']=_hash({k:v for k,v in profile.items()if k!='sha256'})
            (root/'profile/profile.json').write_text(json.dumps(profile))
            with pytest.raises(ValueError,match='saved|selected|validation'):
                m.read_rn18_fusion_profile(root/'profile')
    finally:
        for path,body in before.items():path.write_bytes(body)


@pytest.mark.parametrize('damage',['profile','crop','last','native_metric'])
def test_fresh_read_transaction_rejects_changed_raw_dependency(evaluated,damage):
    import json
    root,fitroot,*_=evaluated
    if damage=='profile':p=root/'profile/profile.json'
    elif damage=='last':p=fitroot/'fit/checkpoints/last.ckpt'
    elif damage=='native_metric':p=root/'test-native/native-metrics.json'
    else:p=next((fitroot/'crops').rglob('*data.mdb'))
    before=p.read_bytes()
    try:
        if damage=='profile':
            from egoemg_repro.classic_emg import _hash
            value=json.loads(before);value['saved_fit_validation']['fit_pose_sha256']='0'*64
            value['sha256']=_hash({k:v for k,v in value.items()if k!='sha256'});p.write_text(json.dumps(value))
        elif damage=='native_metric':
            value=json.loads(before);value['test_mae']+=.1;p.write_text(json.dumps(value))
        else:
            with p.open('r+b')as f:f.seek(-1,2);v=f.read(1);f.seek(-1,2);f.write(bytes([v[0]^1]))
        with pytest.raises(ValueError):api().read_rn18_fusion_pass(root/'test-replay')
    finally:p.write_bytes(before)
