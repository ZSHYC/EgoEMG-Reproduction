"""Native one-consumption point statistics and frozen selected evaluation."""
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('torch')

ROOT=Path(__file__).resolve().parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.emgformer_eval')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emgformer_eval':pytest.fail('native EMGFormer selected evaluation missing')
        raise


def parts(role='test',batch_size=4):
    import torch
    result=[];n=6
    for start in range(0,n,batch_size):
        index=torch.arange(start,min(start+batch_size,n));count=len(index)
        result.append(dict(pred=((index+1).float()/100)[:,None,None].expand(count,22,1).clone(),
            target=torch.zeros(count,22,1),mask=torch.tensor([True,True,False,True,False,True])[index,None],
            index=index.clone(),unique=index.clone(),repeat=torch.zeros(count,dtype=torch.int64),
            center=(index%3)*12000+6000,episode=(index%3).clone(),
            hand=['left'if i<3 else'right'for i in index],
            split=['train'if role=='val'else('user','gesture','both')[i%3]for i in index],
            user=['bob'if i%3==1 else'alice'for i in index],input_sha256='a'*64,
            sample_input_sha256=['b'*64]*count))
    return result


@pytest.mark.parametrize('role',['val','test'])
def test_one_consumption_statistics_keep_native_batch_weighting_and_pooled_valid_angles(role):
    import torch
    raw=parts(role);s=api().emgformer_statistics(raw,samples=6,batch_size=4,role=role)
    weighted=torch.tensor(0.)
    for p in raw:
        mask=p['mask'][:,None,:].expand_as(p['target']);weighted+=(p['pred']-p['target']).abs()[mask].mean()*len(p['pred'])
    assert s['consumed_samples']==s['unique_samples']==6 and s['duplicate_consumptions']==0
    assert s['batch_sizes']==[4,2]and s['native_batch_weighted_mae_rad']==float(weighted/6)
    report=s['all'];all_=report['overall'];assert all_['valid_angle_count']==4*22 and report['invalid_angle_count']==2*22
    expected=sum(float(p['pred'][j,0,0])*22 for p in raw for j in range(len(p['pred']))if p['mask'][j,0])
    assert math.isclose(all_['absolute_error_sum_rad'],expected,rel_tol=1e-14)
    assert not math.isclose(s['native_batch_weighted_mae_rad'],all_['mae_deg']*math.pi/180,rel_tol=1e-4)
    assert sum(v['valid_angle_count']for v in report['by_joint'])==all_['valid_angle_count']
    assert sum(v['valid_angle_count']for v in report['by_hand'].values())==all_['valid_angle_count']
    assert sum(v['valid_angle_count']for v in report['by_split'].values())==all_['valid_angle_count']
    assert sum(v['valid_angle_count']for v in report['by_user'].values())==all_['valid_angle_count']
    assert len(report['by_joint'])==22 and set(report['by_component'])=={'fingers','wrist'}
    assert len(report['by_finger'])==5 and len(report['by_pd_group'])==3
    assert report['user_summary']['user_count']==2 and 'vision_valid'not in s


def test_all_invalid_batch_contributes_native_zero_but_no_pooled_denominator():
    import torch
    raw=parts();raw[0]['mask'].zero_();s=api().emgformer_statistics(raw,samples=6,batch_size=4,role='test')
    assert s['all']['overall']['valid_angle_count']==22
    assert s['native_batch_weighted_mae_rad']==float(raw[1]['pred'][1,0,0]*2/6)
    for p in raw:p['mask'].zero_()
    s=api().emgformer_statistics(raw,samples=6,batch_size=4,role='test')
    assert s['all']['overall']['mae_deg']is None and s['native_batch_weighted_mae_rad']==0


def test_raw_statistics_are_strict_json_values_for_public_report():
    value=api().emgformer_statistics(parts(),samples=6,batch_size=4,role='test')
    assert json.loads(json.dumps(value,allow_nan=False))==value


@pytest.mark.parametrize('damage',['shape','pred_dtype','target_dtype','mask_dtype','nan','inf_masked','index','unique',
    'repeat','hand','split','user','center','episode','input_digest','sample_digest','missing_batch','short_batch','float_index'])
def test_statistics_reject_non_native_points_indices_metadata_and_coverage(damage):
    import torch
    raw=parts();p=raw[0]
    if damage=='shape':p['pred']=p['pred'].expand(4,22,2).clone()
    elif damage=='pred_dtype':p['pred']=p['pred'].double()
    elif damage=='target_dtype':p['target']=p['target'].double()
    elif damage=='mask_dtype':p['mask']=p['mask'].float()
    elif damage=='nan':p['pred'][0,0,0]=float('nan')
    elif damage=='inf_masked':p['target'][2,0,0]=float('inf')
    elif damage=='index':p['index'][0]=1
    elif damage=='unique':p['unique'][0]=1
    elif damage=='repeat':p['repeat'][0]=1
    elif damage=='hand':p['hand'][0]='right'
    elif damage=='split':p['split'][0]='train'
    elif damage=='user':p['user'][0]=''
    elif damage=='center':p['center']=p['center'].float()
    elif damage=='episode':p['episode'][0]=-1
    elif damage=='input_digest':p['input_sha256']='z'*64
    elif damage=='sample_digest':p['sample_input_sha256'][0]=''
    elif damage=='missing_batch':raw.pop()
    elif damage=='short_batch':
        for k,v in p.items():
            if torch.is_tensor(v)or isinstance(v,list):p[k]=v[:-1]
    elif damage=='float_index':p['index']=p['index'].float()
    with pytest.raises(ValueError):api().emgformer_statistics(raw,samples=6,batch_size=4,role='test')


@pytest.mark.parametrize('samples,batch_size,role',[(0,4,'test'),(5,4,'test'),(True,4,'test'),(6,False,'test'),(6,0,'test'),(6,4,'train')])
def test_invalid_evaluation_population_rejects(samples,batch_size,role):
    with pytest.raises(ValueError):api().emgformer_statistics(parts(),samples=samples,batch_size=batch_size,role=role)


@pytest.fixture(scope='module')
def chain(tmp_path_factory,request):
    from test_emgformer_train import native_runs
    m=api()
    for name in ('run_emgformer_pass','read_emgformer_pass','freeze_emgformer_validation',
        'read_emgformer_profile','build_emgformer_report','publish_emgformer_report','verify_emgformer_report'):
        assert hasattr(m,name),'selected point lifecycle missing: '+name
    saved=os.environ.get('EGOEMG_EMGFORMER_ENGINEERING_FIT')
    if saved:
        fit=Path(saved).resolve();data=Path(os.environ.get('EGOEMG_EMGFORMER_ENGINEERING_DATA',str(fit.parent/'data'))).resolve()
        assert fit.is_dir()and data.is_dir()
    else:
        # A clean checkout can bootstrap the same two-epoch fixture. During
        # development use the retained current-source fit, without new updates.
        runs=native_runs.__wrapped__(tmp_path_factory);_,fit,data,_,_=runs
    identity=json.loads((fit/'identity.json').read_text())
    from test_emgformer_data import UPSTREAM
    cached=os.environ.get('EGOEMG_EMGFORMER_EVAL_CHAIN')
    if cached:
        root=Path(cached).resolve();m.verify_emgformer_report(root/'test-replay',root/'report')
        assert json.loads((root/'test-replay/pass.json').read_text())['fit_path']==str(fit)
        return root,fit,data,identity,dict(os.environ),[]
    root=tmp_path_factory.mktemp('native-emgformer-selected-evaluation')
    before={p.name:p.read_bytes()for p in (fit/'checkpoints').glob('*.ckpt')}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    base=[sys.executable,'-B','-m','egoemg_repro.emgformer_eval']
    common=['--fit',str(fit),'--data',str(data),'--upstream',str(UPSTREAM),'--engineering']
    calls=[('val-native',['run',*common,'--output',str(root/'val-native'),'--role','val','--kind','native']),
        ('val-replay',['run',*common,'--output',str(root/'val-replay'),'--role','val','--kind','replay','--reference',str(root/'val-native')]),
        ('freeze',['freeze','--native',str(root/'val-native'),'--replay',str(root/'val-replay'),'--output',str(root/'profile')]),
        ('test-native',['run',*common,'--output',str(root/'test-native'),'--role','test','--kind','native','--profile',str(root/'profile')]),
        ('test-replay',['run',*common,'--output',str(root/'test-replay'),'--role','test','--kind','replay','--reference',str(root/'test-native'),'--profile',str(root/'profile')]),
        ('report',['report','--replay',str(root/'test-replay'),'--output',str(root/'report')]),
        ('verify',['verify','--replay',str(root/'test-replay'),'--output',str(root/'report')])]
    for mode,args in calls:
        p=subprocess.run(base+args,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
        (root/(mode+'.stdout')).write_text(p.stdout);(root/(mode+'.stderr')).write_text(p.stderr)
        assert p.returncode==0,p.stderr[-8000:]
    assert before=={p.name:p.read_bytes()for p in (fit/'checkpoints').glob('*.ckpt')}
    return root,fit,data,identity,env,common


def test_actual_selected_point_validation_freeze_test_replay_and_report(chain,monkeypatch):
    root,fit,_,identity,_,_=chain;m=api()
    monkeypatch.setattr(m,'build_emgformer_runtime',lambda *a,**kw:pytest.fail('raw acceptance/report must not construct a model'))
    reference=m.read_emgformer_pass(root/'val-native');replay=m.read_emgformer_pass(root/'val-replay')
    profile=m.read_emgformer_profile(root/'profile');native=m.read_emgformer_pass(root/'test-native');test=m.read_emgformer_pass(root/'test-replay')
    assert reference['process']!=replay['process']and native['process']!=test['process']
    assert reference['prediction_sha256']==replay['prediction_sha256']and native['prediction_sha256']==test['prediction_sha256']
    assert max(reference['finished_ns'],replay['finished_ns'])<=profile['finished_ns']<=min(native['started_ns'],test['started_ns'])
    for value in (reference,replay,native,test):
        assert value['native_point_center_selection_exact']and value['optimizer_updates']==0
        assert value['model_before_sha256']==value['model_after_sha256']
        assert value['statistics']['unique_samples']==value['samples']and value['statistics']['duplicate_consumptions']==0
        assert value['samples']==identity['cohort']['counts'][value['role']]
        assert not value['scientific_eligible']and not value['full_paper_complete']
    result=m.verify_emgformer_report(root/'test-replay',root/'report')
    assert result==m.build_emgformer_report(root/'test-replay')and not result['scientific_eligible']
    assert result['test']['unique_samples']==identity['cohort']['counts']['test']
    assert all(user.startswith('user_')for user in result['test']['all']['by_user'])
    public=json.dumps(result);assert '/home/'not in public and '/tmp/'not in public
    assert result['training']['selected_best']==json.loads((fit/'fit-result.json').read_text())['selected_best']


@pytest.mark.parametrize('damage',['role','kind','statistics','coverage','selected','model','modelboth','update_bool','source','policy','profile','chronology'])
def test_durable_selected_pass_rejects_resealed_tampered_chain(chain,damage):
    from egoemg_repro.classic_emg import _hash
    root,_,_,_,_,_=chain;path=root/'test-replay/pass.json';old=path.read_bytes();v=json.loads(old)
    if damage=='role':v['role']='val'
    elif damage=='kind':v['kind']='native'
    elif damage=='statistics':v['statistics']['all']['overall']['mae_deg']+=.2
    elif damage=='coverage':v['samples']-=2
    elif damage=='selected':v['selected_best']['file']='last.ckpt'
    elif damage=='model':v['model_after_sha256']='0'*64
    elif damage=='modelboth':v['model_before_sha256']=v['model_after_sha256']='0'*64
    elif damage=='update_bool':v['optimizer_updates']=False
    elif damage=='source':v['sources_sha256']={}
    elif damage=='policy':v['fixed_agreement_policy']['native_mae_atol_rad']=1.
    elif damage=='profile':v['profile_sha256']='0'*64
    elif damage=='chronology':v['started_ns']=1
    v['sha256']=_hash({k:x for k,x in v.items()if k!='sha256'})
    try:
        path.write_text(json.dumps(v))
        with pytest.raises(ValueError):api().read_emgformer_pass(root/'test-replay')
    finally:path.write_bytes(old)


@pytest.mark.parametrize('scenario',['test_without_profile','validation_with_profile','replay_without_reference',
    'wrong_role_reference','same_process_replay','native_with_reference','engineering_as_full','partial_fit'])
def test_invalid_selected_execution_rejects_before_model_construction(chain,tmp_path,monkeypatch,scenario):
    from test_emgformer_data import UPSTREAM,cfg
    from egoemg_repro.emgformer_literal import emgformer_normalization
    from egoemg_repro.emgformer_cohort import admit_emgformer_cohort
    root,fit,data,identity,_,_=chain;m=api();config=cfg()
    cohort=admit_emgformer_cohort(data,upstream=UPSTREAM,config=config,
        normalization=emgformer_normalization(UPSTREAM,config),expected=identity['cohort'])
    monkeypatch.setattr(m,'build_emgformer_runtime',lambda *a,**kw:pytest.fail('invalid evaluation reached model construction'))
    kwargs=dict(role='val',kind='native',require_full=False)
    if scenario=='test_without_profile':kwargs['role']='test'
    elif scenario=='validation_with_profile':kwargs['profile']=root/'profile'
    elif scenario=='replay_without_reference':kwargs['kind']='replay'
    elif scenario=='wrong_role_reference':kwargs.update(kind='replay',reference=root/'test-native')
    elif scenario=='same_process_replay':
        monkeypatch.setattr(m,'_process',lambda:json.loads((root/'val-native/pass.json').read_text())['process'])
        kwargs.update(kind='replay',reference=root/'val-native')
    elif scenario=='native_with_reference':kwargs['reference']=root/'val-native'
    elif scenario=='engineering_as_full':kwargs['require_full']=True
    else:
        result=json.loads((fit/'fit-result.json').read_text());result['stage_complete']=False
        monkeypatch.setattr(m.train,'read_emgformer_fit',lambda *a,**kw:result)
    with pytest.raises(ValueError):m.run_emgformer_pass(cohort,fit,tmp_path/'invalid',**kwargs)
    assert not(tmp_path/'invalid').exists()


@pytest.mark.parametrize('damage',['predictions','native_metrics','profile','metrics_json','metrics_md','completion'])
def test_payload_sidecar_profile_and_public_output_tampering_rejects(chain,damage):
    root,_,_,_,_,_=chain
    path={'predictions':root/'test-replay/predictions.pt','native_metrics':root/'test-native/native-metrics.json',
        'profile':root/'profile/profile.json','metrics_json':root/'report/metrics.json','metrics_md':root/'report/metrics.md',
        'completion':root/'report/completion.json'}[damage];old=path.read_bytes()
    try:
        path.write_bytes(old+b'bad')
        with pytest.raises((ValueError,json.JSONDecodeError)):api().verify_emgformer_report(root/'test-replay',root/'report')
    finally:path.write_bytes(old)


@pytest.mark.parametrize('cycle',['reference_self','profile_to_test','two_pass_cycle'])
def test_reference_graph_cycles_fail_with_bounded_reads(chain,monkeypatch,cycle):
    from egoemg_repro.classic_emg import _hash
    root,_,_,_,_,_=chain;m=api();changes={}
    def alter(path,**values):
        changes[path]=path.read_bytes();v=json.loads(changes[path]);v.update(values)
        v['sha256']=_hash({k:x for k,x in v.items()if k!='sha256'});path.write_text(json.dumps(v))
    original=m._load_pass;loads=0
    def bounded(path):
        nonlocal loads
        loads+=1
        if loads>12:raise RecursionError('bounded test detected cyclic artifact traversal')
        return original(path)
    monkeypatch.setattr(m,'_load_pass',bounded)
    try:
        if cycle=='reference_self':
            alter(root/'test-replay/pass.json',reference_path=str(root/'test-replay'))
            action=lambda:m.read_emgformer_pass(root/'test-replay')
        elif cycle=='profile_to_test':
            alter(root/'profile/profile.json',reference_path=str(root/'test-native'))
            action=lambda:m.read_emgformer_profile(root/'profile')
        else:
            alter(root/'val-native/pass.json',kind='replay',native_metrics={},native_aggregation_agrees=False,
                reference_path=str(root/'val-replay'),reference_sha256=json.loads((root/'val-replay/pass.json').read_text())['sha256'])
            action=lambda:m.read_emgformer_pass(root/'val-replay')
        with pytest.raises(ValueError):action()
        assert loads<=12
    finally:
        for path,old in changes.items():path.write_bytes(old)


def _reseal_file(path,value):
    from egoemg_repro.classic_emg import _hash
    value['sha256']=_hash({k:v for k,v in value.items()if k!='sha256'});path.write_text(json.dumps(value))
    return value


@pytest.mark.parametrize('metric',['test_mae_wrist/flexion','test_loss'])
def test_review_unverified_native_sidecar_metrics_cannot_enter_verified_report(chain,metric):
    root,_,_,_,_,_=chain;paths=[root/'test-native/native-metrics.json',root/'test-native/pass.json',root/'test-replay/pass.json']
    old={p:p.read_bytes()for p in paths}
    try:
        side=json.loads(old[paths[0]]);side[metric]=12345.;paths[0].write_text(json.dumps(side))
        native=json.loads(old[paths[1]]);native['native_metrics']=side;native=_reseal_file(paths[1],native)
        replay=json.loads(old[paths[2]]);replay['reference_sha256']=native['sha256'];_reseal_file(paths[2],replay)
        with pytest.raises(ValueError):api().build_emgformer_report(root/'test-replay')
    finally:
        for p,b in old.items():p.write_bytes(b)


@pytest.mark.parametrize('damage',['episode_count','both_membership','train_subjects'])
def test_review_population_claims_must_match_admitted_sources_and_observed_samples(chain,damage):
    import torch
    from egoemg_repro.classic_emg import _file_hash,_hash
    root,_,_,_,_,_=chain;paths=[root/(role+'/'+name)for role in ('test-native','test-replay')for name in ('predictions.pt','pass.json')]
    old={p:p.read_bytes()for p in paths}
    try:
        native_sha=None
        for mode in ('test-native','test-replay'):
            payload_path=root/mode/'predictions.pt';pass_path=root/mode/'pass.json'
            payload=torch.load(str(payload_path),map_location='cpu',weights_only=True);pop=payload['population']
            if damage=='episode_count':pop['public']['episode_records']=999
            elif damage=='both_membership':pop['test_users_by_split']['both']=[]
            else:pop['users_by_role']['train']=['invented_subject']
            pop['public']=api()._population_public(pop['users_by_role'],pop['test_users_by_split'],pop['public']['episode_records'])
            torch.save(payload,str(payload_path));v=json.loads(old[pass_path]);v.update(payload_sha256=_file_hash(payload_path),
                population_sha256=_hash(pop),population=pop['public'])
            if mode=='test-replay':v['reference_sha256']=native_sha
            v=_reseal_file(pass_path,v)
            if mode=='test-native':native_sha=v['sha256']
        with pytest.raises(ValueError):api().build_emgformer_report(root/'test-replay')
    finally:
        for p,b in old.items():p.write_bytes(b)


@pytest.mark.parametrize('dependency',['profile','reference'])
def test_review_future_dependencies_reject_before_selected_model_loading(chain,tmp_path,monkeypatch,dependency):
    import time
    from test_emgformer_data import UPSTREAM,cfg
    from egoemg_repro.emgformer_literal import emgformer_normalization
    from egoemg_repro.emgformer_cohort import admit_emgformer_cohort
    root,fit,data,identity,_,_=chain;m=api();config=cfg()
    cohort=admit_emgformer_cohort(data,upstream=UPSTREAM,config=config,
        normalization=emgformer_normalization(UPSTREAM,config),expected=identity['cohort'])
    path=root/('profile/profile.json'if dependency=='profile'else'val-native/pass.json');old=path.read_bytes()
    monkeypatch.setattr(m,'build_emgformer_runtime',lambda *a,**kw:pytest.fail('future dependency reached selected model loading'))
    try:
        v=json.loads(old);v['finished_ns']=time.time_ns()+3600*10**9;_reseal_file(path,v)
        kwargs=dict(role='test',kind='native',profile=root/'profile')if dependency=='profile'else dict(role='val',kind='replay',reference=root/'val-native')
        with pytest.raises(ValueError):m.run_emgformer_pass(cohort,fit,tmp_path/'invalid-future',require_full=False,**kwargs)
        assert not(tmp_path/'invalid-future').exists()
    finally:path.write_bytes(old)
