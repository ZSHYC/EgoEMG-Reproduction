"""Real selected DINO checkpoint, pretest validation replay and final test audit."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_vision_literal_train import native_runs,ROOT,UPSTREAM


def api():
    try:return importlib.import_module('egoemg_repro.vision_literal_eval')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.vision_literal_eval':pytest.fail('selected-best native vision evaluation missing')
        raise


def test_statistics_preserve_generated_mask_missing_crops_and_repeat_denominators():
    torch=pytest.importorskip('torch');m=api()
    # Four consumed hands, two unique. Missing right crop must retain its label.
    part=dict(pred=torch.ones(4,22,1),target=torch.zeros(4,22,1),mask=torch.ones(4,1,dtype=torch.bool),
        index=torch.arange(4),unique=torch.tensor([0,1,0,1]),repeat=torch.tensor([0,0,1,1]),
        vision=torch.tensor([[True],[False],[True],[False]]),hand=['left','right']*2,
        split=['user']*4,user=['private-person']*4,center=torch.tensor([3895]*4),
        input_sha256='a'*64)
    value=m.vision_statistics([part],samples=4,batch_size=4,role='test')
    assert value['consumed']['all']['overall']['valid_angle_count']==88
    assert value['unique_first_repeat']['all']['overall']['valid_angle_count']==44
    assert value['consumed']['samples']==dict(all=4,vision_valid=2,missing_crop=2)
    assert value['unique_samples']==2 and value['duplicate_consumptions']==2
    assert value['consumed']['all']['overall']['absolute_error_sum_rad']==88.
    assert value['consumed']['all']['by_joint'][21]['valid_angle_count']==4
    assert value['native_batch_weighted_mae_rad']==1.
    assert value['consumed']['all']['by_split_user_summary']['user']['std_mae_deg']==0.
    broken=copy.deepcopy(part);broken['index'][3]=2
    with pytest.raises(ValueError):m.vision_statistics([broken],samples=4,batch_size=4,role='test')
    # Independent validation is held out from the original train split.
    validation=copy.deepcopy(part);validation['split']=['train']*4
    val=m.vision_statistics([validation],samples=4,batch_size=4,role='val')
    assert val['consumed']['all']['by_split']['train']['valid_angle_count']==88
    with pytest.raises(ValueError):m.vision_statistics([validation],samples=4,batch_size=4,role='test')


def test_native_batch_weighting_is_not_silently_replaced_by_pooled_valid_angles():
    torch=pytest.importorskip('torch');m=api()
    def part(start,errors,valid):
        n=len(errors)
        return dict(pred=torch.tensor(errors).reshape(n,1,1).expand(n,22,1).clone(),target=torch.zeros(n,22,1),
            mask=torch.tensor(valid).reshape(n,1),index=torch.arange(start,start+n),
            unique=torch.arange(start,start+n)%2,repeat=torch.arange(start,start+n)//2,
            vision=torch.ones(n,1,dtype=torch.bool),hand=['left' if (start+i)%2==0 else 'right' for i in range(n)],
            split=['user']*n,user=['p']*n,center=torch.tensor([3895]*n),input_sha256='b'*64)
    parts=[part(0,[1.,100.,3.],[True,False,True]),part(3,[9.],[True])]
    value=m.vision_statistics(parts,samples=4,batch_size=3,role='test')
    assert value['native_batch_weighted_mae_rad']==3.75
    assert value['consumed']['all']['overall']['absolute_error_sum_rad']==286.
    assert value['consumed']['all']['overall']['valid_angle_count']==66


@pytest.fixture(scope='module')
def evaluated(native_runs):
    root,env,_,_=native_runs;m=api();output=root/'evaluation'
    output.mkdir()
    base=[sys.executable,'-B','-m','egoemg_repro.vision_literal_eval']
    shared=['--data',str(root/'data'),'--crops',str(root/'crops'),'--upstream',str(UPSTREAM),
        '--fit',str(root/'control'),'--engineering']
    commands=[('val-native',['pass',*shared,'--role','val','--kind','native','--output',str(output/'val-native')]),
        ('val-replay',['pass',*shared,'--role','val','--kind','replay','--reference',str(output/'val-native'),'--output',str(output/'val-replay')]),
        ('freeze',['freeze','--reference',str(output/'val-native'),'--replay',str(output/'val-replay'),'--output',str(output/'profile')]),
        ('test-native',['pass',*shared,'--role','test','--kind','native','--profile',str(output/'profile'),'--output',str(output/'test-native')]),
        ('test-replay',['pass',*shared,'--role','test','--kind','replay','--profile',str(output/'profile'),
            '--reference',str(output/'test-native'),'--output',str(output/'test-replay')])]
    for name,args in commands:
        child=subprocess.run([*base,*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
        (output/(name+'.stdout')).write_text(child.stdout);(output/(name+'.stderr')).write_text(child.stderr)
        assert child.returncode==0,child.stderr[-6500:]
    return output


def test_actual_native_validation_and_test_use_only_selected_best_and_full_population(evaluated):
    m=api();val=m.read_vision_pass(evaluated/'val-native');native=m.read_vision_pass(evaluated/'test-native')
    replay=m.read_vision_pass(evaluated/'test-replay');profile=m.read_vision_profile(evaluated/'profile')
    assert val['role']=='val' and val['statistics']['consumed']['samples']['all']==4
    assert native['statistics']['consumed']['samples']['all']==12
    assert native['statistics']['unique_samples']==6 and native['statistics']['duplicate_consumptions']==6
    assert native['selected_best']['file']!='last.ckpt'
    assert native['profile_sha256']==profile['sha256'] and replay['reference_sha256']==native['sha256']
    assert native['prediction_sha256']==replay['prediction_sha256']
    assert native['model_before_sha256']==native['model_after_sha256']
    assert native['process']!=replay['process']
    assert native['native_aggregation_agrees'] is True and native['scientific_eligible'] is False
    assert native['training_full_terminated'] is False and native['full_paper_complete'] is False
    assert profile['fixed_agreement_policy']['prediction']=='exact typed prediction/target/mask/input replay'


def test_public_engineering_report_is_sanitized_and_does_not_claim_full_training(evaluated):
    m=api();report=m.build_vision_report(evaluated/'test-replay')
    raw=json.dumps(report)
    assert report['scientific_eligible'] is False and report['full_paper_complete'] is False
    assert report['test']['unique_samples']==6 and report['training']['completed_epochs']==2
    assert report['training']['execution_policy']['activation_checkpointing'] is False
    assert '激活重计算：关闭' in m.render_vision_report(report)
    adapted=copy.deepcopy(report);adapted['training']['execution_policy']['activation_checkpointing']=True
    assert '激活重计算：开启（显式适配）' in m.render_vision_report(adapted)
    assert 'private-person' not in raw and str(ROOT) not in raw and '/tmp/' not in raw
    assert list(report['test']['consumed']['all']['by_user'])==['user_000']
    assert report['population']['unique_subjects_by_role']==dict(train=1,val=1,test=1)
    assert report['population']['train_test_user_intersection_by_split']==dict(user=1,gesture=1,both=1)


def test_population_intersections_come_from_actual_admitted_episode_metadata(tmp_path):
    pytest.importorskip('torch')
    from test_vision_literal_data import sample_store,crop_store,config,release_fixture_readers
    from egoemg_repro.vision_literal_cohort import admit_vision_cohort
    np=pytest.importorskip('numpy');api();data=tmp_path/'data';crops=tmp_path/'crops'
    sample_store(data,n=9*7790);crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*7790,7790,7790,7790]).astype('int8').tofile(data/'frame_split_id.dat')
    try:
        cohort=admit_vision_cohort(data,crops,upstream=UPSTREAM,config=config())
        population=api().vision_population(cohort)
        assert population['public']['unique_subjects_by_role']==dict(train=1,val=1,test=1)
        assert population['public']['train_test_user_intersection_by_split']==dict(user=1,gesture=1,both=1)
        assert population['public']['cross_role_user_intersection_counts']==dict(train_val=1,train_test=1,val_test=1)
        assert population['public']['episode_records']==1
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['payload','statistics','native','profile'])
def test_resealed_sidecar_cannot_override_raw_predictions_native_metrics_or_pretest_profile(evaluated,damage):
    m=api();path=evaluated/'test-replay/pass.json';before=path.read_bytes();value=json.loads(before)
    if damage=='payload':value['payload_sha256']='0'*64
    elif damage=='statistics':value['statistics']['consumed']['all']['overall']['valid_angle_count']-=22
    elif damage=='native':value['native_metrics']={'test_mae':0.}
    elif damage=='profile':value['profile_sha256']='1'*64
    from egoemg_repro.classic_emg import _hash
    value['sha256']=_hash({k:v for k,v in value.items() if k!='sha256'})
    try:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):m.build_vision_report(evaluated/'test-replay')
    finally:path.write_bytes(before)


def test_full_evaluation_refuses_engineering_fit_before_model_or_output(native_runs,monkeypatch,tmp_path):
    from egoemg_repro.vision_literal_cohort import admit_vision_cohort
    from test_vision_literal_data import config
    root,_,_,_=native_runs;m=api();cohort=admit_vision_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config())
    def forbidden(*args,**kwargs):raise AssertionError('must reject before model construction')
    monkeypatch.setattr(m,'build_vision_runtime',forbidden);output=tmp_path/'no-output'
    with pytest.raises(ValueError):m.run_vision_pass(cohort,root/'control',output,role='val',kind='native')
    assert not output.exists()


def test_test_requires_frozen_independent_validation_before_model_or_output(native_runs,monkeypatch,tmp_path):
    from egoemg_repro.vision_literal_cohort import admit_vision_cohort
    from test_vision_literal_data import config
    root,_,_,_=native_runs;m=api();cohort=admit_vision_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config())
    def forbidden(*args,**kwargs):raise AssertionError('must reject before model construction')
    monkeypatch.setattr(m,'build_vision_runtime',forbidden);output=tmp_path/'no-output'
    with pytest.raises(ValueError):m.run_vision_pass(cohort,root/'control',output,role='test',kind='native',require_full=False)
    assert not output.exists()


def test_publication_commits_only_after_raw_chain_and_never_overwrites(evaluated,tmp_path):
    m=api();output=tmp_path/'public'
    report=m.publish_vision_report(evaluated/'test-replay',output)
    assert m.verify_vision_report(evaluated/'test-replay',output)==report
    assert (output/'completion.json').is_file()
    text=(output/'metrics.md').read_text()
    assert 'MAE' in text and '12' in text and '工程' in text
    before={p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises((FileExistsError,ValueError)):m.publish_vision_report(evaluated/'test-replay',output)
    assert {p.name:p.read_bytes() for p in output.iterdir()}==before
    (output/'metrics.md').write_text('changed')
    with pytest.raises(ValueError):m.verify_vision_report(evaluated/'test-replay',output)


@pytest.mark.parametrize('name',['val-native','val-replay'])
def test_resealed_model_claim_must_match_raw_selected_checkpoint(evaluated,name):
    from egoemg_repro.classic_emg import _hash
    m=api();path=evaluated/name/'pass.json';before=path.read_bytes();value=json.loads(before)
    value['model_before_sha256']=value['model_after_sha256']='0'*64
    value['sha256']=_hash({k:v for k,v in value.items() if k!='sha256'})
    try:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError,match='selected.*model|model.*selected'):m.read_vision_pass(path.parent)
    finally:path.write_bytes(before)
