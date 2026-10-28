"""Real selected fusion checkpoint, pretest validation replay and final test audit."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_fusion_literal_train import native_runs as trained_runs,ROOT,UPSTREAM,branch_files,native_packet


def api():
    try:return importlib.import_module('egoemg_repro.fusion_literal_eval')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.fusion_literal_eval':pytest.fail('selected-best native fusion evaluation missing')
        raise


@pytest.fixture(scope='module')
def native_runs(request):
    retained=os.environ.get('EGOEMG_FUSION_EVAL_RETAINED_FIT')
    if retained is None:return request.getfixturevalue('trained_runs')
    from egoemg_repro.fusion_literal_train import read_fusion_fit
    root=Path(retained).resolve();done=read_fusion_fit(root/'resumed',require_full=False)
    assert done['stage_complete'] and done['completed_epochs']==2 and done['optimizer_executions']==4
    prefixes=[p.read_bytes() for p in (root/'resumed').glob('historical-fit-result-*.json')
        if json.loads(p.read_bytes())['completed_epochs']==1]
    assert len(prefixes)==1
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    return root,env,None,prefixes[0],{},None,None


def test_statistics_preserve_generated_mask_missing_crops_and_repeat_denominators():
    torch=pytest.importorskip('torch');m=api()
    # Four consumed hands, two unique. Missing right crop must retain its label.
    part=dict(pred=torch.ones(4,22,1),target=torch.zeros(4,22,1),mask=torch.ones(4,1,dtype=torch.bool),
        index=torch.arange(4),unique=torch.tensor([0,1,0,1]),repeat=torch.tensor([0,0,1,1]),
        vision=torch.tensor([[True],[False],[True],[False]]),hand=['left','right']*2,
        split=['user']*4,user=['private-person']*4,center=torch.tensor([3895]*4),
        input_sha256='a'*64,sample_input_sha256=['a'*64,'b'*64]*2,episode=torch.zeros(4,dtype=torch.int64),window=torch.zeros(4,dtype=torch.int64))
    value=m.fusion_statistics([part],samples=4,batch_size=4,role='test')
    assert value['consumed']['all']['overall']['valid_angle_count']==88
    assert value['unique_first_repeat']['all']['overall']['valid_angle_count']==44
    assert value['consumed']['samples']==dict(all=4,vision_valid=2,missing_crop=2)
    assert value['unique_samples']==2 and value['duplicate_consumptions']==2
    assert value['consumed']['all']['overall']['absolute_error_sum_rad']==88.
    assert value['consumed']['all']['by_joint'][21]['valid_angle_count']==4
    assert value['native_batch_weighted_mae_rad']==1.
    assert value['consumed']['all']['by_split_user_summary']['user']['std_mae_deg']==0.
    broken=copy.deepcopy(part);broken['index'][3]=2
    with pytest.raises(ValueError):m.fusion_statistics([broken],samples=4,batch_size=4,role='test')
    # Independent validation is held out from the original train split.
    validation=copy.deepcopy(part);validation['split']=['train']*4
    val=m.fusion_statistics([validation],samples=4,batch_size=4,role='val')
    assert val['consumed']['all']['by_split']['train']['valid_angle_count']==88
    with pytest.raises(ValueError):m.fusion_statistics([validation],samples=4,batch_size=4,role='test')


def test_native_batch_weighting_is_not_silently_replaced_by_pooled_valid_angles():
    torch=pytest.importorskip('torch');m=api()
    def part(start,errors,valid):
        n=len(errors)
        return dict(pred=torch.tensor(errors).reshape(n,1,1).expand(n,22,1).clone(),target=torch.zeros(n,22,1),
            mask=torch.tensor(valid).reshape(n,1),index=torch.arange(start,start+n),
            unique=torch.arange(start,start+n)%2,repeat=torch.arange(start,start+n)//2,
            vision=torch.ones(n,1,dtype=torch.bool),hand=['left' if (start+i)%2==0 else 'right' for i in range(n)],
            split=['user']*n,user=['p']*n,center=torch.tensor([3895]*n),input_sha256='b'*64,sample_input_sha256=['a'*64]*n,episode=torch.zeros(n,dtype=torch.int64),window=torch.zeros(n,dtype=torch.int64))
    parts=[part(0,[1.,100.,3.],[True,False,True]),part(3,[9.],[False])]
    value=m.fusion_statistics(parts,samples=4,batch_size=3,role='test')
    assert value['native_batch_weighted_mae_rad']==1.5
    assert value['consumed']['all']['overall']['absolute_error_sum_rad']==88.
    assert value['consumed']['all']['overall']['valid_angle_count']==44


@pytest.fixture(scope='module')
def evaluated(native_runs):
    root,env,*_=native_runs;m=api()
    existing=os.environ.get('EGOEMG_FUSION_EVAL_EXISTING_PASSES')
    if existing:
        output=Path(existing);m.read_fusion_pass(output/'test-replay');return output
    output=root/('evaluation-'+m.IMPORTED_SHA[:12]);output.mkdir()
    base=[sys.executable,'-B','-m','egoemg_repro.fusion_literal_eval']
    shared=['--data',str(root/'data'),'--crops',str(root/'crops'),'--upstream',str(UPSTREAM),
        '--fit',str(root/'resumed'),'--engineering']
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
    m=api();val=m.read_fusion_pass(evaluated/'val-native');native=m.read_fusion_pass(evaluated/'test-native')
    replay=m.read_fusion_pass(evaluated/'test-replay');profile=m.read_fusion_profile(evaluated/'profile')
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
    m=api();report=m.build_fusion_report(evaluated/'test-replay',require_full=False)
    raw=json.dumps(report)
    assert report['scientific_eligible'] is False and report['full_paper_complete'] is False
    assert report['test']['unique_samples']==6 and report['training']['completed_epochs']==2
    assert report['training']['execution_policy']['early_stopping'] is False
    assert report['training']['execution_policy']['scheduler_T_max']==150
    assert '工程' in m.render_fusion_report(report)
    assert report['initializers']['scope']=='engineering'
    assert 'private-person' not in raw and str(ROOT) not in raw and '/tmp/' not in raw
    assert list(report['test']['consumed']['all']['by_user'])==['user_000']
    assert report['population']['unique_subjects_by_role']==dict(train=1,val=1,test=1)
    assert report['population']['train_test_user_intersection_by_split']==dict(user=1,gesture=1,both=1)


def test_population_intersections_come_from_actual_admitted_episode_metadata(native_runs):
    from egoemg_repro.fusion_literal_cohort import admit_fusion_cohort
    from egoemg_repro.fusion_literal_data import fusion_normalization
    from test_fusion_literal_data import config
    root,*_=native_runs
    cohort=admit_fusion_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config(),normalization=fusion_normalization(UPSTREAM))
    population=api().fusion_population(cohort)
    assert population['public']['unique_subjects_by_role']==dict(train=1,val=1,test=1)
    assert population['public']['train_test_user_intersection_by_split']==dict(user=1,gesture=1,both=1)
    assert population['public']['cross_role_user_intersection_counts']==dict(train_val=1,train_test=1,val_test=1)
    assert population['public']['episode_records']==1


@pytest.mark.parametrize('damage',['payload','statistics','native','profile','best','population','process','policy','premature'])
def test_resealed_sidecar_cannot_override_raw_predictions_native_metrics_or_pretest_profile(evaluated,damage):
    m=api();path=evaluated/'test-replay/pass.json';before=path.read_bytes();value=json.loads(before)
    if damage=='payload':value['payload_sha256']='0'*64
    elif damage=='statistics':value['statistics']['consumed']['all']['overall']['valid_angle_count']-=22
    elif damage=='native':value['native_metrics']={'test_mae':0.}
    elif damage=='profile':value['profile_sha256']='1'*64
    elif damage=='best':value['selected_best']['file']='last.ckpt'
    elif damage=='population':value['population']['episode_records']+=1
    elif damage=='process':value['process']=m.read_fusion_pass(evaluated/'test-native')['process']
    elif damage=='policy':value['fixed_agreement_policy']['native_mae_atol_rad']=1.
    elif damage=='premature':value['started_ns']=1
    from egoemg_repro.classic_emg import _hash
    value['sha256']=_hash({k:v for k,v in value.items() if k!='sha256'})
    try:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):m.build_fusion_report(evaluated/'test-replay',require_full=False)
    finally:path.write_bytes(before)


def test_full_evaluation_refuses_engineering_fit_before_model_or_output(native_runs,monkeypatch,tmp_path):
    from egoemg_repro.fusion_literal_cohort import admit_fusion_cohort
    from egoemg_repro.fusion_literal_data import fusion_normalization
    from test_fusion_literal_data import config
    root,*_=native_runs;m=api();cohort=admit_fusion_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config(),normalization=fusion_normalization(UPSTREAM))
    def forbidden(*args,**kwargs):raise AssertionError('must reject before model construction')
    monkeypatch.setattr(m,'build_fusion_runtime',forbidden);output=tmp_path/'no-output'
    with pytest.raises(ValueError):m.run_fusion_pass(cohort,root/'resumed',output,role='val',kind='native')
    assert not output.exists()


def test_test_requires_frozen_independent_validation_before_model_or_output(native_runs,monkeypatch,tmp_path):
    from egoemg_repro.fusion_literal_cohort import admit_fusion_cohort
    from egoemg_repro.fusion_literal_data import fusion_normalization
    from test_fusion_literal_data import config
    root,*_=native_runs;m=api();cohort=admit_fusion_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config(),normalization=fusion_normalization(UPSTREAM))
    def forbidden(*args,**kwargs):raise AssertionError('must reject before model construction')
    monkeypatch.setattr(m,'build_fusion_runtime',forbidden);output=tmp_path/'no-output'
    with pytest.raises(ValueError):m.run_fusion_pass(cohort,root/'resumed',output,role='test',kind='native',require_full=False)
    assert not output.exists()


def test_publication_commits_only_after_raw_chain_and_never_overwrites(evaluated,tmp_path):
    m=api();output=tmp_path/'public'
    report=m.publish_fusion_report(evaluated.parent/'resumed',evaluated/'test-native',evaluated/'test-replay',output,require_full=False)
    assert m.verify_fusion_report(evaluated/'test-replay',output,require_full=False)==report
    assert (output/'completion.json').is_file()
    text=(output/'metrics.md').read_text()
    assert 'MAE' in text and '12' in text and '工程' in text
    before={p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises((FileExistsError,ValueError)):m.publish_fusion_report(evaluated.parent/'resumed',evaluated/'test-native',evaluated/'test-replay',output,require_full=False)
    assert {p.name:p.read_bytes() for p in output.iterdir()}==before
    (output/'metrics.md').write_text('changed')
    with pytest.raises(ValueError):m.verify_fusion_report(evaluated/'test-replay',output,require_full=False)


@pytest.mark.parametrize('name',['val-native','val-replay'])
def test_resealed_model_claim_must_match_raw_selected_checkpoint(evaluated,name):
    from egoemg_repro.classic_emg import _hash
    m=api();path=evaluated/name/'pass.json';before=path.read_bytes();value=json.loads(before)
    value['model_before_sha256']=value['model_after_sha256']='0'*64
    value['sha256']=_hash({k:v for k,v in value.items() if k!='sha256'})
    try:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError,match='selected.*model|model.*selected'):m.read_fusion_pass(path.parent)
    finally:path.write_bytes(before)


@pytest.mark.parametrize('damage',['repeat_target','repeat_user','repeat_input','index_dtype','mask_dtype','angles','unknown_index'])
def test_repeated_consumptions_are_paired_to_the_same_typed_multimodal_input(damage):
    torch=pytest.importorskip('torch');m=api()
    part=dict(pred=torch.ones(4,22,1),target=torch.zeros(4,22,1),mask=torch.ones(4,1,dtype=torch.bool),
        index=torch.arange(4),unique=torch.tensor([0,1,0,1]),repeat=torch.tensor([0,0,1,1]),
        vision=torch.tensor([[True],[False],[True],[False]]),hand=['left','right']*2,split=['user']*4,
        user=['private-person']*4,center=torch.tensor([3895]*4),input_sha256='a'*64,
        sample_input_sha256=['a'*64,'b'*64]*2,episode=torch.zeros(4,dtype=torch.int64),window=torch.zeros(4,dtype=torch.int64))
    if damage=='repeat_target':part['target'][2,0,0]=1.
    elif damage=='repeat_user':part['user'][2]='another-private-person'
    elif damage=='repeat_input':part['sample_input_sha256'][2]='c'*64
    elif damage=='index_dtype':part['unique']=part['unique'].float()
    elif damage=='mask_dtype':part['mask']=part['mask'].float()
    elif damage=='angles':part['pred']=part['pred'][:,:20]
    else:part['index'][-1]=4
    with pytest.raises(ValueError):m.fusion_statistics([part],samples=4,batch_size=4,role='test')


def test_default_publication_and_wrappers_reject_engineering_or_unpaired_test(evaluated,native_runs,tmp_path):
    m=api();root,*_=native_runs;output=tmp_path/'no-scientific-report'
    with pytest.raises(ValueError):m.publish_fusion_report(root/'resumed',evaluated/'test-native',evaluated/'test-replay',output)
    assert not output.exists()
    with pytest.raises(ValueError):m.publish_fusion_report(root/'crash',evaluated/'test-native',evaluated/'test-replay',output,require_full=False)
    assert not output.exists()
    with pytest.raises(ValueError):m.publish_fusion_report(root/'resumed',evaluated/'val-native',evaluated/'test-replay',output,require_full=False)
    assert not output.exists()


def test_incomplete_fit_cannot_evaluate_even_in_engineering_mode(native_runs,monkeypatch,tmp_path):
    from egoemg_repro.fusion_literal_cohort import admit_fusion_cohort
    from egoemg_repro.fusion_literal_data import fusion_normalization
    from test_fusion_literal_data import config
    root,_,_,prefix,*_=native_runs;m=api()
    cohort=admit_fusion_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config(),normalization=fusion_normalization(UPSTREAM))
    def forbidden(*args,**kwargs):raise AssertionError('incomplete fit must fail before model construction')
    monkeypatch.setattr(m,'build_fusion_runtime',forbidden)
    path=root/'resumed/fit-result.json';before=path.read_bytes();output=tmp_path/'incomplete'
    try:
        path.write_bytes(prefix)
        with pytest.raises(ValueError):m.evaluate_fusion(cohort,root/'resumed',output,role='val',require_full=False)
        assert not output.exists()
    finally:path.write_bytes(before)


def test_curves_are_bound_to_actual_native_epoch_journal_and_completion(evaluated,native_runs,tmp_path):
    m=api();root,*_=native_runs;output=tmp_path/'curves'
    report=m.publish_fusion_report(root/'resumed',evaluated/'test-native',evaluated/'test-replay',output,require_full=False)
    curves=json.loads((output/'curve-data.json').read_text())
    assert len(curves['epochs'])==report['training']['completed_epochs']==2
    assert [r['optimizer_executions'] for r in curves['epochs']]==[2,4]
    assert curves['selected_best']==report['training']['selected_best']
    assert curves['validation_unit']=='radian; original sample-weighted val_mae'
    assert (output/'training-curves.png').read_bytes().startswith(b'\x89PNG')
    completion=json.loads((output/'completion.json').read_text())
    assert set(completion['files_sha256'])=={'metrics.json','metrics.md','curve-data.json','training-curves.png','training-curves.svg'}
    # Resealing a picture receipt must not replace the plot derived from raw epochs.
    from egoemg_repro.classic_emg import _hash,_file_hash
    (output/'training-curves.svg').write_text('<svg>not the actual training curve</svg>')
    completion['files_sha256']['training-curves.svg']=_file_hash(output/'training-curves.svg')
    completion['sha256']=_hash({k:v for k,v in completion.items() if k!='sha256'})
    (output/'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError):m.verify_fusion_report(evaluated/'test-replay',output,require_full=False)



def test_report_refuses_epoch_journal_change_during_curve_preparation(evaluated,native_runs,monkeypatch):
    m=api();root,*_=native_runs;path=root/'resumed/epochs.jsonl';before=path.read_bytes();original=m._curve_data
    def changed(*args):
        result=original(*args);path.write_bytes(before+b'\n');return result
    monkeypatch.setattr(m,'_curve_data',changed)
    try:
        with pytest.raises(ValueError):m.build_fusion_report(evaluated/'test-replay',require_full=False)
    finally:path.write_bytes(before)


def test_curve_selection_epoch_is_the_actual_selected_checkpoint(evaluated,native_runs):
    torch=pytest.importorskip('torch');root,*_=native_runs;m=api()
    report=m.build_fusion_report(evaluated/'test-replay',require_full=False)
    selected=root/'resumed/checkpoints'/report['training']['selected_best']['file']
    raw=torch.load(str(selected),map_location='cpu',weights_only=False)
    assert report['curve_data']['selected_epoch_zero_based']==raw['fusion_fit']['completed_epochs']-1
    assert report['plot_runtime']['font_sha256'] and report['plot_runtime']['matplotlib_version']
