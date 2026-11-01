"""The real native CLI handoffs and owned-process stop boundaries, without fits."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    try:return importlib.import_module('egoemg_repro.rn18_fusion_experiment')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_fusion_experiment':pytest.fail('native RN18 qualification/full experiment orchestration missing')
        raise


def request(root):
    return {name:str(root/name)for name in ('data','crops','upstream','cohort_receipt','branches','handoff_request')}


def test_native_cli_plan_freezes_validation_before_test_and_uses_fresh_full_fit(tmp_path):
    stages=api().rn18_fusion_experiment_stages(request(tmp_path),tmp_path/'run')
    assert len(stages)==20 and len({s['name']for s in stages})==20
    names=[s['name']for s in stages]
    assert names[:4]==['trained-reopen','qual-prefix','qual-resume','qual-accept']
    assert names[-3:]==['full-test-replay','full-report','full-verify']
    for family in ('qual','full'):
        group={s['name']:s for s in stages if s['name'].startswith(family+'-')}
        assert names.index(family+'-val-replay')<names.index(family+'-freeze')<names.index(family+'-test-native')
        native=group[family+'-test-native']['argv'];replay=group[family+'-test-replay']['argv']
        assert '--profile'in native and '--profile'in replay and '--reference'in replay
        assert native[native.index('--profile')+1]==str(tmp_path/'run'/f'{family}-evaluation/profile')
        report=group[family+'-report']['argv'];verify=group[family+'-verify']['argv']
        assert report[1:3]==['egoemg_repro.rn18_fusion_eval','report']
        assert all(x in report for x in ('--fit','--evaluation','--replay','--output'))
        assert '--recompute'in verify and '--replay'not in verify
        assert ('--engineering'in native)is(family=='qual')
        assert ('--engineering'in report)is(family=='qual')
        assert ('--engineering'in verify)is(family=='qual')
    prefix=stages[1]['argv'];resume=stages[2]['argv'];full=next(s['argv']for s in stages if s['name']=='full-fit')
    assert prefix[prefix.index('--stop-after-epoch')+1]=='1'
    assert prefix[prefix.index('--scope')+1]=='cuda_engineering'and '--resume'in resume
    assert full[full.index('--scope')+1]=='full'and '--resume'not in full and '--stop-after-epoch'not in full
    assert full[full.index('--output')+1]!=prefix[prefix.index('--output')+1]
    assert sum(s['cuda']for s in stages)==11


@pytest.mark.parametrize('damage',['extra','missing','relative','traversal','bool'])
def test_extra_or_unsafe_cli_locators_do_not_reach_a_stage_plan(tmp_path,damage):
    q=request(tmp_path)
    if damage=='extra':q['extra_args']=['--stop-after-epoch','1']
    elif damage=='missing':del q['branches']
    elif damage=='relative':q['data']='relative/data'
    elif damage=='traversal':q['crops']=str(tmp_path/'../crops')
    else:q['upstream']=True
    with pytest.raises(ValueError):api().rn18_fusion_experiment_stages(q,tmp_path/'run')


def test_qualification_acceptance_requires_actual_two_epoch_four_update_six_state_scope():
    v=dict(scope='cuda_engineering',completed_epochs=2,optimizer_executions=4,stage_complete=True,
        full_training_terminated=False,resume_start_exact={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')},
        resume_validation_exact=True,train_samples_per_epoch=[400,400],val_samples_per_epoch=[1052,1052],scientific_eligible=False)
    m=api();m._check_fit_result(v,full=False)
    for key,value in [('scope','cpu_engineering'),('completed_epochs',True),('optimizer_executions',8),
        ('resume_validation_exact',False),('train_samples_per_epoch',[200,200]),('val_samples_per_epoch',[2,2]),
        ('resume_start_exact',{k:True for k in ('model','optimizer','scheduler','callbacks','loops')})]:
        wrong=copy.deepcopy(v);wrong[key]=value
        with pytest.raises(ValueError):m._check_fit_result(wrong,full=False)
    with pytest.raises(ValueError):m._check_fit_result(v,full=True)


def test_full_budget_cannot_be_satisfied_by_partial_epochs_or_engineering_counts():
    v=dict(scope='full',completed_epochs=150,optimizer_executions=46950,stage_complete=True,
        full_training_terminated=True,train_samples_per_epoch=[62470]*150,val_samples_per_epoch=[1052]*150,scientific_eligible=False)
    m=api();m._check_fit_result(v,full=True)
    for key,value in [('completed_epochs',149),('optimizer_executions',46800),('full_training_terminated',False),
        ('train_samples_per_epoch',[400]*150),('val_samples_per_epoch',[1052]*149),('scientific_eligible',True)]:
        wrong=copy.deepcopy(v);wrong[key]=value
        with pytest.raises(ValueError):m._check_fit_result(wrong,full=True)


def tiny_stage(name,code):return dict(name=name,argv=['-c',code],cuda=False)


def test_actual_owned_processes_run_in_order_and_all_consumers_close(tmp_path):
    target=tmp_path/'order.txt';env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    stages=[tiny_stage('one',f"from pathlib import Path;Path({str(target)!r}).write_text('one')"),
        tiny_stage('two',f"from pathlib import Path;p=Path({str(target)!r});assert p.read_text()=='one';p.write_text('one,two')")]
    result=api()._run_stages(stages,tmp_path/'owned',guard=lambda:None,environment=env,after_stage=lambda stage:None)
    assert target.read_text()=='one,two'and len(result)==2
    from egoemg_repro.emg2pose_memory_process import same_process
    for stage in result:
        assert stage['status']=='success'and stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']
        assert not same_process(stage['worker'])and not same_process(stage['guardian'])


def test_actual_child_failure_stops_before_later_full_fit_and_closes_descendants(tmp_path):
    forbidden=tmp_path/'should-not-start';child_record=tmp_path/'descendant.json';env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    spawn="import json,subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);s=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split();Path("+repr(str(child_record))+").write_text(json.dumps(dict(pid=p.pid,start_ticks=s[19])));raise SystemExit(7)"
    stages=[tiny_stage('qual-fail',spawn),
        tiny_stage('full-fit',f"from pathlib import Path;Path({str(forbidden)!r}).touch()")]
    with pytest.raises(RuntimeError):
        api()._run_stages(stages,tmp_path/'owned',guard=lambda:None,environment=env,after_stage=lambda stage:None)
    assert not forbidden.exists()and not(tmp_path/'owned/owned-full-fit').exists()
    stage=json.loads((tmp_path/'owned/owned-qual-fail/stage-result.json').read_text())
    assert stage['worker_exit_code']==7 and stage['all_owned_consumers_exited']
    from egoemg_repro.emg2pose_memory_process import same_process
    assert not same_process(json.loads(child_record.read_text()))


def test_native_cli_output_parent_exists_before_the_owned_stage_starts(tmp_path):
    output=tmp_path/'evaluation/val-native'
    stage=dict(name='val-native',cuda=False,argv=['-c',
        "import sys;from pathlib import Path;p=Path(sys.argv[-1]);assert p.parent.is_dir();p.mkdir();(p/'complete').touch()",
        '--output',str(output)])
    api()._run_stages([stage],tmp_path/'owned',guard=lambda:None,
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''),after_stage=lambda stage:None)
    assert(output/'complete').is_file()


def test_cpu_controller_rejects_cuda_visibility_before_any_owned_child(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-test')
    with pytest.raises(ValueError):api().run_rn18_fusion_experiment(request(tmp_path),tmp_path/'run',gpu_uuid='GPU-test')
    assert not(tmp_path/'run').exists()


def test_source_changed_after_import_is_rejected_before_any_owned_child(tmp_path):
    q=request(tmp_path)
    for name in ('data','crops','upstream'):(tmp_path/name).mkdir()
    handoff={k:q[k]for k in ('data','crops','upstream','cohort_receipt')}
    handoff.update({k:str(tmp_path/k)for k in ('vision_fit','emg_fit','admission','vision_source_root','emg_source_root')})
    branches=dict(vision_fit=handoff['vision_fit'],emg_fit=handoff['emg_fit'],output=handoff['admission'],
        vision_source_root=handoff['vision_source_root'],emg_source_root=handoff['emg_source_root'])
    for name,value in [('cohort_receipt',{}),('branches',branches),('handoff_request',handoff)]:
        (tmp_path/name).write_text(json.dumps(value))
    # The controller's first boundary requires a genuinely ML-free interpreter;
    # earlier fit/evaluation tests may already have imported Torch in pytest.
    code='''import json,sys
from pathlib import Path
import egoemg_repro.rn18_fusion_experiment as m
assert 'torch' not in sys.modules
q=json.loads(sys.argv[1]);root=Path(sys.argv[2])
code=root/'code';code.mkdir();source=code/'rn18_fusion_experiment.py'
source.write_bytes(Path(m.__file__).read_bytes()+b'\\n');m.__file__=str(source)
def forbidden(*a,**k):raise AssertionError('changed imported source reached owned process dispatch')
m._run_stages=forbidden
try:
 m.run_rn18_fusion_experiment(q,root/'run',gpu_uuid='GPU-00000000-0000-0000-0000-000000000000')
except ValueError as exc:
 assert 'imported experiment source' in str(exc),str(exc)
else:raise AssertionError('changed imported source was accepted')
assert not(root/'run').exists()
'''
    result=subprocess.run([sys.executable,'-B','-c',code,json.dumps(q),str(tmp_path)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert not(tmp_path/'run').exists()
