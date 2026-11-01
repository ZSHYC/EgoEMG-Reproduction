"""Owner/exit/source boundary only; a ready queue never admits trained weights."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from egoemg_repro.emg2pose_memory_process import process_identity

STAGES=('qual-prefix','qual-resume','qual-export','qual-reopen','full-fit','full-export','full-reopen')


def api():
    try:return importlib.import_module('egoemg_repro.rn18_fusion_workflow')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_fusion_workflow':pytest.fail('source-bound RN18 producer queue missing')
        raise


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))


@pytest.fixture(scope='module')
def dead_handles():
    result=[]
    for _ in range(16):
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.04)'])
        h=process_identity(p.pid);assert h is not None;p.wait(timeout=3);assert process_identity(p.pid)is None
        result.append(h)
    return result


@pytest.fixture
def producer(tmp_path,dead_handles):
    root=tmp_path/'producer';private=root/'private';private.mkdir(parents=True)
    (root/'source.py').write_text('SOURCE = 1\n')
    body=dict(source_commit='a'*40,files_sha256={'source.py':sha(root/'source.py')},full_paper_complete=False)
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(private/'stage-receipt.json',dict(body,bundle_sha256=digest))
    handles={k:{n:dead_handles[i][n]for n in ('pid','start_ticks')}for i,k in enumerate(('wrapper','child'))}
    state=dict(status='complete',actual_exit_code=0,**copy.deepcopy(handles));write(private/'execution-state.json',state)
    stages=[];observed={}
    for i,label in enumerate(STAGES):
        started=dict(owner=dead_handles[1],worker=dead_handles[2+i*2],guardian=dead_handles[3+i*2])
        result=dict(status='success',worker_exit_code=0,all_owned_consumers_exited=True,**started)
        write(private/('owned-'+label)/'started.json',started);write(private/('owned-'+label)/'stage-result.json',result)
        stages.append(result)
        if label in ('qual-prefix','qual-resume','full-fit'):
            observed[label]=dict(worker=started['worker'],actual_nvml_pid=started['worker']['pid'],gpu_uuid='test-metadata-only')
    fit=dict(scope='full',stage_complete=True,full_training_terminated=True,completed_epochs=150,optimizer_executions=36150)
    workflow=dict(format='actual_rn18_cotrain_original_batch_qualification_and_full150_workflow_v1',
        source_commit='a'*40,bundle_sha256=digest,actual_owned_stages=stages,actual_nvml_observations=observed,
        actual_selected_full_initializer=dict(fit_result=fit,selected_export=dict(full_initializer=True,tensor_inventory=[{}]*110)),
        all_owned_consumers_exited=True,all_frozen_members_unchanged=True,trained_full_initializer_complete=True,
        scientific_settings_added=0,full_paper_complete=False)
    write(private/'workflow-result.json',workflow)
    return root,digest,handles,state,workflow


def status(producer):
    root,digest,handles,*_=producer
    return api().cotrain_dependency_status(root,expected_bundle=digest,expected_handles=handles)


def test_all_seven_actual_records_must_close_before_independent_admission(producer,monkeypatch):
    import builtins
    original=builtins.__import__
    def no_ml(name,*args,**kwargs):
        if name.split('.')[0]in ('torch','pytorch_lightning'):raise AssertionError('queue gate imported ML')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_ml)
    r=status(producer)
    assert r['ready_for_independent_admission']and r['closed_owned_stages']==list(STAGES)
    assert r['closed_owner_handles']==16 and not r['full_trained_handoff_admitted']and not r['scientific_eligible']


@pytest.mark.parametrize('damage',['failed','wrong_handle','capsule','worker_live_flag','stage_bytes','missing_stage','partial_budget','missing_nvml','fake_exit'])
def test_no_terminal_or_source_shortcut_grants_ready(producer,damage):
    root,digest,handles,state,w=producer;p=root/'private'
    if damage=='failed':state.update(status='failed',actual_exit_code=1);write(p/'execution-state.json',state)
    elif damage=='wrong_handle':state['child']['start_ticks']='0';write(p/'execution-state.json',state)
    elif damage=='capsule':(root/'source.py').write_text('SOURCE = 2\n')
    elif damage=='worker_live_flag':w['actual_owned_stages'][0]['all_owned_consumers_exited']=False;write(p/'workflow-result.json',w)
    elif damage=='stage_bytes':v=json.loads((p/'owned-full-fit/stage-result.json').read_text());v['worker_exit_code']=1;write(p/'owned-full-fit/stage-result.json',v)
    elif damage=='missing_stage':(p/'owned-full-reopen/stage-result.json').unlink()
    elif damage=='partial_budget':w['actual_selected_full_initializer']['fit_result']['completed_epochs']=149;write(p/'workflow-result.json',w)
    elif damage=='missing_nvml':w['actual_nvml_observations'].pop('full-fit');write(p/'workflow-result.json',w)
    else:state['actual_exit_code']=False;write(p/'execution-state.json',state)
    with pytest.raises(ValueError):status(producer)


@pytest.fixture
def live_pair():
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:yield process_identity(),process_identity(p.pid)
    finally:p.terminate();p.wait(timeout=3)


@pytest.mark.parametrize('kind',['running','child_finishing','terminal_owners_finishing'])
def test_finalization_is_waited_without_restart_or_early_admission(producer,live_pair,kind):
    root,digest,handles,state,w=producer;parent,child=live_pair
    state={k:{n:h[n]for n in ('pid','start_ticks')}for k,h in zip(('wrapper','child'),(parent,child))}
    if kind=='child_finishing':state['child']=handles['child']
    state['status']='complete'if kind=='terminal_owners_finishing'else'running'
    if kind=='terminal_owners_finishing':state['actual_exit_code']=0
    write(root/'private/execution-state.json',state)
    expected={k:state[k]for k in ('wrapper','child')}
    r=api().cotrain_dependency_status(root,expected_bundle=digest,expected_handles=expected)
    assert not r['ready_for_independent_admission']and r['status']=='waiting'and not r['full_trained_handoff_admitted']


def test_missing_wrapper_with_stale_nonterminal_receipt_is_not_success(producer):
    root,_,_,state,_=producer;state.pop('actual_exit_code');state['status']='running';write(root/'private/execution-state.json',state)
    with pytest.raises(ValueError):status(producer)


def test_cpu_action_rejects_visible_gpu_before_any_import_or_output(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError):api().run_handoff_action('admit',{},tmp_path/'not-created')
    assert not(tmp_path/'not-created').exists()


@pytest.mark.parametrize('race',['source_after_hash','terminal_during_workflow_read'])
def test_source_and_terminal_generation_are_guarded_across_the_complete_read(producer,monkeypatch,race):
    root,*_=producer;m=api()
    if race=='source_after_hash':
        original=m._sha;done=False
        def race_sha(path):
            nonlocal done
            value=original(path)
            if Path(path)==root/'source.py'and not done:
                done=True;(root/'source.py').write_text('SOURCE = 2\n')
            return value
        monkeypatch.setattr(m,'_sha',race_sha)
    else:
        original=m._json;done=False
        def race_json(path):
            nonlocal done
            value=original(path)
            if Path(path)==root/'private/workflow-result.json'and not done:
                done=True;state=json.loads((root/'private/execution-state.json').read_text());state.update(status='failed',actual_exit_code=1)
                write(root/'private/execution-state.json',state)
            return value
        monkeypatch.setattr(m,'_json',race_json)
    with pytest.raises(ValueError):status(producer)


def test_actual_engineering_input_never_reaches_full_handoff_or_constructor(tmp_path,monkeypatch):
    pytest.importorskip('pytorch_lightning')
    import torch,numpy as np
    from test_fusion_literal_data import joint_store,UPSTREAM
    from test_vision_literal_data import crop_store,release_fixture_readers
    from test_rn18_fusion_data import config
    from egoemg_repro.rn18_fusion_cohort import admit_rn18_fusion_cohort
    data=tmp_path/'data';joint_store(data,n=9*12000)
    np.repeat([0,1,2,3],[6*12000,12000,12000,12000]).astype(np.int8).tofile(data/'frame_split_id.dat')
    crops=tmp_path/'crops';crop_store(crops,flat=True)
    c=admit_rn18_fusion_cohort(data,crops,upstream=UPSTREAM,config=config(),require_full=False)
    receipt=tmp_path/'cohort.json';write(receipt,c.identity);attempts=[]
    def deny(*a,**k):attempts.append(True);raise AssertionError('engineering input reached a learned constructor')
    monkeypatch.setattr(torch.nn.Module,'__init__',deny);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    q=dict(data=str(data),crops=str(crops),upstream=str(UPSTREAM),cohort_receipt=str(receipt),
        vision_fit=str(tmp_path/'absent-vision'),emg_fit=str(tmp_path/'absent-emg'),
        vision_source_root=str(tmp_path),emg_source_root=str(tmp_path),admission=str(tmp_path/'not-created-admission'),
        runtime_packet=str(tmp_path/'not-created.pt'),construction_record=str(tmp_path/'not-created.json'))
    try:
        with pytest.raises(ValueError):api().run_handoff_action('admit',q,tmp_path/'not-created-output.json')
        assert not attempts and not(tmp_path/'not-created-admission').exists()and not(tmp_path/'not-created-output.json').exists()
    finally:release_fixture_readers(tmp_path)
