"""Full-only operational handoff must use actual original process evidence."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from egoemg_repro.emg2pose_memory_process import process_identity

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_emg2pose_full_queue.py'
def api():
    spec=importlib.util.spec_from_file_location('full_queue_operator',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def write(p,v):p.write_text(json.dumps(v))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@pytest.fixture
def case(tmp_path):
    root=tmp_path/'source-stage';(root/'private').mkdir(parents=True)
    code=root/'private/controller.py';code.write_text('# immutable unit controller fixture\n')
    owner=process_identity();state=dict(status='running',wrapper=owner,child=owner)
    write(root/'private/execution-state.json',state)
    config=dict(source_stage={'root':str(root),'files_sha256':{'private/controller.py':sha(code)}},
        source_launch={'state':json.loads(json.dumps(state))})
    return root,config,state

def test_import_does_not_load_torch():
    api();code="import importlib.util,sys; s=importlib.util.spec_from_file_location('queue',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules"
    r=subprocess.run([sys.executable,'-c',code,str(SCRIPT)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr

def test_authoritative_live_source_is_a_wait_and_never_opens_partial_result(case):
    m=api();root,config,_=case
    assert m.source_handoff(config) is None
    assert not (root/'private/full-source-result.json').exists()

def test_success_status_with_original_owner_still_alive_is_not_a_handoff(case):
    m=api();root,config,state=case;state.update(status='complete',actual_exit_code=0)
    write(root/'private/execution-state.json',state)
    assert m.source_handoff(config) is None

@pytest.mark.parametrize('fault',['failed','vanished','identity','producer'])
def test_failed_vanished_or_changed_original_source_cannot_start_any_model(case,fault):
    m=api();root,config,state=case
    if fault=='failed':state.update(status='failed',actual_exit_code=1)
    elif fault=='vanished':
        for k in ('wrapper','child'):
            state[k]=dict(state[k],start_ticks='0');config['source_launch']['state'][k]=state[k]
    elif fault=='identity':state['child']=dict(state['child'],start_ticks='0')
    else:(root/'private/controller.py').write_text('# changed\n')
    write(root/'private/execution-state.json',state)
    with pytest.raises((ValueError,RuntimeError)):m.source_handoff(config)

def test_fake_full_result_cannot_admit_engineering_source_marker(case,tmp_path):
    m=api();root,config,state=case
    for k in ('wrapper','child'):
        state[k]=dict(state[k],start_ticks='0');config['source_launch']['state'][k]=state[k]
    state.update(status='complete',actual_exit_code=0);write(root/'private/execution-state.json',state)
    source=tmp_path/'tiny-source';source.mkdir();write(source/'completion.json',{'cohort':'engineering','full_corpus_complete':False})
    write(root/'private/source-output.json',{'source':str(source),'output':str(source.parent)})
    write(root/'private/full-source-result.json',dict(cohort='full',full_corpus_complete=True,source_path=str(source),
        source_pipeline_completion_sha256=sha(source/'completion.json'),native_data_identity=dict(cohort='full',full_corpus_complete=True,
            engineering_only=False,file_count=25253,archive_bytes=462824048640,counts={'split':{'train':17136,'val':1950,'test':6167}})))
    with pytest.raises(ValueError):m.source_handoff(config)

@pytest.mark.parametrize('size',['small','middle','large'])
def test_commands_keep_full_budget_and_model_specific_frozen_qualifications(size,tmp_path):
    m=api();q=dict(upstream='/immutable/upstream',gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',
        training_qualifications={s:{'directory':'/qualified/train-'+s,'sha256':'a'*64} for s in m.SIZES},
        evaluation_qualifications={s:{'directory':'/qualified/eval-'+s,'sha256':'b'*64} for s in m.SIZES},
        controller_qualification={'directory':'/qualified/controller','sha256':'c'*64})
    h={'source_root':'/original/full-source','metadata_csv':'/original/full.csv','completion_sha256':'d'*64}
    argv=m.model_command(q,h,size,tmp_path/size)
    assert argv[argv.index('--scope')+1]=='full' and argv[argv.index('--size')+1]==size
    assert argv[argv.index('--qualification')+1]=='/qualified/train-'+size
    assert argv[argv.index('--evaluation-qualification')+1]=='/qualified/eval-'+size
    assert not any(x in argv for x in ('--stop-after-epoch','--deadline-seconds','--resume','--allow-engineering'))

def test_invalid_or_changed_config_refuses_before_wait_and_output(tmp_path):
    m=api();path=tmp_path/'request.json';write(path,{'format':'wrong'})
    for frozen in ('0'*64,sha(path)):
        with pytest.raises(ValueError):m.read_config(path,frozen)

def test_config_cannot_omit_child_owner_by_adding_an_unrelated_key(case,tmp_path):
    m=api();_,source,_=case
    q=dict(format=m.FORMAT,models=list(m.SIZES),gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',
        training_qualifications={s:{'directory':'/train/'+s,'sha256':'a'*64} for s in m.SIZES},
        evaluation_qualifications={s:{'directory':'/eval/'+s,'sha256':'b'*64} for s in m.SIZES},
        controller_qualification={'directory':'/driver','sha256':'c'*64},**source)
    q['source_launch']['state']={'wrapper':process_identity(),'unexpected':{}}
    path=tmp_path/'malformed.json';write(path,q)
    with pytest.raises(ValueError,match='owner'):m.read_config(path,sha(path))


def completed_full_case(case,tmp_path):
    root,config,state=case
    for k in ('wrapper','child'):
        state[k]=dict(state[k],start_ticks='0');config['source_launch']['state'][k]=state[k]
    state.update(status='complete',actual_exit_code=0);write(root/'private/execution-state.json',state)
    source=tmp_path/'actual-format-source';source.mkdir()
    write(source/'completion.json',dict(cohort='full',full_corpus_complete=True,pipeline_complete=True))
    write(root/'private/source-output.json',{'source':str(source)})
    # This is the actual producer's numeric identity shape. Archive provenance
    # belongs to VerifiedDirectSource, and is intentionally absent here.
    native=dict(format='emg2pose_runtime_data_identity_v1',cohort='full',full_corpus_complete=True,
        engineering_only=False,file_count=25253,rows=3043302266,counts={'split':{'train':17136,'val':1950,'test':6167}})
    owned=dict(worker_exit_code=0,status='success',all_owned_consumers_exited=True,descriptor_bindings={},
        worker=dict(state['wrapper']),guardian=dict(state['child']))
    (root/'private/owned-stage').mkdir();write(root/'private/owned-stage/stage-result.json',owned)
    result=dict(cohort='full',full_corpus_complete=True,source_path=str(source),
        source_pipeline_completion_sha256=sha(source/'completion.json'),native_data_identity=native,
        producer_files_sha256={},actual_owned_stage=owned)
    write(root/'private/full-source-result.json',result)
    return root,config,source,result


def test_actual_raw_numeric_identity_handoff_succeeds_before_verified_archive_admission(case,tmp_path):
    m=api();root,config,source,result=completed_full_case(case,tmp_path)
    assert 'archive_bytes' not in result['native_data_identity']
    handoff=m.source_handoff(config)
    assert handoff['source_root']==str(source) and handoff['completion_sha256']==sha(source/'completion.json')
    assert handoff['source_result_sha256']==sha(root/'private/full-source-result.json')
    # Correcting the raw boundary must not relax the later verified-source gate.
    with pytest.raises(ValueError,match='full source|25253'):m.driver._full_source(result['native_data_identity'])


@pytest.mark.parametrize('damage',['format','cohort','incomplete','engineering','count','roles'])
def test_raw_handoff_still_requires_complete_original_population(case,tmp_path,damage):
    m=api();root,config,_,result=completed_full_case(case,tmp_path);native=result['native_data_identity']
    if damage=='format':native['format']='emg2pose_verified_direct_data_identity_v1'
    elif damage=='cohort':native['cohort']='mini'
    elif damage=='incomplete':native['full_corpus_complete']=False
    elif damage=='engineering':native['engineering_only']=True
    elif damage=='count':native['file_count']=25252
    else:native['counts']['split']['val']=1951
    write(root/'private/full-source-result.json',result)
    with pytest.raises(ValueError,match='native|full|25253'):m.source_handoff(config)
