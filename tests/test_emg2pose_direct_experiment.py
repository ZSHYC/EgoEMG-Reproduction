"""Actual NoTorch orchestration, native phases and scientific completion gates."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
pytest.importorskip('hydra');pytest.importorskip('tensorboard')
from test_emg2pose_direct_evaluate import memory,corpus,UPSTREAM,sha

def api():return importlib.import_module('egoemg_repro.emg2pose_direct_experiment')

def invoke(memory,output,*,fault=None,**changes):
    kwargs=dict(upstream=str(UPSTREAM),source_root=str(memory.source_root),metadata_csv=str(memory.metadata),size='small',
        output=str(output),expected_source_completion_sha256=memory.receipt['completion_sha256'],scope='cpu_engineering')
    kwargs.update(changes)
    # Local host has ~7GiB. This explicit unit-test resource observation is not
    # a server resource qualification; real server orchestration is uninjected.
    code='''import json,sys
from egoemg_repro import emg2pose_direct_experiment as m
m.memory.available_memory_bytes=lambda:2*1024**4
fault=sys.argv[2]
if fault=='initial_pressure':m.memory.available_memory_bytes=lambda:0
if fault=='owned_failure':
 original=m._command
 def command(request,name):
  return [sys.executable,'-c','raise RuntimeError("intentional owned test failure")'] if name=='fit' else original(request,name)
 m._command=command
kwargs=json.loads(sys.argv[1])
if fault=='after_admission':
 def progress(record):
  if record['name']=='admission':raise RuntimeError('intentional parent test failure')
 kwargs['on_progress']=progress
v=m.run_experiment(**kwargs)
assert 'torch' not in sys.modules
print('DRIVER_FINISHED='+json.dumps(v,sort_keys=True))
'''
    r=subprocess.run([sys.executable,'-c',code,json.dumps(kwargs),fault or ''],env=dict(os.environ,
        CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),capture_output=True,text=True,timeout=600)
    output.parent.mkdir(parents=True,exist_ok=True)
    (output.parent/(output.name+'.stdout')).write_text(r.stdout)
    (output.parent/(output.name+'.stderr')).write_text(r.stderr)
    return r

@pytest.fixture(scope='module')
def finished(memory,tmp_path_factory):
    cached=[]
    def load():
        if not cached:
            output=tmp_path_factory.mktemp('driver-real-cpu')/'experiment';r=invoke(memory,output)
            cached.append((output,r))
        output,r=cached[0]
        assert r.returncode==0,r.stdout[-2000:]+r.stderr[-6000:]
        return output
    return load

def test_import_and_parent_phase_plan_do_not_import_torch():
    m=api()
    r=subprocess.run([sys.executable,'-c',"import sys;from egoemg_repro import emg2pose_direct_experiment as m;assert 'torch' not in sys.modules;assert len(m.stage_plan('small','full'))==14"],capture_output=True,text=True)
    assert r.returncode==0,r.stderr

@pytest.mark.parametrize('size',['small','middle','large'])
def test_model_parametric_original_plan_has_fixed_fourteen_stages(size):
    m=api();plan=m.stage_plan(size,'full');names=[p['name'] for p in plan]
    assert names==['admission','fit',*[mode+'-'+name for mode in ('official_literal','metadata_indices')
        for name in ('val-native','val-replay','profile','test-native','test-replay')],'export','verify']
    assert sum(p['cuda'] for p in plan)==9
    assert not any(p['cuda'] for p in m.stage_plan(size,'cpu_engineering'))
    for scope in ('full','cuda_engineering'):assert m.stage_plan(size,scope)==plan

def test_actual_owned_cpu_fit_profiles_both_tests_export_verify_and_fresh_reopen(finished):
    m=api();root=finished();value=m.read_experiment(root,require_full=False)
    assert value['format']=='emg2pose_direct_experiment_driver_v1' and value['scope']=='cpu_engineering'
    assert value['scientific_complete'] is value['full_paper_complete'] is False
    assert value['parent_torch_unimported'] is True and len(value['stages'])==14
    assert value['report_completion_sha256']==sha(root/'report/completion.json')
    assert json.loads((root/'fit/fit-result.json').read_text())['optimizer_executions']==32
    for mode in ('official_literal','metadata_indices'):
        for role in ('val','test'):
            a=json.loads((root/mode/(role+'-native/pass.json')).read_text());b=json.loads((root/mode/(role+'-replay/pass.json')).read_text())
            assert a['typed_input_output_digest']==b['typed_input_output_digest'] and a['statistics_sha256']==b['statistics_sha256']
    with pytest.raises(ValueError,match='engineering|full'):m.read_experiment(root)
    with pytest.raises(ValueError,match='SHA|sha|completion'):
        m.read_experiment(root,expected_completion_sha256='0'*64,require_full=False)

@pytest.mark.parametrize('change',[{'scope':'unknown'},{'size':'huge'},{'gpu_uuid':'GPU-01234567'},
    {'deadline_seconds':0},{'deadline_seconds':True},{'qualification':'missing'},
    {'evaluation_qualification':'missing'},{'controller_qualification':'missing'}])
def test_invalid_policy_or_half_frozen_qualification_refuses_before_output(memory,tmp_path,change):
    api();output=tmp_path/'absent';r=invoke(memory,output,**change)
    assert r.returncode!=0 and not output.exists()
    assert 'ValueError' in r.stderr

def test_complete_engineering_source_cannot_enter_full_before_any_gpu_or_output(memory,tmp_path):
    api();source=str(memory.source_root)
    r=invoke(memory,tmp_path/'absent',scope='full',gpu_uuid='GPU-00000000-0000-0000-0000-000000000000',
        qualification=source,expected_qualification_sha256='0'*64,evaluation_qualification=source,
        expected_evaluation_qualification_sha256='0'*64,controller_qualification=source,expected_controller_qualification_sha256='0'*64)
    assert r.returncode!=0 and not (tmp_path/'absent').exists()
    assert '25253' in r.stderr or 'full source' in r.stderr

@pytest.mark.parametrize('fault',['initial_pressure','after_admission','owned_failure'])
def test_real_pressure_parent_or_owned_failure_retains_no_completion_and_no_consumers(memory,tmp_path,fault):
    m=api();output=tmp_path/fault;r=invoke(memory,output,fault=fault)
    assert r.returncode!=0 and not (output/'driver-completion.json').exists()
    if fault=='initial_pressure':assert not output.exists();return
    status=json.loads((output/'driver-status.json').read_text());assert status['status']=='failed'
    from egoemg_repro.emg2pose_memory_process import same_process,process_identity
    for path in (output/'stages').glob('*/stage-result.json'):
        v=json.loads(path.read_text());assert v['all_owned_consumers_exited']
        for kind in ('worker','guardian'):
            p=v[kind];assert not same_process(p) or process_identity(p['pid'])['state']=='Z'
    if fault=='owned_failure':
        v=json.loads((output/'stages/fit/stage-result.json').read_text());assert v['worker_exit_code']!=0
    with pytest.raises((ValueError,FileNotFoundError)):m.read_experiment(output,require_full=False)

@pytest.mark.parametrize('damage',['request','owned_exit','public_report','resealed_completion'])
def test_driver_reader_reopens_every_actual_stage_and_derived_report(finished,damage):
    m=api();root=finished()
    path={'request':root/'request.json','owned_exit':root/'stages/fit/stage-result.json',
        'public_report':root/'report/metrics.json','resealed_completion':root/'driver-completion.json'}[damage]
    raw=path.read_bytes();value=json.loads(raw)
    if damage=='owned_exit':value['worker_exit_code']=1
    elif damage=='request':value['size']='large'
    else:value['scientific_complete']=True
    if damage=='resealed_completion':value=m._seal({k:v for k,v in value.items() if k!='sha256'})
    try:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):m.read_experiment(root,require_full=False)
    finally:path.write_bytes(raw)

def test_existing_output_and_inherited_archive_rights_cannot_be_reused(memory,finished,tmp_path):
    api();r=invoke(memory,finished());assert r.returncode!=0 and 'FileExistsError' in r.stderr
    code="import os;os.environ['EGOEMG_STAGE_FD_ARCHIVE']='123';from egoemg_repro import emg2pose_direct_experiment as m;m.main(['run'])"
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert r.returncode!=0 and 'inherited archive' in r.stderr
