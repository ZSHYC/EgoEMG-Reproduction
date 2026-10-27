"""Original selected-best inference, actual fresh replay and pretest admission."""
from __future__ import annotations
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning');pytest.importorskip('hydra');pytest.importorskip('tensorboard')
from test_emg2pose_runtime import corpus,UPSTREAM
from test_emg2pose_direct_runtime import pipeline_corpus
from egoemg_repro import emg2pose_data as storage
from egoemg_repro.emg2pose_direct_source import open_source


def api():return importlib.import_module('egoemg_repro.emg2pose_direct_evaluate')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def memory(corpus,tmp_path_factory):
    root=tmp_path_factory.mktemp('eval-distinct-recordings')
    with corpus.metadata_csv.open() as f:rows=list(csv.DictReader(f))
    for i,row in enumerate(rows):
        row['session']='separate-recording-'+str(i)
        path=root/(row['filename']+'.hdf5');shutil.copy2(corpus.root/path.name,path)
        with h5py.File(path,'r+') as f:f['emg2pose'].attrs['session']=row['session']
    metadata=root/'metadata.csv'
    with metadata.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    prepared=pipeline_corpus.__wrapped__(SimpleNamespace(root=root,metadata_csv=metadata),tmp_path_factory)
    return SimpleNamespace(source_root=prepared.root,index=prepared.root/'data',metadata=metadata,receipt={'completion_sha256':sha(prepared.root/'completion.json')},root=root)


def source(memory):
    return open_source(memory.source_root,expected_completion_sha256=memory.receipt['completion_sha256'],
                       upstream=UPSTREAM,metadata_csv=memory.metadata)


@pytest.fixture(scope='module')
def trained(memory,tmp_path_factory):
    from egoemg_repro.emg2pose_direct_train import fit_source,read_fit
    directory=tmp_path_factory.mktemp('eval-genuine-fit')/'fit'
    with source(memory) as handle:
        result=fit_source(UPSTREAM,handle,memory.metadata,'small',directory,scope='cpu_engineering')
    assert result['completed_epochs']==2 and result['optimizer_executions']==32
    assert read_fit(directory,require_full=False)['selected_best']['validated'] is True
    return directory


def run(memory,trained,output,**kwargs):
    with source(memory) as handle:
        return api().run_evaluation_pass(UPSTREAM,handle,memory.metadata,'small',trained,output,
                                        scope='cpu_engineering',**kwargs)


def fresh(memory,trained,output,**kwargs):
    command=[sys.executable,'-m','egoemg_repro.emg2pose_direct_evaluate','pass',
        '--upstream',str(UPSTREAM),'--source-root',str(memory.source_root),
        '--expected-source-completion-sha256',memory.receipt['completion_sha256'],
        '--metadata-csv',str(memory.metadata),'--size','small','--fit',str(trained),
        '--output',str(output),'--scope','cpu_engineering']
    for key,value in kwargs.items():command.extend(['--'+key.replace('_','-'),str(value)])
    # Observe sealed identities without changing the real CLI or its decisions.
    # Report only field paths, never values, for hosted-only subprocess drift.
    diagnostic='''import json,sys
from pathlib import Path
from egoemg_repro import emg2pose_direct_evaluate as evaluation
original=evaluation._seal
def paths(a,b,p=''):
 if type(a) is dict and type(b) is dict:
  return [v for k in sorted(set(a)|set(b)) for v in paths(a.get(k),b.get(k),p+'.'+k)]
 if type(a) is list and type(b) is list and len(a)==len(b):
  return [v for i,(x,y) in enumerate(zip(a,b)) for v in paths(x,y,p+'['+str(i)+']')]
 return [] if a==b else [p]
def observe(value):
 result=original(value)
 if value.get('document')=='identity' and '--reference' in sys.argv:
  reference=Path(sys.argv[sys.argv.index('--reference')+1])/'identity.json'
  differences=paths(json.loads(reference.read_text()),json.loads(json.dumps(result,allow_nan=False)))
  if differences:print('REPLAY_IDENTITY_DIFFERING_FIELDS='+json.dumps(differences[:100]),file=sys.stderr)
 return result
evaluation._seal=observe
evaluation.main()
'''
    command=[sys.executable,'-c',diagnostic,*command[3:]]
    result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
        capture_output=True,text=True,timeout=180)
    output.parent.mkdir(parents=True,exist_ok=True)
    (output.parent/(output.name+'.stdout')).write_text(result.stdout)
    (output.parent/(output.name+'.stderr')).write_text(result.stderr)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-6000:]
    return api().read_evaluation_pass(output,require_full=False)


def make_passes(memory,trained,tmp_path_factory):
    root=tmp_path_factory.mktemp('selected-evaluation-passes')
    m=api()
    for selection in ('official_literal','metadata_indices'):
        native=root/(selection+'-val-native');replay=root/(selection+'-val-replay')
        fresh(memory,trained,native,role='val',kind='native',selection=selection)
        fresh(memory,trained,replay,role='val',kind='replay',selection=selection,reference=native,
              expected_reference_sha256=sha(native/'pass.json'))
        m.freeze_evaluation_profile(native,replay,root/(selection+'-profile'),
            expected_native_sha256=sha(native/'pass.json'),expected_replay_sha256=sha(replay/'pass.json'),require_full=False)
    return root


@pytest.fixture(scope='module')
def passes(memory,trained,tmp_path_factory):
    cached=[]
    def load():
        if not cached:cached.append(make_passes(memory,trained,tmp_path_factory))
        return cached[0]
    return load


def test_actual_selected_best_whole_val_native_and_fresh_replay_agree(passes):
    m=api()
    for mode in ('official_literal','metadata_indices'):
        native=m.read_evaluation_pass(passes()/(mode+'-val-native'),require_full=False)
        replay=m.read_evaluation_pass(passes()/(mode+'-val-replay'),require_full=False)
        assert native['samples_by_condition']=={'user':1}
        assert native['batch_count']==1 and native['stage_complete'] is True
        assert native['model_before_sha256']==native['model_after_sha256']==native['fit_binding']['model_sha256']
        assert native['typed_input_output_digest']==replay['typed_input_output_digest']
        assert native['statistics_sha256']==replay['statistics_sha256']
        assert native['process']!=replay['process']
        assert native['test_performed'] is False and native['scientific_complete'] is False
        assert not (passes()/(mode+'-val-native')/'predictions.pt').exists()
        profile=m.read_evaluation_profile(passes()/(mode+'-profile'),sha(passes()/(mode+'-profile/profile.json')),require_full=False)
        assert profile['selection']==mode and profile['qualified_for_full_test'] is False


def test_profile_and_actual_complete_test_then_fresh_test_replay(memory,trained,passes,tmp_path):
    for mode in ('official_literal','metadata_indices'):
        profile=passes()/(mode+'-profile');native=tmp_path/(mode+'-test-native');replay=tmp_path/(mode+'-test-replay')
        kwargs=dict(role='test',selection=mode,profile=profile,expected_profile_sha256=sha(profile/'profile.json'))
        result=fresh(memory,trained,native,kind='native',**kwargs)
        assert result['samples_by_condition']=={'user_stage':1} and result['test_performed'] is True
        actual=fresh(memory,trained,replay,kind='replay',reference=native,
            expected_reference_sha256=sha(native/'pass.json'),**kwargs)
        assert actual['typed_input_output_digest']==result['typed_input_output_digest']
        assert actual['statistics_sha256']==result['statistics_sha256']


def test_missing_test_profile_refuses_before_model_data_rng_output(memory,trained,tmp_path,monkeypatch):
    from egoemg_repro import emg2pose_memory_runtime as runtime
    before=torch.get_rng_state().clone();out=tmp_path/'absent'
    def forbidden(*_a,**_k):raise AssertionError('no model before pretest profile')
    monkeypatch.setattr(runtime,'reload_runtime',forbidden)
    with pytest.raises(ValueError,match='profile|calibration'):
        run(memory,trained,out,role='test',kind='native')
    assert not out.exists() and torch.equal(before,torch.get_rng_state())


def test_engineering_source_fit_and_profile_never_admit_full(memory,trained,passes,tmp_path):
    m=api();profile=passes()/'official_literal-profile'
    with pytest.raises(ValueError):m.read_evaluation_pass(passes()/'official_literal-val-native')
    with pytest.raises(ValueError):m.read_evaluation_profile(profile,sha(profile/'profile.json'))
    with source(memory) as handle:
        with pytest.raises(ValueError):m.run_evaluation_pass(UPSTREAM,handle,memory.metadata,'small',trained,
            tmp_path/'never',role='val',kind='native')
    assert not (tmp_path/'never').exists()


def test_same_process_replay_wrong_reference_and_cross_mode_profile_are_refused(memory,trained,passes,tmp_path):
    m=api();native=passes()/'official_literal-val-native'
    same_process=tmp_path/'same-process-native'
    run(memory,trained,same_process,role='val',kind='native')
    with pytest.raises(ValueError,match='fresh|process'):
        run(memory,trained,tmp_path/'same',role='val',kind='replay',reference=same_process,
            expected_reference_sha256=sha(same_process/'pass.json'))
    with pytest.raises(ValueError,match='SHA|sha|reference'):
        run(memory,trained,tmp_path/'wrong',role='val',kind='replay',reference=native,expected_reference_sha256='0'*64)
    p=passes()/'metadata_indices-profile'
    with pytest.raises(ValueError,match='mode|selection|profile'):
        run(memory,trained,tmp_path/'cross',role='test',kind='native',selection='official_literal',
            profile=p,expected_profile_sha256=sha(p/'profile.json'))
    assert not any((tmp_path/n).exists() for n in ('same','wrong','cross'))


@pytest.mark.parametrize('damage',['journal','statistics','native_metrics','model_claim','missing_batch','resealed_statistics'])
def test_reader_reopens_raw_sidecars_and_rejects_corruption_even_with_resealed_claims(passes,tmp_path,damage):
    m=api();root=passes()/'official_literal-val-native'
    names=['pass.json','batches.jsonl','statistics.json','native-metrics.json']
    backup={n:(root/n).read_bytes() for n in names}
    try:
        packet=json.loads(backup['pass.json'])
        if damage=='journal':
            with (root/'batches.jsonl').open('ab') as f:f.write(b'changed\n')
        elif damage=='missing_batch':(root/'batches.jsonl').write_bytes(b'')
        elif damage in ('statistics','resealed_statistics'):
            value=json.loads(backup['statistics.json']);value['samples']+=1
            (root/'statistics.json').write_text(json.dumps(value))
            if damage=='resealed_statistics':
                packet['statistics_sha256']=sha(root/'statistics.json')
                packet['sha256']=m._hash({k:v for k,v in packet.items() if k!='sha256'})
                (root/'pass.json').write_text(json.dumps(packet))
        elif damage=='native_metrics':(root/'native-metrics.json').write_text('{}')
        else:
            packet['model_before_sha256']='0'*64
            packet['sha256']=m._hash({k:v for k,v in packet.items() if k!='sha256'})
            (root/'pass.json').write_text(json.dumps(packet))
        with pytest.raises((ValueError,FileNotFoundError)):m.read_evaluation_pass(root,require_full=False)
    finally:
        for name,value in backup.items():(root/name).write_bytes(value)


def test_failed_pass_has_no_success_capsule_and_does_not_change_fit(memory,trained,tmp_path,monkeypatch):
    from egoemg_repro.emg2pose_memory_statistics import Emg2PoseStatistics
    before={str(p.relative_to(trained)):sha(p) for p in trained.rglob('*') if p.is_file()}
    def fail(*_a,**_k):raise RuntimeError('injected comparison failure')
    monkeypatch.setattr(Emg2PoseStatistics,'check_native',fail)
    output=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='injected comparison failure'):
        run(memory,trained,output,role='val',kind='native')
    assert not (output/'pass.json').exists()
    assert {str(p.relative_to(trained)):sha(p) for p in trained.rglob('*') if p.is_file()}==before


def test_existing_outputs_are_never_overwritten_and_unowned_paths_are_refused(memory,trained,passes):
    out=passes()/'official_literal-val-native';before={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):run(memory,trained,out,role='val',kind='native')
    assert {str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}==before
    with pytest.raises(ValueError):run(memory,trained,trained/'nested-eval',role='val',kind='native')
    assert not (trained/'nested-eval').exists()


@pytest.mark.parametrize('samples,zero_valid,valid_times,sums',[
    (2,0,1,[0.0]*20),
    (1,1,0,[1.0]+[0.0]*19),
])
def test_impossible_raw_sample_counts_or_zero_mask_error_sums_are_refused(samples,zero_valid,valid_times,sums):
    from egoemg_repro.emg2pose_memory_statistics import Emg2PoseStatistics
    values={'user':dict(samples=samples,zero_valid=zero_valid,valid_times=valid_times,sums=sums)}
    with pytest.raises(ValueError,match='statistics'):
        api()._merge(Emg2PoseStatistics(),'user',values,{'user':samples})


def test_gpu_prelaunch_margin_is_distinct_from_running_allocation_margin(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    free=[76*1024**3]
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda _device:(free[0],80*1024**3))
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _device:'allocation-fixture')
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda _device:(8,0))
    module=api();before=module._environment('cuda_engineering')
    free[0]=16*1024**3
    with pytest.raises(RuntimeError,match='64GiB'):module._environment('cuda_engineering')
    assert module._environment('cuda_engineering',prelaunch=False)==before
    free[0]=7*1024**3
    with pytest.raises(RuntimeError,match='8GiB'):
        module._environment('cuda_engineering',prelaunch=False)


@pytest.mark.parametrize('damage',['pipeline','identity','kernel'])
def test_direct_pass_binds_actual_ordinary_source_and_refuses_kernel_identity(passes,tmp_path,damage):
    origin=passes()/'metadata_indices-val-native';owned=tmp_path/'changed';shutil.copytree(origin,owned)
    p=owned/'pass.json';v=json.loads(p.read_text())
    if damage=='pipeline':v['source_file_binding']['source_pipeline_completion_sha256']='0'*64
    elif damage=='identity':v['source_file_binding']['data_identity_sha256']='0'*64
    else:v['source_kernel_binding']={'device':0,'inode':0}
    v['sha256']=api()._hash({k:w for k,w in v.items() if k!='sha256'});p.write_text(json.dumps(v))
    with pytest.raises(ValueError):api().read_evaluation_pass(owned,require_full=False)


def test_direct_cli_refuses_archive_right_without_closing_caller(monkeypatch):
    fd=os.open(os.devnull,os.O_RDONLY)
    try:
        monkeypatch.setenv('EGOEMG_STAGE_FD_ARCHIVE',str(fd))
        with pytest.raises(ValueError):api().main([])
        os.fstat(fd)
    finally:os.close(fd)


def test_qualified_journal_pair_algorithms_and_old_globals_unchanged():
    from egoemg_repro import emg2pose_memory_evaluate as old,emg2pose_memory_runtime as old_runtime
    new=api();assert new._journal_statistics.__code__ is old._journal_statistics.__code__ and new._pair.__code__ is old._pair.__code__
    assert old.read_evaluation_pass.__globals__['runtime'] is old_runtime
    assert new.read_evaluation_pass.__globals__ is not old.read_evaluation_pass.__globals__
    assert new.FORMAT!=old.FORMAT and new.PROFILE!=old.PROFILE


def test_actual_empty_descriptor_owned_cli_pass_and_nonempty_rights_refusal(memory,trained,tmp_path):
    from egoemg_repro.emg2pose_memory_process import run_owned_stage
    output=tmp_path/'native';directory=tmp_path/'owned'
    command=[sys.executable,'-m','egoemg_repro.emg2pose_direct_evaluate','pass','--upstream',str(UPSTREAM),
        '--source-root',str(memory.source_root),'--expected-source-completion-sha256',memory.receipt['completion_sha256'],
        '--metadata-csv',str(memory.metadata),'--size','small','--fit',str(trained),'--output',str(output),
        '--scope','cpu_engineering','--selection','metadata_indices','--role','val','--kind','native']
    stage=run_owned_stage(command,directory,descriptors={},guard=lambda:None,environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    assert stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']
    packet=api().read_evaluation_pass(output,require_full=False)
    assert api()._owned(directory,packet)['sha256']==sha(directory/'stage-result.json')
    path=directory/'stage-result.json';value=json.loads(path.read_text());value['descriptor_bindings']={'ARCHIVE':{'fake':True}};path.write_text(json.dumps(value))
    with pytest.raises(ValueError):api()._owned(directory,packet)
