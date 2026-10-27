"""Actual original ordinary-data recovery; inherited TAR rights never enter."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning');pytest.importorskip('hydra');pytest.importorskip('tensorboard')
from test_emg2pose_direct_runtime import pipeline_corpus,native_corpus,UPSTREAM
from egoemg_repro import emg2pose_data as storage


def api():return importlib.import_module('egoemg_repro.emg2pose_direct_recovery')

@pytest.fixture(autouse=True)
def unit_environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.delenv('EGOEMG_STAGE_FD_ARCHIVE',raising=False)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=64*1024**3))

@pytest.fixture(scope='module')
def prefixes():return {}


def prefix(pipeline_corpus,tmp_path_factory,prefixes,size='small'):
    recovery=api()
    if size not in prefixes:
        from egoemg_repro.emg2pose_direct_source import open_source
        out=tmp_path_factory.mktemp('direct-recovery-'+size)/'run'
        with open_source(pipeline_corpus.root,expected_completion_sha256=storage._sha256(pipeline_corpus.root/'completion.json'),
                upstream=UPSTREAM,metadata_csv=pipeline_corpus.csv) as source:
            result=recovery.run_recovery_phase(UPSTREAM,source,pipeline_corpus.csv,size,out,phase='prefix',device='cpu')
            source.assert_current();assert not hasattr(source,'fileno')
        prefixes[size]=(out,result)
    return prefixes[size]


def command(c,out,phase,size='small',expected=None):
    cmd=[sys.executable,'-m','egoemg_repro.emg2pose_direct_recovery','--upstream',str(UPSTREAM),
        '--source-root',str(c.root),'--expected-source-completion-sha256',storage._sha256(c.root/'completion.json'),
        '--metadata-csv',str(c.csv),'--size',size,'--output',str(out),'--phase',phase,'--device','cpu']
    if expected is not None:cmd+=['--expected-prefix-sha256',expected]
    return cmd


@pytest.mark.parametrize('size',['small','middle','large'])
def test_actual_native_prefix_and_fresh_process_restore_all_six_states(pipeline_corpus,tmp_path_factory,prefixes,size):
    out,result=prefix(pipeline_corpus,tmp_path_factory,prefixes,size);frozen=storage._sha256(out/'prefix.json')
    assert result['format']==api().FORMAT and result['format']!='emg2pose_native_memory_recovery_v1'
    assert result['actual_optimizer_updates']==2 and result['global_step']==2 and len(result['validations'])==2
    assert all(v['samples']==1 for v in result['validations'])
    assert result['profile']['scientific_batch_size']==320 and result['profile']['scientific_workers']==12
    checkpoint=torch.load(out/'prefix.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['lr_schedulers'][0]['T_max']==250 and checkpoint['global_step']==2
    assert not any(k.startswith('EarlyStopping') for k in checkpoint['callbacks'])
    states=[v for k,v in checkpoint['callbacks'].items() if k.startswith('ModelCheckpoint{')]
    assert len(states)==1 and len(states[0]['best_k_models'])==2
    r=subprocess.run(command(pipeline_corpus,out,'restore',size,frozen),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),capture_output=True,text=True,timeout=240)
    assert r.returncode==0,r.stdout[-3500:]+r.stderr[-6500:]
    restored=json.loads((out/'restore.json').read_text())
    assert restored['process']!=result['process'] and restored['restore_matches']==dict.fromkeys(('model','optimizer','scheduler','callbacks','loops','rng'),True)
    assert restored['fresh_validation']==result['validations'][-1] and restored['actual_optimizer_updates']==1 and restored['global_step']==3
    assert restored['parameters_changed'] and not restored['scientific_training_complete'] and not restored['uninterrupted_trajectory_qualified']
    assert storage._sha256(out/'prefix.json')==frozen and storage._sha256(out/'prefix.ckpt')==result['checkpoint_sha256']


@pytest.mark.parametrize('fault',['receipt','checkpoint','identity','missing_sha','same_process','old_family'])
def test_bad_or_unfrozen_recovery_never_starts_training(pipeline_corpus,tmp_path_factory,prefixes,tmp_path,fault):
    out,_=prefix(pipeline_corpus,tmp_path_factory,prefixes);copy=tmp_path/'owned';copy.mkdir()
    for name in ('prefix.json','prefix.ckpt'):shutil.copy2(out/name,copy/name)
    p=copy/'prefix.json';value=json.loads(p.read_text())
    if fault=='identity':value['runtime_identity']['size']='large';p.write_text(json.dumps(value))
    elif fault=='old_family':value['format']='emg2pose_native_memory_recovery_v1';p.write_text(json.dumps(value))
    expected=storage._sha256(p)
    if fault=='receipt':p.write_text('{}')
    elif fault=='checkpoint':
        with (copy/'prefix.ckpt').open('ab') as stream:stream.write(b'actual change')
    elif fault=='missing_sha':expected=None
    from egoemg_repro.emg2pose_direct_source import open_source
    with open_source(pipeline_corpus.root,expected_completion_sha256=storage._sha256(pipeline_corpus.root/'completion.json'),upstream=UPSTREAM,metadata_csv=pipeline_corpus.csv) as source:
        with pytest.raises(ValueError):api().run_recovery_phase(UPSTREAM,source,pipeline_corpus.csv,'small',copy,phase='restore',device='cpu',expected_prefix_sha256=expected)
        source.assert_current()
    assert not (copy/'restore.json').exists() and not (copy/'restore-started.json').exists()


def test_old_global_and_core_native_recovery_algorithms_are_unchanged():
    from egoemg_repro import emg2pose_memory_recovery as old,emg2pose_memory_runtime as old_runtime
    new=api();assert new.run_recovery_phase.__code__ is old.run_recovery_phase.__code__
    assert old.run_recovery_phase.__globals__['runtime'] is old_runtime and new.run_recovery_phase.__globals__ is not old.run_recovery_phase.__globals__
    assert new.run_recovery_phase.__globals__['VerifiedSource'] is not old.run_recovery_phase.__globals__['VerifiedSource']


def test_existing_prefix_and_dictionary_source_are_refused(pipeline_corpus,tmp_path_factory,prefixes):
    out,_=prefix(pipeline_corpus,tmp_path_factory,prefixes);before=storage._sha256(out/'prefix.ckpt')
    from egoemg_repro.emg2pose_direct_source import open_source
    with open_source(pipeline_corpus.root,expected_completion_sha256=storage._sha256(pipeline_corpus.root/'completion.json'),upstream=UPSTREAM,metadata_csv=pipeline_corpus.csv) as source:
        with pytest.raises(FileExistsError):api().run_recovery_phase(UPSTREAM,source,pipeline_corpus.csv,'small',out,phase='prefix')
    with pytest.raises(ValueError):api().run_recovery_phase(UPSTREAM,{},pipeline_corpus.csv,'small',out.with_name('fake'),phase='prefix')
    assert storage._sha256(out/'prefix.ckpt')==before


def test_checkpoint_writes_and_deletes_stay_owned_and_prefix_is_frozen(tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'checkpoints').mkdir();io=api().checkpoint_io(root)
    payload={'state_dict':{'x':torch.ones(4)},'global_step':2}
    io.save_checkpoint(payload,root/'checkpoints/a.ckpt');io.save_checkpoint(payload,root/'checkpoints/a.ckpt');io.remove_checkpoint(root/'checkpoints/a.ckpt')
    io.save_checkpoint(payload,root/'prefix.ckpt');expected=storage._sha256(root/'prefix.ckpt')
    with pytest.raises(FileExistsError):io.save_checkpoint(payload,root/'prefix.ckpt')
    with pytest.raises(ValueError):io.remove_checkpoint(root/'prefix.ckpt')
    with pytest.raises(ValueError):io.save_checkpoint(payload,tmp_path/'foreign.ckpt')
    assert torch.equal(api().checkpoint_io(root,expected_checkpoint_sha256=expected).load_checkpoint(root/'prefix.ckpt')['state_dict']['x'],torch.ones(4))


def test_ordinary_cli_refuses_archive_protocol_without_closing_caller_right(monkeypatch):
    fd=os.open(os.devnull,os.O_RDONLY)
    try:
        monkeypatch.setenv('EGOEMG_STAGE_FD_ARCHIVE',str(fd))
        with pytest.raises(ValueError):api().main([])
        os.fstat(fd)
    finally:os.close(fd)


def test_failed_final_receipt_keeps_failure_and_no_success_marker(pipeline_corpus,tmp_path,monkeypatch):
    recovery=api();out=tmp_path/'failed';original=recovery.checks._write_json;failed=[]
    def fail(path,value):
        if Path(path)==out/'status.json' and not failed:failed.append(True);raise OSError('actual final marker failure')
        return original(path,value)
    monkeypatch.setattr(recovery.checks,'_write_json',fail)
    from egoemg_repro.emg2pose_direct_source import open_source
    with open_source(pipeline_corpus.root,expected_completion_sha256=storage._sha256(pipeline_corpus.root/'completion.json'),upstream=UPSTREAM,metadata_csv=pipeline_corpus.csv) as source:
        with pytest.raises(OSError):recovery.run_recovery_phase(UPSTREAM,source,pipeline_corpus.csv,'small',out,phase='prefix')
        source.assert_current()
    assert not (out/'prefix.json').exists() and json.loads((out/'status.json').read_text())['status']=='failed'


def test_changed_model_prediction_is_refused_on_full_fresh_validation_and_not_retried(pipeline_corpus,tmp_path_factory,prefixes,tmp_path):
    out,_=prefix(pipeline_corpus,tmp_path_factory,prefixes);copy=tmp_path/'changed';copy.mkdir();(copy/'checkpoints').mkdir()
    value=json.loads((out/'prefix.json').read_text());checkpoint=torch.load(out/'prefix.ckpt',map_location='cpu',weights_only=False)
    checkpoint['state_dict']['model.head.net.3.bias'].add_(.25);torch.save(checkpoint,copy/'prefix.ckpt')
    value['checkpoint_sha256']=storage._sha256(copy/'prefix.ckpt');(copy/'prefix.json').write_text(json.dumps(value));expected=storage._sha256(copy/'prefix.json')
    r=subprocess.run(command(pipeline_corpus,copy,'restore',expected=expected),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=240)
    assert r.returncode!=0 and 'fresh complete native validation differs' in (copy/'status.json').read_text()
    assert not (copy/'restore.json').exists()
    retry=subprocess.run(command(pipeline_corpus,copy,'restore',expected=expected),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=240)
    assert retry.returncode!=0 and 'never automatically retried' in retry.stderr
