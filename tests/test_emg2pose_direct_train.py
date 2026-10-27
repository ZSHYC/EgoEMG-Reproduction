"""Native independent train/val epochs, selected best, and exact restore boundary."""
import copy
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning')
pytest.importorskip('hydra')
pytest.importorskip('tensorboard')
from test_emg2pose_direct_runtime import pipeline_corpus,native_corpus,UPSTREAM
from egoemg_repro import emg2pose_data as storage
from egoemg_repro.emg2pose_direct_source import open_source


def api():return importlib.import_module('egoemg_repro.emg2pose_direct_train')

@pytest.fixture(scope='module')
def memory(pipeline_corpus):return pipeline_corpus

@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.delenv('EGOEMG_STAGE_FD_ARCHIVE',raising=False)


def call(memory,output,**kwargs):
    with open_source(memory.root,
                     expected_completion_sha256=storage._sha256(memory.root/'completion.json'),
                     upstream=UPSTREAM,metadata_csv=memory.csv) as source:
        return api().fit_source(UPSTREAM,source,memory.csv,'small',output,**kwargs)


def test_full_policy_has_original_250_budget_no_early_stop_and_no_engineering_cap():
    p=api().fit_policy('full')
    assert p['max_epochs']==250 and p['batch_size']==320 and p['workers']==12
    assert p['precision']=='bf16-mixed' and p['scheduler_epochs']==250
    assert p['gradient_clip_val']==1 and p['save_top_k']==3 and p['save_last'] is True
    assert p['monitor']=='val_mae' and p['mode']=='min' and p['early_stopping'] is False
    assert p['limit_train_batches']==p['limit_val_batches']==1.0
    assert p['num_sanity_val_steps']==2 and p['full_paper_complete'] is False


@pytest.mark.parametrize('scope',[None,True,'mini','smoke','full_cpu'])
def test_unknown_or_implicit_fit_scope_fails(scope):
    with pytest.raises(ValueError):api().fit_policy(scope)


def test_tiny_source_cannot_start_full_fit_or_touch_rng_output(memory,tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='complete.*25253|full.*source'):
        call(memory,tmp_path/'never-created',scope='full')
    assert torch.equal(torch.get_rng_state(),before)
    assert not (tmp_path/'never-created').exists()


def make_runs(memory,tmp_path_factory):
    root=tmp_path_factory.mktemp('native-direct-fit')
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('CUDA_VISIBLE_DEVICES','')
        control=call(memory,root/'control',scope='cpu_engineering')
        prefix=call(memory,root/'resumed',scope='cpu_engineering',stop_after_epoch=1)
    last=storage._sha256(root/'resumed/checkpoints/last.ckpt')
    command=[sys.executable,'-m','egoemg_repro.emg2pose_direct_train',
        '--upstream',str(UPSTREAM),'--source-root',str(memory.root),
        '--expected-source-completion-sha256',storage._sha256(memory.root/'completion.json'),
        '--metadata-csv',str(memory.csv),'--size','small',
        '--output',str(root/'resumed'),'--scope','cpu_engineering','--resume',
        '--expected-last-sha256',last]
    result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
        capture_output=True,text=True,timeout=240)
    (root/'resume.stdout').write_text(result.stdout);(root/'resume.stderr').write_text(result.stderr)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr[-7000:]
    return root,control,prefix,last


@pytest.fixture(scope='module')
def runs(memory,tmp_path_factory):
    cached=[]
    def load():
        if not cached:cached.append(make_runs(memory,tmp_path_factory))
        return cached[0]
    return load


def test_actual_original_train_val_epochs_selected_best_and_fresh_native_restore(runs):
    root,control,prefix,last=runs();m=api()
    assert prefix['completed_epochs']==1 and prefix['stage_complete'] is False
    for name in ('control','resumed'):
        result=m.read_fit(root/name,require_full=False)
        assert result['completed_epochs']==2 and result['optimizer_executions']==32
        assert result['train_samples_per_epoch']==[16,16] and result['val_samples_per_epoch']==[1,1]
        assert result['stage_complete'] is True and result['full_training_complete'] is False
        assert result['selected_best']['validated'] is True
        assert result['scientific_complete'] is False and result['test_performed'] is False
        checkpoint=torch.load(root/name/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert checkpoint['lr_schedulers'][0]['T_max']==250 and checkpoint['global_step']==32
        assert not any(k.startswith('EarlyStopping') for k in checkpoint['callbacks'])
        assert checkpoint['emg2pose_fit']['completed_epochs']==2
    restored=m.read_fit(root/'resumed',require_full=False)
    assert restored['resume_matches']==dict.fromkeys(('model','optimizer','scheduler','callbacks','loops','rng'),True)
    assert restored['resume_checkpoint']['sha256']==last
    assert storage._sha256(root/'resumed'/restored['resume_checkpoint']['file'])==last
    identity=json.loads((root/'resumed/identity.json').read_text())
    assert identity['train_unique_windows']==identity['val_samples']==1
    assert identity['train_samples']==16 and identity['train_batches']==16
    assert identity['profile']['fixture']=='independent original train and val roles'
    assert not (root/'resumed/completion.json').exists()


def test_engineering_fit_never_satisfies_full_reader_or_cuda_qualification(runs):
    root,_,_,_=runs()
    with pytest.raises(ValueError,match='full|engineering'):api().read_fit(root/'resumed')
    with pytest.raises(ValueError,match='CUDA|cuda|engineering'):
        api().read_qualification(root/'resumed',storage._sha256(root/'resumed/fit-result.json'))


@pytest.mark.parametrize('damage',['last','best','journal','claim'])
def test_reader_rejects_changed_native_state_coverage_and_full_claims(runs,tmp_path,damage):
    root,_,_,_=runs();owned=root/'control'
    selected=json.loads((owned/'fit-result.json').read_text())['selected_best']['file']
    path={'last':owned/'checkpoints/last.ckpt','best':owned/'checkpoints'/selected,
          'journal':owned/'epochs.jsonl','claim':owned/'fit-result.json'}[damage]
    backup=tmp_path/'original';shutil.copy2(path,backup)
    try:
        if damage=='last':
            with path.open('ab') as h:h.write(b'changed')
        elif damage=='best':path.unlink()
        elif damage=='journal':path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
        else:
            value=json.loads(path.read_text());value['full_training_complete']=True
            value['sha256']=api()._hash({k:v for k,v in value.items() if k!='sha256'})
            path.write_text(json.dumps(value))
        with pytest.raises((ValueError,FileNotFoundError)):api().read_fit(owned,require_full=False)
    finally:shutil.copy2(backup,path)


def test_resume_rejects_changed_last_or_partial_epoch_before_training(memory,runs,tmp_path,monkeypatch):
    root,_,_,_=runs();owned=tmp_path/'owned';shutil.copytree(root/'resumed',owned)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    before={str(p.relative_to(owned)):storage._sha256(p) for p in owned.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='SHA|sha|frozen'):
        call(memory,owned,scope='cpu_engineering',resume=True,expected_last_sha256='0'*64)
    after={str(p.relative_to(owned)):storage._sha256(p) for p in owned.rglob('*') if p.is_file()}
    assert after==before


def test_checkpoint_io_owns_native_paths_and_frozen_resume_sha(tmp_path):
    owned=tmp_path/'owned';owned.mkdir();(owned/'checkpoints').mkdir()
    io=api().checkpoint_io(owned)
    payload={'state_dict':{'weight':torch.arange(3)}}
    io.save_checkpoint(payload,owned/'checkpoints/last.ckpt')
    frozen=storage._sha256(owned/'checkpoints/last.ckpt')
    loaded=api().checkpoint_io(owned,expected_last_sha256=frozen).load_checkpoint(owned/'checkpoints/last.ckpt')
    assert torch.equal(loaded['state_dict']['weight'],torch.arange(3))
    for p in (owned/'history.ckpt',tmp_path/'unowned.ckpt'):
        with pytest.raises(ValueError):io.save_checkpoint(payload,p)
        with pytest.raises(ValueError):io.remove_checkpoint(p)
    with pytest.raises(ValueError):api().checkpoint_io(owned,expected_last_sha256='0'*64).load_checkpoint(owned/'checkpoints/last.ckpt')


def test_wrong_model_resume_preserves_previous_valid_receipt(memory,runs,monkeypatch):
    root,_,_,_=runs();output=root/'control'
    before={str(p.relative_to(output)):storage._sha256(p) for p in output.rglob('*') if p.is_file()}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with open_source(memory.root,
                     expected_completion_sha256=storage._sha256(memory.root/'completion.json'),
                     upstream=UPSTREAM,metadata_csv=memory.csv) as source:
        with pytest.raises(ValueError,match='identity differs'):
            api().fit_source(UPSTREAM,source,memory.csv,'middle',output,
                scope='cpu_engineering',resume=True,expected_last_sha256=before['checkpoints/last.ckpt'])
    after={str(p.relative_to(output)):storage._sha256(p) for p in output.rglob('*') if p.is_file()}
    assert after==before


def test_existing_writer_lock_refuses_without_leaking_open_descriptor(memory,runs,monkeypatch):
    root,_,_,_=runs();output=root/'resumed';lock=os.open(output/'fit.lock',os.O_RDWR)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    def descriptors():
        values={}
        for p in Path('/proc/self/fd').iterdir():
            try:values[int(p.name)]=os.readlink(p)
            except FileNotFoundError:pass
        return values
    try:
        before=descriptors()
        with pytest.raises(BlockingIOError):
            call(memory,output,scope='cpu_engineering',resume=True,
                 expected_last_sha256=storage._sha256(output/'checkpoints/last.ckpt'))
        assert descriptors()==before
    finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)


def test_completed_native_budget_cannot_be_extended_by_resume(memory,runs,monkeypatch):
    root,_,_,_=runs();output=root/'control'
    before={str(p.relative_to(output)):storage._sha256(p) for p in output.rglob('*') if p.is_file()}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError,match='completed native fit'):
        call(memory,output,scope='cpu_engineering',resume=True,
             expected_last_sha256=before['checkpoints/last.ckpt'])
    assert {str(p.relative_to(output)):storage._sha256(p) for p in output.rglob('*') if p.is_file()}==before


def test_exact_fit_and_witness_algorithms_use_private_new_globals():
    from egoemg_repro import emg2pose_memory_train as old,emg2pose_memory_runtime as old_runtime
    new=api();assert new.fit_source.__code__ is old.fit_source.__code__
    assert new.FitWitness is not old.FitWitness and new.FitWitness.check.__code__ is old.FitWitness.check.__code__
    assert old.fit_source.__globals__['runtime'] is old_runtime and old.FitWitness.check.__globals__ is old.fit_source.__globals__
    assert new.FitWitness.check.__globals__ is new.fit_source.__globals__ and new.fit_source.__globals__ is not old.fit_source.__globals__
    assert new.FORMAT!=old.FORMAT


def test_ordinary_fit_cli_refuses_archive_right_without_closing_it(monkeypatch):
    fd=os.open(os.devnull,os.O_RDONLY)
    try:
        monkeypatch.setenv('EGOEMG_STAGE_FD_ARCHIVE',str(fd))
        with pytest.raises(ValueError):api().main([])
        os.fstat(fd)
    finally:os.close(fd)


def test_old_family_resealed_fit_is_refused(runs,tmp_path):
    root,_,_,_=runs();out=tmp_path/'old-family';shutil.copytree(root/'control',out)
    for name in ('identity.json','fit-result.json'):
        p=out/name;v=json.loads(p.read_text());v['format']='emg2pose_native_memory_fit_v1';v['sha256']=api()._hash({k:w for k,w in v.items() if k!='sha256'});p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='format'):api().read_fit(out,require_full=False)


def test_actual_ordinary_val_only_source_cannot_train(native_corpus,tmp_path_factory):
    import csv
    from types import SimpleNamespace
    original=tmp_path_factory.mktemp('ordinary-val-only')
    with native_corpus.metadata_csv.open(newline='') as h:
        reader=csv.DictReader(h);fields=reader.fieldnames;rows=[r for r in reader if r['split']=='val']
    assert len(rows)==1
    shutil.copy2(native_corpus.root/(rows[0]['filename']+'.hdf5'),original/(rows[0]['filename']+'.hdf5'))
    metadata=original/'metadata.csv'
    with metadata.open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    only=pipeline_corpus.__wrapped__(SimpleNamespace(root=original,metadata_csv=metadata),tmp_path_factory)
    out=original/'never-fit';before=torch.get_rng_state().clone()
    with open_source(only.root,expected_completion_sha256=storage._sha256(only.root/'completion.json'),upstream=UPSTREAM,metadata_csv=only.csv) as source:
        assert source.data_identity['counts']['split']=={'val':1}
        with pytest.raises(ValueError,match='independent.*train.*val'):
            api().fit_source(UPSTREAM,source,only.csv,'small',out,scope='cpu_engineering')
        source.assert_current()
    assert not out.exists() and torch.equal(before,torch.get_rng_state())
