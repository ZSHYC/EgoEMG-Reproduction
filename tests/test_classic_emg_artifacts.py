"""Artifact/journal/typed-state contracts only; synthetic states are not GPU evidence."""
import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    try:return importlib.import_module('egoemg_repro.classic_emg_artifacts')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.classic_emg_artifacts':pytest.fail('formal artifact contract missing')
        raise


@pytest.fixture(autouse=True)
def capacity(monkeypatch):
    monkeypatch.setattr(api().shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))


def attempt(batch=0,epoch=0,step=0,executed=True):
    return dict(epoch=epoch,batch=batch,pl_step=step+1,optimizer_step_before=step,
        optimizer_step_after=step+int(executed),optimizer_executed=executed,
        scale_before=65536.,scale_after=65536. if executed else 32768.,
        gradient_finite=executed,gradient_abs_max=1. if executed else None,loss=1.)


def epoch_event(epoch=0):
    return dict(epoch=epoch,full=True,samples=601,expected_samples=601,val_loss=2.,
        batch_weighted_loss=2.,coverage_sha256='b'*64)


def journal(tmp_path):
    return api().FormalJournal(tmp_path/'journal',identity_sha256='a'*64,train_samples=1200,batch_size=600)


def test_journal_bounded_chain_reopen_and_tamper(tmp_path):
    r=api()
    with journal(tmp_path) as j:
        j.append_attempt(attempt());j.append_attempt(attempt(batch=1,step=1));j.append_epoch(epoch_event())
        prefix=j.prefix();carry=j.carry(prefix)
        assert prefix['attempts']==2 and prefix['executions']==2 and prefix['skips']==0
        assert prefix['epochs']==1 and prefix['max_attempts']==200
        assert prefix['total_budget_bytes']==64*1024**2
        for segment in carry['segments']:
            assert max(map(len,segment['data'].splitlines(keepends=True)))<=1024
            assert len(segment['data'])<=16*1024**2
        assert r.verify_journal_carry(carry,prefix,identity_sha256='a'*64) is None
    with r.FormalJournal(tmp_path/'journal',identity_sha256='a'*64,train_samples=1200,batch_size=600,
                         create=False,prefix=prefix) as reopened:
        assert reopened.prefix()==prefix
    changed=copy.deepcopy(carry);changed['segments'][0]['data']=changed['segments'][0]['data'].replace(b'"loss":1.0',b'"loss":9.0')
    with pytest.raises(ValueError):r.verify_journal_carry(changed,prefix,identity_sha256='a'*64)


@pytest.mark.parametrize('damage',['oversize','nan','unknown','step','scale','counter'])
def test_invalid_journal_attempt_never_appends(tmp_path,damage):
    value=attempt()
    if damage=='oversize':value['extra']='x'*1024
    elif damage=='nan':value['loss']=float('nan')
    elif damage=='unknown':value['private_extra']='not a record field'
    elif damage=='step':value['optimizer_step_after']=0
    elif damage=='scale':value['scale_after']=1.
    else:value['pl_step']=10
    with journal(tmp_path) as j:
        before=j.prefix()
        with pytest.raises(ValueError):j.append_attempt(value)
        assert j.prefix()==before


def test_low_space_and_failed_append_leave_exact_committed_prefix(tmp_path,monkeypatch):
    r=api()
    with journal(tmp_path) as j:
        j.append_attempt(attempt());before=j.prefix();saved=j.carry(before)
        monkeypatch.setattr(r.shutil,'disk_usage',lambda _:SimpleNamespace(free=1024**3))
        with pytest.raises(OSError,match='15GiB|reserve'):j.append_attempt(attempt(batch=1,step=1))
        assert j.prefix()==before and j.carry(before)==saved


def test_overflow_is_not_counted_as_optimizer_execution(tmp_path):
    with journal(tmp_path) as j:
        j.append_attempt(attempt());j.append_attempt(attempt(batch=1,step=1,executed=False))
        p=j.prefix();assert (p['attempts'],p['executions'],p['skips'])==(2,1,1)


def test_append_is_streaming_without_rereading_segment(tmp_path,monkeypatch):
    with journal(tmp_path) as j:
        j.append_attempt(attempt())
        monkeypatch.setattr(j.directory,'read',lambda *a:(_ for _ in ()).throw(AssertionError('segment reread per step')))
        j.append_attempt(attempt(batch=1,step=1))
        assert j.prefix()['attempts']==2


def test_fsync_failure_rolls_back_journal_prefix(tmp_path,monkeypatch):
    r=api()
    with journal(tmp_path) as j:
        j.append_attempt(attempt());before=j.prefix();carry=j.carry(before);original=r.os.fsync;calls=[]
        def fail(fd):
            if not calls:calls.append(True);raise OSError('injected journal fsync')
            return original(fd)
        monkeypatch.setattr(r.os,'fsync',fail)
        with pytest.raises(OSError,match='fsync'):j.append_attempt(attempt(batch=1,step=1))
        assert j.prefix()==before and j.carry(before)==carry


def test_scalar_metrics_are_rotated_journal_records_not_selection_events(tmp_path):
    with journal(tmp_path) as j:
        j.append_attempt(attempt());before=j.prefix()
        j.append_metric(dict(pl_step=1,name='lr-AdamW',value=.001))
        after=j.prefix()
        assert after['attempts']==before['attempts'] and after['epochs']==0
        assert after['metric_records']==1 and after['sequence']==before['sequence']+1


def test_journal_read_rejects_outside_name_before_file_open(tmp_path,monkeypatch):
    r=api();directory=r._Directory(tmp_path/'owned',create=True)
    (tmp_path/'neighbor').write_bytes(b'private neighbor sentinel');original=r.os.open;opened=[]
    def observed(name,*args,**kwargs):opened.append(name);return original(name,*args,**kwargs)
    monkeypatch.setattr(r.os,'open',observed)
    try:
        with pytest.raises(ValueError):directory.read('../neighbor',1024)
        assert opened==[]
    finally:directory.close()


def test_directory_public_verification_never_adopts_rebound_neighbor(tmp_path,monkeypatch):
    r=api();output=tmp_path/'owned';neighbor=tmp_path/'neighbor';neighbor.mkdir()
    (neighbor/'marker').write_bytes(b'foreign marker');original=r.os.open;rebound=[]
    def opening(path,flags,*args,**kwargs):
        if Path(path)==output and flags&r.os.O_DIRECTORY and not rebound:
            rebound.append(True);output.rename(tmp_path/'original-owned');neighbor.rename(output)
        return original(path,flags,*args,**kwargs)
    monkeypatch.setattr(r.os,'open',opening)
    directory=None
    try:
        with pytest.raises(ValueError):
            directory=r._Directory(output,create=True)
            r._write_json(directory,'new-artifact.json',{'synthetic':True})
        assert rebound and (output/'marker').read_bytes()==b'foreign marker'
        assert not (output/'new-artifact.json').exists()
    finally:
        if directory is not None:directory.close()


def test_directory_atomic_publication_does_not_replace_racing_destination(tmp_path,monkeypatch):
    r=api();output=tmp_path/'owned';neighbor=tmp_path/'neighbor';neighbor.mkdir()
    (neighbor/'marker').write_bytes(b'foreign marker');original=r.os.mkdir;rebound=[]
    def creating(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if not rebound:
            rebound.append(True);neighbor.rename(output)
        return result
    monkeypatch.setattr(r.os,'mkdir',creating)
    directory=None
    try:
        with pytest.raises(FileExistsError):directory=r._Directory(output,create=True)
        assert rebound and (output/'marker').read_bytes()==b'foreign marker'
        assert list(output.iterdir())==[output/'marker']
        assert set(tmp_path.iterdir())=={output}
    finally:
        if directory is not None:directory.close()


def identity(r):
    """Synthetic contract references and small tensor schema; no GPU qualification."""
    pl=pytest.importorskip('pytorch_lightning');project=Path(__file__).parents[1]
    recipe=json.loads((project/'configs/vemg2pose.json').read_text())
    value=dict(format=r.IDENTITY_FORMAT,scope=r.SCOPE,production_ready=False,scientific_eligible=False,
        recipe=recipe,effective=dict(precision='16-mixed',batch_size=600,global_batch_size=600,world_size=1,rank=0,
            num_workers=0,train_shuffle=True,seed=42,log_every_n_steps=50,max_epochs=100,clip_value=0),
        counts=dict(train=1200,val=601,test=601),
        bindings={k:'a'*64 for k in r.BINDING_KEYS},
        runtime=dict(runtime_profile='canonical_backend_nondeterministic_v1',deterministic=False,cudnn_deterministic=False,
            cudnn_benchmark=False,trainer_deterministic=False,matmul_precision='high',cublas_workspace_config=':4096:8',
            trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False,
            resume_scope='exact typed full checkpoint and RNG state restoration only; no bitwise trajectory guarantee',
            precision_plugin='MixedPrecision',scaler_enabled=True,execution_evidence='contract fixture; not executed'),
        source_files_sha256={name:r.file_sha256(project/name) for name in r.SOURCE_FILES},
        framework_files_sha256={name:r.file_sha256(Path(pl.__file__).resolve().parent/name) for name in r.FRAMEWORK_FILES},
        model_schema={'model.weight':dict(dtype='torch.float32',shape=[2,2])},parameter_names=['model.weight'])
    value['sha256']=r.json_hash(value)
    return value


@pytest.mark.parametrize('damage',['scope','precision','batch','workers','norm','runtime','source','dtype','gate'])
def test_formal_identity_is_exact_and_never_a_run_permit(damage):
    r=api();value=identity(r);expected=copy.deepcopy(value)
    assert r.validate_formal_identity(json.loads(json.dumps(value)),expected=expected) is None
    if damage=='scope':value['scope']='cuda_qualification'
    elif damage=='precision':value['effective']['precision']='32-true'
    elif damage=='batch':value['effective']['batch_size']=1
    elif damage=='workers':value['effective']['num_workers']=12
    elif damage=='norm':value['bindings']['normalization_sha256']='f'*64
    elif damage=='runtime':value['runtime']['deterministic']=True
    elif damage=='source':value['source_files_sha256'][next(iter(value['source_files_sha256']))]='0'*64
    elif damage=='dtype':value['effective']['world_size']=1.
    else:value['production_ready']=True
    value['sha256']=r.json_hash({k:v for k,v in value.items() if k!='sha256'})
    with pytest.raises(ValueError):r.validate_formal_identity(value,expected=expected)


def native_optimizer_scheduler(parameter,variant):
    """Use the pinned original Hydra path; do not hand-build a tuple-beta stand-in."""
    pytest.importorskip('hydra');classic=importlib.import_module('egoemg_repro.classic_emg')
    root=Path(__file__).parents[1]/'upstream';cfg=classic.compose_classic(root,variant)
    classic._activate(root);utils=importlib.import_module('egoemg.utils')
    assert Path(utils.__file__).resolve()==root/'egoemg/utils.py'
    configured=utils.instantiate_optimizer_and_scheduler(iter([parameter]),cfg.optimizer,cfg.lr_scheduler)
    return configured['optimizer'],configured['lr_scheduler']['scheduler']


def state(r,ident,j,directory):
    torch=pytest.importorskip('torch');cuda=importlib.import_module('egoemg_repro.classic_emg_cuda_check')
    parameter=torch.nn.Parameter(torch.ones(2,2))
    optimizer,scheduler=native_optimizer_scheduler(parameter,ident['recipe']['variant'])
    for _ in range(j.prefix()['executions']):parameter.grad=torch.ones_like(parameter);optimizer.step()
    for _ in range(j.prefix()['epochs']):scheduler.step()
    # The CUDA field is a synthetic typed placeholder, never an actual CUDA state.
    rng=dict(python=__import__('random').getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=[torch.get_rng_state()],generators={k:torch.get_rng_state() for k in ('train_loader','train_sampler','val_loader')})
    p=j.prefix();event=epoch_event();best=directory/'best-epoch0.ckpt'
    callbacks={'FormalProtocol':dict(completed_epochs=1,attempts=2,executions=2,skips=0,journal_prefix=p,
        last_validation=event,selected_best=dict(epoch=0,score=2.,event_sha256=r.json_hash(event))),
        "EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}":dict(wait_count=0,stopped_epoch=0,best_score=torch.tensor(2.),patience=50),
        "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min'}":dict(monitor='val_loss',best_model_score=torch.tensor(2.),
            best_model_path=str(best),current_score=torch.tensor(2.),dirpath=str(directory),best_k_models={str(best):torch.tensor(2.)},
            kth_best_model_path=str(best),kth_value=torch.tensor(2.),last_model_path=str(directory/'last.ckpt'))}
    loop={name:{} for name in cuda.LOOP_KEYS}
    loop['epoch_progress']={k:dict(ready=1,started=1,processed=1,completed=0) for k in ('current','total')}
    for k in ('epoch_loop.batch_progress','epoch_loop.val_loop.batch_progress'):
        loop[k]=dict(current=dict(ready=2,started=2,processed=2,completed=2),
                     total=dict(ready=2,started=2,processed=2,completed=2),is_last_batch=True)
    loop['epoch_loop.state_dict']={'_batches_that_stepped':2}
    loop['epoch_loop.scheduler_progress']={k:dict(ready=1,completed=1) for k in ('total','current')}
    loop['epoch_loop.manual_optimization.optim_step_progress']={k:dict(ready=0,completed=0) for k in ('total','current')}
    loop['epoch_loop.automatic_optimization.optim_progress']={'optimizer':{
        'step':{k:dict(ready=2,completed=2) for k in ('total','current')},
        'zero_grad':{k:dict(ready=2,started=2,completed=2) for k in ('total','current')}}}
    zero=dict(ready=0,started=0,processed=0,completed=0)
    inactive={key:dict(batch_progress=dict(current=zero.copy(),total=zero.copy(),is_last_batch=False),state_dict={})
              for key in ('validate_loop','test_loop')}
    inactive['predict_loop']=dict(batch_progress=dict(current=zero.copy(),total=zero.copy()),state_dict={})
    return dict(epoch=0,global_step=2,state_dict={'model.weight':torch.ones(2,2)},
        optimizer_states=[optimizer.state_dict()],lr_schedulers=[scheduler.state_dict()],callbacks=callbacks,loops={'fit_loop':loop,**inactive},
        MixedPrecision=dict(scale=65536.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=2),
        formal=dict(format=r.CHECKPOINT_FORMAT,identity=ident,completed_epochs=1,epoch_boundary=True,rng=rng,
                    journal_prefix=p,selected_best=None,role='pending'))


def prepared(tmp_path):
    r=api();ident=identity(r)
    j=r.FormalJournal(tmp_path/'journal',identity_sha256=ident['sha256'],train_samples=1200,batch_size=600)
    j.append_attempt(attempt());j.append_attempt(attempt(batch=1,step=1));j.append_epoch(epoch_event())
    directory=tmp_path/'checkpoints'
    io=r.FormalCheckpointIO(directory,ident,journal=j)
    value=state(r,ident,j,directory)
    return r,ident,j,io,value,directory


@pytest.mark.parametrize('stream',['torch','train_loader','train_sampler','val_loader'])
def test_unrestorable_cpu_rng_never_publishes_or_changes_global_rng(tmp_path,stream):
    torch=pytest.importorskip('torch');r,ident,j,io,value,directory=prepared(tmp_path)
    before=torch.get_rng_state().clone();malformed=torch.zeros(1,dtype=torch.uint8)
    other_rng=storage_digest((__import__('random').getstate(),np.random.get_state()))
    try:
        with pytest.raises(RuntimeError):torch.Generator(device='cpu').set_state(malformed)
        target=value['formal']['rng'] if stream=='torch' else value['formal']['rng']['generators']
        target[stream]=malformed
        with pytest.raises(ValueError):io.save_checkpoint(value,directory/'best-epoch0.ckpt')
        assert not (directory/'best-epoch0.ckpt').exists()
        assert torch.equal(torch.get_rng_state(),before)
        assert storage_digest((__import__('random').getstate(),np.random.get_state()))==other_rng
    finally:io.close();j.close()


def test_restorable_cpu_rng_survives_save_and_read_without_global_rng_change(tmp_path):
    torch=pytest.importorskip('torch');r,ident,j,io,value,directory=prepared(tmp_path)
    before=torch.get_rng_state().clone();expected={}
    other_rng=storage_digest((__import__('random').getstate(),np.random.get_state()))
    try:
        for seed,name in enumerate(('torch','train_loader','train_sampler','val_loader'),start=123):
            generator=torch.Generator(device='cpu').manual_seed(seed)
            torch.rand(7,generator=generator)
            target=value['formal']['rng'] if name=='torch' else value['formal']['rng']['generators']
            target[name]=generator.get_state()
            expected[name]=torch.rand(5,generator=generator)
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt')
        io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        restored=r.read_formal_last(directory/'last.ckpt',ident)['formal']['rng']
        for name,samples in expected.items():
            generator=torch.Generator(device='cpu')
            generator.set_state(restored['torch'] if name=='torch' else restored['generators'][name])
            assert torch.equal(torch.rand(5,generator=generator),samples)
        assert torch.equal(torch.get_rng_state(),before)
        assert storage_digest((__import__('random').getstate(),np.random.get_state()))==other_rng
    finally:io.close();j.close()


@pytest.mark.parametrize('mutation',['append','symlink','directory'])
def test_read_rechecks_checkpoint_and_namespace_after_final_identity(tmp_path,monkeypatch,mutation):
    r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt')
        io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        path=directory/'last.ckpt';original=r.validate_formal_identity;calls=[];mutated=[]
        def validating(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(True)
            if len(calls)==4:
                if mutation=='append':
                    with path.open('ab') as stream:stream.write(b' ')
                else:
                    moved=tmp_path/'original-checkpoints';directory.rename(moved)
                    if mutation=='symlink':directory.symlink_to(moved,target_is_directory=True)
                    else:
                        directory.mkdir()
                        # Even the identical checkpoint inode cannot authorize a new parent namespace.
                        os.link(moved/'last.ckpt',path)
                mutated.append(True)
            return result
        monkeypatch.setattr(r,'validate_formal_identity',validating)
        with pytest.raises(ValueError):r.read_formal_last(path,ident)
        assert len(calls)==4 and mutated
    finally:io.close();j.close()


def test_read_checks_parent_namespace_after_final_checkpoint_stamp(tmp_path,monkeypatch):
    r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt')
        io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        path=directory/'last.ckpt';neighbor=tmp_path/'preexisting-parent';neighbor.mkdir()
        # Establish the shared inode's final ctime before admission captures its stamp.
        os.link(path,neighbor/'last.ckpt')
        original=r.storage._stamp;calls=[];rebound=[]
        def stamping(candidate):
            result=original(candidate)
            if Path(candidate)==path:
                calls.append(True)
                if len(calls)==2:
                    directory.rename(tmp_path/'original-parent');neighbor.rename(directory);rebound.append(True)
            return result
        monkeypatch.setattr(r.storage,'_stamp',stamping)
        with pytest.raises(ValueError):r.read_formal_last(path,ident)
        assert len(calls)==2 and rebound
    finally:io.close();j.close()


def test_last_carries_nonrecursive_best_and_offline_migratable_journal(tmp_path):
    r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt')
        io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        last=r.read_formal_last(directory/'last.ckpt',ident)
        best=r.read_selected_best(directory/'last.ckpt',ident)
        assert best['formal']['role']=='best' and best['formal']['selected_best'] is None
        assert last['formal']['selected_best']['checkpoint']['formal']['selected_best'] is None
        assert 'formal_carry' not in last['formal']['selected_best']['checkpoint']
        assert last['formal_carry']['segments']
        assert type(last['optimizer_states'][0]['param_groups'][0]['betas']) is type(value['optimizer_states'][0]['param_groups'][0]['betas'])
        assert storage_digest(last['optimizer_states'])==storage_digest(value['optimizer_states'])
        assert storage_digest(last['lr_schedulers'])==storage_digest(value['lr_schedulers'])
        with pytest.raises(ValueError):r.read_formal_last(directory/'best-epoch0.ckpt',ident)
    finally:io.close();j.close()


@pytest.mark.parametrize('damage',['scaler','optimizer','callback','loop','loop_nested','rng','model_dtype','model_key',
                                 'optimizer_dtype','partial','best_path','prefix','cursor_dtype','identity_dtype','recursive',
                                 'validation_dtype','selection_dtype','callback_score_dtype','optimizer_defaults','scheduler_fields'])
def test_checkpoint_corruption_fails_before_publication(tmp_path,damage):
    r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        if damage=='scaler':del value['MixedPrecision']['_growth_tracker']
        elif damage=='optimizer':value['optimizer_states'][0]['state']={}
        elif damage=='callback':del value['callbacks']['FormalProtocol']
        elif damage=='loop':value['loops']['fit_loop']['epoch_progress']['current']['completed']=1
        elif damage=='loop_nested':value['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress']={}
        elif damage=='rng':value['formal']['rng']['numpy']=list(value['formal']['rng']['numpy'])
        elif damage=='model_dtype':value['state_dict']['model.weight']=value['state_dict']['model.weight'].half()
        elif damage=='model_key':value['state_dict']['bad']=value['state_dict'].pop('model.weight')
        elif damage=='optimizer_dtype':value['optimizer_states'][0]['state'][0]['step']=value['optimizer_states'][0]['state'][0]['step'].double()
        elif damage=='cursor_dtype':value['global_step']=2.
        elif damage=='identity_dtype':
            value['formal']['identity']=copy.deepcopy(ident);value['formal']['identity']['production_ready']=0
        elif damage=='validation_dtype':value['callbacks']['FormalProtocol']['last_validation']['epoch']=False
        elif damage=='selection_dtype':value['callbacks']['FormalProtocol']['selected_best']['epoch']=False
        elif damage=='callback_score_dtype':
            next(v for k,v in value['callbacks'].items() if k.startswith('EarlyStopping'))['best_score']=2.
        elif damage=='optimizer_defaults':value['optimizer_states'][0]['param_groups'][0]['betas'][0]=.1
        elif damage=='scheduler_fields':value['lr_schedulers'][0]['unknown']='not original scheduler state'
        elif damage=='recursive':value['formal']['selected_best']={'checkpoint':value,'state_sha256':'a'*64}
        elif damage=='partial':value['formal']['epoch_boundary']=False
        elif damage=='best_path':next(v for k,v in value['callbacks'].items() if k.startswith('ModelCheckpoint'))['best_model_path']='/outside/best.ckpt'
        else:value['formal']['journal_prefix']['chain_sha256']='0'*64
        with pytest.raises(ValueError):io.save_checkpoint(value,directory/'best-epoch0.ckpt')
        assert not (directory/'best-epoch0.ckpt').exists()
    finally:io.close();j.close()


def test_truncated_checkpoint_is_not_legacy_or_best_fallback(tmp_path):
    r=api();ident=identity(r);path=tmp_path/'bad.ckpt';path.write_bytes(b'truncated')
    with pytest.raises(ValueError,match='unreadable|truncated'):r.read_formal_last(path,ident)


def test_offline_last_restores_carried_best_journal_and_callbacks_in_new_directory(tmp_path):
    r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt');io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        admitted=r.read_formal_last(directory/'last.ckpt',ident)
        next_j=r.FormalJournal(tmp_path/'new-journal',identity_sha256=ident['sha256'],train_samples=1200,batch_size=600)
        migrated=r.FormalCheckpointIO(tmp_path/'new-checkpoints',ident,journal=next_j,resume=directory/'last.ckpt',admitted=admitted)
        try:
            loaded=migrated.load_checkpoint(directory/'last.ckpt')
            assert loaded['formal_seal']==r._seal(loaded)
            assert next_j.prefix()==j.prefix()
            assert storage_digest(loaded['optimizer_states'])==storage_digest(admitted['optimizer_states'])
            assert storage_digest(loaded['formal']['rng'])==storage_digest(admitted['formal']['rng'])
            monitor=next(v for k,v in loaded['callbacks'].items() if k.startswith('ModelCheckpoint'))
            assert Path(monitor['best_model_path']).parent==tmp_path/'new-checkpoints'
            assert (tmp_path/'new-checkpoints/recovered-best.ckpt').exists()
        finally:migrated.close();next_j.close()
    finally:io.close();j.close()


def test_checkpoint_capacity_rejects_before_creating_output(tmp_path,monkeypatch):
    r=api();ident=identity(r)
    with r.FormalJournal(tmp_path/'journal',identity_sha256=ident['sha256'],train_samples=1200,batch_size=600) as j:
        monkeypatch.setattr(r.shutil,'disk_usage',lambda _:SimpleNamespace(free=1024**3))
        with pytest.raises(OSError,match='15|space|reserve'):
            r.FormalCheckpointIO(tmp_path/'checkpoints',ident,journal=j)
        assert not (tmp_path/'checkpoints').exists()


def storage_digest(value):return api().storage.state_digest(value)


def test_last_failed_directory_fsync_preserves_previous_resume_bytes(tmp_path,monkeypatch):
    r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt');io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        before=(directory/'last.ckpt').read_bytes();original=r.os.fsync;calls=[]
        def fail(fd):
            if fd==io.directory.fd and not calls:calls.append(True);raise OSError('injected commit fsync')
            return original(fd)
        monkeypatch.setattr(r.os,'fsync',fail)
        with pytest.raises(OSError,match='fsync'):io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        assert (directory/'last.ckpt').read_bytes()==before
    finally:io.close();j.close()


@pytest.mark.parametrize('replacement',['symlink','directory'])
def test_atomic_failure_after_namespace_rebind_never_deletes_neighbor(tmp_path,monkeypatch,replacement):
    r=api();directory=r._Directory(tmp_path/'owned',create=True)
    neighbor=tmp_path/'neighbor';neighbor.mkdir();(neighbor/'marker').write_bytes(b'other valid marker')
    original=r.os.fsync;changed=[]
    def replace(fd):
        if fd==directory.fd and not changed:
            changed.append(True);directory.path.rename(tmp_path/'old-owned')
            if replacement=='symlink':directory.path.symlink_to(neighbor,target_is_directory=True)
            else:neighbor.rename(directory.path)
            raise OSError('injected namespace rebind before commit')
        return original(fd)
    monkeypatch.setattr(r.os,'fsync',replace)
    try:
        with pytest.raises(OSError,match='rebind'):
            r._atomic(directory,'marker',lambda s:s.write(b'new owned bytes'),4096,replace=False)
        target=neighbor if replacement=='symlink' else directory.path
        assert (target/'marker').read_bytes()==b'other valid marker'
        assert not (tmp_path/'old-owned/marker').exists()
    finally:directory.close()


def test_optimizer_counter_observes_real_calls_and_state_not_batch_count():
    torch=pytest.importorskip('torch');r=api();p=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([p],lr=.01)
    with r.OptimizerExecutionCounter(optimizer) as counter:
        counter.begin();p.grad=torch.ones(2);optimizer.step();first=counter.end()
        counter.begin();second=counter.end()  # no optimizer execution, no CUDA overflow claim
    assert first==dict(executions=1,step_before=0,step_after=1)
    assert second==dict(executions=0,step_before=1,step_after=1)


def test_older_selected_best_is_not_replaced_by_last_weights(tmp_path):
    torch=pytest.importorskip('torch');r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt')
        j.append_attempt(attempt(epoch=1,batch=0,step=2));j.append_attempt(attempt(epoch=1,batch=1,step=3));j.append_epoch(epoch_event(1))
        later=state(r,ident,j,directory);later['epoch']=1;later['global_step']=4
        later['state_dict']['model.weight'].fill_(9);later['optimizer_states'][0]['state'][0]['step'].fill_(4)
        later['lr_schedulers'][0].update(last_epoch=2,_step_count=3)
        later['MixedPrecision']['_growth_tracker']=4
        later['formal']['completed_epochs']=2
        p=later['callbacks']['FormalProtocol'];p.update(completed_epochs=2,attempts=4,executions=4,last_validation=epoch_event(1))
        next(v for k,v in later['callbacks'].items() if k.startswith('EarlyStopping'))['wait_count']=1
        loop=later['loops']['fit_loop']
        loop['epoch_progress']={k:dict(ready=2,started=2,processed=2,completed=1) for k in ('total','current')}
        loop['epoch_loop.state_dict']['_batches_that_stepped']=4
        loop['epoch_loop.batch_progress']['total']=dict(ready=4,started=4,processed=4,completed=4)
        loop['epoch_loop.scheduler_progress']['total']=dict(ready=2,completed=2)
        for values in loop['epoch_loop.automatic_optimization.optim_progress']['optimizer'].values():
            values['total']={k:4 for k in values['total']}
        io.save_checkpoint(later,directory/'last.ckpt')
        best=r.read_selected_best(directory/'last.ckpt',ident)
        assert torch.equal(best['state_dict']['model.weight'],torch.ones(2,2))
        assert best['formal']['completed_epochs']==1
    finally:io.close();j.close()


@pytest.fixture(scope='module')
def stock_top1_best_then_nonimproving(tmp_path_factory):
    """Actual stock PL CPU callback states, independent of artifact admission."""
    torch=pytest.importorskip('torch');pl=pytest.importorskip('pytorch_lightning')
    class Tiny(pl.LightningModule):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def training_step(self,batch,batch_idx):return self.weight.square()
        def validation_step(self,batch,batch_idx):
            self.log('val_loss',self.weight.detach().new_tensor(float(self.current_epoch+1)),
                     on_step=False,on_epoch=True,batch_size=1)
        def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.1)
    directory=tmp_path_factory.mktemp('stock-top1')
    monitor=pl.callbacks.ModelCheckpoint(dirpath=directory,monitor='val_loss',mode='min',save_top_k=1,
        save_last=True,save_on_train_epoch_end=True,filename='best-epoch{epoch}',auto_insert_metric_name=False)
    early=pl.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=50)
    trainer=pl.Trainer(accelerator='cpu',devices=1,precision='32-true',max_epochs=2,callbacks=[early,monitor],
        logger=False,num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,
        default_root_dir=directory)
    loader=torch.utils.data.DataLoader(torch.ones(1,1),batch_size=1,generator=torch.Generator().manual_seed(42))
    trainer.fit(Tiny(),train_dataloaders=loader,val_dataloaders=loader)
    best=torch.load(monitor.best_model_path,map_location='cpu',weights_only=False)
    last=torch.load(monitor.last_model_path,map_location='cpu',weights_only=False)
    native=last['callbacks'][monitor.state_key]
    assert type(monitor) is pl.callbacks.ModelCheckpoint
    assert best['epoch']==0 and last['epoch']==1
    assert float(best['callbacks'][monitor.state_key]['current_score'])==1.
    assert float(native['current_score'])==float(native['best_model_score'])==float(monitor.current_score)==1.
    assert float(trainer.callback_metrics['val_loss'])==2.
    print(json.dumps(dict(native_callback=type(monitor).__module__+'.'+type(monitor).__qualname__,
        lightning=pl.__version__,torch=str(torch.__version__),best_epoch=best['epoch'],last_epoch=last['epoch'],
        native_current_score=float(native['current_score']),latest_val_loss=float(trainer.callback_metrics['val_loss']))))
    return best,last,monitor.state_key,early.state_key


@pytest.mark.parametrize('forge_latest_loss',[False,True],ids=['native_top1','forged_latest_loss'])
def test_artifact_current_score_obeys_stock_top1_state(tmp_path,stock_top1_best_then_nonimproving,forge_latest_loss):
    torch=pytest.importorskip('torch');r=api();ident=identity(r)
    native_best,native_last,monitor_key,early_key=stock_top1_best_then_nonimproving
    first=epoch_event();first.update(val_loss=1.,batch_weighted_loss=1.)
    j=r.FormalJournal(tmp_path/'journal',identity_sha256=ident['sha256'],train_samples=1200,batch_size=600)
    j.append_attempt(attempt());j.append_attempt(attempt(batch=1,step=1));j.append_epoch(first)
    directory=tmp_path/'checkpoints';io=r.FormalCheckpointIO(directory,ident,journal=j)
    def callbacks(value,native):
        # Only known file locations adapt; all native numeric callback state is retained.
        monitor=copy.deepcopy(native['callbacks'][monitor_key]);monitor['dirpath']=str(directory)
        for key in ('best_model_path','kth_best_model_path','last_model_path'):
            if monitor[key]:monitor[key]=str(directory/Path(monitor[key]).name)
        monitor['best_k_models']={str(directory/Path(path).name):score for path,score in monitor['best_k_models'].items()}
        protocol=value['callbacks']['FormalProtocol']
        value['callbacks']={'FormalProtocol':protocol,monitor_key:monitor,
                            early_key:copy.deepcopy(native['callbacks'][early_key])}
        return monitor
    try:
        best=state(r,ident,j,directory);callbacks(best,native_best)
        protocol=best['callbacks']['FormalProtocol']
        protocol.update(last_validation=first,selected_best=dict(epoch=0,score=1.,event_sha256=r.json_hash(first)))
        io.save_checkpoint(best,directory/'best-epoch0.ckpt')
        j.append_attempt(attempt(epoch=1,batch=0,step=2));j.append_attempt(attempt(epoch=1,batch=1,step=3))
        j.append_epoch(epoch_event(1))
        later=state(r,ident,j,directory);monitor=callbacks(later,native_last)
        later.update(epoch=1,global_step=4);later['MixedPrecision']['_growth_tracker']=4
        later['formal']['completed_epochs']=2
        later['callbacks']['FormalProtocol'].update(completed_epochs=2,attempts=4,executions=4,
            last_validation=epoch_event(1),selected_best=dict(epoch=0,score=1.,event_sha256=r.json_hash(first)))
        loop=later['loops']['fit_loop']
        loop['epoch_progress']={k:dict(ready=2,started=2,processed=2,completed=1) for k in ('total','current')}
        loop['epoch_loop.state_dict']['_batches_that_stepped']=4
        loop['epoch_loop.batch_progress']['total']=dict(ready=4,started=4,processed=4,completed=4)
        loop['epoch_loop.scheduler_progress']['total']=dict(ready=2,completed=2)
        for progress in loop['epoch_loop.automatic_optimization.optim_progress']['optimizer'].values():
            progress['total']={k:4 for k in progress['total']}
        if forge_latest_loss:
            monitor['current_score']=torch.tensor(2.)
            with pytest.raises(ValueError,match='ModelCheckpoint'):io.save_checkpoint(later,directory/'last.ckpt')
            assert not (directory/'last.ckpt').exists()
        else:
            io.save_checkpoint(later,directory/'last.ckpt')
            admitted=r.read_formal_last(directory/'last.ckpt',ident)
            saved=admitted['callbacks'][monitor_key]
            assert float(saved['current_score'])==1. and admitted['formal']['journal_prefix']['epoch_summaries'][-1]['val_loss']==2.
            assert storage_digest(saved)==storage_digest(monitor)
            assert storage_digest(admitted['callbacks'][early_key])==storage_digest(native_last['callbacks'][early_key])
            assert r.read_selected_best(directory/'last.ckpt',ident)['formal']['completed_epochs']==1
        assert float(native_last['callbacks'][monitor_key]['current_score'])==1.
    finally:io.close();j.close()


@pytest.mark.parametrize('variant',['vemg2pose','neuropose'])
def test_native_hydra_adamw_group_is_admitted_without_rewriting(variant):
    torch=pytest.importorskip('torch');omega=pytest.importorskip('omegaconf');r=api()
    recipe=json.loads((Path(__file__).parents[1]/'configs'/f'{variant}.json').read_text())
    parameter=torch.nn.Parameter(torch.ones(2,2));optimizer,scheduler=native_optimizer_scheduler(parameter,variant)
    for _ in range(2):parameter.grad=torch.ones_like(parameter);optimizer.step()
    scheduler.step()
    value=dict(optimizer_states=[optimizer.state_dict()],lr_schedulers=[scheduler.state_dict()])
    group=value['optimizer_states'][0]['param_groups'][0]
    assert isinstance(group['betas'],omega.ListConfig) and list(group['betas'])==[.9,.999]
    assert value['lr_schedulers'][0]['last_epoch']==1
    before=storage_digest(value)
    r._optimizer_scheduler_state(value,dict(recipe=recipe,parameter_names=['model.weight']),1)
    assert storage_digest(value)==before and isinstance(group['betas'],omega.ListConfig)


@pytest.mark.parametrize('damage',['beta_values','beta_tuple','eps','group_extra',
                                 'scheduler_last_epoch','scheduler_T_max','scheduler_extra'])
def test_native_hydra_optimizer_scheduler_corruption_is_not_published(tmp_path,damage):
    r,ident,j,io,value,directory=prepared(tmp_path)
    try:
        group=value['optimizer_states'][0]['param_groups'][0];scheduler=value['lr_schedulers'][0]
        if damage=='beta_values':group['betas'][1]=.99
        elif damage=='beta_tuple':group['betas']=(.9,.999)
        elif damage=='eps':group['eps']=1e-6
        elif damage=='group_extra':group['unknown']='not original AdamW state'
        elif damage=='scheduler_last_epoch':scheduler['last_epoch']=0
        elif damage=='scheduler_T_max':scheduler['T_max']=99
        else:scheduler['unknown']='not original cosine state'
        expected='cosine' if damage.startswith('scheduler_') else 'AdamW'
        with pytest.raises(ValueError,match=expected):io.save_checkpoint(value,directory/'best-epoch0.ckpt')
        assert not (directory/'best-epoch0.ckpt').exists()
    finally:io.close();j.close()
