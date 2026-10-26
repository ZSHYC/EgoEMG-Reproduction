"""Runner contracts; these CPU tests never constitute actual CUDA acceptance."""
import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


def api():
    try:
        return importlib.import_module('egoemg_repro.classic_emg_formal')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.classic_emg_formal':
            pytest.fail('formal runner is not implemented')
        raise


@pytest.mark.parametrize('variant,batch', [('vemg2pose',600),('neuropose',256)])
def test_scientific_policy_preserves_complete_original_budget_and_batch(variant,batch):
    p=api().formal_policy(variant=variant)
    assert p['batch_size']==p['validation_batch_size']==batch
    assert p['trainer']==dict(accelerator='gpu',devices=1,precision='16-mixed',max_epochs=100,
        max_steps=-1,limit_train_batches=1.0,limit_val_batches=1.0,limit_test_batches=1.0,
        check_val_every_n_epoch=1,val_check_interval=1.0,accumulate_grad_batches=1,
        gradient_clip_val=0,log_every_n_steps=50,num_sanity_val_steps=0,deterministic=False)
    assert p['monitor']=='val_loss' and p['mode']=='min'
    assert p['early_stopping']==dict(patience=50,min_delta=0.,check_finite=True,check_on_train_epoch_end=False)
    assert p['runtime_profile']=='canonical_backend_nondeterministic_v1'
    assert p['scientific_eligible'] is False


@pytest.mark.parametrize('override', [dict(max_steps=4),dict(max_epochs=2),dict(batch_size=1),
    dict(num_workers=12),dict(limit_val_batches=.5),dict(stop_after_epoch=1),dict(device='cpu'),
    dict(precision='bf16-mixed'),dict(deterministic=True),dict(accumulate_grad_batches=2)])
def test_scientific_policy_refuses_unsupported_overrides(override):
    with pytest.raises((ValueError,TypeError)):
        api().formal_policy(variant='vemg2pose',**override)


def test_engineering_pause_has_explicit_separate_scope_and_never_scientific():
    p=api().formal_policy(variant='neuropose',engineering=True,engineering_epochs=2,stop_after_epoch=1)
    assert p['scope']=='formal_engineering_acceptance' and p['production_ready'] is False
    assert p['trainer']['max_epochs']==2 and p['trainer']['max_steps']==-1
    assert p['stop_after_epoch']==1


def test_coverage_checks_real_order_unique_ids_and_unequal_batch_weighting():
    c=api().Coverage(5,ordered=True,batch_size=3)
    c.add([0,1,2],loss=2.)
    c.add([3,4],loss=5.)
    result=c.finish()
    assert result['samples']==5 and result['batch_weighted_loss']==3.2
    assert result['batch_sizes']==[3,2]


@pytest.mark.parametrize('batches', [[[0,1,2],[2,4]],[[0,1,2]],[[0,2,1],[3,4]],[[0,1],[2,3,4]]])
def test_partial_duplicate_reordered_or_regrouped_validation_is_ineligible(batches):
    with pytest.raises(ValueError):
        c=api().Coverage(5,ordered=True,batch_size=3)
        for batch in batches:c.add(batch,loss=1.)
        c.finish()


@pytest.mark.parametrize('loss',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_validation_cannot_enter_selection(loss):
    c=api().Coverage(2,ordered=True,batch_size=2)
    with pytest.raises(ValueError):c.add([0,1],loss=loss)


def test_train_coverage_accepts_shuffle_but_rejects_duplicate_and_missing():
    c=api().Coverage(5,ordered=False,batch_size=3)
    c.add([4,1,0]);c.add([3,2]);assert c.finish()['samples']==5
    with pytest.raises(ValueError):
        c=api().Coverage(5,ordered=False,batch_size=3);c.add([0,1,2]);c.add([3,3])


def test_termination_requires_real_patience_or_completed_hundred_epochs():
    r=api()
    assert r.termination_reason([float(100-i) for i in range(51)]+[50.]*49,
        wait_count=49,stopped_epoch=0,best_score=50.,interrupted=False)=='max_epochs'
    assert r.termination_reason([1.]*51,wait_count=50,stopped_epoch=50,best_score=1.,interrupted=False)=='early_stopping'


@pytest.mark.parametrize('kwargs', [dict(losses=[1.]*2,wait_count=1,stopped_epoch=0,best_score=1.,interrupted=False),
    dict(losses=[1.]*51,wait_count=50,stopped_epoch=0,best_score=1.,interrupted=False),
    dict(losses=[1.]*100,wait_count=99,stopped_epoch=99,best_score=1.,interrupted=True),
    dict(losses=[math.nan]*100,wait_count=50,stopped_epoch=99,best_score=math.nan,interrupted=False)])
def test_external_partial_or_nonfinite_stop_is_not_training_completion(kwargs):
    with pytest.raises(ValueError):api().termination_reason(**kwargs)


def test_caller_passed_boolean_cannot_open_production_before_data_access(tmp_path):
    with pytest.raises(ValueError,match='artifact|qualification|path'):
        api().prepare_formal_run(tmp_path/'absent-data',upstream=tmp_path/'absent-upstream',
            config=tmp_path/'absent-config',partitions={},cohort_receipt={'passed':True},
            qualification={'passed':True})


def test_unterminated_directory_cannot_offer_a_selected_best(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        api().verify_termination_and_select_best(tmp_path)


def test_actual_callbacks_select_first_minimum_loss_despite_opposite_mae_and_unequal_batches(tmp_path):
    torch=pytest.importorskip('torch');pl=pytest.importorskip('pytorch_lightning')
    r=api()
    class Module(pl.LightningModule):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def training_step(self,batch,batch_idx):return self.weight.square()
        def validation_step(self,batch,batch_idx):
            loss=torch.tensor(([2.,5.] if self.current_epoch==0 else [3.2,3.2])[batch_idx])
            self.log('val_loss',loss,on_epoch=True,batch_size=len(batch['emg']))
            self.log('val_mae',torch.tensor(100. if self.current_epoch==0 else 0.),on_epoch=True,batch_size=len(batch['emg']))
            return loss
        def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.01)
    data=[{'emg':torch.zeros(1),'classic_sample_index':i} for i in range(5)]
    loader=torch.utils.data.DataLoader(data,batch_size=3)
    observer=r.ValidationCoverage(samples=5,batch_size=3)
    from omegaconf import OmegaConf
    cfg=OmegaConf.create({'callbacks':[
        {'_target_':'pytorch_lightning.callbacks.ModelCheckpoint','monitor':'val_loss','mode':'min','save_top_k':1,'save_last':True},
        {'_target_':'pytorch_lightning.callbacks.EarlyStopping','monitor':'val_loss','mode':'min','patience':50,'min_delta':0,'check_finite':True,'check_on_train_epoch_end':False},
        {'_target_':'pytorch_lightning.callbacks.LearningRateMonitor','logging_interval':'epoch'}]})
    callbacks=r.build_callbacks(cfg,tmp_path/'checkpoints',observer)
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=2,logger=pl.loggers.CSVLogger(tmp_path/'logs'),
        callbacks=callbacks,num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False)
    trainer.fit(Module(),train_dataloaders=loader,val_dataloaders=loader)
    monitor=next(c for c in callbacks if type(c).__name__=='ModelCheckpoint')
    selected=torch.load(monitor.best_model_path,map_location='cpu',weights_only=False)
    assert selected['epoch']==0
    assert abs(float(monitor.best_model_score)-3.2)<1e-6
    assert observer.selected_best['epoch']==0 and observer.last_validation['samples']==5
    assert observer.last_validation['batch_weighted_loss']==pytest.approx(3.2)


def test_native_callback_restore_compares_only_fields_lightning_restores(tmp_path):
    """Actual CPU Trainer restore, aborted before the first resumed forward."""
    import hashlib
    torch=pytest.importorskip('torch');pl=pytest.importorskip('pytorch_lightning')
    from pytorch_lightning.callbacks import Callback,EarlyStopping,ModelCheckpoint
    r=api();directory=tmp_path/'checkpoints';observed={}
    class StopAtRestore(RuntimeError):pass
    class Module(pl.LightningModule):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def training_step(self,batch,batch_idx):return self.weight.square()
        def validation_step(self,batch,batch_idx):self.log('val_loss',torch.tensor(1.),on_epoch=True,batch_size=len(batch))
        def configure_optimizers(self):
            optimizer=torch.optim.AdamW(self.parameters(),lr=.001)
            return dict(optimizer=optimizer,lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100))
    def callbacks(observer=None):
        result=[EarlyStopping(monitor='val_loss',mode='min',patience=50,check_on_train_epoch_end=False),
            ModelCheckpoint(dirpath=directory,monitor='val_loss',mode='min',save_top_k=1,
                save_last=True,save_on_train_epoch_end=True)]
        return [observer,*result] if observer is not None else result
    def trainer(values,epochs):
        return pl.Trainer(accelerator='cpu',devices=1,max_epochs=epochs,callbacks=values,logger=False,
            num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False)
    loader=torch.utils.data.DataLoader(torch.zeros(2,1),batch_size=1)
    trainer(callbacks(),1).fit(Module(),train_dataloaders=loader,val_dataloaders=loader)
    path=directory/'last.ckpt';before=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    original_callbacks=r.storage.state_digest(checkpoint['callbacks'])
    class ObserveRestore(Callback):
        def on_train_start(self,trainer,pl_module):
            actual=r._callbacks_state(trainer);saved=checkpoint['callbacks']
            key=next(k for k in saved if k.startswith('ModelCheckpoint{'))
            # These are native values, never assigned to satisfy the comparison.
            assert float(saved[key]['current_score'])==1. and actual[key]['current_score'] is None
            assert r.storage.state_digest(pl_module.state_dict())==r.storage.state_digest(checkpoint['state_dict'])
            assert r.storage.state_digest([o.state_dict() for o in trainer.optimizers])==r.storage.state_digest(checkpoint['optimizer_states'])
            assert r.storage.state_digest([c.scheduler.state_dict() for c in trainer.lr_scheduler_configs])==r.storage.state_digest(checkpoint['lr_schedulers'])
            expected=r._canonical_callbacks(saved);restored=r._canonical_callbacks(actual)
            assert r.storage.state_digest(restored)==r.storage.state_digest(expected)
            # All native-restored selection and early-stop fields still matter.
            early=next(k for k in actual if k.startswith('EarlyStopping{'))
            for state_key,field in [(key,'best_model_score'),(key,'kth_value'),(early,'best_score'),(early,'wait_count')]:
                damaged=copy.deepcopy(actual);damaged[state_key][field]+=1
                assert r.storage.state_digest(r._canonical_callbacks(damaged))!=r.storage.state_digest(expected)
            observed['reached']=True
            raise StopAtRestore('no resumed training forward or update')
    with pytest.raises(StopAtRestore):
        trainer(callbacks(ObserveRestore()),2).fit(Module(),train_dataloaders=loader,
            val_dataloaders=loader,ckpt_path=str(path))
    assert observed=={'reached':True}
    assert r.storage.state_digest(checkpoint['callbacks'])==original_callbacks
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_missing_final_source_acceptance_cannot_be_promoted_from_an_engineering_result(tmp_path):
    (tmp_path/'status.json').write_text('{"state":"engineering_finished"}')
    (tmp_path/'result.json').write_text('{"passed":true,"production_ready":true}')
    with pytest.raises((ValueError,FileNotFoundError)):api().verify_formal_acceptance(tmp_path/'result.json')


@pytest.mark.parametrize('variant',['vemg2pose','neuropose'])
def test_original_model_contract_json_roundtrip_is_not_a_different_model(variant):
    import json,os
    pytest.importorskip('torch');pytest.importorskip('hydra')
    upstream=os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:pytest.skip('actual pinned official model metadata required')
    r=api();cfg=r.cuda.models.compose_classic(Path(upstream),variant)
    actual=r.cuda.models.resolved_classic_identity(cfg,r.cuda.models.classic_contract(variant))
    recorded=json.loads(json.dumps(actual))
    assert isinstance(actual['contract']['channel_indices'],tuple)
    assert isinstance(recorded['contract']['channel_indices'],list)
    assert r._model_contract_matches(recorded,actual)
    for field,value in [('parameters',1),('left_context',1),('input_channels',1),('channel_indices',[0])]:
        changed=copy.deepcopy(recorded);changed['contract'][field]=value
        assert not r._model_contract_matches(changed,actual)
    changed=copy.deepcopy(recorded);changed['raw_config_sha256']='0'*64
    assert not r._model_contract_matches(changed,actual)


def test_engineering_diagnostic_never_forwards_the_training_model_or_changes_rng(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    from contextlib import nullcontext
    r=api()
    class Original(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bn=torch.nn.BatchNorm1d(22);self.dropout=torch.nn.Dropout(.5);self.forward_calls=0
        def forward(self,batch):
            self.forward_calls+=1
            pred=self.dropout(self.bn(batch['joint_angles']))
            return pred,batch['joint_angles'],batch['label_valid_mask']
    original=Original().train();original.bn.eval()  # Preserve heterogeneous child modes too.
    module=SimpleNamespace(model=original,state_dict=lambda:original.state_dict())
    prepared=SimpleNamespace(identity={'counts':{'val':1}},module=module,
        policy={'batch_size':256,'variant':'neuropose','scope':'formal_engineering_acceptance'})
    observer=r.FormalProtocol(prepared,{'train_loader':torch.Generator(),'train_sampler':torch.Generator(),
        'val_loader':torch.Generator()},SimpleNamespace(),tmp_path)
    observer.witness_model=copy.deepcopy(original)
    optimizer=torch.optim.AdamW(original.parameters())
    precision=SimpleNamespace(forward_context=nullcontext,state_dict=lambda:{'scale':65536.})
    trainer=SimpleNamespace(precision_plugin=precision,optimizers=[optimizer])
    batch={'emg':torch.zeros(2,16,7790),'joint_angles':torch.ones(2,22,7790),
        'label_valid_mask':torch.ones(2,7790,dtype=torch.bool),'classic_sample_index':torch.arange(2)}
    before=r.storage.state_digest(original.state_dict());rng=r.storage.state_digest(r.storage._rng(observer.generators))
    witness=observer._witness(module,trainer,batch)
    assert original.forward_calls==0 and original.training is True and original.bn.training is False
    assert r.storage.state_digest(original.state_dict())==before
    assert r.storage.state_digest(r.storage._rng(observer.generators))==rng
    assert witness['prediction'].shape==(1,22,7790)


def test_full_budget_with_earlier_patience_exhaustion_cannot_claim_a_hundred_epoch_stop():
    losses=[1.]*60+[.5]*40
    with pytest.raises(ValueError):
        api().termination_reason(losses,wait_count=39,stopped_epoch=0,best_score=.5,interrupted=False)


def test_runner_logger_keeps_finite_metrics_bounded_and_excludes_post_checkpoint_tail(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');pl=pytest.importorskip('pytorch_lightning');r=api()
    monkeypatch.setattr(r.artifacts.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    class Module(pl.LightningModule):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def training_step(self,batch,batch_idx):
            loss=self.weight.square();self.log('train_loss',loss,on_step=True,on_epoch=True);return loss
        def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.01)
    with r.artifacts.FormalJournal(tmp_path/'journal',identity_sha256='a'*64,train_samples=2,batch_size=1) as journal:
        observer=SimpleNamespace(epoch_boundary=False)
        logger=r._JournalLogger(journal,observer)
        trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,logger=logger,
            log_every_n_steps=1,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
        trainer.fit(Module(),train_dataloaders=torch.utils.data.DataLoader(torch.zeros(2,1),batch_size=1))
        assert journal.prefix()['metric_records']>=2
        observer.epoch_boundary=True;prefix=journal.prefix()
        logger.log_metrics({'train_loss_epoch':1.},step=2)
        assert journal.prefix()==prefix and not list(tmp_path.rglob('*.csv'))


@pytest.mark.parametrize('damage',['inf','mask_width','angle_mask','protocol','float16_input'])
def test_runtime_batch_admission_rejects_nonfinite_or_nonliteral_contract(damage):
    torch=pytest.importorskip('torch');r=api()
    batch={'emg':torch.zeros(2,16,7790),'joint_angles':torch.ones(2,22,7790),
        'label_valid_mask':torch.ones(2,7790,dtype=torch.bool),'classic_sample_index':torch.arange(2),
        'mask_policy':['official_literal_loader']*2,'dataset_name':['egoemg']*2,'hand':['left','right']}
    if damage=='inf':batch['emg'][1,0,1]=float('inf')
    elif damage=='mask_width':batch['label_valid_mask']=torch.ones(2,22,7790,dtype=torch.bool)
    elif damage=='angle_mask':batch['angle_valid_mask']=torch.ones(2,22,7790,dtype=torch.bool)
    elif damage=='protocol':batch['mask_policy'][1]='legacy'
    else:batch['emg']=batch['emg'].half()
    with pytest.raises(ValueError):r.admit_formal_batch(batch,r.cuda.models.classic_contract('neuropose'))


@pytest.mark.parametrize('device',['cpu','cuda:1','bf16'])
def test_wrong_device_refuses_before_any_source_or_artifact_read(tmp_path,device):
    with pytest.raises(ValueError,match='CUDA|cuda'):
        api().prepare_formal_run(tmp_path/'missing',upstream=tmp_path/'missing',config=tmp_path/'missing',
            partitions={},cohort_receipt=None,qualification=None,device=device)


def test_measured_nonidentical_next_prediction_has_no_implicit_tolerance():
    torch=pytest.importorskip('torch');r=api()
    record={'input_sha256':'a'*64,'target_sha256':'b'*64,'mask_sha256':'c'*64,
        'prediction':torch.tensor([1.]),'model_sha256':'d'*64,'optimizer_sha256':'e'*64,'scaler_sha256':'f'*64}
    left={'before':copy.deepcopy(record),'after':copy.deepcopy(record),
        'execution':{'executions':1,'step_before':2,'step_after':3}}
    right=copy.deepcopy(left);right['after']['prediction'][0]+=1e-6
    with pytest.raises(ValueError,match='tolerance|differs'):r._compare_witness(left,right)


def test_actual_cuda_final_source_acceptance_is_never_inferred_from_cpu_tests():
    import os
    path=os.environ.get('EGOEMG_FORMAL_GPU_BUNDLE')
    if not path:pytest.skip('root must supply an actual three-process CUDA acceptance bundle')
    result=api().verify_formal_acceptance(Path(path))
    assert result['strict_state_restore'] is True and result['measured_final_states_equal'] is True
    assert result['production_ready'] is False and result['bitwise_trajectory_guaranteed'] is False


def test_nonnull_cohort_supplement_requires_original_probe_and_passes_it_to_full_recheck(tmp_path,monkeypatch):
    import hashlib,json
    r=api();probe=tmp_path/'finite-census.json';probe.write_bytes(b'original finite census\n')
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'supplement':{
        'file_sha256':hashlib.sha256(b'original finite census\n').hexdigest()}}))
    with pytest.raises(ValueError,match='probe|supplement'):
        r._cohort_guard(tmp_path,'partitions',upstream='upstream',config='config',receipt=receipt,finite_probe=None)
    wrong=tmp_path/'wrong.json';wrong.write_bytes(b'changed census')
    with pytest.raises(ValueError,match='probe|supplement'):
        r._cohort_guard(tmp_path,'partitions',upstream='upstream',config='config',receipt=receipt,finite_probe=wrong)
    def full_recheck(root,partitions,*,upstream,config,receipt,finite_probe):
        if finite_probe!=probe:raise AssertionError('original supplement path was not propagated to full cohort verifier')
        raise RuntimeError('reached genuine full-cohort verification boundary')
    monkeypatch.setattr(r.cohort,'cohort_stat_guard',full_recheck)
    with pytest.raises(RuntimeError,match='full-cohort'):
        r._cohort_guard(tmp_path,'partitions',upstream='upstream',config='config',receipt=receipt,finite_probe=probe)


@pytest.mark.parametrize('target',['io','journal'])
@pytest.mark.parametrize('fault',['close_error','source_change'])
def test_final_resource_close_cannot_leave_finished_status_or_bypass_last_guard(tmp_path,monkeypatch,target,fault):
    """Negative tail fixture: admission/fit/checkpoint contents never prove CUDA."""
    import json
    import os
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api()
    monkeypatch.setattr(r.artifacts.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    output=tmp_path/'out';source=tmp_path/'guarded-source.json';source.write_text('{"original":true}\n')
    guard=r.cohort._StatGuard([source]);policy=r.formal_policy(variant='neuropose',engineering=True)
    identity={'sha256':'a'*64,'counts':dict(r.FULL_COUNTS,train=512),
        'recipe':{'normalization':{},'mask_policy':'negative-boundary-fixture'}}
    prepared=SimpleNamespace(data=tmp_path,partitions={},upstream=tmp_path,guard=guard,
        factory_prepared=SimpleNamespace(cfg=SimpleNamespace()),identity=identity,policy=policy,
        module=SimpleNamespace(model=SimpleNamespace()),precision=None,
        manifest={'sha256':'b'*64,'scope':policy['scope']})
    prefix={'attempts':4,'executions':4,'skips':0}
    checkpoint={'formal':{'journal_prefix':prefix},'formal_seal':'c'*64}
    closed=[];revoked=[];native_journal=r.artifacts.FormalJournal
    native_close=r.artifacts._Directory.close;native_fail=r.cuda._fail_owned_output
    def inject(name):
        closed.append(name)
        assert json.loads((output/'status.json').read_text())['state']=='engineering_finished'
        if name==target:
            if fault=='close_error':raise OSError('injected final '+name+' close failure')
            source.write_text('{"changed_during_close":true}\n')
    class Observer:
        def __init__(self,*args):self.completed_epochs=2;self.witnesses={};self.resume_start=None
    class IO:
        def __init__(self,*args,**kwargs):self.directory=r.artifacts._Directory(output/'checkpoints',create=True)
        def close(self):self.directory.close();inject('io')
    class Journal:
        def __init__(self,*args,**kwargs):self.real=native_journal(*args,**kwargs)
        def prefix(self):return prefix
        def close(self):self.real.close();inject('journal')
    class Trainer:
        def __init__(self,**kwargs):self.interrupted=False
        def fit(self,*args,**kwargs):(output/'checkpoints/last.ckpt').write_bytes(b'negative fixture; never parsed as checkpoint')
    def close_directory(directory):
        native_close(directory)
        if directory.path==output:closed.append('owned')
    def revoke(path,exc,*,owned):
        os.fstat(owned.fd)  # The original creation-time capability is still live.
        native_fail(path,exc,owned=owned);revoked.append(type(exc).__name__)
    monkeypatch.setattr(r,'prepare_formal_run',lambda *args,**kwargs:prepared)
    monkeypatch.setattr(r.cuda.datasets,'build_classic_datasets',lambda *args,**kwargs:{k:range(v) for k,v in r.FULL_COUNTS.items()})
    monkeypatch.setattr(r,'FormalProtocol',Observer);monkeypatch.setattr(r,'build_callbacks',lambda *args:[])
    monkeypatch.setattr(r.artifacts,'FormalJournal',Journal);monkeypatch.setattr(r.artifacts,'FormalCheckpointIO',IO)
    monkeypatch.setattr(r.storage.pl,'Trainer',Trainer)
    monkeypatch.setattr(r.artifacts,'validate_formal_identity',lambda *args:None)
    monkeypatch.setattr(r.artifacts,'read_formal_last',lambda *args:checkpoint)
    monkeypatch.setattr(r,'_checkpoint_snapshot',lambda *args:{'scope':'negative fixture'})
    monkeypatch.setattr(r,'_read_run',lambda *args:None)
    monkeypatch.setattr(r.artifacts._Directory,'close',close_directory)
    monkeypatch.setattr(r.cuda,'_fail_owned_output',revoke)
    expected=OSError if fault=='close_error' else ValueError
    with pytest.raises(expected):
        r.run_formal(tmp_path,upstream=tmp_path,config=source,partitions={},cohort_receipt=source,
            qualification={},output=output,engineering=True)
    assert json.loads((output/'status.json').read_text())['state']=='failed'
    assert (output/'checkpoints/last.ckpt').read_bytes()==b'negative fixture; never parsed as checkpoint'
    assert closed==['io','journal','owned'] and revoked==[expected.__name__]
    if fault=='source_change':
        with pytest.raises(ValueError):guard.check()
