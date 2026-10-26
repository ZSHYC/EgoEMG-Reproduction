"""Post-termination sidecar contracts; CPU fixtures are never CUDA qualification."""
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def api():
    try:return importlib.import_module('egoemg_repro.classic_emg_evaluate')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.classic_emg_evaluate':pytest.fail('selected-best evaluation sidecar missing')
        raise


@pytest.mark.parametrize('variant,batch,final',[('vemg2pose',600,106),('neuropose',256,210)])
def test_test_policy_is_complete_canonical_batch_and_original_precision(variant,batch,final):
    p=api().evaluation_policy(variant=variant)
    assert p['batch_size']==batch and p['test_samples']==4306 and p['num_workers']==0
    assert p['precision']=='16-mixed' and p['device']=='cuda:0'
    assert p['limit_test_batches']==1.0 and p['shuffle'] is False and p['drop_last'] is False
    assert p['final_batch_size']==final and p['scientific_eligible'] is False


@pytest.mark.parametrize('device',['cpu','cuda:1','bf16'])
def test_wrong_device_is_rejected_before_any_training_run_read(tmp_path,device):
    with pytest.raises(ValueError,match='cuda|CUDA'):
        api().prepare_selected_best_evaluation(tmp_path/'absent',device=device)


@pytest.mark.parametrize('run',[{'passed':True},{'production_ready':True},True,None])
def test_passed_dict_or_boolean_is_not_a_terminated_run_artifact(run):
    with pytest.raises(ValueError,match='run|path|artifact'):api().prepare_selected_best_evaluation(run)


def test_engineering_or_partial_result_cannot_reach_any_factory_or_test(tmp_path,monkeypatch):
    r=api();run=tmp_path/'run';run.mkdir()
    (run/'identity.json').write_text('{"scope":"formal_engineering_acceptance","passed":true}')
    (run/'status.json').write_text('{"state":"engineering_finished"}')
    (run/'result.json').write_text('{"completed_epochs":2,"passed":true}')
    def forbidden(*args,**kwargs):raise AssertionError('factory/test reached before real termination admission')
    monkeypatch.setattr(r.cuda,'_factory',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)):r.prepare_selected_best_evaluation(run)


def batch(torch,*,index=0,value=1.,length=7790):
    return dict(emg=torch.zeros(1,16,length),joint_angles=torch.full((1,22,length),value),
        label_valid_mask=torch.ones(1,length,dtype=torch.bool),classic_sample_index=torch.tensor([index]),
        mask_policy=['official_literal_loader'],dataset_name=['egoemg'],hand=['left' if index%2==0 else 'right'],
        episode=torch.tensor([0]),start=torch.tensor([0]),end=torch.tensor([length]))


def windows():return [dict(role='test',episode=0,start=0,end=7790)]


def test_consumed_input_fingerprints_require_full_unique_order_and_weight_batch_loss():
    torch=pytest.importorskip('torch');r=api()
    ledger=r.TestInputLedger(windows(),variant='neuropose',batch_size=1)
    for index,loss in [(0,2.),(1,4.)]:
        value=batch(torch,index=index,value=float(index+1));receipt=ledger.prepare(value)
        ledger.commit(value,receipt,loss=loss)
    value=ledger.finish()
    assert value['samples']==2 and value['batch_sizes']==[1,1] and value['batch_weighted_loss']==3.
    assert len(value['target_sha256'])==len(value['mask_sha256'])==len(value['sample_sha256'])==64
    assert value['fingerprint_scope']=='consumed original input target span and generated 2D mask; no train-step prediction-hook claim'


@pytest.mark.parametrize('damage',['missing','duplicate','order','mask','target','metadata','nonfinite_loss'])
def test_partial_or_changed_consumed_test_inputs_cannot_publish(damage):
    torch=pytest.importorskip('torch');r=api()
    ledger=r.TestInputLedger(windows(),variant='neuropose',batch_size=1)
    value=batch(torch);receipt=ledger.prepare(value)
    with pytest.raises(ValueError):
        if damage=='missing':ledger.finish()
        elif damage=='duplicate':ledger.commit(value,receipt,loss=1.);ledger.commit(value,receipt,loss=1.)
        elif damage=='order':
            value=batch(torch,index=1);receipt=ledger.prepare(value);ledger.commit(value,receipt,loss=1.)
        elif damage=='mask':
            value['label_valid_mask']=torch.ones(1,22,7790,dtype=torch.bool);ledger.prepare(value)
        elif damage=='target':
            value['joint_angles'][0,0,0]=float('inf');ledger.prepare(value)
        elif damage=='metadata':
            value['episode'][0]=2;ledger.prepare(value)
        else:ledger.commit(value,receipt,loss=float('nan'))


def test_vemg_fingerprint_span_uses_context1790_and_actual6000_targets():
    torch=pytest.importorskip('torch');r=api()
    ledger=r.TestInputLedger(windows(),variant='vemg2pose',batch_size=1)
    value=batch(torch);value['emg']=torch.zeros(1,8,7790)
    first=ledger.prepare(value)
    assert first['target_shape']==[1,22,6000] and first['mask_shape']==[1,6000]
    outside=batch(torch);outside['emg']=torch.zeros(1,8,7790);outside['joint_angles'][...,0]=99.
    inside=batch(torch);inside['emg']=torch.zeros(1,8,7790);inside['joint_angles'][...,1790]=99.
    assert r.TestInputLedger(windows(),variant='vemg2pose',batch_size=1).prepare(outside)['target_sha256']==first['target_sha256']
    assert r.TestInputLedger(windows(),variant='vemg2pose',batch_size=1).prepare(inside)['target_sha256']!=first['target_sha256']


def test_sidecar_output_must_be_new_and_separate_from_trained_run(tmp_path):
    run=tmp_path/'trained';run.mkdir();(run/'status.json').write_bytes(b'original trained marker')
    for output in (run,run/'new-sidecar'):
        with pytest.raises((ValueError,FileExistsError)):
            api().run_selected_best_test(run,output=output)
    assert (run/'status.json').read_bytes()==b'original trained marker'


def test_actual_original_cpu_test_step_consumes_fixture_without_optimizer_or_weight_change(tmp_path):
    """Two-sample CPU engineering fixture; does not exercise CUDA admission."""
    torch=pytest.importorskip('torch');pl=pytest.importorskip('pytorch_lightning');r=api()
    pytest.importorskip('joblib');pytest.importorskip('pandas')
    torch.set_num_threads(2)
    project=Path(__file__).resolve().parents[1];upstream=project/'upstream'
    if not (upstream/'egoemg/train.py').exists():pytest.skip('pinned upstream is required for native CPU test fixture')
    cfg=r.cuda.models.compose_classic(upstream,'neuropose');r.cuda.models._activate(upstream)
    from egoemg._train_utils import setup_runtime
    from egoemg.train import make_lightning_module
    setup_runtime(cfg);module=make_lightning_module(cfg).eval()
    before=r.storage.state_digest(module.state_dict())
    samples=[]
    for index in (0,1):
        item=batch(torch,index=index,value=.03)
        samples.append({key:(value[0] if torch.is_tensor(value) or isinstance(value,list) else value) for key,value in item.items()})
    ledger=r.TestInputLedger(windows(),variant='neuropose',batch_size=1)
    observer=r.TestCoverageObserver(ledger)
    loader=torch.utils.data.DataLoader(samples,batch_size=1,collate_fn=r._Collate(ledger))
    trainer=pl.Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,callbacks=[observer],
        enable_model_summary=False,enable_progress_bar=False,default_root_dir=str(tmp_path))
    measured=trainer.test(model=module,dataloaders=loader,ckpt_path=None,verbose=False)
    assert observer.result['samples']==2 and observer.result['batch_sizes']==[1,1]
    assert measured[0]['test_loss']==pytest.approx(observer.result['batch_weighted_loss'],rel=2e-6)
    assert trainer.optimizers==[] and trainer.lr_scheduler_configs==[] and trainer.global_step==0
    assert r.storage.state_digest(module.state_dict())==before
    assert all(parameter.grad is None for parameter in module.parameters())


def test_selected_snapshot_reconstruction_keeps_strict_carried_best_and_never_uses_last_weights(tmp_path,monkeypatch):
    """Synthetic typed checkpoint contract only; not termination/CUDA evidence."""
    torch=pytest.importorskip('torch');r=api()
    monkeypatch.setattr(r.artifacts.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    import importlib.util
    path=Path(__file__).with_name('test_classic_emg_artifacts.py')
    spec=importlib.util.spec_from_file_location('evaluation_artifact_contract_fixture',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    a,identity,journal,io,value,directory=helper.prepared(tmp_path)
    try:
        io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt')
        io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
        last=a.read_formal_last(directory/'last.ckpt',identity)
        best=last['formal']['selected_best']['checkpoint'];event=best['formal']['journal_prefix']['epoch_summaries'][-1]
        selection=dict(selected_state_sha256=last['formal']['selected_best']['state_sha256'],
            selected_model_sha256=r.storage.state_digest(best['state_dict']),best_epoch=0,best_val_loss=2.,
            selected_event_sha256=a.json_hash(event))
        (directory/'best-epoch0.ckpt').unlink()
        last['state_dict']['model.weight'].fill_(9)  # Ensure helper cannot fall back to last weights.
        recovered=r._selected_core(last,selection)
        assert torch.equal(recovered['state_dict']['model.weight'],torch.ones(2,2))
        assert a._seal(recovered)==selection['selected_state_sha256']
        assert recovered['formal_carry']['segments'] and recovered['formal']['selected_best'] is None
        separate=tmp_path/'sidecar-best';separate.mkdir();saved=separate/'selected-best.ckpt'
        torch.save(recovered,saved)
        admitted=a._read(saved,identity,'best')
        assert a._seal(admitted)==selection['selected_state_sha256']
        changed=copy.deepcopy(selection);changed['selected_model_sha256']=r.storage.state_digest(last['state_dict'])
        with pytest.raises(ValueError,match='selected best'):r._selected_core(last,changed)
        changed=copy.deepcopy(selection);changed['selected_event_sha256']='f'*64
        with pytest.raises(ValueError,match='selected best'):r._selected_core(last,changed)
    finally:io.close();journal.close()


def test_evaluation_immutability_checks_reject_optimizer_buffers_and_scaler_changes():
    torch=pytest.importorskip('torch');r=api()
    module=torch.nn.BatchNorm1d(2)
    precision=SimpleNamespace(state_dict=lambda:{'scale':65536.})
    trainer=SimpleNamespace(optimizers=[],lr_scheduler_configs=[],global_step=0)
    model_sha=r.storage.state_digest(module.state_dict());scaler_sha=r.storage.state_digest(precision.state_dict())
    r._check_invariance(trainer,module,precision,model_sha,scaler_sha)
    module.running_mean[0]=1.
    with pytest.raises(ValueError,match='buffers|parameters'):r._check_invariance(trainer,module,precision,model_sha,scaler_sha)
    module.running_mean[0]=0.;trainer.optimizers=[object()]
    with pytest.raises(ValueError,match='optimizers'):r._check_invariance(trainer,module,precision,model_sha,scaler_sha)
    trainer.optimizers=[];precision.state_dict=lambda:{'scale':32768.}
    with pytest.raises(ValueError,match='scaler'):r._check_invariance(trainer,module,precision,model_sha,scaler_sha)


@pytest.mark.parametrize('fault',['readback_close_error','source_change_during_close','result_readback','late_validation_source_change','public_rebind'])
def test_sidecar_late_failure_revokes_its_success_and_preserves_training_artifacts(tmp_path,monkeypatch,fault):
    """Negative tail only: admission/fit are stubs; no CPU CUDA-positive path."""
    import json
    import importlib.util
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api()
    monkeypatch.setattr(r.artifacts.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    trained=tmp_path/'trained';trained.mkdir();marker=trained/'status.json';marker.write_bytes(b'immutable trained marker')
    source=tmp_path/'guarded-input';source.write_bytes(b'original guarded input')
    spec=importlib.util.spec_from_file_location('negative_sidecar_artifact_fixture',Path(__file__).with_name('test_classic_emg_artifacts.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    a,identity,journal,io,value,directory=helper.prepared(trained)
    io.save_checkpoint(copy.deepcopy(value),directory/'best-epoch0.ckpt');io.save_checkpoint(copy.deepcopy(value),directory/'last.ckpt')
    last=a.read_formal_last(directory/'last.ckpt',identity)
    core=last['formal']['selected_best']['checkpoint'];event=core['formal']['journal_prefix']['epoch_summaries'][-1]
    selection=dict(selected_state_sha256=last['formal']['selected_best']['state_sha256'],
        selected_model_sha256=r.storage.state_digest(core['state_dict']),best_epoch=0,best_val_loss=2.,selected_event_sha256=a.json_hash(event))
    best=r._selected_core(last,selection)
    model=torch.nn.Module();model.model=torch.nn.Linear(2,2,bias=False);model.load_state_dict(best['state_dict'])
    precision=SimpleNamespace(state_dict=lambda:copy.deepcopy(best['MixedPrecision']))
    guard=r.cohort._StatGuard([source,marker,directory/'last.ckpt',directory/'best-epoch0.ckpt'])
    prepared=SimpleNamespace(run=trained,identity={'sha256':'a'*64,'source_files_sha256':r._sources(),
        'evaluation_framework_files_sha256':r._framework()},selection=selection,selected_checkpoint=best,
        original_best={'present':True},manifest={'sha256':'b'*64,'state_identity':identity},
        policy=r.evaluation_policy(variant='vemg2pose'),guard=guard,module=model,precision=precision,
        datasets={'test':range(4306)},partitions={'windows':[dict(role='test',episode=0,start=i*7790,end=(i+1)*7790) for i in range(2153)]})
    output=tmp_path/'sidecar';moved=tmp_path/'moved-owned';injected=[]
    native_close=r.artifacts._Directory.close;native_write=r.artifacts._write_json
    old_last=(directory/'last.ckpt').read_bytes();old_best=(directory/'best-epoch0.ckpt').read_bytes()
    class Observer:
        def __init__(self,p,ledger):self.model_sha=selection['selected_model_sha256'];self.scaler_sha=r.storage.state_digest(best['MixedPrecision']);self.result=None
    class Trainer:
        def __init__(self,**kwargs):
            self.observer=kwargs['callbacks'][0];self.optimizers=[];self.lr_scheduler_configs=[];self.global_step=0;self.interrupted=False
        def test(self,**kwargs):
            self.observer.result={'test_loss':1.,'samples':4306};return [{'test_loss':1.}]
    def close(directory):
        native_close(directory)
        if directory.path==output and not injected and (output/'status.json').exists():
            if json.loads((output/'status.json').read_text())['state']=='selected_best_test_finished':
                if fault in ('readback_close_error','source_change_during_close'):
                    injected.append(fault)
                    if fault=='readback_close_error':raise OSError('injected final sidecar readback close failure')
                    source.write_bytes(b'changed during actual final readback close')
    def write(directory,name,value,**kwargs):
        native_write(directory,name,value,**kwargs)
        if name=='status.json' and value.get('state')=='selected_best_test_finished' and fault=='result_readback':
            injected.append(fault);(output/'result.json').write_text('{"changed_after_publication":true}')
        if name=='status.json' and value.get('state')=='selected_best_test_finished' and fault=='public_rebind':
            injected.append(fault);output.rename(moved);output.mkdir()
            (output/'status.json').write_bytes(b'foreign replacement status')
            (output/'checkpoint.pt').write_bytes(b'foreign replacement checkpoint')
    native_validate=r.artifacts.validate_formal_identity
    def validate(*args,**kwargs):
        native_validate(*args,**kwargs)
        if fault=='late_validation_source_change' and not injected and (output/'status.json').exists():
            if json.loads((output/'status.json').read_text())['state']=='selected_best_test_finished':
                injected.append(fault);source.write_bytes(b'changed after final native identity validation')
    monkeypatch.setattr(r,'prepare_selected_best_evaluation',lambda *args,**kwargs:prepared)
    monkeypatch.setattr(r,'SelectedBestTestProtocol',Observer);monkeypatch.setattr(r.storage.pl,'Trainer',Trainer)
    monkeypatch.setattr(r.artifacts._Directory,'close',close);monkeypatch.setattr(r.artifacts,'_write_json',write)
    monkeypatch.setattr(r.artifacts,'validate_formal_identity',validate)
    try:
        with pytest.raises(OSError if fault in ('readback_close_error','public_rebind') else ValueError):
            r.run_selected_best_test(trained,output=output)
        assert injected==[fault]
        owned_output=moved if fault=='public_rebind' else output
        if fault=='public_rebind':
            if (moved/'status.json').exists():
                assert json.loads((moved/'status.json').read_text())['state']!='selected_best_test_finished'
            assert (output/'status.json').read_bytes()==b'foreign replacement status'
            assert (output/'checkpoint.pt').read_bytes()==b'foreign replacement checkpoint'
        else:
            assert json.loads((output/'status.json').read_text())['state']=='failed'
            assert json.loads((output/'status.json').read_text())['scope']=='post_termination_selected_best_test'
        assert marker.read_bytes()==b'immutable trained marker'
        assert (directory/'last.ckpt').read_bytes()==old_last and (directory/'best-epoch0.ckpt').read_bytes()==old_best
        assert (owned_output/'selected-best.ckpt').is_file()
        if fault in ('source_change_during_close','late_validation_source_change'):
            with pytest.raises(ValueError):guard.check()
    finally:io.close();journal.close()


@pytest.mark.parametrize('damage',[None,'keys','shape','dtype'])
def test_prepare_loads_distinct_selected_parameters_and_buffers_after_schema_validation(tmp_path,monkeypatch,damage):
    """Actual prepare boundary only; admission is stubbed and scaler sentinel aborts."""
    import json
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');r=api()
    run=tmp_path/'trained';run.mkdir();(run/'checkpoints').mkdir()
    for name in ('identity.json','status.json','result.json'):(run/name).write_text('{}')
    (run/'checkpoints/last.ckpt').write_bytes(b'never parsed: negative admission fixture')
    partitions={'windows':[]};parts=tmp_path/'partitions.json';parts.write_text(json.dumps(partitions))
    config=tmp_path/'config.json';config.write_text('{}')
    cpu={'sha256':'fixture-cohort','data_identity':{'sha256':'fixture-data','model_protocol':'fixture-model'}}
    receipt=tmp_path/'cohort.json';receipt.write_text(json.dumps(cpu))
    module=torch.nn.Module();module.model=torch.nn.Linear(2,2,bias=False)
    module.register_buffer('running',torch.zeros(2))
    with torch.no_grad():module.model.weight.zero_()
    initial=r.storage.state_digest(module.state_dict())
    selected={'model.weight':torch.ones(2,2),'running':torch.full((2,),3.)}
    if damage=='keys':selected.pop('running')
    elif damage=='shape':selected['model.weight']=torch.ones(1,2)
    elif damage=='dtype':selected['running']=torch.full((2,),3.,dtype=torch.float64)
    best={'state_dict':selected,'MixedPrecision':{'scope':'boundary sentinel only'}}
    selection={'selected_model_sha256':r.storage.state_digest(selected)}
    assert initial!=selection['selected_model_sha256']
    manifest={'scope':'formal_training','sources':r.formal._source_snapshot(),'policy':{'variant':'neuropose'},
        'cohort_sha256':cpu['sha256'],'state_identity':{'counts':r.formal.FULL_COUNTS,'runtime':{'fixture':True},'recipe':{},
            'bindings':{'data_sha256':cpu['data_identity']['sha256'],'cohort_receipt_sha256':r.file_sha256(receipt),
                'partition_sha256':r.storage._json_hash(partitions)}},
        'provenance':{'data':str(tmp_path),'upstream':str(tmp_path),'config':str(config),'partitions':str(parts),
            'cohort_receipt':str(receipt),'finite_probe':None}}
    (run/'identity.json').write_text(json.dumps(manifest))
    last={'callbacks':{'ModelCheckpoint{fixture}':{'best_model_path':str(run/'checkpoints/missing-best.ckpt')}}}
    observed=[]
    class StopBeforeAdmissionSuccess(RuntimeError):pass
    class PrecisionBoundary:
        def load_state_dict(self,state):
            assert torch.equal(module.model.weight,torch.ones(2,2))
            assert torch.equal(module.running,torch.full((2,),3.))
            assert r.storage.state_digest(module.state_dict())==selection['selected_model_sha256']
            observed.append(True)
            raise StopBeforeAdmissionSuccess('actual selected values loaded; no CPU CUDA admission is returned')
    monkeypatch.setattr(r.formal,'verify_termination_and_select_best',lambda *_:selection)
    monkeypatch.setattr(r.artifacts,'read_formal_last',lambda *args,**kwargs:last)
    monkeypatch.setattr(r,'_selected_core',lambda *args:best)
    monkeypatch.setattr(r.formal,'_cohort_guard',lambda *args,**kwargs:r.cohort._StatGuard([receipt,parts,config]))
    monkeypatch.setattr(r.cuda,'_profile',lambda:({'fixture':True},PrecisionBoundary()))
    monkeypatch.setattr(r.cuda,'_prepare',lambda *args,**kwargs:SimpleNamespace(identity={'model':'fixture-model'}))
    monkeypatch.setattr(r.cuda,'_factory',lambda *_:module)
    with pytest.raises(ValueError if damage else StopBeforeAdmissionSuccess):r.prepare_selected_best_evaluation(run)
    if damage:
        assert observed==[] and r.storage.state_digest(module.state_dict())==initial
    else:assert observed==[True]


def test_failure_revocation_never_adopts_an_existing_readonly_namespace(tmp_path,monkeypatch):
    r=api();monkeypatch.setattr(r.artifacts.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    foreign=tmp_path/'foreign';foreign.mkdir();status=foreign/'status.json';status.write_bytes(b'foreign original status')
    captured=r.artifacts._Directory(foreign,create=False)
    try:
        r._fail_sidecar(foreign,captured,RuntimeError('not the creator'))
        assert status.read_bytes()==b'foreign original status'
    finally:captured.close()
