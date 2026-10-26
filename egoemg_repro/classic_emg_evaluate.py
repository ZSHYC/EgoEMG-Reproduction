"""Termination-qualified, selected-best test sidecar; no scientific completion.

The immutable training run is read only. Pickles are trusted local artifacts.
Input fingerprints describe consumed original targets/masks, not a forward-hook
observation of the original Lightning _step's prediction tuple. Prediction/report
passes and independent recomputation remain a later stage.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from . import classic_emg_formal as formal

artifacts=formal.artifacts
cohort=formal.cohort
cuda=formal.cuda
storage=formal.storage
file_sha256=formal.file_sha256
Callback=storage.Callback
FORMAT='classic_selected_best_test_v1'
SCOPE='post_termination_selected_best_test'
SOURCE_FILES=(*formal.SOURCES,'egoemg_repro/classic_emg_evaluate.py')
EVALUATION_FRAMEWORK_FILES=('trainer/trainer.py','loops/evaluation_loop.py','strategies/strategy.py',
    'strategies/single_device.py','plugins/precision/precision.py','loops/utilities.py')
IMPORTED_SOURCE_SHA256=file_sha256(Path(__file__))
FINGERPRINT_SCOPE='consumed original input target span and generated 2D mask; no train-step prediction-hook claim'


def evaluation_policy(*,variant,device='cuda:0'):
    if variant not in ('vemg2pose','neuropose') or device!='cuda:0':
        raise ValueError('selected-best testing requires original variant and actual CUDA cuda:0')
    batch=600 if variant=='vemg2pose' else 256
    return dict(scope=SCOPE,scientific_eligible=False,production_ready=False,variant=variant,
        device=device,precision='16-mixed',batch_size=batch,test_samples=4306,num_workers=0,
        world_size=1,shuffle=False,drop_last=False,limit_test_batches=1.0,
        final_batch_size=4306%batch,seed=42,test_loader_seed=45,
        runtime_profile=cuda.BACKEND_PROFILE,trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False,
        fingerprint_scope=FINGERPRINT_SCOPE)


def _sources():
    root=Path(__file__).resolve().parents[1]
    value={name:file_sha256(root/name) for name in SOURCE_FILES}
    if value['egoemg_repro/classic_emg_evaluate.py']!=IMPORTED_SOURCE_SHA256:
        raise ValueError('executed evaluation sidecar source changed after import')
    return value


def _framework():
    storage._runtime();root=Path(storage.pl.__file__).resolve().parent
    return {name:file_sha256(root/name) for name in EVALUATION_FRAMEWORK_FILES}


def _tensor_digest(value):
    value=value.detach().cpu().contiguous()
    digest=hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(value.dtype),shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode())
    digest.update(memoryview(value.numpy()).cast('B'))
    return digest.hexdigest()


class TestInputLedger:
    """Bounded canonical-batch receipts; input data is hashed on CPU before transfer."""
    def __init__(self,windows,*,variant,batch_size):
        self.windows=copy.deepcopy(windows);self.contract=cuda.models.classic_contract(variant)
        if (not self.windows or any(w.get('role')!='test' or w['end']-w['start']!=7790 for w in self.windows) or
            len({(w['episode'],w['start'],w['end']) for w in self.windows})!=len(self.windows)):
            raise ValueError('unique original frozen test windows required')
        self.samples=2*len(windows);self.batch_size=batch_size
        self.coverage=formal.Coverage(self.samples,ordered=True,batch_size=batch_size)
        self.prepared_samples=0;self.prepared_batches=0;self.committed_batches=0;self.pending={}
        self.loss_sum=0.;self.target_hash=hashlib.sha256();self.mask_hash=hashlib.sha256();self.sample_hash=hashlib.sha256()
    def prepare(self,batch):
        torch=storage._runtime();formal.admit_formal_batch(batch,self.contract)
        if batch['joint_angles'].device.type!='cpu' or batch['label_valid_mask'].device.type!='cpu':
            raise ValueError('original test input receipt must be captured before device transfer')
        ids=batch['classic_sample_index'].tolist();size=len(ids)
        expected=min(self.batch_size,self.samples-self.prepared_samples)
        if size!=expected or ids!=list(range(self.prepared_samples,self.prepared_samples+size)):
            raise ValueError('canonical complete test grouping/order differs')
        descriptors=[]
        for offset,index in enumerate(ids):
            w=self.windows[index//2];hand=('left','right')[index%2]
            for key in ('episode','start','end'):
                value=batch.get(key)
                if not torch.is_tensor(value) or value.shape!=(size,) or int(value[offset])!=w[key]:
                    raise ValueError('consumed original test sample metadata differs: '+key)
            if batch['hand'][offset]!=hand:raise ValueError('consumed original test hand identity differs')
            descriptors.append(dict(sample_index=index,episode=w['episode'],start=w['start'],end=w['end'],hand=hand))
        span=slice(self.contract.left_context,7790-self.contract.right_context)
        target=batch['joint_angles'][...,span];mask=batch['label_valid_mask'][...,span]
        receipt=dict(sequence=self.prepared_batches,batch_size=size,indices=ids,
            target_shape=list(target.shape),mask_shape=list(mask.shape),target_dtype=str(target.dtype),mask_dtype=str(mask.dtype),
            target_sha256=_tensor_digest(target),mask_sha256=_tensor_digest(mask),sample_sha256=artifacts.json_hash(descriptors))
        self.pending[receipt['sequence']]=copy.deepcopy(receipt)
        self.prepared_batches+=1;self.prepared_samples+=size
        return receipt
    def commit(self,batch,receipt,*,loss):
        if type(loss) is not float or not math.isfinite(loss):raise ValueError('finite original test loss required')
        if (not isinstance(receipt,dict) or receipt.get('sequence')!=self.committed_batches or
            self.pending.get(self.committed_batches)!=receipt or batch['classic_sample_index'].tolist()!=receipt['indices']):
            raise ValueError('test output must consume its exact prepared original input receipt once')
        self.coverage.add(receipt['indices']);self.loss_sum+=receipt['batch_size']*loss
        for digest,key in ((self.target_hash,'target_sha256'),(self.mask_hash,'mask_sha256'),(self.sample_hash,'sample_sha256')):
            header=artifacts.json_hash(dict(sequence=receipt['sequence'],batch_size=receipt['batch_size'],digest=receipt[key]))
            digest.update(bytes.fromhex(header))
        del self.pending[self.committed_batches];self.committed_batches+=1
    def finish(self):
        coverage=self.coverage.finish()
        if self.pending or self.prepared_samples!=self.samples or self.prepared_batches!=self.committed_batches:
            raise ValueError('unconsumed or incomplete original test inputs')
        return dict(coverage,batch_weighted_loss=self.loss_sum/self.samples,
            target_sha256=self.target_hash.hexdigest(),mask_sha256=self.mask_hash.hexdigest(),sample_sha256=self.sample_hash.hexdigest(),
            fingerprint_encoding='ordered chain of typed canonical-batch target/mask and frozen-sample receipts v1',
            fingerprint_scope=FINGERPRINT_SCOPE)


class _Collate:
    def __init__(self,ledger):self.ledger=ledger
    def __call__(self,samples):
        batch=storage._runtime().utils.data.default_collate(samples)
        batch['classic_test_receipt']=self.ledger.prepare(batch)
        return batch


class TestCoverageObserver(Callback):
    """Generic input/loss observer; CPU fixture use does not admit CUDA evaluation."""
    def __init__(self,ledger):self.ledger=ledger;self.result=None
    def on_test_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
        self.ledger.commit(batch,batch['classic_test_receipt'],loss=float(outputs.detach()))
    def on_test_end(self,trainer,pl_module):
        result=self.ledger.finish();actual=float(trainer.callback_metrics['test_loss'])
        if not math.isfinite(actual) or not math.isclose(actual,result['batch_weighted_loss'],rel_tol=2e-6,abs_tol=1e-7):
            raise ValueError('native original test_loss differs from complete batch-weighted loss')
        self.result=dict(result,test_loss=actual)


def _selected_core(last,selection):
    """Recover only the strictly admitted carried best; never last's model weights."""
    try:
        carried=last['formal']['selected_best'];best=copy.deepcopy(carried['checkpoint'])
        prefix=best['formal']['journal_prefix'];event=prefix['epoch_summaries'][-1]
        if (carried['state_sha256']!=selection['selected_state_sha256'] or artifacts._seal(best)!=carried['state_sha256'] or
            storage.state_digest(best['state_dict'])!=selection['selected_model_sha256'] or best['epoch']!=selection['best_epoch'] or
            event['epoch']!=selection['best_epoch'] or event['val_loss']!=selection['best_val_loss'] or
            artifacts.json_hash(event)!=selection['selected_event_sha256']):
            raise ValueError('selected best state/model/complete validation event differs')
        identity=last['formal']['identity']
        best['formal_carry']=dict(identity_sha256=identity['sha256'],segments=[
            dict(name=blob['name'],data=blob['data'][:descriptor['bytes']])
            for blob,descriptor in zip(last['formal_carry']['segments'],prefix['segments'])])
        artifacts._checkpoint(best,identity,'best',best['formal_carry'])
        best['formal_seal']=artifacts._seal(best)
        return best
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError('missing verified carried selected-best checkpoint') from exc


@dataclass
class PreparedEvaluation:
    run:Path
    manifest:dict
    selection:dict
    selected_checkpoint:dict
    original_best:dict
    policy:dict
    identity:dict
    guard:object
    module:object
    precision:object
    datasets:dict
    partitions:dict


def prepare_selected_best_evaluation(run,*,device='cuda:0'):
    if device!='cuda:0':raise ValueError('selected-best test requires actual CUDA cuda:0')
    if not isinstance(run,(str,os.PathLike)):raise ValueError('a terminated training run artifact path is required')
    run=artifacts._safe(run)
    documents=[run/name for name in ('identity.json','status.json','result.json','checkpoints/last.ckpt')]
    root=Path(__file__).resolve().parents[1]
    documents.extend(root/name for name in SOURCE_FILES)
    guard=cohort._StatGuard(documents)
    sources=_sources();guard.check()
    try:
        selection=formal.verify_termination_and_select_best(run)
        manifest=formal._json(run/'identity.json')
        policy=evaluation_policy(variant=manifest['policy']['variant'],device=device)
        state_identity=manifest['state_identity']
        if ({name:sources[name] for name in formal.SOURCES}!=manifest['sources'] or
            manifest['scope']!='formal_training' or state_identity['counts']!=formal.FULL_COUNTS):
            raise ValueError('terminated full-cohort formal training source identity required')
        last=artifacts.read_formal_last(run/'checkpoints/last.ckpt',state_identity)
        best=_selected_core(last,selection)
        monitor=next(value for key,value in last['callbacks'].items() if key.startswith('ModelCheckpoint{'))
        path=artifacts._safe(monitor['best_model_path'])
        if path.name=='last.ckpt':raise ValueError('last weights cannot substitute for selected best')
        original_best=dict(path=str(path),present=False,file_sha256=None)
        if path.exists():
            guard.add(path);original=artifacts._read(path,state_identity,'best')
            if artifacts._seal(original)!=selection['selected_state_sha256']:
                raise ValueError('existing best file is not the exact verified carried best')
            original_best.update(present=True,file_sha256=file_sha256(path))
        provenance=manifest['provenance']
        if not provenance['partitions']:raise ValueError('original partition artifact provenance required for frozen test')
        complete=formal._cohort_guard(provenance['data'],provenance['partitions'],upstream=provenance['upstream'],
            config=provenance['config'],receipt=provenance['cohort_receipt'],finite_probe=provenance['finite_probe'])
        guard.check();complete.paths.update(guard.paths);complete.parents.update(guard.parents);guard=complete
        cpu=formal._json(provenance['cohort_receipt'])
        if (cpu['sha256']!=manifest['cohort_sha256'] or cpu['data_identity']['sha256']!=state_identity['bindings']['data_sha256'] or
            file_sha256(provenance['cohort_receipt'])!=state_identity['bindings']['cohort_receipt_sha256']):
            raise ValueError('original full-cohort identity differs from terminated training')
        partitions=formal._json(provenance['partitions'],limit=256*1024**2)
        if storage._json_hash(partitions)!=state_identity['bindings']['partition_sha256']:
            raise ValueError('original frozen partition identity differs')
        runtime,precision=cuda._profile()
        if runtime!=state_identity['runtime']:raise ValueError('actual evaluation runtime/precision differs from qualified training')
        for name in EVALUATION_FRAMEWORK_FILES:guard.add(Path(storage.pl.__file__).resolve().parent/name)
        framework=_framework();guard.check()
        prepared=cuda._prepare(provenance['data'],provenance['upstream'],provenance['config'],partitions,runtime,
            expected_recipe=state_identity['recipe'])
        if prepared.identity['model']!=cpu['data_identity']['model_protocol']:
            raise ValueError('actual original factory/source/config differs from full-cohort model identity')
        module=cuda._factory(prepared)
        storage._strict_state(module,best)
        module.load_state_dict(best['state_dict'],strict=True)
        if storage.state_digest(module.state_dict())!=selection['selected_model_sha256']:
            raise ValueError('fresh original module did not load the exact selected best model')
        precision.load_state_dict(best['MixedPrecision'])
        datasets=cuda.datasets.build_classic_datasets(prepared.data,partitions,upstream=prepared.upstream,config=prepared.cfg,
            normalization=state_identity['recipe']['normalization'],mask_policy=state_identity['recipe']['mask_policy'])
        if {role:len(dataset) for role,dataset in datasets.items()}!=formal.FULL_COUNTS:
            raise ValueError('actual complete frozen dataset counts differ')
        identity=dict(format=FORMAT,scope=SCOPE,scientific_eligible=False,production_ready=False,
            training_manifest_sha256=manifest['sha256'],training_result_sha256=file_sha256(run/'result.json'),
            last_checkpoint_sha256=file_sha256(run/'checkpoints/last.ckpt'),selection=selection,original_best=original_best,
            policy=policy,runtime=runtime,source_files_sha256=sources,evaluation_framework_files_sha256=framework,
            cohort_receipt_sha256=file_sha256(provenance['cohort_receipt']),cohort_sha256=cpu['sha256'],
            data_sha256=cpu['data_identity']['sha256'],partition_sha256=storage._json_hash(partitions),
            fingerprint_scope=FINGERPRINT_SCOPE,
            prediction_pass='not executed here; later post-termination report/recomputation stage')
        identity['sha256']=artifacts.json_hash(identity)
        guard.check()
        return PreparedEvaluation(run,manifest,selection,best,original_best,policy,identity,guard,module,precision,datasets,partitions)
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError('incomplete or ineligible terminated training run artifact') from exc


def _check_invariance(trainer,module,precision,model_sha,scaler_sha):
    if trainer.optimizers or trainer.lr_scheduler_configs or trainer.global_step!=0:
        raise ValueError('evaluation must not configure or execute training optimizers/schedulers')
    if storage.state_digest(module.state_dict())!=model_sha:
        raise ValueError('selected model parameters or registered buffers changed during test')
    if storage.state_digest(precision.state_dict())!=scaler_sha:
        raise ValueError('selected checkpoint scaler changed during test')
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise ValueError('test must not create parameter gradients')


class SelectedBestTestProtocol(TestCoverageObserver):
    def __init__(self,prepared,ledger):
        super().__init__(ledger);self.prepared=prepared
        self.model_sha=prepared.selection['selected_model_sha256']
        self.scaler_sha=storage.state_digest(prepared.selected_checkpoint['MixedPrecision'])
    def on_test_start(self,trainer,pl_module):
        self.prepared.guard.check()
        if (type(trainer.precision_plugin).__name__!='MixedPrecision' or
            trainer.precision_plugin.precision!='16-mixed' or not trainer.precision_plugin.scaler.is_enabled() or
            next(pl_module.parameters()).device.type!='cuda' or not pl_module.automatic_optimization or pl_module.model.training):
            raise ValueError('genuine original CUDA MixedPrecision test/eval path required')
        _check_invariance(trainer,pl_module,trainer.precision_plugin,self.model_sha,self.scaler_sha)
    def on_test_batch_start(self,trainer,pl_module,batch,batch_idx,dataloader_idx=0):
        self.prepared.guard.check();formal.admit_formal_batch(batch,self.ledger.contract)
    def on_test_end(self,trainer,pl_module):
        super().on_test_end(trainer,pl_module)
        _check_invariance(trainer,pl_module,trainer.precision_plugin,self.model_sha,self.scaler_sha)
        self.prepared.guard.check()


def _sidecar_output(run,output):
    if not isinstance(run,(str,os.PathLike)) or not isinstance(output,(str,os.PathLike)):
        raise ValueError('real training run and new sidecar output paths required')
    run=artifacts._safe(run);output=artifacts._safe(output)
    if output==run or run in output.parents:raise ValueError('evaluation sidecar output must be separate from the immutable trained run')
    if output.exists():raise FileExistsError('sidecar output exists; no overwrite or implicit rerun')
    return output


def _fail_sidecar(output,owned,exc):
    """Revoke only this newly created sidecar through its retained capability."""
    owner=owned.owner
    if not isinstance(owner,artifacts._CreatedDirectory) or owned.path!=output or owner.path!=output:return
    try:
        held=os.fstat(owner.fd)
        if not stat.S_ISDIR(held.st_mode) or (held.st_dev,held.st_ino)!=owner.identity:return
    except (OSError,TypeError,ValueError):return
    try:
        try:os.unlink('status.json',dir_fd=owned.fd)
        except FileNotFoundError:pass
        os.fsync(owned.fd)
    except OSError:pass
    if not owner.is_current(output):return
    try:
        artifacts._write_json(owned,'status.json',dict(state='failed',scope=SCOPE,
            scientific_eligible=False,production_ready=False,error_type=type(exc).__name__,error=str(exc)))
    except (OSError,ValueError):pass


def run_selected_best_test(run,*,output,device='cuda:0'):
    output=_sidecar_output(run,output)
    prepared=prepare_selected_best_evaluation(run,device=device)
    windows=[w for w in prepared.partitions['windows'] if w['role']=='test']
    ledger=TestInputLedger(windows,variant=prepared.policy['variant'],batch_size=prepared.policy['batch_size'])
    if ledger.samples!=4306:raise ValueError('full original4306 test hand samples required')
    torch=storage._runtime();owned=None
    try:
        owned=artifacts._Directory(output,create=True)
        artifacts._write_json(owned,'identity.json',prepared.identity)
        artifacts._write_json(owned,'status.json',dict(state='running',scope=SCOPE,scientific_eligible=False,production_ready=False))
        checkpoint=prepared.selected_checkpoint;selected_path=output/'selected-best.ckpt'
        artifacts._atomic(owned,selected_path.name,lambda stream:torch.save(checkpoint,stream),artifacts._payload_budget(checkpoint),replace=False)
        readback=artifacts._read(selected_path,prepared.manifest['state_identity'],'best')
        if artifacts._seal(readback)!=prepared.selection['selected_state_sha256']:
            raise ValueError('sidecar immutable selected checkpoint reconstruction differs')
        reference=dict(format=FORMAT,scope='immutable_selected_best_reference',scientific_eligible=False,
            training_manifest_sha256=prepared.manifest['sha256'],selected_state_sha256=prepared.selection['selected_state_sha256'],
            selected_model_sha256=prepared.selection['selected_model_sha256'],selected_event_sha256=prepared.selection['selected_event_sha256'],
            checkpoint_name=selected_path.name,checkpoint_sha256=file_sha256(selected_path),original_best=prepared.original_best)
        artifacts._write_json(owned,'selected-best.json',reference)
        for path in (selected_path,output/'identity.json',output/'selected-best.json'):prepared.guard.add(path)
        used=formal._Dataset(prepared.datasets['test'],prepared,4306)
        loader=torch.utils.data.DataLoader(used,batch_size=prepared.policy['batch_size'],num_workers=0,shuffle=False,
            drop_last=False,generator=torch.Generator().manual_seed(45),collate_fn=_Collate(ledger))
        observer=SelectedBestTestProtocol(prepared,ledger)
        prepared.module.eval()
        trainer=storage.pl.Trainer(accelerator='gpu',devices=1,plugins=[prepared.precision],logger=False,
            callbacks=[observer],enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
            limit_test_batches=1.0,inference_mode=True,num_sanity_val_steps=0,log_every_n_steps=50,
            deterministic=False,default_root_dir=str(output))
        measured=trainer.test(model=prepared.module,dataloaders=loader,ckpt_path=None,verbose=False)
        if len(measured)!=1 or observer.result is None or getattr(trainer,'interrupted',False):
            raise ValueError('complete original selected-best test did not finish')
        _check_invariance(trainer,prepared.module,prepared.precision,observer.model_sha,observer.scaler_sha)
        metrics={name:float(value) for name,value in measured[0].items()}
        if 'test_loss' not in metrics or any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError('finite original official test metrics missing')
        if not math.isclose(metrics['test_loss'],observer.result['test_loss'],rel_tol=2e-6,abs_tol=1e-7):
            raise ValueError('returned native test metrics differ from complete observed test loss')
        result=dict(format=FORMAT,scope=SCOPE,scientific_eligible=False,production_ready=False,
            identity_sha256=prepared.identity['sha256'],selection=reference,metrics=metrics,coverage=observer.result,
            original_test_step=True,optimizer_executions=0,model_state_unchanged=True,scaler_state_unchanged=True,
            prediction_pass='not executed; subsequent report pass remains separate',
            process=cuda._process_identity('selected-best-test'))
        prepared.guard.check();owned.check()
        artifacts._write_json(owned,'result.json',result)
        artifacts._write_json(owned,'status.json',dict(state='selected_best_test_finished',scope=SCOPE,
            scientific_eligible=False,production_ready=False,result_sha256=file_sha256(output/'result.json')),replace=True)
        if formal._json(output/'result.json')!=result or formal._json(output/'selected-best.json')!=reference:
            raise ValueError('sidecar result/immutable selection readback differs')
        if formal._json(output/'status.json')['result_sha256']!=file_sha256(output/'result.json'):
            raise ValueError('sidecar final status/result binding differs')
        prepared.guard.check()
        if (_sources()!=prepared.identity['source_files_sha256'] or
            _framework()!=prepared.identity['evaluation_framework_files_sha256']):
            raise ValueError('executed sidecar or evaluation framework source changed before publication')
        artifacts.validate_formal_identity(prepared.manifest['state_identity']);prepared.guard.check();owned.check()
        return result
    except BaseException as exc:
        if owned is not None:_fail_sidecar(output,owned,exc)
        raise
    finally:
        if owned is not None:owned.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv)
    print(json.dumps(run_selected_best_test(args.run,output=args.output,device=args.device),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
