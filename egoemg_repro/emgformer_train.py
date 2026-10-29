"""Native EgoEMG EMGFormer200/B500/six-worker fit and point validation.

Engineering diagnostics never claim full training. Original callbacks, model,
optimizer and stochastic loader behavior remain native; parent-state recovery
does not promise bitwise continuation of persistent-worker stochastic training.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from . import classic_emg_train as storage
from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard
from .emgformer_literal import compose_emgformer,emgformer_protocol,emgformer_normalization
from .emgformer_cohort import admit_emgformer_cohort,native_emgformer_loaders,FULL_COUNTS
from .emgformer_runtime import build_emgformer_runtime,emgformer_optimizer_identity,emgformer_state_sha256
from .vision_literal_runtime import _schema
from .vision_literal_check import _native_restored_callbacks
from .vision_literal_train import (_json,_write,_sealed,_check_seal,_read_epochs,_epoch_bytes,_native_callback,
    _adam_steps,_loop_sources,_loop_paths)

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_egoemg_emgformer_native_fit_v1'
SCOPES=('full','cpu_engineering','cuda_engineering')
LOCAL_SOURCES=('egoemg_repro/emgformer_train.py','egoemg_repro/emgformer_runtime.py',
    'egoemg_repro/emgformer_cohort.py','egoemg_repro/emgformer_literal.py',
    'egoemg_repro/emgformer_data.py','egoemg_repro/classic_emg_train.py',
    'egoemg_repro/vision_literal_runtime.py','egoemg_repro/vision_literal_train.py','egoemg_repro/vision_literal_check.py')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported EMGFormer fit source changed')
    return {name:_file_hash(ROOT/name) for name in LOCAL_SOURCES}


def emgformer_fit_policy(upstream,config,*,scope='full'):
    if type(scope)is not str or scope not in SCOPES:raise ValueError('explicit supported EMGFormer scope required')
    from hydra.utils import instantiate
    protocol=emgformer_protocol(upstream,config);full=scope=='full';cpu=scope=='cpu_engineering'
    checkpoint=next(c for c in protocol['callbacks']if c['_target_']=='pytorch_lightning.callbacks.ModelCheckpoint')
    early=next(c for c in protocol['callbacks']if c['_target_']=='pytorch_lightning.callbacks.EarlyStopping')
    native=instantiate(early)
    return _sealed(dict(format='original_egoemg_emgformer_fit_policy_v1',scope=scope,size=protocol['size'],protocol_sha256=protocol['sha256'],
        max_epochs=200 if full else 2,precision='32-true'if cpu else 'bf16-mixed',
        batch_size_by_role={r:2 if cpu else 500 for r in ('train','val')},workers=0 if cpu else 6,
        prefetch_factor=None if cpu else 1,persistent_workers=not cpu,pin_memory=False,
        gradient_clip_val=protocol['trainer']['gradient_clip_val'],scheduler_T_max=200,scheduler_eta_min=1e-5,
        monitor=checkpoint['monitor'],monitor_mode=checkpoint['mode'],save_top_k=checkpoint['save_top_k'],save_last=checkpoint['save_last'],
        early_stopping=dict(monitor=native.monitor,mode=native.mode,patience=native.patience,state_key=native.state_key,
            check_on_train_epoch_end=native._check_on_train_epoch_end),native_callback_declarations=protocol['callbacks'],
        effective_loss_weights=protocol['effective_loss_weights'],training_target_shape=[22,12000],validation_target_shape=[22,1],
        num_sanity_val_steps=2 if full else 0,engineering_train_samples=None if full else 4 if cpu else 1000,
        native_global_loader_rng=True,single_rank=True,scientific_eligible=False,full_paper_complete=False,
        logger='original TensorBoardLogger; owned output path/version only',bitwise_training_trajectory_guaranteed=False))


def emgformer_stage_complete(policy,records):
    return (len(records)==policy['max_epochs']and all(math.isfinite(r['train_loss'])and
        math.isfinite(r['validation']['val_mae'])for r in records))


def _runtime_device(policy):
    import torch
    if policy['scope']=='cpu_engineering':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU engineering requires CUDA hidden at startup')
        return dict(accelerator='cpu',cuda_initialized=False)
    if torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise ValueError('actual single visible CUDA bf16 GPU required')
    free,total=torch.cuda.mem_get_info(0)
    if free<64*1024**3:raise ValueError('64GiB actual GPU memory reserve required before EMGFormer fit')
    return dict(accelerator='gpu',name=torch.cuda.get_device_name(0),free_memory_before=free,total_memory=total,
        visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version())


def _receipt(cohort,path,required):
    if path is None:
        if required:raise ValueError('full EMGFormer requires independently frozen complete numeric receipt')
        return None,None
    path=_safe(path)
    if _json(path)!=cohort.identity:raise ValueError('EMGFormer frozen receipt differs from actual complete admission')
    return path,dict(path=str(path),file_sha256=_file_hash(path),identity_sha256=cohort.identity['sha256'])


def _check_receipt(identity):
    binding=identity['cohort_receipt']
    if binding is None:
        if identity['policy']['scope']=='full':raise ValueError('full EMGFormer lacks frozen input receipt')
    elif (_file_hash(_safe(binding['path']))!=binding['file_sha256'] or _json(binding['path'])!=identity['cohort'] or
        binding['identity_sha256']!=identity['cohort']['sha256']):raise ValueError('EMGFormer frozen numeric receipt changed')


def _snap(module,trainer):
    d=storage.state_digest
    return dict(model=d(module.state_dict()),optimizer=d([o.state_dict() for o in trainer.optimizers]),
        scheduler=d([x.scheduler.state_dict() for x in trainer.lr_scheduler_configs]),
        callbacks=d(_native_restored_callbacks({cb.state_key:cb.state_dict() for cb in trainer.callbacks if cb.state_dict()})),
        loops=d(trainer.fit_loop.state_dict()),rng=d(storage._rng({})))


def _checkpoint_snap(c):
    d=storage.state_digest
    return dict(model=d(c['state_dict']),optimizer=d(c['optimizer_states']),scheduler=d(c['lr_schedulers']),
        callbacks=d(_native_restored_callbacks(c['callbacks'])),loops=d(c['loops']['fit_loop']),rng=d(c['emgformer_fit']['rng']))


def _batch(cohort,policy,batch,role,seen,allowed):
    import torch
    ids=batch['emgformer_sample_index'];limit=len(allowed)
    if ids.dtype!=torch.int64 or ids.ndim!=1:raise ValueError('actual EMGFormer consumption indices required')
    ids=ids.detach().cpu().numpy();n=len(ids);size=policy['batch_size_by_role'][role]
    if(n!=min(size,limit-len(seen))or len(set(map(int,ids)))!=n or any(int(i)not in allowed or int(i)in seen for i in ids)):
        raise ValueError('EMGFormer batch is repeated, partial or outside its population')
    if role=='val'and ids.tolist()!=list(range(len(seen),len(seen)+n)):raise ValueError('native EMGFormer validation order differs')
    t=12000 if role=='train'else 1
    if(batch['emg'].shape!=(n,8,12000)or batch['joint_angles'].shape!=(n,22,t)or batch['label_valid_mask'].shape!=(n,t)or
        batch['emg'].dtype!=torch.float32 or batch['joint_angles'].dtype!=torch.float32 or batch['label_valid_mask'].dtype!=torch.bool or
        batch['emg_channel_mask'].shape!=(n,8)or not bool(batch['emg_channel_mask'].all())or
        not bool(torch.isfinite(batch['emg']).all())or not bool(torch.isfinite(batch['joint_angles']).all())):
        raise ValueError('EMGFormer full/point input shape, mask, dtype or finiteness differs')
    ds=cohort.datasets[role];anchors=len(ds.windows)
    for j,index in enumerate(ids):
        hand,window=divmod(int(index),anchors);w=ds.windows[window];offset=int(batch['jitter_offset'][j]);name=('left','right')[hand]
        if(batch['role'][j]!=role or batch['split'][j]!=w['split']or batch['hand'][j]!=name or
            batch['target_hand'][j]!=name or int(batch['target_hand_index'][j])!=hand or batch['dataset_name'][j]!='egoemg'or
            int(batch['emgformer_repeat_index'][j])!=0 or int(batch['emgformer_window_index'][j])!=window or
            not 0<=offset<=w['jitter_max_offset']or int(batch['start'][j])!=w['start']+offset or
            int(batch['end'][j])!=w['end']+offset or int(batch['center'][j])!=w['center']+offset):
            raise ValueError('EMGFormer original anchor/jitter/hand binding differs')
    seen.update(map(int,ids));return n


def _hands(ids,anchors):
    left=sum(int(i)<anchors for i in ids);return dict(left=left,right=len(ids)-left)


def _close_loaders(loaders):
    for loader in loaders.values():
        iterator=getattr(loader,'_iterator',None)
        if iterator is not None:
            workers=list(iterator._workers);iterator._shutdown_workers();loader._iterator=None
            if any(w.is_alive()or w.exitcode is None for w in workers):raise RuntimeError('owned native loader worker did not exit')


class ValidationWitness(storage.Callback):
    """Digest native point validation and observe the same raw model forward."""
    def __init__(self,cohort,policy):self.cohort=cohort;self.policy=policy;self.phase=None;self.validation=None

    def _observe_forward(self,module):
        self.original_forward=module.forward
        def forward(batch):
            if self.phase!='val':return self.original_forward(batch)
            import torch
            captured=[];native_model_forward=module.model.forward
            def observe(*args,**kwargs):
                value=native_model_forward(*args,**kwargs);captured.append(value);return value
            module.model.forward=observe
            try:out=self.original_forward(batch)
            finally:module.model.forward=native_model_forward
            pred,target,mask=out
            if(len(captured)!=1 or not isinstance(captured[0],torch.Tensor)or pred.shape!=target.shape or
                pred.shape[1:]!=(22,1)or mask.shape!=(len(pred),1)or not bool(torch.isfinite(pred).all())):
                raise ValueError('original EMGFormer point validation differs/nonfinite')
            raw=captured[0];center=raw.shape[-1]//2
            if(not torch.equal(pred,raw[...,center:center+1])or not torch.equal(target,batch['joint_angles'])or
                not torch.equal(mask,batch['label_valid_mask'])):raise ValueError('native EMGFormer prediction center/target/mask selection differs')
            self.parts.append(storage.state_digest(dict(pred=pred,target=target,mask=mask,emg=batch['emg'],sample_index=batch['emgformer_sample_index'])))
            return out
        module.forward=forward

    def on_validation_start(self,trainer,module):
        if not hasattr(self,'original_forward'):self._observe_forward(module)

    def on_validation_epoch_start(self,trainer,module):self.parts=[];self.val_seen=set()

    def on_validation_batch_start(self,trainer,module,batch,batch_idx,dataloader_idx=0):
        if not trainer.sanity_checking:
            self.cohort.guard.check();self.phase='val'
            _batch(self.cohort,self.policy,batch,'val',self.val_seen,set(range(len(self.cohort.datasets['val']))))

    def on_validation_end(self,trainer,module):
        self.phase=None
        if trainer.sanity_checking:return
        score=float(trainer.callback_metrics['val_mae'])
        if len(self.val_seen)!=len(self.cohort.datasets['val'])or not math.isfinite(score):raise ValueError('incomplete/nonfinite EMGFormer validation')
        self.validation=dict(samples=len(self.val_seen),val_mae=score,prediction_sha256=_hash(self.parts),target_shape=[22,1],
            native_center_selection_exact=True,hand_samples=_hands(self.val_seen,len(self.cohort.datasets['val'].windows)))
        self.parts=[]


class EmgformerFitWitness(ValidationWitness):
    def __init__(self,prepared,cohort,identity,output,stop_after,receipt_path):
        super().__init__(cohort,identity['policy']);self.prepared=prepared;self.identity=identity;self.output=output
        self.stop_after=stop_after;self.steps=0;self.completed_epochs=0;self.records=[];self.pending=None;self.exact_restore=None
        self.source_guard=_StatGuard([*[ROOT/name for name in identity['fit_sources_sha256']],*_loop_paths().values(),
            *([]if receipt_path is None else[receipt_path])])

    def state_dict(self):return dict(steps=self.steps,completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records))

    def load_state_dict(self,state):
        if set(state)!={'steps','completed_epochs','records'}:raise ValueError('EMGFormer callback schema differs')
        self.steps=state['steps'];self.completed_epochs=state['completed_epochs'];self.records=copy.deepcopy(state['records'])

    def check(self):self.cohort.guard.check();self.source_guard.check();storage._space(self.output)

    def on_load_checkpoint(self,trainer,module,checkpoint):
        p=checkpoint['emgformer_fit']
        if p['identity']!=self.identity or p['epoch_boundary']is not True:raise ValueError('owned complete EMGFormer epoch required')
        self.pending=checkpoint

    def on_fit_start(self,trainer,module):
        if self.pending is not None:storage._restore_rng(self.pending['emgformer_fit']['rng'],{})
        self._observe_forward(module);self.step_hook=trainer.optimizers[0].register_step_post_hook(lambda *args:self._executed())

    def _executed(self):self.steps+=1

    def on_train_start(self,trainer,module):
        emgformer_optimizer_identity(self.prepared,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
        if(getattr(trainer.precision_plugin,'scaler',None)is not None or trainer.gradient_clip_val!=1 or trainer.accumulate_grad_batches!=1):
            raise ValueError('native EMGFormer precision/clip/accumulation differs')

    def on_train_epoch_start(self,trainer,module):
        self.check();self.cohort.check()
        if trainer.current_epoch!=self.completed_epochs:raise ValueError('resumed EMGFormer epoch differs from complete prefix')
        self.train_seen=set();self.train_loss_sum=0.;self.train_batches=0;self.validation=None

    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        self.check();self.phase='train';ids=self.identity['train_indices']
        allowed=set(range(len(self.cohort.datasets['train'])))if ids is None else set(ids)
        self.batch_n=_batch(self.cohort,self.policy,batch,'train',self.train_seen,allowed)

    def on_before_optimizer_step(self,trainer,module,optimizer):
        import torch
        grads=[p.grad for p in module.parameters()if p.grad is not None]
        if not grads or not bool(torch.stack([torch.isfinite(g).all()for g in grads]).all()):raise ValueError('EMGFormer gradients absent/nonfinite')

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        loss=float(outputs['loss'])
        if not math.isfinite(loss)or self.steps!=trainer.global_step:raise ValueError('native EMGFormer loss/update count differs')
        self.train_loss_sum+=self.batch_n*loss;self.train_batches+=1

    def on_train_epoch_end(self,trainer,module):
        self.check();self.cohort.check();samples=self.identity['train_samples'];batches=math.ceil(samples/self.policy['batch_size_by_role']['train'])
        if(len(self.train_seen)!=samples or self.validation is None or self.train_batches!=batches or self.steps!=(self.completed_epochs+1)*batches):
            raise ValueError('partial EMGFormer epoch cannot be committed')
        _adam_steps(trainer.optimizers[0],self.steps)
        record=dict(epoch=self.completed_epochs,optimizer_executions=self.steps,global_step=trainer.global_step,
            train_samples=samples,train_batches=batches,train_loss=self.train_loss_sum/samples,
            train_hand_samples=_hands(self.train_seen,len(self.cohort.datasets['train'].windows)),
            validation=copy.deepcopy(self.validation),model_sha256=emgformer_state_sha256(module.state_dict()))
        previous=_epoch_bytes(self.records,self.identity['sha256']);self.records.append(record);self.completed_epochs+=1
        raw=_epoch_bytes(self.records,self.identity['sha256']);path=self.output/'epochs.jsonl'
        if len(raw)>2*1024**2 or path.read_bytes()!=previous:raise ValueError('EMGFormer epoch journal prefix/bound differs')
        with path.open('ab')as f:f.write(raw[len(previous):]);f.flush();os.fsync(f.fileno())
        if self.stop_after==self.completed_epochs:trainer.should_stop=True

    def on_save_checkpoint(self,trainer,module,checkpoint):
        self.check();checkpoint['emgformer_fit']=dict(format=FORMAT,identity=self.identity,epoch_boundary=self.validation is not None,
            completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records),rng=storage._rng({}))

    def on_fit_end(self,trainer,module):module.forward=self.original_forward;self.step_hook.remove()


def _frozen_validation(prepared,cohort,policy,loader,device):
    # A complete fresh validation pass precedes resuming training; its RNG effects
    # do not alter the saved native checkpoint restoration boundary.
    before=storage._rng({});witness=ValidationWitness(cohort,policy)
    trainer=storage.pl.Trainer(accelerator=device['accelerator'],devices=1,precision=policy['precision'],
        callbacks=[witness],logger=False,enable_progress_bar=False,enable_model_summary=False,enable_checkpointing=False)
    try:
        trainer.validate(prepared.module,dataloaders=loader,verbose=False)
        return witness.validation
    finally:
        if hasattr(witness,'original_forward'):prepared.module.forward=witness.original_forward
        storage._restore_rng(before,{})


def read_emgformer_fit(output,*,require_full=True):
    output=_safe(output);identity=_json(output/'identity.json');result=_json(output/'fit-result.json')
    return _validate_fit(output,identity,result,require_full=require_full)


def _validate_fit(output,identity,result,*,require_full):
    """Validate a candidate against native durable files before publishing it."""
    import torch
    _check_seal(identity,FORMAT);_check_seal(result,FORMAT);_check_seal(identity['policy'],'original_egoemg_emgformer_fit_policy_v1')
    if (result['identity_sha256']!=identity['sha256'] or identity['fit_sources_sha256']!=_sources() or
        identity['native_loop_sources_sha256']!=_loop_sources() or result['scientific_eligible'] is not False or
        result['full_paper_complete'] is not False):raise ValueError('EMGFormer fit/source/result identity differs')
    _check_receipt(identity);policy=identity['policy']
    if result['scope']!=policy['scope'] or (require_full and policy['scope']!='full'):raise ValueError('engineering EMGFormer is not complete full training')
    last_path=output/'checkpoints/last.ckpt'
    if _file_hash(last_path)!=result['last_checkpoint_sha256']:raise ValueError('EMGFormer last checkpoint bytes changed')
    last=torch.load(str(last_path),map_location='cpu',weights_only=False);packet=last['emgformer_fit']
    records,_=_read_epochs(output/'epochs.jsonl',identity['sha256']);samples=identity['train_samples']
    ids=identity['train_indices'];anchors=identity['cohort']['counts']['train']//2
    expected_hands=dict(left=anchors,right=anchors)if ids is None else _hands(ids,anchors)
    batches=math.ceil(samples/policy['batch_size_by_role']['train']);steps=batches*len(records)
    if (not records or packet['identity']!=identity or packet['records']!=records or packet['epoch_boundary'] is not True or
        packet['completed_epochs']!=len(records) or result['completed_epochs']!=len(records) or
        result['optimizer_executions']!=steps or last['global_step']!=steps):raise ValueError('EMGFormer actual epoch/optimizer/journal carry differs')
    for i,r in enumerate(records):
        if (r['train_samples']!=samples or r['train_batches']!=batches or r['optimizer_executions']!=(i+1)*batches or
            r['global_step']!=(i+1)*batches or r['validation']['samples']!=identity['cohort']['counts']['val'] or
            r['train_hand_samples']!=expected_hands or r['validation']['target_shape']!=[22,1]or
            r['validation']['native_center_selection_exact']is not True or
            r['validation']['hand_samples']!=dict(left=identity['cohort']['counts']['val']//2,right=identity['cohort']['counts']['val']//2)or
            not math.isfinite(r['train_loss']) or not math.isfinite(r['validation']['val_mae'])):
            raise ValueError('partial/nonfinite EMGFormer epoch')
    early=last['callbacks'].get(policy['early_stopping']['state_key'])
    if early is None or early['patience']!=200 or float(early['best_score'])!=min(r['validation']['val_mae']for r in records):
        raise ValueError('native EMGFormer early stopping state differs')
    if (_schema(last['state_dict'])!=identity['construction']['model_schema'] or
        emgformer_state_sha256(last['state_dict'])!=records[-1]['model_sha256'] or
        result['final_model_state_sha256']!=records[-1]['model_sha256']):raise ValueError('terminal EMGFormer tensor state differs')
    native=_native_callback(last,'ModelCheckpoint{');best=Path(native['best_model_path']);score=float(native['best_model_score'])
    if (best.parent!=output/'checkpoints' or best.is_symlink() or best.name=='last.ckpt' or
        result['selected_best']!=dict(file=best.name,score=score,checkpoint_sha256=_file_hash(best),validated=True) or
        score!=min(r['validation']['val_mae'] for r in records)):raise ValueError('EMGFormer selected checkpoint differs from actual val best')
    selected=torch.load(str(best),map_location='cpu',weights_only=False);p=selected['emgformer_fit'];epoch=p['completed_epochs']-1
    if (p['identity']!=identity or p['epoch_boundary'] is not True or not 0<=epoch<len(records) or
        p['records']!=records[:epoch+1] or records[epoch]['validation']['val_mae']!=score or
        selected['global_step']!=records[epoch]['global_step'] or _schema(selected['state_dict'])!=identity['construction']['model_schema'] or
        emgformer_state_sha256(selected['state_dict'])!=records[epoch]['model_sha256']):raise ValueError('selected EMGFormer state/epoch differs')
    done=emgformer_stage_complete(policy,records);full=done and policy['scope']=='full'
    if (result['stage_complete'] is not done or result['full_training_terminated'] is not full or
        result['train_samples_per_epoch']!=[r['train_samples'] for r in records] or
        result['val_samples_per_epoch']!=[r['validation']['samples']for r in records]or
        result['train_hand_samples_per_epoch']!=[r['train_hand_samples']for r in records]):raise ValueError('EMGFormer completion/coverage differs')
    if require_full and (not full or identity['cohort']['metadata_matches_registered_legacy_release'] is not True or
        identity['cohort']['counts']!=FULL_COUNTS or samples!=FULL_COUNTS['train']):raise ValueError('full EMGFormer budget/population incomplete')
    return result


def _historical_result(output,identity):
    path=output/'fit-result.json'
    if not path.exists():return
    raw=path.read_bytes();value=_json(path);_check_seal(value,FORMAT)
    if (value['identity_sha256']!=identity['sha256'] or value['scope']!=identity['policy']['scope'] or
        type(value['stage_complete']) is not bool or value['scientific_eligible'] is not False or value['full_paper_complete'] is not False):
        raise ValueError('prior EMGFormer result does not belong to this owned fit')
    if value['stage_complete']:
        read_emgformer_fit(output,require_full=False)
        raise ValueError('completed EMGFormer cannot be retrained/reselected')
    # An incomplete result describes an older observation, not necessarily the
    # latest checkpoint. Preserve exact bytes before reconciling durable state.
    import hashlib
    historical=output/('historical-fit-result-'+hashlib.sha256(raw).hexdigest()+'.json')
    if historical.exists():
        if historical.read_bytes()!=raw:raise ValueError('historical EMGFormer result changed')
    else:
        with historical.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def _completed_checkpoint_result(output,identity,checkpoint,device,validation_exact,recovery_started):
    import torch
    packet=checkpoint['emgformer_fit'];records=packet['records'];policy=identity['policy']
    if len(records)!=policy['max_epochs'] or not emgformer_stage_complete(policy,records):
        raise ValueError('saved EMGFormer budget is not complete')
    native=_native_callback(checkpoint,'ModelCheckpoint{');best=Path(native['best_model_path'])
    if best.parent!=output/'checkpoints' or best.name=='last.ckpt':raise ValueError('saved native best location differs')
    result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=policy['scope'],completed_epochs=len(records),
        optimizer_executions=checkpoint['global_step'],stage_complete=True,full_training_terminated=policy['scope']=='full',
        scientific_eligible=False,full_paper_complete=False,resume_start_exact=None,resume_validation_exact=validation_exact,
        completion_recovered_from_native_checkpoint=True,additional_optimizer_updates_during_recovery=0,
        final_model_state_sha256=emgformer_state_sha256(checkpoint['state_dict']),last_checkpoint_sha256=_file_hash(output/'checkpoints/last.ckpt'),
        selected_best=dict(file=best.name,score=float(native['best_model_score']),checkpoint_sha256=_file_hash(best),validated=True),
        train_samples_per_epoch=[r['train_samples'] for r in records],val_samples_per_epoch=[r['validation']['samples']for r in records],train_hand_samples_per_epoch=[r['train_hand_samples']for r in records],
        cuda_initialized=torch.cuda.is_initialized(),runtime_device=device,elapsed_fit_seconds=0.,elapsed_recovery_seconds=time.monotonic()-recovery_started,
        bitwise_training_trajectory_guaranteed=False,
        rng_restore_contract='complete saved budget independently accepted; no new training or optimizer restoration attempted',
        callback_restore_policy='selected state read from original native checkpoint'))
    _validate_fit(output,identity,result,require_full=policy['scope']=='full')
    _write(output/'fit-result.json',result,replace=True);return result


def fit_emgformer(cohort,output,*,scope='full',resume=False,stop_after_epoch=None,cohort_receipt=None):
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint
    recovery_started=time.monotonic();cohort.check();policy=emgformer_fit_policy(cohort.upstream,cohort.config,scope=scope)
    receipt_path,binding=_receipt(cohort,cohort_receipt,scope=='full')
    if scope=='full' and (cohort.identity['metadata_matches_registered_legacy_release'] is not True or cohort.identity['counts']!=FULL_COUNTS):
        raise ValueError('full EMGFormer requires complete registered legacy fold')
    if stop_after_epoch is not None and (scope=='full' or type(stop_after_epoch) is not int or not 1<=stop_after_epoch<=2):
        raise ValueError('bounded EMGFormer epoch pause is engineering only')
    device=_runtime_device(policy);output=_safe(output,exists=resume);storage._space(output.parent)
    if not resume:output.mkdir(mode=0o700,exist_ok=False)
    lock=os.open(output/'fit.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    loaders={};prepared=None;witness=None
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);checkpoint=None;identity=None
        if resume:
            identity=_json(output/'identity.json');_check_seal(identity,FORMAT)
            if (identity['policy']!=policy or identity['cohort']!=cohort.identity or identity['fit_sources_sha256']!=_sources() or
                identity['native_loop_sources_sha256']!=_loop_sources() or identity['cohort_receipt']!=binding):raise ValueError('EMGFormer resume source/config/cohort differs')
            _historical_result(output,identity)
            checkpoint=torch.load(str(output/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False);p=checkpoint['emgformer_fit']
            if p['identity']!=identity or p['epoch_boundary'] is not True:raise ValueError('owned complete EMGFormer epoch required')
            if (type(p['completed_epochs']) is not int or not 1<=p['completed_epochs']<=policy['max_epochs'] or
                len(p['records'])!=p['completed_epochs']):raise ValueError('saved EMGFormer epoch is outside owned budget')
            carry=_epoch_bytes(p['records'],identity['sha256']);path=output/'epochs.jsonl';raw=path.read_bytes()
            if not raw.startswith(carry):raise ValueError('EMGFormer checkpoint/journal prefix differs')
            if len(raw)>len(carry):
                (output/('uncommitted-epochs-'+uuid.uuid4().hex+'.jsonl')).write_bytes(raw[len(carry):])
                with path.open('wb') as f:f.write(carry);f.flush();os.fsync(f.fileno())
        prepared=build_emgformer_runtime(cohort.upstream,cohort.config,**({} if checkpoint is None else dict(
            state=checkpoint['state_dict'],expected_initialization=identity['construction'])))
        loaders={r:l for r,l in native_emgformer_loaders(cohort,cpu_engineering=scope=='cpu_engineering').items()if r in('train','val')};samples=len(cohort.datasets['train']);indices=None
        if scope!='full':
            from torch.utils.data import Subset
            from egoemg.datamodule import make_data_module
            dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=policy['workers'];dm.batch_size=policy['batch_size_by_role']['train']
            half=min(len(cohort.datasets['train'].windows),policy['engineering_train_samples']//2)
            anchors=len(cohort.datasets['train'].windows);indices=list(range(half))+list(range(anchors,anchors+half));samples=len(indices)
            loaders={r:dm._make_loader(Subset(ds,indices)if r=='train'else ds,shuffle=r=='train',num_workers=policy['workers'])
                for r,ds in cohort.datasets.items()if r in('train','val')}
        if identity is None:
            identity=_sealed(dict(format=FORMAT,owner=uuid.uuid4().hex,policy=policy,cohort=cohort.identity,cohort_receipt=binding,
                construction=prepared.identity,fit_sources_sha256=_sources(),native_loop_sources_sha256=_loop_sources(),train_samples=samples,train_indices=indices,
                scientific_eligible=False,full_paper_complete=False))
            _write(output/'identity.json',identity)
            with (output/'epochs.jsonl').open('xb'):pass
        elif identity['train_samples']!=samples or identity['train_indices']!=indices:raise ValueError('resumed EMGFormer population differs')
        validation_exact=None
        if checkpoint is not None:
            replay=_frozen_validation(prepared,cohort,policy,loaders['val'],device)
            validation_exact=replay==checkpoint['emgformer_fit']['records'][-1]['validation']
            if not validation_exact:raise ValueError('fresh EMGFormer frozen validation differs before continuation')
            if checkpoint['emgformer_fit']['completed_epochs']==policy['max_epochs']:
                _close_loaders(loaders);return _completed_checkpoint_result(output,identity,checkpoint,device,validation_exact,recovery_started)
        witness=EmgformerFitWitness(prepared,cohort,identity,output,stop_after_epoch,receipt_path)
        from egoemg._train_utils import build_callbacks,build_logger
        callbacks=build_callbacks(cohort.config,ensure_model_summary=True)+[witness]
        for cb in callbacks:
            if isinstance(cb,ModelCheckpoint):cb.dirpath=str(output/'checkpoints')
        effective=copy.deepcopy(cohort.config);effective.logger.save_dir=str(output);effective.logger.name='logs';effective.logger.version=uuid.uuid4().hex
        logger=build_logger(effective);kwargs=OmegaConf.to_container(cohort.config.trainer,resolve=True)
        kwargs.update(devices=1,accelerator=device['accelerator'],precision=policy['precision'],max_epochs=policy['max_epochs'],
            num_sanity_val_steps=policy['num_sanity_val_steps'],enable_progress_bar=False,callbacks=callbacks,logger=logger)
        trainer=storage.pl.Trainer(**kwargs);connector=trainer._checkpoint_connector;original=connector.restore_training_state
        def restore():
            value=original()
            if checkpoint is not None:
                actual=_snap(prepared.module,trainer);expected=_checkpoint_snap(checkpoint)
                witness.exact_restore={k:actual[k]==expected[k] for k in expected}
                if not all(witness.exact_restore.values()):raise ValueError('native EMGFormer six-state restore differs: '+str(witness.exact_restore))
            return value
        connector.restore_training_state=restore;started=time.monotonic()
        try:trainer.fit(prepared.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'],
            ckpt_path=None if checkpoint is None else str(output/'checkpoints/last.ckpt'))
        finally:connector.restore_training_state=original
        _close_loaders(loaders);cohort.check();witness.check();prepared.check();native=next(cb for cb in trainer.callbacks if isinstance(cb,ModelCheckpoint));best=Path(native.best_model_path)
        done=emgformer_stage_complete(policy,witness.records)
        result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,completed_epochs=witness.completed_epochs,
            optimizer_executions=witness.steps,stage_complete=done,full_training_terminated=done and scope=='full',
            scientific_eligible=False,full_paper_complete=False,resume_start_exact=witness.exact_restore,resume_validation_exact=validation_exact,
            final_model_state_sha256=emgformer_state_sha256(prepared.module.state_dict()),last_checkpoint_sha256=_file_hash(output/'checkpoints/last.ckpt'),
            selected_best=dict(file=best.name,score=float(native.best_model_score),checkpoint_sha256=_file_hash(best),validated=True),
            train_samples_per_epoch=[r['train_samples'] for r in witness.records],val_samples_per_epoch=[r['validation']['samples']for r in witness.records],train_hand_samples_per_epoch=[r['train_hand_samples']for r in witness.records],
            cuda_initialized=torch.cuda.is_initialized(),runtime_device=device,elapsed_fit_seconds=time.monotonic()-started,
            bitwise_training_trajectory_guaranteed=False,
            rng_restore_contract='exact after native restore_training_state, before iterator reconstruction; resumed global loader RNG can diverge from uninterrupted training',
            callback_restore_policy='native fields except ModelCheckpoint.current_score, not restored by native Lightning'))
        _validate_fit(output,identity,result,require_full=scope=='full' and done)
        _write(output/'fit-result.json',result,replace=resume);read_emgformer_fit(output,require_full=scope=='full' and done)
        return result
    finally:
        try:
            _close_loaders(locals().get('loaders',{}))
            if witness is not None:
                if hasattr(witness,'original_forward'):prepared.module.forward=witness.original_forward
                if hasattr(witness,'step_hook'):witness.step_hook.remove()
        finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)



def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True);p=sub.add_parser('fit')
    for name in('data','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--size',choices=('small','middle','large'),required=True)
    p.add_argument('--scope',choices=SCOPES,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-epoch',type=int)
    p.add_argument('--cohort-receipt',type=Path);args=vars(parser.parse_args());args.pop('action');data=args.pop('data');upstream=args.pop('upstream');size=args.pop('size')
    cfg=compose_emgformer(upstream,size);expected=None if args['cohort_receipt']is None else _json(args['cohort_receipt'])
    if args['scope']=='full'and expected is None:parser.error('--cohort-receipt required for complete EMGFormer training')
    if args['resume']:
        saved=_json(args['output']/'identity.json')['cohort']
        if expected is not None and expected!=saved:raise ValueError('resume frozen EMGFormer cohort differs')
        expected=saved
    cohort=admit_emgformer_cohort(data,upstream=upstream,config=cfg,normalization=emgformer_normalization(upstream,cfg),expected=expected)
    result=fit_emgformer(cohort,**args)
    print(json.dumps({k:result[k]for k in('scope','completed_epochs','optimizer_executions','stage_complete','full_training_terminated')}))


if __name__=='__main__':main()
