"""Original ViT-S/EMG fusion epochs and native complete-state recovery.

Full mode consumes the admitted complete legacy fusion fold at batch200/bf16 for150
epochs, with no early stopping. Engineering modes are explicit two-epoch
diagnostics, never paper scores or a full fusion score. Training never iterates the test role.
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
from .fusion_literal import compose_fusion,fusion_protocol
from .fusion_literal_data import fusion_normalization
from .fusion_initializer_handoff import FusionInitializers,admit_fusion_initializers,engineering_fusion_initializers
from .fusion_literal_cohort import admit_fusion_cohort,native_fusion_loaders,FULL_COUNTS
from .fusion_literal_runtime import build_fusion_runtime,fusion_optimizer_identity,fusion_state_sha256
from .vision_literal_runtime import _schema
from .vision_literal_check import _native_restored_callbacks
from .vision_literal_train import (_json,_write,_sealed,_check_seal,_read_epochs,_epoch_bytes,_native_callback,
    _adam_steps,_loop_sources,_loop_paths)

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_fusion_native_fit_v1'
SCOPES=('full','cpu_engineering','cuda_engineering')
LOCAL_SOURCES=('egoemg_repro/fusion_literal_train.py','egoemg_repro/fusion_literal_runtime.py',
    'egoemg_repro/fusion_literal_cohort.py','egoemg_repro/fusion_initializer_handoff.py','egoemg_repro/classic_emg_train.py',
    'egoemg_repro/vision_literal_runtime.py','egoemg_repro/vision_literal_train.py','egoemg_repro/vision_literal_check.py')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported fusion fit source changed')
    return {name:_file_hash(ROOT/name) for name in LOCAL_SOURCES}


def fusion_fit_policy(upstream,config,*,scope='full'):
    if type(scope) is not str or scope not in SCOPES:raise ValueError('explicit supported fusion scope required')
    protocol=fusion_protocol(upstream,config);full=scope=='full';cpu=scope=='cpu_engineering'
    callbacks=protocol['callbacks'];checkpoint=callbacks[0]
    if ([c['_target_'] for c in callbacks]!=['pytorch_lightning.callbacks.ModelCheckpoint',
        'pytorch_lightning.callbacks.LearningRateMonitor']):raise ValueError('original fusion checkpoint and LR monitor required')
    return _sealed(dict(format='original_fusion_fit_policy_v1',scope=scope,protocol_sha256=protocol['sha256'],
        max_epochs=150 if full else 2,precision='32-true' if cpu else 'bf16-mixed',
        batch_size_by_role={r:2 if cpu else 200 for r in ('train','val','test')},
        gradient_clip_val=protocol['trainer']['gradient_clip_val'],scheduler_T_max=150,scheduler_eta_min=1e-5,
        monitor=checkpoint['monitor'],monitor_mode=checkpoint['mode'],save_top_k=checkpoint['save_top_k'],save_last=checkpoint['save_last'],
        early_stopping=False,num_sanity_val_steps=2 if full else 0,engineering_train_samples=None if full else 4 if cpu else 400,
        native_global_loader_rng=True,workers=0,single_rank=True,scientific_eligible=False,full_paper_complete=False,
        logger='original TensorBoardLogger; owned output path/version only',bitwise_training_trajectory_guaranteed=False))


def fusion_stage_complete(policy,records):
    return (len(records)==policy['max_epochs'] and policy['early_stopping'] is False and
        all(math.isfinite(r['train_loss']) and math.isfinite(r['validation']['val_mae']) for r in records))


def _runtime_device(policy):
    import torch
    if policy['scope']=='cpu_engineering':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU engineering requires CUDA hidden at startup')
        return dict(accelerator='cpu',cuda_initialized=False)
    if torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise ValueError('actual single visible CUDA bf16 GPU required')
    free,total=torch.cuda.mem_get_info(0)
    if free<64*1024**3:raise ValueError('64GiB actual GPU memory reserve required before fusion fit')
    return dict(accelerator='gpu',name=torch.cuda.get_device_name(0),free_memory_before=free,total_memory=total,
        visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version())


def _receipt(cohort,path,required):
    if path is None:
        if required:raise ValueError('full fusion requires independently frozen complete numeric receipt')
        return None,None
    path=_safe(path)
    if _json(path)!=cohort.identity:raise ValueError('fusion frozen receipt differs from actual complete admission')
    return path,dict(path=str(path),file_sha256=_file_hash(path),identity_sha256=cohort.identity['sha256'])


def _check_receipt(identity):
    binding=identity['cohort_receipt']
    if binding is None:
        if identity['policy']['scope']=='full':raise ValueError('full fusion lacks frozen input receipt')
    elif (_file_hash(_safe(binding['path']))!=binding['file_sha256'] or _json(binding['path'])!=identity['cohort'] or
        binding['identity_sha256']!=identity['cohort']['sha256']):raise ValueError('fusion frozen numeric receipt changed')


def _initializer_descriptor(initializers):
    initializers.check()
    return dict(identity=copy.deepcopy(initializers.identity),
        files={name:str(value) if isinstance(value,Path) else value for name,value in initializers.runtime_kwargs.items()},
        producers=copy.deepcopy(initializers.producer_sources))


def _data_locations(cohort):
    return dict(data=str(cohort.datasets['train'].store.root),crops=str(cohort.datasets['train'].views[0].crops),
        upstream=str(cohort.upstream))


def _verify_recorded_inputs(identity,cohort=None,initializers=None,*,require_full):
    """Recompute data and completed producer evidence when reopening a fit.

    Only the in-progress owning fit can supply its already admitted live
    objects; the public read always performs a fresh full content admission.
    """
    from omegaconf import OmegaConf
    if (cohort is None)!=(initializers is None):raise ValueError('both live admitted fusion inputs required together')
    descriptor=identity['initializer_descriptor'];locations=identity['data_locations']
    if cohort is None:
        cfg=OmegaConf.create(identity['saved_config'])
        cohort=admit_fusion_cohort(locations['data'],locations['crops'],upstream=locations['upstream'],config=cfg,
            normalization=fusion_normalization(locations['upstream']),expected=identity['cohort'])
        saved=descriptor['identity'];producers=descriptor['producers']
        if saved['scope']=='full':
            if not isinstance(producers,dict) or set(producers)!={'vision_fit','vision_source_root','emg_fit','emg_export'}:
                raise ValueError('full fusion requires actual producer source locators')
            initializers=admit_fusion_initializers(cohort,**producers)
        elif saved['scope']=='engineering':
            if producers is not None:raise ValueError('engineering fusion cannot claim full producer locators')
            initializers=engineering_fusion_initializers(cohort,**descriptor['files'])
        else:raise ValueError('unknown fusion initializer scope')
    cohort.check();initializers.check(require_full=require_full)
    if (cohort.identity!=identity['cohort'] or _data_locations(cohort)!=locations or
        _initializer_descriptor(initializers)!=descriptor or initializers._cohort is not cohort or
        fusion_fit_policy(cohort.upstream,cohort.config,scope=identity['policy']['scope'])!=identity['policy'] or
        OmegaConf.to_container(cohort.config,resolve=True)!=identity['saved_config']):
        raise ValueError('fusion actual data/producer/config binding differs from saved fit')
    init=identity['construction']['initialization']
    for role in ('vision','emg'):
        if init[role+'_file']['sha256']!=descriptor['identity'][role]['checkpoint_sha256']:
            raise ValueError('native initial branch tensor receipt differs from verified producer file')
    expected=len(cohort.datasets['train'])
    if identity['policy']['scope']!='full':expected=min(expected,identity['policy']['engineering_train_samples'])
    if identity['train_samples']!=expected:raise ValueError('fusion actual training population differs')
    return cohort,initializers


def _snap(module,trainer):
    d=storage.state_digest
    return dict(model=d(module.state_dict()),optimizer=d([o.state_dict() for o in trainer.optimizers]),
        scheduler=d([x.scheduler.state_dict() for x in trainer.lr_scheduler_configs]),
        callbacks=d(_native_restored_callbacks({cb.state_key:cb.state_dict() for cb in trainer.callbacks if cb.state_dict()})),
        loops=d(trainer.fit_loop.state_dict()),rng=d(storage._rng({})))


def _checkpoint_snap(c):
    d=storage.state_digest
    return dict(model=d(c['state_dict']),optimizer=d(c['optimizer_states']),scheduler=d(c['lr_schedulers']),
        callbacks=d(_native_restored_callbacks(c['callbacks'])),loops=d(c['loops']['fit_loop']),rng=d(c['fusion_fit']['rng']))


def _batch(cohort,policy,batch,role,seen,limit):
    import torch
    ids=batch['fusion_sample_index']
    if ids.dtype!=torch.int64 or ids.ndim!=1:raise ValueError('actual fusion consumption indices required')
    ids=ids.detach().cpu().numpy();n=len(ids);size=policy['batch_size_by_role'][role]
    if (n!=min(size,limit-len(seen)) or np.any(ids<0) or np.any(ids>=limit) or len(np.unique(ids))!=n or
        any(int(i) in seen for i in ids)):raise ValueError('fusion batch is repeated, partial or out of range')
    if role=='val' and ids.tolist()!=list(range(len(seen),len(seen)+n)):raise ValueError('native fusion validation order differs')
    if (batch['emg'].shape!=(n,16,7790) or batch['vision_img'].shape!=(n,3,256,256) or
        batch['joint_angles'].shape!=(n,22,1) or batch['label_valid_mask'].shape!=(n,1) or
        batch['label_valid_mask'].dtype!=torch.bool or batch['vision_valid_mask'].shape!=(n,1) or
        batch['vision_valid_mask'].dtype!=torch.bool or
        any(not bool(torch.isfinite(batch[k]).all()) for k in ('emg','vision_img','joint_angles'))):
        raise ValueError('fusion complete input/center target/masks malformed or nonfinite')
    ds=cohort.datasets[role];anchors=len(ds.windows)
    for j,index in enumerate(ids):
        unique=int(index)%(2*anchors);repeat=int(index)//(2*anchors);hand,window=divmod(unique,anchors)
        w=ds.windows[window];offset=int(batch['jitter_offset'][j])
        if (batch['role'][j]!=role or batch['split'][j]!=w['split'] or batch['hand'][j]!=('left','right')[hand] or
            int(batch['fusion_repeat_index'][j])!=repeat or int(batch['fusion_unique_sample_index'][j])!=unique or
            int(batch['fusion_window_index'][j])!=window or not 0<=offset<=w['jitter_max_offset'] or
            int(batch['start'][j])!=w['start']+offset or int(batch['end'][j])!=w['end']+offset or
            int(batch['center'][j])!=w['center']+offset or int(batch['episode'][j])!=w['episode'] or
            batch['frame_mapping_field'][j]!=ds.frame_mapping_field):
            raise ValueError('fusion original anchor/jitter/hand/repeat binding differs')
    seen.update(map(int,ids));return n


class ValidationWitness(storage.Callback):
    """Digest one entire original validation pass without retaining predictions."""
    def __init__(self,cohort,policy):self.cohort=cohort;self.policy=policy;self.phase=None;self.validation=None

    def _observe_forward(self,module):
        self.original_forward=module.forward
        def forward(batch):
            out=self.original_forward(batch)
            if self.phase=='val':
                import torch
                pred,target,mask=out
                if (pred.shape!=target.shape or pred.shape[1:]!=(22,1) or mask.shape!=(len(pred),1) or
                    not bool(torch.isfinite(pred).all())):raise ValueError('original fusion validation trajectory differs/nonfinite')
                self.parts.append(storage.state_digest(dict(pred=pred,target=target,mask=mask,emg=batch['emg'],
                    sample_index=batch['fusion_sample_index'],vision_img=batch['vision_img'],vision_valid_mask=batch['vision_valid_mask'])))
            return out
        module.forward=forward

    def on_validation_start(self,trainer,module):
        if not hasattr(self,'original_forward'):self._observe_forward(module)

    def on_validation_epoch_start(self,trainer,module):self.parts=[];self.val_seen=set()

    def on_validation_batch_start(self,trainer,module,batch,batch_idx,dataloader_idx=0):
        if not trainer.sanity_checking:
            self.cohort.guard.check();self.phase='val'
            _batch(self.cohort,self.policy,batch,'val',self.val_seen,len(self.cohort.datasets['val']))

    def on_validation_end(self,trainer,module):
        self.phase=None
        if trainer.sanity_checking:return
        score=float(trainer.callback_metrics['val_mae'])
        if len(self.val_seen)!=len(self.cohort.datasets['val']) or not math.isfinite(score):raise ValueError('incomplete/nonfinite fusion validation')
        self.validation=dict(samples=len(self.val_seen),val_mae=score,prediction_sha256=_hash(self.parts))
        self.parts=[]


class FusionFitWitness(ValidationWitness):
    def __init__(self,prepared,cohort,initializers,identity,output,stop_after,receipt_path):
        super().__init__(cohort,identity['policy']);self.prepared=prepared;self.initializers=initializers;self.identity=identity;self.output=output
        self.stop_after=stop_after;self.steps=0;self.completed_epochs=0;self.records=[];self.pending=None;self.exact_restore=None
        self.source_guard=_StatGuard([*[ROOT/name for name in identity['fit_sources_sha256']],*_loop_paths().values(),
            *([] if receipt_path is None else [receipt_path])])

    def state_dict(self):return dict(steps=self.steps,completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records))

    def load_state_dict(self,state):
        if set(state)!={'steps','completed_epochs','records'}:raise ValueError('fusion callback schema differs')
        self.steps=state['steps'];self.completed_epochs=state['completed_epochs'];self.records=copy.deepcopy(state['records'])

    def check(self):self.cohort.guard.check();self.initializers._guard.check();self.source_guard.check();storage._space(self.output)

    def on_load_checkpoint(self,trainer,module,checkpoint):
        p=checkpoint['fusion_fit']
        if p['identity']!=self.identity or p['epoch_boundary'] is not True:raise ValueError('owned complete fusion epoch required')
        self.pending=checkpoint

    def on_fit_start(self,trainer,module):
        if self.pending is not None:storage._restore_rng(self.pending['fusion_fit']['rng'],{})
        self._observe_forward(module)
        self.step_hook=trainer.optimizers[0].register_step_post_hook(lambda *args:self._executed())

    def _executed(self):self.steps+=1

    def on_train_start(self,trainer,module):
        fusion_optimizer_identity(self.prepared,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
        if (getattr(trainer.precision_plugin,'scaler',None) is not None or trainer.gradient_clip_val!=1 or
            trainer.accumulate_grad_batches!=1):raise ValueError('native fusion precision/clip/accumulation differs')

    def on_train_epoch_start(self,trainer,module):
        self.check()
        if trainer.current_epoch!=self.completed_epochs:raise ValueError('resumed fusion epoch differs from complete prefix')
        self.train_seen=set();self.train_loss_sum=0.;self.train_batches=0;self.validation=None

    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        self.check();self.phase='train'
        self.batch_n=_batch(self.cohort,self.policy,batch,'train',self.train_seen,self.identity['train_samples'])

    def on_before_optimizer_step(self,trainer,module,optimizer):
        import torch
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):raise ValueError('fusion gradients absent/nonfinite')

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        loss=float(outputs['loss'])
        if not math.isfinite(loss) or self.steps!=trainer.global_step:raise ValueError('native fusion loss/update count differs')
        self.train_loss_sum+=self.batch_n*loss;self.train_batches+=1

    def on_train_epoch_end(self,trainer,module):
        self.check();samples=self.identity['train_samples'];batches=math.ceil(samples/self.policy['batch_size_by_role']['train'])
        if (len(self.train_seen)!=samples or self.validation is None or self.train_batches!=batches or
            self.steps!=(self.completed_epochs+1)*batches):raise ValueError('partial fusion epoch cannot be committed')
        _adam_steps(trainer.optimizers[0],self.steps)
        record=dict(epoch=self.completed_epochs,optimizer_executions=self.steps,global_step=trainer.global_step,
            train_samples=samples,train_batches=batches,train_loss=self.train_loss_sum/samples,
            validation=copy.deepcopy(self.validation),model_sha256=fusion_state_sha256(module.state_dict()))
        previous=_epoch_bytes(self.records,self.identity['sha256']);self.records.append(record);self.completed_epochs+=1
        raw=_epoch_bytes(self.records,self.identity['sha256']);path=self.output/'epochs.jsonl'
        if len(raw)>2*1024**2 or path.read_bytes()!=previous:raise ValueError('fusion epoch journal prefix/bound differs')
        with path.open('ab') as f:f.write(raw[len(previous):]);f.flush();os.fsync(f.fileno())
        if self.stop_after==self.completed_epochs:trainer.should_stop=True

    def on_save_checkpoint(self,trainer,module,checkpoint):
        self.check();checkpoint['fusion_fit']=dict(format=FORMAT,identity=self.identity,
            epoch_boundary=self.validation is not None,completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records),rng=storage._rng({}))

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


def read_fusion_fit(output,*,require_full=True):
    output=_safe(output);identity=_json(output/'identity.json');result=_json(output/'fit-result.json')
    return _validate_fit(output,identity,result,require_full=require_full)


def _validate_fit(output,identity,result,*,require_full,cohort=None,initializers=None):
    """Validate a candidate against native durable files before publishing it."""
    import torch
    _check_seal(identity,FORMAT);_check_seal(result,FORMAT);_check_seal(identity['policy'],'original_fusion_fit_policy_v1')
    if (result['identity_sha256']!=identity['sha256'] or identity['fit_sources_sha256']!=_sources() or
        identity['native_loop_sources_sha256']!=_loop_sources() or result['scientific_eligible'] is not False or
        result['full_paper_complete'] is not False):raise ValueError('fusion fit/source/result identity differs')
    _check_receipt(identity);policy=identity['policy']
    if result['scope']!=policy['scope'] or (require_full and policy['scope']!='full'):raise ValueError('engineering fusion is not complete full training')
    _verify_recorded_inputs(identity,cohort,initializers,require_full=policy['scope']=='full')
    last_path=output/'checkpoints/last.ckpt'
    if _file_hash(last_path)!=result['last_checkpoint_sha256']:raise ValueError('fusion last checkpoint bytes changed')
    last=torch.load(str(last_path),map_location='cpu',weights_only=False);packet=last['fusion_fit']
    records,_=_read_epochs(output/'epochs.jsonl',identity['sha256']);samples=identity['train_samples']
    batches=math.ceil(samples/policy['batch_size_by_role']['train']);steps=batches*len(records)
    if (not records or packet['identity']!=identity or packet['records']!=records or packet['epoch_boundary'] is not True or
        packet['completed_epochs']!=len(records) or result['completed_epochs']!=len(records) or
        result['optimizer_executions']!=steps or last['global_step']!=steps):raise ValueError('fusion actual epoch/optimizer/journal carry differs')
    for i,r in enumerate(records):
        if (r['train_samples']!=samples or r['train_batches']!=batches or r['optimizer_executions']!=(i+1)*batches or
            r['global_step']!=(i+1)*batches or r['validation']['samples']!=identity['cohort']['counts']['val'] or
            not math.isfinite(r['train_loss']) or not math.isfinite(r['validation']['val_mae'])):
            raise ValueError('partial/nonfinite fusion epoch')
    if any(k.startswith('EarlyStopping') for k in last['callbacks']):raise ValueError('fusion cannot use early stopping')
    if (_schema(last['state_dict'])!=identity['construction']['model_schema'] or
        fusion_state_sha256(last['state_dict'])!=records[-1]['model_sha256'] or
        result['final_model_state_sha256']!=records[-1]['model_sha256']):raise ValueError('terminal fusion tensor state differs')
    native=_native_callback(last,'ModelCheckpoint{');best=Path(native['best_model_path']);score=float(native['best_model_score'])
    if (best.parent!=output/'checkpoints' or best.is_symlink() or best.name=='last.ckpt' or
        result['selected_best']!=dict(file=best.name,score=score,checkpoint_sha256=_file_hash(best),validated=True) or
        score!=min(r['validation']['val_mae'] for r in records)):raise ValueError('fusion selected checkpoint differs from actual val best')
    selected=torch.load(str(best),map_location='cpu',weights_only=False);p=selected['fusion_fit'];epoch=p['completed_epochs']-1
    if (p['identity']!=identity or p['epoch_boundary'] is not True or not 0<=epoch<len(records) or
        p['records']!=records[:epoch+1] or records[epoch]['validation']['val_mae']!=score or
        selected['global_step']!=records[epoch]['global_step'] or _schema(selected['state_dict'])!=identity['construction']['model_schema'] or
        fusion_state_sha256(selected['state_dict'])!=records[epoch]['model_sha256']):raise ValueError('selected fusion state/epoch differs')
    done=fusion_stage_complete(policy,records);full=done and policy['scope']=='full'
    if (result['stage_complete'] is not done or result['full_training_terminated'] is not full or
        result['train_samples_per_epoch']!=[r['train_samples'] for r in records] or
        result['val_samples_per_epoch']!=[r['validation']['samples'] for r in records]):raise ValueError('fusion completion/coverage differs')
    if require_full and (not full or identity['cohort']['metadata_matches_registered_legacy_release'] is not True or
        identity['cohort']['counts']!=FULL_COUNTS or samples!=FULL_COUNTS['train']):raise ValueError('full fusion budget/population incomplete')
    return result


def _historical_result(output,identity):
    path=output/'fit-result.json'
    if not path.exists():return
    raw=path.read_bytes();value=_json(path);_check_seal(value,FORMAT)
    if (value['identity_sha256']!=identity['sha256'] or value['scope']!=identity['policy']['scope'] or
        type(value['stage_complete']) is not bool or value['scientific_eligible'] is not False or value['full_paper_complete'] is not False):
        raise ValueError('prior fusion result does not belong to this owned fit')
    if value['stage_complete']:
        read_fusion_fit(output,require_full=False)
        raise ValueError('completed fusion cannot be retrained/reselected')
    # An incomplete result describes an older observation, not necessarily the
    # latest checkpoint. Preserve exact bytes before reconciling durable state.
    import hashlib
    historical=output/('historical-fit-result-'+hashlib.sha256(raw).hexdigest()+'.json')
    if historical.exists():
        if historical.read_bytes()!=raw:raise ValueError('historical fusion result changed')
    else:
        with historical.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def _completed_checkpoint_result(output,identity,checkpoint,device,validation_exact,recovery_started):
    import torch
    packet=checkpoint['fusion_fit'];records=packet['records'];policy=identity['policy']
    if len(records)!=policy['max_epochs'] or not fusion_stage_complete(policy,records):
        raise ValueError('saved fusion budget is not complete')
    native=_native_callback(checkpoint,'ModelCheckpoint{');best=Path(native['best_model_path'])
    if best.parent!=output/'checkpoints' or best.name=='last.ckpt':raise ValueError('saved native best location differs')
    result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=policy['scope'],completed_epochs=len(records),
        optimizer_executions=checkpoint['global_step'],stage_complete=True,full_training_terminated=policy['scope']=='full',
        scientific_eligible=False,full_paper_complete=False,resume_start_exact=None,resume_validation_exact=validation_exact,
        completion_recovered_from_native_checkpoint=True,additional_optimizer_updates_during_recovery=0,
        final_model_state_sha256=fusion_state_sha256(checkpoint['state_dict']),last_checkpoint_sha256=_file_hash(output/'checkpoints/last.ckpt'),
        selected_best=dict(file=best.name,score=float(native['best_model_score']),checkpoint_sha256=_file_hash(best),validated=True),
        train_samples_per_epoch=[r['train_samples'] for r in records],val_samples_per_epoch=[r['validation']['samples'] for r in records],
        cuda_initialized=torch.cuda.is_initialized(),runtime_device=device,elapsed_fit_seconds=0.,elapsed_recovery_seconds=time.monotonic()-recovery_started,
        bitwise_training_trajectory_guaranteed=False,
        rng_restore_contract='complete saved budget independently accepted; no new training or optimizer restoration attempted',
        callback_restore_policy='selected state read from original native checkpoint'))
    _validate_fit(output,identity,result,require_full=policy['scope']=='full')
    _write(output/'fit-result.json',result,replace=True);return result


def fit_fusion(cohort,initializers,output,*,scope='full',resume=False,stop_after_epoch=None,cohort_receipt=None):
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint
    recovery_started=time.monotonic();cohort.check();policy=fusion_fit_policy(cohort.upstream,cohort.config,scope=scope)
    if type(initializers) is not FusionInitializers:raise ValueError('actual verified fusion initializer handoff required')
    initializers.check(require_full=scope=='full')
    if initializers._cohort is not cohort:raise ValueError('fusion initializers bind a different admitted cohort')
    descriptor=_initializer_descriptor(initializers)
    receipt_path,binding=_receipt(cohort,cohort_receipt,scope=='full')
    if scope=='full' and (cohort.identity['metadata_matches_registered_legacy_release'] is not True or cohort.identity['counts']!=FULL_COUNTS):
        raise ValueError('full fusion requires complete registered legacy fold')
    if stop_after_epoch is not None and (scope=='full' or type(stop_after_epoch) is not int or not 1<=stop_after_epoch<=2):
        raise ValueError('bounded fusion epoch pause is engineering only')
    device=_runtime_device(policy);output=_safe(output,exists=resume);storage._space(output.parent)
    if not resume:output.mkdir(mode=0o700,exist_ok=False)
    lock=os.open(output/'fit.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);checkpoint=None;identity=None
        if resume:
            identity=_json(output/'identity.json');_check_seal(identity,FORMAT)
            if (identity['policy']!=policy or identity['cohort']!=cohort.identity or identity['fit_sources_sha256']!=_sources() or
                identity['native_loop_sources_sha256']!=_loop_sources() or identity['cohort_receipt']!=binding or
                identity['initializer_descriptor']!=descriptor):raise ValueError('fusion resume source/config/cohort differs')
            _historical_result(output,identity)
            checkpoint=torch.load(str(output/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False);p=checkpoint['fusion_fit']
            if p['identity']!=identity or p['epoch_boundary'] is not True:raise ValueError('owned complete fusion epoch required')
            if (type(p['completed_epochs']) is not int or not 1<=p['completed_epochs']<=policy['max_epochs'] or
                len(p['records'])!=p['completed_epochs']):raise ValueError('saved fusion epoch is outside owned budget')
            carry=_epoch_bytes(p['records'],identity['sha256']);path=output/'epochs.jsonl';raw=path.read_bytes()
            if not raw.startswith(carry):raise ValueError('fusion checkpoint/journal prefix differs')
            if len(raw)>len(carry):
                (output/('uncommitted-epochs-'+uuid.uuid4().hex+'.jsonl')).write_bytes(raw[len(carry):])
                with path.open('wb') as f:f.write(carry);f.flush();os.fsync(f.fileno())
        prepared=build_fusion_runtime(cohort.upstream,cohort.config,**(initializers.runtime_kwargs if checkpoint is None else dict(
            state=checkpoint['state_dict'],expected_initialization=identity['construction'])))
        loaders=native_fusion_loaders(cohort);samples=len(cohort.datasets['train'])
        if scope!='full':
            from torch.utils.data import Subset
            from egoemg.datamodule import make_data_module
            dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0;dm.batch_size=policy['batch_size_by_role']['train']
            samples=min(samples,policy['engineering_train_samples'])
            loaders={r:dm._make_loader(Subset(ds,list(range(samples))) if r=='train' else ds,shuffle=r=='train') for r,ds in cohort.datasets.items()}
        if identity is None:
            identity=_sealed(dict(format=FORMAT,owner=uuid.uuid4().hex,policy=policy,cohort=cohort.identity,cohort_receipt=binding,
                construction=prepared.identity,initializer_descriptor=descriptor,data_locations=_data_locations(cohort),
                saved_config=OmegaConf.to_container(cohort.config,resolve=True),
                fit_sources_sha256=_sources(),native_loop_sources_sha256=_loop_sources(),train_samples=samples,
                scientific_eligible=False,full_paper_complete=False))
            _write(output/'identity.json',identity)
            with (output/'epochs.jsonl').open('xb'):pass
        elif identity['train_samples']!=samples:raise ValueError('resumed fusion population differs')
        validation_exact=None
        if checkpoint is not None:
            replay=_frozen_validation(prepared,cohort,policy,loaders['val'],device)
            validation_exact=replay==checkpoint['fusion_fit']['records'][-1]['validation']
            if not validation_exact:raise ValueError('fresh fusion frozen validation differs before continuation')
            if checkpoint['fusion_fit']['completed_epochs']==policy['max_epochs']:
                return _completed_checkpoint_result(output,identity,checkpoint,device,validation_exact,recovery_started)
        witness=FusionFitWitness(prepared,cohort,initializers,identity,output,stop_after_epoch,receipt_path)
        from egoemg._train_utils import build_callbacks,build_logger
        callbacks=build_callbacks(cohort.config)+[witness]
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
                if not all(witness.exact_restore.values()):raise ValueError('native fusion six-state restore differs: '+str(witness.exact_restore))
            return value
        connector.restore_training_state=restore;started=time.monotonic()
        try:trainer.fit(prepared.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'],
            ckpt_path=None if checkpoint is None else str(output/'checkpoints/last.ckpt'))
        finally:connector.restore_training_state=original
        cohort.check();witness.check();prepared.check();native=next(cb for cb in trainer.callbacks if isinstance(cb,ModelCheckpoint));best=Path(native.best_model_path)
        done=fusion_stage_complete(policy,witness.records)
        result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,completed_epochs=witness.completed_epochs,
            optimizer_executions=witness.steps,stage_complete=done,full_training_terminated=done and scope=='full',
            scientific_eligible=False,full_paper_complete=False,resume_start_exact=witness.exact_restore,resume_validation_exact=validation_exact,
            final_model_state_sha256=fusion_state_sha256(prepared.module.state_dict()),last_checkpoint_sha256=_file_hash(output/'checkpoints/last.ckpt'),
            selected_best=dict(file=best.name,score=float(native.best_model_score),checkpoint_sha256=_file_hash(best),validated=True),
            train_samples_per_epoch=[r['train_samples'] for r in witness.records],val_samples_per_epoch=[r['validation']['samples'] for r in witness.records],
            cuda_initialized=torch.cuda.is_initialized(),runtime_device=device,elapsed_fit_seconds=time.monotonic()-started,
            bitwise_training_trajectory_guaranteed=False,
            rng_restore_contract='exact after native restore_training_state, before iterator reconstruction; resumed global loader RNG can diverge from uninterrupted training',
            callback_restore_policy='native fields except ModelCheckpoint.current_score, not restored by native Lightning'))
        _validate_fit(output,identity,result,require_full=scope=='full' and done,cohort=cohort,initializers=initializers)
        _write(output/'fit-result.json',result,replace=resume);read_fusion_fit(output,require_full=scope=='full' and done)
        return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)

