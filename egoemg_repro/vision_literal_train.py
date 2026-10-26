"""Original vision fit with complete epoch coverage and native epoch restoration.

Full mode uses genuine CUDA bf16, original150/patience30/360 and a complete
registered legacy input receipt. Engineering modes are explicit bounded
diagnostics. Neither a completed fit nor an engineering run is a final model
score: selected-best evaluation and independent reporting are separate stages.
Run the reviewed GPU capacity/restore protocol before starting full mode.
All checkpoint inputs must be trusted local artifacts.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from . import classic_emg_train as storage
from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _StatGuard,_safe
from .vision_literal import compose_vision,vision_protocol
from .vision_literal_cohort import admit_vision_cohort,native_vision_loaders,FULL_COUNTS
from .vision_literal_runtime import build_vision_runtime,vision_optimizer_identity,vision_state_sha256,_schema
from .vision_literal_check import _native_restored_callbacks


FORMAT='original_vision_native_fit_v1'
ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
SCOPES=('full','cpu_engineering','cuda_engineering')
Callback=storage.Callback
MAX_EPOCH_BYTES=2*1024**2
LOOP_MODULES=('pytorch_lightning.loops.fit_loop','pytorch_lightning.loops.training_epoch_loop',
    'pytorch_lightning.loops.fetchers','pytorch_lightning.trainer.connectors.checkpoint_connector',
    'torch.utils.data.dataloader','torch.utils.data.sampler','torch.utils.checkpoint',
    'timm.models._manipulate','timm.layers.config')


def _loop_paths():return {name:Path(importlib.import_module(name).__file__).resolve() for name in LOOP_MODULES}


def _loop_sources():return {name:_file_hash(path) for name,path in _loop_paths().items()}


def _native_checkpoint_use_reentrant():
    from timm.models._manipulate import use_reentrant_ckpt
    value=use_reentrant_ckpt()
    if type(value) is not bool:raise ValueError('native checkpoint reentrant choice must be boolean')
    return value


def _configure_activation_checkpointing(prepared,enabled):
    from timm.models.vision_transformer import VisionTransformer
    if type(enabled) is not bool:raise ValueError('activation checkpointing requires explicit boolean')
    backbone=prepared.module.model.backbone
    if type(backbone) is not VisionTransformer:raise ValueError('native DINO VisionTransformer required for checkpointing')
    backbone.set_grad_checkpointing(enabled)
    if backbone.grad_checkpointing is not enabled:raise ValueError('native activation checkpoint switch was not applied')


def _check_activation_checkpointing(prepared,policy):
    if prepared.module.model.backbone.grad_checkpointing is not policy['activation_checkpointing']:
        raise ValueError('native activation checkpoint switch changed during fit')
    if policy['activation_checkpointing'] and _native_checkpoint_use_reentrant()!=policy['activation_checkpoint_use_reentrant']:
        raise ValueError('native activation checkpoint reentrant choice changed during fit')


def vision_fit_policy(upstream,config,*,scope='full',activation_checkpointing=False):
    if type(scope) is not str or scope not in SCOPES:raise ValueError('explicit supported vision fit scope required')
    if type(activation_checkpointing) is not bool:raise ValueError('activation checkpointing requires explicit boolean')
    p=vision_protocol(upstream,config);full=scope=='full';cpu=scope=='cpu_engineering'
    checkpoint,_,early=p['callbacks']
    result=dict(format='original_vision_fit_policy_v1',scope=scope,variant=p['variant'],protocol_sha256=p['sha256'],
        max_epochs=150 if full else 2,precision='32-true' if cpu else 'bf16-mixed',
        batch_size_by_role={role:2 if cpu else 360 for role in ('train','val','test')},
        gradient_clip_val=p['trainer']['gradient_clip_val'],scheduler_T_max=p['lr_scheduler']['scheduler']['T_max'],
        scheduler_eta_min=p['lr_scheduler']['scheduler']['eta_min'],monitor=checkpoint['monitor'],monitor_mode=checkpoint['mode'],
        patience=early['patience'],save_top_k=checkpoint['save_top_k'],save_last=checkpoint['save_last'],
        num_sanity_val_steps=2 if full else 0,engineering_train_samples=None if full else (4 if cpu else 720),
        native_global_loader_rng=True,workers=0,single_rank=True,scientific_eligible=False,full_paper_complete=False,
        activation_checkpointing=activation_checkpointing,
        activation_checkpoint_use_reentrant=_native_checkpoint_use_reentrant() if activation_checkpointing else None,
        performance_is_paper_gpu_hours=False,bitwise_training_trajectory_guaranteed=False)
    result['sha256']=_hash(result);return result


def _json(path):
    return json.loads(_safe(path).read_text(),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _write(path,value,*,replace=False):
    path=Path(path);raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    tmp=path.parent/('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    try:
        if replace:os.replace(tmp,path)
        else:os.link(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def _sealed(value):
    # Native pretrained metadata contains tuples. Normalize once before sealing
    # so JSON files and native torch checkpoint packets carry identical types.
    result=json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
    result['sha256']=_hash(result);return result


def _check_seal(value,format):
    if (not isinstance(value,dict) or value.get('format')!=format or
        value.get('sha256')!=_hash({k:v for k,v in value.items() if k!='sha256'})):
        raise ValueError('vision fit document seal/format differs')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported vision fit source changed')
    names=('egoemg_repro/vision_literal_train.py','egoemg_repro/vision_literal_cohort.py',
        'egoemg_repro/classic_emg_train.py','egoemg_repro/vision_literal_check.py')
    return {name:_file_hash(ROOT/name) for name in names}


def _space(path):
    if shutil.disk_usage(path).free<15*1024**3:raise OSError('15GiB reserve required for owned vision fit')


def _runtime_device(policy):
    import torch
    if policy['scope']=='cpu_engineering':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scope requires CUDA_VISIBLE_DEVICES empty at startup')
        return dict(accelerator='cpu',cuda_initialized=False)
    if torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise ValueError('actual single visible CUDA bf16 device required')
    free,total=torch.cuda.mem_get_info(0)
    if free<64*1024**3:raise ValueError('64GiB free GPU memory required before original vision fit')
    prop=torch.cuda.get_device_properties(0)
    return dict(accelerator='gpu',name=prop.name,total_memory=total,free_memory_before=free,
        visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),cuda_initialized=torch.cuda.is_initialized())


def _read_epochs(path,identity_sha):
    raw=Path(path).read_bytes()
    if len(raw)>MAX_EPOCH_BYTES or (raw and not raw.endswith(b'\n')):raise ValueError('bounded complete vision epoch journal required')
    records=[];previous='0'*64
    for line in raw.splitlines():
        envelope=json.loads(line);record=envelope['record']
        expected=dict(previous_sha256=previous,identity_sha256=identity_sha,record=record)
        digest=_hash(expected)
        if envelope!={**expected,'sha256':digest} or type(record['epoch']) is not int or record['epoch']!=len(records):
            raise ValueError('vision epoch journal chain/order differs')
        records.append(record);previous=digest
    return records,previous


def _epoch_bytes(records,identity_sha):
    previous='0'*64;lines=[]
    for record in records:
        body=dict(previous_sha256=previous,identity_sha256=identity_sha,record=record);previous=_hash(body)
        lines.append(json.dumps({**body,'sha256':previous},sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    return ''.join(lines).encode()


def _callbacks(trainer):return {cb.state_key:cb.state_dict() for cb in trainer.callbacks if cb.state_dict()}


def _snap(module,trainer):
    d=storage.state_digest
    return dict(model=d(module.state_dict()),optimizer=d([o.state_dict() for o in trainer.optimizers]),
        scheduler=d([v.scheduler.state_dict() for v in trainer.lr_scheduler_configs]),
        callbacks=d(_native_restored_callbacks(_callbacks(trainer))),loops=d(trainer.fit_loop.state_dict()),rng=d(storage._rng({})))


def _checkpoint_snap(checkpoint):
    d=storage.state_digest
    return dict(model=d(checkpoint['state_dict']),optimizer=d(checkpoint['optimizer_states']),scheduler=d(checkpoint['lr_schedulers']),
        callbacks=d(_native_restored_callbacks(checkpoint['callbacks'])),loops=d(checkpoint['loops']['fit_loop']),
        rng=d(checkpoint['vision_fit']['rng']))


def _adam_steps(optimizer,expected):
    import torch
    for group in optimizer.param_groups:
        for p in group['params']:
            value=optimizer.state.get(p,{}).get('step')
            if not isinstance(value,torch.Tensor) or value.numel()!=1 or not bool(torch.isfinite(value)) or float(value)!=expected:
                raise ValueError('actual native AdamW moment step counter differs')


def vision_stage_complete(policy,records,early):
    """Natural original-budget completion, never an external/premature stop."""
    n=len(records)
    if not 0<n<=policy['max_epochs'] or not math.isfinite(float(early['best_score'])):
        return False
    if any(not math.isfinite(r['train_loss']) or not math.isfinite(r['validation']['val_mae']) for r in records):
        return False
    if policy['scope']!='full':return n==policy['max_epochs']
    if policy['max_epochs']!=150 or policy['patience']!=30:return False
    return n==150 or (n>=31 and type(early['wait_count']) is int and early['wait_count']>=30 and
        type(early['stopped_epoch']) is int and early['stopped_epoch']==n-1)


def _cohort_receipt(cohort,path,*,required):
    if path is None:
        if required:raise ValueError('full fit requires an independently frozen complete legacy cohort receipt')
        return None,None
    path=_safe(path);frozen=_json(path)
    if frozen!=cohort.identity:raise ValueError('frozen complete legacy cohort receipt differs from actual rehashed inputs')
    return path,dict(path=str(path),file_sha256=_file_hash(path),identity_sha256=frozen['sha256'])


def _verify_cohort_receipt(identity):
    binding=identity['cohort_receipt']
    if binding is None:
        if identity['policy']['scope']=='full':raise ValueError('full native fit lacks frozen cohort receipt binding')
        return
    if (_file_hash(_safe(binding['path']))!=binding['file_sha256'] or
        _json(binding['path'])!=identity['cohort'] or binding['identity_sha256']!=identity['cohort']['sha256']):
        raise ValueError('frozen complete legacy cohort receipt bytes or identity changed')


class FitWitness(Callback):
    def __init__(self,prepared,cohort,identity,output,train_samples,stop_after_epoch,receipt_path=None):
        self.prepared=prepared;self.cohort=cohort;self.identity=identity;self.output=output
        self.train_samples=train_samples;self.stop_after=stop_after_epoch;self.steps=0;self.records=[]
        self.completed_epochs=0;self.pending=None;self.exact_restore=None;self.phase=None;self.validation=None
        self.source_guard=_StatGuard([*[ROOT/name for name in identity['fit_sources_sha256']],*_loop_paths().values(),
            *([] if receipt_path is None else [receipt_path])])

    def state_dict(self):return dict(steps=self.steps,completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records))

    def load_state_dict(self,state):
        if set(state)!={'steps','completed_epochs','records'}:raise ValueError('native vision witness callback schema differs')
        self.steps=state['steps'];self.completed_epochs=state['completed_epochs'];self.records=copy.deepcopy(state['records'])

    def check(self):
        _check_activation_checkpointing(self.prepared,self.identity['policy'])
        self.cohort.guard.check();self.source_guard.check();_space(self.output)

    def on_load_checkpoint(self,trainer,module,checkpoint):
        packet=checkpoint['vision_fit']
        if packet['identity']!=self.identity or packet['epoch_boundary'] is not True:
            raise ValueError('only owned complete epoch vision checkpoints may resume')
        self.pending=checkpoint

    def on_fit_start(self,trainer,module):
        if self.pending is not None:storage._restore_rng(self.pending['vision_fit']['rng'],{})
        self.original_forward=module.forward
        def observed(batch):
            out=self.original_forward(batch)
            if self.phase=='val' and not trainer.sanity_checking:
                pred,target,mask=out
                import torch
                if pred.shape!=target.shape or pred.shape[1:]!=(22,1) or mask.shape!=(len(pred),1) or not bool(torch.isfinite(pred).all()):
                    raise ValueError('original full validation output is nonfinite or malformed')
                self.parts.append(dict(pred=pred.detach().cpu().clone(),target=target.detach().cpu().clone(),mask=mask.detach().cpu().clone()))
            return out
        module.forward=observed
        self.step_hook=trainer.optimizers[0].register_step_post_hook(lambda *args:self._executed())

    def _executed(self):self.steps+=1

    def on_train_start(self,trainer,module):
        vision_optimizer_identity(self.prepared,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
        if getattr(trainer.precision_plugin,'scaler',None) is not None:raise ValueError('native bf16/CPU32 must not invent an FP16 scaler')
        if trainer.gradient_clip_val!=1 or trainer.accumulate_grad_batches!=1:raise ValueError('original clip/accumulation differs')

    def _batch(self,batch,role):
        import torch
        ids=batch['vision_sample_index']
        if ids.dtype!=torch.int64 or ids.ndim!=1:raise ValueError('actual vision consumption indices required')
        ids=ids.detach().cpu().numpy();n=len(ids);limit=self.train_samples if role=='train' else len(self.cohort.datasets[role])
        batch_size=self.identity['policy']['batch_size_by_role'][role]
        seen=self.train_seen if role=='train' else self.val_seen
        if n!=min(batch_size,limit-len(seen)) or np.any(ids<0) or np.any(ids>=limit) or len(np.unique(ids))!=n or any(int(i) in seen for i in ids):
            raise ValueError('vision native batch/population is partial, repeated or out of range')
        if role!='train' and ids.tolist()!=list(range(len(seen),len(seen)+n)):raise ValueError('validation order differs')
        if (batch['vision_img'].shape!=(n,3,256,256) or batch['joint_angles'].shape!=(n,22,1) or
            batch['label_valid_mask'].shape!=(n,1) or batch['label_valid_mask'].dtype!=torch.bool or
            batch['vision_valid_mask'].shape!=(n,1) or batch['emg'].shape!=(n,16,0)):
            raise ValueError('original vision input shape/mask differs')
        ds=self.cohort.datasets[role];windows=ds.windows;anchors=len(windows)
        for j,index in enumerate(ids):
            w=windows[int(index)%anchors];offset=int(batch['jitter_offset'][j])
            if (batch['role'][j]!=role or batch['split'][j]!=w['split'] or
                int(batch['target_hand_index'][j])!=(int(index)%(2*anchors))//anchors or
                int(batch['vision_repeat_index'][j])!=int(index)//(2*anchors) or
                not 0<=offset<=w['jitter_max_offset'] or int(batch['start'][j])!=w['start']+offset or
                int(batch['end'][j])!=w['end']+offset or int(batch['center'][j])!=w['center']+offset):
                raise ValueError('vision anchor/jitter/hand/repeat binding differs')
        seen.update(map(int,ids))
        return n

    def on_train_epoch_start(self,trainer,module):
        self.check()
        if trainer.current_epoch!=self.completed_epochs:raise ValueError('native resumed epoch differs from saved complete prefix')
        self.train_seen=set();self.train_loss_sum=0.;self.train_batches=0;self.validation=None

    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        self.check();self.phase='train';self.batch_n=self._batch(batch,'train')

    def on_before_optimizer_step(self,trainer,module,optimizer):
        import torch
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
            raise ValueError('original vision gradients are absent/nonfinite')

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        value=float(outputs['loss'])
        if not math.isfinite(value):raise ValueError('nonfinite original vision loss')
        self.train_loss_sum+=self.batch_n*value;self.train_batches+=1
        if self.steps!=trainer.global_step:raise ValueError('native bf16 optimizer call/global step differs')

    def on_validation_epoch_start(self,trainer,module):
        self.parts=[];self.val_seen=set()

    def on_validation_batch_start(self,trainer,module,batch,batch_idx,dataloader_idx=0):
        self.phase='val'
        if not trainer.sanity_checking:self.check();self._batch(batch,'val')

    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:self.phase=None;return
        import torch
        if len(self.val_seen)!=len(self.cohort.datasets['val']):raise ValueError('complete repeated validation is missing')
        value=float(trainer.callback_metrics['val_mae'])
        if not math.isfinite(value):raise ValueError('nonfinite original validation MAE')
        ref={name:torch.cat([p[name] for p in self.parts]) for name in ('pred','target','mask')}
        self.validation=dict(samples=len(self.val_seen),val_mae=value,prediction_sha256=storage.state_digest(ref))
        self.parts=[];self.phase=None

    def on_train_epoch_end(self,trainer,module):
        self.check()
        if len(self.train_seen)!=self.train_samples or self.validation is None:raise ValueError('partial vision epoch cannot be committed')
        expected_batches=math.ceil(self.train_samples/self.identity['policy']['batch_size_by_role']['train'])
        if self.train_batches!=expected_batches or self.steps!=(self.completed_epochs+1)*expected_batches:
            raise ValueError('native full epoch optimizer count differs')
        _adam_steps(trainer.optimizers[0],self.steps)
        from pytorch_lightning.callbacks import EarlyStopping
        early=next(cb for cb in trainer.callbacks if isinstance(cb,EarlyStopping))
        record=dict(epoch=self.completed_epochs,optimizer_executions=self.steps,global_step=trainer.global_step,
            train_samples=len(self.train_seen),train_batches=self.train_batches,train_loss=self.train_loss_sum/self.train_samples,
            validation=copy.deepcopy(self.validation),model_sha256=vision_state_sha256(module.state_dict()),
            early=dict(wait_count=early.wait_count,stopped_epoch=early.stopped_epoch,best_score=float(early.best_score)))
        self.records.append(record);self.completed_epochs+=1
        raw=_epoch_bytes(self.records,self.identity['sha256'])
        if len(raw)>MAX_EPOCH_BYTES:raise ValueError('vision epoch journal byte bound exceeded')
        path=self.output/'epochs.jsonl'
        with path.open('ab') as f:
            previous=_epoch_bytes(self.records[:-1],self.identity['sha256'])
            if path.stat().st_size!=len(previous):raise ValueError('owned epoch journal prefix changed')
            f.write(raw[len(previous):]);f.flush();os.fsync(f.fileno())
        if self.stop_after==self.completed_epochs:trainer.should_stop=True

    def on_save_checkpoint(self,trainer,module,checkpoint):
        self.check()
        checkpoint['vision_fit']=dict(format=FORMAT,identity=self.identity,epoch_boundary=self.validation is not None,
            completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records),rng=storage._rng({}))

    def on_fit_end(self,trainer,module):
        module.forward=self.original_forward;self.step_hook.remove()


def _native_callback(checkpoint,prefix):
    matches=[v for k,v in checkpoint['callbacks'].items() if k.startswith(prefix)]
    if len(matches)!=1:raise ValueError('one exact native callback required')
    return matches[0]


def read_vision_fit(output,*,require_full=True):
    import torch
    output=_safe(output);identity=_json(output/'identity.json');result=_json(output/'fit-result.json')
    _check_seal(identity,FORMAT);_check_seal(result,FORMAT)
    if result['identity_sha256']!=identity['sha256'] or result['scientific_eligible'] is not False or result['full_paper_complete'] is not False:
        raise ValueError('fit-only result identity/scientific claims differ')
    if identity['fit_sources_sha256']!=_sources():raise ValueError('vision fit source changed')
    if identity['native_loop_sources_sha256']!=_loop_sources():raise ValueError('native iterator/restore source changed')
    _check_seal(identity['policy'],'original_vision_fit_policy_v1')
    _verify_cohort_receipt(identity)
    if (result['scope']!=identity['policy']['scope'] or type(result['completed_epochs']) is not int or
        type(result['selected_best']['score']) is not float or result['native_loop_sources_sha256']!=identity['native_loop_sources_sha256']):
        raise ValueError('typed vision fit scope/epoch/score or native loop identity differs')
    if require_full and identity['policy']['scope']!='full':raise ValueError('engineering fit is not full training')
    last_path=output/'checkpoints/last.ckpt'
    if _file_hash(last_path)!=result['last_checkpoint_sha256']:raise ValueError('native last checkpoint bytes differ')
    last=torch.load(str(last_path),map_location='cpu',weights_only=False)
    packet=last['vision_fit']
    records,chain=_read_epochs(output/'epochs.jsonl',identity['sha256'])
    if packet['identity']!=identity or packet['epoch_boundary'] is not True or packet['records']!=records or packet['completed_epochs']!=len(records):
        raise ValueError('actual native checkpoint epoch carry differs from journal')
    policy=identity['policy'];batch=policy['batch_size_by_role']['train'];train=identity['train_samples']
    steps=math.ceil(train/batch)*len(records)
    if (not records or type(result['optimizer_executions']) is not int or result['optimizer_executions']!=steps or
        last['global_step']!=steps or result['completed_epochs']!=len(records)):
        raise ValueError('completed optimizer/epoch counters differ from actual native checkpoint')
    for i,record in enumerate(records):
        if (record['train_samples']!=train or record['train_batches']!=math.ceil(train/batch) or
            record['optimizer_executions']!=(i+1)*math.ceil(train/batch) or
            record['validation']['samples']!=identity['cohort']['counts']['val'] or
            not math.isfinite(record['validation']['val_mae']) or not math.isfinite(record['train_loss'])):
            raise ValueError('partial/nonfinite vision epoch cannot complete')
    if _schema(last['state_dict'])!=identity['construction']['model_schema'] or vision_state_sha256(last['state_dict'])!=records[-1]['model_sha256']:
        raise ValueError('native terminal model state differs from actual final epoch')
    native=_native_callback(last,'ModelCheckpoint{');best=Path(native['best_model_path'])
    if (best.parent!=output/'checkpoints' or best.is_symlink() or best.name=='last.ckpt' or
        result['selected_best']['file']!=best.name or _file_hash(best)!=result['selected_best']['checkpoint_sha256']):
        raise ValueError('actual native selected best differs; last fallback forbidden')
    score=float(native['best_model_score']);minimum=min(r['validation']['val_mae'] for r in records)
    if not math.isfinite(score) or score!=minimum or result['selected_best']['score']!=score:
        raise ValueError('actual native best score differs from complete validation history')
    selected=torch.load(str(best),map_location='cpu',weights_only=False);source=selected['vision_fit']
    epoch=source['completed_epochs']-1
    if (source['identity']!=identity or not 0<=epoch<len(records) or source['records']!=records[:epoch+1] or
        selected['global_step']!=records[epoch]['global_step'] or records[epoch]['validation']['val_mae']!=score or
        vision_state_sha256(selected['state_dict'])!=records[epoch]['model_sha256']):
        raise ValueError('selected checkpoint state/epoch is not the native validation best')
    early=_native_callback(last,'EarlyStopping{');done=vision_stage_complete(policy,records,early)
    if policy['scope']=='full':
        if not identity['cohort']['metadata_matches_registered_legacy_release'] or identity['cohort']['counts']!=FULL_COUNTS:
            raise ValueError('full fit requires complete registered legacy cohort')
    full=done and policy['scope']=='full'
    if result['stage_complete'] is not done or result['full_training_terminated'] is not full or (require_full and not full):
        raise ValueError('premature/external termination is not a full native fit')
    if result['final_model_state_sha256']!=records[-1]['model_sha256'] or result['selected_best']['validated'] is not True:
        raise ValueError('terminal model or selected-best claim differs')
    if result['train_samples_per_epoch']!=[r['train_samples'] for r in records] or result['val_samples_per_epoch']!=[r['validation']['samples'] for r in records]:
        raise ValueError('actual full coverage differs from fit summary')
    return result


def fit_vision(cohort,output,*,scope='full',resume=False,stop_after_epoch=None,cohort_receipt=None,activation_checkpointing=False):
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint
    cohort.check();policy=vision_fit_policy(cohort.upstream,cohort.config,scope=scope,activation_checkpointing=activation_checkpointing)
    receipt_path,receipt_binding=_cohort_receipt(cohort,cohort_receipt,required=scope=='full')
    if scope=='full' and (not cohort.identity['metadata_matches_registered_legacy_release'] or cohort.identity['counts']!=FULL_COUNTS):
        raise ValueError('full vision fit requires complete registered legacy cohort')
    if stop_after_epoch is not None and (scope=='full' or type(stop_after_epoch) is not int or not 1<=stop_after_epoch<=2):
        raise ValueError('stop_after_epoch is an explicit bounded engineering pause only')
    device=_runtime_device(policy);output=_safe(output,exists=resume);_space(output.parent)
    if not resume:output.mkdir(mode=0o700,exist_ok=False)
    lock=os.open(output/'fit.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        checkpoint=None;identity=None
        if resume:
            identity=_json(output/'identity.json');_check_seal(identity,FORMAT)
            if (identity['policy']!=policy or identity['cohort']!=cohort.identity or identity['fit_sources_sha256']!=_sources() or
                identity['native_loop_sources_sha256']!=_loop_sources() or identity['cohort_receipt']!=receipt_binding):
                raise ValueError('owned resume source/config/cohort differs')
            if (output/'fit-result.json').exists():
                previous=read_vision_fit(output,require_full=False)
                if previous['stage_complete']:raise ValueError('completed native fit cannot be reselected or retrained')
            checkpoint=torch.load(str(output/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False)
            packet=checkpoint['vision_fit']
            if packet['identity']!=identity or packet['epoch_boundary'] is not True:raise ValueError('owned complete native epoch required')
            carry=_epoch_bytes(packet['records'],identity['sha256']);path=output/'epochs.jsonl';raw=path.read_bytes()
            if not raw.startswith(carry):raise ValueError('native checkpoint epoch carry differs from owned journal prefix')
            if len(raw)>len(carry):
                with (output/('uncommitted-epochs-'+uuid.uuid4().hex+'.jsonl')).open('xb') as f:f.write(raw[len(carry):])
                with path.open('wb') as f:f.write(carry);f.flush();os.fsync(f.fileno())
        prepared=build_vision_runtime(cohort.upstream,cohort.config,**({} if checkpoint is None else dict(
            state=checkpoint['state_dict'],expected_initialization=identity['construction'])))
        _configure_activation_checkpointing(prepared,policy['activation_checkpointing'])
        loaders=native_vision_loaders(cohort)
        train_samples=len(cohort.datasets['train'])
        if scope!='full':
            from torch.utils.data import Subset
            from egoemg.datamodule import make_data_module
            dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0
            dm.batch_size=policy['batch_size_by_role']['train'];train_samples=min(policy['engineering_train_samples'],train_samples)
            loaders={role:dm._make_loader(Subset(ds,list(range(train_samples))) if role=='train' else ds,shuffle=role=='train') for role,ds in cohort.datasets.items()}
        if identity is None:
            identity=_sealed(dict(format=FORMAT,owner=uuid.uuid4().hex,policy=policy,cohort=cohort.identity,
                cohort_receipt=receipt_binding,
                construction=prepared.identity,fit_sources_sha256=_sources(),native_loop_sources_sha256=_loop_sources(),train_samples=train_samples,
                scientific_eligible=False,full_paper_complete=False))
            _write(output/'identity.json',identity)
            with (output/'epochs.jsonl').open('xb'):pass
        elif identity['train_samples']!=train_samples:raise ValueError('actual resume population differs')
        witness=FitWitness(prepared,cohort,identity,output,train_samples,stop_after_epoch,receipt_path)
        from egoemg._train_utils import build_callbacks
        callbacks=build_callbacks(cohort.config)+[witness]
        for cb in callbacks:
            if isinstance(cb,ModelCheckpoint):cb.dirpath=str(output/'checkpoints')
        logger=storage.CSVLogger(save_dir=str(output),name='logs',version=uuid.uuid4().hex)
        kwargs=OmegaConf.to_container(cohort.config.trainer,resolve=True)
        kwargs.update(devices=1,accelerator=device['accelerator'],precision=policy['precision'],max_epochs=policy['max_epochs'],
            num_sanity_val_steps=policy['num_sanity_val_steps'],enable_progress_bar=False,callbacks=callbacks,logger=logger)
        trainer=storage.pl.Trainer(**kwargs)
        connector=trainer._checkpoint_connector;original=connector.restore_training_state
        def observed_restore():
            result=original()
            if checkpoint is not None:
                actual=_snap(prepared.module,trainer);expected=_checkpoint_snap(checkpoint)
                witness.exact_restore={k:actual[k]==expected[k] for k in expected}
                if not all(witness.exact_restore.values()):raise ValueError('actual complete native restore differs: '+str(witness.exact_restore))
            return result
        connector.restore_training_state=observed_restore;start=time.monotonic()
        try:trainer.fit(prepared.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'],
                ckpt_path=None if checkpoint is None else str(output/'checkpoints/last.ckpt'))
        finally:connector.restore_training_state=original
        cohort.check();witness.check();prepared.check()
        native=next(cb for cb in trainer.callbacks if isinstance(cb,ModelCheckpoint));best=Path(native.best_model_path)
        from pytorch_lightning.callbacks import EarlyStopping
        early=next(cb for cb in trainer.callbacks if isinstance(cb,EarlyStopping))
        stage_complete=vision_stage_complete(policy,witness.records,early.state_dict())
        result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,completed_epochs=witness.completed_epochs,
            optimizer_executions=witness.steps,stage_complete=stage_complete,full_training_terminated=stage_complete and scope=='full',
            scientific_eligible=False,full_paper_complete=False,resume_start_exact=witness.exact_restore,
            final_model_state_sha256=vision_state_sha256(prepared.module.state_dict()),
            last_checkpoint_sha256=_file_hash(output/'checkpoints/last.ckpt'),
            selected_best=dict(file=best.name,score=float(native.best_model_score),checkpoint_sha256=_file_hash(best),validated=True),
            train_samples_per_epoch=[r['train_samples'] for r in witness.records],val_samples_per_epoch=[r['validation']['samples'] for r in witness.records],
            cuda_initialized=torch.cuda.is_initialized(),runtime_device=device,elapsed_fit_seconds=time.monotonic()-start,
            native_loop_sources_sha256=identity['native_loop_sources_sha256'],
            rng_restore_contract='exact after original restore_training_state, before native iterator reconstruction; native reentry can advance global RNG differently from uninterrupted training',
            callback_restore_policy='native fields except ModelCheckpoint.current_score, which native PL never restores',
            bitwise_training_trajectory_guaranteed=False,performance_is_paper_gpu_hours=False))
        _write(output/'fit-result.json',result,replace=resume)
        read_vision_fit(output,require_full=scope=='full' and stage_complete)
        return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    fit=sub.add_parser('fit')
    for name in ('data','crops','upstream','output'):fit.add_argument('--'+name,type=Path,required=True)
    fit.add_argument('--variant',choices=('vit_small','vit_base','vit_large'),default='vit_small')
    fit.add_argument('--scope',choices=SCOPES,required=True);fit.add_argument('--resume',action='store_true')
    fit.add_argument('--cohort-receipt',type=Path)
    fit.add_argument('--activation-checkpointing',action='store_true',
        help='explicit native timm activation recomputation; preserve original batch and training objective')
    fit.add_argument('--stop-after-epoch',type=int)
    args=vars(parser.parse_args());args.pop('action');data=args.pop('data');crops=args.pop('crops');upstream=args.pop('upstream');variant=args.pop('variant')
    if args['scope']=='full' and args['cohort_receipt'] is None:
        parser.error('--cohort-receipt is required for full training; freeze and qualify actual complete inputs first')
    cfg=compose_vision(upstream,variant);expected=None if args['cohort_receipt'] is None else _json(args['cohort_receipt'])
    if args['resume']:
        saved=_json(args['output']/'identity.json')['cohort']
        if expected is not None and expected!=saved:raise ValueError('frozen cohort receipt differs from owned resume identity')
        expected=saved
    cohort=admit_vision_cohort(data,crops,upstream=upstream,config=cfg,expected=expected)
    result=fit_vision(cohort,**args)
    print(json.dumps({k:result[k] for k in ('scope','completed_epochs','optimizer_executions','stage_complete','full_training_terminated')}))


if __name__=='__main__':main()
