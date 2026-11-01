"""Original ResNet fit with native validation-end checkpoints and exact restore.

Odd epochs contain training evidence only. Native validation/checkpoint hooks
run every two epochs; resumed pending epoch-end hooks cannot add another epoch.
Fit-only and all engineering records remain ineligible as scientific results.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
import uuid

import numpy as np

from . import classic_emg_train as storage
from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _StatGuard,_safe
from .resnet_literal import resnet_protocol,compose_resnet
from .resnet_data import ResnetCohort,admit_resnet_cohort,native_resnet_loaders,FULL_COUNTS
from .resnet_runtime import build_resnet_runtime,resnet_optimizer_identity
from .vision_literal_runtime import vision_state_sha256,_schema
from .vision_literal_check import _native_restored_callbacks
from .vision_literal_train import (_json,_write,_sealed,_check_seal,_space,_runtime_device,
    _read_epochs,_epoch_bytes,_callbacks,_snap,_adam_steps,_native_callback,
    _cohort_receipt,_verify_cohort_receipt,FitWitness as _PointWitness,
    _loop_paths as _point_loop_paths)


SCOPES=('full','cuda_engineering','cpu_engineering')
FORMAT='original_resnet_native_fit_v1'
BOUNDARY='native_validation_end_before_train_epoch_end'
ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
MAX_EPOCH_BYTES=2*1024**2


def _plain(value):
    from omegaconf import OmegaConf
    if OmegaConf.is_config(value):return OmegaConf.to_container(value,resolve=True)
    if isinstance(value,dict):return{k:_plain(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return[_plain(v)for v in value]
    return value


def _factory_hyperparameters(config):
    """Exact argument mapping of pinned egoemg.train.make_lightning_module."""
    c=_plain(config)
    return dict(module_conf=c['module'],optimizer_conf=c['optimizer'],lr_scheduler_conf=c['lr_scheduler'],
        loss_weights=c['loss_weights'],pretrained_checkpoint=c.get('pretrained_checkpoint'),
        pretrained_strict=c.get('pretrained_strict',False),freeze_backbone=c.get('freeze_backbone',False),
        pretrained_emg_checkpoint=c.get('pretrained_emg_checkpoint'),stage2_vision_checkpoint=c.get('stage2_vision_checkpoint'),
        component_lr_scales=c.get('component_lr_scales'),ignore_head_tail_dims=c.get('ignore_head_tail_dims',0),
        datamodule=c['datamodule'],batch_augmentation=c.get('batch_augmentation'),
        val_episode_name_mapping=c.get('val_episode_name_mapping'),anchor_loss_weight=c.get('anchor_loss_weight',0.),
        anchor_shuffle_fraction=c.get('anchor_shuffle_fraction',0.))


def _validate_hyperparameters(checkpoint,identity):
    packet=checkpoint['resnet_fit'];fresh=packet['factory_pretrained']
    if type(fresh)is not bool or packet['initialization_mode']!=('fresh_native_pretrained'if fresh else'strict_self_contained_restore'):
        raise ValueError('native ResNet initialization/complete-restore mode differs')
    expected=copy.deepcopy(identity['native_hyperparameters'])
    expected['module_conf']['pretrained']=fresh
    if _hash(_plain(checkpoint['hyper_parameters']))!=_hash(expected):
        raise ValueError('native ResNet saved objective/model/factory hyperparameters differ')


def resnet_fit_policy(upstream,config,*,scope='full'):
    if type(scope) is not str or scope not in SCOPES:
        raise ValueError('explicit supported original ResNet fit scope required')
    p=resnet_protocol(upstream,config);full=scope=='full';cpu=scope=='cpu_engineering'
    checkpoint,_,early=p['callbacks']
    result=dict(format='original_resnet_fit_policy_v1',scope=scope,variant=p['variant'],
        protocol_sha256=p['sha256'],max_epochs=p['trainer']['max_epochs'] if full else 4,
        check_val_every_n_epoch=p['trainer']['check_val_every_n_epoch'],
        precision='32-true' if cpu else p['trainer']['precision'],
        batch_size_by_role={r:2 if cpu else v for r,v in p['batch_size_by_role'].items()},
        workers_by_role={r:0 if cpu else v for r,v in p['workers_by_role'].items()},
        monitor=checkpoint['monitor'],monitor_mode=checkpoint['mode'],patience=early['patience'],
        patience_counts='actual validation checks only',
        check_on_train_epoch_end=early['check_on_train_epoch_end'],
        save_top_k=checkpoint['save_top_k'],save_last=checkpoint['save_last'],
        gradient_clip_val=p['trainer']['gradient_clip_val'],
        scheduler_T_max=p['lr_scheduler']['scheduler']['T_max'],
        scheduler_eta_min=p['lr_scheduler']['scheduler']['eta_min'],
        num_sanity_val_steps=2 if full else 0,engineering_train_samples=None if full else
            4 if cpu else 2*p['batch_size_by_role']['train'],
        engineering_eval_samples=4 if cpu else None,
        native_global_loader_rng=True,single_rank=True,declared_devices=p['trainer']['devices'],
        scientific_eligible=False,full_paper_complete=False,
        bitwise_training_trajectory_guaranteed=False,performance_is_paper_gpu_hours=False)
    result['sha256']=_hash(result);return result


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported ResNet fit source changed')
    names=('egoemg_repro/resnet_train.py','egoemg_repro/resnet_data.py',
        'egoemg_repro/resnet_runtime.py','egoemg_repro/resnet_literal.py',
        'egoemg_repro/vision_literal_train.py','egoemg_repro/vision_literal_check.py',
        'egoemg_repro/classic_emg_train.py','egoemg_repro/classic_emg_cohort.py')
    return {name:_file_hash(ROOT/name)for name in names}


def _loop_paths():
    paths=_point_loop_paths()
    for name in ('pytorch_lightning.callbacks.model_checkpoint',
                 'pytorch_lightning.callbacks.early_stopping','pytorch_lightning.callbacks.lr_monitor'):
        paths[name]=Path(importlib.import_module(name).__file__).resolve()
    return paths


def _loop_sources():return {n:_file_hash(p)for n,p in _loop_paths().items()}


def _digest(value):return type(value)is str and re.fullmatch('[0-9a-f]{64}',value)is not None


def _validate_records(policy,records,*,train_samples,val_samples):
    """Recompute coverage and native patience from genuine validation events."""
    if (type(records)is not list or not records or len(records)>policy['max_epochs']
        or any(type(n)is not int or n<=0 for n in (train_samples,val_samples))):
        raise ValueError('bounded nonempty complete ResNet epoch history required')
    batch=policy['batch_size_by_role']['train'];batches=math.ceil(train_samples/batch)
    best=None;wait=0;stopped=0;checks=0
    try:
        for i,r in enumerate(records):
            if (type(r['epoch'])is not int or r['epoch']!=i
                or any(type(r[k])is not int for k in ('optimizer_executions','global_step','train_samples','train_batches'))
                or r['optimizer_executions']!=(i+1)*batches or r['global_step']!=(i+1)*batches
                or r['train_samples']!=train_samples or r['train_batches']!=batches
                or type(r['train_loss'])is not float or not math.isfinite(r['train_loss']) or r['train_loss']<0
                or not _digest(r['model_sha256']) or (stopped and i>stopped)):
                raise ValueError('partial, duplicated or malformed ResNet epoch evidence')
            v=r['validation']
            if (i+1)%policy['check_val_every_n_epoch']:
                if v is not None:raise ValueError('odd epoch cannot claim a fresh validation')
            else:
                if (type(v)is not dict or type(v['samples'])is not int or v['samples']!=val_samples
                    or type(v['val_mae'])is not float or not math.isfinite(v['val_mae']) or v['val_mae']<0
                    or not _digest(v['prediction_sha256'])):
                    raise ValueError('complete genuine even-epoch validation required')
                checks+=1;score=v['val_mae']
                if best is None or score<best:best=score;wait=0
                else:
                    wait+=1
                    if wait>=policy['patience']:stopped=i
            early=r['early']
            if (type(early['wait_count'])is not int or type(early['stopped_epoch'])is not int
                or(best is not None and type(early['best_score'])is not float)
                or early!=dict(wait_count=wait,stopped_epoch=stopped,best_score=best)):
                raise ValueError('native patience/best must advance on genuine validations only')
    except (KeyError,TypeError,IndexError)as exc:raise ValueError('malformed ResNet epoch evidence')from exc
    return dict(completed_epochs=len(records),optimizer_executions=batches*len(records),
        validation_checks=checks,minimum_val_mae=best)


def resnet_stage_complete(policy,records,early,*,train_samples,val_samples):
    value=_validate_records(policy,records,train_samples=train_samples,val_samples=val_samples)
    n=value['completed_epochs']
    if n%2 or value['minimum_val_mae']is None:return False
    actual=dict(wait_count=early['wait_count'],stopped_epoch=early['stopped_epoch'],best_score=float(early['best_score']))
    if actual!=records[-1]['early']:raise ValueError('terminal native EarlyStopping state differs from history')
    if policy['scope']!='full':return n==policy['max_epochs']
    return n==150 or (value['validation_checks']>=31 and actual['wait_count']>=30 and actual['stopped_epoch']==n-1)


def _checkpoint_snap(checkpoint):
    d=storage.state_digest
    return dict(model=d(checkpoint['state_dict']),optimizer=d(checkpoint['optimizer_states']),
        scheduler=d(checkpoint['lr_schedulers']),callbacks=d(_native_restored_callbacks(checkpoint['callbacks'])),
        loops=d(checkpoint['loops']['fit_loop']),rng=d(checkpoint['resnet_fit']['rng']))


def _validate_rng(value,*,cpu):
    import torch
    if type(value)is not dict or set(value)!={'python','numpy','torch','cuda','generators'} or value['generators']!={}:
        raise ValueError('complete native global RNG packet required')
    try:
        random.Random().setstate(value['python']);np.random.RandomState().set_state(value['numpy'])
        torch.Generator().set_state(value['torch'])
    except (TypeError,ValueError,RuntimeError)as exc:raise ValueError('native RNG packet cannot be restored')from exc
    if (type(value['cuda'])is not list or len(value['cuda'])!=(0 if cpu else 1)
        or any(not isinstance(t,torch.Tensor)or t.dtype!=torch.uint8 or t.ndim!=1 or not t.numel()for t in value['cuda'])):
        raise ValueError('actual native CPU/CUDA RNG domains differ')


def _validate_native_boundary(checkpoint,identity,records):
    """The author's checkpoint is after validation, before epoch-end completion."""
    p=identity['policy'];n=len(records);batches=math.ceil(identity['train_samples']/p['batch_size_by_role']['train'])
    val_batches=math.ceil(identity['val_samples']/p['batch_size_by_role']['val'])
    packet=checkpoint['resnet_fit'];loops=checkpoint['loops']['fit_loop']
    if (n%2 or type(checkpoint['epoch'])is not int or type(checkpoint['global_step'])is not int
        or type(packet['completed_epochs'])is not int
        or packet['format']!=FORMAT or packet['boundary_kind']!=BOUNDARY or packet['epoch_boundary']is not True
        or packet['identity']!=identity or packet['records']!=records or packet['completed_epochs']!=n
        or checkpoint['epoch']!=n-1 or checkpoint['global_step']!=n*batches):
        raise ValueError('owned complete native ResNet validation boundary required')
    epoch=loops['epoch_progress']
    expected=dict(completed=n-1,processed=n-1,ready=n,started=n)
    if (epoch['current']!=expected or epoch['total']!=expected
        or any(type(v)is not int for k in ('current','total')for v in epoch[k].values())):
        raise ValueError('original validation-end pending epoch progress differs')
    for key,size in (('epoch_loop.batch_progress',batches),('epoch_loop.val_loop.batch_progress',val_batches)):
        progress=loops[key]
        if (progress['current']!={k:size for k in ('completed','processed','ready','started')}
            or any(type(v)is not int for v in progress['current'].values())or progress['is_last_batch']is not True):
            raise ValueError('native complete training/validation batch progress differs')
    if (loops['epoch_loop.batch_progress']['total']!={k:n*batches for k in ('completed','processed','ready','started')}
        or loops['epoch_loop.state_dict']['_batches_that_stepped']!=n*batches-1
        or loops['epoch_loop.scheduler_progress']['total']!=dict(completed=n,ready=n)):
        raise ValueError('native total loop/optimizer/scheduler progress differs')
    _validate_rng(packet['rng'],cpu=p['scope']=='cpu_engineering')


def _validate_saved_optimizer(checkpoint,identity,steps,epochs):
    import torch
    if len(checkpoint['optimizer_states'])!=1 or len(checkpoint['lr_schedulers'])!=1:
        raise ValueError('one complete original optimizer/scheduler required')
    opt=checkpoint['optimizer_states'][0];sched=checkpoint['lr_schedulers'][0]
    if (sched['T_max']!=150 or sched['eta_min']!=1e-5 or sched['last_epoch']!=epochs
        or sched['base_lrs']!=[1e-4,1e-4] or sched['_step_count']!=epochs+1
        or type(sched['_last_lr'])is not list or len(sched['_last_lr'])!=2 or len(opt['param_groups'])!=2):
        raise ValueError('original ResNet cosine checkpoint differs')
    lr=1e-5+(1e-4-1e-5)*(1+math.cos(math.pi*epochs/150))/2
    names=identity['construction']['parameter_names'];schema=identity['construction']['model_schema'];seen=[]
    for group,prefix in zip(opt['param_groups'],('head','backbone')):
        expected=[n for n in names if n.startswith('model.'+prefix+'.')]
        if (group.get('name')!=prefix or len(group['params'])!=len(expected)
            or group['initial_lr']!=1e-4 or group['weight_decay']!=.01 or tuple(group['betas'])!=(.9,.999)
            or group['eps']!=1e-8 or group['amsgrad']is not False
            or any(group.get(k,False)is not False for k in ('maximize','capturable','differentiable'))
            or any(group.get(k)is not None for k in ('foreach','fused'))
            or not math.isclose(group['lr'],lr,rel_tol=1e-10,abs_tol=1e-14)):
            raise ValueError('saved original ResNet AdamW groups differ')
        for index,name in zip(group['params'],expected):
            state=opt['state'][index]
            if type(index)is not int or index in seen or set(state)!={'step','exp_avg','exp_avg_sq'}:
                raise ValueError('complete native AdamW steps/moments required')
            step=state['step']
            if not isinstance(step,torch.Tensor)or step.numel()!=1 or float(step)!=steps:
                raise ValueError('complete native AdamW steps/moments required')
            for key in ('exp_avg','exp_avg_sq'):
                t=state[key]
                if list(t.shape)!=schema[name]['shape']or str(t.dtype)!=schema[name]['dtype']or not bool(torch.isfinite(t).all()):
                    raise ValueError('saved named AdamW moment shape/dtype/content differs')
            seen.append(index)
    if set(seen)!=set(opt['state'])or len(seen)!=len(names):raise ValueError('saved AdamW parameter coverage differs')
    if any(not math.isclose(v,lr,rel_tol=1e-10,abs_tol=1e-14)for v in sched['_last_lr']):
        raise ValueError('native stored scheduler learning rates differ')


class ResnetFitWitness(storage.Callback):
    # These unchanged methods observe generic point batches and native hooks.
    state_dict=_PointWitness.state_dict
    load_state_dict=_PointWitness.load_state_dict
    _executed=_PointWitness._executed
    on_before_optimizer_step=_PointWitness.on_before_optimizer_step
    on_train_batch_end=_PointWitness.on_train_batch_end
    on_validation_epoch_start=_PointWitness.on_validation_epoch_start
    on_validation_batch_start=_PointWitness.on_validation_batch_start
    on_fit_end=_PointWitness.on_fit_end

    def __init__(self,prepared,cohort,identity,output,stop_after_epoch,receipt_path):
        self.prepared=prepared;self.cohort=cohort;self.identity=identity;self.output=output
        self.stop_after=stop_after_epoch;self.train_samples=identity['train_samples'];self.val_samples=identity['val_samples']
        self.steps=0;self.completed_epochs=0;self.records=[];self.pending=None;self.pending_close=None
        self.committed_current=False;self.exact_restore=None;self.phase=None;self.validation=None
        self.source_guard=_StatGuard([*[ROOT/n for n in identity['fit_sources_sha256']],*_loop_paths().values(),
            *([]if receipt_path is None else [receipt_path])])

    def check(self):
        self.cohort.guard.check();self.source_guard.check();_space(self.output)

    def on_load_checkpoint(self,trainer,module,checkpoint):
        packet=checkpoint['resnet_fit']
        _validate_native_boundary(checkpoint,self.identity,packet['records'])
        self.pending=checkpoint;self.pending_close=packet['completed_epochs']-1

    def on_fit_start(self,trainer,module):
        import torch
        if self.pending is not None:storage._restore_rng(self.pending['resnet_fit']['rng'],{})
        self.original_forward=module.forward
        def observed(batch):
            out=self.original_forward(batch)
            if self.phase=='val'and not trainer.sanity_checking:
                pred,target,mask=out
                if (pred.shape!=target.shape or pred.shape[1:]!=(22,1)or mask.shape!=(len(pred),1)
                    or not bool(torch.isfinite(pred).all())):
                    raise ValueError('original ResNet complete validation output differs or is nonfinite')
                self.parts.append(dict(pred=pred.detach().cpu().clone(),target=target.detach().cpu().clone(),mask=mask.detach().cpu().clone()))
            return out
        module.forward=observed
        self.step_hook=trainer.optimizers[0].register_step_post_hook(lambda *a:self._executed())

    def on_train_start(self,trainer,module):
        resnet_optimizer_identity(self.prepared,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
        if getattr(trainer.precision_plugin,'scaler',None)is not None:
            raise ValueError('native BF16/CPU32 has no FP16 scaler')
        if trainer.gradient_clip_val!=1 or trainer.accumulate_grad_batches!=1:
            raise ValueError('original ResNet clip/accumulation differs')

    def _batch(self,batch,role):
        import torch
        ids=batch['vision_sample_index']
        if ids.dtype!=torch.int64 or ids.ndim!=1:raise ValueError('actual native point consumption indices required')
        ids=ids.detach().cpu().numpy();n=len(ids)
        limit=self.train_samples if role=='train'else self.val_samples
        seen=self.train_seen if role=='train'else self.val_seen
        size=self.identity['policy']['batch_size_by_role'][role]
        if (n!=min(size,limit-len(seen))or np.any(ids<0)or np.any(ids>=limit)
            or len(np.unique(ids))!=n or any(int(i)in seen for i in ids)):
            raise ValueError('native ResNet batch/population is partial, repeated or out of range')
        if role!='train'and ids.tolist()!=list(range(len(seen),len(seen)+n)):
            raise ValueError('original ResNet validation order differs')
        if (batch['vision_img'].shape!=(n,3,256,256)or batch['joint_angles'].shape!=(n,22,1)
            or batch['label_valid_mask'].shape!=(n,1)or batch['label_valid_mask'].dtype!=torch.bool
            or batch['vision_valid_mask'].shape!=(n,1)or batch['emg'].shape!=(n,16,0)):
            raise ValueError('original ResNet point input/target/mask layout differs')
        ds=self.cohort.datasets[role];anchors=len(ds.windows)
        for j,index in enumerate(ids):
            w=ds.windows[int(index)%anchors];offset=int(batch['jitter_offset'][j])
            if (batch['role'][j]!=role or batch['split'][j]!=w['split']
                or int(batch['target_hand_index'][j])!=(int(index)%(2*anchors))//anchors
                or int(batch['vision_repeat_index'][j])!=int(index)//(2*anchors)
                or not 0<=offset<=w['jitter_max_offset']or int(batch['start'][j])!=w['start']+offset
                or int(batch['end'][j])!=w['end']+offset or int(batch['center'][j])!=w['center']+offset):
                raise ValueError('original ResNet anchor/jitter/hand/repeat identity differs')
        seen.update(map(int,ids));return n

    def on_train_epoch_start(self,trainer,module):
        self.check()
        if trainer.current_epoch!=self.completed_epochs or self.pending_close is not None:
            raise ValueError('native resumed ResNet epoch differs from committed prefix')
        self.train_seen=set();self.train_loss_sum=0.;self.train_batches=0;self.validation=None
        self.committed_current=False

    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        self.check();self.phase='train';self.batch_n=self._batch(batch,'train')

    def _commit_epoch(self,trainer,module):
        from pytorch_lightning.callbacks import EarlyStopping
        self.check();i=trainer.current_epoch
        if i!=self.completed_epochs or len(self.train_seen)!=self.train_samples:
            raise ValueError('only a new complete ResNet training epoch can be committed')
        batches=math.ceil(self.train_samples/self.identity['policy']['batch_size_by_role']['train'])
        if self.train_batches!=batches or self.steps!=(i+1)*batches or trainer.global_step!=self.steps:
            raise ValueError('actual native ResNet optimizer execution count differs')
        if (i%2==1)!=(self.validation is not None):raise ValueError('original every-two-epoch validation cadence differs')
        _adam_steps(trainer.optimizers[0],self.steps)
        early=next(c for c in trainer.callbacks if isinstance(c,EarlyStopping));best=float(early.best_score)
        record=dict(epoch=i,optimizer_executions=self.steps,global_step=trainer.global_step,
            train_samples=len(self.train_seen),train_batches=self.train_batches,
            train_loss=self.train_loss_sum/self.train_samples,validation=copy.deepcopy(self.validation),
            model_sha256=vision_state_sha256(module.state_dict()),early=dict(wait_count=early.wait_count,
                stopped_epoch=early.stopped_epoch,best_score=best if math.isfinite(best)else None))
        candidate=[*self.records,record]
        _validate_records(self.identity['policy'],candidate,train_samples=self.train_samples,val_samples=self.val_samples)
        raw=_epoch_bytes(candidate,self.identity['sha256']);previous=_epoch_bytes(self.records,self.identity['sha256'])
        if len(raw)>MAX_EPOCH_BYTES:raise ValueError('bounded ResNet epoch journal required')
        path=self.output/'epochs.jsonl'
        if path.read_bytes()!=previous:raise ValueError('owned ResNet epoch journal prefix changed')
        with path.open('ab')as f:f.write(raw[len(previous):]);f.flush();os.fsync(f.fileno())
        self.records=candidate;self.completed_epochs+=1;self.committed_current=True

    def on_validation_end(self,trainer,module):
        import torch
        if trainer.sanity_checking:self.phase=None;return
        if len(self.val_seen)!=self.val_samples:raise ValueError('complete ResNet validation population missing')
        value=float(trainer.callback_metrics['val_mae'])
        if not math.isfinite(value):raise ValueError('nonfinite native ResNet validation')
        ref={n:torch.cat([part[n]for part in self.parts])for n in ('pred','target','mask')}
        self.validation=dict(samples=len(self.val_seen),val_mae=value,prediction_sha256=storage.state_digest(ref))
        self.parts=[];self.phase=None
        # Original ModelCheckpoint runs next, still before train_epoch_end.
        self._commit_epoch(trainer,module)

    def on_train_epoch_end(self,trainer,module):
        self.check();i=trainer.current_epoch
        if self.completed_epochs==i:
            if i%2:raise ValueError('even epoch lost its native validation-end commit')
            self._commit_epoch(trainer,module)
        elif (self.completed_epochs==i+1 and self.records[-1]['epoch']==i
              and (self.committed_current or self.pending_close==i)):
            if self.steps!=trainer.global_step or self.records[-1]['global_step']!=trainer.global_step:
                raise ValueError('pending native epoch-end changed optimizer steps')
            self.pending_close=None;self.committed_current=False
        else:raise ValueError('unowned or duplicate native ResNet epoch-end event')
        if self.stop_after==self.completed_epochs:trainer.should_stop=True

    def on_save_checkpoint(self,trainer,module,checkpoint):
        self.check()
        if not self.committed_current or self.completed_epochs%2:
            raise ValueError('native ResNet checkpoint must follow genuine complete validation')
        checkpoint['resnet_fit']=dict(format=FORMAT,identity=self.identity,epoch_boundary=True,boundary_kind=BOUNDARY,
            factory_pretrained=self.prepared.factory_pretrained,initialization_mode=self.prepared.mode,
            completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records),rng=storage._rng({}))
        _validate_native_boundary(checkpoint,self.identity,self.records)


def _trusted_checkpoint(path):
    import torch
    path=_safe(path)
    if not path.is_file()or path.is_symlink():raise ValueError('owned regular native checkpoint required')
    return torch.load(str(path),map_location='cpu',weights_only=False)


def _validate_identity(identity):
    _check_seal(identity,FORMAT)
    if identity['fit_sources_sha256']!=_sources()or identity['native_loop_sources_sha256']!=_loop_sources():
        raise ValueError('ResNet fit source/native loop generation differs')
    config=compose_resnet(identity['upstream'],identity['policy']['variant'])
    policy=resnet_fit_policy(identity['upstream'],config,scope=identity['policy']['scope'])
    if policy!=identity['policy']or resnet_protocol(identity['upstream'],config)['sha256']!=identity['construction']['protocol_sha256']:
        raise ValueError('actual original ResNet fit policy/config differs')
    if identity['native_hyperparameters']!=_factory_hyperparameters(config):
        raise ValueError('native original ResNet factory hyperparameter identity differs')
    if identity['cohort']['projection']['resnet_protocol_sha256']!=policy['protocol_sha256']:
        raise ValueError('native ResNet data/model protocol binding differs')
    _verify_cohort_receipt(identity)
    return policy


def _checkpoint_contents(checkpoint,identity,records):
    summary=_validate_records(identity['policy'],records,train_samples=identity['train_samples'],val_samples=identity['val_samples'])
    _validate_native_boundary(checkpoint,identity,records)
    _validate_hyperparameters(checkpoint,identity)
    if (_schema(checkpoint['state_dict'])!=identity['construction']['model_schema']
        or vision_state_sha256(checkpoint['state_dict'])!=records[-1]['model_sha256']):
        raise ValueError('native ResNet checkpoint tensors differ from complete epoch evidence')
    _validate_saved_optimizer(checkpoint,identity,summary['optimizer_executions'],summary['completed_epochs'])
    callback=_native_callback(checkpoint,'ResnetFitWitness')
    if callback!=dict(steps=summary['optimizer_executions'],completed_epochs=len(records),records=records):
        raise ValueError('native observer callback state differs from checkpoint carry')
    early=_native_callback(checkpoint,'EarlyStopping{')
    if (type(early['patience'])is not int or early['patience']!=identity['policy']['patience']
        or dict(wait_count=early['wait_count'],stopped_epoch=early['stopped_epoch'],best_score=float(early['best_score']))!=records[-1]['early']):
        raise ValueError('native saved EarlyStopping state differs from actual validation history')
    return summary


def _validate_checkpoint(checkpoint,identity,records):
    try:return _checkpoint_contents(checkpoint,identity,records)
    except (KeyError,TypeError,IndexError,AttributeError)as exc:
        raise ValueError('complete native ResNet checkpoint schema required')from exc


def _verify_resume(output,identity,result,records):
    evidence=result['resume_evidence']
    if evidence is None:
        if result['resume_start_exact']is not None:raise ValueError('fresh fit cannot invent native resume evidence')
        return
    path=output/evidence['parent_file'];proof_path=output/evidence['observation_file']
    if (path.parent!=output/'parents'or proof_path.parent!=output/'resume-observations'
        or _file_hash(path)!=evidence['parent_checkpoint_sha256']or _file_hash(proof_path)!=evidence['observation_sha256']):
        raise ValueError('preserved native resume parent or observation bytes differ')
    parent=_trusted_checkpoint(path);carry=parent['resnet_fit']['records'];_validate_checkpoint(parent,identity,carry)
    if carry!=records[:len(carry)]or len(carry)>=len(records):raise ValueError('native resumed parent is not a strict complete prefix')
    proof=_json(proof_path);expected=_checkpoint_snap(parent)
    if (proof['identity_sha256']!=identity['sha256']or proof['parent_checkpoint_sha256']!=evidence['parent_checkpoint_sha256']
        or proof['expected']!=expected or proof['observed']!=expected
        or result['resume_start_exact']!={k:True for k in expected}):
        raise ValueError('six original restored state domains differ from preserved parent')


def read_resnet_fit(output,*,require_full=True):
    if type(require_full)is not bool:raise ValueError('full requirement must be boolean')
    output=_safe(output);identity=_json(output/'identity.json');result=_json(output/'fit-result.json')
    policy=_validate_identity(identity);_check_seal(result,FORMAT)
    if (result['identity_sha256']!=identity['sha256']or result['scope']!=policy['scope']
        or result['scientific_eligible']is not False or result['full_paper_complete']is not False
        or result['native_loop_sources_sha256']!=identity['native_loop_sources_sha256']):
        raise ValueError('ResNet fit-only identity/scope/scientific claim differs')
    if require_full and policy['scope']!='full':raise ValueError('engineering ResNet fit is not full training')
    last_path=output/'checkpoints/last.ckpt'
    if _file_hash(last_path)!=result['last_checkpoint_sha256']:raise ValueError('native last ResNet checkpoint bytes differ')
    records,chain=_read_epochs(output/'epochs.jsonl',identity['sha256']);last=_trusted_checkpoint(last_path)
    summary=_validate_checkpoint(last,identity,records)
    if (type(result['completed_epochs'])is not int or result['completed_epochs']!=summary['completed_epochs']
        or type(result['optimizer_executions'])is not int or result['optimizer_executions']!=summary['optimizer_executions']
        or result['epoch_journal_chain_sha256']!=chain):
        raise ValueError('actual ResNet epoch/update counters or journal differ')
    native=_native_callback(last,'ModelCheckpoint{');best=Path(native['best_model_path'])
    if (best.parent!=output/'checkpoints'or best.is_symlink()or best.name=='last.ckpt'
        or result['selected_best']['file']!=best.name or _file_hash(best)!=result['selected_best']['checkpoint_sha256']):
        raise ValueError('actual native ResNet selected-best file differs; last fallback forbidden')
    score=float(native['best_model_score']);selected=_trusted_checkpoint(best);carry=selected['resnet_fit']['records']
    _validate_checkpoint(selected,identity,carry)
    if (type(result['selected_best']['score'])is not float or not math.isfinite(score)
        or score!=summary['minimum_val_mae']or result['selected_best']['score']!=score
        or carry!=records[:len(carry)]or carry[-1]['validation']['val_mae']!=score
        or result['selected_best']['completed_epochs']!=len(carry)or result['selected_best']['validated']is not True):
        raise ValueError('native ResNet validation minimum/selected epoch differs')
    early=_native_callback(last,'EarlyStopping{')
    done=resnet_stage_complete(policy,records,early,train_samples=identity['train_samples'],val_samples=identity['val_samples'])
    if policy['scope']!='cpu_engineering'and (not identity['cohort']['registered_full_legacy']or identity['cohort']['counts']!=FULL_COUNTS):
        raise ValueError('actual complete registered ResNet cohort required')
    full=done and policy['scope']=='full'
    if (result['stage_complete']is not done or result['full_training_terminated']is not full or(require_full and not full)
        or result['final_model_state_sha256']!=records[-1]['model_sha256']
        or result['train_samples_per_epoch']!=[r['train_samples']for r in records]
        or result['val_samples_per_epoch']!=[None if r['validation']is None else r['validation']['samples']for r in records]):
        raise ValueError('complete native ResNet termination/coverage evidence differs')
    _verify_resume(output,identity,result,records)
    return result


def fit_resnet(cohort,output,*,scope='full',resume=False,stop_after_epoch=None,cohort_receipt=None):
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
    if type(resume)is not bool:raise ValueError('resume requires explicit boolean')
    if type(cohort)is not ResnetCohort:raise ValueError('actual admitted original ResNet cohort required')
    policy=resnet_fit_policy(cohort.upstream,cohort.config,scope=scope);cohort.check(require_full=scope!='cpu_engineering')
    if stop_after_epoch is not None and(scope=='full'or type(stop_after_epoch)is not int or stop_after_epoch!=2):
        raise ValueError('only complete engineering epoch2 can explicitly pause this four-epoch protocol')
    receipt_path,binding=_cohort_receipt(cohort,cohort_receipt,required=scope=='full')
    device=_runtime_device(policy);output=_safe(output,exists=resume);_space(output.parent)
    if not resume:output.mkdir(mode=0o700,exist_ok=False)
    lock=os.open(output/'fit.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        checkpoint=None;identity=None;resume_evidence=None
        if resume:
            identity=_json(output/'identity.json');_validate_identity(identity)
            if identity['policy']!=policy or identity['cohort']!=cohort.identity or identity['cohort_receipt']!=binding:
                raise ValueError('owned ResNet resume source/config/cohort differs')
            if(output/'fit-result.json').exists()and read_resnet_fit(output,require_full=False)['stage_complete']:
                raise ValueError('complete ResNet fit cannot be retrained or reselected')
            checkpoint=_trusted_checkpoint(output/'checkpoints/last.ckpt');carry=checkpoint['resnet_fit']['records']
            _validate_checkpoint(checkpoint,identity,carry)
            prefix=_epoch_bytes(carry,identity['sha256']);path=output/'epochs.jsonl';raw=path.read_bytes()
            if not raw.startswith(prefix):raise ValueError('owned native ResNet journal carry differs')
            if raw!=prefix:
                with(output/('uncommitted-epochs-'+uuid.uuid4().hex+'.jsonl')).open('xb')as f:f.write(raw[len(prefix):])
                with path.open('wb')as f:f.write(prefix);f.flush();os.fsync(f.fileno())
            parent=output/'parents'/('parent-'+uuid.uuid4().hex+'.ckpt');parent.parent.mkdir(exist_ok=True)
            shutil.copyfile(output/'checkpoints/last.ckpt',parent)
            parent_hash=_file_hash(parent)
            if parent_hash!=_file_hash(output/'checkpoints/last.ckpt'):raise ValueError('native resume parent copy differs')
            observations=output/'resume-observations';observations.mkdir(exist_ok=True)
            observation=observations/(uuid.uuid4().hex+'.json')
            resume_evidence=dict(parent_file=str(parent.relative_to(output)),parent_checkpoint_sha256=parent_hash,
                observation_file=str(observation.relative_to(output)))
        prepared=build_resnet_runtime(cohort.upstream,cohort.config,**({}if checkpoint is None else
            dict(state=checkpoint['state_dict'],expected_initialization=identity['construction'])))
        loaders=native_resnet_loaders(cohort,scope=scope)
        train_samples=len(loaders['train'].dataset);val_samples=len(loaders['val'].dataset)
        if identity is None:
            identity=_sealed(dict(format=FORMAT,owner=uuid.uuid4().hex,upstream=str(cohort.upstream),policy=policy,
                cohort=cohort.identity,cohort_receipt=binding,construction=prepared.identity,
                native_hyperparameters=_factory_hyperparameters(cohort.config),
                fit_sources_sha256=_sources(),native_loop_sources_sha256=_loop_sources(),
                train_samples=train_samples,val_samples=val_samples,scientific_eligible=False,full_paper_complete=False))
            _write(output/'identity.json',identity)
            with(output/'epochs.jsonl').open('xb'):pass
        elif identity['train_samples']!=train_samples or identity['val_samples']!=val_samples:
            raise ValueError('actual original ResNet resume loader populations differ')
        witness=ResnetFitWitness(prepared,cohort,identity,output,stop_after_epoch,receipt_path)
        from egoemg._train_utils import build_callbacks
        callbacks=build_callbacks(cohort.config)+[witness]
        for cb in callbacks:
            if isinstance(cb,ModelCheckpoint):cb.dirpath=str(output/'checkpoints')
        kwargs=OmegaConf.to_container(cohort.config.trainer,resolve=True)
        kwargs.update(devices=1,accelerator=device['accelerator'],precision=policy['precision'],max_epochs=policy['max_epochs'],
            num_sanity_val_steps=policy['num_sanity_val_steps'],enable_progress_bar=False,callbacks=callbacks,
            logger=storage.CSVLogger(save_dir=str(output),name='logs',version=uuid.uuid4().hex))
        trainer=storage.pl.Trainer(**kwargs)
        native=next(c for c in trainer.callbacks if isinstance(c,ModelCheckpoint))
        if trainer.check_val_every_n_epoch!=2 or native._should_save_on_train_epoch_end(trainer)is not False:
            raise ValueError('original ResNet validation/checkpoint event cadence differs')
        connector=trainer._checkpoint_connector;original=connector.restore_training_state
        def observe_restore():
            value=original()
            if checkpoint is not None:
                actual=_snap(prepared.module,trainer);expected=_checkpoint_snap(checkpoint)
                witness.exact_restore={k:actual[k]==expected[k]for k in expected}
                if not all(witness.exact_restore.values()):raise ValueError('native six-domain ResNet restore differs: '+str(witness.exact_restore))
                s=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
                _write(observation,dict(identity_sha256=identity['sha256'],parent_checkpoint_sha256=parent_hash,
                    expected=expected,observed=actual,process=dict(pid=os.getpid(),start_ticks=s[19],uid=os.getuid())))
                resume_evidence['observation_sha256']=_file_hash(observation)
            return value
        connector.restore_training_state=observe_restore;start=time.monotonic()
        try:trainer.fit(prepared.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'],
                ckpt_path=None if checkpoint is None else str(output/'checkpoints/last.ckpt'))
        finally:connector.restore_training_state=original
        cohort.check(require_full=scope!='cpu_engineering');witness.check();prepared.check()
        early=next(c for c in trainer.callbacks if isinstance(c,EarlyStopping))
        done=resnet_stage_complete(policy,witness.records,early.state_dict(),train_samples=train_samples,val_samples=val_samples)
        best=Path(native.best_model_path);selected=_trusted_checkpoint(best);_,chain=_read_epochs(output/'epochs.jsonl',identity['sha256'])
        result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,
            completed_epochs=witness.completed_epochs,optimizer_executions=witness.steps,
            stage_complete=done,full_training_terminated=done and scope=='full',scientific_eligible=False,full_paper_complete=False,
            resume_start_exact=witness.exact_restore,resume_evidence=resume_evidence,
            final_model_state_sha256=vision_state_sha256(prepared.module.state_dict()),
            last_checkpoint_sha256=_file_hash(output/'checkpoints/last.ckpt'),epoch_journal_chain_sha256=chain,
            selected_best=dict(file=best.name,score=float(native.best_model_score),checkpoint_sha256=_file_hash(best),
                completed_epochs=selected['resnet_fit']['completed_epochs'],validated=True),
            train_samples_per_epoch=[r['train_samples']for r in witness.records],
            val_samples_per_epoch=[None if r['validation']is None else r['validation']['samples']for r in witness.records],
            native_checkpoint_boundary=BOUNDARY,cuda_initialized=torch.cuda.is_initialized(),runtime_device=device,
            elapsed_fit_seconds=time.monotonic()-start,native_loop_sources_sha256=identity['native_loop_sources_sha256'],
            rng_restore_contract='exact six domains after original restore_training_state, before native iterator reconstruction',
            callback_restore_policy='native fields except ModelCheckpoint.current_score, which native PL never restores',
            bitwise_training_trajectory_guaranteed=False,performance_is_paper_gpu_hours=False))
        _write(output/'fit-result.json',result,replace=resume)
        read_resnet_fit(output,require_full=scope=='full'and done)
        return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for action in ('fit','admit'):
        p=sub.add_parser(action)
        for name in ('data','crops','upstream','output'):p.add_argument('--'+name,required=True,type=Path)
        p.add_argument('--variant',required=True,choices=('resnet18','resnet50','resnet152'))
        if action=='fit':
            p.add_argument('--scope',choices=SCOPES,default='full');p.add_argument('--resume',action='store_true')
            p.add_argument('--stop-after-epoch',type=int);p.add_argument('--cohort-receipt',type=Path)
        else:p.add_argument('--engineering',action='store_true')
    p=sub.add_parser('verify');p.add_argument('--fit',required=True,type=Path);p.add_argument('--engineering',action='store_true')
    a=parser.parse_args()
    if a.action=='verify':value=read_resnet_fit(a.fit,require_full=not a.engineering)
    else:
        if a.action=='fit'and a.scope=='full'and a.cohort_receipt is None:
            parser.error('full original ResNet fit requires --cohort-receipt before input admission')
        full=not a.engineering if a.action=='admit'else a.scope!='cpu_engineering'
        expected=_json(a.cohort_receipt)if a.action=='fit'and a.cohort_receipt is not None else None
        cohort=admit_resnet_cohort(a.data,a.crops,upstream=a.upstream,config=compose_resnet(a.upstream,a.variant),
            expected=expected,require_full=full)
        if a.action=='admit':
            output=_safe(a.output,exists=False);_space(output.parent);cohort.check(require_full=full)
            _write(output,cohort.identity);value=cohort.identity
        else:value=fit_resnet(cohort,a.output,scope=a.scope,resume=a.resume,
            stop_after_epoch=a.stop_after_epoch,cohort_receipt=a.cohort_receipt)
    print(__import__('json').dumps(dict(format=value['format'],sha256=value['sha256'],scientific_eligible=value['scientific_eligible'])))


if __name__=='__main__':main()
