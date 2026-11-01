"""Original RN18/EMGFormer-S fusion epochs and native complete-state recovery.

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
from .rn18_fusion_literal import compose_rn18_fusion,rn18_fusion_protocol
from .rn18_fusion_runtime import EngineeringRn18FusionInitializers,engineering_rn18_fusion_initializers
from .rn18_fusion_trained_handoff import TrainedRn18FusionInitializers,read_rn18_fusion_initializers
from .rn18_fusion_cohort import admit_rn18_fusion_cohort,FULL_COUNTS
from .rn18_fusion_runtime import build_rn18_fusion_runtime
from .vision_literal_runtime import vision_state_sha256 as fusion_state_sha256
from .resnet_train import _factory_hyperparameters,_plain,_validate_rng
from .vision_literal_runtime import _schema
from .vision_literal_check import _native_restored_callbacks
from .vision_literal_train import (_json,_write,_sealed,_check_seal,_read_epochs,_epoch_bytes,_native_callback,
    _adam_steps,_loop_sources,_loop_paths)

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_rn18_fusion_native_fit_v1'
SCOPES=('full','cpu_engineering','cuda_engineering')
LOCAL_SOURCES=('egoemg_repro/rn18_fusion_train.py','egoemg_repro/rn18_fusion_runtime.py',
    'egoemg_repro/rn18_fusion_cohort.py','egoemg_repro/rn18_fusion_data.py',
    'egoemg_repro/rn18_fusion_trained_handoff.py','egoemg_repro/rn18_fusion_vision_producer.py',
    'egoemg_repro/classic_emg_train.py','egoemg_repro/vision_literal_runtime.py',
    'egoemg_repro/vision_literal_train.py','egoemg_repro/vision_literal_check.py','egoemg_repro/resnet_train.py')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported fusion fit source changed')
    return {name:_file_hash(ROOT/name) for name in LOCAL_SOURCES}


def rn18_fusion_fit_policy(upstream,config,*,scope='full'):
    if type(scope) is not str or scope not in SCOPES:raise ValueError('explicit supported fusion scope required')
    protocol=rn18_fusion_protocol(upstream,config);full=scope=='full';cpu=scope=='cpu_engineering'
    callbacks=protocol['callbacks'];checkpoint=callbacks[0]
    if ([c['_target_'] for c in callbacks]!=['pytorch_lightning.callbacks.ModelCheckpoint',
        'pytorch_lightning.callbacks.LearningRateMonitor']):raise ValueError('original fusion checkpoint and LR monitor required')
    return _sealed(dict(format='original_rn18_fusion_fit_policy_v1',scope=scope,protocol_sha256=protocol['sha256'],
        max_epochs=150 if full else 2,precision='32-true' if cpu else 'bf16-mixed',
        batch_size_by_role={r:2 if cpu else 200 for r in ('train','val','test')},
        gradient_clip_val=protocol['trainer']['gradient_clip_val'],scheduler_T_max=150,scheduler_eta_min=1e-5,
        monitor=checkpoint['monitor'],monitor_mode=checkpoint['mode'],save_top_k=checkpoint['save_top_k'],save_last=checkpoint['save_last'],
        early_stopping=False,num_sanity_val_steps=2 if full else 0,engineering_train_samples=None if full else 4 if cpu else 400,
        native_global_loader_rng=True,workers=0 if cpu else 10,
        workers_by_role={r:0 if cpu else 10 for r in ('train','val','test')},
        loader=dict(pin_memory=not cpu,persistent_workers=not cpu,prefetch_factor=None if cpu else 2),
        optimizer_class='torch.optim.adam.Adam',optimizer_lr=.0006,optimizer_weight_decay=.0001,
        check_val_every_n_epoch=1,single_rank=True,scientific_eligible=False,full_paper_complete=False,
        logger='original TensorBoardLogger; owned output path/version only',bitwise_training_trajectory_guaranteed=False))


def rn18_fusion_stage_complete(policy,records):
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
    files={}
    for role,(path,_) in zip(('vision','emg'),initializers._assets):
        files[role+'_checkpoint']=str(path)
        files[role+'_sha256']=initializers.identity['files'][role]['sha256']
    return dict(identity=copy.deepcopy(initializers.identity),files=files,
        producers=copy.deepcopy(initializers.producer_sources)if type(initializers)is TrainedRn18FusionInitializers else None)


def _data_locations(cohort):
    return dict(data=str(cohort.datasets['train'].store.root),crops=str(cohort.datasets['train'].views[0].crops),upstream=str(cohort.upstream))


def _check_policy(policy,cohort):
    _check_seal(policy,'original_rn18_fusion_fit_policy_v1')
    scope=policy['scope'];cpu=scope=='cpu_engineering'
    if(type(scope)is not str or scope not in SCOPES or policy['protocol_sha256']!=cohort['protocol_sha256']
        or policy['max_epochs']!=(150 if scope=='full'else 2)
        or policy['batch_size_by_role']!={r:2 if cpu else 200 for r in ('train','val','test')}
        or policy['workers_by_role']!={r:0 if cpu else 10 for r in ('train','val','test')}
        or policy['loader']!=dict(pin_memory=not cpu,persistent_workers=not cpu,prefetch_factor=None if cpu else 2)
        or policy['optimizer_class']!='torch.optim.adam.Adam'or policy['optimizer_lr']!=.0006
        or policy['optimizer_weight_decay']!=.0001 or policy['scheduler_T_max']!=150 or policy['scheduler_eta_min']!=1e-5
        or policy['engineering_train_samples']!=(None if scope=='full'else 4 if cpu else 400)
        or policy['precision']!=('32-true'if cpu else'bf16-mixed')or policy['gradient_clip_val']!=1
        or policy['early_stopping']is not False or policy['check_val_every_n_epoch']!=1
        or policy['num_sanity_val_steps']!=(2 if scope=='full'else 0)
        or policy['save_top_k']!=3 or policy['save_last']is not True or policy['monitor']!='val_mae'or policy['monitor_mode']!='min'):
        raise ValueError('exact original RN18 fusion budget/optimizer/loader policy required')


def _verify_recorded_inputs(identity,cohort=None,initializers=None,*,require_full):
    from omegaconf import OmegaConf
    if (cohort is None)!=(initializers is None):raise ValueError('both registered RN18 inputs required together')
    descriptor=identity['initializer_descriptor'];locations=identity['data_locations']
    if cohort is None:
        cfg=OmegaConf.create(identity['saved_config'])
        cohort=admit_rn18_fusion_cohort(locations['data'],locations['crops'],upstream=locations['upstream'],config=cfg,
            expected=identity['cohort'],require_full=identity['policy']['scope']!='cpu_engineering')
        saved=descriptor['identity'];producers=descriptor['producers']
        if saved['scope']=='full':
            if type(producers)is not dict or set(producers)!={'vision_fit','emg_fit','output','vision_source_root','emg_source_root'}:
                raise ValueError('full RN18 fusion needs actual completed producer locators')
            initializers=read_rn18_fusion_initializers(cohort,**producers)
        elif saved['scope']=='engineering':
            if producers is not None:raise ValueError('engineering branches cannot invent full producer sources')
            initializers=engineering_rn18_fusion_initializers(cohort.upstream,cohort.config,**descriptor['files'])
        else:raise ValueError('unknown RN18 initializer scope')
    cohort.check(require_full=identity['policy']['scope']!='cpu_engineering');initializers.check(require_full=require_full)
    _check_policy(identity['policy'],cohort.identity)
    if(cohort.identity!=identity['cohort']or _data_locations(cohort)!=locations or _initializer_descriptor(initializers)!=descriptor
        or(type(initializers)is TrainedRn18FusionInitializers and initializers._cohort is not cohort)
        or rn18_fusion_fit_policy(cohort.upstream,cohort.config,scope=identity['policy']['scope'])!=identity['policy']
        or OmegaConf.to_container(cohort.config,resolve=True)!=identity['saved_config']):
        raise ValueError('RN18 actual data/producer/config binding differs')
    init=identity['construction']
    if init['initializers_identity_sha256']!=descriptor['identity']['sha256']:
        raise ValueError('native construction binds different RN18 initializers')
    for role in ('vision','emg'):
        if init['initialization'][role+'_file']['sha256']!=descriptor['identity']['files'][role]['sha256']:
            raise ValueError('RN18 construction source checkpoint differs')
    expected=len(cohort.datasets['train'])
    if identity['policy']['scope']!='full':expected=min(expected,identity['policy']['engineering_train_samples'])
    if identity['train_samples']!=expected:raise ValueError('RN18 actual training population differs')
    return cohort,initializers


def _fit_loaders(cohort,policy):
    from torch.utils.data import Subset
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cohort.config);cpu=policy['scope']=='cpu_engineering'
    if cpu:
        dm.batch_size=2;dm.num_workers=dm.eval_num_workers=0;dm.pin_memory=False;dm.persistent_workers=False;dm.prefetch_factor=None
    samples=len(cohort.datasets['train'])
    if policy['scope']!='full':samples=min(samples,policy['engineering_train_samples'])
    loaders={r:dm._make_loader(Subset(ds,list(range(samples)))if r=='train'and samples<len(ds)else ds,
        shuffle=r=='train',num_workers=policy['workers_by_role'][r])for r,ds in cohort.datasets.items()}
    for role,l in loaders.items():
        if(l.batch_size!=policy['batch_size_by_role'][role]or l.num_workers!=policy['workers_by_role'][role]
            or l.pin_memory is not policy['loader']['pin_memory']or l.persistent_workers is not policy['loader']['persistent_workers']
            or l.prefetch_factor!=policy['loader']['prefetch_factor']or l.generator is not None or l.drop_last):
            raise ValueError('actual native RN18 loader flags differ')
    return loaders,samples


def _close_loaders(loaders):
    for loader in loaders.values():
        iterator=getattr(loader,'_iterator',None)
        if iterator is None:continue
        workers=list(getattr(iterator,'_workers',[]));iterator._shutdown_workers()
        for worker in workers:
            worker.join(timeout=15)
            if worker.is_alive():raise ValueError('owned RN18 loader worker still live')
        loader._iterator=None


def _parameter_groups(module):
    names={id(p):n for n,p in module.named_parameters()}
    groups=[dict(name=k,parameter_names=[names[id(p)]for p in ps if p.requires_grad])for k,ps in module._params_by_component().items()]
    groups=[g for g in groups if g['parameter_names']]
    flat=[n for g in groups for n in g['parameter_names']]
    if len(set(flat))!=len(names)or set(flat)!=set(names.values())or not all(p.requires_grad for p in module.parameters()):
        raise ValueError('every native RN18 parameter must remain unfrozen and grouped exactly once')
    if not any(n.startswith('model.temporal_attn.')for n in groups[-1]['parameter_names']):
        raise ValueError('native temporal attention parameter group missing')
    return groups


def rn18_fusion_optimizer_identity(prepared,optimizer,scheduler):
    import torch
    if type(optimizer)is not torch.optim.Adam or type(scheduler)is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError('native RN18 Adam/cosine required')
    expected=_parameter_groups(prepared.module);names={id(p):n for n,p in prepared.module.named_parameters()}
    actual=[dict(name=g['name'],parameter_names=[names[id(p)]for p in g['params']])for g in optimizer.param_groups]
    if(actual!=expected or any(g['initial_lr']!=.0006 or g['weight_decay']!=.0001 or tuple(g['betas'])!=(.9,.999)
        or g['eps']!=1e-8 or g['amsgrad']for g in optimizer.param_groups)or scheduler.T_max!=150 or scheduler.eta_min!=1e-5):
        raise ValueError('native RN18 optimizer all-parameter groups/options differ')
    return expected


def _normalized_hyperparameters(value):
    value=copy.deepcopy(_plain(value))
    value['module_conf']['vision_pretrained_checkpoint']='<verified RN18 branch or saved-state extraction>'
    value['pretrained_emg_checkpoint']='<verified EMG branch or self-contained restore>'
    return value


def _check_checkpoint_domains(checkpoint,identity,epochs,steps):
    import torch
    try:
        packet=checkpoint['rn18_fusion_fit'];schema=identity['construction']['model_schema']
        if(packet['optimizer_class']!='torch.optim.adam.Adam'or checkpoint['epoch']!=epochs-1
            or packet['format']!=FORMAT or _normalized_hyperparameters(checkpoint['hyper_parameters'])!=identity['native_hyperparameters']):
            raise ValueError('RN18 native factory objective or optimizer class differs')
        mode=packet['initialization_mode'];actual=_plain(checkpoint['hyper_parameters'])
        if mode=='strict_self_contained_restore':
            v=Path(actual['module_conf']['vision_pretrained_checkpoint'])
            if actual['pretrained_emg_checkpoint']is not None or v.name!='saved-vision.pt'or not v.parent.name.startswith('egoemg-rn18-fusion-restore-'):
                raise ValueError('self-contained RN18 restore factory inputs differ')
        elif mode in('fresh_engineering_branch_initialization','fresh_trained_branch_initialization'):
            files=identity['initializer_descriptor']['files']
            if(actual['module_conf']['vision_pretrained_checkpoint']!=files['vision_checkpoint']
                or actual['pretrained_emg_checkpoint']!=files['emg_checkpoint']
                or(mode=='fresh_trained_branch_initialization')!=(identity['initializer_descriptor']['identity']['scope']=='full')):
                raise ValueError('fresh RN18 factory source branches differ')
        else:raise ValueError('unknown RN18 construction mode')
        if len(checkpoint['optimizer_states'])!=1 or len(checkpoint['lr_schedulers'])!=1:raise ValueError('one native Adam/cosine required')
        native_parameters={n for n in schema if not n.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
        if(set(identity['parameter_names'])!=native_parameters or len(identity['parameter_names'])!=len(native_parameters)
            or sum(math.prod(schema[n]['shape'])for n in native_parameters)!=identity['construction']['parameters']):
            raise ValueError('complete native RN18 learned-parameter inventory required')
        from egoemg.lightning import EmgPredictionModule
        by_component={k:[]for k in EmgPredictionModule._COMPONENT_PREFIX_MAP};unassigned=[]
        for name in identity['parameter_names']:
            for key,prefixes in EmgPredictionModule._COMPONENT_PREFIX_MAP.items():
                if any(name.startswith('model.'+prefix)for prefix in prefixes):by_component[key].append(name);break
            else:unassigned.append(name)
        if unassigned:by_component['_unassigned']=unassigned
        expected_groups=[dict(name=k,parameter_names=v)for k,v in by_component.items()if v]
        if identity['parameter_groups']!=expected_groups:raise ValueError('native component parameter inventory/order differs')
        witness=checkpoint['callbacks']['FusionFitWitness']
        if witness!=dict(steps=steps,completed_epochs=epochs,records=packet['records']):
            raise ValueError('native saved RN18 fit callback state differs')
        opt=checkpoint['optimizer_states'][0];sched=checkpoint['lr_schedulers'][0];groups=opt['param_groups'];expected=identity['parameter_groups']
        if len(groups)!=len(expected):raise ValueError('native RN18 parameter groups missing')
        ids=[];names=[];lr=1e-5+(.0006-1e-5)*(1+math.cos(math.pi*epochs/150))/2
        for g,e in zip(groups,expected):
            groupids=list(range(len(ids),len(ids)+len(e['parameter_names'])))
            if(g['name']!=e['name']or g['params']!=groupids or list(g['betas'])!=[.9,.999]or g['eps']!=1e-8
                or g['weight_decay']!=.0001 or g['initial_lr']!=.0006 or g['amsgrad']is not False
                or any(g[k]is not False for k in('maximize','capturable','differentiable'))
                or not math.isclose(g['lr'],lr,rel_tol=1e-10,abs_tol=1e-14)):
                raise ValueError('saved native Adam coverage/options differ')
            ids.extend(groupids);names.extend(e['parameter_names'])
        if len(set(names))!=len(names)or set(names)!=set(identity['parameter_names'])or set(opt['state'])!=set(ids):
            raise ValueError('native RN18 optimizer parameter or moment coverage differs')
        for index,name in zip(ids,names):
            state=opt['state'][index];step=state['step']
            if(set(state)!={'step','exp_avg','exp_avg_sq'}or not isinstance(step,torch.Tensor)
                or step.dtype!=torch.float32 or step.ndim!=0 or float(step)!=steps):raise ValueError('native Adam saved update counter differs')
            for key in('exp_avg','exp_avg_sq'):
                v=state[key]
                if(not isinstance(v,torch.Tensor)or list(v.shape)!=schema[name]['shape']or str(v.dtype)!=schema[name]['dtype']
                    or not bool(torch.isfinite(v).all())):raise ValueError('native Adam moment shape/dtype/finite state differs')
        if(sched['T_max']!=150 or sched['eta_min']!=1e-5 or sched['last_epoch']!=epochs or sched['_step_count']!=epochs+1
            or sched['base_lrs']!=[.0006]*len(groups)or len(sched['_last_lr'])!=len(groups)
            or any(not math.isclose(x,lr,rel_tol=1e-10,abs_tol=1e-14)for x in sched['_last_lr'])):
            raise ValueError('native cosine saved epoch/learning rate differs')
        loops=checkpoint['loops']['fit_loop'];batches=math.ceil(identity['train_samples']/identity['policy']['batch_size_by_role']['train'])
        if(loops['epoch_loop.state_dict']['_batches_that_stepped']!=steps
            or loops['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']!=dict(ready=steps,completed=steps)
            or loops['epoch_loop.scheduler_progress']['total']!=dict(ready=epochs,completed=epochs)):
            raise ValueError('native loop optimizer/scheduler counters differ')
        for which,count in(('total',steps),('current',batches)):
            if loops['epoch_loop.batch_progress'][which]!={k:count for k in('ready','started','processed','completed')}:
                raise ValueError('native loop batch coverage differs')
        expected_epoch=dict(ready=epochs,started=epochs,processed=epochs,completed=epochs-1)
        if any(loops['epoch_progress'][which]!=expected_epoch for which in('total','current')):
            raise ValueError('native train-epoch-end boundary differs')
        _validate_rng(packet['rng'],cpu=identity['policy']['scope']=='cpu_engineering')
    except(KeyError,TypeError,IndexError,AttributeError,RuntimeError)as exc:
        raise ValueError('complete RN18 Adam/scheduler/loop/RNG checkpoint required')from exc


def _verify_resume(output,identity,result,records):
    evidence=result['resume_evidence']
    if evidence is None:
        if result['resume_start_exact']is not None:raise ValueError('fresh RN18 fit cannot invent resume state')
        return
    parent_path=output/evidence['parent_file'];proof_path=output/evidence['observation_file']
    if(parent_path.parent!=output/'parents'or proof_path.parent!=output/'resume-observations'
        or parent_path.is_symlink()or proof_path.is_symlink()or _file_hash(parent_path)!=evidence['parent_checkpoint_sha256']
        or _file_hash(proof_path)!=evidence['observation_sha256']):raise ValueError('preserved RN18 parent/restore proof differs')
    import torch
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);packet=parent['rn18_fusion_fit'];carry=packet['records']
    if(packet['identity']!=identity or carry!=records[:len(carry)]or not 0<len(carry)<len(records)
        or packet['completed_epochs']!=len(carry)or packet['epoch_boundary']is not True):raise ValueError('RN18 parent is not complete strict prefix')
    _check_checkpoint_domains(parent,identity,len(carry),carry[-1]['global_step'])
    if fusion_state_sha256(parent['state_dict'])!=carry[-1]['model_sha256']:raise ValueError('RN18 parent model differs')
    proof=_json(proof_path);expected=_checkpoint_snap(parent)
    if(proof['identity_sha256']!=identity['sha256']or proof['parent_checkpoint_sha256']!=evidence['parent_checkpoint_sha256']
        or proof['expected']!=expected or proof['observed']!=expected or result['resume_start_exact']!={k:True for k in expected}
        or result['resume_validation_exact']is not True):raise ValueError('RN18 restored six-domain proof differs')


def _snap(module,trainer):
    d=storage.state_digest
    return dict(model=d(module.state_dict()),optimizer=d([o.state_dict() for o in trainer.optimizers]),
        scheduler=d([x.scheduler.state_dict() for x in trainer.lr_scheduler_configs]),
        callbacks=d(_native_restored_callbacks({cb.state_key:cb.state_dict() for cb in trainer.callbacks if cb.state_dict()})),
        loops=d(trainer.fit_loop.state_dict()),rng=d(storage._rng({})))


def _checkpoint_snap(c):
    d=storage.state_digest
    return dict(model=d(c['state_dict']),optimizer=d(c['optimizer_states']),scheduler=d(c['lr_schedulers']),
        callbacks=d(_native_restored_callbacks(c['callbacks'])),loops=d(c['loops']['fit_loop']),rng=d(c['rn18_fusion_fit']['rng']))


def _batch(cohort,policy,batch,role,seen,limit):
    import torch
    ids=batch['rn18_fusion_sample_index']
    if ids.dtype!=torch.int64 or ids.ndim!=1:raise ValueError('actual fusion consumption indices required')
    ids=ids.detach().cpu().numpy();n=len(ids);size=policy['batch_size_by_role'][role]
    if (n!=min(size,limit-len(seen)) or np.any(ids<0) or np.any(ids>=limit) or len(np.unique(ids))!=n or
        any(int(i) in seen for i in ids)):raise ValueError('fusion batch is repeated, partial or out of range')
    if role=='val' and ids.tolist()!=list(range(len(seen),len(seen)+n)):raise ValueError('native fusion validation order differs')
    if (batch['emg'].shape!=(n,8,12000) or batch['vision_img'].shape!=(n,3,256,256) or
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
            int(batch['rn18_fusion_repeat_index'][j])!=repeat or int(batch['rn18_fusion_unique_sample_index'][j])!=unique or
            int(batch['rn18_fusion_window_index'][j])!=window or not 0<=offset<=max(0,w['jitter_upper']-1) or
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
                self.pose_parts.append({k:v.detach().cpu().clone()for k,v in dict(pred=pred,target=target,mask=mask).items()})
                self.parts.append(storage.state_digest(dict(pred=pred,target=target,mask=mask,emg=batch['emg'],
                    sample_index=batch['rn18_fusion_sample_index'],vision_img=batch['vision_img'],vision_valid_mask=batch['vision_valid_mask'])))
            return out
        module.forward=forward

    def on_validation_start(self,trainer,module):
        if not hasattr(self,'original_forward'):self._observe_forward(module)

    def on_validation_epoch_start(self,trainer,module):self.parts=[];self.pose_parts=[];self.val_seen=set()

    def on_validation_batch_start(self,trainer,module,batch,batch_idx,dataloader_idx=0):
        if not trainer.sanity_checking:
            self.cohort.guard.check();self.phase='val'
            _batch(self.cohort,self.policy,batch,'val',self.val_seen,len(self.cohort.datasets['val']))

    def on_validation_end(self,trainer,module):
        self.phase=None
        if trainer.sanity_checking:return
        score=float(trainer.callback_metrics['val_mae'])
        if len(self.val_seen)!=len(self.cohort.datasets['val']) or not math.isfinite(score):raise ValueError('incomplete/nonfinite fusion validation')
        import torch
        pose=storage.state_digest({k:torch.cat([p[k]for p in self.pose_parts])for k in ('pred','target','mask')})
        self.validation=dict(samples=len(self.val_seen),val_mae=score,prediction_sha256=_hash(self.parts),pose_sha256=pose)
        self.parts=[];self.pose_parts=[]


class FusionFitWitness(ValidationWitness):
    def __init__(self,prepared,cohort,initializers,identity,output,stop_after,receipt_path):
        super().__init__(cohort,identity['policy']);self.prepared=prepared;self.initializers=initializers;self.identity=identity;self.output=output
        self.stop_after=stop_after;self.steps=0;self.completed_epochs=0;self.records=[];self.pending=None;self.exact_restore=None
        self.asset_guard=_StatGuard([x[0]for x in initializers._assets]);self.source_guard=_StatGuard([*[ROOT/name for name in identity['fit_sources_sha256']],*_loop_paths().values(),
            *([] if receipt_path is None else [receipt_path])])

    def state_dict(self):return dict(steps=self.steps,completed_epochs=self.completed_epochs,records=copy.deepcopy(self.records))

    def load_state_dict(self,state):
        if set(state)!={'steps','completed_epochs','records'}:raise ValueError('fusion callback schema differs')
        self.steps=state['steps'];self.completed_epochs=state['completed_epochs'];self.records=copy.deepcopy(state['records'])

    def check(self):self.cohort.guard.check();self.asset_guard.check();self.source_guard.check();storage._space(self.output)

    def on_load_checkpoint(self,trainer,module,checkpoint):
        p=checkpoint['rn18_fusion_fit']
        if p['identity']!=self.identity or p['epoch_boundary'] is not True:raise ValueError('owned complete fusion epoch required')
        self.pending=checkpoint

    def on_fit_start(self,trainer,module):
        if self.pending is not None:storage._restore_rng(self.pending['rn18_fusion_fit']['rng'],{})
        self._observe_forward(module)
        self.step_hook=trainer.optimizers[0].register_step_post_hook(lambda *args:self._executed())

    def _executed(self):self.steps+=1

    def on_train_start(self,trainer,module):
        rn18_fusion_optimizer_identity(self.prepared,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
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
        self.check();checkpoint['rn18_fusion_fit']=dict(format=FORMAT,identity=self.identity,
            optimizer_class='torch.optim.adam.Adam',initialization_mode=self.prepared.mode,
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


def _read_fit_context(output,*,require_full=True):
    """Reopen all evidence once and retain its actual input-generation guards."""
    if type(require_full)is not bool:raise ValueError('explicit RN18 full requirement must be boolean')
    output=_safe(output);identity=_json(output/'identity.json');result=_json(output/'fit-result.json')
    _check_seal(identity,FORMAT);_check_seal(result,FORMAT)
    if identity['fit_sources_sha256']!=_sources():raise ValueError('RN18 fit source differs')
    paths=[output/n for n in ('identity.json','fit-result.json','epochs.jsonl','checkpoints/last.ckpt')]
    guard=_StatGuard([*paths,*[ROOT/n for n in _sources()],*_loop_paths().values()])
    c,initializers=_verify_recorded_inputs(identity,require_full=identity['policy']['scope']=='full')
    result=_validate_fit(output,identity,result,require_full=require_full,cohort=c,initializers=initializers)
    guard.add(output/'checkpoints'/result['selected_best']['file'])
    for owner in (c,initializers,getattr(initializers,'_vision',None),getattr(initializers,'_emg',None)):
        existing=getattr(owner,'guard',getattr(owner,'_guard',None))
        if existing is not None:
            existing.check()
            for path,stamp in existing.paths.items():
                guard.add(path)
                if guard.paths[path]!=stamp:raise ValueError('RN18 reopened source/input generation differs')
    for path,_ in initializers._assets:guard.add(path)
    if result['resume_evidence']is not None:
        for key in ('parent_file','observation_file'):guard.add(output/result['resume_evidence'][key])
    guard.check();return result,identity,c,initializers,guard


def read_rn18_fusion_fit(output,*,require_full=True):
    return _read_fit_context(output,require_full=require_full)[0]


def _validate_fit(output,identity,result,*,require_full,cohort=None,initializers=None):
    """Validate a candidate against native durable files before publishing it."""
    import torch
    _check_seal(identity,FORMAT);_check_seal(result,FORMAT);_check_seal(identity['policy'],'original_rn18_fusion_fit_policy_v1')
    if (result['identity_sha256']!=identity['sha256'] or identity['fit_sources_sha256']!=_sources() or
        identity['native_loop_sources_sha256']!=_loop_sources() or result['scientific_eligible'] is not False or
        result['full_paper_complete'] is not False):raise ValueError('fusion fit/source/result identity differs')
    _check_receipt(identity);policy=identity['policy']
    if result['scope']!=policy['scope'] or (require_full and policy['scope']!='full'):raise ValueError('engineering fusion is not complete full training')
    _verify_recorded_inputs(identity,cohort,initializers,require_full=policy['scope']=='full');_check_policy(policy,identity['cohort'])
    last_path=output/'checkpoints/last.ckpt'
    if _file_hash(last_path)!=result['last_checkpoint_sha256']:raise ValueError('fusion last checkpoint bytes changed')
    last=torch.load(str(last_path),map_location='cpu',weights_only=False);packet=last['rn18_fusion_fit']
    records,_=_read_epochs(output/'epochs.jsonl',identity['sha256']);samples=identity['train_samples']
    batches=math.ceil(samples/policy['batch_size_by_role']['train']);steps=batches*len(records)
    if (not records or packet['identity']!=identity or packet['records']!=records or packet['epoch_boundary'] is not True or
        packet['completed_epochs']!=len(records) or result['completed_epochs']!=len(records) or
        result['optimizer_executions']!=steps or last['global_step']!=steps):raise ValueError('fusion actual epoch/optimizer/journal carry differs')
    for i,r in enumerate(records):
        if (r['epoch']!=i or type(r['epoch'])is not int or r['train_samples']!=samples or r['train_batches']!=batches or r['optimizer_executions']!=(i+1)*batches or
            r['global_step']!=(i+1)*batches or r['validation']['samples']!=identity['cohort']['counts']['val'] or
            not math.isfinite(r['train_loss']) or not math.isfinite(r['validation']['val_mae'])):
            raise ValueError('partial/nonfinite fusion epoch')
    _check_checkpoint_domains(last,identity,len(records),steps)
    _verify_resume(output,identity,result,records)
    if any(k.startswith('EarlyStopping') for k in last['callbacks']):raise ValueError('fusion cannot use early stopping')
    if (_schema(last['state_dict'])!=identity['construction']['model_schema'] or
        fusion_state_sha256(last['state_dict'])!=records[-1]['model_sha256'] or
        result['final_model_state_sha256']!=records[-1]['model_sha256']):raise ValueError('terminal fusion tensor state differs')
    native=_native_callback(last,'ModelCheckpoint{');best=Path(native['best_model_path']);score=float(native['best_model_score'])
    if (best.parent!=output/'checkpoints' or best.is_symlink() or best.name=='last.ckpt' or
        result['selected_best']!=dict(file=best.name,score=score,checkpoint_sha256=_file_hash(best),validated=True) or
        score!=min(r['validation']['val_mae'] for r in records)):raise ValueError('fusion selected checkpoint differs from actual val best')
    selected=torch.load(str(best),map_location='cpu',weights_only=False);p=selected['rn18_fusion_fit'];epoch=p['completed_epochs']-1
    if (p['identity']!=identity or p['epoch_boundary'] is not True or not 0<=epoch<len(records) or
        p['records']!=records[:epoch+1] or records[epoch]['validation']['val_mae']!=score or
        selected['global_step']!=records[epoch]['global_step'] or _schema(selected['state_dict'])!=identity['construction']['model_schema'] or
        fusion_state_sha256(selected['state_dict'])!=records[epoch]['model_sha256']):raise ValueError('selected fusion state/epoch differs')
    _check_checkpoint_domains(selected,identity,epoch+1,records[epoch]['global_step'])
    done=rn18_fusion_stage_complete(policy,records);full=done and policy['scope']=='full'
    if (result['stage_complete'] is not done or result['full_training_terminated'] is not full or
        result['train_samples_per_epoch']!=[r['train_samples'] for r in records] or
        result['val_samples_per_epoch']!=[r['validation']['samples'] for r in records]):raise ValueError('fusion completion/coverage differs')
    if require_full and (not full or identity['cohort']['registered_full_legacy'] is not True or
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
        read_rn18_fusion_fit(output,require_full=False)
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
    packet=checkpoint['rn18_fusion_fit'];records=packet['records'];policy=identity['policy']
    if len(records)!=policy['max_epochs'] or not rn18_fusion_stage_complete(policy,records):
        raise ValueError('saved fusion budget is not complete')
    native=_native_callback(checkpoint,'ModelCheckpoint{');best=Path(native['best_model_path'])
    if best.parent!=output/'checkpoints' or best.name=='last.ckpt':raise ValueError('saved native best location differs')
    result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=policy['scope'],completed_epochs=len(records),
        optimizer_executions=checkpoint['global_step'],stage_complete=True,full_training_terminated=policy['scope']=='full',
        scientific_eligible=False,full_paper_complete=False,resume_start_exact=None,resume_validation_exact=validation_exact,resume_evidence=None,
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


def fit_rn18_fusion(cohort,initializers,output,*,scope='full',resume=False,stop_after_epoch=None,cohort_receipt=None):
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint
    recovery_started=time.monotonic();policy=rn18_fusion_fit_policy(cohort.upstream,cohort.config,scope=scope)
    if type(initializers)not in(EngineeringRn18FusionInitializers,TrainedRn18FusionInitializers):
        raise ValueError('registered RN18 fusion initializer object required')
    initializers.check(require_full=scope=='full');cohort.check(require_full=scope!='cpu_engineering')
    if type(initializers)is TrainedRn18FusionInitializers and initializers._cohort is not cohort:
        raise ValueError('RN18 initializers bind different actual cohort')
    descriptor=_initializer_descriptor(initializers)
    receipt_path,binding=_receipt(cohort,cohort_receipt,scope=='full')
    if scope=='full' and (cohort.identity['registered_full_legacy'] is not True or cohort.identity['counts']!=FULL_COUNTS):
        raise ValueError('full fusion requires complete registered legacy fold')
    if stop_after_epoch is not None and (scope=='full' or type(stop_after_epoch) is not int or not 1<=stop_after_epoch<=2):
        raise ValueError('bounded fusion epoch pause is engineering only')
    device=_runtime_device(policy);output=_safe(output,exists=resume);storage._space(output.parent)
    if not resume:output.mkdir(mode=0o700,exist_ok=False)
    loaders={}
    lock=os.open(output/'fit.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);checkpoint=None;identity=None;resume_evidence=None
        if resume:
            identity=_json(output/'identity.json');_check_seal(identity,FORMAT)
            if (identity['policy']!=policy or identity['cohort']!=cohort.identity or identity['fit_sources_sha256']!=_sources() or
                identity['native_loop_sources_sha256']!=_loop_sources() or identity['cohort_receipt']!=binding or
                identity['initializer_descriptor']!=descriptor):raise ValueError('fusion resume source/config/cohort differs')
            _historical_result(output,identity)
            checkpoint=torch.load(str(output/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False);p=checkpoint['rn18_fusion_fit']
            if p['identity']!=identity or p['epoch_boundary'] is not True:raise ValueError('owned complete fusion epoch required')
            if (type(p['completed_epochs']) is not int or not 1<=p['completed_epochs']<=policy['max_epochs'] or
                len(p['records'])!=p['completed_epochs']):raise ValueError('saved fusion epoch is outside owned budget')
            carry=_epoch_bytes(p['records'],identity['sha256']);path=output/'epochs.jsonl';raw=path.read_bytes()
            if not raw.startswith(carry):raise ValueError('fusion checkpoint/journal prefix differs')
            if len(raw)>len(carry):
                (output/('uncommitted-epochs-'+uuid.uuid4().hex+'.jsonl')).write_bytes(raw[len(carry):])
                with path.open('wb') as f:f.write(carry);f.flush();os.fsync(f.fileno())
        if checkpoint is not None:
            _check_checkpoint_domains(checkpoint,identity,p['completed_epochs'],checkpoint['global_step'])
        prepared=build_rn18_fusion_runtime(cohort.upstream,cohort.config,**(dict(initializers=initializers)if checkpoint is None else dict(
            state=checkpoint['state_dict'],expected_initialization=identity['construction'])))
        loaders,samples=_fit_loaders(cohort,policy)
        if identity is None:
            identity=_sealed(dict(format=FORMAT,owner=uuid.uuid4().hex,policy=policy,cohort=cohort.identity,cohort_receipt=binding,
                construction=prepared.identity,initializer_descriptor=descriptor,data_locations=_data_locations(cohort),
                saved_config=OmegaConf.to_container(cohort.config,resolve=True),
                native_hyperparameters=_normalized_hyperparameters(_factory_hyperparameters(cohort.config)),
                parameter_names=[n for n,p in prepared.module.named_parameters()],parameter_groups=_parameter_groups(prepared.module),
                fit_sources_sha256=_sources(),native_loop_sources_sha256=_loop_sources(),train_samples=samples,
                scientific_eligible=False,full_paper_complete=False))
            _write(output/'identity.json',identity)
            with (output/'epochs.jsonl').open('xb'):pass
        elif identity['train_samples']!=samples:raise ValueError('resumed fusion population differs')
        validation_exact=None
        if checkpoint is not None:
            replay=_frozen_validation(prepared,cohort,policy,loaders['val'],device)
            validation_exact=replay==checkpoint['rn18_fusion_fit']['records'][-1]['validation']
            if not validation_exact:raise ValueError('fresh fusion frozen validation differs before continuation')
            if checkpoint['rn18_fusion_fit']['completed_epochs']==policy['max_epochs']:
                return _completed_checkpoint_result(output,identity,checkpoint,device,validation_exact,recovery_started)
        if checkpoint is not None:
            parentdir=output/'parents';parentdir.mkdir(exist_ok=True);parent=parentdir/(uuid.uuid4().hex+'.ckpt')
            import shutil
            shutil.copyfile(output/'checkpoints/last.ckpt',parent)
            observedir=output/'resume-observations';observedir.mkdir(exist_ok=True);observation=observedir/(uuid.uuid4().hex+'.json')
            resume_evidence=dict(parent_file=str(parent.relative_to(output)),parent_checkpoint_sha256=_file_hash(parent),
                observation_file=str(observation.relative_to(output)))
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
                _write(observation,dict(identity_sha256=identity['sha256'],parent_checkpoint_sha256=resume_evidence['parent_checkpoint_sha256'],
                    expected=expected,observed=actual))
                resume_evidence['observation_sha256']=_file_hash(observation)
                if not all(witness.exact_restore.values()):raise ValueError('native fusion six-state restore differs: '+str(witness.exact_restore))
            return value
        connector.restore_training_state=restore;started=time.monotonic()
        try:trainer.fit(prepared.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'],
            ckpt_path=None if checkpoint is None else str(output/'checkpoints/last.ckpt'))
        finally:connector.restore_training_state=original
        cohort.check(require_full=scope!='cpu_engineering');initializers.check(require_full=scope=='full');witness.check();prepared.check();native=next(cb for cb in trainer.callbacks if isinstance(cb,ModelCheckpoint));best=Path(native.best_model_path)
        done=rn18_fusion_stage_complete(policy,witness.records)
        result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,completed_epochs=witness.completed_epochs,
            optimizer_executions=witness.steps,stage_complete=done,full_training_terminated=done and scope=='full',
            scientific_eligible=False,full_paper_complete=False,resume_start_exact=witness.exact_restore,resume_validation_exact=validation_exact,resume_evidence=resume_evidence,
            final_model_state_sha256=fusion_state_sha256(prepared.module.state_dict()),last_checkpoint_sha256=_file_hash(output/'checkpoints/last.ckpt'),
            selected_best=dict(file=best.name,score=float(native.best_model_score),checkpoint_sha256=_file_hash(best),validated=True),
            train_samples_per_epoch=[r['train_samples'] for r in witness.records],val_samples_per_epoch=[r['validation']['samples'] for r in witness.records],
            cuda_initialized=torch.cuda.is_initialized(),runtime_device=device,elapsed_fit_seconds=time.monotonic()-started,
            bitwise_training_trajectory_guaranteed=False,
            rng_restore_contract='exact after native restore_training_state, before iterator reconstruction; resumed global loader RNG can diverge from uninterrupted training',
            callback_restore_policy='native fields except ModelCheckpoint.current_score, not restored by native Lightning'))
        _validate_fit(output,identity,result,require_full=scope=='full' and done,cohort=cohort,initializers=initializers)
        _write(output/'fit-result.json',result,replace=resume);read_rn18_fusion_fit(output,require_full=scope=='full' and done)
        return result
    finally:
        try:_close_loaders(loaders)
        finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)



def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('fit')
    for name in ('data','crops','upstream','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--scope',choices=SCOPES,default='full');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epoch',type=int);p.add_argument('--cohort-receipt',type=Path)
    p.add_argument('--branches',required=True,type=Path,help='engineering branch files or registered full producer locators')
    p=sub.add_parser('verify');p.add_argument('--fit',required=True,type=Path);p.add_argument('--engineering',action='store_true')
    a=parser.parse_args()
    if a.action=='verify':value=read_rn18_fusion_fit(a.fit,require_full=not a.engineering)
    else:
        if a.scope=='full'and a.cohort_receipt is None:parser.error('full RN18 fit requires frozen --cohort-receipt')
        cfg=compose_rn18_fusion(a.upstream);expected=None if a.cohort_receipt is None else _json(a.cohort_receipt)
        c=admit_rn18_fusion_cohort(a.data,a.crops,upstream=a.upstream,config=cfg,expected=expected,require_full=a.scope!='cpu_engineering')
        locators=_json(a.branches)
        if set(locators)=={'vision_fit','emg_fit','output','vision_source_root','emg_source_root'}:
            initializers=read_rn18_fusion_initializers(c,**locators)
        else:
            if a.scope=='full':parser.error('full RN18 fit requires independently registered trained producers')
            initializers=engineering_rn18_fusion_initializers(c.upstream,c.config,**locators)
        value=fit_rn18_fusion(c,initializers,a.output,scope=a.scope,resume=a.resume,stop_after_epoch=a.stop_after_epoch,cohort_receipt=a.cohort_receipt)
    print(__import__('json').dumps(dict(format=value['format'],sha256=value['sha256'],stage_complete=value['stage_complete'],scientific_eligible=False)))


if __name__=='__main__':main()
