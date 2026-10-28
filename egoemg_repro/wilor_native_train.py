"""Source-bound WiLoR data and fresh native fit boundaries.

The vision-S companion admits only unchanged point-reader data semantics. It
does not initialize a DINO model or grant WiLoR training/CUDA qualification.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import weakref

from .classic_emg import _activate,_file_hash,_hash
from .wilor_literal import _validated,wilor_protocol

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FULL_COUNTS=dict(train=202020,val=3152,test=8612)
ROLES=('train','val','test')
SCOPES=('full','cuda_engineering','cpu_engineering')
_COHORTS=weakref.WeakSet()
_DATA_FIELDS=('dataset','datamodule','transforms','egoemg_unified_memmap_dir',
    'egoemg_emg_layout','egoemg_emg2pose_channel_indices','egoemg_channel_interpolate',
    'skip_emg_loading','per_episode_crops_dir','vision_num_frames',
    'vision_frame_selection','vision_patch_size','center_frame_eval')


def _sealed(value):
    value=copy.deepcopy(value);value['sha256']=_hash(value);return value


def wilor_policy(*,scope='full'):
    if type(scope) is not str or scope not in SCOPES:raise ValueError('explicit supported WiLoR scope required')
    full=scope=='full';cpu=scope=='cpu_engineering'
    return _sealed(dict(format='native_wilor_fit_policy_v1',scope=scope,max_epochs=30 if full else 2,
        precision='32-true' if cpu else 'bf16-mixed',batch_size_by_role={r:1 if cpu else 18 for r in ROLES},
        engineering_train_samples=None if full else 2 if cpu else 36,engineering_eval_samples=2 if cpu else None,
        full_counts=FULL_COUNTS.copy(),monitor='val_mae',monitor_mode='min',patience=30,
        save_top_k=3,save_last=True,gradient_clip_val=1,scheduler_T_max=30,scheduler_eta_min=1e-5,
        scheduler_behavior='constant_initial_lr_equals_eta_min',num_sanity_val_steps=2 if full else 0,
        native_global_loader_rng=True,num_workers=0,single_rank=True,checkpoint_continuation_supported=False,
        bitwise_training_trajectory_guaranteed=False,scientific_eligible=False,full_paper_complete=False))


def _data_projection(upstream,config):
    from .vision_literal import compose_vision,_validated as validate_vision,vision_protocol
    root,actual=_validated(upstream,config)
    companion=compose_vision(root,'vit_small');_,_,expected=validate_vision(root,companion)
    left={name:actual[name] for name in _DATA_FIELDS};right={name:expected[name] for name in _DATA_FIELDS}
    if _hash(left)!=_hash(right):raise ValueError('WiLoR and original S companion data projection differs')
    return companion,_sealed(dict(format='native_wilor_data_projection_v1',
        companion_role='original vision-S data only; no DINO model/training/CUDA permit',
        compared_fields=list(_DATA_FIELDS),data_fields_sha256=_hash(left),
        wilor_protocol_sha256=wilor_protocol(root,config)['sha256'],
        companion_protocol_sha256=vision_protocol(root,companion)['sha256'],
        scientific_eligible=False,full_paper_complete=False))


def _data_sources(root):
    from .vision_literal_cohort import _sources
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported WiLoR training source changed')
    paths=_sources(root)
    for name in ('egoemg_repro/wilor_native_train.py','egoemg_repro/wilor_literal.py'):
        paths[name]=ROOT/name
    return paths


def _data_identity(common,projection,sources):
    v=common.identity
    registered=(v['metadata_matches_registered_legacy_release'] is True and v['counts']==FULL_COUNTS and
        v['episode_records']==41 and v['repeat_by_role']=={r:2 for r in ROLES})
    return _sealed(dict(format='native_wilor_data_admission_v1',
        scope='complete actual native point data only; no model/GPU/training or paper equivalence',
        projection=projection,common_data=copy.deepcopy(v),sources_sha256=sources,
        counts=v['counts'].copy(),unique_counts=v['unique_counts'].copy(),
        full_payload_content_hashed=v['full_payload_content_hashed'],registered_full_legacy=registered,
        scientific_eligible=False,production_ready=False,full_paper_complete=False))


def _check_readers(common):
    import numpy as np
    from .vision_literal_data import VisionLiteralDataset,_PointView
    from .vision_literal import vision_protocol
    protocol=vision_protocol(common.upstream,common.config)
    for role,ds in common.datasets.items():
        if (type(ds) is not VisionLiteralDataset or ds.role!=role or ds.repeat!=2 or ds.protocol!=protocol or
            len(ds.views)!=2 or ds.frame_mapping_field not in ('image_head_frame_index','image_webcam_frame_index')):
            raise ValueError('actual original WiLoR companion dataset layout changed')
        for hand,view in zip(('left','right'),ds.views):
            if (type(view) is not _PointView or view.target_hand!=hand or view.store is not ds.store or
                view.native is not ds.native or view.vision_patch_size!=256 or view.emg_layout!='emg2pose_interpolate16' or
                not np.array_equal(view.vision_mean,255.*np.array([.485,.456,.406],np.float32)) or
                not np.array_equal(view.vision_std,255.*np.array([.229,.224,.225],np.float32))):
                raise ValueError('actual original WiLoR point-reader normalization/layout changed')


@dataclass(frozen=True,eq=False)
class WilorCohort:
    identity:dict
    common:object
    upstream:Path
    config:object
    _seal:str

    @property
    def datasets(self):return self.common.datasets
    @property
    def partitions(self):return self.common.partitions
    @property
    def guard(self):return self.common.guard

    def check(self,*,require_full=False):
        if type(require_full) is not bool:raise ValueError('full requirement must be boolean')
        if self not in _COHORTS:raise ValueError('actual admitted WiLoR data object required')
        self.common.check();_check_readers(self.common)
        companion,projection=_data_projection(self.upstream,self.config)
        from omegaconf import OmegaConf
        if OmegaConf.to_container(companion,resolve=True)!=OmegaConf.to_container(self.common.config,resolve=True):
            raise ValueError('WiLoR common data companion configuration changed')
        sources={name:_file_hash(path) for name,path in _data_sources(self.upstream).items()}
        expected=_data_identity(self.common,projection,sources)
        if self.identity!=expected or expected['sha256']!=self._seal:raise ValueError('WiLoR outer data identity changed')
        if require_full and not expected['registered_full_legacy']:
            raise ValueError('complete registered full legacy WiLoR data required')
        self.guard.check()


def admit_wilor_cohort(data,crops,*,upstream,config,expected=None,require_full=True,chunk_rows=262144):
    if type(require_full) is not bool:raise ValueError('full requirement must be boolean')
    root,_=_validated(upstream,config);companion,projection=_data_projection(root,config)
    from .classic_emg_cohort import _StatGuard
    from .vision_literal_cohort import admit_vision_cohort
    paths=_data_sources(root);outer=_StatGuard(paths.values())
    hashes={name:_file_hash(path) for name,path in paths.items()}
    # expected is always compared with freshly recomputed real content below.
    common=admit_vision_cohort(data,crops,upstream=root,config=companion,chunk_rows=chunk_rows)
    outer.check()
    for path,stamp in outer.paths.items():
        if path in common.guard.paths and common.guard.paths[path]!=stamp:raise ValueError('data source generation changed')
        common.guard.paths[path]=stamp
    common.guard.parents.update(outer.parents)
    identity=_data_identity(common,projection,hashes)
    if expected is not None and expected!=identity:raise ValueError('actual WiLoR input content identity differs from receipt')
    result=WilorCohort(identity,common,root,copy.deepcopy(config),identity['sha256']);_COHORTS.add(result)
    result.check(require_full=require_full);return result


def native_wilor_loaders(cohort,*,scope='full'):
    policy=wilor_policy(scope=scope)
    if type(cohort) is not WilorCohort or cohort not in _COHORTS:raise ValueError('actual admitted WiLoR cohort required')
    cohort.check(require_full=scope!='cpu_engineering');_activate(cohort.upstream)
    from egoemg.datamodule import make_data_module
    from torch.utils.data import Subset
    dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0
    dm.batch_size=policy['batch_size_by_role']['train'];loaders={}
    for role,ds in cohort.datasets.items():
        limit=policy['engineering_train_samples'] if role=='train' else policy['engineering_eval_samples']
        if limit is not None:ds=Subset(ds,range(min(len(ds),limit)))
        loaders[role]=dm._make_loader(ds,shuffle=role=='train')
    if any(loader.batch_size!=policy['batch_size_by_role'][role] or loader.generator is not None or
        loader.num_workers!=0 or loader.drop_last for role,loader in loaders.items()):
        raise ValueError('original WiLoR native loader execution differs')
    cohort.check(require_full=scope!='cpu_engineering');return loaders


FORMAT='native_wilor_uninterrupted_fit_v1'
_FITS=weakref.WeakSet()
_CHECKPOINT_WORK_BYTES=8*1024**3


def _process():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=fields[19],uid=os.getuid())


_FIT_SOURCE_NAMES=('egoemg_repro/wilor_native_train.py','egoemg_repro/wilor_literal_runtime.py',
        'egoemg_repro/wilor_literal.py','configs/wilor-assets.json','egoemg_repro/vision_literal_cohort.py',
        'egoemg_repro/vision_literal_data.py','egoemg_repro/classic_emg_cohort.py',
        'egoemg_repro/classic_emg_train.py','egoemg_repro/vision_literal_train.py',
        'egoemg_repro/vision_literal_runtime.py','egoemg_repro/vision_literal_check.py')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported native WiLoR training source changed')

    return {name:_file_hash(ROOT/name) for name in _FIT_SOURCE_NAMES}


def _capture_fit_sources():
    from .classic_emg_cohort import _StatGuard
    from .vision_literal_train import _loop_paths,_loop_sources
    guard=_StatGuard([*[ROOT/name for name in _FIT_SOURCE_NAMES],*_loop_paths().values()])
    sources=_sources();loops=_loop_sources();guard.check()
    return guard,sources,loops


def _guard_merge(*guards):
    from .classic_emg_cohort import _StatGuard
    out=_StatGuard([])
    for guard in guards:
        guard.check()
        for path,stamp in guard.paths.items():
            if path in out.paths and out.paths[path]!=stamp:raise ValueError('native source/input generation changed')
            out.paths[path]=stamp
        out.parents.update(guard.parents)
    out.check();return out


def _rng_packet(*,cpu):
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=[] if cpu else torch.cuda.get_rng_state_all(),generators={})


def _validate_rng(value,*,cpu):
    import numpy as np
    import torch
    try:
        if (type(cpu) is not bool or type(value) is not dict or
            set(value)!={'python','numpy','torch','cuda','generators'} or value['generators']!={} or
            type(value['cuda']) is not list or len(value['cuda'])!=(0 if cpu else 1)):
            raise ValueError('complete original global RNG streams required')
        for tensor in [value['torch'],*value['cuda']]:
            if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.uint8 or tensor.ndim!=1 or not tensor.numel() or tensor.device.type!='cpu':
                raise ValueError('typed CPU/native CUDA RNG states required')
        # Pinned torch2.2.2 CUDAGeneratorImpl::get_state: uint64 seed followed
        # by int64 Philox offset; its setter requires offset divisible by4.
        # Inspect bytes on CPU, without constructing/restoring a CUDA generator.
        for tensor in value['cuda']:
            if (tensor.numel()!=16 or not tensor.is_contiguous() or
                int.from_bytes(bytes(tensor.tolist())[8:],sys.byteorder)%4):
                raise ValueError('complete pinned CUDA RNG seed/Philox state required')
        random.Random(0).setstate(value['python']);np.random.RandomState(0).set_state(value['numpy'])
        torch.Generator(device='cpu').set_state(value['torch'])
    except (KeyError,TypeError,ValueError,RuntimeError) as exc:raise ValueError('native saved RNG state malformed') from exc


def wilor_stage_complete(policy,records,*,train_samples,val_samples,interrupted=False):
    """Budget/coverage predicate only; never a model or qualification certificate."""
    try:
        scope=policy['scope']
        if (type(interrupted) is not bool or interrupted or policy!=wilor_policy(scope=scope) or
            type(records) is not list or len(records)!=policy['max_epochs']):return False
        expected=(202020,3152) if scope=='full' else (2,2) if scope=='cpu_engineering' else (36,3152)
        if (type(train_samples) is not int or type(val_samples) is not int or (train_samples,val_samples)!=expected):return False
        batches=math.ceil(train_samples/policy['batch_size_by_role']['train'])
        for epoch,row in enumerate(records):
            pairs=dict(epoch=epoch,train_samples=train_samples,train_batches=batches,
                optimizer_executions=(epoch+1)*batches,global_step=(epoch+1)*batches)
            if (any(type(row[k]) is not int or row[k]!=v for k,v in pairs.items()) or
                type(row['validation']['samples']) is not int or row['validation']['samples']!=val_samples or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in (row['train_loss'],row['validation']['val_mae']))):return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError):return False


def _raw_qualification(path):
    if not isinstance(path,(str,os.PathLike)):raise ValueError('actual raw new WiLoR CUDA qualification artifact required')
    try:
        from .wilor_native_eval import read_wilor_qualification
        return read_wilor_qualification(path)
    except (ImportError,OSError,KeyError,TypeError,ValueError) as exc:
        raise ValueError('actual raw current-source WiLoR CUDA qualification artifact required') from exc


def _qualification_binding(qualification,cohort,construction):
    if (qualification.get('format')!='native_wilor_raw_cuda_qualification_v1' or
        qualification.get('training_sources_sha256')!=_sources() or
        qualification.get('cohort_sha256')!=cohort.identity['sha256'] or
        qualification.get('wilor_protocol_sha256')!=construction['protocol_sha256'] or
        qualification.get('construction_identity_sha256')!=construction['sha256'] or
        qualification.get('runtime_versions')!=construction['runtime_versions']):
        raise ValueError('actual CUDA qualification source/cohort/runtime/initialization binding differs')


def _runtime_device(policy):
    import torch
    if policy['scope']=='cpu_engineering':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU engineering requires CUDA hidden at startup')
        return dict(accelerator='cpu',cuda_initialized=False)
    if (torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or
        not torch.cuda.is_bf16_supported()):raise ValueError('actual single visible CUDA BF16 device required')
    free,total=torch.cuda.mem_get_info(0)
    if free<64*1024**3:raise ValueError('actual64GiB free GPU reserve required before native WiLoR fit')
    return dict(accelerator='gpu',name=torch.cuda.get_device_name(0),free_memory_before=free,total_memory=total,
        visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version())


def _batch(cohort,policy,batch,role,seen,limit):
    import numpy as np
    import torch
    if (type(cohort) is not WilorCohort or cohort not in _COHORTS or role not in ROLES or
        policy!=wilor_policy(scope=policy['scope']) or type(limit) is not int or not 0<limit<=len(cohort.datasets[role])):
        raise ValueError('actual native WiLoR data/role/population required')
    cohort.guard.check();ids=batch['vision_sample_index']
    if ids.dtype!=torch.int64 or ids.ndim!=1:raise ValueError('typed actual native WiLoR sample indices required')
    ids=ids.detach().cpu().numpy();n=len(ids)
    if (n!=min(policy['batch_size_by_role'][role],limit-len(seen)) or np.any(ids<0) or np.any(ids>=limit) or
        len(np.unique(ids))!=n or any(int(i) in seen for i in ids) or
        (role!='train' and ids.tolist()!=list(range(len(seen),len(seen)+n)))):
        raise ValueError('native WiLoR population/order/partial batch differs')
    if (batch['vision_img'].shape!=(n,3,256,256) or batch['vision_img'].dtype!=torch.float32 or
        batch['joint_angles'].shape!=(n,22,1) or batch['joint_angles'].dtype!=torch.float32 or
        batch['label_valid_mask'].shape!=(n,1) or batch['label_valid_mask'].dtype!=torch.bool or
        batch['vision_valid_mask'].shape!=(n,1) or batch['vision_valid_mask'].dtype!=torch.bool or
        batch['emg'].shape!=(n,16,0) or
        any(not bool(torch.isfinite(batch[k]).all()) for k in ('vision_img','joint_angles'))):
        raise ValueError('native WiLoR complete input/target/generated mask malformed or nonfinite')
    ds=cohort.datasets[role];anchors=len(ds.windows)
    for j,index in enumerate(ids):
        unique=int(index)%(2*anchors);repeat=int(index)//(2*anchors);hand,window=divmod(unique,anchors)
        w=ds.windows[window];offset=int(batch['jitter_offset'][j])
        expected=dict(start=w['start']+offset,end=w['end']+offset,center=w['center']+offset,
            episode=w['episode'],target_hand_index=hand,vision_window_index=window,
            vision_unique_sample_index=unique,vision_repeat_index=repeat)
        if (any(int(batch[k][j])!=v for k,v in expected.items()) or not 0<=offset<=w['jitter_max_offset'] or
            batch['role'][j]!=role or batch['split'][j]!=w['split'] or batch['hand'][j]!=('left','right')[hand] or
            batch['dataset_name'][j]!='egoemg' or batch['frame_mapping_field'][j]!=ds.frame_mapping_field):
            raise ValueError('native WiLoR original anchor/jitter/hand/repeat/input binding differs')
    seen.update(map(int,ids));return n


def _prediction_part(batch,out,module):
    import torch
    from .classic_emg_train import state_digest
    if type(out) is not tuple or len(out)!=3:raise ValueError('original WiLoR prediction tuple required')
    pred,target,mask=out;n=len(batch['vision_sample_index']);device=next(module.parameters()).device
    if (pred.shape!=(n,22,1) or target.shape!=pred.shape or pred.dtype not in (torch.float32,torch.bfloat16) or
        mask.shape!=(n,1) or mask.dtype!=torch.bool or not bool(torch.isfinite(pred).all()) or
        not torch.equal(target,batch['joint_angles']) or not torch.equal(mask,batch['label_valid_mask']) or
        batch['vision_img'].device!=device or pred.device!=device):
        raise ValueError('actual original WiLoR prediction/target/mask or per-forward device differs')
    value={k:batch[k] for k in ('vision_img','vision_valid_mask','vision_sample_index','vision_unique_sample_index',
        'vision_repeat_index','center','episode','jitter_offset','target_hand_index')}
    value.update(prediction=pred,target=target,mask=mask)
    return state_digest(value),str(device)


def _finite(value):
    import torch
    if torch.is_tensor(value):
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(torch.isfinite(value).all()):
            raise ValueError('native checkpoint tensor state nonfinite')
    elif isinstance(value,dict):
        for item in value.values():_finite(item)
    elif isinstance(value,(tuple,list)):
        for item in value:_finite(item)
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError('native checkpoint scalar state nonfinite')


def _frozen_names(construction):
    prefixes=tuple('model.backbone.'+n+'.' for n in construction['model_contract']['frozen_backbone_components'])
    return [n for n in construction['parameter_names'] if n.startswith(prefixes)]


def _frozen_sha(state,construction):
    from .vision_literal_runtime import vision_state_sha256
    return vision_state_sha256({n:state[n] for n in _frozen_names(construction)})


def _saved_optimizer(checkpoint,identity,epochs,steps):
    import torch
    construction=identity['construction'];frozen=set(_frozen_names(construction));names=construction['parameter_names']
    groups={g:[n for n in names if n.startswith('model.'+g+'.') and n not in frozen] for g in ('head','backbone')}
    optimizer=checkpoint['optimizer_states'][0];scheduler=checkpoint['lr_schedulers'][0];states=optimizer['state'];cursor=0
    if len(optimizer['param_groups'])!=2:raise ValueError('original two native WiLoR optimizer groups required')
    for group_name,group in zip(('head','backbone'),optimizer['param_groups']):
        expected=groups[group_name];ids=list(range(cursor,cursor+len(expected)));cursor+=len(expected)
        if (group.get('name')!=group_name or group['params']!=ids or group['lr']!=1e-5 or group.get('initial_lr')!=1e-5 or
            group['weight_decay']!=.01 or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group['amsgrad'] is not False or
            any(group.get(k,False) is not False for k in ('maximize','capturable','differentiable')) or
            any(group.get(k) is not None for k in ('foreach','fused'))):raise ValueError('actual native WiLoR optimizer recipe/membership differs')
        for index,name in zip(ids,expected):
            s=states[index];shape=tuple(construction['model_schema'][name]['shape'])
            if (set(s)!={'step','exp_avg','exp_avg_sq'} or not torch.is_tensor(s['step']) or s['step'].numel()!=1 or float(s['step'])!=steps or
                any(not torch.is_tensor(s[k]) or s[k].shape!=shape or s[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq'))):
                raise ValueError('actual native WiLoR AdamW moments/step state differs')
    if (set(states)!=set(range(cursor)) or cursor!=393 or scheduler['T_max']!=30 or scheduler['eta_min']!=1e-5 or
        scheduler['base_lrs']!=[1e-5,1e-5] or scheduler['_last_lr']!=[1e-5,1e-5] or scheduler['last_epoch']!=epochs):
        raise ValueError('complete native WiLoR optimizer/constant scheduler state differs')
    for k in ('optimizer_states','lr_schedulers'):_finite(checkpoint[k])


def _validate_loops(loops,*,epochs,steps,batches,val_batches):
    """Pinned Lightning2.5 native automatic-optimization checkpoint boundary."""
    def progress(total,current,fields=('ready','completed')):
        return dict(total={k:total for k in fields},current={k:current for k in fields})
    def batch(total,current,last):
        return dict(**progress(total,current,('ready','completed','started','processed')),is_last_batch=last)
    expected=dict(fit_loop={
        'state_dict':{},'epoch_loop.state_dict':{'_batches_that_stepped':steps},
        'epoch_loop.batch_progress':batch(steps,batches,True),
        'epoch_loop.scheduler_progress':progress(epochs,1),
        'epoch_loop.automatic_optimization.state_dict':{},
        'epoch_loop.automatic_optimization.optim_progress':{'optimizer':{
            'step':progress(steps,batches),'zero_grad':progress(steps,batches,('ready','completed','started'))}},
        'epoch_loop.manual_optimization.state_dict':{},
        'epoch_loop.manual_optimization.optim_step_progress':progress(0,0),
        'epoch_loop.val_loop.state_dict':{},
        'epoch_loop.val_loop.batch_progress':batch(val_batches,val_batches,True),
        # ModelCheckpoint saves after epoch processed, before epoch completed.
        'epoch_progress':{k:dict(ready=epochs,started=epochs,processed=epochs,completed=epochs-1)
                          for k in ('total','current')}},
        validate_loop={'state_dict':{},'batch_progress':batch(0,0,False)},
        test_loop={'state_dict':{},'batch_progress':batch(0,0,False)},
        predict_loop={'state_dict':{},'batch_progress':progress(0,0,('ready','completed','started','processed'))})
    # Canonical JSON comparison also distinguishes bool/float from int counters.
    try:
        if type(loops) is not dict or _hash(loops)!=_hash(expected):
            raise ValueError('complete native Lightning loop structure/counters differ')
    except (TypeError,OverflowError) as exc:
        raise ValueError('malformed native Lightning checkpoint loops') from exc


def _validate_checkpoint(checkpoint,identity,records):
    from .vision_literal_runtime import _schema,vision_state_sha256
    try:
        if (type(checkpoint) is not dict or type(identity) is not dict or identity.get('format')!=FORMAT or
            type(records) is not list or not records or any(k not in checkpoint for k in
                ('native_wilor','state_dict','optimizer_states','lr_schedulers','loops','callbacks','epoch','global_step'))):
            raise ValueError('complete original native WiLoR checkpoint required')
        p=checkpoint['native_wilor'];execution=p['execution'];process=execution['process'];epochs=len(records)
        if (set(execution)!={'process','started_ns'} or set(process)!={'pid','start_ticks','uid'} or
            type(process['pid']) is not int or process['pid']<=0 or type(process['uid']) is not int or
            type(process['start_ticks']) is not str or not process['start_ticks'].isdigit() or
            type(execution['started_ns']) is not int or execution['started_ns']<=0 or
            p['format']!=FORMAT or p['identity']!=identity or p['epoch_boundary'] is not True or p['records']!=records or
            type(p['completed_epochs']) is not int or p['completed_epochs']!=epochs or
            type(checkpoint['epoch']) is not int or checkpoint['epoch']!=epochs-1 or
            checkpoint['global_step']!=records[-1]['optimizer_executions'] or
            p['initial_model_sha256']!=identity['construction']['initial_state_sha256']):
            raise ValueError('native WiLoR checkpoint epoch/owner/source/initial state differs')
        if (len(checkpoint['optimizer_states'])!=1 or len(checkpoint['lr_schedulers'])!=1 or
            any(k in checkpoint for k in ('MixedPrecision','native_amp_scaling_state','amp_scaling_state'))):
            raise ValueError('complete original BF16/CPU32 optimizer state without an invented FP16 scaler required')
        _validate_rng(p['rng'],cpu=identity['policy']['scope']=='cpu_engineering')
        train=identity['counts']['train'];val=identity['counts']['val'];batch=identity['policy']['batch_size_by_role']['train'];batches=math.ceil(train/batch)
        if not 1<=epochs<=identity['policy']['max_epochs']:raise ValueError('native WiLoR checkpoint epoch outside budget')
        for i,row in enumerate(records):
            expected=dict(epoch=i,train_samples=train,train_batches=batches,optimizer_executions=(i+1)*batches,global_step=(i+1)*batches)
            v=row['validation'];v_batches=math.ceil(val/identity['policy']['batch_size_by_role']['val']);gpu=identity['policy']['scope']!='cpu_engineering'
            if (any(type(row[k]) is not int or row[k]!=x for k,x in expected.items()) or v['samples']!=val or
                not math.isfinite(row['train_loss']) or not math.isfinite(v['val_mae']) or v['forward_batches']!=v_batches or
                v['cuda_forward_batches']!=(v_batches if gpu else 0) or v['forward_device']!=('cuda:0' if gpu else 'cpu') or
                v['input_and_module_device_agreed'] is not True):raise ValueError('native WiLoR epoch coverage/update/device/finite values differ')
        _validate_loops(checkpoint['loops'],epochs=epochs,steps=records[-1]['optimizer_executions'],batches=batches,
                        val_batches=math.ceil(val/identity['policy']['batch_size_by_role']['val']))
        construction=identity['construction'];state=checkpoint['state_dict'];schema=_schema(state)
        if (schema!=construction['model_schema'] or len([k for k in state if k.startswith('model.backbone.')])!=404 or
            sum(math.prod(schema[n]['shape']) for n in construction['parameter_names'])!=631631657 or
            sum(math.prod(schema[n]['shape']) for n in _frozen_names(construction))!=52499 or
            any(schema[n]['dtype']!='torch.float32' for n in construction['parameter_names']) or
            vision_state_sha256(state)!=records[-1]['model_sha256'] or _frozen_sha(state,construction)!=identity['frozen_state_sha256']):
            raise ValueError('complete actual native WiLoR model schema/count/state/frozen identity differs')
        _saved_optimizer(checkpoint,identity,epochs,records[-1]['optimizer_executions'])
    except (KeyError,TypeError,IndexError,AttributeError) as exc:raise ValueError('incomplete or malformed native WiLoR checkpoint') from exc


@dataclass(eq=False)
class PreparedWilorFit:
    identity:dict
    cohort:WilorCohort
    runtime:object
    guard:object
    _seal:str
    selection:dict|None=None
    checkpoint:dict|None=None
    run:Path|None=None

    @property
    def module(self):return self.runtime.module
    @property
    def datasets(self):return self.cohort.datasets
    @property
    def policy(self):return self.identity['policy']
    @property
    def cfg(self):return self.cohort.config

    def check(self):
        if self not in _FITS:raise ValueError('actual prepared native WiLoR fit required')
        self.guard.check();self.cohort.check(require_full=self.policy['scope']!='cpu_engineering');self.runtime.check()
        if (self.identity.get('sha256')!=self._seal or _hash({k:v for k,v in self.identity.items() if k!='sha256'})!=self._seal or
            self.identity['policy']!=wilor_policy(scope=self.policy['scope']) or self.identity['training_sources_sha256']!=_sources() or
            self.identity['cohort']!=self.cohort.identity or self.identity['construction']!=self.runtime.identity):
            raise ValueError('actual native WiLoR fit source/config/input/runtime identity changed')


def _observer_type():
    from . import classic_emg_train as storage
    from .vision_literal_train import _epoch_bytes,_adam_steps
    from .vision_literal_runtime import vision_state_sha256
    from .wilor_literal_runtime import wilor_optimizer_identity
    class NativeWilorObserver(storage.Callback):
        def __init__(self,prepared,output):
            self.prepared=prepared;self.output=output;self.records=[];self.steps=0;self.phase=None;self.hook=None
            self.execution=dict(process=_process(),started_ns=time.time_ns());self.validation=None
        def check(self):
            self.prepared.guard.check();storage._space(self.output)
        def restore(self,module):
            if hasattr(self,'original_forward'):module.forward=self.original_forward
            if self.hook is not None:self.hook.remove();self.hook=None
        def on_fit_start(self,trainer,module):
            self.original_forward=module.forward
            def forward(batch):
                out=self.original_forward(batch)
                if self.phase=='val' and not trainer.sanity_checking:
                    digest,device=_prediction_part(batch,out,module);self.parts.append(digest);self.devices.add(device)
                return out
            module.forward=forward
            self.hook=trainer.optimizers[0].register_step_post_hook(lambda *a,**k:self._executed())
        def _executed(self):self.steps+=1
        def on_train_start(self,trainer,module):
            from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor
            if len(trainer.optimizers)!=1 or len(trainer.lr_scheduler_configs)!=1:raise ValueError('one original native optimizer/scheduler required')
            wilor_optimizer_identity(self.prepared.runtime,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
            checkpoints=[c for c in trainer.callbacks if isinstance(c,ModelCheckpoint)];early=[c for c in trainer.callbacks if isinstance(c,EarlyStopping)]
            if (len(checkpoints)!=1 or len(early)!=1 or sum(isinstance(c,LearningRateMonitor) for c in trainer.callbacks)!=1 or
                checkpoints[0].monitor!='val_mae' or checkpoints[0].mode!='min' or checkpoints[0].save_top_k!=3 or checkpoints[0].save_last is not True or
                early[0].monitor!='val_mae' or early[0].mode!='min' or early[0].patience!=30 or early[0]._check_on_train_epoch_end is not False or
                getattr(trainer.precision_plugin,'scaler',None) is not None or trainer.gradient_clip_val!=1 or trainer.accumulate_grad_batches!=1 or
                trainer.limit_train_batches!=1.0 or trainer.limit_val_batches!=1.0):raise ValueError('original WiLoR callbacks/clip/precision/full loader policy differs')
        def on_train_epoch_start(self,trainer,module):
            self.prepared.check()
            if trainer.current_epoch!=len(self.records):raise ValueError('fresh native WiLoR epoch order differs')
            self.train_seen=set();self.train_batches=0;self.loss_sum=0.;self.validation=None
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            self.check();self.phase='train';self.batch_n=_batch(self.prepared.cohort,self.prepared.policy,batch,'train',self.train_seen,self.prepared.identity['counts']['train'])
        def on_before_optimizer_step(self,trainer,module,optimizer):
            import torch
            gradients=[p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):raise ValueError('native WiLoR gradients absent/nonfinite')
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            loss=float(outputs['loss'])
            if not math.isfinite(loss) or self.steps!=trainer.global_step:raise ValueError('native WiLoR finite loss/actual optimizer count differs')
            self.loss_sum+=self.batch_n*loss;self.train_batches+=1
        def on_validation_epoch_start(self,trainer,module):
            self.parts=[];self.devices=set();self.val_seen=set()
        def on_validation_batch_start(self,trainer,module,batch,batch_idx,dataloader_idx=0):
            self.phase=None if trainer.sanity_checking else 'val'
            if not trainer.sanity_checking:
                self.check();_batch(self.prepared.cohort,self.prepared.policy,batch,'val',self.val_seen,self.prepared.identity['counts']['val'])
        def on_validation_end(self,trainer,module):
            self.phase=None
            if trainer.sanity_checking:return
            n=self.prepared.identity['counts']['val'];score=float(trainer.callback_metrics['val_mae'])
            device='cpu' if self.prepared.policy['scope']=='cpu_engineering' else 'cuda:0'
            if len(self.val_seen)!=n or not math.isfinite(score) or self.devices!={device}:raise ValueError('native WiLoR complete validation/device differs')
            self.validation=dict(samples=n,val_mae=score,prediction_sha256=_hash(self.parts),forward_batches=len(self.parts),
                cuda_forward_batches=len(self.parts) if device=='cuda:0' else 0,forward_device=device,input_and_module_device_agreed=True)
        def on_train_epoch_end(self,trainer,module):
            self.prepared.check();n=self.prepared.identity['counts']['train'];batches=math.ceil(n/self.prepared.policy['batch_size_by_role']['train'])
            if len(self.train_seen)!=n or self.train_batches!=batches or self.steps!=(len(self.records)+1)*batches or self.validation is None:
                raise ValueError('partial native WiLoR epoch cannot be committed')
            _adam_steps(trainer.optimizers[0],self.steps)
            if _frozen_sha(module.state_dict(),self.prepared.runtime.identity)!=self.prepared.identity['frozen_state_sha256']:
                raise ValueError('original frozen WiLoR parameters changed')
            record=dict(epoch=len(self.records),optimizer_executions=self.steps,global_step=trainer.global_step,
                train_samples=n,train_batches=batches,train_loss=self.loss_sum/n,validation=copy.deepcopy(self.validation),
                model_sha256=vision_state_sha256(module.state_dict()))
            previous=_epoch_bytes(self.records,self.prepared.identity['sha256']);self.records.append(record)
            raw=_epoch_bytes(self.records,self.prepared.identity['sha256']);path=self.output/'epochs.jsonl'
            if len(raw)>2*1024**2 or path.read_bytes()!=previous:raise ValueError('native WiLoR journal prefix/bound differs')
            with path.open('ab') as f:f.write(raw[len(previous):]);f.flush();os.fsync(f.fileno())
        def on_save_checkpoint(self,trainer,module,checkpoint):
            self.check();storage._space(self.output,budget=_CHECKPOINT_WORK_BYTES)
            checkpoint['native_wilor']=dict(format=FORMAT,identity=self.prepared.identity,epoch_boundary=self.validation is not None,
                records=copy.deepcopy(self.records),completed_epochs=len(self.records),execution=self.execution,
                initial_model_sha256=self.prepared.runtime.identity['initial_state_sha256'],
                rng=_rng_packet(cpu=self.prepared.policy['scope']=='cpu_engineering'))
    return NativeWilorObserver


def fit_wilor(data,crops,*,upstream,config,assets,output,scope='full',qualification=None):
    policy=wilor_policy(scope=scope)
    qualified=_raw_qualification(qualification) if scope=='full' else None
    if scope!='full' and qualification is not None:raise ValueError('CPU/CUDA engineering cannot inherit a qualification')
    from .classic_emg_cohort import _safe,_StatGuard
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('native WiLoR output exists; no overwrite/resume/implicit restart')
    from .wilor_literal_runtime import WilorAssets,build_wilor_runtime
    if type(assets) is not WilorAssets or assets.restore:raise ValueError('actual admitted fresh pretrained WiLoR assets required')
    from . import classic_emg_train as storage
    from .vision_literal_train import _loop_paths,_loop_sources,_write,_read_epochs
    from omegaconf import OmegaConf
    storage._space(output.parent,budget=_CHECKPOINT_WORK_BYTES*(5 if scope=='full' else 4))
    root,_=_validated(upstream,config);assets.check();device=_runtime_device(policy)
    source_guard,sources,loop_sources=_capture_fit_sources()
    cohort=admit_wilor_cohort(data,crops,upstream=root,config=config,require_full=scope!='cpu_engineering')
    runtime=build_wilor_runtime(root,config,assets=assets)
    if qualified is not None:_qualification_binding(qualified,cohort,runtime.identity)
    loaders=native_wilor_loaders(cohort,scope=scope);counts={r:len(l.dataset) for r,l in loaders.items()}
    identity=_sealed(dict(format=FORMAT,policy=policy,cohort=cohort.identity,counts=counts,construction=runtime.identity,
        frozen_state_sha256=_frozen_sha(runtime.module.state_dict(),runtime.identity),
        saved_config=OmegaConf.to_container(config,resolve=True),training_sources_sha256=sources,native_loop_sources_sha256=loop_sources,
        locations=dict(data=str(cohort.datasets['train'].store.root),crops=str(cohort.datasets['train'].views[0].crops),
            upstream=str(root),assets_runtime=str(assets.runtime),qualification=str(_safe(qualification)) if qualified is not None else None),
        qualification=qualified,runtime_device=device,scientific_eligible=False,full_paper_complete=False))
    guard=_guard_merge(source_guard,cohort.guard,runtime._source_guard,assets._guard)
    prepared=PreparedWilorFit(identity,cohort,runtime,guard,identity['sha256']);_FITS.add(prepared);prepared.check()
    output.mkdir(mode=0o700);_write(output/'identity.json',identity);prepared.guard.add(output/'identity.json')
    with (output/'epochs.jsonl').open('xb'):pass
    witness=_observer_type()(prepared,output)
    from egoemg._train_utils import build_callbacks,build_logger
    from pytorch_lightning.callbacks import ModelCheckpoint
    callbacks=build_callbacks(config)+[witness]
    for cb in callbacks:
        if isinstance(cb,ModelCheckpoint):cb.dirpath=str(output/'checkpoints')
    effective=copy.deepcopy(config);effective.logger.save_dir=str(output);effective.logger.name='logs';effective.logger.version='native-wilor'
    logger=build_logger(effective);kwargs=OmegaConf.to_container(config.trainer,resolve=True)
    kwargs.update(accelerator=device['accelerator'],devices=1,precision=policy['precision'],max_epochs=policy['max_epochs'],
        num_sanity_val_steps=policy['num_sanity_val_steps'],enable_progress_bar=False,callbacks=callbacks,logger=logger)
    trainer=storage.pl.Trainer(**kwargs);started=time.monotonic()
    try:trainer.fit(runtime.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'])
    finally:witness.restore(runtime.module)
    prepared.check()
    if not wilor_stage_complete(policy,witness.records,train_samples=counts['train'],val_samples=counts['val'],interrupted=bool(trainer.interrupted)):
        raise ValueError('native WiLoR fit did not complete its genuine full budget')
    import torch
    last_path=output/'checkpoints/last.ckpt';last=torch.load(str(last_path),map_location='cpu',weights_only=False,mmap=True)
    records,_=_read_epochs(output/'epochs.jsonl',identity['sha256']);_validate_checkpoint(last,identity,records)
    selection=_selection(output,identity,last,records)
    result=_sealed(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,completed_epochs=len(records),
        optimizer_executions=witness.steps,train_samples_per_epoch=[r['train_samples'] for r in records],
        val_samples_per_epoch=[r['validation']['samples'] for r in records],initial_model_sha256=runtime.identity['initial_state_sha256'],
        final_model_state_sha256=records[-1]['model_sha256'],last_checkpoint_sha256=_file_hash(last_path),selected_best=selection,
        journal_sha256=_file_hash(output/'epochs.jsonl'),full_training_terminated=scope=='full',stage_complete=True,
        execution=witness.execution,seconds=time.monotonic()-started,finished_ns=time.time_ns(),
        checkpoint_continuation_supported=False,scientific_eligible=False,full_paper_complete=False))
    _write(output/'fit-result.json',result)
    return read_wilor_fit(output,require_full=scope=='full')


def _selection(output,identity,last,records):
    from .classic_emg_cohort import _safe
    from .vision_literal_train import _native_callback
    from .vision_literal_runtime import vision_state_sha256
    import torch
    cb=_native_callback(last,'ModelCheckpoint{');path=_safe(cb['best_model_path']);score=float(cb['best_model_score'])
    if (path.parent!=output/'checkpoints' or path.name=='last.ckpt' or cb['monitor']!='val_mae' or
        Path(cb['last_model_path'])!=output/'checkpoints/last.ckpt' or not math.isfinite(score) or
        score!=min(r['validation']['val_mae'] for r in records) or path not in map(Path,cb['best_k_models'])):
        raise ValueError('actual native WiLoR validation best selection differs')
    if (len(cb['best_k_models'])!=min(3,len(records)) or
        sorted(float(v) for v in cb['best_k_models'].values())!=sorted(r['validation']['val_mae'] for r in records)[:3]):
        raise ValueError('native WiLoR top3 validation selection differs')
    best=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True);n=best['native_wilor']['completed_epochs']
    if type(n) is not int or not 1<=n<=len(records):raise ValueError('native selected WiLoR best epoch outside journal')
    _validate_checkpoint(best,identity,records[:n])
    if records[n-1]['validation']['val_mae']!=score:raise ValueError('native selected WiLoR checkpoint is not validation best')
    early=_native_callback(last,'EarlyStopping{')
    if float(early['best_score'])!=score:raise ValueError('original native early-stopping validation history differs')
    return dict(file=path.name,epoch=n-1,score=score,checkpoint_sha256=_file_hash(path),
        model_sha256=vision_state_sha256(best['state_dict']),validation_prediction_sha256=records[n-1]['validation']['prediction_sha256'])


def _read_fit(output,*,require_full):
    if type(require_full) is not bool:raise ValueError('full native WiLoR requirement must be boolean')
    from .classic_emg_cohort import _safe,_StatGuard
    from .vision_literal_train import _json,_read_epochs,_loop_paths,_loop_sources
    from .wilor_literal_runtime import admit_wilor_assets,_sources as runtime_sources,_versions,_restore_identity
    from .vision_literal_runtime import vision_state_sha256
    from omegaconf import OmegaConf
    import torch
    output=_safe(output);guard=_StatGuard([output/n for n in ('identity.json','fit-result.json','epochs.jsonl','checkpoints/last.ckpt')])
    identity=_json(output/'identity.json');result=_json(output/'fit-result.json')
    for value in (identity,result):
        if (value.get('format')!=FORMAT or value.get('sha256')!=_hash({k:v for k,v in value.items() if k!='sha256'}) or
            value.get('scientific_eligible') is not False or value.get('full_paper_complete') is not False):raise ValueError('native WiLoR fit identity/result seal or scope differs')
    policy=wilor_policy(scope=identity['policy']['scope']);scope=policy['scope']
    source_guard,sources,loop_sources=_capture_fit_sources();guard=_guard_merge(guard,source_guard)
    if identity['policy']!=policy or identity['training_sources_sha256']!=sources:raise ValueError('actual native WiLoR fit source/policy differs')
    if require_full and scope!='full':raise ValueError('engineering fit is not full native WiLoR training')
    for p in [*[ROOT/n for n in identity['training_sources_sha256']],*_loop_paths().values()]:guard.add(p)
    if identity['native_loop_sources_sha256']!=loop_sources:raise ValueError('native WiLoR training iterator/checkpoint source changed')
    locations=identity['locations'];qualified=_raw_qualification(locations['qualification']) if scope=='full' else None
    if qualified!=identity['qualification']:raise ValueError('native WiLoR saved raw qualification changed')
    cfg=OmegaConf.create(identity['saved_config'])
    cohort=admit_wilor_cohort(locations['data'],locations['crops'],upstream=locations['upstream'],config=cfg,
        expected=identity['cohort'],require_full=scope!='cpu_engineering')
    counts=cohort.identity['counts'].copy()
    if scope!='full':counts['train']=min(counts['train'],policy['engineering_train_samples'])
    if scope=='cpu_engineering':counts.update({r:min(counts[r],2) for r in ('val','test')})
    if counts!=identity['counts']:raise ValueError('actual native WiLoR fit population differs')
    assets=admit_wilor_assets(locations['assets_runtime'],restore=True)
    records,_=_read_epochs(output/'epochs.jsonl',identity['sha256'])
    if not wilor_stage_complete(policy,records,train_samples=counts['train'],val_samples=counts['val']):raise ValueError('incomplete native WiLoR full or engineering budget')
    last_path=output/'checkpoints/last.ckpt'
    if _file_hash(last_path)!=result['last_checkpoint_sha256']:raise ValueError('native last WiLoR checkpoint bytes changed')
    last=torch.load(str(last_path),map_location='cpu',weights_only=False,mmap=True)
    protocol=wilor_protocol(locations['upstream'],cfg)
    _restore_identity(identity['construction'],protocol,runtime_sources(Path(locations['upstream']),protocol,assets),_versions(),last['state_dict'])
    if qualified is not None:_qualification_binding(qualified,cohort,identity['construction'])
    _validate_checkpoint(last,identity,records)
    from .vision_literal_train import _native_callback
    best_path=_safe(_native_callback(last,'ModelCheckpoint{')['best_model_path']);guard.add(best_path)
    selection=_selection(output,identity,last,records)
    expected=dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,completed_epochs=len(records),
        optimizer_executions=records[-1]['optimizer_executions'],train_samples_per_epoch=[r['train_samples'] for r in records],
        val_samples_per_epoch=[r['validation']['samples'] for r in records],initial_model_sha256=identity['construction']['initial_state_sha256'],
        final_model_state_sha256=records[-1]['model_sha256'],last_checkpoint_sha256=_file_hash(last_path),selected_best=selection,
        journal_sha256=_file_hash(output/'epochs.jsonl'),full_training_terminated=scope=='full',stage_complete=True,
        execution=last['native_wilor']['execution'],checkpoint_continuation_supported=False,scientific_eligible=False,full_paper_complete=False)
    if (set(result)!=set(expected)|{'seconds','finished_ns','sha256'} or any(result.get(k)!=v for k,v in expected.items()) or
        type(result['seconds']) not in (int,float) or not math.isfinite(result['seconds']) or result['seconds']<0 or
        type(result['finished_ns']) is not int or result['finished_ns']<=expected['execution']['started_ns'] or
        result['initial_model_sha256']==result['final_model_state_sha256']):raise ValueError('native WiLoR fit result differs from actual complete checkpoint/journal/selection')
    guard=_guard_merge(guard,cohort.guard,assets._guard)
    return result,identity,last,cohort,assets,guard


def read_wilor_fit(output,*,require_full=True):
    if type(require_full) is not bool:raise ValueError('full native WiLoR requirement must be boolean')
    return _read_fit(output,require_full=require_full)[0]


def reopen_wilor_best(output,*,require_full=True):
    if type(require_full) is not bool:raise ValueError('full native WiLoR requirement must be boolean')
    from .wilor_literal_runtime import build_wilor_runtime
    from .vision_literal_runtime import vision_state_sha256
    import torch
    result,identity,last,cohort,assets,guard=_read_fit(output,require_full=require_full)
    path=Path(output)/'checkpoints'/result['selected_best']['file']
    checkpoint=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
    runtime=build_wilor_runtime(cohort.upstream,cohort.config,assets=assets,state=checkpoint['state_dict'],expected_initialization=identity['construction'])
    if vision_state_sha256(runtime.module.state_dict())!=result['selected_best']['model_sha256']:raise ValueError('strict native WiLoR selected best reload differs')
    prepared=PreparedWilorFit(identity,cohort,runtime,_guard_merge(guard,runtime._source_guard),identity['sha256'],
        selection=result['selected_best'],checkpoint=checkpoint,run=Path(output))
    _FITS.add(prepared);prepared.check();return prepared
