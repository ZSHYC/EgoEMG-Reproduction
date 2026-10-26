"""Bounded, actual CUDA FP16/scaler qualification; never a formal training gate.

Each variant uses exactly four original training hand samples and two validation
hand samples. A prefix pauses after epoch one; a separate process resumes epoch
two. The one deliberate Inf gradient is confined to this qualification protocol.
Trusted local checkpoints contain pickle and must not be supplied by strangers.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import uuid

import numpy as np

from . import classic_emg as models
from . import classic_emg_data as datasets
from . import classic_emg_train as storage
from .data import LegacyStore, file_sha256

pl, Callback, CheckpointIO = storage.pl, storage.Callback, storage.CheckpointIO
FORMAT = 'classic_cuda_qualification_v1'
SCOPE = 'cuda_qualification'
BACKEND_PROFILE = 'canonical_backend_nondeterministic_v1'
RESUME_SCOPE = 'exact typed full checkpoint and RNG state restoration only; no bitwise trajectory guarantee'
IMPORTED_SOURCE_SHA256 = file_sha256(Path(__file__))
RUNTIME_FILES = ('egoemg_repro/__init__.py', 'egoemg_repro/classic_emg.py',
    'egoemg_repro/classic_emg_data.py', 'egoemg_repro/classic_emg_train.py',
    'egoemg_repro/classic_emg_cuda_check.py', 'egoemg_repro/data.py', 'egoemg_repro/partitions.py')
LOOP_KEYS = {'state_dict','epoch_loop.state_dict','epoch_loop.batch_progress','epoch_loop.scheduler_progress',
    'epoch_loop.automatic_optimization.state_dict','epoch_loop.automatic_optimization.optim_progress',
    'epoch_loop.manual_optimization.state_dict','epoch_loop.manual_optimization.optim_step_progress',
    'epoch_loop.val_loop.state_dict','epoch_loop.val_loop.batch_progress','epoch_progress'}
SCALER_KEYS = {'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}


def cuda_check_policy(*, variant, device='cuda:0', phase='prefix'):
    if variant not in ('vemg2pose','neuropose') or device != 'cuda:0' or phase not in ('prefix','resume'):
        raise ValueError('CUDA qualification permits only two base variants, cuda:0, and prefix/resume')
    return dict(scope=SCOPE, scientific_eligible=False, production_ready=False, variant=variant,
        device=device, precision='16-mixed', batch_size=1, num_workers=0, planned_epochs=2,
        train_samples=4, val_samples=2, max_attempts=8, world_size=1, clip_value=0,
        runtime_profile=BACKEND_PROFILE, deterministic=False, cudnn_deterministic=False, cudnn_benchmark=False,
        trainer_deterministic=False, matmul_precision='high', cublas_workspace_config=':4096:8',
        resume_scope=RESUME_SCOPE, trajectory_agreement_qualified=False, bitwise_trajectory_guaranteed=False,
        sample_selection='first four train hand samples; first two val hand samples; no filtering or replacement',
        fault='one Inf parameter-gradient hook after first actual update in prefix; original loss unchanged',
        loader_rng='separate loader/sampler generators; epoch-boundary workers0 restore',
        seed_policy=models.SEED_POLICY, validation='complete fixed two-sample engineering validation only',
        limits='no formal training, best selection, test, batch-capacity qualification or scientific completion')


def gradient_observation(gradients):
    """Read original unscaled gradients; never sanitize the scaler's input."""
    torch = storage._runtime()
    values = [g.detach() for g in gradients if g is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in values)
    maximum = max((float(g.abs().max()) for g in values if g.numel()), default=0.) if finite else None
    return dict(finite=finite, absolute_max=maximum)


def _tensor_schema(state):
    torch = storage._runtime()
    if not isinstance(state, dict) or not state:
        raise ValueError('missing model tensor state')
    result = {}
    for key,value in state.items():
        if not isinstance(key,str) or not torch.is_tensor(value):
            raise ValueError('invalid model tensor state')
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite model tensor state')
        result[key] = dict(dtype=str(value.dtype),shape=list(value.shape))
    return result


def optimizer_schema(states):
    torch = storage._runtime()
    schema = storage._optimizer_schema(states)
    optimizer = states[0]
    ids = optimizer['param_groups'][0]['params']
    if (not ids or len(set(ids)) != len(ids) or set(optimizer['state']) != set(ids)):
        raise ValueError('optimizer parameters/state are incomplete')
    for values in optimizer['state'].values():
        for value in values.values():
            if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
                raise ValueError('optimizer requires finite typed tensor state')
        step = values['step']
        if step.numel()!=1 or float(step) < 1 or float(step) != int(float(step)):
            raise ValueError('optimizer actual step state invalid')
        if values['exp_avg'].shape != values['exp_avg_sq'].shape or values['exp_avg'].dtype != values['exp_avg_sq'].dtype:
            raise ValueError('optimizer moment schema differs')
        if values['exp_avg'].dtype != torch.float32:
            raise ValueError('original AMP AdamW moments must remain float32')
    return schema


def scaler_schema(state):
    if not isinstance(state,dict) or set(state)!=SCALER_KEYS:
        raise ValueError('actual GradScaler state fields missing')
    for key in ('scale','growth_factor','backoff_factor'):
        if type(state[key]) is not float or not math.isfinite(state[key]) or state[key]<=0:
            raise ValueError('invalid GradScaler float state')
    if (state['growth_factor'] != 2. or state['backoff_factor'] != .5 or
        type(state['growth_interval']) is not int or state['growth_interval'] != 2000):
        raise ValueError('qualification must retain default GradScaler settings')
    if type(state['_growth_tracker']) is not int or not 0<=state['_growth_tracker']<2000:
        raise ValueError('invalid GradScaler growth tracker')
    return {key:type(value).__name__ for key,value in state.items()}


def _rng_valid(state):
    torch = storage._runtime()
    if (set(state)!= {'python','numpy','torch','cuda','generators'} or
        type(state['python']) is not tuple or type(state['numpy']) is not tuple or
        type(state['cuda']) is not list or len(state['cuda'])!=1 or
        set(state['generators']) != {'train_loader','train_sampler','val_loader'}):
        raise ValueError('missing typed CUDA/Python/NumPy/loader RNG state')
    # Validate tuple contents without mutating the current process RNG streams.
    if (len(state['numpy'])!=5 or not isinstance(state['numpy'][1],np.ndarray) or
        state['numpy'][1].dtype!=np.dtype('uint32') or state['numpy'][1].shape!=(624,)):
        raise ValueError('NumPy RNG must retain its typed MT19937 array')
    import random
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state(state['numpy'])
    for value in [state['torch'], *state['cuda'], *state['generators'].values()]:
        if not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1 or value.numel()==0:
            raise ValueError('invalid RNG byte tensor')


def validate_checkpoint_state(checkpoint, identity):
    """Structural state admission only. This does not issue a CUDA pass receipt."""
    try:
        meta = checkpoint['cuda_check']
        if (meta['format']!=FORMAT or meta['scope']!=SCOPE or meta['identity']!=identity or
            meta['role']!='qualification_last' or meta['epoch_boundary'] is not True):
            raise ValueError('CUDA checkpoint identity/scope/epoch boundary differs')
        epochs = meta['completed_epochs']
        if type(epochs) is not int or epochs not in (1,2) or checkpoint['epoch']!=epochs-1:
            raise ValueError('CUDA checkpoint epoch differs')
        if _tensor_schema(checkpoint['state_dict']) != meta['model_schema']:
            raise ValueError('CUDA model key/shape/dtype schema differs')
        if optimizer_schema(checkpoint['optimizer_states']) != meta['optimizer_schema']:
            raise ValueError('CUDA optimizer schema differs')
        if scaler_schema(checkpoint['MixedPrecision']) != meta['scaler_schema']:
            raise ValueError('CUDA actual scaler schema differs')
        schedulers=checkpoint['lr_schedulers']
        if (len(schedulers)!=1 or type(schedulers[0].get('T_max')) is not int or schedulers[0]['T_max']!=100 or
            type(schedulers[0].get('last_epoch')) is not int or schedulers[0]['last_epoch']!=epochs):
            raise ValueError('CUDA original epoch scheduler missing or reset')
        loop=checkpoint['loops']['fit_loop']
        if set(loop)!=LOOP_KEYS:
            raise ValueError('CUDA complete Lightning fit loop keys differ')
        expected=dict(ready=epochs,started=epochs,processed=epochs,completed=epochs-1)
        for category in ('current','total'):
            if loop['epoch_progress'][category]!=expected:
                raise ValueError('CUDA not a genuine Lightning epoch boundary')
        for key,count in (('epoch_loop.batch_progress',4),('epoch_loop.val_loop.batch_progress',2)):
            if loop[key]['current']!=dict(ready=count,started=count,processed=count,completed=count) or loop[key]['is_last_batch'] is not True:
                raise ValueError('CUDA partial train/validation loop')
        protocol=checkpoint['callbacks']['CudaProtocol']
        if (set(protocol)!={'events','validation_events','executions','injection_count','completed_epochs'} or
            protocol['completed_epochs']!=epochs or len(protocol['events'])!=4*epochs or
            type(checkpoint['global_step']) is not int or checkpoint['global_step']!=4*epochs or
            type(protocol['injection_count']) is not int or protocol['injection_count']!=1 or
            type(protocol['executions']) is not int or not 1<=protocol['executions']<=4*epochs-1):
            raise ValueError('CUDA callback attempts/update/injection state differs')
        if any(float(v['step'])!=protocol['executions'] for v in checkpoint['optimizer_states'][0]['state'].values()):
            raise ValueError('CUDA actual AdamW per-parameter steps differ from execution counter')
        validation=protocol['validation_events']
        if len(validation)!=epochs:
            raise ValueError('CUDA fixed validation callback state missing')
        for epoch,event in enumerate(validation):
            if (event['epoch']!=epoch or event['samples']!=2 or event['full_fixed_validation'] is not True or
                event['full_cohort_validation'] is not False or event['scientific_eligible'] is not False or
                not math.isfinite(event['val_loss']) or
                not math.isclose(event['val_loss'],event['batch_weighted_loss'],rel_tol=2e-6,abs_tol=1e-7)):
                raise ValueError('CUDA callback original validation evidence differs')
        monitors=[v for k,v in checkpoint['callbacks'].items() if k.startswith('ModelCheckpoint{')]
        if len(monitors)!=1:
            raise ValueError('CUDA checkpoint callback missing')
        directory=Path(monitors[0]['dirpath'])
        last=monitors[0]['last_model_path']
        if not directory.is_absolute() or '..' in directory.parts or (last and Path(last)!=directory/'last.ckpt'):
            raise ValueError('CUDA checkpoint callback path escapes owned directory')
        _rng_valid(meta['rng'])
    except (KeyError,TypeError,AttributeError,IndexError,RuntimeError) as exc:
        raise ValueError('CUDA checkpoint full state schema is invalid') from exc


def _seal(checkpoint):
    return storage.state_digest({k:v for k,v in checkpoint.items() if k!='cuda_check_seal'})


def _check_identity(identity):
    try:
        if (identity['format']!=FORMAT or identity['scope']!=SCOPE or
            identity['sha256']!=storage._json_hash({k:v for k,v in identity.items() if k!='sha256'}) or
            storage._json_hash(identity['policy'])!=storage._json_hash(cuda_check_policy(variant=identity['recipe']['variant']))):
            raise ValueError('CUDA qualification identity differs')
        _check_backend_profile(identity['runtime'],identity['policy'])
        root=Path(__file__).resolve().parents[1]
        if file_sha256(Path(__file__))!=IMPORTED_SOURCE_SHA256:
            raise ValueError('CUDA source changed after import')
        if identity['runtime_files_sha256']!={name:file_sha256(root/name) for name in RUNTIME_FILES}:
            raise ValueError('CUDA runtime source identity changed')
        framework=Path(pl.__file__).resolve().parent
        if identity['framework_files_sha256']!={name:file_sha256(framework/name) for name in storage.FRAMEWORK_FILES}:
            raise ValueError('actual Lightning implementation identity changed')
        if identity['runtime']['precision_plugin']!='MixedPrecision' or identity['runtime']['scaler_enabled'] is not True:
            raise ValueError('CUDA identity requires actual enabled FP16 scaler')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('invalid CUDA qualification identity') from exc


def _check_backend_profile(runtime,policy):
    for key in ('deterministic','cudnn_deterministic','cudnn_benchmark','trainer_deterministic',
                'trajectory_agreement_qualified','bitwise_trajectory_guaranteed'):
        if type(runtime[key]) is not bool or type(policy[key]) is not bool or runtime[key] is not policy[key]:
            raise ValueError('CUDA backend profile flags differ from exact policy')
    for key in ('runtime_profile','matmul_precision','cublas_workspace_config','resume_scope'):
        if type(runtime[key]) is not str or runtime[key]!=policy[key]:
            raise ValueError('CUDA backend runtime profile/restore scope differs from policy')


def read_cuda_checkpoint(path, identity):
    torch=storage._runtime(); path=storage._safe(path); before=storage._stamp(path)
    try:
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    except Exception as exc:
        raise ValueError('CUDA checkpoint is truncated or unreadable') from exc
    _check_identity(identity)
    validate_checkpoint_state(checkpoint,identity)
    if checkpoint.get('cuda_check_seal')!=_seal(checkpoint):
        raise ValueError('CUDA checkpoint integrity seal differs')
    if storage._stamp(path)!=before:
        raise ValueError('CUDA checkpoint changed during admission')
    return checkpoint


def _profile():
    torch=storage._runtime()
    if not torch.cuda.is_available():
        raise RuntimeError('actual CUDA is required; CPU fallback cannot qualify FP16')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if torch.cuda.device_count()!=1 or not visible.startswith('GPU-') or ',' in visible:
        raise ValueError('qualification requires exactly one root-selected CUDA_VISIBLE_DEVICES GPU UUID')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('CUBLAS_WORKSPACE_CONFIG=:4096:8 must precede Torch/CUDA initialization')
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False
    torch.set_float32_matmul_precision('high')
    from pytorch_lightning.plugins.precision import MixedPrecision
    precision=MixedPrecision('16-mixed',device='cuda')  # Default real scaler, no tuning.
    if not precision.scaler.is_enabled():
        raise RuntimeError('actual CUDA GradScaler is disabled')
    properties=torch.cuda.get_device_properties(0)
    query=subprocess.run(['nvidia-smi','--id='+visible,'--query-gpu=uuid,driver_version','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=10)
    rows=[row.strip().split(', ') for row in query.stdout.strip().splitlines()]
    if len(rows)!=1 or len(rows[0])!=2 or rows[0][0]!=visible:
        raise ValueError('actual driver/device UUID mapping differs from root-selected GPU')
    versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','hydra-core',
        'omegaconf','pytorch-lightning','torchmetrics','lightning-utilities','pandas','joblib')}
    result=dict(runtime_profile=BACKEND_PROFILE,resume_scope=RESUME_SCOPE,trajectory_agreement_qualified=False,
        bitwise_trajectory_guaranteed=False,trainer_deterministic=False,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        versions=versions,python=platform.python_version(),torch_build=str(torch.__version__),
        cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version(),visible_uuid=visible,driver_version=rows[0][1],
        device_name=properties.name,capability=list(torch.cuda.get_device_capability(0)),total_memory=properties.total_memory,
        deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
        precision_plugin=type(precision).__qualname__,scaler_class=type(precision.scaler).__module__+'.'+type(precision.scaler).__qualname__,
        scaler_enabled=precision.scaler.is_enabled(),default_scaler_state=precision.state_dict())
    torch_root=Path(torch.__file__).resolve().parent
    result['torch_files_sha256']={name:file_sha256(torch_root/name) for name in
        ('cuda/amp/grad_scaler.py','optim/adamw.py','utils/data/dataloader.py','cuda/random.py')}
    result['torch_config']=torch.__config__.show()
    scaler_schema(result['default_scaler_state'])
    return result,precision


class _Prepared:
    pass


def _check_prepared(prepared):
    _check_identity(prepared.identity)
    for relative,stamp in prepared.stamps.items():
        if storage._stamp(datasets._safe_path(prepared.data,relative))!=stamp:
            raise ValueError('bounded qualification data changed: '+relative)
    models._validated(prepared.upstream,prepared.cfg)


def _prepare(data,upstream,config,partitions,profile,*,expected_recipe):
    recipe=datasets.load_classic_config(config,upstream)
    if storage._json_hash(recipe)!=storage._json_hash(expected_recipe):
        raise ValueError('classic config changed after pre-CUDA metadata admission')
    if partitions.get('settings',{}).get('val_fraction')!=recipe['val_fraction']:
        raise ValueError('partition fraction differs from original classic recipe')
    cfg=models.compose_classic(upstream,recipe['variant'])
    root=storage._safe(data).resolve()
    # The token explicitly covers selected spans, not an unperformed full scan.
    store=LegacyStore(root); paths=datasets._schema(store)
    paths.update({name:datasets._safe_path(root,name) for name in ('manifest.json','metadata.npz')})
    relative={name:str(path.relative_to(root)) for name,path in paths.items()}
    stamps={relative[name]:storage._stamp(path) for name,path in paths.items()}
    datasets._partitions(root,partitions,upstream,recipe['mask_policy'])
    selected={role:[w for w in partitions['windows'] if w['role']==role][:count]
              for role,count in (('train',2),('val',1))}
    if len(selected['train'])!=2 or len(selected['val'])!=1:
        raise ValueError('qualification requires first four train and first two val hand samples')
    fields={}
    import hashlib
    for name in store.fields:
        if name not in paths:
            continue
        digest=hashlib.sha256(); spans=[]
        for role,windows in selected.items():
            for w in windows:
                value=np.asarray(store.field(name)[w['start']:w['end']])
                if value.dtype.kind=='f' and not np.isfinite(value).all():
                    raise ValueError('qualification raw fields must be finite; no replacement: '+name)
                span=dict(role=role,start=w['start'],end=w['end'],dtype=value.dtype.str,shape=list(value.shape))
                digest.update(json.dumps(span,sort_keys=True).encode());digest.update(value.tobytes(order='C'))
                spans.append(span)
        fields[name]=dict(selected_span_sha256=digest.hexdigest(),schema=store.fields[name],spans=spans)
    for name,path in paths.items():
        if storage._stamp(path)!=stamps[relative[name]]:
            raise ValueError('qualification source changed during span hashing')
    model=models.resolved_classic_identity(cfg,models.classic_contract(recipe['variant']))
    project=Path(__file__).resolve().parents[1]
    identity=dict(format=FORMAT,scope=SCOPE,recipe=recipe,model=model,policy=cuda_check_policy(variant=recipe['variant']),
        runtime=profile,runtime_files_sha256={name:file_sha256(project/name) for name in RUNTIME_FILES},
        framework_files_sha256={name:file_sha256(Path(pl.__file__).resolve().parent/name) for name in storage.FRAMEWORK_FILES},
        data=dict(scope='bounded selected spans only; no full-cohort admission',fields=fields,
            manifest_sha256=file_sha256(paths['manifest.json']),metadata_sha256=file_sha256(paths['metadata.npz']),
            partition_sha256=storage._json_hash(partitions),selected_windows=selected),
        declared_devices=list(cfg.trainer.devices),declared_batch_size=recipe['batch_size'],
        source_adaptations=['batch1/workers0; single CUDA rank', 'CSVLogger replaces TensorBoard logging only',
          'canonical nondeterministic backend with fixed CUBLAS workspace; no bitwise trajectory promise',
          'one deliberate Inf gradient; no formal best or test'])
    identity=json.loads(json.dumps(identity,sort_keys=True,allow_nan=False));identity['sha256']=storage._json_hash(identity)
    p=_Prepared();p.data,p.upstream,p.cfg,p.recipe,p.partitions,p.identity,p.stamps=root,Path(upstream).resolve(),cfg,recipe,copy.deepcopy(partitions),identity,stamps
    _check_prepared(p)
    return p


class _FixedDataset:
    def __init__(self,dataset,prepared,count):
        self.dataset,self.prepared,self.count=dataset,prepared,count
        if len(dataset)<count:
            raise ValueError('fixed qualification sample prefix unavailable')
    def __len__(self): return self.count
    def __getitem__(self,index):
        if not 0<=index<self.count: raise IndexError(index)
        _check_prepared(self.prepared)
        item=self.dataset[index];datasets.admit_classic_sample(item,self.dataset.contract)
        span=slice(self.dataset.contract.left_context,7790-self.dataset.contract.right_context)
        if not bool(item['label_valid_mask'][span].any()):
            raise ValueError('fixed qualification sample has no supervision; cannot search for replacement')
        _check_prepared(self.prepared)
        item['classic_sample_index']=index
        return item


def _factory(prepared):
    _check_prepared(prepared)
    root,contract,_,_=models._validated(prepared.upstream,prepared.cfg)
    if storage._json_hash(models.resolved_classic_identity(prepared.cfg,contract))!=storage._json_hash(prepared.identity['model']):
        raise ValueError('CUDA model/config identity differs before original factory')
    models._activate(root)
    from egoemg._train_utils import setup_runtime
    from egoemg.train import make_lightning_module
    setup_runtime(prepared.cfg)
    module=make_lightning_module(prepared.cfg)
    model=module.model
    if (sum(p.numel() for p in model.parameters())!=contract.parameters or
        (model.left_context,model.right_context)!=(contract.left_context,contract.right_context) or
        model.out_channels!=20 or model.provide_initial_pos is not False or not module.automatic_optimization):
        raise ValueError('actual CUDA factory differs from classic contract')
    _check_prepared(prepared)
    return module


def _witness(module,batch,precision,contract):
    torch=storage._runtime();storage._admit_batch(batch,contract)
    batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
    module.to('cuda:0');was_training=module.model.training
    try:
        module.model.eval()
        with torch.no_grad(),precision.forward_context(): output=module.model.forward(batch)
    finally:
        module.model.train(was_training)
    if type(output) is not tuple or len(output)!=3:
        raise ValueError('actual CUDA model tuple differs')
    pred,target,mask=output;length=7790-contract.left_context-contract.right_context
    span=slice(contract.left_context,7790-contract.right_context)
    if (pred.shape!=(1,22,length) or pred.dtype not in (torch.float16,torch.float32) or target.shape!=pred.shape or
        mask.shape!=(1,length) or mask.dtype!=torch.bool or not torch.equal(target,batch['joint_angles'][...,span]) or
        not torch.equal(mask,batch['label_valid_mask'][...,span]) or not bool(torch.isfinite(pred).all())):
        raise ValueError('actual CUDA tuple/context/mask/dtype/finite witness differs')
    return dict(prediction_dtype=str(pred.dtype),target_dtype=str(target.dtype),prediction_shape=list(pred.shape),
                mask_shape=list(mask.shape),mask_dtype=str(mask.dtype))


def _resume_snapshot(checkpoint):
    return {name:storage.state_digest(checkpoint[key]) for name,key in
            (('model','state_dict'),('optimizer','optimizer_states'),('scheduler','lr_schedulers'),('scaler','MixedPrecision'))} | {
            'rng':storage.state_digest(checkpoint['cuda_check']['rng']),
            'protocol':storage.state_digest(checkpoint['callbacks']['CudaProtocol'])}


class CudaProtocol(Callback):
    """Observer of real automatic optimization, plus the single explicit fault."""
    def __init__(self,prepared,generators,phase):
        self.prepared,self.generators,self.phase=prepared,generators,phase
        self.events=[];self.validation_events=[];self.executions=0;self.injection_count=0;self.completed_epochs=0
        self.pending_rng=None;self.resume_start=None;self.epoch_boundary=False
        self.inject_next=False;self.injected=False
    def state_dict(self):
        return dict(events=self.events,validation_events=self.validation_events,executions=self.executions,
                    injection_count=self.injection_count,completed_epochs=self.completed_epochs)
    def load_state_dict(self,state):
        for name in ('events','validation_events','executions','injection_count','completed_epochs'):
            setattr(self,name,state[name])
    def on_load_checkpoint(self,trainer,pl_module,checkpoint):
        self.pending_rng=checkpoint['cuda_check']['rng'];self.expected_resume=_resume_snapshot(checkpoint)
    def on_train_start(self,trainer,pl_module):
        optimizer=trainer.optimizers[0]
        if type(optimizer).__name__!='AdamW' or len(trainer.lr_scheduler_configs)!=1 or type(trainer.lr_scheduler_configs[0].scheduler).__name__!='CosineAnnealingLR':
            raise ValueError('actual original AdamW/cosine optimizer differs')
        self.step_hook=optimizer.register_step_post_hook(lambda *_:setattr(self,'executions',self.executions+1))
        parameter=next(p for p in pl_module.parameters() if p.requires_grad)
        def inject(gradient):
            if self.inject_next:
                self.inject_next=False;self.injected=True;self.injection_count+=1
                return gradient.new_full(gradient.shape,float('inf'))
            return gradient
        self.gradient_hook=parameter.register_hook(inject)
    def on_train_epoch_start(self,trainer,pl_module):
        if self.pending_rng is not None:
            storage._restore_rng(self.pending_rng,self.generators);self.pending_rng=None
            snapshot=dict(model=storage.state_digest(pl_module.state_dict()),optimizer=storage.state_digest([o.state_dict() for o in trainer.optimizers]),
                scheduler=storage.state_digest([c.scheduler.state_dict() for c in trainer.lr_scheduler_configs]),
                scaler=storage.state_digest(trainer.precision_plugin.state_dict()),rng=storage.state_digest(storage._rng(self.generators)),
                protocol=storage.state_digest(self.state_dict()))
            if snapshot!=self.expected_resume:
                raise ValueError('actual CUDA model/optimizer/scheduler/scaler/callback/RNG restore differs')
            self.resume_start=snapshot
        self.seen=[];self.epoch_boundary=False
    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        _check_prepared(self.prepared);storage._admit_batch(batch,models.classic_contract(self.prepared.recipe['variant']))
        if len(self.events)>=8: raise ValueError('qualification attempt budget exceeded')
        self.seen.extend(batch['classic_sample_index'].tolist());self.injected=False
        self.inject_next=self.phase=='prefix' and self.executions>=1 and self.injection_count==0
        self.before=dict(parameters=storage.state_digest(dict(pl_module.named_parameters())),step=storage._optimizer_step(trainer.optimizers[0]),
            executions=self.executions,scale=float(trainer.precision_plugin.scaler.get_scale()))
        self.gradients=None
    def on_before_backward(self,trainer,pl_module,loss):
        if not bool(storage._runtime().isfinite(loss)):
            raise ValueError('original CUDA loss must be finite before backward')
    def on_before_optimizer_step(self,trainer,pl_module,optimizer):
        self.gradients=gradient_observation([p.grad for p in pl_module.parameters() if p.grad is not None])
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        step=storage._optimizer_step(trainer.optimizers[0]);executed=self.executions-self.before['executions']
        after=storage.state_digest(dict(pl_module.named_parameters()));scale=float(trainer.precision_plugin.scaler.get_scale())
        if self.gradients is None or executed not in (0,1): raise ValueError('missing actual scaler/optimizer observation')
        if executed:
            if (step!=self.before['step']+1 or after==self.before['parameters'] or not self.gradients['finite'] or
                scale!=self.before['scale']):
                raise ValueError('actual finite optimizer update witness failed')
        elif step!=self.before['step'] or after!=self.before['parameters'] or scale!=self.before['scale']*.5 or self.gradients['finite']:
            raise ValueError('actual overflow skip/backoff/parameter witness failed')
        if self.injected and executed: raise ValueError('injected Inf was not skipped by actual GradScaler')
        loss=float(outputs['loss'].detach())
        if not math.isfinite(loss): raise ValueError('nonfinite original CUDA batch loss')
        self.events.append(dict(epoch=trainer.current_epoch,batch=batch_idx,global_step=trainer.global_step,
            optimizer_executed=bool(executed),optimizer_step_before=self.before['step'],optimizer_step_after=step,
            parameters_sha256_before=self.before['parameters'],parameters_sha256_after=after,
            gradient=self.gradients,scale_before=self.before['scale'],scale_after=scale,injected=self.injected,
            input_sha256=storage.state_digest({k:batch[k] for k in ('emg','joint_angles','label_valid_mask','classic_sample_index')}),loss=loss))
    def on_validation_epoch_start(self,trainer,pl_module): self.val_seen=[];self.val_losses=[]
    def on_validation_batch_start(self,trainer,pl_module,batch,batch_idx,dataloader_idx=0):
        _check_prepared(self.prepared);storage._admit_batch(batch,models.classic_contract(self.prepared.recipe['variant']))
    def on_validation_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
        loss=float(outputs.detach())
        if not math.isfinite(loss): raise ValueError('original CUDA validation loss nonfinite')
        self.val_seen.extend(batch['classic_sample_index'].tolist());self.val_losses.append((1,loss,int(batch['label_valid_mask'].sum())))
    def on_validation_end(self,trainer,pl_module):
        if trainer.sanity_checking: return
        loss=float(trainer.callback_metrics['val_loss']);weighted=storage.batch_weighted_loss(self.val_losses)
        if self.val_seen!=[0,1] or not math.isfinite(loss) or not math.isclose(loss,weighted,rel_tol=2e-6,abs_tol=1e-7):
            raise ValueError('fixed original validation coverage/weighted loss differs')
        self.validation_events.append(dict(epoch=trainer.current_epoch,samples=2,full_fixed_validation=True,
            full_cohort_validation=False,val_loss=loss,batch_weighted_loss=weighted,scientific_eligible=False))
    def on_train_epoch_end(self,trainer,pl_module):
        if sorted(self.seen)!=list(range(4)) or len(self.validation_events)!=trainer.current_epoch+1:
            raise ValueError('fixed qualification epoch incomplete')
        self.completed_epochs=trainer.current_epoch+1;self.epoch_boundary=True
        if self.phase=='prefix':
            if self.injection_count!=1 or self.executions<1:
                raise ValueError('fixed prefix did not achieve update then actual injected skip; no retry')
            trainer.should_stop=True
    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        checkpoint['cuda_check']=dict(format=FORMAT,scope=SCOPE,identity=self.prepared.identity,role='qualification_last',
            completed_epochs=self.completed_epochs,epoch_boundary=self.epoch_boundary,rng=storage._rng(self.generators),
            model_schema=_tensor_schema(checkpoint['state_dict']),optimizer_schema=optimizer_schema(checkpoint['optimizer_states']),
            scaler_schema=scaler_schema(checkpoint['MixedPrecision']),resume_start=self.resume_start)
    def teardown(self,trainer,pl_module,stage):
        for name in ('step_hook','gradient_hook'):
            hook=getattr(self,name,None)
            if hook is not None: hook.remove()


class CudaCheckpointIO(CheckpointIO):
    def __init__(self,directory,identity,resume=None,admitted=None):
        self.directory,self.identity=storage._safe(directory),identity
        self.resume=storage._safe(resume) if resume else None;self.admitted=admitted
    def _owned(self,path):
        path=storage._safe(path)
        if path!=self.directory/'last.ckpt': raise ValueError('CUDA checkpoint may write only this owned last.ckpt')
        return path
    def save_checkpoint(self,checkpoint,path,storage_options=None):
        if storage_options is not None: raise ValueError('remote CUDA checkpoint storage unsupported')
        path=self._owned(path);validate_checkpoint_state(checkpoint,self.identity)
        checkpoint['cuda_check_seal']=_seal(checkpoint)
        budget=2*storage._payload_bytes(checkpoint)+storage.IO_OVERHEAD
        storage._atomic(path,lambda stream:storage._runtime().save(checkpoint,stream),budget)
    def load_checkpoint(self,path,map_location=None):
        path=storage._safe(path)
        if self.resume is None or path!=self.resume or self.admitted is None: raise ValueError('unadmitted CUDA resume path')
        checkpoint=read_cuda_checkpoint(path,self.identity)
        if checkpoint['cuda_check_seal']!=self.admitted['cuda_check_seal']:
            raise ValueError('CUDA resume changed after pre-factory admission')
        value=copy.deepcopy(checkpoint)
        for name,state in value['callbacks'].items():
            if name.startswith('ModelCheckpoint{'):
                state['dirpath']=str(self.directory);state['last_model_path']=str(self.directory/'last.ckpt')
        return value
    def remove_checkpoint(self,path):
        self._owned(path).unlink(missing_ok=True);storage._fsync_dir(self.directory)


def _publish_result(output,result):
    storage._write_json(output/'result.json',result)
    storage._write_json(output/'status.json',dict(state='qualification_phase_finished',scope=SCOPE,
        scientific_eligible=False,production_ready=False,result_sha256=file_sha256(output/'result.json')))


class _OwnedOutput:
    """Creation-time directory identity; all failure mutations use this dirfd."""
    def __init__(self,path):
        self.path=storage._safe(path)
        self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        item=os.fstat(self.fd);self.identity=(item.st_dev,item.st_ino)
        if not self.is_current(self.path):
            self.close();raise ValueError('new CUDA output directory identity changed during capture')
    def is_current(self,path):
        try:
            path=storage._safe(path)
            if path!=self.path or self.fd is None: return False
            current=os.stat(path,follow_symlinks=False);opened=os.fstat(self.fd)
            return (stat.S_ISDIR(current.st_mode) and (current.st_dev,current.st_ino)==self.identity and
                    (opened.st_dev,opened.st_ino)==self.identity and storage._safe(path)==self.path)
        except (OSError,ValueError): return False
    def space(self,budget=0):
        # statvfs accepts a directory fd on Linux. Never follow a rebound path.
        if storage.shutil.disk_usage(self.fd).free<storage.RESERVE_BYTES+budget:
            raise OSError('classic output requires 15GiB reserve plus temporary payload space')
    def close(self):
        if self.fd is not None: os.close(self.fd);self.fd=None
    def __enter__(self): return self
    def __exit__(self,*_): self.close()


def _capture_owned_output(output):
    """Capture immediately after this run exclusively creates its directory."""
    return _OwnedOutput(output)


def _write_failed_status(output,owned,exc):
    if not owned.is_current(output): return
    payload=(json.dumps(dict(state='failed',scope=SCOPE,scientific_eligible=False,production_ready=False,
        error_type=type(exc).__name__,error=str(exc)),sort_keys=True,allow_nan=False)+'\n').encode()
    if len(payload)>storage.LOG_LIMIT_BYTES: raise OSError('CUDA failed status exceeds bounded log size')
    owned.space(len(payload)+4096)
    temporary='.failed-status.'+uuid.uuid4().hex+'.incomplete'
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=owned.fd)
    try:
        try:
            for offset in range(0,len(payload),65536):
                chunk=payload[offset:offset+65536];owned.space(len(chunk))
                if os.write(descriptor,chunk)!=len(chunk): raise OSError('short CUDA failed-status write')
                owned.space()
            os.fsync(descriptor)
        finally: os.close(descriptor)
        if not owned.is_current(output): return
        owned.space()
        os.replace(temporary,'status.json',src_dir_fd=owned.fd,dst_dir_fd=owned.fd)
        os.fsync(owned.fd)
    finally:
        try: os.unlink(temporary,dir_fd=owned.fd)
        except FileNotFoundError: pass


def _fail_owned_output(output,exc,*,owned=None):
    # No creation-time capability means no authority to delete or publish.
    # Even a rebind after this guard can mutate only the captured directory.
    if not isinstance(owned,_OwnedOutput) or not owned.is_current(output): return
    try:
        try: os.unlink('status.json',dir_fd=owned.fd)
        except FileNotFoundError: pass
        os.fsync(owned.fd)
    except OSError: pass
    try: _write_failed_status(output,owned,exc)
    except (OSError,ValueError): pass


def _process_identity(phase):
    raw=Path('/proc/self/stat').read_text()
    return dict(pid=os.getpid(),start_ticks=int(raw.rsplit(')',1)[1].split()[19]),phase=phase)


def run_cuda_check(data,*,upstream,config,partitions,output,device='cuda:0',phase='prefix',resume=None):
    if phase not in ('prefix','resume'): raise ValueError('qualification phase must be prefix or resume')
    if phase=='resume' and resume is None: raise ValueError('resume phase requires an admitted prefix checkpoint')
    if phase=='prefix' and resume is not None: raise ValueError('prefix phase cannot consume a resume checkpoint')
    if device!='cuda:0': raise ValueError('only root-selected cuda:0 is permitted')
    output=storage._safe(output);storage._space(output,storage.IO_OVERHEAD)
    if output.exists(): raise FileExistsError('CUDA qualification output exists; no overwrite or automatic restart')
    recipe=datasets.load_classic_config(config,upstream)  # Config-only, no context/scaler/data/factory.
    cuda_check_policy(variant=recipe['variant'],device=device,phase=phase)
    profile,precision=_profile()  # Metadata rejection precedes actual GPU allocation.
    prepared=_prepare(data,upstream,config,partitions,profile,expected_recipe=recipe)
    admitted=read_cuda_checkpoint(resume,prepared.identity) if resume else None
    if admitted is not None and admitted['cuda_check']['completed_epochs']!=1:
        raise ValueError('resume must begin at exactly epoch one of the same two-epoch plan')
    module=_factory(prepared)
    if admitted is not None: storage._strict_state(module,admitted)
    full=datasets.build_classic_datasets(prepared.data,prepared.partitions,upstream=prepared.upstream,
        config=prepared.cfg,normalization=prepared.recipe['normalization'],mask_policy=prepared.recipe['mask_policy'])
    fixed={role:_FixedDataset(full[role],prepared,n) for role,n in (('train',4),('val',2))}
    torch=storage._runtime()
    witness=_witness(module,torch.utils.data.default_collate([fixed['val'][0]]),precision,models.classic_contract(prepared.recipe['variant']))
    generators={name:torch.Generator().manual_seed(prepared.recipe['seed']+n) for n,name in enumerate(('train_loader','train_sampler','val_loader'))}
    train=torch.utils.data.DataLoader(fixed['train'],batch_size=1,num_workers=0,generator=generators['train_loader'],
        sampler=torch.utils.data.RandomSampler(fixed['train'],generator=generators['train_sampler']))
    val=torch.utils.data.DataLoader(fixed['val'],batch_size=1,num_workers=0,shuffle=False,generator=generators['val_loader'])
    observer=CudaProtocol(prepared,generators,phase)
    from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
    monitor=ModelCheckpoint(dirpath=str(output/'checkpoints'),save_top_k=0,save_last=True,save_on_train_epoch_end=True)
    io=CudaCheckpointIO(output/'checkpoints',prepared.identity,resume,admitted)
    # Budget covers full model + AdamW moments + duplicate serialization staging.
    storage._space(output,2*sum(p.numel()*p.element_size() for p in module.parameters())*4+storage.IO_OVERHEAD)
    output.mkdir(parents=True,exist_ok=False)
    owned=_capture_owned_output(output)
    try:
        os.mkdir('checkpoints',dir_fd=owned.fd)
        if not owned.is_current(output): raise ValueError('CUDA output namespace no longer owned')
        storage._write_json(output/'identity.json',prepared.identity)
        storage._write_json(output/'provenance.json',dict(data=str(prepared.data),upstream=str(prepared.upstream),
            config=str(Path(config).absolute()),python=sys.executable,sys_path=list(sys.path),source_stamps=prepared.stamps))
        storage._write_json(output/'status.json',dict(state='running',scope=SCOPE,scientific_eligible=False,production_ready=False))
        logger=storage.ClassicCSVLogger(str(output),name='lightning',version='cuda-qualification')
        trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=2,max_steps=-1,gradient_clip_val=0,
            callbacks=[observer,LearningRateMonitor(logging_interval='epoch'),monitor],plugins=[io,precision],logger=logger,
            num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1,
            check_val_every_n_epoch=1,limit_train_batches=1.0,limit_val_batches=1.0,
            default_root_dir=str(output),deterministic=False)
        torch.cuda.reset_peak_memory_stats(0)
        trainer.fit(module,train_dataloaders=train,val_dataloaders=val,ckpt_path=str(storage._safe(resume)) if resume else None)
        _check_prepared(prepared)
        checkpoint=read_cuda_checkpoint(output/'checkpoints/last.ckpt',prepared.identity)
        expected_epochs=1 if phase=='prefix' else 2
        if observer.completed_epochs!=expected_epochs or len(observer.events)!=expected_epochs*4 or observer.executions<(1 if phase=='prefix' else 2):
            raise ValueError('bounded CUDA phase lacks actual required completed updates/epochs')
        result=dict(format=FORMAT,scope=SCOPE,scientific_eligible=False,production_ready=False,phase=phase,
            runtime_profile=BACKEND_PROFILE,resume_scope=RESUME_SCOPE,
            trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False,
            identity=prepared.identity,completed_epochs=observer.completed_epochs,attempts=len(observer.events),
            optimizer_executions=observer.executions,injection_count=observer.injection_count,events=observer.events,
            validation_events=observer.validation_events,witness=witness,resume_start=observer.resume_start,
            checkpoint_sha256=file_sha256(output/'checkpoints/last.ckpt'),checkpoint_seal=checkpoint['cuda_check_seal'],
            checkpoint_end_snapshot=_resume_snapshot(checkpoint),rng_sha256=storage.state_digest(checkpoint['cuda_check']['rng']),
            process=_process_identity(phase),memory=dict(max_allocated=torch.cuda.max_memory_allocated(0),
                max_reserved=torch.cuda.max_memory_reserved(0),free_and_total=list(torch.cuda.mem_get_info(0))),
            limitations=prepared.identity['policy']['limits'])
        _publish_result(output,result)
        return result
    except BaseException as exc:
        _fail_owned_output(output,exc,owned=owned)
        raise
    finally: owned.close()


def _read_phase(directory,phase):
    directory=storage._safe(directory)
    records={name:datasets._safe_path(directory,name) for name in ('status.json','result.json','identity.json','checkpoints/last.ckpt')}
    stamps={name:storage._stamp(path) for name,path in records.items()}
    status=json.loads(records['status.json'].read_text());result=json.loads(records['result.json'].read_text())
    identity=json.loads(records['identity.json'].read_text());_check_identity(identity)
    if (status.get('state')!='qualification_phase_finished' or status.get('scientific_eligible') is not False or
        status.get('production_ready') is not False or status.get('result_sha256')!=file_sha256(records['result.json']) or
        result['format']!=FORMAT or result['scope']!=SCOPE or result['phase']!=phase or result['identity']!=identity or
        result['scientific_eligible'] is not False or result['production_ready'] is not False):
        raise ValueError('CUDA phase final artifact identity/status differs')
    checkpoint=read_cuda_checkpoint(records['checkpoints/last.ckpt'],identity)
    if (result['checkpoint_sha256']!=file_sha256(records['checkpoints/last.ckpt']) or
        result['checkpoint_seal']!=checkpoint['cuda_check_seal'] or result['events']!=checkpoint['callbacks']['CudaProtocol']['events'] or
        result['checkpoint_end_snapshot']!=_resume_snapshot(checkpoint)):
        raise ValueError('CUDA result/checkpoint observations differ')
    for name,path in records.items():
        if storage._stamp(datasets._safe_path(directory,name))!=stamps[name]:
            raise ValueError('CUDA phase files changed during verification')
    return result,checkpoint


def verify_cuda_check(prefix,resumed):
    """Read-only verification of real artifacts, never CPU/fabricated credentials."""
    a,ca=_read_phase(prefix,'prefix');b,cb=_read_phase(resumed,'resume')
    if a['identity']!=b['identity'] or a['completed_epochs']!=1 or b['completed_epochs']!=2:
        raise ValueError('CUDA prefix/resume identity or epoch mismatch')
    if (a['process']['pid'],a['process']['start_ticks'])==(b['process']['pid'],b['process']['start_ticks']):
        raise ValueError('strict CUDA reload requires a separate process')
    if b['resume_start']!=a['checkpoint_end_snapshot'] or cb['cuda_check']['resume_start']!=b['resume_start']:
        raise ValueError('CUDA strict full-state epoch reload evidence differs')
    events=b['events']
    if len(events)!=8 or a['events']!=events[:4] or b['attempts']!=8 or a['attempts']!=4:
        raise ValueError('CUDA bounded attempt history differs')
    injected=[e for e in events if e['injected']]
    if len(injected)!=1 or b['injection_count']!=1:
        raise ValueError('CUDA exactly one injected skip is required')
    fault=injected[0];index=events.index(fault)
    if (fault['optimizer_executed'] or fault['gradient']['finite'] or fault['gradient']['absolute_max'] is not None or
        fault['optimizer_step_before']!=fault['optimizer_step_after'] or
        fault['parameters_sha256_before']!=fault['parameters_sha256_after'] or fault['scale_after']!=fault['scale_before']*.5 or
        not any(e['optimizer_executed'] for e in events[:index]) or not any(e['optimizer_executed'] for e in events[index+1:])):
        raise ValueError('CUDA actual optimizer skip/backoff and subsequent finite update evidence failed')
    executions=sum(e['optimizer_executed'] for e in events)
    if executions<2 or executions!=b['optimizer_executions'] or executions!=cb['callbacks']['CudaProtocol']['executions']:
        raise ValueError('CUDA actual optimizer execution count differs')
    for result,checkpoint,count in ((a,ca,1),(b,cb,2)):
        for key in ('runtime_profile','resume_scope','trajectory_agreement_qualified','bitwise_trajectory_guaranteed'):
            if storage._json_hash(result[key])!=storage._json_hash(result['identity']['policy'][key]):
                raise ValueError('CUDA phase profile/trajectory claim differs from policy')
        if len(result['validation_events'])!=count or any(e['samples']!=2 or e['full_fixed_validation'] is not True or
            e['full_cohort_validation'] is not False for e in result['validation_events']):
            raise ValueError('CUDA fixed original validation evidence incomplete')
        if result['rng_sha256']!=storage.state_digest(checkpoint['cuda_check']['rng']):
            raise ValueError('CUDA RNG evidence differs')
        if result['witness']['prediction_dtype'] not in ('torch.float16','torch.float32'):
            raise ValueError('CUDA actual prediction dtype missing')
    return dict(format=FORMAT,scope=SCOPE,cuda_numeric_passed=True,scientific_eligible=False,production_ready=False,
        runtime_profile=BACKEND_PROFILE,resume_scope=RESUME_SCOPE,trajectory_agreement_qualified=False,
        bitwise_trajectory_guaranteed=False,strict_state_restore=True,
        identity_sha256=b['identity']['sha256'],attempts=8,optimizer_executions=executions,injected_skips=1,
        strict_resume=True,prefix_checkpoint_sha256=a['checkpoint_sha256'],resumed_checkpoint_sha256=b['checkpoint_sha256'],
        limitations=b['limitations'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for phase in ('prefix','resume'):
        p=sub.add_parser(phase)
        for name in ('data','upstream','config','partitions','output'): p.add_argument('--'+name,required=True,type=Path)
        p.add_argument('--device',default='cuda:0')
        if phase=='resume': p.add_argument('--checkpoint',required=True,type=Path)
    p=sub.add_parser('verify');p.add_argument('--prefix',required=True,type=Path);p.add_argument('--resumed',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.command=='verify': result=verify_cuda_check(args.prefix,args.resumed)
    else: result=run_cuda_check(args.data,upstream=args.upstream,config=args.config,
        partitions=json.loads(args.partitions.read_text()),output=args.output,device=args.device,phase=args.command,
        resume=getattr(args,'checkpoint',None))
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
