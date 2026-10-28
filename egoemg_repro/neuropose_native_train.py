"""Fresh original NeuroPose epochs; no checkpoint continuation interface.

The old resumed-trajectory comparison is neither changed nor accepted here.
Full execution requires a separate actual CUDA frozen-model qualification.
Trusted local native checkpoint pickle only; engineering is never scientific.
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
import time

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FORMAT='native_neuropose_uninterrupted_fit_v1'
COUNTS=dict(train=101714,val=1576,test=4306)
SCOPES=('full','cuda_engineering','cpu_engineering')


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def _sealed(value):
    value=copy.deepcopy(value);value['sha256']=_hash(value);return value


def _check(value):
    if value.get('sha256')!=_hash({k:v for k,v in value.items() if k!='sha256'}):raise ValueError('native NeuroPose seal differs')


def _json(path):
    from .classic_emg_cohort import _safe
    p=_safe(path)
    if p.stat().st_size>64*1024**2:raise ValueError('bounded native JSON required')
    return json.loads(p.read_text())


def _write(path,value):
    from . import classic_emg_train as storage
    p=storage._safe(path)
    if p.exists():raise FileExistsError('native evidence already exists')
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    storage._atomic(p,lambda f:f.write(raw),len(raw)+4096,replace=False)


def _rng_packet(generators,*,cpu):
    from . import classic_emg_train as storage
    if not cpu:return storage._rng(generators)
    import random,numpy as np
    torch=storage._runtime()
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=[],generators={name:g.get_state() for name,g in generators.items()})


def _sources():
    from . import classic_emg_formal as formal
    if _sha(__file__)!=IMPORTED_SHA:raise ValueError('imported native NeuroPose source changed')
    return dict(formal._source_snapshot(),**{'egoemg_repro/neuropose_native_train.py':IMPORTED_SHA})


def neuropose_policy(*,scope='full'):
    if type(scope) is not str or scope not in SCOPES:raise ValueError('explicit native NeuroPose scope required')
    cpu=scope=='cpu_engineering';full=scope=='full'
    return dict(scope=scope,variant='neuropose',scientific_eligible=False,batch_size=2 if cpu else 256,
        counts=COUNTS.copy() if full else None,engineering_train_limit=None if full else 4 if cpu else 512,
        num_workers=0,monitor='val_loss',mode='min',patience=50,
        checkpoint_continuation_supported=False,training_trajectory_agreement_claimed=False,
        trainer=dict(accelerator='cpu' if cpu else 'gpu',devices=1,precision='32-true' if cpu else '16-mixed',
            max_epochs=100 if full else 2,max_steps=-1,limit_train_batches=1.0,limit_val_batches=1.0,
            limit_test_batches=1.0,check_val_every_n_epoch=1,val_check_interval=1.0,
            accumulate_grad_batches=1,gradient_clip_val=0,log_every_n_steps=50,num_sanity_val_steps=0,deterministic=False))


def _precondition_paths(preconditions):
    from .classic_emg_cohort import _safe
    if (not isinstance(preconditions,dict) or set(preconditions)!={'numeric_prefix','numeric_resume','capacity'} or
        not isinstance(preconditions['capacity'],list) or len(preconditions['capacity'])!=3 or
        any(not isinstance(v,(str,os.PathLike)) for v in [preconditions['numeric_prefix'],preconditions['numeric_resume'],*preconditions['capacity']])):
        raise ValueError('original raw numeric and three capacity precondition artifact paths required')
    return {k:([str(_safe(x)) for x in v] if k=='capacity' else str(_safe(v))) for k,v in preconditions.items()}


def _raw_preconditions(paths):
    files={}
    for role in ('numeric_prefix','numeric_resume'):
        for n in ('identity.json','result.json','status.json','checkpoints/last.ckpt'):
            p=Path(paths[role])/n;files[str(p)]=_sha(p)
    for directory in paths['capacity']:
        for n in ('identity.json','result.json','status.json'):
            p=Path(directory)/n;files[str(p)]=_sha(p)
    return files


@dataclass
class PreparedNeuro:
    identity:dict
    policy:dict
    cfg:object
    module:object
    datasets:dict
    guard:object
    precision:object=None
    selected_checkpoint:dict|None=None
    selection:dict|None=None
    run:Path|None=None

    def check(self):
        self.guard.check();_check(self.identity)
        if self.identity['sources']!=_sources() or self.policy!=neuropose_policy(scope=self.identity['policy']['scope']):
            raise ValueError('native prepared source/policy differs')


def _prepare(data,*,upstream,config,partitions,scope,preconditions,cohort_receipt,finite_probe,qualification,construct=True):
    from . import classic_emg_train as storage,classic_emg_formal as formal
    from .classic_emg_cohort import _safe,_StatGuard
    from .classic_emg_data import load_classic_config,build_classic_datasets
    from omegaconf import OmegaConf
    policy=neuropose_policy(scope=scope);cpu=scope=='cpu_engineering';qual=None;paths=None
    # Full qualification is checked before any data/model/output access.
    if scope=='full':
        if not isinstance(qualification,(str,os.PathLike)):raise ValueError('actual new raw CUDA qualification artifact required')
        from .neuropose_native_eval import read_neuropose_qualification
        qual=read_neuropose_qualification(qualification)
    if not cpu:
        paths=_precondition_paths(preconditions)
        if not isinstance(cohort_receipt,(str,os.PathLike)):raise ValueError('original complete cohort receipt artifact required')
    elif any(x is not None for x in (preconditions,cohort_receipt,finite_probe,qualification)):
        raise ValueError('CPU engineering cannot inherit a CUDA/full qualification')
    recipe=load_classic_config(config,upstream)
    if recipe['variant']!='neuropose':raise ValueError('original NeuroPose variant required')
    if not isinstance(partitions,(str,os.PathLike)):raise ValueError('frozen partition file artifact required')
    locations=dict(data=str(_safe(data)),upstream=str(_safe(upstream)),config=str(_safe(config)),partitions=str(_safe(partitions)),
        cohort_receipt=str(_safe(cohort_receipt)) if cohort_receipt is not None else None,
        finite_probe=str(_safe(finite_probe)) if finite_probe is not None else None,
        qualification=str(_safe(qualification)) if qualification is not None else None)
    partition=_json(locations['partitions'])
    sources=_sources();guard=_StatGuard([ROOT/n for n in sources]+[locations['config'],locations['partitions']])
    module=None;precision=None
    if cpu:
        old_policy=storage.runtime_policy(engineering=True,batch_size=2,engineering_epochs=2)
        base=storage.prepare_classic_run(data,upstream=upstream,config=config,partitions=partition,policy=old_policy)
        cfg=base.cfg;input_identity=base.identity['data'];runtime=base.identity['versions'];framework=base.identity['framework_files_sha256']
        for name in base.stamps:guard.add(base.data/name)
        from .classic_emg import _activate
        _activate(base.upstream)
        if construct:
            from egoemg._train_utils import setup_runtime
            from egoemg.train import make_lightning_module
            storage.pl.seed_everything(42,workers=True);setup_runtime(cfg);module=make_lightning_module(cfg)
    else:
        if construct:
            base=formal.prepare_formal_run(data,upstream=upstream,config=config,partitions=locations['partitions'],
                cohort_receipt=cohort_receipt,finite_probe=finite_probe,qualification=dict(paths,final_acceptance=None),engineering=True)
            cfg=base.factory_prepared.cfg;module=base.module;precision=base.precision
            runtime=base.identity['runtime'];framework=base.identity['framework_files_sha256']
            for p in base.guard.paths:guard.add(p)
        else:
            # CPU-safe read path: reopen actual complete inputs and raw evidence.
            complete=formal._cohort_guard(data,locations['partitions'],upstream=upstream,config=config,
                receipt=cohort_receipt,finite_probe=finite_probe)
            for p in complete.paths:guard.add(p)
            from .classic_emg import compose_classic
            cfg=compose_classic(upstream,'neuropose');runtime=None;framework=None
        receipt=_json(cohort_receipt);input_identity=receipt
        if {k:v['hand_samples'] for k,v in receipt['footprints'].items()}!=COUNTS:raise ValueError('actual complete registered NeuroPose cohort required')
    datasets=build_classic_datasets(data,partition,upstream=upstream,config=cfg,
        normalization=recipe['normalization'],mask_policy=recipe['mask_policy'])
    actual={r:len(ds) for r,ds in datasets.items()}
    if not cpu and actual!=COUNTS:raise ValueError('actual complete native dataset counts differ')
    counts=dict(actual)
    if scope!='full':counts['train']=min(counts['train'],policy['engineering_train_limit'])
    if min(counts.values())<=0:raise ValueError('all native roles must be nonempty')
    raw=_raw_preconditions(paths) if paths is not None else None
    if raw is not None:
        for p in raw:guard.add(p)
    identity=_sealed(dict(format=FORMAT,policy=policy,recipe=recipe,input_identity=input_identity,
        counts=counts,underlying_counts=actual,locations=locations,preconditions=paths,precondition_files_sha256=raw,
        qualification=qual,sources=sources,runtime=runtime,framework=framework,
        resolved_config=OmegaConf.to_container(cfg,resolve=True),scientific_eligible=False,full_paper_complete=False))
    if qual is not None:
        # Task2's reader supplies this explicit source/cohort/precondition binding.
        if (qual['training_sources']!=sources or qual['cohort_sha256']!=input_identity['sha256'] or
            qual['precondition_files_sha256']!=raw):raise ValueError('actual new CUDA qualification differs from full inputs/source/preconditions')
    guard.check();return PreparedNeuro(identity,policy,cfg,module,datasets,guard,precision)


def _part(batch,output):
    from . import classic_emg_train as storage
    torch=storage._runtime()
    if type(output) is not tuple or len(output)!=3:raise ValueError('original NeuroPose tuple required')
    pred,target,mask=output;n=len(batch['emg'])
    if (pred.shape!=target.shape or pred.shape!=(n,22,7790) or mask.shape!=(n,7790) or mask.dtype!=torch.bool or
        target.dtype!=torch.float32 or pred.dtype not in (torch.float16,torch.float32) or not bool(torch.isfinite(pred).all()) or
        not torch.equal(target,batch['joint_angles']) or not torch.equal(mask,batch['label_valid_mask'])):
        raise ValueError('original full NeuroPose prediction/target/generated mask differs')
    return storage.state_digest(dict(prediction=pred,target=target,mask=mask,emg=batch['emg'],indices=batch['classic_sample_index']))


def _observer_type():
    from . import classic_emg_formal as formal,classic_emg_train as storage
    class NativeObserver(formal.ValidationCoverage):
        def __init__(self,prepared,output=None,generators=None):
            super().__init__(samples=prepared.identity['counts']['val'],batch_size=prepared.policy['batch_size'])
            self.prepared=prepared;self.output=output;self.generators=generators or {};self.records=[]
            self.attempts=0;self.steps=0;self.boundary=False;self.phase=None;self.step_hook=None
            self.initial_model_sha256=storage.state_digest(prepared.module.state_dict())
        def on_fit_start(self,trainer,module):self._observe(module)
        def on_validation_start(self,trainer,module):
            if not hasattr(self,'original_forward'):self._observe(module)
        def _observe(self,module):
            self.original_forward=module.forward
            def forward(batch):
                value=self.original_forward(batch)
                if self.phase=='val':self.parts.append(_part(batch,value))
                return value
            module.forward=forward
        def restore_forward(self,module):
            if hasattr(self,'original_forward'):module.forward=self.original_forward
            if self.step_hook is not None:self.step_hook.remove();self.step_hook=None
        def on_train_start(self,trainer,module):
            torch=storage._runtime();self.prepared.check()
            if (len(trainer.optimizers)!=1 or type(trainer.optimizers[0]).__name__!='AdamW' or
                len(trainer.lr_scheduler_configs)!=1 or type(trainer.lr_scheduler_configs[0].scheduler).__name__!='CosineAnnealingLR'):
                raise ValueError('original native AdamW/Cosine required')
            ids=[id(p) for g in trainer.optimizers[0].param_groups for p in g['params']]
            if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in module.parameters() if p.requires_grad}:
                raise ValueError('all original trainable parameters required exactly once')
            scaler=getattr(trainer.precision_plugin,'scaler',None);cpu=self.prepared.policy['scope']=='cpu_engineering'
            if (cpu and scaler is not None) or (not cpu and (scaler is None or not scaler.is_enabled() or next(module.parameters()).device.type!='cuda')):
                raise ValueError('actual declared CPU32 or original CUDA16/scaler required')
            self.step_hook=trainer.optimizers[0].register_step_post_hook(lambda *a:self._executed())
        def _executed(self):self.steps+=1
        def on_train_epoch_start(self,trainer,module):
            self.boundary=False;self.phase='train';self.train_loss=0.
            self.training=formal.Coverage(self.prepared.identity['counts']['train'],ordered=False,batch_size=self.prepared.policy['batch_size'])
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            self.prepared.check();formal.admit_formal_batch(batch,formal.cuda.models.classic_contract('neuropose'))
            self.training.add(batch['classic_sample_index'].tolist())
        def on_before_backward(self,trainer,module,loss):
            if not bool(storage._runtime().isfinite(loss)):raise ValueError('native training loss nonfinite')
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            loss=float(outputs['loss'].detach())
            if not math.isfinite(loss):raise ValueError('native training loss nonfinite')
            self.attempts+=1;self.train_loss+=len(batch['emg'])*loss
        def on_validation_epoch_start(self,trainer,module):
            self.prepared.check();self.phase='val';self.parts=[]
            super().on_validation_epoch_start(trainer,module)
        def on_validation_batch_start(self,trainer,module,batch,batch_idx,dataloader_idx=0):
            self.prepared.check();formal.admit_formal_batch(batch,formal.cuda.models.classic_contract('neuropose'))
        def validation_completed(self,trainer,module,event):pass
        def on_validation_end(self,trainer,module):
            super().on_validation_end(trainer,module)
            self.validation=dict(samples=self.validation_samples,val_loss=float(trainer.callback_metrics['val_loss']),
                prediction_sha256=_hash(self.parts),model_sha256=storage.state_digest(module.state_dict()))
            self.phase=None
        def on_train_epoch_end(self,trainer,module):
            coverage=self.training.finish();row=dict(epoch=int(trainer.current_epoch),train_samples=coverage['samples'],
                train_batches=len(coverage['batch_sizes']),train_loss=self.train_loss/coverage['samples'],
                attempts=self.attempts,optimizer_executions=self.steps,validation=copy.deepcopy(self.validation))
            if row['epoch']!=len(self.records):raise ValueError('native complete epoch order differs')
            self.records.append(row);self.boundary=True
            raw=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
            with (self.output/'epochs.jsonl').open('ab') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        def on_save_checkpoint(self,trainer,module,checkpoint):
            if not self.boundary or not self.records:raise ValueError('native complete epoch checkpoint required')
            self.prepared.check()
            checkpoint['native_neuropose']=dict(identity=self.prepared.identity,epoch_boundary=True,records=copy.deepcopy(self.records),
                completed_epochs=len(self.records),attempts=self.attempts,optimizer_executions=self.steps,
                initial_model_sha256=self.initial_model_sha256,
                rng=_rng_packet(self.generators,cpu=self.prepared.policy['scope']=='cpu_engineering'))
            storage._space(self.output,2*storage._payload_bytes(checkpoint)+storage.IO_OVERHEAD)
    return NativeObserver


def _loaders(prepared):
    from . import classic_emg_train as storage,classic_emg_formal as formal
    torch=storage._runtime();generators={k:torch.Generator().manual_seed(42+i) for i,k in enumerate(('train_loader','train_sampler','val_loader'))}
    datasets={r:formal._Dataset(prepared.datasets[r],prepared,prepared.identity['counts'][r]) for r in ('train','val')}
    train=torch.utils.data.DataLoader(datasets['train'],batch_size=prepared.policy['batch_size'],num_workers=0,
        generator=generators['train_loader'],sampler=torch.utils.data.RandomSampler(datasets['train'],generator=generators['train_sampler']))
    val=torch.utils.data.DataLoader(datasets['val'],batch_size=prepared.policy['batch_size'],num_workers=0,shuffle=False,generator=generators['val_loader'])
    return dict(train=train,val=val),generators


def _callback(checkpoint,name):
    matches=[v for k,v in checkpoint['callbacks'].items() if k.startswith(name)]
    if len(matches)!=1:raise ValueError('exact native '+name+' callback required')
    return matches[0]


def _read_epochs(output):
    raw=(output/'epochs.jsonl').read_bytes()
    if not raw.endswith(b'\n'):raise ValueError('complete native epoch journal required')
    return [json.loads(x) for x in raw.splitlines()]


def _validate_checkpoint(checkpoint,identity,records):
    from . import classic_emg_train as storage
    packet=checkpoint['native_neuropose'];epochs=len(records)
    if (packet['identity']!=identity or packet['epoch_boundary'] is not True or packet['records']!=records or
        packet['completed_epochs']!=epochs or packet['optimizer_executions']!=records[-1]['optimizer_executions'] or
        packet['attempts']!=records[-1]['attempts'] or checkpoint['epoch']!=epochs-1 or
        storage.state_digest(checkpoint['state_dict'])!=records[-1]['validation']['model_sha256']):
        raise ValueError('native checkpoint/actual complete epoch/model binding differs')
    if any(k not in checkpoint for k in ('optimizer_states','lr_schedulers','loops','callbacks')):
        raise ValueError('complete ordinary native training state required')
    torch=storage._runtime()
    def finite(value):
        if torch.is_tensor(value):
            if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(torch.isfinite(value).all()):
                raise ValueError('native model/optimizer/scheduler state nonfinite')
        elif isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,(tuple,list)):
            for item in value:finite(item)
        elif isinstance(value,float) and not math.isfinite(value):raise ValueError('native training state nonfinite')
    for name in ('state_dict','optimizer_states','lr_schedulers'):finite(checkpoint[name])
    if len(checkpoint['optimizer_states'])!=1 or len(checkpoint['lr_schedulers'])!=1:raise ValueError('native single optimizer/scheduler state required')
    optimizer=checkpoint['optimizer_states'][0]
    steps=[float(state['step']) for state in optimizer['state'].values()]
    if (not steps or max(steps)!=packet['optimizer_executions'] or
        any(not n.is_integer() or n<=0 or n>packet['optimizer_executions'] for n in steps) or
        checkpoint['global_step']!=packet['attempts']):raise ValueError('actual native optimizer/attempt counters differ')
    if 'rng' not in packet or not isinstance(packet['rng'],dict):raise ValueError('complete native RNG state required')
    rng=packet['rng']
    if (set(rng)!={'python','numpy','torch','cuda','generators'} or
        set(rng['generators'])!={'train_loader','train_sampler','val_loader'} or
        not isinstance(rng['cuda'],list) or
        any(not torch.is_tensor(t) or t.dtype!=torch.uint8 or t.ndim!=1 or not t.numel()
            for t in [rng['torch'],*rng['cuda'],*rng['generators'].values()])):
        raise ValueError('complete native typed RNG streams required')
    try:
        import random,numpy as np
        random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
        for t in [rng['torch'],*rng['generators'].values()]:torch.Generator().set_state(t)
    except (TypeError,ValueError,RuntimeError) as exc:raise ValueError('native saved RNG state malformed') from exc
    initial=packet.get('initial_model_sha256')
    if not isinstance(initial,str) or len(initial)!=64 or any(c not in '0123456789abcdef' for c in initial):
        raise ValueError('native actual initial model fingerprint required')
    if identity['policy']['scope']!='cpu_engineering' and 'native_amp_scaling_state' not in checkpoint and 'MixedPrecision' not in checkpoint:
        raise ValueError('original native FP16 scaler state missing')


def _selection(output,identity,last,records):
    from . import classic_emg_train as storage
    monitor=_callback(last,'ModelCheckpoint');path=storage._safe(monitor['best_model_path'])
    if path.parent!=output/'checkpoints' or path.name=='last.ckpt':raise ValueError('actual selected best checkpoint required')
    torch=storage._runtime();best=torch.load(str(path),map_location='cpu',weights_only=False)
    n=best['native_neuropose']['completed_epochs']
    if type(n) is not int or not 1<=n<=len(records):raise ValueError('selected native best epoch outside complete journal')
    _validate_checkpoint(best,identity,records[:n]);score=records[n-1]['validation']['val_loss']
    if best['native_neuropose']['initial_model_sha256']!=last['native_neuropose']['initial_model_sha256']:
        raise ValueError('native selected best/last initialization differs')
    if score!=min(row['validation']['val_loss'] for row in records) or float(monitor['best_model_score'])!=score:
        raise ValueError('actual native validation-loss minimum differs')
    return dict(file=path.name,checkpoint_sha256=_sha(path),epoch=n-1,score=score,
        model_sha256=storage.state_digest(best['state_dict']),validation_prediction_sha256=records[n-1]['validation']['prediction_sha256'])


def fit_neuropose(data,*,upstream,config,partitions,output,scope='full',preconditions=None,cohort_receipt=None,finite_probe=None,qualification=None):
    policy=neuropose_policy(scope=scope)
    if scope=='full' and not isinstance(qualification,(str,os.PathLike)):raise ValueError('actual new raw CUDA qualification artifact required')
    from . import classic_emg_train as storage,classic_emg_formal as formal
    from .classic_emg_cohort import _safe
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('native fit output exists; no overwrite or implicit restart')
    storage._space(output.parent)
    prepared=_prepare(data,upstream=upstream,config=config,partitions=partitions,scope=scope,preconditions=preconditions,
        cohort_receipt=cohort_receipt,finite_probe=finite_probe,qualification=qualification)
    prepared.check();output.mkdir(mode=0o700);_write(output/'identity.json',prepared.identity)
    with (output/'epochs.jsonl').open('xb'):pass
    initial=storage.state_digest(prepared.module.state_dict());loaders,generators=_loaders(prepared)
    witness=_observer_type()(prepared,output,generators);callbacks=formal.build_callbacks(prepared.cfg,output/'checkpoints',witness)
    kwargs=policy['trainer'].copy();plugins=[]
    if prepared.precision is not None:kwargs.pop('precision');plugins.append(prepared.precision)
    trainer=storage.pl.Trainer(**kwargs,callbacks=callbacks,logger=storage.ClassicCSVLogger(str(output/'logs')),
        plugins=plugins,enable_progress_bar=False,enable_model_summary=False,default_root_dir=str(output))
    started=time.monotonic()
    try:
        trainer.fit(prepared.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'])
    finally:witness.restore_forward(prepared.module)
    prepared.check();torch=storage._runtime();last=torch.load(str(output/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False)
    records=_read_epochs(output);_validate_checkpoint(last,prepared.identity,records)
    if bool(trainer.interrupted):raise ValueError('interrupted native fit is incomplete')
    if scope!='full':
        if len(records)!=2:raise ValueError('native engineering requires complete two epochs')
        reason='engineering-two-epochs'
    else:
        stop=_callback(last,'EarlyStopping')
        reason=formal.termination_reason([r['validation']['val_loss'] for r in records],wait_count=stop['wait_count'],
            stopped_epoch=stop['stopped_epoch'],best_score=float(stop['best_score']),interrupted=False)
    result=_sealed(dict(format=FORMAT,identity_sha256=prepared.identity['sha256'],scope=scope,completed_epochs=len(records),
        optimizer_executions=witness.steps,attempts=witness.attempts,train_samples_per_epoch=[r['train_samples'] for r in records],
        val_samples_per_epoch=[r['validation']['samples'] for r in records],initial_model_sha256=initial,
        last_model_sha256=storage.state_digest(last['state_dict']),last_checkpoint_sha256=_sha(output/'checkpoints/last.ckpt'),
        selected_best=_selection(output,prepared.identity,last,records),journal_sha256=_sha(output/'epochs.jsonl'),
        stop_reason=reason,full_training_terminated=scope=='full',seconds=time.monotonic()-started,
        scientific_eligible=False,full_paper_complete=False))
    _write(output/'fit-result.json',result);return read_neuropose_fit(output,require_full=scope=='full')


def _read_fit(output,*,require_full):
    from . import classic_emg_train as storage
    from .classic_emg_cohort import _safe,_StatGuard
    if type(require_full) is not bool:raise ValueError('explicit full fit requirement must be boolean')
    output=_safe(output);identity=_json(output/'identity.json');result=_json(output/'fit-result.json');_check(identity);_check(result)
    policy=neuropose_policy(scope=identity['policy']['scope'])
    if identity['format']!=FORMAT or identity['policy']!=policy or identity['sources']!=_sources():raise ValueError('actual native fit source/policy differs')
    if require_full and policy['scope']!='full':raise ValueError('engineering fit is not full native training')
    paths=[output/n for n in ('identity.json','fit-result.json','epochs.jsonl','checkpoints/last.ckpt')]
    guard=_StatGuard(paths+[ROOT/n for n in identity['sources']])
    p=identity['locations'];actual=_prepare(p['data'],upstream=p['upstream'],config=p['config'],partitions=p['partitions'],
        scope=policy['scope'],preconditions=identity['preconditions'],cohort_receipt=p['cohort_receipt'],finite_probe=p['finite_probe'],
        qualification=p['qualification'],construct=False)
    # CUDA readback retains recorded runtime metadata, without initializing CUDA.
    reproduced=copy.deepcopy(actual.identity)
    if policy['scope']!='cpu_engineering':
        reproduced['runtime']=identity['runtime'];reproduced['framework']=identity['framework'];reproduced.pop('sha256');reproduced=_sealed(reproduced)
    if reproduced!=identity:raise ValueError('actual original native input/config/qualification identity differs')
    records=_read_epochs(output);epochs=len(records);torch=storage._runtime()
    if not 1<=epochs<=policy['trainer']['max_epochs']:raise ValueError('native actual epoch count outside declared budget')
    for i,row in enumerate(records):
        n=identity['counts']['train']
        if (row['epoch']!=i or row['train_samples']!=n or row['train_batches']!=math.ceil(n/policy['batch_size']) or
            row['validation']['samples']!=identity['counts']['val'] or
            row['attempts']!=(i+1)*math.ceil(n/policy['batch_size']) or
            type(row['optimizer_executions']) is not int or not 0<=row['optimizer_executions']<=row['attempts'] or
            not all(math.isfinite(row[k]) for k in ('train_loss',)) or not math.isfinite(row['validation']['val_loss'])):
            raise ValueError('native actual epoch coverage/counter/finite metrics differ')
    last=torch.load(str(output/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False);_validate_checkpoint(last,identity,records)
    selected=_selection(output,identity,last,records);guard.add(output/'checkpoints'/selected['file'])
    if policy['scope']=='full':
        from .classic_emg_formal import termination_reason
        stop=_callback(last,'EarlyStopping');reason=termination_reason([r['validation']['val_loss'] for r in records],
            wait_count=stop['wait_count'],stopped_epoch=stop['stopped_epoch'],best_score=float(stop['best_score']),interrupted=False)
    else:
        if epochs!=2:raise ValueError('native engineering fit incomplete')
        reason='engineering-two-epochs'
    expected=dict(format=FORMAT,identity_sha256=identity['sha256'],scope=policy['scope'],completed_epochs=epochs,
        optimizer_executions=records[-1]['optimizer_executions'],attempts=records[-1]['attempts'],
        train_samples_per_epoch=[r['train_samples'] for r in records],val_samples_per_epoch=[r['validation']['samples'] for r in records],
        last_model_sha256=storage.state_digest(last['state_dict']),last_checkpoint_sha256=_sha(output/'checkpoints/last.ckpt'),
        selected_best=selected,journal_sha256=_sha(output/'epochs.jsonl'),stop_reason=reason,full_training_terminated=policy['scope']=='full',
        initial_model_sha256=last['native_neuropose']['initial_model_sha256'],scientific_eligible=False,full_paper_complete=False)
    if (any(result.get(k)!=v for k,v in expected.items()) or set(result)!=set(expected)|{'seconds','sha256'} or
        not isinstance(result['seconds'],(int,float)) or not math.isfinite(result['seconds']) or result['seconds']<0 or
        result['optimizer_executions']<=0):raise ValueError('native fit result differs from raw checkpoint/journal/selection')
    actual.guard.check();guard.check();return result,identity,last,guard


def read_neuropose_fit(output,*,require_full=True):
    if type(require_full) is not bool:raise ValueError('explicit full fit requirement must be boolean')
    return _read_fit(output,require_full=require_full)[0]


def reopen_neuropose_best(output,*,require_full=True):
    from . import classic_emg_train as storage
    result,identity,last,guard=_read_fit(output,require_full=require_full);p=identity['locations']
    prepared=_prepare(p['data'],upstream=p['upstream'],config=p['config'],partitions=p['partitions'],scope=identity['policy']['scope'],
        preconditions=identity['preconditions'],cohort_receipt=p['cohort_receipt'],finite_probe=p['finite_probe'],qualification=p['qualification'])
    if prepared.identity!=identity:raise ValueError('new original runtime/input identity differs from selected fit')
    path=Path(output)/'checkpoints'/result['selected_best']['file'];checkpoint=storage._runtime().load(str(path),map_location='cpu',weights_only=False)
    prepared.module.load_state_dict(checkpoint['state_dict'],strict=True)
    if storage.state_digest(prepared.module.state_dict())!=result['selected_best']['model_sha256']:raise ValueError('strict native selected best model differs')
    if prepared.precision is not None:prepared.precision.load_state_dict(checkpoint['MixedPrecision'])
    for q in guard.paths:prepared.guard.add(q)
    prepared.selected_checkpoint=checkpoint;prepared.selection=result['selected_best'];prepared.run=Path(output);prepared.check();return prepared


def _validation(prepared):
    from . import classic_emg_train as storage
    prepared.check();loader=_loaders(prepared)[0]['val'];witness=_observer_type()(prepared)
    kwargs=prepared.policy['trainer'].copy();plugins=[]
    if prepared.precision is not None:kwargs.pop('precision');plugins=[prepared.precision]
    trainer=storage.pl.Trainer(**kwargs,callbacks=[witness],logger=False,plugins=plugins,enable_checkpointing=False,
        enable_progress_bar=False,enable_model_summary=False)
    before=storage.state_digest(prepared.module.state_dict())
    try:trainer.validate(prepared.module,dataloaders=loader,verbose=False)
    finally:witness.restore_forward(prepared.module)
    value=witness.validation
    if (before!=value['model_sha256'] or value['prediction_sha256']!=prepared.selection['validation_prediction_sha256'] or
        not math.isclose(value['val_loss'],prepared.selection['score'],rel_tol=2e-6,abs_tol=1e-7)):
        raise ValueError('strict selected native full validation differs from complete selected epoch')
    prepared.check();return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','upstream','config','partitions','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scope',choices=SCOPES,default='full')
    for name in ('preconditions','cohort-receipt','finite-probe','qualification'):p.add_argument('--'+name,type=Path)
    args=vars(p.parse_args())
    if args['preconditions'] is not None:args['preconditions']=_json(args['preconditions'])
    print(json.dumps(fit_neuropose(**args),sort_keys=True))


if __name__=='__main__':main()
