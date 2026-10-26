"""Original classic Lightning training with explicit evidence and execution scopes.

This module never tests or exports a scientific completion capsule. Engineering
runs exercise this entry and genuine epoch checkpoints, but do not qualify full
training. All pickle inputs must be trusted local artifacts.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from . import classic_emg_artifacts as artifacts
from . import classic_emg_cohort as cohort
from . import classic_emg_cuda_check as cuda
from .data import file_sha256

storage=cuda.storage
Callback=storage.Callback
FORMAT='classic_formal_runner_v1'
IMPORTED_SOURCE_SHA256=file_sha256(Path(__file__))
CAPACITY_HELPER_SHA256='b890de97c1ebafcb5ddb88dde91785a40a439dce904e919e433f82f8389f7d87'
FULL_COUNTS=dict(train=101714,val=1576,test=4306)
SOURCES=(*artifacts.SOURCE_FILES,'tools/classic_capacity.py','egoemg_repro/classic_emg_formal.py')


def formal_policy(*,variant,engineering=False,engineering_epochs=2,stop_after_epoch=None,device='cuda:0'):
    """Only full canonical training or a separately named bounded acceptance."""
    if variant not in ('vemg2pose','neuropose') or device!='cuda:0' or type(engineering) is not bool:
        raise ValueError('formal policy requires known original variant and actual cuda:0')
    if type(engineering_epochs) is not int or engineering_epochs!=2:
        raise ValueError('engineering acceptance has exactly the declared two-epoch budget')
    if stop_after_epoch is not None and (not engineering or type(stop_after_epoch) is not int or stop_after_epoch!=1):
        raise ValueError('only engineering epoch-one pause is permitted')
    batch=600 if variant=='vemg2pose' else 256
    trainer=dict(accelerator='gpu',devices=1,precision='16-mixed',max_epochs=2 if engineering else 100,
        max_steps=-1,limit_train_batches=1.0,limit_val_batches=1.0,limit_test_batches=1.0,
        check_val_every_n_epoch=1,val_check_interval=1.0,accumulate_grad_batches=1,gradient_clip_val=0,
        log_every_n_steps=50,num_sanity_val_steps=0,deterministic=False)
    return dict(scope='formal_engineering_acceptance' if engineering else 'formal_training',
        scientific_eligible=False,production_ready=False,variant=variant,batch_size=batch,validation_batch_size=batch,
        num_workers=0,world_size=1,train_shuffle=True,trainer=trainer,monitor='val_loss',mode='min',
        early_stopping=dict(patience=50,min_delta=0.,check_finite=True,check_on_train_epoch_end=False),
        stop_after_epoch=stop_after_epoch,runtime_profile=cuda.BACKEND_PROFILE,
        trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False,
        resume_scope=cuda.RESUME_SCOPE)


class Coverage:
    """One bounded bitset and ordered digest; no retained row histories."""
    def __init__(self,samples,*,ordered,batch_size):
        if any(type(v) is not int or v<=0 for v in (samples,batch_size)) or type(ordered) is not bool:
            raise ValueError('positive exact coverage dimensions required')
        self.expected=samples;self.batch_size=batch_size;self.ordered=ordered
        self.seen=bytearray(samples);self.samples=0;self.total=0.;self.has_loss=False
        self.digest=hashlib.sha256();self.batch_sizes=[]
    def add(self,indices,*,loss=None):
        ids=list(indices);size=len(ids)
        if size!=min(self.batch_size,self.expected-self.samples) or not size:
            raise ValueError('original batch grouping or sample count differs')
        if loss is not None and (type(loss) is not float or not math.isfinite(loss)):
            raise ValueError('finite original validation loss required')
        if len(set(ids))!=size:raise ValueError('duplicate sample identity inside batch')
        for offset,index in enumerate(ids):
            if type(index) is not int or not 0<=index<self.expected or self.seen[index]:
                raise ValueError('duplicate or unknown sample identity')
            if self.ordered and index!=self.samples+offset:raise ValueError('validation order differs')
        for index in ids:
            self.seen[index]=1;self.digest.update(index.to_bytes(8,'big'))
        self.samples+=size;self.batch_sizes.append(size)
        if loss is not None:self.total+=size*loss;self.has_loss=True
    def finish(self):
        if self.samples!=self.expected or not all(self.seen):raise ValueError('missing full cohort sample identities')
        return dict(samples=self.samples,batch_sizes=self.batch_sizes.copy(),coverage_sha256=self.digest.hexdigest(),
            batch_weighted_loss=self.total/self.samples if self.has_loss else None)


def termination_reason(losses,*,wait_count,stopped_epoch,best_score,interrupted):
    """Check real finite min/patience history; should_stop is never sufficient."""
    if (interrupted or not losses or len(losses)>100 or
        any(type(x) is not float or not math.isfinite(x) for x in losses) or
        type(wait_count) is not int or type(stopped_epoch) is not int or
        not math.isfinite(float(best_score))):raise ValueError('external, partial, or nonfinite termination')
    best=min(losses);index=losses.index(best);wait=len(losses)-index-1
    running_best=float('inf');running_wait=0
    for epoch,loss in enumerate(losses):
        if loss<running_best:running_best=loss;running_wait=0
        else:running_wait+=1
        if running_wait>=50 and epoch!=len(losses)-1:
            raise ValueError('history continued after original finite patience was exhausted')
    if float(best_score)!=best or wait_count!=wait or wait>50:
        raise ValueError('actual EarlyStopping state differs from full validation history')
    if wait==50 and stopped_epoch==len(losses)-1:return 'early_stopping'
    if len(losses)==100 and stopped_epoch==0:return 'max_epochs'
    raise ValueError('training has not reached finite patience or the full 100 epochs')


def _json(path,limit=8*1024**2):
    path=artifacts._safe(path)
    with _ReadDirectory(path.parent) as directory:
        return json.loads(directory.read(path.name,limit),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


class _ReadDirectory:
    def __init__(self,path):self.value=artifacts._Directory(path,create=False)
    def __enter__(self):return self.value
    def __exit__(self,*_):self.value.close()


def _source_snapshot():
    root=Path(__file__).resolve().parents[1]
    value={name:file_sha256(root/name) for name in SOURCES}
    if value['egoemg_repro/classic_emg_formal.py']!=IMPORTED_SOURCE_SHA256:
        raise ValueError('formal executed source changed after import')
    return value


def _cohort_guard(data,partitions,*,upstream,config,receipt,finite_probe):
    """Keep the actual receipt's supplemental evidence in the strict recheck."""
    supplement=_json(receipt).get('supplement')
    if supplement is not None:
        if not isinstance(finite_probe,(str,os.PathLike)):
            raise ValueError('cohort supplement requires its original finite probe artifact path')
        probe=artifacts._safe(finite_probe)
        if file_sha256(probe)!=supplement['file_sha256']:
            raise ValueError('original supplemental finite probe bytes differ')
    elif finite_probe is not None:
        raise ValueError('finite probe supplied for a receipt without that supplement')
    return cohort.cohort_stat_guard(data,partitions,upstream=upstream,config=config,
        receipt=receipt,finite_probe=finite_probe)


def _qualification_paths(qualification,engineering):
    if not isinstance(qualification,dict) or set(qualification)!={'numeric_prefix','numeric_resume','capacity','final_acceptance'}:
        raise ValueError('qualification requires original numeric/capacity/final acceptance artifact paths')
    if not isinstance(qualification['capacity'],list) or len(qualification['capacity'])!=3:
        raise ValueError('all three original capacity rung artifact paths are required')
    paths=[qualification['numeric_prefix'],qualification['numeric_resume'],*qualification['capacity']]
    if any(not isinstance(x,(str,os.PathLike)) for x in paths):raise ValueError('real qualification artifact paths required')
    if not engineering and not isinstance(qualification['final_acceptance'],(str,os.PathLike)):
        raise ValueError('final formal-source actual GPU acceptance artifact path required')
    if engineering and qualification['final_acceptance'] is not None:
        raise ValueError('engineering cannot inherit or assert production acceptance')
    return {k:([artifacts._safe(x) for x in v] if k=='capacity' else artifacts._safe(v) if v is not None else None)
            for k,v in qualification.items()}


def _capacity(paths,*,variant,numeric,numeric_verification,cohort_path,cpu,guard):
    """Recheck the immutable reviewed helper's raw artifacts, not a pass bool."""
    root=Path(__file__).resolve().parents[1];helper=root/'tools/classic_capacity.py'
    if file_sha256(helper)!=CAPACITY_HELPER_SHA256:raise ValueError('reviewed capacity helper source is missing or differs')
    guard.add(helper)
    spec=importlib.util.spec_from_file_location('_classic_reviewed_capacity_contract',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.IMPORTED_HELPER_SHA!=CAPACITY_HELPER_SHA256 or file_sha256(helper)!=CAPACITY_HELPER_SHA256:
        raise ValueError('executed reviewed capacity helper source differs')
    expected_sources={name:file_sha256(root/name) for name in set(cuda.RUNTIME_FILES)|{'egoemg_repro/classic_emg_cohort.py'}}
    expected_sources['capacity_helper']=CAPACITY_HELPER_SHA256
    previous=None;references=[];processes=set()
    for path,batch in zip(paths,module.LADDERS[variant]):
        for name in ('identity.json','result.json','status.json'):guard.add(path/name)
        identity=_json(path/'identity.json');result=_json(path/'result.json');status=_json(path/'status.json')
        module.validate_result(result,module.policy(variant,batch))
        if (result['identity']!=identity or status.get('state')!='capacity_entry_finished' or
            status.get('result_sha256')!=file_sha256(path/'result.json') or identity['runtime']!=numeric['runtime'] or
            identity['sources']!=expected_sources or identity['cohort_receipt_sha256']!=file_sha256(cohort_path) or
            identity['data_identity_sha256']!=cpu['data_identity']['sha256'] or
            identity['numeric_identity_sha256']!=numeric['sha256'] or identity['recipe']!=numeric['recipe'] or
            identity['numeric_artifacts']['verification']!=numeric_verification or identity['previous']!=previous):
            raise ValueError('capacity original artifact chain/source/runtime/cohort/numeric binding differs')
        process=result['process'];process_key=(process['pid'],process['start_ticks'])
        if process_key in processes:raise ValueError('capacity rungs must be separate actual processes')
        processes.add(process_key)
        previous=dict(batch_size=batch,result_sha256=file_sha256(path/'result.json'),process=process)
        references.append(dict(directory=str(path),**previous))
    guard.check();return references


@dataclass
class PreparedFormal:
    data:Path
    upstream:Path
    config:Path
    partitions:dict
    cohort_receipt:dict
    policy:dict
    identity:dict
    manifest:dict
    guard:object
    factory_prepared:object
    precision:object
    module:object


def prepare_formal_run(data,*,upstream,config,partitions,cohort_receipt,qualification,
                       finite_probe=None,engineering=False,stop_after_epoch=None,device='cuda:0'):
    if device!='cuda:0':raise ValueError('formal training requires actual CUDA cuda:0; no CPU or alternate precision device')
    if type(engineering) is not bool:raise ValueError('explicit engineering scope must be boolean')
    if stop_after_epoch is not None and (not engineering or type(stop_after_epoch) is not int or stop_after_epoch!=1):
        raise ValueError('only explicit engineering epoch-one pause is supported')
    paths=_qualification_paths(qualification,engineering)
    if not isinstance(cohort_receipt,(str,os.PathLike)):raise ValueError('original cohort receipt artifact path required')
    sources=_source_snapshot()
    documents=[Path(cohort_receipt),Path(config),*(Path(__file__).resolve().parents[1]/name for name in SOURCES)]
    if finite_probe is not None:documents.append(Path(finite_probe))
    if isinstance(partitions,(str,os.PathLike)):documents.append(Path(partitions))
    documents.extend(directory/name for directory in (paths['numeric_prefix'],paths['numeric_resume'])
        for name in ('identity.json','result.json','status.json','checkpoints/last.ckpt'))
    documents.extend(directory/name for directory in paths['capacity'] for name in ('identity.json','result.json','status.json'))
    if paths['final_acceptance'] is not None:documents.append(paths['final_acceptance'])
    entry_guard=cohort._StatGuard(documents)
    recipe=cuda.datasets.load_classic_config(config,upstream)
    policy=formal_policy(variant=recipe['variant'],engineering=engineering,stop_after_epoch=stop_after_epoch,device=device)
    numeric=cuda.verify_cuda_check(paths['numeric_prefix'],paths['numeric_resume'])
    numeric_identity=_json(paths['numeric_prefix']/'identity.json')
    if numeric_identity['recipe']!=recipe:raise ValueError('numeric qualification recipe differs')
    guard=_cohort_guard(data,partitions,upstream=upstream,config=config,receipt=cohort_receipt,finite_probe=finite_probe)
    entry_guard.check();guard.paths.update(entry_guard.paths);guard.parents.update(entry_guard.parents)
    cpu=_json(cohort_receipt)
    if {k:v['hand_samples'] for k,v in cpu['footprints'].items()}!=FULL_COUNTS:
        raise ValueError('exact fully admitted classic cohort required')
    partition=_json(partitions,limit=256*1024**2) if isinstance(partitions,(str,os.PathLike)) else copy.deepcopy(partitions)
    for directory in (paths['numeric_prefix'],paths['numeric_resume']):
        for name in ('identity.json','result.json','status.json','checkpoints/last.ckpt'):guard.add(directory/name)
    references=_capacity(paths['capacity'],variant=recipe['variant'],numeric=numeric_identity,
        numeric_verification=numeric,cohort_path=cohort_receipt,cpu=cpu,guard=guard)
    for reference in references:
        identity=_json(Path(reference['directory'])/'identity.json')
        if identity['numeric_artifacts']!={'verification':numeric,
            'prefix_result_sha256':file_sha256(paths['numeric_prefix']/'result.json'),
            'resumed_result_sha256':file_sha256(paths['numeric_resume']/'result.json')}:
            raise ValueError('capacity raw numeric artifacts differ')
    if sources!=_source_snapshot():raise ValueError('formal sources changed during evidence admission')
    for name in SOURCES:guard.add(Path(__file__).resolve().parents[1]/name)
    acceptance=None
    if not engineering:
        acceptance=verify_formal_acceptance(paths['final_acceptance'])
        if (acceptance['sources']!=sources or acceptance['runtime']!=numeric_identity['runtime'] or
            acceptance['cohort_sha256']!=cpu['sha256'] or acceptance['variant']!=recipe['variant'] or
            acceptance['capacity']!=references):raise ValueError('final-source GPU acceptance bindings differ')
        guard.add(paths['final_acceptance'])
    guard.check()
    runtime,precision=cuda._profile()  # No CUDA positive path is simulated in CPU tests.
    if runtime!=numeric_identity['runtime']:raise ValueError('actual formal CUDA runtime differs from qualified runtime')
    prepared=cuda._prepare(data,upstream,config,partition,runtime,expected_recipe=recipe)
    if prepared.identity!=numeric_identity:raise ValueError('original factory/data/source numeric identity differs')
    module=cuda._factory(prepared)
    counts=FULL_COUNTS.copy()
    if engineering:counts['train']=2*policy['batch_size']
    binding=artifacts.json_hash(dict(numeric=numeric,capacity=references,sources=sources,
        acceptance=acceptance,scope=policy['scope']))
    identity=dict(format=artifacts.IDENTITY_FORMAT,scope=artifacts.SCOPE,production_ready=False,scientific_eligible=False,
        recipe=recipe,effective=dict(precision='16-mixed',batch_size=policy['batch_size'],global_batch_size=policy['batch_size'],
        world_size=1,rank=0,num_workers=0,train_shuffle=True,seed=42,log_every_n_steps=50,max_epochs=100,clip_value=0),
        counts=counts,bindings=dict(cohort_receipt_sha256=file_sha256(cohort_receipt),data_sha256=cpu['data_identity']['sha256'],
        partition_sha256=storage._json_hash(partition),normalization_sha256=storage._json_hash(recipe['normalization']),
        numeric_qualification_artifact_sha256=binding),runtime=runtime,
        source_files_sha256={n:sources[n] for n in artifacts.SOURCE_FILES},
        framework_files_sha256={n:file_sha256(Path(storage.pl.__file__).resolve().parent/n) for n in artifacts.FRAMEWORK_FILES},
        model_schema=cuda._tensor_schema(module.state_dict()),parameter_names=[name for name,_ in module.named_parameters()])
    identity=json.loads(json.dumps(identity,allow_nan=False));identity['sha256']=artifacts.json_hash(identity)
    artifacts.validate_formal_identity(identity)
    manifest=dict(format=FORMAT,scope=policy['scope'],scientific_eligible=False,production_ready=False,
        state_identity=identity,policy=policy,sources=sources,capacity=references,cohort_sha256=cpu['sha256'],
        numeric_artifacts={k:str(paths[k]) for k in ('numeric_prefix','numeric_resume')},final_acceptance=acceptance,
        provenance=dict(data=str(Path(data).absolute()),upstream=str(Path(upstream).absolute()),config=str(Path(config).absolute()),
            partitions=str(Path(partitions).absolute()) if isinstance(partitions,(str,os.PathLike)) else None,
            cohort_receipt=str(Path(cohort_receipt).absolute()),
            finite_probe=str(Path(finite_probe).absolute()) if finite_probe is not None else None),
        adaptations=dict(declared_devices=list(prepared.cfg.trainer.devices),effective_devices=[0],world_size=1,
            declared_per_rank_batch=recipe['batch_size'],effective_per_rank_batch=policy['batch_size'],
            effective_global_batch=policy['batch_size'],declared_num_workers=12,effective_num_workers=0,
            training_samples=counts['train'],full_training_samples=FULL_COUNTS['train'],
            original_loss='real batch-union FK and original 22D tuple/2D mask; effective fingertip weight .02',
            checkpoint_timing='native ModelCheckpoint at completed training epoch before PL epoch.completed increment',
            callback_restore='known owned checkpoint paths rebased by strict IO; other native callback values restored exactly',
            recurrent_state='original VEMG outer forward resets SequentialLSTM.hidden before every train/val batch; transient hidden is not checkpoint state',
            engineering_prediction='independent original model copy, eval/no_grad, first1 diagnostic before/after actual full-batch optimizer; all RNG restored',
            limitation='single rank is not original six-rank global batch or equivalent NeuroPose BatchNorm'))
    manifest['sha256']=artifacts.json_hash(manifest);guard.check()
    return PreparedFormal(Path(data),Path(upstream),Path(config),partition,cpu,policy,identity,manifest,guard,prepared,precision,module)


class ValidationCoverage(Callback):
    """Complete validation gate runs before native EarlyStopping/Checkpoint."""
    def __init__(self,*,samples,batch_size):
        self.validation_samples=samples;self.batch_size=batch_size
        self.last_validation=None;self.selected_best=None
    def on_validation_epoch_start(self,trainer,pl_module):
        self.validation_coverage=Coverage(self.validation_samples,ordered=True,batch_size=self.batch_size)
    def on_validation_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
        if trainer.sanity_checking:return
        self.validation_coverage.add(batch['classic_sample_index'].tolist(),loss=float(outputs.detach()))
    def on_validation_end(self,trainer,pl_module):
        if trainer.sanity_checking:return
        observed=self.validation_coverage.finish();loss=float(trainer.callback_metrics['val_loss'])
        if not math.isfinite(loss) or not math.isclose(loss,observed['batch_weighted_loss'],rel_tol=2e-6,abs_tol=1e-7):
            raise ValueError('actual Lightning original val_loss differs from complete batch-weighted loss')
        event=dict(epoch=int(trainer.current_epoch),full=True,samples=observed['samples'],
            expected_samples=self.validation_samples,val_loss=loss,batch_weighted_loss=float(observed['batch_weighted_loss']),
            coverage_sha256=observed['coverage_sha256'])
        self.last_validation=event
        if self.selected_best is None or loss<self.selected_best['score']:
            self.selected_best=dict(epoch=event['epoch'],score=loss,event_sha256=artifacts.json_hash(event))
        self.validation_completed(trainer,pl_module,event)
    def validation_completed(self,trainer,pl_module,event):
        """Extension point for committing an admitted event to the epoch journal."""


def build_callbacks(cfg,directory,observer):
    from hydra.utils import instantiate
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor
    callbacks=[observer];kinds=[]
    for entry in cfg.callbacks:
        target=str(entry['_target_'])
        if target.endswith('.ModelCheckpoint'):
            callback=instantiate(entry,dirpath=str(Path(directory).absolute()),save_on_train_epoch_end=True)
            if callback.monitor!='val_loss' or callback.mode!='min' or callback.save_top_k!=1 or not callback.save_last:
                raise ValueError('original loss/min top-one and last checkpoints required')
        elif target.endswith('.EarlyStopping'):
            callback=instantiate(entry)
            if (callback.monitor!='val_loss' or callback.mode!='min' or callback.patience!=50 or
                callback.min_delta!=0 or not callback.check_finite or callback._check_on_train_epoch_end is not False):
                raise ValueError('original finite loss/min patience50 early stopping required')
        elif target.endswith('.LearningRateMonitor'):
            callback=instantiate(entry)
        else:raise ValueError('unknown original callback')
        kinds.append(type(callback));callbacks.append(callback)
    if sorted(c.__name__ for c in kinds)!=sorted(c.__name__ for c in (ModelCheckpoint,EarlyStopping,LearningRateMonitor)):
        raise ValueError('all three original callbacks required exactly once')
    return callbacks


class _Dataset:
    def __init__(self,dataset,prepared,count):
        self.dataset=dataset;self.prepared=prepared;self.count=count
        if not 0<count<=len(dataset):raise ValueError('run dataset count differs from frozen cohort')
    def __len__(self):return self.count
    def __getitem__(self,index):
        if not 0<=index<self.count:raise IndexError(index)
        item=self.dataset[index]
        item['classic_sample_index']=index
        return item


def admit_formal_batch(batch,contract):
    """Vectorized equivalent of literal sample admission, before optimization.

    Full source guards bracket consumed batches; neither data hashing nor an
    O(source files) stat scan is repeated once per individual hand sample.
    """
    torch=storage._runtime()
    emg=batch.get('emg')
    if not torch.is_tensor(emg) or emg.ndim!=3 or len(emg)==0:raise ValueError('original EMG batch schema differs')
    count=len(emg)
    for name,shape,dtype in (('emg',(count,contract.input_channels,7790),torch.float32),
        ('joint_angles',(count,22,7790),torch.float32),('label_valid_mask',(count,7790),torch.bool)):
        value=batch.get(name)
        if not torch.is_tensor(value) or tuple(value.shape)!=shape or value.dtype!=dtype:
            raise ValueError('original batch shape/dtype differs: '+name)
        if dtype==torch.float32 and not bool(torch.isfinite(value).all()):
            raise ValueError('original batch must be finite; no cleanup: '+name)
    if ('angle_valid_mask' in batch or batch.get('mask_policy')!=['official_literal_loader']*count or
        batch.get('dataset_name')!=['egoemg']*count or len(batch.get('hand',[]))!=count or
        any(hand not in ('left','right') for hand in batch['hand'])):
        raise ValueError('original literal 2D-mask batch protocol differs')
    indices=batch.get('classic_sample_index')
    if not torch.is_tensor(indices) or indices.shape!=(count,) or indices.dtype!=torch.int64:
        raise ValueError('exact frozen hand sample identity batch required')


def _callbacks_state(trainer):
    return {callback.state_key:copy.deepcopy(callback.state_dict()) for callback in trainer.callbacks if callback.state_dict()}


def _canonical_callbacks(states):
    """Compare native-restored fields, with the strict IO's owned paths rebased."""
    result=copy.deepcopy(states)
    for key,state in result.items():
        if key.startswith('ModelCheckpoint{'):
            # PL saves current_score but never restores it. The artifact reader
            # still validates this field in the original saved checkpoint.
            state['current_score']=None
            state['dirpath']='OWNED_CHECKPOINT_DIRECTORY'
            for name in ('best_model_path','kth_best_model_path'):
                state[name]='OWNED_SELECTED_BEST' if state[name] else ''
            state['last_model_path']='OWNED_LAST' if state['last_model_path'] else ''
            state['best_k_models']={'OWNED_SELECTED_BEST':score for score in state['best_k_models'].values()}
    return result


def _snapshot(module,trainer,protocol,*,rng=None):
    return dict(model=storage.state_digest(module.state_dict()),
        optimizer=storage.state_digest([o.state_dict() for o in trainer.optimizers]),
        scheduler=storage.state_digest([c.scheduler.state_dict() for c in trainer.lr_scheduler_configs]),
        scaler=storage.state_digest(trainer.precision_plugin.state_dict()),
        callbacks=storage.state_digest(_canonical_callbacks(_callbacks_state(trainer))),
        rng=storage.state_digest(storage._rng(protocol.generators) if rng is None else rng))


def _checkpoint_snapshot(checkpoint):
    return dict(model=storage.state_digest(checkpoint['state_dict']),
        optimizer=storage.state_digest(checkpoint['optimizer_states']),scheduler=storage.state_digest(checkpoint['lr_schedulers']),
        scaler=storage.state_digest(checkpoint['MixedPrecision']),callbacks=storage.state_digest(_canonical_callbacks(checkpoint['callbacks'])),
        rng=storage.state_digest(checkpoint['formal']['rng']))


class FormalProtocol(ValidationCoverage):
    def __init__(self,prepared,generators,journal,output):
        super().__init__(samples=prepared.identity['counts']['val'],batch_size=prepared.policy['batch_size'])
        self.prepared=prepared;self.generators=generators;self.journal=journal;self.output=output
        self.completed_epochs=0;self.epoch_boundary=False;self.pending_checkpoint=None
        self.resume_start=None;self.witnesses={};self.first_batch=None
        self.witness_model=copy.deepcopy(prepared.module.model) if prepared.policy['scope']=='formal_engineering_acceptance' else None
    def state_dict(self):
        p=self.journal.prefix()
        return dict(completed_epochs=self.completed_epochs,attempts=p['attempts'],executions=p['executions'],
            skips=p['skips'],journal_prefix=p,last_validation=self.last_validation,selected_best=self.selected_best)
    def load_state_dict(self,state):
        if artifacts.json_hash(state['journal_prefix'])!=artifacts.json_hash(self.journal.prefix()):
            raise ValueError('loaded callback journal prefix differs from recovered carry')
        self.completed_epochs=state['completed_epochs'];self.last_validation=state['last_validation'];self.selected_best=state['selected_best']
    def on_load_checkpoint(self,trainer,pl_module,checkpoint):
        self.pending_checkpoint=checkpoint
    def on_fit_start(self,trainer,pl_module):
        if self.pending_checkpoint is not None:
            # SingleDeviceStrategy has restored module/callbacks here; optimizer
            # restoration follows before setup_data creates the native iterator.
            storage._restore_rng(self.pending_checkpoint['formal']['rng'],self.generators)
            if storage.state_digest(storage._rng(self.generators))!=storage.state_digest(self.pending_checkpoint['formal']['rng']):
                raise ValueError('actual typed complete RNG restore differs before loader iterator creation')
    def on_train_start(self,trainer,pl_module):
        optimizer=trainer.optimizers[0]
        if (type(optimizer).__name__!='AdamW' or len(trainer.optimizers)!=1 or
            len(trainer.lr_scheduler_configs)!=1 or type(trainer.lr_scheduler_configs[0].scheduler).__name__!='CosineAnnealingLR'):
            raise ValueError('original automatic AdamW/Cosine optimizer required')
        if trainer.precision_plugin.scaler is None or not trainer.precision_plugin.scaler.is_enabled():
            raise ValueError('genuine enabled CUDA GradScaler required')
        self.counter=artifacts.OptimizerExecutionCounter(optimizer)
        if self.pending_checkpoint is not None:
            checkpoint=self.pending_checkpoint;expected=_checkpoint_snapshot(checkpoint)
            actual_rng=storage._rng(self.generators);saved=checkpoint['formal']['rng']
            # Native setup_data already creates one train iterator before
            # on_train_start. workers0 consumes exactly one loader base seed.
            generator=storage._runtime().Generator();generator.set_state(saved['generators']['train_loader'])
            storage._runtime().empty((),dtype=storage._runtime().int64).random_(generator=generator)
            expected_rng=copy.deepcopy(saved);expected_rng['generators']['train_loader']=generator.get_state()
            if storage.state_digest(actual_rng)!=storage.state_digest(expected_rng):
                raise ValueError('native resumed loader/RNG stream advancement differs before first forward')
            snapshot=_snapshot(pl_module,trainer,self,rng=saved)
            if snapshot!=expected:raise ValueError('actual model/optimizer/scheduler/scaler/callback full restore differs')
            self.resume_start=snapshot;self.pending_checkpoint=None
    def on_train_epoch_start(self,trainer,pl_module):
        self.epoch_boundary=False
        self.train_coverage=Coverage(self.prepared.identity['counts']['train'],ordered=False,batch_size=self.batch_size)
        self.first_batch=None
    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        self.prepared.guard.check();admit_formal_batch(batch,cuda.models.classic_contract(self.prepared.policy['variant']))
        self.train_coverage.add(batch['classic_sample_index'].tolist())
        self.counter.begin();self.scale_before=float(trainer.precision_plugin.scaler.get_scale());self.gradient=None
        if self.prepared.policy['scope']=='formal_engineering_acceptance' and batch_idx==0:
            self.first_batch={k:v.detach().clone() if storage._runtime().is_tensor(v) else copy.deepcopy(v) for k,v in batch.items()}
            self.witnesses[str(trainer.current_epoch)]={'before':self._witness(pl_module,trainer,self.first_batch)}
    def on_before_backward(self,trainer,pl_module,loss):
        if not bool(storage._runtime().isfinite(loss)):raise ValueError('original training loss is nonfinite')
    def on_before_optimizer_step(self,trainer,pl_module,optimizer):
        self.gradient=cuda.gradient_observation([p.grad for p in pl_module.parameters() if p.grad is not None])
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        result=self.counter.end()
        if self.gradient is None:raise ValueError('actual unscaled gradient observation missing')
        loss=float(outputs['loss'].detach());executed=bool(result['executions'])
        self.journal.append_attempt(dict(epoch=int(trainer.current_epoch),batch=batch_idx,pl_step=int(trainer.global_step),
            optimizer_step_before=result['step_before'],optimizer_step_after=result['step_after'],optimizer_executed=executed,
            scale_before=self.scale_before,scale_after=float(trainer.precision_plugin.scaler.get_scale()),
            gradient_finite=self.gradient['finite'],gradient_abs_max=self.gradient['absolute_max'],loss=loss))
        if self.first_batch is not None and batch_idx==0:
            self.witnesses[str(trainer.current_epoch)]['after']=self._witness(pl_module,trainer,self.first_batch)
            self.witnesses[str(trainer.current_epoch)]['execution']=result
            self.first_batch=None
    def on_validation_batch_start(self,trainer,pl_module,batch,batch_idx,dataloader_idx=0):
        self.prepared.guard.check();admit_formal_batch(batch,cuda.models.classic_contract(self.prepared.policy['variant']))
    def validation_completed(self,trainer,pl_module,event):
        self.train_coverage.finish()  # A complete val cannot conceal partial training.
        self.journal.append_epoch(event)
        # Journal stores epoch summaries without the transport kind.
        self.last_validation=self.journal.prefix()['epoch_summaries'][-1]
    def on_train_epoch_end(self,trainer,pl_module):
        self.train_coverage.finish();self.completed_epochs=int(trainer.current_epoch)+1
        if self.journal.prefix()['epochs']!=self.completed_epochs:raise ValueError('completed full validation epoch missing')
        self.epoch_boundary=True
        if self.prepared.policy['stop_after_epoch']==self.completed_epochs:trainer.should_stop=True
    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        checkpoint['formal']=dict(format=artifacts.CHECKPOINT_FORMAT,identity=self.prepared.identity,
            completed_epochs=self.completed_epochs,epoch_boundary=self.epoch_boundary,rng=storage._rng(self.generators),
            journal_prefix=self.journal.prefix(),selected_best=None,role='last')
    def _witness(self,module,trainer,batch):
        torch=storage._runtime();rng=storage._rng(self.generators)
        if self.witness_model is None:raise ValueError('diagnostic copy exists only in explicit engineering scope')
        full_input_sha=storage.state_digest({k:batch[k] for k in ('emg','joint_angles','label_valid_mask','classic_sample_index')})
        diagnostic={k:v[:1] if torch.is_tensor(v) else v for k,v in batch.items()}
        try:
            self.witness_model.to(batch['emg'].device)
            self.witness_model.load_state_dict(module.model.state_dict(),strict=True)
            self.witness_model.eval()
            with torch.no_grad(),trainer.precision_plugin.forward_context():out=self.witness_model.forward(diagnostic)
            if type(out) is not tuple or len(out)!=3:raise ValueError('original model must return actual tuple')
            pred,target,mask=out
            contract=cuda.models.classic_contract(self.prepared.policy['variant'])
            length=7790-contract.left_context-contract.right_context
            span=slice(contract.left_context,7790-contract.right_context)
            if (pred.shape!=target.shape or pred.shape[1:]!=(22,length) or mask.shape!=(len(pred),length) or
                mask.dtype!=torch.bool or target.dtype!=torch.float32 or pred.dtype not in (torch.float16,torch.float32) or
                not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(target).all()) or
                not torch.equal(target,diagnostic['joint_angles'][...,span]) or
                not torch.equal(mask,diagnostic['label_valid_mask'][...,span])):
                raise ValueError('actual original 22D tuple/generated 2D mask/dtype/finite witness differs')
            return dict(input_sha256=full_input_sha,
                target_sha256=storage.state_digest(target),mask_sha256=storage.state_digest(mask),prediction=pred.detach().cpu(),
                model_sha256=storage.state_digest(module.state_dict()),optimizer_sha256=storage.state_digest(trainer.optimizers[0].state_dict()),
                scaler_sha256=storage.state_digest(trainer.precision_plugin.state_dict()))
        finally:
            storage._restore_rng(rng,self.generators)
    def teardown(self,trainer,pl_module,stage):
        counter=getattr(self,'counter',None)
        if counter is not None:counter.close()


class _JournalLogger(storage.CSVLogger):
    """Use Lightning's logger API with bounded scalar journal writes only."""
    def __init__(self,journal,observer):
        # No CSV experiment or rewritten growing table is created.
        self.journal=journal;self.observer=observer
    @property
    def name(self):return 'classic-formal-journal'
    @property
    def version(self):return '1'
    @property
    def save_dir(self):return str(self.journal.directory.path)
    @property
    def log_dir(self):return self.save_dir
    @property
    def experiment(self):return self
    def log_hyperparams(self,params):pass
    def log_metrics(self,metrics,step=None):
        if self.observer.epoch_boundary:return
        for name,value in metrics.items():
            if name=='epoch':continue
            scalar=float(value)
            if not math.isfinite(scalar):raise ValueError('nonfinite original scalar logger metric')
            self.journal.append_metric(dict(pl_step=self.journal.prefix()['attempts'],name=str(name),value=scalar))
    def save(self):pass
    def finalize(self,status):pass


def _termination(checkpoint,*,interrupted=False):
    events=checkpoint['formal']['journal_prefix']['epoch_summaries']
    state=next(v for k,v in checkpoint['callbacks'].items() if k.startswith('EarlyStopping{'))
    return termination_reason([e['val_loss'] for e in events],wait_count=state['wait_count'],
        stopped_epoch=state['stopped_epoch'],best_score=state['best_score'],interrupted=interrupted)


def _read_run(run):
    run=artifacts._safe(run);manifest=_json(run/'identity.json');status=_json(run/'status.json');result=_json(run/'result.json')
    if (manifest.get('format')!=FORMAT or manifest.get('sha256')!=artifacts.json_hash({k:v for k,v in manifest.items() if k!='sha256'}) or
        manifest['sources']!=_source_snapshot() or result.get('manifest_sha256')!=manifest['sha256'] or
        status.get('result_sha256')!=file_sha256(run/'result.json') or result['scientific_eligible'] is not False):
        raise ValueError('formal run source/manifest/status/result binding differs')
    checkpoint=artifacts.read_formal_last(run/'checkpoints/last.ckpt',manifest['state_identity'])
    if (result['checkpoint_sha256']!=file_sha256(run/'checkpoints/last.ckpt') or
        result['checkpoint_seal']!=checkpoint['formal_seal'] or result['completed_epochs']!=checkpoint['formal']['completed_epochs'] or
        result['checkpoint_end_snapshot']!=_checkpoint_snapshot(checkpoint)):
        raise ValueError('run actual last checkpoint/result state binding differs')
    return manifest,status,result,checkpoint


def verify_termination_and_select_best(run):
    """Only a verified full run may expose its carried finite-loss selected core."""
    manifest,status,result,checkpoint=_read_run(run)
    if (manifest['scope']!='formal_training' or manifest['policy']!=formal_policy(variant=manifest['policy']['variant']) or
        manifest['state_identity']['counts']!=FULL_COUNTS or status['state']!='training_finished' or
        result['scope']!='formal_training' or result['interrupted'] is not False):
        raise ValueError('only complete full-cohort formal training can select formal best')
    reason=_termination(checkpoint,interrupted=result['interrupted'])
    if result['stop_reason']!=reason:raise ValueError('termination reason differs from actual original callback state')
    selected=checkpoint['formal']['selected_best'];core=selected['checkpoint']
    return dict(format=FORMAT,scope='terminated_formal_training_selection',scientific_eligible=False,production_ready=False,
        stop_reason=reason,completed_epochs=checkpoint['formal']['completed_epochs'],
        best_epoch=core['epoch'],best_val_loss=core['formal']['journal_prefix']['epoch_summaries'][-1]['val_loss'],
        selected_state_sha256=selected['state_sha256'],selected_model_sha256=storage.state_digest(core['state_dict']),
        selected_event_sha256=checkpoint['callbacks']['FormalProtocol']['selected_best']['event_sha256'],
        last_checkpoint_sha256=file_sha256(Path(run)/'checkpoints/last.ckpt'))


def _terminal_resume(checkpoint):
    try:_termination(checkpoint)
    except ValueError:return False
    return True


def run_formal(data,*,upstream,config,partitions,cohort_receipt,qualification,output,
               finite_probe=None,engineering=False,stop_after_epoch=None,resume=None,device='cuda:0'):
    """Genuine original automatic optimization; full training opens only by evidence."""
    output=artifacts._safe(output)
    if output.exists():raise FileExistsError('formal output exists; never overwrite or implicitly restart')
    if resume is not None and not isinstance(resume,(str,os.PathLike)):raise ValueError('real last checkpoint resume path required')
    if stop_after_epoch is not None and resume is not None:raise ValueError('resumed acceptance must finish its declared epoch budget')
    prepared=prepare_formal_run(data,upstream=upstream,config=config,partitions=partitions,
        cohort_receipt=cohort_receipt,qualification=qualification,finite_probe=finite_probe,engineering=engineering,
        stop_after_epoch=stop_after_epoch,device=device)
    admitted=artifacts.read_formal_last(resume,prepared.identity) if resume else None
    if admitted is not None:
        if _terminal_resume(admitted) or admitted['formal']['completed_epochs']>=prepared.policy['trainer']['max_epochs']:
            raise ValueError('resume checkpoint already reached a genuine terminal or engineering budget')
        storage._strict_state(prepared.module,admitted)
    full=cuda.datasets.build_classic_datasets(prepared.data,prepared.partitions,upstream=prepared.upstream,
        config=prepared.factory_prepared.cfg,normalization=prepared.identity['recipe']['normalization'],
        mask_policy=prepared.identity['recipe']['mask_policy'])
    if {role:len(dataset) for role,dataset in full.items()}!=FULL_COUNTS:
        raise ValueError('actual complete dataset sample counts differ from admitted cohort')
    used={role:_Dataset(full[role],prepared,prepared.identity['counts'][role]) for role in ('train','val')}
    torch=storage._runtime()
    generators={name:torch.Generator().manual_seed(42+i) for i,name in enumerate(('train_loader','train_sampler','val_loader'))}
    train=torch.utils.data.DataLoader(used['train'],batch_size=prepared.policy['batch_size'],num_workers=0,
        generator=generators['train_loader'],sampler=torch.utils.data.RandomSampler(used['train'],generator=generators['train_sampler']))
    val=torch.utils.data.DataLoader(used['val'],batch_size=prepared.policy['batch_size'],num_workers=0,
        generator=generators['val_loader'],shuffle=False)
    prepared.guard.check();owned=None;journal=None;io=None
    try:
        owned=artifacts._Directory(output,create=True)
        with ExitStack() as resources:
            artifacts._write_json(owned,'identity.json',prepared.manifest)
            artifacts._write_json(owned,'status.json',dict(state='running',scope=prepared.policy['scope'],scientific_eligible=False,production_ready=False))
            journal=artifacts.FormalJournal(output/'journal',identity_sha256=prepared.identity['sha256'],
                train_samples=prepared.identity['counts']['train'],batch_size=prepared.policy['batch_size'])
            resources.callback(journal.close)
            observer=FormalProtocol(prepared,generators,journal,output)
            callbacks=build_callbacks(prepared.factory_prepared.cfg,output/'checkpoints',observer)
            io=artifacts.FormalCheckpointIO(output/'checkpoints',prepared.identity,journal=journal,resume=resume,admitted=admitted)
            resources.callback(io.close)
            logger=_JournalLogger(journal,observer)
            kwargs=prepared.policy['trainer'].copy();kwargs.pop('precision')
            trainer=storage.pl.Trainer(**kwargs,callbacks=callbacks,logger=logger,plugins=[io,prepared.precision],
                enable_progress_bar=False,enable_model_summary=False,default_root_dir=str(output))
            trainer.fit(prepared.module,train_dataloaders=train,val_dataloaders=val,ckpt_path=str(artifacts._safe(resume)) if resume else None)
            prepared.guard.check();artifacts.validate_formal_identity(prepared.identity)
            checkpoint=artifacts.read_formal_last(output/'checkpoints/last.ckpt',prepared.identity)
            if checkpoint['formal']['journal_prefix']!=journal.prefix():raise ValueError('last checkpoint and committed journal boundary differ')
            interrupted=bool(getattr(trainer,'interrupted',False))
            if engineering:
                expected=1 if stop_after_epoch==1 else 2
                if interrupted or observer.completed_epochs!=expected:raise ValueError('engineering entry did not reach its exact full epoch boundary')
                reason='engineering-epoch-pause' if stop_after_epoch==1 else 'engineering-epoch-budget'
            else:reason=_termination(checkpoint,interrupted=interrupted)
            witness_sha=None
            if engineering:
                budget=2*storage._payload_bytes(observer.witnesses)+storage.IO_OVERHEAD
                artifacts._atomic(owned,'engineering-witnesses.pt',lambda stream:torch.save(observer.witnesses,stream),budget,replace=False)
                witness_sha=file_sha256(output/'engineering-witnesses.pt')
            prefix=checkpoint['formal']['journal_prefix']
            result=dict(format=FORMAT,scope=prepared.policy['scope'],scientific_eligible=False,production_ready=False,
                manifest_sha256=prepared.manifest['sha256'],completed_epochs=observer.completed_epochs,
                attempts=prefix['attempts'],optimizer_executions=prefix['executions'],natural_overflows=prefix['skips'],
                stop_reason=reason,interrupted=interrupted,selection_scope='engineering' if engineering else 'terminated-full-validation-loss',
                checkpoint_sha256=file_sha256(output/'checkpoints/last.ckpt'),checkpoint_seal=checkpoint['formal_seal'],
                checkpoint_end_snapshot=_checkpoint_snapshot(checkpoint),resume_start=observer.resume_start,
                engineering_witnesses_sha256=witness_sha,process=cuda._process_identity(prepared.policy['scope']),
                runtime_profile=cuda.BACKEND_PROFILE,trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False,
                logging='bounded scalar journal; native post-checkpoint train-epoch emissions excluded from committed journal',
                limitations=['training stage only; no frozen test or paper completion',
                    'single CUDA rank/global batch600 or256; workers0 adaptation',cuda.RESUME_SCOPE])
            artifacts._write_json(owned,'result.json',result)
            artifacts._write_json(owned,'status.json',dict(state='engineering_finished' if engineering else 'training_finished',
                scope=prepared.policy['scope'],scientific_eligible=False,production_ready=False,
                result_sha256=file_sha256(output/'result.json')),replace=True)
        prepared.guard.check();owned.check();_read_run(output);prepared.guard.check();owned.check()
        return result
    except BaseException as exc:
        if owned is not None:
            cuda._fail_owned_output(output,exc,owned=owned.owner)
        raise
    finally:
        if owned is not None:owned.close()


def _witness_file(run,result):
    path=artifacts._safe(Path(run)/'engineering-witnesses.pt')
    before=storage._stamp(path)
    if before[2]>256*1024**2 or file_sha256(path)!=result['engineering_witnesses_sha256']:
        raise ValueError('bounded original engineering prediction witnesses missing or changed')
    value=storage._runtime().load(path,map_location='cpu',weights_only=False)
    if storage._stamp(path)!=before:raise ValueError('engineering witness file changed during verification')
    return value


def _compare_witness(left,right):
    torch=storage._runtime()
    if set(left)!=set(right) or set(left)!={'before','after','execution'} or left['execution']!=right['execution']:
        raise ValueError('original next optimizer execution evidence differs')
    if left['execution']['executions']!=1:raise ValueError('acceptance next attempt must genuinely execute optimizer')
    measurements={}
    for phase in ('before','after'):
        a=left[phase];b=right[phase]
        expected={'input_sha256','target_sha256','mask_sha256','prediction','model_sha256','optimizer_sha256','scaler_sha256'}
        if set(a)!=expected or set(b)!=expected or any(a[k]!=b[k] for k in expected-{'prediction'}):
            raise ValueError('original next input/model/optimizer/scaler/target/mask state differs')
        x=a['prediction'];y=b['prediction']
        if (not torch.is_tensor(x) or not torch.is_tensor(y) or x.dtype!=y.dtype or x.shape!=y.shape or
            not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all())):
            raise ValueError('original next prediction dtype/shape/finite evidence differs')
        maximum=float((x.float()-y.float()).abs().max())
        measurements[phase]=dict(prediction_max_abs_difference=maximum,measured_equal=bool(torch.equal(x,y)))
        if not torch.equal(x,y):
            raise ValueError('canonical backend next prediction differs; measured max_abs='+str(maximum)+
                '; no numerical tolerance has been calibrated or authorized by this profile')
    return measurements


def _acceptance(uninterrupted,prefix,resumed):
    directories=[artifacts._safe(x) for x in (uninterrupted,prefix,resumed)]
    guard=cohort._StatGuard([directory/name for directory in directories for name in
        ('identity.json','status.json','result.json','checkpoints/last.ckpt','engineering-witnesses.pt')])
    runs=[_read_run(x) for x in directories]
    identities=[run[0] for run in runs];results=[run[2] for run in runs];checkpoints=[run[3] for run in runs]
    a,p,b=identities;ra,rp,rb=results;ca,cp,cb=checkpoints
    variant=a['policy']['variant']
    if any(m['provenance']!=a['provenance'] or m['numeric_artifacts']!=a['numeric_artifacts'] for m in identities):
        raise ValueError('actual engineering source/data/numeric artifact paths differ')
    provenance=a['provenance']
    if not provenance['partitions']:
        raise ValueError('final-source acceptance requires original partition file provenance')
    numeric_paths={k:artifacts._safe(v) for k,v in a['numeric_artifacts'].items()}
    for directory in numeric_paths.values():
        for name in ('identity.json','status.json','result.json','checkpoints/last.ckpt'):guard.add(directory/name)
    guard.add(Path(provenance['cohort_receipt']))
    numeric_verified=cuda.verify_cuda_check(numeric_paths['numeric_prefix'],numeric_paths['numeric_resume'])
    numeric_identity=_json(numeric_paths['numeric_prefix']/'identity.json')
    if provenance['finite_probe'] is not None:guard.add(Path(provenance['finite_probe']))
    full_guard=_cohort_guard(provenance['data'],provenance['partitions'],upstream=provenance['upstream'],
        config=provenance['config'],receipt=provenance['cohort_receipt'],finite_probe=provenance['finite_probe'])
    guard.check();full_guard.paths.update(guard.paths);full_guard.parents.update(guard.parents);guard=full_guard
    cpu=_json(provenance['cohort_receipt'])
    actual_capacity=_capacity([Path(x['directory']) for x in a['capacity']],variant=variant,numeric=numeric_identity,
        numeric_verification=numeric_verified,cohort_path=provenance['cohort_receipt'],cpu=cpu,guard=guard)
    model_config=cuda.models.compose_classic(Path(provenance['upstream']),variant)
    actual_model=cuda.models.resolved_classic_identity(model_config,cuda.models.classic_contract(variant))
    expected_binding=artifacts.json_hash(dict(numeric=numeric_verified,capacity=actual_capacity,sources=a['sources'],
        acceptance=None,scope='formal_engineering_acceptance'))
    if (actual_capacity!=a['capacity'] or a['cohort_sha256']!=cpu['sha256'] or
        numeric_identity['model']!=actual_model or a['state_identity']['runtime']!=numeric_identity['runtime'] or
        a['state_identity']['recipe']!=numeric_identity['recipe'] or
        a['state_identity']['bindings']!=dict(cohort_receipt_sha256=file_sha256(provenance['cohort_receipt']),
            data_sha256=cpu['data_identity']['sha256'],partition_sha256=storage._json_hash(_json(provenance['partitions'],limit=256*1024**2)),
            normalization_sha256=storage._json_hash(numeric_identity['recipe']['normalization']),
            numeric_qualification_artifact_sha256=expected_binding)):
        raise ValueError('raw full-cohort/numeric/capacity/original-model/source bindings differ')
    for manifest,status,result,checkpoint in runs:
        policy=formal_policy(variant=variant,engineering=True,stop_after_epoch=manifest['policy']['stop_after_epoch'])
        if (manifest['scope']!='formal_engineering_acceptance' or manifest['policy']!=policy or status['state']!='engineering_finished' or
            result['scope']!='formal_engineering_acceptance' or result['interrupted'] is not False or
            manifest['state_identity']['counts']!=dict(train=2*policy['batch_size'],val=1576,test=4306)):
            raise ValueError('only actual declared bounded engineering entries can qualify final source')
    if (a['state_identity']!=p['state_identity'] or p['state_identity']!=b['state_identity'] or
        a['capacity']!=p['capacity'] or a['capacity']!=b['capacity'] or
        [r['completed_epochs'] for r in results]!=[2,1,2] or
        [r['stop_reason'] for r in results]!=['engineering-epoch-budget','engineering-epoch-pause','engineering-epoch-budget'] or
        rb['resume_start']!=rp['checkpoint_end_snapshot'] or ra['resume_start'] is not None or rp['resume_start'] is not None):
        raise ValueError('final-source uninterrupted/prefix/fresh-resume identity or strict restore differs')
    processes=[(r['process']['pid'],r['process']['start_ticks']) for r in results]
    if len(set(processes))!=3:raise ValueError('final-source acceptance requires three separate actual processes')
    if storage.state_digest(ca['state_dict'])!=storage.state_digest(cb['state_dict']) or any(
        storage.state_digest(ca[key])!=storage.state_digest(cb[key]) for key in ('optimizer_states','lr_schedulers','MixedPrecision')):
        raise ValueError('measured canonical uninterrupted/resumed final parameter/optimizer/scheduler/scaler states differ')
    if storage.state_digest(ca['formal']['rng'])!=storage.state_digest(cb['formal']['rng']):
        raise ValueError('uninterrupted/resumed actual final input/RNG state differs')
    wa,wp,wb=[_witness_file(d,r) for d,r in zip(directories,results)]
    if set(wa)!={'0','1'} or set(wp)!={'0'} or set(wb)!={'1'}:
        raise ValueError('engineering original before/after next-step prediction evidence incomplete')
    first=_compare_witness(wa['0'],wp['0']);next_step=_compare_witness(wa['1'],wb['1'])
    guard.check()
    return dict(format='classic_formal_source_gpu_acceptance_v1',scope='bounded_final_source_cuda_acceptance',
        scientific_eligible=False,production_ready=False,variant=variant,sources=a['sources'],runtime=a['state_identity']['runtime'],
        cohort_sha256=a['cohort_sha256'],capacity=a['capacity'],strict_state_restore=True,
        measured_first_step=first,measured_next_step=next_step,measured_final_states_equal=True,
        trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False,
        comparison_profile='exact observed comparisons only; no trajectory guarantee or calibrated tolerance',
        artifacts=[dict(directory=str(d),result_sha256=file_sha256(d/'result.json'),
            checkpoint_sha256=file_sha256(d/'checkpoints/last.ckpt'),witness_sha256=r['engineering_witnesses_sha256'])
            for d,r in zip(directories,results)])


def verify_formal_acceptance(bundle):
    """Recompute the original three-process CUDA evidence behind a saved bundle."""
    guard=cohort._StatGuard([bundle])
    expected=_json(bundle)
    if (expected.get('format')!='classic_formal_source_gpu_acceptance_v1' or
        expected.get('scope')!='bounded_final_source_cuda_acceptance' or len(expected.get('artifacts',[]))!=3):
        raise ValueError('not a final formal-source actual GPU acceptance artifact')
    actual=_acceptance(*(x['directory'] for x in expected['artifacts']))
    if expected!=actual:raise ValueError('final-source acceptance differs from raw verified artifacts')
    guard.check()
    return actual


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run')
    for name in ('data','upstream','config','partitions','cohort-receipt','qualification','output'):
        run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--engineering',action='store_true');run.add_argument('--stop-after-epoch',type=int)
    run.add_argument('--finite-probe',type=Path)
    run.add_argument('--resume',type=Path);run.add_argument('--device',default='cuda:0')
    verify=sub.add_parser('verify-termination');verify.add_argument('--run',type=Path,required=True)
    accept=sub.add_parser('verify-acceptance');accept.add_argument('--bundle',type=Path,required=True)
    build=sub.add_parser('build-acceptance')
    for name in ('uninterrupted','prefix','resumed','output'):build.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='run':
        values=vars(args).copy();values.pop('command');values['qualification']=_json(values['qualification'])
        value=run_formal(**values)
    elif args.command=='verify-termination':value=verify_termination_and_select_best(args.run)
    elif args.command=='verify-acceptance':value=verify_formal_acceptance(args.bundle)
    else:
        value=_acceptance(args.uninterrupted,args.prefix,args.resumed)
        path=artifacts._safe(args.output)
        with _ReadDirectory(path.parent) as directory:artifacts._write_json(directory,path.name,value)
        verify_formal_acceptance(path)
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
