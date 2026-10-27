"""Original EMG2Pose fit, complete epoch journal and validation-selected best.

Full mode requires the entire admitted source and this driver's actual CUDA
engineering qualification. Engineering has independent native train/val roles;
the val-only mini is never silently used as a training population here.
Checkpoints are trusted local artifacts. Final test/reporting is separate.
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
import re
import time
import uuid

from . import classic_emg_train as tensors
from . import emg2pose_check as checks, emg2pose_data as storage
from . import emg2pose_memory_runtime as runtime
from . import emg2pose_memory_recovery as recovery
from .emg2pose_memory_index import VerifiedSource, open_source


FORMAT='emg2pose_native_memory_fit_v1'
_IMPORTED_SHA=storage._sha256(__file__)
_HASH=re.compile('[0-9a-f]{64}')
_LOOPS=('pytorch_lightning.loops.fit_loop','pytorch_lightning.loops.training_epoch_loop',
        'pytorch_lightning.trainer.connectors.checkpoint_connector',
        'pytorch_lightning.callbacks.model_checkpoint','pytorch_lightning.plugins.precision.amp',
        'torch.utils.data.dataloader','torch.utils.data.sampler','torch.optim.adamw','torch.optim.lr_scheduler')
Callback=tensors.Callback


def fit_policy(scope='full'):
    if type(scope) is not str or scope not in ('full','cpu_engineering','cuda_engineering'):
        raise ValueError('explicit full, cpu_engineering or cuda_engineering scope required')
    cpu=scope=='cpu_engineering'
    return dict(scope=scope,max_epochs=250 if scope=='full' else 2,batch_size=1 if cpu else 320,
        workers=0 if cpu else 12,precision='32-true' if cpu else 'bf16-mixed',scheduler_epochs=250,
        gradient_clip_val=1,save_top_k=3,save_last=True,monitor='val_mae',mode='min',early_stopping=False,
        limit_train_batches=1.,limit_val_batches=1.,num_sanity_val_steps=2 if scope=='full' else 0,
        scientific_complete=False,test_performed=False,full_paper_complete=False)


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _seal(value):
    value=json.loads(json.dumps(value,sort_keys=True,allow_nan=False));value['sha256']=_hash(value);return value


def _read(path):return storage._json(storage._safe_path(path,file=True))


def _check(value):
    if (not isinstance(value,dict) or value.get('format')!=FORMAT
            or value.get('sha256')!=_hash({k:v for k,v in value.items() if k!='sha256'})):
        raise ValueError('native fit document format or seal differs')


def _write(path,value,*,replace=False):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    checks._atomic_write(path,lambda h:h.write(raw),len(raw)+4096,replace=replace)


def _sources():
    if storage._sha256(__file__)!=_IMPORTED_SHA:raise ValueError('imported fit implementation changed')
    root=Path(__file__).parent
    return {name:storage._sha256(root/name) for name in
        ('emg2pose_memory_train.py','emg2pose_memory_recovery.py','classic_emg_train.py','emg2pose_check.py')}


def _loop_paths():return {n:Path(importlib.import_module(n).__file__).resolve() for n in _LOOPS}
def _loops():return {n:storage._sha256(p) for n,p in _loop_paths().items()}


def _stamp(path):
    s=storage._safe_path(path,file=True).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _journal(records,identity_sha):
    previous=identity_sha;lines=[]
    for sequence,record in enumerate(records):
        body=dict(sequence=sequence,previous_sha256=previous,record=record)
        body['sha256']=_hash(body);previous=body['sha256']
        lines.append(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    raw=''.join(lines).encode()
    if len(raw)>2*1024**2:raise ValueError('native epoch journal exceeds bounded format')
    return raw


def _read_journal(path,identity_sha):
    raw=storage._safe_path(path,file=True).read_bytes()
    if len(raw)>2*1024**2:raise ValueError('native epoch journal exceeds bounded format')
    try:records=[json.loads(line)['record'] for line in raw.splitlines()]
    except (ValueError,KeyError) as e:raise ValueError('invalid native epoch journal') from e
    if raw!=_journal(records,identity_sha):raise ValueError('native epoch journal chain differs')
    return records


def checkpoint_io(output,*,expected_last_sha256=None):
    import torch
    from pytorch_lightning.plugins.io import CheckpointIO
    output=storage._safe_path(output);directory=output/'checkpoints'
    def owned(path):
        path=storage._safe_path(path)
        if path.parent!=directory or path.suffix!='.ckpt':raise ValueError('unowned native checkpoint path')
        return path
    class NativeFitIO(CheckpointIO):
        def save_checkpoint(self,checkpoint,path,storage_options=None):
            if storage_options is not None:raise ValueError('remote checkpoint storage unsupported')
            path=owned(path)
            checks._atomic_write(path,lambda h:torch.save(checkpoint,h),
                checks._tensor_payload_bytes(checkpoint)*2+32*1024**2)
        def load_checkpoint(self,path,map_location=None):
            path=owned(path)
            if (path!=directory/'last.ckpt' or not isinstance(expected_last_sha256,str)
                    or not _HASH.fullmatch(expected_last_sha256) or storage._sha256(path)!=expected_last_sha256):
                raise ValueError('native resume requires separately frozen last checkpoint SHA256')
            return torch.load(path,map_location='cpu' if map_location is None else map_location,weights_only=False)
        def remove_checkpoint(self,path):
            owned(path).unlink(missing_ok=True);checks._sync_directory(directory)
    return NativeFitIO()


def _native_callback(checkpoint):
    matches=[v for k,v in checkpoint['callbacks'].items() if k.startswith('ModelCheckpoint{')]
    if len(matches)!=1 or any(k.startswith('EarlyStopping') for k in checkpoint['callbacks']):
        raise ValueError('native top3+last/noearlystop callbacks required')
    return matches[0]


def _checkpoint_snapshot(checkpoint):
    d=tensors.state_digest
    return dict(model=d(checkpoint['state_dict']),optimizer=d(checkpoint['optimizer_states']),
        scheduler=d(checkpoint['lr_schedulers']),callbacks=d(recovery._callbacks(checkpoint['callbacks'])),
        loops=d(checkpoint['loops']['fit_loop']),rng=d(checkpoint['emg2pose_fit']['rng']))


def _adam(optimizer,steps):
    import torch
    if not optimizer.state:raise ValueError('native AdamW has no actual moment state')
    for value in optimizer.state.values():
        step=value.get('step')
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or float(step)!=steps:
            raise ValueError('native AdamW step counter differs from actual updates')


class FitWitness(Callback):
    def __init__(self,identity,source,output,device,stop_after,records):
        self.identity=identity;self.source=source;self.output=output;self.device=device;self.stop_after=stop_after
        self.records=copy.deepcopy(records);self.steps=0 if not records else records[-1]['optimizer_executions']
        self.pending=None;self.phase=None;self.validation=None;self.forward=None;self.hook=None
        self.resume_matches=None;self.resume_observation=None;self.minimum_gpu_free=None
        self.loop_stamps={p:_stamp(p) for p in _loop_paths().values()}

    def state_dict(self):return dict(steps=self.steps,records=copy.deepcopy(self.records))
    def load_state_dict(self,value):
        if set(value)!={'steps','records'}:raise ValueError('native fit witness state schema differs')
        self.steps=value['steps'];self.records=copy.deepcopy(value['records'])

    def check(self):
        import torch
        self.source.assert_current();storage._space(self.output)
        if _sources()!=self.identity['fit_sources'] or any(_stamp(p)!=s for p,s in self.loop_stamps.items()):
            raise ValueError('native fit source or loop implementation changed')
        if self.device.type=='cuda':
            free,_=torch.cuda.mem_get_info(self.device)
            self.minimum_gpu_free=free if self.minimum_gpu_free is None else min(free,self.minimum_gpu_free)
            if free<8*1024**3:raise RuntimeError('observed8GiB GPU margin lost')

    def on_load_checkpoint(self,trainer,module,checkpoint):
        packet=checkpoint['emg2pose_fit']
        if packet['identity']!=self.identity or packet['epoch_boundary'] is not True:
            raise ValueError('only exact owned complete native epoch checkpoints may resume')
        self.pending=checkpoint

    def attach(self,module,trainer):
        import torch
        self.forward=module.forward
        def observed(batch):
            out=self.forward(batch)
            if self.phase=='val' and not trainer.sanity_checking:
                pred,target,mask=out
                if pred.shape!=target.shape or tuple(pred.shape[1:])!=(20,7280) or tuple(mask.shape)!=(len(pred),7280):
                    raise ValueError('malformed original native validation output')
                if not bool(torch.isfinite(pred).all()):raise ValueError('nonfinite original native prediction')
                self.inputs.update(bytes.fromhex(tensors.state_digest(batch)))
                self.outputs.update(bytes.fromhex(tensors.state_digest(out)))
                self.val_n+=len(pred);self.val_batches+=1
            return out
        module.forward=observed

    def detach(self,module):
        if self.forward is not None:module.forward=self.forward;self.forward=None
        if self.hook is not None:self.hook.remove();self.hook=None

    def on_fit_start(self,trainer,module):
        if self.pending is not None:checks._restore_rng(self.pending['emg2pose_fit']['rng'],self.device)
        self.attach(module,trainer)
        def stepped(*args):self.steps+=1
        self.hook=trainer.optimizers[0].register_step_post_hook(stepped)

    def on_train_start(self,trainer,module):
        import torch
        if (len(trainer.optimizers)!=1 or type(trainer.optimizers[0]) is not torch.optim.AdamW
                or trainer.lr_scheduler_configs[0].scheduler.T_max!=250 or trainer.gradient_clip_val!=1
                or trainer.accumulate_grad_batches!=1 or getattr(trainer.precision_plugin,'scaler',None) is not None):
            raise ValueError('original native AdamW/cosine250/clip1/noFP16scaler differs')

    def on_train_epoch_start(self,trainer,module):
        self.check()
        if trainer.current_epoch!=len(self.records):raise ValueError('native restored complete epoch counter differs')
        self.train_n=0;self.train_batches=0;self.train_sum=0.;self.validation=None

    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        self.check();self.phase='train'
        shape=tuple(batch['emg'].shape)
        if len(shape)!=3 or shape[1:]!=(16,7790):raise ValueError('native original training input differs')
        self.batch_n=shape[0]

    def on_before_optimizer_step(self,trainer,module,optimizer):
        import torch
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(v).all()) for v in grads):raise ValueError('nonfinite native gradients')

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        loss=float(outputs['loss'])
        if not math.isfinite(loss) or self.steps!=trainer.global_step:raise ValueError('native loss/update observation differs')
        self.train_n+=self.batch_n;self.train_batches+=1;self.train_sum+=loss*self.batch_n

    def on_validation_epoch_start(self,trainer,module):
        self.phase='val';self.inputs=hashlib.sha256();self.outputs=hashlib.sha256();self.val_n=0;self.val_batches=0

    def on_validation_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):
        if not trainer.sanity_checking:self.check()

    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:self.phase=None;return
        values={k:float(v) for k,v in trainer.callback_metrics.items() if k.startswith('val_')}
        if self.val_n!=self.identity['val_samples'] or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('complete finite original validation is missing')
        self.validation=dict(samples=self.val_n,batches=self.val_batches,val_mae=values['val_mae'],
            inputs_sha256=self.inputs.hexdigest(),outputs_sha256=self.outputs.hexdigest(),native_metrics_sha256=_hash(values))
        self.phase=None

    def on_train_epoch_end(self,trainer,module):
        self.check();n=len(self.records);batches=self.identity['train_batches']
        if (self.train_n!=self.identity['train_samples'] or self.train_batches!=batches or self.validation is None
                or self.steps!=(n+1)*batches or trainer.global_step!=self.steps):
            raise ValueError('partial native training/validation epoch cannot be committed: '+str(dict(
                train_n=self.train_n,expected_n=self.identity['train_samples'],train_batches=self.train_batches,
                expected_batches=batches,validation_present=self.validation is not None,
                steps=self.steps,global_step=trainer.global_step,expected_steps=(n+1)*batches)))
        _adam(trainer.optimizers[0],self.steps)
        record=dict(epoch=n,optimizer_executions=self.steps,global_step=trainer.global_step,
            train_samples=self.train_n,train_batches=self.train_batches,train_loss=self.train_sum/self.train_n,
            validation=copy.deepcopy(self.validation),model_sha256=_state_sha(module.state_dict()))
        old=_journal(self.records,self.identity['sha256']);self.records.append(record)
        raw=_journal(self.records,self.identity['sha256']);path=self.output/'epochs.jsonl'
        with path.open('r+b') as h:
            if h.read()!=old:raise ValueError('owned epoch journal prefix changed')
            h.write(raw[len(old):]);h.flush();os.fsync(h.fileno())
        if self.stop_after==len(self.records):trainer.should_stop=True

    def on_save_checkpoint(self,trainer,module,checkpoint):
        self.check()
        checkpoint['emg2pose_fit']=dict(format=FORMAT,identity=self.identity,epoch_boundary=self.validation is not None,
            completed_epochs=len(self.records),records=copy.deepcopy(self.records),rng=checks._rng_state(self.device),
            resume_matches=self.resume_matches,resume_observation=self.resume_observation)

    def on_fit_end(self,trainer,module):self.detach(module)


def _full_source(identity):
    if (identity.get('cohort')!='full' or identity.get('full_corpus_complete') is not True
            or identity.get('engineering_only') is not False or identity.get('file_count')!=25253
            or identity.get('counts',{}).get('split')!={'train':17136,'val':1950,'test':6167}
            or identity.get('archive_bytes')!=462824048640):
        raise ValueError('full fit requires the complete admitted25253 source with independent original roles')


def _load(path):
    import torch
    return torch.load(storage._safe_path(path,file=True),map_location='cpu',weights_only=False)


def read_fit(output,*,require_full=True):
    output=storage._safe_path(output);identity=_read(output/'identity.json');result=_read(output/'fit-result.json')
    _check(identity);_check(result)
    if (result['identity_sha256']!=identity['sha256'] or identity['fit_sources']!=_sources()
            or identity['loop_sources']!=_loops() or identity['policy']!=fit_policy(result['scope'])):
        raise ValueError('native fit identity/source/policy differs')
    if result['scientific_complete'] is not False or result['test_performed'] is not False or result['full_paper_complete'] is not False:
        raise ValueError('fit-only result cannot claim final scientific test completion')
    if require_full and result['scope']!='full':raise ValueError('engineering fit is not full training')
    records=_read_journal(output/'epochs.jsonl',identity['sha256'])
    if not records:raise ValueError('native epoch journal is empty')
    if storage._sha256(output/'checkpoints/last.ckpt')!=result['last_checkpoint_sha256']:
        raise ValueError('native terminal last checkpoint bytes differ')
    checkpoint=_load(output/'checkpoints/last.ckpt');packet=checkpoint['emg2pose_fit']
    if (packet['identity']!=identity or packet['records']!=records or packet['epoch_boundary'] is not True
            or packet['completed_epochs']!=len(records)):
        raise ValueError('native checkpoint complete epoch carry differs from journal')
    policy=identity['policy'];batches=identity['train_batches'];steps=len(records)*batches
    expected_samples=math.ceil(identity['train_unique_windows']/(policy['batch_size']*16))*policy['batch_size']*16
    if identity['train_samples']!=expected_samples or batches!=expected_samples//policy['batch_size']:
        raise ValueError('original native session-block padding differs')
    for i,r in enumerate(records):
        if (r['epoch']!=i or r['train_samples']!=identity['train_samples'] or r['train_batches']!=batches
                or r['optimizer_executions']!=(i+1)*batches or r['global_step']!=r['optimizer_executions']
                or r['validation']['samples']!=identity['val_samples'] or not math.isfinite(r['validation']['val_mae'])
                or not math.isfinite(r['train_loss'])):
            raise ValueError('partial or nonfinite native epoch coverage')
    # State bytes are checked using the same ordered state-dictionary algorithm
    # as the live module, without reconstructing or reseeding another model.
    if (checkpoint['global_step']!=steps or result['optimizer_executions']!=steps or result['completed_epochs']!=len(records)
            or _state_sha(checkpoint['state_dict'])!=records[-1]['model_sha256']
            or result['final_model_sha256']!=records[-1]['model_sha256']):
        raise ValueError('native terminal model/update counters differ')
    native=_native_callback(checkpoint);selected=result['selected_best'];best=storage._safe_path(native['best_model_path'],file=True)
    if best.parent!=output/'checkpoints' or best.name=='last.ckpt' or selected['file']!=best.name or storage._sha256(best)!=selected['checkpoint_sha256']:
        raise ValueError('native selected best checkpoint binding differs')
    score=float(native['best_model_score']);minimum=min(r['validation']['val_mae'] for r in records)
    if score!=minimum or selected['score']!=score or selected['validated'] is not True:raise ValueError('native selected best is not validation minimum')
    best_state=_load(best);selected_packet=best_state['emg2pose_fit'];epoch=selected_packet['completed_epochs']-1
    if (selected_packet['identity']!=identity or not 0<=epoch<len(records) or selected_packet['records']!=records[:epoch+1]
            or records[epoch]['validation']['val_mae']!=score or _state_sha(best_state['state_dict'])!=records[epoch]['model_sha256']
            or best_state['global_step']!=records[epoch]['global_step']):
        raise ValueError('native selected best state/epoch differs from actual validation history')
    for item in checkpoint['lr_schedulers']:
        if item['T_max']!=250:raise ValueError('native cosine scheduler horizon differs')
    complete=len(records)==policy['max_epochs']
    if result['stage_complete'] is not complete or result['full_training_complete'] is not (complete and result['scope']=='full'):
        raise ValueError('premature or engineering termination is not full training')
    if result['scope']=='full':_full_source(identity['runtime']['data'])
    if require_full and not result['full_training_complete']:raise ValueError('full native250 epoch training incomplete')
    if result['train_samples_per_epoch']!=[r['train_samples'] for r in records] or result['val_samples_per_epoch']!=[r['validation']['samples'] for r in records]:
        raise ValueError('native fit coverage summary differs')
    prefix=result['resume_checkpoint']
    if prefix is not None:
        path=storage._safe_path(output/prefix['file'],file=True)
        if path.parent!=output/'history' or storage._sha256(path)!=prefix['sha256']:raise ValueError('frozen resume prefix differs')
        before=_load(path)
        observation=packet['resume_observation']
        if (packet['resume_matches']!=result['resume_matches'] or observation['expected']!=_checkpoint_snapshot(before)
                or observation['actual']!=observation['expected'] or not all(result['resume_matches'].values())):
            raise ValueError('actual six-state native resume observation differs')
    return result


def _state_sha(state):
    return tensors.state_digest(state)


def read_qualification(output,expected_sha256):
    output=storage._safe_path(output)
    if (not isinstance(expected_sha256,str) or not _HASH.fullmatch(expected_sha256)
            or storage._sha256(output/'fit-result.json')!=expected_sha256):raise ValueError('CUDA qualification receipt SHA256 differs')
    result=read_fit(output,require_full=False);identity=_read(output/'identity.json')
    if (result['scope']!='cuda_engineering' or result['stage_complete'] is not True or result['cuda_initialized'] is not True
            or result['resume_matches']!={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')}
            or identity['profile']['batch_size']!=320 or identity['profile']['workers']!=12
            or identity['profile']['precision']!='bf16-mixed'):
        raise ValueError('actual CUDA engineering complete epochs and native six-state restore required')
    return identity


def fit_source(upstream,source,metadata_csv,size,output,*,scope='full',resume=False,expected_last_sha256=None,
               stop_after_epoch=None,qualification=None,expected_qualification_sha256=None):
    import torch
    policy=fit_policy(scope)
    if type(source) is not VerifiedSource:raise ValueError('actual admitted source required')
    source.assert_current();data=source.data_identity
    if scope=='full':_full_source(data)
    if not all(data.get('counts',{}).get('split',{}).get(role,0)>0 for role in ('train','val')):
        raise ValueError('independent original train and val source roles required; val-only mini cannot train')
    if stop_after_epoch is not None and (scope=='full' or type(stop_after_epoch) is not int or not 1<=stop_after_epoch<=2):
        raise ValueError('stop_after_epoch is only an explicit engineering pause')
    cfg=runtime.compose_config(upstream,size,source.index,source=source,metadata_csv=metadata_csv)
    device='cpu' if scope=='cpu_engineering' else 'cuda:0'
    profile=recovery._profile(cfg,device);profile['fixture']='independent original train and val roles'
    # CUDA UUID is private invocation evidence. Equivalent devices must still
    # have the same actual accelerator name/profile and runtime implementation.
    visible=profile.pop('visible_device')
    native_identity=runtime.runtime_identity(upstream,cfg,source=source)
    qualified=None
    if scope=='full':
        if qualification is None:raise ValueError('full fit requires actual frozen CUDA engineering qualification')
        qualified=read_qualification(qualification,expected_qualification_sha256)
        expected={k:v for k,v in native_identity.items() if k!='data'}
        if ({k:v for k,v in qualified['runtime'].items() if k!='data'}!=expected or qualified['profile']!=profile):
            raise ValueError('CUDA qualification scientific recipe/runtime/hardware differs')
    output=storage._safe_path(output)
    if not resume and output.exists():raise FileExistsError('native fit output already exists')
    if resume:
        # Refuse a bad frozen input before changing any existing run artifact.
        checkpoint=checkpoint_io(output,expected_last_sha256=expected_last_sha256).load_checkpoint(output/'checkpoints/last.ckpt')
    else:
        checkpoint=None;output.mkdir(mode=0o700);(output/'checkpoints').mkdir();(output/'history').mkdir()
    lock=os.open(output/'fit.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    dm=module=trainer=witness=None;attempt=uuid.uuid4().hex;begun=False;result_written=False
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        os.close(lock)
        raise
    try:
        identity=None;records=[];prefix=None
        if resume:
            identity=_read(output/'identity.json');_check(identity)
            if (identity['runtime']!=native_identity or identity['policy']!=policy or identity['profile']!=profile
                    or identity['fit_sources']!=_sources() or identity['loop_sources']!=_loops()
                    or checkpoint['emg2pose_fit']['identity']!=identity or checkpoint['emg2pose_fit']['epoch_boundary'] is not True):
                raise ValueError('owned native resume source/profile/complete epoch identity differs')
            records=checkpoint['emg2pose_fit']['records']
            carry=_journal(records,identity['sha256']);path=output/'epochs.jsonl';raw=path.read_bytes()
            if not raw.startswith(carry):raise ValueError('native checkpoint journal prefix differs')
            if len(records)>=policy['max_epochs']:raise ValueError('completed native fit cannot resume beyond original budget')
            begun=True
            if raw!=carry:
                (output/'history'/('uncommitted-'+attempt+'.jsonl')).write_bytes(raw[len(carry):])
                checks._atomic_write(path,lambda h:h.write(carry),len(carry)+4096)
            prefix=dict(file='history/resume-'+attempt+'.ckpt',sha256=expected_last_sha256)
            os.link(output/'checkpoints/last.ckpt',output/prefix['file'])
            if (output/'fit-result.json').exists():os.rename(output/'fit-result.json',output/'history'/('fit-result-'+attempt+'.json'))
        else:begun=True
        _write(output/('attempt-'+attempt+'.json'),dict(scope=scope,resume=resume,visible_device=visible,process=recovery._process()))
        if checkpoint is None:module,dm=runtime.build_runtime(upstream,cfg,source=source)
        else:module,dm=runtime.reload_runtime(upstream,cfg,checkpoint['state_dict'],identity=native_identity,source=source)
        dm.setup('fit')
        if scope=='cpu_engineering':dm.batch_size=1;dm.num_workers=dm.eval_num_workers=0
        train_loader,val_loader=dm.train_dataloader(),dm.val_dataloader()
        unique_train,val_n=len(dm.train_dataset),len(dm.val_dataset)
        if unique_train<=0 or val_n<=0 or dm.train_dataset is dm.val_dataset:raise ValueError('nonempty independent native train/val datasets required')
        from egoemg.datamodule import SessionShuffleSampler
        sampler=train_loader.sampler
        train_n=len(sampler);train_batches=len(train_loader)
        if (type(sampler) is not SessionShuffleSampler or sampler.num_replicas!=1 or sampler.rank!=0
                or sampler.block_batches!=16 or sampler.batch_size!=policy['batch_size']
                or train_n!=math.ceil(unique_train/(policy['batch_size']*16))*policy['batch_size']*16
                or train_batches!=train_n//policy['batch_size']):
            raise ValueError('original native session-shuffle padding/single-rank profile differs')
        for loader in (train_loader,val_loader):
            if loader.batch_size!=policy['batch_size'] or loader.num_workers!=policy['workers']:raise ValueError('actual original native loader profile differs')
        if identity is None:
            identity=_seal(dict(format=FORMAT,owner=uuid.uuid4().hex,runtime=native_identity,policy=policy,profile=profile,
                fit_sources=_sources(),loop_sources=_loops(),train_samples=train_n,train_unique_windows=unique_train,
                train_batches=train_batches,val_samples=val_n,
                qualification_sha256=expected_qualification_sha256 if scope=='full' else None,
                scientific_complete=False,test_performed=False,full_paper_complete=False))
            _write(output/'identity.json',identity);(output/'epochs.jsonl').touch(exist_ok=False)
        elif (identity['train_samples']!=train_n or identity['train_unique_windows']!=unique_train
              or identity['train_batches']!=train_batches or identity['val_samples']!=val_n):
            raise ValueError('native resumed actual role population or sampler padding differs')
        witness=FitWitness(identity,source,output,torch.device(device),stop_after_epoch,records)
        from egoemg._train_utils import build_callbacks,build_logger,build_trainer
        from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
        from omegaconf import OmegaConf
        callbacks=build_callbacks(cfg,extra_callbacks=[witness],ensure_model_summary=True)
        native=[v for v in callbacks if isinstance(v,ModelCheckpoint)]
        if (len(native)!=1 or native[0].monitor!='val_mae' or native[0].mode!='min' or native[0].save_top_k!=3
                or native[0].save_last is not True or any(isinstance(v,EarlyStopping) for v in callbacks)):
            raise ValueError('original native top3+last/noearlystop policy differs')
        native[0].dirpath=str(output/'checkpoints')
        operational=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        operational.logger.save_dir=str(output);operational.logger.version='native-fit'
        for key,value in dict(max_epochs=policy['max_epochs'],limit_train_batches=1.,limit_val_batches=1.,
                              num_sanity_val_steps=policy['num_sanity_val_steps'],enable_progress_bar=False).items():
            OmegaConf.update(operational.trainer,key,value,force_add=True)
        trainer=build_trainer(operational,callbacks,build_logger(operational))
        trainer.strategy.checkpoint_io=checkpoint_io(output,expected_last_sha256=expected_last_sha256)
        connector=trainer._checkpoint_connector;original=connector.restore_training_state
        def observed_restore():
            value=original()
            if checkpoint is not None:
                actual=recovery._snapshot(module,trainer,torch.device(device));expected=_checkpoint_snapshot(checkpoint)
                witness.resume_matches={k:actual[k]==expected[k] for k in expected}
                witness.resume_observation=dict(expected=expected,actual=actual)
                if not all(witness.resume_matches.values()):raise ValueError('actual native six-state restore differs: '+str(witness.resume_matches))
            return value
        connector.restore_training_state=observed_restore;start=time.monotonic()
        try:trainer.fit(module,train_dataloaders=train_loader,val_dataloaders=val_loader,
                        ckpt_path=None if checkpoint is None else str(output/'checkpoints/last.ckpt'))
        finally:connector.restore_training_state=original
        witness.check()
        if runtime.runtime_identity(upstream,cfg,source=source)!=native_identity:raise ValueError('native runtime changed during fit')
        if checkpoint is not None and not witness.resume_matches:raise ValueError('native restoration boundary unobserved')
        n=len(witness.records);complete=n==policy['max_epochs'];native=native[0];best=Path(native.best_model_path)
        if not complete and stop_after_epoch!=n:raise ValueError('unexpected incomplete native fit')
        result=_seal(dict(format=FORMAT,identity_sha256=identity['sha256'],scope=scope,completed_epochs=n,
            optimizer_executions=witness.steps,stage_complete=complete,full_training_complete=complete and scope=='full',
            scientific_complete=False,test_performed=False,full_paper_complete=False,completion_requires_owned_stage=True,
            final_model_sha256=_state_sha(module.state_dict()),last_checkpoint_sha256=storage._sha256(output/'checkpoints/last.ckpt'),
            selected_best=dict(file=best.name,score=float(native.best_model_score),checkpoint_sha256=storage._sha256(best),validated=True),
            train_samples_per_epoch=[v['train_samples'] for v in witness.records],val_samples_per_epoch=[v['validation']['samples'] for v in witness.records],
            resume_matches=witness.resume_matches,resume_checkpoint=prefix,cuda_initialized=torch.cuda.is_initialized(),
            runtime_device=str(module.device),minimum_observed_gpu_free_bytes=witness.minimum_gpu_free,seconds=time.monotonic()-start,
            uninterrupted_trajectory_qualified=False,callback_restore_policy='stock fields except unrestored ModelCheckpoint.current_score'))
        # Finalize potentially failing logger/dataset work before publishing.
        trainer.logger.finalize('success');runtime.close_data_module(dm)
        _write(output/'status.json',dict(status='fit-only-complete' if complete else 'engineering-paused',scientific_complete=False),replace=True)
        _write(output/'fit-result.json',result)
        result_written=True
        read_fit(output,require_full=scope=='full')
        return result
    except BaseException as error:
        if result_written:(output/'fit-result.json').unlink(missing_ok=True)
        if begun:
            _write(output/'status.json',dict(status='failed',error_type=type(error).__name__,error=str(error),scientific_complete=False),replace=True)
        raise
    finally:
        if witness is not None and module is not None:witness.detach(module)
        if dm is not None:runtime.close_data_module(dm)
        fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)


def main(argv=None):
    fd=None
    try:
        fd=int(os.environ['EGOEMG_STAGE_FD_ARCHIVE']);p=argparse.ArgumentParser(description=__doc__)
        for name in ('upstream','index','metadata-csv','output'):p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--expected-index-completion-sha256',required=True)
        p.add_argument('--size',choices=tuple(runtime.native.SIZES),required=True)
        p.add_argument('--scope',choices=('full','cpu_engineering','cuda_engineering'),default='full')
        p.add_argument('--resume',action='store_true');p.add_argument('--expected-last-sha256')
        p.add_argument('--stop-after-epoch',type=int);p.add_argument('--qualification',type=Path)
        p.add_argument('--expected-qualification-sha256');a=p.parse_args(argv)
        with open_source(fd,a.index,expected_completion_sha256=a.expected_index_completion_sha256,
                         upstream=a.upstream,metadata_csv=a.metadata_csv) as source:
            fit_source(a.upstream,source,a.metadata_csv,a.size,a.output,scope=a.scope,resume=a.resume,
                expected_last_sha256=a.expected_last_sha256,stop_after_epoch=a.stop_after_epoch,
                qualification=a.qualification,expected_qualification_sha256=a.expected_qualification_sha256)
    finally:
        if fd is not None:os.close(fd)


if __name__=='__main__':main()
