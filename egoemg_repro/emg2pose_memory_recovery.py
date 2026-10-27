"""Bounded native checkpoint recovery over an admitted immutable source.

This deliberately reuses val as an engineering training fixture. It preserves
the native model/optimizer/callbacks, but never publishes scientific completion.
The owner must supervise each CLI process and its data workers separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from . import emg2pose_check as checks, emg2pose_data as storage
from . import emg2pose_memory_runtime as runtime
from .emg2pose_memory_index import VerifiedSource, open_source
from .emg2pose_memory_process import process_identity


FORMAT = 'emg2pose_native_memory_recovery_v1'
_HASH = re.compile('[0-9a-f]{64}')
_IMPORTED_SHA = storage._sha256(__file__)


def checkpoint_io(output, *, expected_checkpoint_sha256=None):
    """Atomic native IO, restricted to this run; frozen files are immutable."""
    import torch
    from pytorch_lightning.plugins.io import CheckpointIO
    output = storage._safe_path(output)
    native = output/'checkpoints'

    def owned(path, *, automatic=False):
        path = storage._safe_path(path)
        if path.parent == native and path.suffix == '.ckpt':return path
        if not automatic and path in (output/'prefix.ckpt',output/'restored.ckpt'):return path
        raise ValueError('checkpoint operation is outside owned native checkpoint paths')

    class NativeRecoveryIO(CheckpointIO):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            if storage_options is not None:raise ValueError('remote checkpoint storage is unsupported')
            path = owned(path)
            automatic = path.parent == native
            budget = checks._tensor_payload_bytes(checkpoint)*2+32*1024**2
            checks._atomic_write(path,lambda h:torch.save(checkpoint,h),budget,replace=automatic)

        def load_checkpoint(self, path, map_location=None):
            path = owned(path)
            if (path != output/'prefix.ckpt' or not isinstance(expected_checkpoint_sha256,str)
                    or not _HASH.fullmatch(expected_checkpoint_sha256)
                    or storage._sha256(path) != expected_checkpoint_sha256):
                raise ValueError('native load requires the frozen prefix checkpoint SHA256')
            return torch.load(path,map_location='cpu' if map_location is None else map_location,
                              weights_only=False)

        def remove_checkpoint(self, path):
            path = owned(path,automatic=True)
            path.unlink(missing_ok=True)
            checks._sync_directory(native)

    return NativeRecoveryIO()


def _digest(value):
    # Reuse the existing typed tensor/state digest, not pickle storage IDs.
    from .classic_emg_train import state_digest
    return state_digest(value)


def _callbacks(states):
    states = copy.deepcopy(states)
    for key,value in states.items():
        if key.startswith('ModelCheckpoint{'):
            # Pinned PL2.5 saves this field but intentionally never loads it.
            value.pop('current_score',None)
    return states


def _snapshot(module,trainer,device):
    return dict(model=_digest(module.state_dict()),
        optimizer=_digest([v.state_dict() for v in trainer.optimizers]),
        scheduler=_digest([v.scheduler.state_dict() for v in trainer.lr_scheduler_configs]),
        callbacks=_digest(_callbacks({v.state_key:v.state_dict() for v in trainer.callbacks if v.state_dict()})),
        loops=_digest(trainer.fit_loop.state_dict()),rng=_digest(checks._rng_state(device)))


def _checkpoint_snapshot(checkpoint):
    return dict(model=_digest(checkpoint['state_dict']),optimizer=_digest(checkpoint['optimizer_states']),
        scheduler=_digest(checkpoint['lr_schedulers']),callbacks=_digest(_callbacks(checkpoint['callbacks'])),
        loops=_digest(checkpoint['loops']['fit_loop']),rng=_digest(checkpoint['emg2pose_recovery']['rng']))


def _process():
    value = process_identity()
    if value is None:raise RuntimeError('actual process identity is unavailable')
    return {key:value[key] for key in ('pid','start_ticks')}


def _producer():
    if storage._sha256(__file__) != _IMPORTED_SHA:
        raise ValueError('imported recovery implementation changed on disk')
    root = Path(__file__).parent
    return {name:storage._sha256(root/name) for name in ('emg2pose_memory_recovery.py','classic_emg_train.py')}


def _profile(cfg,device):
    import torch
    if device not in ('cpu','cuda:0'):raise ValueError('device must explicitly be cpu or cuda:0')
    gpu = device == 'cuda:0'
    if not gpu and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU recovery requires CUDA_VISIBLE_DEVICES empty before framework import')
    if gpu:
        if not re.fullmatch('GPU-[0-9a-fA-F-]+',os.environ.get('CUDA_VISIBLE_DEVICES','')):
            raise ValueError('GPU recovery requires exactly one explicitly selected GPU UUID')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('selected actual CUDA device is unavailable')
        torch.cuda.set_device(0)
    return dict(device=device,visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device_name=torch.cuda.get_device_name(0) if gpu else 'CPU',
        batch_size=320 if gpu else 1,workers=12 if gpu else 0,
        precision=str(cfg.trainer.precision) if gpu else '32-true',
        scientific_batch_size=320,scientific_workers=12,scheduler_epochs=250,gradient_clip_val=1,
        logger='pytorch_lightning.loggers.TensorBoardLogger',
        fixture='entire original val dataset reused as engineering train; no independent validation',
        uninterrupted_trajectory_qualified=False)


def _witness(identity,profile,source,device,samples):
    import torch
    from pytorch_lightning.callbacks import Callback

    class Witness(Callback):
        def __init__(self):
            self.steps=0;self.validations=[];self.pending=None;self.restore_matches=None
            self.forward=None;self.hook=None;self.phase=None;self.minimum_gpu_free=None

        def check(self):
            source.assert_current()
            if _producer() != identity['producer']:raise ValueError('recovery producer changed during phase')
            if device.type == 'cuda':
                free,_=torch.cuda.mem_get_info(device)
                self.minimum_gpu_free=free if self.minimum_gpu_free is None else min(free,self.minimum_gpu_free)
                if free < 8*1024**3:raise RuntimeError('observed8GiB GPU reserve lost')

        def attach(self,module):
            if self.forward is not None:return
            self.forward=module.forward
            def observed(batch):
                output=self.forward(batch)
                if self.phase == 'val':
                    pred,target,mask=output
                    if (pred.shape != target.shape or tuple(pred.shape[1:]) != (20,7280)
                            or tuple(mask.shape) != (len(pred),7280)
                            or not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(target).all())):
                        raise ValueError('nonfinite or malformed native validation output')
                    self.inputs.update(bytes.fromhex(_digest(batch)))
                    self.outputs.update(bytes.fromhex(_digest(output)))
                    self.sizes.append(len(pred));self.count+=len(pred)
                return output
            module.forward=observed

        def detach(self,module):
            if self.forward is not None:module.forward=self.forward;self.forward=None
            if self.hook is not None:self.hook.remove();self.hook=None

        def on_load_checkpoint(self,trainer,module,checkpoint):
            if checkpoint['emg2pose_recovery']['identity'] != identity:
                raise ValueError('native checkpoint recovery identity differs')
            self.pending=checkpoint

        def on_fit_start(self,trainer,module):
            self.check()
            if self.pending is not None:
                checks._restore_rng(self.pending['emg2pose_recovery']['rng'],device)
            self.attach(module)
            def stepped(*args):self.steps+=1
            self.hook=trainer.optimizers[0].register_step_post_hook(stepped)

        def on_train_start(self,trainer,module):
            if (len(trainer.optimizers) != 1 or type(trainer.optimizers[0]) is not torch.optim.AdamW
                    or len(trainer.lr_scheduler_configs) != 1
                    or trainer.lr_scheduler_configs[0].scheduler.T_max != 250
                    or trainer.gradient_clip_val != 1 or trainer.accumulate_grad_batches != 1
                    or getattr(trainer.precision_plugin,'scaler',None) is not None):
                raise ValueError('actual native AdamW/cosine250/clip1/bf16-no-scaler contract differs')

        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            self.phase='train';self.check()
            if tuple(batch['emg'].shape) != (profile['batch_size'],16,7790):
                raise ValueError('actual engineering training batch differs')

        def on_before_backward(self,trainer,module,loss):
            if loss.ndim or not bool(torch.isfinite(loss)):raise ValueError('nonfinite native training loss')

        def on_after_backward(self,trainer,module):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise ValueError('missing or nonfinite native gradients')

        def on_validation_epoch_start(self,trainer,module):
            self.check();self.phase='val';self.count=0;self.sizes=[]
            self.inputs=hashlib.sha256();self.outputs=hashlib.sha256()

        def on_validation_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):self.check()

        def on_validation_end(self,trainer,module):
            metrics={k:float(v) for k,v in trainer.callback_metrics.items() if k.startswith('val_')}
            if self.count != samples or not metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError('complete finite original validation is required')
            self.validations.append(dict(samples=self.count,batch_sizes=self.sizes,
                inputs_sha256=self.inputs.hexdigest(),outputs_sha256=self.outputs.hexdigest(),metrics=metrics))
            self.phase=None

        def on_save_checkpoint(self,trainer,module,checkpoint):
            self.check()
            checkpoint['emg2pose_recovery']=dict(format=FORMAT,identity=identity,
                runtime_identity=identity['runtime'],rng=checks._rng_state(device),
                validations=copy.deepcopy(self.validations),process=_process())

        def on_fit_end(self,trainer,module):self.detach(module)

    return Witness()


def run_recovery_phase(upstream,source,metadata_csv,size,output,*,phase,device='cpu',expected_prefix_sha256=None):
    """Run one bounded native phase; source lifetime belongs to the caller."""
    if phase not in ('prefix','restore'):raise ValueError('phase must be prefix or restore')
    output=storage._safe_path(output)
    if phase == 'prefix' and output.exists():raise FileExistsError('existing recovery output is never overwritten')
    if type(source) is not VerifiedSource:raise ValueError('an actual admitted source is required')
    source.assert_current()
    prefix=None;checkpoint=None
    if phase == 'restore':
        if (not isinstance(expected_prefix_sha256,str) or not _HASH.fullmatch(expected_prefix_sha256)
                or storage._sha256(output/'prefix.json') != expected_prefix_sha256):
            raise ValueError('prefix receipt differs from separately frozen SHA256')
        prefix=storage._json(output/'prefix.json')
        if (prefix.get('format') != FORMAT or prefix.get('phase') != 'prefix'
                or prefix.get('scientific_training_complete') is not False
                or storage._sha256(output/'prefix.ckpt') != prefix.get('checkpoint_sha256')):
            raise ValueError('frozen prefix checkpoint or receipt differs')
        if (output/'restore-started.json').exists() or (output/'restore.json').exists():
            raise FileExistsError('an attempted recovery is never automatically retried')
    cfg=runtime.compose_config(upstream,size,source.index,source=source,metadata_csv=metadata_csv)
    profile=_profile(cfg,device)
    identity=dict(format=FORMAT,runtime=runtime.runtime_identity(upstream,cfg,source=source),
                  profile=profile,producer=_producer())
    if prefix is not None:
        if prefix.get('runtime_identity') != identity['runtime'] or prefix.get('identity') != identity:
            raise ValueError('prefix runtime, data or recovery profile identity differs')
        if prefix.get('process') == _process():
            raise ValueError('recovery requires a fresh actual process')
    if phase == 'prefix':
        output.mkdir(mode=0o700);(output/'checkpoints').mkdir(mode=0o700)
    checks._write_json(output/(phase+'-started.json'),dict(phase=phase,process=_process(),identity=identity))
    dm=module=trainer=witness=None
    start=time.monotonic()
    try:
        import torch
        from omegaconf import OmegaConf
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        torch_device=torch.device(device)
        if prefix is None:module,dm=runtime.build_runtime(upstream,cfg,source=source)
        else:
            checkpoint=checkpoint_io(output,expected_checkpoint_sha256=prefix['checkpoint_sha256']).load_checkpoint(output/'prefix.ckpt')
            if checkpoint['emg2pose_recovery']['identity'] != identity:
                raise ValueError('frozen checkpoint recovery identity differs')
            module,dm=runtime.reload_runtime(upstream,cfg,checkpoint['state_dict'],identity=identity['runtime'],source=source)
        from egoemg._train_utils import build_callbacks, build_logger, build_trainer
        dm.setup('validate')
        if len(dm.val_dataset) < profile['batch_size']:raise ValueError('complete engineering fixture cannot supply a full native batch')
        dm.train_dataset=dm.val_dataset
        if device == 'cpu':
            dm.batch_size=1;dm.num_workers=0;dm.eval_num_workers=0
        train_loader,val_loader=dm.train_dataloader(),dm.val_dataloader()
        for loader in (train_loader,val_loader):
            if loader.batch_size != profile['batch_size'] or loader.num_workers != profile['workers']:
                raise ValueError('actual native dataloader profile differs')
        witness=_witness(identity,profile,source,torch_device,len(dm.val_dataset))
        callbacks=build_callbacks(cfg,extra_callbacks=[witness],ensure_model_summary=True)
        native=[v for v in callbacks if isinstance(v,ModelCheckpoint)]
        if (len(native) != 1 or native[0].monitor != 'val_mae' or native[0].mode != 'min'
                or native[0].save_top_k != 3 or native[0].save_last is not True
                or any(isinstance(v,EarlyStopping) for v in callbacks)):
            raise ValueError('original EMG2Pose top3+last/noearlystop callback contract differs')
        native[0].dirpath=str(output/'checkpoints')
        operational=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        operational.logger.save_dir=str(output);operational.logger.version='native-recovery'
        t=operational.trainer
        t.max_steps=2 if phase == 'prefix' else 3
        for key,value in dict(limit_train_batches=1,limit_val_batches=1.,num_sanity_val_steps=0,
                              enable_progress_bar=False).items():
            OmegaConf.update(t,key,value,force_add=True)
        # Plugin instances are deliberately outside OmegaConf and the scientific
        # identity; build_trainer still consumes the native trainer configuration.
        logger=build_logger(operational)
        trainer=build_trainer(operational,callbacks,logger)
        trainer.strategy.checkpoint_io=checkpoint_io(output,expected_checkpoint_sha256=None if prefix is None else prefix['checkpoint_sha256'])
        if torch_device.type == 'cuda':torch.cuda.reset_peak_memory_stats(torch_device)
        fresh=None
        if checkpoint is not None:
            witness.attach(module)
            try:trainer.validate(module,dataloaders=val_loader,verbose=False)
            finally:witness.detach(module)
            fresh=witness.validations.pop()
            if fresh != prefix['validations'][-1]:raise ValueError('fresh complete native validation differs from frozen prefix')
        before=checks._parameters_sha(module)
        connector=trainer._checkpoint_connector;original=connector.restore_training_state
        def observed_restore():
            value=original()
            if checkpoint is not None:
                expected=_checkpoint_snapshot(checkpoint);actual=_snapshot(module,trainer,torch_device)
                witness.restore_matches={k:actual[k] == expected[k] for k in expected}
                if not all(witness.restore_matches.values()):
                    raise ValueError('actual native recovery state differs: '+str(witness.restore_matches))
            return value
        connector.restore_training_state=observed_restore
        try:
            trainer.fit(module,train_dataloaders=train_loader,val_dataloaders=val_loader,
                        ckpt_path=None if prefix is None else str(output/'prefix.ckpt'))
        finally:connector.restore_training_state=original
        expected_steps=2 if phase == 'prefix' else 1
        if (witness.steps != expected_steps or trainer.global_step != (2 if prefix is None else 3)
                or len(witness.validations) != expected_steps or checks._parameters_sha(module) == before
                or (prefix is not None and (witness.restore_matches is None or not all(witness.restore_matches.values())))):
            raise ValueError('bounded native recovery did not perform every required actual update/validation')
        frozen=output/('prefix.ckpt' if prefix is None else 'restored.ckpt')
        trainer.save_checkpoint(frozen)
        witness.check()
        if runtime.runtime_identity(upstream,cfg,source=source) != identity['runtime']:
            raise ValueError('runtime changed during recovery phase')
        result=dict(format=FORMAT,phase=phase,identity=identity,runtime_identity=identity['runtime'],profile=profile,
            process=_process(),actual_optimizer_updates=witness.steps,global_step=trainer.global_step,
            validations=witness.validations,fresh_validation=fresh,restore_matches=witness.restore_matches,
            parameters_changed=True,checkpoint_sha256=storage._sha256(frozen),seconds=time.monotonic()-start,
            minimum_observed_gpu_free_bytes=witness.minimum_gpu_free,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(torch_device) if torch_device.type == 'cuda' else None,
            restore_observation='after unchanged native restore_training_state, before iterator reconstruction',
            uninterrupted_trajectory_qualified=False,scientific_training_complete=False,
            full_source_acquired=False,selected_best_or_test_performed=False,full_paper_complete=False)
        checks._write_json(output/(phase+'.json'),result)
        checks._write_json(output/'status.json',dict(phase=phase,status='engineering-phase-complete',scientific_training_complete=False))
        return result
    except BaseException as error:
        (output/(phase+'.json')).unlink(missing_ok=True)
        checks._write_json(output/'status.json',dict(phase=phase,status='failed',error_type=type(error).__name__,error=str(error),scientific_training_complete=False))
        raise
    finally:
        if witness is not None and module is not None:witness.detach(module)
        if trainer is not None and trainer.logger is not None:trainer.logger.finalize('finished' if (output/(phase+'.json')).exists() else 'failed')
        if dm is not None:runtime.close_data_module(dm)


def main(argv=None):
    fd=None
    try:
        fd=int(os.environ['EGOEMG_STAGE_FD_ARCHIVE'])
        parser=argparse.ArgumentParser(description=__doc__)
        for name in ('upstream','index','metadata-csv','output'):parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--expected-index-completion-sha256',required=True)
        parser.add_argument('--size',choices=tuple(runtime.native.SIZES),required=True)
        parser.add_argument('--phase',choices=('prefix','restore'),required=True)
        parser.add_argument('--device',choices=('cpu','cuda:0'),required=True)
        parser.add_argument('--expected-prefix-sha256')
        args=parser.parse_args(argv)
        with open_source(fd,args.index,expected_completion_sha256=args.expected_index_completion_sha256,
                         upstream=args.upstream,metadata_csv=args.metadata_csv) as source:
            run_recovery_phase(args.upstream,source,args.metadata_csv,args.size,args.output,phase=args.phase,
                device=args.device,expected_prefix_sha256=args.expected_prefix_sha256)
    finally:
        if fd is not None:os.close(fd)


if __name__ == '__main__':main()
