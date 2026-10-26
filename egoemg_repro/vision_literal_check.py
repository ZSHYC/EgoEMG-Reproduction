"""Bounded original Lightning vision fit, complete validation and fresh resume.

This engineering entry deliberately truncates training to two batches, then one
batch in a new process. It never tests the test role or qualifies full training.
Checkpoint inputs are trusted local artifacts. CPU uses32-true; the explicit
CUDA entry uses original bf16-mixed, with no FP16 GradScaler invented for it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil

from . import classic_emg_train as storage
from .classic_emg import _file_hash,_hash
from .vision_literal import compose_vision
from .vision_literal_data import VisionLiteralDataset,build_vision_partitions,_crop_path
from .vision_literal_runtime import build_vision_runtime,vision_optimizer_identity,vision_state_sha256,_stamp
from .data import LegacyStore


FORMAT='native_vision_bounded_engineering_v1'
ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
Callback=storage.Callback


def _process_identity():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=fields[19])


def _json_create(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def _datasets(data,crops,upstream,config):
    store=LegacyStore(data);partitions=build_vision_partitions(store)
    result={role:VisionLiteralDataset(data,partitions,crops,upstream=upstream,config=config,role=role)
        for role in ('train','val')}
    if not all(len(ds)>0 for ds in result.values()):raise ValueError('nonempty independent train/val roles required')
    return result


def _data_identity(datasets):
    train=datasets['train'];store=train.store;root=store.root
    files={root/'manifest.json',root/'metadata.npz'}
    # The release manifest also declares unacquired raw EMG. Bind the actual
    # point-reader memmaps and partition field; vision never opens those EMG
    # declarations. Use the opened arrays' paths rather than guessing filenames.
    files.add(Path(store.field('frame_split_id').filename))
    files.update(Path(array.filename) for ds in datasets.values() for view in ds.views
        for array in view._frame_memmaps.values())
    for ds in datasets.values():
        for w in ds.windows:
            path=_crop_path(ds.views[0].crops,store.episodes[w['episode']]['id'])
            files.add(path/'data.mdb' if path.is_dir() else path)
    records=[(str(p.resolve()),list(_stamp(p))) for p in sorted(files)]
    return dict(partition_sha256=train.partitions['provenance']['partition_sha256'],
        manifest_sha256=_file_hash(root/'manifest.json'),metadata_sha256=_file_hash(root/'metadata.npz'),
        file_stat_sha256=_hash(records),file_count=len(files),
        consumed={role:len(ds) for role,ds in datasets.items()},
        unique={role:len(ds)//ds.repeat for role,ds in datasets.items()},
        admission='engineering stat/metadata binding only; full payload content admission remains required')


def _check_identity(datasets,device,variant):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported checker source changed')
    result=dict(format=FORMAT,variant=variant,device=device,data=_data_identity(datasets),
        source_sha256={name:_file_hash(ROOT/name) for name in (
            'egoemg_repro/vision_literal_check.py','egoemg_repro/classic_emg_train.py')},
        batch_size=2,workers=0,single_rank=True,precision='32-true' if device=='cpu' else 'bf16-mixed',
        train_batches_by_phase=dict(prefix=2,resume=1),full_validation=True,clip=1,
        scientific_eligible=False,production_ready=False,full_paper_complete=False)
    result['sha256']=_hash(result)
    return result


def _callback_states(trainer):
    return {cb.state_key:cb.state_dict() for cb in trainer.callbacks if cb.state_dict()}


def _native_restored_callbacks(states):
    result=copy.deepcopy(states)
    for key,state in result.items():
        if key.startswith('ModelCheckpoint{'):
            # PL2.5 saves this transient observation but its unchanged
            # load_state_dict deliberately never restores it. All selection,
            # score, path and top-k fields are still compared exactly.
            state.pop('current_score')
    return result


def _snapshots(module,trainer,generators):
    digest=storage.state_digest
    return dict(model=digest(module.state_dict()),optimizer=digest([o.state_dict() for o in trainer.optimizers]),
        scheduler=digest([c.scheduler.state_dict() for c in trainer.lr_scheduler_configs]),
        callbacks=digest(_native_restored_callbacks(_callback_states(trainer))),loops=digest(trainer.fit_loop.state_dict()),
        rng=digest(storage._rng(generators)))


class EngineeringWitness(Callback):
    def __init__(self,prepared,identity,datasets,generators):
        self.prepared=prepared;self.identity=identity;self.datasets=datasets;self.generators=generators
        self.steps=0;self.gradient_finite=True;self.train_losses=[];self.validation_samples=0
        self.reference_sha=None;self.reference=None;self.pending=None;self.parts=[];self.phase=None
        self.exact_restore=None;self.hook=None

    def state_dict(self):
        return dict(steps=self.steps,gradient_finite=self.gradient_finite,train_losses=self.train_losses,
            validation_samples=self.validation_samples,reference_sha=self.reference_sha)

    def load_state_dict(self,state):
        for name,value in state.items():setattr(self,name,value)

    def on_load_checkpoint(self,trainer,pl_module,checkpoint):
        packet=checkpoint['vision_engineering']
        if packet['check_identity']!=self.identity or packet['initialization']!=self.prepared.identity:
            raise ValueError('native checkpoint engineering source/data identity differs')
        self.pending=checkpoint

    def on_fit_start(self,trainer,pl_module):
        if self.pending is not None:storage._restore_rng(self.pending['vision_engineering']['rng'],self.generators)
        self.original_forward=pl_module.forward
        def observed_forward(batch):
            output=self.original_forward(batch)
            if self.phase!='val':return output
            import torch
            pred,target,mask=output
            if pred.shape!=target.shape or pred.shape[1:]!=(22,1) or mask.shape!=(len(pred),1):
                raise ValueError('original validation output shape differs')
            if not all(bool(torch.isfinite(x).all()) for x in (pred,target)):
                raise ValueError('nonfinite original validation output')
            self.parts.append(dict(pred=pred.detach().cpu().clone(),target=target.detach().cpu().clone(),
                mask=mask.detach().cpu().clone(),indices=self.batch['vision_sample_index'].cpu().clone(),
                vision=self.batch['vision_valid_mask'].cpu().clone()))
            return output
        # Original _step calls self.forward and self.model.forward directly,
        # bypassing nn.Module.__call__ and its standard forward hooks.
        pl_module.forward=observed_forward
        self.step_hook=trainer.optimizers[0].register_step_post_hook(self._executed)

    def _executed(self,*args):self.steps+=1

    def on_train_start(self,trainer,pl_module):
        vision_optimizer_identity(self.prepared,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
        if getattr(trainer.precision_plugin,'scaler',None) is not None:
            raise ValueError('CPU32/bf16 engineering must not use an FP16 scaler')
        if trainer.gradient_clip_val!=1:raise ValueError('original gradient clip1 required')

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):self.phase='train'

    def on_before_optimizer_step(self,trainer,pl_module,optimizer):
        import torch
        gradients=[p.grad for p in pl_module.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            self.gradient_finite=False;raise ValueError('original vision gradients are absent/nonfinite')

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        value=float(outputs['loss'])
        if not math.isfinite(value):raise ValueError('nonfinite original training loss')
        self.train_losses.append(value)

    def on_validation_epoch_start(self,trainer,pl_module):self.parts=[]

    def on_validation_batch_start(self,trainer,pl_module,batch,batch_idx,dataloader_idx=0):
        self.phase='val';self.batch=batch

    def on_validation_end(self,trainer,pl_module):
        import torch
        ref={name:torch.cat([p[name] for p in self.parts]) for name in self.parts[0]}
        if ref['indices'].tolist()!=list(range(len(self.datasets['val']))):
            raise ValueError('validation must consume every repeated sample in exact original order')
        self.validation_samples=len(ref['indices']);self.reference=ref;self.reference_sha=storage.state_digest(ref)
        if not math.isfinite(float(trainer.callback_metrics['val_mae'])):raise ValueError('nonfinite native validation MAE')
        self.phase=None

    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        checkpoint['vision_engineering']=dict(format=FORMAT,check_identity=self.identity,
            initialization=self.prepared.identity,rng=storage._rng(self.generators),process=_process_identity())

    def on_fit_end(self,trainer,pl_module):
        pl_module.forward=self.original_forward
        self.step_hook.remove()


def _loaders(datasets,generators):
    from torch.utils.data import DataLoader
    return {role:DataLoader(ds,batch_size=2,shuffle=role=='train',num_workers=0,drop_last=False,
        generator=generators[role+'_loader']) for role,ds in datasets.items()}


def _replay_validation(prepared,loader,expected,device):
    import torch
    module=prepared.module.to(device).eval();parts=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
        for batch in loader:
            value={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            pred,target,mask=module.forward(value)
            parts.append(dict(pred=pred.detach().cpu(),target=target.detach().cpu(),mask=mask.detach().cpu(),
                indices=batch['vision_sample_index'],vision=batch['vision_valid_mask']))
    actual={name:torch.cat([p[name] for p in parts]) for name in expected}
    if storage.state_digest(actual)!=storage.state_digest(expected):
        raise ValueError('complete validation prediction/target/mask/order bytes differ on fresh reload')


def run_check(phase,*,data,crops,upstream,output,variant='vit_small',device='cpu'):
    import torch
    if phase not in ('prefix','resume') or device not in ('cpu','cuda'):raise ValueError('invalid engineering phase/device')
    if device=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU engineering requires explicit CUDA_VISIBLE_DEVICES empty before process startup')
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('actual CUDA bf16 support is required')
    output=Path(output);process=_process_identity()
    if phase=='prefix':
        if shutil.disk_usage(output.parent).free<15*1024**3:raise ValueError('15GiB disk reserve required')
        output.mkdir(exist_ok=False)
    elif (output/'resume.json').exists():raise FileExistsError('resume result already exists')
    # Verify trusted artifact bytes before creating any resumed model/trainer.
    checkpoint=None;receipt=None
    if phase=='resume':
        receipt=json.loads((output/'prefix.json').read_text())
        previous=receipt.get('process')
        if (not isinstance(previous,dict) or set(previous)!=set(process) or type(previous['pid']) is not int or
            type(previous['start_ticks']) is not str or not previous['start_ticks'].isdigit()):
            raise ValueError('original process identity is required for fresh restoration')
        if previous==process:raise ValueError('resume requires an actual different process')
        if _file_hash(output/'engineering-final.ckpt')!=receipt['checkpoint_sha256']:
            raise ValueError('checkpoint bytes differ from the original engineering receipt')
        if _file_hash(output/'validation-reference.pt')!=receipt['reference_file_sha256']:
            raise ValueError('validation reference bytes differ')
        checkpoint=torch.load(str(output/'engineering-final.ckpt'),map_location='cpu',weights_only=False)
        if checkpoint['vision_engineering']['process']!=previous:
            raise ValueError('checkpoint process differs from its engineering receipt')
    config=compose_vision(upstream,variant);datasets=_datasets(data,crops,upstream,config)
    identity=_check_identity(datasets,device,variant)
    if checkpoint is not None and (identity!=checkpoint['vision_engineering']['check_identity'] or
        identity!=receipt['check_identity']):raise ValueError('engineering data/source/config changed before resume')
    prepared=build_vision_runtime(upstream,config,**({} if checkpoint is None else dict(
        state=checkpoint['state_dict'],expected_initialization=checkpoint['vision_engineering']['initialization'])))
    generators={name+'_loader':torch.Generator().manual_seed(42) for name in ('train','val')}
    loaders=_loaders(datasets,generators)
    if checkpoint is not None:
        reference=torch.load(str(output/'validation-reference.pt'),map_location='cpu',weights_only=False)
        _replay_validation(prepared,loaders['val'],reference,device)
    witness=EngineeringWitness(prepared,identity,datasets,generators)
    from egoemg._train_utils import build_callbacks
    callbacks=build_callbacks(config)+[witness]
    from pytorch_lightning.callbacks import ModelCheckpoint
    for callback in callbacks:
        if isinstance(callback,ModelCheckpoint):callback.dirpath=str(output.resolve()/'native-checkpoints')
    logger=storage.CSVLogger(save_dir=str(output),name='native',version=phase)
    trainer=storage.pl.Trainer(accelerator='cpu' if device=='cpu' else 'gpu',devices=1,
        precision=identity['precision'],max_epochs=1 if phase=='prefix' else 2,
        limit_train_batches=2 if phase=='prefix' else 1,limit_val_batches=1.,num_sanity_val_steps=0,
        gradient_clip_val=1,callbacks=callbacks,logger=logger,enable_progress_bar=False,
        enable_model_summary=False,log_every_n_steps=1)
    before=vision_state_sha256(prepared.module.state_dict())
    connector=trainer._checkpoint_connector;original_restore=connector.restore_training_state
    def observed_restore():
        result=original_restore()
        if checkpoint is not None:
            expected=dict(model=storage.state_digest(checkpoint['state_dict']),
                optimizer=storage.state_digest(checkpoint['optimizer_states']),
                scheduler=storage.state_digest(checkpoint['lr_schedulers']),
                callbacks=storage.state_digest(_native_restored_callbacks(checkpoint['callbacks'])),
                loops=storage.state_digest(checkpoint['loops']['fit_loop']),
                rng=storage.state_digest(checkpoint['vision_engineering']['rng']))
            actual=_snapshots(prepared.module,trainer,generators)
            witness.exact_restore={name:actual[name]==expected[name] for name in expected}
            if not all(witness.exact_restore.values()):
                raise ValueError('actual native full checkpoint restoration differs: '+str(witness.exact_restore))
        return result
    connector.restore_training_state=observed_restore
    try:
        trainer.fit(prepared.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'],
            ckpt_path=None if checkpoint is None else str(output/'engineering-final.ckpt'))
    finally:connector.restore_training_state=original_restore
    expected_steps=2 if phase=='prefix' else 3
    if witness.steps!=expected_steps or trainer.global_step!=expected_steps:
        raise ValueError('actual native optimizer execution count differs from bounded protocol')
    if witness.validation_samples!=len(datasets['val']):raise ValueError('complete independent validation is missing')
    if phase=='resume' and (witness.exact_restore is None or not all(witness.exact_restore.values())):
        raise ValueError('actual native restore boundary was not observed')
    if _check_identity(datasets,device,variant)!=identity:raise ValueError('source/data changed during engineering')
    prepared.check()
    optimizer=vision_optimizer_identity(prepared,trainer.optimizers[0],trainer.lr_scheduler_configs[0].scheduler)
    final=output/('engineering-final.ckpt' if phase=='prefix' else 'engineering-resumed.ckpt')
    trainer.save_checkpoint(final)
    result=dict(format=FORMAT,phase=phase,scientific_eligible=False,production_ready=False,full_paper_complete=False,
        check_identity=identity,initialization_sha256=prepared.identity['sha256'],actual_optimizer_updates=witness.steps,
        new_actual_updates=2 if phase=='prefix' else 1,full_validation_samples=witness.validation_samples,
        validation_unique_hand_samples=len(datasets['val'])//datasets['val'].repeat,
        gradient_finite=witness.gradient_finite,train_losses=witness.train_losses,
        trained_weights_changed=before!=vision_state_sha256(prepared.module.state_dict()),
        native_gradient_clip_val=trainer.gradient_clip_val,precision=identity['precision'],optimizer=optimizer,
        checkpoint_sha256=_file_hash(final),cuda_initialized=torch.cuda.is_initialized(),
        process=process,restored_from_process=None if receipt is None else receipt['process'],
        fresh_process_restore=phase=='resume',full_validation_prediction_equal=phase=='resume',
        exact_restore=witness.exact_restore,
        callback_restore_policy='all native saved fields except ModelCheckpoint.current_score, which PL2.5 never restores',
        output_path_policy='phase-specific CSV logs; one stable owned native checkpoint directory for exact callback restore',
        restore_observation='immediately after unchanged native restore_training_state; before data iterator',
        loop_scope='complete truncated engineering epochs; not full-data sampler/trajectory equivalence')
    if phase=='prefix':
        torch.save(witness.reference,output/'validation-reference.pt')
        result['reference_file_sha256']=_file_hash(output/'validation-reference.pt')
    _json_create(output/(phase+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prefix','resume'))
    for name in ('data','crops','upstream','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--variant',choices=('vit_small','vit_base','vit_large'),default='vit_small')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=vars(parser.parse_args());result=run_check(**args)
    print(json.dumps({k:result[k] for k in ('phase','actual_optimizer_updates','full_validation_samples','scientific_eligible')}))


if __name__=='__main__':main()
