"""Explicit native WiLoR recovery; historical fits retain their own identity."""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import weakref

from .emg2pose_memory_process import same_process


ROOT = Path(__file__).resolve().parents[1]
_ADMITTED = weakref.WeakSet()
_FULL_COUNTS = dict(train=202020,val=3152,test=8612)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _seal(value):
    return dict(value,sha256=_hash(value))


def _path(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('parent/checkpoint path contains a symlink')
    return path.resolve()


def _stamp(path):
    s=Path(path).lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('parent/checkpoint must be a regular file')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]


def _json(path,limit=4*1024**2):
    _path(path);stamp=_stamp(path)
    if stamp[2]>limit:
        raise ValueError('bounded parent identity/terminal metadata required')
    value=json.loads(Path(path).read_bytes())
    if _stamp(path)!=stamp:
        raise ValueError('parent metadata changed during admission')
    return value


def _producer(parent):
    folder=parent.parent/'owned-fit'
    if not (folder/'stage-result.json').exists():
        raise ValueError('parent fit has no terminal owned-stage evidence')
    result=_json(folder/'stage-result.json');started=_json(folder/'started.json')
    if (result.get('status') not in ('success','failed') or type(result.get('worker_exit_code')) is not int
            or result.get('all_owned_consumers_exited') is not True
            or any(result.get(k)!=started.get(k) for k in ('owner','worker','guardian'))):
        raise ValueError('parent terminal/registration differs or is unclean')
    for key in ('worker','guardian'):
        owner=result.get(key)
        if (type(owner) is not dict or type(owner.get('pid')) is not int or owner['pid']<=0
                or type(owner.get('start_ticks')) is not str or not owner['start_ticks'].isdigit()
                or type(owner.get('uid')) is not int or owner['uid']!=os.getuid() or same_process(owner)):
            raise ValueError('parent fit owner is live or has invalid ownership')
    return started,result


def _original_parent(parent):
    if os.path.lexists(parent/'recovery-identity.json'):
        raise ValueError('recovered parent lineage requires an explicit recursive continuation reader')
    result=parent/'fit-result.json'
    if result.exists():
        value=_json(result)
        if type(value)is not dict or str(value.get('format','')).startswith('explicit_native_wilor_'):
            raise ValueError('recovered continuation cannot be admitted as an original parent')


def _parent_metadata(parent,checkpoint=None):
    """Cheap rejection only: this does not admit native checkpoint tensors."""
    parent=_path(parent);_original_parent(parent)
    metadata_paths=[parent/'identity.json',parent/'epochs.jsonl',
        parent.parent/'owned-fit/started.json',parent.parent/'owned-fit/stage-result.json']
    if not metadata_paths[-1].exists():
        raise ValueError('parent fit has no terminal owned-stage evidence')
    stamps={str(p):_stamp(p)for p in metadata_paths}
    identity=_json(parent/'identity.json')
    if (type(identity) is not dict or identity.get('format')!='native_wilor_uninterrupted_fit_v1'
            or identity.get('sha256')!=_hash({k:v for k,v in identity.items() if k!='sha256'})
            or identity.get('scientific_eligible') is not False or identity.get('full_paper_complete') is not False):
        raise ValueError('parent identity seal/format differs')
    policy=identity.get('policy',{});scope=policy.get('scope')
    if scope not in ('full','cuda_engineering','cpu_engineering'):
        raise ValueError('parent scope differs')
    counts=identity.get('counts',{});expected=_FULL_COUNTS if scope=='full' else (
        dict(train=2,val=2,test=2) if scope=='cpu_engineering' else dict(_FULL_COUNTS,train=36))
    if (type(counts) is not dict or set(counts)!=set(expected)
            or any(type(counts[k]) is not int or counts[k]!=v for k,v in expected.items())):
        raise ValueError('parent counts differ from its original scope')
    maximum=30 if scope=='full' else 2
    if type(policy.get('max_epochs')) is not int or policy['max_epochs']!=maximum:
        raise ValueError('parent original epoch budget differs')
    path=parent/'epochs.jsonl';_stamp(path);raw=path.read_bytes()
    if not raw or len(raw)>2*1024**2 or not raw.endswith(b'\n'):
        raise ValueError('parent requires a bounded complete epoch journal')
    records=[];previous='0'*64
    for line in raw.splitlines():
        value=json.loads(line);record=value.get('record',{})
        body=dict(previous_sha256=previous,identity_sha256=identity['sha256'],record=record)
        if (value!=_seal(body) or type(record.get('epoch')) is not int or record['epoch']!=len(records)):
            raise ValueError('parent journal chain/order differs')
        records.append(record);previous=value['sha256']
    if len(records)>maximum or (len(records)==maximum and checkpoint is None):
        raise ValueError('parent budget is already complete; no implicit recovery')
    if checkpoint is not None and scope=='full':
        raise ValueError('full checkpoint selection must use the newest complete packet')
    started,terminal=_producer(parent)
    paths=sorted((parent/'checkpoints').glob('*.ckpt'))
    if not paths or len(paths)>64:
        raise ValueError('parent requires bounded native checkpoint candidates')
    for p in paths:
        _path(p);stamp=_stamp(p)
        if stamp[6]!=os.getuid():
            raise ValueError('parent checkpoint is not owned by the current user')
    if checkpoint is not None:
        chosen=_path(checkpoint)
        if chosen not in paths:
            raise ValueError('explicit engineering checkpoint is outside parent checkpoints')
        paths=[chosen]
    if any(_stamp(p)!=s for p,s in stamps.items()):
        raise ValueError('parent metadata changed during inspection')
    return dict(parent=parent,identity=identity,records=records,journal_bytes=raw,
        started=started,terminal=terminal,candidates=paths,metadata_stamps=stamps,raw_checkpoint_admitted=False)


def _checkpoint_callbacks(checkpoint,parent,records):
    """The native selection mode is encoded in each callback's state key."""
    try:
        callbacks=checkpoint['callbacks'];states={}
        expected={
            'ModelCheckpoint':dict(monitor='val_mae',mode='min',every_n_train_steps=0,every_n_epochs=1,train_time_interval=None),
            'EarlyStopping':dict(monitor='val_mae',mode='min')}
        for name,fields in expected.items():
            matches=[(k,v)for k,v in callbacks.items()if type(k)is str and k.startswith(name+'{')]
            if len(matches)!=1:
                raise ValueError('parent callback key is missing or duplicated')
            key,value=matches[0]
            if _hash(ast.literal_eval(key[len(name):]))!=_hash(fields):
                raise ValueError('parent callback monitor/mode/interval differs')
            states[name]=value
        cb=states['ModelCheckpoint'];early=states['EarlyStopping'];n=len(records)
        values=[r['validation']['val_mae']for r in records];best=min(values)
        if (cb.get('monitor')!='val_mae' or Path(cb.get('dirpath',''))!=parent/'checkpoints'
                or not isinstance(cb.get('best_k_models'),dict) or len(cb['best_k_models'])!=min(3,n)
                or sorted(float(x)for x in cb['best_k_models'].values())!=sorted(values)[:3]
                or type(cb['best_model_score'])is bool or float(cb['best_model_score'])!=best
                or cb['best_model_path']not in cb['best_k_models'] or float(early['best_score'])!=best
                or any(type(early.get(k))is not int for k in ('patience','wait_count','stopped_epoch'))
                or early['patience']!=30 or early['wait_count']!=n-1-values.index(best) or early['stopped_epoch']!=0):
            raise ValueError('parent native top3/early-stopping callback history differs')
        required={_path(p):float(v)for p,v in cb['best_k_models'].items()}
        if cb.get('last_model_path'):required.setdefault(_path(cb['last_model_path']),None)
        return required
    except (KeyError,TypeError,IndexError,AttributeError,SyntaxError) as exc:
        raise ValueError('malformed parent native callback state')from exc


@dataclass(frozen=True,eq=False)
class AdmittedWilorRecovery:
    identity:dict
    records:list
    checkpoint:dict
    parent:Path
    checkpoint_path:Path
    callback_files:dict
    manifest:dict
    guard:object
    _seal:str

    def check(self):
        if (self not in _ADMITTED or self.manifest.get('sha256')!=self._seal
                or _hash({k:v for k,v in self.manifest.items()if k!='sha256'})!=self._seal
                or self.identity!=self.manifest['historical_identity']
                or self.records!=self.manifest['prefix_records']):
            raise ValueError('parent recovery admission identity changed')
        self.guard.check();_original_parent(self.parent);_producer(self.parent)


def admit_wilor_recovery(parent,*,checkpoint=None):
    metadata=_parent_metadata(parent,checkpoint)
    from . import wilor_native_train as native
    from .classic_emg_cohort import _StatGuard
    from .wilor_literal_runtime import _versions
    source_guard,sources,loops=native._capture_fit_sources()
    parent=metadata['parent'];identity=metadata['identity'];scope=identity['policy']['scope']
    if (identity['policy']!=native.wilor_policy(scope=scope)
            or identity.get('training_sources_sha256')!=sources
            or identity.get('native_loop_sources_sha256')!=loops
            or identity.get('construction',{}).get('runtime_versions')!=_versions()):
        raise ValueError('parent current native source/policy/runtime identity differs')
    guard=_StatGuard([parent/'identity.json',parent/'epochs.jsonl',
        parent.parent/'owned-fit/started.json',parent.parent/'owned-fit/stage-result.json',
        Path(__file__),*metadata['candidates']])
    guard=native._guard_merge(source_guard,guard);guard.check()
    if any(_stamp(p)!=s for p,s in metadata['metadata_stamps'].items()):
        raise ValueError('parent metadata changed before native packet loading')
    import torch
    candidates={}
    for p in metadata['candidates']:
        raw=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
        if type(raw)is not dict or 'explicit_wilor_recovery'in raw:
            raise ValueError('recovered raw parent requires explicit recursive continuation lineage')
        packet=raw.get('native_wilor',{});n=packet.get('completed_epochs')
        if (type(n) is not int or not 1<=n<=len(metadata['records'])
                or packet.get('identity')!=identity or packet.get('records')!=metadata['records'][:n]
                or type(raw.get('epoch')) is not int or raw['epoch']!=n-1):
            raise ValueError('parent native checkpoint identity/journal prefix differs')
        candidates[p]=(n,raw)
    # Prefer the actual current epoch packet over a lagging last save.
    selected=max(candidates,key=lambda p:(candidates[p][0],p.name!='last.ckpt',p.name))
    n,raw=candidates[selected];records=metadata['records'][:n]
    if n>=identity['policy']['max_epochs']:
        raise ValueError('parent selected checkpoint has already completed its budget')
    native._validate_checkpoint(raw,identity,records)
    owner=metadata['started']['worker'];process=raw['native_wilor']['execution']['process']
    if process!={k:owner[k]for k in ('pid','start_ticks','uid')}:
        raise ValueError('parent checkpoint process differs from its actual terminal owner')
    required=_checkpoint_callbacks(raw,parent,records)
    required.setdefault(selected,None);file_hashes={};callback_files={}
    for p,score in required.items():
        if p.parent!=parent/'checkpoints' or p not in candidates:
            raise ValueError('parent callback references an unavailable checkpoint')
        epochs,packet=candidates[p]
        if epochs>n:
            raise ValueError('parent callback references a future checkpoint')
        if p!=selected:native._validate_checkpoint(packet,identity,metadata['records'][:epochs])
        if (packet['native_wilor']['execution']!=raw['native_wilor']['execution'] or
                (score is not None and metadata['records'][epochs-1]['validation']['val_mae']!=score)):
            raise ValueError('parent callback checkpoint owner/score differs')
        digest=_sha(p);file_hashes[str(p)]=digest
        callback_files[p.name]=dict(parent_path=str(p),sha256=digest,bytes=p.stat().st_size,completed_epochs=epochs)
        guard.check()
    for p in (parent/'identity.json',parent/'epochs.jsonl',parent.parent/'owned-fit/started.json',parent.parent/'owned-fit/stage-result.json'):
        file_hashes[str(p)]=_sha(p)
    manifest=_seal(dict(format='explicit_native_wilor_recovery_admission_v1',parent=str(parent),historical_identity=copy.deepcopy(identity),
        parent_checkpoint=str(selected),parent_checkpoint_sha256=file_hashes[str(selected)],
        prefix_records=copy.deepcopy(records),parent_journal_records=len(metadata['records']),
        parent_complete_epochs=n,parent_optimizer_executions=records[-1]['optimizer_executions'],
        original_process=process,original_execution=copy.deepcopy(raw['native_wilor']['execution']),
        callback_files=callback_files,parent_files_sha256=file_hashes,
        core_sources_sha256=sources,native_loop_sources_sha256=loops,recovery_source_sha256=_sha(__file__),
        complete_current_input_admission=False,raw_checkpoint_admitted=True,
        scientific_eligible=False,full_paper_complete=False))
    result=AdmittedWilorRecovery(identity,records,raw,parent,selected,callback_files,manifest,guard,manifest['sha256'])
    _ADMITTED.add(result);result.check();return result


def _relocate_checkpoint(checkpoint,parent,output,callback_files,records):
    """A bootstrap view: only native callback output paths are changed."""
    parent=_path(parent);output=_path(output)
    required=_checkpoint_callbacks(checkpoint,parent,records)
    def relocated(value):
        if not value:return value
        source=_path(value);entry=callback_files.get(source.name)
        if (source.parent!=parent/'checkpoints' or entry is None
                or entry.get('parent_path')!=str(source)):
            raise ValueError('bootstrap callback checkpoint is outside its admitted copy map')
        return str(output/'checkpoints'/source.name)
    for source in required:relocated(str(source))
    value=dict(checkpoint);value['callbacks']=copy.deepcopy(checkpoint['callbacks'])
    for name,state in value['callbacks'].items():
        if not name.startswith('ModelCheckpoint{'):continue
        state['dirpath']=str(output/'checkpoints')
        for key in ('best_model_path','last_model_path','kth_best_model_path'):
            state[key]=relocated(state[key])
        state['best_k_models']={relocated(p):score for p,score in state['best_k_models'].items()}
    return value


def _filesystem_type(path):
    path=_path(path)
    while not path.exists():path=path.parent
    candidates=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before,separator,after=line.partition(' - ')
        if not separator:continue
        mount=re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),before.split()[4])
        point=Path(mount)
        if path.is_relative_to(point):candidates.append((len(point.parts),after.split()[0]))
    if not candidates:raise OSError('checkpoint filesystem type is unavailable')
    return max(candidates)[1]


def _claim_recovery_output(output,*,full,parent=None):
    if type(full)is not bool:raise ValueError('recovery output scope must be boolean')
    output=_path(output)
    if parent is not None:
        parent=_path(parent)
        capsule=parent.parent.parent if parent.parent.name=='private'else parent.parent
        if output.is_relative_to(capsule):
            raise ValueError('recovery output must be outside the historical parent capsule')
    if output.exists():raise FileExistsError('recovery output exists; preserve the prior attempt')
    if full and _filesystem_type(output)in ('tmpfs','ramfs'):
        raise OSError('full WiLoR continuation requires durable checkpoint storage')
    from .classic_emg_train import _space,_fsync_dir
    from .wilor_native_train import _CHECKPOINT_WORK_BYTES
    _space(output.parent,budget=_CHECKPOINT_WORK_BYTES*(5 if full else 4))
    # mkdir is the exclusive, durable attempt claim; a failed attempt is retained.
    output.mkdir(mode=0o700)
    _fsync_dir(output.parent)
    return output


def _recovery_device_admission(historical,actual):
    """Keep native hardware/runtime and reserves; record physical UUID changes."""
    if type(historical)is not dict or type(actual)is not dict:
        raise ValueError('actual native recovery device dictionaries required')
    if historical.get('accelerator')=='cpu':
        if _hash(historical)!=_hash(dict(accelerator='cpu',cuda_initialized=False))or _hash(actual)!=_hash(historical):
            raise ValueError('CPU recovery device/runtime differs')
    else:
        fields={'accelerator','name','free_memory_before','total_memory','visible_device','cuda_runtime','cudnn'}
        if set(historical)!=fields or set(actual)!=fields:
            raise ValueError('complete original CUDA recovery device/runtime required')
        for value in (historical,actual):
            if (value['accelerator']!='gpu' or type(value['free_memory_before'])is not int
                    or value['free_memory_before']<64*1024**3 or type(value['total_memory'])is not int
                    or type(value['visible_device'])is not str
                    or re.fullmatch(r'GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}',value['visible_device'])is None):
                raise ValueError('native recovery CUDA UUID/device64GiB reserve differs')
        retained=fields-{'visible_device','free_memory_before'}
        if _hash({k:historical[k]for k in retained})!=_hash({k:actual[k]for k in retained}):
            raise ValueError('native recovery GPU hardware/runtime differs')
    return dict(historical_device=copy.deepcopy(historical),actual_device=copy.deepcopy(actual),
        same_native_hardware_runtime=True,cuda_uuid_changed=historical.get('visible_device')!=actual.get('visible_device'))


def _packet_snapshot(checkpoint):
    from .classic_emg_train import state_digest as digest
    from .vision_literal_check import _native_restored_callbacks
    return dict(model=digest(checkpoint['state_dict']),optimizer=digest(checkpoint['optimizer_states']),
        scheduler=digest(checkpoint['lr_schedulers']),callbacks=digest(_native_restored_callbacks(checkpoint['callbacks'])),
        loops=digest(checkpoint['loops']['fit_loop']),rng=digest(checkpoint['native_wilor']['rng']))


def _install_restore_gate(trainer,module,checkpoint,*,guard):
    """Observe native restoration before PL2.5 constructs any train iterator."""
    from . import classic_emg_train as storage
    from .vision_literal_train import _snap
    expected=_packet_snapshot(checkpoint);connector=trainer._checkpoint_connector
    original=connector.restore_training_state
    receipt=dict(calls=0,expected=expected,restored_before_iterator=False)
    def restore():
        if receipt['calls']!=0:raise ValueError('native restoration must run exactly once')
        guard()
        if trainer.fit_loop._combined_loader is not None or trainer.fit_loop._data_fetcher is not None:
            raise ValueError('native restoration gate reached after iterator construction')
        receipt['calls']+=1
        result=original()
        storage._restore_rng(checkpoint['native_wilor']['rng'],{})
        actual=_snap(module,trainer);equal={k:actual[k]==v for k,v in expected.items()}
        receipt.update(actual=actual,equal=equal)
        if not all(equal.values()):raise ValueError('native checkpoint restoration domains differ')
        guard();receipt['restored_before_iterator']=True
        return result
    connector.restore_training_state=restore
    return receipt


def _validate_restore_receipt(receipt,expected):
    fields={'model','optimizer','scheduler','callbacks','loops','rng'}
    if (type(receipt)is not dict or type(receipt.get('calls'))is not int or receipt['calls']!=1
            or receipt.get('restored_before_iterator')is not True or type(expected)is not dict or set(expected)!=fields
            or type(receipt.get('equal'))is not dict or set(receipt['equal'])!=fields
            or any(value is not True for value in receipt['equal'].values())):
        raise ValueError('complete typed native restoration receipt required')
    for values in (expected,receipt.get('expected'),receipt.get('actual')):
        if (type(values)is not dict or set(values)!=fields or any(type(v)is not str or re.fullmatch('[0-9a-f]{64}',v)is None for v in values.values())
                or values!=expected):
            raise ValueError('native restoration receipt differs from original parent state')


RECOVERY_IDENTITY='explicit_native_wilor_continuation_identity_v1'
RECOVERY_FIT='explicit_native_wilor_recovered_fit_v1'
RECOVERY_PACKET='explicit_native_wilor_continuation_checkpoint_v1'


def _copy_callback_files(callback_files,output,*,guard):
    import errno
    import fcntl
    import shutil
    from .classic_emg_train import _space,_fsync_dir
    directory=_path(output)/'checkpoints';directory.mkdir(mode=0o700)
    result={}
    for name,entry in callback_files.items():
        if type(name)is not str or Path(name).name!=name or not name.endswith('.ckpt'):
            raise ValueError('invalid callback checkpoint filename')
        guard();source=_path(entry['parent_path']);before=_stamp(source)
        if type(entry.get('bytes'))is not int or before[2]!=entry['bytes']:
            raise ValueError('callback checkpoint byte count differs')
        _space(directory,budget=entry['bytes'])
        destination=directory/name;cloned=True
        with source.open('rb')as reader,destination.open('xb')as writer:
            try:fcntl.ioctl(writer.fileno(),0x40049409,reader.fileno())
            except OSError as error:
                if error.errno not in (errno.EXDEV,errno.EOPNOTSUPP,errno.EINVAL,errno.ENOTTY,errno.ENOSYS):raise
                cloned=False;shutil.copyfileobj(reader,writer,8*1024**2)
            writer.flush();os.fsync(writer.fileno())
        if _sha(destination)!=entry['sha256']or _stamp(source)!=before:
            raise ValueError('callback checkpoint copy/source bytes differ')
        guard();result[name]=dict(sha256=entry['sha256'],bytes=entry['bytes'],reflink=cloned)
    _fsync_dir(directory);return result


def _write(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _frozen_validation(prepared,loader,expected):
    import torch
    from contextlib import nullcontext
    from . import classic_emg_train as storage,wilor_native_train as native
    from .vision_literal_runtime import vision_state_sha256
    module=prepared.runtime.module;device=next(module.parameters()).device
    before=storage._rng({});was_training=module.training;state=vision_state_sha256(module.state_dict());parts=[];samples=0
    def move(value):
        if torch.is_tensor(value):return value.to(device)
        if isinstance(value,dict):return {k:move(v)for k,v in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(move(v)for v in value)
        return value
    module.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                prepared.check();batch=move(batch)
                context=torch.autocast('cuda',dtype=torch.bfloat16)if device.type=='cuda'else nullcontext()
                with context:out=module(batch)
                digest,observed=native._prediction_part(batch,out,module);parts.append(digest);samples+=len(batch['vision_img'])
                if observed!=str(device):raise ValueError('frozen native validation device differs')
    finally:module.train(was_training);storage._restore_rng(before,{})
    actual=native._hash(parts)
    if samples!=expected['samples']or actual!=expected['prediction_sha256']or vision_state_sha256(module.state_dict())!=state:
        raise ValueError('frozen validation differs from the actual parent checkpoint')
    return dict(samples=samples,prediction_sha256=actual,parent_prediction_exact=True,model_state_unchanged=True)


def _bootstrap_io(admitted,bootstrap):
    from pytorch_lightning.plugins.io import TorchCheckpointIO
    class NativeBootstrapIO(TorchCheckpointIO):
        def load_checkpoint(self,path,map_location=None):
            admitted.check()
            if _path(path)!=admitted.checkpoint_path:
                raise ValueError('native bootstrap path differs from admitted parent')
            return dict(bootstrap)
    return NativeBootstrapIO()


def resume_wilor(admitted,output):
    if type(admitted)is not AdmittedWilorRecovery or admitted not in _ADMITTED:
        raise ValueError('producer-admitted native WiLoR recovery required')
    admitted.check()
    from . import wilor_native_train as native,classic_emg_train as storage
    from .wilor_literal_runtime import admit_wilor_assets,build_wilor_runtime
    from .classic_emg_cohort import _StatGuard
    from .vision_literal_train import _epoch_bytes,_read_epochs
    from omegaconf import OmegaConf
    import time
    import torch
    identity=admitted.identity;policy=identity['policy'];scope=policy['scope'];full=scope=='full'
    output=_claim_recovery_output(output,full=full,parent=admitted.parent)
    native._validate_checkpoint(admitted.checkpoint,identity,admitted.records)
    cfg=OmegaConf.create(identity['saved_config']);locations=identity['locations']
    device=native._runtime_device(policy);native._validated(locations['upstream'],cfg)
    device_admission=_recovery_device_admission(identity['runtime_device'],device)
    cohort=native.admit_wilor_cohort(locations['data'],locations['crops'],upstream=locations['upstream'],config=cfg,
        expected=identity['cohort'],require_full=scope!='cpu_engineering')
    qualified=native._raw_qualification(locations['qualification'])if full else None
    if qualified!=identity['qualification']:raise ValueError('parent native CUDA qualification changed')
    assets=admit_wilor_assets(locations['assets_runtime'],restore=True)
    copies=_copy_callback_files(admitted.callback_files,output,guard=admitted.check)
    # Preserve the original identity and exact journal prefix; the continuation
    # receives its own format rather than relabeling the historical fit.
    with(output/'identity.json').open('xb')as stream:stream.write((admitted.parent/'identity.json').read_bytes())
    prefix=_epoch_bytes(admitted.records,identity['sha256'])
    if not(admitted.parent/'epochs.jsonl').read_bytes().startswith(prefix):raise ValueError('parent journal prefix changed')
    with(output/'epochs.jsonl').open('xb')as stream:stream.write(prefix);stream.flush();os.fsync(stream.fileno())
    continuation=_seal(dict(format=RECOVERY_IDENTITY,historical_identity_sha256=identity['sha256'],
        parent_admission=copy.deepcopy(admitted.manifest),parent_file_stamps={p:_stamp(p)for p in admitted.manifest['parent_files_sha256']},
        prefix_epochs=len(admitted.records),prefix_optimizer_executions=admitted.records[-1]['optimizer_executions'],
        callback_copies=copies,recovery_source_path=str(Path(__file__).resolve()),recovery_source_sha256=_sha(__file__),
        new_process=native._process(),started_ns=time.time_ns(),actual_runtime_device=device,device_admission=device_admission,
        actual_current_input_admission_complete=True,uninterrupted_execution=False,scientific_eligible=False,full_paper_complete=False))
    _write(output/'recovery-identity.json',continuation)
    runtime=build_wilor_runtime(cohort.upstream,cfg,assets=assets,state=admitted.checkpoint['state_dict'],expected_initialization=identity['construction'])
    if qualified is not None:native._qualification_binding(qualified,cohort,runtime.identity)
    guard=native._guard_merge(admitted.guard,cohort.guard,assets._guard,runtime._source_guard,
        _StatGuard([output/'identity.json',output/'recovery-identity.json',Path(__file__)]))
    prepared=native.PreparedWilorFit(identity,cohort,runtime,guard,identity['sha256']);native._FITS.add(prepared);prepared.check()
    loaders=native.native_wilor_loaders(cohort,scope=scope)
    runtime.module.to(torch.device('cuda',0)if full or scope=='cuda_engineering'else torch.device('cpu'))
    frozen=_frozen_validation(prepared,loaders['val'],admitted.records[-1]['validation'])
    bootstrap=_relocate_checkpoint(admitted.checkpoint,admitted.parent,output,admitted.callback_files,admitted.records)
    restore_receipt={}
    Base=native._observer_type()
    class RecoveryObserver(Base):
        def on_save_checkpoint(self,trainer,module,checkpoint):
            super().on_save_checkpoint(trainer,module,checkpoint)
            if restore_receipt.get('restored_before_iterator')is not True:
                raise ValueError('recovery checkpoint has no verified native restore boundary')
            checkpoint['explicit_wilor_recovery']=dict(format=RECOVERY_PACKET,continuation_identity_sha256=continuation['sha256'],
                prefix_epochs=len(admitted.records),prefix_optimizer_executions=continuation['prefix_optimizer_executions'],
                native_restore=copy.deepcopy(restore_receipt),frozen_validation=copy.deepcopy(frozen))
    witness=RecoveryObserver(prepared,output);witness.records=copy.deepcopy(admitted.records);witness.steps=continuation['prefix_optimizer_executions']
    from egoemg._train_utils import build_callbacks,build_logger
    from pytorch_lightning.callbacks import ModelCheckpoint
    callbacks=build_callbacks(cfg)+[witness]
    for callback in callbacks:
        if isinstance(callback,ModelCheckpoint):callback.dirpath=str(output/'checkpoints')
    effective=copy.deepcopy(cfg);effective.logger.save_dir=str(output);effective.logger.name='logs';effective.logger.version='native-wilor-recovery'
    kwargs=OmegaConf.to_container(cfg.trainer,resolve=True)
    kwargs.update(accelerator=device['accelerator'],devices=1,precision=policy['precision'],max_epochs=policy['max_epochs'],
        num_sanity_val_steps=policy['num_sanity_val_steps'],enable_progress_bar=False,callbacks=callbacks,
        logger=build_logger(effective),plugins=[_bootstrap_io(admitted,bootstrap)])
    trainer=storage.pl.Trainer(**kwargs)
    receipt=_install_restore_gate(trainer,runtime.module,bootstrap,guard=prepared.check)
    restore_receipt=receipt
    started=time.monotonic()
    try:trainer.fit(runtime.module,train_dataloaders=loaders['train'],val_dataloaders=loaders['val'],ckpt_path=str(admitted.checkpoint_path))
    finally:witness.restore(runtime.module)
    prepared.check();admitted.check()
    if not native.wilor_stage_complete(policy,witness.records,train_samples=identity['counts']['train'],val_samples=identity['counts']['val'],interrupted=bool(trainer.interrupted)):
        raise ValueError('recovered native WiLoR did not complete its original budget')
    last_path=output/'checkpoints/last.ckpt';last=torch.load(str(last_path),map_location='cpu',weights_only=False,mmap=True)
    records,_=_read_epochs(output/'epochs.jsonl',identity['sha256']);native._validate_checkpoint(last,identity,records)
    result=_seal(dict(format=RECOVERY_FIT,identity_sha256=identity['sha256'],continuation_identity_sha256=continuation['sha256'],scope=scope,
        completed_epochs=len(records),optimizer_executions=witness.steps,prefix_epochs=len(admitted.records),
        prefix_optimizer_executions=continuation['prefix_optimizer_executions'],segment_optimizer_executions=witness.steps-continuation['prefix_optimizer_executions'],
        train_samples_per_epoch=[r['train_samples']for r in records],val_samples_per_epoch=[r['validation']['samples']for r in records],
        initial_model_sha256=identity['construction']['initial_state_sha256'],final_model_state_sha256=records[-1]['model_sha256'],
        last_checkpoint_sha256=_sha(last_path),selected_best=native._selection(output,identity,last,records),journal_sha256=_sha(output/'epochs.jsonl'),
        full_training_terminated=full,stage_complete=True,execution=witness.execution,native_restore=receipt,frozen_validation=frozen,
        seconds=time.monotonic()-started,finished_ns=time.time_ns(),checkpoint_continuation_supported=True,uninterrupted_execution=False,
        scientific_eligible=False,full_paper_complete=False))
    _write(output/'fit-result.json',result)
    return read_wilor_recovery(output,require_full=full)


def _read_recovery(output,*,require_full):
    if type(require_full)is not bool:raise ValueError('full recovered fit requirement must be boolean')
    output=_path(output)
    if not(output/'recovery-identity.json').exists():raise ValueError('explicit recovery identity is required')
    from . import wilor_native_train as native
    from .classic_emg_cohort import _StatGuard
    from .vision_literal_train import _read_epochs,_epoch_bytes
    from .wilor_literal_runtime import admit_wilor_assets,_restore_identity,_sources as runtime_sources,_versions
    from omegaconf import OmegaConf
    import torch
    paths=[output/n for n in ('identity.json','recovery-identity.json','fit-result.json','epochs.jsonl','checkpoints/last.ckpt')]
    guard=_StatGuard([*paths,Path(__file__)]);identity=_json(paths[0]);continuation=_json(paths[1]);result=_json(paths[2])
    for value,kind in ((identity,native.FORMAT),(continuation,RECOVERY_IDENTITY),(result,RECOVERY_FIT)):
        if (value.get('format')!=kind or value.get('sha256')!=_hash({k:v for k,v in value.items()if k!='sha256'})
                or value.get('scientific_eligible')is not False or value.get('full_paper_complete')is not False):
            raise ValueError('explicit recovery format/identity/result seal differs')
    source_guard,sources,loops=native._capture_fit_sources();guard=native._guard_merge(guard,source_guard)
    if (identity['policy']!=native.wilor_policy(scope=identity['policy']['scope'])or identity['training_sources_sha256']!=sources
            or identity['native_loop_sources_sha256']!=loops or continuation['historical_identity_sha256']!=identity['sha256']
            or continuation['recovery_source_sha256']!=_sha(__file__)or continuation['uninterrupted_execution']is not False
            or result['continuation_identity_sha256']!=continuation['sha256']or result['identity_sha256']!=identity['sha256']):
        raise ValueError('explicit recovery current source or historical identity differs')
    parent=continuation['parent_admission']
    if parent.get('sha256')!=_hash({k:v for k,v in parent.items()if k!='sha256'})or parent['historical_identity']!=identity:
        raise ValueError('explicit recovery parent admission differs')
    _original_parent(_path(parent['parent']))
    if continuation.get('device_admission')!=_recovery_device_admission(identity['runtime_device'],continuation['actual_runtime_device']):
        raise ValueError('explicit recovery device admission differs from historical hardware/runtime')
    for p,s in continuation['parent_file_stamps'].items():
        if _stamp(p)!=s:raise ValueError('explicit recovery parent generation changed')
        guard.add(p)
    prefix=parent['prefix_records'];records,_=_read_epochs(output/'epochs.jsonl',identity['sha256'])
    if (records[:len(prefix)]!=prefix or not(output/'epochs.jsonl').read_bytes().startswith(_epoch_bytes(prefix,identity['sha256']))
            or continuation['prefix_epochs']!=len(prefix)or continuation['prefix_optimizer_executions']!=prefix[-1]['optimizer_executions']):
        raise ValueError('explicit recovery parent journal prefix differs')
    scope=identity['policy']['scope'];full=scope=='full'
    if require_full and not full:raise ValueError('engineering recovery is not a full native WiLoR fit')
    if not native.wilor_stage_complete(identity['policy'],records,train_samples=identity['counts']['train'],val_samples=identity['counts']['val']):
        raise ValueError('explicit recovery did not complete original budget')
    locations=identity['locations'];cfg=OmegaConf.create(identity['saved_config'])
    cohort=native.admit_wilor_cohort(locations['data'],locations['crops'],upstream=locations['upstream'],config=cfg,expected=identity['cohort'],require_full=scope!='cpu_engineering')
    assets=admit_wilor_assets(locations['assets_runtime'],restore=True)
    qualified=native._raw_qualification(locations['qualification'])if full else None
    if qualified!=identity['qualification']:raise ValueError('explicit recovery parent qualification differs')
    last_path=output/'checkpoints/last.ckpt'
    if _sha(last_path)!=result['last_checkpoint_sha256']:raise ValueError('explicit recovery last checkpoint changed')
    last=torch.load(str(last_path),map_location='cpu',weights_only=False,mmap=True);native._validate_checkpoint(last,identity,records)
    protocol=native.wilor_protocol(locations['upstream'],cfg)
    _restore_identity(identity['construction'],protocol,runtime_sources(Path(locations['upstream']),protocol,assets),_versions(),last['state_dict'])
    packet=last.get('explicit_wilor_recovery',{})
    receipt=packet.get('native_restore',{})
    parent_checkpoint=_path(parent['parent_checkpoint'])
    if str(parent_checkpoint)not in continuation['parent_file_stamps']or _sha(parent_checkpoint)!=parent['parent_checkpoint_sha256']:
        raise ValueError('explicit recovery original parent packet changed')
    original=torch.load(str(parent_checkpoint),map_location='cpu',weights_only=False,mmap=True)
    native._validate_checkpoint(original,identity,prefix)
    if original['native_wilor']['execution']!=parent['original_execution']:
        raise ValueError('explicit recovery original process lineage differs')
    bootstrap=_relocate_checkpoint(original,Path(parent['parent']),output,parent['callback_files'],prefix)
    _validate_restore_receipt(receipt,_packet_snapshot(bootstrap))
    del original,bootstrap
    if (packet.get('format')!=RECOVERY_PACKET or packet.get('continuation_identity_sha256')!=continuation['sha256']
            or type(packet.get('prefix_epochs'))is not int or packet['prefix_epochs']!=len(prefix)
            or type(packet.get('prefix_optimizer_executions'))is not int or packet['prefix_optimizer_executions']!=continuation['prefix_optimizer_executions']):
        raise ValueError('explicit recovery native restoration evidence differs')
    frozen=dict(samples=prefix[-1]['validation']['samples'],prediction_sha256=prefix[-1]['validation']['prediction_sha256'],
        parent_prediction_exact=True,model_state_unchanged=True)
    if _hash(packet.get('frozen_validation'))!=_hash(frozen):raise ValueError('explicit recovery frozen parent validation differs')
    selection=native._selection(output,identity,last,records);guard.add(output/'checkpoints'/selection['file'])
    expected=dict(format=RECOVERY_FIT,identity_sha256=identity['sha256'],continuation_identity_sha256=continuation['sha256'],scope=scope,
        completed_epochs=len(records),optimizer_executions=records[-1]['optimizer_executions'],prefix_epochs=len(prefix),
        prefix_optimizer_executions=continuation['prefix_optimizer_executions'],segment_optimizer_executions=records[-1]['optimizer_executions']-continuation['prefix_optimizer_executions'],
        train_samples_per_epoch=[r['train_samples']for r in records],val_samples_per_epoch=[r['validation']['samples']for r in records],
        initial_model_sha256=identity['construction']['initial_state_sha256'],final_model_state_sha256=records[-1]['model_sha256'],
        last_checkpoint_sha256=_sha(last_path),selected_best=selection,journal_sha256=_sha(output/'epochs.jsonl'),
        full_training_terminated=full,stage_complete=True,execution=last['native_wilor']['execution'],native_restore=receipt,
        frozen_validation=packet['frozen_validation'],checkpoint_continuation_supported=True,uninterrupted_execution=False,scientific_eligible=False,full_paper_complete=False)
    if (set(result)!=set(expected)|{'seconds','finished_ns','sha256'}or _hash({k:result[k]for k in expected})!=_hash(expected)
            or type(result['seconds'])not in(int,float)or not math.isfinite(result['seconds'])or result['seconds']<0
            or type(result['finished_ns'])is not int or result['finished_ns']<=result['execution']['started_ns']
            or result['execution']['process']!=continuation['new_process']or result['segment_optimizer_executions']<=0
            or result['frozen_validation'].get('parent_prediction_exact')is not True):
        raise ValueError('explicit recovered result differs from actual raw checkpoint/journal')
    guard=native._guard_merge(guard,cohort.guard,assets._guard);guard.check()
    return result,identity,last,cohort,assets,guard


def read_wilor_recovery(output,*,require_full=True):
    return _read_recovery(output,require_full=require_full)[0]


def reopen_wilor_recovery(output,*,require_full=True):
    result,identity,last,cohort,assets,guard=_read_recovery(output,require_full=require_full)
    import torch
    from . import wilor_native_train as native
    from .wilor_literal_runtime import build_wilor_runtime
    path=_path(output)/'checkpoints'/result['selected_best']['file']
    checkpoint=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
    runtime=build_wilor_runtime(cohort.upstream,cohort.config,assets=assets,state=checkpoint['state_dict'],expected_initialization=identity['construction'])
    prepared=native.PreparedWilorFit(identity,cohort,runtime,native._guard_merge(guard,runtime._source_guard),identity['sha256'],
        selection=result['selected_best'],checkpoint=checkpoint,run=_path(output))
    native._FITS.add(prepared);prepared.check();return prepared
