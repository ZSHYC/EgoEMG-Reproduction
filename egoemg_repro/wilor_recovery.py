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


def _parent_metadata(parent,checkpoint=None):
    """Cheap rejection only: this does not admit native checkpoint tensors."""
    parent=_path(parent)
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
        self.guard.check();_producer(self.parent)


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


def _claim_recovery_output(output,*,full):
    if type(full)is not bool:raise ValueError('recovery output scope must be boolean')
    output=_path(output)
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
