"""Bounded formal-state contracts. No training entry or production authorization.

Checkpoint pickle is trusted local input only. Synthetic contract tests do not
qualify CUDA. Actual full-cohort/final-source/precision acceptance belongs to the
future runner; numeric qualification references alone are never a run permit.
"""
from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import uuid

import numpy as np

from . import classic_emg_cuda_check as cuda
from .data import file_sha256

storage=cuda.storage
IDENTITY_FORMAT='classic_formal_training_v1'
CHECKPOINT_FORMAT='classic_formal_epoch_v1'
SCOPE='formal_training_state_contract'
MAX_RECORD_BYTES=1024
MAX_SEGMENT_BYTES=16*1024**2
MIN_LOG_BUDGET=64*1024**2
BINDING_KEYS=('cohort_receipt_sha256','data_sha256','partition_sha256','normalization_sha256',
              'numeric_qualification_artifact_sha256')
SOURCE_FILES=(*cuda.RUNTIME_FILES,'egoemg_repro/classic_emg_cohort.py','egoemg_repro/classic_emg_artifacts.py',
              'configs/vemg2pose.json','configs/neuropose.json')
FRAMEWORK_FILES=storage.FRAMEWORK_FILES
IMPORTED_SOURCE_SHA256=file_sha256(Path(__file__))
CheckpointIO=storage.CheckpointIO


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _hex(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid content digest')


def _safe(path):
    if '..' in Path(path).parts: raise ValueError('unsafe parent traversal')
    return storage._safe(path)


def _native(value):
    if type(value) in (str,bool,int,type(None)): return
    if type(value) is float and math.isfinite(value): return
    if type(value) is list:
        for v in value: _native(v)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for v in value.values(): _native(v)
        return
    raise ValueError('identity must retain JSON-native types and finite numbers')


def validate_formal_identity(identity,*,expected=None):
    """Validate exact state identity, not evidence granting a production permit."""
    try:
        _native(identity)
        keys={'format','scope','production_ready','scientific_eligible','recipe','effective','counts','bindings',
              'runtime','source_files_sha256','framework_files_sha256','model_schema','parameter_names','sha256'}
        if set(identity)!=keys or identity['format']!=IDENTITY_FORMAT or identity['scope']!=SCOPE:
            raise ValueError('formal artifact identity/scope differs')
        if identity['production_ready'] is not False or identity['scientific_eligible'] is not False:
            raise ValueError('artifact identity cannot authorize production/scientific execution')
        if identity['sha256']!=json_hash({k:v for k,v in identity.items() if k!='sha256'}):
            raise ValueError('formal identity digest differs')
        recipe=identity['recipe'];variant=recipe['variant'];batch=600 if variant=='vemg2pose' else 256
        if variant not in ('vemg2pose','neuropose'): raise ValueError('unknown formal variant')
        # Compare complete recipe to its frozen public JSON, not a partial allowlist.
        root=Path(__file__).resolve().parents[1]
        canonical=json.loads((root/'configs'/f'{variant}.json').read_text())
        if json_hash(recipe)!=json_hash(canonical): raise ValueError('exact formal recipe differs')
        effective=dict(precision='16-mixed',batch_size=batch,global_batch_size=batch,world_size=1,rank=0,
            num_workers=0,train_shuffle=True,seed=42,log_every_n_steps=50,max_epochs=100,clip_value=0)
        if json_hash(identity['effective'])!=json_hash(effective): raise ValueError('formal effective policy differs')
        if set(identity['counts'])!={'train','val','test'} or any(type(n) is not int or n<=0 for n in identity['counts'].values()):
            raise ValueError('formal complete cohort counts missing')
        if set(identity['bindings'])!=set(BINDING_KEYS): raise ValueError('formal bindings incomplete')
        for digest in identity['bindings'].values(): _hex(digest)
        cuda._check_backend_profile(identity['runtime'],cuda.cuda_check_policy(variant=variant))
        if identity['runtime']['precision_plugin']!='MixedPrecision' or identity['runtime']['scaler_enabled'] is not True:
            raise ValueError('formal precision contract requires actual FP16 scaler state')
        if file_sha256(Path(__file__))!=IMPORTED_SOURCE_SHA256:
            raise ValueError('executed artifacts source changed after import')
        if identity['source_files_sha256']!={n:file_sha256(root/n) for n in SOURCE_FILES}:
            raise ValueError('formal executed source identity differs')
        if storage.pl is None: raise ValueError('actual Lightning source unavailable')
        framework=Path(storage.pl.__file__).resolve().parent
        if identity['framework_files_sha256']!={n:file_sha256(framework/n) for n in FRAMEWORK_FILES}:
            raise ValueError('formal framework source differs')
        schema=identity['model_schema'];names=identity['parameter_names']
        if not schema or not names or len(set(names))!=len(names) or any(n not in schema for n in names):
            raise ValueError('complete model/parameter schema required')
        for name,spec in schema.items():
            if (not isinstance(name,str) or set(spec)!={'dtype','shape'} or not isinstance(spec['dtype'],str) or
                not isinstance(spec['shape'],list) or any(type(v) is not int or v<0 for v in spec['shape'])):
                raise ValueError('invalid model schema')
        if any(schema[n]['dtype']!='torch.float32' for n in names): raise ValueError('formal AMP parameters remain float32')
        if expected is not None and json_hash(identity)!=json_hash(expected): raise ValueError('formal resume identity mismatch')
    except (KeyError,TypeError,AttributeError,OSError) as exc:
        raise ValueError('incomplete formal identity') from exc


def _rename_noreplace(directory,source,destination):
    """Require Linux atomic publication; never fall back to overwriting."""
    if sys.platform!='linux': raise RuntimeError('Linux atomic NOREPLACE publication is required')
    libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None)
    if rename is None: raise RuntimeError('atomic renameat2 NOREPLACE is unsupported')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(directory,os.fsencode(source),directory,os.fsencode(destination),1):
        code=ctypes.get_errno()
        if code in (errno.ENOSYS,errno.EINVAL,errno.EOPNOTSUPP):
            raise RuntimeError('atomic renameat2 NOREPLACE is unsupported')
        raise OSError(code,os.strerror(code),destination)


class _CreatedDirectory(cuda._OwnedOutput):
    """Hold private staging ownership before publishing a fresh public name.

    The caller supplies a trusted parent and must protect the private random
    staging name during mkdir/open acquisition: Linux mkdir returns no fd.
    A later public-name open only verifies the held inode; it cannot adopt one.
    """
    def __init__(self,path):
        self.path=_safe(path);self.fd=None
        parent_fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        parent=os.fstat(parent_fd);staging='.formal-artifacts-stage-'+uuid.uuid4().hex;published=False
        try:
            os.mkdir(staging,mode=0o700,dir_fd=parent_fd)
            self.fd=os.open(staging,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent_fd)
            own=os.fstat(self.fd);self.identity=(own.st_dev,own.st_ino)
            current=_safe(self.path.parent).stat();staged=os.stat(staging,dir_fd=parent_fd,follow_symlinks=False)
            if ((current.st_dev,current.st_ino)!=(parent.st_dev,parent.st_ino) or
                    (staged.st_dev,staged.st_ino)!=self.identity or not stat.S_ISDIR(staged.st_mode)):
                raise ValueError('private staging directory identity changed')
            _rename_noreplace(parent_fd,staging,self.path.name);published=True;os.fsync(parent_fd)
            public_fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
            try:
                current=os.fstat(public_fd)
                if (current.st_dev,current.st_ino)!=self.identity or not self.is_current(self.path):
                    raise ValueError('published owned artifact directory identity changed')
            finally: os.close(public_fd)
        except BaseException:
            if not published and self.fd is not None:
                try:
                    current=os.stat(staging,dir_fd=parent_fd,follow_symlinks=False)
                    if (current.st_dev,current.st_ino)==self.identity:
                        os.rmdir(staging,dir_fd=parent_fd);os.fsync(parent_fd)
                except OSError: pass
            self.close();raise
        finally: os.close(parent_fd)


class _Directory:
    def __init__(self,path,*,create):
        self.path=_safe(path)
        if create: self.owner=_CreatedDirectory(self.path)
        else:
            if not self.path.is_dir(): raise ValueError('owned artifact directory absent')
            self.owner=cuda._capture_owned_output(self.path)
    @property
    def fd(self): return self.owner.fd
    def check(self,extra=0):
        if not self.owner.is_current(self.path): raise ValueError('owned artifact namespace changed')
        if shutil.disk_usage(self.fd).free<storage.RESERVE_BYTES+extra:
            raise OSError('formal artifacts require 15GiB reserve plus payload space')
    def close(self): self.owner.close()
    def read(self,name,limit):
        if not isinstance(name,str) or Path(name).name!=name or name in ('','.','..'):
            raise ValueError('artifact reads require one owned relative filename')
        if not self.owner.is_current(self.path): raise ValueError('owned artifact namespace changed')
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=self.fd)
        try:
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size>limit: raise ValueError('bounded regular artifact required')
            chunks=[];left=before.st_size
            while left:
                block=os.read(fd,min(left,65536))
                if not block: raise ValueError('short artifact read')
                chunks.append(block);left-=len(block)
            after=os.fstat(fd);current=os.stat(name,dir_fd=self.fd,follow_symlinks=False)
            if storage._stamp(self.path/name)!=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) or after!=current:
                raise ValueError('artifact changed during read')
            if not self.owner.is_current(self.path): raise ValueError('owned artifact namespace changed')
            return b''.join(chunks)
        finally: os.close(fd)


def _atomic(directory,name,writer,budget,*,replace):
    """Relative-dirfd publication and rollback; never mutate rebound neighbors."""
    directory.check(budget)
    token=uuid.uuid4().hex;temp='.'+name+'.'+token+'.incomplete';backup='.'+name+'.'+token+'.rollback'
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory.fd)
    own=os.fstat(fd);published=False;backed=False;committed=False
    try:
        class Writer:
            size=0
            def write(self,value):
                if self.size+len(value)>budget: raise OSError('artifact exceeded conservative budget')
                directory.check(len(value))
                n=os.write(fd,value)
                if n!=len(value): raise OSError('short artifact write')
                self.size+=n;directory.check();return n
            def flush(self): pass
            def tell(self): return self.size
        writer(Writer());os.fsync(fd);directory.check()
        if replace:
            try:
                old=os.stat(name,dir_fd=directory.fd,follow_symlinks=False)
                if not stat.S_ISREG(old.st_mode): raise ValueError('target must be an owned regular artifact')
                os.link(name,backup,src_dir_fd=directory.fd,dst_dir_fd=directory.fd,follow_symlinks=False);backed=True
            except FileNotFoundError: pass
            os.replace(temp,name,src_dir_fd=directory.fd,dst_dir_fd=directory.fd)
        else: os.link(temp,name,src_dir_fd=directory.fd,dst_dir_fd=directory.fd,follow_symlinks=False)
        published=True;os.fsync(directory.fd);directory.check();committed=True
    except BaseException:
        if published:
            try:
                actual=os.stat(name,dir_fd=directory.fd,follow_symlinks=False)
                if (actual.st_dev,actual.st_ino)==(own.st_dev,own.st_ino):
                    if backed: os.replace(backup,name,src_dir_fd=directory.fd,dst_dir_fd=directory.fd);backed=False
                    else: os.unlink(name,dir_fd=directory.fd)
                    os.fsync(directory.fd)
            except OSError: pass
        raise
    finally:
        os.close(fd)
        for path in (temp,backup):
            try: os.unlink(path,dir_fd=directory.fd)
            except FileNotFoundError: pass
        if committed:
            try: os.fsync(directory.fd)
            except OSError: pass  # checkpoint commit was already fsynced; optional temp cleanup only


def _write_json(directory,name,value,*,replace=False):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    _atomic(directory,name,lambda stream:stream.write(raw),len(raw)+4096,replace=replace)


def _journal_policy(identity,train,batch):
    _hex(identity)
    if any(type(v) is not int or v<=0 for v in (train,batch)): raise ValueError('invalid journal cohort/batch')
    attempts=100*math.ceil(train/batch)
    return dict(identity_sha256=identity,train_samples=train,batch_size=batch,max_attempts=attempts,
                total_budget_bytes=max(MIN_LOG_BUDGET,2*1024*attempts))


def _empty(): return dict(attempts=0,executions=0,skips=0,epochs=0,sequence=0,chain_sha256='0'*64,
                          metric_records=0,scale=None,growth_tracker=0,epoch_summaries=[])


def _record(record,state,policy):
    """Validate a flat bounded event without altering input or current state."""
    kind=record.get('kind');batches=math.ceil(policy['train_samples']/policy['batch_size'])
    if kind=='attempt':
        fields={'kind','epoch','batch','pl_step','optimizer_step_before','optimizer_step_after','optimizer_executed',
                'scale_before','scale_after','gradient_finite','gradient_abs_max','loss'}
        if set(record)!=fields: raise ValueError('attempt record fields differ')
        ints=('epoch','batch','pl_step','optimizer_step_before','optimizer_step_after')
        if any(type(record[k]) is not int for k in ints) or any(type(record[k]) is not bool for k in ('optimizer_executed','gradient_finite')):
            raise ValueError('attempt record types differ')
        executed=record['optimizer_executed']
        if (record['epoch']!=state['epochs'] or record['batch']!=state['attempts']%batches or
            record['pl_step']!=state['attempts']+1 or record['optimizer_step_before']!=state['executions'] or
            record['optimizer_step_after']!=state['executions']+int(executed) or state['attempts']>=policy['max_attempts']):
            raise ValueError('attempt counters/epoch/order differ')
        for k in ('scale_before','scale_after','loss'):
            if type(record[k]) is not float or not math.isfinite(record[k]) or (k!='loss' and record[k]<=0):
                raise ValueError('finite typed original loss/scaler observation required')
        maximum=record['gradient_abs_max']
        if record['scale_before']!=(65536. if state['scale'] is None else state['scale']):
            raise ValueError('journal scaler transition is not continuous from default state')
        if executed:
            tracker=state['growth_tracker']+1
            expected_scale=record['scale_before']*(2 if tracker==2000 else 1)
            if (record['gradient_finite'] is not True or type(maximum) is not float or not math.isfinite(maximum) or maximum<0 or
                record['scale_after']!=expected_scale):
                raise ValueError('finite actual optimizer execution evidence differs')
            state['growth_tracker']=0 if tracker==2000 else tracker
        elif record['gradient_finite'] is not False or maximum is not None or record['scale_after']!=record['scale_before']*.5:
            raise ValueError('overflow skip/backoff observation differs')
        else: state['growth_tracker']=0
        state['scale']=record['scale_after']
        state['attempts']+=1;state['executions']+=int(executed);state['skips']+=int(not executed)
    elif kind=='epoch':
        fields={'kind','epoch','full','samples','expected_samples','val_loss','batch_weighted_loss','coverage_sha256'}
        if set(record)!=fields or record['full'] is not True: raise ValueError('full validation event required')
        if (type(record['epoch']) is not int or record['epoch']!=state['epochs'] or state['epochs']>=100 or
            state['attempts']!=(state['epochs']+1)*batches or type(record['samples']) is not int or
            type(record['expected_samples']) is not int or record['samples']!=record['expected_samples'] or record['samples']<=0):
            raise ValueError('partial validation or epoch boundary')
        _hex(record['coverage_sha256'])
        if any(type(record[k]) is not float or not math.isfinite(record[k]) for k in ('val_loss','batch_weighted_loss')) or not math.isclose(record['val_loss'],record['batch_weighted_loss'],rel_tol=2e-6,abs_tol=1e-7):
            raise ValueError('finite batch-weighted original validation loss differs')
        state['epochs']+=1;state['epoch_summaries'].append({k:v for k,v in record.items() if k!='kind'})
    elif kind=='metric':
        name=record.get('name')
        if (set(record)!={'kind','pl_step','name','value'} or type(record['pl_step']) is not int or
            not 0<=record['pl_step']<=state['attempts'] or not isinstance(name,str) or not 0<len(name)<=100 or
            any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/.-' for c in name) or
            type(record['value']) is not float or not math.isfinite(record['value'])):
            raise ValueError('bounded scalar logger record required')
        state['metric_records']+=1
    else: raise ValueError('unknown journal record kind')


def _line(record,state,policy):
    _native(record)
    candidate=dict(sequence=state['sequence'],previous_sha256=state['chain_sha256'],record=record)
    raw=json.dumps(candidate,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    digest=hashlib.sha256(bytes.fromhex(state['chain_sha256'])+len(raw).to_bytes(8,'big')+raw).hexdigest()
    candidate['sha256']=digest
    line=(json.dumps(candidate,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(line)>MAX_RECORD_BYTES: raise ValueError('journal record exceeds 1KiB bound')
    _record(record,state,policy);state['sequence']+=1;state['chain_sha256']=digest
    return line


def _prefix(policy,state,segments):
    value=dict(format='formal_journal_prefix_v1',**policy,**copy.deepcopy(state),segments=copy.deepcopy(segments))
    value['sha256']=json_hash(value);return value


def verify_journal_carry(carry,prefix,*,identity_sha256,allow_prefix=False):
    try:
        if prefix['sha256']!=json_hash({k:v for k,v in prefix.items() if k!='sha256'}): raise ValueError('journal prefix seal differs')
        policy=_journal_policy(identity_sha256,prefix['train_samples'],prefix['batch_size'])
        if any(prefix[k]!=v for k,v in policy.items()) or set(carry)!={'identity_sha256','segments'} or carry['identity_sha256']!=identity_sha256:
            raise ValueError('journal identity/policy differs')
        blobs=carry['segments'];descriptors=prefix['segments'];state=_empty();total=0
        if len(blobs)<len(descriptors) or (not allow_prefix and len(blobs)!=len(descriptors)):
            raise ValueError('journal carry missing prefix or has uncommitted segments')
        if sum(len(x['data']) for x in blobs)>policy['total_budget_bytes']: raise ValueError('journal carry exceeds total budget')
        for index,(blob,descriptor) in enumerate(zip(blobs,descriptors)):
            name=f'events-{index:05d}.jsonl'
            if (set(blob)!={'name','data'} or set(descriptor)!={'name','bytes','sha256'} or blob['name']!=name or descriptor['name']!=name or
                type(blob['data']) is not bytes or not 0<descriptor['bytes']<=MAX_SEGMENT_BYTES or len(blob['data'])<descriptor['bytes'] or
                (not allow_prefix and len(blob['data'])!=descriptor['bytes'])):
                raise ValueError('journal segment schema differs')
            raw=blob['data'][:descriptor['bytes']];total+=len(raw)
            if hashlib.sha256(raw).hexdigest()!=descriptor['sha256'] or not raw.endswith(b'\n'): raise ValueError('journal segment hash/truncation')
            for line in raw.splitlines(keepends=True):
                if len(line)>MAX_RECORD_BYTES: raise ValueError('journal record exceeds 1KiB')
                parsed=json.loads(line);record=parsed['record'];expected=_line(record,state,policy)
                if line!=expected: raise ValueError('journal chain/order/typed canonical bytes differ')
        actual=_prefix(policy,state,descriptors)
        if json_hash(actual)!=json_hash(prefix): raise ValueError('stale journal prefix/counters differ')
    except (KeyError,TypeError,UnicodeError,json.JSONDecodeError,OverflowError) as exc:
        raise ValueError('invalid bounded journal carry') from exc


class FormalJournal:
    def __init__(self,directory,*,identity_sha256,train_samples,batch_size,create=True,prefix=None):
        self.policy=_journal_policy(identity_sha256,train_samples,batch_size);self.state=_empty();self.segments=[]
        self._segment_hash=hashlib.sha256();self._segment_stamps={}
        storage._space(_safe(directory),4096)
        self.directory=_Directory(directory,create=create)
        try:
            if create:
                _write_json(self.directory,'journal.json',dict(format='formal_journal_v1',**self.policy))
                if prefix is not None: raise ValueError('new journal must import explicit verified carry')
            else:
                header=json.loads(self.directory.read('journal.json',65536))
                if header!=dict(format='formal_journal_v1',**self.policy) or prefix is None: raise ValueError('journal header/prefix required')
                carry={'identity_sha256':identity_sha256,'segments':[]}
                for item in prefix['segments']:
                    carry['segments'].append(dict(name=item['name'],data=self.directory.read(item['name'],MAX_SEGMENT_BYTES)))
                self._adopt(carry,prefix)
                if any(len(blob['data'])!=item['bytes'] for blob,item in zip(carry['segments'],prefix['segments'])):
                    raise ValueError('journal contains uncommitted tail beyond checkpoint prefix')
        except BaseException: self.close();raise
    def _adopt(self,carry,prefix):
        verify_journal_carry(carry,prefix,identity_sha256=self.policy['identity_sha256'])
        if any(prefix[k]!=v for k,v in self.policy.items()): raise ValueError('journal migration policy differs')
        self.state={k:copy.deepcopy(prefix[k]) for k in _empty()};self.segments=copy.deepcopy(prefix['segments'])
        if self.segments:
            self._segment_hash=hashlib.sha256(carry['segments'][-1]['data'][:self.segments[-1]['bytes']])
            for item in self.segments:
                value=os.stat(item['name'],dir_fd=self.directory.fd,follow_symlinks=False)
                self._segment_stamps[item['name']]=(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    def prefix(self): return _prefix(self.policy,self.state,self.segments)
    def _append(self,value):
        self.directory.check();state=copy.deepcopy(self.state);line=_line(value,state,self.policy)
        if sum(s['bytes'] for s in self.segments)+len(line)>self.policy['total_budget_bytes']:
            raise OSError('journal total budget exceeded')
        new=not self.segments or self.segments[-1]['bytes']+len(line)>MAX_SEGMENT_BYTES
        index=len(self.segments) if new else len(self.segments)-1;name=f'events-{index:05d}.jsonl'
        self.directory.check(len(line));fd=os.open(name,os.O_RDWR|os.O_NOFOLLOW|(os.O_CREAT|os.O_EXCL if new else 0),0o600,dir_fd=self.directory.fd)
        item=os.fstat(fd);before=item.st_size
        stamp=(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
        if not new and stamp!=self._segment_stamps.get(name):
            os.close(fd);raise ValueError('journal segment changed since committed append')
        try:
            os.lseek(fd,0,os.SEEK_END)
            if os.write(fd,line)!=len(line): raise OSError('short journal append')
            self.directory.check();os.fsync(fd);os.fsync(self.directory.fd);self.directory.check()
            digest=hashlib.sha256() if new else self._segment_hash.copy();digest.update(line)
            descriptor=dict(name=name,bytes=before+len(line),sha256=digest.hexdigest())
            item=os.fstat(fd);written_stamp=(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
            named=os.stat(name,dir_fd=self.directory.fd,follow_symlinks=False)
            if item.st_size!=before+len(line) or (named.st_dev,named.st_ino)!=(item.st_dev,item.st_ino):
                raise ValueError('journal segment replaced or concurrently appended')
        except BaseException:
            os.ftruncate(fd,before)
            try: os.fsync(fd)
            except OSError: pass
            if new:
                try:
                    named=os.stat(name,dir_fd=self.directory.fd,follow_symlinks=False)
                    if (named.st_dev,named.st_ino)==(item.st_dev,item.st_ino): os.unlink(name,dir_fd=self.directory.fd)
                except FileNotFoundError: pass
            else:
                item=os.fstat(fd)
                self._segment_stamps[name]=(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
            raise
        finally: os.close(fd)
        if new:self.segments.append(descriptor)
        else:self.segments[-1]=descriptor
        self.state=state;self._segment_hash=digest;self._segment_stamps[name]=written_stamp
    def append_attempt(self,record): self._append(dict(record,kind='attempt'))
    def append_epoch(self,record): self._append(dict(record,kind='epoch'))
    def append_metric(self,record): self._append(dict(record,kind='metric'))
    def carry(self,prefix):
        blobs=[dict(name=x['name'],data=self.directory.read(x['name'],MAX_SEGMENT_BYTES)[:x['bytes']]) for x in prefix['segments']]
        result=dict(identity_sha256=self.policy['identity_sha256'],segments=blobs)
        verify_journal_carry(result,prefix,identity_sha256=self.policy['identity_sha256']);return result
    def import_carry(self,carry,prefix):
        if self.state['sequence'] or self.segments: raise ValueError('only a new empty journal may recover carried prefix')
        verify_journal_carry(carry,prefix,identity_sha256=self.policy['identity_sha256'])
        for blob,item in zip(carry['segments'],prefix['segments']):
            raw=blob['data'][:item['bytes']]
            _atomic(self.directory,item['name'],lambda stream,raw=raw:stream.write(raw),len(raw)+4096,replace=False)
        self._adopt(carry,prefix)
    def close(self): self.directory.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()


class OptimizerExecutionCounter:
    """Real post-step calls crossed with AdamW state; not a CUDA proof."""
    def __init__(self,optimizer):
        if type(optimizer).__name__!='AdamW': raise ValueError('original AdamW required')
        self.optimizer=optimizer;self.calls=0;self.pending=None
        self.hook=optimizer.register_step_post_hook(self._post)
    def _post(self,*_): self.calls+=1
    def begin(self):
        if self.pending is not None: raise ValueError('unfinished optimizer observation')
        self.pending=(self.calls,storage._optimizer_step(self.optimizer))
    def end(self):
        if self.pending is None: raise ValueError('optimizer observation missing before state')
        calls,before=self.pending;self.pending=None;after=storage._optimizer_step(self.optimizer);delta=self.calls-calls
        if delta not in (0,1) or after-before!=delta: raise ValueError('post-step call/state advance disagreement')
        return dict(executions=delta,step_before=before,step_after=after)
    def close(self): self.hook.remove()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()


def _seal(value): return storage.state_digest({k:v for k,v in value.items() if k not in ('formal_seal','formal_carry')})


def _nonrecursive(value):
    try:
        selected=value['formal']['selected_best']
        if selected is not None:
            child=selected['checkpoint']
            if child is value or child['formal']['selected_best'] is not None or 'formal_carry' in child:
                raise ValueError('selected best must be a nonrecursive checkpoint core')
    except (KeyError,TypeError,AttributeError) as exc:raise ValueError('invalid selected-best core') from exc


def _model_bytes(identity):
    sizes={'torch.float32':4,'torch.int64':8,'torch.int32':4,'torch.bool':1,'torch.float64':8,'torch.float16':2}
    try:return sum(math.prod(v['shape'])*sizes[v['dtype']] for v in identity['model_schema'].values())
    except KeyError as exc:raise ValueError('unsupported model schema dtype') from exc


def _journal_bound(identity):
    return _journal_policy(identity['sha256'],identity['counts']['train'],identity['effective']['batch_size'])


def _checkpoint_limit(identity): return 16*_model_bytes(identity)+8*_journal_bound(identity)['total_budget_bytes']+128*1024**2


def _payload_budget(value):
    carry=value.get('formal_carry',{})
    raw=sum(len(blob['data']) for blob in carry.get('segments',[]))
    return 2*storage._payload_bytes(value)+2*raw+storage.IO_OVERHEAD


def _expected_loops(epochs,train,val):
    def progress(n,keys): return {k:n for k in keys}
    four=('ready','started','processed','completed');two=('ready','completed');three=('ready','started','completed')
    epoch=dict(ready=epochs,started=epochs,processed=epochs,completed=epochs-1)
    loop={name:{} for name in cuda.LOOP_KEYS}
    loop['epoch_progress']={k:epoch.copy() for k in ('current','total')}
    loop['epoch_loop.state_dict']={'_batches_that_stepped':epochs*train}
    for key,n,total in (('epoch_loop.batch_progress',train,epochs*train),('epoch_loop.val_loop.batch_progress',val,val)):
        loop[key]=dict(current=progress(n,four),total=progress(total,four),is_last_batch=True)
    loop['epoch_loop.scheduler_progress']=dict(current=progress(1,two),total=progress(epochs,two))
    loop['epoch_loop.automatic_optimization.optim_progress']={'optimizer':{
        'step':dict(current=progress(train,two),total=progress(epochs*train,two)),
        'zero_grad':dict(current=progress(train,three),total=progress(epochs*train,three))}}
    loop['epoch_loop.manual_optimization.optim_step_progress']={k:progress(0,two) for k in ('current','total')}
    inactive={key:dict(batch_progress=dict(current=progress(0,four),total=progress(0,four),is_last_batch=False),state_dict={})
              for key in ('validate_loop','test_loop')}
    inactive['predict_loop']=dict(batch_progress=dict(current=progress(0,four),total=progress(0,four)),state_dict={})
    return dict(fit_loop=loop,**inactive)


def _optimizer_scheduler_state(value,identity,epochs):
    """Check original raw AdamW/Cosine state against this actual Torch runtime."""
    torch=storage._runtime();recipe=identity['recipe']
    reference=torch.optim.AdamW([torch.nn.Parameter(torch.empty(0))],lr=recipe['lr'],weight_decay=recipe['weight_decay'])
    cosine=torch.optim.lr_scheduler.CosineAnnealingLR(reference,T_max=100,eta_min=recipe['eta_min'])
    expected=reference.state_dict()['param_groups'][0];group=value['optimizer_states'][0]['param_groups'][0]
    expected['params']=list(range(len(identity['parameter_names'])))
    expected['lr']=recipe['eta_min']+(recipe['lr']-recipe['eta_min'])*(1+math.cos(math.pi*epochs/100))/2
    if (set(group)!=set(expected) or any(storage.state_digest(group[k])!=storage.state_digest(v)
            for k,v in expected.items() if k!='lr') or type(group['lr']) is not float or
            not math.isclose(group['lr'],expected['lr'],rel_tol=1e-12,abs_tol=1e-15)):
        raise ValueError('original raw AdamW parameter group state differs')
    schedulers=value['lr_schedulers'];expected_scheduler=cosine.state_dict()
    expected_scheduler.update(last_epoch=epochs,_step_count=epochs+1,_last_lr=[group['lr']])
    if len(schedulers)!=1 or storage.state_digest(schedulers[0])!=storage.state_digest(expected_scheduler):
        raise ValueError('original raw cosine epoch scheduler state differs')


def _score(value,expected):
    torch=storage._runtime()
    return (torch.is_tensor(value) and value.dtype==torch.float32 and value.ndim==0 and
            bool(torch.isfinite(value)) and float(value)==expected)


def _formal_rng_valid(state):
    """Restore-admit CPU streams in isolation; CUDA bytes remain structural only."""
    cuda._rng_valid(state);torch=storage._runtime()
    for value in (state['torch'],*state['generators'].values()):
        torch.Generator(device='cpu').set_state(value)


def _checkpoint(value,identity,role,carry,*,nested=False):
    torch=storage._runtime();validate_formal_identity(identity)
    try:
        meta=value['formal'];epochs=meta['completed_epochs'];prefix=meta['journal_prefix']
        if set(meta)!={'format','identity','completed_epochs','epoch_boundary','rng','journal_prefix','selected_best','role'}:
            raise ValueError('unbounded/unknown formal checkpoint metadata fields')
        if (meta['format']!=CHECKPOINT_FORMAT or json_hash(meta['identity'])!=json_hash(identity) or meta['role']!=role or
            meta['epoch_boundary'] is not True or type(epochs) is not int or not 1<=epochs<=100 or
            type(value['epoch']) is not int or type(value['global_step']) is not int or value['epoch']!=epochs-1):
            raise ValueError('formal full epoch/identity/role differs')
        if cuda._tensor_schema(value['state_dict'])!=identity['model_schema']: raise ValueError('model key/dtype/shape differs')
        schema=cuda.optimizer_schema(value['optimizer_states']);opt=value['optimizer_states'][0]
        if len(schema['parameters'])!=len(identity['parameter_names']): raise ValueError('optimizer parameters incomplete')
        for index,name in zip(schema['parameters'],identity['parameter_names']):
            if opt['state'][index]['step'].dtype!=torch.float32 or opt['state'][index]['step'].ndim!=0:
                raise ValueError('AdamW step tensor dtype/shape differs')
            for key in ('exp_avg','exp_avg_sq'):
                if list(opt['state'][index][key].shape)!=identity['model_schema'][name]['shape']:
                    raise ValueError('optimizer tensor shape differs from actual parameter schema')
        cuda.scaler_schema(value['MixedPrecision']);_formal_rng_valid(meta['rng'])
        if any(prefix[k]!=v for k,v in _journal_bound(identity).items()): raise ValueError('checkpoint journal policy differs')
        verify_journal_carry(carry,prefix,identity_sha256=identity['sha256'],allow_prefix=nested)
        if (value['MixedPrecision']['scale']!=prefix['scale'] or
            value['MixedPrecision']['_growth_tracker']!=prefix['growth_tracker']):
            raise ValueError('actual scaler state disagrees with journal transitions')
        if prefix['epochs']!=epochs or prefix['attempts']!=value['global_step']: raise ValueError('journal/PL epoch-step mismatch')
        if any(float(v['step'])!=prefix['executions'] for v in opt['state'].values()): raise ValueError('actual optimizer execution state differs')
        batches=[math.ceil(identity['counts'][r]/identity['effective']['batch_size']) for r in ('train','val')]
        if json_hash(value['loops'])!=json_hash(_expected_loops(epochs,*batches)):
            raise ValueError('incomplete or reset original Lightning loop/progress state')
        _optimizer_scheduler_state(value,identity,epochs)
        callbacks=value['callbacks'];p=callbacks['FormalProtocol']
        if set(p)!={'completed_epochs','attempts','executions','skips','journal_prefix','last_validation','selected_best'}:
            raise ValueError('bounded formal callback fields differ')
        if (any(type(p[k]) is not int or p[k]!=prefix[k] for k in ('attempts','executions','skips')) or
            type(p['completed_epochs']) is not int or p['completed_epochs']!=epochs or json_hash(p['journal_prefix'])!=json_hash(prefix)):
            raise ValueError('formal callback/journal counters differ')
        events=prefix['epoch_summaries'];last=events[-1]
        if json_hash(p['last_validation'])!=json_hash(last) or last['samples']!=identity['counts']['val']:
            raise ValueError('complete original validation evidence differs')
        minimum=min(e['val_loss'] for e in events);best_epoch=next(e['epoch'] for e in events if e['val_loss']==minimum)
        selection=dict(epoch=best_epoch,score=minimum,event_sha256=json_hash(events[best_epoch]))
        if json_hash(p['selected_best'])!=json_hash(selection): raise ValueError('best is not first minimum full val_loss')
        early=[v for k,v in callbacks.items() if k.startswith('EarlyStopping{')]
        monitors=[v for k,v in callbacks.items() if k.startswith('ModelCheckpoint{')]
        if len(early)!=1 or len(monitors)!=1 or len(callbacks)!=3: raise ValueError('original callback states missing/extra')
        e=early[0];wait=epochs-1-best_epoch
        if (set(e)!={'wait_count','stopped_epoch','best_score','patience'} or
            any(type(e[k]) is not int for k in ('wait_count','stopped_epoch','patience')) or e['patience']!=50 or wait>50 or
            e['wait_count']!=wait or e['stopped_epoch']!=(epochs-1 if wait==50 else 0) or not _score(e['best_score'],minimum)):
            raise ValueError('original EarlyStopping state differs')
        monitor=monitors[0]
        if set(monitor)!={'monitor','best_model_score','best_model_path','current_score','dirpath','best_k_models',
                          'kth_best_model_path','kth_value','last_model_path'} or monitor['monitor']!='val_loss':
            raise ValueError('original ModelCheckpoint state schema differs')
        parent=Path(monitor['dirpath']);best=Path(monitor['best_model_path'])
        if not parent.is_absolute() or '..' in parent.parts or best.parent!=parent or best.name=='last.ckpt':
            raise ValueError('selected best callback path escapes owned directory')
        for key in ('kth_best_model_path','last_model_path'):
            if monitor[key] and (Path(monitor[key]).parent!=parent or '..' in Path(monitor[key]).parts):
                raise ValueError('callback path outside original run')
        if (set(monitor['best_k_models'])!={str(best)} or not _score(monitor['best_model_score'],minimum) or
            not _score(monitor['best_k_models'][str(best)],minimum) or not _score(monitor['kth_value'],minimum) or
            not _score(monitor['current_score'],minimum)):
            raise ValueError('ModelCheckpoint full-validation best score differs')
        if value.get('formal_seal') is not None and value['formal_seal']!=_seal(value): raise ValueError('checkpoint typed state integrity differs')
        if role=='best':
            if meta['selected_best'] is not None or best_epoch!=epochs-1: raise ValueError('best snapshot recursive or not newly selected')
        elif role=='last':
            selected=meta['selected_best']
            if not selected or set(selected)!={'state_sha256','checkpoint'}: raise ValueError('last requires carried best, never last-weight fallback')
            best_core=selected['checkpoint']
            if 'formal_carry' in best_core or selected['state_sha256']!=_seal(best_core): raise ValueError('best must be nonrecursive core with bound journal prefix')
            _checkpoint(best_core,identity,'best',carry,nested=True)
            if best_core['formal']['completed_epochs']!=best_epoch+1: raise ValueError('carried best epoch differs from selector')
        else: raise ValueError('unknown formal checkpoint role')
    except (KeyError,TypeError,AttributeError,IndexError,RuntimeError) as exc:
        raise ValueError('truncated formal full-state schema') from exc


def _read(path,identity,role):
    torch=storage._runtime();validate_formal_identity(identity);path=_safe(path)
    with cuda._capture_owned_output(path.parent) as namespace:
        before=storage._stamp(path)
        if before[2]>_checkpoint_limit(identity): raise ValueError('checkpoint exceeds bounded state budget')
        try:value=torch.load(path,map_location='cpu',weights_only=False)
        except Exception as exc:raise ValueError('formal checkpoint truncated or unreadable') from exc
        try:
            _nonrecursive(value)
            if value.get('formal_seal')!=_seal(value): raise ValueError('formal typed checkpoint seal differs')
            carry=value['formal_carry'];_checkpoint(value,identity,role,carry)
        except (KeyError,TypeError,AttributeError) as exc:raise ValueError('incomplete formal checkpoint schema') from exc
        if carry['identity_sha256']!=identity['sha256']: raise ValueError('journal binding differs')
        validate_formal_identity(identity)
        # Point-in-time admission: check the namespace after all file/path work.
        if storage._stamp(_safe(path))!=before or not namespace.is_current(path.parent):
            raise ValueError('checkpoint or namespace changed during admission')
        return value


def read_formal_last(path,identity): return _read(path,identity,'last')


def read_selected_best(path,identity):
    value=read_formal_last(path,identity)
    return copy.deepcopy(value['formal']['selected_best']['checkpoint'])


class FormalCheckpointIO(CheckpointIO):
    def __init__(self,directory,identity,*,journal,resume=None,admitted=None):
        validate_formal_identity(identity);self.identity=copy.deepcopy(identity);self.journal=journal
        if journal.policy!=_journal_bound(identity): raise ValueError('journal and formal identity differ')
        storage._space(_safe(directory),_checkpoint_limit(identity))
        self.directory=_Directory(directory,create=True);self.resume=_safe(resume) if resume else None
        self.admitted=admitted;self.resume_sha=file_sha256(self.resume) if resume else None;self.owned={}
    def _name(self,path):
        path=_safe(path)
        if path.parent!=self.directory.path or path.suffix!='.ckpt': raise ValueError('checkpoint path outside owned output')
        return path.name
    def save_checkpoint(self,checkpoint,path,storage_options=None):
        if storage_options is not None: raise ValueError('remote formal checkpoint IO forbidden')
        validate_formal_identity(self.identity);name=self._name(path);value=copy.deepcopy(checkpoint)
        _nonrecursive(value)
        role='last' if name=='last.ckpt' else 'best';meta=value['formal'];meta['role']=role
        if json_hash(meta['identity'])!=json_hash(self.identity) or json_hash(meta['journal_prefix'])!=json_hash(self.journal.prefix()):
            raise ValueError('checkpoint identity/journal prefix differs')
        carry=self.journal.carry(meta['journal_prefix']);value['formal_carry']=carry
        if role=='last':
            monitors=[s for n,s in value['callbacks'].items() if n.startswith('ModelCheckpoint{')]
            if len(monitors)!=1: raise ValueError('selected best callback missing')
            best_path=Path(monitors[0]['best_model_path']);best_name=self._name(best_path)
            best=_read(self.directory.path/best_name,self.identity,'best');best.pop('formal_carry')
            meta['selected_best']=dict(state_sha256=_seal(best),checkpoint=best)
        else:meta['selected_best']=None
        _checkpoint(value,self.identity,role,carry);value['formal_seal']=_seal(value)
        budget=_payload_budget(value)
        current=sum(os.stat(n,dir_fd=self.directory.fd,follow_symlinks=False).st_size for n in os.listdir(self.directory.fd))
        if current+budget>_checkpoint_limit(self.identity): raise OSError('formal checkpoint directory budget exceeded')
        if name in self.owned:
            prior=os.stat(name,dir_fd=self.directory.fd,follow_symlinks=False)
            if (prior.st_dev,prior.st_ino)!=self.owned[name]: raise ValueError('checkpoint no longer owned')
        _atomic(self.directory,name,lambda stream:storage._runtime().save(value,stream),budget,replace=name in self.owned)
        item=os.stat(name,dir_fd=self.directory.fd,follow_symlinks=False);self.owned[name]=(item.st_dev,item.st_ino)
    def load_checkpoint(self,path,map_location=None):
        path=_safe(path)
        if self.resume is None or path!=self.resume or self.admitted is None: raise ValueError('unadmitted formal last resume')
        if file_sha256(path)!=self.resume_sha: raise ValueError('formal resume file changed after admission')
        value=read_formal_last(path,self.identity)
        if value['formal_seal']!=self.admitted['formal_seal']: raise ValueError('formal resume state changed')
        self.journal.import_carry(value['formal_carry'],value['formal']['journal_prefix'])
        # Recover carried selected best into only this NEW owned directory.
        selected=copy.deepcopy(value['formal']['selected_best']['checkpoint']);best_name='recovered-best.ckpt'
        prefix=selected['formal']['journal_prefix']
        selected['formal_carry']=dict(identity_sha256=self.identity['sha256'],segments=[
            dict(name=blob['name'],data=blob['data'][:item['bytes']]) for blob,item in zip(value['formal_carry']['segments'],prefix['segments'])])
        self._rebase(selected,best_name);selected['formal_seal']=_seal(selected)
        budget=_payload_budget(selected)
        _atomic(self.directory,best_name,lambda stream:storage._runtime().save(selected,stream),budget,replace=False)
        item=os.stat(best_name,dir_fd=self.directory.fd,follow_symlinks=False);self.owned[best_name]=(item.st_dev,item.st_ino)
        value=copy.deepcopy(value);self._rebase(value,best_name)
        # Only known callback paths adapt; the returned state's new seal records
        # that adaptation. Admission/original file bytes retain their old seal.
        value['formal_seal']=_seal(value)
        return value
    def _rebase(self,value,best_name):
        for key,state in value['callbacks'].items():
            if key.startswith('ModelCheckpoint{'):
                path=str(self.directory.path/best_name);state.update(dirpath=str(self.directory.path),best_model_path=path,
                    best_k_models={path:state['best_model_score']},kth_best_model_path=path,last_model_path=str(self.directory.path/'last.ckpt'))
    def remove_checkpoint(self,path):
        name=self._name(path);self.directory.check()
        if name not in self.owned: raise ValueError('cannot remove checkpoint not created by this IO')
        item=os.stat(name,dir_fd=self.directory.fd,follow_symlinks=False)
        if (item.st_dev,item.st_ino)!=self.owned[name]: raise ValueError('owned checkpoint inode changed')
        os.unlink(name,dir_fd=self.directory.fd);os.fsync(self.directory.fd);del self.owned[name]
    def close(self):self.directory.close()
