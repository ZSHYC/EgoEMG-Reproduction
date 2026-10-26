"""External bounded legacy worker. Formal evaluation is deliberately unavailable.

Only a fresh child may import the admitted old runtime. Completed checkpoints are
trusted local pickle, admitted by their immutable whole-file SHA before unpickle.
CPU helpers and synthetic tests never qualify the original CUDA bf16 experiment.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import ctypes
import errno
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import stat
import subprocess
import sys
import uuid

if __package__ in (None,''):  # Private child bootstrap; this is not an inference CLI.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from tools.legacy_derived import preflight, reducers
else:
    from . import preflight, reducers

FORMAT='legacy_derived_bounded_worker_v1'
COHORT_FORMAT='legacy_derived_fixed_engineering_cohort_v1'
RESERVE=15*1024**3
MAX_OUTPUT=32*1024**2
FACTORY_TEMP_BUDGET=256*1024**2  # Original fusion writes one temporary vision state.
TASK_A={'__init__.py':'153e29991bb492ff673bfcbecb3b452de0a6dfc07c068ef56b59ce1eca2471f9',
    'preflight.py':'f5f7aee91eebfb7fcf77cb3a403c0808149d732a6f37084930f112acad2301d8',
    'reducers.py':'4e8792d42cc8041440acb3d7737be2bd1bd69e16138c51e969000f701ba7b818',
    'cli.py':'582851717efb4d6b6994f39a19c005c9bd1634333d1c2ed8341c05321aaef42e'}
CHECKPOINTS={
 's':{'last.pt':'c3e1f844bc47cfee3a2422ce924fbf08a5a81afb46605e9824d47d29dd26e75d','best.pt':'be50bf2a1eb262c49d68fb17bbb4f56043ed14c100f68ebab326ca97d601b365'},
 'm':{'last.pt':'ed43852218f9cc81b757386d3955f1966fce6ef528a549eb6546a64cae7c095b','best.pt':'4a0ed2020a00483b52b6dea13071ff802bcab367d841e8545434eb90dc663321'},
 'rn18':{'last.pt':'e9bf7c7fa2abee7997c2c7c50f750137ded4ae8dad1cba998f52392516db5fc0','best.pt':'85acf8d823b95bee2754022916978d928c4d5d270e49edfe58844519adc924b7'},
 'fusion':{'last.pt':'a305c998dcb57e52b9289e141b1d6c5d882e33a7ecc1f04fee0dd321682a43b2','best.pt':'874adb63ecef78d969e5705924fa3ec7170949985aadeb8eba5b9f85bba81ac0'}}
CROP_SHA='a72f06683858e1d0101ad4a7fa1ab10c59980baf95a65b6b9f38957e7ee79249'
WORKER_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _tools():
    root=Path(__file__).resolve().parent
    values={name:preflight._hash(preflight._read_small(root/name)) for name in TASK_A}
    if values!=TASK_A or preflight._hash(preflight._read_small(Path(__file__)))!=WORKER_SHA:
        raise ValueError('executed derived tool source changed')
    return dict(task_a=values,worker_sha256=WORKER_SHA)


def _whole_sha(path):
    path=preflight._safe(path);before=preflight._file_stamp(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        if preflight._stamp(os.fstat(stream.fileno()))!=before:raise ValueError('input replaced before hashing')
        for value in iter(lambda:stream.read(8*1024**2),b''):digest.update(value)
        if preflight._stamp(os.fstat(stream.fileno()))!=before:raise ValueError('input changed during hashing')
    if preflight._file_stamp(path)!=before:raise ValueError('input replaced after hashing')
    return digest.hexdigest()


class InputSnapshot:
    """Cheap safe namespace + full stat witnesses; not a whole-payload hash."""
    def __init__(self,paths):
        self.values={preflight._safe(p):preflight._file_stamp(p) for p in paths}
    def check(self):
        if any(preflight._file_stamp(path)!=before for path,before in self.values.items()):
            raise ValueError('input changed after admission')


class NamespaceSnapshot:
    """Bind directories as well as files through the parent's final publication."""
    def __init__(self,roots):
        self.trees={preflight._safe(root):_tree_files(root) for root in roots}
        self.files=InputSnapshot([path for values in self.trees.values() for path in values])
        self.directories={path:preflight._stamp(path.stat()) for root in self.trees
                          for path in [root,*sorted(p for p in root.rglob('*') if p.is_dir())]}
    def check(self):
        self.files.check()
        if (any(_tree_files(root)!=values for root,values in self.trees.items()) or
            any(preflight._stamp(preflight._safe(path).stat())!=before for path,before in self.directories.items())):
            raise ValueError('input namespace changed after admission')


def _official_inputs(root):
    root=preflight._safe(root)
    return sorted({*root.glob('egoemg/**/*.py'),*root.glob('config/**/*.yaml'),
                   *root.glob('egoemg/UmeTrack/**/*.json')})


class RunInputGuard:
    """The SAME pre-admission witnesses survive the final output commit."""
    def __init__(self,spec):
        legacy=preflight._safe(spec['legacy_root']);run=preflight._safe(spec['run_spec']['run'])
        self.official=preflight._safe(spec['upstream_root'])
        self.official_paths=_official_inputs(self.official)
        tool=Path(__file__).resolve().parent
        paths=[legacy/name for name in preflight.LEGACY_FILES]
        paths += [run/name for name in preflight.SMALL_FILES|{'last.pt','best.pt','progress.jsonl'}]
        paths += [tool/name for name in TASK_A]+[Path(__file__).resolve(),*self.official_paths]
        self.files=InputSnapshot(paths)
        self.payload=NamespaceSnapshot([spec['data'],spec['crops']])
        self.check()
    def add(self,path):
        path=preflight._safe(path);value=preflight._file_stamp(path)
        if self.files.values.setdefault(path,value)!=value:
            raise ValueError('guarded input changed before reuse')
        return value
    def check(self):
        self.files.check()
        if _official_inputs(self.official)!=self.official_paths:
            raise ValueError('official input source namespace changed')
        self.payload.check()
        self.files.check()


def _tree_files(root):
    root=preflight._safe(root)
    if not root.is_dir():raise ValueError('required input namespace unavailable')
    values=[]
    for path in sorted(root.rglob('*')):
        preflight._safe(path)
        if path.is_file():values.append(path)
        elif not path.is_dir():raise ValueError('unsupported input namespace member')
    return values


def _contract(spec,cohort,bounded):
    if bounded is not True:raise ValueError('formal derived execution is not enabled in this stage')
    if (not isinstance(spec,dict) or set(spec)!={'format','run_spec','legacy_root','upstream_root','data','crops','output','device'} or
        spec['format']!=FORMAT or spec['device']!='cuda:0'):
        raise ValueError('worker spec/precision/device override is not permitted')
    run=spec['run_spec']
    if not isinstance(run,dict) or set(run)!={'model','run','files_sha256'} or run['model'] not in CHECKPOINTS:
        raise ValueError('unknown selected model or run spec')
    expected=dict(format=COHORT_FORMAT,partition_sha256=preflight.REFERENCE['partition_sha256'],
        partition_raw_sha256=preflight.RAW_FILES[run['model']]['partitions.json'],window_ordinals=[0,1])
    if preflight._canonical(cohort)!=preflight._canonical(expected):
        raise ValueError('only the fixed preselected engineering windows are allowed')
    for name in ('legacy_root','upstream_root','data','crops','output'):preflight._safe(spec[name])
    output=preflight._safe(spec['output'])
    if any(output.is_relative_to(preflight._safe(root)) for root in
           [spec[name] for name in ('legacy_root','upstream_root','data','crops')]+[run['run']]):
        raise ValueError('attempt output must be outside every frozen input namespace')
    return run


def _admit_inputs(spec,cohort,*,bounded):
    run=_contract(spec,cohort,bounded);_tools()
    source=preflight.validate_source_manifest(spec['legacy_root'],spec['upstream_root'])
    root=preflight._safe(spec['legacy_root']);directory=preflight._safe(run['run'])
    paths=[root/name for name in preflight.LEGACY_FILES]+[directory/name for name in preflight.SMALL_FILES]
    paths += [directory/name for name in ('progress.jsonl','last.pt','best.pt')]
    snapshot=InputSnapshot(paths)
    metadata=preflight._run(run,root)
    for name,expected in CHECKPOINTS[run['model']].items():
        if _whole_sha(directory/name)!=expected:raise ValueError('locked checkpoint whole SHA differs')
    snapshot.check();_tools()
    if preflight.validate_source_manifest(spec['legacy_root'],spec['upstream_root'])!=source:
        raise ValueError('source changed before tensor admission')
    return dict(format=FORMAT,scope='bounded_engineering',benchmark_eligible=False,cohort=cohort,
        model=run['model'],source=source,tool_source=_tools(),checkpoint_sha256=CHECKPOINTS[run['model']]['last.pt'],
        selected_file_sha256=CHECKPOINTS[run['model']]['best.pt'],recipe_raw_sha256=metadata['recipe_sha256'],
        effective_config_sha256=metadata['config_sha256'],summary_sha256=metadata['summary_sha256'])


def _load_checkpoint(path,expected,torch):
    """Hash the same opened descriptor before unpickling any trusted tensor."""
    path=preflight._safe(path);before=preflight._file_stamp(path)
    with path.open('rb') as stream:
        if preflight._stamp(os.fstat(stream.fileno()))!=before:raise ValueError('checkpoint replaced before read')
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:raise ValueError('authoritative checkpoint whole SHA differs before tensor read')
        if preflight._stamp(os.fstat(stream.fileno()))!=before:raise ValueError('checkpoint changed before tensor read')
        stream.seek(0);value=torch.load(stream,map_location='cpu',weights_only=False)
        if preflight._stamp(os.fstat(stream.fileno()))!=before:raise ValueError('checkpoint changed during tensor read')
    if preflight._file_stamp(path)!=before:raise ValueError('checkpoint replaced during tensor admission')
    return value


def typed_state_sha256(state):
    import torch
    if not isinstance(state,dict) or not state:raise ValueError('empty typed model state')
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if not isinstance(name,str) or not torch.is_tensor(value) or value.layout!=torch.strided:
            raise ValueError('model state requires named dense tensors')
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite selected model state')
        for part in (name.encode(),str(value.dtype).encode(),json.dumps(list(value.shape)).encode(),
                     value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()):
            digest.update(len(part).to_bytes(8,'big'));digest.update(part)
    return digest.hexdigest()


def strict_load_state(model,state):
    import torch
    expected=model.state_dict()
    if not isinstance(state,dict) or set(expected)!=set(state):raise ValueError('selected model tensor keys differ')
    for name,value in expected.items():
        other=state[name]
        if not torch.is_tensor(other) or other.shape!=value.shape or other.dtype!=value.dtype:
            raise ValueError('selected model tensor shape/dtype differs')
    token=typed_state_sha256(state)  # Refuse nonfinite state before changing the model.
    model.load_state_dict(state,strict=True)
    if typed_state_sha256(model.state_dict())!=token:raise ValueError('strict restored model tensor bytes differ')
    return token


def pre_builder_state_sha256(state):
    """Pre-dtype gate, not a claim to know the original full shape template.

    The unmodified visual builder internally strict-loads resume_state. Reject
    castable dtypes before that call; its own strict key/shape checks and our
    subsequent complete schema/digest check remain mandatory before inference.
    """
    import torch
    token=typed_state_sha256(state)
    for name,value in state.items():
        if value.dtype==torch.float32:continue
        if (name=='num_batches_tracked' or name.endswith('.num_batches_tracked')) and value.dtype==torch.int64 and value.ndim==0:
            continue
        raise ValueError('unsupported tensor dtype before original builder load')
    return token


@contextmanager
def offline_transport():
    """Refuse Python network transport, including an accidental weight download."""
    original={name:getattr(socket,name) for name in ('create_connection','getaddrinfo')}
    connect,connect_ex=socket.socket.connect,socket.socket.connect_ex
    def denied(*_,**__):raise RuntimeError('legacy worker is offline; network transport refused')
    def local_connect(stream,*args,**kwargs):
        if stream.family!=socket.AF_UNIX:return denied()
        return connect(stream,*args,**kwargs)
    def local_connect_ex(stream,*args,**kwargs):
        if stream.family!=socket.AF_UNIX:return denied()
        return connect_ex(stream,*args,**kwargs)
    socket.create_connection=socket.getaddrinfo=denied
    socket.socket.connect,socket.socket.connect_ex=local_connect,local_connect_ex
    try:yield
    finally:
        for name,value in original.items():setattr(socket,name,value)
        socket.socket.connect,socket.socket.connect_ex=connect,connect_ex


def _runtime_origins(legacy,official):
    legacy,official=preflight._safe(legacy),preflight._safe(official)
    for name,module in tuple(sys.modules.items()):
        if name=='egoemg_repro' or name.startswith('egoemg_repro.'):
            path=Path(getattr(module,'__file__','')).absolute()
            try:relative=str(path.relative_to(legacy))
            except ValueError:raise ValueError('loaded reproduction namespace is not the old checkout') from None
            if relative not in preflight.LEGACY_FILES or _whole_sha(path)!=preflight.LEGACY_FILES[relative]:
                raise ValueError('loaded reproduction file is not in the admitted old manifest')
        elif name=='egoemg' or name.startswith('egoemg.'):
            path=Path(getattr(module,'__file__','')).absolute()
            if not path.is_relative_to(official/'egoemg'):raise ValueError('loaded official namespace differs')
            preflight._safe(path)


def _source_check(legacy,official):
    _tools()
    for name,expected in preflight.LEGACY_FILES.items():
        if _whole_sha(Path(legacy)/name)!=expected:raise ValueError('old runtime source drifted')
    if preflight._upstream(official)!={'revision':preflight.UPSTREAM_COMMIT,'sha256':preflight.UPSTREAM_SHA}:
        raise ValueError('official source drifted')
    _runtime_origins(legacy,official)


def _load_runtime(legacy,official):
    preflight.validate_source_manifest(legacy,official)  # Includes preimport rejection.
    legacy,official=preflight._safe(legacy),preflight._safe(official)
    def external(entry):
        if not entry:return False
        path=Path(entry).absolute()
        return not ((path/'egoemg_repro').exists() or (path/'egoemg').exists())
    sys.path[:]=[str(legacy),str(official)]+[entry for entry in sys.path if external(entry)]
    runtime=SimpleRuntime()
    for name in ('train','audit','report','data','metrics','crops'):
        setattr(runtime,name,importlib.import_module('egoemg_repro.'+name))
    _source_check(legacy,official)
    return runtime


class SimpleRuntime:
    pass


def _precision_guard(torch,config,device):
    if device!='cuda:0' or config.get('precision')!='bfloat16':raise ValueError('original CUDA bf16 precision is required')
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA is required; no CPU precision substitution')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if torch.cuda.device_count()!=1 or not visible.startswith('GPU-') or ',' in visible:
        raise ValueError('one actually root-reserved GPU UUID must be visible')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('actual device lacks original bf16 precision')


def numeric_environment(torch,environment,*,visual):
    """Compare only actually recorded numeric versions; report absent evidence."""
    import numpy as np
    observed=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                  cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(key not in environment or environment[key]!=value for key,value in observed.items()):
        raise ValueError('original numeric runtime version differs')
    recorded=environment.get('torchvision') or environment.get('initialization',{}).get(
        'official_builder',{}).get('pretrained_weights',{}).get('torchvision_version')
    if visual:
        import torchvision
        observed['torchvision']=str(torchvision.__version__)
        if recorded is not None and recorded!=observed['torchvision']:
            raise ValueError('original recorded vision runtime differs')
    return dict(observed=observed,torchvision_recorded=recorded is not None,
        limitation='unrecorded historical torchvision version is not inferred from another model')


def _free(path):return shutil.disk_usage(path).free


def _rename_noreplace(directory,source,destination):
    """Linux only; an unsupported primitive never falls back to overwrite."""
    if sys.platform!='linux':raise RuntimeError('Linux atomic NOREPLACE publication is required')
    libc=ctypes.CDLL(None,use_errno=True)
    rename=getattr(libc,'renameat2',None)
    if rename is None:raise RuntimeError('atomic renameat2 NOREPLACE is unsupported')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(directory,os.fsencode(source),directory,os.fsencode(destination),1):
        code=ctypes.get_errno()
        if code in (errno.ENOSYS,errno.EINVAL,errno.EOPNOTSUPP):
            raise RuntimeError('atomic renameat2 NOREPLACE is unsupported')
        raise OSError(code,os.strerror(code),destination)


class Attempt:
    """Fd-owned private staging, then atomic NOREPLACE public-name publication.

    The caller supplies a trusted parent. Its private random staging namespace
    must not be modified during mkdir/open acquisition: Linux mkdir returns no
    fd. Ownership comes from that held staging fd, never from a public-name open.
    """
    def __init__(self,output,identity):
        self.path=preflight._safe(output);self.fd=None;self.lock=None;self.marker_fd=None;self.owned={}
        if self.path.exists():raise FileExistsError('attempt output already exists')
        if sys.platform!='linux':raise RuntimeError('Linux atomic NOREPLACE publication is required')
        parent=self.path.parent
        if _free(parent)<RESERVE+MAX_OUTPUT+FACTORY_TEMP_BUDGET:raise OSError('15GiB reserve plus bounded output/factory temporary required')
        parent_fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        parent_value=os.fstat(parent_fd);parent_id=(parent_value.st_dev,parent_value.st_ino)
        staging='.legacy-derived-stage-'+uuid.uuid4().hex
        published=False
        try:
            os.mkdir(staging,mode=0o700,dir_fd=parent_fd)
            self.fd=os.open(staging,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
            value=os.fstat(self.fd);self.identity=(value.st_dev,value.st_ino)
            current=preflight._safe(parent).stat()
            staged=os.stat(staging,dir_fd=parent_fd,follow_symlinks=False)
            if ((current.st_dev,current.st_ino)!=parent_id or
                    (staged.st_dev,staged.st_ino)!=self.identity or not stat.S_ISDIR(staged.st_mode)):
                raise ValueError('private staging directory identity changed')
            _rename_noreplace(parent_fd,staging,self.path.name);published=True
            os.fsync(parent_fd)
            # This public-name open only verifies the already captured identity.
            # Keep it real so a rebind here is rejected without adopting a neighbor.
            public_fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                current=os.fstat(public_fd)
                if (current.st_dev,current.st_ino)!=self.identity or not self.current():
                    raise ValueError('published owned directory identity changed')
            finally:os.close(public_fd)
            self.lock=os.open('attempt.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600,dir_fd=self.fd)
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.write('attempt.json',self.json(dict(format=FORMAT,scope='bounded_engineering',benchmark_eligible=False,
                attempt_uuid=uuid.uuid4().hex,pid=os.getpid(),identity=identity)))
        except BaseException:
            if not published and self.fd is not None:
                try:
                    current=os.stat(staging,dir_fd=parent_fd,follow_symlinks=False)
                    if (current.st_dev,current.st_ino)==self.identity:
                        os.rmdir(staging,dir_fd=parent_fd);os.fsync(parent_fd)
                except OSError:pass
            self.close();raise
        finally:os.close(parent_fd)
    @staticmethod
    def json(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    def current(self):
        try:
            value=preflight._safe(self.path).stat()
            return self.fd is not None and stat.S_ISDIR(value.st_mode) and (value.st_dev,value.st_ino)==self.identity
        except (OSError,ValueError):return False
    def write(self,name,payload):
        if name not in ('attempt.json','request.json','result.json','status.json','centers.npz') or not self.current():
            raise ValueError('publication path is not owned by this attempt')
        if not isinstance(payload,bytes) or len(payload)>MAX_OUTPUT:raise ValueError('bounded private payload exceeded')
        if _free(self.fd)<RESERVE+len(payload)+4096:raise OSError('15GiB reserve prevents publication')
        temporary='.'+name+'.'+uuid.uuid4().hex+'.incomplete'
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
        marker_fd=None
        try:
            try:
                for start in range(0,len(payload),65536):
                    chunk=payload[start:start+65536]
                    if _free(self.fd)<RESERVE+len(chunk):raise OSError('15GiB reserve prevents write')
                    if os.write(descriptor,chunk)!=len(chunk):raise OSError('short bounded artifact write')
                os.fsync(descriptor)
                value=os.fstat(descriptor);identity=(value.st_dev,value.st_ino)
                # Holding the marker inode prevents unlink/recreate inode reuse
                # from making a foreign replacement look owned during rollback.
                if name=='status.json':marker_fd=os.dup(descriptor)
            finally:os.close(descriptor)
            if not self.current():raise ValueError('attempt directory changed before publication')
            if _free(self.fd)<RESERVE:raise OSError('15GiB reserve prevents commit')
            # An attempt owns these fresh names; never replace its immutable start/request.
            if name in ('attempt.json','request.json','status.json'):
                os.link(temporary,name,src_dir_fd=self.fd,dst_dir_fd=self.fd,follow_symlinks=False)
            else:os.replace(temporary,name,src_dir_fd=self.fd,dst_dir_fd=self.fd)
            self.owned[name]=identity
            if marker_fd is not None:
                if self.marker_fd is not None:os.close(self.marker_fd)
                self.marker_fd=marker_fd;marker_fd=None
            os.fsync(self.fd)
        finally:
            if marker_fd is not None:os.close(marker_fd)
            try:os.unlink(temporary,dir_fd=self.fd)
            except FileNotFoundError:pass
    def finish(self,result,*,guard=None):
        if result.get('benchmark_eligible') is not False:raise ValueError('engineering cannot publish a benchmark')
        if guard is not None:guard.check()
        self.write('result.json',self.json(result))
        if guard is not None:guard.check()
        self.write('status.json',self.json(dict(state='engineering_finished',benchmark_eligible=False,
            result_sha256=preflight._hash(self.json(result)))))
        if not self.current():raise ValueError('owned output directory changed during completion')
        if guard is not None:guard.check()
    def fail(self):
        if self.fd is None:return
        try:
            value=os.stat('status.json',dir_fd=self.fd,follow_symlinks=False)
            if (not stat.S_ISREG(value.st_mode) or
                    self.owned.get('status.json')!=(value.st_dev,value.st_ino)):
                return  # A foreign replacement is not this attempt's marker.
            os.unlink('status.json',dir_fd=self.fd);os.fsync(self.fd)
        except FileNotFoundError:pass
        except OSError:return
        if self.marker_fd is not None:os.close(self.marker_fd);self.marker_fd=None
        self.owned.pop('status.json',None)
        if not self.current():return
        try:self.write('status.json',self.json(dict(state='failed',benchmark_eligible=False)))
        except (OSError,ValueError):pass
    def close(self):
        if self.marker_fd is not None:os.close(self.marker_fd);self.marker_fd=None
        if self.lock is not None:os.close(self.lock);self.lock=None
        if self.fd is not None:os.close(self.fd);self.fd=None
    def __enter__(self):return self
    def __exit__(self,kind,*_):
        if kind is not None:self.fail()
        self.close()


def _forward_chunk(runtime,model,batch,config,device):
    """The original align path; synthetic CPU tests of this helper are not a pass."""
    import torch
    with torch.no_grad(),runtime.train.precision_context(device,config):
        prediction,target,valid=runtime.train.aligned_training_batch(model,batch,config)
    if prediction.shape!=target.shape or valid.shape!=target.shape or valid.dtype!=torch.bool:
        raise ValueError('original aligned prediction/target/mask schema differs')
    if target.dtype!=torch.float32 or prediction.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('original bf16 prediction/float32 target schema differs')
    arrays=tuple(v.float().cpu().numpy() if v.dtype!=torch.bool else v.cpu().numpy() for v in (prediction,target,valid))
    if arrays[0].ndim!=3 or arrays[0].shape[1]!=22 or not all(__import__('numpy').isfinite(v[arrays[2]]).all() for v in arrays[:2]):
        raise ValueError('original aligned valid angles are nonfinite')
    return arrays


def _user_key(value):
    # Original audit conversion, memory-only; never persist or print the source ID.
    return str(value.item() if hasattr(value,'item') else value)


def result_identity(admitted):
    """Only hashes/selected-model identity, no private window/user ordinal list."""
    return dict(admission_sha256=preflight._canonical(admitted),model=admitted['model'],
        identities=dict(admitted['source'],tool_source_sha256=preflight._canonical(admitted['tool_source']),
            cohort_selection_sha256=preflight._canonical(admitted['cohort']),
            checkpoint_sha256=admitted['checkpoint_sha256'],selected_file_sha256=admitted['selected_file_sha256'],
            recipe_raw_sha256=admitted['recipe_raw_sha256'],effective_config_sha256=admitted['effective_config_sha256'],
            summary_sha256=admitted['summary_sha256']))


def _infer(runtime,model,evidence,spec,cohort,snapshot):
    import numpy as np
    import torch
    train=runtime.train;config=evidence['config'];visual=train.model_kind(config)!='emg'
    full=train.build_datasets(spec['data'],evidence['manifest'],evidence['norm'],config,crops=spec['crops'])
    if not all(len(v) for v in full.values()):raise ValueError('original full role population is empty')
    dataset=full['test'];windows=dataset.windows
    if len(windows)!=1396:raise ValueError('frozen test window population differs')
    # Actual original dataset, immutable first two windows; no replacement/search.
    selected=torch.utils.data.Subset(dataset,[0,1,2,3])
    loader=torch.utils.data.DataLoader(selected,batch_size=config['batch_size'],num_workers=config['num_workers'],
        shuffle=False,drop_last=False,pin_memory=True,generator=torch.Generator().manual_seed(config['seed']))
    users=sorted({_user_key(dataset.store.episodes[int(w['episode'])]['subject']) for w in windows})
    ordinals={user:i for i,user in enumerate(users)}
    reader=runtime.crops.CropStore(spec['crops']) if not visual else None
    original=runtime.metrics.AngleAccumulator('center' if visual else 'trajectory')
    center_meter=runtime.audit.GroupedCenterAccumulator();population=runtime.audit.CenterPopulation(evidence['provenance']['partition_sha256'])
    trajectory={};targets=[];masks=[];predictions=[];metadata={key:[] for key in reducers.COHORT_DTYPES}
    model.eval();offset=0
    for batch in loader:
        snapshot.check();_source_check(spec['legacy_root'],spec['upstream_root'])
        batch=train.move_batch(batch,spec['device']);pred,target,valid=_forward_chunk(runtime,model,batch,config,spec['device'])
        for i in range(len(pred)):
            window=windows[offset//2];hand=('left','right')[offset%2];user=_user_key(batch['subject'][i])
            if batch['hand'][i]!=hand or batch['split'][i]!=window['split'] or int(batch['center'][i])!=window['center']:
                raise ValueError('original paired sample order differs')
            center=0 if visual else 12000//2-model.left_context
            if (not visual and (model.left_context,model.right_context,pred.shape[-1])!=(510,0,11490)) or (visual and pred.shape[-1]!=1):
                raise ValueError('original context/physical center differs')
            original.update(pred[i],target[i],valid[i],center_index=center,hand=hand,split=window['split'],user=user)
            cp,ct,cv=(v[i,:,center:center+1] for v in (pred,target,valid))
            center_meter.update(cp,ct,cv,hand=hand,split=window['split'],user=user)
            present=bool(batch['vision_valid_mask'][i].bool().any()) if visual else runtime.audit.crop_present_at_center(dataset.store,window,reader,hand)
            population.update(center=window['center'],hand=hand,split=window['split'],crop_present=present)
            row=window['center'];store=dataset.store
            if store.split_labels[int(store.field('frame_split_id')[row])]!=window['split']:raise ValueError('physical center split differs')
            gid=int(store.field('label_gesture_class')[row]);active=bool(store.field('label_gesture_active')[row])
            if not -1<=gid<=59:raise ValueError('physical center gesture is outside pinned vocabulary')
            for key,value in dict(ordinal=offset,hand=offset%2,split=reducers.SPLITS.index(window['split']),gesture=gid,
                active=active,crop_present=present,user_ordinal=ordinals[user]).items():metadata[key].append(value)
            predictions.append(cp[:,0]);targets.append(ct[:,0]);masks.append(cv[:,0])
            if not visual:reducers.reduce_trajectory_chunk(pred[i],target[i],valid[i],
                dict(hand=offset%2,split=reducers.SPLITS.index(window['split']),user_ordinal=ordinals[user]),trajectory)
            offset+=1
        snapshot.check()
    if offset!=4 or len(loader)!=1:raise ValueError('bounded original-batch traversal differs')
    target=np.asarray(targets,dtype='<f4');valid=np.asarray(masks,dtype=bool)
    metadata={key:np.asarray(value,dtype=reducers.COHORT_DTYPES[key]) for key,value in metadata.items()}
    identity=reducers.cohort_identity(target,valid,metadata)
    cache=io.BytesIO();np.savez(cache,prediction=np.asarray(predictions,dtype='<f4'),target=target,valid=valid,
        **metadata,cohort_sha256=np.array(identity),model=np.array(spec['run_spec']['model']))
    # Raw user labels stay only in these in-memory objects; expose no per-user mapping.
    aggregate=original.report();safe_overall=dict(aggregate['overall'])
    return cache.getvalue(),dict(samples=offset,batches=1,cohort_sha256=identity,original_overall=safe_overall,
        trajectory= reducers.trajectory_report(trajectory) if trajectory else None,
        population=population.report(),baseline_agreement=None,full_cohort=False,
        original_batch_size=config['batch_size'],original_num_workers=config['num_workers'],
        engineering_partial_batch=True)


def _child(spec,cohort,admitted):
    # This is reached only through the fresh isolated interpreter below.
    if _admit_inputs(spec,cohort,bounded=True)!=admitted:raise ValueError('child metadata differs from parent admission')
    roots=[spec['data'],spec['crops']]
    trees={root:_tree_files(root) for root in roots}
    run=Path(spec['run_spec']['run']);paths=[Path(spec['legacy_root'])/n for n in preflight.LEGACY_FILES]
    paths += [run/n for n in preflight.SMALL_FILES|{'last.pt','best.pt','progress.jsonl'}]
    paths += [p for values in trees.values() for p in values]
    snapshot=InputSnapshot(paths)
    with offline_transport(),open(os.devnull,'w') as quiet,redirect_stdout(quiet),redirect_stderr(quiet):
        runtime=_load_runtime(spec['legacy_root'],spec['upstream_root']);import torch
        config=preflight.read_json(run/'config.json');_precision_guard(torch,config,spec['device'])
        environment=preflight.read_json(run/'environment.json')
        profile=numeric_environment(torch,environment,visual=runtime.train.model_kind(config)!='emg')
        anchors=CHECKPOINTS[spec['run_spec']['model']]
        last=_load_checkpoint(run/'last.pt',anchors['last.pt'],torch)
        convenience=_load_checkpoint(run/'best.pt',anchors['best.pt'],torch)
        evidence=runtime.audit.validate_artifacts(run,last,device=spec['device'])
        loaded={run/'last.pt':last,run/'best.pt':convenience}
        exported=runtime.report.build_export(run,spec['run_spec']['model'],checkpoint_loader=lambda path:loaded[Path(path)])
        if (exported['identities']['authoritative_checkpoint_file_sha256']!=anchors['last.pt'] or
            exported['identities']['selected_checkpoint_file_sha256']!=anchors['best.pt']):raise ValueError('old export selected files differ')
        train=runtime.train
        current=dict(source=train.validate_upstream(spec['upstream_root']),data=train.data_identity(spec['data']))
        crop=train.visual_data_identity(spec['data'],spec['crops'],evidence['manifest'],allow_missing_episodes=False)
        if crop.get('sha256')!=CROP_SHA:raise ValueError('frozen crop identity differs')
        if train.model_kind(config)!='emg':current['visual_data']=crop
        train.assert_compatible_provenance(evidence['provenance'],current)
        snapshot.check();_source_check(spec['legacy_root'],spec['upstream_root'])
        train.seed_everything(config['seed']);torch.set_float32_matmul_precision(evidence['matmul_precision'])
        selected_token=pre_builder_state_sha256(evidence['best']['model'])
        model=train.build_training_model(spec['upstream_root'],config,resume_state=evidence['best']['model'])
        typed= strict_load_state(model,evidence['best']['model']);model=model.to(spec['device'])
        if typed!=selected_token:raise ValueError('post-builder complete typed identity differs')
        cache,statistics=_infer(runtime,model,evidence,spec,cohort,snapshot)
        snapshot.check();_source_check(spec['legacy_root'],spec['upstream_root'])
        if any(_tree_files(root)!=values for root,values in trees.items()):raise ValueError('input namespace inventory changed')
    return cache,dict(format=FORMAT,status='bounded_engineering_only',benchmark_eligible=False,scientific_eligible=False,
        **result_identity(admitted),selected_tensor_sha256=typed,statistics=statistics,
        center_cache_sha256=preflight._hash(cache),evaluation_profile=dict(profile,device=spec['device'],
            precision=config['precision'],matmul_precision=torch.get_float32_matmul_precision(),
            visible_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(0)),
        full_payload_rehashed=False,this_derived_full_passes_per_model=0,historical_training_final_test_passes=1,
        historical_locked_center_verification_passes=1)


def _launch(request,attempt):
    roots=[]
    for entry in sys.path:
        if entry and Path(entry).is_dir() and not ((Path(entry)/'egoemg_repro').exists() or (Path(entry)/'egoemg').exists()):
            roots.append(str(Path(entry).absolute()))
    request=dict(request,dependency_roots=roots,owned_identity=list(attempt.identity))
    attempt.write('request.json',Attempt.json(request))
    if not attempt.current():raise ValueError('attempt namespace changed before launch')
    for name in ('private-cache','private-temp'):os.mkdir(name,mode=0o700,dir_fd=attempt.fd)
    env={key:value for key,value in os.environ.items() if key in ('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')}
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TRANSFORMERS_OFFLINE='1',
        EGOEMG_NO_PRETRAINED_DOWNLOAD='1',TMPDIR=str(attempt.path/'private-temp'),
        TORCH_HOME=str(attempt.path/'private-cache'),HF_HOME=str(attempt.path/'private-cache'),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    process=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).absolute()),'--_worker_request',str(attempt.path/'request.json')],
        env=env,cwd='/',capture_output=True,text=True)
    if process.returncode:raise RuntimeError('isolated bounded legacy worker refused; exit='+str(process.returncode))
    if process.stdout.strip()!='bounded engineering child finished':raise ValueError('child protocol differs')
    return preflight.read_json(attempt.path/'result.json')


def run_model_pass(spec,cohort,*,bounded):
    """Spawn one bounded isolated worker. Formal mode is unavailable, with no CLI."""
    _contract(spec,cohort,bounded)
    guard=RunInputGuard(spec)
    admitted=_admit_inputs(spec,cohort,bounded=bounded)
    guard.check()
    with Attempt(spec['output'],admitted) as attempt:
        try:
            result=_launch(dict(spec=spec,cohort=cohort,admitted=admitted),attempt)
            guard.check()
            if (result.get('admission_sha256')!=preflight._canonical(admitted) or result.get('benchmark_eligible') is not False or
                result.get('scientific_eligible') is not False or result.get('format')!=FORMAT or
                result.get('status')!='bounded_engineering_only' or result.get('model')!=spec['run_spec']['model'] or
                result.get('this_derived_full_passes_per_model')!=0):raise ValueError('child returned an incompatible scope')
            cache=preflight._safe(Path(spec['output'])/'centers.npz')
            cache_stamp=guard.add(cache)
            try:
                value=cache.stat()
                if (value.st_size>MAX_OUTPUT or value.st_mode&0o777!=0o600 or value.st_uid!=os.getuid() or
                    _whole_sha(cache)!=result.get('center_cache_sha256')):
                    raise ValueError('private center cache identity differs')
            except OSError:raise ValueError('private center cache is missing or partial') from None
            if _admit_inputs(spec,cohort,bounded=True)!=admitted:raise ValueError('parent final admission differs')
            guard.check()
            if preflight._file_stamp(cache)!=cache_stamp:raise ValueError('private center cache changed before publication')
            attempt.finish(result,guard=guard)
            return result
        except BaseException:
            attempt.fail();raise


def _private_main(argv):
    if len(argv)!=2 or argv[0]!='--_worker_request':raise ValueError('only the private bounded child protocol exists')
    path=preflight._safe(argv[1])
    if path.stat().st_mode&0o777!=0o600:raise ValueError('child request must be private0600')
    request=preflight.read_json(path)
    if set(request)!={'spec','cohort','admitted','dependency_roots','owned_identity'}:raise ValueError('private request schema differs')
    for entry in request['dependency_roots']:
        entry=preflight._safe(entry)
        if (entry/'egoemg_repro').exists() or (entry/'egoemg').exists():raise ValueError('dependency root contains a shadow runtime')
        sys.path.append(str(entry))
    _tools();spec=request['spec'];output=preflight._safe(spec['output'])
    # Child has no authority to create a replacement attempt or change the start record.
    attempt=Attempt.__new__(Attempt);attempt.path=output;attempt.lock=None;attempt.marker_fd=None;attempt.owned={}
    attempt.fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);attempt.identity=tuple(request['owned_identity'])
    try:
        if not attempt.current():raise ValueError('child output is no longer the parent-owned directory')
        cache,result=_child(spec,request['cohort'],request['admitted'])
        attempt.write('centers.npz',cache);attempt.write('result.json',Attempt.json(result))
        # No finished marker here: parent must re-admit inputs before final publication.
    finally:attempt.close()
    print('bounded engineering child finished')
    return 0


if __name__=='__main__':
    try:raise SystemExit(_private_main(sys.argv[1:]))
    except Exception:
        # No path, participant or exception payload may leak to launcher logs.
        print('bounded legacy child refused',file=sys.stderr);raise SystemExit(1)
