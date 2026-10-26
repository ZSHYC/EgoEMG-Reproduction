#!/usr/bin/env python3
"""Private, review-before-launch CUDA capacity diagnostic; never formal training.

Only --budget and CPU contract tests are safe to run before root authorizes CUDA.
The run subcommand uses a new process for one fixed rung and never retries OOM.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import stat
import sys
import time
import uuid

FORMAT = 'private_classic_cuda_capacity_v1'
SCOPE = 'bounded_cuda_capacity_diagnostic'
LADDERS = {'vemg2pose': (32, 128, 600), 'neuropose': (16, 64, 256)}
QUALIFIER_SHA = 'cfbccea940016fd1981aeebdfa7144399ba804ced7c66adec81cd7a32e8e8c54'
COHORT_SHA = '72267d660f55e9f5fbbd2e1e17edf07627ef01d2ba09e92fca1086c50a16dbd3'
CPU_SHA = '52b1ce655f3c4e107e3d809e403cd507fbcfd655f7ac9ea7bc0b9a592b18b0d8'
RESERVE = 15 * 1024**3
LOG_LIMIT = 16 * 1024**2
JSON_LIMIT = 2 * 1024**2
IMPORTED_HELPER_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def safe(path, *, exists=True):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError('a real filesystem artifact path is required')
    path = Path(path).expanduser().absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('no traversal or symlink paths are admitted')
    return path.resolve(strict=exists)


def policy(variant, batch):
    if variant not in LADDERS or type(batch) is not int or batch not in LADDERS[variant]:
        raise ValueError('only the reviewed fixed capacity ladder is admitted')
    return dict(format=FORMAT, scope=SCOPE, scientific_eligible=False, production_ready=False,
        variant=variant, batch_size=batch, validation_batch_size=batch, num_workers=0,
        world_size=1, precision='16-mixed', train_attempts=2, train_samples=2*batch,
        validation_samples=1576, validation_batches=math.ceil(1576/batch), max_epochs=1,
        gradient_clip_val=0, accumulate_grad_batches=1, runtime_profile='canonical_backend_nondeterministic_v1',
        deterministic=False, cudnn_deterministic=False, cudnn_benchmark=False,
        matmul_precision='high', cublas_workspace_config=':4096:8',
        train_selection='first 2B unique frozen train hand indices, shuffled once without replacement',
        validation='all frozen val hand indices once in original order; same B as train; original batch-union FK',
        scaler='actual Lightning MixedPrecision default GradScaler; no tuning or injected faults',
        memory_threshold='each phase peak reserved <= 85% total; after second train attempt free >= 4GiB',
        trajectory_agreement_qualified=False, bitwise_trajectory_guaranteed=False,
        limitations='capacity engineering only; no full training, scientific result, best/ES selection, test or production permit')


def work_budget():
    return dict(children=6, train_attempts=12,
        train_hand_samples=sum(2*b for ladder in LADDERS.values() for b in ladder),
        validation_hand_samples=6*1576,
        validation_batches=sum(math.ceil(1576/b) for ladder in LADDERS.values() for b in ladder))


def check_memory(observation):
    for key in ('peak_allocated', 'peak_reserved', 'free', 'total'):
        value = observation.get(key)
        if type(value) is not int or value < 0:
            raise ValueError('actual memory counters must be nonnegative integer bytes')
    if (observation['total'] <= 0 or observation['peak_allocated'] > observation['peak_reserved'] or
        observation['peak_reserved'] > observation['total']*0.85 or observation['free'] < 4*1024**3):
        raise ValueError('capacity memory margin failed; no batch reduction or retry is authorized')


def require_qualification_files(prefix, resumed):
    # A passed:true dict, exported summary or log is deliberately insufficient.
    for directory in (prefix, resumed):
        directory = safe(directory)
        for name in ('identity.json', 'status.json', 'result.json', 'checkpoints/last.ckpt'):
            if not safe(directory/name).is_file():
                raise ValueError('complete actual numeric qualification artifacts are required')


def _rename_noreplace(parent_fd,source,destination):
    """Linux atomic fresh-name publication; never fall back to overwrite."""
    if sys.platform!='linux':
        raise RuntimeError('Linux atomic NOREPLACE publication is required')
    rename=getattr(ctypes.CDLL(None,use_errno=True),'renameat2',None)
    if rename is None:
        raise RuntimeError('atomic renameat2 NOREPLACE is unsupported')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(parent_fd,os.fsencode(source),parent_fd,os.fsencode(destination),1):
        code=ctypes.get_errno()
        if code in (errno.ENOSYS,errno.EINVAL,errno.EOPNOTSUPP):
            raise RuntimeError('atomic renameat2 NOREPLACE is unsupported')
        raise OSError(code,os.strerror(code),destination)


class OwnedOutput:
    """Hold a private staging fd before public-name publication.

    The parent and private random mkdir/open acquisition namespace are trusted.
    Linux mkdir returns no fd; hostile interference inside that acquisition
    namespace is outside this assumption. A public-name open only verifies the
    already held capability and never establishes ownership.
    """
    def __init__(self, path, fd):
        self.path, self.fd = path, fd
        current = os.fstat(fd)
        self.identity = current.st_dev, current.st_ino

    @classmethod
    def claim(cls, path):
        path = safe(path, exists=False)
        if not path.parent.is_dir():
            raise ValueError('output parent must already exist')
        if path.exists():
            raise FileExistsError('capacity output already exists')
        if sys.platform!='linux':
            raise RuntimeError('Linux atomic NOREPLACE publication is required')
        if shutil.disk_usage(path.parent).free < RESERVE+LOG_LIMIT+4*JSON_LIMIT:
            raise OSError('15GiB output reserve plus bounded diagnostics is required')
        parent_fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        parent_stat=os.fstat(parent_fd)
        parent_identity=(parent_stat.st_dev,parent_stat.st_ino)
        staging='.classic-capacity-stage-'+uuid.uuid4().hex
        own=None;published=False
        try:
            os.mkdir(staging,mode=0o700,dir_fd=parent_fd)
            fd=os.open(staging,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent_fd)
            own=cls(path,fd)
            parent_now=os.stat(safe(path.parent),follow_symlinks=False)
            staged=os.stat(staging,dir_fd=parent_fd,follow_symlinks=False)
            if ((parent_now.st_dev,parent_now.st_ino)!=parent_identity or
                not stat.S_ISDIR(staged.st_mode) or (staged.st_dev,staged.st_ino)!=own.identity):
                raise ValueError('private capacity staging directory identity changed')
            _rename_noreplace(parent_fd,staging,path.name);published=True
            os.fsync(parent_fd)
            public_fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
            try:
                public=os.fstat(public_fd)
                if (public.st_dev,public.st_ino)!=own.identity:
                    raise ValueError('published capacity output directory identity changed')
                own.check()
            finally:
                os.close(public_fd)
            os.fsync(own.fd)
            return own
        except BaseException:
            if own is not None:
                if not published:
                    try:
                        staged=os.stat(staging,dir_fd=parent_fd,follow_symlinks=False)
                        if (staged.st_dev,staged.st_ino)==own.identity:
                            os.rmdir(staging,dir_fd=parent_fd);os.fsync(parent_fd)
                    except OSError:
                        pass
                own.close()
            raise
        finally:
            os.close(parent_fd)

    def check(self, budget=0):
        if self.fd is None:
            raise ValueError('output capability is closed')
        current = os.stat(safe(self.path), follow_symlinks=False)
        opened = os.fstat(self.fd)
        if (not stat.S_ISDIR(current.st_mode) or
            (current.st_dev,current.st_ino) != self.identity or
            (opened.st_dev,opened.st_ino) != self.identity):
            raise ValueError('output namespace was rebound')
        if shutil.disk_usage(self.fd).free < RESERVE+budget:
            raise OSError('15GiB output reserve would be breached')

    def write_json(self, name, value):
        if name not in ('identity.json', 'result.json', 'status.json', 'provenance.json'):
            raise ValueError('unowned output name')
        payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        if len(payload)>JSON_LIMIT:
            raise OSError('bounded diagnostic JSON limit exceeded')
        self.check(len(payload)+4096)
        temporary = '.capacity-'+uuid.uuid4().hex+'.incomplete'
        fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600, dir_fd=self.fd)
        try:
            try:
                if os.write(fd, payload) != len(payload):
                    raise OSError('short diagnostic JSON write')
                os.fsync(fd)
            finally:
                os.close(fd)
            self.check()
            os.replace(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            os.fsync(self.fd)
            return hashlib.sha256(payload).hexdigest()
        finally:
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def verify_json(self, name, expected):
        self.check()
        if name not in ('identity.json','result.json','status.json'):
            raise ValueError('unowned verification name')
        fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=self.fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size>JSON_LIMIT:
                raise ValueError('published diagnostic file is invalid')
            with os.fdopen(fd,'rb',closefd=False) as stream:
                actual = hashlib.sha256(stream.read(JSON_LIMIT+1)).hexdigest()
            after = os.fstat(fd)
            current = os.stat(name,dir_fd=self.fd,follow_symlinks=False)
            stamp = lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
            if stamp(before)!=stamp(after) or stamp(after)!=stamp(current) or actual!=expected:
                raise ValueError('published diagnostic JSON changed')
        finally:
            os.close(fd)
        self.check()

    def fail(self, exc):
        # Even after a namespace rebind, only this run's original directory is touched.
        if self.fd is None:
            return
        for name in ('result.json', 'status.json'):
            try:
                os.unlink(name, dir_fd=self.fd)
            except OSError:
                # Still attempt status revocation if result removal itself fails.
                pass
        try:
            os.fsync(self.fd)
            self.write_json('status.json', dict(state='failed', scope=SCOPE,
                scientific_eligible=False, production_ready=False,
                error_type=type(exc).__name__, error=str(exc)[:4096]))
        except (OSError, ValueError):
            pass

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CappedLog(io.TextIOBase):
    def __init__(self, owned):
        self.owned, self.written, self.failed = owned, 0, None
        self.fd = os.open('child.log', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600, dir_fd=owned.fd)
    def writable(self):
        return True
    def write(self, text):
        try:
            self.check()
            payload = text.encode('utf-8', errors='replace')
            if self.written+len(payload)>LOG_LIMIT:
                raise OSError('16MiB child log cap exceeded; stop without truncation')
            self.owned.check(len(payload))
            if os.write(self.fd,payload)!=len(payload):
                raise OSError('short child log write')
            self.written += len(payload)
            return len(text)
        except BaseException as exc:
            # logging.StreamHandler may swallow write errors. Preserve failure as state.
            self.failed = exc
            raise
    def check(self):
        if self.failed is not None:
            raise OSError('child log has a prior cap/write failure') from self.failed
    def flush(self):
        self.check()
        if getattr(self,'fd',None) is not None:
            os.fsync(self.fd)
    def close(self):
        if getattr(self,'fd',None) is not None:
            self.flush()
            os.close(self.fd)
            self.fd=None
        super().close()


def close_log(log, owned):
    """A final flush/close failure must revoke this entry's success too."""
    try:
        log.close()
    except BaseException as exc:
        owned.fail(exc)
        if getattr(log, 'fd', None) is not None:
            os.close(log.fd)
            log.fd = None
        raise


def _read_json(path):
    path = safe(path)
    if path.stat().st_size > JSON_LIMIT:
        raise ValueError('oversize diagnostic document')
    return json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _source_snapshot(root, qualifier):
    names = set(qualifier.RUNTIME_FILES) | {'egoemg_repro/classic_emg_cohort.py'}
    snapshot = {name:file_sha(root/name) for name in sorted(names)}
    snapshot['capacity_helper'] = file_sha(Path(__file__))
    if (snapshot['egoemg_repro/classic_emg_cuda_check.py']!=QUALIFIER_SHA or
        snapshot['egoemg_repro/classic_emg_cohort.py']!=COHORT_SHA or
        snapshot['egoemg_repro/classic_emg_train.py']!=CPU_SHA or
        snapshot['capacity_helper']!=IMPORTED_HELPER_SHA):
        raise ValueError('frozen qualifier/cohort/CPU/helper source differs')
    return snapshot


def validate_result(result,p):
    """Engineering artifact consistency only, never an independent GPU permit."""
    def finite(value, *, positive=False):
        return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else True)
    try:
        identity=result['identity']
        if (result['format']!=FORMAT or result['scope']!=SCOPE or result['capacity_passed'] is not True or
            result['scientific_eligible'] is not False or result['production_ready'] is not False or
            identity['policy']!=p or identity['sha256']!=digest({k:v for k,v in identity.items() if k!='sha256'}) or
            identity['scientific_eligible'] is not False or identity['production_ready'] is not False or
            result['attempts']!=2 or len(result['events'])!=2):
            raise ValueError('capacity result scope/policy/identity/attempt count differs')
        executions=0
        previous=None
        for index,event in enumerate(result['events']):
            executed=event['optimizer_executed']
            gradient=event['gradient']
            if (event['batch']!=index or event['size']!=p['batch_size'] or not finite(event['loss']) or
                type(executed) is not bool or event['natural_overflow'] is not (not executed) or
                type(gradient['finite']) is not bool or
                any(type(event[k]) is not int or event[k]<0 for k in ('optimizer_step_before','optimizer_step_after')) or
                not finite(event['scale_before'],positive=True) or not finite(event['scale_after'],positive=True)):
                raise ValueError('capacity attempt fields differ')
            if executed:
                if (event['optimizer_step_after']!=event['optimizer_step_before']+1 or not gradient['finite'] or
                    not finite(gradient['absolute_max']) or event['scale_after']!=event['scale_before']):
                    raise ValueError('capacity finite optimizer execution differs')
            elif (event['optimizer_step_after']!=event['optimizer_step_before'] or gradient['finite'] or
                  gradient['absolute_max'] is not None or event['scale_after']!=event['scale_before']*.5):
                raise ValueError('capacity natural overflow witness differs')
            if previous is not None and (event['optimizer_step_before']!=previous['optimizer_step_after'] or
                                        event['scale_before']!=previous['scale_after']):
                raise ValueError('capacity optimizer/scaler progression differs')
            previous=event;executions+=executed
        if executions<1 or result['optimizer_executions']!=executions or result['natural_overflows']!=2-executions:
            raise ValueError('capacity actual finite execution is absent or counts differ')
        expected_sizes=[min(p['batch_size'],1576-index*p['batch_size']) for index in range(p['validation_batches'])]
        val=result['validation'];events=result['validation_events']
        if (val['samples']!=1576 or val['batches']!=len(expected_sizes) or val['batch_sizes']!=expected_sizes or
            val['full_frozen_val_coverage'] is not True or val['original_batch_union_fk'] is not True or
            val['loss_selection_performed'] is not False or val['scientific_eligible'] is not False or
            len(events)!=len(expected_sizes) or not finite(val['val_loss']) or not finite(val['batch_weighted_loss']) or
            result['validation_order_sha256']!=digest(list(range(1576))) or
            result['train_coverage']['samples']!=p['train_samples'] or
            result['train_coverage']['unique_samples']!=p['train_samples'] or
            result['train_coverage']['sorted_indices_sha256']!=digest(list(range(p['train_samples'])))):
            raise ValueError('capacity original validation/train coverage differs')
        for index,(event,size) in enumerate(zip(events,expected_sizes)):
            if event['batch']!=index or event['size']!=size or not finite(event['loss']):
                raise ValueError('capacity validation event differs')
        weighted=sum(event['size']*event['loss'] for event in events)/1576
        if not math.isclose(weighted,val['val_loss'],rel_tol=2e-6,abs_tol=1e-7) or weighted!=val['batch_weighted_loss']:
            raise ValueError('capacity original batch-weighted validation loss differs')
        for phase in ('train','validation'):
            check_memory(result['memory'][phase])
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('incomplete capacity engineering artifact') from exc


def _previous(directory, p, runtime, sources, cohort_sha):
    ladder = LADDERS[p['variant']]
    position = ladder.index(p['batch_size'])
    if position == 0:
        if directory is not None:
            raise ValueError('the first rung starts from actual batch1 numeric qualification only')
        return None
    if directory is None:
        raise ValueError('the immediate preceding reviewed rung must have completed first')
    directory = safe(directory)
    status, result = _read_json(directory/'status.json'), _read_json(directory/'result.json')
    if (status.get('state')!='capacity_entry_finished' or status.get('result_sha256')!=file_sha(directory/'result.json') or
        result.get('format')!=FORMAT or result.get('scope')!=SCOPE or
        result.get('scientific_eligible') is not False or result.get('production_ready') is not False):
        raise ValueError('previous rung has no source-bound completed engineering artifact')
    identity = result['identity']
    validate_result(result,policy(p['variant'],ladder[position-1]))
    if (_read_json(directory/'identity.json')!=identity or identity['runtime']!=runtime or
        identity['sources']!=sources or identity['cohort_receipt_sha256']!=cohort_sha or
        result['process']['pid']==os.getpid()):
        raise ValueError('previous rung identity/process/coverage differs')
    for phase in ('train', 'validation'):
        check_memory(result['memory'][phase])
    return dict(batch_size=ladder[position-1],result_sha256=file_sha(directory/'result.json'),process=result['process'])


def run(args):
    p = policy(args.variant,args.batch_size)
    helper = safe(__file__)
    if args.reviewed_helper_sha256!=IMPORTED_HELPER_SHA or file_sha(helper)!=IMPORTED_HELPER_SHA:
        raise ValueError('root must supply the reviewed exact helper SHA before CUDA execution')
    root = helper.parent.parent
    sys.path.insert(0,str(root))
    require_qualification_files(args.numeric_prefix,args.numeric_resume)
    with OwnedOutput.claim(args.output) as owned:
        log = CappedLog(owned)
        log_finalized = False
        try:
            owned.write_json('status.json',dict(state='running',scope=SCOPE,scientific_eligible=False,production_ready=False))
            # Import inside redirection: Lightning creates handlers capturing sys.stderr on import.
            with redirect_stdout(log), redirect_stderr(log):
                from egoemg_repro import classic_emg_cuda_check as q, classic_emg_cohort as cohort
                log.check()
                sources = _source_snapshot(root,q)
                numeric_paths = [safe(directory)/name for directory in (args.numeric_prefix,args.numeric_resume)
                    for name in ('identity.json','status.json','result.json','checkpoints/last.ckpt')]
                # Capture BEFORE verification, retain through the complete CPU scan and GPU entry.
                entry_guard = cohort._StatGuard([*numeric_paths,helper,safe(args.cohort_receipt),
                    safe(args.config),safe(args.partitions),*(root/name for name in q.RUNTIME_FILES)])
                qverified = q.verify_cuda_check(args.numeric_prefix,args.numeric_resume)
                numeric = _read_json(safe(args.numeric_prefix)/'identity.json')
                entry_guard.check()
                if (qverified['cuda_numeric_passed'] is not True or qverified['strict_state_restore'] is not True or
                    qverified['identity_sha256']!=numeric['sha256'] or numeric['recipe']['variant']!=args.variant):
                    raise ValueError('actual batch1 numeric qualification differs')
                recipe = q.datasets.load_classic_config(safe(args.config),safe(args.upstream))
                if recipe!=numeric['recipe']:
                    raise ValueError('capacity base recipe differs from actual numeric qualifier')
                partitions = _read_partition(args.partitions)
                preparation_start = time.monotonic()
                guard = cohort.cohort_stat_guard(args.data,args.partitions,upstream=args.upstream,
                    config=args.config,receipt=args.cohort_receipt,finite_probe=args.finite_probe)
                cpu = cohort._json(args.cohort_receipt)
                entry_guard.check()
                guard.paths.update(entry_guard.paths);guard.parents.update(entry_guard.parents)
                guard.check()
                if cpu['recipe']!=recipe or {k:v['hand_samples'] for k,v in cpu['footprints'].items()} != dict(train=101714,val=1576,test=4306):
                    raise ValueError('capacity requires the exact fully admitted frozen cohort')
                cpu_seconds = time.monotonic()-preparation_start
                log.check()
                runtime,precision = q._profile()  # First CUDA initialization; root-selected UUID only.
                if runtime!=numeric['runtime']:
                    raise ValueError('actual complete backend/runtime signature differs from batch1 qualification')
                framework = Path(q.pl.__file__).resolve().parent
                for name in q.storage.FRAMEWORK_FILES:
                    guard.add(framework/name)
                torch_root = Path(q.storage._runtime().__file__).resolve().parent
                for name in runtime['torch_files_sha256']:
                    guard.add(torch_root/name)
                if args.previous is not None:
                    for name in ('identity.json','status.json','result.json'):
                        guard.add(safe(args.previous)/name)
                previous = _previous(args.previous,p,runtime,sources,file_sha(args.cohort_receipt))
                guard.check()
                prepared = q._prepare(args.data,args.upstream,args.config,partitions,runtime,expected_recipe=recipe)
                if prepared.identity!=numeric:
                    raise ValueError('actual original factory input/source/data identity differs from numeric qualification')
                module = q._factory(prepared)
                identity = dict(format=FORMAT,scope=SCOPE,scientific_eligible=False,production_ready=False,
                    policy=p,runtime=runtime,sources=sources,cohort_receipt_sha256=file_sha(args.cohort_receipt),
                    data_identity_sha256=cpu['data_identity']['sha256'],numeric_identity_sha256=numeric['sha256'],
                    numeric_artifacts=dict(verification=qverified,prefix_result_sha256=file_sha(safe(args.numeric_prefix)/'result.json'),
                        resumed_result_sha256=file_sha(safe(args.numeric_resume)/'result.json')),
                    previous=previous,model=numeric['model'],recipe=recipe)
                identity['sha256']=digest(identity)
                identity_file_sha=owned.write_json('identity.json',identity)
                result = _fit(q,cohort,prepared,module,precision,guard,p,owned,log)
                guard.check()
                if _source_snapshot(root,q)!=sources:
                    raise ValueError('source changed during bounded capacity execution')
                q._check_prepared(prepared)
                result.update(format=FORMAT,scope=SCOPE,scientific_eligible=False,production_ready=False,
                    capacity_passed=True,identity=identity,cpu_full_verification_seconds=cpu_seconds,
                    process=q._process_identity('capacity-'+str(p['batch_size'])),
                    completed_utc=datetime.now(timezone.utc).isoformat())
                if previous and result['process']['pid']==previous['process']['pid'] and result['process']['start_ticks']==previous['process']['start_ticks']:
                    raise ValueError('capacity rungs require fresh processes')
                validate_result(result,p)
                result_sha=owned.write_json('result.json',result)
                guard.check();owned.check()
                status_sha=owned.write_json('status.json',dict(state='capacity_entry_finished',scope=SCOPE,
                    scientific_eligible=False,production_ready=False,result_sha256=result_sha))
                guard.check();owned.check()
            log.flush()
            entry_guard.check();guard.check()
            for name,expected in (('identity.json',identity_file_sha),('result.json',result_sha),('status.json',status_sha)):
                owned.verify_json(name,expected)
                entry_guard.check();guard.check()
            close_log(log,owned)
            log_finalized = True
            # Close/readback callbacks execute before success. Retain the SAME
            # source witnesses and output capability until the final checks.
            owned.check();entry_guard.check();guard.check()
            return dict(scope=SCOPE,variant=args.variant,batch_size=args.batch_size,
                capacity_passed=True,scientific_eligible=False,production_ready=False,
                result_sha256=result_sha)
        except BaseException as exc:
            owned.fail(exc)
            raise
        finally:
            if not log_finalized:
                close_log(log,owned)


def _read_partition(path):
    # The genuine cohort partition is larger than the small output-document limit.
    path = safe(path)
    if path.stat().st_size>256*1024**2:
        raise ValueError('partition document exceeds bounded metadata allowance')
    return json.loads(path.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite partition')))


def _fit(q,cohort,prepared,module,precision,guard,p,owned,log):
    torch = q.storage._runtime()
    contract = q.models.classic_contract(p['variant'])
    guard.check()
    full = {role:q.datasets.ClassicDataset(prepared.data,[w for w in prepared.partitions['windows'] if w['role']==role],
        upstream=prepared.upstream,config=prepared.cfg,normalization=prepared.recipe['normalization'],role=role,
        mask_policy=prepared.recipe['mask_policy']) for role in ('train','val')}
    class GuardedDataset:
        def __init__(self,dataset,count):
            self.dataset,self.count=dataset,count
            if len(dataset)<count:
                raise ValueError('required unique frozen hand samples unavailable')
        def __len__(self):
            return self.count
        def __getitem__(self,index):
            if not 0<=index<self.count:
                raise IndexError(index)
            with guard.read():
                owned.check();log.check()
                item=self.dataset[index]
                q.datasets.admit_classic_sample(item,contract)
                item['classic_sample_index']=index
                return item
    fixed = {role:GuardedDataset(full[role],p['train_samples'] if role=='train' else p['validation_samples']) for role in ('train','val')}
    if len(full['val'])!=1576:
        raise ValueError('original full validation hand sample count differs')
    generators = {name:torch.Generator().manual_seed(prepared.recipe['seed']+offset) for offset,name in enumerate(('train_loader','train_sampler','val_loader'))}
    train = torch.utils.data.DataLoader(fixed['train'],batch_size=p['batch_size'],num_workers=0,
        sampler=torch.utils.data.RandomSampler(fixed['train'],replacement=False,generator=generators['train_sampler']),
        generator=generators['train_loader'],drop_last=False)
    val = torch.utils.data.DataLoader(fixed['val'],batch_size=p['batch_size'],num_workers=0,shuffle=False,
        generator=generators['val_loader'],drop_last=False)
    class Observer(q.Callback):
        def __init__(self):
            self.events=[];self.val_events=[];self.executions=0;self.seen=[];self.val_seen=[];self.memory={};self.hook=None
        def snapshot(self):
            torch.cuda.synchronize(0)
            free,total=torch.cuda.mem_get_info(0)
            return dict(peak_allocated=torch.cuda.max_memory_allocated(0),peak_reserved=torch.cuda.max_memory_reserved(0),
                allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0),free=free,total=total,
                host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                host_rss_bytes=_rss(),elapsed_fit_seconds=time.monotonic()-self.fit_start)
        def on_train_start(self,trainer,pl_module):
            optimizer=trainer.optimizers[0]
            if (type(optimizer).__name__!='AdamW' or len(trainer.lr_scheduler_configs)!=1 or
                type(trainer.lr_scheduler_configs[0].scheduler).__name__!='CosineAnnealingLR' or
                trainer.lr_scheduler_configs[0].scheduler.T_max!=100 or not pl_module.automatic_optimization):
                raise ValueError('original automatic AdamW/Cosine T_max100 contract differs')
            self.hook=optimizer.register_step_post_hook(lambda *_:setattr(self,'executions',self.executions+1))
            self.fit_start=time.monotonic();torch.cuda.reset_peak_memory_stats(0)
        def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
            guard.check();owned.check();log.check();q.storage._admit_batch(batch,contract)
            ids=batch['classic_sample_index'].tolist()
            if len(self.events)>=2 or len(ids)!=p['batch_size'] or len(set(ids))!=len(ids):
                raise ValueError('exact two unique original-size train batches required')
            self.seen.extend(ids);self.started=time.monotonic();self.gradient=None
            self.before=dict(step=q.storage._optimizer_step(trainer.optimizers[0]),executions=self.executions,
                scale=float(trainer.precision_plugin.scaler.get_scale()))
        def on_before_backward(self,trainer,pl_module,loss):
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite original training loss; no cleanup or retry')
        def on_before_optimizer_step(self,trainer,pl_module,optimizer):
            self.gradient=q.gradient_observation([parameter.grad for parameter in pl_module.parameters() if parameter.grad is not None])
        def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
            step=q.storage._optimizer_step(trainer.optimizers[0]);executed=self.executions-self.before['executions']
            scale=float(trainer.precision_plugin.scaler.get_scale());loss=float(outputs['loss'].detach())
            if not math.isfinite(loss) or self.gradient is None or executed not in (0,1):
                raise ValueError('missing actual optimizer/scaler/loss observation')
            if executed:
                if step!=self.before['step']+1 or not self.gradient['finite'] or scale!=self.before['scale']:
                    raise ValueError('genuine finite optimizer step observation differs')
            elif step!=self.before['step'] or self.gradient['finite'] or scale!=self.before['scale']*.5:
                raise ValueError('natural overflow skip observation differs')
            observed=self.snapshot()
            self.events.append(dict(batch=batch_idx,size=len(batch['classic_sample_index']),loss=loss,
                optimizer_executed=bool(executed),optimizer_step_before=self.before['step'],optimizer_step_after=step,
                scale_before=self.before['scale'],scale_after=scale,gradient=self.gradient,
                natural_overflow=not bool(executed),step_seconds=time.monotonic()-self.started,memory=observed))
            if len(self.events)==2:
                check_memory(observed);self.memory['train']=observed
            guard.check();owned.check();log.check()
        def on_validation_epoch_start(self,trainer,pl_module):
            if trainer.sanity_checking:
                raise ValueError('sanity validation is not in this bounded protocol')
            self.val_started=time.monotonic();torch.cuda.reset_peak_memory_stats(0)
        def on_validation_batch_start(self,trainer,pl_module,batch,batch_idx,dataloader_idx=0):
            guard.check();owned.check();log.check();q.storage._admit_batch(batch,contract)
            ids=batch['classic_sample_index'].tolist();self.val_seen.extend(ids)
            expected=min(p['batch_size'],1576-batch_idx*p['batch_size'])
            if len(ids)!=expected or ids!=list(range(batch_idx*p['batch_size'],batch_idx*p['batch_size']+expected)):
                raise ValueError('full original validation batch grouping/coverage differs')
            self.val_batch_started=time.monotonic()
        def on_validation_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
            loss=float(outputs.detach())
            if not math.isfinite(loss):
                raise ValueError('nonfinite original full-batch validation loss')
            torch.cuda.synchronize(0)
            self.val_events.append(dict(batch=batch_idx,size=len(batch['classic_sample_index']),loss=loss,
                seconds=time.monotonic()-self.val_batch_started))
            guard.check();owned.check();log.check()
        def on_validation_end(self,trainer,pl_module):
            loss=float(trainer.callback_metrics['val_loss'])
            weighted=sum(event['size']*event['loss'] for event in self.val_events)/1576
            if (self.val_seen!=list(range(1576)) or len(self.val_events)!=p['validation_batches'] or
                not math.isfinite(loss) or not math.isclose(loss,weighted,rel_tol=2e-6,abs_tol=1e-7)):
                raise ValueError('original validation coverage or batch-weighted loss differs')
            observed=self.snapshot();check_memory(observed);self.memory['validation']=observed
            self.validation=dict(samples=1576,batches=len(self.val_events),batch_sizes=[e['size'] for e in self.val_events],
                val_loss=loss,batch_weighted_loss=weighted,seconds=time.monotonic()-self.val_started,
                full_frozen_val_coverage=True,original_batch_union_fk=True,
                scientific_eligible=False,loss_selection_performed=False)
        def on_train_epoch_end(self,trainer,pl_module):
            if sorted(self.seen)!=list(range(p['train_samples'])) or len(self.events)!=2 or self.executions<1 or not hasattr(self,'validation'):
                raise ValueError('bounded training lacks finite genuine update or complete original validation')
        def teardown(self,trainer,pl_module,stage):
            if self.hook is not None:
                self.hook.remove();self.hook=None
    observer=Observer()
    trainer=q.pl.Trainer(accelerator='gpu',devices=1,max_epochs=1,max_steps=-1,
        gradient_clip_val=0,accumulate_grad_batches=1,plugins=[precision],
        callbacks=[observer],logger=False,enable_checkpointing=False,enable_progress_bar=False,
        enable_model_summary=False,num_sanity_val_steps=0,log_every_n_steps=50,
        check_val_every_n_epoch=1,val_check_interval=1.0,limit_train_batches=1.0,limit_val_batches=1.0,
        default_root_dir=str(owned.path),deterministic=False)
    trainer.fit(module,train_dataloaders=train,val_dataloaders=val)
    if (len(observer.events)!=2 or observer.executions<1 or not hasattr(observer,'validation') or
        len(observer.val_events)!=p['validation_batches']):
        raise ValueError('actual bounded fit was incomplete')
    return dict(attempts=2,optimizer_executions=observer.executions,
        natural_overflows=sum(event['natural_overflow'] for event in observer.events),
        events=observer.events,validation=observer.validation,validation_events=observer.val_events,
        memory=observer.memory,fit_seconds=time.monotonic()-observer.fit_start,
        train_coverage=dict(samples=len(observer.seen),unique_samples=len(set(observer.seen)),
            sorted_indices_sha256=digest(sorted(observer.seen))),
        train_order_sha256=digest(observer.seen),validation_order_sha256=digest(observer.val_seen))


def _rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    raise RuntimeError('actual Linux host RSS observation unavailable')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('budget')
    launch=sub.add_parser('run')
    launch.add_argument('--variant',required=True,choices=tuple(LADDERS))
    launch.add_argument('--batch-size',required=True,type=int)
    for key in ('data','upstream','config','partitions','cohort-receipt','numeric-prefix','numeric-resume','output','reviewed-helper-sha256'):
        launch.add_argument('--'+key,required=True)
    launch.add_argument('--finite-probe')
    launch.add_argument('--previous')
    args=parser.parse_args(argv)
    if args.command=='budget':
        print(json.dumps(work_budget(),sort_keys=True,allow_nan=False))
    else:
        # Final success is the owned artifact, avoiding a later stdout BrokenPipe failure.
        run(args)

if __name__=='__main__':
    main()
