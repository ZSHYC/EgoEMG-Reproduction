"""Real process/descriptor regressions for the volatile-cache supervisor.

Omitting owner-EOF handling must fail the SIGKILL case; omitting session cleanup
must fail both descendant-exit cases; omitting guard polling must fail pressure.
No model, network, large allocation, or simulated positive process is used.
"""
import ctypes
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


CORPUS = b'sealed-corpus-bytes' * 17
REPO = Path(__file__).resolve().parents[1]
WORKER = '''import fcntl,json,os,signal,sys,time
from pathlib import Path
mode,ready=sys.argv[1:]
fd=int(os.environ['EGOEMG_STAGE_FD_ARCHIVE'])
data=os.pread(fd,65536,0)
grandchild=None
if mode in ('descendant','failure','wait','resistant'):
 grandchild=os.fork()
 if grandchild==0:
  if mode=='resistant':signal.signal(signal.SIGTERM,signal.SIG_IGN)
  while True:time.sleep(.02)
if mode=='resistant':signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(ready).write_text(json.dumps(dict(pid=os.getpid(),grandchild=grandchild,fd=fd,
 bytes=data.hex(),sid=os.getsid(0),pgrp=os.getpgrp(),seals=fcntl.fcntl(fd,1034))))
print('worker read sealed bytes',flush=True)
if mode=='failure':raise SystemExit(7)
if mode in ('wait','resistant'):
 while True:time.sleep(.02)
'''
OWNER = '''import ctypes,fcntl,os,sys
from pathlib import Path
from egoemg_repro.emg2pose_memory_process import run_owned_stage
worker,mode,ready,output,pressure=sys.argv[1:]
fd=ctypes.CDLL(None,use_errno=True).memfd_create(b'egoemg-process-fixture',3)
if fd<0:raise OSError(ctypes.get_errno(),'memfd_create')
os.write(fd,b'sealed-corpus-bytes'*17);fcntl.fcntl(fd,1033,15)
def guard():
 if Path(pressure).exists():raise RuntimeError('observed capacity pressure')
try:
 run_owned_stage([sys.executable,worker,mode,ready],Path(output),
   descriptors={'ARCHIVE':fd},guard=guard,poll_seconds=.02,terminate_grace=.15)
finally:os.close(fd)
'''


def memfd(name):
    fd = ctypes.CDLL(None,use_errno=True).memfd_create(name.encode(),3)
    if fd < 0:
        raise OSError(ctypes.get_errno(),'memfd_create')
    return fd


@pytest.fixture
def sealed():
    fd = memfd('egoemg-process-fixture')
    os.write(fd, CORPUS)
    fcntl.fcntl(fd, 1033, 15)
    try:
        yield fd
    finally:
        os.close(fd)


@pytest.fixture
def worker(tmp_path):
    path = tmp_path/'worker.py'
    path.write_text(WORKER)
    return path


def wait_for(path, timeout=10):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(.02)
    raise AssertionError(f'no complete receipt at {path}')


def live(pid):
    p = Path('/proc')/str(pid)/'stat'
    if not p.exists():
        return False
    return p.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def assert_consumers_gone(receipt):
    deadline = time.monotonic()+5
    pids = [receipt['pid'], receipt['grandchild']]
    while any(live(p) for p in pids if p) and time.monotonic() < deadline:
        time.sleep(.02)
    assert not any(live(p) for p in pids if p), pids


def owner_process(tmp_path, worker, mode):
    owner = tmp_path/'owner.py'
    owner.write_text(OWNER)
    ready, output, pressure = (tmp_path/n for n in ('ready.json','stage','pressure'))
    env = dict(os.environ, PYTHONPATH=str(REPO))
    log = (tmp_path/'owner.log').open('wb')
    proc = subprocess.Popen([sys.executable,str(owner),str(worker),mode,str(ready),
                             str(output),str(pressure)],env=env,stdout=log,
                            stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    return proc, ready, output, pressure


def test_named_rights_read_actual_sealed_bytes_and_keep_caller_fd(sealed,worker,tmp_path):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    ready, output = tmp_path/'ready.json', tmp_path/'stage'
    result = api.run_owned_stage([sys.executable,str(worker),'read',str(ready)],output,
        descriptors={'ARCHIVE':sealed},guard=lambda:None,poll_seconds=.02,terminate_grace=.15)
    actual = wait_for(ready)
    assert bytes.fromhex(actual['bytes']) == CORPUS
    assert actual['seals'] & 15 == 15
    assert result['status'] == 'success' and result['worker_exit_code'] == 0
    assert result['scientific_complete'] is False
    assert actual['sid'] == actual['pgrp'] == actual['pid']
    assert os.pread(sealed,len(CORPUS),0) == CORPUS
    assert not (output/'completion.json').exists()
    assert 'worker read sealed bytes' in (output/'worker.log').read_text()
    assert json.loads((output/'stage-result.json').read_text()) == result


@pytest.mark.parametrize('mode',['descendant','failure'])
def test_worker_exit_cleans_still_live_cache_consumers_before_return(sealed,worker,tmp_path,mode):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    ready, output = tmp_path/'ready.json', tmp_path/'stage'
    command = [sys.executable,str(worker),mode,str(ready)]
    if mode == 'failure':
        with pytest.raises(api.OwnedStageError):
            api.run_owned_stage(command,output,descriptors={'ARCHIVE':sealed},
                guard=lambda:None,poll_seconds=.02,terminate_grace=.15)
    else:
        api.run_owned_stage(command,output,descriptors={'ARCHIVE':sealed},
            guard=lambda:None,poll_seconds=.02,terminate_grace=.15)
    assert_consumers_gone(wait_for(ready))
    result = json.loads((output/'stage-result.json').read_text())
    assert result['worker_exit_code'] == (7 if mode == 'failure' else 0)
    assert result['status'] == ('failed' if mode == 'failure' else 'success')
    assert result['all_owned_consumers_exited'] is True


@pytest.mark.parametrize('exception',[RuntimeError,OSError])
def test_live_pressure_or_unknown_capacity_aborts_and_closes_worker_rights(sealed,worker,tmp_path,exception):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    ready, output = tmp_path/'ready.json', tmp_path/'stage'
    def guard():
        if ready.exists():
            raise exception('capacity is no longer qualified')
    with pytest.raises(exception,match='capacity is no longer qualified'):
        api.run_owned_stage([sys.executable,str(worker),'wait',str(ready)],output,
            descriptors={'ARCHIVE':sealed},guard=guard,poll_seconds=.02,terminate_grace=.15)
    assert_consumers_gone(wait_for(ready))
    result = json.loads((output/'stage-result.json').read_text())
    assert result['status'] == 'failed' and result['all_owned_consumers_exited']
    assert not (output/'completion.json').exists()


@pytest.mark.parametrize('signum',[signal.SIGKILL,signal.SIGTERM,signal.SIGINT])
def test_holder_death_releases_worker_and_grandchild_without_success(worker,tmp_path,signum):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    proc, ready, output, _ = owner_process(tmp_path,worker,'wait')
    try:
        consumers = wait_for(ready)
        started = wait_for(output/'started.json')
        guardian = started['guardian']
        assert api.same_process(guardian)
        # A resident guardian copy would keep the large memfd alive after cleanup.
        target = Path('/proc')/str(consumers['pid'])/'fd'/str(consumers['fd'])
        expected = target.stat()
        for p in (Path('/proc')/str(guardian['pid'])/'fd').iterdir():
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            assert (st.st_dev,st.st_ino) != (expected.st_dev,expected.st_ino)
        os.kill(proc.pid,signum)
        proc.wait(timeout=10)
        assert_consumers_gone(consumers)
        result = wait_for(output/'guardian-result.json')
        assert result['all_owned_consumers_exited']
        assert result['status'] == 'aborted'
        if (output/'stage-result.json').exists():
            assert json.loads((output/'stage-result.json').read_text())['status'] == 'failed'
        assert not (output/'completion.json').exists()
    finally:
        if proc.poll() is None:
            proc.kill();proc.wait(timeout=10)


def test_term_resistant_owned_session_is_killed_on_pressure(worker,tmp_path):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    proc, ready, output, pressure = owner_process(tmp_path,worker,'resistant')
    try:
        consumers = wait_for(ready)
        pressure.touch()
        proc.wait(timeout=10)
        assert proc.returncode != 0
        assert_consumers_gone(consumers)
        result = wait_for(output/'guardian-result.json')
        assert result['all_owned_consumers_exited']
        assert result['kill_escalation_used'] is True
    finally:
        if proc.poll() is None:
            proc.kill();proc.wait(timeout=10)


@pytest.mark.parametrize('mode',['wait','resistant','nested'])
def test_guardian_only_death_preserves_owned_identity_and_cleans_every_consumer(worker,tmp_path,mode):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    if mode == 'nested':
        nested = tmp_path/'nested.py'
        nested.write_text('''import os,sys
from pathlib import Path
from egoemg_repro.emg2pose_memory_process import run_owned_stage
run_owned_stage([sys.executable,WORKER,'wait',sys.argv[2]],Path(sys.argv[2]).parent/'inner-stage',
 descriptors={'ARCHIVE':int(os.environ['EGOEMG_STAGE_FD_ARCHIVE'])},guard=lambda:None,
 poll_seconds=.02,terminate_grace=.15)
'''.replace('WORKER',repr(str(worker))))
        worker = nested
    proc,ready,output,_ = owner_process(tmp_path,worker,mode)
    consumers,started = None,None
    sentinel = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        consumers = wait_for(ready);started = wait_for(output/'started.json')
        if mode == 'nested':wait_for(tmp_path/'inner-stage/started.json')
        os.kill(started['guardian']['pid'],signal.SIGKILL)
        proc.wait(timeout=10)
        assert proc.returncode != 0
        result = wait_for(output/'stage-result.json')
        assert result['worker'] == started['worker']
        assert result['all_owned_consumers_exited'] is True
        assert result['status'] == 'failed' and result['scientific_complete'] is False
        assert_consumers_gone(consumers)
        assert not live(started['worker']['pid'])
        if mode == 'nested':
            inner = wait_for(tmp_path/'inner-stage/guardian-result.json')
            assert inner['all_owned_consumers_exited'] and inner['status'] == 'aborted'
        assert sentinel.poll() is None
    finally:
        # The RED run must not retain its intentionally orphaned test workers.
        pids = [p for p in ((consumers or {}).get('grandchild'),(consumers or {}).get('pid'),
            ((started or {}).get('worker') or {}).get('pid')) if p]
        for pid in pids:
            if live(pid):
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        if proc.poll() is None:proc.kill();proc.wait(timeout=10)
        sentinel.terminate();sentinel.wait(timeout=10)


def test_guardian_loss_before_owner_receives_start_packet_uses_registered_identity(sealed,worker,tmp_path):
    api=importlib.import_module('egoemg_repro.emg2pose_memory_process')
    output,ready=tmp_path/'stage',tmp_path/'ready.json'
    registration={}
    def guard():
        if output.exists():
            registration.update(wait_for(output/'started.json'))
            os.kill(registration['guardian']['pid'],signal.SIGKILL)
            raise RuntimeError('actual guardian lost before owner reads packet')
    with pytest.raises(RuntimeError,match='before owner reads packet'):
        api.run_owned_stage([sys.executable,str(worker),'wait',str(ready)],output,
            descriptors={'ARCHIVE':sealed},guard=guard,poll_seconds=.02,terminate_grace=.15)
    result=wait_for(output/'stage-result.json')
    assert result['worker']==registration['worker'] and result['all_owned_consumers_exited']
    assert result['status']=='failed' and not live(registration['worker']['pid'])
    if ready.exists():assert_consumers_gone(wait_for(ready))


def test_registered_fallback_refuses_a_reused_or_stale_leader_identity():
    api=importlib.import_module('egoemg_repro.emg2pose_memory_process')
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        stale=dict(api.process_identity(sentinel.pid),start_ticks='0')
        with pytest.raises(api.OwnedStageError,match='reused'):
            api._clean_session(stale,.01,.01,registered=True)
        assert sentinel.poll() is None
    finally:sentinel.terminate();sentinel.wait(timeout=10)


def test_stale_start_ticks_cannot_signal_unrelated_live_sentinel():
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    proc = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        current = api.process_identity(proc.pid)
        assert api.same_process(current)
        stale = dict(current,start_ticks=str(int(current['start_ticks'])+1))
        assert not api.same_process(stale)
        assert api._signal_process(stale,signal.SIGKILL) is False
        assert proc.poll() is None
    finally:
        proc.terminate();proc.wait(timeout=10)


@pytest.mark.parametrize('signum',[0,signal.SIGTERM])
def test_real_pidfd_signal_works_without_python_wrappers(monkeypatch,signum):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    proc = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        identity = api.process_identity(proc.pid)
        monkeypatch.delattr(os,'pidfd_open',raising=False)
        monkeypatch.delattr(signal,'pidfd_send_signal',raising=False)
        assert api._signal_process(identity,signum) is True
        if signum == 0:
            assert proc.poll() is None
        else:
            assert proc.wait(timeout=10) == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.terminate();proc.wait(timeout=10)


def test_initial_guard_failure_never_launches_worker(sealed,worker,tmp_path):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    ready, output = tmp_path/'ready.json', tmp_path/'stage'
    def guard():
        raise RuntimeError('initial reserve refused')
    with pytest.raises(RuntimeError,match='initial reserve refused'):
        api.run_owned_stage([sys.executable,str(worker),'read',str(ready)],output,
            descriptors={'ARCHIVE':sealed},guard=guard)
    assert not ready.exists()
    assert os.pread(sealed,len(CORPUS),0) == CORPUS


def test_existing_output_is_not_overwritten_or_executed(sealed,worker,tmp_path):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    output, ready = tmp_path/'stage', tmp_path/'ready.json'
    output.mkdir();(output/'sentinel').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        api.run_owned_stage([sys.executable,str(worker),'read',str(ready)],output,
            descriptors={'ARCHIVE':sealed},guard=lambda:None)
    assert (output/'sentinel').read_bytes() == b'keep'
    assert not ready.exists()


def test_unsealed_descriptor_is_rejected_before_worker_launch(worker,tmp_path):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    fd = memfd('unsealed')
    try:
        with pytest.raises(ValueError,match='seal'):
            api.run_owned_stage([sys.executable,str(worker),'read',str(tmp_path/'ready.json')],
                tmp_path/'stage',descriptors={'ARCHIVE':fd},guard=lambda:None)
        assert not (tmp_path/'ready.json').exists()
    finally:
        os.close(fd)


@pytest.mark.parametrize('options',[{'poll_seconds':0},{'poll_seconds':float('nan')},
    {'terminate_grace':-1},{'descriptors':{'bad-key':1}},{'argv':[]}])
def test_invalid_policy_refuses_before_creating_stage(sealed,tmp_path,options):
    api = importlib.import_module('egoemg_repro.emg2pose_memory_process')
    kwargs=dict(argv=[sys.executable,'-c','raise SystemExit(0)'],output=tmp_path/'stage',
                descriptors={'ARCHIVE':sealed},guard=lambda:None)
    kwargs.update(options)
    with pytest.raises((TypeError,ValueError)):
        api.run_owned_stage(**kwargs)
    assert not (tmp_path/'stage').exists()
