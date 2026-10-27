"""Own one cache-consuming process session, including loss of its caller.

The guardian parents the worker and closes transferred cache rights immediately
after launch. Its private control socket is held only by the owner. A worker's
unreaped exit pins its session identity until all consumers have really exited.
These private phase receipts never constitute scientific completion.
"""
from __future__ import annotations

import array
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time


_MAX_PACKET = 65536
_IDENTITY = ('pid', 'start_ticks', 'uid', 'sid', 'pgrp')
_FD_ENV = 'EGOEMG_STAGE_FD_'


class OwnedStageError(RuntimeError):
    """An owned phase failed; its private receipts retain the actual outcome."""


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    if type(pid) is not int or pid <= 0:
        return None
    try:
        root = Path('/proc')/str(pid)
        fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], uid=root.stat().st_uid,
                    sid=int(fields[3]), pgrp=int(fields[2]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(identity):
    if not isinstance(identity, dict) or any(k not in identity for k in _IDENTITY):
        return False
    if any(type(identity[k]) is not int or identity[k] < 0 for k in ('pid','uid','sid','pgrp')):
        return False
    if (identity['pid'] <= 0 or not isinstance(identity['start_ticks'], str)
            or not identity['start_ticks'].isdigit()):
        return False
    current = process_identity(identity['pid'])
    return current is not None and all(current[k] == identity[k] for k in _IDENTITY)


def _linux_pidfd_call(number, *arguments):
    # Some Conda Python/libc builds omit both wrappers on a modern kernel.
    # x86_64 Linux UAPI unistd_64.h defines pidfd_open=434/send_signal=424.
    # Unknown architectures fail; never replace a pidfd signal with numeric kill.
    if os.uname().machine != 'x86_64':
        raise OwnedStageError('pidfd wrappers absent on an unsupported architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    return _linux_pidfd_call(434, ctypes.c_int(pid), ctypes.c_uint(0))


def _pidfd_signal(descriptor, signum):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(descriptor, signum)
    else:
        _linux_pidfd_call(424, ctypes.c_int(descriptor), ctypes.c_int(signum),
                         ctypes.c_void_p(None), ctypes.c_uint(0))


def _signal_process(identity, signum):
    """Never signal a recycled numeric PID, even across the final proc check."""
    if not same_process(identity):
        return False
    try:
        descriptor = _pidfd_open(identity['pid'])
    except ProcessLookupError:
        return False
    try:
        if not same_process(identity):
            return False
        try:
            _pidfd_signal(descriptor, signum)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(descriptor)


def _binding(fd):
    if type(fd) is not int or fd < 0:
        raise ValueError('a live sealed descriptor is required')
    try:
        info = os.fstat(fd)
        seals = fcntl.fcntl(fd, 1034)
    except OSError as error:
        raise ValueError('a live sealed descriptor is required') from error
    if not stat.S_ISREG(info.st_mode) or seals & 15 != 15:
        raise ValueError('all four immutable seals are required')
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, seals=seals)


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _send(channel, value, descriptors=()):
    payload = json.dumps(value, sort_keys=True).encode()
    if len(payload) > _MAX_PACKET:
        raise ValueError('owned phase control packet exceeds 64 KiB')
    ancillary = []
    if descriptors:
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', descriptors))]
    if channel.sendmsg([payload], ancillary) != len(payload):
        raise OwnedStageError('incomplete owned phase control packet')


def _receive(channel):
    rights = []
    try:
        payload, ancillary, flags, _ = channel.recvmsg(
            _MAX_PACKET+1, socket.CMSG_SPACE(32*array.array('i').itemsize),
            socket.MSG_CMSG_CLOEXEC)
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array('i')
                values.frombytes(data[:len(data)//values.itemsize*values.itemsize])
                rights.extend(values)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(payload) > _MAX_PACKET:
            raise OwnedStageError('truncated owned phase control packet')
        return (None if not payload else json.loads(payload)), rights
    except BaseException:
        for fd in rights:
            os.close(fd)
        raise


def _session_members(leader):
    if not same_process(leader):
        raise OwnedStageError('owned worker leader identity is no longer pinned')
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        identity = process_identity(int(entry.name))
        if (identity is not None and identity['sid'] == leader['sid']
                and identity['uid'] == leader['uid'] and identity['state'] != 'Z'):
            members.append(identity)
    return members


def _clean_session(leader, grace, poll):
    started, terminated, escalated = time.monotonic(), set(), False
    while True:
        members = _session_members(leader)
        if not members:
            return escalated
        force = time.monotonic()-started >= grace
        for member in members:
            key = member['pid'], member['start_ticks']
            if force:
                escalated = _signal_process(member, signal.SIGKILL) or escalated
            elif key not in terminated:
                _signal_process(member, signal.SIGTERM)
                terminated.add(key)
        time.sleep(min(poll, .05))


def _worker_exited(child):
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _guardian(channel, output, owner, grace, poll):
    child, leader, rights = None, None, []
    guardian = process_identity()
    try:
        request, rights = _receive(channel)
        if request is None or not same_process(owner):
            raise OwnedStageError('cache owner disappeared before worker launch')
        names = request['descriptor_names']
        if len(names) != len(rights) or len(set(names)) != len(names):
            raise OwnedStageError('received descriptor rights do not match names')
        actual = {name:_binding(fd) for name,fd in zip(names,rights)}
        if actual != request['descriptor_bindings']:
            raise OwnedStageError('received source descriptor identity changed')
        environment = {k:v for k,v in request['environment'].items() if not k.startswith(_FD_ENV)}
        environment.update({_FD_ENV+name:str(fd) for name,fd in zip(names,rights)})
        try:
            with (output/'worker.log').open('xb') as log:
                child = subprocess.Popen(request['argv'], env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True, pass_fds=rights)
        finally:
            for fd in rights:
                os.close(fd)
            rights = []
        leader = process_identity(child.pid)
        if (leader is None or leader['sid'] != child.pid or leader['pgrp'] != child.pid
                or leader['uid'] != owner['uid']):
            raise OwnedStageError('worker did not establish its owned session')
        started = dict(owner=owner, guardian=guardian, worker=leader,
                       descriptor_bindings=actual)
        _atomic(output/'started.json', started)
        _send(channel, dict(event='started', **started))
        status, reason = 'aborted', 'owner_lost'
        while True:
            if not same_process(owner):
                break
            if select.select([channel], [], [], poll)[0]:
                packet, extra = _receive(channel)
                for fd in extra:
                    os.close(fd)
                if packet is None:
                    break
                if packet.get('event') == 'abort':
                    reason = 'owner_abort'
                    break
                raise OwnedStageError('unexpected owner control message')
            if _worker_exited(child):
                status, reason = 'finished', 'worker_exit'
                break
        escalation = _clean_session(leader, grace, poll)
        code = child.wait()
        if status == 'finished':
            status = 'success' if code == 0 else 'failed'
        result = dict(format='egoemg_owned_phase_guardian_v1', status=status, reason=reason,
            owner=owner, guardian=guardian, worker=leader, worker_exit_code=code,
            all_owned_consumers_exited=True, kill_escalation_used=escalation,
            scientific_complete=False)
        _atomic(output/'guardian-result.json', result)
        try:
            _send(channel, dict(event='result', result=result))
        except (BrokenPipeError, ConnectionResetError):
            pass
        return 0
    except BaseException as error:
        clean, code = child is None, None
        if child is not None and leader is not None:
            _clean_session(leader, grace, poll)
            code = child.wait(); clean = True
        result = dict(format='egoemg_owned_phase_guardian_v1', status='failed',
            reason='guardian_error', error_type=type(error).__name__, owner=owner,
            guardian=guardian, worker=leader, worker_exit_code=code,
            all_owned_consumers_exited=clean, scientific_complete=False)
        _atomic(output/'guardian-result.json', result)
        try:
            _send(channel, dict(event='result', result=result))
        except (BrokenPipeError, ConnectionResetError):
            pass
        return 1
    finally:
        for fd in rights:
            os.close(fd)
        channel.close()


def run_owned_stage(argv, output, *, descriptors, guard, environment=None,
                    poll_seconds=.25, terminate_grace=5.):
    """Run one phase; return only after all its session consumers have exited.

    The guard raises when source/resource admission is lost. Named immutable
    rights reach the worker as EGOEMG_STAGE_FD_<NAME>; input fds remain caller
    owned. SIGKILL of this caller is handled independently by its guardian.
    """
    if (not isinstance(argv, (list,tuple)) or not argv
            or any(not isinstance(v,str) or not v or '\0' in v for v in argv)):
        raise ValueError('a nonempty command argument sequence is required')
    for value, name, zero in ((poll_seconds,'poll_seconds',False),(terminate_grace,'terminate_grace',True)):
        if (type(value) not in (int,float) or not math.isfinite(value)
                or value < 0 or (not zero and value == 0)):
            raise ValueError(name+' must be finite and positive (grace may be zero)')
    if not callable(guard) or not isinstance(descriptors, dict) or len(descriptors) > 32:
        raise ValueError('a guard and at most 32 named descriptor rights are required')
    if any(not isinstance(k,str) or re.fullmatch('[A-Z][A-Z0-9_]*',k) is None for k in descriptors):
        raise ValueError('descriptor names must be uppercase identifiers')
    bindings = {name:_binding(fd) for name,fd in descriptors.items()}
    if not all((hasattr(os,'WNOWAIT'),hasattr(socket,'MSG_CMSG_CLOEXEC'))):
        raise OwnedStageError('Linux pidfd/waitid/descriptor-rights support is required')
    descriptor = _pidfd_open(os.getpid())
    try:
        _pidfd_signal(descriptor, 0)
    finally:
        os.close(descriptor)
    child_environment = dict(os.environ if environment is None else environment)
    if any(not isinstance(k,str) or not isinstance(v,str) or '\0' in k+v or '=' in k
           for k,v in child_environment.items()):
        raise ValueError('worker environment must contain valid string pairs')
    request = dict(argv=list(argv),environment=child_environment,
                   descriptor_names=list(descriptors),descriptor_bindings=bindings)
    if len(json.dumps(request,sort_keys=True).encode()) > _MAX_PACKET:
        raise ValueError('owned phase control packet exceeds 64 KiB')
    guard()
    output = Path(output).resolve(); output.mkdir(mode=0o700)
    owner, guardian, result, started = process_identity(), None, None, None
    channel, remote = socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
    handlers = {}
    def interrupted(signum, _):
        raise OwnedStageError('owner received signal '+str(signum))
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM,signal.SIGINT):
                handlers[signum] = signal.signal(signum,interrupted)
        guardian_environment = dict(os.environ,CUDA_VISIBLE_DEVICES='')
        guardian_environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')
        with (output/'guardian.log').open('xb') as log:
            guardian = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--guardian',
                str(remote.fileno()),str(output),json.dumps(owner),str(terminate_grace),
                str(poll_seconds)],pass_fds=(remote.fileno(),),close_fds=True,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=guardian_environment)
        remote.close()
        _send(channel,request,tuple(descriptors.values()))
        while result is None:
            guard()
            if select.select([channel],[],[],poll_seconds)[0]:
                packet, extra = _receive(channel)
                for fd in extra:
                    os.close(fd)
                if packet is None:
                    raise OwnedStageError('guardian exited without a terminal receipt')
                if packet['event'] == 'started':
                    started = packet
                elif packet['event'] == 'result':
                    result = packet['result']
                else:
                    raise OwnedStageError('unexpected guardian message')
            elif guardian.poll() is not None:
                raise OwnedStageError('guardian exited before a terminal receipt')
        while guardian.poll() is None:
            guard(); time.sleep(min(poll_seconds,.05))
        guard()
        if (guardian.returncode != 0 or result['status'] != 'success'
                or not result['all_owned_consumers_exited']):
            raise OwnedStageError('owned worker failed or was aborted')
        result = dict(result,format='egoemg_owned_stage_v1',descriptor_bindings=bindings)
        _atomic(output/'stage-result.json',result)
        return result
    except BaseException as error:
        for signum in handlers:
            signal.signal(signum,signal.SIG_IGN)
        try:
            if guardian is not None and guardian.poll() is None:
                try:
                    _send(channel,dict(event='abort'))
                except (BrokenPipeError,ConnectionResetError):
                    pass
            channel.close(); remote.close()
            if guardian is not None:
                while guardian.poll() is None:
                    time.sleep(min(poll_seconds,.05))
            receipt = output/'guardian-result.json'
            result = json.loads(receipt.read_text()) if receipt.exists() else dict(
                owner=owner,worker=None,worker_exit_code=None,all_owned_consumers_exited=guardian is None)
            result = dict(result,format='egoemg_owned_stage_v1',status='failed',
                          error_type=type(error).__name__,scientific_complete=False,
                          descriptor_bindings=bindings)
            _atomic(output/'stage-result.json',result)
        finally:
            raise
    finally:
        channel.close(); remote.close()
        for signum, handler in handlers.items():
            signal.signal(signum,handler)


if __name__ == '__main__':
    if len(sys.argv) != 7 or sys.argv[1] != '--guardian':
        raise SystemExit('private guardian entry point; use run_owned_stage')
    raise SystemExit(_guardian(socket.socket(fileno=int(sys.argv[2])),Path(sys.argv[3]),
                              json.loads(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])))
