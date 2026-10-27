"""Sealed full-byte caches: real HTTP, kernel ownership and memory admission."""
from __future__ import annotations

import fcntl
import hashlib
import importlib
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


def module():
    return importlib.import_module('egoemg_repro.emg2pose_memory_archive')


def owned_fds():
    result = set()
    for path in Path('/proc/self/fd').iterdir():
        try:
            if os.readlink(path).startswith('/memfd:egoemg-emg2pose-'):
                result.add(int(path.name))
        except FileNotFoundError:
            pass
    return result


@pytest.fixture
def endpoint(monkeypatch):
    # Non-round length forces an exact short final range. Negative zero/NaN-like
    # bit patterns and arbitrary trailing bytes must survive unmodified.
    data = bytes.fromhex('000000800100c07f') * 4001 + bytes(range(251))
    state = SimpleNamespace(data=data, mode='ok', requests=[], finished=[], heads=0,
                            lock=threading.Lock(), active=0, max_active=0)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            pass

        def do_HEAD(self):
            with state.lock:
                state.heads += 1
                state.requests.append(('HEAD', dict(self.headers)))
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Content-Type', 'application/x-tar')
            self.send_header('ETag', '"changed"' if state.mode == 'final_head' and state.heads > 1 else '"stable"')
            self.send_header('Last-Modified', 'Thu, 01 Jan 2026 00:00:00 GMT')
            self.end_headers()

        def do_GET(self):
            with state.lock:
                state.requests.append(('GET', dict(self.headers)))
                state.active += 1
                state.max_active = max(state.max_active, state.active)
            try:
                start, end = map(int, self.headers['Range'].removeprefix('bytes=').split('-'))
                if start == 0 and state.mode == 'ok':
                    time.sleep(.12)  # Later ranges actually finish first.
                self.send_response(200 if state.mode == 'status200' else 206)
                self.send_header('Content-Length', str(end-start+1))
                self.send_header('Content-Type', 'application/x-tar')
                self.send_header('Content-Range', f'bytes {start+(state.mode == "range")}-{end}/{len(data)}')
                self.send_header('ETag', '"changed"' if state.mode == 'etag' else '"stable"')
                self.send_header('Last-Modified', 'Thu, 01 Jan 2026 00:00:00 GMT')
                self.end_headers()
                body = data[start:end+1]
                self.wfile.write(body[:len(body)//2] if state.mode == 'short' else body)
                if state.mode == 'short':
                    self.close_connection = True
                with state.lock:
                    state.finished.append(start)
            finally:
                with state.lock:
                    state.active -= 1

    class Server(ThreadingHTTPServer):
        def handle_error(self, *_args):
            pass  # Protocol rejection may close a connection before body write.

    from egoemg_repro import emg2pose_archive
    server = Server(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    worker.start()
    monkeypatch.setitem(emg2pose_archive._ASSETS, 'full',
                        (f'http://127.0.0.1:{server.server_port}/source.tar', len(data)))
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def acquire(monkeypatch, **kwargs):
    memory = module()
    monkeypatch.setattr(memory, 'available_memory_bytes', lambda: 2**40)
    return memory.acquire_archive('full', workers=3, chunk_bytes=7001, reserve_bytes=0, **kwargs)


def test_parallel_out_of_order_ranges_preserve_every_original_byte_and_seal(endpoint, monkeypatch):
    before = owned_fds()
    events = []
    with acquire(monkeypatch, on_progress=events.append) as cache:
        descriptor = cache.fileno()
        assert os.pread(descriptor, len(endpoint.data)+100, 0) == endpoint.data
        assert os.fstat(descriptor).st_size == len(endpoint.data)
        result = cache.receipt
        assert result['complete'] is True and result['full_corpus_complete'] is False
        assert result['source_bytes_received'] == len(endpoint.data)
        assert result['full_sha256'] == hashlib.sha256(endpoint.data).hexdigest()
        assert [(x['offset'], x['raw_bytes']) for x in result['chunks']] == [
            (0, 7001), (7001, 7001), (14002, 7001), (21003, 7001), (28004, 4255)]
        assert all(x['raw_sha256'] == hashlib.sha256(endpoint.data[x['offset']:x['offset']+x['raw_bytes']]).hexdigest()
                   for x in result['chunks'])
        with pytest.raises(OSError):
            os.pwrite(descriptor, b'bad', 0)
        with pytest.raises(OSError):
            os.ftruncate(descriptor, len(endpoint.data)+1)
        # Linux UAPI F_GET_SEALS=1024+10; SEAL/SHRINK/GROW/WRITE bits=1/2/4/8.
        # The bundled portable Python omits their symbolic fcntl constants.
        assert fcntl.fcntl(descriptor, 1034) & 15 == 15
        # A caller changing a progress/receipt object cannot change identity.
        result['source']['etag'] = 'tampered'
        assert cache.receipt['source']['etag'] == '"stable"'
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert owned_fds() == before
    assert endpoint.finished[0] != 0
    assert 1 < endpoint.max_active <= 3
    gets = [headers for method, headers in endpoint.requests if method == 'GET']
    assert len(gets) == 5
    assert all(h['If-Match'] == '"stable"' and h['Accept-Encoding'] == 'identity' for h in gets)
    assert [e['source_bytes_received'] for e in events] == sorted(e['source_bytes_received'] for e in events)
    assert all(not e['complete'] and e['full_sha256'] is None for e in events[:-1])
    assert events[-1]['complete'] is True


@pytest.mark.parametrize('mode', ['etag', 'range', 'status200', 'short', 'final_head'])
def test_protocol_failures_never_return_a_ready_cache_or_leak_fd(endpoint, monkeypatch, mode):
    before = owned_fds()
    endpoint.mode = mode
    memory = module()
    events = []
    with pytest.raises(memory.ArchiveError):
        acquire(monkeypatch, on_progress=events.append)
    assert owned_fds() == before
    assert not any(e['complete'] for e in events)


@pytest.mark.parametrize('when', ['first', 'sealed'])
def test_observer_exception_releases_its_owned_cache(endpoint, monkeypatch, when):
    before = owned_fds()

    def observer(event):
        if when == 'first' or event['complete']:
            raise RuntimeError('observer interrupted')

    with pytest.raises(RuntimeError, match='observer interrupted'):
        acquire(monkeypatch, on_progress=observer)
    assert owned_fds() == before


def test_capacity_admission_requires_full_source_plus_reserve_and_buffers(endpoint, monkeypatch):
    memory = module()
    monkeypatch.setattr(memory, 'available_memory_bytes', lambda: len(endpoint.data)+1000-1)
    before = owned_fds()
    with pytest.raises(memory.ArchiveCapacityError):
        memory.acquire_archive('full', workers=3, chunk_bytes=7001, reserve_bytes=1000)
    assert owned_fds() == before
    assert not any(method == 'GET' for method, _ in endpoint.requests)


def test_runtime_memory_pressure_stops_acquisition_and_closes_fd(endpoint, monkeypatch):
    memory = module()
    free = [2**40]
    monkeypatch.setattr(memory, 'available_memory_bytes', lambda: free[0])
    before = owned_fds()
    events = []

    def observer(event):
        events.append(event)
        free[0] = 0

    with pytest.raises(memory.ArchiveCapacityError):
        memory.acquire_archive('full', workers=1, chunk_bytes=7001, reserve_bytes=1000, on_progress=observer)
    assert owned_fds() == before
    assert len(events) == 1 and events[0]['complete'] is False


def test_short_kernel_writes_are_completed_and_read_back(endpoint, monkeypatch):
    memory = module()
    original = memory.os.pwrite
    monkeypatch.setattr(memory.os, 'pwrite', lambda fd, data, offset: original(fd, data[:13], offset))
    with acquire(monkeypatch) as cache:
        assert os.pread(cache.fileno(), len(endpoint.data), 0) == endpoint.data


def test_python_without_builtin_memfd_uses_same_sealed_linux_object(endpoint, monkeypatch):
    memory = module()
    # The actual server's Conda Python exposes neither os.memfd_create nor MFD
    # flags, although its running kernel and libc implement the operation.
    for name in ('memfd_create', 'MFD_ALLOW_SEALING', 'MFD_CLOEXEC'):
        monkeypatch.delattr(memory.os, name, raising=False)
    before = owned_fds()
    with acquire(monkeypatch) as cache:
        descriptor = cache.fileno()
        assert os.pread(descriptor, len(endpoint.data), 0) == endpoint.data
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        with pytest.raises(OSError):
            os.pwrite(descriptor, b'forbidden', 0)
    assert owned_fds() == before


@pytest.mark.parametrize('fault', ['zero', 'corrupt'])
def test_bad_memory_writes_cannot_receive_complete_hash_receipt(endpoint, monkeypatch, fault):
    memory = module()
    original = memory.os.pwrite

    def write(fd, data, offset):
        return 0 if fault == 'zero' else original(fd, b'X'*len(data), offset)

    monkeypatch.setattr(memory.os, 'pwrite', write)
    before = owned_fds()
    with pytest.raises(memory.ArchiveError):
        acquire(monkeypatch)
    assert owned_fds() == before


@pytest.mark.parametrize('key,value', [('workers', 0), ('workers', True), ('workers', 33),
                                      ('chunk_bytes', 0), ('chunk_bytes', 1.5),
                                      ('chunk_bytes', 2**30), ('reserve_bytes', -1),
                                      ('reserve_bytes', False)])
def test_invalid_policy_is_rejected_before_network_io(endpoint, key, value):
    with pytest.raises(ValueError):
        module().acquire_archive('full', **{key: value})
    assert endpoint.requests == []


def memory_files(root):
    meminfo = root/'meminfo'
    meminfo.write_text('MemTotal: 1000000 kB\nMemAvailable: 200 kB\n')
    membership = root/'cgroup'
    membership.write_text('0::/parent/leaf\n')
    cg = root/'hierarchy'
    (cg/'parent/leaf').mkdir(parents=True)
    (cg/'cgroup.controllers').write_text('cpuset cpu io memory pids\n')
    (cg/'memory.current').write_text('1000\n')
    for part in [cg/'parent', cg/'parent/leaf']:
        (part/'memory.max').write_text('max\n')
        (part/'memory.current').write_text('1000\n')
    return dict(meminfo=meminfo, cgroup_file=membership, cgroup_root=cg)


def test_tighter_parent_cgroup_limits_available_memory(tmp_path):
    paths = memory_files(tmp_path)
    parent = paths['cgroup_root']/'parent'
    (parent/'memory.max').write_text('100000\n')
    (parent/'memory.current').write_text('40000\n')
    assert module().available_memory_bytes(**paths) == 60000
    (parent/'memory.current').write_text('100001\n')
    assert module().available_memory_bytes(**paths) == 0


def test_unlimited_cgroups_still_observe_host_available_memory(tmp_path):
    paths = memory_files(tmp_path)
    assert module().available_memory_bytes(**paths) == 204800


def test_missing_memory_controller_at_cgroup_root_is_not_assumed_unlimited(tmp_path):
    paths = memory_files(tmp_path)
    paths['cgroup_file'].write_text('0::/\n')
    (paths['cgroup_root']/'memory.current').unlink()
    (paths['cgroup_root']/'cgroup.controllers').write_text('cpu io pids\n')
    with pytest.raises(module().ArchiveCapacityError):
        module().available_memory_bytes(**paths)


def test_global_v2_root_without_memory_files_still_checks_all_nonroot_limits(tmp_path):
    paths = memory_files(tmp_path)
    (paths['cgroup_root']/'memory.current').unlink()
    parent = paths['cgroup_root']/'parent'
    (parent/'memory.max').write_text('100000\n')
    (parent/'memory.current').write_text('40000\n')
    assert module().available_memory_bytes(**paths) == 60000


def test_finite_namespaced_root_limit_is_not_exempted(tmp_path):
    paths = memory_files(tmp_path)
    (paths['cgroup_root']/'memory.max').write_text('50000\n')
    (paths['cgroup_root']/'memory.current').write_text('30000\n')
    assert module().available_memory_bytes(**paths) == 20000


@pytest.mark.parametrize('fault', ['missing_max', 'bad_max', 'bad_current', 'no_v2', 'traversal', 'no_available'])
def test_unknown_memory_capacity_fails_closed(tmp_path, fault):
    paths = memory_files(tmp_path)
    leaf = paths['cgroup_root']/'parent/leaf'
    if fault == 'missing_max':
        (leaf/'memory.max').unlink()
    elif fault == 'bad_max':
        (leaf/'memory.max').write_text('unlimited')
    elif fault == 'bad_current':
        (leaf/'memory.current').write_text('-5')
    elif fault == 'no_v2':
        paths['cgroup_file'].write_text('1:memory:/parent/leaf\n')
    elif fault == 'traversal':
        paths['cgroup_file'].write_text('0::/parent/../../outside\n')
    else:
        paths['meminfo'].write_text('MemTotal: 1000000 kB\n')
    with pytest.raises(module().ArchiveCapacityError):
        module().available_memory_bytes(**paths)
