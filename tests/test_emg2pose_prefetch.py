"""Offline transport at the HTTP boundary; the real frozen archive writer runs."""
from __future__ import annotations

import hashlib
import importlib
import json
import lzma
import random
import threading
import types
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import requests

from egoemg_repro import emg2pose_archive as archive


class Response:
    def __init__(self, body, *, status=206, headers=None, fail=None, gate=None):
        self.body, self.status_code = body, status
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.fail, self.gate = fail, gate
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.closed = True
        if self.gate:
            self.gate.set()

    def iter_content(self, chunk_size):
        if self.gate:
            assert self.gate.wait(3), 'fixture gate did not release'
        if self.fail:
            raise self.fail
        for i in range(0, len(self.body), chunk_size):
            if self.closed:
                raise requests.ConnectionError('closed fixture response')
            yield self.body[i:i+chunk_size]


class Transport:
    """No socket: independently specified official-shaped responses and bytes."""
    def __init__(self):
        self.data = random.Random(43).randbytes(12_315)
        self.etag, self.modified = '"offline-fixture"', 'Thu, 01 Jan 2026 00:00:00 GMT'
        self.calls, self.responses, self.sessions, self.completed = [], [], [], []
        self.mode, self.hook = 'ok', None
        self.lock = threading.Lock()

    def factory(self):
        owner = self

        class Session:
            def __init__(self):
                self.trust_env, self.auth = True, ('ambient', 'forbidden')
                self.cookies, self.proxies = {'ambient': 'forbidden'}, {'ambient': 'forbidden'}
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

            def close(self):
                self.closed = True

            def request(self, method, url, **kwargs):
                with owner.lock:
                    owner.calls.append((method, url, dict(kwargs), threading.get_ident()))
                assert not self.trust_env and self.auth is None and not self.cookies and not self.proxies
                if method == 'HEAD':
                    body, start, end, status = b'', 0, len(owner.data)-1, 200
                    size = len(owner.data)
                else:
                    start, end = map(int, kwargs['headers']['Range'][6:].split('-'))
                    body, status, size = owner.data[start:end+1], 206, end-start+1
                headers = {'Content-Length': str(size), 'Content-Type': 'application/x-tar',
                           'ETag': owner.etag, 'Last-Modified': owner.modified,
                           'X-Actual-Fixture': f'actual-{method}-{start}'}
                if method == 'GET':
                    headers['Content-Range'] = f'bytes {start}-{end}/{len(owner.data)}'
                result = Response(body, status=status, headers=headers)
                if method == 'GET' and owner.hook:
                    owner.hook(start, result)
                owner.responses.append(result)
                return result

        session = Session()
        self.sessions.append(session)
        return session


@pytest.fixture
def prefetch():
    return importlib.import_module('egoemg_repro.emg2pose_prefetch')


@pytest.fixture
def source(prefetch, monkeypatch):
    transport = Transport()
    monkeypatch.setitem(archive._ASSETS, 'full', (archive._ASSETS['full'][0], len(transport.data)))
    monkeypatch.setattr(prefetch, '_new_session', transport.factory)
    monkeypatch.setattr(archive.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**12))
    return transport


def acquire(prefetch, path, **kwargs):
    settings = dict(route=prefetch.ExplicitRoute('socks5h://127.0.0.1:12345'),
                    chunk_bytes=4096, budget_bytes=10**8, reserve_bytes=0)
    settings.update(kwargs)
    return prefetch.acquire_prefetched(path, **settings)


def snapshot():
    return (archive.acquire_archive, archive.acquire_archive.__code__, archive._receive_range,
            archive._head, archive._request, archive.requests, requests.Session,
            dict(archive.acquire_archive.__globals__))


def test_single_worker_preserves_exact_original_writer_and_actual_headers(prefetch, source, tmp_path):
    # Baseline is the original writer with a test-local Session factory, never a
    # patched global. Expected raw bytes and xz6 streams are independent literals.
    original = archive.acquire_archive
    scope = dict(original.__globals__, requests=SimpleNamespace(Session=source.factory))
    baseline = types.FunctionType(original.__code__, scope, argdefs=original.__defaults__)
    baseline.__kwdefaults__ = original.__kwdefaults__.copy()
    a = baseline(tmp_path/'baseline', chunk_bytes=4096, budget_bytes=10**8, reserve_bytes=0)
    before = snapshot()
    receipt = {}
    b = acquire(prefetch, tmp_path/'prefetched', workers=1, receipt=receipt)
    assert a == b and b['complete'] and b['full_sha256'] == hashlib.sha256(source.data).hexdigest()
    assert snapshot() == before
    for entry in b['chunks']:
        raw = source.data[entry['offset']:entry['offset']+entry['raw_bytes']]
        want = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
        assert (tmp_path/'prefetched'/entry['path']).read_bytes() == want
        assert (tmp_path/'baseline'/entry['path']).read_bytes() == want
    with archive.read_archive(tmp_path/'prefetched') as stream:
        assert stream.read() == source.data
    assert receipt['delivered_raw_bytes'] == 12_315
    assert all(x['headers']['X-Actual-Fixture'] == f"actual-GET-{x['offset']}" for x in receipt['ranges'])
    assert all(s.closed for s in source.sessions) and all(r.closed for r in source.responses)


@pytest.mark.parametrize('drift', ['hash', 'signature', 'code'])
def test_archive_drift_refused_before_session_or_output(prefetch, source, tmp_path, monkeypatch, drift):
    if drift == 'hash':
        monkeypatch.setattr(prefetch, 'ARCHIVE_SHA256', '0'*64)
    elif drift == 'signature':
        monkeypatch.setattr(archive, 'acquire_archive', lambda output: None)
    else:
        original = archive.acquire_archive
        scope = dict(original.__globals__)
        changed = types.FunctionType(original.__code__.replace(co_consts=original.__code__.co_consts+('drift',)),
                                     scope, argdefs=original.__defaults__)
        changed.__kwdefaults__ = original.__kwdefaults__.copy()
        monkeypatch.setattr(archive, 'acquire_archive', changed)
    with pytest.raises(archive.ArchiveError, match='pin|signature|code|hash'):
        acquire(prefetch, tmp_path/'no-output')
    assert not source.sessions and not (tmp_path/'no-output').exists()


def test_held_writer_lock_does_not_create_transport(prefetch, source, tmp_path):
    with archive._lock(tmp_path/'locked', write=True):
        with pytest.raises(archive.ArchiveError, match='lock'):
            acquire(prefetch, tmp_path/'locked')
    assert not source.sessions and not source.calls


def test_failed_strict_reader_leaves_globals_and_no_complete(prefetch, source, tmp_path):
    source.hook = lambda _, response: setattr(response, 'status_code', 200)
    before = snapshot()
    with pytest.raises(archive.ArchiveError, match='206'):
        acquire(prefetch, tmp_path/'failed', max_chunks=1)
    assert snapshot() == before
    m = json.loads((tmp_path/'failed/manifest.json').read_text())
    assert m['chunks'] == [] and not m['complete'] and m['full_sha256'] is None
    assert all(s.closed for s in source.sessions) and all(r.closed for r in source.responses)


@pytest.mark.parametrize('route', ['http://127.0.0.1:1234', 'socks5h://example.org:1234',
                                  'socks5h://u:p@127.0.0.1:1234', 'socks5h://127.0.0.1:1234/x'])
def test_route_refuses_ambient_or_remote_credentials(prefetch, route):
    with pytest.raises(ValueError):
        prefetch.ExplicitRoute(route)


def test_two_workers_out_of_order_are_bounded_and_writer_stays_ordered(prefetch, source, tmp_path):
    first_gate, second_finished = threading.Event(), threading.Event()
    def hook(start, response):
        if start == 0:
            response.gate = first_gate
        original = response.iter_content
        def content(chunk_size):
            yield from original(chunk_size)
            source.completed.append(start)
            if start == 4096:
                second_finished.set()
        response.iter_content = content
    source.hook = hook
    receipt = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acquire, prefetch, tmp_path/'two', workers=2, receipt=receipt,
                             raw_memory_budget_bytes=32_768)
        try:
            assert second_finished.wait(2), 'second worker did not independently finish'
            gets = [call for call in source.calls if call[0] == 'GET']
            assert len(gets) == 2  # no third range while first delivery is blocked
        finally:
            first_gate.set()
        m = future.result(timeout=5)
    assert source.completed[:2] == [4096, 0]
    assert [x['offset'] for x in m['chunks']] == [0, 4096, 8192, 12288]
    assert receipt['peak_pending_chunks'] == 2 and receipt['peak_reserved_raw_bytes'] <= 32_768
    assert receipt['pending_chunks_at_close'] == receipt['reserved_raw_bytes_at_close'] == 0
    with archive.read_archive(tmp_path/'two') as stream:
        assert stream.read() == source.data
    for method, _, kwargs, _ in source.calls:
        assert kwargs['timeout'] == (15, 60) and kwargs['allow_redirects'] is False
        assert kwargs['stream'] is True and kwargs['verify'] is True
        assert kwargs['proxies'] == {'http': 'socks5h://127.0.0.1:12345',
                                      'https': 'socks5h://127.0.0.1:12345'}
        if method == 'GET':
            assert kwargs['headers']['If-Match'] == source.etag
            assert kwargs['headers']['Accept-Encoding'] == 'identity'


def test_two_workers_never_fetch_beyond_explicit_chunk_limit(prefetch, source, tmp_path):
    receipt = {}
    m = acquire(prefetch, tmp_path/'one-only', workers=2, max_chunks=1, receipt=receipt)
    assert not m['complete'] and m['source_bytes_received'] == 4096
    assert [x[2]['headers']['Range'] for x in source.calls if x[0] == 'GET'] == ['bytes=0-4095']
    assert receipt['peak_pending_chunks'] == 1


@pytest.mark.parametrize('kwargs', [{'workers': 3}, {'workers': 2, 'max_pending_chunks': 1},
                                   {'workers': 2, 'max_pending_chunks': 3},
                                   {'workers': 2, 'raw_memory_budget_bytes': 32_767},
                                   {'workers': 1, 'max_pending_chunks': 1,
                                    'raw_memory_budget_bytes': 16_384}])
def test_unsafe_concurrency_or_raw_budget_fails_before_output(prefetch, source, tmp_path, kwargs):
    with pytest.raises(ValueError):
        acquire(prefetch, tmp_path/'unsafe', **kwargs)
    assert not source.calls and not (tmp_path/'unsafe').exists()


def test_first_get_cannot_switch_to_other_official_url(prefetch, source):
    receipt = {'wire_body_bytes': 0, 'delivered_raw_bytes': 0, 'ranges': []}
    with prefetch._PrefetchSession(route=prefetch.ExplicitRoute('socks5h://127.0.0.1:12345'),
                                   chunk_bytes=4096, max_chunks=1, receipt=receipt) as session:
        identity = archive._head(session, archive._ASSETS['full'][0], len(source.data))
        identity['url'] = archive._ASSETS['mini'][0]
        with pytest.raises(archive.ArchiveError, match='source|URL'):
            archive._receive_range(session, identity, 0, 4096)
    assert not [x for x in source.calls if x[0] == 'GET']


@pytest.mark.parametrize('bad', ['duplicate', 'skip', 'size', 'etag', 'method'])
def test_provider_rejects_unexpected_get_without_fetch(prefetch, source, bad):
    receipt = {'wire_body_bytes': 0, 'delivered_raw_bytes': 0, 'ranges': []}
    with prefetch._PrefetchSession(route=prefetch.ExplicitRoute('socks5h://127.0.0.1:12345'),
                                   chunk_bytes=4096, max_chunks=1, receipt=receipt) as session:
        identity = archive._head(session, archive._ASSETS['full'][0], len(source.data))
        assert archive._receive_range(session, identity, 0, 4096) == source.data[:4096]
        headers = {'Range': 'bytes=4096-8191', 'If-Match': source.etag, 'Accept-Encoding': 'identity'}
        method = 'GET'
        if bad == 'duplicate': headers['Range'] = 'bytes=0-4095'
        if bad == 'skip': headers['Range'] = 'bytes=8192-12287'
        if bad == 'size': headers['Range'] = 'bytes=4096-4100'
        if bad == 'etag': headers['If-Match'] = '"changed"'
        if bad == 'method': method = 'POST'
        before = len(source.calls)
        with pytest.raises(archive.ArchiveError):
            session.request(method, identity['url'], headers=headers, timeout=(15, 60),
                            stream=True, allow_redirects=False)
        assert len(source.calls) == before


@pytest.mark.parametrize('fault', ['status', 'range', 'total', 'length', 'etag', 'modified',
                                  'encoding', 'transfer', 'html', 'short', 'long', 'interrupt', 'timeout'])
def test_actual_strict_protocol_fault_never_delivers_or_commits(prefetch, source, tmp_path, fault):
    def hook(start, response):
        if fault == 'status': response.status_code = 200
        if fault == 'range': response.headers['Content-Range'] = 'bytes 1-4096/12315'
        if fault == 'total': response.headers['Content-Range'] = 'bytes 0-4095/12316'
        if fault == 'length': response.headers['Content-Length'] = '4097'
        if fault == 'etag': response.headers['ETag'] = '"changed"'
        if fault == 'modified': response.headers['Last-Modified'] = 'changed'
        if fault == 'encoding': response.headers['Content-Encoding'] = 'gzip'
        if fault == 'transfer': response.headers['Transfer-Encoding'] = 'chunked'
        if fault == 'html': response.body = b'<!DOCTYPE html>'.ljust(4096, b' ')
        if fault == 'short': response.body = response.body[:-1]
        if fault == 'long': response.body += b'X'
        if fault == 'interrupt': response.fail = requests.ConnectionError('fixture interruption')
        if fault == 'timeout': raise requests.Timeout('fixture timeout')
    source.hook = hook
    receipt = {}
    with pytest.raises(archive.ArchiveError):
        acquire(prefetch, tmp_path/fault, max_chunks=1, receipt=receipt)
    m = json.loads((tmp_path/fault/'manifest.json').read_text())
    assert not m['complete'] and m['chunks'] == [] and m['source_bytes_received'] == 0
    assert m['full_sha256'] is None and receipt['delivered_raw_bytes'] == 0
    assert receipt['pending_chunks_at_close'] == receipt['reserved_raw_bytes_at_close'] == 0
    assert all(s.closed for s in source.sessions) and all(r.closed for r in source.responses)


def test_main_original_reader_rechecks_actual_buffered_headers(prefetch, source, tmp_path, monkeypatch):
    original = prefetch._BufferedResponse.__init__
    def tampered(self, data, evidence):
        original(self, data, evidence)
        self.headers['ETag'] = '"tampered-after-worker"'
    monkeypatch.setattr(prefetch._BufferedResponse, '__init__', tampered)
    with pytest.raises(archive.ArchiveError, match='identity'):
        acquire(prefetch, tmp_path/'main-recheck', max_chunks=1)
    m = json.loads((tmp_path/'main-recheck/manifest.json').read_text())
    assert m['source_bytes_received'] == 0 and not m['complete']


def test_background_failure_stops_new_ranges_and_cancels_inflight(prefetch, source, tmp_path):
    first_gate, second_failed = threading.Event(), threading.Event()
    def hook(start, response):
        if start == 0:
            response.gate = first_gate
        elif start == 4096:
            response.status_code = 200
            second_failed.set()
    source.hook = hook
    receipt = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acquire, prefetch, tmp_path/'fail-fast', receipt=receipt)
        try:
            assert second_failed.wait(2)
        finally:
            first_gate.set()
        with pytest.raises(archive.ArchiveError):
            future.result(timeout=5)
    assert len([x for x in source.calls if x[0] == 'GET']) == 2
    assert all(s.closed for s in source.sessions) and all(r.closed for r in source.responses)
    assert receipt['pending_chunks_at_close'] == receipt['reserved_raw_bytes_at_close'] == 0


def test_worker_failure_closes_blocked_peer_without_waiting_for_main(prefetch, source, tmp_path):
    first_gate, first_started, second_failed = threading.Event(), threading.Event(), threading.Event()
    def hook(start, response):
        if start == 0:
            response.gate = first_gate
            first_started.set()
        elif start == 4096:
            assert first_started.wait(2)
            response.status_code = 200
            second_failed.set()
    source.hook = hook
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acquire, prefetch, tmp_path/'cancel-peer')
        try:
            assert second_failed.wait(2)
            assert first_gate.wait(1), 'failed worker did not close the blocked peer response'
        finally:
            first_gate.set()
        with pytest.raises(archive.ArchiveError):
            future.result(timeout=5)
    assert not json.loads((tmp_path/'cancel-peer/manifest.json').read_text())['complete']


def test_original_reserve_rejects_before_any_payload(prefetch, source, tmp_path, monkeypatch):
    reserve = 15*1024**3
    monkeypatch.setattr(archive.shutil, 'disk_usage', lambda _: SimpleNamespace(free=reserve-1))
    with pytest.raises(archive.ArchiveCapacityError):
        acquire(prefetch, tmp_path/'reserve', reserve_bytes=reserve)
    assert not [x for x in source.calls if x[0] == 'GET']
    assert not (tmp_path/'reserve/manifest.json').exists()


def test_dynamic_reserve_preserves_only_durable_prefix_and_resume(prefetch, source, tmp_path, monkeypatch):
    root = tmp_path/'dynamic'; reserve = 15*1024**3
    def free(_):
        path = root/'manifest.json'
        committed = path.exists() and len(json.loads(path.read_text())['chunks']) >= 1
        return SimpleNamespace(free=reserve if committed else 10**12)
    monkeypatch.setattr(archive.shutil, 'disk_usage', free)
    receipt = {};m = acquire(prefetch, root, reserve_bytes=reserve, receipt=receipt)
    disk = json.loads((root/'manifest.json').read_text())
    assert m['stop_reason'] == 'reserve_bytes' and m['stop_reason_persisted'] is False
    assert disk['source_bytes_received'] == 4096 and not disk['complete'] and disk['full_sha256'] is None
    first = (root/'blocks/00000000.xz').read_bytes()
    monkeypatch.setattr(archive.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**12))
    source.calls.clear()
    final = acquire(prefetch, root)
    assert final['complete'] and (root/'blocks/00000000.xz').read_bytes() == first
    assert all(x[2]['headers']['Range'] != 'bytes=0-4095' for x in source.calls if x[0] == 'GET')
    with archive.read_archive(root) as stream:
        assert stream.read() == source.data


def test_original_budget_stop_never_credits_ready_ranges(prefetch, source, tmp_path):
    m = acquire(prefetch, tmp_path/'budget', budget_bytes=4096)
    assert m['stop_reason'] == 'budget_bytes' and m['source_bytes_received'] == 0 and not m['complete']
    assert json.loads((tmp_path/'budget/manifest.json').read_text())['chunks'] == []


@pytest.mark.parametrize('failure', ['block_write', 'block_fsync', 'manifest'])
def test_original_transaction_failure_quarantines_uncredited_bytes(prefetch, source, tmp_path, monkeypatch, failure):
    root = tmp_path/failure
    with monkeypatch.context() as patch:
        if failure == 'block_write':
            original = archive._Storage.write
            def write(self, path, content):
                if path.parent.name == 'blocks':
                    original(self, path, content[:31])
                    raise OSError('fixture write failure')
                return original(self, path, content)
            patch.setattr(archive._Storage, 'write', write)
        elif failure == 'block_fsync':
            original = archive._fsync_dir
            def fsync(path):
                if path.name == 'blocks': raise OSError('fixture block directory fsync failure')
                return original(path)
            patch.setattr(archive, '_fsync_dir', fsync)
        else:
            original = archive._atomic_manifest
            def manifest(root, value, storage):
                if value['chunks']: raise OSError('fixture manifest failure')
                return original(root, value, storage)
            patch.setattr(archive, '_atomic_manifest', manifest)
        with pytest.raises(OSError):
            acquire(prefetch, root)
    m = json.loads((root/'manifest.json').read_text())
    assert m['chunks'] == [] and not m['complete'] and m['source_bytes_received'] == 0
    source.calls.clear()
    with pytest.raises(archive.ArchiveError, match='quarantine'):
        acquire(prefetch, root)
    assert not source.calls and list((root/'quarantine').iterdir())
    m = acquire(prefetch, root)
    assert m['complete']
    with archive.read_archive(root) as stream:
        assert stream.read() == source.data


def test_keyboard_interrupt_closes_inflight_and_releases_raw_slots(prefetch, source, tmp_path, monkeypatch):
    second_started = threading.Event(); second_gate = threading.Event()
    def hook(start, response):
        if start == 4096:
            response.gate = second_gate
            second_started.set()
        elif start == 0:
            original = response.iter_content
            def content(chunk_size):
                assert second_started.wait(2)
                yield from original(chunk_size)
            response.iter_content = content
    source.hook = hook
    original = prefetch._PrefetchSession.request
    def interrupt(self, method, *args, **kwargs):
        response = original(self, method, *args, **kwargs)
        if method == 'GET': raise KeyboardInterrupt('fixture user cancellation')
        return response
    monkeypatch.setattr(prefetch._PrefetchSession, 'request', interrupt)
    receipt = {}
    with pytest.raises(KeyboardInterrupt):
        acquire(prefetch, tmp_path/'cancel', receipt=receipt)
    assert second_gate.is_set() and all(x.closed for x in source.responses)
    assert all(s.closed for s in source.sessions)
    assert receipt['pending_chunks_at_close'] == receipt['reserved_raw_bytes_at_close'] == 0
    assert not json.loads((tmp_path/'cancel/manifest.json').read_text())['complete']


def test_resume_rejects_changed_etag_before_any_prefetch(prefetch, source, tmp_path):
    root = tmp_path/'changed'
    acquire(prefetch, root, max_chunks=1)
    before = (root/'manifest.json').read_bytes()
    source.etag = '"changed"'; source.calls.clear()
    with pytest.raises(archive.ArchiveError, match='identity'):
        acquire(prefetch, root)
    assert (root/'manifest.json').read_bytes() == before
    assert not [x for x in source.calls if x[0] == 'GET']


@pytest.mark.parametrize('helper', ['_request', '_identity_headers', '_head', '_receive_range'])
def test_in_memory_strict_helper_drift_is_rejected(prefetch, source, tmp_path, monkeypatch, helper):
    monkeypatch.setattr(archive, helper, lambda *args, **kwargs: None)
    with pytest.raises(archive.ArchiveError, match='pin|code|helper'):
        acquire(prefetch, tmp_path/'helper-drift')
    assert not source.sessions and not (tmp_path/'helper-drift').exists()


def test_bridge_executes_same_code_with_only_session_namespace_changed(prefetch, source, tmp_path):
    observed = []
    original = archive.acquire_archive
    old_profile = sys.getprofile()
    def profile(frame, event, _):
        if event == 'call' and frame.f_code is original.__code__:
            scope = frame.f_globals
            assert scope is not original.__globals__
            assert {k for k in scope if scope[k] is not original.__globals__[k]} == {'requests'}
            assert scope['_receive_range'] is archive._receive_range
            assert scope['_receive_range'].__globals__ is vars(archive)
            observed.append(True)
    try:
        sys.setprofile(profile)
        acquire(prefetch, tmp_path/'profile', max_chunks=1)
    finally:
        sys.setprofile(old_profile)
    assert observed == [True]


def test_two_workers_and_sequential_have_identical_manifest_and_xz(prefetch, source, tmp_path):
    sequential = acquire(prefetch, tmp_path/'sequential', workers=1)
    parallel = acquire(prefetch, tmp_path/'parallel', workers=2)
    assert sequential == parallel
    assert (tmp_path/'sequential/manifest.json').read_bytes() == (tmp_path/'parallel/manifest.json').read_bytes()
    for entry in sequential['chunks']:
        assert (tmp_path/'sequential'/entry['path']).read_bytes() == (tmp_path/'parallel'/entry['path']).read_bytes()


@pytest.mark.parametrize('fault', ['size', 'weak_etag', 'no_modified', 'encoding', 'transfer', 'redirect'])
def test_invalid_head_never_starts_worker_or_credits_manifest(prefetch, source, tmp_path, monkeypatch, fault):
    original = source.factory
    def factory():
        session = original()
        request = session.request
        def changed(method, url, **kwargs):
            response = request(method, url, **kwargs)
            if method == 'HEAD':
                if fault == 'size': response.headers['Content-Length'] = '12316'
                if fault == 'weak_etag': response.headers['ETag'] = 'W/"weak"'
                if fault == 'no_modified': response.headers.pop('Last-Modified')
                if fault == 'encoding': response.headers['Content-Encoding'] = 'gzip'
                if fault == 'transfer': response.headers['Transfer-Encoding'] = 'chunked'
                if fault == 'redirect': response.status_code = 302
            return response
        session.request = changed
        return session
    monkeypatch.setattr(prefetch, '_new_session', factory)
    with pytest.raises(archive.ArchiveError):
        acquire(prefetch, tmp_path/fault)
    assert not [x for x in source.calls if x[0] == 'GET']
    assert not (tmp_path/fault/'manifest.json').exists()
    assert all(x.closed for x in source.sessions) and all(x.closed for x in source.responses)


def test_workers_reuse_private_sessions_and_only_original_main_writes(prefetch, source, tmp_path, monkeypatch):
    writes = []; main_thread = threading.get_ident()
    original = archive._Storage.write
    def write(self, path, content):
        writes.append(threading.get_ident())
        return original(self, path, content)
    monkeypatch.setattr(archive._Storage, 'write', write)
    result = acquire(prefetch, tmp_path/'ownership')
    assert result['complete'] and writes and set(writes) == {main_thread}
    assert len(source.sessions) <= 3  # HEAD + at most one private session per worker
    assert all(x.closed for x in source.sessions)
    with archive.read_archive(tmp_path/'ownership') as stream:
        assert stream.read() == source.data


def test_worker_session_creation_failure_also_cancels_blocked_peer(prefetch, source, tmp_path, monkeypatch):
    first_gate, first_started, creation_failed = threading.Event(), threading.Event(), threading.Event()
    def hook(start, response):
        if start == 0:
            response.gate = first_gate
            first_started.set()
    source.hook = hook
    original = source.factory
    def factory():
        if len(source.sessions) >= 2:
            assert first_started.wait(2)
            creation_failed.set()
            raise RuntimeError('fixture Session construction failure')
        return original()
    monkeypatch.setattr(prefetch, '_new_session', factory)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acquire, prefetch, tmp_path/'session-failure')
        try:
            assert creation_failed.wait(2)
            assert first_gate.wait(1), 'Session construction failure did not cancel blocked peer'
        finally:
            first_gate.set()
        with pytest.raises((archive.ArchiveError, RuntimeError)):
            future.result(timeout=5)
    assert all(x.closed for x in source.sessions)


def test_receipt_times_observation_and_separates_delivery_from_commit(prefetch, source, tmp_path):
    receipt = {}; result = acquire(prefetch, tmp_path/'timing', max_chunks=1, receipt=receipt)
    assert receipt['invocation_status'] == 'returned'
    assert receipt['returned_committed_raw_bytes'] == result['source_bytes_received'] == 4096
    assert receipt['request_attempts'] == 2  # one HEAD and one bounded GET
    for entry in receipt['ranges']:
        assert (receipt['started_monotonic_ns'] <= entry['request_started_ns'] <=
                entry['headers_observed_ns'] <= entry['body_verified_ns'] <=
                entry['ordered_delivery_ns'] <= receipt['finished_monotonic_ns'])
    assert receipt['wire_body_bytes'] == receipt['delivered_raw_bytes'] == 4096
    assert not result['complete'] and receipt['returned_archive_complete'] is False
