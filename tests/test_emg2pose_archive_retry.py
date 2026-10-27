"""Real interrupted HTTP bodies must preserve ranges and archive transactions."""
from __future__ import annotations

import hashlib
import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import random
import threading
from types import SimpleNamespace

import pytest
import requests

from egoemg_repro import emg2pose_archive as archive


def retry(*args, **kwargs):
    return importlib.import_module('egoemg_repro.emg2pose_archive_retry').receive_with_retries(
        *args, **kwargs)


@pytest.fixture
def endpoint(monkeypatch):
    state = SimpleNamespace(data=random.Random(42).randbytes(300_000), requests=[],
                            actions=[], default='ok')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(state.data)))
            self.send_header('Content-Type', 'application/x-tar')
            self.send_header('ETag', '"fixed-object"')
            self.send_header('Last-Modified', 'Thu, 01 Jan 2026 00:00:00 GMT')
            self.end_headers()

        def do_GET(self):
            state.requests.append(dict(self.headers))
            action = state.actions.pop(0) if state.actions else state.default
            start, end = map(int, self.headers['Range'].removeprefix('bytes=').split('-'))
            body = state.data[start:end+1]
            if action == 'html':
                body = b'<html>error</html>'.ljust(len(body), b' ')
            self.send_response(200 if action == 'status' else 206)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Range',
                             f'bytes {start+(action == "range")}-{end}/{len(state.data)}')
            self.send_header('ETag', '"changed"' if action == 'etag' else '"fixed-object"')
            self.send_header('Last-Modified', 'changed' if action == 'modified' else
                             'Thu, 01 Jan 2026 00:00:00 GMT')
            if action == 'encoded':
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(body[:len(body)//2] if action == 'truncate' else body)
            self.wfile.flush()
            if action == 'truncate':
                self.close_connection = True

    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, *_args):
            pass

    server = QuietServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever,
                              kwargs={'poll_interval': .01}, daemon=True)
    worker.start()
    state.url = f'http://127.0.0.1:{server.server_port}/dataset.tar'
    monkeypatch.setitem(archive._ASSETS, 'full', (state.url, len(state.data)))
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def receive(endpoint, *, guard=lambda: None, **kwargs):
    with requests.Session() as session:
        session.trust_env = False
        source = archive._head(session, endpoint.url, len(endpoint.data))
        return retry(archive._receive_range, session, source, 1024, 131072,
                     guard=guard, delays=(0., 0.), **kwargs)


def test_real_partial_response_retries_exact_range_and_strong_identity(endpoint):
    endpoint.actions = ['truncate', 'ok']
    guards, events = [], []
    body = receive(endpoint, guard=lambda: guards.append(True), on_retry=events.append)
    assert body == endpoint.data[1024:132096]
    assert len(endpoint.requests) == 2
    assert endpoint.requests[0] == endpoint.requests[1]
    assert endpoint.requests[0]['Range'] == 'bytes=1024-132095'
    assert endpoint.requests[0]['If-Match'] == '"fixed-object"'
    assert endpoint.requests[0]['Accept-Encoding'] == 'identity'
    assert len(guards) == 3  # before first, before backoff, before next attempt
    assert events == [dict(failed_attempt=1, next_attempt=2, start=1024, size=131072,
                          delay_seconds=0., error_type='ChunkedEncodingError')]


def test_three_body_attempts_then_preserves_actual_last_failure(endpoint):
    endpoint.default = 'truncate'
    with pytest.raises(archive.ArchiveError, match='no chunk committed') as caught:
        receive(endpoint)
    assert isinstance(caught.value.__cause__, requests.exceptions.ChunkedEncodingError)
    assert len(endpoint.requests) == 3
    assert len({r['Range'] for r in endpoint.requests}) == 1


@pytest.mark.parametrize('action', ['etag', 'modified', 'range', 'status', 'html', 'encoded'])
def test_protocol_and_content_errors_fail_once(endpoint, action):
    endpoint.actions = [action, 'ok']
    with pytest.raises(archive.ArchiveError):
        receive(endpoint)
    assert len(endpoint.requests) == 1


def test_later_attempt_still_checks_object_identity(endpoint):
    endpoint.actions = ['truncate', 'etag', 'ok']
    with pytest.raises(archive.ArchiveError, match='identity changed'):
        receive(endpoint)
    assert len(endpoint.requests) == 2


def test_no_partial_chunk_commits_and_explicit_resume_is_byte_exact(endpoint, monkeypatch, tmp_path):
    native = archive._receive_range
    monkeypatch.setattr(archive, '_receive_range', lambda *a: retry(
        native, *a, guard=lambda: None, delays=(0., 0.)))
    root = tmp_path/'archive'
    endpoint.actions = ['ok', 'truncate', 'truncate', 'truncate']
    options = dict(chunk_bytes=131072, budget_bytes=10*1024**2, reserve_bytes=0)
    with pytest.raises(archive.ArchiveError, match='no chunk committed'):
        archive.acquire_archive(root, **options)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['source_bytes_received'] == 131072
    assert len(manifest['chunks']) == 1 and manifest['complete'] is False
    assert len(list((root/'blocks').iterdir())) == 1
    assert [r['Range'] for r in endpoint.requests] == [
        'bytes=0-131071', *['bytes=131072-262143']*3]
    result = archive.acquire_archive(root, **options)
    assert result['complete'] is True
    assert result['full_sha256'] == hashlib.sha256(endpoint.data).hexdigest()
    with archive.read_archive(root) as reader:
        assert reader.read() == endpoint.data


def test_resource_failure_before_first_attempt_makes_no_request(endpoint):
    def guard():
        raise archive.ArchiveCapacityError('reserve_bytes')
    with pytest.raises(archive.ArchiveCapacityError, match='reserve_bytes'):
        receive(endpoint, guard=guard)
    assert endpoint.requests == []


def test_resource_pressure_prevents_retry(endpoint):
    endpoint.actions = ['truncate', 'ok']
    calls = []
    def guard():
        calls.append(True)
        if len(calls) == 2:
            raise archive.ArchiveCapacityError('reserve_bytes')
    with pytest.raises(archive.ArchiveCapacityError, match='reserve_bytes'):
        receive(endpoint, guard=guard)
    assert len(endpoint.requests) == 1


def test_guard_runs_after_backoff_before_next_request(endpoint, monkeypatch):
    module = importlib.import_module('egoemg_repro.emg2pose_archive_retry')
    endpoint.actions = ['truncate', 'ok']
    pressure = []
    monkeypatch.setattr(module.time, 'sleep', lambda _delay: pressure.append(True))
    def guard():
        if pressure:
            raise archive.ArchiveCapacityError('reserve_bytes')
    with pytest.raises(archive.ArchiveCapacityError):
        receive(endpoint, guard=guard)
    assert len(endpoint.requests) == 1


@pytest.mark.parametrize('cause', [requests.exceptions.SSLError('tls'),
                                  requests.exceptions.ContentDecodingError('decode'),
                                  requests.RequestException('unknown'), OSError('local')])
def test_only_explicit_network_body_causes_are_retryable(cause):
    calls = []
    def broken(*_args):
        calls.append(True)
        raise archive.ArchiveError('incomplete or interrupted range body; no chunk committed') from cause
    with pytest.raises(archive.ArchiveError) as caught:
        retry(broken, None, {}, 0, 1, guard=lambda: None, delays=(0., 0.))
    assert caught.value.__cause__ is cause
    assert len(calls) == 1


@pytest.mark.parametrize('cause', [requests.ConnectionError('body'), requests.ReadTimeout('body')])
def test_actual_requests_read_timeout_and_connection_failure_are_bounded(cause):
    calls = []
    def broken(*_args):
        calls.append(True)
        raise archive.ArchiveError('incomplete or interrupted range body; no chunk committed') from cause
    with pytest.raises(archive.ArchiveError):
        retry(broken, None, {}, 0, 1, guard=lambda: None, delays=(0., 0.))
    assert len(calls) == 3


@pytest.mark.parametrize('error', [OSError('local'), KeyboardInterrupt(),
                                  archive.ArchiveError('range body exceeds exact expected bytes'),
                                  archive.ArchiveError('incomplete or interrupted range body; no chunk committed')])
def test_other_failures_and_interruptions_are_not_retried(error):
    calls = []
    def broken(*_args):
        calls.append(True)
        raise error
    with pytest.raises(type(error)) as caught:
        retry(broken, None, {}, 0, 1, guard=lambda: None, delays=(0., 0.))
    assert caught.value is error and len(calls) == 1


@pytest.mark.parametrize('delays', [(float('nan'),), (float('inf'),), (-1.,),
                                   (True,), ('0',), (0., 0., 0.), iter([0.])])
def test_invalid_or_unbounded_retry_policies_fail_before_work(delays):
    calls = []
    with pytest.raises(ValueError):
        retry(lambda *_: calls.append(True), None, {}, 0, 1,
              guard=lambda: calls.append(True), delays=delays)
    assert not calls


def test_retry_observer_failure_propagates_without_new_request(endpoint):
    endpoint.actions = ['truncate', 'ok']
    def observer(_event):
        raise OSError('log disk failed')
    with pytest.raises(OSError, match='log disk failed'):
        receive(endpoint, on_retry=observer)
    assert len(endpoint.requests) == 1


def test_caller_source_mutation_does_not_change_retry_identity(endpoint):
    endpoint.actions = ['truncate', 'ok']
    with requests.Session() as session:
        session.trust_env = False
        source = archive._head(session, endpoint.url, len(endpoint.data))
        def observer(_event):
            source['etag'] = '"changed-by-caller"'
        result = retry(archive._receive_range, session, source, 1024, 131072,
                       guard=lambda: None, delays=(0., 0.), on_retry=observer)
    assert result == endpoint.data[1024:132096]
    assert [r['If-Match'] for r in endpoint.requests] == ['"fixed-object"']*2


@pytest.mark.parametrize('delays', [(), (0.,)])
def test_smaller_explicit_retry_budget_is_respected(endpoint, delays):
    endpoint.default = 'truncate'
    with requests.Session() as session:
        session.trust_env = False
        source = archive._head(session, endpoint.url, len(endpoint.data))
        with pytest.raises(archive.ArchiveError, match='no chunk committed'):
            retry(archive._receive_range, session, source, 1024, 131072,
                  guard=lambda: None, delays=delays)
    assert len(endpoint.requests) == len(delays)+1
