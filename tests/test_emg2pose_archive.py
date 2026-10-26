"""Real HTTP/framing, filesystem transactions and byte-level archive contracts."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import random
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


def test_archive_public_api_exists():
    module = importlib.import_module('egoemg_repro.emg2pose_archive')
    assert callable(module.acquire_archive) and callable(module.read_archive)


@pytest.fixture
def archive():
    return importlib.import_module('egoemg_repro.emg2pose_archive')


@pytest.fixture
def raw_tar():
    # Includes NaN payload/sign bits, negative zero, arbitrary file contents and
    # nonzero tar member padding: comparisons must be on original bytes.
    payload = bytes.fromhex('0100c07f00000080000000000100c0ff') * 7
    payload += random.Random(42).randbytes(48_123)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as tf:
        member = tarfile.TarInfo('dataset/session.hdf5')
        member.size = len(payload)
        tf.addfile(member, io.BytesIO(payload))
    source = bytearray(stream.getvalue())
    # Tar readers ignore member padding, but our archive must retain it.
    source[512 + len(payload):((512 + len(payload) + 511)//512)*512] = b'P' * (
        ((512 + len(payload) + 511)//512)*512 - 512 - len(payload))
    return bytes(source), payload


@pytest.fixture
def endpoint(archive, monkeypatch, raw_tar):
    data, _ = raw_tar
    state = SimpleNamespace(data=data, etag='"fixture-v1"', mode='ok', requests=[], gets=0,
                            transient_remaining=0, fail_after=None)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            pass

        def do_HEAD(self):
            state.requests.append(('HEAD', dict(self.headers)))
            self.send_response(200)
            self.send_header('Content-Length', str(len(state.data) + (state.mode == 'head_size')))
            self.send_header('Content-Type', 'application/x-tar')
            if state.mode != 'no_etag':
                self.send_header('ETag', 'W/"weak"' if state.mode == 'weak_etag' else state.etag)
            self.send_header('Last-Modified', 'Thu, 01 Jan 2026 00:00:00 GMT')
            if state.mode == 'head_encoded':
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()

        def do_GET(self):
            state.gets += 1
            state.requests.append(('GET', dict(self.headers)))
            if state.transient_remaining or (state.fail_after is not None and state.gets > state.fail_after):
                state.transient_remaining = max(0, state.transient_remaining-1)
                self.send_response(503)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            start, end = map(int, self.headers['Range'].removeprefix('bytes=').split('-'))
            body = state.data[start:end+1]
            if state.mode == 'html':
                body = b'<!DOCTYPE html><html>error</html>'.ljust(len(body), b' ')
            self.send_response(200 if state.mode == 'status200' else 206)
            self.send_header('Content-Length', str(len(body)+(state.mode == 'long')))
            offset = 1 if state.mode == 'bad_range' else 0
            total = len(state.data) + (state.mode == 'bad_total')
            self.send_header('Content-Range', f'bytes {start+offset}-{end}/{total}')
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('ETag', '"changed"' if state.mode == 'etag_get' else state.etag)
            self.send_header('Last-Modified', 'Thu, 01 Jan 2026 00:00:00 GMT')
            if state.mode == 'encoded':
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            if state.mode == 'short':
                self.wfile.write(body[:len(body)//2])
                self.close_connection = True
            else:
                self.wfile.write(body + (b'X' if state.mode == 'long' else b''))

    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, *_args):
            # Invalid-header tests deliberately close before the server writes.
            pass
    server = QuietServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    worker.start()
    monkeypatch.setitem(archive._ASSETS, 'full', (f'http://127.0.0.1:{server.server_port}/dataset.tar', len(data)))
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def acquire(archive, path, **kwargs):
    return archive.acquire_archive(path, chunk_bytes=4096, reserve_bytes=0,
                                   budget_bytes=10*1024**2, **kwargs)


def load_manifest(root):
    return json.loads((root/'manifest.json').read_text())


def test_original_tar_bytes_roundtrip_and_streaming_extract(archive, endpoint, raw_tar, tmp_path):
    root = tmp_path/'archive'
    receipt = acquire(archive, root)
    assert receipt['complete'] is True
    assert receipt['source_bytes_received'] == len(raw_tar[0])
    assert receipt['full_sha256'] == hashlib.sha256(raw_tar[0]).hexdigest()
    assert receipt['projection_is_guarantee'] is False
    with archive.read_archive(root) as reader:
        assert reader.complete is True
        assert reader.read() == raw_tar[0]
    with archive.read_archive(root) as reader:
        with tarfile.open(mode='r|', fileobj=reader) as tf:
            member = next(iter(tf))
            assert tf.extractfile(member).read() == raw_tar[1]
    for method, headers in endpoint.requests:
        assert headers['Accept-Encoding'] == 'identity'
        if method == 'GET':
            assert headers['If-Match'] == endpoint.etag


def test_explicit_bounded_prefix_never_claims_complete(archive, endpoint, tmp_path):
    root = tmp_path/'archive'
    receipt = acquire(archive, root, max_chunks=1)
    assert receipt['complete'] is False
    assert receipt['stop_reason'] == 'max_chunks'
    assert receipt['full_sha256'] is None
    assert receipt['source_bytes_received'] == 4096
    with pytest.raises(archive.ArchiveError, match='incomplete'):
        archive.read_archive(root)
    with archive.read_archive(root, allow_incomplete=True) as reader:
        assert reader.complete is False
        assert reader.read() == endpoint.data[:4096]


def test_interrupt_then_resume_checks_prefix_and_rebuilds_full_hash(archive, endpoint, tmp_path):
    root = tmp_path/'archive'
    endpoint.fail_after = 1
    with pytest.raises(archive.ArchiveError):
        acquire(archive, root)
    before = load_manifest(root)
    assert len(before['chunks']) == 1 and not before['complete']
    first_path = root/before['chunks'][0]['path']
    saved = first_path.read_bytes()
    endpoint.fail_after = None
    endpoint.requests.clear()
    result = acquire(archive, root)
    assert result['complete']
    assert first_path.read_bytes() == saved
    requested = [h['Range'] for m, h in endpoint.requests if m == 'GET']
    assert requested[0] == 'bytes=4096-8191'
    assert result['full_sha256'] == hashlib.sha256(endpoint.data).hexdigest()


@pytest.mark.parametrize('mode', ['head_size', 'head_encoded', 'no_etag', 'weak_etag',
                                  'status200', 'bad_range', 'bad_total', 'html', 'short',
                                  'long', 'encoded', 'etag_get'])
def test_bad_http_never_commits_a_chunk(archive, endpoint, tmp_path, mode):
    endpoint.mode = mode
    root = tmp_path/'archive'
    with pytest.raises(archive.ArchiveError):
        acquire(archive, root, max_chunks=1)
    if (root/'manifest.json').exists():
        manifest = load_manifest(root)
        assert manifest['chunks'] == [] and not manifest['complete']
    assert not list(root.glob('blocks/*.xz'))


def test_source_etag_changed_before_resume_is_rejected(archive, endpoint, tmp_path):
    root = tmp_path/'archive'
    acquire(archive, root, max_chunks=1)
    before = (root/'manifest.json').read_bytes()
    endpoint.etag = '"fixture-v2"'
    with pytest.raises(archive.ArchiveError, match='identity'):
        acquire(archive, root)
    assert (root/'manifest.json').read_bytes() == before


def test_transient_get_failures_have_a_finite_retry_budget(archive, endpoint, tmp_path):
    endpoint.transient_remaining = 2
    result = acquire(archive, tmp_path/'ok', max_chunks=1)
    assert len(result['chunks']) == 1 and endpoint.gets == 3
    endpoint.gets = 0
    endpoint.transient_remaining = 10
    with pytest.raises(archive.ArchiveError):
        acquire(archive, tmp_path/'failed', max_chunks=1)
    assert endpoint.gets == 3


@pytest.mark.parametrize('orphan', ['blocks/00000000.xz', 'blocks/chunk.part', 'manifest.json.part'])
def test_unledgered_files_are_quarantined_not_credited_or_overwritten(archive, endpoint, tmp_path, orphan):
    root = tmp_path/'archive'
    bad = root/orphan
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b'uncommitted original bytes')
    with pytest.raises(archive.ArchiveError, match='quarantin'):
        acquire(archive, root, max_chunks=1)
    assert not bad.exists()
    preserved = [p for p in (root/'quarantine').iterdir() if p.is_file()]
    assert any(p.read_bytes() == b'uncommitted original bytes' for p in preserved)
    assert endpoint.gets == 0
    result = acquire(archive, root, max_chunks=1)
    assert len(result['chunks']) == 1 and not result['complete']


@pytest.mark.parametrize('tamper', ['compressed', 'raw_hash', 'range_gap', 'reordered', 'missing',
                                    'path_traversal', 'false_complete', 'xz_trailing_stream'])
def test_reader_and_resume_reject_tampering(archive, endpoint, tmp_path, tamper):
    root = tmp_path/'archive'
    acquire(archive, root, max_chunks=2)
    manifest = load_manifest(root)
    record = manifest['chunks'][0]
    path = root/record['path']
    if tamper == 'compressed':
        data = bytearray(path.read_bytes());data[len(data)//2] ^= 1;path.write_bytes(data)
    elif tamper == 'raw_hash':
        record['raw_sha256'] = '0'*64
    elif tamper == 'range_gap':
        record['offset'] = 1
    elif tamper == 'reordered':
        manifest['chunks'].reverse()
    elif tamper == 'missing':
        path.unlink()
    elif tamper == 'path_traversal':
        record['path'] = '../outside.xz'
    elif tamper == 'false_complete':
        manifest['complete'] = True
    elif tamper == 'xz_trailing_stream':
        data = path.read_bytes()+lzma.compress(b'not original data')
        path.write_bytes(data)
        record['compressed_bytes'] = len(data)
        record['compressed_sha256'] = hashlib.sha256(data).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(archive.ArchiveError):
        archive.read_archive(root, allow_incomplete=True)
    with pytest.raises(archive.ArchiveError):
        acquire(archive, root)


def _hold_lock(root, ready, release):
    import fcntl
    with (Path(root)/'archive.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=10)


def test_competing_writer_is_rejected_before_network(archive, endpoint, tmp_path):
    root = tmp_path/'archive';root.mkdir()
    ctx = multiprocessing.get_context('spawn')
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_hold_lock, args=(root, ready, release))
    process.start()
    try:
        assert ready.wait(timeout=5)
        with pytest.raises(archive.ArchiveError, match='lock|writer'):
            acquire(archive, root)
        assert endpoint.requests == []
    finally:
        release.set();process.join(timeout=5)
        if process.is_alive(): process.terminate();process.join()


def test_reader_holds_shared_lock_against_writer(archive, endpoint, tmp_path):
    root = tmp_path/'archive';acquire(archive, root, max_chunks=1)
    with archive.read_archive(root, allow_incomplete=True):
        with pytest.raises(archive.ArchiveError, match='lock|writer'):
            acquire(archive, root)
    assert acquire(archive, root, max_chunks=1)['source_bytes_received'] == 8192


def test_capacity_projection_stops_without_shrinking_source(archive, endpoint, tmp_path):
    root = tmp_path/'archive'
    result = archive.acquire_archive(root, chunk_bytes=4096, budget_bytes=20_000, reserve_bytes=0)
    assert not result['complete']
    assert result['stop_reason'] in {'budget_bytes', 'projected_capacity'}
    assert 0 < result['source_bytes_received'] < len(endpoint.data)
    assert result['source']['expected_bytes'] == len(endpoint.data)
    assert result['full_sha256'] is None
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= 20_000
    with archive.read_archive(root, allow_incomplete=True) as reader:
        assert reader.read() == endpoint.data[:result['source_bytes_received']]


def test_actual_disk_reserve_is_rechecked_and_stops_with_prefix(archive, endpoint, tmp_path, monkeypatch):
    root = tmp_path/'archive';acquire(archive, root, max_chunks=1)
    before = load_manifest(root)
    monkeypatch.setattr(archive.shutil, 'disk_usage', lambda _p: SimpleNamespace(free=15*1024**3))
    result = archive.acquire_archive(root, chunk_bytes=4096, budget_bytes=10*1024**2,
                                     reserve_bytes=15*1024**3)
    assert not result['complete'] and result['stop_reason'] == 'reserve_bytes'
    assert result['chunks'] == before['chunks']


def test_default_asset_registry_and_argument_bounds(archive, tmp_path):
    assert archive._ASSETS['full'] == ('https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar', 462824048640)
    assert archive._ASSETS['mini'][1] == 647198208
    for kw in ({'asset':'other'}, {'chunk_bytes':0}, {'chunk_bytes':16*1024**2+1},
               {'max_chunks':0}, {'max_chunks':True}, {'budget_bytes':0}, {'reserve_bytes':-1}):
        with pytest.raises(ValueError):
            archive.acquire_archive(tmp_path/'invalid', **kw)


def test_symlink_output_is_rejected(archive, endpoint, tmp_path):
    target = tmp_path/'target';target.mkdir()
    link = tmp_path/'link';link.symlink_to(target, target_is_directory=True)
    with pytest.raises(archive.ArchiveError, match='symlink'):
        acquire(archive, link)
    assert not list(target.iterdir())


@pytest.mark.parametrize('field,value', [('full_sha256', '0'*64), ('projection_is_guarantee', True),
                                         ('projected_compressed_bytes', 1)])
def test_completed_manifest_rejects_forged_hash_or_capacity_claim(archive, endpoint, tmp_path, field, value):
    root = tmp_path/'archive';acquire(archive, root)
    manifest = load_manifest(root)
    manifest[field] = value
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(archive.ArchiveError):
        archive.read_archive(root)


def test_manifest_commit_interruption_preserves_orphans_and_requires_explicit_resume(archive, endpoint, tmp_path, monkeypatch):
    root = tmp_path/'archive'
    original_replace = os.replace
    def interrupted_replace(source, destination):
        if Path(destination) == root/'manifest.json' and list(root.glob('blocks/*.xz')):
            raise OSError('simulated manifest publication interruption')
        return original_replace(source, destination)
    with monkeypatch.context() as fault:
        fault.setattr(os, 'replace', interrupted_replace)
        with pytest.raises(OSError, match='simulated'):
            acquire(archive, root, max_chunks=1)
    assert load_manifest(root)['chunks'] == []
    orphan_bytes = next(root.glob('blocks/*.xz')).read_bytes()
    with pytest.raises(archive.ArchiveError, match='quarantin'):
        acquire(archive, root, max_chunks=1)
    assert any(p.read_bytes() == orphan_bytes for p in (root/'quarantine').iterdir())
    assert acquire(archive, root, max_chunks=1)['source_bytes_received'] == 4096


def test_disk_reserve_drop_during_block_write_keeps_verified_prefix(archive, endpoint, tmp_path, monkeypatch):
    endpoint.data = random.Random(19).randbytes(768*1024)
    url, _size = archive._ASSETS['full']
    monkeypatch.setitem(archive._ASSETS, 'full', (url, len(endpoint.data)))
    root = tmp_path/'archive'
    initial = archive.acquire_archive(root, chunk_bytes=256*1024, max_chunks=1,
                                      reserve_bytes=0, budget_bytes=10*1024**2)
    reserve = 15*1024**3
    def disk_usage(_root):
        partials = list(root.glob('blocks/*.part-*'))
        low = any(p.stat().st_size >= 64*1024 for p in partials)
        return SimpleNamespace(free=reserve if low else reserve+10*1024**2)
    monkeypatch.setattr(archive.shutil, 'disk_usage', disk_usage)
    result = archive.acquire_archive(root, chunk_bytes=256*1024, reserve_bytes=reserve,
                                     budget_bytes=10*1024**2)
    assert result['stop_reason'] == 'reserve_bytes' and not result['complete']
    assert result['chunks'] == initial['chunks']
    assert load_manifest(root)['chunks'] == initial['chunks']
    assert list(root.glob('blocks/*.part-*'))


def test_reader_rejects_xz_expansion_beyond_recorded_raw_size(archive, endpoint, tmp_path):
    root = tmp_path/'archive';acquire(archive, root, max_chunks=1)
    manifest = load_manifest(root);record = manifest['chunks'][0]
    bomb = lzma.compress(b'X'*(16*1024**2))
    (root/record['path']).write_bytes(bomb)
    record['compressed_bytes'] = len(bomb)
    record['compressed_sha256'] = hashlib.sha256(bomb).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(archive.ArchiveError):
        archive.read_archive(root, allow_incomplete=True)


def test_quarantine_move_fsyncs_source_and_destination_directories(archive, endpoint, tmp_path, monkeypatch):
    root = tmp_path/'archive';(root/'blocks').mkdir(parents=True)
    (root/'blocks'/'chunk.part').write_bytes(b'keep these bytes')
    synced = set();actual = os.fsync
    def record_fsync(descriptor):
        synced.add(Path(os.readlink(f'/proc/self/fd/{descriptor}')))
        return actual(descriptor)
    monkeypatch.setattr(os, 'fsync', record_fsync)
    with pytest.raises(archive.ArchiveError, match='quarantin'):
        acquire(archive, root)
    assert {root, root/'blocks', root/'quarantine'} <= synced
