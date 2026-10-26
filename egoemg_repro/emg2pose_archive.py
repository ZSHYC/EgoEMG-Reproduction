"""Byte-exact, resumable official EMG2Pose tar acquisition in independent xz6 blocks.

This stores the original HTTP object, including tar headers/padding and HDF5
layout bytes. ETags identify the transport object; they are not publisher SHA256
checksums. Capacity projections are observations, never full-corpus guarantees.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import requests

_ASSETS = {
    'full': ('https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar', 462824048640),
    'mini': ('https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar', 647198208),
}
CHUNK_BYTES = 16 * 1024**2
_FORMAT = 'emg2pose_byte_exact_xz_v1'
_HASH = re.compile(r'[0-9a-f]{64}')
_RETRIES = 3


class ArchiveError(RuntimeError):
    """A protocol, source-identity, integrity or transaction check failed."""


class ArchiveCapacityError(ArchiveError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(f'archive capacity stop: {reason}')


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'{name} must be an integer >= {0 if zero else 1}')


def _reject_symlinks(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ArchiveError('symlink in managed archive directory')


@contextmanager
def _lock(root, *, write):
    if root.is_symlink():
        raise ArchiveError('symlink archive output')
    if write:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ArchiveError('archive directory does not exist')
    _reject_symlinks(root)
    descriptor = os.open(root/'archive.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_EX if write else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ArchiveError('archive lock is held by another reader/writer') from error
        yield
    finally:
        os.close(descriptor)


def _load_manifest(root):
    path = root/'manifest.json'
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, ValueError, UnicodeError) as error:
        raise ArchiveError('invalid archive manifest JSON') from error
    if not isinstance(value, dict):
        raise ArchiveError('archive manifest must be an object')
    return value


def _inspect_orphans(root, manifest, *, quarantine):
    """An unledgered block is never automatically trusted, even if valid xz."""
    _reject_symlinks(root)
    records = manifest.get('chunks', []) if manifest else []
    if not isinstance(records, list):
        raise ArchiveError('invalid chunk ledger')
    credited = set()
    for entry in records:
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise ArchiveError('invalid chunk path')
        if not re.fullmatch(r'blocks/[0-9]{8}\.xz', entry['path']):
            raise ArchiveError('unsafe or noncanonical chunk path')
        credited.add(entry['path'])
    orphans = []
    for path in root.rglob('*'):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            if relative not in {'blocks', 'quarantine'}:
                raise ArchiveError('unexpected directory in archive')
        elif path.is_file():
            if relative in {'manifest.json', 'archive.lock'} or relative in credited or relative.startswith('quarantine/'):
                continue
            orphans.append(path)
        else:
            raise ArchiveError('nonregular file in archive')
    if not orphans:
        return
    if quarantine:
        destination = root/'quarantine'
        destination.mkdir(exist_ok=True)
        changed_directories = {root, destination}
        for path in orphans:
            changed_directories.add(path.parent)
            path.rename(destination/(path.name+'.'+uuid.uuid4().hex+'.uncommitted'))
        for directory in changed_directories:
            _fsync_dir(directory)
        raise ArchiveError(f'{len(orphans)} unledgered files preserved in quarantine; inspect before explicit resume')
    raise ArchiveError('unledgered files require writer quarantine before reading')


def _fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Storage:
    def __init__(self, root, budget, reserve):
        self.root, self.budget, self.reserve = root, budget, reserve
        self.usage = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

    def check(self, extra):
        if self.usage + extra > self.budget:
            raise ArchiveCapacityError('budget_bytes')
        if shutil.disk_usage(self.root).free - extra < self.reserve:
            raise ArchiveCapacityError('reserve_bytes')

    def write(self, path, content):
        self.check(len(content))
        with path.open('xb') as output:
            for offset in range(0, len(content), 64*1024):
                part = content[offset:offset+64*1024]
                self.check(len(part))
                written = output.write(part)
                self.usage += written
                if written != len(part):
                    raise ArchiveError('short local archive write')
            output.flush()
            os.fsync(output.fileno())


def _encoded(manifest):
    return (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _atomic_manifest(root, manifest, storage):
    encoded = _encoded(manifest)
    old_size = (root/'manifest.json').stat().st_size if (root/'manifest.json').exists() else 0
    part = root/('manifest.json.part-'+uuid.uuid4().hex)
    storage.write(part, encoded)
    part.replace(root/'manifest.json')
    storage.usage -= old_size
    _fsync_dir(root)


def _request(session, method, url, headers):
    last_error = None
    for attempt in range(_RETRIES):
        try:
            response = session.request(method, url, headers=headers, timeout=(15, 60),
                                       stream=True, allow_redirects=False)
            if response.status_code in {408, 429, 500, 502, 503, 504}:
                status = response.status_code
                response.close()
                last_error = ArchiveError(f'transient HTTP {status}')
            else:
                return response
        except (requests.ConnectionError, requests.Timeout) as error:
            last_error = error
        if attempt + 1 < _RETRIES:
            time.sleep(.25 * 2**attempt)
    raise ArchiveError(f'HTTP {method} failed after {_RETRIES} attempts') from last_error


def _identity_headers(response, expected_length):
    if response.headers.get('Content-Encoding', 'identity').strip().lower() != 'identity':
        raise ArchiveError('encoded HTTP response cannot preserve byte ranges')
    if response.headers.get('Transfer-Encoding'):
        raise ArchiveError('ambiguous HTTP transfer encoding')
    if response.headers.get('Content-Length') != str(expected_length):
        raise ArchiveError('HTTP Content-Length differs from exact expected bytes')
    kind = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
    if kind not in {'application/octet-stream', 'binary/octet-stream', 'application/x-tar', 'application/x-gtar'}:
        raise ArchiveError('HTTP response is not a binary tar object')


def _head(session, url, expected):
    with _request(session, 'HEAD', url, {'Accept-Encoding':'identity'}) as response:
        if response.status_code != 200:
            raise ArchiveError('source HEAD did not return HTTP 200')
        _identity_headers(response, expected)
        etag = response.headers.get('ETag', '')
        modified = response.headers.get('Last-Modified', '')
        if not re.fullmatch(r'"[^"\r\n]+"', etag) or not modified.strip():
            raise ArchiveError('source requires a strong ETag and Last-Modified')
        return {'url':url, 'expected_bytes':expected, 'etag':etag, 'last_modified':modified}


def _receive_range(session, source, start, size):
    end = start+size-1
    headers = {'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity', 'If-Match':source['etag']}
    with _request(session, 'GET', source['url'], headers) as response:
        if response.status_code != 206:
            raise ArchiveError('source range did not return HTTP 206')
        _identity_headers(response, size)
        if response.headers.get('Content-Range') != f"bytes {start}-{end}/{source['expected_bytes']}":
            raise ArchiveError('HTTP Content-Range differs from requested range/total')
        if response.headers.get('ETag') != source['etag'] or response.headers.get('Last-Modified') != source['last_modified']:
            raise ArchiveError('source object identity changed during range acquisition')
        body = bytearray()
        try:
            for part in response.iter_content(chunk_size=64*1024):
                if len(body)+len(part) > size:
                    raise ArchiveError('range body exceeds exact expected bytes')
                body.extend(part)
        except requests.RequestException as error:
            raise ArchiveError('incomplete or interrupted range body; no chunk committed') from error
        if len(body) != size:
            raise ArchiveError('range body shorter than exact expected bytes')
        prefix = body.lstrip(b'\xef\xbb\xbf \t\r\n').lower()
        if prefix.startswith((b'<!doctype html', b'<html')):
            raise ArchiveError('HTML masquerading as binary range')
        return bytes(body)


def _decompress(root, record):
    path = root/record['path']
    raw_bytes, compressed_bytes = record['raw_bytes'], record['compressed_bytes']
    if not path.is_file() or path.is_symlink() or path.stat().st_size != compressed_bytes:
        raise ArchiveError('missing or wrong-sized compressed block')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record['compressed_sha256']:
        raise ArchiveError('compressed block SHA256 mismatch')
    try:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=256*1024**2)
        raw = decoder.decompress(content, max_length=raw_bytes+1)
    except lzma.LZMAError as error:
        raise ArchiveError('invalid xz block') from error
    if len(raw) != raw_bytes or not decoder.eof or decoder.unused_data:
        raise ArchiveError('xz block raw size, EOF or trailing stream mismatch')
    if hashlib.sha256(raw).hexdigest() != record['raw_sha256']:
        raise ArchiveError('decompressed raw block SHA256 mismatch')
    return raw


def _validate_archive(root, manifest):
    if manifest.get('format') != _FORMAT or manifest.get('schema_version') != 1:
        raise ArchiveError('unknown archive manifest format')
    asset = manifest.get('asset')
    if asset not in _ASSETS:
        raise ArchiveError('unknown archive source asset')
    source = manifest.get('source')
    if not isinstance(source, dict) or (source.get('url'), source.get('expected_bytes')) != _ASSETS[asset]:
        raise ArchiveError('archive source identity does not match fixed official asset')
    if not re.fullmatch(r'"[^"\r\n]+"', str(source.get('etag', ''))) or not source.get('last_modified'):
        raise ArchiveError('invalid source identity')
    total = source['expected_bytes']
    chunk = manifest.get('chunk_bytes')
    if type(chunk) is not int or not 1 <= chunk <= CHUNK_BYTES:
        raise ArchiveError('invalid archive chunk size')
    if manifest.get('compression') != {'format':'xz', 'preset':6}:
        raise ArchiveError('unknown compression contract')
    records = manifest.get('chunks')
    if not isinstance(records, list) or len(records) > math.ceil(total/chunk):
        raise ArchiveError('invalid chunk ledger length')
    offset = compressed = 0
    hasher = hashlib.sha256()
    for index, entry in enumerate(records):
        if not isinstance(entry, dict):
            raise ArchiveError('invalid chunk record')
        expected = min(chunk, total-offset)
        if (type(entry.get('index')) is not int or entry['index'] != index or
                type(entry.get('offset')) is not int or entry['offset'] != offset or
                type(entry.get('raw_bytes')) is not int or entry['raw_bytes'] != expected or
                entry.get('path') != f'blocks/{index:08d}.xz'):
            raise ArchiveError('chunk ledger is reordered, discontinuous or unsafe')
        size = entry.get('compressed_bytes')
        if type(size) is not int or not 1 <= size <= expected+1024**2:
            raise ArchiveError('unbounded compressed block size')
        if any(not isinstance(entry.get(key), str) or not _HASH.fullmatch(entry[key]) for key in ('raw_sha256', 'compressed_sha256')):
            raise ArchiveError('invalid chunk digest')
        hasher.update(_decompress(root, entry))
        offset += expected
        compressed += size
    complete = offset == total
    if (type(manifest.get('complete')) is not bool or manifest['complete'] != complete or
            manifest.get('status') != ('complete' if complete else 'incomplete') or
            type(manifest.get('source_bytes_received')) is not int or manifest['source_bytes_received'] != offset or
            type(manifest.get('compressed_bytes')) is not int or manifest['compressed_bytes'] != compressed):
        raise ArchiveError('manifest coverage/completion totals disagree with verified blocks')
    expected_hash = hasher.hexdigest() if complete else None
    if manifest.get('full_sha256') != expected_hash:
        raise ArchiveError('full SHA256 is wrong or claimed for an incomplete archive')
    ratio = compressed/offset if offset else None
    projected = math.ceil(ratio*total) if offset else None
    if (manifest.get('projection_is_guarantee') is not False or
            manifest.get('observed_compression_ratio') != ratio or
            manifest.get('projected_compressed_bytes') != projected):
        raise ArchiveError('capacity projection differs from actual verified bytes or claims a guarantee')
    return hasher


def _projection(manifest):
    received = manifest['source_bytes_received']
    manifest['observed_compression_ratio'] = manifest['compressed_bytes']/received if received else None
    manifest['projected_compressed_bytes'] = math.ceil(manifest['observed_compression_ratio']*manifest['source']['expected_bytes']) if received else None
    manifest['projection_is_guarantee'] = False


def _stop(root, manifest, reason, storage):
    result = dict(manifest, stop_reason=reason)
    try:
        _atomic_manifest(root, result, storage)
    except ArchiveCapacityError:
        # Changing the receipt must not consume the reserve or exceed the hard
        # budget. The previously committed incomplete prefix remains valid.
        result['stop_reason_persisted'] = False
    return result


def acquire_archive(output, *, asset='full', chunk_bytes=CHUNK_BYTES, max_chunks=None,
                    budget_bytes=340*1024**3, reserve_bytes=15*1024**3):
    """Acquire exact ranges; ``max_chunks`` limits NEW blocks in this invocation.

    The hard budget includes all managed files, quarantine and temporary writes.
    No admission check requires room for the entire uncompressed source. A
    projection stop preserves a verified prefix and can be explicitly resumed
    with revised capacity; it never reduces the expected source/cohort.
    """
    if asset not in _ASSETS:
        raise ValueError('asset must be full or mini')
    _positive(chunk_bytes, 'chunk_bytes')
    if chunk_bytes > CHUNK_BYTES:
        raise ValueError('chunk_bytes cannot exceed 16 MiB')
    if max_chunks is not None:
        _positive(max_chunks, 'max_chunks')
    _positive(budget_bytes, 'budget_bytes')
    _positive(reserve_bytes, 'reserve_bytes', zero=True)
    root = Path(output)
    with _lock(root, write=True):
        manifest = _load_manifest(root)
        _inspect_orphans(root, manifest, quarantine=True)
        hasher = _validate_archive(root, manifest) if manifest else hashlib.sha256()
        if manifest and (manifest['asset'] != asset or manifest['chunk_bytes'] != chunk_bytes):
            raise ArchiveError('resume asset/chunk identity differs from committed manifest')
        storage = _Storage(root, budget_bytes, reserve_bytes)
        with requests.Session() as session:
            session.trust_env = False
            session.auth = None
            session.cookies.clear()
            session.proxies.clear()
            url, total = _ASSETS[asset]
            identity = _head(session, url, total)
            if manifest and manifest['source'] != identity:
                raise ArchiveError('source identity changed since archive creation')
            if manifest is None:
                manifest = {'format':_FORMAT, 'schema_version':1, 'asset':asset, 'source':identity,
                            'source_identity_policy':'ETag is transport identity, not a publisher checksum; SHA256 identifies received bytes',
                            'compression':{'format':'xz', 'preset':6}, 'chunk_bytes':chunk_bytes,
                            'chunks':[], 'source_bytes_received':0, 'compressed_bytes':0,
                            'complete':False, 'status':'incomplete', 'full_sha256':None,
                            'stop_reason':'acquiring'}
                _projection(manifest)
                _atomic_manifest(root, manifest, storage)
            if manifest['complete']:
                return manifest
            added = 0
            while manifest['source_bytes_received'] < total:
                if max_chunks is not None and added >= max_chunks:
                    return _stop(root, manifest, 'max_chunks', storage)
                try:
                    storage.check(4096)
                    projection = manifest['projected_compressed_bytes']
                    if projection is not None:
                        metadata_estimate = math.ceil(total/chunk_bytes)*768+4096
                        projected_archive = projection+metadata_estimate
                        quarantine_size = sum(p.stat().st_size for p in (root/'quarantine').glob('*') if p.is_file())
                        if projected_archive+quarantine_size > budget_bytes:
                            return _stop(root, manifest, 'projected_capacity', storage)
                        remaining = max(0, projected_archive+quarantine_size-storage.usage)
                        if shutil.disk_usage(root).free-remaining < reserve_bytes:
                            return _stop(root, manifest, 'projected_capacity', storage)
                    start = manifest['source_bytes_received']
                    raw = _receive_range(session, identity, start, min(chunk_bytes, total-start))
                    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
                    candidate_hasher = hasher.copy()
                    candidate_hasher.update(raw)
                    index = len(manifest['chunks'])
                    record = {'index':index, 'offset':start, 'raw_bytes':len(raw),
                              'raw_sha256':hashlib.sha256(raw).hexdigest(),
                              'path':f'blocks/{index:08d}.xz', 'compressed_bytes':len(compressed),
                              'compressed_sha256':hashlib.sha256(compressed).hexdigest()}
                    candidate = dict(manifest, chunks=manifest['chunks']+[record],
                                     source_bytes_received=start+len(raw),
                                     compressed_bytes=manifest['compressed_bytes']+len(compressed),
                                     stop_reason='acquiring')
                    candidate['complete'] = candidate['source_bytes_received'] == total
                    candidate['status'] = 'complete' if candidate['complete'] else 'incomplete'
                    candidate['full_sha256'] = candidate_hasher.hexdigest() if candidate['complete'] else None
                    _projection(candidate)
                    storage.check(len(compressed)+len(_encoded(candidate)))
                    blocks = root/'blocks'
                    blocks.mkdir(exist_ok=True)
                    destination = root/record['path']
                    if destination.exists():
                        raise ArchiveError('uncommitted block destination already exists')
                    part = blocks/(destination.name+'.part-'+uuid.uuid4().hex)
                    storage.write(part, compressed)
                    part.replace(destination)
                    _fsync_dir(blocks)
                    _atomic_manifest(root, candidate, storage)
                    manifest, hasher = candidate, candidate_hasher
                    added += 1
                except ArchiveCapacityError as error:
                    return _stop(root, manifest, error.reason, storage)
            manifest = dict(manifest, stop_reason='complete')
            _atomic_manifest(root, manifest, storage)
            return manifest


class ArchiveReader(io.RawIOBase):
    """Sequential verified reader; use bounded reads/tar streaming for full data.

    Opening validates the entire committed prefix, including raw hashes. Blocks
    are verified again as read. A shared lock lasts until close/context exit.
    """
    def __init__(self, root, manifest, lock):
        super().__init__()
        self.manifest = manifest
        self.complete = manifest['complete']
        self._root, self._lock = root, lock
        self._index = 0
        self._buffer = b''
        self._position = 0

    def readable(self):
        return True

    def readinto(self, target):
        self._checkClosed()
        view = memoryview(target).cast('B')
        count = 0
        while count < len(view):
            if self._position == len(self._buffer):
                if self._index >= len(self.manifest['chunks']):
                    break
                self._buffer = _decompress(self._root, self.manifest['chunks'][self._index])
                self._index += 1
                self._position = 0
            size = min(len(view)-count, len(self._buffer)-self._position)
            view[count:count+size] = self._buffer[self._position:self._position+size]
            count += size
            self._position += size
        return count

    def close(self):
        if not self.closed:
            self._lock.__exit__(None, None, None)
            self._buffer = b''
        super().close()


def read_archive(output, *, allow_incomplete=False):
    """Open complete source, or an explicitly marked engineering prefix.

    Does not create a restored tar/HDF5 on disk. ``read()`` without a size has
    normal file semantics and collects all bytes; full-corpus consumers should
    use bounded reads (for example ``tarfile.open(mode='r|', fileobj=reader)``).
    """
    if type(allow_incomplete) is not bool:
        raise ValueError('allow_incomplete must be an explicit boolean')
    root = Path(output)
    lock = _lock(root, write=False)
    lock.__enter__()
    try:
        manifest = _load_manifest(root)
        if manifest is None:
            raise ArchiveError('archive manifest missing')
        _inspect_orphans(root, manifest, quarantine=False)
        _validate_archive(root, manifest)
        if not manifest['complete'] and not allow_incomplete:
            raise ArchiveError('archive is incomplete; engineering prefix requires allow_incomplete=True')
        return ArchiveReader(root, manifest, lock)
    except BaseException:
        lock.__exit__(None, None, None)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--asset', choices=tuple(_ASSETS), default='full')
    parser.add_argument('--max-chunks', required=True, type=int, help='Explicit bounded acquisition; not a full-data launch')
    parser.add_argument('--budget-bytes', type=int, default=340*1024**3)
    parser.add_argument('--reserve-bytes', type=int, default=15*1024**3)
    args = parser.parse_args(argv)
    print(json.dumps(acquire_archive(args.output, asset=args.asset, max_chunks=args.max_chunks,
                                     budget_bytes=args.budget_bytes, reserve_bytes=args.reserve_bytes), indent=2))


if __name__ == '__main__':
    main()
