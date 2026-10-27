"""Complete byte-exact HTTP objects in owned, sealed, volatile Linux memory.

This is an acquisition/storage primitive, not a completed EMG2Pose dataset or
training result. No bytes are extracted or discarded. The persistent xz archive
is independent and never modified here. Consumers must keep the returned owner
alive (or explicitly inherit its fd) while reading it.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import hashlib
import os
from pathlib import Path
import re
import time

import requests

from . import emg2pose_archive as transport
from .emg2pose_archive import ArchiveCapacityError, ArchiveError


RESERVE_BYTES = 256 * 1024**3
CHUNK_BYTES = 16 * 1024**2
# Linux UAPI (linux/fcntl.h): some portable Python builds omit these names even
# when memfd_create and the running kernel support the exact same operations.
_ADD_SEALS = getattr(fcntl, 'F_ADD_SEALS', 1033)
_GET_SEALS = getattr(fcntl, 'F_GET_SEALS', 1034)
_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008


def available_memory_bytes(*, meminfo=Path('/proc/meminfo'),
                           cgroup_file=Path('/proc/self/cgroup'),
                           cgroup_root=Path('/sys/fs/cgroup')) -> int:
    """Conservatively observe host availability and all visible v2 limits.

    This observation is not an operating-system reservation. Unknown/malformed
    capacity fails closed; an unlimited leaf cannot hide a finite parent.
    Explicit file locations also support inspection of a mounted host hierarchy.
    """
    try:
        matches = re.findall(r'^MemAvailable:\s+([0-9]+) kB\s*$',
                             Path(meminfo).read_text(), flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('missing or ambiguous MemAvailable')
        available = int(matches[0]) * 1024
        groups = [line[3:] for line in Path(cgroup_file).read_text().splitlines()
                  if line.startswith('0::')]
        if len(groups) != 1 or not groups[0].startswith('/'):
            raise ValueError('a single unified cgroup v2 hierarchy is required')
        parts = groups[0].split('/')[1:]
        if any(p in {'.', '..'} for p in parts):
            raise ValueError('unsafe cgroup membership')
        root = Path(cgroup_root)
        if not root.is_dir():
            raise ValueError('missing cgroup hierarchy root')
        if not (root/'memory.max').exists() and 'memory' not in (root/'cgroup.controllers').read_text().split():
            raise ValueError('memory controller is not exposed at cgroup root')
        current = root.joinpath(*[p for p in parts if p])
        while True:
            maximum_path = current/'memory.max'
            # Kernel cgroup-v2 docs define memory.current/max on non-root
            # groups. A real global root can have neither; controller presence
            # above distinguishes that case from an unknown/unmounted hierarchy.
            # A finite namespaced root exposes memory.max and is checked below.
            if current == root and not maximum_path.exists():
                break
            used = (current/'memory.current').read_text().strip()
            if not re.fullmatch(r'[0-9]+', used):
                raise ValueError('invalid cgroup memory.current')
            maximum = maximum_path.read_text().strip()
            if maximum != 'max':
                if not re.fullmatch(r'[0-9]+', maximum):
                    raise ValueError('invalid cgroup memory.max')
                available = min(available, max(0, int(maximum)-int(used)))
            if current == root:
                break
            current = current.parent
        return available
    except (OSError, ValueError) as error:
        raise ArchiveCapacityError('unknown host/cgroup memory capacity') from error


def _policy(workers, chunk_bytes, reserve_bytes, on_progress):
    transport._positive(workers, 'workers')
    transport._positive(chunk_bytes, 'chunk_bytes')
    transport._positive(reserve_bytes, 'reserve_bytes', zero=True)
    if workers > 32 or chunk_bytes > 64*1024**2:
        raise ValueError('at most 32 workers and 64 MiB per range are permitted')
    if on_progress is not None and not callable(on_progress):
        raise ValueError('on_progress must be callable or None')


def _guard(reserve_bytes, extra_bytes):
    available = available_memory_bytes()
    if type(available) is not int or available < reserve_bytes + extra_bytes:
        raise ArchiveCapacityError('observed memory reserve would be exhausted')


class SealedArchive:
    """Own an immutable full-byte cache; close releases this owner's fd."""

    def __init__(self, descriptor, receipt):
        self._descriptor = descriptor
        self._receipt = deepcopy(receipt)

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    def fileno(self):
        if self._descriptor is None:
            raise ValueError('memory archive owner is closed')
        return self._descriptor

    def close(self):
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            os.close(descriptor)

    def __enter__(self):
        self.fileno()
        return self

    def __exit__(self, *_):
        self.close()


def _fetch(source, start, size):
    # requests.Session is not shared across threads. Each worker's connection
    # and response are closed before it hands the bounded body to the owner.
    with requests.Session() as session:
        return transport._receive_range(session, source, start, size)


def _final_head(source):
    # The transport worker owns only its HTTP session, never the cache FD.
    with requests.Session() as session:
        return transport._head(session, source['url'], source['expected_bytes'])


def _write_all(descriptor, body, offset):
    view = memoryview(body)
    done = 0
    while done < len(view):
        written = os.pwrite(descriptor, view[done:], offset+done)
        if written <= 0:
            raise ArchiveError('short memory archive write made no progress')
        done += written


def _create_memfd():
    # Both paths call the same Linux kernel operation, with CLOEXEC=1 and
    # ALLOW_SEALING=2 (linux/memfd.h). Conda builds targeting an old glibc may
    # omit Python's wrapper even when the host libc exports memfd_create.
    if hasattr(os, 'memfd_create'):
        return os.memfd_create('egoemg-emg2pose-cache', 0x0001 | 0x0002)
    import ctypes
    try:
        create = ctypes.CDLL(None, use_errno=True).memfd_create
    except AttributeError as error:
        raise ArchiveError('Linux memfd_create is unavailable; no storage fallback') from error
    create.argtypes, create.restype = (ctypes.c_char_p, ctypes.c_uint), ctypes.c_int
    descriptor = create(b'egoemg-emg2pose-cache', 0x0001 | 0x0002)
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return descriptor


def _acquire_source(source, *, workers, chunk_bytes, reserve_bytes, on_progress=None):
    """Internal engineering boundary; public acquisition pins the asset HEAD."""
    _policy(workers, chunk_bytes, reserve_bytes, on_progress)
    source = deepcopy(source)
    length = source['expected_bytes']
    transport._positive(length, 'source expected bytes')
    # receive_range's bytearray, immutable return, temporary prefix and in-flight
    # completed futures are all bounded. Never queue one future per full object.
    working_bytes = 4 * workers * chunk_bytes
    _guard(reserve_bytes, length+working_bytes)
    descriptor = None
    started = time.monotonic()
    try:
        descriptor = _create_memfd()
        os.fchmod(descriptor, 0o600)
        os.ftruncate(descriptor, length)
        records = {}
        received = next_offset = 0
        inflight = {}
        head = None
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='emg2pose-range')
        try:
            while next_offset < length or inflight:
                _guard(reserve_bytes, working_bytes)
                while len(inflight) < workers and next_offset < length:
                    size = min(chunk_bytes, length-next_offset)
                    future = pool.submit(_fetch, source, next_offset, size)
                    inflight[future] = (next_offset, size)
                    next_offset += size
                done, _ = wait(inflight, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    offset, size = inflight.pop(future)
                    body = future.result()
                    if len(body) != size:
                        raise ArchiveError('range body differs from planned memory extent')
                    _guard(reserve_bytes, working_bytes)
                    _write_all(descriptor, body, offset)
                    record = {'offset': offset, 'raw_bytes': size,
                              'raw_sha256': hashlib.sha256(body).hexdigest()}
                    records[offset] = record
                    received += size
                    if on_progress:
                        on_progress(deepcopy({'complete': False, 'full_sha256': None,
                                              'full_corpus_complete': False,
                                              'source_bytes_received': received,
                                              'source': source, 'chunk': record}))
                done.clear()
            if received != length or len(records) != (length+chunk_bytes-1)//chunk_bytes:
                raise ArchiveError('incomplete memory archive extent ledger')
            head = pool.submit(_final_head, source)
            while not head.done():
                _guard(reserve_bytes, working_bytes)
                wait((head,), timeout=1)
            _guard(reserve_bytes, working_bytes)
            if head.result() != source:
                raise ArchiveError('source object identity changed at final HEAD')
        except BaseException:
            # Free the potentially large owned cache immediately, before waiting
            # for bounded HTTP workers to unwind. They never write to this fd.
            os.close(descriptor)
            descriptor = None
            for future in inflight:
                future.cancel()
            if head is not None:
                head.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        _guard(reserve_bytes, working_bytes)
        # Seal before final readback so the verified digest cannot race mutation.
        fcntl.fcntl(descriptor, _ADD_SEALS, _SEALS)
        if fcntl.fcntl(descriptor, _GET_SEALS) & _SEALS != _SEALS:
            raise ArchiveError('memory archive is not fully immutable')
        digest = hashlib.sha256()
        for offset in range(0, length, chunk_bytes):
            _guard(reserve_bytes, working_bytes)
            record = records[offset]
            body = os.pread(descriptor, record['raw_bytes'], offset)
            if len(body) != record['raw_bytes'] or hashlib.sha256(body).hexdigest() != record['raw_sha256']:
                raise ArchiveError('sealed memory range readback differs from received bytes')
            digest.update(body)
        if os.fstat(descriptor).st_size != length:
            raise ArchiveError('sealed memory object has incorrect size')
        result = {'format': 'emg2pose_sealed_memory_archive_v1', 'complete': True,
                  'full_corpus_complete': False, 'source': source,
                  'source_bytes_received': received, 'full_sha256': digest.hexdigest(),
                  'chunks': [records[o] for o in sorted(records)], 'seals': _SEALS,
                  'elapsed_seconds': time.monotonic()-started,
                  'storage': 'volatile owned memfd; every original HTTP object byte retained',
                  'source_identity_policy': 'ETag is transport identity, not a publisher checksum; SHA256 identifies received bytes'}
        if on_progress:
            on_progress(deepcopy(result))
        owner = SealedArchive(descriptor, result)
        descriptor = None
        return owner
    finally:
        if descriptor is not None:
            os.close(descriptor)


def acquire_archive(asset: str, *, workers=8, chunk_bytes=CHUNK_BYTES,
                    reserve_bytes=RESERVE_BYTES, on_progress=None) -> SealedArchive:
    """Acquire an exact official asset; only a fully read-back sealed fd returns.

    Callers own the returned context manager and must close it. This function
    writes no data or receipts to disk and launches no training process. All
    errors release its owned fd; the independent persistent receiver is untouched.
    """
    _policy(workers, chunk_bytes, reserve_bytes, on_progress)
    if not isinstance(asset, str) or asset not in transport._ASSETS:
        raise ValueError('asset must be full or mini')
    url, expected = transport._ASSETS[asset]
    with requests.Session() as session:
        source = transport._head(session, url, expected)
    return _acquire_source(source, workers=workers, chunk_bytes=chunk_bytes,
                           reserve_bytes=reserve_bytes, on_progress=on_progress)
