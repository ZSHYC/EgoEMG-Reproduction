"""Pinned 9fb Session adapter; only the original archive writer touches disk.

This API is for a separately reviewed future receiver. Importing it neither
changes the archive module nor starts a thread/request. Raw-buffer accounting
does not claim a process-RSS bound or a measured network speed improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import re
import threading
import time
import types
from urllib.parse import urlsplit

import requests

from . import emg2pose_archive as archive

ARCHIVE_SHA256 = '9fb6681fc4e3090573cfb01b04ad73d2052fb441085c39ed9dfec7b6939059e1'
_URLS = frozenset({
    'https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar',
    'https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar',
})
_SIGNATURE = "(output, *, asset='full', chunk_bytes=16777216, max_chunks=None, budget_bytes=365072220160, reserve_bytes=16106127360)"
_new_session = requests.Session


@dataclass(frozen=True)
class ExplicitRoute:
    proxy_url: str

    def __post_init__(self):
        try:
            url = urlsplit(self.proxy_url)
            valid = (url.scheme == 'socks5h' and url.hostname == '127.0.0.1' and
                     url.port is not None and 1 <= url.port <= 65535 and
                     url.username is None and url.password is None and
                     not url.path and not url.query and not url.fragment)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('route must be an explicit credential-free loopback socks5h endpoint')


def _code_shape(code):
    """Compare executable content without deployment filename/line offsets."""
    return (code.co_code, tuple(_code_shape(x) if isinstance(x, types.CodeType) else x
                               for x in code.co_consts), code.co_names, code.co_varnames,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_flags, code.co_freevars, code.co_cellvars)


def _pinned_acquire():
    source = Path(archive.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != ARCHIVE_SHA256:
        raise archive.ArchiveError('archive source hash pin differs')
    original = archive.acquire_archive
    if not isinstance(original, types.FunctionType) or str(inspect.signature(original)) != _SIGNATURE:
        raise archive.ArchiveError('archive signature pin differs')
    # Compile, do not execute, the exact pinned source. This also detects an
    # in-memory replacement before/after this adapter was imported.
    module_code = compile(source, archive.__file__, 'exec', dont_inherit=True)
    expected_codes = {x.co_name: x for x in module_code.co_consts if isinstance(x, types.CodeType)}
    expected = expected_codes['acquire_archive']
    if (original.__globals__ is not vars(archive) or original.__code__.co_freevars or
            _code_shape(original.__code__) != _code_shape(expected)):
        raise archive.ArchiveError('archive executable code pin differs')
    for name in ('_request', '_identity_headers', '_head', '_receive_range'):
        helper = getattr(archive, name)
        if (not isinstance(helper, types.FunctionType) or helper.__globals__ is not vars(archive) or
                _code_shape(helper.__code__) != _code_shape(expected_codes[name])):
            raise archive.ArchiveError('archive strict helper code pin differs: '+name)
    return original


def _headers(response):
    values = dict(response.headers)
    if len(values) > 64 or sum(len(str(k))+len(str(v)) for k, v in values.items()) > 16_384:
        raise archive.ArchiveError('response header evidence exceeds bounded size')
    return values


class _ObservedResponse:
    def __init__(self, response, owner):
        self.response, self.owner = response, owner
        self.status_code, self.headers = response.status_code, response.headers
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def iter_content(self, chunk_size):
        for part in self.response.iter_content(chunk_size=chunk_size):
            with self.owner.counter_lock:
                self.owner.receipt['wire_body_bytes'] += len(part)
            if self.owner.cancelled.is_set():
                raise archive.ArchiveError('prefetch cancelled')
            yield part

    def close(self):
        with self.owner.counter_lock:
            if self.closed:
                return
            self.closed = True
            self.owner.active.discard(self)
        self.response.close()


class _Transport:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False
        self.session = _new_session()
        self.session.trust_env = False
        self.session.auth = None
        self.session.cookies.clear()
        self.session.proxies.clear()
        self.last = None

    def request(self, method, url, *, headers, timeout, stream, allow_redirects):
        if self.owner.cancelled.is_set():
            raise archive.ArchiveError('prefetch cancelled')
        if (url not in _URLS or method not in {'HEAD', 'GET'} or timeout != (15, 60) or
                stream is not True or allow_redirects is not False):
            raise archive.ArchiveError('unapproved transport request')
        self.session.cookies.clear()
        self.session.proxies.clear()
        started = time.monotonic_ns()
        with self.owner.counter_lock:
            self.owner.receipt['request_attempts'] += 1
        raw = self.session.request(method, url, headers=headers, timeout=timeout, stream=True,
                                   allow_redirects=False, verify=True,
                                   proxies={'http': self.owner.route.proxy_url,
                                            'https': self.owner.route.proxy_url})
        response = _ObservedResponse(raw, self.owner)
        try:
            self.last = {'status': raw.status_code, 'headers': _headers(raw),
                         'request_started_ns': started, 'headers_observed_ns': time.monotonic_ns()}
        except BaseException:
            response.close()
            raise
        with self.owner.counter_lock:
            cancelled = self.owner.cancelled.is_set()
            if not cancelled:
                self.owner.active.add(response)
        if cancelled:
            response.close()
            raise archive.ArchiveError('prefetch cancelled')
        return response

    def close(self):
        with self.owner.counter_lock:
            if self.closed:
                return
            self.closed = True
        self.session.close()


class _BufferedResponse:
    def __init__(self, data, evidence):
        self.data = data
        self.status_code = evidence['status']
        self.headers = requests.structures.CaseInsensitiveDict(evidence['headers'])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.data), chunk_size):
            yield self.data[offset:offset+chunk_size]

    def close(self):
        self.data = b''


class _PrefetchSession:
    def __init__(self, *, route, chunk_bytes, max_chunks, receipt, workers=1, max_pending_chunks=2):
        self.route, self.chunk_bytes, self.max_chunks = route, chunk_bytes, max_chunks
        self.workers, self.max_pending = workers, max_pending_chunks
        self.receipt = receipt
        self.receipt.update(peak_pending_chunks=0, peak_reserved_raw_bytes=0,
                            pending_chunks_at_close=0, reserved_raw_bytes_at_close=0,
                            verified_ranges=0, discarded_ranges=0)
        self.receipt.setdefault('request_attempts', 0)
        self.trust_env, self.auth, self.cookies, self.proxies = False, None, {}, {}
        self.cancelled = threading.Event()
        self.counter_lock = threading.Lock()
        self.active = set()
        self.head_transport = None
        self.source, self.next_offset = None, None
        self.head_url = None
        self.executor = None
        self.transports = set()
        self.worker_local = threading.local()
        self.pending = {}
        self.borrowed = None
        self.schedule_next = self.limit = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancelled.set()
        with self.counter_lock:
            active = tuple(self.active)
            transports = tuple(self.transports)
        for response in active:
            response.close()
        for transport in transports:
            transport.close()
        if self.head_transport:
            self.head_transport.close()
        if self.executor:
            for _, future in self.pending.values():
                future.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)
        # A worker may have been constructing its Session when cancellation
        # took the first snapshot. Joining precedes this final resource sweep.
        for transport in tuple(self.transports):
            transport.close()
        self.receipt['discarded_ranges'] += len(self.pending)
        self.pending.clear()
        self.borrowed = None
        self.receipt['pending_chunks_at_close'] = 0
        self.receipt['reserved_raw_bytes_at_close'] = 0

    def _worker_transport(self):
        if self.cancelled.is_set():
            raise archive.ArchiveError('prefetch cancelled')
        transport = getattr(self.worker_local, 'transport', None)
        if transport is None:
            transport = _Transport(self)
            with self.counter_lock:
                cancelled = self.cancelled.is_set()
                if not cancelled:
                    self.transports.add(transport)
            if cancelled:
                transport.close()
                raise archive.ArchiveError('prefetch cancelled')
            self.worker_local.transport = transport
        return transport

    def _fetch(self, start, size):
        try:
            transport = self._worker_transport()
            data = archive._receive_range(transport, self.source, start, size)
            if self.cancelled.is_set():
                raise archive.ArchiveError('prefetch cancelled')
            with self.counter_lock:
                self.receipt['verified_ranges'] += 1
            return data, dict(transport.last, body_verified_ns=time.monotonic_ns())
        except BaseException:
            # Do not wait behind a slow earlier offset before propagating a
            # later worker's failure. Stop scheduling/retries and unblock every
            # active response; only the owning main thread joins the executor.
            self.cancelled.set()
            with self.counter_lock:
                active = tuple(self.active)
            for response in active:
                response.close()
            raise

    def _fill(self):
        # A delivered slot stays borrowed until the NEXT main GET. Refilling
        # while the original strict reader is copying it would break the raw
        # copy budget even if the ready queue itself contained only two items.
        while len(self.pending)+(self.borrowed is not None) < self.max_pending and self.schedule_next < self.limit:
            if self.cancelled.is_set():
                raise archive.ArchiveError('prefetch cancelled')
            start = self.schedule_next
            size = min(self.chunk_bytes, self.source['expected_bytes']-start)
            self.schedule_next += size
            self.pending[start] = (size, self.executor.submit(self._fetch, start, size))
            count = len(self.pending)+(self.borrowed is not None)
            reserved = 4*(sum(n for n, _ in self.pending.values())+
                          (self.borrowed[1] if self.borrowed else 0))
            if self.max_pending == 1:
                reserved += self.chunk_bytes  # previous writer raw survives the next GET
            self.receipt['peak_pending_chunks'] = max(self.receipt['peak_pending_chunks'], count)
            self.receipt['peak_reserved_raw_bytes'] = max(self.receipt['peak_reserved_raw_bytes'], reserved)

    def request(self, method, url, *, headers, timeout, stream, allow_redirects):
        if self.closed or timeout != (15, 60) or stream is not True or allow_redirects is not False:
            raise archive.ArchiveError('closed or unapproved prefetch request')
        if method == 'HEAD':
            if (self.source is not None or headers != {'Accept-Encoding': 'identity'} or
                    self.head_url is not None and url != self.head_url):
                raise archive.ArchiveError('unexpected HEAD request')
            self.head_url = url
            if self.head_transport is None:
                self.head_transport = _Transport(self)
            return self.head_transport.request(method, url, headers=headers, timeout=timeout,
                                               stream=stream, allow_redirects=allow_redirects)
        if method != 'GET' or self.head_transport is None:
            raise archive.ArchiveError('unexpected prefetch method or GET before HEAD')
        if url != self.head_url:
            raise archive.ArchiveError('GET source URL differs from verified HEAD')
        actual = self.head_transport.last
        identity = requests.structures.CaseInsensitiveDict(actual['headers'])
        source = {'url': url, 'expected_bytes': int(identity['Content-Length']),
                  'etag': identity['ETag'], 'last_modified': identity['Last-Modified']}
        match = re.fullmatch(r'bytes=([0-9]+)-([0-9]+)', str(headers.get('Range', '')))
        if match is None or actual['status'] != 200:
            raise archive.ArchiveError('invalid Range/HEAD identity')
        start, end = map(int, match.groups())
        size = min(self.chunk_bytes, source['expected_bytes']-start)
        if (start % self.chunk_bytes or size <= 0 or end != start+size-1 or
                headers != {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                            'If-Match': source['etag']} or
                self.source is not None and (self.source != source or start != self.next_offset)):
            raise archive.ArchiveError('unexpected range order, size or source identity')
        if self.source is None:
            self.source = source
            self.schedule_next = start
            self.limit = source['expected_bytes'] if self.max_chunks is None else min(
                source['expected_bytes'], start+self.max_chunks*self.chunk_bytes)
            self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='emg2pose-prefetch')
        if start >= self.limit:
            raise archive.ArchiveError('GET exceeds explicit new-chunk bound')
        self.borrowed = None
        self._fill()
        if start not in self.pending:
            raise archive.ArchiveError('requested range is absent from bounded prefetch queue')
        pending_size, future = self.pending[start]
        if pending_size != size:
            raise archive.ArchiveError('prefetch entry size differs')
        data, evidence = future.result()
        del self.pending[start]
        self.borrowed = (start, size)
        self.next_offset = start+size
        self.receipt['delivered_raw_bytes'] += len(data)
        entry = {'offset': start, 'size': size, 'sha256': hashlib.sha256(data).hexdigest(),
                 'ordered_delivery_ns': time.monotonic_ns(), **evidence}
        self.receipt['ranges'].append(entry)
        del self.receipt['ranges'][:-64]
        return _BufferedResponse(data, evidence)


def acquire_prefetched(output, *, route, asset='full', chunk_bytes=archive.CHUNK_BYTES,
                       max_chunks=None, budget_bytes=340*1024**3, reserve_bytes=15*1024**3,
                       workers=2, max_pending_chunks=2, raw_memory_budget_bytes=128*1024**2,
                       receipt=None):
    """Run the exact original writer with an invocation-local Session factory.

    ``receipt`` is a bounded private in-memory diagnostic object, never a new
    archive-manifest field. No file is written except by the original writer.
    """
    original = _pinned_acquire()
    if not isinstance(route, ExplicitRoute):
        raise ValueError('an explicit route is required')
    if type(workers) is not int or workers not in {1, 2}:
        raise ValueError('workers must be 1 or 2')
    if type(max_pending_chunks) is not int or not 1 <= max_pending_chunks <= 2:
        raise ValueError('max_pending_chunks must be 1 or 2')
    if workers > max_pending_chunks:
        raise ValueError('pending slots must cover all workers')
    archive._positive(chunk_bytes, 'chunk_bytes')
    if chunk_bytes > archive.CHUNK_BYTES:
        raise ValueError('chunk_bytes cannot exceed 16 MiB')
    archive._positive(raw_memory_budget_bytes, 'raw_memory_budget_bytes')
    raw_copy_bound = max(4*max_pending_chunks, 5)*chunk_bytes
    if raw_copy_bound > raw_memory_budget_bytes:
        raise ValueError('raw-buffer budget cannot cover bounded reader copies')
    if receipt is None:
        receipt = {}
    if not isinstance(receipt, dict) or receipt:
        raise ValueError('receipt must be an empty dictionary')
    receipt.update(archive_source_sha256=ARCHIVE_SHA256, wire_body_bytes=0, delivered_raw_bytes=0,
                   started_monotonic_ns=time.monotonic_ns(),
                   ranges=[], workers=workers, max_pending_chunks=max_pending_chunks,
                   raw_memory_budget_bytes=raw_memory_budget_bytes,
                   raw_copy_budget_bound_bytes=raw_copy_bound,
                   wire_body_bytes_scope='transport iter_content bytes including discarded/retried bodies; not TCP wire or durable progress',
                   delivered_raw_bytes_scope='ordered provider delivery; not original-writer commit')
    def factory():
        return _PrefetchSession(route=route, chunk_bytes=chunk_bytes, max_chunks=max_chunks,
                                workers=workers, max_pending_chunks=max_pending_chunks, receipt=receipt)
    scope = dict(original.__globals__)
    scope['requests'] = types.SimpleNamespace(Session=factory)
    adapted = types.FunctionType(original.__code__, scope, original.__name__, original.__defaults__)
    adapted.__kwdefaults__ = original.__kwdefaults__.copy()
    try:
        result = adapted(output, asset=asset, chunk_bytes=chunk_bytes, max_chunks=max_chunks,
                         budget_bytes=budget_bytes, reserve_bytes=reserve_bytes)
    except BaseException:
        receipt['invocation_status'] = 'failed'
        raise
    else:
        receipt.update(invocation_status='returned', returned_committed_raw_bytes=result['source_bytes_received'],
                       returned_archive_complete=result['complete'])
        return result
    finally:
        receipt['finished_monotonic_ns'] = time.monotonic_ns()
