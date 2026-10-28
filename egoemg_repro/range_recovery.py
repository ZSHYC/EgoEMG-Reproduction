"""Bounded transport recovery around the unchanged strict range implementation.

The caller must own the destination exclusively, as for StrictRangeDownloader.
Only complete committed chunks can advance a retry budget. Protocol, capacity,
certificate and unrelated errors are never converted into retryable failures.
"""
from __future__ import annotations

from pathlib import Path
import time

import requests

from .download import DownloadError
from . import range_download as strict


def _stamp(path):
    try:
        value=path.stat()
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    except FileNotFoundError:
        return None


def _size(stamp):
    return 0 if stamp is None else stamp[2]


def _transient(error):
    if isinstance(error,requests.exceptions.ProxyError):
        return False
    if isinstance(error,requests.exceptions.SSLError):
        # Other SSL errors include certificate validation and protocol failures.
        # Limit SSL recovery to the actual observed temporary EOF condition.
        message=str(error).casefold()
        return ('unexpected_eof_while_reading' in message or
                'eof occurred in violation of protocol' in message or
                'ssleoferror' in message)
    return isinstance(error,(requests.exceptions.ConnectionError,requests.exceptions.Timeout,
                             requests.exceptions.ChunkedEncodingError))


def _committed_progress(before,after,*,chunk_bytes,expected_bytes):
    initial,current=_size(before),_size(after)
    if current<initial or current>expected_bytes:
        raise DownloadError('partial size changed outside the strict range bounds')
    if before is not None and (after is None or before[:2]!=after[:2]):
        raise DownloadError('partial replaced during an interrupted strict range')
    if current==initial and before!=after:
        raise DownloadError('partial changed without a committed strict chunk')
    if current>initial and current!=expected_bytes and (current-initial)%chunk_bytes:
        raise DownloadError('partial advanced outside complete strict chunks')


class BoundedRangeDownloader:
    """Three attempts per stalled offset, backoff 1/2 s, native writer intact.

    ``sleep`` and ``session_factory`` permit deterministic offline fault tests.
    Real intake uses the default sleeper and anonymous native requests sessions.
    Evidence/cancellation callback failures propagate immediately.
    """

    def __init__(self,asset,*,chunk_bytes=strict.CHUNK_BYTES,session_factory=None,
                 sleep=time.sleep,on_retry=None):
        self.asset=asset
        self.native=strict.StrictRangeDownloader(asset,chunk_bytes=chunk_bytes,session_factory=session_factory)
        self.chunk_bytes=chunk_bytes
        self.sleep,self.on_retry=sleep,on_retry
        self._statistics={}
        self._reset_statistics()

    def _reset_statistics(self):
        self._statistics=dict(transient_failures=0,scheduled_retries=0,exhausted=False,last_failure_offset=None)

    @property
    def statistics(self):
        return dict(self._statistics)

    def __call__(self,drive_id,part):
        if drive_id!=self.asset.drive_id:
            raise DownloadError('range transport asset identity mismatch')
        part=Path(part)
        # The native helper preserves recognized quota documents. Do this before
        # taking an identity snapshot, so its intentional move is not a shrink.
        strict.quarantine_quota_html(part)
        self._reset_statistics()
        last_offset=None
        failures=0
        waiting=False
        expected_stamp=None
        while True:
            before=_stamp(part)
            if waiting and before!=expected_stamp:
                raise DownloadError('partial changed during recovery wait')
            try:
                return self.native(drive_id,part)
            except requests.exceptions.RequestException as error:
                if not _transient(error):
                    raise
                self._statistics['transient_failures']+=1
                after=_stamp(part)
                _committed_progress(before,after,chunk_bytes=self.chunk_bytes,
                                    expected_bytes=self.asset.expected_bytes)
                offset=_size(after)
                failures=failures+1 if offset==last_offset else 1
                last_offset=offset
                self._statistics['last_failure_offset']=offset
                will_retry=failures<3
                delay=failures if will_retry else 0
                self._statistics['exhausted']=not will_retry
                if will_retry:
                    self._statistics['scheduled_retries']+=1
                event=dict(offset=offset,failure_number=failures,will_retry=will_retry,
                           wait_seconds=delay,exception_type=type(error).__name__)
                if self.on_retry is not None:
                    self.on_retry(event)
                if not will_retry:
                    raise
                self.sleep(delay)
                expected_stamp=after
                waiting=True
