"""Bounded same-range body retries around the unchanged strict archive reader.

This helper never writes a block or advances the archive manifest. The supplied
reader must still validate every HTTP response and return only complete bytes.
Its existing request/header retries remain independently bounded.
"""
from __future__ import annotations

import math
import time

import requests

from .emg2pose_archive import ArchiveError


def receive_with_retries(receive, session, source, start, size, *, guard,
                         delays=(1., 2.), on_retry=None):
    """Allow at most three body attempts for one identity-bound original range.

    Only an interrupted-body error explicitly caused by a requests connection,
    timeout or truncated-transfer exception is retryable. TLS, protocol, content,
    storage and interruption failures propagate. The guard runs before every
    attempt and before backoff; a guard or observer failure also propagates.
    """
    if (type(delays) not in (tuple, list) or len(delays) > 2 or
            any(type(d) not in (int, float) or not math.isfinite(d) or d < 0
                for d in delays)):
        raise ValueError('delays must contain at most two finite nonnegative numbers')
    if not callable(receive) or not callable(guard) or (on_retry is not None and not callable(on_retry)):
        raise ValueError('receive, guard and any retry observer must be callable')
    delays = tuple(delays)
    source = dict(source)
    for attempt in range(1, len(delays)+2):
        guard()
        try:
            return receive(session, source, start, size)
        except ArchiveError as error:
            cause = error.__cause__
            retryable = (
                str(error) == 'incomplete or interrupted range body; no chunk committed' and
                isinstance(cause, (requests.ConnectionError, requests.Timeout,
                                   requests.exceptions.ChunkedEncodingError)) and
                not isinstance(cause, requests.exceptions.SSLError))
            if not retryable or attempt > len(delays):
                raise
            guard()
            delay = delays[attempt-1]
            if on_retry is not None:
                on_retry(dict(failed_attempt=attempt, next_attempt=attempt+1,
                              start=start, size=size, delay_seconds=delay,
                              error_type=type(cause).__name__))
            time.sleep(delay)
