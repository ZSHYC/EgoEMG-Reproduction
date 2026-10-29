"""Bind the recorder's compact JSON to its actual complete parent identity."""
from __future__ import annotations

import os

from egoemg_repro.emg2pose_memory_process import process_identity,same_process


def recorder_owner(snapshot):
    if (type(snapshot)is not dict or set(snapshot)!={'pid','start_ticks'}or
            type(snapshot['pid'])is not int or snapshot['pid']<=0 or
            type(snapshot['start_ticks'])is not str or not snapshot['start_ticks'].isdigit()):
        raise ValueError('compact recorder PID/start identity is malformed')
    actual=process_identity(os.getppid())
    if (actual is None or actual['uid']!=os.getuid()or
            any(actual[k]!=snapshot[k]for k in('pid','start_ticks'))or not same_process(actual)):
        raise ValueError('recorded recorder is not the actual live same-user parent')
    return actual
