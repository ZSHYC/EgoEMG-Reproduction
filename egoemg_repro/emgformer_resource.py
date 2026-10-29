"""Idle-device selection only; not a GPU/model/training qualification."""
from __future__ import annotations

import re


RESERVE_MIB=65536
_UUID=re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def select_idle_emgformer_gpu(gpus,applications,*,candidates):
    """Choose a permitted empty device from one complete NVML observation.

    The caller supplies the independently fixed candidate/reservation policy.
    Any compute application, regardless of owner, makes its device unavailable.
    Malformed or incomplete observations raise instead of implying idle capacity.
    """
    if (type(gpus)is not list or type(applications)is not list or
        type(candidates)not in (list,tuple)or not candidates or
        any(type(u)is not str or _UUID.fullmatch(u)is None for u in candidates)or
        len(set(candidates))!=len(candidates)):
        raise ValueError('complete typed GPU inventory and distinct explicit candidates required')
    devices={};indices=set()
    for gpu in gpus:
        if (type(gpu)is not dict or set(gpu)!={'index','uuid','free_mib','total_mib'}or
            type(gpu['uuid'])is not str or _UUID.fullmatch(gpu['uuid'])is None or
            any(type(gpu[k])is not int for k in ('index','free_mib','total_mib'))or
            gpu['index']<0 or not 0<=gpu['free_mib']<=gpu['total_mib']or gpu['total_mib']<=0 or
            gpu['uuid']in devices or gpu['index']in indices):
            raise ValueError('complete distinct typed GPU records required')
        devices[gpu['uuid']]=gpu;indices.add(gpu['index'])
    if any(u not in devices for u in candidates):
        raise ValueError('candidate missing from complete GPU inventory')
    seen=set();busy=set()
    for app in applications:
        if (type(app)is not dict or set(app)!={'pid','uuid'}or type(app['pid'])is not int or
            app['pid']<=0 or type(app['uuid'])is not str or app['uuid']not in devices or
            (app['pid'],app['uuid'])in seen):
            raise ValueError('complete unique typed compute application records required')
        seen.add((app['pid'],app['uuid']));busy.add(app['uuid'])
    for uuid in candidates:
        if uuid not in busy and devices[uuid]['free_mib']>=RESERVE_MIB:
            return dict(devices[uuid])
    return None
