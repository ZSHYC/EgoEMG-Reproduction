"""Owned aria2 byte acquisition into a persistent ordinary TAR file.

This is a separate transport receipt, never scientific corpus admission.
Preallocated file length is not received-byte progress. Explicit resumes bind
both the failed partial bytes and the native aria2 control file before mutation.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import time
import urllib.error
import urllib.request
import uuid

import requests

from . import emg2pose_archive as transport,emg2pose_data as storage
from .emg2pose_memory_archive import available_memory_bytes,RESERVE_BYTES
from .emg2pose_memory_process import run_owned_stage,_atomic


FORMAT='emg2pose_disk_byte_acquisition_v1'
DISK_RESERVE=15*1024**3
WORKSPACE=256*1024**2
_IMPORTED_SHA=storage._sha256(__file__)
_HASH=re.compile('[0-9a-f]{64}')


def _resources(parent,extra=0):
    if available_memory_bytes()<RESERVE_BYTES+WORKSPACE:raise RuntimeError('unchanged256GiB host reserve required')
    if shutil.disk_usage(parent).free<DISK_RESERVE+extra:raise RuntimeError('unchanged15GiB disk reserve required')
    if storage._sha256(__file__)!=_IMPORTED_SHA:raise ValueError('imported disk acquisition producer changed')


def _file_sha(path,guard=None):
    digest=hashlib.sha256()
    with storage._safe_path(path,file=True).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            if guard is not None:guard()
            digest.update(block)
    return digest.hexdigest()


def acquire_disk_archive(asset,output,*,workers=8,resume=False,expected_failure_sha256=None,on_progress=None):
    """Acquire the original official mini/full TAR; do not inspect/train it."""
    if asset not in transport._ASSETS:raise ValueError('explicit official mini or full asset required')
    url,length=transport._ASSETS[asset]
    with requests.Session() as session:
        session.trust_env=False;source=transport._head(session,url,length)
    expected=storage.MINI_SHA256 if asset=='mini' else None
    return _acquire_source(source,output,asset=asset,workers=workers,expected_sha256=expected,
        resume=resume,expected_failure_sha256=expected_failure_sha256,on_progress=on_progress)


def _acquire_source(source,output,*,asset,workers=8,min_split_bytes=16*1024**2,expected_sha256=None,
                    resume=False,expected_failure_sha256=None,on_progress=None):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('one through eight actual connections required')
    if type(min_split_bytes) is not int or not 1024**2<=min_split_bytes<=16*1024**2:raise ValueError('bounded split bytes required')
    if type(resume) is not bool or (on_progress is not None and not callable(on_progress)):raise ValueError('explicit resume/callback required')
    if expected_sha256 is not None and (type(expected_sha256) is not str or not _HASH.fullmatch(expected_sha256)):
        raise ValueError('separately frozen complete SHA256 required')
    if (type(source) is not dict or set(source)!={'url','expected_bytes','etag','last_modified'} or
            type(source['expected_bytes']) is not int or not 0<source['expected_bytes']<=transport._ASSETS['full'][1]):
        raise ValueError('exact source HEAD identity required')
    output=storage._safe_path(output);parent=storage._safe_path(output.parent)
    if not parent.is_dir():raise ValueError('existing output parent required')
    binary=shutil.which('aria2c')
    if binary is None:raise RuntimeError('installed aria2c required for disk acquisition')
    policy=dict(workers=workers,min_split_bytes=min_split_bytes,file_allocation='falloc',connect_timeout=15,
        timeout=60,max_tries=3,retry_wait=2,expected_sha256=expected_sha256,binary_sha256=_file_sha(binary),
        producer_sha256=_IMPORTED_SHA,host_reserve_bytes=RESERVE_BYTES,disk_reserve_bytes=DISK_RESERVE)
    archive=output/'archive.tar';control=output/'archive.tar.aria2';previous=None
    if not resume:
        if output.exists():raise FileExistsError('existing download output is never overwritten')
        _resources(parent,source['expected_bytes']+32*1024**2);output.mkdir(mode=0o700)
    elif not output.is_dir():raise ValueError('actual failed owned output required for resume')
    lock=os.open(output/'archive.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise FileExistsError('download still has a live owner') from error
        if resume:
            failure=output/'failure.json'
            if ((output/'receipt.json').exists() or type(expected_failure_sha256) is not str or
                    not _HASH.fullmatch(expected_failure_sha256) or _file_sha(failure)!=expected_failure_sha256):
                raise ValueError('separately frozen actual failure receipt required')
            previous=storage._json(failure)
            if (previous.get('source')!=source or previous.get('policy')!=policy or
                    previous.get('complete_bytes') is not False or previous.get('actual_owned_consumers_exited') is not True):
                raise ValueError('failed source/policy/actual terminal ownership differs')
            for path,key in ((archive,'partial_sha256'),(control,'control_sha256')):
                if previous.get(key) is None or _file_sha(path,lambda:_resources(parent))!=previous[key]:
                    raise ValueError('failed partial/control bytes differ before resume')
            _resources(parent,max(0,source['expected_bytes']-archive.stat().st_blocks*512)+32*1024**2)
        with requests.Session() as session:
            session.trust_env=False
            if transport._head(session,source['url'],source['expected_bytes'])!=source:
                raise ValueError('source changed before acquisition/resume')
        attempts=output/'attempts';attempts.mkdir(mode=0o700,exist_ok=True)
        attempt=uuid.uuid4().hex;stage=attempts/attempt
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        token=secrets.token_hex(32);config=output/('rpc-'+attempt+'.conf')
        text=f'enable-rpc=true\nrpc-listen-all=false\nrpc-allow-origin-all=false\nrpc-listen-port={port}\nrpc-secret={token}\n'
        fd=os.open(config,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'w') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def rpc(method,*params):
            body=json.dumps(dict(jsonrpc='2.0',id=1,method=method,params=['token:'+token,*params])).encode()
            request=urllib.request.Request(f'http://127.0.0.1:{port}/jsonrpc',data=body,headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=2) as response:packet=json.loads(response.read(1024**2))
            if 'error' in packet:raise RuntimeError('owned aria2 RPC refused command')
            return packet['result']
        progress=dict(source_bytes_received=0,total_bytes=source['expected_bytes'],maximum_connections_observed=0)
        started=time.monotonic();last=None;shutdown=False
        def guard():
            nonlocal last,shutdown
            _resources(parent)
            log=stage/'worker.log'
            if log.exists() and log.stat().st_size>16*1024**2:raise RuntimeError('owned download log bound exceeded')
            if time.monotonic()-started>7*24*3600:raise RuntimeError('owned acquisition phase time budget exceeded')
            if shutdown:return
            try:
                jobs=rpc('aria2.tellActive')+rpc('aria2.tellWaiting',0,10)+rpc('aria2.tellStopped',0,10)
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                if time.monotonic()-started<30:return
                raise RuntimeError('owned aria2 RPC unavailable beyond startup bound')
            if not jobs:
                if time.monotonic()-started<30:return
                raise RuntimeError('owned aria2 lost its original source job')
            # The same GID can move from active to stopped between RPC calls.
            jobs={job['gid']:job for job in jobs}
            if len(jobs)!=1:raise RuntimeError('owned acquisition must have exactly one source job')
            job=next(iter(jobs.values()));total=int(job['totalLength']);done=int(job['completedLength'])
            if total not in (0,source['expected_bytes']) or not 0<=done<=source['expected_bytes']:
                raise RuntimeError('actual aria2 byte accounting differs from source')
            progress.update(source_bytes_received=done,maximum_connections_observed=max(
                progress['maximum_connections_observed'],int(job.get('connections','0'))))
            value=(done,job['status'],progress['maximum_connections_observed'])
            if value!=last:
                _atomic(output/'progress.json',dict(format=FORMAT,status=job['status'],**progress));last=value
                if on_progress is not None:on_progress(dict(progress))
            if job['status']=='error':raise RuntimeError('owned aria2 source download failed: '+job.get('errorCode','unknown'))
            if job['status']=='complete':
                if done!=source['expected_bytes']:raise RuntimeError('aria2 reported incomplete complete bytes')
                rpc('aria2.shutdown');shutdown=True
        argv=[binary,'--conf-path='+str(config),'--no-netrc=true','--all-proxy=','--http-proxy=','--https-proxy=',
            '--file-allocation=falloc','--split='+str(workers),'--max-connection-per-server='+str(workers),
            '--min-split-size='+str(min_split_bytes),'--piece-length='+str(min_split_bytes),'--max-concurrent-downloads=1',
            '--max-tries=3','--retry-wait=2','--connect-timeout=15','--timeout=60','--auto-save-interval=1',
            '--auto-file-renaming=false','--allow-overwrite=false','--continue='+str(resume).lower(),
            '--check-certificate=true','--http-accept-gzip=false','--header=Accept-Encoding: identity',
            '--header=If-Match: '+source['etag'],'--header=If-Unmodified-Since: '+source['last_modified'],
            '--show-console-readout=false','--summary-interval=60','--enable-color=false',
            '--dir='+str(output),'--out=archive.tar']
        if expected_sha256 is not None:argv+=['--check-integrity=true','--checksum=sha-256='+expected_sha256]
        argv.append(source['url'])
        try:
            owned=run_owned_stage(argv,stage,descriptors={},guard=guard,poll_seconds=.2,terminate_grace=5.)
            if not shutdown or archive.stat().st_size!=source['expected_bytes']:
                raise RuntimeError('actual complete byte transfer and owned exit required')
            digest=_file_sha(archive,lambda:_resources(parent))
            if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('complete received SHA256 differs')
            with requests.Session() as session:
                session.trust_env=False;final=transport._head(session,source['url'],source['expected_bytes'])
            if final!=source:raise ValueError('source changed at final HEAD')
            _resources(parent);os.chmod(archive,0o400)
            receipt=dict(format=FORMAT,asset=asset,source=source,final_source=final,complete_bytes=True,
                source_bytes_received=source['expected_bytes'],full_sha256=digest,policy=policy,owned_stage=owned,
                maximum_connections_observed=progress['maximum_connections_observed'],
                resumed_previous_failure_sha256=expected_failure_sha256 if resume else None,
                seconds=time.monotonic()-started,source_qualified=False,full_corpus_complete=False,
                scientific_complete=False,full_paper_complete=False)
            _atomic(output/'receipt.json',receipt);return receipt
        except BaseException as error:
            stage_result=storage._json(stage/'stage-result.json') if (stage/'stage-result.json').is_file() else None
            exited=stage_result is not None and stage_result.get('all_owned_consumers_exited') is True
            partial=_file_sha(archive) if archive.is_file() and exited else None
            native=_file_sha(control) if control.is_file() and exited else None
            failure=dict(format=FORMAT,asset=asset,source=source,policy=policy,complete_bytes=False,
                error_type=type(error).__name__,error=str(error),actual_owned_consumers_exited=exited,
                partial_sha256=partial,control_sha256=native,**progress)
            _atomic(output/'failure.json',failure)
            _atomic(attempts/(attempt+'-failure.json'),failure);raise
    finally:os.close(lock)
