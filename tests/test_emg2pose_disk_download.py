"""Actual installed aria2, bounded HTTP bytes and owned persistent acquisition."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import threading
import time
from types import SimpleNamespace

import pytest

ARIA2=shutil.which('aria2c')
pytestmark=pytest.mark.skipif(ARIA2 is None,reason='actual installed aria2 binary required; separately executed on eeg-server')


def api():return importlib.import_module('egoemg_repro.emg2pose_disk_download')
def sha(body):return hashlib.sha256(body).hexdigest()


@pytest.fixture(autouse=True)
def bounded_fixture_capacity(monkeypatch):
    # Only the 5MiB loopback unit fixture gets synthetic resource observations.
    # The actual official mini/full controller uses the unmodified real probes.
    if ARIA2 is None:return
    module=api()
    monkeypatch.setattr(module,'available_memory_bytes',lambda:2*1024**4)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _:SimpleNamespace(free=64*1024**3))


@pytest.fixture
def http():
    raw=bytes(range(251))*(5*1024**2//251)+b'ragged-final';state=dict(mode='ok',requests=[],fetched=False,delay=.002)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*_):pass
        def send_binary_headers(self,n,status=200,start=None,end=None):
            self.send_response(status);self.send_header('Content-Type','application/octet-stream')
            self.send_header('Content-Length',str(n));self.send_header('Accept-Ranges','bytes')
            self.send_header('ETag','"changed"' if state['mode']=='final_head' and state['fetched'] and self.command=='HEAD' else '"fixed"')
            self.send_header('Last-Modified','Wed, 12 Jun 2024 14:31:55 GMT')
            if start is not None:self.send_header('Content-Range',f'bytes {start}-{end}/{len(raw)}')
            self.end_headers()
        def do_HEAD(self):self.send_binary_headers(len(raw))
        def do_GET(self):
            state['requests'].append(dict(range=self.headers.get('Range'),if_match=self.headers.get('If-Match')))
            state['fetched']=True
            if state['mode']=='wrong_identity' or self.headers.get('If-Match')!='"fixed"':self.send_binary_headers(0,412);return
            start,end=0,len(raw)-1;requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',requested);assert match
                start=int(match[1]);end=int(match[2]) if match[2] else end
                end=min(end,len(raw)-1)
            self.send_binary_headers(end-start+1,206 if requested else 200,start if requested else None,end)
            try:
                for i in range(start,end+1,65536):
                    self.wfile.write(raw[i:min(end+1,i+65536)]);self.wfile.flush();time.sleep(state['delay'])
                    if state['mode']=='truncate' and start<1024**2:self.connection.shutdown(socket.SHUT_RDWR);self.connection.close();break
            except (BrokenPipeError,ConnectionResetError,OSError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    source=dict(url=f'http://127.0.0.1:{server.server_port}/archive.tar',expected_bytes=len(raw),etag='"fixed"',last_modified='Wed, 12 Jun 2024 14:31:55 GMT')
    try:yield SimpleNamespace(raw=raw,source=source,state=state)
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


def acquire(http,path,**options):
    return api()._acquire_source(http.source,path,asset='engineering',workers=4,min_split_bytes=1024**2,
        expected_sha256=options.pop('expected_sha256',sha(http.raw)),**options)


def test_actual_parallel_ragged_bytes_progress_and_credential_free_receipt(http,tmp_path):
    # Keep real connections open across a real RPC polling interval.
    http.state['delay']=.1;output=tmp_path/'archive';events=[]
    receipt=acquire(http,output,on_progress=events.append)
    assert (output/'archive.tar').read_bytes()==http.raw
    assert receipt['full_sha256']==sha(http.raw) and receipt['source_bytes_received']==len(http.raw)
    assert receipt['complete_bytes'] is True and receipt['source_qualified'] is False
    assert receipt['scientific_complete'] is False and receipt['full_paper_complete'] is False
    assert receipt['owned_stage']['worker_exit_code']==0 and receipt['owned_stage']['all_owned_consumers_exited'] is True
    assert receipt['maximum_connections_observed']>=2
    assert len(http.state['requests'])>1 and all(x['if_match']=='"fixed"' for x in http.state['requests'])
    assert events and max(e['source_bytes_received'] for e in events)==len(http.raw)
    assert all(0<=e['source_bytes_received']<=len(http.raw) for e in events)
    public=json.dumps(receipt);assert 'rpc-secret' not in public and 'token:' not in public
    assert not (output/'completion.json').exists()


@pytest.mark.parametrize('mode',['wrong_identity','final_head','truncate'])
def test_actual_transport_failure_never_publishes_complete_byte_receipt(http,tmp_path,mode):
    http.state['mode']=mode;output=tmp_path/'failed'
    with pytest.raises(Exception):acquire(http,output)
    assert not (output/'receipt.json').exists()
    failure=json.loads((output/'failure.json').read_text());assert failure['complete_bytes'] is False
    assert failure['actual_owned_consumers_exited'] is True
    assert failure['source']==http.source
    if mode=='truncate':
        # A successfully written high-offset piece extends the whole file while
        # the low-offset response keeps failing. Size cannot establish progress.
        assert (output/'archive.tar').stat().st_size==len(http.raw)
        assert 0<failure['source_bytes_received']<len(http.raw)


def test_wrong_independently_frozen_hash_is_refused(http,tmp_path):
    output=tmp_path/'wrong-hash'
    with pytest.raises(Exception):acquire(http,output,expected_sha256='0'*64)
    assert not (output/'receipt.json').exists()
    assert json.loads((output/'failure.json').read_text())['complete_bytes'] is False


def test_existing_active_or_complete_output_is_never_overwritten(http,tmp_path):
    output=tmp_path/'already';output.mkdir();(output/'sentinel').write_bytes(b'keep')
    with pytest.raises(FileExistsError):acquire(http,output)
    assert (output/'sentinel').read_bytes()==b'keep'


@pytest.mark.parametrize('kind',['disk','memory'])
def test_pressure_refuses_before_any_download_file_is_created(http,tmp_path,monkeypatch,kind):
    if kind=='memory':monkeypatch.setattr(api(),'available_memory_bytes',lambda:0)
    else:monkeypatch.setattr(api().shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    output=tmp_path/'no-allocation'
    with pytest.raises(Exception):acquire(http,output)
    assert not output.exists() and not http.state['requests']


@pytest.mark.parametrize('workers',[0,True,9])
def test_invalid_concurrency_refuses_before_allocating(http,tmp_path,workers):
    output=tmp_path/'never'
    with pytest.raises(ValueError):api()._acquire_source(http.source,output,asset='engineering',workers=workers)
    assert not output.exists() and not http.state['requests']


def test_failure_resume_requires_frozen_partial_and_control_bytes(http,tmp_path):
    output=tmp_path/'resume';http.state['mode']='truncate'
    with pytest.raises(Exception):acquire(http,output)
    failure=output/'failure.json';frozen=sha(failure.read_bytes());packet=json.loads(failure.read_text())
    assert packet['partial_sha256'] is not None and packet['control_sha256'] is not None
    http.state['mode']='ok';receipt=acquire(http,output,resume=True,expected_failure_sha256=frozen)
    assert (output/'archive.tar').read_bytes()==http.raw and receipt['complete_bytes'] is True
    assert receipt['resumed_previous_failure_sha256']==frozen
    assert (output/'attempts').is_dir()


@pytest.mark.parametrize('damage',['receipt','partial','control'])
def test_changed_failed_input_is_refused_before_resume_requests(http,tmp_path,damage):
    output=tmp_path/'resume-change';http.state['mode']='truncate'
    with pytest.raises(Exception):acquire(http,output)
    failure=output/'failure.json';frozen=sha(failure.read_bytes())
    path={'receipt':failure,'partial':output/'archive.tar','control':output/'archive.tar.aria2'}[damage]
    with path.open('ab') as stream:stream.write(b'changed')
    before=len(http.state['requests']);http.state['mode']='ok'
    with pytest.raises(ValueError):acquire(http,output,resume=True,expected_failure_sha256=frozen)
    assert len(http.state['requests'])==before and not (output/'receipt.json').exists()
