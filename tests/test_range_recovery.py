"""Bounded retry around the real strict reader and original final-file writer."""
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path

import pytest
import requests

from egoemg_repro.download import Asset, DownloadError, download_one
from test_range_download import PublicServer, QUOTA, ample_transport_fixture_capacity


EOF = 'EOF occurred in violation of protocol (_ssl.c:1010)'


def adapter(asset, server, *, sleep=None, on_retry=None, chunk_bytes=4):
    assert importlib.util.find_spec('egoemg_repro.range_recovery') is not None, 'bounded recovery is missing'
    from egoemg_repro.range_recovery import BoundedRangeDownloader
    return BoundedRangeDownloader(asset, chunk_bytes=chunk_bytes, session_factory=server.session,
                                  sleep=sleep or (lambda _: None), on_retry=on_retry)


def test_eof_resumes_only_after_verified_complete_chunks(tmp_path):
    payload=b'\x00abcdefghijklmno';asset=Asset('tiny.dat','public-id',len(payload))
    calls=Counter();events=[];delays=[]
    def fault(response,start,end):
        calls[start]+=1
        if start==4 and calls[start]==1:raise requests.exceptions.SSLError(EOF)
    server=PublicServer(payload,alter=fault)
    receive=adapter(asset,server,sleep=delays.append,on_retry=events.append)
    receipt=download_one(asset,tmp_path,downloader=receive)
    assert (tmp_path/asset.filename).read_bytes()==payload
    assert receipt['sha256']==hashlib.sha256(payload).hexdigest()
    assert [s for s,e in server.requests]==[0,4,4,8,12]
    assert delays==[1] and events[0]['offset']==4 and events[0]['failure_number']==1
    assert events[0]['will_retry'] is True
    assert receive.statistics['transient_failures']==receive.statistics['scheduled_retries']==1
    assert not receive.statistics['exhausted']


def test_stream_disconnect_never_appends_unverified_partial_response(tmp_path):
    payload=b'\x00abcdefghijk';asset=Asset('tiny.dat','public-id',len(payload));calls=Counter()
    observed=[]
    def fault(response,start,end):
        calls[start]+=1
        if start==4 and calls[start]==1:
            def broken(chunk_size):
                yield response.body[:2]
                raise requests.exceptions.ConnectionError('connection reset mid-body')
            response.iter_content=broken
    server=PublicServer(payload,alter=fault)
    receive=adapter(asset,server,on_retry=lambda event:observed.append((tmp_path/'tiny.dat.part').read_bytes()))
    download_one(asset,tmp_path,downloader=receive)
    assert observed==[payload[:4]] and (tmp_path/'tiny.dat').read_bytes()==payload
    assert [s for s,e in server.requests]==[0,4,4,8]
    assert all(r.closed for r in server.responses)


def test_retry_budget_resets_at_later_verified_offsets(tmp_path):
    payload=b'\x00abcdefghijk';asset=Asset('tiny.dat','public-id',len(payload));calls=Counter();delays=[]
    def fault(response,start,end):
        calls[start]+=1
        if calls[start]<3:raise requests.exceptions.ReadTimeout('temporary read timeout')
    server=PublicServer(payload,alter=fault);receive=adapter(asset,server,sleep=delays.append)
    download_one(asset,tmp_path,downloader=receive)
    assert [s for s,e in server.requests]==[0,0,0,4,4,4,8,8,8]
    assert delays==[1,2]*3
    assert receive.statistics['transient_failures']==receive.statistics['scheduled_retries']==6
    assert (tmp_path/'tiny.dat').read_bytes()==payload


def test_three_stalled_attempts_stop_and_keep_original_verified_prefix(tmp_path):
    payload=b'\x00abcdefghijk';asset=Asset('tiny.dat','public-id',len(payload));delays=[];events=[]
    def fault(response,start,end):
        if start==4:raise requests.exceptions.SSLError(EOF)
    server=PublicServer(payload,alter=fault);receive=adapter(asset,server,sleep=delays.append,on_retry=events.append)
    with pytest.raises(DownloadError,match='interrupted'):download_one(asset,tmp_path,downloader=receive)
    assert [s for s,e in server.requests]==[0,4,4,4]
    assert delays==[1,2] and [e['will_retry'] for e in events]==[True,True,False]
    assert (tmp_path/'tiny.dat.part').read_bytes()==payload[:4]
    assert not (tmp_path/'tiny.dat').exists() and not (tmp_path/'acquisition_ledger.json').exists()
    assert receive.statistics['exhausted'] and receive.statistics['transient_failures']==3


@pytest.mark.parametrize('error',[
    requests.exceptions.SSLError('CERTIFICATE_VERIFY_FAILED'),
    requests.exceptions.SSLError('WRONG_VERSION_NUMBER'),
    requests.exceptions.ProxyError('bad proxy'),
    requests.exceptions.InvalidURL('bad URL'),
    OSError('local write or unrelated OS failure'),
])
def test_permanent_errors_are_not_retried(tmp_path,error):
    asset=Asset('tiny.dat','public-id',8);server=PublicServer(b'\x00abcdefg',alter=lambda *args:(_ for _ in ()).throw(error))
    delays=[];receive=adapter(asset,server,sleep=delays.append)
    with pytest.raises(DownloadError):download_one(asset,tmp_path,downloader=receive)
    assert len(server.requests)==1 and delays==[] and receive.statistics['scheduled_retries']==0


@pytest.mark.parametrize('headers,status,body',[
    ({},200,None),({},403,None),({},429,None),
    ({'Content-Type':'text/html'},206,QUOTA),
    ({'Content-Range':'bytes 0-3/999'},206,None),
    ({'Content-Range':'bytes 1-4/16'},206,None),
    ({'Content-Length':'3'},206,None),
    ({'Content-Encoding':'gzip'},206,None),
    ({},206,b'x'),({},206,b'five!'),
])
def test_native_protocol_errors_never_trigger_retry(tmp_path,headers,status,body):
    payload=b'\x00abcdefghijklmno';asset=Asset('tiny.dat','public-id',len(payload));delays=[]
    def fault(response,start,end):
        response.status_code=status;response.headers.update(headers)
        if body is not None:response.body=body
    server=PublicServer(payload,alter=fault);receive=adapter(asset,server,sleep=delays.append)
    with pytest.raises(DownloadError):download_one(asset,tmp_path,downloader=receive)
    assert server.requests==[(0,3)] and not delays
    assert not (tmp_path/'tiny.dat').exists() and not (tmp_path/'acquisition_ledger.json').exists()


def test_arbitrary_binary_prefix_and_same_final_hash_gate_are_preserved(tmp_path):
    payload=b'\x00abcdefghijk';asset=Asset('tiny.dat','public-id',len(payload),sha256=hashlib.sha256(payload).hexdigest())
    part=tmp_path/'tiny.dat.part';part.write_bytes(payload[:3]);server=PublicServer(payload)
    download_one(asset,tmp_path,downloader=adapter(asset,server))
    assert server.requests[0]==(3,6) and (tmp_path/'tiny.dat').read_bytes()==payload
    wrong=Asset('wrong.dat','public-id',len(payload),sha256='0'*64)
    with pytest.raises(DownloadError,match='SHA256'):download_one(wrong,tmp_path,downloader=adapter(wrong,PublicServer(payload)))
    assert (tmp_path/'wrong.dat.part').read_bytes()==payload and not (tmp_path/'wrong.dat').exists()


def test_known_quota_document_is_preserved_before_offset_snapshot(tmp_path):
    payload=b'\x00'+b'x'*512;asset=Asset('tiny.dat','public-id',len(payload));part=tmp_path/'tiny.dat.part';part.write_bytes(QUOTA)
    server=PublicServer(payload);download_one(asset,tmp_path,downloader=adapter(asset,server,chunk_bytes=128))
    assert server.requests[0][0]==0 and (tmp_path/'tiny.dat').read_bytes()==payload
    preserved=list((tmp_path/'.quota-quarantine').glob('*.html'))
    assert len(preserved)==1 and preserved[0].read_bytes()==QUOTA


def test_wrong_id_refused_before_quota_or_network_mutation(tmp_path):
    asset=Asset('tiny.dat','public-id',512);part=tmp_path/'tiny.dat.part';part.write_bytes(QUOTA)
    server=PublicServer(b'x'*512);receive=adapter(asset,server)
    with pytest.raises(DownloadError):receive('another-id',part)
    assert part.read_bytes()==QUOTA and not server.requests and not (tmp_path/'.quota-quarantine').exists()


@pytest.mark.parametrize('mutation',['append','replace'])
def test_partial_changed_during_backoff_is_not_consumed(tmp_path,mutation):
    payload=b'\x00abcdefghijk';asset=Asset('tiny.dat','public-id',len(payload));part=tmp_path/'tiny.dat.part'
    def fault(response,start,end):
        if start==4:raise requests.exceptions.ConnectionError('temporary failure')
    def change(_):
        if mutation=='append':
            with part.open('ab') as f:f.write(b'x')
        else:
            other=tmp_path/'replacement';other.write_bytes(payload[:4]);other.replace(part)
    server=PublicServer(payload,alter=fault);receive=adapter(asset,server,sleep=change)
    with pytest.raises(DownloadError,match='partial'):download_one(asset,tmp_path,downloader=receive)
    assert [s for s,e in server.requests]==[0,4] and not (tmp_path/'tiny.dat').exists()


def test_shrunk_partial_after_network_error_fails_without_retry(tmp_path):
    payload=b'\x00abcdefghijk';asset=Asset('tiny.dat','public-id',len(payload));part=tmp_path/'tiny.dat.part';part.write_bytes(payload[:3]);delays=[]
    def fault(response,start,end):
        part.write_bytes(payload[:2]);raise requests.exceptions.ConnectionError('temporary failure')
    server=PublicServer(payload,alter=fault);receive=adapter(asset,server,sleep=delays.append)
    with pytest.raises(DownloadError,match='partial'):download_one(asset,tmp_path,downloader=receive)
    assert server.requests==[(3,6)] and not delays


def test_callback_cancellation_propagates_before_backoff(tmp_path):
    payload=b'\x00abcdefghijk';asset=Asset('tiny.dat','public-id',len(payload));delays=[]
    def fault(response,start,end):raise requests.exceptions.ConnectionError('temporary failure')
    def cancel(_):raise KeyboardInterrupt('cancel owned intake')
    server=PublicServer(payload,alter=fault);receive=adapter(asset,server,sleep=delays.append,on_retry=cancel)
    with pytest.raises(KeyboardInterrupt):download_one(asset,tmp_path,downloader=receive)
    assert len(server.requests)==1 and not delays and not (tmp_path/'tiny.dat').exists()


def test_capacity_failure_is_not_retried_or_hidden(tmp_path,monkeypatch):
    from egoemg_repro import range_download
    from types import SimpleNamespace
    monkeypatch.setattr(range_download.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    asset=Asset('tiny.dat','public-id',8);server=PublicServer(b'\x00abcdefg');delays=[]
    receive=adapter(asset,server,sleep=delays.append)
    with pytest.raises(DownloadError):download_one(asset,tmp_path,downloader=receive)
    assert not server.requests and not delays
