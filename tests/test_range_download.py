"""Tiny local HTTP-boundary fixtures; no network or release payload downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from egoemg_repro.download import Asset, DownloadError, MIN_FREE_BYTES, download_one


QUOTA = (b'<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title></head>'
         b'<body>Too many users have viewed or downloaded this file recently.</body></html>')


def api():
    assert importlib.util.find_spec('egoemg_repro.range_download') is not None, 'strict range recovery is missing'
    from egoemg_repro import range_download
    return range_download


@pytest.fixture(autouse=True)
def ample_transport_fixture_capacity(monkeypatch):
    """Keep real reserve arithmetic; make synthetic transfers host-independent."""
    from egoemg_repro import download
    module = api()
    checked_capacity = download.ensure_capacity

    def ample_disk(_path):
        return SimpleNamespace(total=100*1024**3, used=10*1024**3, free=90*1024**3)

    def fixture_capacity(root, assets, reserve_bytes=MIN_FREE_BYTES, disk_usage=None):
        # download_one calls the frozen function without a disk_usage keyword;
        # that function's original default captured the real host callable.
        # Forward explicitly instead, retaining the full production calculation.
        return checked_capacity(root, assets, reserve_bytes=reserve_bytes,
                                disk_usage=disk_usage or module.shutil.disk_usage)

    monkeypatch.setattr(download, 'ensure_capacity', fixture_capacity)
    # The adapter supplies this current callable explicitly for every chunk.
    # Intentional low-space tests override it below, for both boundaries.
    monkeypatch.setattr(module.shutil, 'disk_usage', ample_disk)


class Response:
    def __init__(self, body, start, end, total, *, headers=None, status=206, interrupt=False):
        self.body, self.status_code, self.interrupt = body, status, interrupt
        self.headers = {'Content-Type':'application/octet-stream', 'Content-Length':str(end-start+1),
                        'Content-Range':f'bytes {start}-{end}/{total}'}
        self.headers.update(headers or {})
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, chunk_size):
        for index in range(0,len(self.body),2):
            yield self.body[index:index+2]
            if self.interrupt:
                raise OSError('simulated disconnect')


class PublicServer:
    def __init__(self, payload, *, alter=None):
        self.payload, self.alter = payload, alter
        self.requests, self.responses = [], []

    def session(self):
        server=self

        class Session:
            def __init__(self):
                self.trust_env=True
                self.cookies={'old':'must-clear'}
                self.auth=('must','clear')
                self.proxies={'https':'must-clear'}

            def __enter__(self):
                return self

            def __exit__(self,*_):
                pass

            def get(self,url,*,params,headers,stream,timeout,allow_redirects):
                assert self.trust_env is False
                assert not self.cookies and self.auth is None and not self.proxies
                assert url=='https://drive.usercontent.google.com/download'
                assert params=={'id':'public-id','export':'download','confirm':'t'}
                assert headers['User-Agent'].startswith('python-requests/')
                assert headers['Accept-Encoding']=='identity'
                assert stream and allow_redirects is False
                span=re.fullmatch(r'bytes=(\d+)-(\d+)',headers['Range'])
                assert span
                start,end=map(int,span.groups())
                server.requests.append((start,end))
                response=Response(server.payload[start:end+1],start,end,len(server.payload))
                if server.alter:
                    server.alter(response,start,end)
                server.responses.append(response)
                return response
        return Session()


def transfer(asset,server,*,chunk_bytes=4):
    return api().StrictRangeDownloader(asset,chunk_bytes=chunk_bytes,session_factory=server.session)


def test_exact_public_ranges_resume_binary_partial_and_bind_existing_receipt(tmp_path):
    payload=b'\x00binary-payload17'
    asset=Asset('tiny.data.mdb','public-id',len(payload))
    part=tmp_path/(asset.filename+'.part')
    part.write_bytes(payload[:3])
    server=PublicServer(payload)
    entry=download_one(asset,tmp_path,downloader=transfer(asset,server))
    assert (tmp_path/asset.filename).read_bytes()==payload
    assert entry['sha256']==hashlib.sha256(payload).hexdigest()
    assert server.requests[0]==(3,6)
    assert server.requests[-1][1]==len(payload)-1
    assert all(response.closed for response in server.responses)
    assert json.loads((tmp_path/'acquisition_ledger.json').read_text())['files'][asset.filename]==entry


def test_interrupted_range_keeps_only_complete_chunks_then_resumes(tmp_path):
    payload=b'\x00twelvebytes'
    asset=Asset('tiny.data.mdb','public-id',len(payload))
    def interrupt(response,start,end):
        if start==4:
            response.interrupt=True
    failing=PublicServer(payload,alter=interrupt)
    with pytest.raises(DownloadError,match='interrupted'):
        download_one(asset,tmp_path,downloader=transfer(asset,failing))
    part=tmp_path/(asset.filename+'.part')
    assert part.read_bytes()==payload[:4]
    assert not (tmp_path/'acquisition_ledger.json').exists()
    working=PublicServer(payload)
    download_one(asset,tmp_path,downloader=transfer(asset,working))
    assert working.requests[0][0]==4
    assert (tmp_path/asset.filename).read_bytes()==payload


@pytest.mark.parametrize('headers,status,body', [
    ({},200,None),
    ({'Content-Type':'text/html'},206,QUOTA),
    ({'Content-Range':'bytes 0-3/999'},206,None),
    ({'Content-Range':'bytes 1-4/16'},206,None),
    ({'Content-Range':'bytes 0-3/*'},206,None),
    ({'Content-Length':'3'},206,None),
    ({'Content-Encoding':'gzip'},206,None),
    ({},206,b'x'),
    ({},206,b'five!'),
    ({},403,None),
])
def test_bad_status_headers_or_body_never_append(tmp_path,headers,status,body):
    payload=b'\x00binarypayload!!'
    asset=Asset('tiny.data.mdb','public-id',len(payload))
    def corrupt(response,start,end):
        response.headers.update(headers)
        response.status_code=status
        if body is not None:response.body=body
    server=PublicServer(payload,alter=corrupt)
    with pytest.raises(DownloadError):
        download_one(asset,tmp_path,downloader=transfer(asset,server))
    part=tmp_path/(asset.filename+'.part')
    assert not part.exists() or part.stat().st_size==0
    assert not (tmp_path/asset.filename).exists()
    assert not (tmp_path/'acquisition_ledger.json').exists()
    assert server.requests == [(0,3)]  # It must fail at the protocol boundary.
    assert all(response.closed for response in server.responses)


def test_html_with_false_binary_headers_is_not_written(tmp_path):
    asset=Asset('tiny.data.mdb','public-id',len(QUOTA))
    server=PublicServer(QUOTA)
    with pytest.raises(DownloadError,match='HTML'):
        download_one(asset,tmp_path,downloader=transfer(asset,server,chunk_bytes=len(QUOTA)))
    assert not (tmp_path/(asset.filename+'.part')).exists()


def test_known_quota_page_is_quarantined_with_hash_receipt_before_fresh_bytes(tmp_path):
    payload=b'\x00binarypayload!!'
    asset=Asset('tiny.data.mdb','public-id',len(payload)+len(QUOTA))
    part=tmp_path/(asset.filename+'.part')
    part.write_bytes(QUOTA)
    result=api().quarantine_quota_html(part)
    assert result is not None
    assert not part.exists()
    preserved=Path(result['path'])
    assert preserved.read_bytes()==QUOTA
    receipt=json.loads(preserved.with_suffix('.json').read_text())
    assert receipt['sha256']==hashlib.sha256(QUOTA).hexdigest()
    assert receipt['bytes']==len(QUOTA)
    assert receipt['original_partial']==part.name
    assert receipt['diagnosis']=='google_drive_quota_html'


def test_unknown_html_is_retained_and_binary_partial_is_unchanged(tmp_path):
    part=tmp_path/'tiny.data.mdb.part'
    part.write_bytes(b'<html><title>Sign in</title></html>')
    with pytest.raises(DownloadError,match='unrecognized HTML'):
        api().quarantine_quota_html(part)
    assert part.read_bytes()==b'<html><title>Sign in</title></html>'
    part.write_bytes(b'\x00genuine binary prefix')
    assert api().quarantine_quota_html(part) is None
    assert part.read_bytes()==b'\x00genuine binary prefix'


def test_reserve_failure_precedes_any_request_or_append(tmp_path,monkeypatch):
    module=api()
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _p:SimpleNamespace(free=MIN_FREE_BYTES))
    payload=b'\x00binarypayload!!'
    asset=Asset('tiny.data.mdb','public-id',len(payload))
    part=tmp_path/(asset.filename+'.part')
    part.write_bytes(payload[:4])
    server=PublicServer(payload)
    with pytest.raises(DownloadError,match='reserve'):
        transfer(asset,server)(asset.drive_id,part)
    assert not server.requests
    assert part.read_bytes()==payload[:4]


def test_wrong_offset_response_does_not_damage_existing_valid_prefix(tmp_path):
    payload=b'\x00binarypayload!!'
    asset=Asset('tiny.data.mdb','public-id',len(payload))
    part=tmp_path/(asset.filename+'.part')
    part.write_bytes(payload[:4])
    def wrong_offset(response,start,end):
        response.headers['Content-Range']=f'bytes 0-3/{len(payload)}'
    server=PublicServer(payload,alter=wrong_offset)
    with pytest.raises(DownloadError,match='Content-Range'):
        download_one(asset,tmp_path,downloader=transfer(asset,server))
    assert part.read_bytes()==payload[:4]


def test_quarantine_never_counts_html_bytes_as_retained_payload_space(tmp_path,monkeypatch):
    module=api()
    expected=len(QUOTA)+100
    inventory={'files':[{'episode_id':'episode_0','filename':'episode_0.data.mdb',
                         'drive_id':'public-id','expected_bytes':expected}]}
    part=tmp_path/'episode_0.data.mdb.part'
    part.write_bytes(QUOTA)
    # Before quarantine, the frozen reservation credits the partial's length.
    # Preserving that HTML on disk must not keep that credit for a fresh file.
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _p:SimpleNamespace(free=MIN_FREE_BYTES+100))
    server=PublicServer(b'\x00'*expected)
    with pytest.raises(DownloadError,match='reserve'):
        module.recover_assets(inventory,tmp_path,session_factory=server.session)
    assert not server.requests
    assert not part.exists()
    preserved=list((tmp_path/'.quota-quarantine').glob('*.html'))
    assert len(preserved)==1 and preserved[0].read_bytes()==QUOTA


def test_no_implicit_retry_on_quota_or_redirect_and_no_auth_handling(tmp_path):
    payload=b'\x00binarypayload!!'
    asset=Asset('tiny.data.mdb','public-id',len(payload))
    def redirect(response,start,end):
        response.status_code=302
        response.headers['Location']='https://accounts.google.com/'
    server=PublicServer(payload,alter=redirect)
    with pytest.raises(DownloadError,match='206'):
        download_one(asset,tmp_path,downloader=transfer(asset,server))
    assert len(server.requests)==1
    assert not (tmp_path/'acquisition_ledger.json').exists()


def test_recover_inventory_validates_real_lmdb_and_preserves_quota_evidence(tmp_path):
    lmdb=pytest.importorskip('lmdb')
    source=tmp_path/'source.data.mdb'
    env=lmdb.open(str(source),subdir=False,map_size=2**20)
    with env.begin(write=True) as transaction:
        transaction.put(b'00000017_L',b'jpeg-fixture')
    env.close()
    payload=source.read_bytes()
    inventory={'files':[{'episode_id':'episode_0','filename':'episode_0.data.mdb','drive_id':'public-id','expected_bytes':len(payload)}]}
    root=tmp_path/'crops'
    root.mkdir()
    (root/'episode_0.data.mdb.part').write_bytes(QUOTA)
    server=PublicServer(payload)
    result=api().recover_assets(inventory,root,episodes=['episode_0'],chunk_bytes=4096,session_factory=server.session)
    assert result['validated_files']==1
    assert result['lmdb_entries']==1
    assert (root/'episode_0.data.mdb').read_bytes()==payload
    receipts=list((root/'.quota-quarantine').glob('*.json'))
    assert len(receipts)==1
    assert json.loads(receipts[0].read_text())['sha256']==hashlib.sha256(QUOTA).hexdigest()
    assert server.requests[0][0]==0
    # Existing verified files are rehashed/reopened, with no new network request.
    second=PublicServer(payload)
    assert api().recover_assets(inventory,root,session_factory=second.session)['lmdb_entries']==1
    assert not second.requests


def test_complete_wrong_lmdb_is_not_reported_as_success(tmp_path):
    lmdb=pytest.importorskip('lmdb')
    payload=b'\x00not-a-valid-lmdb-file'
    inventory={'files':[{'episode_id':'episode_0','filename':'episode_0.data.mdb',
                         'drive_id':'public-id','expected_bytes':len(payload)}]}
    with pytest.raises(lmdb.Error):
        api().recover_assets(inventory,tmp_path,session_factory=PublicServer(payload).session)
    # The existing ledger's scope is the byte transfer, not LMDB validity.
    assert (tmp_path/'acquisition_ledger.json').exists()
    assert (tmp_path/'episode_0.data.mdb').read_bytes()==payload


def test_cli_explains_explicit_direct_policy_without_site_packages():
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-S','-m','egoemg_repro.range_download','--help'],
                          cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--direct-public' in result.stdout
    assert '--inventory' in result.stdout and '--episodes' in result.stdout


def test_cli_requires_explicit_direct_public_choice_before_any_transfer(tmp_path):
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-m','egoemg_repro.range_download','--inventory','absent.json',
                           '--root',str(tmp_path/'new-root')],capture_output=True,text=True)
    assert result.returncode==2
    assert '--direct-public' in result.stderr
    assert not (tmp_path/'new-root').exists()


def test_transport_suite_is_independent_of_outer_eight_gib_host_capacity():
    """Exercise all other cases under low host/default-bound disk boundaries."""
    import subprocess
    import sys
    assert MIN_FREE_BYTES == 15*1024**3
    code = '''
import shutil
from types import SimpleNamespace
import pytest
from egoemg_repro import download
low_disk = lambda _path: SimpleNamespace(total=16*1024**3, used=8*1024**3, free=8*1024**3)
shutil.disk_usage = low_disk
download.ensure_capacity.__defaults__ = (download.MIN_FREE_BYTES, low_disk)
raise SystemExit(pytest.main([
    "tests/test_range_download.py", "-q", "-k",
    "not transport_suite_is_independent_of_outer_eight_gib_host_capacity"
]))
'''
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'passed' in result.stdout  # Optional LMDB cases may explicitly skip.
