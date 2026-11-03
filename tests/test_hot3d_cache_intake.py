"""Transport-only fixtures; these do not qualify an actual HOT3D download."""
import hashlib
import importlib
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    try: return importlib.import_module('egoemg_repro.hot3d_cache_intake')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.hot3d_cache_intake': pytest.fail('read-only HOT3D cache integration missing')
        raise


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
def generation(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
def actor(pid):
    fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=fields[19])
def gone(handle):
    p=Path('/proc',str(handle['pid']),'stat')
    return not p.exists() or p.read_text().rsplit(')',1)[1].split()[19] != str(handle['start_ticks'])


@pytest.fixture(autouse=True)
def transport_fixture_capacity(monkeypatch):
    """Tiny transport fixtures do not allocate a real225GiB source corpus.

    Hosted CI exercises admission against an explicit capacity fixture. The
    source-bound server run opts into the actual disk query; its acquisition
    preflight and real receiver always use the actual filesystem as well.
    """
    if os.environ.get('EGOEMG_REQUIRE_HOT3D_TRANSPORT_CAPACITY')=='1':return
    import shutil
    from types import SimpleNamespace
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=225*1024**3+1024**2))


@pytest.fixture
def closed_source(tmp_path):
    root=tmp_path/'old';root.mkdir()
    payloads=[b'first',b'second',b'new']
    assets=[dict(path='train_aria/clip-%06d.tar'%i,expected_bytes=len(b),
        lfs_sha256=hashlib.sha256(b).hexdigest(),url='https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/30fe9674782f32e1e5edba98476b6ff4300132c5/train_aria/clip-%06d.tar'%i)
        for i,b in enumerate(payloads,1)]
    owners=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(.05)'])for _ in range(2)]
    handles=[actor(p.pid)for p in owners]
    for p in owners: assert p.wait()==0
    for a,b in zip(assets[:2],payloads):
        path=root/'archives'/a['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    # A partial/control pair is preserved and never admitted by file size alone.
    partial=root/'archives'/assets[2]['path'];partial.write_bytes(b'new')
    partial.with_name(partial.name+'.aria2').write_bytes(b'control')
    dump(root/'manifest.json',dict(files=assets))
    dump(root/'config.json',dict(source_revision='30fe9674782f32e1e5edba98476b6ff4300132c5'))
    (root/'worker.py').write_text('# original closed fixture source\n')
    body=dict(source_commit='engineering-fixture-not-actual-HOT3D',
        files_sha256={n:sha(root/n)for n in('manifest.json','config.json','worker.py')},full_paper_complete=False)
    bundle=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(root/'private/stage-receipt.json',dict(body,bundle_sha256=bundle))
    state=dict(status='failed',actual_exit_code=1,wrapper=handles[0],child=handles[1])
    dump(root/'private/execution-state.json',state)
    row=dict(path=assets[0]['path'],bytes=5,sha256=assets[0]['lfs_sha256'],
        matches_publisher_lfs_sha256=True,reused_closed_qualification_archive=True,native_worker=None,native_exit_code=None)
    (root/'receipts.jsonl').write_text(json.dumps(row)+'\n')
    dump(root/'failed.json',dict(actor=handles[1],completed_archive_count=1,completed_archive_bytes=5,
        owned_native_worker_exited=True,automatic_retry=False))
    return root,dict(root=str(root),source_commit=body['source_commit'],bundle_sha256=bundle,actual_state=state),assets


def prepare_cache(closed_source, tmp_path):
    root,launch,assets=closed_source;m=api()
    descriptor=m.cache_from_closed_source(root,launch,assets)
    path=tmp_path/'cache.json';dump(path,descriptor)
    return m.load_readonly_cache(path,expected_sha256=sha(path),assets=assets),descriptor,path


def test_closed_cache_candidates_are_not_copied_and_partial_is_preserved(closed_source,tmp_path):
    root,launch,assets=closed_source;before={str(p):generation(p)for p in root.rglob('*')if p.is_file()}
    cache,document,_=prepare_cache(closed_source,tmp_path)
    assert document['source_journal_archives']==1
    assert set(document['entries'])=={'train_aria/clip-000001.tar','train_aria/clip-000002.tar'}
    assert cache.path_for(assets[0])==root/'archives/train_aria/clip-000001.tar'
    assert cache.path_for(assets[2]) is None
    assert cache.verified_archives==2 and cache.verified_bytes==11
    assert {str(p):generation(p)for p in root.rglob('*')if p.is_file()}==before
    assert not (tmp_path/'archives').exists()


@pytest.mark.parametrize('damage',['live_owner','bundle','source_bytes','journal_tail','journal_digest'])
def test_bad_or_live_closed_source_cannot_produce_cache(damage,closed_source):
    root,launch,assets=closed_source
    if damage=='live_owner':
        state=json.loads((root/'private/execution-state.json').read_text());state['child']=actor(os.getpid())
        launch['actual_state']=state;dump(root/'private/execution-state.json',state)
    elif damage=='bundle': launch['bundle_sha256']='0'*64
    elif damage=='source_bytes': (root/'worker.py').write_text('# changed\n')
    elif damage=='journal_tail':
        p=root/'receipts.jsonl';p.write_bytes(p.read_bytes().rstrip(b'\n'))
    else:
        p=root/'receipts.jsonl';row=json.loads(p.read_text());row['sha256']='0'*64;p.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError): api().cache_from_closed_source(root,launch,assets)


@pytest.mark.parametrize('damage',['cache_digest','payload_bytes','payload_generation','source_metadata','symlink_ancestor'])
def test_cache_rejects_wrong_current_bytes_and_generations(damage,closed_source,tmp_path):
    root,launch,assets=closed_source;m=api()
    payload=root/'archives'/assets[0]['path']
    # Wrong bytes BEFORE the snapshot give a stable matching generation. Only
    # actually reading the complete publisher SHA can reject this case.
    if damage=='payload_bytes':payload.write_bytes(b'WRONG')
    document=m.cache_from_closed_source(root,launch,assets)
    path=tmp_path/'cache.json';dump(path,document);digest=sha(path)
    if damage=='cache_digest': digest='0'*64
    elif damage=='payload_bytes': pass
    elif damage=='payload_generation': payload.touch()
    elif damage=='source_metadata': (root/'failed.json').write_text('{}\n')
    else:
        original=root/'archives/train_aria';moved=root/'old-physical';original.rename(moved);original.symlink_to(moved,target_is_directory=True)
    with pytest.raises(ValueError):m.load_readonly_cache(path,expected_sha256=digest,assets=assets)


@pytest.mark.parametrize('target',['payload','closed_metadata','cache_map'])
def test_reuse_guard_detects_changes_after_whole_SHA_admission(target,closed_source,tmp_path):
    cache,_,path=prepare_cache(closed_source,tmp_path);root,_,assets=closed_source
    if target=='payload': (root/'archives'/assets[1]['path']).write_bytes(b'CHANGED')
    elif target=='closed_metadata':(root/'failed.json').write_text('{}\n')
    else:path.write_text('{}\n')
    with pytest.raises(ValueError):cache.check()


def test_next_generation_flattens_external_paths_and_keeps_original_ancestry(closed_source,tmp_path):
    old,_,assets=closed_source;cache,document,_=prepare_cache(closed_source,tmp_path);m=api()
    new=tmp_path/'next';new.mkdir()
    dump(new/'private/cache.json',document)
    dump(new/'config.json',dict(source_revision='30fe9674782f32e1e5edba98476b6ff4300132c5',cache_sha256=sha(new/'private/cache.json')))
    dump(new/'manifest.json',dict(files=assets));(new/'worker.py').write_text('# next closed fixture\n')
    source_launch=closed_source[1];state=source_launch['actual_state'];dump(new/'private/execution-state.json',state)
    rows=[dict(path=a['path'],bytes=a['expected_bytes'],sha256=a['lfs_sha256'],matches_publisher_lfs_sha256=True,
        reused_readonly_cache=True,payload_path=str(old/'archives'/a['path']),native_worker=None,native_exit_code=None)for a in assets[:2]]
    (new/'receipts.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
    dump(new/'failed.json',dict(actor=state['child'],completed_archive_count=2,completed_archive_bytes=11,owned_native_worker_exited=True,automatic_retry=False))
    body=dict(source_commit='next-engineering-fixture',files_sha256={n:sha(new/n)for n in('config.json','manifest.json','worker.py','private/cache.json')},full_paper_complete=False)
    bundle=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest();dump(new/'private/stage-receipt.json',dict(body,bundle_sha256=bundle))
    launch=dict(root=str(new),source_commit=body['source_commit'],bundle_sha256=bundle,actual_state=state)
    flattened=m.cache_from_closed_source(new,launch,assets);p=tmp_path/'flattened.json';dump(p,flattened)
    next_cache=m.load_readonly_cache(p,expected_sha256=sha(p),assets=assets)
    assert next_cache.path_for(assets[0])==old/'archives'/assets[0]['path']
    assert not (new/'archives').exists()
    (old/'failed.json').write_text('{}\n')
    with pytest.raises(ValueError):next_cache.check()


def native_fixture(tmp_path, *, fail=False, wait=False):
    p=tmp_path/'native-fixture';p.write_text('#!'+sys.executable+'\n'+'''import json,os,pathlib,sys,time
a=sys.argv[1:]
for flag in ['--no-conf=true','--max-tries=1','--max-connection-per-server=1','--split=1','--max-download-limit=32M','--continue=false','--allow-overwrite=false','--auto-file-renaming=false','--check-integrity=true']:
 assert flag in a,flag
assert not any('certificate=false' in x for x in a)
out=pathlib.Path(next(x[6:]for x in a if x.startswith('--dir=')))/next(x[6:]for x in a if x.startswith('--out='))
with (out.parent/'invocations').open('a') as f:f.write('one native process\\n')
out.write_bytes(b'PARTIAL' if '''+str(fail or wait)+''' else b'new')
(out.parent/'native-pid').write_text(str(os.getpid()))
'''+('time.sleep(30)\n'if wait else'')+('raise SystemExit(1)\n'if fail else''))
    p.chmod(0o700);return p


@pytest.mark.parametrize('fail',[False,True])
def test_transport_reuses_whole_cache_and_runs_each_new_native_child_once(fail,closed_source,tmp_path):
    m=api();cache,_,_=prepare_cache(closed_source,tmp_path);old,_,assets=closed_source
    out=tmp_path/'intake';out.mkdir();binary=native_fixture(tmp_path,fail=fail)
    before={str(p):generation(p)for p in old.rglob('*')if p.is_file()}
    if fail:
        with pytest.raises(ValueError):m.receive_assets(out,assets,cache,binary=binary,guard=cache.check)
        assert (out/'archives'/assets[2]['path']).read_bytes()==b'PARTIAL'
    else:
        result=m.receive_assets(out,assets,cache,binary=binary,guard=cache.check)
        assert result['archive_count']==3 and result['archive_bytes']==14
        assert result['reused_cache_archives']==2 and result['new_download_archives']==1
        assert len(result['native_workers'])==1 and all(gone(h)for h in result['native_workers'])
    rows=[json.loads(s)for s in(out/'receipts.jsonl').read_text().splitlines()]
    assert len(rows)==(2 if fail else 3)
    assert not (out/'archives'/assets[0]['path']).exists()
    assert (out/'archives/train_aria/invocations').read_text().splitlines()==['one native process']
    assert not (out/'completed.json').exists()
    assert {str(p):generation(p)for p in old.rglob('*')if p.is_file()}==before


def test_capacity_failure_preserves_cache_and_creates_no_native_payload(closed_source,tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    cache,_,_=prepare_cache(closed_source,tmp_path);out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=225*1024**3-1))
    with pytest.raises(RuntimeError):api().receive_assets(out,closed_source[2],cache,binary='never-run',guard=cache.check)
    assert not (out/'archives').exists()


def test_guard_failure_reaps_only_owned_native_child(closed_source,tmp_path):
    m=api();cache,_,_=prepare_cache(closed_source,tmp_path);out=tmp_path/'out';out.mkdir()
    binary=native_fixture(tmp_path,wait=True);sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    def guard():
        cache.check()
        if (out/'archives/train_aria/native-pid').exists():raise RuntimeError('injected source guard change')
    try:
        with pytest.raises(RuntimeError):m.receive_assets(out,closed_source[2],cache,binary=binary,guard=guard)
        pid=int((out/'archives/train_aria/native-pid').read_text())
        assert not Path('/proc',str(pid)).exists()
        assert sentinel.poll() is None
        assert (out/'archives/train_aria/clip-000003.tar').read_bytes()==b'PARTIAL'
    finally:sentinel.terminate();sentinel.wait()


def test_public_complete_manifest_is_fixed_and_no_source_numeric_completion(tmp_path):
    m=api();inventory=json.loads((ROOT/'assets_hot3d_train_manifest.json').read_text())
    assert len(m.checked_assets(inventory))==2804
    assert m.MANIFEST_SHA256=='3495c287c68dfc36abcabd91bdd03fc461c54cac00b8f9bd2cb5ca59aadb39ee'
    assert m.FULL_BYTES==294201743360
    # Small transport fixtures have no full-source completion permit.
    with pytest.raises(ValueError):m._require_full_totals(dict(archive_count=3,archive_bytes=14))
