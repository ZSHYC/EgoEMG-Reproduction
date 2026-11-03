"""Transport regressions from the single review, never actual HOT3D admission."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_hot3d_cache_intake import api,actor,closed_source,dump,generation,prepare_cache,sha,native_fixture,gone,transport_fixture_capacity


def successor(closed_source,tmp_path):
    old,launch,assets=closed_source
    _,document,_=prepare_cache(closed_source,tmp_path)
    root=tmp_path/'successor';root.mkdir()
    dump(root/'private/cache.json',document)
    dump(root/'config.json',dict(source_revision='30fe9674782f32e1e5edba98476b6ff4300132c5',cache_sha256=sha(root/'private/cache.json')))
    dump(root/'manifest.json',dict(files=assets))
    (root/'worker.py').write_text('# closed transport fixture\n')
    state=launch['actual_state'];dump(root/'private/execution-state.json',state)
    a=assets[0];payload=old/'archives'/a['path']
    row=dict(path=a['path'],bytes=a['expected_bytes'],sha256=a['lfs_sha256'],matches_publisher_lfs_sha256=True,
        reused_readonly_cache=True,payload_path=str(payload),payload_generation=generation(payload),native_worker=None,native_exit_code=None)
    (root/'receipts.jsonl').write_text(json.dumps(row)+'\n')
    dump(root/'failed.json',dict(actor=state['child'],completed_archive_count=1,completed_archive_bytes=5,owned_native_worker_exited=True))
    body=dict(source_commit='closed-cache-successor-engineering-only',files_sha256={n:sha(root/n)for n in('config.json','manifest.json','worker.py','private/cache.json')},full_paper_complete=False)
    bundle=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(root/'private/stage-receipt.json',dict(body,bundle_sha256=bundle))
    return root,dict(root=str(root),source_commit=body['source_commit'],bundle_sha256=bundle,actual_state=state),assets


@pytest.mark.parametrize('damage',['owned_receipt','external_receipt','external_future','receipt_only'])
def test_equal_bytes_cannot_replace_the_recorded_cache_or_receipt_generation(closed_source,tmp_path,damage):
    old,launch,assets=closed_source
    if damage=='owned_receipt':
        root=old;row=json.loads((root/'receipts.jsonl').read_text())
        row['payload_generation']=generation(old/'archives'/assets[0]['path'])
        (root/'receipts.jsonl').write_text(json.dumps(row)+'\n')
    else:root,launch,assets=successor(closed_source,tmp_path)
    if damage=='receipt_only':
        row=json.loads((root/'receipts.jsonl').read_text());row['payload_generation'][3]+=1
        (root/'receipts.jsonl').write_text(json.dumps(row)+'\n')
    else:
        payload=old/'archives'/assets[1 if damage=='external_future'else 0]['path']
        old_sha=sha(payload);s=payload.stat();os.utime(payload,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000))
        assert sha(payload)==old_sha  # Only generation provenance, not SHA, changed.
    with pytest.raises(ValueError):api().cache_from_closed_source(root,launch,assets)


def test_unchanged_inherited_receipt_and_future_seed_preserve_original_generations(closed_source,tmp_path):
    old,_,_=closed_source;root,launch,assets=successor(closed_source,tmp_path)
    value=api().cache_from_closed_source(root,launch,assets)
    assert value['source_journal_archives']==1 and value['candidate_archives']==2
    for asset in assets[:2]:
        path=old/'archives'/asset['path'];record=value['entries'][asset['path']]
        assert record['physical_path']==str(path)and record['generation']==generation(path)
    assert not(root/'archives').exists()


def control(path,kind):
    p=path.with_name(path.name+'.aria2')
    if kind=='file':p.write_bytes(b'native-control-preserve')
    else:p.symlink_to(path.parent/'absent-control-target')
    return p


@pytest.mark.parametrize('kind',['file','dangling_symlink'])
def test_cache_continuously_refuses_new_control_files_without_changing_payloads(closed_source,tmp_path,kind):
    cache,_,_=prepare_cache(closed_source,tmp_path);root,_,assets=closed_source
    path=root/'archives'/assets[0]['path'];before=generation(path);p=control(path,kind)
    with pytest.raises(ValueError):cache.check()
    with pytest.raises(ValueError):cache.path_for(assets[0])
    assert generation(path)==before and(p.exists()or p.is_symlink())


def test_cache_initial_admission_refuses_dangling_native_control(closed_source,tmp_path):
    root,launch,assets=closed_source;doc=api().cache_from_closed_source(root,launch,assets)
    path=tmp_path/'candidate.json';dump(path,doc);p=control(root/'archives'/assets[0]['path'],'dangling_symlink')
    with pytest.raises(ValueError):api().load_readonly_cache(path,expected_sha256=sha(path),assets=assets)
    assert p.is_symlink()


def test_candidate_snapshot_refuses_dangling_native_control(closed_source):
    root,launch,assets=closed_source;p=control(root/'archives'/assets[0]['path'],'dangling_symlink')
    with pytest.raises(ValueError):api().cache_from_closed_source(root,launch,assets)
    assert p.is_symlink()


@pytest.mark.parametrize('kind',['file','dangling_symlink'])
def test_successful_native_exit_with_matching_payload_and_control_cannot_be_journaled(closed_source,tmp_path,kind):
    cache,_,_=prepare_cache(closed_source,tmp_path);_,_,assets=closed_source
    out=tmp_path/'intake';out.mkdir();binary=native_fixture(tmp_path)
    ending="out.with_name(out.name+'.aria2').write_bytes(b'control')\n"if kind=='file'else"out.with_name(out.name+'.aria2').symlink_to(out.parent/'absent-native-control-target')\n"
    binary.write_text(binary.read_text()+ending)
    with pytest.raises(ValueError):api().receive_assets(out,assets,cache,binary=binary,guard=cache.check)
    target=out/'archives'/assets[2]['path'];p=target.with_name(target.name+'.aria2')
    assert target.read_bytes()==b'new'and(p.exists()or p.is_symlink())
    assert len((out/'receipts.jsonl').read_text().splitlines())==2
    assert not(out/'completed.json').exists()


def test_newly_completed_owned_payloads_keep_control_absence_guarded(closed_source,tmp_path):
    cache,_,_=prepare_cache(closed_source,tmp_path);_,_,assets=closed_source
    out=tmp_path/'intake';out.mkdir();binary=native_fixture(tmp_path);made=[]
    def guard():
        cache.check();journal=out/'receipts.jsonl'
        if not made and journal.exists()and len(journal.read_text().splitlines())==3:
            made.append(control(out/'archives'/assets[2]['path'],'file'))
    with pytest.raises(ValueError):api().receive_assets(out,assets,cache,binary=binary,guard=guard)
    assert len(made)==1 and made[0].read_bytes()==b'native-control-preserve'


def test_dangling_control_is_rejected_before_starting_a_new_native_worker(closed_source,tmp_path):
    cache,_,_=prepare_cache(closed_source,tmp_path);_,_,assets=closed_source
    out=tmp_path/'intake';out.mkdir();binary=native_fixture(tmp_path);made=[]
    def guard():
        cache.check();journal=out/'receipts.jsonl'
        if not made and journal.exists()and len(journal.read_text().splitlines())==2:
            target=out/'archives'/assets[2]['path'];target.parent.mkdir(parents=True)
            made.append(control(target,'dangling_symlink'))
    with pytest.raises(ValueError):api().receive_assets(out,assets,cache,binary=binary,guard=guard)
    assert len(made)==1 and made[0].is_symlink()
    assert not(out/'archives/train_aria/native-pid').exists()


@pytest.mark.parametrize('cleanup',['unresolved','kill_fallback_closed'])
def test_failed_cleanup_retains_actual_native_identity_and_truthful_failure_receipt(closed_source,tmp_path,monkeypatch,cleanup):
    m=api();cache,_,_=prepare_cache(closed_source,tmp_path);_,_,assets=closed_source
    out=tmp_path/'intake';out.mkdir();binary=native_fixture(tmp_path,wait=True)
    real_popen=subprocess.Popen;real_terminate=real_popen.terminate;real_wait=real_popen.wait
    sentinel=real_popen([sys.executable,'-c','import time;time.sleep(30)']);children=[]
    def start(*a,**k):
        child=real_popen(*a,**k);children.append(child)
        def refuse_terminate():raise OSError('injected terminate failure')
        child.terminate=refuse_terminate
        if cleanup=='unresolved':
            def refuse_kill():raise OSError('injected kill failure')
            def refuse_wait(*a,**k):raise subprocess.TimeoutExpired('owned native fixture',k.get('timeout',0))
            child.kill=refuse_kill;child.wait=refuse_wait
        else:
            calls=[]
            def first_wait_times_out(*a,**k):
                calls.append(True)
                if len(calls)==1:raise subprocess.TimeoutExpired('owned native fixture',k.get('timeout',0))
                return real_wait(child,*a,**k)
            child.wait=first_wait_times_out
        return child
    def guard():
        cache.check()
        if(out/'archives/train_aria/native-pid').exists():raise RuntimeError('actual owned guard rejected the fixture')
    monkeypatch.setattr(m.subprocess,'Popen',start)
    try:
        with pytest.raises((RuntimeError,OSError))as caught:m.receive_assets(out,assets,cache,binary=binary,guard=guard)
        assert isinstance(caught.value,RuntimeError)  # Cleanup must not mask the initiating failure.
        detail=caught.value.hot3d_owned_native_cleanup
        assert len(children)==1 and detail['native_worker']['pid']==children[0].pid
        assert detail['cleanup_errors']and sentinel.poll()is None
        m._publish_failed_intake(out,actor(os.getpid()),caught.value)
        failed=json.loads((out/'failed.json').read_text());expected=cleanup=='kill_fallback_closed'
        assert failed['owned_native_worker_exited']is expected
        assert failed['current_worker']==detail['native_worker']and failed['native_cleanup']==detail
        assert detail['worker_reaped']is expected and detail['physically_closed']is expected
        assert len((out/'receipts.jsonl').read_text().splitlines())==2
        assert(out/'archives'/assets[2]['path']).read_bytes()==b'PARTIAL'
        if expected:m._closed(detail['native_worker'])
        else:
            assert children[0].poll()is None
            with pytest.raises(ValueError):m._closed(detail['native_worker'])
    finally:
        for child in children:
            if child.poll()is None:real_terminate(child)
            real_wait(child,timeout=5)
        real_terminate(sentinel);real_wait(sentinel,timeout=5)
    for child in children:assert not Path('/proc',str(child.pid)).exists()


def test_unidentified_unresolved_native_worker_cannot_be_reused_as_a_closed_source(closed_source):
    root,launch,assets=closed_source;value=json.loads((root/'failed.json').read_text())
    value.update(owned_native_worker_exited=False,current_worker=None,native_cleanup={'native_child_created':True})
    dump(root/'failed.json',value)
    with pytest.raises(ValueError):api().cache_from_closed_source(root,launch,assets)


def test_pre_child_failure_receipt_distinguishes_no_native_child_from_unresolved_cleanup(tmp_path):
    api()._publish_failed_intake(tmp_path,actor(os.getpid()),ValueError('before native creation'))
    failed=json.loads((tmp_path/'failed.json').read_text())
    assert failed['owned_native_worker_exited']is True and failed['current_worker']is None
    assert failed['native_cleanup']['native_child_created']is False
    assert failed['completed_archive_count']==0 and failed['completed_archive_bytes']==0
    assert failed['full_paper_complete']is False
