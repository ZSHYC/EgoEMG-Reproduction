"""Read-only reuse of closed HOT3D archive generations; no numeric permit."""
from pathlib import Path
import hashlib
import json
import os
import stat
import subprocess
import time
from types import MappingProxyType

from scripts.paper.hot3d_full_train_receiver import (
    REVISION, actor, atomic, checked_assets, exclusive, need, now, sha,
    verify_capacity, verify_payload,
)

MANIFEST_SHA256 = '3495c287c68dfc36abcabd91bdd03fc461c54cac00b8f9bd2cb5ca59aadb39ee'
NATIVE_SHA256 = 'e218243e2e742aedb78803e273e7d8a2b3e00286caf96c0bcf4fb3086b5a26e8'
LEGACY_RECEIVER_SHA256 = 'cc94d103f95980161a9a4d1e16b8903dc24ad4c5588010f5a6c2ee98a3342381'
FULL_BYTES = 294201743360
RESERVE = 225*1024**3


def _path(path):
    p=Path(path).absolute()
    need('..'not in p.parts and not any(a.is_symlink()for a in(p,*p.parents)), 'symlink/traversal in source path')
    return p.resolve(strict=True)


def _generation(path):
    p=_path(path);s=p.stat();need(stat.S_ISREG(s.st_mode),'regular physical file required')
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink)


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _closed(handle):
    need(isinstance(handle,dict)and type(handle.get('pid'))is int and handle['pid']>0
        and str(handle.get('start_ticks','')).isdigit(),'actual PID/start_ticks handle required')
    p=Path('/proc',str(handle['pid']),'stat')
    try: current=p.read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError): return
    need(current!=str(handle['start_ticks']),'original source actor is not physically closed')


def _check_pins(pins):
    for path,generation in pins:
        need(_generation(path)==tuple(generation),'source file generation changed')


def cache_from_closed_source(root, launch, assets):
    """Snapshot authenticated closed metadata and candidates, without copying.

    A candidate is NOT currently SHA-qualified by this function. The new
    receiver must hash all candidates in load_readonly_cache before download.
    """
    root=_path(root);need(root==_path(launch['root']),'source launch root differs')
    pins={};handles={};lookup={a['path']:a for a in assets}
    def add(path):
        p=_path(path);g=_generation(p);h=sha(p);need(_generation(p)==g,'metadata changed during read')
        if str(p)in pins:need(pins[str(p)]==dict(path=str(p),generation=list(g),sha256=h),'ancestry metadata differs')
        pins[str(p)]=dict(path=str(p),generation=list(g),sha256=h)
        return p
    def read(path):return json.loads(add(path).read_text())
    def close(h):
        _closed(h);handles[(h['pid'],str(h['start_ticks']))]=dict(pid=h['pid'],start_ticks=str(h['start_ticks']))
    state=read(root/'private/execution-state.json')
    need(state.get('status')in('failed','complete')and state.get('actual_exit_code')==(0 if state['status']=='complete'else 1), 'terminal source exit required')
    for n in('wrapper','child'):
        need(state[n]==launch['actual_state'][n],'original source owner differs');close(state[n])
    manifest=read(root/'private/stage-receipt.json')
    body={n:manifest[n]for n in('source_commit','files_sha256','full_paper_complete')}
    need(_hash(body)==manifest['bundle_sha256']==launch['bundle_sha256']
        and manifest['source_commit']==launch['source_commit'],'original source bundle differs')
    for name,digest in manifest['files_sha256'].items():
        rel=Path(name);need(not rel.is_absolute()and'..'not in rel.parts,'unsafe source member')
        need(sha(add(root/rel))==digest,'frozen source member changed')
    config=read(root/'config.json');inventory=read(root/'manifest.json')
    need(config['source_revision']==REVISION and inventory['files']==assets,'original source revision/inventory differs')
    if 'native_binary'in config:
        need(sha(add(config['native_binary']))==config['native_binary_sha256'],'native source binary changed')
    inherited={}
    cache_file=root/'private/cache.json'
    if cache_file.exists():
        need(sha(add(cache_file))==config['cache_sha256'],'prior source cache identity changed')
        prior=read(cache_file);inherited=prior['entries']
        for record in prior['source_metadata']:
            p=_path(record['path']);need(_generation(p)==tuple(record['generation'])and sha(p)==record['sha256'],'prior source ancestry changed');add(p)
        for h in prior['closed_actors']:close(h)
    for p in sorted((root/'private').glob('owned-*/started.json')):
        stage=read(p)
        for n in('worker','guardian'):close(stage[n])
        result=p.with_name('stage-result.json')
        if result.exists():read(result)
    for name in('failed.json','completed.json','progress.json'):
        p=root/name
        if p.exists():
            value=read(p)
            for n in('actor','current_worker'):
                if value.get(n)is not None:close(value[n])
    journal=add(root/'receipts.jsonl');raw=journal.read_bytes()
    need(not raw or raw.endswith(b'\n'),'partial terminal journal tail needs separate audit')
    rows=[json.loads(line)for line in raw.splitlines()];entries={}
    need(len({r['path']for r in rows})==len(rows),'duplicate source journal path')
    def candidate(name,path,kind):
        a=lookup[name];p=_path(path);g=_generation(p)
        need(g[2]==a['expected_bytes']and not p.with_name(p.name+'.aria2').exists(),'incomplete source candidate')
        entries[name]=dict(physical_path=str(p),bytes=a['expected_bytes'],sha256=a['lfs_sha256'],generation=list(g),candidate_origin=kind)
    for row in rows:
        need(row['path']in lookup,'unknown source receipt');a=lookup[row['path']]
        need(row['bytes']==a['expected_bytes']and row['sha256']==a['lfs_sha256']and row['matches_publisher_lfs_sha256']is True,'source receipt LFS identity differs')
        h=row.get('native_worker')
        if h is not None:need(row['native_exit_code']==0,'unsuccessful native receipt');close(h)
        else:need(row.get('reused_readonly_cache')or row.get('reused_closed_qualification_archive'),'missing original native owner')
        if row.get('reused_readonly_cache'):
            need(row['path']in inherited and row['payload_path']==inherited[row['path']]['physical_path'],'prior physical cache path differs')
            candidate(row['path'],row['payload_path'],'closed_external_cache_receipt')
        else:
            p=root/'archives'/row['path']
            need('payload_path'not in row or _path(row['payload_path'])==_path(p),'owned source receipt path differs')
            candidate(row['path'],p,'closed_owned_receipt')
    terminal=read(root/('completed.json'if state['status']=='complete'else'failed.json'))
    count_key,bytes_key=('archive_count','archive_bytes')if state['status']=='complete'else('completed_archive_count','completed_archive_bytes')
    need(terminal[count_key]==len(rows)and terminal[bytes_key]==sum(r['bytes']for r in rows),'terminal source journal totals differ')
    # Preserve complete future seeds and a possible completed unjournaled file;
    # partial/control pairs remain untouched and are never candidates.
    for name,a in lookup.items():
        if name in entries:continue
        if name in inherited:
            candidate(name,inherited[name]['physical_path'],'closed_future_external_candidate');continue
        p=root/'archives'/name
        if p.is_file()and p.stat().st_size==a['expected_bytes']and not p.with_name(p.name+'.aria2').exists():
            candidate(name,p,'complete_size_unjournaled_candidate')
    _check_pins((r['path'],r['generation'])for r in pins.values())
    for h in handles.values():_closed(h)
    return dict(format='hot3d_readonly_completed_cache_v1',dataset_revision=REVISION,
        manifest_sha256=sha(root/'manifest.json'),source_journal_archives=len(rows),source_journal_bytes=sum(r['bytes']for r in rows),
        candidate_archives=len(entries),entries=entries,source_metadata=list(pins.values()),closed_actors=list(handles.values()),
        candidate_payloads_rehashed=False,whole_publisher_SHA_admission_still_required=True,
        copied_archive_bytes=0,numeric_qualification_complete=False,full_paper_complete=False)


class ReadOnlyCache:
    def __init__(self, entries, pins, handles):
        self._entries=MappingProxyType(entries);self._pins=tuple(pins);self._handles=tuple(handles)
    @property
    def verified_archives(self):return len(self._entries)
    @property
    def verified_bytes(self):return sum(r[1]for r in self._entries.values())
    def check(self):
        _check_pins(self._pins)
        for h in self._handles:_closed(h)
    def path_for(self, asset):
        self.check();record=self._entries.get(asset['path'])
        if record is None:return None
        need((record[1],record[2])==(asset['expected_bytes'],asset['lfs_sha256']),'cache requested asset identity differs')
        return record[0]


def load_readonly_cache(path, *, expected_sha256, assets):
    try:
        path=_path(path);stamp=_generation(path);need(sha(path)==expected_sha256,'cache map digest differs')
        document=json.loads(path.read_text());pins=[(path,stamp)]
        need(document['format']=='hot3d_readonly_completed_cache_v1'and document['dataset_revision']==REVISION,'cache format/revision differs')
        lookup={a['path']:a for a in assets};entries={}
        for h in document['closed_actors']:_closed(h)
        for record in document['source_metadata']:
            p=_path(record['path']);g=tuple(record['generation'])
            need(_generation(p)==g and sha(p)==record['sha256']and _generation(p)==g,'closed source metadata changed')
            pins.append((p,g))
        for name,record in document['entries'].items():
            need(name in lookup,'cache contains unknown archive');a=lookup[name];p=_path(record['physical_path']);g=tuple(record['generation'])
            need(record['bytes']==a['expected_bytes']and record['sha256']==a['lfs_sha256'],'cache LFS descriptor differs')
            need(_generation(p)==g and not p.with_name(p.name+'.aria2').exists(),'cache payload generation/control changed')
            verify_payload(p,a)
            need(_generation(p)==g,'cache generation changed while hashing')
            pins.append((p,g));entries[name]=(p,a['expected_bytes'],a['lfs_sha256'])
        cache=ReadOnlyCache(entries,pins,document['closed_actors']);cache.check();return cache
    except (OSError,KeyError,TypeError)as exc:raise ValueError('read-only source cache admission failed')from exc


def _native_command(binary, asset, target):
    return [str(binary),'--no-conf=true','--max-tries=1','--connect-timeout=15','--timeout=60',
        '--max-connection-per-server=1','--split=1','--max-download-limit=32M','--file-allocation=none',
        '--allow-overwrite=false','--auto-file-renaming=false','--continue=false','--check-integrity=true',
        '--checksum=sha-256='+asset['lfs_sha256'],'--console-log-level=warn','--summary-interval=60',
        '--stop-with-process='+str(os.getpid()),'--dir='+str(target.parent),'--out='+target.name,asset['url']]


def receive_assets(root, assets, cache, *, binary, guard):
    """Transport helper only. Small inputs cannot publish full completion."""
    root=_path(root);guard();cache.check()
    verify_capacity(root,remaining=sum(a['expected_bytes']for a in assets if cache.path_for(a)is None),reserve=RESERVE)
    need(not (root/'archives').exists(),'new owned archive root must not exist')
    child=None;workers=[];count=received=reused=0;own_pins=[]
    def check():
        guard();cache.check();_check_pins(own_pins);verify_capacity(root,remaining=0,reserve=RESERVE)
    def progress(current):
        atomic(root/'progress.json',dict(observed_utc=now(),actor=actor(os.getpid()),completed_archive_count=count,
            completed_archive_bytes=received,current_archive_path=current,
            current_worker=actor(child.pid)if child is not None and child.poll()is None else None,
            target_archive_count=len(assets),target_archive_bytes=sum(a['expected_bytes']for a in assets),
            reused_readonly_cache_archives=reused,numeric_qualification_complete=False,full_paper_complete=False))
    try:
        with(root/'receipts.jsonl').open('x')as journal,(root/'aria2.log').open('xb')as log:
            for asset in assets:
                check();target=cache.path_for(asset);worker=None;cached=target is not None
                if not cached:
                    target=root/'archives'/asset['path'];need(not target.exists()and not target.with_name(target.name+'.aria2').exists(),'existing partial requires explicit closed audit')
                    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                    child=subprocess.Popen(_native_command(binary,asset,target),cwd=root,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    worker=actor(child.pid);workers.append(worker);progress(asset['path'])
                    while child.poll()is None:
                        check();need((root/'aria2.log').stat().st_size<=32*1024**2,'native log exceeded bound')
                        progress(asset['path']);time.sleep(1)
                    need(child.returncode==0,'native source download failed; partial preserved without retry')
                    g=_generation(target);verify_payload(target,asset);need(_generation(target)==g,'new payload generation changed')
                    own_pins.append((target,g));_closed(worker)
                check();count+=1;received+=asset['expected_bytes'];reused+=int(cached)
                row=dict(observed_utc=now(),path=asset['path'],bytes=asset['expected_bytes'],sha256=asset['lfs_sha256'],
                    matches_publisher_lfs_sha256=True,reused_readonly_cache=cached,payload_path=str(target),
                    payload_generation=list(_generation(target)),native_worker=worker,native_exit_code=0 if worker else None)
                journal.write(json.dumps(row,sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno());progress(asset['path'])
            check()
        return dict(archive_count=count,archive_bytes=received,reused_cache_archives=reused,
            new_download_archives=count-reused,native_workers=workers,numeric_qualification_complete=False,full_paper_complete=False)
    except BaseException:
        if child is not None and child.poll()is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        raise


def _require_full_totals(result):
    need(result['archive_count']==2804 and result['archive_bytes']==FULL_BYTES,'complete public source totals required')


def main(root):
    root=_path(root);os.nice(10);owner=actor(os.getpid());cache=None
    try:
        config=json.loads((root/'config.json').read_text())
        stage=json.loads((root/'private/stage-receipt.json').read_text())
        pins=[]
        for name,digest in stage['files_sha256'].items():
            p=_path(root/name);need(sha(p)==digest,'frozen receiver source changed');pins.append((p,_generation(p)))
        need(sha(root/'manifest.json')==MANIFEST_SHA256,'fixed complete public manifest differs')
        assets=checked_assets(json.loads((root/'manifest.json').read_text()))
        legacy=root/'scripts/paper/hot3d_full_train_receiver.py'
        need(sha(legacy)==LEGACY_RECEIVER_SHA256,'original source receiver changed')
        binary=_path(config['native_binary']);need(sha(binary)==NATIVE_SHA256==config['native_binary_sha256'],'native binary differs')
        pins.append((binary,_generation(binary)))
        document=json.loads((root/'private/cache.json').read_text())
        need(document['manifest_sha256']==MANIFEST_SHA256,'cache source manifest differs')
        cache=load_readonly_cache(root/'private/cache.json',expected_sha256=config['cache_sha256'],assets=assets)
        def guard():_check_pins(pins);cache.check()
        guard();exclusive(root/'cache-admission.json',dict(observed_utc=now(),actor=owner,
            candidate_archives=cache.verified_archives,whole_cache_bytes_rehashed=cache.verified_bytes,
            all_publisher_LFS_sha256_exact=True,old_file_generations_unchanged=True,copied_archive_bytes=0,
            numeric_qualification_complete=False,full_paper_complete=False))
        exclusive(root/'started.json',dict(observed_utc=now(),actor=owner,source_revision=REVISION,
            manifest_sha256=MANIFEST_SHA256,target_archives=2804,target_bytes=FULL_BYTES,
            reserve_bytes=RESERVE,automatic_retry=False,actual_gpu_actions=0,full_paper_complete=False))
        result=receive_assets(root,assets,cache,binary=binary,guard=guard);_require_full_totals(result);guard()
        exclusive(root/'completed.json',dict(result,observed_utc=now(),actor=owner,source_revision=REVISION,
            manifest_sha256=MANIFEST_SHA256,all_publisher_lfs_sha256_exact=True,
            source_numeric_qualification_complete=False,author_mixture_reconstructed=False,actual_gpu_actions=0,actual_model_runs=0))
    except BaseException as exc:
        rows=[];journal=root/'receipts.jsonl'
        if journal.exists():
            raw=journal.read_bytes();lines=raw.splitlines();lines=lines if not raw or raw.endswith(b'\n')else lines[:-1]
            rows=[json.loads(s)for s in lines]
        exclusive(root/'failed.json',dict(observed_utc=now(),actor=owner,error_type=type(exc).__name__,message=str(exc)[:400],
            completed_archive_count=len(rows),completed_archive_bytes=sum(r['bytes']for r in rows),
            owned_native_worker_exited=True,original_payloads_and_partials_preserved=True,automatic_retry=False,
            numeric_qualification_complete=False,full_paper_complete=False))
        raise
