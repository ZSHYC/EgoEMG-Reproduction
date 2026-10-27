"""Prepare actual owned original files; retire only reconstructable TAR scratch.

All original HDF5 objects, attributes, extra files and opaque TAR bytes survive.
No receipt here constitutes model training or whole-paper completion.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from . import emg2pose_data as storage,emg2pose_disk_download as disk
from . import emg2pose_tar_preservation as retained
from .emg2pose_memory_process import same_process

FORMAT='emg2pose_original_disk_preparation_v1'
RETIREMENT='emg2pose_reconstructable_tar_retirement_v1'
_CODE=(Path(__file__),Path(disk.__file__),Path(retained.__file__),Path(storage.__file__))
_IMPORTED={p.name:storage._sha256(p) for p in _CODE}
_STAMPS={p:retained._stamp(p) for p in _CODE}


def _guard(parent):
    retained._guard(parent)
    if any(retained._stamp(p)!=s for p,s in _STAMPS.items()):raise ValueError('imported preparation producer changed')


def _frozen(path,expected):
    if not retained._digest(expected) or storage._sha256(path)!=expected:raise ValueError('separately frozen preparation/source identity differs')
    return storage._json(path)


def _process_identity(value):
    return (type(value) is dict and all(type(value.get(k)) is int and value[k]>=0 for k in ('pid','uid','sid','pgrp')) and
            value['pid']>0 and type(value.get('start_ticks')) is str and value['start_ticks'].isdigit())


def _publish_preparation(path,value):
    temporary=path.with_name('.'+path.name+'.incomplete')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);owned=os.fstat(fd)
    try:
        with os.fdopen(fd,'wb') as stream:
            body=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
            if stream.write(body)!=len(body):raise OSError('short preparation marker write')
            stream.flush();os.fsync(stream.fileno())
        _guard(path.parent);os.link(temporary,path,follow_symlinks=False);temporary.unlink();retained._sync(path.parent)
    except BaseException:
        for p in (temporary,path):
            try:s=p.lstat()
            except FileNotFoundError:continue
            if (s.st_dev,s.st_ino)==(owned.st_dev,owned.st_ino):p.unlink()
        raise


@contextmanager
def _lock(download):
    download=storage._safe_path(download)
    path=storage._safe_path(download/'archive.lock',file=True)
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise FileExistsError('original download still has an active owner') from error
        yield download
    finally:os.close(fd)


def _download(download,expected,cohort):
    receipt_path=download/'receipt.json';r=_frozen(receipt_path,expected)
    source=r.get('source');policy=r.get('policy',{});owned=r.get('owned_stage',{})
    if (r.get('format')!=disk.FORMAT or r.get('complete_bytes') is not True or type(source) is not dict or
            set(source)!={'url','expected_bytes','etag','last_modified'} or source!=r.get('final_source') or
            not retained._integer(source.get('expected_bytes'),1024) or source['expected_bytes']%512 or
            not retained._digest(r.get('full_sha256')) or type(source.get('url')) is not str or
            type(source.get('etag')) is not str or not source['etag'].startswith('"') or not source['etag'].endswith('"') or
            type(source.get('last_modified')) is not str or not source['last_modified'] or
            type(r.get('source_bytes_received')) is not int or r['source_bytes_received']!=source['expected_bytes'] or
            any(r.get(k) is not False for k in ('source_qualified','full_corpus_complete','scientific_complete','full_paper_complete')) or
            policy.get('producer_sha256')!=_IMPORTED[Path(disk.__file__).name] or
            type(policy.get('host_reserve_bytes')) is not int or policy['host_reserve_bytes']!=256*1024**3 or
            type(policy.get('disk_reserve_bytes')) is not int or policy['disk_reserve_bytes']!=15*1024**3):
        raise ValueError('actual completed original disk-byte receipt/source/policy required')
    asset=r.get('asset')
    if cohort in ('mini','full'):
        url,length=disk.transport._ASSETS[cohort]
        if asset!=cohort or (source['url'],source['expected_bytes'])!=(url,length):raise ValueError('official complete asset/cohort differs')
        if cohort=='mini' and r['full_sha256']!=storage.MINI_SHA256:raise ValueError('complete original mini SHA differs')
    elif (cohort!='engineering' or asset!='engineering' or source['expected_bytes']>storage.MINI_BYTES or
            source['url'] in {v[0] for v in disk.transport._ASSETS.values()}):
        raise ValueError('nonofficial bounded engineering cannot replace an official full/mini source')
    if (owned.get('format')!='egoemg_owned_stage_v1' or owned.get('status')!='success' or
            owned.get('all_owned_consumers_exited') is not True or type(owned.get('worker_exit_code')) is not int or
            owned['worker_exit_code']!=0 or owned.get('descriptor_bindings')!={} or owned.get('scientific_complete') is not False):
        raise ValueError('actual successful owned disk stage required')
    if any(not _process_identity(owned.get(k)) or same_process(owned[k]) for k in ('worker','guardian')):
        raise ValueError('original download consumer/guardian must actually have exited')
    matches=[]
    for path in (download/'attempts').glob('*/stage-result.json'):
        storage._safe_path(path,file=True)
        if storage._json(path)==owned:matches.append(path)
    if len(matches)!=1:raise ValueError('unique actual owned stage result required')
    archive=storage._safe_path(download/'archive.tar',file=True);s=archive.stat()
    if (not stat.S_ISREG(s.st_mode) or s.st_uid!=os.geteuid() or s.st_nlink!=1 or s.st_mode&0o222 or
            s.st_size!=source['expected_bytes']):raise ValueError('single-link owned readonly original TAR required')
    _guard(download)
    return r,archive,matches[0]


def _members(preserved):
    manifest=storage._json(preserved/'manifest.json')
    return [preserved/'contents'/m['name'] for m in manifest['members'] if m['kind']=='file' and Path(m['name']).suffix=='.hdf5']


def _inspect(prepared,upstream,csv,cohort):
    acquisition=prepared/'mini-acquisition.json' if cohort=='mini' else None
    return storage.inspect_corpus(_members(prepared/'preserved'),csv,cohort=cohort,upstream=upstream,acquisition_receipt=acquisition)


def prepare_disk_source(download,output,*,upstream,metadata_csv,cohort,expected_download_receipt_sha256,on_progress=None):
    """Preserve every original byte and inventory every original HDF5 file."""
    if cohort not in ('engineering','mini','full'):raise ValueError('explicit source cohort required')
    if on_progress is not None and not callable(on_progress):raise ValueError('callable progress observer required')
    output=storage._safe_path(output);csv=storage._safe_path(metadata_csv,file=True)
    if output.exists():raise FileExistsError('existing source preparation is never overwritten')
    if not output.parent.is_dir():raise ValueError('existing preparation parent required')
    upstream,_,_=storage._official(upstream)
    with _lock(download) as download:
        receipt,archive,stage=_download(download,expected_download_receipt_sha256,cohort)
        stamp=retained._stamp(archive);csv_sha=storage._sha256(csv);download_sha=storage._sha256(download/'receipt.json');stage_sha=storage._sha256(stage)
        output.mkdir(mode=0o700)
        with (output/'download-receipt.json').open('xb') as stream:
            stream.write((download/'receipt.json').read_bytes());stream.flush();os.fsync(stream.fileno())
        preserved=retained.preserve_tar(archive,output/'preserved',expected_bytes=receipt['source']['expected_bytes'],
            expected_sha256=receipt['full_sha256'],on_progress=on_progress)
        manifest=storage._json(output/'preserved/manifest.json')
        if cohort=='mini':
            entries={s['name']:s for s in manifest['segments'] if s['kind']=='payload'}
            files=[dict(name=m['name'],bytes=m['bytes'],offset_data=m['offset_data'],sha256=entries[m['name']]['sha256']) for m in manifest['members'] if m['kind']=='file']
            storage._write_json(output/'mini-acquisition.json',dict(complete=True,bytes=storage.MINI_BYTES,expected_bytes=storage.MINI_BYTES,
                sha256=storage.MINI_SHA256,archive=str(archive),files=files))
        inventory=_inspect(output,upstream,csv,cohort)
        storage._write_json(output/'source-inventory.json',inventory)
        _guard(output)
        if (retained._stamp(archive)!=stamp or storage._sha256(csv)!=csv_sha or storage._sha256(stage)!=stage_sha or
                storage._sha256(download/'receipt.json')!=download_sha or storage._sha256(output/'download-receipt.json')!=download_sha):
            raise ValueError('original source/download/metadata changed during preparation')
        result=dict(format=FORMAT,preparation_complete=True,cohort=cohort,download_directory=str(download),
            download_receipt_sha256=download_sha,download_stage_path=str(stage),download_stage_sha256=stage_sha,
            archive_bytes=receipt['source']['expected_bytes'],archive_sha256=receipt['full_sha256'],archive_stamp=list(stamp),
            preservation_completion_sha256=preserved['completion_sha256'],source_inventory_sha256=storage._sha256(output/'source-inventory.json'),
            metadata_csv=str(csv),metadata_csv_sha256=csv_sha,upstream=str(upstream),producer_files_sha256=_IMPORTED,
            original_hdf5_files=len(inventory['sessions']),rows=inventory['total_rows'],counts=inventory['counts'],
            full_corpus_complete=inventory['full_corpus_complete'],scientific_complete=False,full_paper_complete=False)
        _publish_preparation(output/'preparation.json',result);return result


def retire_managed_tar(prepared,*,expected_preparation_sha256,on_progress=None):
    """Remove only the bound redundant archive, after fresh complete proofs."""
    prepared=storage._safe_path(prepared)
    if any((prepared/n).exists() for n in ('retirement-started.json','retirement.json')):
        raise FileExistsError('an attempted scratch retirement is never automatically retried')
    if on_progress is not None and not callable(on_progress):raise ValueError('callable progress observer required')
    proof=_frozen(prepared/'preparation.json',expected_preparation_sha256)
    if (proof.get('format')!=FORMAT or proof.get('preparation_complete') is not True or
            proof.get('producer_files_sha256')!=_IMPORTED or
            any(proof.get(k) is not False for k in ('scientific_complete','full_paper_complete'))):
        raise ValueError('complete original preparation/scope/producer required')
    with _lock(proof['download_directory']) as download:
        receipt,archive,stage=_download(download,proof['download_receipt_sha256'],proof['cohort'])
        if (list(retained._stamp(archive))!=proof['archive_stamp'] or receipt['full_sha256']!=proof['archive_sha256'] or
                receipt['source']['expected_bytes']!=proof['archive_bytes'] or str(stage)!=proof['download_stage_path'] or
                storage._sha256(stage)!=proof['download_stage_sha256'] or
                storage._sha256(prepared/'download-receipt.json')!=proof['download_receipt_sha256']):
            raise ValueError('bound owned original TAR or source stage differs')
        inventory=_frozen(prepared/'source-inventory.json',proof['source_inventory_sha256'])
        csv=storage._safe_path(proof['metadata_csv'],file=True)
        if storage._sha256(csv)!=proof['metadata_csv_sha256']:raise ValueError('source metadata CSV differs')
        if (inventory.get('cohort')!=proof['cohort'] or inventory.get('full_corpus_complete') is not proof['full_corpus_complete'] or
                len(inventory['sessions'])!=proof['original_hdf5_files'] or inventory['total_rows']!=proof['rows'] or inventory['counts']!=proof['counts']):
            raise ValueError('complete source inventory/scope differs')
        paths=[prepared/'preparation.json',prepared/'source-inventory.json',prepared/'download-receipt.json',
            prepared/'preserved/completion.json',csv,download/'receipt.json',stage]
        if proof['cohort']=='mini':paths.append(prepared/'mini-acquisition.json')
        bindings={p:retained._stamp(p) for p in paths};stamp=tuple(proof['archive_stamp'])
        _,retained_stamps=retained._tree(prepared/'preserved',storage._json(prepared/'preserved/manifest.json'))
        fd=os.open(archive,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            def current():
                _guard(prepared);retained._source_guard(fd,archive,stamp)
                if any(retained._stamp(p)!=v for p,v in bindings.items()):raise ValueError('frozen source preparation binding changed')
            current()
            retained.verify_preservation(prepared/'preserved',expected_completion_sha256=proof['preservation_completion_sha256'],on_progress=on_progress)
            digest=hashlib.sha256();done=0
            for chunk in retained._chunks(fd,0,proof['archive_bytes'],current):
                digest.update(chunk);done+=len(chunk)
                if on_progress is not None:on_progress(dict(phase='original-before-retirement',bytes_processed=done,total_bytes=proof['archive_bytes']))
            if digest.hexdigest()!=proof['archive_sha256']:raise ValueError('actual readonly original TAR SHA differs')
            if _inspect(prepared,Path(proof['upstream']),csv,proof['cohort'])!=inventory:raise ValueError('fresh original complete inventory differs')
            current()
            # Recheck actual ownership/mode/link count and successful source stage.
            _download(download,proof['download_receipt_sha256'],proof['cohort']);current()
            def final_current():
                current()
                if any(retained._stamp(p)!=v for p,v in retained_stamps.items()):
                    raise ValueError('retained original bytes/tree changed after complete verification')
            final_current()
            witness=dict(format=RETIREMENT,preparation_sha256=expected_preparation_sha256,
                archive_bytes=proof['archive_bytes'],archive_sha256=proof['archive_sha256'],archive_stamp=list(stamp),
                preservation_completion_sha256=proof['preservation_completion_sha256'],source_inventory_sha256=proof['source_inventory_sha256'],
                original_hdf5_files=proof['original_hdf5_files'],all_original_bytes_reconstructable=True,
                full_corpus_complete=proof['full_corpus_complete'],scientific_complete=False,full_paper_complete=False)
            storage._write_json(prepared/'retirement-started.json',dict(witness,redundant_tar_retired=False));retained._sync(prepared)
            final_current();directory=os.open(download,os.O_RDONLY|os.O_DIRECTORY)
            try:
                final_current();os.unlink('archive.tar',dir_fd=directory);os.fsync(directory)
            finally:os.close(directory)
            result=dict(witness,redundant_tar_retired=True)
            storage._write_json(prepared/'retirement.json',result);retained._sync(prepared);return result
        finally:os.close(fd)
