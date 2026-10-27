"""Private source/index preparation inside an owned, guarded CPU process.

Full, mini and engineering identities remain distinct. No source is extracted,
no scientific model is trained, and caller-owned cache descriptors are borrowed.
The worker closes only its inherited rights; the owner can keep the cache for
later native model phases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from . import emg2pose_archive as transport, emg2pose_data as storage
from . import emg2pose_memory_data as memory, emg2pose_memory_index as index
from .emg2pose_memory_archive import RESERVE_BYTES, SealedArchive, available_memory_bytes
from .emg2pose_memory_process import _atomic, _binding, run_owned_stage


_HASH = re.compile('[0-9a-f]{64}')
_IMPORTED_SHA = storage._sha256(__file__)
_CODE = ('__init__.py','download.py','emg2pose_archive.py','emg2pose_data.py',
         'emg2pose_memory_archive.py','emg2pose_memory_data.py','emg2pose_memory_index.py',
         'emg2pose_runtime.py','emg2pose_memory_process.py','emg2pose_memory_prepare.py')
_FULL_SPLITS = {'train':17136,'val':1950,'test':6167}


def _policy(cohort, reserve_bytes):
    if cohort not in {'full','mini','engineering'}:
        raise ValueError('cohort must explicitly be full, mini or engineering')
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('memory reserve must be a nonnegative integer')
    if cohort != 'engineering' and reserve_bytes < RESERVE_BYTES:
        raise ValueError('official source preparation retains at least 256 GiB')


def _code_paths():
    if storage._sha256(__file__) != _IMPORTED_SHA:
        raise ValueError('imported preparation producer changed on disk')
    return {name:Path(__file__).parent/name for name in _CODE}


def _stamp(path):
    value = storage._safe_path(path, file=True).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def _inputs(fd, acquisition, metadata, upstream):
    upstream,_,_ = storage._official(upstream)
    code = _code_paths()
    paths = [acquisition,metadata,*code.values(),*(upstream/name for name in storage.SOURCE_HASHES)]
    return (_binding(fd),{path:_stamp(path) for path in paths},
            {'egoemg_repro/'+name:storage._sha256(path) for name,path in code.items()})


def _current(fd, bindings):
    original,stamps,producer = bindings
    if _binding(fd) != original or any(_stamp(path) != stamp for path,stamp in stamps.items()):
        raise ValueError('preparation source, producer or input binding changed')
    _code_paths()


def _receipt(fd, receipt, cohort):
    _policy(cohort,RESERVE_BYTES)
    info = _binding(fd)
    if not isinstance(receipt,dict) or not isinstance(receipt.get('source'),dict):
        raise ValueError('complete sealed-memory acquisition receipt is required')
    source = receipt['source']
    if (receipt.get('format') != 'emg2pose_sealed_memory_archive_v1'
            or receipt.get('complete') is not True or receipt.get('full_corpus_complete') is not False
            or type(receipt.get('source_bytes_received')) is not int
            or receipt['source_bytes_received'] != info['size']
            or source.get('expected_bytes') != info['size']
            or not isinstance(receipt.get('full_sha256'),str)
            or not _HASH.fullmatch(receipt['full_sha256']) or receipt.get('seals') != 15
            or not isinstance(source.get('etag'),str) or not re.fullmatch(r'"[^"\r\n]+"',source['etag'])
            or not isinstance(source.get('last_modified'),str) or not source['last_modified'].strip()
            or not isinstance(source.get('url'),str) or not source['url']):
        raise ValueError('complete source receipt differs from the immutable object')
    memory._acquisition(cohort,info['size'],receipt['full_sha256'],receipt)
    return receipt


def _identity(identity, cohort, receipt):
    if (identity.get('cohort') != cohort or identity.get('full_corpus_complete') is not (cohort=='full')
            or identity.get('engineering_only') is not (cohort!='full')
            or identity.get('archive_bytes') != receipt['source_bytes_received']
            or identity.get('archive_sha256') != receipt['full_sha256']
            or identity.get('input_channels') != 16 or identity.get('angle_dimensions') != 20):
        raise ValueError('prepared corpus identity differs from acquired source/cohort')
    if cohort == 'full' and (identity.get('file_count') != storage.FULL_FILE_COUNT
                            or identity.get('counts',{}).get('split') != _FULL_SPLITS):
        raise ValueError('complete source requires all 25253 official sessions/roles')
    if cohort == 'mini' and identity.get('file_count') != 30:
        raise ValueError('mini requires all30 sessions and remains engineering-only')


def prepare_source(fd, output, *, acquisition_receipt, expected_acquisition_sha256,
                   metadata_csv, upstream, cohort='full'):
    """Independently inspect, index and admit a borrowed immutable source.

    A caller supervising this long operation must continue resource checks. The
    owned wrapper below runs it as a child, keeping the parent's guard responsive.
    """
    _policy(cohort,RESERVE_BYTES)
    output = storage._safe_path(output)
    stage = output.with_name(output.name+'.incomplete')
    if output.exists() or stage.exists():
        raise FileExistsError('existing prepared source or incomplete work is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('preparation parent must already exist')
    acquisition = storage._safe_path(acquisition_receipt,file=True)
    metadata = storage._safe_path(metadata_csv,file=True)
    if (not isinstance(expected_acquisition_sha256,str)
            or not _HASH.fullmatch(expected_acquisition_sha256)
            or storage._sha256(acquisition) != expected_acquisition_sha256):
        raise ValueError('acquisition file differs from separately frozen SHA256')
    receipt = _receipt(fd,storage._json(acquisition),cohort)
    bindings = _inputs(fd,acquisition,metadata,upstream)
    storage._space(output.parent,4*1024**2)
    stage.mkdir(mode=0o700)
    try:
        _atomic(stage/'status.json',dict(preparation_complete=False,phase='inventory'))
        inventory = memory.inspect_archive(fd,expected_bytes=receipt['source_bytes_received'],
            expected_sha256=receipt['full_sha256'],metadata_csv=metadata,upstream=upstream,
            cohort=cohort,acquisition_receipt=receipt)
        _current(fd,bindings)
        _atomic(stage/'status.json',dict(preparation_complete=False,phase='index'))
        prepared = index.prepare_index(fd,inventory,stage/'index',upstream=upstream,metadata_csv=metadata)
        with index.open_source(fd,stage/'index',expected_completion_sha256=prepared['completion_sha256'],
                               upstream=upstream,metadata_csv=metadata) as verified:
            identity = verified.data_identity
            if identity != prepared['data_identity']:
                raise ValueError('independently admitted index differs')
            _identity(identity,cohort,receipt)
            result = dict(format='emg2pose_prepared_memory_source_v1',preparation_complete=True,
                cohort=cohort,full_corpus_complete=cohort=='full',scientific_complete=False,
                full_paper_complete=False,completion_requires_owned_phase=True,
                acquisition_receipt_sha256=expected_acquisition_sha256,index_directory='index',
                index_completion_sha256=prepared['completion_sha256'],data_identity=identity,
                producer_files_sha256=bindings[2])
            storage._space(stage)
            _current(fd,bindings)
            storage._write_json(stage/'preparation.json',result)
            _atomic(stage/'status.json',dict(preparation_complete=True,phase='prepared'))
            _current(fd,bindings)
            verified.assert_current()
            storage._publish_directory(stage,output)
        return result
    except BaseException as error:
        if stage.exists():
            (stage/'preparation.json').unlink(missing_ok=True)
            _atomic(stage/'status.json',dict(preparation_complete=False,phase='failed',
                error_type=type(error).__name__,error=str(error),
                policy='owned incomplete index retained; caller source never removed'))
        raise


def _read_prepared(output, cohort, receipt, acquisition_sha, producer):
    result = storage._json(output/'preparation.json')
    if (result.get('format') != 'emg2pose_prepared_memory_source_v1'
            or result.get('preparation_complete') is not True or result.get('cohort') != cohort
            or result.get('scientific_complete') is not False or result.get('full_paper_complete') is not False
            or result.get('completion_requires_owned_phase') is not True
            or result.get('acquisition_receipt_sha256') != acquisition_sha
            or result.get('producer_files_sha256') != producer
            or result.get('index_directory') != 'index'
            or result.get('index_completion_sha256') != storage._sha256(output/'index/completion.json')):
        raise ValueError('owned worker did not publish a matching prepared source')
    completion = storage._json(output/'index/completion.json')
    if (completion.get('format') != 'emg2pose_memory_index_v1' or completion.get('complete') is not True
            or completion.get('data_identity') != result['data_identity']):
        raise ValueError('prepared receipt/index identities disagree')
    _identity(result['data_identity'],cohort,receipt)
    return result


def prepare_owned_source(cache, output, *, upstream, metadata_csv, cohort='full', reserve_bytes=RESERVE_BYTES):
    """Supervise source preparation without blocking the cache owner's guard."""
    _policy(cohort,reserve_bytes)
    if type(cache) is not SealedArchive:
        raise ValueError('an actual owned SealedArchive is required')
    fd = cache.fileno()
    receipt = _receipt(fd,cache.receipt,cohort)
    output = storage._safe_path(output)
    if output.exists():
        raise FileExistsError('owned preparation output is never overwritten')
    output.mkdir(mode=0o700)
    try:
        acquisition = output/'acquisition.json'
        storage._write_json(acquisition,receipt)
        acquisition_sha = storage._sha256(acquisition)
        metadata = storage._safe_path(metadata_csv,file=True)
        bindings = _inputs(fd,acquisition,metadata,upstream)
        def guard():
            available = available_memory_bytes()
            if type(available) is not int or available < reserve_bytes:
                raise transport.ArchiveCapacityError('source preparation memory reserve lost')
            if cache.fileno() != fd:
                raise ValueError('caller cache descriptor changed')
            _current(fd,bindings)
        guard()
        _atomic(output/'status.json',dict(preparation_complete=False,phase='owned-worker'))
        bootstrap = ('import sys;sys.path.insert(0,sys.argv.pop(1));'
                     'from egoemg_repro.emg2pose_memory_prepare import main;raise SystemExit(main())')
        command = [sys.executable,'-c',bootstrap,str(Path(__file__).resolve().parents[1]),'--cohort',cohort,
            '--upstream',str(upstream),'--metadata-csv',str(metadata),
            '--acquisition-receipt',str(acquisition),'--expected-acquisition-sha256',acquisition_sha,
            '--output',str(output/'prepared')]
        phase = run_owned_stage(command,output/'owned-phase',descriptors={'ARCHIVE':fd},guard=guard,
                                environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        guard()
        if phase['status'] != 'success' or phase['worker_exit_code'] != 0 or not phase['all_owned_consumers_exited']:
            raise ValueError('source preparation requires a successful owned phase')
        prepared = _read_prepared(output/'prepared',cohort,receipt,acquisition_sha,bindings[2])
        result = dict(format='emg2pose_owned_preparation_v1',preparation_complete=True,
            scientific_complete=False,full_paper_complete=False,completion_requires_owned_phase=True,
            prepared_directory='prepared',preparation_sha256=storage._sha256(output/'prepared/preparation.json'),
            index_completion_sha256=prepared['index_completion_sha256'],data_identity=prepared['data_identity'],
            phase_receipt_sha256=storage._sha256(output/'owned-phase/stage-result.json'),
            acquisition_receipt_sha256=acquisition_sha,producer_files_sha256=bindings[2])
        storage._space(output)
        guard()
        _atomic(output/'status.json',dict(preparation_complete=True,phase='prepared'))
        guard()
        storage._write_json(output/'owned-preparation.json',result)
        return result
    except BaseException as error:
        (output/'owned-preparation.json').unlink(missing_ok=True)
        _atomic(output/'status.json',dict(preparation_complete=False,phase='failed',
                                       error_type=type(error).__name__,error=str(error)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['full','mini','engineering'],default='full')
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--metadata-csv',type=Path,required=True)
    parser.add_argument('--acquisition-receipt',type=Path,required=True)
    parser.add_argument('--expected-acquisition-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    value = os.environ.get('EGOEMG_STAGE_FD_ARCHIVE','')
    if not value.isdigit():
        raise ValueError('an inherited named archive descriptor is required')
    fd = int(value)
    try:
        prepare_source(fd,args.output,acquisition_receipt=args.acquisition_receipt,
            expected_acquisition_sha256=args.expected_acquisition_sha256,
            metadata_csv=args.metadata_csv,upstream=args.upstream,cohort=args.cohort)
        return 0
    finally:
        os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
