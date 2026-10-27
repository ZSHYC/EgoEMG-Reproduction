"""Separate compact metadata transaction and explicit verified source ownership."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

from . import emg2pose_data as storage,emg2pose_memory_index as reference
from . import emg2pose_compact_data as data,emg2pose_compact_archive as archive


FORMAT='emg2pose_compact_index_v1'
DATASET='emg2pose_compact_tar_dataset_v1'
_TOKEN=object()
_HASH=re.compile('[0-9a-f]{64}')
_FILES=('source-inventory.json','metadata.npz','manifest.json','completion.json')
_CODE=(*data._CODE,'emg2pose_memory_index.py','emg2pose_compact_index.py')
_stamp=reference._stamp


def _sources():return {name:storage._sha256(Path(__file__).parent/name) for name in _CODE}
_IMPORTED=_sources()


def _current():
    if _sources()!=_IMPORTED:raise ValueError('imported compact index producer changed')


def _identity(inventory,arrays):
    value=reference._identity(inventory,arrays)
    value.update(format='emg2pose_compact_data_identity_v1',
        inspection_core_sha256=inventory['inspection_core_sha256'],
        reader_sources_sha256=inventory['reader_sources_sha256'])
    return value


def _manifest(inventory,arrays,inventory_sha,metadata_sha):
    return dict(format=DATASET,total_rows=inventory['total_rows'],sessions=reference._sessions(inventory),
        fields=inventory['fields'],archive_bytes=inventory['source_index']['archive_bytes'],
        archive_sha256=inventory['source_index']['archive_sha256'],source_inventory_sha256=inventory_sha,
        metadata_sha256=metadata_sha,metadata_arrays=arrays)


def prepare_index(reader,inventory,output,*,upstream,metadata_csv):
    """Reinspect all actual fields before publishing only four native sidecars."""
    data._reader(reader);_current();output=storage._safe_path(output)
    stage=output.with_name(output.name+'.incomplete')
    if output.exists() or stage.exists():raise FileExistsError('existing compact transaction is never overwritten')
    if not output.parent.is_dir():raise ValueError('compact index parent must already exist')
    if type(inventory) is not dict or inventory.get('format')!=data.FORMAT or inventory.get('complete') is not True:
        raise ValueError('complete compact source inventory required')
    try:
        actual=data.inspect_compact(reader,expected_receipt_sha256=reader.receipt['sha256'],
            metadata_csv=metadata_csv,upstream=upstream,cohort=inventory['cohort'],
            origin_receipt=(inventory.get('acquisition') or {}).get('receipt'))
        if reference._logical_inventory(actual)!=reference._logical_inventory(inventory):
            raise ValueError('compact inventory differs from independently inspected actual source')
    except (KeyError,TypeError) as error:raise ValueError('invalid compact inventory') from error
    metadata=storage._sidecar(actual['sessions'],actual['blocks']);arrays=reference._arrays(metadata)
    identity=_identity(actual,arrays);code=_sources()
    storage._space(output.parent,sum(v.nbytes for v in metadata.values())+len(json.dumps(actual).encode())+4*1024**2)
    stage.mkdir(mode=0o700)
    try:
        storage._write_json(stage/'source-inventory.json',actual)
        np.savez(stage/'metadata.npz',**metadata)
        with np.load(stage/'metadata.npz',allow_pickle=False) as opened:
            if set(opened.files)!=set(metadata) or any(opened[k].dtype!=v.dtype or opened[k].tobytes()!=v.tobytes() for k,v in metadata.items()):
                raise ValueError('serialized compact native metadata differs')
        with (stage/'metadata.npz').open('rb') as stream:
            import os
            os.fsync(stream.fileno())
        manifest=_manifest(actual,arrays,storage._sha256(stage/'source-inventory.json'),storage._sha256(stage/'metadata.npz'))
        storage._write_json(stage/'manifest.json',manifest)
        completion=dict(format=FORMAT,complete=True,manifest_sha256=storage._sha256(stage/'manifest.json'),
            source_inventory_sha256=manifest['source_inventory_sha256'],metadata_sha256=manifest['metadata_sha256'],
            producer_files_sha256=code,data_identity=identity)
        storage._space(stage);_current();reader._current()
        if code!=_sources() or storage._sha256(metadata_csv)!=actual['metadata_csv']['sha256']:
            raise ValueError('compact source code or metadata changed during transaction')
        storage._write_json(stage/'completion.json',completion)
        receipt=dict(completion_sha256=storage._sha256(stage/'completion.json'),data_identity=deepcopy(identity))
        storage._publish_directory(stage,output);return receipt
    except BaseException as error:
        if stage.exists():
            (stage/'completion.json').unlink(missing_ok=True)
            storage._write_json(stage/'status.json',dict(complete=False,error_type=type(error).__name__,
                error=str(error),policy='owned incomplete compact index retained; source untouched'))
        raise


def _load_index(index,expected,upstream,metadata_csv):
    _current()
    if type(expected) is not str or not _HASH.fullmatch(expected):raise ValueError('separately frozen compact completion SHA required')
    index=storage._safe_path(index);metadata_csv=storage._safe_path(metadata_csv,file=True)
    if index.name.endswith('.incomplete') or set(p.name for p in index.iterdir())!=set(_FILES):
        raise ValueError('exact completed compact index required')
    upstream,official,_=storage._official(upstream)
    paths=[index/name for name in _FILES]+[metadata_csv]
    paths+=[Path(__file__).parent/name for name in _CODE]+[upstream/name for name in storage.SOURCE_HASHES]
    stamps={p:_stamp(p) for p in paths}
    if storage._sha256(index/'completion.json')!=expected:raise ValueError('frozen compact completion differs')
    completion=storage._json(index/'completion.json');manifest=storage._json(index/'manifest.json')
    inventory=storage._json(index/'source-inventory.json')
    if (completion.get('format')!=FORMAT or completion.get('complete') is not True or
            manifest.get('format')!=DATASET or inventory.get('format')!=data.FORMAT or inventory.get('complete') is not True):
        raise ValueError('compact index formats or completion differ')
    for name,key in [('manifest.json','manifest_sha256'),('metadata.npz','metadata_sha256'),('source-inventory.json','source_inventory_sha256')]:
        if completion.get(key)!=storage._sha256(index/name):raise ValueError('compact transaction bytes differ')
    if completion.get('producer_files_sha256')!=_sources():raise ValueError('compact index producer differs')
    if (inventory['upstream']['revision']!=official['revision'] or inventory['upstream']['files_sha256']!=official['files_sha256'] or
            inventory.get('reader_sha256')!=storage._sha256(Path(data.__file__)) or
            inventory.get('reader_sources_sha256')!=data._sources() or
            inventory.get('inspection_core_sha256')!=storage._sha256(Path(data.original.__file__)) or
            inventory.get('storage_helpers_sha256')!=storage._sha256(Path(storage.__file__)) or
            inventory.get('scientific_complete') is not False or inventory.get('full_paper_complete') is not False):
        raise ValueError('compact inventory source/helper/completion identity differs')
    cohort=inventory['cohort']
    if cohort not in {'engineering','mini','full'}:raise ValueError('unknown compact source cohort')
    for name,value in [('full_corpus_complete',cohort=='full'),('engineering_only',cohort!='full'),('inspected_file_set_complete',cohort!='engineering')]:
        if inventory.get(name) is not value:raise ValueError('compact source cohort claims disagree')
    csv_hash=storage._sha256(metadata_csv);rows=storage._csv_rows(metadata_csv)
    if csv_hash!=inventory['metadata_csv']['sha256'] or len(rows)!=inventory['metadata_csv']['filename_count']:
        raise ValueError('compact metadata CSV differs')
    if cohort!='engineering' and (csv_hash!=storage.CSV_SHA256 or len(rows)!=storage.FULL_FILE_COUNT):
        raise ValueError('official compact cohort requires pinned full metadata')
    from .emg2pose_runtime import _source_records
    sessions,total_rows,names=_source_records(inventory,rows)
    if (total_rows!=inventory['total_rows'] or (cohort=='full' and names!=set(rows)) or
            (cohort=='mini' and len(sessions)!=30)):
        raise ValueError('compact original source population or row count differs')
    source_index=inventory['source_index'];packet=inventory['compact_receipt']
    archive._receipt(packet,packet['sha256'])
    if source_index.get('format')!=data.INDEX_FORMAT:raise ValueError('logical compact TAR index required')
    origin=data._origin(cohort,source_index['archive_bytes'],source_index['archive_sha256'],
                        (inventory.get('acquisition') or {}).get('receipt'),compact_receipt=packet)
    if origin!=inventory.get('acquisition'):raise ValueError('compact source origin differs')
    shapes=dict(emg=[total_rows,16],joint_angles=[total_rows,20],time=[total_rows],valid_mask=[total_rows])
    if set(inventory['fields'])!=set(shapes):raise ValueError('compact source requires exactly four original fields')
    for key,shape in shapes.items():
        spec=inventory['fields'][key]
        if (spec['shape']!=shape or np.dtype(spec['dtype'])!=storage.FIELD_DTYPES[key] or
                type(spec['sha256']) is not str or not _HASH.fullmatch(spec['sha256'])):
            raise ValueError('compact field schema differs from original16ch/20D')
    counts={key:dict(Counter(s[key] for s in sessions)) for key in ('split','generalization','side')}
    if inventory['counts']!=counts:raise ValueError('compact source metadata counts differ')
    if cohort=='full' and counts['split']!={'train':17136,'val':1950,'test':6167}:
        raise ValueError('full compact source requires all25253 original roles')
    previous=0
    for block in inventory['blocks']:
        if type(block) is not list or len(block)!=3 or any(type(x) is not int for x in block):raise ValueError('invalid compact source block')
        si,a,b=block
        if si<0 or si>=len(sessions) or a<previous or b<=a or a<sessions[si]['start_idx'] or b>sessions[si]['end_idx']:
            raise ValueError('compact source blocks overlap or cross sessions')
        previous=b
    if sum(b-a for _,a,b in inventory['blocks'])!=inventory['valid_rows']:raise ValueError('compact raw valid count differs')
    expected_arrays=storage._sidecar(sessions,inventory['blocks'])
    with np.load(index/'metadata.npz',allow_pickle=False) as opened:
        if set(opened.files)!=set(expected_arrays) or any(opened[k].dtype!=v.dtype or opened[k].tobytes()!=v.tobytes() for k,v in expected_arrays.items()):
            raise ValueError('compact native metadata arrays differ')
    arrays=reference._arrays(expected_arrays);identity=_identity(inventory,arrays)
    if manifest!=_manifest(inventory,arrays,completion['source_inventory_sha256'],completion['metadata_sha256']) or completion['data_identity']!=identity:
        raise ValueError('compact native manifest or content identity differs')
    if any(_stamp(p)!=stamp for p,stamp in stamps.items()):raise ValueError('compact index changed during admission')
    return index,manifest,identity,stamps


class VerifiedCompactSource:
    """Own the two immutable source objects and independently frozen metadata."""
    def __init__(self,reader,index,manifest,identity,stamps,*,_token):
        self._reader=None
        if _token is not _TOKEN:raise TypeError('use open_source for actual compact admission')
        self.index=index;self._manifest=deepcopy(manifest);self._identity=deepcopy(identity);self._stamps=stamps
        def binding(fd):
            device,inode,size=archive._stamp(fd)
            return dict(device=device,inode=inode,size=size,seals=archive._SEALS)
        self._handle=dict(format='emg2pose_verified_compact_handle_v1',compact_receipt=reader.receipt,
            payload=binding(reader.payload_fileno()),block_index=binding(reader.index_fileno()),
            index_completion_sha256=storage._sha256(index/'completion.json'),
            binding_stamps={str(p):list(v) for p,v in stamps.items()})
        self._reader=reader
    def assert_current(self):
        if self._reader is None:raise ValueError('verified compact source is closed')
        self._reader._current()
        if any(_stamp(p)!=v for p,v in self._stamps.items()):raise ValueError('verified compact index binding changed')
    def payload_fileno(self):self.assert_current();return self._reader.payload_fileno()
    def block_index_fileno(self):self.assert_current();return self._reader.index_fileno()
    def pread(self,size,offset):self.assert_current();return self._reader.pread(size,offset)
    def slice(self,offset,length):self.assert_current();return self._reader.slice(offset,length)
    @property
    def manifest(self):self.assert_current();return deepcopy(self._manifest)
    @property
    def data_identity(self):self.assert_current();return deepcopy(self._identity)
    @property
    def handle_identity(self):self.assert_current();return deepcopy(self._handle)
    def close(self):
        if self._reader is not None:
            reader,self._reader=self._reader,None;reader.close()
    def __enter__(self):self.assert_current();return self
    def __exit__(self,*_args):self.close()
    def __del__(self):
        if getattr(self,'_reader',None) is not None:self.close()


def open_source(reader,index,*,expected_completion_sha256,upstream,metadata_csv):
    """Bind an already deep-verified exact-class compact reader to native sidecars."""
    data._reader(reader);clone=None
    try:
        index,manifest,identity,stamps=_load_index(index,expected_completion_sha256,upstream,metadata_csv)
        receipt=reader.receipt
        if (receipt['raw_bytes'],receipt['raw_sha256'])!=(manifest['archive_bytes'],manifest['archive_sha256']):
            raise ValueError('actual complete compact bytes differ from native index')
        clone=reader.clone();source=VerifiedCompactSource(clone,index,manifest,identity,stamps,_token=_TOKEN);clone=None
        try:source.assert_current();return source
        except BaseException:source.close();raise
    except (OSError,KeyError,TypeError,EOFError) as error:
        raise ValueError('invalid completed compact index or source') from error
    finally:
        if clone is not None:clone.close()
