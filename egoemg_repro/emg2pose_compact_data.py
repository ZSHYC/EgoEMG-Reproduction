"""Independent compact-source TAR/HDF inventory using unchanged inspection code.

Only source access is privately rebound. No raw-fd validator, original helper,
dataset algorithm or imported module global is changed.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import tarfile
from types import FunctionType

from . import emg2pose_data as storage,emg2pose_memory_data as original
from . import emg2pose_compact_archive as compact,emg2pose_archive as transport


FORMAT='emg2pose_compact_tar_inventory_v1'
INDEX_FORMAT='emg2pose_compact_tar_index_v1'
HTTP_ORIGIN='emg2pose_compact_http_origin_v1'
_CODE=('__init__.py','download.py','emg2pose_archive.py','emg2pose_data.py',
       'emg2pose_memory_archive.py','emg2pose_memory_data.py','emg2pose_compact_codec.py',
       'emg2pose_compact_archive.py','emg2pose_compact_data.py')


def _sources():return {name:storage._sha256(Path(__file__).parent/name) for name in _CODE}
_IMPORTED=_sources()


def _current():
    if _sources()!=_IMPORTED:raise ValueError('imported compact inspection source changed')


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _reader(reader):
    if type(reader) is not compact.CompactReader:raise ValueError('actual complete compact reader required')
    reader._current();return reader.receipt


def _range_sha(reader,offset,length):
    with reader.slice(offset,length) as view:return original._digest(view)


def index_archive(reader,*,expected_bytes=None,expected_sha256=None):
    """Account for every original byte/member; never expose a raw descriptor."""
    receipt=_reader(reader);_current();expected=receipt['raw_bytes']
    if ((expected_bytes is not None and expected_bytes!=expected) or
            (expected_sha256 is not None and expected_sha256!=receipt['raw_sha256'])):
        raise ValueError('logical compact TAR differs from separately expected bytes/hash')
    if expected<1024 or expected%512:raise ValueError('complete aligned original TAR bytes required')
    members=[];names=set();end=0
    with reader.slice(0,expected) as source:
        if original._digest(source)!=receipt['raw_sha256']:raise ValueError('complete original compact TAR SHA256 differs')
        source.seek(0)
        try:
            with tarfile.open(fileobj=source,mode='r:') as archive:
                for member in archive:
                    name=original._member_name(member)
                    if name in names:raise ValueError('duplicate compact TAR member name')
                    names.add(name)
                    if (member.sparse is not None or member.type==tarfile.GNUTYPE_SPARSE or
                            any(k.startswith('GNU.sparse') for k in member.pax_headers)):
                        raise ValueError('sparse compact TAR members are unsupported')
                    if not (member.isdir() or member.isreg()) or (member.isdir() and member.size):
                        raise ValueError('unsupported compact TAR member type')
                    if (member.offset<end or member.offset_data<member.offset+512 or
                            member.offset_data%512 or member.size<0):
                        raise ValueError('invalid or overlapping compact TAR extent')
                    end=member.offset_data+((member.size+511)//512)*512
                    if end+1024>expected:raise ValueError('truncated compact TAR member or trailer')
                    members.append(dict(name=name,kind='directory' if member.isdir() else 'file',
                        offset=member.offset,offset_data=member.offset_data,bytes=member.size,
                        pax_headers=dict(member.pax_headers),sha256=_range_sha(reader,member.offset_data,member.size),
                        headers_sha256=_range_sha(reader,member.offset,member.offset_data-member.offset)))
        except (tarfile.TarError,EOFError,OSError) as error:
            raise ValueError('invalid or truncated original compact TAR') from error
        if expected-end<1024:raise ValueError('missing complete compact TAR trailer')
        source.seek(end)
        for body in iter(lambda:source.read(8*1024**2),b''):
            if body.strip(b'\0'):raise ValueError('nonzero or appended compact TAR trailer data')
    _reader(reader);_current()
    return dict(format=INDEX_FORMAT,archive_bytes=expected,archive_sha256=receipt['raw_sha256'],
                trailer_bytes=expected-end,members=members)


def _origin(cohort,expected_bytes,expected_sha256,receipt,*,compact_receipt):
    if (compact_receipt['raw_bytes']!=expected_bytes or compact_receipt['raw_sha256']!=expected_sha256):
        raise ValueError('compact logical bytes differ from origin')
    if cohort=='engineering':return None
    if type(receipt) is not dict:raise ValueError('complete official source origin required')
    receipt=deepcopy(receipt)
    if receipt.get('format')=='emg2pose_sealed_memory_archive_v1':
        # This describes the actual ORIGINAL acquisition before packing. It is
        # not a claim that either compact descriptor is a raw archive fd.
        witness=original._acquisition(cohort,expected_bytes,expected_sha256,receipt)
        return dict(format='emg2pose_compact_source_origin_v1',kind='original_raw_acquisition',**witness)
    fields={'format','source','final_source','source_bytes_received','full_sha256',
            'compact_receipt_sha256','complete','full_corpus_complete'}
    source=receipt.get('source');url,count=transport._ASSETS['mini' if cohort=='mini' else 'full']
    if (set(receipt)!=fields or receipt.get('format')!=HTTP_ORIGIN or type(source) is not dict or
            set(source)!={'url','expected_bytes','etag','last_modified'} or source!=receipt.get('final_source') or
            source['url']!=url or type(source['expected_bytes']) is not int or source['expected_bytes']!=count or
            type(source['etag']) is not str or not re.fullmatch(r'"[^"\r\n]+"',source['etag']) or
            type(source['last_modified']) is not str or not source['last_modified'].strip() or
            type(receipt['source_bytes_received']) is not int or receipt['source_bytes_received']!=count or
            expected_bytes!=count or receipt['full_sha256']!=expected_sha256 or
            receipt['compact_receipt_sha256']!=compact_receipt['sha256'] or receipt['complete'] is not True or
            receipt['full_corpus_complete'] is not False):
        raise ValueError('official compact HTTP origin is incomplete or differs')
    if cohort=='mini' and (expected_bytes!=storage.MINI_BYTES or expected_sha256!=storage.MINI_SHA256):
        raise ValueError('original mini origin differs from known byte identity')
    return dict(format='emg2pose_compact_source_origin_v1',kind='compact_http_acquisition',
                receipt_sha256=_hash(receipt),receipt=receipt)


def inspect_compact(reader,*,expected_receipt_sha256,metadata_csv,upstream,cohort='engineering',
                    origin_receipt=None,chunk_rows=262144):
    """Reuse the original exact inventory code with logical compact slices."""
    receipt=_reader(reader);compact._receipt(receipt,expected_receipt_sha256);_current()
    bound=FunctionType(original.inspect_archive.__code__,dict(original.inspect_archive.__globals__,
        FdSlice=lambda source,offset,length:source.slice(offset,length),index_archive=index_archive,
        _acquisition=partial(_origin,compact_receipt=receipt)),
        name='inspect_compact_core',argdefs=original.inspect_archive.__defaults__,
        closure=original.inspect_archive.__closure__)
    assert bound.__code__ is original.inspect_archive.__code__
    value=bound(reader,expected_bytes=receipt['raw_bytes'],expected_sha256=receipt['raw_sha256'],
        metadata_csv=metadata_csv,upstream=upstream,cohort=cohort,
        acquisition_receipt=origin_receipt,chunk_rows=chunk_rows)
    value.update(format=FORMAT,compact_receipt=deepcopy(receipt),
        inspection_core_sha256=storage._sha256(Path(original.__file__)),
        reader_sha256=storage._sha256(Path(__file__)),reader_sources_sha256=_sources(),
        source_retention='all original logical TAR/HDF bytes remain in separate sealed compact payload/index',
        scientific_complete=False,full_paper_complete=False)
    _reader(reader);_current();return value
