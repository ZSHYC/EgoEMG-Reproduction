"""Separate compact inventories preserve the actual original inspection logic."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import tarfile

import pytest

pytest.importorskip('h5py')
from test_emg2pose_memory_data import corpus_fixture,sealed,inspect,make_tar,UPSTREAM


def api():return importlib.import_module('egoemg_repro.emg2pose_compact_data')
def sha(raw):return hashlib.sha256(raw).hexdigest()


def packed(monkeypatch,raw):
    from egoemg_repro import emg2pose_compact_archive as archive
    monkeypatch.setattr(archive,'available_memory_bytes',lambda:2*1024**4)
    return archive.pack_stream([raw],expected_bytes=len(raw),expected_sha256=sha(raw),max_encoded_bytes=2*len(raw)+65536)


@pytest.fixture
def case(tmp_path,monkeypatch):
    paths,metadata,arrays,old,converted,raw=corpus_fixture(tmp_path,monkeypatch)
    with sealed(raw) as fd:original=inspect(fd,raw,metadata,cohort='engineering')
    with packed(monkeypatch,raw) as reader:
        yield dict(reader=reader,metadata=metadata,raw=raw,original=original,converted=converted,root=tmp_path)


def scan(case,**options):
    reader=options.pop('reader',case['reader'])
    return api().inspect_compact(reader,expected_receipt_sha256=options.pop('expected',case['reader'].receipt['sha256']),
        metadata_csv=case['metadata'],upstream=UPSTREAM,**options)


def test_all_original_fields_masks_sessions_and_blocks_match_without_raw_global_changes(case):
    from egoemg_repro import emg2pose_memory_data as original
    before=(original.FdSlice,original.index_archive,original._acquisition,original.inspect_archive.__code__)
    result=scan(case,chunk_rows=3)
    assert result['format']=='emg2pose_compact_tar_inventory_v1'
    for key in ('sessions','blocks','counts','fields','session_field_hashes','total_rows','valid_rows','metadata_csv','upstream'):
        assert result[key]==case['original'][key],key
    for key in ('archive_bytes','archive_sha256','members','trailer_bytes'):
        assert result['source_index'][key]==case['original']['source_index'][key]
    assert result['source_index']['format']=='emg2pose_compact_tar_index_v1'
    assert result['full_corpus_complete'] is False and result['engineering_only'] is True
    assert result['compact_receipt']==case['reader'].receipt
    assert result['inspection_core_sha256']==hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
    assert before==(original.FdSlice,original.index_archive,original._acquisition,original.inspect_archive.__code__)
    assert case['reader'].pread(6,0)==b'corpus'


def test_chunk_boundaries_do_not_change_ik_or_finite_mask_statistics(case):
    a=scan(case,chunk_rows=1);b=scan(case,chunk_rows=7)
    assert a['fields']==b['fields']==case['original']['fields']
    assert a['blocks']==b['blocks']==case['original']['blocks']


def test_explicit_ik_mask_and_finite_filter_preserve_all_original_bits(tmp_path,monkeypatch):
    _paths,metadata,_arrays,_old,_converted,raw=corpus_fixture(tmp_path,monkeypatch,explicit_ik=True)
    with sealed(raw) as fd:expected=inspect(fd,raw,metadata,cohort='engineering')
    with packed(monkeypatch,raw) as reader:
        actual=api().inspect_compact(reader,expected_receipt_sha256=reader.receipt['sha256'],
            metadata_csv=metadata,upstream=UPSTREAM,chunk_rows=3)
    assert actual['valid_rows']==expected['valid_rows']==16
    assert actual['fields']==expected['fields'] and actual['blocks']==expected['blocks']
    assert all(s['explicit_ik_mask'] for s in actual['sessions'])


def test_inventory_does_not_apply_the_native_all_false_window_fallback(tmp_path,monkeypatch):
    import h5py
    from test_emg2pose_data import fixture
    paths,metadata,_arrays=fixture(tmp_path,explicit_ik=True)
    for path in paths:
        with h5py.File(path,'r+') as file:file['emg2pose/ik_failure_mask'][:]=True
    raw=make_tar([('corpus/'+p.name,tarfile.REGTYPE,p.read_bytes()) for p in paths])
    with packed(monkeypatch,raw) as reader:
        actual=api().inspect_compact(reader,expected_receipt_sha256=reader.receipt['sha256'],
            metadata_csv=metadata,upstream=UPSTREAM,chunk_rows=3)
    assert actual['total_rows']==32 and actual['valid_rows']==0 and actual['blocks']==[]
    assert actual['fields']['valid_mask']['sha256']==sha(b'\0'*32)


@pytest.mark.parametrize('options',[{'expected':'0'*64},{'cohort':'full'},{'cohort':'mini'},
                                    {'cohort':'unknown'},{'chunk_rows':True},{'chunk_rows':0}])
def test_wrong_frozen_identity_unknown_scope_and_fake_official_cohort_are_refused(case,options):
    with pytest.raises(ValueError):scan(case,**options)


def test_raw_fd_or_duck_type_never_substitutes_for_admitted_compact_reader(case):
    from types import SimpleNamespace
    for reader in (case['reader'].payload_fileno(),SimpleNamespace(receipt=case['reader'].receipt)):
        with pytest.raises(ValueError):scan(case,reader=reader,expected=case['reader'].receipt['sha256'])


@pytest.mark.parametrize('name',['/outside','../outside','corpus/../x','corpus//x','corpus\\x'])
def test_unsafe_tar_names_are_refused(monkeypatch,name):
    raw=make_tar([(name,tarfile.REGTYPE,b'x')])
    with packed(monkeypatch,raw) as reader:
        with pytest.raises(ValueError):api().index_archive(reader)


@pytest.mark.parametrize('kind',[tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.CHRTYPE,tarfile.FIFOTYPE,tarfile.GNUTYPE_SPARSE])
def test_unsupported_tar_member_types_are_refused(monkeypatch,kind):
    raw=make_tar([('corpus/x',kind,b'')])
    with packed(monkeypatch,raw) as reader:
        with pytest.raises(ValueError):api().index_archive(reader)


@pytest.mark.parametrize('kind',['duplicate','hidden_archive','nonzero_trailer','short'])
def test_ambiguous_truncated_or_unaccounted_tar_bytes_are_refused(monkeypatch,kind):
    if kind=='duplicate':raw=make_tar([('corpus/x',tarfile.REGTYPE,b'a'),('corpus/x',tarfile.REGTYPE,b'b')])
    elif kind=='hidden_archive':raw=make_tar([('corpus/a',tarfile.REGTYPE,b'a')])+make_tar([('corpus/b',tarfile.REGTYPE,b'b')])
    elif kind=='nonzero_trailer':raw=make_tar([('corpus/x',tarfile.REGTYPE,b'a')])[:-1]+b'x'
    else:raw=b'\0'*512
    with packed(monkeypatch,raw) as reader:
        with pytest.raises(ValueError):api().index_archive(reader)


def test_original_member_padding_is_preserved_without_becoming_a_hidden_member(monkeypatch):
    raw=bytearray(make_tar([('corpus/x',tarfile.REGTYPE,b'x')]))
    raw[513:1024]=b'P'*511;raw=bytes(raw)
    with packed(monkeypatch,raw) as reader:
        index=api().index_archive(reader)
        assert index['archive_sha256']==sha(raw)
        assert index['members'][0]['bytes']==1 and index['members'][0]['sha256']==sha(b'x')
        assert reader.pread(len(raw),0)==raw


def http_origin():
    from egoemg_repro import emg2pose_data as storage,emg2pose_archive as transport
    source=dict(url=transport._ASSETS['mini'][0],expected_bytes=storage.MINI_BYTES,
                etag='"independent-schema-test"',last_modified='Wed, 12 Jun 2024 14:31:55 GMT')
    packet=dict(raw_bytes=storage.MINI_BYTES,raw_sha256=storage.MINI_SHA256,sha256='a'*64)
    origin=dict(format='emg2pose_compact_http_origin_v1',source=source,final_source=deepcopy(source),
        source_bytes_received=storage.MINI_BYTES,full_sha256=storage.MINI_SHA256,
        compact_receipt_sha256=packet['sha256'],complete=True,full_corpus_complete=False)
    return packet,origin


def test_transport_witness_schema_has_no_independent_dataset_qualification_authority():
    packet,origin=http_origin()
    result=api()._origin('mini',packet['raw_bytes'],packet['raw_sha256'],origin,compact_receipt=packet)
    assert result['kind']=='compact_http_acquisition' and result['receipt']==origin
    assert result['receipt_sha256']==sha(json.dumps(origin,sort_keys=True,separators=(',',':')).encode())
    # This exercises only a synthetic transport schema. Actual admission above
    # still requires a real deep-verified reader, exact bytes and all HDF files.
    assert result['receipt']['full_corpus_complete'] is False


@pytest.mark.parametrize('damage',['url','weak_etag','changed_final','short','wrong_raw_hash','wrong_packet',
                                    'not_complete','fake_full','extra','wrong_format','wrong_logical_length'])
def test_origin_must_bind_original_object_both_heads_and_actual_compact_packet(damage):
    packet,origin=http_origin()
    if damage=='url':origin['source']['url']='https://example.invalid/unrelated'
    elif damage=='weak_etag':origin['source']['etag']='W/"weak"';origin['final_source']=deepcopy(origin['source'])
    elif damage=='changed_final':origin['final_source']['etag']='"changed"'
    elif damage=='short':origin['source_bytes_received']-=1
    elif damage=='wrong_raw_hash':origin['full_sha256']='0'*64
    elif damage=='wrong_packet':origin['compact_receipt_sha256']='0'*64
    elif damage=='not_complete':origin['complete']=False
    elif damage=='fake_full':origin['full_corpus_complete']=True
    elif damage=='extra':origin['unexpected']='extra'
    elif damage=='wrong_format':origin['format']='unknown'
    else:packet['raw_bytes']+=1
    from egoemg_repro import emg2pose_data as storage
    with pytest.raises(ValueError):
        api()._origin('mini',storage.MINI_BYTES,storage.MINI_SHA256,origin,compact_receipt=packet)
