"""Sealed TAR/HDF5 bytes and native converter semantics, without extraction."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1]/'upstream'))


def module():
    return importlib.import_module('egoemg_repro.emg2pose_memory_data')


@contextmanager
def sealed(payload, *, seal=True):
    from egoemg_repro.emg2pose_memory_archive import _create_memfd
    descriptor = _create_memfd()
    try:
        for start in range(0, len(payload), 8192):
            part = payload[start:start+8192]
            assert os.pwrite(descriptor, part, start) == len(part)
        if seal:
            fcntl.fcntl(descriptor, 1033, 15)
        yield descriptor
    finally:
        os.close(descriptor)


def make_tar(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=tarfile.GNU_FORMAT) as archive:
        for name, kind, payload in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(payload) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = 'outside.hdf5'
            archive.addfile(member, io.BytesIO(payload) if member.size else None)
    return stream.getvalue()


def index(fd, payload):
    return module().index_archive(fd, expected_bytes=len(payload),
                                  expected_sha256=hashlib.sha256(payload).hexdigest())


def test_independent_cursors_and_slice_eof_never_read_neighbour_bytes():
    with sealed(b'BEFOREabcdeAFTER') as fd:
        with module().FdSlice(fd, 6, 5) as first, module().FdSlice(fd, 6, 5) as second:
            assert first.read(2) == b'ab'
            os.lseek(fd, 999, os.SEEK_SET)
            assert second.read() == b'abcde'
            assert first.tell() == 2 and first.read(99) == b'cde'
            assert first.read(1) == b''
            assert first.seek(-2, os.SEEK_END) == 3
            assert first.read() == b'de'
            first.seek(0)
            buffer = bytearray(8)
            assert first.readinto(buffer) == 5
            assert buffer == b'abcde\x00\x00\x00'
            assert first.readinto(buffer) == 0
        assert os.pread(fd, 16, 0) == b'BEFOREabcdeAFTER'


def test_reader_close_only_closes_its_owned_duplicate():
    with sealed(b'original') as fd:
        reader = module().FdSlice(fd, 0, 8)
        duplicate = reader.fileno()
        assert reader.readable() and reader.seekable() and not reader.writable()
        reader.close()
        reader.close()
        assert reader.closed
        with pytest.raises(OSError):
            os.fstat(duplicate)
        with pytest.raises(ValueError):
            reader.read(1)
        assert os.pread(fd, 8, 0) == b'original'


@pytest.mark.parametrize('offset,length', [(-1, 1), (0, -1), (4, 9), (True, 1), (0, 1.5)])
def test_invalid_slice_extent_is_rejected(offset, length):
    with sealed(b'12345') as fd:
        with pytest.raises(ValueError):
            module().FdSlice(fd, offset, length)


def test_unsealed_source_is_never_exposed_as_immutable_reader():
    with sealed(b'12345', seal=False) as fd:
        with pytest.raises(ValueError, match='seal|immutable'):
            module().FdSlice(fd, 0, 5)


def test_negative_seek_and_write_are_rejected_without_moving_cursor():
    with sealed(b'12345') as fd, module().FdSlice(fd, 0, 5) as reader:
        reader.seek(2)
        with pytest.raises((ValueError, OSError)):
            reader.seek(-1)
        assert reader.tell() == 2
        with pytest.raises((ValueError, OSError)):
            reader.write(b'bad')
        assert reader.read() == b'345'


def test_index_preserves_original_padding_and_accounts_for_every_member():
    raw = bytearray(make_tar([('corpus', tarfile.DIRTYPE, b''),
                              ('corpus/a.hdf5', tarfile.REGTYPE, b'raw\x00\xff'),
                              ('corpus/metadata.csv', tarfile.REGTYPE, b'col\nvalue\n')]))
    raw[1029:1536] = b'P' * 507
    raw = bytes(raw)
    with sealed(raw) as fd:
        result = index(fd, raw)
        assert result['format'] == 'emg2pose_sealed_tar_index_v1'
        assert result['archive_bytes'] == len(raw)
        assert result['archive_sha256'] == hashlib.sha256(raw).hexdigest()
        assert result['trailer_bytes'] == len(raw)-2560
        assert [x['name'] for x in result['members']] == ['corpus', 'corpus/a.hdf5', 'corpus/metadata.csv']
        entry = result['members'][1]
        assert entry['offset_data'] == 1024 and entry['bytes'] == 5
        assert entry['sha256'] == hashlib.sha256(b'raw\x00\xff').hexdigest()
        assert os.pread(fd, len(raw), 0) == raw


@pytest.mark.parametrize('name', ['/outside', '../outside', 'corpus/../a', 'corpus/./a', 'corpus//a', 'corpus\\a'])
def test_unsafe_or_ambiguous_tar_paths_are_rejected(name):
    raw = make_tar([(name, tarfile.REGTYPE, b'data')])
    with sealed(raw) as fd, pytest.raises(ValueError):
        index(fd, raw)


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                                 tarfile.FIFOTYPE, tarfile.GNUTYPE_SPARSE])
def test_links_devices_and_sparse_members_are_rejected(kind):
    raw = make_tar([('corpus/a', kind, b'')])
    with sealed(raw) as fd, pytest.raises(ValueError):
        index(fd, raw)


def test_duplicate_tar_name_cannot_hide_another_file():
    raw = make_tar([('corpus/a', tarfile.REGTYPE, b'one'), ('corpus/a', tarfile.REGTYPE, b'two')])
    with sealed(raw) as fd, pytest.raises(ValueError, match='duplicate'):
        index(fd, raw)


@pytest.mark.parametrize('format', [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_long_utf8_member_names_bind_the_whole_extension_header(format):
    name = 'corpus/'+'session_'*25+'µ.hdf5'
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=format) as archive:
        member = tarfile.TarInfo(name)
        member.size = 7
        archive.addfile(member, io.BytesIO(b'payload'))
    raw = stream.getvalue()
    with sealed(raw) as fd:
        result = index(fd, raw)
        member = result['members'][0]
        assert member['name'] == name and member['offset_data'] >= 1536
        assert member['headers_sha256'] == hashlib.sha256(raw[:member['offset_data']]).hexdigest()
        with module().FdSlice(fd, member['offset_data'], member['bytes']) as source:
            assert source.read() == b'payload'


@pytest.mark.parametrize('fault', ['appended_archive', 'one_zero_block', 'truncated_body', 'wrong_hash', 'wrong_size'])
def test_incomplete_or_appended_archives_and_identity_mismatches_are_rejected(fault):
    raw = make_tar([('a', tarfile.REGTYPE, b'payload')])
    if fault == 'appended_archive':
        raw += make_tar([('hidden', tarfile.REGTYPE, b'other')])
    elif fault == 'one_zero_block':
        raw = raw[:1536]
    elif fault == 'truncated_body':
        raw = raw[:518]
    with sealed(raw) as fd, pytest.raises(ValueError):
        module().index_archive(fd, expected_bytes=len(raw)+(fault == 'wrong_size'),
                               expected_sha256='0'*64 if fault == 'wrong_hash' else hashlib.sha256(raw).hexdigest())


def corpus_fixture(tmp_path, monkeypatch, *, explicit_ik=False):
    pytest.importorskip('h5py')
    pytest.importorskip('numpy')
    from test_emg2pose_data import fixture
    from egoemg_repro import emg2pose_data as reference
    paths, metadata, arrays = fixture(tmp_path, explicit_ik=explicit_ik)
    monkeypatch.setattr(reference.shutil, 'disk_usage', lambda _: SimpleNamespace(free=64*1024**3))
    inventory = reference.inspect_corpus(paths, metadata, cohort='engineering', upstream=UPSTREAM)
    receipt = tmp_path/'old-receipt.json'
    receipt.write_text(json.dumps(inventory, allow_nan=False))
    converted = reference.convert_corpus(receipt, tmp_path/'old-memmap', upstream=UPSTREAM, chunk_rows=3)
    raw = make_tar([('corpus', tarfile.DIRTYPE, b'')] + [
        ('corpus/'+p.name, tarfile.REGTYPE, p.read_bytes()) for p in reversed(paths)] + [
        ('corpus/metadata.csv', tarfile.REGTYPE, metadata.read_bytes())])
    return paths, metadata, arrays, inventory, converted, raw


def inspect(fd, raw, metadata, **kwargs):
    return module().inspect_archive(fd, expected_bytes=len(raw), expected_sha256=hashlib.sha256(raw).hexdigest(),
                                    metadata_csv=metadata, upstream=UPSTREAM, chunk_rows=3, **kwargs)


def test_real_hdf5_callback_reads_only_its_nonzero_offset_member(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    np = pytest.importorskip('numpy')
    paths, metadata, arrays, old, converted, raw = corpus_fixture(tmp_path, monkeypatch)
    with sealed(raw) as fd:
        entry = next(x for x in index(fd, raw)['members'] if x['name'].endswith(paths[0].name))
        with module().FdSlice(fd, entry['offset_data'], entry['bytes']) as source:
            with h5py.File(source, 'r') as f:
                assert f['emg2pose/timeseries'][:].tobytes() == arrays[paths[0].stem].tobytes()
                assert f['emg2pose/preserved_extra'][:].tolist() == [7, 9]
                typed = f['emg2pose'].attrs['typed_numeric']
                assert typed.dtype == np.dtype('<i2') and typed.tolist() == [1, 2]
                assert f['emg2pose'].attrs['user'] == 'private-user-中文'
        assert os.pread(fd, 6, 0) == b'corpus'


@pytest.mark.parametrize('explicit_ik', [False, True])
def test_memory_inventory_matches_original_converter_fields_blocks_attrs_and_order(tmp_path, monkeypatch, explicit_ik):
    np = pytest.importorskip('numpy')
    paths, metadata, arrays, old, converted, raw = corpus_fixture(tmp_path, monkeypatch, explicit_ik=explicit_ik)
    with sealed(raw) as fd:
        result = inspect(fd, raw, metadata, cohort='engineering')
        assert result['format'] == 'emg2pose_sealed_tar_inventory_v1'
        assert result['complete'] is True and result['full_corpus_complete'] is False
        assert result['engineering_only'] is True and result['total_rows'] == 32
        assert result['counts'] == old['counts']
        assert [s['filename'] for s in result['sessions']] == ['b-train', 'a-val', 'c-stage', 'd-both']
        for actual, expected in zip(result['sessions'], old['sessions']):
            for key in ['filename', 'session_id', 'rows', 'bytes', 'sha256', 'split', 'generalization',
                        'start_idx', 'end_idx', 'explicit_ik_mask', 'hdf5_objects']:
                assert actual[key] == expected[key], key
        for name in ['emg', 'joint_angles', 'time', 'valid_mask']:
            assert result['fields'][name]['sha256'] == converted['fields'][name]['sha256']
            assert result['fields'][name]['shape'] == converted['fields'][name]['shape']
        assert result['session_field_hashes'] == converted['session_field_hashes']
        with np.load(tmp_path/'old-memmap/metadata.npz', allow_pickle=False) as md:
            expected_blocks = [list(x) for x in zip(md['blocks_session_idx'].tolist(), md['blocks_start'].tolist(), md['blocks_end'].tolist())]
        assert result['blocks'] == expected_blocks
        assert json.loads(json.dumps(result, allow_nan=False)) == result
        assert os.pread(fd, len(raw), 0) == raw


@pytest.mark.parametrize('fault', ['wrong_attrs', 'external_link', 'missing_h5', 'extra_h5', 'duplicate_basename'])
def test_invalid_physical_corpus_never_produces_complete_inventory(tmp_path, monkeypatch, fault):
    h5py = pytest.importorskip('h5py')
    paths, metadata, arrays, old, converted, raw = corpus_fixture(tmp_path, monkeypatch)
    if fault in ('wrong_attrs', 'external_link'):
        with h5py.File(paths[0], 'r+') as f:
            if fault == 'wrong_attrs':
                f['emg2pose'].attrs['split'] = 'train'
            else:
                f['emg2pose']['outside'] = h5py.ExternalLink(str(paths[1]), '/emg2pose')
    entries = [('corpus/'+p.name, tarfile.REGTYPE, p.read_bytes()) for p in paths]
    if fault == 'missing_h5':
        entries = []
    elif fault == 'extra_h5':
        entries.append(('corpus/unknown.hdf5', tarfile.REGTYPE, paths[0].read_bytes()))
    elif fault == 'duplicate_basename':
        entries.append(('another/'+paths[0].name, tarfile.REGTYPE, paths[0].read_bytes()))
    raw = make_tar(entries+[('corpus/metadata.csv', tarfile.REGTYPE, metadata.read_bytes())])
    with sealed(raw) as fd, pytest.raises(ValueError):
        inspect(fd, raw, metadata, cohort='engineering')


@pytest.mark.parametrize('cohort', ['mini', 'full'])
def test_arbitrary_fixture_cannot_claim_official_cohort(tmp_path, monkeypatch, cohort):
    paths, metadata, arrays, old, converted, raw = corpus_fixture(tmp_path, monkeypatch)
    with sealed(raw) as fd, pytest.raises(ValueError, match='official|metadata|acquisition'):
        inspect(fd, raw, metadata, cohort=cohort)


@pytest.mark.parametrize('fault', ['healthy', 'incomplete', 'wrong_url', 'wrong_bytes', 'wrong_sha', 'missing_file'])
def test_full_admission_binds_complete_acquisition_and_exact_metadata_population(tmp_path, monkeypatch, fault):
    # Replace only external asset identities for a small real-HDF5 gate test;
    # this is never an actual official full-corpus receipt or performance result.
    paths, metadata, arrays, old, converted, raw = corpus_fixture(tmp_path, monkeypatch)
    from egoemg_repro import emg2pose_data as storage, emg2pose_archive as transport
    monkeypatch.setattr(storage, 'CSV_SHA256', hashlib.sha256(metadata.read_bytes()).hexdigest())
    monkeypatch.setattr(storage, 'FULL_FILE_COUNT', 4)
    if fault == 'missing_file':
        raw = make_tar([('corpus/'+p.name, tarfile.REGTYPE, p.read_bytes()) for p in paths[:-1]])
    url = 'https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar'
    monkeypatch.setitem(transport._ASSETS, 'full', (url, len(raw)))
    receipt = {'format': 'emg2pose_sealed_memory_archive_v1', 'complete': True,
               'full_corpus_complete': False, 'source_bytes_received': len(raw), 'seals': 15,
               'full_sha256': hashlib.sha256(raw).hexdigest(),
               'source': {'url': url, 'expected_bytes': len(raw), 'etag': '"fixture-stable"',
                          'last_modified': 'Thu, 01 Jan 2026 00:00:00 GMT'}}
    if fault == 'incomplete':
        receipt['complete'] = False
    elif fault == 'wrong_url':
        receipt['source']['url'] = 'https://example.invalid/another.tar'
    elif fault == 'wrong_bytes':
        receipt['source_bytes_received'] -= 512
    elif fault == 'wrong_sha':
        receipt['full_sha256'] = '0'*64
    with sealed(raw) as fd:
        if fault == 'healthy':
            result = inspect(fd, raw, metadata, cohort='full', acquisition_receipt=receipt)
            assert result['full_corpus_complete'] is True and result['engineering_only'] is False
        else:
            with pytest.raises(ValueError):
                inspect(fd, raw, metadata, cohort='full', acquisition_receipt=receipt)
