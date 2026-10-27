"""Small index transactions and explicitly owned, verified sealed sources."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('h5py')
UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1]/'upstream'))


def api():
    return importlib.import_module('egoemg_repro.emg2pose_memory_index')


@pytest.fixture
def case(tmp_path, monkeypatch):
    from test_emg2pose_memory_data import corpus_fixture, sealed, inspect
    paths, metadata, arrays, old, converted, raw = corpus_fixture(tmp_path, monkeypatch)
    with sealed(raw) as fd:
        inventory = inspect(fd, raw, metadata, cohort='engineering')
        yield dict(fd=fd, raw=raw, metadata=metadata, inventory=inventory,
                   converted=converted, root=tmp_path, output=tmp_path/'memory-index')


def prepare(case, **kwargs):
    return api().prepare_index(case['fd'], kwargs.pop('inventory', case['inventory']),
                               kwargs.pop('output', case['output']), upstream=UPSTREAM,
                               metadata_csv=kwargs.pop('metadata_csv', case['metadata']), **kwargs)


def opened(case, receipt, **kwargs):
    return api().open_source(kwargs.pop('fd', case['fd']), kwargs.pop('index', case['output']),
                             expected_completion_sha256=kwargs.pop('expected', receipt['completion_sha256']),
                             upstream=UPSTREAM, metadata_csv=kwargs.pop('metadata_csv', case['metadata']), **kwargs)


def test_index_contains_only_small_native_sidecars_and_separate_completion(case):
    receipt = prepare(case)
    assert sorted(p.name for p in case['output'].iterdir()) == [
        'completion.json', 'manifest.json', 'metadata.npz', 'source-inventory.json']
    assert receipt['completion_sha256'] == hashlib.sha256((case['output']/'completion.json').read_bytes()).hexdigest()
    with opened(case, receipt) as source:
        assert source.manifest['format'] == 'emg2pose_sealed_tar_dataset_v1'
        assert source.data_identity['rows'] == 32 and source.data_identity['file_count'] == 4
        assert source.data_identity['full_corpus_complete'] is False
        assert source.data_identity['input_channels'] == 16 and source.data_identity['angle_dimensions'] == 20
        assert source.data_identity['fields']['emg']['sha256'] == case['converted']['fields']['emg']['sha256']
        assert source.assert_current() is None
    from egoemg_repro.emg2pose_runtime import validate_completed_data
    with pytest.raises(ValueError):
        validate_completed_data(UPSTREAM, case['output'], case['output']/'source-inventory.json',
                                metadata_csv=case['metadata'])


def test_verified_owner_keeps_source_alive_after_original_fd_closes(case):
    from test_emg2pose_memory_data import sealed
    receipt = prepare(case)
    with sealed(case['raw']) as other:
        source = opened(case, receipt, fd=other)
        duplicate = source.fileno()
        assert duplicate != other
    try:
        assert os.pread(source.fileno(), 6, 0) == b'corpus'
        assert source.assert_current() is None
        identity = source.handle_identity
        assert identity['size'] == len(case['raw']) and identity['seals'] & 15 == 15
    finally:
        source.close()
    source.close()
    with pytest.raises(OSError):
        os.fstat(duplicate)
    with pytest.raises(ValueError):
        source.fileno()


def test_returned_dicts_cannot_mutate_verified_identity_or_manifest(case):
    receipt = prepare(case)
    with opened(case, receipt) as source:
        data = source.data_identity
        data['rows'] = 999
        manifest = source.manifest
        manifest['sessions'][0]['start_idx'] = 999
        assert source.data_identity['rows'] == 32
        assert source.manifest['sessions'][0]['start_idx'] == 0


@pytest.mark.parametrize('fault', ['fields', 'blocks', 'cohort'])
def test_preparation_rechecks_inventory_against_actual_source(case, fault):
    inventory = deepcopy(case['inventory'])
    if fault == 'fields':
        inventory['fields']['emg']['sha256'] = '0'*64
    elif fault == 'blocks':
        inventory['blocks'] = []
    else:
        inventory['cohort'] = 'full'
        inventory['full_corpus_complete'] = True
    with pytest.raises(ValueError):
        prepare(case, inventory=inventory)
    assert not case['output'].exists()


def test_existing_output_is_never_overwritten(case):
    prepare(case)
    before = {p.name:p.read_bytes() for p in case['output'].iterdir()}
    with pytest.raises(FileExistsError):
        prepare(case)
    assert {p.name:p.read_bytes() for p in case['output'].iterdir()} == before


@pytest.mark.parametrize('kind', ['manifest', 'source', 'metadata', 'consistent_completion'])
def test_mutated_index_is_rejected_against_separately_frozen_receipt(case, kind):
    receipt = prepare(case)
    root = case['output']
    if kind == 'manifest':
        p = root/'manifest.json'
        value = json.loads(p.read_text());value['sessions'][0]['member_offset'] += 512
        p.write_text(json.dumps(value))
    elif kind == 'source':
        p = root/'source-inventory.json'
        value = json.loads(p.read_text());value['blocks'] = []
        p.write_text(json.dumps(value))
    elif kind == 'metadata':
        with np.load(root/'metadata.npz',allow_pickle=False) as f:
            values = {k:f[k] for k in f.files}
        values['session_start_idx'][0] = 999
        np.savez(root/'metadata.npz',**values)
    else:
        manifest = json.loads((root/'manifest.json').read_text())
        manifest['sessions'][0]['member_offset'] += 512
        (root/'manifest.json').write_text(json.dumps(manifest))
        completion = json.loads((root/'completion.json').read_text())
        completion['manifest_sha256'] = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
        (root/'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError):
        opened(case, receipt)


@pytest.mark.parametrize('fault', ['wrong_bytes', 'unsealed', 'wrong_expected_receipt'])
def test_source_admission_requires_actual_bytes_seals_and_frozen_completion(case, fault):
    from test_emg2pose_memory_data import sealed
    receipt = prepare(case)
    raw = bytearray(case['raw'])
    if fault == 'wrong_bytes':
        raw[1024] ^= 1
    with sealed(bytes(raw), seal=fault != 'unsealed') as other:
        with pytest.raises(ValueError):
            opened(case, receipt, fd=other, expected='0'*64 if fault == 'wrong_expected_receipt' else receipt['completion_sha256'])


def test_live_binding_detects_changed_small_transaction_files(case):
    receipt = prepare(case)
    with opened(case, receipt) as source:
        path = case['output']/'metadata.npz'
        path.write_bytes(path.read_bytes())
        with pytest.raises(ValueError):
            source.assert_current()


def test_identical_relocated_index_and_rebuilt_cache_preserve_content_identity(case):
    from test_emg2pose_memory_data import sealed
    receipt = prepare(case)
    with opened(case, receipt) as original:
        identity = original.data_identity
        old_handle = original.handle_identity
    relocated = case['root']/'relocated'
    shutil.copytree(case['output'], relocated)
    csv = case['root']/'relocated-metadata.csv';csv.write_bytes(case['metadata'].read_bytes())
    with sealed(case['raw']) as other:
        with opened(case, receipt, fd=other, index=relocated, metadata_csv=csv) as restored:
            assert restored.data_identity == identity
            assert restored.handle_identity['inode'] != old_handle['inode']
        new = prepare(case, output=case['root']/'rebuilt', metadata_csv=csv)
        with opened(case, new, index=case['root']/'rebuilt', metadata_csv=csv) as rebuilt:
            assert rebuilt.data_identity == identity


def test_failed_atomic_publication_retains_incomplete_evidence_without_completion(case, monkeypatch):
    from egoemg_repro import emg2pose_data as storage
    def fail(*args):
        raise RuntimeError('owned publish failed')
    monkeypatch.setattr(storage, '_publish_directory', fail)
    with pytest.raises(RuntimeError, match='owned publish failed'):
        prepare(case)
    assert not case['output'].exists()
    stage = case['output'].with_name(case['output'].name+'.incomplete')
    assert stage.is_dir() and not (stage/'completion.json').exists()
    assert json.loads((stage/'status.json').read_text())['complete'] is False


def test_admission_failure_never_double_closes_a_reused_unrelated_fd(case):
    receipt = prepare(case)
    archive = case['root']/'source-for-child.tar'
    archive.write_bytes(case['raw'])
    script = r'''
import fcntl,gc,json,os,sys
from pathlib import Path
from egoemg_repro import emg2pose_memory_index as api,emg2pose_memory_data as memory
from egoemg_repro.emg2pose_memory_archive import _create_memfd
p=json.loads(sys.argv[1]);fd=_create_memfd();body=Path(p['archive']).read_bytes()
assert os.pwrite(fd,body,0)==len(body)
fcntl.fcntl(fd,1033,15)
original=memory._digest
def change_after_hash(stream):
 value=original(stream)
 path=Path(p['index'])/'metadata.npz'
 path.write_bytes(path.read_bytes())
 return value
memory._digest=change_after_hash
failure=None
try:
 api.open_source(fd,Path(p['index']),expected_completion_sha256=p['digest'],
                 upstream=Path(p['upstream']),metadata_csv=Path(p['metadata']))
except ValueError as error:
 failure=error
else:raise AssertionError('changed index was admitted')
# Retain the exception traceback, reuse just-freed numbers, then release it.
sentinels=[os.open('/dev/null',os.O_RDONLY) for _ in range(16)]
failure=None;gc.collect()
for descriptor in sentinels:os.fstat(descriptor)
for descriptor in sentinels:os.close(descriptor)
os.close(fd)
'''
    result = subprocess.run([sys.executable,'-c',script,json.dumps({
        'archive':str(archive),'index':str(case['output']),'digest':receipt['completion_sha256'],
        'upstream':str(UPSTREAM),'metadata':str(case['metadata'])})],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
