"""Independent native metadata transaction and explicit compact source owner."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_emg2pose_compact_data import case,packed,UPSTREAM


def api():return importlib.import_module('egoemg_repro.emg2pose_compact_index')
def data():return importlib.import_module('egoemg_repro.emg2pose_compact_data')
def inventory(case):
    return data().inspect_compact(case['reader'],expected_receipt_sha256=case['reader'].receipt['sha256'],
        metadata_csv=case['metadata'],upstream=UPSTREAM,chunk_rows=3)
def prepare(case,**changes):
    return api().prepare_index(case['reader'],changes.pop('inventory',inventory(case)),
        changes.pop('output',case['root']/'compact-index'),upstream=UPSTREAM,metadata_csv=case['metadata'],**changes)
def opened(case,receipt,**changes):
    return api().open_source(changes.pop('reader',case['reader']),changes.pop('index',case['root']/'compact-index'),
        expected_completion_sha256=changes.pop('expected',receipt['completion_sha256']),upstream=UPSTREAM,
        metadata_csv=changes.pop('metadata_csv',case['metadata']),**changes)


def test_only_four_small_sidecars_and_native_metadata_bytes_match_original(case):
    receipt=prepare(case);root=case['root']/'compact-index'
    assert sorted(p.name for p in root.iterdir())==['completion.json','manifest.json','metadata.npz','source-inventory.json']
    assert receipt['completion_sha256']==hashlib.sha256((root/'completion.json').read_bytes()).hexdigest()
    with np.load(root/'metadata.npz',allow_pickle=False) as actual,np.load(case['root']/'old-memmap/metadata.npz',allow_pickle=False) as old:
        assert set(actual.files)==set(old.files)
        for name in old.files:assert actual[name].dtype==old[name].dtype and actual[name].tobytes()==old[name].tobytes()
    with opened(case,receipt) as source:
        assert source.manifest['format']=='emg2pose_compact_tar_dataset_v1'
        identity=source.data_identity
        assert identity['format']=='emg2pose_compact_data_identity_v1' and identity['rows']==32 and identity['file_count']==4
        assert identity['fields']==case['original']['fields']
        assert identity['full_corpus_complete'] is False and identity['engineering_only'] is True
        assert source.pread(6,0)==b'corpus' and source.assert_current() is None
        assert not hasattr(source,'fileno')
        proof=source.handle_identity
        assert proof['format']=='emg2pose_verified_compact_handle_v1'
        assert proof['payload']['inode']!=proof['block_index']['inode']


def test_source_owns_both_objects_after_original_reader_closes(case):
    receipt=prepare(case);source=opened(case,receipt)
    fds=(source.payload_fileno(),source.block_index_fileno());case['reader'].close()
    try:
        assert source.pread(6,0)==b'corpus'
        with source.slice(0,6) as view:assert view.read()==b'corpus'
    finally:source.close()
    source.close()
    for fd in fds:
        with pytest.raises(OSError):os.fstat(fd)
    with pytest.raises(ValueError):source.assert_current()


def test_returned_metadata_cannot_mutate_the_owner(case):
    receipt=prepare(case)
    with opened(case,receipt) as source:
        identity=source.data_identity;identity['rows']=999
        manifest=source.manifest;manifest['sessions'][0]['member_offset']+=512
        proof=source.handle_identity;proof['payload']['inode']=0
        assert source.data_identity['rows']==32 and source.manifest['sessions'][0]['member_offset']==case['original']['sessions'][0]['member_offset']
        assert source.handle_identity['payload']['inode']>0


@pytest.mark.parametrize('damage',['fields','blocks','cohort','receipt'])
def test_prepare_reinspects_actual_fields_masks_roles_and_packet_identity(case,damage):
    value=inventory(case)
    if damage=='fields':value['fields']['emg']['sha256']='0'*64
    elif damage=='blocks':value['blocks']=[]
    elif damage=='cohort':value.update(cohort='full',full_corpus_complete=True)
    else:value['compact_receipt']['raw_sha256']='0'*64
    with pytest.raises(ValueError):prepare(case,inventory=value)
    assert not (case['root']/'compact-index').exists()


def test_existing_or_failed_transactions_are_never_overwritten(case):
    prepare(case);root=case['root']/'compact-index';before={p.name:p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError):prepare(case)
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}
    other=case['root']/'other-index';other.with_name(other.name+'.incomplete').mkdir()
    with pytest.raises(FileExistsError):prepare(case,output=other)


@pytest.mark.parametrize('damage',['manifest','inventory','npz','completion','extra','symlink'])
def test_mutated_or_ambiguous_index_never_passes_frozen_admission(case,damage):
    receipt=prepare(case);root=case['root']/'compact-index'
    if damage=='extra':(root/'unexpected').write_text('extra')
    elif damage=='symlink':
        original=root/'manifest.json';saved=case['root']/'moved.json';original.rename(saved);original.symlink_to(saved)
    elif damage=='npz':
        with np.load(root/'metadata.npz',allow_pickle=False) as f:values={k:f[k] for k in f.files}
        values['session_start_idx'][0]=999;np.savez(root/'metadata.npz',**values)
    else:
        path=root/({'manifest':'manifest.json','inventory':'source-inventory.json','completion':'completion.json'}[damage])
        value=json.loads(path.read_text())
        if damage=='manifest':value['sessions'][0]['member_offset']+=512
        elif damage=='inventory':value['blocks']=[]
        else:value['data_identity']['rows']=999
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):opened(case,receipt)


def test_separately_frozen_completion_and_exact_source_bytes_are_required(case,monkeypatch):
    receipt=prepare(case)
    with pytest.raises(ValueError):opened(case,receipt,expected='0'*64)
    raw=bytearray(case['raw']);raw[1024]^=1
    with packed(monkeypatch,bytes(raw)) as other:
        with pytest.raises(ValueError):opened(case,receipt,reader=other)


def test_relocated_index_and_repacked_bytes_keep_content_identity(case,monkeypatch):
    receipt=prepare(case)
    with opened(case,receipt) as source:identity=source.data_identity;binding=source.handle_identity
    location=case['root']/'relocated';shutil.copytree(case['root']/'compact-index',location)
    csv=case['root']/'relocated.csv';csv.write_bytes(case['metadata'].read_bytes())
    with packed(monkeypatch,case['raw']) as other:
        with opened(case,receipt,reader=other,index=location,metadata_csv=csv) as source:
            assert source.data_identity==identity
            assert source.handle_identity['payload']['inode']!=binding['payload']['inode']
            assert source.handle_identity['binding_stamps']!=binding['binding_stamps']


def test_live_binding_rejects_even_same_byte_replacement(case):
    receipt=prepare(case)
    with opened(case,receipt) as source:
        path=case['root']/'compact-index/metadata.npz';path.write_bytes(path.read_bytes())
        with pytest.raises(ValueError):source.assert_current()


def test_raw_and_original_memmap_admission_are_not_broadened(case):
    receipt=prepare(case)
    from egoemg_repro.emg2pose_memory_index import open_source
    from egoemg_repro.emg2pose_runtime import validate_completed_data
    root=case['root']/'compact-index'
    with pytest.raises(ValueError):
        open_source(case['reader'].payload_fileno(),root,expected_completion_sha256=receipt['completion_sha256'],
                    upstream=UPSTREAM,metadata_csv=case['metadata'])
    with pytest.raises(ValueError):validate_completed_data(UPSTREAM,root,root/'source-inventory.json',metadata_csv=case['metadata'])


@pytest.mark.parametrize('damage',['manifest','metadata','cohort','identity','producer'])
def test_independent_native_reconstruction_rejects_self_consistent_resealed_claims(case,damage):
    receipt=prepare(case);root=case['root']/'compact-index'
    completion=json.loads((root/'completion.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    value=json.loads((root/'source-inventory.json').read_text())
    if damage=='manifest':manifest['sessions'][0]['member_offset']+=512
    elif damage=='metadata':
        with np.load(root/'metadata.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        arrays['session_start_idx'][0]=999;np.savez(root/'metadata.npz',**arrays)
        manifest['metadata_sha256']=completion['metadata_sha256']=hashlib.sha256((root/'metadata.npz').read_bytes()).hexdigest()
    elif damage=='cohort':
        value['full_corpus_complete']=True;(root/'source-inventory.json').write_text(json.dumps(value))
        manifest['source_inventory_sha256']=completion['source_inventory_sha256']=hashlib.sha256((root/'source-inventory.json').read_bytes()).hexdigest()
    elif damage=='identity':completion['data_identity']['rows']=999
    else:completion['producer_files_sha256']['emg2pose_compact_index.py']='0'*64
    (root/'manifest.json').write_text(json.dumps(manifest))
    completion['manifest_sha256']=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    (root/'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError):opened(case,receipt,expected=hashlib.sha256((root/'completion.json').read_bytes()).hexdigest())


def test_disk_pressure_and_interrupted_publication_never_publish_completion(case,monkeypatch):
    from types import SimpleNamespace
    from egoemg_repro import emg2pose_data as storage
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=15*1024**3-1))
    with pytest.raises(RuntimeError,match='15 GiB'):prepare(case)
    assert not (case['root']/'compact-index').exists()
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=64*1024**3))
    def fail(*_args):raise RuntimeError('actual publication failure')
    monkeypatch.setattr(storage,'_publish_directory',fail)
    with pytest.raises(RuntimeError,match='publication failure'):prepare(case)
    stage=case['root']/'compact-index.incomplete'
    assert stage.exists() and not (stage/'completion.json').exists()
    assert json.loads((stage/'status.json').read_text())['complete'] is False
    assert case['reader'].pread(6,0)==b'corpus'
