"""Only owned completed redundant TARs may retire; original source bits survive."""
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream'))
def api():return importlib.import_module('egoemg_repro.emg2pose_disk_prepare')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@pytest.fixture
def case(tmp_path,monkeypatch):
    from test_emg2pose_memory_data import corpus_fixture
    from egoemg_repro import emg2pose_tar_preservation as kept,emg2pose_disk_download as disk
    from egoemg_repro.emg2pose_memory_process import run_owned_stage
    monkeypatch.setattr(kept,'available_memory_bytes',lambda:2*1024**4)
    monkeypatch.setattr(kept.storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=64*1024**3))
    paths,csv,arrays,inventory,converted,raw=corpus_fixture(tmp_path,monkeypatch)
    download=tmp_path/'download';download.mkdir(mode=0o700)
    archive=download/'archive.tar';archive.write_bytes(raw);archive.chmod(0o400)
    (download/'archive.lock').touch(mode=0o600)
    (download/'attempts').mkdir(mode=0o700)
    owned=run_owned_stage([sys.executable,'-c','pass'],download/'attempts/fixture',descriptors={},guard=lambda:None)
    source=dict(url='https://example.invalid/engineering.tar',expected_bytes=len(raw),etag='"owned-fixture-v1"',last_modified='Thu, 01 Jan 1970 00:00:00 GMT')
    receipt=dict(format=disk.FORMAT,asset='engineering',source=source,final_source=source,complete_bytes=True,
        source_bytes_received=len(raw),full_sha256=hashlib.sha256(raw).hexdigest(),owned_stage=owned,
        policy=dict(producer_sha256=sha(disk.__file__),host_reserve_bytes=256*1024**3,disk_reserve_bytes=15*1024**3),
        source_qualified=False,full_corpus_complete=False,scientific_complete=False,full_paper_complete=False)
    (download/'receipt.json').write_text(json.dumps(receipt))
    return dict(root=tmp_path,download=download,archive=archive,raw=raw,csv=csv,output=tmp_path/'prepared',receipt=receipt,converted=converted)

def prepare(c,**kwargs):
    return api().prepare_disk_source(c['download'],c['output'],upstream=UPSTREAM,metadata_csv=c['csv'],
        cohort=kwargs.pop('cohort','engineering'),expected_download_receipt_sha256=kwargs.pop('expected',sha(c['download']/'receipt.json')),**kwargs)

def retire(c,**kwargs):
    return api().retire_managed_tar(c['output'],expected_preparation_sha256=kwargs.pop('expected',sha(c['output']/'preparation.json')),**kwargs)

def test_prepare_then_retire_then_unchanged_conversion_and_exact_rebuild(case):
    from egoemg_repro import emg2pose_data as old,emg2pose_tar_preservation as kept
    result=prepare(case);assert result['preparation_complete'] is True and result['full_corpus_complete'] is False
    assert result==old._json(case['output']/'preparation.json') and case['archive'].read_bytes()==case['raw']
    retired=retire(case);assert retired['redundant_tar_retired'] is True and retired['scientific_complete'] is False
    assert not case['archive'].exists() and len(list((case['output']/'preserved/contents').rglob('*.hdf5')))==4
    converted=old.convert_corpus(case['output']/'source-inventory.json',case['root']/'converted-after-retire',upstream=UPSTREAM)
    for name,spec in case['converted']['fields'].items():assert converted['fields'][name]['sha256']==spec['sha256']
    kept.reconstruct_tar(case['output']/'preserved',case['root']/'rebuilt.tar',expected_completion_sha256=result['preservation_completion_sha256'])
    assert (case['root']/'rebuilt.tar').read_bytes()==case['raw']
    assert not retired['full_paper_complete']

@pytest.mark.parametrize('fault',['format','incomplete','received','source','producer','reserve','claims_full','stage_missing','stage_changed','worker_live','writable','hardlink'])
def test_invalid_download_or_ownership_never_prepares_or_deletes(case,fault):
    c=case;r=c['receipt'];p=c['download']/'receipt.json'
    if fault=='format':r['format']='other'
    elif fault=='incomplete':r['complete_bytes']=False
    elif fault=='received':r['source_bytes_received']-=1
    elif fault=='source':r['final_source']={**r['source'],'etag':'"other"'}
    elif fault=='producer':r['policy']['producer_sha256']='0'*64
    elif fault=='reserve':r['policy']['host_reserve_bytes']=0
    elif fault=='claims_full':r['full_corpus_complete']=True
    elif fault=='stage_missing':(c['download']/'attempts/fixture/stage-result.json').unlink()
    elif fault=='stage_changed':(c['download']/'attempts/fixture/stage-result.json').write_text('{}')
    elif fault=='worker_live':
        from egoemg_repro.emg2pose_memory_process import process_identity
        r['owned_stage']['worker']=process_identity();(c['download']/'attempts/fixture/stage-result.json').write_text(json.dumps(r['owned_stage']))
    elif fault=='writable':c['archive'].chmod(0o600)
    else:os.link(c['archive'],c['root']/'foreign-link.tar')
    p.write_text(json.dumps(r))
    with pytest.raises((ValueError,FileExistsError)):prepare(c)
    assert not c['output'].exists() and c['archive'].read_bytes()==c['raw']

@pytest.mark.parametrize('cohort',['mini','full'])
def test_nonofficial_tiny_source_cannot_be_promoted(case,cohort):
    with pytest.raises(ValueError):prepare(case,cohort=cohort)
    assert case['archive'].exists() and not case['output'].exists()

@pytest.mark.parametrize('stage',['prepare','retire'])
def test_frozen_identity_and_existing_output_are_required(case,stage):
    if stage=='prepare':
        with pytest.raises(ValueError):prepare(case,expected='0'*64)
        prepare(case)
        with pytest.raises(FileExistsError):prepare(case)
    else:
        prepare(case)
        with pytest.raises(ValueError):retire(case,expected='0'*64)
        retire(case)
        with pytest.raises(FileExistsError):retire(case)

@pytest.mark.parametrize('stage',['prepare','retire'])
def test_original_download_lock_prevents_concurrent_actions(case,stage):
    if stage=='retire':prepare(case)
    with (case['download']/'archive.lock').open('r+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(FileExistsError):(prepare if stage=='prepare' else retire)(case)
    assert case['archive'].exists()

@pytest.mark.parametrize('fault',['payload','manifest','csv','inventory','receipt','source','replacement'])
def test_late_mutations_cannot_retire_the_owned_source(case,fault):
    prepare(case);root=case['output']
    paths={'payload':next((root/'preserved/contents').rglob('*.hdf5')),'manifest':root/'preserved/manifest.json',
        'csv':case['csv'],'inventory':root/'source-inventory.json','receipt':case['download']/'receipt.json','source':case['archive']}
    if fault=='replacement':
        case['archive'].rename(case['root']/'saved-original.tar');case['archive'].write_bytes(case['raw']);case['archive'].chmod(0o400)
    else:
        p=paths[fault];p.chmod(0o600);b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
    with pytest.raises((ValueError,RuntimeError)):retire(case)
    assert case['archive'].exists() and not (root/'retirement.json').exists()

def test_progress_mutation_and_pressure_leave_original_present(case,monkeypatch):
    prepare(case);seen=False
    def pressure(event):
        nonlocal seen
        if not seen:
            seen=True;monkeypatch.setattr(api().retained,'available_memory_bytes',lambda:0)
    with pytest.raises(RuntimeError):retire(case,on_progress=pressure)
    assert seen and case['archive'].exists() and not (case['output']/'retirement.json').exists()

def test_final_retirement_marker_failure_preserves_reconstructable_source(case,monkeypatch):
    from egoemg_repro import emg2pose_tar_preservation as kept
    prepared=prepare(case);original=api().storage._write_json
    def fail(path,value):
        if Path(path).name=='retirement.json':raise OSError('actual late publication failure')
        return original(path,value)
    monkeypatch.setattr(api().storage,'_write_json',fail)
    with pytest.raises(OSError):retire(case)
    assert (case['output']/'retirement-started.json').is_file() and not (case['output']/'retirement.json').exists()
    kept.reconstruct_tar(case['output']/'preserved',case['root']/'recovered.tar',expected_completion_sha256=prepared['preservation_completion_sha256'])
    assert (case['root']/'recovered.tar').read_bytes()==case['raw']

@pytest.mark.parametrize('role',['worker','guardian'])
@pytest.mark.parametrize('field,value',[('pid',0),('start_ticks','not-start-ticks'),('uid',True)])
def test_malformed_terminal_process_identity_cannot_prove_an_owned_download(case,role,field,value):
    r=case['receipt'];r['owned_stage'][role][field]=value
    (case['download']/'receipt.json').write_text(json.dumps(r))
    (case['download']/'attempts/fixture/stage-result.json').write_text(json.dumps(r['owned_stage']))
    with pytest.raises(ValueError):prepare(case)
    assert case['archive'].exists() and not case['output'].exists()

@pytest.mark.parametrize('fault',['extra','overhead','payload-restored','manifest-restored'])
def test_retained_byte_changes_after_deep_verification_still_prevent_retirement(case,fault):
    prepare(case);root=case['output']/'preserved';seen=False
    if fault=='extra':p=root/'contents/corpus/metadata.csv'
    elif fault=='overhead':p=root/'tar-overhead.bin'
    elif fault=='payload-restored':p=next((root/'contents').rglob('*.hdf5'))
    else:p=root/'manifest.json'
    def change(event):
        nonlocal seen
        if event['phase']=='original-before-retirement' and not seen:
            seen=True;raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            if fault.endswith('restored'):p.write_bytes(raw)
    with pytest.raises(ValueError):retire(case,on_progress=change)
    assert seen and case['archive'].exists() and not (case['output']/'retirement.json').exists()

def test_late_preparation_directory_sync_failure_never_leaves_ready_marker(case,monkeypatch):
    original=api().retained._sync
    def fail(path):
        if Path(path)==case['output'] and (case['output']/'preparation.json').exists():raise OSError('actual preparation directory sync failure')
        return original(path)
    monkeypatch.setattr(api().retained,'_sync',fail)
    with pytest.raises(OSError):prepare(case)
    assert case['archive'].exists() and not (case['output']/'preparation.json').exists()

def test_failed_preparation_publication_never_removes_a_foreign_replacement(case,monkeypatch):
    original=api().retained._sync;marker=case['output']/'preparation.json'
    def replace(path):
        if Path(path)==case['output'] and marker.exists():
            marker.rename(case['output']/'owned-marker-moved.json');marker.write_text('foreign sentinel')
            raise OSError('actual foreign replacement at late sync')
        return original(path)
    monkeypatch.setattr(api().retained,'_sync',replace)
    with pytest.raises(OSError):prepare(case)
    assert case['archive'].exists() and marker.read_text()=='foreign sentinel'
