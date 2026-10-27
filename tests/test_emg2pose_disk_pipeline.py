"""Original source -> verified retirement -> unchanged conversion -> admission."""
import hashlib
import importlib
from pathlib import Path

import pytest
from test_emg2pose_disk_prepare import case as scratch_fixture,UPSTREAM


def api():return importlib.import_module('egoemg_repro.emg2pose_disk_pipeline')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@pytest.fixture
def case(tmp_path,monkeypatch):
    return scratch_fixture.__wrapped__(tmp_path,monkeypatch)

def run(c,**kwargs):
    return api().prepare_original_data(c['download'],c['root']/'pipeline',upstream=UPSTREAM,metadata_csv=c['csv'],
        cohort=kwargs.pop('cohort','engineering'),expected_download_receipt_sha256=kwargs.pop('expected',sha(c['download']/'receipt.json')),**kwargs)

def test_entire_original_engineering_source_retirement_conversion_admission(case):
    from egoemg_repro import emg2pose_data as old
    result=run(case);root=case['root']/'pipeline'
    assert result['pipeline_complete'] is True and result['full_corpus_complete'] is False
    assert result['data_identity']['file_count']==4 and result['data_identity']['rows']==32
    assert not result['scientific_complete'] and not result['full_paper_complete'] and not result['model_training_performed']
    assert result==old._json(root/'completion.json') and not case['archive'].exists()
    for name,spec in case['converted']['fields'].items():assert result['data_identity']['fields'][name]['sha256']==spec['sha256']
    assert result['preparation_sha256']==sha(root/'prepared/preparation.json')
    assert result['retirement_sha256']==sha(root/'prepared/retirement.json')
    assert result['conversion_completion_sha256']==sha(root/'data/completion.json')

@pytest.mark.parametrize('stage',['conversion','admission'])
def test_later_failure_keeps_all_source_bits_reconstructable_without_ready_marker(case,monkeypatch,stage):
    from egoemg_repro import emg2pose_tar_preservation as kept,emg2pose_data as old
    m=api()
    def fail(*a,**k):raise RuntimeError('actual requested stage failure')
    if stage=='conversion':monkeypatch.setattr(m.storage,'convert_corpus',fail)
    else:monkeypatch.setattr(m.native,'validate_completed_data',fail)
    with pytest.raises(RuntimeError):run(case)
    root=case['root']/'pipeline';assert not (root/'completion.json').exists()
    proof=old._json(root/'prepared/preparation.json')
    kept.reconstruct_tar(root/'prepared/preserved',case['root']/'failure-recovery.tar',expected_completion_sha256=proof['preservation_completion_sha256'])
    assert (case['root']/'failure-recovery.tar').read_bytes()==case['raw']

@pytest.mark.parametrize('cohort',['mini','full'])
def test_tiny_fixture_never_becomes_an_official_source(case,cohort):
    with pytest.raises(ValueError):run(case,cohort=cohort)
    assert case['archive'].exists() and not (case['root']/'pipeline/completion.json').exists()

def test_frozen_byte_receipt_is_required_before_any_source_retirement(case):
    with pytest.raises(ValueError):run(case,expected='0'*64)
    assert case['archive'].exists() and not (case['root']/'pipeline/completion.json').exists()

def test_existing_pipeline_is_never_overwritten(case):
    p=case['root']/'pipeline';p.mkdir();(p/'foreign').write_text('sentinel')
    with pytest.raises(FileExistsError):run(case)
    assert (p/'foreign').read_text()=='sentinel' and case['archive'].exists()

def test_callback_failure_keeps_source_and_no_ready_marker(case):
    def stop(_):raise RuntimeError('actual observer stop')
    with pytest.raises(RuntimeError):run(case,on_progress=stop)
    assert case['archive'].exists() and not (case['root']/'pipeline/completion.json').exists()

def test_source_completion_sync_failure_retracts_only_owned_marker(case,monkeypatch):
    m=api();original=m.preparation.retained._sync;root=case['root']/'pipeline'
    def stop(path):
        if Path(path)==root and (root/'completion.json').exists():raise OSError('actual source-ready directory sync failure')
        return original(path)
    monkeypatch.setattr(m.preparation.retained,'_sync',stop)
    with pytest.raises(OSError):run(case)
    assert not (root/'completion.json').exists() and (root/'prepared/preserved/completion.json').exists()
