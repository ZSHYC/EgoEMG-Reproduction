"""Source/index admission transactions and actual owned worker execution.

The failures caught here are admitting a changed/unsealed/partial source,
publishing before final guards, or treating a failed worker as a ready source.
All positive fixtures remain explicit engineering cohorts.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).resolve().parents[1]/'upstream'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api():
    return importlib.import_module('egoemg_repro.emg2pose_memory_prepare')


@pytest.fixture
def case(tmp_path,monkeypatch):
    from test_emg2pose_memory_data import corpus_fixture,sealed
    paths,metadata,arrays,old,converted,raw = corpus_fixture(tmp_path,monkeypatch)
    receipt=dict(format='emg2pose_sealed_memory_archive_v1',complete=True,
        full_corpus_complete=False,source_bytes_received=len(raw),
        full_sha256=hashlib.sha256(raw).hexdigest(),seals=15,chunks=[],
        source=dict(url='https://example.invalid/engineering.tar',expected_bytes=len(raw),
                    etag='"fixture-v1"',last_modified='Thu, 01 Jan 1970 00:00:00 GMT'))
    acquisition=tmp_path/'acquisition.json'
    acquisition.write_text(json.dumps(receipt,sort_keys=True))
    with sealed(raw) as fd:
        yield dict(fd=fd,raw=raw,metadata=metadata,converted=converted,root=tmp_path,
                   output=tmp_path/'prepared',acquisition=acquisition,receipt=receipt)


def prepare(case,**kwargs):
    return api().prepare_source(kwargs.pop('fd',case['fd']),kwargs.pop('output',case['output']),
        upstream=UPSTREAM,metadata_csv=kwargs.pop('metadata_csv',case['metadata']),
        acquisition_receipt=case['acquisition'],
        expected_acquisition_sha256=kwargs.pop('expected',sha(case['acquisition'])),
        cohort=kwargs.pop('cohort','engineering'),**kwargs)


def test_preparation_publishes_compact_verified_index_matching_original_fields(case):
    result=prepare(case)
    assert result['preparation_complete'] is True
    assert result['full_corpus_complete'] is False and result['scientific_complete'] is False
    assert result['index_directory']=='index'
    assert result['acquisition_receipt_sha256']==sha(case['acquisition'])
    assert result==json.loads((case['output']/'preparation.json').read_text())
    location=case['output']/'index'
    assert sorted(p.name for p in location.iterdir())==[
        'completion.json','manifest.json','metadata.npz','source-inventory.json']
    assert result['index_completion_sha256']==sha(location/'completion.json')
    assert not list(case['output'].rglob('*.dat')) and not list(case['output'].rglob('*.hdf5'))
    from egoemg_repro.emg2pose_memory_index import open_source
    with open_source(case['fd'],location,expected_completion_sha256=result['index_completion_sha256'],
                     upstream=UPSTREAM,metadata_csv=case['metadata']) as source:
        assert source.data_identity==result['data_identity']
        assert source.data_identity['file_count']==4 and source.data_identity['rows']==32
        for field in ('emg','joint_angles','time','valid_mask'):
            for key in ('dtype','shape','sha256'):
                assert source.data_identity['fields'][field][key]==case['converted']['fields'][field][key]
    assert os.pread(case['fd'],6,0)==b'corpus'


def test_prepared_index_remains_content_bound_after_directory_relocation(case):
    result=prepare(case)
    relocated=case['root']/'relocated';shutil.copytree(case['output'],relocated)
    from egoemg_repro.emg2pose_memory_index import open_source
    with open_source(case['fd'],relocated/'index',expected_completion_sha256=result['index_completion_sha256'],
                     upstream=UPSTREAM,metadata_csv=case['metadata']) as source:
        assert source.data_identity==result['data_identity']


@pytest.mark.parametrize('fault',['format','incomplete','claims_corpus','bad_digest','wrong_digest',
                                'source_size','received_size','seals','weak_etag'])
def test_invalid_acquisition_never_publishes_a_prepared_source(case,fault):
    value=deepcopy(case['receipt'])
    if fault=='format':value['format']='some-other-format'
    elif fault=='incomplete':value['complete']=False
    elif fault=='claims_corpus':value['full_corpus_complete']=True
    elif fault=='bad_digest':value['full_sha256']='bad'
    elif fault=='wrong_digest':value['full_sha256']='0'*64
    elif fault=='source_size':value['source']['expected_bytes']+=512
    elif fault=='received_size':value['source_bytes_received']+=512
    elif fault=='seals':value['seals']=0
    else:value['source']['etag']='W/"weak"'
    case['acquisition'].write_text(json.dumps(value))
    with pytest.raises(ValueError):prepare(case)
    assert not case['output'].exists()
    assert os.pread(case['fd'],6,0)==b'corpus'


def test_separately_frozen_acquisition_file_sha_is_required(case):
    with pytest.raises(ValueError):prepare(case,expected='0'*64)
    assert not case['output'].exists()


@pytest.mark.parametrize('cohort',['full','mini'])
def test_fixture_cannot_be_relabelled_as_an_official_complete_source(case,cohort):
    with pytest.raises(ValueError):prepare(case,cohort=cohort)
    assert not case['output'].exists()


def test_unsealed_source_rejected_without_closing_borrowed_descriptor(case):
    from test_emg2pose_memory_data import sealed
    with sealed(case['raw'],seal=False) as fd:
        with pytest.raises(ValueError):prepare(case,fd=fd)
        assert os.pread(fd,6,0)==b'corpus'
    assert not case['output'].exists()


@pytest.mark.parametrize('existing',['published','incomplete'])
def test_existing_transaction_or_incomplete_work_is_never_overwritten(case,existing):
    path=case['output'] if existing=='published' else case['output'].with_name('prepared.incomplete')
    path.mkdir();(path/'sentinel').write_bytes(b'keep')
    with pytest.raises(FileExistsError):prepare(case)
    assert (path/'sentinel').read_bytes()==b'keep'


def test_late_acquisition_change_retains_incomplete_without_preparation_marker(case,monkeypatch):
    from egoemg_repro import emg2pose_memory_index as index
    original=index.open_source
    @contextmanager
    def change_after_admission(*args,**kwargs):
        with original(*args,**kwargs) as source:
            case['acquisition'].write_bytes(case['acquisition'].read_bytes()+b' ')
            yield source
    monkeypatch.setattr(index,'open_source',change_after_admission)
    with pytest.raises(ValueError):prepare(case)
    stage=case['output'].with_name('prepared.incomplete')
    assert not case['output'].exists() and not (stage/'preparation.json').exists()
    assert json.loads((stage/'status.json').read_text())['preparation_complete'] is False
    assert os.pread(case['fd'],6,0)==b'corpus'


def test_late_disk_pressure_prevents_publication_but_keeps_owned_index(case,monkeypatch):
    from egoemg_repro import emg2pose_data as storage
    stage=case['output'].with_name('prepared.incomplete')
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(
        free=0 if (stage/'index/completion.json').exists() else 64*1024**3))
    with pytest.raises(RuntimeError):prepare(case)
    assert not case['output'].exists() and (stage/'index/completion.json').exists()
    assert not (stage/'preparation.json').exists()
    assert os.pread(case['fd'],6,0)==b'corpus'


def test_changed_csv_is_rejected_instead_of_reusing_existing_inventory(case):
    text=case['metadata'].read_text().replace('private-user-中文','different-user')
    case['metadata'].write_text(text)
    with pytest.raises(ValueError):prepare(case)
    assert not case['output'].exists()


def cache_owner(case,receipt=None):
    from egoemg_repro.emg2pose_memory_archive import SealedArchive
    return SealedArchive(os.dup(case['fd']),case['receipt'] if receipt is None else receipt)


def test_owned_preparation_runs_actual_worker_and_freezes_completed_index(case):
    output=case['root']/'owned'
    with cache_owner(case) as cache:
        result=api().prepare_owned_source(cache,output,cohort='engineering',
            upstream=UPSTREAM,metadata_csv=case['metadata'],reserve_bytes=0)
        assert os.pread(cache.fileno(),6,0)==b'corpus'
    phase=json.loads((output/'owned-phase/stage-result.json').read_text())
    assert phase['status']=='success' and phase['worker_exit_code']==0
    assert phase['all_owned_consumers_exited'] is True
    assert result==json.loads((output/'owned-preparation.json').read_text())
    assert result['preparation_complete'] is True and result['scientific_complete'] is False
    assert result['data_identity']['rows']==32 and result['data_identity']['engineering_only']
    assert result['preparation_sha256']==sha(output/'prepared/preparation.json')
    assert result['index_completion_sha256']==sha(output/'prepared/index/completion.json')
    assert not (output/'completion.json').exists()


def test_failed_owned_worker_has_no_owned_preparation_success(case):
    from egoemg_repro.emg2pose_memory_process import OwnedStageError
    receipt=deepcopy(case['receipt']);receipt['full_sha256']='0'*64
    output=case['root']/'owned'
    with cache_owner(case,receipt) as cache:
        with pytest.raises(OwnedStageError):
            api().prepare_owned_source(cache,output,cohort='engineering',
                upstream=UPSTREAM,metadata_csv=case['metadata'],reserve_bytes=0)
        assert os.pread(cache.fileno(),6,0)==b'corpus'
    phase=json.loads((output/'owned-phase/stage-result.json').read_text())
    assert phase['status']=='failed' and phase['all_owned_consumers_exited']
    assert not (output/'owned-preparation.json').exists()


def test_owned_worker_uses_actual_source_bundle_from_unrelated_working_directory(case,monkeypatch):
    module=api()
    unrelated=case['root']/'unrelated';unrelated.mkdir()
    shadow=unrelated/'egoemg_repro';shadow.mkdir()
    (shadow/'__init__.py').write_text("raise AssertionError('wrong source bundle selected')\n")
    monkeypatch.chdir(unrelated)
    monkeypatch.setenv('PYTHONPATH',str(unrelated))
    with cache_owner(case) as cache:
        result=module.prepare_owned_source(cache,case['root']/'owned',cohort='engineering',
            upstream=UPSTREAM,metadata_csv=case['metadata'],reserve_bytes=0)
        assert result['preparation_complete'] and result['data_identity']['rows']==32
    inner=json.loads((case['root']/'owned/prepared/preparation.json').read_text())
    assert inner['producer_files_sha256']==result['producer_files_sha256']


def test_parent_capacity_poll_refuses_live_preparation_without_success(case,monkeypatch):
    from egoemg_repro.emg2pose_archive import ArchiveCapacityError
    output=case['root']/'owned'
    monkeypatch.setattr(api(),'available_memory_bytes',lambda:
        0 if (output/'owned-phase/started.json').exists() else 64*1024**3)
    with cache_owner(case) as cache:
        with pytest.raises(ArchiveCapacityError):
            api().prepare_owned_source(cache,output,cohort='engineering',
                upstream=UPSTREAM,metadata_csv=case['metadata'],reserve_bytes=1)
        assert os.pread(cache.fileno(),6,0)==b'corpus'
    assert not (output/'owned-preparation.json').exists()
    assert json.loads((output/'owned-phase/guardian-result.json').read_text())['all_owned_consumers_exited']


def test_parent_rejects_a_worker_receipt_with_different_producer_identity(case,monkeypatch):
    module=api();output=case['root']/'owned';changed=[]
    def observation():
        path=output/'prepared/preparation.json'
        if not changed and path.exists() and (output/'owned-phase/stage-result.json').exists():
            value=json.loads(path.read_text())
            value['producer_files_sha256']['egoemg_repro/emg2pose_memory_prepare.py']='0'*64
            path.write_text(json.dumps(value));changed.append(True)
        return 64*1024**3
    monkeypatch.setattr(module,'available_memory_bytes',observation)
    with cache_owner(case) as cache:
        with pytest.raises(ValueError):
            module.prepare_owned_source(cache,output,cohort='engineering',upstream=UPSTREAM,
                                        metadata_csv=case['metadata'],reserve_bytes=0)
    assert changed and not (output/'owned-preparation.json').exists()


@pytest.mark.parametrize('cohort',['full','mini'])
def test_official_cohort_reserve_floor_cannot_be_lowered(cohort):
    with pytest.raises(ValueError):api()._policy(cohort,0)


def test_cli_closes_inherited_right_after_success_in_real_process(case):
    args=['--cohort','engineering','--upstream',str(UPSTREAM),'--metadata-csv',str(case['metadata']),
          '--acquisition-receipt',str(case['acquisition']),'--expected-acquisition-sha256',sha(case['acquisition']),
          '--output',str(case['output'])]
    code='''import os,sys
from egoemg_repro.emg2pose_memory_prepare import main
fd=int(os.environ['EGOEMG_STAGE_FD_ARCHIVE'])
assert main(sys.argv[1:])==0
try:os.fstat(fd)
except OSError:pass
else:raise AssertionError('worker retained inherited cache descriptor')
'''
    result=subprocess.run([sys.executable,'-c',code,*args],env=dict(os.environ,
        EGOEMG_STAGE_FD_ARCHIVE=str(case['fd'])),pass_fds=(case['fd'],),capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    assert (case['output']/'preparation.json').exists()
    assert os.pread(case['fd'],6,0)==b'corpus'
