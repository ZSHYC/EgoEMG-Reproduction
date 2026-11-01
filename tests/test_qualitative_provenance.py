"""Actual completed source capsules must exist and retain their accepted bytes."""
import json
import os
from pathlib import Path
import shutil

import pytest

from test_occlusion_dense_data import UPSTREAM

ROOT=Path(__file__).resolve().parents[1]
PAIRED=Path(os.environ.get('EGOEMG_QUALITATIVE_PAIRED_ROOT','/data/shihao/egoemg-qualitative-ordinary-paired-io-3ddc1a6b03eb-20261101'))
PRED=Path('/data/shihao/egoemg-vits-ordinary-frozen-full-cpu-7fe8ea87a630-20261101')
pytestmark=pytest.mark.skipif(not(PAIRED.exists()and PRED.exists()),reason='actual completed source required, no fake full positive')


def graph(tmp_path,*,full=True):
    destination=tmp_path/'paired';destination.mkdir()
    names=['private/workflow-result.json','private/execution-state.json','private/stage-receipt.json','private/vits_s-a/identity.json','private/vits_s-a/samples.jsonl']
    if full:names+=list(json.loads((PAIRED/'private/stage-receipt.json').read_text())['files_sha256'])
    for name in names:
        target=destination/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(PAIRED/name,target)
    return destination


def load(path,**kwargs):
    from egoemg_repro.qualitative_run import load_qualitative_evidence
    return load_qualitative_evidence(evidence=ROOT/'results/native-qualitative',paired_root=path,upstream=UPSTREAM,**kwargs)


# This catches an opaque retained bundle field permitting missing or changed
# historical producer bytes, even though the accepted identity/journal match.
@pytest.mark.parametrize('damage',['all_capsule_absent','changed_manifest_map','missing_file','changed_file'])
def test_accepted_metadata_cannot_replace_the_actual_paired_source_capsule(tmp_path,damage):
    root=graph(tmp_path,full=damage!='all_capsule_absent')
    source=root/'egoemg_repro/qualitative_inputs.py'
    if damage=='changed_manifest_map':
        path=root/'private/stage-receipt.json';value=json.loads(path.read_text())
        value['files_sha256']={'missing-source.py':'0'*64};path.write_text(json.dumps(value))
    elif damage=='missing_file':source.unlink()
    elif damage=='changed_file':source.write_bytes(source.read_bytes()+b'\n# changed historical producer\n')
    with pytest.raises((ValueError,FileNotFoundError)):load(root)


def test_every_admitted_paired_capsule_file_remains_guarded_after_admission(tmp_path):
    root=graph(tmp_path);e=load(root);assert len(e.keys())==2322
    source=root/'egoemg_repro/qualitative_inputs.py';source.write_bytes(source.read_bytes()+b'\n')
    with pytest.raises(ValueError):e.check()


def test_stronger_reader_can_open_exact_accepted_historical_predictions_without_new_forward():
    from egoemg_repro.classic_emg import _hash
    from egoemg_repro.data import file_sha256
    from egoemg_repro.qualitative_run import read_qualitative_pass
    e=load(PAIRED,producer_root=PRED)
    identity=json.loads((PRED/'private/pass-a/identity.json').read_text())
    assert _hash(e.identity)==identity['evidence_identity_sha256']
    assert e.identity['source_files_sha256']==identity['source_files_sha256']
    assert e.reader_provenance['current_source_files_sha256']['egoemg_repro/qualitative_run.py']==file_sha256(ROOT/'egoemg_repro/qualitative_run.py')
    assert e.reader_provenance['historical_producer_source_files_sha256']==identity['source_files_sha256']
    raw=read_qualitative_pass(PRED/'private/pass-a',evidence=e)
    assert raw['identity']==identity and raw['arrays']['targets'].shape==(2322,22)


def test_another_capsule_cannot_supply_the_accepted_historical_predictor_identity():
    with pytest.raises(ValueError):load(PAIRED,producer_root=PAIRED)
