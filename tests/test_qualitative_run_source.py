"""Actual closed standard-grid source admissions; no fake full positives."""
import copy
import json
import os
from pathlib import Path

import pytest

from test_occlusion_dense_data import UPSTREAM
from egoemg_repro.qualitative_run import (load_qualitative_evidence,admit_qualitative_run_inputs,
    run_qualitative_pass,read_qualitative_pass,PROOF)

ROOT=Path(__file__).resolve().parents[1]
PAIRED=Path(os.environ.get('EGOEMG_QUALITATIVE_PAIRED_ROOT',
    '/data/shihao/egoemg-qualitative-ordinary-paired-io-3ddc1a6b03eb-20261101'))
pytestmark=pytest.mark.skipif(not PAIRED.exists(),reason='actual server source required; no simulated full positive')


@pytest.fixture(scope='module')
def evidence():
    return load_qualitative_evidence(evidence=ROOT/'results/native-qualitative',paired_root=PAIRED,upstream=UPSTREAM)


def test_real_closed_ordinary_source_has_all2322_keys_38_batches_and_original_counts(evidence):
    p=evidence;p.check();v=p.identity;keys=p.keys()
    assert len(keys)==2322 and len(v['batch_ranges'])==38
    assert v['batch_ranges'][-1]==[2289,2322] # right group starts2161:64+64+33
    assert v['crop_missing_retained']==80 and v['label_invalid_retained']==64
    assert v['paired_input_identity']['windows_crossing_center_split']==230
    assert v['ordinary_dense_collation_AST_exact']
    assert not v['trained_model_or_inference_permit']and not v['figure7_reproduced']


def test_copied_objects_and_mutated_exported_metadata_do_not_change_live_source(evidence):
    original=evidence.keys()[0];evidence.keys()[0].clear();evidence.identity.clear();evidence.records()[0].clear()
    assert evidence.keys()[0]==original
    with pytest.raises(ValueError,match='registered'):copy.copy(evidence).check()


def copied_graph(tmp_path):
    d=tmp_path/'copied-source';(d/'private/vits_s-a').mkdir(parents=True)
    for n in('workflow-result.json','execution-state.json','stage-receipt.json'):
        (d/'private'/n).write_bytes((PAIRED/'private'/n).read_bytes())
    for n in('identity.json','samples.jsonl'):
        (d/'private/vits_s-a'/n).write_bytes((PAIRED/'private/vits_s-a'/n).read_bytes())
    return d


@pytest.mark.parametrize('damage',['source_commit','live_owner','changed_journal','resealed_proof'])
def test_current_graph_drift_or_resealed_evidence_cannot_authorize_model_inputs(tmp_path,damage):
    d=copied_graph(tmp_path);proofdir=ROOT/'results/native-qualitative'
    if damage=='source_commit':
        p=d/'private/workflow-result.json';v=json.loads(p.read_text());v['source_commit']='a'*40;p.write_text(json.dumps(v))
    elif damage=='live_owner':
        from egoemg_repro.emg2pose_memory_process import process_identity
        p=d/'private/execution-state.json';v=json.loads(p.read_text());v['wrapper']=process_identity();p.write_text(json.dumps(v))
    elif damage=='changed_journal':
        p=d/'private/vits_s-a/samples.jsonl';rows=p.read_text().splitlines();v=json.loads(rows[0]);v['key']['center_idx']+=1;rows[0]=json.dumps(v);p.write_text('\n'.join(rows)+'\n')
    else:
        proofdir=tmp_path/'proof';proofdir.mkdir();v=json.loads((ROOT/'results/native-qualitative'/PROOF).read_text());v['recipes']['vits_s']['selected_hand_points']=1;(proofdir/PROOF).write_text(json.dumps(v))
    with pytest.raises(ValueError):load_qualitative_evidence(evidence=proofdir,paired_root=d,upstream=UPSTREAM)


def test_full_evidence_is_not_a_registered_model_input_or_prediction_result(evidence,tmp_path):
    with pytest.raises(ValueError,match='registered'):run_qualitative_pass(evidence,'/missing',tmp_path/'result')
    with pytest.raises(ValueError,match='registered'):admit_qualitative_run_inputs(object(),data='/missing',crops='/missing')
    with pytest.raises(ValueError,match='registered'):read_qualitative_pass(tmp_path,evidence=copy.copy(evidence))
