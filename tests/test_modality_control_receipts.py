"""Receipt-only engineering fixtures never establish a completed neural run.

These catch promotion of unverified population/provenance and a receipt made
by someone other than the actual closed numeric worker. Raw neural rebuilding
is the one slow dependency replaced here; file IO and verification are real.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from egoemg_repro import modality_control_run as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def actor():
    code = ('import json;from egoemg_repro.emg2pose_memory_process import process_identity;'
            'print(json.dumps(process_identity()))')
    p = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=True)
    result = json.loads(p.stdout)
    m._closed(result)
    return result


@pytest.fixture
def receipts(tmp_path):
    evidence = SimpleNamespace(identity={'sha256':'e'*64}, keys=lambda:list(range(17168)), check=lambda:None)
    report_actor, verifier_actor = actor(), actor()
    for directory in ('report', 'report-rebuilt'):
        p = tmp_path/directory
        p.mkdir()
        (p/'metrics.json').write_text('{"engineering_fixture_only":true}\n')
        (p/'condition-results.csv').write_text('engineering_fixture_only\ntrue\n')
    body = dict(format='pending_complete_control_report_v1', evidence_identity_sha256='e'*64,
        actual_points=17168, actual_conditions=6, first_result_file_sha256='a'*64,
        second_result_file_sha256='b'*64, actual_complete_raw_replay_exact=True,
        artifacts_sha256={n:m._sha(tmp_path/'report'/n) for n in ('metrics.json','condition-results.csv')},
        report_actor=report_actor, scientific_eligible=False, full_paper_complete=False)
    original = m._seal(body)
    rebuilt = m._seal(dict(body, report_actor=verifier_actor))
    write(tmp_path/'report/pending-report.json', original)
    write(tmp_path/'report-rebuilt/pending-report.json', rebuilt)
    return tmp_path, evidence, original, rebuilt, report_actor, verifier_actor


@pytest.mark.parametrize('damage', ['points','evidence','conditions','first_pass','second_pass',
    'replay','eligibility','paper_complete','extra','artifact_path'])
def test_verifier_rejects_resealed_pending_metadata_that_its_rebuild_did_not_establish(receipts, monkeypatch, damage):
    root, evidence, original, rebuilt, _, _ = receipts
    bad = copy.deepcopy(original)
    if damage=='points': bad['actual_points']=1
    elif damage=='evidence': bad['evidence_identity_sha256']='c'*64
    elif damage=='conditions': bad['actual_conditions']=1
    elif damage=='first_pass': bad['first_result_file_sha256']='c'*64
    elif damage=='second_pass': bad['second_result_file_sha256']='c'*64
    elif damage=='replay': bad['actual_complete_raw_replay_exact']=False
    elif damage=='eligibility': bad['scientific_eligible']=True
    elif damage=='paper_complete': bad['full_paper_complete']=True
    elif damage=='extra': bad['unverified_population']=True
    else:
        bad['artifacts_sha256']['../unowned.json']='c'*64
    write(root/'report/pending-report.json',m._seal({k:v for k,v in bad.items() if k!='sha256'}))
    # Only expensive raw neural report reconstruction is replaced. The actual
    # verifier must reject differing metadata even if its arithmetic is equal.
    monkeypatch.setattr(m,'build_control_report',lambda *a,**k:rebuilt)
    with pytest.raises(ValueError):
        m.verify_control_report('first','second',root/'report',evidence=evidence,output=root/'report-rebuilt')
    assert not(root/'report-rebuilt/verification.json').exists()


def verification(root, evidence, original, rebuilt, verifier_actor):
    return m._seal(dict(format='independently_rebuilt_complete_control_report_v1',
        pending_report_file_sha256=m._sha(root/'report/pending-report.json'),
        rebuilt_pending_report_file_sha256=m._sha(root/'report-rebuilt/pending-report.json'),
        artifacts_sha256=original['artifacts_sha256'],evidence_identity_sha256=evidence.identity['sha256'],
        all_numeric_json_and_csv_bytes_exact=True, actual_points=17168, actual_conditions=6,
        first_result_file_sha256='a'*64,second_result_file_sha256='b'*64,
        verifier_actor=verifier_actor,scientific_eligible=False,full_paper_complete=False))


def test_real_verifier_seals_rebuilt_metadata_and_its_own_actor(receipts,monkeypatch):
    from egoemg_repro.emg2pose_memory_process import process_identity
    root,e,original,rebuilt,_,_=receipts
    rebuilt=m._seal(dict({k:v for k,v in rebuilt.items()if k!='sha256'},report_actor=process_identity()))
    write(root/'report-rebuilt/pending-report.json',rebuilt)
    monkeypatch.setattr(m,'build_control_report',lambda *a,**k:rebuilt)
    result=m.verify_control_report('first','second',root/'report',evidence=e,output=root/'report-rebuilt')
    assert result['actual_points']==17168 and result['actual_conditions']==6
    assert result['rebuilt_pending_report_file_sha256']==m._sha(root/'report-rebuilt/pending-report.json')
    assert result['verifier_actor']['pid']==process_identity()['pid']
    assert result['scientific_eligible']is False


def test_valid_receipt_promotion_requires_both_actual_closed_numeric_workers(receipts):
    root,e,original,rebuilt,report_actor,verifier_actor=receipts
    write(root/'report-rebuilt/verification.json',verification(root,e,original,rebuilt,verifier_actor))
    value=m._promote_verified_control_report(root,evidence=e,report_actor=report_actor,verifier_actor=verifier_actor)
    assert value['actual_points']==17168 and value['actual_conditions']==6
    assert value['scientific_eligible']is True and value['full_paper_complete']is False
    assert not(root/'report/report.json').exists()  # Pure admission, no publishing side effect.


@pytest.mark.parametrize('damage',['report_actor','rebuilt_actor','verifier_actor','verification_points',
    'rebuilt_pass','verification_rebuilt_sha','rebuilt_artifact'])
def test_promotion_rejects_unowned_or_changed_numeric_evidence(receipts, damage):
    root,e,original,rebuilt,report_actor,verifier_actor=receipts
    value=verification(root,e,original,rebuilt,verifier_actor)
    if damage=='report_actor': original['report_actor']=actor()
    elif damage=='rebuilt_actor': rebuilt['report_actor']=actor()
    elif damage=='verifier_actor': value['verifier_actor']=actor()
    elif damage=='verification_points': value['actual_points']=1
    elif damage=='rebuilt_pass': rebuilt['first_result_file_sha256']='c'*64
    elif damage=='verification_rebuilt_sha': value['rebuilt_pending_report_file_sha256']='c'*64
    else:(root/'report-rebuilt/metrics.json').write_text('{}\n')
    write(root/'report/pending-report.json',m._seal({k:v for k,v in original.items() if k!='sha256'}))
    write(root/'report-rebuilt/pending-report.json',m._seal({k:v for k,v in rebuilt.items() if k!='sha256'}))
    # Recompute simple receipt/file seals so the test reaches semantic binding.
    value['pending_report_file_sha256']=m._sha(root/'report/pending-report.json')
    if damage!='verification_rebuilt_sha':value['rebuilt_pending_report_file_sha256']=m._sha(root/'report-rebuilt/pending-report.json')
    write(root/'report-rebuilt/verification.json',m._seal({k:v for k,v in value.items() if k!='sha256'}))
    with pytest.raises(ValueError):
        m._promote_verified_control_report(root,evidence=e,report_actor=report_actor,verifier_actor=verifier_actor)
    assert not(root/'report/report.json').exists()
