"""Nested admission must match the genuine complete original input receipt."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_modality_control_run import real_control_evidence

RGB=dict(vision_mean=[123.67500305175781,116.27999877929688,103.52999877929688],
    vision_std=[58.39500045776367,57.12000274658203,57.375])


def api():return importlib.import_module('egoemg_repro.modality_control_run')


def admission(evidence):
    from egoemg_repro.occlusion_dense_run import _seal
    original=json.loads((Path(evidence.request['baseline'])/'full-a/identity.json').read_text())['input_admission']
    return _seal(dict(format='registered_complete_control_input_admission_v1',
        evidence_identity_sha256=evidence.identity['sha256'],native_input_admission=original,
        normalization={h:copy.deepcopy(RGB)for h in('left','right')},
        full_inference_permit=False,full_paper_complete=False))


def test_correct_nested_original_admission_is_valid_metadata_only(real_control_evidence):
    v=admission(real_control_evidence)
    result=api()._validate_control_input_identity(real_control_evidence,v)
    assert result==v and not result['full_inference_permit']
    assert result['native_input_admission']['actual_complete_payload_files']==54


@pytest.mark.parametrize('damage',['nested_unsealed','nested_resealed_bytes','nested_resealed_prehash',
    'nested_resealed_payloadmap','outer_permit','outer_evidence','normalization','outer_unknown_field'])
def test_resealed_partial_or_false_input_admission_is_rejected(real_control_evidence,damage):
    from egoemg_repro.occlusion_dense_run import _seal
    v=admission(real_control_evidence);native=v['native_input_admission']
    if damage=='nested_unsealed':native['actual_complete_payload_bytes']=1
    elif damage=='nested_resealed_bytes':native['actual_complete_payload_bytes']=1
    elif damage=='nested_resealed_prehash':native['complete_pre_forward_payload_hashes_verified']=False
    elif damage=='nested_resealed_payloadmap':native['complete_payload_sha256'].pop(next(iter(native['complete_payload_sha256'])))
    elif damage=='outer_permit':v['full_inference_permit']=True
    elif damage=='outer_evidence':v['evidence_identity_sha256']='a'*64
    elif damage=='normalization':v['normalization']['left']['vision_mean'][0]=.485
    else:v['max_samples']=1
    if damage.startswith('nested_resealed_'):
        v['native_input_admission']=_seal({k:x for k,x in native.items()if k!='sha256'})
    v=_seal({k:x for k,x in v.items()if k!='sha256'})
    with pytest.raises(ValueError):api()._validate_control_input_identity(real_control_evidence,v)


def test_unregistered_object_cannot_validate_input_metadata():
    with pytest.raises(ValueError):api()._validate_control_input_identity(SimpleNamespace(check=lambda:None),{})
