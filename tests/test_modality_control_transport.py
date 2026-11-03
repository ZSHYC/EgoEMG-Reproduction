"""Persisted condition fingerprints must prove the actual deterministic inputs.

The independent oracle uses native samples plus literal indexing/zero/black
operations, without calling the production input transformation or any model.
"""
import copy
import json

import pytest
import torch

from test_modality_control_workflow import actual_inputs
from egoemg_repro import modality_control_run as m
from egoemg_repro.classic_emg import _hash
from egoemg_repro.occlusion_dense_data import _typed_sha
from egoemg_repro.occlusion_dense_run import _collate_pairs


@pytest.fixture(scope='module')
def transport(actual_inputs):
    p=actual_inputs;keys=p.evidence.keys()[:64]
    samples=[p[i]for i in range(64)]
    batch=_collate_pairs(samples,upstream=p.evidence.native.upstream)['fusion']
    fingerprints={'row:'+str(i):_hash(json.loads(p.evidence.native.records[i])['typed_input_sha256'])for i in range(64)}
    for c in range(6):
        tensors={n:x.clone()for n,x in batch.items()if isinstance(x,torch.Tensor)}
        if c//2==1:tensors['emg'].zero_()
        if c//2==2:
            # Independent indexing oracle: native 7790 -> +3895 circular roll.
            tensors['emg']=torch.cat((batch['emg'][...,3895:],batch['emg'][...,:3895]),dim=-1)
        if c%2:
            black=-torch.tensor([123.67500305175781,116.27999877929688,103.52999877929688])/torch.tensor([58.39500045776367,57.12000274658203,57.375])
            shape=[1]*tensors['vision_img'].ndim;shape[-3]=3
            tensors['vision_img']=black.reshape(shape).expand_as(tensors['vision_img']).clone()
            tensors['vision_valid_mask'].zero_()
        fingerprints.update({'condition:'+str(c)+':'+n:_typed_sha(x.numpy())for n,x in tensors.items()})
    return p,keys,fingerprints


def test_actual_native_B64_condition_transport_fingerprints_are_admissible_without_models(transport,monkeypatch):
    p,keys,fp=transport
    def forbidden(*a,**k):raise AssertionError('transport admission constructed a model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    m._validate_control_fingerprints(p,keys,fp)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['all_conditions','one_condition','one_tensor','extra_tensor','extra_row',
    'missing_row','digest_type','digest_length','wrong_zero','wrong_roll','wrong_black','wrong_target'])
def test_qualification_cannot_admit_incomplete_or_altered_condition_fingerprints(transport,damage):
    p,keys,original=transport;fp=copy.deepcopy(original)
    if damage=='all_conditions':fp={k:v for k,v in fp.items()if k.startswith('row:')}
    elif damage=='one_condition':fp={k:v for k,v in fp.items()if not k.startswith('condition:5:')}
    elif damage=='one_tensor':fp.pop('condition:3:vision_img')
    elif damage=='extra_tensor':fp['condition:6:emg']='a'*64
    elif damage=='extra_row':fp['row:17168']='a'*64
    elif damage=='missing_row':fp.pop('row:1')
    elif damage=='digest_type':fp['condition:2:emg']=None
    elif damage=='digest_length':fp['condition:2:emg']='a'*63
    elif damage=='wrong_zero':fp['condition:2:emg']=fp['condition:0:emg']
    elif damage=='wrong_roll':fp['condition:4:emg']=fp['condition:0:emg']
    elif damage=='wrong_black':fp['condition:1:vision_img']=fp['condition:0:vision_img']
    else:fp['condition:5:joint_angles']='a'*64
    with pytest.raises(ValueError):m._validate_control_fingerprints(p,keys,fp)
