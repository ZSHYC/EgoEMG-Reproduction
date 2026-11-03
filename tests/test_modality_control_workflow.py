"""Whole-input and native batch execution boundaries, without fake full models."""
import copy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_modality_control_run import arguments


def api():
    return importlib.import_module('egoemg_repro.modality_control_run')


def test_unregistered_source_cannot_start_bulk_input_or_learned_execution(monkeypatch,tmp_path):
    def forbidden(*a,**k):raise AssertionError('invalid source constructed model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    with pytest.raises(ValueError):api().admit_control_inputs(SimpleNamespace(check=lambda:None))
    with pytest.raises(ValueError):api().run_control_pass(SimpleNamespace(check=lambda:None),tmp_path/'new',scope='complete')
    assert not(tmp_path/'new').exists()


def test_workflow_request_cannot_silently_limit_population_or_change_batch_size(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('invalid request constructed model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    for field in('max_samples','batch_size','limit_batches'):
        with pytest.raises(ValueError):api().run_control_workflow(dict(arguments(),**{field:1}),tmp_path/field)
        assert not(tmp_path/field).exists()


@pytest.fixture(scope='module')
def actual_inputs():
    if not Path(arguments()['paired_root']).is_dir():
        if os.environ.get('EGOEMG_REQUIRE_CONTROL_WORKFLOW')=='1':pytest.fail('actual complete control sources required')
        pytest.skip('actual complete inputs unavailable; no fake positive')
    e=api().load_control_evidence(arguments())
    original=torch.nn.Module.__init__
    def forbidden(*a,**k):raise AssertionError('full input admission constructed a learned module')
    torch.nn.Module.__init__=forbidden
    try:return api().admit_control_inputs(e)
    finally:torch.nn.Module.__init__=original


def test_actual_whole_inputs_bind_native_normalization_payloads_and_exact_batch_grids(actual_inputs):
    m=api();p=actual_inputs;p.check()
    assert len(p)==17168
    assert p.identity['native_input_admission']['actual_complete_payload_files']==54
    assert p.identity['native_input_admission']['actual_complete_payload_bytes']==61379659366
    assert p.identity['full_inference_permit']is False
    for hand,view in zip(('left','right'),p.native.dataset.fusion_views):
        assert p.normalization[hand]==dict(vision_mean=view.vision_mean.tolist(),vision_std=view.vision_std.tolist())
    complete=m.control_batch_ranges(p.evidence,'complete');qual=m.control_batch_ranges(p.evidence,'qualification')
    assert len(complete)==272 and sum(b-a for a,b in complete)==17168
    assert len(qual)==12 and [b-a for a,b in qual]==[64,45,64,45,64,19,64,19,64,8,64,8]


@pytest.mark.parametrize('damage',['copy','evidence','native','normalization','identity'])
def test_replacing_admitted_inputs_or_resealing_norm_cannot_reuse_full_permit(actual_inputs,damage,monkeypatch):
    from egoemg_repro.classic_emg import _hash
    p=actual_inputs
    if damage=='copy':
        with pytest.raises(ValueError,match='registered'):copy.copy(p).check()
        return
    old=copy.deepcopy(p.identity)
    with monkeypatch.context()as patch:
        if damage=='evidence':patch.setattr(p,'evidence',copy.copy(p.evidence))
        elif damage=='native':patch.setattr(p,'native',copy.copy(p.native))
        elif damage=='normalization':patch.setattr(p,'normalization',dict(left=dict(vision_mean=[0,0,0],vision_std=[1,1,1]),right=p.normalization['right']))
        else:p.identity['full_inference_permit']=True;p.identity['sha256']=_hash({k:v for k,v in p.identity.items()if k!='sha256'})
        try:
            with pytest.raises(ValueError):p.check()
        finally:p.identity.clear();p.identity.update(old)
    p.check()


def test_complete_control_needs_its_own_real_six_condition_qualification_before_model_creation(actual_inputs,tmp_path,monkeypatch):
    import egoemg_repro.occlusion_dense_models as models
    def forbidden(*a,**k):raise AssertionError('unqualified full control run constructed model')
    monkeypatch.setattr(models,'load_dense_trained_models',forbidden)
    for proof in(None,Path(actual_inputs.evidence.request['baseline'])/'qualification.json'):
        with pytest.raises(ValueError):api().run_control_pass(actual_inputs,tmp_path/'full',scope='complete',qualification=proof)
        assert not(tmp_path/'full').exists()


def test_empty_directory_cannot_be_accepted_as_a_complete_control_pass(actual_inputs,tmp_path):
    with pytest.raises(ValueError):api().read_control_pass(tmp_path,evidence=actual_inputs.evidence)


@pytest.fixture(scope='module')
def actual_trained(actual_inputs):
    original=torch.optim.Optimizer.__init__
    def forbidden(*a,**k):raise AssertionError('control native batch constructed optimizer')
    torch.optim.Optimizer.__init__=forbidden
    try:yield api().load_control_models(actual_inputs)
    finally:torch.optim.Optimizer.__init__=original


def test_genuine_B64_and_original_tail_six_condition_packets_match_complete_native_baseline(actual_inputs,actual_trained,tmp_path):
    from egoemg_repro.occlusion_dense_run import _managed_loader,_actor_closed
    p=actual_inputs;keys=p.evidence.keys();ranges=api().control_batch_ranges(p.evidence,'qualification')[:2]
    indices=[i for a,b in ranges for i in range(a,b)];seen=[]
    with _managed_loader(p,indices,upstream=p.evidence.native.upstream)as(iterator,workers,pinned):
        for ordinal,batch in enumerate(iterator):
            output=api().predict_control_batch_set(p,actual_trained,batch)
            a,b=ranges[ordinal];assert output['keys']==keys[a:b]
            assert output['actual_forward_condition_count']==6 and not output['full_inference_permit']
            baseline=p.evidence.baseline_packet(output['baseline_ordinal'])
            np.testing.assert_array_equal(output['arrays']['fusion_prediction'][0],baseline['fusion_prediction'])
            record=api().write_control_batch(tmp_path,ordinal,keys=output['keys'],arrays=output['arrays'],
                baseline=baseline,baseline_ordinal=output['baseline_ordinal'],inputs_sha256=output['inputs_sha256'])
            again=api().read_control_batch(tmp_path,record,expected_keys=output['keys'],baseline=baseline,baseline_ordinal=output['baseline_ordinal'])
            assert again['fusion_prediction'].shape==(6,b-a,22)
            seen.append(b-a)
    assert seen==[64,45]and len(workers)==4
    for worker in workers:_actor_closed(worker)
    actual_trained.check();p.check();assert not torch.cuda.is_initialized()


def test_collated_input_change_is_rejected_before_any_control_forward(actual_inputs,actual_trained,monkeypatch):
    from egoemg_repro.occlusion_dense_run import _managed_loader
    import egoemg_repro.modality_control_inference as adapter
    p=actual_inputs
    def forbidden(*a,**k):raise AssertionError('changed collated input reached learned forward')
    with _managed_loader(p,list(range(64)),upstream=p.evidence.native.upstream)as(iterator,workers,pinned):
        batch=next(iterator);batch['fusion']['emg'][0,0,0]+=1
        monkeypatch.setattr(adapter,'predict_control_batch',forbidden)
        with pytest.raises(ValueError):api().predict_control_batch_set(p,actual_trained,batch)
