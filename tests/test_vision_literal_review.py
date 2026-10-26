"""Whole-vision review regressions: prior content admission and natural stops."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_vision_literal_data import sample_store,crop_store,config,UPSTREAM,release_fixture_readers


@pytest.fixture
def admitted(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('hydra')
    from egoemg_repro.vision_literal_cohort import admit_vision_cohort
    np=pytest.importorskip('numpy');data=tmp_path/'data';crops=tmp_path/'crops'
    sample_store(data,n=9*7790);crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*7790,7790,7790,7790]).astype('int8').tofile(data/'frame_split_id.dat')
    cohort=admit_vision_cohort(data,crops,upstream=UPSTREAM,config=config())
    receipt=tmp_path/'frozen-cohort.json';receipt.write_text(json.dumps(cohort.identity))
    try:yield cohort,receipt,data,crops
    finally:release_fixture_readers(crops)


def test_full_callable_demands_prior_frozen_receipt_before_device_model_or_output(admitted,monkeypatch,tmp_path):
    import egoemg_repro.vision_literal_train as m
    cohort,_,_,_=admitted;output=tmp_path/'never-created'
    def forbidden(*args,**kwargs):raise AssertionError('receipt admission must precede CUDA/model work')
    monkeypatch.setattr(m,'_runtime_device',forbidden);monkeypatch.setattr(m,'build_vision_runtime',forbidden)
    with pytest.raises(ValueError,match='frozen.*receipt'):m.fit_vision(cohort,output,scope='full')
    assert not output.exists()


def test_changed_valid_payload_and_unchanged_metadata_cannot_replace_frozen_full_inputs(admitted,monkeypatch,tmp_path):
    import numpy as np
    import egoemg_repro.vision_literal_train as m
    from egoemg_repro.vision_literal_cohort import admit_vision_cohort
    cohort,receipt,data,crops=admitted
    metadata=(data/'metadata.npz').read_bytes();manifest=(data/'manifest.json').read_bytes()
    target=np.memmap(data/'generated_joint_angles_left.dat',mode='r+',dtype=np.float32,shape=(9*7790,20))
    target[3895,0]+=.01;target.flush();del target
    assert (data/'metadata.npz').read_bytes()==metadata and (data/'manifest.json').read_bytes()==manifest
    changed=admit_vision_cohort(data,crops,upstream=UPSTREAM,config=config())
    assert changed.identity['counts']==cohort.identity['counts']
    def forbidden(*args,**kwargs):raise AssertionError('changed frozen payload must reject before runtime')
    monkeypatch.setattr(m,'_runtime_device',forbidden);monkeypatch.setattr(m,'build_vision_runtime',forbidden)
    output=tmp_path/'never-created'
    with pytest.raises(ValueError,match='frozen.*receipt'):
        m.fit_vision(changed,output,scope='full',cohort_receipt=receipt)
    assert not output.exists()


def test_matching_tiny_receipt_cannot_promote_an_incomplete_cohort(admitted,monkeypatch,tmp_path):
    import egoemg_repro.vision_literal_train as m
    cohort,receipt,_,_=admitted
    def forbidden(*args,**kwargs):raise AssertionError('unregistered tiny cohort must reject before runtime')
    monkeypatch.setattr(m,'_runtime_device',forbidden)
    with pytest.raises(ValueError,match='registered|complete legacy'):
        m.fit_vision(cohort,tmp_path/'never-created',scope='full',cohort_receipt=receipt)
    assert not (tmp_path/'never-created').exists()


def test_full_cli_demands_receipt_before_reading_unavailable_input_paths(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('hydra')
    command=[sys.executable,'-B','-m','egoemg_repro.vision_literal_train','fit','--scope','full',
        '--data',str(tmp_path/'absent-data'),'--crops',str(tmp_path/'absent-crops'),
        '--upstream',str(UPSTREAM),'--output',str(tmp_path/'never-created')]
    child=subprocess.run(command,capture_output=True,text=True,timeout=45,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    assert child.returncode==2 and '--cohort-receipt' in child.stderr
    assert not (tmp_path/'never-created').exists()


@pytest.mark.parametrize('epochs,wait,stopped,want',[
    (0,0,0,False),(30,30,29,False),(31,29,30,False),(31,30,29,False),
    (31,30,30,True),(149,29,148,False),(150,0,0,True),(151,30,150,False)])
def test_full_natural_completion_requires_original_cap_or_actual_patience_boundary(epochs,wait,stopped,want):
    pytest.importorskip('torch');pytest.importorskip('hydra')
    m=importlib.import_module('egoemg_repro.vision_literal_train')
    policy=m.vision_fit_policy(UPSTREAM,config())
    records=[dict(train_loss=1.,validation=dict(val_mae=1.)) for _ in range(epochs)]
    early=dict(wait_count=wait,stopped_epoch=stopped,best_score=1.)
    assert m.vision_stage_complete(policy,records,early) is want


@pytest.mark.parametrize('damage',['val_nan','loss_inf','early_nan'])
def test_nonfinite_evidence_never_becomes_full_completion_at_epoch_cap(damage):
    pytest.importorskip('torch');pytest.importorskip('hydra')
    m=importlib.import_module('egoemg_repro.vision_literal_train');policy=m.vision_fit_policy(UPSTREAM,config())
    records=[dict(train_loss=1.,validation=dict(val_mae=1.)) for _ in range(150)]
    early=dict(wait_count=0,stopped_epoch=0,best_score=1.)
    if damage=='val_nan':records[-1]['validation']['val_mae']=float('nan')
    elif damage=='loss_inf':records[-1]['train_loss']=float('inf')
    else:early['best_score']=float('nan')
    assert m.vision_stage_complete(policy,records,early) is False
