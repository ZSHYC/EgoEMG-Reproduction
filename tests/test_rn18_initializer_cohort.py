"""Complete RN18 initializer full-trajectory inputs, no fit/constructor permit."""
import copy
from dataclasses import replace
import importlib
import random

import numpy as np
import pytest

from egoemg_repro.classic_emg import _hash
from egoemg_repro.data import LegacyStore
from test_emg_initializer_data import source
from test_emg_initializer_literal import UPSTREAM,config


def api():return importlib.import_module('egoemg_repro.rn18_initializer_cohort')


def args(root):
    from egoemg_repro.emg_initializer_literal import emg_initializer_normalization
    return dict(data=root,upstream=UPSTREAM,config=config(),normalization=emg_initializer_normalization(UPSTREAM),require_full=False)


def admit(root,**kw):
    a=args(root);a.update(kw);return api().admit_rn18_initializer_cohort(**a)


def test_complete_numeric_targets_and_actual_content_identity_reopen_without_rng_or_models(tmp_path):
    import torch
    m=api();source(tmp_path,optional_labels=True);a=args(tmp_path)
    py=random.getstate();ns=np.random.get_state();ts=torch.get_rng_state().clone()
    c=m.admit_rn18_initializer_cohort(**a);c.check();again=m.admit_rn18_initializer_cohort(expected=c.identity,**a)
    assert again.identity==c.identity and not c.identity['registered_full_legacy']
    assert c.identity['full_payload_content_hashed']and c.identity['full_trajectory_target_nonfinite_entries']==0
    assert c.identity['consumed_emg_nonfinite_entries']==0 and set(c.datasets)=={'train','val'}
    for role,ds in c.datasets.items():
        assert c.identity['counts'][role]==len(ds)and c.identity['unique_counts'][role]==len(ds)
        assert c.identity['finite_footprints'][role]['full_trajectory_target_elements']==c.identity['finite_footprints'][role]['unique_possible_rows']*44
    assert random.getstate()==py and ns[0]==np.random.get_state()[0]and np.array_equal(ns[1],np.random.get_state()[1])and ns[2:]==np.random.get_state()[2:]
    assert torch.equal(ts,torch.get_rng_state())and not torch.cuda.is_initialized()
    assert not c.identity['model_or_training_permit']


@pytest.mark.parametrize('damage',['emg','target','masked_target','wrist'])
def test_full_jitter_noncenter_nonfinite_values_reject_even_when_masked_false(tmp_path,damage):
    m=api();source(tmp_path)
    from egoemg_repro.rn18_initializer_data import build_rn18_initializer_partitions
    p=build_rn18_initializer_partitions(LegacyStore(tmp_path));w=next(w for w in p['windows']if w['role']=='train');row=w['footprint_end']-2
    if damage=='emg':name='emg_left_filtered';shape=(len(LegacyStore(tmp_path).field('frame_split_id')),8)
    elif damage=='wrist':name='mocap_left_wrist_yaw';shape=(len(LegacyStore(tmp_path).field('frame_split_id')),)
    else:name='generated_joint_angles_left';shape=(len(LegacyStore(tmp_path).field('frame_split_id')),20)
    a=np.memmap(tmp_path/(name+'.dat'),mode='r+',dtype=np.float32,shape=shape);a[row]=np.nan;a.flush()
    if damage=='masked_target':
        mask=np.memmap(tmp_path/'generated_label_valid.dat',mode='r+',dtype=bool,shape=(shape[0],2));mask[row,0]=False;mask.flush()
    with pytest.raises(ValueError,match='nonfinite'):admit(tmp_path)


@pytest.mark.parametrize('damage',['clone','identity','partitions','norm_dtype','mapping','config','window','center_view','generation'])
def test_registered_live_cohort_rejects_resealed_or_changed_bindings(tmp_path,damage):
    m=api();source(tmp_path);c=admit(tmp_path);c.check()
    if damage=='clone':c=replace(c)
    elif damage=='identity':c.identity['counts']['train']+=2;c.identity['sha256']=_hash({k:v for k,v in c.identity.items()if k!='sha256'})
    elif damage=='partitions':c.partitions['windows'][0]['start']+=1
    elif damage=='norm_dtype':c.datasets['train'].views[0]._emg_mean=c.datasets['train'].views[0]._emg_mean.astype(np.float64)
    elif damage=='mapping':
        d=c.datasets['val'];key=next(k for k in d.views[0]._frame_memmaps if k.startswith('emg_left_filtered'))
        d.views[0]._frame_memmaps[key]=np.asarray(d.views[0]._frame_memmaps[key])
    elif damage=='config':c.config.trainer.max_epochs=200
    elif damage=='window':c.datasets['train'].windows=[]
    elif damage=='center_view':c.datasets['val'].views[1]._center_target_only=True
    else:
        p=tmp_path/'generated_joint_angles_left.dat';other=tmp_path/'replacement';other.write_bytes(p.read_bytes());other.replace(p)
    with pytest.raises(ValueError):c.check()


def test_resealed_expected_receipt_is_not_a_full_data_admission(tmp_path):
    m=api();source(tmp_path);c=admit(tmp_path)
    bad=copy.deepcopy(c.identity);bad['counts']['val']+=1;bad['sha256']=_hash({k:v for k,v in bad.items()if k!='sha256'})
    with pytest.raises(ValueError):admit(tmp_path,expected=bad)
    with pytest.raises(ValueError):admit(tmp_path,require_full=True)
    with pytest.raises(ValueError):admit(tmp_path,chunk_rows=0)


def test_per_hand_normalization_source_binding_cannot_be_replaced_by_same_numbers_float64(tmp_path):
    m=api();source(tmp_path);a=args(tmp_path);a['normalization']['hands']['left']['dtype']='float64'
    with pytest.raises(ValueError):m.admit_rn18_initializer_cohort(**a)


def test_new_provider_membership_or_content_change_cannot_be_guard_only(tmp_path,monkeypatch):
    m=api();source(tmp_path);c=admit(tmp_path);p=tmp_path/'provider.py';p.write_text('x=1\n');original=m._sources
    monkeypatch.setattr(m,'_sources',lambda root:{**original(root),'new-provider':p})
    with pytest.raises(ValueError):c.check()


def test_full_future_partition_and_counts_are_exact_registered_rn18_family_constants():
    m=api();assert m.FULL_COUNTS==dict(train=62470,val=1052)
    assert m.FULL_FUTURE_PARTITION_SHA=='44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0'
