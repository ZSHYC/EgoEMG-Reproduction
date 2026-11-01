"""New ordinary-grid byte joins; synthetic tensors are never trained sources."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_occlusion_dense_data import UPSTREAM
from egoemg_repro.classic_emg import _hash
from egoemg_repro.occlusion_dense_data import _typed_sha
from egoemg_repro.occlusion_dense_inference import source_collate_fusion
from egoemg_repro.occlusion_dense_run import write_dense_batch


def api():
    from egoemg_repro import qualitative_run
    return qualitative_run


def samples():
    packets=[];records=[]
    for i in range(5):
        hand='left'if i<3 else'right'
        key=dict(sequence=i,split='user',hand=hand,ep_idx=0,center_idx=i*7790+3895,
            window_start=i*7790,window_end=(i+1)*7790,native_index=i if i<3 else i-3,group_index=0)
        target=(np.arange(22,dtype=np.float32)/11+i)[:,None]
        mask=np.array([i%2==0],bool)
        f=dict(emg=np.full((16,7790),i/7,np.float32),joint_angles=target,
            label_valid_mask=mask,emg_channel_mask=np.ones(16,bool),
            vision_img=np.full((3,256,256),i/13,np.float32),vision_valid_mask=~mask,
            vision_frame_indices=np.array([i],np.int64),target_hand=hand)
        e=dict(emg=np.full((8,7790),i/17,np.float32),joint_angles=target.copy(),
            label_valid_mask=mask.copy(),emg_channel_mask=np.ones(8,bool),target_hand=hand)
        typed={'fusion:'+n:_typed_sha(f[n])for n in('emg','emg_channel_mask','vision_img','vision_valid_mask','vision_frame_indices','joint_angles','label_valid_mask')}
        typed.update({'standalone:'+n:_typed_sha(e[n])for n in('emg','emg_channel_mask','joint_angles','label_valid_mask')})
        packets.append(dict(key=key,fusion=f,standalone_emg=e,typed_input_sha256=typed,verified_input_sha256=_hash(typed)))
        records.append(dict(key=copy.deepcopy(key),typed_input_sha256=typed,verified_input_sha256=_hash(typed),crop_valid=bool(f['vision_valid_mask'][0]),label_valid=bool(mask[0]),fusion_emg_shape=[16,7790],standalone_emg_shape=[8,7790]))
    return packets,records


def batch(packets):
    return dict(keys=[p['key']for p in packets],verified_input_sha256=[p['verified_input_sha256']for p in packets],
        fusion=source_collate_fusion([p['fusion']for p in packets],upstream=UPSTREAM),
        standalone=source_collate_fusion([p['standalone_emg']for p in packets],upstream=UPSTREAM))


def raw(packets):
    target=np.stack([p['fusion']['joint_angles'][:,0]for p in packets]);n=len(packets)
    a=dict(targets=target,vision_prediction=target+np.float32(.3),fusion_prediction=target+np.float32(.1),
        emg_prediction=target-np.float32(.4),label_valid_mask=np.stack([p['fusion']['label_valid_mask']for p in packets]),
        crop_valid_mask=np.stack([p['fusion']['vision_valid_mask']for p in packets]))
    a['emg_full_prediction']=np.repeat(a['emg_prediction'][:,:,None],146,axis=2)
    for role in('vision','fusion','emg'):
        a['mae_'+role+'_rad']=np.array([float((torch.from_numpy(a[role+'_prediction'][i])-torch.from_numpy(target[i])).abs().mean())for i in range(n)],np.float64)
    return a


def saved(tmp_path):
    packets,expected=samples();records=[];arrays=[]
    for ordinal,group in enumerate((packets[:3],packets[3:])):
        a=raw(group);arrays.append(a)
        records.append(write_dense_batch(tmp_path,ordinal,keys=[p['key']for p in group],arrays=a,
            inputs_sha256={'row:'+str(p['key']['sequence']):p['verified_input_sha256']for p in group}))
    return packets,expected,records,arrays


# Catches trusting a copied per-row proof after a later collator changes the
# actual tensors, or feeding the fusion16 channels to the standalone8 branch.
def test_collated_bytes_equal_every_retained_original_field_and_source_metadata():
    packets,expected=samples();b=batch(packets[:3]);v=api().verify_qualitative_collated_batch(b,expected[:3])
    assert {k:v[k]for k in('row:0','row:1','row:2')}=={'row:'+str(i):expected[i]['verified_input_sha256']for i in range(3)}
    assert v['fusion:emg']==_typed_sha(b['fusion']['emg'].numpy())
    assert v['standalone:emg']==_typed_sha(b['standalone']['emg'].numpy())
    assert 'emg_channel_mask'not in b['fusion'] # Literal original collator omits it.


@pytest.mark.parametrize('damage',['fusion_emg','standalone_emg','target','crop_mask','dtype','hand','keys','claimed_hash','extra_tensor'])
def test_poisoned_collated_tensors_reject_even_when_self_reported_row_hash_is_unchanged(damage):
    p,r=samples();b=batch(p[:3])
    if damage=='fusion_emg':b['fusion']['emg'][0,0,0]+=1
    elif damage=='standalone_emg':b['standalone']['emg']=b['fusion']['emg']
    elif damage=='target':b['fusion']['joint_angles'][0,0,0]+=1
    elif damage=='crop_mask':b['fusion']['vision_valid_mask'][0]=~b['fusion']['vision_valid_mask'][0]
    elif damage=='dtype':b['fusion']['vision_img']=b['fusion']['vision_img'].double()
    elif damage=='hand':b['fusion']['target_hand'][0]='right'
    elif damage=='keys':b['keys'][0]['center_idx']+=1
    elif damage=='claimed_hash':b['verified_input_sha256'][0]='a'*64
    else:b['fusion']['vision_features']=torch.zeros((3,128))
    with pytest.raises(ValueError):api().verify_qualitative_collated_batch(b,r[:3])


def test_saved_source_packets_join_both_group_tails_without_dropping_false_masks(tmp_path):
    p,r,records,arrays=saved(tmp_path)
    value=api().read_qualitative_batches(tmp_path,records,expected_records=r)
    for name in arrays[0]:np.testing.assert_array_equal(value['arrays'][name],np.concatenate([a[name]for a in arrays]))
    assert value['counts']==dict(points=5,batches=2,crop_missing=3,label_invalid=2)
    assert not value['trained_source_admitted']


@pytest.mark.parametrize('damage',['missing_batch','extra_batch','reordered_batch','source_order','row_hash','changed_target','changed_mask','payload_bytes'])
def test_changed_source_or_resealed_raw_packets_do_not_pass_the_original_input_join(tmp_path,damage):
    p,r,records,arrays=saved(tmp_path)
    if damage=='missing_batch':records.pop()
    elif damage=='extra_batch':records.append(copy.deepcopy(records[-1]))
    elif damage=='reordered_batch':records.reverse()
    elif damage=='source_order':r[0],r[1]=r[1],r[0]
    elif damage=='row_hash':
        records[0]['inputs_sha256']['row:0']='a'*64
        v=records[0];v['sha256']=_hash({k:x for k,x in v.items()if k!='sha256'})
    elif damage=='payload_bytes':
        path=tmp_path/records[0]['file'];path.write_bytes(path.read_bytes()+b'x')
    else:
        # A fully internally valid, freshly resealed packet must still join its
        # original target/mask, not the writer's new self-consistent assertion.
        a=arrays[0]
        if damage=='changed_target':
            a['targets'][0,0]+=1
            for role in('vision','emg','fusion'):
                a['mae_'+role+'_rad']=np.array([float((torch.from_numpy(a[role+'_prediction'][i])-torch.from_numpy(a['targets'][i])).abs().mean())for i in range(3)],np.float64)
        else:a['label_valid_mask'][0]=~a['label_valid_mask'][0]
        (tmp_path/records[0]['file']).unlink()
        records[0]=write_dense_batch(tmp_path,0,keys=[x['key']for x in p[:3]],arrays=a,inputs_sha256=records[0]['inputs_sha256'])
    with pytest.raises(ValueError):api().read_qualitative_batches(tmp_path,records,expected_records=r)


def test_unregistered_input_or_evidence_cannot_start_any_native_model(tmp_path,monkeypatch):
    def denied(*a,**k):raise AssertionError('unregistered data attempted model construction')
    monkeypatch.setattr(torch.nn.Module,'__init__',denied)
    with pytest.raises(ValueError,match='registered'):api().run_qualitative_pass(object(),'/missing',tmp_path/'out')
    with pytest.raises(ValueError,match='registered'):api().read_qualitative_pass(tmp_path,evidence=object())
