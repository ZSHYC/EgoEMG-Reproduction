"""The new actual-file flag adapter must keep all original selected keys."""
import copy

import numpy as np
import pytest


def api():
    from egoemg_repro import qualitative_flags
    return qualitative_flags


def data():
    n=8*7790;centers=[i*7790+3895 for i in range(6)]
    keys=[dict(sequence=i,split=('user','gesture','both')[i//2],hand='left'if i%2==0 else'right',
        ep_idx=0,center_idx=c,window_start=c-3895,window_end=c-3895+7790,group_index=i//2,native_index=0)
        for i,c in enumerate(centers)]
    fields=dict(mocap_webcam_tracked=np.ones(n,bool),image_webcam_stale=np.zeros(n,bool),
        mocap_left_valid=np.zeros((n,21),bool),mocap_right_valid=np.zeros((n,21),bool))
    fields['mocap_webcam_tracked'][centers[2]]=False;fields['image_webcam_stale'][centers[3]]=True
    fields['mocap_left_valid'][centers[0],0]=True;fields['mocap_right_valid'][centers[1],20]=True
    fields['mocap_left_valid'][centers[2],1]=True;fields['mocap_right_valid'][centers[3],2]=True
    fields['mocap_right_valid'][centers[5]]=True
    return keys,fields


def test_any_marker_is_per_actual_hand_and_stale_tracking_flags_do_not_drop_or_reorder():
    keys,fields=data();saved=copy.deepcopy((keys,fields));rows=api().extract_qualitative_flags(keys,fields)
    assert len(rows)==6
    assert [r['mocap_tracked']for r in rows]==[True,True,False,True,True,True]
    assert [r['image_stale']for r in rows]==[False,False,False,True,False,False]
    assert [r['markers_valid']for r in rows]==[True,True,True,True,False,True]
    assert [r['sequence']for r in rows if r['mocap_tracked']and not r['image_stale']and r['markers_valid']]==[0,1,5]
    for k,row in zip(keys,rows):
        for name in('sequence','split','hand','ep_idx','center_idx','window_start','window_end'):assert row[name]==k[name]
    assert keys==saved[0]
    for n,a in fields.items():np.testing.assert_array_equal(a,saved[1][n])


@pytest.mark.parametrize('damage',['missing','extra_label_mask','not_bool','wrong_markers','wrong_total','center_outside','duplicate','wrong_hand','out_of_order'])
def test_incomplete_flags_or_invalid_key_cannot_default_to_false_or_change_population(damage):
    k,a=data()
    if damage=='missing':a.pop('mocap_left_valid')
    elif damage=='extra_label_mask':a['generated_label_valid']=np.zeros((len(a['image_webcam_stale']),2),bool)
    elif damage=='not_bool':a['mocap_right_valid']=a['mocap_right_valid'].astype(np.uint8)
    elif damage=='wrong_markers':a['mocap_left_valid']=a['mocap_left_valid'][:,:20]
    elif damage=='wrong_total':a['image_webcam_stale']=a['image_webcam_stale'][:-1]
    elif damage=='center_outside':k[-1]['center_idx']=len(a['image_webcam_stale'])
    elif damage=='duplicate':k[-1]=copy.deepcopy(k[-2]);k[-1]['sequence']=5
    elif damage=='wrong_hand':k[0]['hand']='unknown'
    else:k[0],k[1]=k[1],k[0]
    with pytest.raises(ValueError):api().extract_qualitative_flags(k,a)


def test_unregistered_source_is_rejected_before_reading_a_manifest(tmp_path):
    with pytest.raises(ValueError,match='registered'):
        api().admit_qualitative_flags(object(),data=tmp_path/'missing',geometry=tmp_path/'missing')
