"""New image/pose/error assembly and one shared display view per sample."""
import copy
import json

import numpy as np
import pytest
import torch

from test_occlusion_dense_data import UPSTREAM
from egoemg_repro.classic_emg import _hash
from egoemg_repro.occlusion_dense_data import _typed_sha


def api():
    from egoemg_repro import qualitative_render
    return qualitative_render


def fixture():
    key=dict(sequence=0,split='user',hand='left',ep_idx=0,center_idx=3895,
        window_start=0,window_end=7790,native_index=0,group_index=0)
    rgb=np.zeros((256,256,3),np.uint8);rgb[:,:128,0]=255;rgb[:,128:,2]=255
    image=rgb.transpose(2,0,1).astype(np.float32)
    mean=255*np.array([.485,.456,.406],np.float32);std=255*np.array([.229,.224,.225],np.float32)
    for c in range(3):image[c]=(image[c]-mean[c])/std[c]
    target=np.linspace(.05,.7,22,dtype=np.float32)
    f=dict(emg=np.zeros((16,7790),np.float32),emg_channel_mask=np.ones(16,bool),
        joint_angles=target[:,None].copy(),label_valid_mask=np.array([False],bool),
        vision_img=image,vision_valid_mask=np.array([True],bool),vision_frame_indices=np.array([17],np.int64))
    e=dict(emg=np.zeros((8,7790),np.float32),emg_channel_mask=np.ones(8,bool),
        joint_angles=target[:,None].copy(),label_valid_mask=np.array([False],bool))
    sample=dict(key=copy.deepcopy(key),fusion=f,standalone_emg=e)
    reseal(sample)
    arrays=dict(targets=target,vision_prediction=target+np.float32(.4),fusion_prediction=target+np.float32(.1),
        emg_prediction=target-np.float32(.2),label_valid_mask=np.array([False],bool),crop_valid_mask=np.array([True],bool))
    arrays['emg_full_prediction']=np.repeat(arrays['emg_prediction'][:,None],146,axis=1)
    for role in('vision','emg','fusion'):
        arrays['mae_'+role+'_rad']=np.float64(float((torch.from_numpy(arrays[role+'_prediction'])-torch.from_numpy(target)).abs().mean()))
    row=dict(key=copy.deepcopy(key),**key,source_filter_pass=True,crop_valid=True,label_valid=False,
        mocap_tracked=True,image_stale=False,markers_valid=True,
        **{'mae_'+role:float(arrays['mae_'+role+'_rad'])for role in('vision','emg','fusion')})
    row['delta_mae']=row['mae_vision']-row['mae_fusion']
    return sample,dict(key=copy.deepcopy(key),arrays=arrays),row,rgb


def reseal(sample):
    typed={'fusion:'+n:_typed_sha(sample['fusion'][n])for n in('emg','emg_channel_mask','joint_angles','label_valid_mask','vision_img','vision_valid_mask','vision_frame_indices')}
    typed.update({'standalone:'+n:_typed_sha(sample['standalone_emg'][n])for n in('emg','emg_channel_mask','joint_angles','label_valid_mask')})
    sample.update(typed_input_sha256=typed,verified_input_sha256=_hash(typed))


# Catches swapping any modality, dropping the two numeric wrist angles,
# mirroring an already canonical left RGB, or rendering with independent views.
def test_actual_pose_mapping_and_lossless_canonical_rgb_are_bound_to_one_native_key():
    s,p,r,rgb=fixture();before=copy.deepcopy((s,p,r))
    panel=api().assemble_qualitative_panel(s,p,r,upstream=UPSTREAM)
    np.testing.assert_array_equal(panel['rgb'],rgb)
    for role,field in(('gt','targets'),('emg','emg_prediction'),('vision','vision_prediction'),('fusion','fusion_prediction')):
        np.testing.assert_array_equal(panel['poses'][role],p['arrays'][field])
        assert panel['meshes'][role]['source_pose22_sha256']==_typed_sha(p['arrays'][field])
    m=panel['metadata'];assert m['key']==s['key']and m['label_valid']is False
    assert m['numeric_error_angles']==22 and m['mesh_finger_angles']==20
    assert m['errors_rad']=={role:float(p['arrays']['mae_'+role+'_rad'])for role in('emg','vision','fusion')}
    assert m['errors_deg']=={role:float(p['arrays']['mae_'+role+'_rad'])*57.3 for role in('emg','vision','fusion')}
    assert set(m['shared_view'])>={'elevation','azimuth','projection','limits'}
    assert not m['physical_rgb_projection_proven']and not m['trained_source_admitted']
    assert s['key']==before[0]['key']and r==before[2]
    for name in p['arrays']:np.testing.assert_array_equal(p['arrays'][name],before[1]['arrays'][name])


@pytest.mark.parametrize('damage',['sample_key','prediction_key','row_key','target','label_mask','crop_mask','missing_crop','untracked_row','saved_error','rank_error','midpoint','missing_emg','poisoned_emg','non_image_rgb'])
def test_panel_does_not_join_another_point_or_repair_missing_or_changed_inputs(damage):
    s,p,r,_=fixture()
    if damage=='sample_key':s['key']['center_idx']+=1
    elif damage=='prediction_key':p['key']['hand']='right'
    elif damage=='row_key':r['key']['window_start']+=1
    elif damage=='target':p['arrays']['targets'][0]+=1
    elif damage=='label_mask':p['arrays']['label_valid_mask'][0]=True
    elif damage=='crop_mask':p['arrays']['crop_valid_mask'][0]=False
    elif damage=='missing_crop':s['fusion']['vision_valid_mask'][0]=False;reseal(s)
    elif damage=='untracked_row':r['source_filter_pass']=False;r['mocap_tracked']=False
    elif damage=='saved_error':p['arrays']['mae_fusion_rad']+=.001
    elif damage=='rank_error':r['mae_vision']+=.001
    elif damage=='midpoint':p['arrays']['emg_full_prediction'][0,73]+=1
    elif damage=='missing_emg':p['arrays'].pop('emg_prediction')
    elif damage=='poisoned_emg':s['standalone_emg']['emg'][0,0]+=1
    else:s['fusion']['vision_img'][0,0,0]+=.001;reseal(s)
    with pytest.raises(ValueError):api().assemble_qualitative_panel(s,p,r,upstream=UPSTREAM)


def test_render_saves_the_bound_pixels_meshes_and_error_metadata_without_overwrite(tmp_path):
    from PIL import Image
    s,p,r,_=fixture();panel=api().assemble_qualitative_panel(s,p,r,upstream=UPSTREAM)
    result=api().render_qualitative_panel(panel,tmp_path/'sample.png',caption='Candidate 1')
    assert Image.open(tmp_path/'sample.png').size==(2160,512)
    assert result['panel_metadata']==panel['metadata']
    assert result['file_sha256']and result['png_pixels_sha256']
    assert not result['trained_source_admitted']
    with pytest.raises(FileExistsError):api().render_qualitative_panel(panel,tmp_path/'sample.png')


def test_late_pose_or_mesh_mutation_cannot_be_drawn_under_old_metadata(tmp_path):
    s,p,r,_=fixture();panel=api().assemble_qualitative_panel(s,p,r,upstream=UPSTREAM)
    panel['meshes']['vision']['vertices'][0,0]+=.01
    with pytest.raises(ValueError):api().render_qualitative_panel(panel,tmp_path/'invalid.png')


def test_candidate_export_requires_real_registered_inputs_before_file_or_model_io(tmp_path):
    with pytest.raises(ValueError,match='registered'):
        api().export_qualitative_candidates(object(),'/missing','/missing2',flags=object(),output=tmp_path/'out',limit=1)
