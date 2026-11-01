"""Typed IO boundaries for real native geometry; synthetic fixtures grant none."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

np=pytest.importorskip('numpy')
RUNNER=Path(__file__).resolve().parents[1]/'scripts/paper/run_native_point_geometry.py'


def api():
    assert RUNNER.is_file(),'native point geometry runner is not implemented'
    name='native_point_geometry_under_test'
    spec=importlib.util.spec_from_file_location(name,RUNNER);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def point_files(tmp_path,m,monkeypatch,damage=None):
    metadata=tmp_path/'metadata.npz'
    np.savez(metadata,episode_start_idx=np.array([0,50],np.int64),episode_end_idx=np.array([50,100],np.int64),
        episode_length=np.array([50,50],np.int64),episode_beta_idx=np.array([0,1],np.int32),
        episode_webcam_video_path=np.array([b'episode0.mp4',b'episode1.mp4']))
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(camera_matrix=[[1696.,0.,1910.],[0.,1697.,1669.],[0.,0.,1.]],
        distortion_coefficients=[0.]*14,image_width=3840,image_height=3360)))
    rotations=np.tile(np.r_[np.eye(3).ravel(),[.1,.2,.3]],(2,1)).astype(np.float32)
    arrays=dict(center=np.array([10,60,10,60],np.int64),episode=np.array([0,1,0,1],np.int64),
        hand=np.array([0,0,1,1],np.int8),camera_transform=rotations.copy(),
        camera_tracked=np.array([False,True]),image_stale=np.array([False,False]),
        image_frame_index=np.array([3,7],np.int32))
    for hand,base in [('left',1.),('right',2.)]:
        arrays[hand+'_pose']=np.full((2,48),base,np.float32)
        arrays[hand+'_world_transform']=rotations.copy()
        arrays[hand+'_marker_valid']=np.ones((2,21),bool)
        arrays[hand+'_beta']=np.tile(np.arange(10,dtype=np.float32)+base,(2,1))
    if damage=='dtype':arrays['left_pose']=arrays['left_pose'].astype(np.float64)
    elif damage=='shape':arrays['right_pose']=arrays['right_pose'][:,:45]
    elif damage=='order':arrays['hand']=np.array([1,1,0,0],np.int8)
    elif damage=='centers':arrays['center'][2]+=1
    elif damage=='episode':arrays['episode'][0]=1
    elif damage=='transform_nan':arrays['camera_transform'][0,0]=np.nan
    elif damage=='beta_nan':arrays['left_beta'][0,0]=np.nan
    elif damage=='missing_beta':del arrays['right_beta']
    elif damage=='unknown_array':arrays['unbound_extra']=np.zeros(2)
    elif damage=='zero_left_pose':arrays['left_pose'][0]=0.
    points=tmp_path/'native-test-geometry.npz';np.savez_compressed(points,**arrays)
    value=dict(format='actual_legacy_complete_geometry_numeric_scan_v1',total_rows=100,episode_count=2,
        full_numeric_scan_complete=True,structural_admission_passed=True,scientific_eligible=False,full_paper_complete=False,
        metadata_sha256=sha(metadata),camera=dict(declared_calibration_resolution=[3840,3360],matrix=[[1696.,0.,1910.],[0.,1697.,1669.],[0.,0.,1.]],distortion_coefficient_count=14),
        point_payloads=dict(test=dict(file=points.name,sha256=sha(points),bytes=points.stat().st_size)),
        native_point_geometry_roles=dict(test=dict(unique_hand_samples=4,repeated_consumptions=8,by_hand=dict(left=dict(unique_hand_centers=2),right=dict(unique_hand_centers=2)))))
    admission=tmp_path/'admission.json';admission.write_text(json.dumps(value))
    monkeypatch.setattr(m,'ADMISSION_SHA',sha(admission));monkeypatch.setattr(m,'CALIBRATION_SHA',sha(calibration))
    monkeypatch.setattr(m,'ROLE_COUNTS',dict(val=4,test=4))
    return points,admission,metadata,calibration


def load(m,files,role='test'):return m.load_point_inputs(*files,role=role)


def test_typed_point_admission_preserves_actual_betas_and_cannot_import_torch(tmp_path,monkeypatch):
    import builtins
    m=api();files=point_files(tmp_path,m,monkeypatch);original=builtins.__import__
    def no_torch(name,*a,**k):
        if name=='torch'or name.startswith('torch.'):raise AssertionError('typed geometry data must not construct/import Torch')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',no_torch)
    data=load(m,files)
    assert data['role']=='test'and data['unique_hand_samples']==4
    np.testing.assert_array_equal(data['arrays']['left_beta'][0],np.arange(10,dtype=np.float32)+1)
    np.testing.assert_array_equal(data['arrays']['right_beta'][0],np.arange(10,dtype=np.float32)+2)
    assert all(not a.flags.writeable for a in data['arrays'].values())


@pytest.mark.parametrize('damage',['dtype','shape','order','centers','episode','transform_nan','beta_nan','missing_beta','unknown_array'])
def test_resealed_synthetic_point_packet_must_still_match_typed_native_schema(tmp_path,monkeypatch,damage):
    m=api();files=point_files(tmp_path,m,monkeypatch,damage)
    with pytest.raises(ValueError):load(m,files)


@pytest.mark.parametrize('target',['points','admission','metadata','calibration'])
def test_changed_input_bytes_reject_before_geometry_construction(tmp_path,monkeypatch,target):
    m=api();files=point_files(tmp_path,m,monkeypatch);p=files[['points','admission','metadata','calibration'].index(target)]
    p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError):load(m,files)


def test_complete_order_and_bilateral_probe_selection_are_explicit_and_deterministic(tmp_path,monkeypatch):
    m=api();data=load(m,point_files(tmp_path,m,monkeypatch))
    assert m.select_points(data,'complete_role')==[('left',0),('left',1),('right',0),('right',1)]
    assert m.select_points(data,'bilateral_probe')==[('left',1),('right',1)]
    for mode in (None,'first2','dense'):
        with pytest.raises(ValueError):m.select_points(data,mode)


@pytest.mark.parametrize('role',[None,'train','both'])
def test_unknown_role_rejects_before_file_io(role):
    with pytest.raises(ValueError):api().load_point_inputs(None,None,None,None,role=role)


def test_post_admission_memmap_style_dtype_or_input_file_mutations_reject(tmp_path,monkeypatch):
    m=api();files=point_files(tmp_path,m,monkeypatch);data=load(m,files)
    data['arrays']['left_pose'].dtype=np.int32
    with pytest.raises(ValueError):m.select_points(data,'complete_role')
    data=load(m,files);files[0].write_bytes(files[0].read_bytes()+b'new generation')
    with pytest.raises(ValueError):m.select_points(data,'bilateral_probe')


@pytest.mark.parametrize('policy',[None,'actual_video','1920x1080',''])
def test_resolution_policy_is_explicit_and_never_an_observed_video_claim(policy):
    m=api()
    with pytest.raises(ValueError):m.validate_resolution_policy(policy)


class MissingNativeMesh:
    def __init__(self):self.calls=[]
    def get_mano_verts_world(self,*args):
        self.calls.append(args);return None,None


def test_zero_pose_is_retained_and_native_mesh_none_is_not_a_fabricated_score(tmp_path,monkeypatch):
    m=api();data=load(m,point_files(tmp_path,m,monkeypatch,'zero_left_pose'));native=MissingNativeMesh()
    row,arrays=m.geometry_frame(native,data,'left',0,video_root=tmp_path,device='cpu')
    assert row['status']=='invalid_zero_pose'and row['occlusion_score']is None and arrays is None
    assert row['center_idx']==10 and row['hand']=='left'and len(native.calls)==1


@pytest.mark.parametrize('hand,base',[('left',1.),('right',2.)])
def test_valid_pose_native_failure_raises_with_actual_hand_beta_and_no_zero_substitute(tmp_path,monkeypatch,hand,base):
    m=api();data=load(m,point_files(tmp_path,m,monkeypatch));native=MissingNativeMesh()
    with pytest.raises(RuntimeError,match='native'):m.geometry_frame(native,data,hand,0,video_root=tmp_path,device='cpu')
    assert len(native.calls)==1
    np.testing.assert_array_equal(native.calls[0][1],np.arange(10,dtype=np.float32)+base)
    assert native.calls[0][4]is(hand=='left')


@pytest.mark.parametrize('hand,index',[(None,0),('middle',0),('left',-1),('right',999),('left',True)])
def test_frame_selector_rejects_unknown_hand_or_index_before_native_call(tmp_path,monkeypatch,hand,index):
    m=api();data=load(m,point_files(tmp_path,m,monkeypatch));native=MissingNativeMesh()
    with pytest.raises(ValueError):m.geometry_frame(native,data,hand,index,video_root=tmp_path,device='cpu')
    assert not native.calls


def test_run_rejects_implicit_resolution_before_any_runtime_or_input_io():
    from types import SimpleNamespace
    with pytest.raises(ValueError):api().run_geometry(SimpleNamespace(resolution_policy=None))


def test_verifier_rejects_partial_output_without_importing_a_runtime(tmp_path):
    with pytest.raises(ValueError):api().verify_geometry(tmp_path)


@pytest.mark.skipif(not os.environ.get('EGOEMG_GEOMETRY_NATIVE_TEST_REQUEST'),reason='licensed assets and authentic point request required')
def test_real_native_bilateral_geometry_and_raw_packet_corruption(tmp_path):
    from types import SimpleNamespace
    q=json.loads(Path(os.environ['EGOEMG_GEOMETRY_NATIVE_TEST_REQUEST']).read_text())
    q['output']=tmp_path/'native-probe';assert q['mode']=='bilateral_probe'
    m=api();result=m.run_geometry(SimpleNamespace(**q))
    assert result['actual_mesh_calls']==2 and result['actual_occlusion_calls']==2
    assert result['actual_right_mano_geometry_layers_constructed']==1
    assert result['learned_pose_models_constructed']==0 and result['optimizer_updates']==0 and not result['cuda_initialized']
    assert not result['complete_standard_role']and not result['scientific_eligible']
    assert m.verify_geometry(q['output'])==result
    point=next((q['output']/'points').glob('*.npz'));point.write_bytes(point.read_bytes()+b'changed packet')
    with pytest.raises(ValueError):m.verify_geometry(q['output'])
