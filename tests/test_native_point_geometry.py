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


def engineering_output(tmp_path,monkeypatch,zero=False):
    """Synthetic integration packet; it grants no native execution evidence."""
    from egoemg_repro.classic_emg_cohort import _source_paths
    m=api();files=point_files(tmp_path,m,monkeypatch,'zero_left_pose'if zero else None);data=load(m,files)
    spec=importlib.util.spec_from_file_location('engineering_original_kernel',RUNNER.parents[2]/'upstream/egoemg/occlusion.py')
    kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
    class Native:
        def get_mano_verts_world(self,pose,beta,R,tr,flip,device):
            if not np.any(pose):return None,None
            cam=np.ones((778,3),np.float64);cam[:,:2]=.01;cam[0,0]=.02;cam[1,1]=.02
            return cam+tr,np.tile(np.array([0,1,2],np.uint32),(1538,1))
        def get_episode_intrinsics(self,*args):
            active_x0=(1920.-1080.*(3840/3360))/2.;active_x1=active_x0+1080.*(3840/3360)
            K=np.eye(3);K[0,0]=1696.*(active_x1-active_x0)/3840.;K[1,1]=1697.*1080./3360.
            K[0,2]=1910.*(active_x1-active_x0)/3840.+active_x0;K[1,2]=1669.*1080./3360.
            return K,1920,1080
        def compute_self_occlusion(self,cam,faces,K,h,w,**kwargs):
            u,v,valid=kernel._project_vertices(cam,K);weights=kernel.compute_vertex_area_weights(cam,faces)
            return dict(occlusion_score=0.,visible_ratio=1.,visible=np.ones(778,bool),area_weights=weights,
                u_proj=np.round(u).astype(np.int32),v_proj=np.round(v).astype(np.int32),zbuffer=np.ones((1080,1920)))
    assets=tmp_path/'assets';assets.mkdir();(assets/'MANO_RIGHT.pkl').write_bytes(b'engineering model placeholder')
    (assets/'LICENSE.txt').write_bytes(b'engineering license placeholder')
    ch=tmp_path/'chumpy';ch.mkdir();(ch/'__init__.py').write_text('# engineering boundary fixture\n')
    sm=tmp_path/'smplx';sm.mkdir();(sm/'__init__.py').write_text('# engineering boundary fixture\n')
    monkeypatch.setattr(m,'MANO_ASSET_SHA',sha(assets/'MANO_RIGHT.pkl'),raising=False)
    monkeypatch.setattr(m,'MANO_LICENSE_SHA',sha(assets/'LICENSE.txt'),raising=False)
    upstream=(RUNNER.parents[2]/'upstream').resolve()
    paths={*_source_paths(upstream),upstream/'scripts/eval/evaluate_occlusion_fusion.py',RUNNER,*ch.glob('**/*.py'),*sm.glob('**/*.py'),*assets.iterdir()}
    sources={str(p):sha(p)for p in sorted(paths)}
    runtime=dict(upstream=str(upstream),upstream_revision=m.UPSTREAM_REVISION,mano_assets=str(assets),
        chumpy_dir=str(ch.parent),smplx_dir=str(sm),source_files_sha256=sources,source_bundle_sha256=m._json_sha(sources),
        mano_asset_sha256=sha(assets/'MANO_RIGHT.pkl'),original_model_source_files_modified=False,
        actual_right_mano_geometry_layers_constructed=1,versions=dict(torch='engineering',numpy=np.__version__,chumpy='0.70'),
        numpy_aliases=[],inspect_getargspec_backfilled=False)
    policy=m.validate_resolution_policy(m.RESOLUTION_POLICY)
    geometry_policy=dict(right_mano_for_both_hands=True,flat_hand_mean=False,global_orient='native zero',
        hand_pose_slice=[3,48],left_mirror_axis='X',left_faces_reverse=True,
        legacy_manifest_convention='flip_local_z; source script X mirror retained and not claimed reconciled',
        depth_eps=.005,window_half=2,actual_betas=True,zero_beta_fallback=False)
    mode='complete_role'if zero else'bilateral_probe';selected=m.select_points(data,mode)
    identity=dict(format='original_native_point_geometry_identity_v1',role='test',mode=mode,
        source_runner_sha256=m.IMPORTED_SHA,runtime=runtime,input_paths={k:str(p)for k,p in zip(('points','admission','metadata','calibration'),files)},
        input_sha256=data['files_sha256'],input_token=data['_token'],selection=selected,resolution_policy=policy,
        geometry_policy=geometry_policy,scientific_eligible=False,full_paper_complete=False)
    output=tmp_path/'output';output.mkdir();(output/'points').mkdir();rows=[]
    identity['sha256']=m._json_sha(identity)
    for seq,(hand,index)in enumerate(selected):
        row,arrays=m.geometry_frame(Native(),data,hand,index,video_root=tmp_path,device='cpu')
        if arrays is not None:
            packet=output/'points'/f'{seq:06d}-{hand}-{row["center_idx"]}.npz';np.savez_compressed(packet,**arrays)
            row.update(packet_file=str(packet.relative_to(output)),packet_file_sha256=m.file_sha(packet))
        row.update(sequence=seq,identity_sha256=identity['sha256']);rows.append(row)
    invalid=sum(r['status']=='invalid_zero_pose'for r in rows)
    result=dict(format='original_native_point_geometry_result_v1',identity_sha256=identity['sha256'],role='test',mode=mode,
        rows=len(rows),actual_mesh_calls=len(rows),actual_occlusion_calls=len(rows)-invalid,invalid_zero_pose_rows=invalid,
        complete_standard_role=zero,actual_right_mano_geometry_layers_constructed=1,learned_pose_models_constructed=0,
        optimizer_updates=0,cuda_initialized=False,resolution_policy=policy,all_input_and_source_generations_unchanged=True,
        seconds=.1,dense_grid_or_paired_gain_complete=False,scientific_eligible=False,full_paper_complete=False)
    def seal():
        identity.pop('sha256',None);identity['sha256']=m._json_sha(identity)
        for r in rows:r['identity_sha256']=identity['sha256']
        result['identity_sha256']=identity['sha256'];m._write_json(output/'identity.json',identity)
        (output/'points.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        result['journal_file_sha256']=m.file_sha(output/'points.jsonl');result.pop('sha256',None);result['sha256']=m._json_sha(result)
        m._write_json(output/'result.json',result)
    seal();return m,output,identity,result,rows,seal


def test_engineering_packet_baseline_needs_no_torch(tmp_path,monkeypatch):
    import builtins
    m,out,identity,result,rows,seal=engineering_output(tmp_path,monkeypatch)
    original=builtins.__import__
    def forbid(name,*a,**kw):
        if name=='torch'or name.startswith('torch.'):raise AssertionError('verification cannot import Torch')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',forbid)
    assert m.verify_geometry(out)==result


@pytest.mark.parametrize('damage',['K_nan','K_changed','projection','face_dtype','face_index','vertex_nan','zbuffer_nan','zbuffer_negative','extra_array','missing_array'])
def test_resealed_packet_must_bind_native_schema_intrinsics_and_projection(tmp_path,monkeypatch,damage):
    m,out,identity,result,rows,seal=engineering_output(tmp_path,monkeypatch)
    assert m.verify_geometry(out)==result
    packet=out/rows[0]['packet_file']
    with np.load(packet)as f:arrays={k:f[k]for k in f.files}
    if damage=='K_nan':arrays['K']=np.array([np.nan])
    elif damage=='K_changed':arrays['K'][0,0]+=1
    elif damage=='projection':
        arrays['u_proj'][:]=-1;arrays['v_proj'][:]=-1;arrays['visible'][:]=False
        rows[0].update(visible_ratio=0.,occlusion_score=1.,author_visible_ratio_round6=0.,author_occlusion_score_round6=1.,visible_vertices=0)
    elif damage=='face_dtype':arrays['faces']=arrays['faces'].astype(np.int64)
    elif damage=='face_index':arrays['faces'][0,0]=778
    elif damage=='vertex_nan':arrays['verts_world'][0,0]=np.nan
    elif damage=='zbuffer_nan':arrays['zbuffer'][0,0]=np.nan
    elif damage=='zbuffer_negative':arrays['zbuffer'][0,0]=-1.
    elif damage=='extra_array':arrays['unbound_extra']=np.zeros(1)
    elif damage=='missing_array':del arrays['K']
    np.savez_compressed(packet,**arrays);rows[0]['arrays_sha256']={k:m._array_sha(v)for k,v in arrays.items()}
    rows[0]['packet_file_sha256']=m.file_sha(packet);seal()
    with pytest.raises(ValueError):m.verify_geometry(out)


@pytest.mark.parametrize('target,key,value',[
    ('identity','format','other'),('result','format','other'),('result','role','val'),('result','mode','complete_role'),
    ('identity','scientific_eligible',True),('identity','full_paper_complete',True),('result','dense_grid_or_paired_gain_complete',True),
    ('result','all_input_and_source_generations_unchanged',False),('result','actual_right_mano_geometry_layers_constructed',0),
    ('result','optimizer_updates',False),('result','rows',True),('runtime','source_files_sha256',{}),
    ('runtime','upstream_revision','wrong'),('runtime','source_bundle_sha256','wrong'),('runtime','mano_asset_sha256','wrong'),
    ('runtime','original_model_source_files_modified',True),('runtime','actual_right_mano_geometry_layers_constructed',0),
    ('geometry','depth_eps',.01),('identity_policy','actual_episode_video_dimensions_proven',True),
    ('result_policy','width',1),('result_policy','original_intrinsics_scaling_unchanged',False),('result_policy','distortion_applied',True),
    ('row','mocap_tracked',False),('row','image_stale',True),('row','markers_valid',False),('row','marker_valid_count',0),
    ('row','image_frame_index',-999),('row','world_transform_sha256','wrong'),('row','camera_transform_sha256','wrong'),
    ('row','width',1),('row','height',1),('row','visible_vertices',0),('row','positive_depth_vertices',0),
    ('row','area_weight_sum',999.),('row','author_visible_ratio_round6',.5),('row','point_index',True)])
def test_resealed_false_provenance_scope_and_population_reject(tmp_path,monkeypatch,target,key,value):
    m,out,identity,result,rows,seal=engineering_output(tmp_path,monkeypatch)
    assert m.verify_geometry(out)==result
    obj=dict(identity=identity,result=result,runtime=identity['runtime'],geometry=identity['geometry_policy'],
        identity_policy=identity['resolution_policy'],result_policy=result['resolution_policy'],row=rows[0])[target]
    obj[key]=value;seal()
    with pytest.raises(ValueError):m.verify_geometry(out)


def test_zero_pose_rejects_native_statistics_even_when_resealed(tmp_path,monkeypatch):
    m,out,identity,result,rows,seal=engineering_output(tmp_path,monkeypatch,zero=True)
    assert m.verify_geometry(out)==result
    rows[0]['positive_depth_vertices']=778;seal()
    with pytest.raises(ValueError):m.verify_geometry(out)


def test_packet_guard_precedes_first_hash(tmp_path,monkeypatch):
    m,out,identity,result,rows,seal=engineering_output(tmp_path,monkeypatch);packet=out/rows[0]['packet_file']
    assert m.verify_geometry(out)==result
    original=m.file_sha;changed=[]
    def race(p):
        digest=original(p)
        if Path(p)==packet and not changed:
            with packet.open('ab')as f:f.write(b'changed after SHA read')
            changed.append(True)
        return digest
    monkeypatch.setattr(m,'file_sha',race)
    with pytest.raises(ValueError):m.verify_geometry(out)
    assert changed


def test_source_import_guard_precedes_execution_and_layer_creation(tmp_path,monkeypatch):
    from types import SimpleNamespace,ModuleType
    import egoemg_repro.classic_emg as emg
    import egoemg_repro.classic_emg_cohort as cohort
    m=api()
    # The production entry owns a separate process. This in-process source-race
    # fixture must restore its process-wide network guard on every exit.
    monkeypatch.setattr(m.socket.socket,'connect',m.socket.socket.connect)
    monkeypatch.setattr(m.socket,'create_connection',m.socket.create_connection)
    upstream=tmp_path/'upstream';source=upstream/'scripts/eval/evaluate_occlusion_fusion.py'
    source.parent.mkdir(parents=True);source.write_text('generation = "before import"\n')
    assets=tmp_path/'assets';assets.mkdir();(assets/'MANO_RIGHT.pkl').write_bytes(b'fixture');(assets/'LICENSE.txt').write_bytes(b'fixture')
    monkeypatch.setattr(m,'MANO_ASSET_SHA','45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767',raising=False)
    monkeypatch.setattr(m,'MANO_LICENSE_SHA','75fc4da03259504ce9a4ed2418ab98dd59647add7ff95e7691c4d55fb0e3cf63',raising=False)
    ch=tmp_path/'deps/chumpy';ch.mkdir(parents=True);(ch/'__init__.py').write_text('# fixture\n')
    sm=tmp_path/'smplx';sm.mkdir();(sm/'__init__.py').write_text('# fixture\n')
    c=ModuleType('chumpy');c.__file__=str(ch/'__init__.py');c.__version__='0.70'
    s=ModuleType('smplx');s.__file__=str(sm/'__init__.py')
    class MANO:faces=np.zeros((1538,3),np.uint32)
    s.MANO=MANO
    t=ModuleType('torch');t.__version__='inert';t.set_num_threads=lambda n:None;t.cuda=SimpleNamespace(is_initialized=lambda:False)
    class Optimizer:pass
    t.optim=SimpleNamespace(AdamW=Optimizer)
    for n,v in [('chumpy',c),('smplx',s),('torch',t)]:monkeypatch.setitem(sys.modules,n,v)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(emg,'_activate',lambda p:None)
    monkeypatch.setattr(cohort,'_source_paths',lambda p:[source])
    monkeypatch.setattr(m.subprocess,'check_output',lambda a,**kw:m.UPSTREAM_REVISION+'\n'if'rev-parse'in a else'')
    calls=[]
    def race(name,path):
        source.read_bytes();source.write_text('generation = "after import"\n')
        def layer(device):calls.append(True);return MANO()
        return SimpleNamespace(_MANO_LAYER=None,_get_mano_layer=layer,
            ResNetVisionPose=type('R',(),{}),VisionViTPose=type('V',(),{}),Emg2PoseFormer=type('E',(),{}))
    monkeypatch.setattr(m,'_load_source',race)
    # The current production constants are patched only for inert fixture assets.
    original_sha=m.file_sha
    def fixture_hash(p):
        if Path(p)==assets/'MANO_RIGHT.pkl':return getattr(m,'MANO_ASSET_SHA')
        if Path(p)==assets/'LICENSE.txt':return getattr(m,'MANO_LICENSE_SHA')
        return original_sha(p)
    monkeypatch.setattr(m,'file_sha',fixture_hash)
    with pytest.raises(ValueError):m._native_runtime(SimpleNamespace(upstream=upstream,mano_assets=assets,chumpy_dir=ch.parent))
    assert not calls
