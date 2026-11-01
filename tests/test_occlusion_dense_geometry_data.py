"""Real tiny numeric files; explicit synthetic reference pins, no asset/model."""
import builtins
import copy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.classic_emg import _hash
from egoemg_repro.occlusion_dense_data import COLUMNS,_typed_sha


def api():return importlib.import_module('egoemg_repro.occlusion_dense_geometry_data')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')


def values():
    n=100;meta=dict(episode_start_idx=np.array([0,50],np.int64),episode_end_idx=np.array([50,100],np.int64),
        episode_length=np.array([50,50],np.int64),episode_beta_idx=np.array([0,1],np.int64),
        episode_id=np.array(['a','b']),episode_webcam_video_path=np.array(['a.mp4','b.mp4']))
    raw=dict(sequence=[0,1,2,3],episode=[0,0,1,1],start=[6,6,56,56],end=[14,14,64,64],center=[10,10,60,60],
        split_id=[1,1,3,3],hand_index=[0,1,0,1],native_index=[0]*4,group_index=[0,0,2,2])
    points={k:np.asarray(raw[k],dtype=d)for k,d in COLUMNS.items()}
    fields={};identity=np.r_[np.eye(3).reshape(-1),[0,0,0]].astype(np.float32)
    for hi,h in enumerate(('left','right')):
        pose=np.full((n,48),.1+hi,np.float32);pose[10]=0
        fields['generated_mano_'+h+'_pose']=pose
        fields['mocap_mano_'+h+'_world_transform']=np.tile(identity,(n,1))
        fields['mocap_'+h+'_valid']=np.ones((n,21),bool);fields['mocap_'+h+'_valid'][10]=False
    fields['mocap_webcam_transform']=np.tile(identity,(n,1));fields['mocap_webcam_tracked']=np.ones(n,bool)
    fields['mocap_webcam_tracked'][10]=False
    fields['image_webcam_stale']=np.zeros(n,bool);fields['image_webcam_stale'][10]=True
    fields['image_webcam_frame_index']=np.arange(n,dtype=np.int32);fields['image_webcam_frame_index'][10]=-1
    betas={h:np.full((2,10),hi+1,np.float32)for hi,h in enumerate(('left','right'))}
    betas['left'][1]=3;betas['right'][1]=4
    return points,fields,betas,meta


def fixture(tmp_path,monkeypatch):
    m=api();points,fields,betas,meta=values()
    data=tmp_path/'data';geo=tmp_path/'geo';index=tmp_path/'index'
    for p in(data,geo,geo/'meta',index):p.mkdir()
    np.savez(data/'metadata.npz',**meta)
    split=np.zeros(100,np.int8);split[10]=1;split[60]=3;split.tofile(data/'frame_split_id.dat')
    spec={};bulk=[]
    for name,a in fields.items():
        parent=data if name.startswith('image_')else geo
        a.tofile(parent/(name+'.dat'));spec[name]=dict(filename=name+'.dat',dtype=a.dtype.name,shape=list(a.shape))
        if parent==geo:bulk.append(dict(spec[name],field=name,sha256=sha(parent/(name+'.dat')),
            actual_full_sha256=sha(parent/(name+'.dat')),expected_bytes=a.nbytes))
    spec['frame_split_id']=dict(filename='frame_split_id.dat',dtype='int8',shape=[100])
    manifest=dict(total_rows=100,num_episodes=2,frame_split_labels=['train','user','gesture','both'],fields=spec)
    write(data/'manifest.json',manifest)
    for h,a in betas.items():a.tofile(geo/'meta'/('generated_mano_'+h+'_beta.dat'))
    calib=dict(image_width=3840,image_height=3360,camera_matrix=[[1696.,0.,1910.],[0.,1697.,1669.],[0.,0.,1.]],
        distortion_coefficients=[0.]*14)
    write(geo/'meta/GX010023_standard_calibration.json',calib)
    gr=dict(format='actual_legacy_complete_geometry_numeric_scan_v1',full_numeric_scan_complete=True,
        structural_admission_passed=True,actual_beta_fields_used=True,zero_beta_fallback_used=False,
        scientific_eligible=False,full_paper_complete=False,total_rows=100,episode_count=2,
        manifest_sha256=sha(data/'manifest.json'),metadata_sha256=sha(data/'metadata.npz'),bulk_files=bulk,
        episode_beta_index_equals_episode_index=True,episode_intervals_contiguous_complete=True,
        camera=dict(matrix=calib['camera_matrix'],declared_calibration_resolution=[3840,3360],distortion_coefficient_count=14))
    inputs={n:sha(data/f)for n,f in dict(manifest='manifest.json',metadata='metadata.npz',frame_split_id='frame_split_id.dat').items()}
    ref=dict(format='actual_complete_native_dense_index_admission_verified_v1',actual_exit_code=0,
        actual_complete_index_passes=4,all_ten_original_owners_closed=True,exact_typed_point_arrays_and_identities_rebuilt=True,
        full_paper_complete=False,recipes={})
    for recipe in('vits_s','rn18_s'):
        p=index/('private/'+recipe+'-a');p.mkdir(parents=True)
        i=dict(format='original_dense_occlusion_complete_index_v1',protocol=dict(recipe=recipe,window_length=8,dense_stride=2),
            total_rows=100,episode_records=2,unique_hand_points=4,input_files_sha256=inputs,
            arrays_sha256={k:_typed_sha(a)for k,a in points.items()})
        i['sha256']=_hash(i);write(p/'identity.json',i);np.savez(p/'points.npz',**points)
        ref['recipes'][recipe]=dict(unique_hand_points=4,episode_records=2,total_rows=100,window_length=8,dense_stride=2,
            identity_sha256=i['sha256'],identity_files_sha256=[sha(p/'identity.json')]*2,
            point_files_sha256=[sha(p/'points.npz')]*2,input_files_sha256=inputs)
    grpath=tmp_path/'geometry-reference.json';irpath=tmp_path/'index-reference.json';write(grpath,gr);write(irpath,ref)
    monkeypatch.setattr(m,'GEOMETRY_REFERENCE_SHA',sha(grpath));monkeypatch.setattr(m,'INDEX_REFERENCE_SHA',sha(irpath))
    monkeypatch.setattr(m,'CALIBRATION_SHA',sha(geo/'meta/GX010023_standard_calibration.json'))
    monkeypatch.setattr(m,'BETA_SHA',{h:sha(geo/'meta'/('generated_mano_'+h+'_beta.dat'))for h in('left','right')})
    return dict(data=data,geometry=geo,index_root=index,geometry_reference=grpath,index_reference=irpath)


def test_pure_selection_uses_actual_episode_beta_and_keeps_zero_flags_and_native_order():
    m=api();p,f,b,md=values();md['episode_beta_idx']=np.array([1,0],np.int64)
    a,s=m.extract_dense_point_arrays(p,f,b,md)
    assert s['sequence'].tolist()==[0,1,2,3]and s['pool_index'].tolist()==[0,0,1,1]
    assert a['center'].tolist()==[10,60,10,60]and a['hand'].tolist()==[0,0,1,1]
    np.testing.assert_array_equal(a['left_beta'][:,0],[3.,1.]);np.testing.assert_array_equal(a['right_beta'][:,0],[4.,2.])
    assert not a['left_pose'][0].any()and a['image_stale'].tolist()==[True,False]
    assert a['camera_tracked'].tolist()==[False,True]and a['image_frame_index'].tolist()==[-1,60]
    assert not a['left_marker_valid'][0].any()and a['left_marker_valid'][1].all()
    assert all(not z.flags.writeable for z in(*a.values(),*s.values()))


@pytest.mark.parametrize('damage',['unpaired','point_dtype','pose_dtype','pose_nan','camera_singular','beta_missing',
    'beta_nan','beta_index','center_episode','marker_shape','frame_dtype'])
def test_typed_extraction_cannot_repair_or_drop_bad_geometry(damage):
    m=api();p,f,b,md=values();m.extract_dense_point_arrays(p,f,b,md)
    if damage=='unpaired':p['hand_index'][1]=0
    elif damage=='point_dtype':p['center']=p['center'].astype(np.float64)
    elif damage=='pose_dtype':f['generated_mano_left_pose']=f['generated_mano_left_pose'].astype(np.float64)
    elif damage=='pose_nan':f['generated_mano_right_pose'][60,0]=np.nan
    elif damage=='camera_singular':f['mocap_webcam_transform'][60,:9]=0
    elif damage=='beta_missing':del b['right']
    elif damage=='beta_nan':b['left'][1,0]=np.nan
    elif damage=='beta_index':md['episode_beta_idx'][1]=99
    elif damage=='center_episode':p['episode'][2:]=0
    elif damage=='marker_shape':f['mocap_left_valid']=f['mocap_left_valid'][:,:20]
    else:f['image_webcam_frame_index']=f['image_webcam_frame_index'].astype(np.int64)
    with pytest.raises(ValueError):m.extract_dense_point_arrays(p,f,b,md)


def test_full_live_admission_and_source_order_native_pool_are_bound(tmp_path,monkeypatch):
    m=api();kw=fixture(tmp_path,monkeypatch);v=m.admit_dense_geometry_inputs(**kw);v.check()
    assert v.identity['native_geometry_executed']is False and v.identity['learned_pose_models_constructed']==0
    for recipe in('vits_s','rn18_s'):
        d=v.data_by_recipe[recipe];s=v.selection_by_recipe[recipe]
        assert d['unique_hand_samples']==4 and s['pool_index'].tolist()==[0,0,1,1]
        assert d['arrays']['left_beta'][1,0]==3. and d['arrays']['right_beta'][1,0]==4.
        assert d['role']=='native_dense_'+recipe


@pytest.mark.parametrize('damage',['raw_bulk','beta_bytes','calibration','index_packet','identity','field_link','data_link','metadata'])
def test_bound_full_sources_reject_byte_drift_or_symlinks(tmp_path,monkeypatch,damage):
    m=api();kw=fixture(tmp_path,monkeypatch);m.admit_dense_geometry_inputs(**kw).check()
    if damage=='raw_bulk':p=kw['geometry']/'generated_mano_left_pose.dat'
    elif damage=='beta_bytes':p=kw['geometry']/'meta/generated_mano_right_beta.dat'
    elif damage=='calibration':p=kw['geometry']/'meta/GX010023_standard_calibration.json'
    elif damage=='index_packet':p=kw['index_root']/'private/vits_s-a/points.npz'
    elif damage=='identity':p=kw['index_root']/'private/rn18_s-a/identity.json'
    elif damage=='metadata':p=kw['data']/'metadata.npz'
    elif damage=='field_link':
        p=kw['geometry']/'generated_mano_left_pose.dat';p.rename(p.with_suffix('.original'));p.symlink_to(p.with_suffix('.original'))
    else:
        alias=tmp_path/'data-alias';alias.symlink_to(kw['data'],target_is_directory=True);kw['data']=alias
    if damage not in('field_link','data_link'):
        with p.open('r+b')as f:f.write(b'X')
    with pytest.raises(ValueError):m.admit_dense_geometry_inputs(**kw)


def test_forged_reference_rejects_before_any_private_io(tmp_path):
    m=api();g=tmp_path/'forged.json';write(g,dict(full_numeric_scan_complete=True,all_ten_original_owners_closed=True))
    with pytest.raises(ValueError):m.admit_dense_geometry_inputs(tmp_path/'missing',tmp_path/'missing',tmp_path/'missing',geometry_reference=g,index_reference=g)


@pytest.mark.parametrize('damage',['unregistered_clone','native_token','selection_reseal','identity_reseal','source_generation'])
def test_live_registry_and_raw_generation_prevent_resealed_admission(tmp_path,monkeypatch,damage):
    m=api();kw=fixture(tmp_path,monkeypatch);v=m.admit_dense_geometry_inputs(**kw);v.check()
    if damage=='unregistered_clone':v=replace(v)
    elif damage=='native_token':v.data_by_recipe['vits_s']['_token']='0'*64
    elif damage=='selection_reseal':
        a=v.selection_by_recipe['vits_s']['pool_index'];a.setflags(write=True);a[0]=1;a.setflags(write=False)
    elif damage=='identity_reseal':v.identity['learned_pose_models_constructed']=1;v.identity['sha256']=_hash({k:x for k,x in v.identity.items()if k!='sha256'})
    else:
        p=kw['geometry']/'generated_mano_left_pose.dat';new=p.with_suffix('.replacement');new.write_bytes(p.read_bytes());new.replace(p)
    with pytest.raises(ValueError):v.check()


def test_private_export_and_independent_reconstruction_match_and_tamper_rejects(tmp_path,monkeypatch):
    m=api();kw=fixture(tmp_path,monkeypatch);out=tmp_path/'export'
    a=m.export_dense_geometry_inputs(output=out,**kw);b=m.verify_dense_geometry_export(output=out,**kw)
    assert a==b and len(list(out.glob('*.npz')))==4
    p=out/'vits_s-points.npz'
    with np.load(p,allow_pickle=False)as f:v={k:f[k]for k in f.files}
    v['left_beta'][0,0]+=1;np.savez(p,**v)
    with pytest.raises(ValueError):m.verify_dense_geometry_export(output=out,**kw)
    with pytest.raises(ValueError):m.export_dense_geometry_inputs(output=out,**kw)


def test_full_input_and_export_never_import_torch(tmp_path,monkeypatch):
    m=api();kw=fixture(tmp_path,monkeypatch);old=builtins.__import__
    def forbid(name,*a,**k):
        if name=='torch'or name.startswith('torch.'):raise AssertionError('geometry IO must not import Torch')
        return old(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',forbid)
    m.export_dense_geometry_inputs(output=tmp_path/'out',**kw)


def test_complete_raw_hashing_never_materializes_bulk_files_in_memory(tmp_path,monkeypatch):
    m=api();kw=fixture(tmp_path,monkeypatch);original=Path.read_bytes
    def reject_bulk(path):
        if path.suffix=='.dat':raise AssertionError('full bulk byte hashes must stream, not allocate entire files')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',reject_bulk)
    m.admit_dense_geometry_inputs(**kw).check()
