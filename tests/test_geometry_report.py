"""Population semantics on explicitly synthetic, non-native geometry fixtures."""
import copy
import importlib
import json

import numpy as np
import pytest


def api():return importlib.import_module('egoemg_repro.geometry_report')


def samples():
    rows=[];packets={}
    for seq,(hand,index,kind)in enumerate([('left',0,'half'),('left',1,'zero'),('right',0,'behind'),('right',1,'visible')]):
        row=dict(sequence=seq,hand=hand,point_index=index,center_idx=10+index,mocap_tracked=kind not in('zero','behind'),
            image_stale=kind=='behind',markers_valid=kind not in('zero','behind'))
        if kind=='zero':row.update(status='invalid_zero_pose',occlusion_score=None,visible_ratio=None);rows.append(row);continue
        cam=np.ones((778,3),np.float64);u=np.full(778,960,np.int32);v=np.full(778,540,np.int32)
        visible=np.zeros(778,bool);weights=np.ones(778,np.float64)
        if kind=='half':visible[:389]=True;cam[-2:,2]=-1;u[-2:]=v[-2:]=-1;u[400:402]=2000
        elif kind=='behind':cam[:,2]=-1;u[:]=v[:]=-1
        else:visible[:]=True
        ratio=float(weights[visible].sum()/weights.sum())
        row.update(status='native_geometry',visible_ratio=ratio,occlusion_score=1.-ratio,positive_depth_vertices=int((cam[:,2]>1e-8).sum()),
            visible_vertices=int(visible.sum()),area_weight_sum=float(weights.sum()),width=1920,height=1080)
        packets[seq]=dict(verts_cam=cam,u_proj=u,v_proj=v,visible=visible,area_weights=weights);rows.append(row)
    return rows,packets


def test_unique_request_scored_denominators_and_projection_populations():
    m=api();rows,packets=samples();seen=[]
    def read(row):seen.append(row['sequence']);return packets[row['sequence']]
    p=m.summarize_rows(rows,read)
    assert seen==[0,2,3]and p['requests']==4 and p['native_geometry_rows']==3 and p['invalid_zero_pose_rows']==1
    assert p['vertices_total']==2334 and p['vertices_positive_depth']==1554
    assert p['vertices_behind_or_near_camera']==780 and p['vertices_front_in_image']==1552
    assert p['vertices_front_offscreen']==2 and p['vertices_visible']==1167
    assert p['native_rows_behind_camera_only']==1 and p['native_rows_with_offscreen_vertices']==1
    assert p['native_rows_all_front_vertices_offscreen']==0 and p['native_rows_with_visible_vertices']==2
    assert p['mocap_tracked_rows']==2 and p['image_stale_rows']==1 and p['markers_valid_rows']==2
    assert p['native_occlusion']==dict(count=3,mean=.5,sample_sd=.5,minimum=0.,median=.5,maximum=1.)
    assert sum(p['occlusion_histogram']['counts'])==3 and len(p['occlusion_histogram']['counts'])==20


def test_zero_pose_rows_keep_flags_without_inventing_score_or_native_packet():
    rows,_=samples();rows=[dict(rows[1],sequence=0)]
    def forbidden(row):raise AssertionError('invalid zero pose must not read a native packet')
    p=api().summarize_rows(rows,forbidden)
    assert p['requests']==1 and p['native_geometry_rows']==0 and p['invalid_zero_pose_rows']==1
    assert p['native_occlusion']==dict(count=0,mean=None,sample_sd=None,minimum=None,median=None,maximum=None)
    assert p['vertices_total']==0 and sum(p['occlusion_histogram']['counts'])==0


@pytest.mark.parametrize('damage',['duplicate','sequence','status','zero_score','ratio','positive_count','visible_count','weights_sum',
    'cam_nan','projection_dtype','projection_shape','weights_negative','visible_behind','visible_offscreen'])
def test_bad_population_or_packet_cannot_be_exported_as_valid_summary(damage):
    rows,packets=samples()
    if damage=='duplicate':rows[3].update(hand=rows[0]['hand'],point_index=rows[0]['point_index'])
    elif damage=='sequence':rows[0]['sequence']=True
    elif damage=='status':rows[0]['status']='skipped'
    elif damage=='zero_score':rows[1]['occlusion_score']=1.
    elif damage=='ratio':rows[0]['occlusion_score']=.9
    elif damage=='positive_count':rows[0]['positive_depth_vertices']=778
    elif damage=='visible_count':rows[0]['visible_vertices']=0
    elif damage=='weights_sum':rows[0]['area_weight_sum']=0.
    elif damage=='cam_nan':packets[0]['verts_cam'][0,0]=np.nan
    elif damage=='projection_dtype':packets[0]['u_proj']=packets[0]['u_proj'].astype(np.int64)
    elif damage=='projection_shape':packets[0]['v_proj']=packets[0]['v_proj'][:-1]
    elif damage=='weights_negative':packets[0]['area_weights'][0]=-1.
    elif damage=='visible_behind':packets[2]['visible'][0]=True
    else:packets[0]['visible'][400]=True
    with pytest.raises(ValueError):api().summarize_rows(rows,lambda r:packets[r['sequence']])


def test_packet_reader_hash_failures_propagate_without_partial_summary():
    rows,_=samples()
    def corrupt(row):raise ValueError('actual packet SHA differs')
    with pytest.raises(ValueError,match='SHA'):api().summarize_rows(rows,corrupt)


def test_unpinned_evidence_rejects_before_private_input_io_or_output_creation(tmp_path):
    p=tmp_path/'forged.json';p.write_text(json.dumps(dict(both_standard_role_computations_complete=True)))
    with pytest.raises(ValueError):api().build_geometry_report(tmp_path/'absent-val',tmp_path/'absent-test',tmp_path/'out',evidence=p)
    assert not(tmp_path/'out').exists()


def test_summary_is_numpy_only_and_never_imports_torch(monkeypatch):
    import builtins
    m=api();rows,packets=samples();original=builtins.__import__
    def reject(name,*a,**kw):
        if name=='torch'or name.startswith('torch.'):raise AssertionError('geometry report cannot import Torch')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',reject)
    assert m.summarize_rows(rows,lambda r:packets[r['sequence']])['native_geometry_rows']==3
