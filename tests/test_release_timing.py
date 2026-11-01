"""Released clock-field diagnostics; raw recorded differences are not model dt."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=(ROOT/'upstream').resolve()
FIELDS={'timestamp_us':('int64','timestamp_us.dat'),'episode_index':('int64','episode_index.dat'),
 'frame_index':('int64','frame_index.dat'),'image_webcam_frame_index':('int32','image_webcam_frame_index.dat'),
 'image_webcam_delta_ms':('int32','image_webcam_delta_ms.dat'),'image_webcam_stale':('bool','image_webcam_stale.dat')}


def api():
    try:return importlib.import_module('egoemg_repro.release_timing')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.release_timing':pytest.fail('full released timing reader missing')
        raise


def arrays():
    return dict(timestamp_us=np.array([1000,1000,2500,2000,8000,8500,9000,9500],dtype='int64'),
        episode_index=np.repeat([0,1],4).astype('int64'),frame_index=np.tile(np.arange(4,dtype='int64'),2),
        image_webcam_frame_index=np.array([0,0,-1,2,4,4,5,5],dtype='int32'),
        image_webcam_delta_ms=np.array([0,10,2147483647,-10,5,5,5,100],dtype='int32'),
        image_webcam_stale=np.array([0,0,1,1,0,0,0,1],dtype='uint8'))


def fixture(root):
    root.mkdir();a=arrays();fields={}
    for k,(dtype,name)in FIELDS.items():
        a[k].tofile(root/name);fields[k]=dict(dtype=dtype,filename=name,shape=[8])
    (root/'manifest.json').write_text(json.dumps(dict(format_version='egoemg_v2_memmap',total_rows=8,num_episodes=2,fields=fields)))
    np.savez(root/'metadata.npz',episode_start_idx=np.array([0,4],dtype='int64'),episode_end_idx=np.array([4,8],dtype='int64'),episode_length=np.array([4,4],dtype='int64'))
    return root


def test_exact_histograms_keep_ties_backward_steps_sentinels_and_chunk_carry():
    m=api();a=arrays();r=m.timing_statistics(a,[(0,4),(4,8)],chunk_rows=1)
    assert r==m.timing_statistics(a,[(0,4),(4,8)],chunk_rows=3)
    assert r['rows']==8 and r['episodes']==2
    g=r['global'];t=g['timestamp_step_us']
    assert t['count']==6 and t['sum']==2500 and t['quantiles']=={'p0':-500,'p50':500,'p95':1250,'p99':1450,'p100':1500}
    assert g['emg_frame_step']['count']==6 and g['emg_frame_step']['min']==g['emg_frame_step']['max']==1
    assert g['camera_frame_step']['count']==4 and g['camera_frame_step']['sum']==1
    assert g['camera']==dict(stale_rows=3,nonstale_rows=5,index_negative_rows=1,index_zero_rows=2,
        metadata_usable_rows=5,int32max_delta_rows=1,stale_and_index_negative_rows=1,nonstale_and_index_negative_rows=0)
    assert g['recorded_delta_ms']['sum']==2147483762 and g['recorded_delta_ms']['count']==8
    assert g['metadata_usable_delta_ms']['count']==5 and g['metadata_usable_delta_ms']['mean']==5
    assert g['metadata_usable_delta_ms']['quantiles']['p95']==9
    assert {x['value']:x['count']for x in r['histograms']if x['scope']=='all'and x['measure']=='timestamp_step_us'}=={-500:1,0:1,500:3,1500:1}


def test_singletons_and_empty_camera_subsets_keep_null_without_cross_episode_pair():
    a={k:v[:2].copy()for k,v in arrays().items()};a['episode_index'][:]=[0,1]
    a['image_webcam_frame_index'][:]=-1;a['image_webcam_stale'][:]=1
    g=api().timing_statistics(a,[(0,1),(1,2)],chunk_rows=1)['global']
    for key in ('timestamp_step_us','emg_frame_step','camera_frame_step','metadata_usable_delta_ms'):
        assert g[key]['count']==0 and g[key]['mean']is None and all(x is None for x in g[key]['quantiles'].values())


@pytest.mark.parametrize('damage',['stale_byte','dtype','shape','gap','overlap','bool_boundary','episode_id','overflow','bad_chunk'])
def test_malformed_layout_coverage_and_overflow_never_produce_a_report(damage):
    a=arrays();bounds=[(0,4),(4,8)];chunk=2
    if damage=='stale_byte':a['image_webcam_stale'][0]=2
    elif damage=='dtype':a['timestamp_us']=a['timestamp_us'].astype('float64')
    elif damage=='shape':a['frame_index']=a['frame_index'][:,None]
    elif damage=='gap':bounds=[(0,3),(4,8)]
    elif damage=='overlap':bounds=[(0,5),(4,8)]
    elif damage=='bool_boundary':bounds=[(False,4),(4,8)]
    elif damage=='episode_id':a['episode_index'][4]=0
    elif damage=='overflow':a['timestamp_us'][:2]=[np.iinfo('int64').min,np.iinfo('int64').max]
    else:chunk=True
    with pytest.raises(ValueError):api().timing_statistics(a,bounds,chunk_rows=chunk)


def test_actual_readonly_fixture_is_explicitly_partial_and_cannot_be_full(tmp_path):
    data=fixture(tmp_path/'data');m=api();r=m.scan_release_timing(data,UPSTREAM,require_full=False,chunk_rows=3)
    assert r['analysis_complete']and not r['full_legacy_population_verified']and not r['full_paper_complete']
    assert not r['hardware_synchronization_accuracy_established']and not r['timestamps_used_as_model_dt']
    assert r['statistics']['rows']==8 and len(r['input_files_sha256'])==8
    assert r['statistics']==m.timing_statistics(arrays(),[(0,4),(4,8)],chunk_rows=1)
    with pytest.raises(ValueError):m.scan_release_timing(data,UPSTREAM)


def test_source_replacement_inside_hash_cannot_create_a_mixed_generation(tmp_path,monkeypatch):
    m=api();data=fixture(tmp_path/'data');p=data/'timestamp_us.dat';original=m._sha;done=False
    def changed(path):
        nonlocal done
        digest=original(path)
        if Path(path)==p and not done:
            done=True;a=np.memmap(p,dtype='int64',mode='r+');a[0]+=1;a.flush();a._mmap.close()
        return digest
    monkeypatch.setattr(m,'_sha',changed)
    with pytest.raises(ValueError):m.scan_release_timing(data,UPSTREAM,require_full=False)
    assert done


def test_declared_episode_count_must_match_the_complete_numeric_metadata(tmp_path):
    data=fixture(tmp_path/'data');p=data/'manifest.json';value=json.loads(p.read_text());value['num_episodes']=1;p.write_text(json.dumps(value))
    with pytest.raises(ValueError):api().scan_release_timing(data,UPSTREAM,require_full=False)
