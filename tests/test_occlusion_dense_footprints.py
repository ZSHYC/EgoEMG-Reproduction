"""Half-open input intersections, independently of models or native permits."""
import copy
import importlib

import numpy as np
import pytest


def api():return importlib.import_module('egoemg_repro.occlusion_dense_footprints')


def fixture():
    from egoemg_repro.occlusion_dense_data import COLUMNS
    values=dict(sequence=[0,1,2,3],episode=[0]*4,start=[4,4,9,9],end=[8,8,13,13],center=[6,6,11,11],
        split_id=[1]*4,hand_index=[0,1,0,1],native_index=[0,0,1,1],group_index=[0]*4)
    a={k:np.asarray(values[k],dtype=d)for k,d in COLUMNS.items()}
    p=dict(windows=[dict(role=r,footprint_start=s,footprint_end=e)for r,s,e in
        [('train',0,4),('train',8,10),('val',5,6),('test',13,20)]])
    return a,p


def test_half_open_rows_not_doubled_by_hands_and_overlapping_requests_separate():
    a,p=fixture();v=api().summarize_dense_footprints(a,p)
    assert v['hand_requests']==4 and v['unique_temporal_windows']==2 and v['dense_input_union_rows']==8
    assert v['roles']['train']==dict(producer_input_union_rows=6,intersection_rows=1,overlapping_hand_requests=2,centers_in_producer_rows=0)
    assert v['roles']['val']==dict(producer_input_union_rows=1,intersection_rows=1,overlapping_hand_requests=2,centers_in_producer_rows=0)
    assert v['roles']['test']['intersection_rows']==0
    assert v['any_training_input_overlap']and v['model_or_geometry_permit']is False


def test_touching_endpoints_and_duplicate_producer_windows_do_not_inflate_overlap():
    a,p=fixture();p['windows']=[dict(role='train',footprint_start=s,footprint_end=e)for s,e in[(0,4),(8,9),(13,20)]]
    assert api().summarize_dense_footprints(a,p)['roles']['train']['intersection_rows']==0
    p['windows']=[dict(role='train',footprint_start=9,footprint_end=10)]*2
    v=api().summarize_dense_footprints(a,p)['roles']['train']
    assert v['intersection_rows']==v['producer_input_union_rows']==1 and v['overlapping_hand_requests']==2


@pytest.mark.parametrize('damage',['duplicate_hand','unpaired','float_column','sequence','paired_bounds','missing_bound','bool_bound','unknown_role'])
def test_invalid_join_or_producer_bounds_cannot_hide_overlap(damage):
    a,p=fixture();api().summarize_dense_footprints(a,p)
    if damage=='duplicate_hand':a['hand_index'][1]=0
    elif damage=='unpaired':a={k:v[:-1]for k,v in a.items()}
    elif damage=='float_column':a['start']=a['start'].astype(np.float32)
    elif damage=='sequence':a['sequence'][1]=0
    elif damage=='paired_bounds':a['start'][1]=3
    elif damage=='missing_bound':del p['windows'][0]['footprint_end']
    elif damage=='bool_bound':p['windows'][0]['footprint_start']=False
    else:p['windows'][0]['role']='all'
    with pytest.raises(ValueError):api().summarize_dense_footprints(a,p)


def test_empty_selection_preserves_zero_denominators():
    a,p=fixture();a={k:v[:0]for k,v in a.items()};v=api().summarize_dense_footprints(a,p)
    assert v['hand_requests']==v['unique_temporal_windows']==v['dense_input_union_rows']==0
    assert not v['any_training_input_overlap']
    assert all(x['intersection_rows']==x['overlapping_hand_requests']==0 for x in v['roles'].values())
