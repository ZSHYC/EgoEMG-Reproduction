"""Table aggregation definitions and raw-statistic/publication boundaries."""
import copy
import importlib
import json
import math

import numpy as np
import pytest

from egoemg_repro.emg2pose_memory_statistics import Emg2PoseStatistics

def api():return importlib.import_module('egoemg_repro.emg2pose_memory_report')

def native(report):
    rows={};users={};factor=math.pi/180
    for c,r in report['by_condition'].items():
        if not r['overall']['valid_angle_count']:rows[c]={};users[c]={};continue
        rows[c]=dict(test_mae=r['overall']['mae_deg']*factor,
            **{'test_mae_'+g:v['mae_deg']*factor for g,v in r['by_group'].items()})
        p=r['per_user'];users[c]=dict(per_user_count=p['user_count'],
            per_user_mae_mean=p['mean_mae_deg']*factor,per_user_mae_std=p['std_mae_deg']*factor)
        for g,v in p['by_group'].items():
            users[c]['per_user_mae_'+g+'_mean']=v['mean_mae_deg']*factor
            users[c]['per_user_mae_'+g+'_std']=v['std_mae_deg']*factor
    return rows,users

@pytest.fixture
def inputs():
    s=Emg2PoseStatistics()
    for c,errors,masks,users in [
        ('stage',[1.,3.],[[1,1,1,1],[1,0,0,0]],['private-A','private-B']),
        ('user',[2.],[[1,1,1,1]],['private-A']),
        ('user_stage',[4.,4.,4.],[[1,1,1,1],[1,0,0,0],[0,0,0,0]],['private-C','private-C','private-D'])]:
        target=np.zeros((len(errors),20,4),np.float32)
        pred=np.broadcast_to(np.asarray(errors,np.float32)[:,None,None],target.shape).copy()
        s.update(pred,target,np.asarray(masks,bool),condition=c,users=users)
    report=s.report();return report,*native(report)

def test_table13_uses_condition_sample_weights_and_separates_valid_angle_pooling(inputs):
    value=api().summarize_tables(*inputs);factor=180/math.pi
    table=value['table13'];assert table['condition_sample_weights']=={'stage':2,'user':1,'user_stage':3}
    assert table['consumed_samples']==6
    assert table['native_group_mae_deg']['thumb']==pytest.approx(2.8*factor)
    assert table['independent_group_mae_deg']['thumb']==pytest.approx(2.8*factor)
    assert value['all_condition_consumptions']['overall']['mae_deg']==pytest.approx(2.5*factor)
    assert table['native_group_mae_deg']['thumb']!=pytest.approx(((1.4+2+4)/3)*factor)
    assert value['scientific_complete'] is False and value['full_paper_complete'] is False
    assert value['condition_labels_complete'] is True

def test_table11_keeps_unweighted_native_user_mean_std_and_exact_count(inputs):
    value=api().summarize_tables(*inputs);row=value['table11']['stage'];factor=180/math.pi
    assert row['native_user_count']==2
    assert row['native_mean_mae_deg']==pytest.approx(2*factor)
    assert row['native_std_mae_deg']==pytest.approx(factor)
    assert row['independent_mean_mae_deg']==pytest.approx(2*factor)
    assert row['native_mean_mae_deg']!=pytest.approx(value['by_condition']['stage']['overall']['mae_deg'])
    assert value['table11']['user_stage']['native_user_count']==1

def test_public_users_are_anonymous_consistent_and_inputs_remain_unchanged(inputs):
    before=copy.deepcopy(inputs);value=api().summarize_tables(*inputs)
    assert inputs==before and 'private-' not in json.dumps(value)
    a=set(value['by_condition']['stage']['users']);b=set(value['by_condition']['user']['users'])
    assert len(a&b)==1 and len(a)==2
    assert all(u.startswith('user-') for u in a|b)
    zero=[u for u in value['by_condition']['user_stage']['users'].values() if u['zero_valid_samples']==1]
    assert len(zero)==1 and zero[0]['overall']['valid_angle_count']==0

@pytest.mark.parametrize('damage',['cached_mae','group_count','raw_sum','joint_count','joint_name','bool_samples','negative_sum','false_completion'])
def test_rebuilds_raw_joint_statistics_and_rejects_resealed_cached_claims(inputs,damage):
    report,rows,users=copy.deepcopy(inputs);r=report['by_condition']['stage'];u=r['users']['private-A']
    if damage=='cached_mae':r['overall']['mae_deg']+=1
    elif damage=='group_count':r['by_group']['thumb']['valid_angle_count']+=1
    elif damage=='raw_sum':u['by_joint'][0]['absolute_error_sum_rad']+=1
    elif damage=='joint_count':u['by_joint'][0]['valid_angle_count']+=1
    elif damage=='joint_name':u['by_joint'][0]['name']='incorrect'
    elif damage=='bool_samples':u['samples']=True
    elif damage=='negative_sum':u['by_joint'][0]['absolute_error_sum_rad']=-1.
    else:report['scientific_complete']=True
    with pytest.raises(ValueError):api().summarize_tables(report,rows,users)

def test_original_native_disagreement_is_not_hidden_by_independent_statistics(inputs):
    report,rows,users=copy.deepcopy(inputs);users['stage']['per_user_mae_mean']+=.001
    with pytest.raises(ValueError,match='native|FP32'):api().summarize_tables(report,rows,users)

def test_missing_conditions_require_explicit_partial_arithmetic_and_never_qualify_full():
    s=Emg2PoseStatistics();x=np.ones((1,20,2),np.float32)
    s.update(x,np.zeros_like(x),np.ones((1,2),bool),condition='user',users=['private-only'])
    r=s.report();n,u=native(r)
    with pytest.raises(ValueError):api().summarize_tables(r,n,u)
    value=api().summarize_tables(r,n,u,require_all_conditions=False)
    assert value['condition_labels_complete'] is False and value['scientific_complete'] is False
    assert value['table13']['condition_sample_weights']=={'user':1}

def test_unknown_condition_and_zero_valid_condition_cannot_be_table_results():
    for name,valid in [('unknown',True),('user',False)]:
        s=Emg2PoseStatistics();x=np.ones((1,20,2),np.float32)
        s.update(x,np.zeros_like(x),np.full((1,2),valid,bool),condition=name,users=['private-only'])
        r=s.report();n,u=native(r)
        with pytest.raises(ValueError):api().summarize_tables(r,n,u,require_all_conditions=False)

def test_all_eight_joint_groups_use_exact_unrounded_angle_sums():
    s=Emg2PoseStatistics();x=np.broadcast_to(np.arange(1,21,dtype=np.float32)[None,:,None],(1,20,2)).copy()
    s.update(x,np.zeros_like(x),np.ones((1,2),bool),condition='user',users=['private-only'])
    r=s.report();n,u=native(r);value=api().summarize_tables(r,n,u,require_all_conditions=False)
    factor=180/math.pi;g=value['table13']['independent_group_mae_deg']
    assert set(g)=={'thumb','index','middle','ring','pinky','proximal','mid','distal'}
    assert g['thumb']==pytest.approx(2.5*factor)
    assert g['index']==pytest.approx(6.5*factor)
    assert g['proximal']==pytest.approx(np.mean([1,2,5,6,9,10,13,14,17,18])*factor)
    assert g['mid']==pytest.approx(11*factor) and g['distal']==pytest.approx(12*factor)
