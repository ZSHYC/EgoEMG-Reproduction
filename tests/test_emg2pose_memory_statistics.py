"""Independent 20D sums and actual unchanged official arithmetic, not model scores."""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def api():
    return importlib.import_module('egoemg_repro.emg2pose_memory_statistics')


def batch():
    p = np.broadcast_to(np.arange(1,21,dtype=np.float32)[None,:,None],(3,20,5)).copy()
    p[1] *= 3
    p[2] *= 8
    t = np.zeros_like(p)
    mask = np.array([[1,1,1,1,1],[1,0,0,0,0],[0,0,0,0,0]],dtype=np.bool_)
    return p,t,mask


def test_unequal_users_valid_times_and_zero_masks_have_distinct_denominators():
    m=api();s=m.Emg2PoseStatistics();p,t,mask=batch()
    s.update(p,t,mask,condition='user',users=['alice','bob','empty'])
    report=s.report();r=report['by_condition']['user'];degrees=180/math.pi
    assert report['scope']=='arithmetic only; no source, coverage, selected-best or calibration qualification'
    assert report['samples']==r['samples']==3 and r['zero_valid_samples']==1
    assert r['overall']['valid_angle_count']==120
    assert r['overall']['absolute_error_sum_rad']==1680
    assert r['overall']['mae_deg']==14*degrees
    assert r['per_user']['user_count']==2
    assert r['per_user']['mean_mae_deg']==21*degrees
    assert math.isclose(r['per_user']['std_mae_deg'],10.5*degrees)
    assert r['users']['empty']['samples']==1
    assert r['users']['empty']['overall']['mae_deg'] is None
    assert report['scientific_complete'] is False and report['full_paper_complete'] is False
    assert len(r['by_joint'])==20
    assert set(r['by_group'])==set(m.GROUPS)
    assert all('wrist' not in j['name'].lower() for j in r['by_joint'])
    assert r['by_joint'][0]['absolute_error_sum_rad']==8
    assert r['by_joint'][0]['valid_angle_count']==6


def test_streaming_batches_conditions_and_all_partitions_conserve():
    s=api().Emg2PoseStatistics();p,t,mask=batch()
    s.update(p[:1],t[:1],mask[:1],condition='user',users=['alice'])
    s.update(p[1:],t[1:],mask[1:],condition='user',users=['bob','empty'])
    s.update(p,t,mask,condition='stage',users=['alice','alice','empty'])
    report=s.report();assert report['samples']==6
    for r in report['by_condition'].values():
        total=r['overall']
        for records in (r['by_joint'],[u['overall'] for u in r['users'].values()],
                        [r['by_group'][g] for g in ('thumb','index','middle','ring','pinky')],
                        [r['by_group'][g] for g in ('proximal','mid','distal')]):
            assert sum(v['valid_angle_count'] for v in records)==total['valid_angle_count']
            assert math.fsum(v['absolute_error_sum_rad'] for v in records)==total['absolute_error_sum_rad']
    assert report['by_condition']['stage']['per_user']['user_count']==1
    assert report['by_condition']['stage']['per_user']['std_mae_deg']==0
    json.dumps(report,allow_nan=False)
    report['by_condition']['user']['overall']['mae_deg']=-1
    assert s.report()['by_condition']['user']['overall']['mae_deg']>0


def test_all_false_masks_and_empty_accumulator_never_publish_nan():
    s=api().Emg2PoseStatistics();assert s.report()['by_condition']=={}
    p,t,mask=batch();mask[:]=False
    s.update(p,t,mask,condition='user',users=['alice','bob','empty'])
    r=s.report()['by_condition']['user']
    assert r['per_user']['user_count']==0 and r['per_user']['mean_mae_deg'] is None
    assert r['per_user']['std_mae_deg'] is None
    assert r['overall']['valid_angle_count']==0
    assert s.check_native({'user':{}},{'user':{}})['native_metrics_agree'] is True
    json.dumps(s.report(),allow_nan=False)


def test_inputs_are_unchanged_and_not_retained_as_prediction_history():
    s=api().Emg2PoseStatistics();p,t,mask=batch()
    before=[v.copy() for v in (p,t,mask)]
    s.update(p,t,mask,condition='user',users=['alice','bob','empty'])
    assert all(np.array_equal(a,b) for a,b in zip((p,t,mask),before))
    report=s.report()
    p.fill(np.nan);t.fill(100);mask[:]=False
    assert s.report()==report


@pytest.mark.parametrize('damage',['channels','pred_dtype','target_dtype','mask_dtype','mask_shape',
                                   'nonfinite_pred','nonfinite_target','empty','users','empty_user','condition'])
def test_bad_batches_are_rejected_before_any_counter_changes(damage):
    s=api().Emg2PoseStatistics();p,t,mask=batch();users=['alice','bob','empty'];condition='user'
    if damage=='channels':p=p[:,:19];t=t[:,:19]
    elif damage=='pred_dtype':p=p.astype(np.float64)
    elif damage=='target_dtype':t=t.astype(np.float16)
    elif damage=='mask_dtype':mask=mask.astype(np.int32)
    elif damage=='mask_shape':mask=mask[:,None,:]
    elif damage=='nonfinite_pred':p[2,0,0]=np.nan
    elif damage=='nonfinite_target':t[2,0,0]=np.inf
    elif damage=='empty':p=p[:0];t=t[:0];mask=mask[:0];users=[]
    elif damage=='users':users=['alice']
    elif damage=='empty_user':users[1]=''
    elif damage=='condition':condition=''
    before=s.report()
    with pytest.raises(ValueError):s.update(p,t,mask,condition=condition,users=users)
    assert s.report()==before


def aligned():
    t=np.broadcast_to(np.arange(7790,dtype=np.float32)[None,None,:],(2,20,7790)).copy()
    mask=np.ones((2,7790),dtype=np.bool_);mask[0,700]=False
    target=t[...,510:];return (target.copy(),target.copy(),mask[:,510:].copy()),t,mask


def test_original_20d_tuple_preserves_exact_510_context_and_target_bits():
    output,t,mask=aligned()
    assert api().validate_original_output(output,original_target=t,original_mask=mask)==dict(
        input_length=7790,left_context=510,right_context=0,output_length=7280,channels=20)


@pytest.mark.parametrize('damage',['tuple','context','mask','target','signed_zero','channels','precision','input'])
def test_original_alignment_refuses_crops_relabels_and_wrong_types(damage):
    output,t,mask=aligned();p,target,outmask=output
    if damage=='tuple':output=list(output)
    elif damage=='context':output=(p[...,:-1],target[...,:-1],outmask[:,:-1])
    elif damage=='mask':outmask[0,190]=True
    elif damage=='target':target[0,0,0]+=1
    elif damage=='signed_zero':t[0,0,510]=0.;target[0,0,0]=-0.
    elif damage=='channels':output=(p[:,:19],target[:,:19],outmask)
    elif damage=='precision':output=(p.astype(np.float16),target,outmask)
    elif damage=='input':t=t.astype(np.float64)
    with pytest.raises(ValueError):api().validate_original_output(output,original_target=t,original_mask=mask)


def native_numbers(s):
    rows={};users={}
    for condition,r in s.report()['by_condition'].items():
        rows[condition]={'test_mae':r['overall']['mae_deg']*math.pi/180}
        rows[condition].update({'test_mae_'+g:v['mae_deg']*math.pi/180 for g,v in r['by_group'].items()})
        u=r['per_user'];users[condition]=dict(per_user_count=u['user_count'],
            per_user_mae_mean=u['mean_mae_deg']*math.pi/180,
            per_user_mae_std=u['std_mae_deg']*math.pi/180)
        for g,v in u['by_group'].items():
            users[condition]['per_user_mae_'+g+'_mean']=v['mean_mae_deg']*math.pi/180
            users[condition]['per_user_mae_'+g+'_std']=v['std_mae_deg']*math.pi/180
    return rows,users


def test_native_summation_bound_is_fixed_and_not_a_prediction_tolerance():
    s=api().Emg2PoseStatistics();s.update(*batch(),condition='user',users=['alice','bob','empty'])
    rows,users=native_numbers(s)
    rows['user']['test_mae']+=1e-6
    result=s.check_native(rows,users)
    assert result['native_metrics_agree'] is True
    assert result['policy']['native_mae_atol_rad']==result['policy']['native_mae_rtol']==2e-6
    assert result['policy']['prediction_replay']=='not qualified by arithmetic comparison'
    rows['user']['test_mae']+=.1
    with pytest.raises(ValueError):s.check_native(rows,users)


@pytest.mark.parametrize('damage',['condition','missing_group','unknown_group','user_count','mean','std','nonfinite'])
def test_native_comparison_requires_all_groups_users_and_finite_fields(damage):
    s=api().Emg2PoseStatistics();s.update(*batch(),condition='user',users=['alice','bob','empty'])
    rows,users=native_numbers(s)
    if damage=='condition':rows['other']=rows.pop('user')
    elif damage=='missing_group':rows['user'].pop('test_mae_thumb')
    elif damage=='unknown_group':rows['user']['test_mae_wrist']=0.
    elif damage=='user_count':users['user']['per_user_count']=3
    elif damage=='mean':users['user']['per_user_mae_thumb_mean']+=.1
    elif damage=='std':users['user']['per_user_mae_std']+=.1
    elif damage=='nonfinite':rows['user']['test_mae']=float('nan')
    with pytest.raises(ValueError):s.check_native(rows,users)


def _native_oracle(upstream):
    import torch
    from egoemg_repro import emg2pose_runtime as runtime
    runtime.storage._official(upstream)
    runtime._activate(upstream)
    from egoemg.test_analysis import EMG2PoseEvaluation
    from egoemg.constants import JOINTS,FINGERS,PD_GROUPS
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    m=api();assert m.JOINT_NAMES==tuple(j.name for j in JOINTS)
    assert m.GROUPS=={g:tuple(j.index for j in JOINTS if g in j.groups) for g in [*FINGERS,*PD_GROUPS]}
    # A recording ID is not a unique stage-file row. The native loader selects
    # both matching rows unless its existing index filter is also supplied.
    selection=object.__new__(Emg2PoseDataset)
    selection._session_id=np.array(['recording','recording'])
    selection._session_filename=np.array(['stage_a.hdf5','stage_b.hdf5'])
    selection.allowed_sessions=['recording']
    assert selection._filter_session_indices().tolist()==[0,1]
    selection.allowed_session_indices=(0,)
    assert selection._filter_session_indices().tolist()==[0]
    torch.set_num_threads(2);assert not torch.cuda.is_available()
    rng=np.random.default_rng(42)
    t=rng.normal(scale=.5,size=(3,20,7280)).astype(np.float32)
    p=(t+rng.normal(scale=.2,size=t.shape)).astype(np.float32)
    mask=rng.random((3,7280))>.1
    mask[1,1:]=False;mask[1,0]=True;mask[2,:]=False
    s=m.Emg2PoseStatistics()
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,scale):self.scale=scale
        def __len__(self):return 3
        def __getitem__(self,i):
            return dict(pred=torch.from_numpy(p[i]*self.scale),target=torch.from_numpy(t[i].copy()),
                        mask=torch.from_numpy(mask[i].copy()),user=['alice','bob','empty'][i])
    class Module(torch.nn.Module):
        def forward(self,b):return b['pred'],b['target'],b['mask']
    context=type('Context',(),{})();context.module=Module();context.batch_size=2;context.num_workers=0
    context._emg2pose_datasets=[Dataset(1),Dataset(.25)]
    rows,users=EMG2PoseEvaluation._evaluate_emg2pose_single_pass(context,[{},{}])
    for condition,scale in [('user',1),('user_stage',.25)]:
        s.update(p*scale,t,mask,condition=condition,users=['alice','bob','empty'])
    result=s.check_native(dict(zip(('user','user_stage'),rows)),dict(zip(('user','user_stage'),users)))
    assert result['native_metrics_agree'] is True
    assert not torch.cuda.is_initialized()
    print('EGOEMG_20D_ARITHMETIC='+json.dumps(dict(conditions=2,groups=8,joints=20,
        unchanged_native_method=True,cuda_initialized=False,model_performance=False)))


def test_actual_unchanged_official_single_pass_and_all_group_constants_in_fresh_cpu_process(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('hydra');pytest.importorskip('pytorch_lightning')
    upstream=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).resolve().parents[1]/'upstream'))
    if not (upstream/'egoemg/test_analysis.py').is_file():pytest.skip('pinned upstream fixture unavailable')
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+environment.get('PYTHONPATH','')
    run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'native',str(upstream)],
                       env=environment,capture_output=True,text=True,timeout=120)
    (tmp_path/'native.stdout').write_text(run.stdout);(tmp_path/'native.stderr').write_text(run.stderr)
    assert run.returncode==0,run.stdout[-2000:]+run.stderr[-5000:]
    assert '"unchanged_native_method": true' in run.stdout


if __name__=='__main__':
    assert sys.argv[1]=='native'
    _native_oracle(Path(sys.argv[2]))
