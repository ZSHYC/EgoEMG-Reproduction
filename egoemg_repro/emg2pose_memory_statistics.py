"""Independent streaming 20-angle arithmetic; no source or inference qualification.

Reports retain private user keys. No full prediction history is retained, and
an arithmetic agreement cannot qualify a checkpoint, replay or scientific run.
"""
from __future__ import annotations

import math

import numpy as np


JOINT_NAMES=(
    'THUMB_CMC_FE','THUMB_CMC_AA','THUMB_MCP_FE','THUMB_IP_FE',
    'INDEX_MCP_AA','INDEX_MCP_FE','INDEX_PIP_FE','INDEX_DIP_FE',
    'MIDDLE_MCP_AA','MIDDLE_MCP_FE','MIDDLE_PIP_FE','MIDDLE_DIP_FE',
    'RING_MCP_AA','RING_MCP_FE','RING_PIP_FE','RING_DIP_FE',
    'PINKY_MCP_AA','PINKY_MCP_FE','PINKY_PIP_FE','PINKY_DIP_FE')
GROUPS={name:tuple(range(4*i,4*i+4)) for i,name in enumerate(('thumb','index','middle','ring','pinky'))}
GROUPS.update(proximal=(0,1,4,5,8,9,12,13,16,17),mid=(2,6,10,14,18),distal=(3,7,11,15,19))
SCOPE='arithmetic only; no source, coverage, selected-best or calibration qualification'


def comparison_policy():
    return dict(native_mae_atol_rad=2e-6,native_mae_rtol=2e-6,
                prediction_replay='not qualified by arithmetic comparison',
                aggregation='float64 sums / exact valid-angle counts; unweighted user means and ddof0 std',
                selection='fixed before validation or test')


def _arrays(prediction,target,mask):
    if (not isinstance(prediction,np.ndarray) or prediction.ndim!=3 or
            prediction.shape[0]<1 or prediction.shape[1]!=20 or prediction.shape[2]<1 or
            prediction.dtype!=np.float32 or not isinstance(target,np.ndarray) or
            target.shape!=prediction.shape or target.dtype!=np.float32 or
            not isinstance(mask,np.ndarray) or mask.shape!=(len(target),target.shape[2]) or
            mask.dtype!=np.bool_ or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError('finite original FP32 Bx20xT prediction/target and bool BxT mask required')


def validate_original_output(output,*,original_target,original_mask):
    if (type(output) is not tuple or len(output)!=3 or
            not isinstance(original_target,np.ndarray) or original_target.ndim!=3 or
            original_target.shape[1:]!=(20,7790) or original_target.dtype!=np.float32 or
            not isinstance(original_mask,np.ndarray) or original_mask.dtype!=np.bool_ or
            original_mask.shape!=(len(original_target),7790) or not np.isfinite(original_target).all()):
        raise ValueError('original20x7790 FP32 target/bool mask and three-part tuple required')
    prediction,target,mask=output
    _arrays(prediction,target,mask)
    if (target.shape!=(len(original_target),20,7280) or
            not np.array_equal(target.view(np.uint32),original_target[...,510:].view(np.uint32)) or
            not np.array_equal(mask,original_mask[:,510:])):
        raise ValueError('original left510/right0 target bits or generated mask alignment differs')
    return dict(input_length=7790,left_context=510,right_context=0,output_length=7280,channels=20)


def _record(error,count):
    return dict(absolute_error_sum_rad=float(error),valid_angle_count=int(count),
                mae_deg=float(error)/int(count)*180/math.pi if count else None)


def _cell_report(sums,valid_times,samples,zero_valid):
    return dict(samples=int(samples),zero_valid_samples=int(zero_valid),
        overall=_record(math.fsum(sums),20*valid_times),
        by_joint=[dict(index=i,name=name,**_record(sums[i],valid_times)) for i,name in enumerate(JOINT_NAMES)],
        by_group={g:_record(math.fsum(sums[i] for i in ids),len(ids)*valid_times) for g,ids in GROUPS.items()})


def _user_summary(records):
    values=[v['mae_deg'] for v in records if v['valid_angle_count']]
    return dict(user_count=len(values),mean_mae_deg=float(np.mean(values)) if values else None,
                std_mae_deg=float(np.std(values,ddof=0)) if values else None)


class Emg2PoseStatistics:
    """Consume bounded batches; retain only per-user 20D sufficient statistics."""
    def __init__(self):
        self._conditions={}

    def update(self,prediction,target,mask,*,condition,users):
        _arrays(prediction,target,mask)
        if (not isinstance(condition,str) or not condition.strip() or type(users) not in (tuple,list) or
                len(users)!=len(target) or any(not isinstance(u,str) or not u.strip() for u in users)):
            raise ValueError('one actual condition and nonempty user key per sample required')
        cells=self._conditions.setdefault(condition,{})
        for i,user in enumerate(users):
            cell=cells.setdefault(user,dict(sums=np.zeros(20,np.float64),valid_times=0,samples=0,zero_valid=0))
            valid=mask[i];count=int(valid.sum())
            # Per-sample temporaries stay bounded even for original batch320.
            error=np.subtract(prediction[i][:,valid],target[i][:,valid],dtype=np.float64)
            np.abs(error,out=error)
            cell['sums']+=error.sum(axis=1,dtype=np.float64)
            cell['valid_times']+=count;cell['samples']+=1;cell['zero_valid']+=int(count==0)

    def report(self):
        conditions={};samples=0
        for name,cells in sorted(self._conditions.items()):
            users={u:_cell_report(v['sums'],v['valid_times'],v['samples'],v['zero_valid']) for u,v in sorted(cells.items())}
            sums=np.zeros(20,np.float64)
            for user in sorted(cells):sums+=cells[user]['sums']
            result=_cell_report(sums,sum(v['valid_times'] for v in cells.values()),
                                sum(v['samples'] for v in cells.values()),sum(v['zero_valid'] for v in cells.values()))
            result['users']=users
            result['per_user']=dict(_user_summary([v['overall'] for v in users.values()]),
                by_group={g:_user_summary([v['by_group'][g] for v in users.values()]) for g in GROUPS})
            samples+=result['samples'];conditions[name]=result
        return dict(format='emg2pose_independent_20d_statistics_v1',scope=SCOPE,samples=samples,
                    by_condition=conditions,scientific_complete=False,full_paper_complete=False)

    def check_native(self,rows,users):
        """Compare native summation only; prediction identity is a separate gate."""
        report=self.report();conditions=report['by_condition']
        if (not conditions or not isinstance(rows,dict) or not isinstance(users,dict) or
                set(rows)!=set(conditions) or set(users)!=set(conditions)):
            raise ValueError('complete consumed native condition groups required')
        policy=comparison_policy();checked=0
        def compare(actual,expected):
            nonlocal checked
            if not isinstance(actual,dict) or set(actual)!=set(expected):
                raise ValueError('native metric/user fields differ')
            for key,value in expected.items():
                other=actual[key]
                if key=='per_user_count':
                    if type(other) is not int or other!=value:raise ValueError('native user count differs')
                elif (type(other) is not float or not math.isfinite(other) or
                        not math.isclose(other,value,abs_tol=policy['native_mae_atol_rad'],rel_tol=policy['native_mae_rtol'])):
                    raise ValueError('native FP32 and independent FP64 metric differ: '+key)
                checked+=1
        for name,r in conditions.items():
            expected={};expected_users={}
            if r['overall']['valid_angle_count']:
                expected['test_mae']=r['overall']['mae_deg']*math.pi/180
                expected.update({'test_mae_'+g:v['mae_deg']*math.pi/180 for g,v in r['by_group'].items()})
                per_user=r['per_user']
                expected_users=dict(per_user_count=per_user['user_count'],
                    per_user_mae_mean=per_user['mean_mae_deg']*math.pi/180,
                    per_user_mae_std=per_user['std_mae_deg']*math.pi/180)
                for g,value in per_user['by_group'].items():
                    expected_users['per_user_mae_'+g+'_mean']=value['mean_mae_deg']*math.pi/180
                    expected_users['per_user_mae_'+g+'_std']=value['std_mae_deg']*math.pi/180
            compare(rows[name],expected);compare(users[name],expected_users)
        return dict(scope=SCOPE,native_metrics_agree=True,condition_count=len(conditions),
                    metric_fields_checked=checked,policy=policy,scientific_complete=False)
