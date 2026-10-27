"""Sanitized Table11/13 arithmetic, never scientific completion admission.

Source, selected-best, inference, profile and owned-exit evidence must be
admitted separately by a final report controller. Condition labels alone do not
prove a benchmark. Literal condition populations may consume the same windows.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .emg2pose_memory_statistics import (Emg2PoseStatistics,JOINT_NAMES,GROUPS,
    comparison_policy,_cell_report)


CONDITIONS=('stage','user','user_stage')
SCOPE='arithmetic only; no source, model, inference, profile or completion admission'


def _raw_statistics(value):
    """Rebuild all cached statistics from per-user raw20D sums/counts."""
    rebuilt=Emg2PoseStatistics()
    try:
        conditions=value['by_condition']
        if not isinstance(conditions,dict) or not conditions or not set(conditions)<=set(CONDITIONS):
            raise ValueError('known nonempty test conditions required')
        for condition,record in sorted(conditions.items()):
            users=record['users']
            if not isinstance(users,dict) or not users:raise ValueError('actual raw user statistics required')
            cells={}
            for user,r in sorted(users.items()):
                if not isinstance(user,str) or not user.strip():raise ValueError('nonempty private user key required')
                samples=r['samples'];zero=r['zero_valid_samples'];joints=r['by_joint']
                if (type(samples) is not int or samples<=0 or type(zero) is not int or not 0<=zero<=samples or
                        not isinstance(joints,list) or len(joints)!=20):
                    raise ValueError('raw sample/zero-valid counts and20 joints required')
                counts=[];sums=[]
                for i,(joint,name) in enumerate(zip(joints,JOINT_NAMES)):
                    count=joint['valid_angle_count'];error=joint['absolute_error_sum_rad']
                    if (type(joint['index']) is not int or joint['index']!=i or joint['name']!=name or
                            type(count) is not int or count<0 or type(error) not in (int,float) or
                            not math.isfinite(error) or error<0):
                        raise ValueError('original joint order, finite sums and exact counts required')
                    counts.append(count);sums.append(float(error))
                times=counts[0]
                if (len(set(counts))!=1 or not samples-zero<=times<=(samples-zero)*7280 or
                        (times==0 and any(sums))):
                    raise ValueError('raw generated-mask/sample conservation differs')
                cells[user]=dict(sums=np.asarray(sums,np.float64),valid_times=times,samples=samples,zero_valid=zero)
            rebuilt._conditions[condition]=cells
        if rebuilt.report()!=value:raise ValueError('raw joints disagree with cached condition/group/user statistics')
    except (KeyError,TypeError,IndexError,AttributeError,OverflowError) as error:
        raise ValueError('malformed raw20D statistics') from error
    return rebuilt


def summarize_tables(statistics,native_rows,native_users,*,require_all_conditions=True):
    """Return anonymous arithmetic views with explicitly different weights.

    Table11 uses native unweighted user statistics. Table13 weights each native
    condition's group MAE by its consumed window count, following the PDF's
    sample-weighted split caption; its independent FP64 counterpart is separate.
    An additional valid-angle-pooled view does not replace either definition.
    """
    if type(require_all_conditions) is not bool:raise ValueError('explicit boolean condition coverage policy required')
    raw=_raw_statistics(statistics);report=raw.report();conditions=report['by_condition']
    complete=set(conditions)==set(CONDITIONS)
    if require_all_conditions and not complete:raise ValueError('all three original test condition labels required')
    if any(not r['overall']['valid_angle_count'] for r in conditions.values()):
        raise ValueError('table results require valid labels in every consumed condition')
    raw.check_native(native_rows,native_users)
    aliases={u:'user-'+str(i+1).zfill(5) for i,u in enumerate(sorted({u for r in conditions.values() for u in r['users']}))}
    public=deepcopy(conditions);table11={};factor=180/math.pi
    for name,r in public.items():
        r['users']={aliases[u]:v for u,v in r['users'].items()}
        native=native_users[name];independent=r['per_user']
        table11[name]=dict(native_user_count=native['per_user_count'],
            native_mean_mae_deg=native['per_user_mae_mean']*factor,
            native_std_mae_deg=native['per_user_mae_std']*factor,
            independent_user_count=independent['user_count'],
            independent_mean_mae_deg=independent['mean_mae_deg'],
            independent_std_mae_deg=independent['std_mae_deg'],
            aggregation='unweighted users; population standard deviation ddof0')
    weights={c:r['samples'] for c,r in conditions.items()};samples=sum(weights.values())
    table13=dict(condition_sample_weights=weights,consumed_samples=samples,
        aggregation='sum(condition consumed windows * condition group MAE) / sum(condition consumed windows)',
        native_group_mae_deg={g:math.fsum(native_rows[c]['test_mae_'+g]*weights[c] for c in conditions)/samples*factor for g in GROUPS},
        independent_group_mae_deg={g:math.fsum(conditions[c]['by_group'][g]['mae_deg']*weights[c] for c in conditions)/samples for g in GROUPS})
    sums=[math.fsum(r['by_joint'][i]['absolute_error_sum_rad'] for r in conditions.values()) for i in range(20)]
    pooled=_cell_report(sums,sum(r['overall']['valid_angle_count']//20 for r in conditions.values()),
        samples,sum(r['zero_valid_samples'] for r in conditions.values()))
    return dict(format='emg2pose_table_arithmetic_v1',scope=SCOPE,
        condition_labels_complete=complete,by_condition=public,table11=table11,table13=table13,
        all_condition_consumptions=pooled,
        population_basis='ordered native condition consumptions; overlapping windows are not deduplicated',
        original_metric_policy=comparison_policy(),scientific_complete=False,full_paper_complete=False)
