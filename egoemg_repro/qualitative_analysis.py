"""Original Figure7 errors, tracking filter, stable ordering and CSV precision.

Numeric joins do not authenticate a trained producer, input cohort or render.
Keep all selected rows and both masks, even when they are not source-filtered.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess

import numpy as np

from . import UPSTREAM_COMMIT
from .classic_emg import _approved_upstream, _hash
from .classic_emg_cohort import _safe
from .data import file_sha256
from .native_ik_inputs import _Guard
from .occlusion_dense_inference import native_batch_ranges
from .occlusion_dense_predictions import KEY_FIELDS, WINDOW_LENGTHS, _key

IMPORTED_SHA=file_sha256(__file__)
SOURCE='scripts/eval/find_fusion_wins.py'


def rank_qualitative_predictions(keys,arrays,flags,*,recipe,upstream):
    """Reject changed/missing keys or arrays; never filter by label/crop mask."""
    import torch
    if recipe not in WINDOW_LENGTHS:raise ValueError('original qualitative recipe required')
    root=_approved_upstream(upstream);path=_safe(root/SOURCE)
    guard=_Guard([_safe(__file__),path],[])
    source=path.read_bytes()
    original=subprocess.run(['git','-C',str(root),'show',UPSTREAM_COMMIT+':'+SOURCE],capture_output=True,timeout=30)
    if original.returncode or source!=original.stdout or file_sha256(__file__)!=IMPORTED_SHA:
        raise ValueError('pinned original qualitative analysis source required')
    keys=copy.deepcopy(list(keys));flags=copy.deepcopy(list(flags));n=len(keys)
    native_batch_ranges(keys)
    if (not n or len(flags)!=n or not isinstance(arrays,dict)
        or any(k['window_end']-k['window_start']!=WINDOW_LENGTHS[recipe]
            or any(type(k.get(v))is not int or k[v]<0 for v in ('ep_idx','center_idx','window_start','window_end','native_index'))for k in keys)):
        raise ValueError('complete ordered ordinary keys, flags and recipe windows required')
    required={'targets','vision_prediction','emg_prediction','fusion_prediction','emg_full_prediction',
        'label_valid_mask','crop_valid_mask','mae_vision_rad','mae_emg_rad','mae_fusion_rad'}
    if set(arrays)!=required:raise ValueError('all original prediction, target, mask and error arrays required')
    for name,a in arrays.items():
        if not isinstance(a,np.ndarray)or not a.flags.c_contiguous:
            raise ValueError('actual contiguous raw array required')
        if name in ('label_valid_mask','crop_valid_mask'):
            okay=a.dtype==np.bool_ and a.shape==(n,1)
        elif name.startswith('mae_'):
            okay=a.dtype==np.float64 and a.shape==(n,)and np.isfinite(a).all()and np.all(a>=0)
        else:
            okay=(a.dtype==np.float32 and np.isfinite(a).all()
                and (a.ndim==3 and a.shape[:2]==(n,22)and a.shape[2]>0
                     if name=='emg_full_prediction'else a.shape==(n,22)))
        if not okay:raise ValueError('finite FP32 original22 predictions/targets and exact boolean/error shapes required')
    full=arrays['emg_full_prediction'];mid=full[:,:,full.shape[2]//2]
    if mid.tobytes()!=arrays['emg_prediction'].tobytes():
        raise ValueError('actual original full-EMG midpoint prediction required')
    rows=[]
    for i,(k,f)in enumerate(zip(keys,flags)):
        if (_key(k,KEY_FIELDS)!=_key(f,KEY_FIELDS)
            or any(type(f.get(v))is not bool for v in ('mocap_tracked','image_stale','markers_valid'))):
            raise ValueError('same complete ordered key and actual boolean tracking flags required')
        row=dict(key=k,**k,mocap_tracked=f['mocap_tracked'],image_stale=f['image_stale'],
            markers_valid=f['markers_valid'],label_valid=bool(arrays['label_valid_mask'][i,0]),
            crop_valid=bool(arrays['crop_valid_mask'][i,0]))
        for role in ('vision','emg','fusion'):
            value=float((torch.from_numpy(arrays[role+'_prediction'][i])
                -torch.from_numpy(arrays['targets'][i])).abs().mean())
            if value!=arrays['mae_'+role+'_rad'][i]:
                raise ValueError('saved error differs from original FP32 unmasked22 reduction')
            row['mae_'+role]=value
        row['delta_mae']=row['mae_vision']-row['mae_fusion']
        row['source_filter_pass']=row['mocap_tracked']and not row['image_stale']and row['markers_valid']
        rows.append(row)
    # Original Python sort is stable and compares unrounded radian deltas.
    ranked=sorted((r for r in rows if r['source_filter_pass']),key=lambda r:r['delta_mae'],reverse=True)
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    csv=next(n.value for n in main.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='csv_rows'for t in n.targets))
    csv_rows=eval(compile(ast.Expression(body=csv),str(path),'eval'),dict(all_results=ranked))
    result=dict(format='original_qualitative_unmasked22_filter_ranking_v1',recipe=recipe,
        source_files_sha256={SOURCE:file_sha256(path),'qualitative_analysis.py':IMPORTED_SHA},
        source_key_sha256=_hash(keys),raw_rows=rows,ranked_rows=ranked,csv_rows=csv_rows,
        counts=dict(selected=n,tracked_filter_pass=len(ranked),rejected=n-len(ranked),
            crop_invalid_retained=sum(not r['crop_valid']for r in ranked),
            label_invalid_retained=sum(not r['label_valid']for r in ranked)),
        ranking_available=bool(ranked),degree_factor=57.3,csv_error_round_decimals=2,
        source_error_uses_mask=False,error_reduction='FP32 per-sample mean of all22 absolute angle errors',
        ranking='stable descending unrounded vision-minus-fusion radians',
        source_filter='tracked and not image-stale and any hand marker valid',
        trained_model_or_full_input_admitted=False,figure7_reproduced=False,
        scientific_eligible=False,full_paper_complete=False)
    guard.check();return result
