"""Pinned original geometry on admitted standard legacy point payloads.

The input command validates data only. Native mesh execution is a separate
action; neither an input receipt nor a bilateral probe is a complete role.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
ADMISSION_SHA='757eef807837c8c44a6be6cbe0b237473f30953a543477dfc13fc32543c96b6a'
CALIBRATION_SHA='2b94a3c887e5aff600dc2d555121b74ac6cbd6670667e2f4c5c8688a434b3f8a'
ROLE_COUNTS=dict(val=1576,test=4306)
RESOLUTION_POLICY='author_fallback_1920x1080'
UPSTREAM_REVISION='f7cbd048b7092139852270491d3859cd472e0311'


def file_sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


IMPORTED_SHA=file_sha(__file__)


def read_json(path):
    def reject(value):raise ValueError('nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def validate_resolution_policy(policy):
    if type(policy)is not str or policy!=RESOLUTION_POLICY:
        raise ValueError('explicit unobserved author_fallback_1920x1080 policy required')
    return dict(name=policy,width=1920,height=1080,actual_episode_video_dimensions_proven=False,
        original_intrinsics_scaling_unchanged=True,distortion_applied=False)


def _array_sha(array):
    import numpy as np
    h=hashlib.sha256(json.dumps([array.dtype.str,list(array.shape)],separators=(',',':')).encode())
    h.update(np.ascontiguousarray(array).tobytes());return h.hexdigest()


def _token(data):
    value={k:data[k]for k in ('role','unique_hand_samples','admission','calibration','files_sha256')}
    value.update(arrays={k:_array_sha(a)for k,a in data['arrays'].items()},
        metadata={k:_array_sha(a)for k,a in data['metadata'].items()})
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_inputs(data):
    data['_guard'].check()
    if file_sha(__file__)!=IMPORTED_SHA or _token(data)!=data['_token']:
        raise ValueError('admitted native geometry input content/schema/source changed')
    if any(a.flags.writeable for a in (*data['arrays'].values(),*data['metadata'].values())):
        raise ValueError('admitted native geometry arrays must remain readonly')


def _typed_schema(arrays,n):
    import numpy as np
    expected=dict(center=(np.int64,(2*n,)),episode=(np.int64,(2*n,)),hand=(np.int8,(2*n,)),
        camera_transform=(np.float32,(n,12)),camera_tracked=(np.bool_,(n,)),
        image_stale=(np.bool_,(n,)),image_frame_index=(np.int32,(n,)))
    for hand in ('left','right'):
        expected.update({hand+'_pose':(np.float32,(n,48)),hand+'_world_transform':(np.float32,(n,12)),
            hand+'_marker_valid':(np.bool_,(n,21)),hand+'_beta':(np.float32,(n,10))})
    if set(arrays)!=set(expected):raise ValueError('complete exact native point fields required; beta cannot fall back')
    for name,(dtype,shape)in expected.items():
        a=arrays[name]
        if a.dtype!=np.dtype(dtype)or a.shape!=shape or(a.dtype.kind=='f'and not np.isfinite(a).all()):
            raise ValueError('native point dtype/shape/finite content differs: '+name)
    if (not np.array_equal(arrays['hand'],np.repeat(np.array([0,1],np.int8),n))
        or not np.array_equal(arrays['center'][:n],arrays['center'][n:])
        or not np.array_equal(arrays['episode'][:n],arrays['episode'][n:])
        or len(np.unique(arrays['center'][:n]))!=n):
        raise ValueError('native complete point hand/center ordering differs')


def load_point_inputs(points,admission,metadata,calibration,role):
    if type(role)is not str or role not in ROLE_COUNTS:
        raise ValueError('explicit val/test geometry role required')
    import numpy as np
    from egoemg_repro.classic_emg_cohort import _safe,_StatGuard
    paths={k:_safe(v)for k,v in dict(points=points,admission=admission,metadata=metadata,calibration=calibration).items()}
    guard=_StatGuard([*paths.values(),Path(__file__)])
    digests={k:file_sha(p)for k,p in paths.items()};guard.check()
    if digests['admission']!=ADMISSION_SHA or digests['calibration']!=CALIBRATION_SHA:
        raise ValueError('fixed admitted geometry reference/calibration bytes differ')
    reference=read_json(paths['admission'])
    if (reference['format']!='actual_legacy_complete_geometry_numeric_scan_v1'
        or reference['full_numeric_scan_complete']is not True or reference['structural_admission_passed']is not True
        or reference['scientific_eligible']is not False or reference['full_paper_complete']is not False):
        raise ValueError('actual complete numeric admission required, not a scientific result')
    packet=reference['point_payloads'][role];population=reference['native_point_geometry_roles'][role]
    if (digests['points']!=packet['sha256']or paths['points'].stat().st_size!=packet['bytes']
        or digests['metadata']!=reference['metadata_sha256']
        or population['unique_hand_samples']!=ROLE_COUNTS[role]
        or population['repeated_consumptions']!=2*ROLE_COUNTS[role]):
        raise ValueError('complete original geometry role/file/metadata binding differs')
    n=ROLE_COUNTS[role]//2
    if any(population['by_hand'][h]['unique_hand_centers']!=n for h in ('left','right')):
        raise ValueError('complete native geometry hand counts differ')
    with np.load(paths['points'],allow_pickle=False)as f:arrays={k:f[k].copy()for k in f.files}
    _typed_schema(arrays,n)
    with np.load(paths['metadata'],allow_pickle=False)as f:meta={k:f[k].copy()for k in f.files}
    starts=meta['episode_start_idx'];ends=meta['episode_end_idx'];lengths=meta['episode_length'];beta_ids=meta['episode_beta_idx']
    episodes=reference['episode_count']
    if (any(a.shape!=(episodes,)or a.dtype.kind not in 'iu'for a in (starts,ends,lengths,beta_ids))
        or starts[0]!=0 or ends[-1]!=reference['total_rows']or not np.array_equal(ends-starts,lengths)
        or not np.array_equal(starts[1:],ends[:-1])or not np.array_equal(beta_ids,np.arange(episodes))):
        raise ValueError('actual complete native geometry episode/beta indexing differs')
    center=arrays['center'][:n];ep=arrays['episode'][:n]
    if np.any(ep<0)or np.any(ep>=episodes)or np.any(center<starts[ep])or np.any(center>=ends[ep]):
        raise ValueError('native point center is outside its recorded episode')
    video_field='episode_head_video_path'if'episode_head_video_path'in meta else'episode_webcam_video_path'
    videos=meta[video_field]
    if videos.shape!=(episodes,)or videos.dtype.kind not in 'SU':raise ValueError('actual episode video path mapping required')
    for value in videos:
        p=Path(bytes(value).decode().rstrip('\x00')if videos.dtype.kind=='S'else str(value))
        if p.is_absolute()or'..'in p.parts:raise ValueError('episode video mapping must remain a relative released path')
    calib=read_json(paths['calibration']);K=np.asarray(calib['camera_matrix'],np.float64)
    distortion=np.asarray(calib['distortion_coefficients'],np.float64)
    camera=reference['camera']
    if (K.shape!=(3,3)or not np.isfinite(K).all()or not np.isfinite(distortion).all()
        or K.tolist()!=camera['matrix']or distortion.size!=camera['distortion_coefficient_count']
        or [calib['image_width'],calib['image_height']]!=camera['declared_calibration_resolution']):
        raise ValueError('actual calibration matrix/resolution/distortion binding differs')
    for a in (*arrays.values(),*meta.values()):a.flags.writeable=False
    data=dict(role=role,unique_hand_samples=ROLE_COUNTS[role],arrays=arrays,metadata=meta,calibration=calib,
        admission=reference,files_sha256=digests,_paths=paths,_guard=guard,_video_field=video_field)
    data['_token']=_token(data);check_inputs(data);return data


def select_points(data,mode):
    if type(mode)is not str or mode not in ('complete_role','bilateral_probe'):
        raise ValueError('explicit complete_role or bilateral_probe geometry selection required')
    import numpy as np
    check_inputs(data);a=data['arrays'];n=data['unique_hand_samples']//2
    if mode=='complete_role':return[(h,i)for h in ('left','right')for i in range(n)]
    chosen=[]
    for hand in ('left','right'):
        good=np.any(a[hand+'_pose']!=0,axis=1)&a['camera_tracked']&a[hand+'_marker_valid'].any(axis=1)
        rows=np.flatnonzero(good)
        if not len(rows):raise ValueError('bilateral native probe needs one actual valid tracked pose per hand')
        chosen.append((hand,int(rows[0])))
    return chosen


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    validate=sub.add_parser('validate',help='data validation only; never a geometry execution')
    for name in ('points','admission','metadata','calibration'):validate.add_argument('--'+name,required=True,type=Path)
    validate.add_argument('--role',required=True,choices=('val','test'))
    validate.add_argument('--mode',required=True,choices=('bilateral_probe','complete_role'))
    validate.add_argument('--resolution-policy',required=True,choices=(RESOLUTION_POLICY,))
    return p


def main():
    a=parser().parse_args();policy=validate_resolution_policy(a.resolution_policy)
    data=load_point_inputs(a.points,a.admission,a.metadata,a.calibration,a.role);selection=select_points(data,a.mode)
    print(json.dumps(dict(format='native_point_geometry_input_contract_v1',role=a.role,mode=a.mode,
        actual_selected_points=len(selection),input_sha256=data['files_sha256'],resolution_policy=policy,
        native_geometry_executed=False,scientific_eligible=False,full_paper_complete=False)))


if __name__=='__main__':main()
