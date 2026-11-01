"""Pinned original geometry on admitted standard legacy point payloads.

The input command validates data only. Native mesh execution is a separate
action; neither an input receipt nor a bilateral probe is a complete role.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.machinery import PathFinder
import inspect
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
ADMISSION_SHA='757eef807837c8c44a6be6cbe0b237473f30953a543477dfc13fc32543c96b6a'
CALIBRATION_SHA='2b94a3c887e5aff600dc2d555121b74ac6cbd6670667e2f4c5c8688a434b3f8a'
ROLE_COUNTS=dict(val=1576,test=4306)
RESOLUTION_POLICY='author_fallback_1920x1080'
UPSTREAM_REVISION='f7cbd048b7092139852270491d3859cd472e0311'
MANO_ASSET_SHA='45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767'
MANO_LICENSE_SHA='75fc4da03259504ce9a4ed2418ab98dd59647add7ff95e7691c4d55fb0e3cf63'


def geometry_policy():
    return dict(right_mano_for_both_hands=True,flat_hand_mean=False,global_orient='native zero',
        hand_pose_slice=[3,48],left_mirror_axis='X',left_faces_reverse=True,
        legacy_manifest_convention='flip_local_z; source script X mirror retained and not claimed reconciled',
        depth_eps=.005,window_half=2,actual_betas=True,zero_beta_fallback=False)


def _require_equal(value,expected,label):
    """JSON equality with strict scalar types (in particular bool is not int)."""
    if type(value)is not type(expected):raise ValueError(label+' type differs')
    if isinstance(expected,dict):
        if value.keys()!=expected.keys():raise ValueError(label+' fields differ')
        for key in expected:_require_equal(value[key],expected[key],label+'.'+key)
    elif isinstance(expected,list):
        if len(value)!=len(expected):raise ValueError(label+' length differs')
        for actual,wanted in zip(value,expected):_require_equal(actual,wanted,label)
    elif value!=expected:raise ValueError(label+' differs')


def _require_fields(value,fields,label):
    if type(value)is not dict or set(value)!=set(fields):raise ValueError(label+' exact fields required')


def _check_upstream(upstream):
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM_REVISION:
        raise ValueError('pinned original geometry revision required')
    if subprocess.check_output(['git','-C',str(upstream),'status','--porcelain','--untracked-files=all'],text=True).strip():
        raise ValueError('original geometry source checkout changed')


def _module_directory(name):
    """Locate dependency providers before executing their source."""
    from egoemg_repro.classic_emg_cohort import _safe
    loaded=sys.modules.get(name)
    source=getattr(loaded,'__file__',None)if loaded is not None else None
    if source is None:
        spec=PathFinder.find_spec(name,sys.path)
        source=spec.origin if spec is not None else None
    if not source:raise ValueError('native geometry dependency provider missing: '+name)
    return _safe(source).parent


def _runtime_source_paths(upstream,assets,chumpy_dir,smplx_dir):
    from egoemg_repro.classic_emg_cohort import _safe,_source_paths
    return {_safe(p)for p in {*_source_paths(upstream),upstream/'scripts/eval/evaluate_occlusion_fusion.py',
        *chumpy_dir.glob('chumpy/**/*.py'),*smplx_dir.glob('**/*.py'),Path(__file__),
        assets/'MANO_RIGHT.pkl',assets/'LICENSE.txt'}}


def _fallback_intrinsics(calib):
    """Original get_episode_intrinsics scaling/crop, without importing Torch."""
    import numpy as np
    video_w,video_h=1920,1080;calib_w=int(calib['image_width']);calib_h=int(calib['image_height'])
    raw=np.asarray(calib['camera_matrix'],np.float64);calib_aspect=calib_w/calib_h;video_aspect=video_w/video_h
    if abs(video_aspect-calib_aspect)<.02:active_x0=0.;active_x1=float(video_w)
    else:
        active_h=float(video_h);active_w=active_h*calib_aspect
        active_x0=(video_w-active_w)/2.;active_x1=active_x0+active_w
    K=np.eye(3,dtype=np.float64)
    K[0,0]=raw[0,0]*(active_x1-active_x0)/calib_w;K[1,1]=raw[1,1]*video_h/calib_h
    K[0,2]=raw[0,2]*(active_x1-active_x0)/calib_w+active_x0;K[1,2]=raw[1,2]*video_h/calib_h
    return K


def _row_inputs(data,hand,index):
    a=data['arrays']
    return dict(hand=hand,point_index=index,ep_idx=int(a['episode'][index]),center_idx=int(a['center'][index]),
        image_frame_index=int(a['image_frame_index'][index]),mocap_tracked=bool(a['camera_tracked'][index]),
        image_stale=bool(a['image_stale'][index]),markers_valid=bool(a[hand+'_marker_valid'][index].any()),
        marker_valid_count=int(a[hand+'_marker_valid'][index].sum()),pose_sha256=_array_sha(a[hand+'_pose'][index]),
        beta_sha256=_array_sha(a[hand+'_beta'][index]),world_transform_sha256=_array_sha(a[hand+'_world_transform'][index]),
        camera_transform_sha256=_array_sha(a['camera_transform'][index]))


def _packet_schema(arrays):
    import numpy as np
    expected=dict(verts_world=(np.float64,(778,3)),verts_cam=(np.float64,(778,3)),faces=(np.uint32,(1538,3)),
        K=(np.float64,(3,3)),pose=(np.float32,(48,)),beta=(np.float32,(10,)),world_transform=(np.float32,(12,)),
        camera_transform=(np.float32,(12,)),visible=(np.bool_,(778,)),area_weights=(np.float64,(778,)),
        zbuffer=(np.float64,(1080,1920)),u_proj=(np.int32,(778,)),v_proj=(np.int32,(778,)))
    _require_fields(arrays,expected,'native geometry packet')
    for name,(dtype,shape)in expected.items():
        a=arrays[name]
        if a.dtype!=np.dtype(dtype)or a.shape!=shape:raise ValueError('native output dtype/shape differs: '+name)
        if a.dtype.kind=='f'and name!='zbuffer'and not np.isfinite(a).all():raise ValueError('nonfinite native output: '+name)
    if np.any(arrays['faces']>=778)or np.any(arrays['area_weights']<0):raise ValueError('invalid native face/area weights')
    z=arrays['zbuffer']
    if np.any(np.isnan(z))or np.any(z<=0):raise ValueError('invalid native z-buffer depths')


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


def _json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.writing')
    with temp.open('x')as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def _load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def geometry_frame(native,data,hand,index,*,video_root,device):
    import numpy as np
    if type(hand)is not str or hand not in ('left','right')or type(index)is not int or not 0<=index<data['unique_hand_samples']//2:
        raise ValueError('explicit native hand and bounded point index required')
    check_inputs(data);a=data['arrays'];pose=a[hand+'_pose'][index];beta=a[hand+'_beta'][index]
    world=np.asarray(a[hand+'_world_transform'][index],np.float64)
    camera=np.asarray(a['camera_transform'][index],np.float64)
    ep=int(a['episode'][index]);row=_row_inputs(data,hand,index)
    verts_world,faces=native.get_mano_verts_world(np.asarray(pose,np.float64),beta,world[:9].reshape(3,3),world[9:12],hand=='left',device)
    if not np.any(pose):
        if verts_world is not None or faces is not None:raise RuntimeError('native zero pose unexpectedly produced geometry')
        return dict(row,status='invalid_zero_pose',occlusion_score=None,visible_ratio=None),None
    if verts_world is None or faces is None:raise RuntimeError('native mesh function failed for a valid admitted pose')
    if verts_world.shape!=(778,3)or faces.shape!=(1538,3)or not np.isfinite(verts_world).all():
        raise RuntimeError('native mesh layout or finite vertices differ')
    dataset=SimpleNamespace(_episode_head_video_path=data['metadata'][data['_video_field']])
    K,width,height=native.get_episode_intrinsics(ep,data['_paths']['metadata'].parent,Path(video_root),dataset,data['calibration'])
    if (width,height)!=(1920,1080):raise ValueError('explicit author fallback did not return its declared resolution')
    # Unmodified four-line world-to-camera operation from evaluate_all_samples.
    T_W_C=np.eye(4,dtype=np.float64);T_W_C[:3,:3]=camera[:9].reshape(3,3);T_W_C[:3,3]=camera[9:12]
    T_C_W=np.linalg.inv(T_W_C)
    R_C_W=T_C_W[:3,:3].astype(np.float64)
    t_C_W=T_C_W[:3,3].astype(np.float64)
    verts_cam=(R_C_W@verts_world.T).T+t_C_W
    occ=native.compute_self_occlusion(verts_cam,faces,K,height,width,depth_eps=.005,window_half=2)
    score=float(occ['occlusion_score']);ratio=float(occ['visible_ratio'])
    if not math.isfinite(score)or not 0.0<=score<=1.0 or score!=1.0-ratio:
        raise RuntimeError('native self-occlusion produced an invalid area-weighted result')
    arrays=dict(verts_world=verts_world,verts_cam=verts_cam,faces=faces,K=K,
        pose=pose.copy(),beta=beta.copy(),world_transform=a[hand+'_world_transform'][index].copy(),
        camera_transform=a['camera_transform'][index].copy(),
        **{k:occ[k]for k in ('visible','area_weights','zbuffer','u_proj','v_proj')})
    _packet_schema(arrays)
    if not np.array_equal(K,_fallback_intrinsics(data['calibration'])):
        raise ValueError('original native fallback intrinsics differ from admitted calibration')
    row.update(status='native_geometry',occlusion_score=score,visible_ratio=ratio,
        author_occlusion_score_round6=round(score,6),author_visible_ratio_round6=round(ratio,6),
        width=width,height=height,positive_depth_vertices=int(np.count_nonzero(verts_cam[:,2]>1e-8)),
        visible_vertices=int(np.count_nonzero(occ['visible'])),area_weight_sum=float(occ['area_weights'].sum()),
        arrays_sha256={k:_array_sha(v)for k,v in arrays.items()})
    return row,arrays


def _native_runtime(args):
    from egoemg_repro.classic_emg import _activate
    from egoemg_repro.classic_emg_cohort import _safe,_StatGuard,_source_paths
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('native geometry requires explicitly hidden CUDA')
    upstream=_safe(args.upstream);assets=_safe(args.mano_assets);chumpy_dir=_safe(args.chumpy_dir)
    sys.path.insert(0,str(chumpy_dir))
    chumpy_package=_module_directory('chumpy');smplx_dir=_module_directory('smplx')
    if chumpy_package!=chumpy_dir/'chumpy':raise ValueError('admitted Chumpy provider required')
    paths=_runtime_source_paths(upstream,assets,chumpy_dir,smplx_dir)
    guard=_StatGuard(paths)
    _check_upstream(upstream)
    model=assets/'MANO_RIGHT.pkl';license=assets/'LICENSE.txt'
    if file_sha(model)!=MANO_ASSET_SHA or file_sha(license)!=MANO_LICENSE_SHA:
        raise ValueError('previously admitted licensed native MANO asset required')
    sources={str(p):file_sha(p)for p in sorted(paths)};guard.check()
    import numpy as np
    aliases=[]
    for name,value in dict(bool=bool,int=int,float=float,complex=complex,object=object,unicode=str,str=str).items():
        if name not in np.__dict__:setattr(np,name,value);aliases.append(name)
    inspect_alias=not hasattr(inspect,'getargspec')
    if inspect_alias:inspect.getargspec=inspect.getfullargspec
    import chumpy,smplx,torch
    guard.check()
    if not Path(chumpy.__file__).resolve().is_relative_to(chumpy_dir)or chumpy.__version__!='0.70':
        raise ValueError('admitted original Chumpy0.70 source required')
    if _module_directory('chumpy')!=chumpy_package or _module_directory('smplx')!=smplx_dir:
        raise ValueError('native geometry dependency provider changed during import')
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('native geometry cannot initialize CUDA')
    def deny(*a,**k):raise RuntimeError('native geometry cannot download or construct a learned pose model/optimizer')
    socket.socket.connect=deny;socket.create_connection=deny;torch.optim.AdamW.__init__=deny
    _activate(upstream)
    native=_load_source('_pinned_author_geometry_evaluation',upstream/'scripts/eval/evaluate_occlusion_fusion.py')
    guard.check()
    if _runtime_source_paths(upstream,assets,chumpy_dir,smplx_dir)!=paths:
        raise ValueError('native geometry source membership changed during imports')
    for cls in (native.ResNetVisionPose,native.VisionViTPose,native.Emg2PoseFormer):cls.__init__=deny
    native.MANO_MODEL_PATH=str(assets)
    if native._MANO_LAYER is not None:raise RuntimeError('native mesh layer was constructed before the owned geometry call')
    layer=native._get_mano_layer('cpu')
    guard.check()
    if _runtime_source_paths(upstream,assets,chumpy_dir,smplx_dir)!=paths:
        raise ValueError('native geometry source membership changed during layer construction')
    if not isinstance(layer,smplx.MANO)or layer.faces.shape!=(1538,3):raise RuntimeError('actual original right MANO geometry layer required')
    return native,guard,dict(source_files_sha256=sources,source_bundle_sha256=_json_sha(sources),
        upstream=str(upstream),upstream_revision=UPSTREAM_REVISION,mano_assets=str(assets),chumpy_dir=str(chumpy_dir),
        smplx_dir=str(smplx_dir),mano_asset_sha256=MANO_ASSET_SHA,
        versions=dict(torch=torch.__version__,numpy=np.__version__,chumpy=chumpy.__version__),
        numpy_aliases=aliases,inspect_getargspec_backfilled=inspect_alias,
        original_model_source_files_modified=False,actual_right_mano_geometry_layers_constructed=1)


def run_geometry(args):
    policy=validate_resolution_policy(args.resolution_policy)
    import numpy as np
    from egoemg_repro.classic_emg_cohort import _safe
    data=load_point_inputs(args.points,args.admission,args.metadata,args.calibration,args.role)
    selected=select_points(data,args.mode);output=_safe(args.output,exists=False)
    if output.exists():raise ValueError('new native geometry output required; existing attempts are preserved')
    output.mkdir(mode=0o700);(output/'points').mkdir(mode=0o700);video_root=output/'unavailable-videos';video_root.mkdir(mode=0o700)
    started=time.monotonic();native,guard,runtime=_native_runtime(args)
    identity=dict(format='original_native_point_geometry_identity_v1',role=args.role,mode=args.mode,
        source_runner_sha256=IMPORTED_SHA,input_paths={k:str(v)for k,v in data['_paths'].items()},
        input_sha256=data['files_sha256'],input_token=data['_token'],selection=selected,
        resolution_policy=policy,runtime=runtime,
        geometry_policy=geometry_policy(),
        scientific_eligible=False,full_paper_complete=False)
    identity['sha256']=_json_sha(identity);_write_json(output/'identity.json',identity)
    counts=dict(actual_mesh_calls=0,actual_occlusion_calls=0,invalid_zero_pose_rows=0);rows=[]
    with(output/'points.jsonl').open('x')as journal:
        for seq,(hand,index)in enumerate(selected):
            guard.check();row,arrays=geometry_frame(native,data,hand,index,video_root=video_root,device='cpu')
            counts['actual_mesh_calls']+=1
            if arrays is None:counts['invalid_zero_pose_rows']+=1
            else:
                counts['actual_occlusion_calls']+=1;packet=output/'points'/f'{seq:06d}-{hand}-{row["center_idx"]}.npz'
                np.savez_compressed(packet,**arrays);packet.chmod(0o600)
                row.update(packet_file=str(packet.relative_to(output)),packet_file_sha256=file_sha(packet))
            row.update(sequence=seq,identity_sha256=identity['sha256']);rows.append(row)
            journal.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
            _write_json(output/'progress.json',dict(completed_points=len(rows),expected_points=len(selected),mode=args.mode))
    check_inputs(data);guard.check()
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('native geometry initialized CUDA')
    result=dict(format='original_native_point_geometry_result_v1',identity_sha256=identity['sha256'],
        journal_file_sha256=file_sha(output/'points.jsonl'),role=args.role,mode=args.mode,rows=len(rows),
        complete_standard_role=args.mode=='complete_role'and len(rows)==ROLE_COUNTS[args.role],
        **counts,actual_right_mano_geometry_layers_constructed=runtime['actual_right_mano_geometry_layers_constructed'],
        learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,
        resolution_policy=policy,all_input_and_source_generations_unchanged=True,seconds=time.monotonic()-started,
        dense_grid_or_paired_gain_complete=False,scientific_eligible=False,full_paper_complete=False)
    result['sha256']=_json_sha(result);_write_json(output/'result.json',result);return verify_geometry(output)


def _verify_runtime(runtime,guard):
    from egoemg_repro.classic_emg_cohort import _safe
    _require_fields(runtime,('source_files_sha256 source_bundle_sha256 upstream upstream_revision mano_assets chumpy_dir '
        'smplx_dir mano_asset_sha256 versions numpy_aliases inspect_getargspec_backfilled '
        'original_model_source_files_modified actual_right_mano_geometry_layers_constructed').split(),'native runtime')
    upstream=_safe(runtime['upstream']);assets=_safe(runtime['mano_assets'])
    chumpy_dir=_safe(runtime['chumpy_dir']);smplx_dir=_safe(runtime['smplx_dir'])
    required=_runtime_source_paths(upstream,assets,chumpy_dir,smplx_dir);sources=runtime['source_files_sha256']
    if type(sources)is not dict or set(sources)!={str(p)for p in required}:
        raise ValueError('complete native geometry source/provider coverage differs')
    for p in required:guard.add(p)
    _check_upstream(upstream)
    _require_equal(runtime['upstream_revision'],UPSTREAM_REVISION,'upstream revision')
    _require_equal(runtime['source_bundle_sha256'],_json_sha(sources),'source bundle')
    _require_equal(runtime['mano_asset_sha256'],MANO_ASSET_SHA,'admitted MANO hash')
    _require_equal(runtime['original_model_source_files_modified'],False,'source modification declaration')
    _require_equal(runtime['actual_right_mano_geometry_layers_constructed'],1,'native geometry layer count')
    if (type(runtime['versions'])is not dict or set(runtime['versions'])!={'torch','numpy','chumpy'}
        or any(type(v)is not str or not v for v in runtime['versions'].values())or runtime['versions']['chumpy']!='0.70'
        or type(runtime['numpy_aliases'])is not list or any(type(v)is not str for v in runtime['numpy_aliases'])
        or type(runtime['inspect_getargspec_backfilled'])is not bool):raise ValueError('native dependency/compatibility declaration differs')
    for p in required:
        if file_sha(p)!=sources[str(p)]:raise ValueError('original native geometry helper/asset source differs')
    if file_sha(assets/'MANO_RIGHT.pkl')!=MANO_ASSET_SHA or file_sha(assets/'LICENSE.txt')!=MANO_LICENSE_SHA:
        raise ValueError('previously admitted licensed native MANO asset differs')
    guard.check();return upstream/'egoemg/occlusion.py'


def verify_geometry(output):
    import numpy as np
    from egoemg_repro.classic_emg_cohort import _safe,_StatGuard
    output=_safe(output)
    if not(output/'identity.json').is_file()or not(output/'result.json').is_file():raise ValueError('complete native geometry output required')
    guard=_StatGuard([output/'identity.json',output/'result.json',output/'points.jsonl'])
    identity=read_json(output/'identity.json');result=read_json(output/'result.json')
    _require_fields(identity,('format role mode source_runner_sha256 input_paths input_sha256 input_token selection '
        'resolution_policy runtime geometry_policy scientific_eligible full_paper_complete sha256').split(),'native identity')
    _require_fields(result,('format identity_sha256 journal_file_sha256 role mode rows complete_standard_role actual_mesh_calls '
        'actual_occlusion_calls invalid_zero_pose_rows actual_right_mano_geometry_layers_constructed learned_pose_models_constructed '
        'optimizer_updates cuda_initialized resolution_policy all_input_and_source_generations_unchanged seconds '
        'dense_grid_or_paired_gain_complete scientific_eligible full_paper_complete sha256').split(),'native result')
    for value in (identity,result):
        if value['sha256']!=_json_sha({k:v for k,v in value.items()if k!='sha256'}):raise ValueError('native geometry result seal differs')
    _require_equal(identity['format'],'original_native_point_geometry_identity_v1','native identity format')
    _require_equal(result['format'],'original_native_point_geometry_result_v1','native result format')
    policy=validate_resolution_policy(RESOLUTION_POLICY)
    for doc in (identity,result):
        _require_equal(doc['resolution_policy'],policy,'canonical fallback resolution')
        _require_equal(doc['scientific_eligible'],False,'scientific scope')
        _require_equal(doc['full_paper_complete'],False,'full paper scope')
    _require_equal(identity['geometry_policy'],geometry_policy(),'original geometry policy')
    if identity['source_runner_sha256']!=IMPORTED_SHA or file_sha(__file__)!=IMPORTED_SHA:
        raise ValueError('native geometry runner source differs')
    source=_verify_runtime(identity['runtime'],guard)
    paths=identity['input_paths'];_require_fields(paths,('points','admission','metadata','calibration'),'native input paths')
    data=load_point_inputs(paths['points'],paths['admission'],paths['metadata'],paths['calibration'],identity['role'])
    selected=select_points(data,identity['mode'])
    _require_equal(paths,{k:str(v)for k,v in data['_paths'].items()},'canonical native input paths')
    _require_equal(identity['input_sha256'],data['files_sha256'],'native input digests')
    _require_equal(identity['input_token'],data['_token'],'native input token')
    _require_equal(identity['selection'],[list(k)for k in selected],'complete native point selection')
    if result['identity_sha256']!=identity['sha256']or result['journal_file_sha256']!=file_sha(output/'points.jsonl'):
        raise ValueError('native geometry result/journal binding differs')
    rows=[json.loads(line)for line in(output/'points.jsonl').read_text().splitlines()]
    if len(rows)!=len(selected):raise ValueError('complete native geometry point population differs')
    kernel=_load_source('_pinned_geometry_verification_kernel',source);guard.check()
    native_rows=0;invalid=0
    native_fields=('author_occlusion_score_round6 author_visible_ratio_round6 width height positive_depth_vertices '
        'visible_vertices area_weight_sum arrays_sha256 packet_file packet_file_sha256').split()
    for seq,((hand,index),row)in enumerate(zip(selected,rows)):
        a=data['arrays'];pose=a[hand+'_pose'][index];beta=a[hand+'_beta'][index]
        expected=_row_inputs(data,hand,index);expected.update(sequence=seq,identity_sha256=identity['sha256'])
        common=set(expected)|{'status','occlusion_score','visible_ratio'}
        _require_fields(row,common if not np.any(pose)else common|set(native_fields),'native point row')
        for key,value in expected.items():_require_equal(row[key],value,'admitted point '+key)
        if not np.any(pose):
            _require_equal(row['status'],'invalid_zero_pose','zero pose status')
            _require_equal(row['occlusion_score'],None,'zero pose occlusion')
            _require_equal(row['visible_ratio'],None,'zero pose visibility')
            invalid+=1;continue
        expected_packet=f'points/{seq:06d}-{hand}-{row["center_idx"]}.npz'
        _require_equal(row['packet_file'],expected_packet,'native packet path')
        packet=_safe(output/row['packet_file']);guard.add(packet)
        if file_sha(packet)!=row['packet_file_sha256']:raise ValueError('actual native geometry packet bytes differ')
        with np.load(packet,allow_pickle=False)as f:arrays={k:f[k]for k in f.files}
        guard.check();_packet_schema(arrays)
        _require_equal(row['arrays_sha256'],{k:_array_sha(v)for k,v in arrays.items()},'typed native packet')
        for name,wanted in dict(pose=pose,beta=beta,world_transform=a[hand+'_world_transform'][index],
            camera_transform=a['camera_transform'][index],K=_fallback_intrinsics(data['calibration'])).items():
            if not np.array_equal(arrays[name],wanted):raise ValueError('admitted native packet '+name+' differs')
        camera=a['camera_transform'][index]
        T=np.eye(4);T[:3,:3]=camera[:9].reshape(3,3);T[:3,3]=camera[9:12];inverse=np.linalg.inv(T)
        expected_cam=(inverse[:3,:3].astype(np.float64)@arrays['verts_world'].T).T+inverse[:3,3].astype(np.float64)
        if not np.array_equal(expected_cam,arrays['verts_cam']):raise ValueError('actual native world-to-camera operation differs')
        u_float,v_float,valid_z=kernel._project_vertices(arrays['verts_cam'],arrays['K'])
        u=np.full(778,-1,dtype=np.int32);v=np.full(778,-1,dtype=np.int32)
        u[valid_z]=np.round(u_float[valid_z]).astype(np.int32);v[valid_z]=np.round(v_float[valid_z]).astype(np.int32)
        if not np.array_equal(u,arrays['u_proj'])or not np.array_equal(v,arrays['v_proj']):
            raise ValueError('native calibrated rounded projections differ')
        weights=kernel.compute_vertex_area_weights(arrays['verts_cam'],arrays['faces'])
        if not np.array_equal(weights,arrays['area_weights']):raise ValueError('native triangle-area weights differ')
        visible=np.zeros(778,bool);z=arrays['verts_cam'][:,2]
        for i in range(778):
            if valid_z[i]and 0<=u[i]<1920 and 0<=v[i]<1080:
                visible[i]=kernel.check_vertex_visibility(int(u[i]),int(v[i]),float(z[i]),arrays['zbuffer'],depth_eps=.005,window_half=2)
        total=float(weights.sum());ratio=0. if total<1e-12 else float(weights[visible].sum())/total
        if not np.array_equal(visible,arrays['visible']):raise ValueError('native visibility differs')
        statistics=dict(status='native_geometry',visible_ratio=ratio,occlusion_score=1.-ratio,
            author_occlusion_score_round6=round(1.-ratio,6),author_visible_ratio_round6=round(ratio,6),width=1920,height=1080,
            positive_depth_vertices=int(np.count_nonzero(valid_z)),visible_vertices=int(np.count_nonzero(visible)),area_weight_sum=total)
        for key,value in statistics.items():_require_equal(row[key],value,'native point '+key)
        native_rows+=1
    expected_result=dict(role=identity['role'],mode=identity['mode'],rows=len(rows),actual_mesh_calls=len(rows),
        actual_occlusion_calls=native_rows,invalid_zero_pose_rows=invalid,
        complete_standard_role=identity['mode']=='complete_role'and len(rows)==ROLE_COUNTS[identity['role']],
        actual_right_mano_geometry_layers_constructed=identity['runtime']['actual_right_mano_geometry_layers_constructed'],
        learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,
        all_input_and_source_generations_unchanged=True,dense_grid_or_paired_gain_complete=False)
    for key,value in expected_result.items():_require_equal(result[key],value,'native result '+key)
    if type(result['seconds'])is not float or not math.isfinite(result['seconds'])or result['seconds']<0:
        raise ValueError('finite native execution duration required')
    check_inputs(data);guard.check();return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('validate','run'):
        command=sub.add_parser(action)
        for name in ('points','admission','metadata','calibration'):command.add_argument('--'+name,required=True,type=Path)
        command.add_argument('--role',required=True,choices=('val','test'))
        command.add_argument('--mode',required=True,choices=('bilateral_probe','complete_role'))
        command.add_argument('--resolution-policy',required=True,choices=(RESOLUTION_POLICY,))
        if action=='run':
            for name in ('upstream','mano-assets','chumpy-dir','output'):command.add_argument('--'+name,required=True,type=Path)
    verify=sub.add_parser('verify');verify.add_argument('--output',required=True,type=Path)
    return p


def main():
    a=parser().parse_args()
    if a.action=='run':print(json.dumps(run_geometry(a)));return
    if a.action=='verify':print(json.dumps(verify_geometry(a.output)));return
    policy=validate_resolution_policy(a.resolution_policy)
    data=load_point_inputs(a.points,a.admission,a.metadata,a.calibration,a.role);selection=select_points(data,a.mode)
    print(json.dumps(dict(format='native_point_geometry_input_contract_v1',role=a.role,mode=a.mode,
        actual_selected_points=len(selection),input_sha256=data['files_sha256'],resolution_policy=policy,
        native_geometry_executed=False,scientific_eligible=False,full_paper_complete=False)))


if __name__=='__main__':main()
