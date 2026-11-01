"""Pinned original Figure7 local meshes and per-pose display normalization.

Compile the actual helper bodies; adapt only the migrated generic-profile path
and avoid unrelated video/MANO imports. This does not prove RGB projection.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import types

import numpy as np

from .classic_emg import _activate, _approved_upstream
from .classic_emg_cohort import _safe
from .data import file_sha256
from .native_ik_inputs import _Guard
from .occlusion_dense_data import _typed_sha

IMPORTED_SHA=file_sha256(__file__)
SCRIPT='scripts/eval/find_fusion_wins.py'
ASSET='egoemg/UmeTrack/dataset/generic_hand_model.json'
ASSET_SHA='2eb8029a8e1cedc2cf5b40c14c0955e2004277024df3c6304ce0bb74106a7c31'
LOADER='egoemg/UmeTrack/lib/tracker/video_pose_data.py'
COMMON='egoemg/UmeTrack/lib/common/'


def _providers(root,skin,so3,hand):
    for module in (skin,so3,hand):
        path=_safe(module.__file__)
        if not path.is_relative_to(root):raise ValueError('pinned original UmeTrack provider required')
        if module is hand:continue
        source=path.read_bytes();tree=ast.parse(source)
        compiled=compile(source,str(path),'exec',dont_inherit=True)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef):
                original=next(c for c in compiled.co_consts if isinstance(c,types.CodeType)and c.co_name==node.name)
                current=getattr(module,node.name,None)
                if not inspect.isfunction(current)or current.__code__!=original:
                    raise ValueError('unmodified original UmeTrack skinning function required')
    if skin.so3_exp_map is not so3.so3_exp_map or skin.HandModel is not hand.HandModel:
        raise ValueError('original UmeTrack function/type provider binding required')
    if any(getattr(skin,n)!=getattr(hand,n)for n in ('DOF_PER_FINGER','NUM_DIGITS','NUM_JOINT_FRAMES')):
        raise ValueError('original UmeTrack skinning constants required')


def native_qualitative_meshes(poses,*,hand,upstream):
    import torch
    if hand not in ('left','right')or not isinstance(poses,dict)or set(poses)!={'gt','emg','vision','fusion'}:
        raise ValueError('explicit hand and all four actual22-angle poses required')
    if any(not isinstance(a,np.ndarray)or a.dtype!=np.float32 or a.shape!=(22,)
        or not np.isfinite(a).all()for a in poses.values()):
        raise ValueError('complete finite FP32 original22-angle poses required; no casting/repair')
    if torch.get_default_dtype()!=torch.float32 or torch.cuda.is_initialized():
        raise ValueError('original FP32 CPU geometry defaults required')
    root=_approved_upstream(upstream)
    names=(SCRIPT,ASSET,LOADER,COMMON+'hand.py',COMMON+'hand_skinning.py',COMMON+'pytorch3d_transforms_so3.py')
    paths={n:_safe(root/n)for n in names};guard=_Guard([_safe(__file__),*paths.values()],[])
    hashes={n:file_sha256(p)for n,p in paths.items()}
    if file_sha256(__file__)!=IMPORTED_SHA or hashes[ASSET]!=ASSET_SHA:
        raise ValueError('pinned original qualitative geometry/profile source required')
    _activate(root)
    from egoemg.UmeTrack.lib.common import hand_skinning as skin, pytorch3d_transforms_so3 as so3
    from egoemg.UmeTrack.lib.common import hand as hand_module
    _providers(root,skin,so3,hand_module)
    env=dict(np=np,torch=torch,json=json,HandModel=hand_module.HandModel,
        _skin_points=skin._skin_points,_UMETRACK_HAND_MODEL=None,UMETRACK_HAND_MODEL_PATH=str(paths[ASSET]))
    loader=ast.parse(paths[LOADER].read_bytes());tree=ast.parse(paths[SCRIPT].read_bytes())
    nodes=[n for n in loader.body if isinstance(n,ast.FunctionDef)and n.name=='load_hand_model_from_dict']
    nodes.extend(n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.name in ('_get_umetrack_hand_model','_mirror_hand_model','skin_fk_mesh'))
    if len(nodes)!=4:raise ValueError('complete original UmeTrack helper definitions required')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(paths[SCRIPT]),'exec',dont_inherit=True),env)
    result={}
    with torch.no_grad(),torch.device('cpu'):
        for role,a in poses.items():
            raw,faces=env['skin_fk_mesh'](joint_angles=a[:20],flip=hand=='left')
            vertices=raw.copy()
            span=np.median(vertices.max(axis=0)-vertices.min(axis=0))
            if span>1e-6:vertices=vertices*(0.09/span)
            if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
                or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(faces).all()
                or not np.equal(faces,np.floor(faces)).all()or faces.min()<0 or faces.max()>=len(vertices)):
                raise ValueError('actual complete finite original mesh vertices/triangles required')
            result[role]=dict(vertices=vertices,faces=faces,unscaled_vertices=raw,
                unscaled_median_span=float(span),source_pose22_sha256=_typed_sha(a),
                source_finger20_sha256=_typed_sha(a[:20]),
                vertices_sha256=_typed_sha(vertices),faces_sha256=_typed_sha(faces))
    _providers(root,skin,so3,hand_module);guard.check()
    return dict(format='original_first20_qualitative_local_umetrack_meshes_v1',hand=hand,meshes=result,
        source_files_sha256=hashes,adapter_source_sha256=IMPORTED_SHA,
        generic_profile_path_substitution='original stale emg2pose/UmeTrack path -> same-revision egoemg/UmeTrack asset',
        finger_angles_used=20,numeric_error_angles=22,wrist_angles_displayed=False,
        source_display_normalization='each pose separately: median vertex span -> 0.09 when span >1e-6',
        mirrored_profile_and_reversed_winding=hand=='left',
        physical_rgb_projection_proven=False,trained_model_or_full_input_admitted=False,
        figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False)
