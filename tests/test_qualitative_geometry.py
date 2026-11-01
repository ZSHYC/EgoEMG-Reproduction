"""New first20/hand/normalization dispatch, compared with unchanged F7 AST."""
import ast
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from test_occlusion_dense_data import UPSTREAM


def api():
    from egoemg_repro import qualitative_geometry
    return qualitative_geometry


def poses():
    x=np.linspace(-.3,.6,22,dtype=np.float32)
    return dict(gt=x,emg=x-np.float32(.5),vision=x+np.float32(.2),fusion=x+np.float32(.1))


def original_namespace():
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    from egoemg.UmeTrack.lib.common.hand import HandModel
    from egoemg.UmeTrack.lib.common.hand_skinning import _skin_points
    env=dict(np=np,torch=torch,json=json,HandModel=HandModel,_skin_points=_skin_points,
        _UMETRACK_HAND_MODEL=None,UMETRACK_HAND_MODEL_PATH=str(UPSTREAM/'egoemg/UmeTrack/dataset/generic_hand_model.json'))
    loader=ast.parse((UPSTREAM/'egoemg/UmeTrack/lib/tracker/video_pose_data.py').read_text())
    tree=ast.parse((UPSTREAM/'scripts/eval/find_fusion_wins.py').read_text())
    nodes=[n for n in loader.body if isinstance(n,ast.FunctionDef)and n.name=='load_hand_model_from_dict']
    nodes.extend(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('_get_umetrack_hand_model','_mirror_hand_model','skin_fk_mesh'))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-F7-local-mesh-helpers>','exec'),env)
    visual=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='visualize_sample')
    wanted=[]
    for n in visual.body:
        if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('gt_ja','gt_fk_local','fk_span'):wanted.append(n)
        elif isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Tuple)and ast.unparse(n.targets[0])=='(gt_fk_local, gt_fk_faces)':wanted.append(n)
        elif isinstance(n,ast.If)and ast.unparse(n.test)=='fk_span > 1e-06':wanted.append(n)
    assert len(wanted)==5
    return env,compile(ast.Module(body=wanted,type_ignores=[]),'<original-F7-first20-and-span-normalization>','exec')


# Catches passing22angles, dropping handedness/winding, sharing GT's scale for
# all predictions, or using a replacement skinner rather than the original.
@pytest.mark.parametrize('hand',['left','right'])
def test_all_four_actual_meshes_equal_original_first20_and_per_pose_normalization(hand):
    p=poses();env,code=original_namespace();v=api().native_qualitative_meshes(p,hand=hand,upstream=UPSTREAM)
    assert set(v['meshes'])==set(p)
    spans=[]
    for role,a in p.items():
        env.update(result=dict(target=a),flip=hand=='left');exec(code,env)
        packet=v['meshes'][role]
        np.testing.assert_array_equal(packet['vertices'],env['gt_fk_local'])
        np.testing.assert_array_equal(packet['faces'],env['gt_fk_faces'])
        assert packet['unscaled_median_span']==float(env['fk_span'])
        spans.append(packet['unscaled_median_span'])
        assert np.isclose(np.median(np.ptp(packet['vertices'],axis=0)),.09,atol=2e-8,rtol=0)
    assert len(set(spans))>1
    assert v['finger_angles_used']==20 and v['numeric_error_angles']==22
    assert not v['physical_rgb_projection_proven']and not v['figure7_reproduced']


def test_wrist_display_is_source_ignored_without_mutating_inputs_or_rng():
    p=poses();saved=copy.deepcopy(p);before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    a=api().native_qualitative_meshes(p,hand='left',upstream=UPSTREAM)
    for x in p.values():x[20:]=[7,-8]
    b=api().native_qualitative_meshes(p,hand='left',upstream=UPSTREAM)
    for role in p:
        np.testing.assert_array_equal(a['meshes'][role]['vertices'],b['meshes'][role]['vertices'])
        np.testing.assert_array_equal(p[role][:20],saved[role][:20])
    assert before[0]==random.getstate();now=np.random.get_state()
    assert before[1][0]==now[0]and np.array_equal(before[1][1],now[1])and before[1][2:]==now[2:]
    assert torch.equal(before[2],torch.get_rng_state())and not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['missing','extra','float64','short20','nonfinite','hand'])
def test_incomplete_or_repaired_pose_cannot_generate_panels(damage):
    p=poses();hand='right'
    if damage=='missing':p.pop('emg')
    elif damage=='extra':p['replacement']=p['gt']
    elif damage=='float64':p['gt']=p['gt'].astype(np.float64)
    elif damage=='short20':p['gt']=p['gt'][:20]
    elif damage=='nonfinite':p['gt'][0]=np.nan
    else:hand='unknown'
    with pytest.raises(ValueError):api().native_qualitative_meshes(p,hand=hand,upstream=UPSTREAM)


def test_foreign_or_runtime_patched_native_skinning_is_rejected(monkeypatch):
    original_namespace()
    from egoemg.UmeTrack.lib.common import hand_skinning
    monkeypatch.setattr(hand_skinning,'_skin_points',lambda *a,**k:torch.zeros((1,3)))
    with pytest.raises(ValueError):api().native_qualitative_meshes(poses(),hand='right',upstream=UPSTREAM)
