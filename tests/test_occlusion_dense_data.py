"""Native quarter-stride selection on synthetic real readonly memmaps."""
import copy
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()


def api():return importlib.import_module('egoemg_repro.occlusion_dense_data')


def config(recipe):
    if recipe=='vits_s':
        from egoemg_repro.fusion_literal import compose_fusion
        return compose_fusion(UPSTREAM)
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    return compose_rn18_fusion(UPSTREAM)


def store(root,w=7790,*,empty=False):
    root.mkdir(exist_ok=True);n=8*w+2000;cut=3*w+111
    split=np.zeros(n,np.int8)
    if not empty:
        split[w:2*w+17]=1;split[2*w+17:3*w+700]=2;split[3*w+700:]=3
    split.tofile(root/'frame_split_id.dat')
    manifest=dict(total_rows=n,frame_split_labels=['train','user','gesture','both'],
        fields=dict(frame_split_id=dict(filename='frame_split_id.dat',shape=[n],dtype='int8')))
    (root/'manifest.json').write_text(json.dumps(manifest))
    np.savez(root/'metadata.npz',episode_id=np.array(['a','b']),episode_subject=np.array(['s0','s1']),
        episode_start_idx=np.array([0,cut]),episode_end_idx=np.array([cut,n]),episode_length=np.array([cut,n-cut]),
        splits_split_id=np.arange(4),splits_split=np.array(['train','user','gesture','both']))
    return split,[(0,cut),(cut,n)]


def points(root,recipe='vits_s',**kwargs):
    return api().native_dense_points(root,upstream=UPSTREAM,config=config(recipe),recipe=recipe,**kwargs)


@pytest.mark.parametrize('recipe,w,stride',[('vits_s',7790,1947),('rn18_s',12000,3000)])
def test_exact_recipe_quarter_stride_uses_integer_floor_and_no_shared_center_grid(recipe,w,stride):
    p=api().dense_occlusion_protocol(UPSTREAM,config(recipe),recipe=recipe)
    assert p['window_length']==w and p['dense_stride']==stride and p['eval_center_stride']is None
    assert p['splits']==['user','gesture','both']and p['hands']==['left','right']
    assert p['repeat']==1 and p['max_samples_per_split']==0 and p['jitter']is False
    assert p['selection']=='native episode-start grid, center split only'
    assert p['model_or_geometry_permit']is False and p['full_paper_complete']is False


@pytest.mark.parametrize('recipe,w,stride',[('vits_s',7790,1947),('rn18_s',12000,3000)])
def test_exact_native_keys_order_centers_and_cross_split_windows_are_retained(tmp_path,recipe,w,stride):
    split,episodes=store(tmp_path,w);v=points(tmp_path,recipe);a=v['points'];expected=[]
    for group,sid in enumerate((1,2,3)):
        for hand in(0,1):
            native_index=0
            for episode,(lo,hi)in enumerate(episodes):
                for start in range(lo,hi-w+1,stride):
                    center=start+w//2
                    if split[center]!=sid:continue
                    expected.append((len(expected),episode,start,start+w,center,sid,hand,native_index,group))
                    native_index+=1
    names=['sequence','episode','start','end','center','split_id','hand_index','native_index','group_index']
    assert list(a)==names
    assert [tuple(int(a[n][i])for n in names)for i in range(len(a['sequence']))]==expected
    assert any(np.any(split[start:end]!=sid)for _,_,start,end,_,sid,*_ in expected)
    assert len({(ep,center,hand)for _,ep,_,_,center,_,hand,*_ in expected})==len(expected)
    assert v['identity']['unique_hand_points']==len(expected)
    assert v['identity']['windows_crossing_center_split']>0
    assert not v['identity']['independent_unseen_input_footprints_proven']
    assert all(not arr.flags.writeable and arr.dtype.kind in'iu'for arr in a.values())


def test_native_odd_window_first_last_and_empty_split_edges(tmp_path):
    split,episodes=store(tmp_path,w=5)
    from egoemg_repro.data import LegacyStore
    v=api()._native_view(LegacyStore(tmp_path),UPSTREAM,5,2,'both','right')
    oracle=[(ep,start+2)for ep,(lo,hi)in enumerate(episodes)for start in range(lo,hi-5+1,2)if split[start+2]==3]
    assert [v._resolve_index_to_center(i)for i in range(len(v))]==oracle
    assert all(lo<=c-2 and c-2+5<=hi for ep,c in oracle for lo,hi in[episodes[ep]])
    empty=api()._native_view(LegacyStore(tmp_path),UPSTREAM,5,2,'train','left')
    assert len(empty)>0
    with pytest.raises(IndexError):v._resolve_index_to_center(len(v))


def test_all_train_fixture_retains_six_empty_heldout_groups(tmp_path):
    store(tmp_path,empty=True);v=points(tmp_path)
    assert v['identity']['unique_hand_points']==0 and len(v['identity']['groups'])==6
    assert all(g['points']==0 for g in v['identity']['groups'])
    assert all(arr.shape==(0,)for arr in v['points'].values())


@pytest.mark.parametrize('damage',['unknown_recipe','other_config','stride','window'])
def test_recipe_and_config_drift_reject_before_index_input_io(tmp_path,damage):
    m=api();cfg=config('vits_s');recipe='vits_s'
    if damage=='unknown_recipe':recipe='rn50_m'
    elif damage=='other_config':cfg=config('rn18_s')
    elif damage=='stride':cfg.datamodule.val_test_stride=7800
    else:cfg.datamodule.val_test_window_length=12000
    with pytest.raises(ValueError):m.native_dense_points(tmp_path/'missing',upstream=UPSTREAM,config=cfg,recipe=recipe)


@pytest.mark.parametrize('damage',['float_split','split_shape','unknown_split_value','labels','overlap','metadata_split','field_link','root_link'])
def test_malformed_index_inputs_cannot_create_dense_receipt(tmp_path,damage):
    root=tmp_path/'data';split,_=store(root)
    assert points(root)['identity']['unique_hand_points']>0
    if damage in('float_split','split_shape','labels','field_link'):
        p=root/'manifest.json';v=json.loads(p.read_text())
        if damage=='float_split':split.astype(np.float32).tofile(root/'frame_split_id.dat');v['fields']['frame_split_id']['dtype']='float32'
        elif damage=='split_shape':v['fields']['frame_split_id']['shape']=[len(split),1]
        elif damage=='labels':v['frame_split_labels']=['train','user','user','both']
        else:
            (root/'frame_split_id.dat').rename(root/'original.dat');(root/'frame_split_id.dat').symlink_to(root/'original.dat')
        p.write_text(json.dumps(v))
    elif damage=='unknown_split_value':split[0]=99;split.tofile(root/'frame_split_id.dat')
    elif damage in('overlap','metadata_split'):
        with np.load(root/'metadata.npz')as f:v={k:f[k]for k in f.files}
        if damage=='overlap':v['episode_start_idx'][1]=100;v['episode_length'][1]=v['episode_end_idx'][1]-100
        else:v['splits_split']=np.array(['train','gesture','user','both'])
        np.savez(root/'metadata.npz',**v)
    else:
        alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True);root=alias
    with pytest.raises(ValueError):points(root)


def test_symlink_upstream_rejects_before_import_or_input_read(tmp_path):
    alias=tmp_path/'upstream';alias.symlink_to(UPSTREAM,target_is_directory=True)
    with pytest.raises(ValueError):api().native_dense_points(tmp_path/'missing',upstream=alias,config=config('vits_s'),recipe='vits_s')


def test_fresh_identity_rebuild_and_resealed_expected_forgery(tmp_path):
    from egoemg_repro.classic_emg import _hash
    store(tmp_path);v=points(tmp_path);second=points(tmp_path,expected=v['identity'])
    assert second['identity']==v['identity']
    bad=copy.deepcopy(v['identity']);bad['unique_hand_points']+=1
    bad['sha256']=_hash({k:x for k,x in bad.items()if k!='sha256'})
    with pytest.raises(ValueError,match='identity|receipt'):points(tmp_path,expected=bad)


def test_no_model_constructors_or_random_state_changes(tmp_path,monkeypatch):
    import torch
    store(tmp_path);cfg=config('vits_s');m=api()
    py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state().clone()
    def forbidden(*a,**kw):raise AssertionError('dense selection cannot construct a learned model')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    v=m.native_dense_points(tmp_path,upstream=UPSTREAM,config=cfg,recipe='vits_s')
    assert random.getstate()==py and npstate[0]==np.random.get_state()[0]
    assert np.array_equal(npstate[1],np.random.get_state()[1])and npstate[2:]==np.random.get_state()[2:]
    assert torch.equal(torch.get_rng_state(),ts)and not torch.cuda.is_initialized()
    assert v['identity']['learned_pose_models_constructed']==0 and v['identity']['optimizer_updates']==0


def test_input_bytes_changed_during_native_index_construction_reject(tmp_path,monkeypatch):
    store(tmp_path);m=api();original=m._native_view;changed=False
    assert points(tmp_path)['identity']['unique_hand_points']>0
    def mutate(*a,**kw):
        nonlocal changed
        if not changed:
            with(tmp_path/'frame_split_id.dat').open('r+b')as f:f.write(b'\x01')
            changed=True
        return original(*a,**kw)
    monkeypatch.setattr(m,'_native_view',mutate)
    with pytest.raises(ValueError):points(tmp_path)
