"""Ordinary Figure7 grid and exact original source subsampling on real files."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import importlib.util
import pytest

# Pure metadata checks remain available without the optional model runtime.
requires_official_runtime = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason='uses the complete optional official Torch reader/model/evaluation runtime')

from test_occlusion_dense_data import config, store, UPSTREAM
from egoemg_repro import qualitative_data as qualitative


def admit(root, recipe='vits_s', **kwargs):
    return qualitative.admit_qualitative_selection(root, upstream=UPSTREAM,
        config=config(recipe), recipe=recipe, engineering=True, **kwargs)


def original_sample(n, cap, seed):
    """Compile the actual source's per-dataset RNG/reset/subsample statements."""
    tree = ast.parse((UPSTREAM/'scripts/eval/find_fusion_wins.py').read_text())
    main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
    wanted = []
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('n_total', 'rng') for t in node.targets):
            wanted.append(node)
        if isinstance(node, ast.If) and 'max_samples_per_split' in ast.unparse(node.test):
            wanted.append(node)
    assert len(wanted) == 3
    scope = dict(np=np, dataset=range(n), args=SimpleNamespace(max_samples_per_split=cap, seed=seed))
    exec(compile(ast.Module(body=sorted(wanted, key=lambda x:x.lineno), type_ignores=[]),
                 '<actual-original-Figure7-subsampling>', 'exec'), scope)
    return scope['indices']


@pytest.mark.parametrize('recipe,w', [('vits_s',7790), ('rn18_s',12000)])
def test_protocol_preserves_original_standard_stride_and_defaults(recipe, w):
    p = qualitative.qualitative_protocol(UPSTREAM, config(recipe), recipe=recipe)
    assert p['window_length'] == p['stride'] == w
    assert p['max_samples_per_split'] == 500 and p['seed'] == 42
    assert p['rng_reset_each_split_and_hand'] is True
    assert p['selection'] == 'native episode-start grid, center split only'
    assert p['model_or_geometry_permit'] is False


@requires_official_runtime
@pytest.mark.parametrize('recipe,w', [('vits_s',7790), ('rn18_s',12000)])
@pytest.mark.parametrize('cap', [0, 1, 500])
def test_full_native_membership_and_sampling_match_independent_rows_and_actual_source(tmp_path, recipe, w, cap):
    split, episodes = store(tmp_path, w)
    selected = admit(tmp_path, recipe, max_samples_per_split=cap)
    a = selected.points; expected = []
    for group, sid in enumerate((1, 2, 3)):
        for hand in (0, 1):
            native = [(ep, start, start+w, start+w//2) for ep,(lo,hi) in enumerate(episodes)
                      for start in range(lo,hi-w+1,w) if split[start+w//2] == sid]
            for index in original_sample(len(native), cap, 42):
                ep, start, end, center = native[index]
                expected.append((len(expected),ep,start,end,center,sid,hand,index,group))
    names = list(qualitative.COLUMNS)
    assert [tuple(int(a[n][i]) for n in names) for i in range(len(a['sequence']))] == expected
    assert selected.identity['selected_hand_points'] == len(expected)
    assert selected.identity['full_registered_index_source'] is False
    assert not selected.identity['model_or_geometry_permit']
    assert selected.identity['figure7_reproduced'] is False


@requires_official_runtime
def test_default_rng_is_reset_for_each_group_and_original_cap500_is_real(tmp_path):
    w = 7790; n = w*1803
    split = np.repeat(np.array([1,2,3], np.int8), n//3)
    split.tofile(tmp_path/'frame_split_id.dat')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(total_rows=n,
        frame_split_labels=['train','user','gesture','both'],
        fields=dict(frame_split_id=dict(filename='frame_split_id.dat',shape=[n],dtype='int8')))))
    np.savez(tmp_path/'metadata.npz',episode_id=np.array(['one']),episode_subject=np.array(['s0']),
        episode_start_idx=np.array([0]),episode_end_idx=np.array([n]),episode_length=np.array([n]),
        splits_split_id=np.arange(4),splits_split=np.array(['train','user','gesture','both']))
    selected = admit(tmp_path)
    assert selected.identity['selected_hand_points'] == 3000
    a = selected.points
    for group in range(3):
        left = a['native_index'][(a['group_index']==group)&(a['hand_index']==0)]
        right = a['native_index'][(a['group_index']==group)&(a['hand_index']==1)]
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left, original_sample(601,500,42))


@requires_official_runtime
def test_bytes_rng_and_readonly_copies_are_preserved(tmp_path):
    import torch
    store(tmp_path)
    before = {p.name:(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in tmp_path.iterdir()}
    py, npr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    selected = admit(tmp_path, max_samples_per_split=1)
    original = selected.identity
    identity = selected.identity; identity['selected_hand_points'] = 999
    arrays = selected.points
    assert all(not a.flags.writeable for a in arrays.values())
    arrays['center'].flags.writeable=True; arrays['center'][:] = 999
    assert selected.identity == original and not (selected.points['center']==999).any()
    assert before == {p.name:(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in tmp_path.iterdir()}
    assert random.getstate()==py and torch.equal(torch.get_rng_state(),tr)
    current=np.random.get_state();assert npr[0]==current[0] and np.array_equal(npr[1],current[1]) and npr[2:]==current[2:]
    assert not torch.cuda.is_initialized()


@requires_official_runtime
def test_crossing_windows_and_empty_groups_are_not_redefined(tmp_path):
    split, _ = store(tmp_path)
    selected=admit(tmp_path, max_samples_per_split=0)
    a=selected.points
    count=sum(np.any(split[a['start'][i]:a['end'][i]]!=a['split_id'][i]) for i in range(len(a['sequence'])))
    assert count>0 and selected.identity['windows_crossing_center_split']==count
    empty=tmp_path/'empty';store(empty,empty=True)
    assert admit(empty).identity['selected_hand_points']==0


@pytest.mark.parametrize('kwargs', [dict(max_samples_per_split=True),dict(max_samples_per_split=-1),
    dict(seed=True),dict(seed=-1),dict(seed=1.5)])
def test_invalid_sampling_arguments_rejected_before_dataset(tmp_path, kwargs):
    with pytest.raises(ValueError,match='seed|sample|integer'):
        admit(tmp_path,**kwargs)


def test_unregistered_or_engineering_source_cannot_claim_actual_release(tmp_path):
    with pytest.raises(ValueError,match='registered'):
        qualitative.QualitativeSelection().check()
    store(tmp_path)
    with pytest.raises(ValueError,match='registered|released|complete|reference|full'):
        qualitative.admit_qualitative_selection(tmp_path,upstream=UPSTREAM,config=config('vits_s'),recipe='vits_s')


@requires_official_runtime
@pytest.mark.parametrize('mutation',['append','hardlink','manifest'])
def test_source_generation_changes_invalidate_registered_selection(tmp_path,mutation):
    store(tmp_path);selected=admit(tmp_path)
    if mutation=='append':
        with(tmp_path/'frame_split_id.dat').open('ab')as f:f.write(b'\0')
    elif mutation=='hardlink':os.link(tmp_path/'frame_split_id.dat',tmp_path/'another-link')
    else:(tmp_path/'manifest.json').write_text((tmp_path/'manifest.json').read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        selected.check()


def test_symlink_input_and_malformed_split_shape_rejected(tmp_path):
    root=tmp_path/'real';store(root)
    alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):admit(alias)
    manifest=json.loads((root/'manifest.json').read_text());manifest['fields']['frame_split_id']['shape']=[1]
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='split|schema|shape'):admit(root)


def test_original_source_replaced_between_config_validation_and_first_guard_is_rejected(tmp_path,monkeypatch):
    # A disposable real f7 checkout, never the shared upstream working tree.
    clone=tmp_path/'upstream'
    subprocess.run(['git','clone','--shared','--no-hardlinks','-q',str(UPSTREAM),str(clone)],check=True,capture_output=True)
    from egoemg_repro.fusion_literal import compose_fusion
    cfg=compose_fusion(clone);original=qualitative._paths
    def paths(root):
        value=original(root)
        script=clone/'scripts/eval/find_fusion_wins.py'
        script.write_text(script.read_text()+'\n# source changed after initial clean-tree validation\n')
        return value
    monkeypatch.setattr(qualitative,'_paths',paths)
    with pytest.raises(ValueError,match='clean|source|upstream'):
        qualitative.qualitative_protocol(clone,cfg,recipe='vits_s')


def test_data_change_during_first_hash_is_rejected_before_native_provider(tmp_path,monkeypatch):
    store(tmp_path);original=qualitative.file_sha256;changed=False
    target=tmp_path/'frame_split_id.dat'
    def digest(path):
        nonlocal changed
        sha=original(path)
        if Path(path)==target and not changed:
            changed=True
            with target.open('r+b')as stream:stream.write(bytes([3]))
        return sha
    monkeypatch.setattr(qualitative,'file_sha256',digest)
    with pytest.raises(ValueError,match='changed'):admit(tmp_path)
    assert changed
