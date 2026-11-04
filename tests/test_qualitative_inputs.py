"""Ordinary selected windows must reach the actual native paired readers."""
import copy
import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="native tensor checks require the optional Torch tier")

from test_occlusion_dense_data import config, UPSTREAM
from test_occlusion_dense_inputs import corpus
from test_vision_literal_data import crop_store, release_fixture_readers
from egoemg_repro.qualitative_data import admit_qualitative_selection


def api():
    from egoemg_repro import qualitative_inputs
    return qualitative_inputs


def test_unregistered_selection_is_rejected_before_any_input_io(tmp_path):
    with pytest.raises(ValueError,match='registered'):
        api().admit_qualitative_inputs(object(),data=tmp_path/'missing',crops=tmp_path/'missing',
            upstream=UPSTREAM,config=None,recipe='vits_s')


def build(tmp_path, recipe='vits_s', cap=1, *, bad_right=False):
    torch.set_num_threads(2)
    data=tmp_path/'data'; arrays=corpus(data); crops=tmp_path/'crops'; crop_store(crops,flat=True)
    if bad_right:
        arrays['generated_label_valid'][:,1]=False
        arrays['generated_label_valid'].tofile(data/'generated_label_valid.dat')
    cfg=config(recipe)
    selected=admit_qualitative_selection(data,upstream=UPSTREAM,config=cfg,recipe=recipe,
        max_samples_per_split=cap,engineering=True)
    ds=api().admit_qualitative_inputs(selected,data=data,crops=crops,
        upstream=UPSTREAM,config=cfg,recipe=recipe)
    return ds,arrays,crops,selected


# Catches replacing the selected ordinary points by dense points, losing the
# per-group sampled native_index, or feeding the fusion layout to standaloneEMG.
@pytest.mark.parametrize('recipe,w,channels',[('vits_s',7790,16),('rn18_s',12000,8)])
@pytest.mark.parametrize('cap',[0,1])
def test_exact_selected_windows_match_original_full_reader_and_independent_channel_values(tmp_path,recipe,w,channels,cap):
    ds,arrays,crops,selected=build(tmp_path,recipe,cap)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from hydra.utils import instantiate
    from egoemg.transforms import Compose
    from egoemg_repro.emg_initializer_literal import compose_emg_initializer
    try:
        points=selected.points; assert len(ds)==selected.identity['selected_hand_points']
        assert ds.identity['selection_identity_sha256']==hashlib.sha256(json.dumps(selected.identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        for group,split in enumerate(('user','gesture','both')):
            for hi,hand in enumerate(('left','right')):
                norm=ds.identity['standalone_normalization']
                ref=EgoEmgMemmapDataset(tmp_path/'data',w,stride=w,allowed_splits=[split],
                    modalities=['emg','joint_angles','labels'],target_hand=hand,
                    emg_field_preference='filtered_paper',emg_layout='target_hand',norm_mode='per-dataset',
                    norm_stats_path=str(UPSTREAM/norm['asset']),dataset_name='egoemg',
                    transform=Compose([instantiate(t)for t in compose_emg_initializer(UPSTREAM).transforms.val]),
                    jitter=False,center_target_only=True)
                choices=np.flatnonzero((points['group_index']==group)&(points['hand_index']==hi))
                for i in choices:
                    i=int(i); packet=ds[i]; k=packet['key'];f=packet['fusion'];e=packet['standalone_emg']
                    native=ref[int(points['native_index'][i])]
                    assert k==dict(sequence=i,split=split,hand=hand,ep_idx=int(points['episode'][i]),
                        center_idx=int(points['center'][i]),window_start=int(points['start'][i]),
                        window_end=int(points['end'][i]),native_index=int(points['native_index'][i]),group_index=group)
                    ep,center=ref._resolve_index_to_center(int(points['native_index'][i]))
                    assert (ep,center)==(k['ep_idx'],k['center_idx'])
                    assert (center-w//2,center-w//2+w)==(k['window_start'],k['window_end'])
                    for n in ('emg','joint_angles','label_valid_mask'):
                        np.testing.assert_array_equal(e[n],native[n])
                    assert f['emg'].shape==(channels,w) and e['emg'].shape==(8,w)
                    raw=arrays[f'emg_{hand}_filtered'][k['window_start']:k['window_end']]
                    n=norm['hands'][hand]
                    np.testing.assert_array_equal(e['emg'],((raw-np.asarray(n['mean'],np.float32))/(np.asarray(n['std'],np.float32)+1e-6)).T)
                    if recipe=='vits_s':
                        mapped=np.zeros((w,16),np.float32);mapped[:,[10,12,0,1,2,4,5,6]]=raw
                        n=ds.identity['fusion_normalization']
                        np.testing.assert_array_equal(f['emg'],((mapped-n['mean'])/(n['std']+1e-6)).T)
                    np.testing.assert_array_equal(f['joint_angles'],e['joint_angles'])
                    np.testing.assert_array_equal(f['label_valid_mask'],e['label_valid_mask'])
        assert ds.identity['windows_crossing_center_split']==selected.identity['windows_crossing_center_split']
    finally:release_fixture_readers(crops)


# Catches an added crop/label filter, a second left flip, RNG consumption, or
# returning mutable internal points/identity/payload maps to later consumers.
def test_missing_crop_false_labels_and_canonical_left_rgb_are_retained_with_copy_isolation(tmp_path):
    random.seed(9);np.random.seed(8);torch.manual_seed(7)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    ds,_,crops,selection=build(tmp_path,bad_right=True)
    try:
        assert len(ds)==6
        left=ds[0]['fusion'];right=ds[1]['fusion']
        assert left['vision_img'][0,30,30]>2 and left['vision_img'][2,30,200]>2
        assert not right['vision_valid_mask'].any() and not right['label_valid_mask'].any()
        assert right['joint_angles'].shape==(22,1)
        np.testing.assert_allclose(right['vision_img'][:,0,0],[-.485/.229,-.456/.224,-.406/.225],atol=1e-6)
        original=ds[0];ds.identity.clear();ds.paths.clear();selection.points['center'].setflags(write=True)
        original['fusion']['joint_angles'][:]=100
        assert not np.any(ds[0]['fusion']['joint_angles']==100)
        payloads=ds.hash_payloads()
        assert payloads=={n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in ds.paths.items()}
        assert not ds.identity['trained_model_or_inference_permit'] and not ds.identity['figure7_reproduced']
        assert random.getstate()==before[0];after=np.random.get_state()
        assert before[1][0]==after[0]and np.array_equal(before[1][1],after[1])and before[1][2:]==after[2:]
        assert torch.equal(before[2],torch.get_rng_state())and not torch.cuda.is_initialized()
        with pytest.raises(dataclasses.FrozenInstanceError):ds.selection=selection
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['unregistered','copied_root','config','recipe'])
def test_wrong_selection_or_source_does_not_open_a_substitute_corpus(tmp_path,damage):
    import shutil
    ds,_,crops,selection=build(tmp_path)
    try:
        data=tmp_path/'data';cfg=config('vits_s');recipe='vits_s'
        if damage=='unregistered':selection=copy.copy(selection)
        elif damage=='copied_root':shutil.copytree(data,tmp_path/'other');data=tmp_path/'other'
        elif damage=='config':cfg.datamodule.val_test_stride=1947
        else:recipe='rn18_s';cfg=config(recipe)
        with pytest.raises(ValueError):api().admit_qualitative_inputs(selection,data=data,crops=crops,upstream=UPSTREAM,config=cfg,recipe=recipe)
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['emg_generation','crop_generation','new_hardlink','layout','norm','reader_patch'])
def test_live_input_admission_detects_payload_or_reader_semantic_changes(tmp_path,damage,monkeypatch):
    ds,_,crops,_=build(tmp_path)
    try:
        ds[0]
        if damage.endswith('_generation'):
            p=next(p for n,p in ds.paths.items()if n.startswith('crop:')) if damage=='crop_generation' else tmp_path/'data/emg_left_filtered.dat'
            replacement=p.with_suffix('.replacement');replacement.write_bytes(p.read_bytes());replacement.replace(p)
        elif damage=='new_hardlink':os.link(tmp_path/'data/emg_left_filtered.dat',tmp_path/'extra')
        elif damage=='reader_patch':
            from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
            monkeypatch.setattr(EgoEmgMemmapDataset,'_getitem_center_supervised',lambda *a,**k:{})
        else:
            state=api()._ADMITTED[ds];view=state['io'].fusion_views[0]
            if damage=='layout':view.emg_layout='target_hand'
            else:view._emg_mean=123.0
        with pytest.raises(ValueError):ds[0]
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('field,channels',[('emg_left_filtered',8),('generated_joint_angles_left',20)])
def test_stable_nonfinite_input_is_rejected_without_native_repair(tmp_path,field,channels):
    ds,_,crops,selection=build(tmp_path)
    try:
        k=ds[0]['key'];p=tmp_path/'data'/(field+'.dat')
        n=json.loads((tmp_path/'data/manifest.json').read_text())['total_rows']
        a=np.memmap(p,mode='r+',dtype=np.float32,shape=(n,channels));a[k['center_idx'],0]=np.nan;a.flush();del a
        current=api().admit_qualitative_inputs(selection,data=tmp_path/'data',crops=crops,upstream=UPSTREAM,config=config('vits_s'),recipe='vits_s')
        with pytest.raises(ValueError,match='finite'):current[0]
    finally:release_fixture_readers(crops)
