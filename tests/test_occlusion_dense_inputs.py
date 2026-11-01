"""Real readonly small memmaps/LMDB and unchanged native reader references."""
import copy
import importlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest

from test_vision_literal_data import sample_store,crop_store,release_fixture_readers
from test_occlusion_dense_data import config,UPSTREAM


def api():
    try:return importlib.import_module('egoemg_repro.occlusion_dense_inputs')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.occlusion_dense_inputs':pytest.fail('dense producer-correct input reader missing')
        raise


def corpus(root):
    n=8*12000;a=sample_store(root,n=n);splits=np.repeat([0,1,2,3],[24000]*4).astype(np.int8)
    splits.tofile(root/'frame_split_id.dat');a['frame_split_id']=splits
    manifest=json.loads((root/'manifest.json').read_text())
    for hand in('left','right'):
        manifest['fields'][f'emg_{hand}_filtered_paper']=copy.deepcopy(manifest['fields'][f'emg_{hand}_filtered'])
    (root/'manifest.json').write_text(json.dumps(manifest));return a


def build(tmp_path,recipe='vits_s',*,false_right_labels=False):
    torch=pytest.importorskip('torch');pytest.importorskip('hydra');torch.set_num_threads(2)
    data=tmp_path/'data';arrays=corpus(data);crops=tmp_path/'crops';crop_store(crops,flat=True)
    if false_right_labels:
        arrays['generated_label_valid'][:,1]=False
        arrays['generated_label_valid'].tofile(data/'generated_label_valid.dat')
    ds=api().DensePredictionDataset(data,crops,upstream=UPSTREAM,config=config(recipe),recipe=recipe)
    return ds,arrays,crops


@pytest.mark.parametrize('recipe,channels,length',[('vits_s',16,7790),('rn18_s',8,12000)])
def test_every_native_key_has_same_window_two_independently_normalized_source_views(tmp_path,recipe,channels,length):
    ds,arrays,crops=build(tmp_path,recipe)
    try:
        p=ds.index['points'];assert len(ds)==ds.index['identity']['unique_hand_points']
        assert not ds.identity['full_input_or_model_admitted']and not ds.identity['figure13_reproduced']
        chosen=[next(i for i in range(len(ds))if p['group_index'][i]==g and p['hand_index'][i]==h)for g in range(3)for h in range(2)]
        for i in chosen:
            out=ds[i];key=out['key'];fusion,emg=out['fusion'],out['standalone_emg'];hand=key['hand']
            assert key==dict(sequence=i,split=('user','gesture','both')[int(p['group_index'][i])],hand=hand,
                ep_idx=int(p['episode'][i]),center_idx=int(p['center'][i]),window_start=int(p['start'][i]),
                window_end=int(p['end'][i]),native_index=int(p['native_index'][i]),group_index=int(p['group_index'][i]))
            assert fusion['emg'].shape==(channels,length)and emg['emg'].shape==(8,length)
            assert fusion['emg'].dtype==emg['emg'].dtype==np.float32
            np.testing.assert_array_equal(fusion['joint_angles'],emg['joint_angles'])
            np.testing.assert_array_equal(fusion['label_valid_mask'],emg['label_valid_mask'])
            norms=ds.identity['standalone_normalization']['hands'][hand]
            raw=arrays[f'emg_{hand}_filtered'][key['window_start']:key['window_end']]
            expected=((raw-np.asarray(norms['mean'],np.float32))/(np.asarray(norms['std'],np.float32)+1e-6)).T
            np.testing.assert_array_equal(emg['emg'],expected)
            if recipe=='vits_s':
                scalar=ds.identity['fusion_normalization'];mapped=np.zeros((length,16),np.float32)
                mapped[:,[10,12,0,1,2,4,5,6]]=raw
                expected_fusion=((mapped-scalar['mean'])/(scalar['std']+1e-6)).T
                np.testing.assert_array_equal(fusion['emg'],expected_fusion)
                assert not np.array_equal(emg['emg'],fusion['emg'][[10,12,0,1,2,4,5,6]])
        assert ds.identity['windows_crossing_center_split']==ds.index['identity']['windows_crossing_center_split']>0
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('recipe',['vits_s','rn18_s'])
def test_standalone_emg_matches_genuine_original_full_reader_on_exact_selected_windows(tmp_path,recipe):
    ds,_,crops=build(tmp_path,recipe)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from hydra.utils import instantiate
    from egoemg.transforms import Compose
    try:
        for hand in('left','right'):
            cfg=ds.emg_config
            ref=EgoEmgMemmapDataset(ds.store.root,ds.identity['window_length'],stride=ds.identity['dense_stride'],
                allowed_splits=['user'],modalities=['emg','joint_angles','labels'],target_hand=hand,
                emg_field_preference='filtered_paper',emg_layout='target_hand',norm_mode='per-dataset',
                norm_stats_path=str(UPSTREAM/ds.identity['standalone_normalization']['asset']),dataset_name='egoemg',
                transform=Compose([instantiate(t)for t in cfg.transforms.val]),jitter=False)
            choices=[i for i in range(len(ds))if ds.index['points']['group_index'][i]==0 and ds.index['points']['hand_index'][i]==(hand=='right')]
            assert len(choices)==len(ref)
            for i in choices[:2]:
                packet=ds[i];idx=packet['key']['native_index'];native=ref[idx]
                assert int(native['window_start_idx'])==packet['key']['window_start']
                assert int(native['window_end_idx'])==packet['key']['window_end']
                np.testing.assert_array_equal(packet['standalone_emg']['emg'],native['emg'])
    finally:release_fixture_readers(crops)


def test_missing_crop_and_false_label_are_retained_without_rng_or_model_construction(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');m=api()
    def forbidden(*a,**kw):raise AssertionError('input IO constructed a learned module')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    random.seed(9);np.random.seed(8);torch.manual_seed(7)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    ds,_,crops=build(tmp_path,false_right_labels=True)
    try:
        p=ds.index['points'];i=next(i for i in range(len(ds))if p['hand_index'][i]==1)
        out=ds[i];assert not out['fusion']['vision_valid_mask'].any()
        assert not out['fusion']['label_valid_mask'].any()and not out['standalone_emg']['label_valid_mask'].any()
        assert out['fusion']['emg'].shape[1]==7790 and len(ds)>0
        np.testing.assert_allclose(out['fusion']['vision_img'][:,0,0],[-.485/.229,-.456/.224,-.406/.225],atol=1e-6)
        assert before[0]==random.getstate();after=np.random.get_state()
        assert before[1][0]==after[0]and np.array_equal(before[1][1],after[1])and before[1][2:]==after[2:]
        assert torch.equal(before[2],torch.get_rng_state())and not torch.cuda.is_initialized()
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['nan_emg','nan_target','payload_generation','index_bytes','layout','standalone_norm'])
def test_mutated_data_keys_or_reader_semantics_fail_instead_of_repairing(tmp_path,damage):
    ds,_,crops=build(tmp_path)
    try:
        out=ds[0];k=out['key'];hand=k['hand']
        if damage in('nan_emg','nan_target','payload_generation'):
            name=f'emg_{hand}_filtered'if damage!='nan_target'else f'generated_joint_angles_{hand}'
            p=ds.store.root/(name+'.dat')
            if damage=='payload_generation':
                replacement=p.with_suffix('.replacement');replacement.write_bytes(p.read_bytes());replacement.replace(p)
            else:
                a=np.memmap(p,mode='r+',dtype=np.float32,shape=(ds.store.total_rows,8 if damage=='nan_emg' else 20))
                a[k['center_idx'],0]=np.nan;a.flush();del a
        elif damage=='index_bytes':
            a=ds.index['points']['center'];a.setflags(write=True);a[0]+=1;a.setflags(write=False)
        elif damage=='layout':ds.fusion_views[0].emg_layout='target_hand'
        else:ds.standalone_views[0]._emg_mean=np.zeros(8,np.float32)
        with pytest.raises(ValueError):ds[0]
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('field,channels',[('emg_left_filtered',8),('generated_joint_angles_left',20)])
def test_preexisting_nonfinite_values_cannot_be_repaired_by_the_native_reader(tmp_path,field,channels):
    ds,_,crops=build(tmp_path)
    try:
        center=ds[0]['key']['center_idx'];p=ds.store.root/(field+'.dat')
        a=np.memmap(p,mode='r+',dtype=np.float32,shape=(ds.store.total_rows,channels));a[center,0]=np.nan;a.flush();del a
        # A fresh IO binding sees the stable corrupted generation. Rejection
        # must now come from numeric validation, not the old generation guard.
        current=api().DensePredictionDataset(ds.store.root,crops,upstream=UPSTREAM,config=config('vits_s'),recipe='vits_s')
        with pytest.raises(ValueError,match='finite'):current[0]
    finally:release_fixture_readers(crops)
