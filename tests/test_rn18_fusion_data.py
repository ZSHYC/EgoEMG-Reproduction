"""Native RN18 fusion8-channel/12000 center samples on real small IO fixtures."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egoemg_repro.data import LegacyStore
from test_fusion_literal_data import joint_store,UPSTREAM
from test_vision_literal_data import crop_store,release_fixture_readers

W=12000


def api():return importlib.import_module('egoemg_repro.rn18_fusion_data')


def config():
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    return compose_rn18_fusion(UPSTREAM)


def dataset(data,crops,role,*,fraction=.12,norm=None,partitions=None):
    pytest.importorskip('torch')
    m=api();cfg=config()
    return m.Rn18FusionDataset(data,partitions or m.build_rn18_fusion_partitions(LegacyStore(data),fraction),crops,
        upstream=UPSTREAM,config=cfg,normalization=m.rn18_fusion_normalization(UPSTREAM,cfg)if norm is None else norm,role=role)


def test_native_hand_vector_normalization_and_failed_declared_namespace_are_explicit():
    pytest.importorskip('torch')
    m=api();n=m.rn18_fusion_normalization(UPSTREAM,config())
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    assert n['canonical_normalization_errors_by_hand']==dict(left='KeyError',right='KeyError')
    assert n['normalization_namespace']=='egoemg'and not n['estimated_from_project_train_rows']
    for hand in('left','right'):
        v=SimpleNamespace(norm_mode='per-dataset',norm_stats_path=str(UPSTREAM/n['asset']),dataset_name='egoemg',
            emg_field_preference='filtered_paper',emg_layout='target_hand',target_hand=hand)
        EgoEmgMemmapDataset._load_norm_stats(v)
        assert n['hands'][hand]['dtype']=='float32'and np.asarray(n['hands'][hand]['mean']).shape==(8,)
        np.testing.assert_array_equal(np.asarray(n['hands'][hand]['mean'],np.float32),v._emg_mean)
        np.testing.assert_array_equal(np.asarray(n['hands'][hand]['std'],np.float32),v._emg_std)
    assert n['hands']['left']['std']!=n['hands']['right']['std']and n['scientific_eligible']is False


def test_closed_12000_provider_preserves_all_native_jitter_and_disjoint_roles(tmp_path):
    m=api();joint_store(tmp_path,n=11*W)
    np.repeat([0,1,2,3],[8*W,W,W,W]).astype(np.int8).tofile(tmp_path/'frame_split_id.dat')
    store=LegacyStore(tmp_path);p=m.build_rn18_fusion_partitions(store)
    from egoemg_repro.emgformer_data import build_emgformer_partitions
    assert p['windows']==build_emgformer_partitions(store)['windows']
    assert p['settings']['window_length']==W and p['settings']['train_stride']==1200
    assert p['settings']['repeat_by_role']==dict(train=1,val=1,test=1)
    assert p['settings']['center_target_only_by_role']==dict(train=True,val=True,test=True)
    footprints={r:set()for r in('train','val','test')}
    for w in p['windows']:
        footprints[w['role']].update(range(w['footprint_start'],w['footprint_end']))
        if w['role']=='train':
            assert w['start']%1200==0 and w['jitter_upper']==min(1200,11*W-w['end'])
            assert w['footprint_end']==w['end']+max(0,w['jitter_upper']-1)
        assert p['coverage'][w['role']]['consumed_hand_samples']==2*p['coverage'][w['role']]['windows']
    assert all(footprints[a]and footprints[b]and not footprints[a]&footprints[b]for a,b in(('train','val'),('train','test'),('val','test')))


@pytest.mark.parametrize('role',['train','val','test'])
@pytest.mark.parametrize('hand',['left','right'])
def test_actual_original_center_reader_values_masks_transforms_and_rng_match(tmp_path,role,hand):
    torch=pytest.importorskip('torch');m=api();data=tmp_path/'data';joint_store(data,n=3*W,test=role=='test')
    crops=tmp_path/'crops';crop_store(crops);cfg=config();ds=dataset(data,crops,role);n=len(ds.windows)
    from hydra.utils import instantiate
    from egoemg.transforms import Compose
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset,_release_readonly_lmdb
    original=EgoEmgMemmapDataset(data,W,stride=1200,target_hand=hand,modalities=['emg','joint_angles','labels'],
        fields=[f'emg_{hand}_filtered_paper'],emg_field_preference='filtered_paper',emg_layout='target_hand',
        emg2pose_channel_indices=None,channel_interpolate=False,norm_mode='per-dataset',
        norm_stats_path=str(UPSTREAM/'assets/per_dataset_norm_stats_repro_filtered_paper_alias.json'),dataset_name='egoemg',
        vision_num_frames=1,center_target_only=True,per_episode_crops_dir=str(crops),jitter=role=='train',
        transform=Compose([instantiate(x)for x in cfg.transforms[role]]))
    w=ds.windows[0];np.random.seed(42);torch.manual_seed(42)
    expected=original[0]if role=='train'else original._getitem_center_supervised(0,w['start'],w['end'],w['center'])
    nr=np.random.get_state();tr=torch.get_rng_state()
    for path,env,txn in list(original._pec_lmdb_cache.values()):txn.abort();_release_readonly_lmdb(path,env)
    original._pec_lmdb_cache.clear()
    try:
        np.random.seed(42);torch.manual_seed(42);got=ds[0 if hand=='left'else n];actual=np.random.get_state()
        for key in('emg','emg_channel_mask','vision_img','joint_angles','label_valid_mask','vision_valid_mask','vision_frame_indices'):
            assert got[key].dtype==expected[key].dtype and got[key].shape==expected[key].shape
            assert got[key].tobytes()==np.ascontiguousarray(expected[key]).tobytes(),key
        assert nr[0]==actual[0]and np.array_equal(nr[1],actual[1])and nr[2:]==actual[2:]
        assert torch.equal(tr,torch.get_rng_state())
        assert got['emg'].shape==(8,W)and got['emg_channel_mask'].tolist()==[True]*8
        assert got['joint_angles'].shape==(22,1)and got['label_valid_mask'].tolist()==[True]
        assert got['vision_valid_mask'].tolist()==[hand=='left']and got['dataset_name']=='egoemg'
        assert len(ds)==2*n and got['rn18_fusion_repeat_index']==0 and not torch.cuda.is_initialized()
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('leftover',[0,1,17,1200])
def test_episode_tail_preserves_exact_randint_upper_and_offset(tmp_path,leftover):
    m=api();data=tmp_path/'data';joint_store(data,n=W+leftover);crops=tmp_path/'crops';crop_store(crops)
    ds=dataset(data,crops,'train',fraction=0)
    try:
        assert ds.windows[0]['jitter_upper']==min(1200,leftover)
        for seed in(0,42):
            np.random.seed(seed);expected=int(np.random.randint(0,min(1200,leftover)))if leftover>0 else 0
            np.random.seed(seed);got=ds[0];assert got['jitter_offset']==expected and got['end']<=W+leftover
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['norm_vector','partition','dtype','layout_after_admission','file_after_admission'])
def test_changed_native_inputs_or_cached_layout_cannot_pass(tmp_path,damage):
    pytest.importorskip('torch')
    m=api();data=tmp_path/'data';joint_store(data,n=3*W);crops=tmp_path/'crops';crop_store(crops)
    if damage=='norm_vector':
        norm=m.rn18_fusion_normalization(UPSTREAM,config());norm['hands']['left']['std']=[1.]
        with pytest.raises(ValueError):dataset(data,crops,'val',norm=norm)
        return
    if damage=='partition':
        part=m.build_rn18_fusion_partitions(LegacyStore(data));part['windows'][0]['jitter_upper']=0
        with pytest.raises(ValueError):dataset(data,crops,'train',partitions=part)
        return
    if damage=='dtype':
        manifest=json.loads((data/'manifest.json').read_text());manifest['fields']['emg_left_filtered']['dtype']='int32'
        (data/'manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError):dataset(data,crops,'val')
        return
    ds=dataset(data,crops,'val')
    try:
        if damage=='layout_after_admission':ds.views[0]._frame_memmaps['emg_left_filtered_paper'].dtype=np.int32
        else:
            with(data/'emg_left_filtered.dat').open('ab')as f:f.write(b'changed generation')
        with pytest.raises(ValueError):ds[0]
    finally:release_fixture_readers(crops)


def test_flat_frame_alias_no_repeat_and_missing_crop_keep_generated_mask(tmp_path):
    m=api();data=tmp_path/'data';joint_store(data,n=3*W,mapping='image_webcam_frame_index');crops=tmp_path/'crops';crop_store(crops,flat=True)
    ds=dataset(data,crops,'val')
    try:
        rows=[ds[i]for i in range(len(ds))];n=len(ds.windows)
        assert [r['hand']for r in rows]==['left']*n+['right']*n
        assert [r['rn18_fusion_sample_index']for r in rows]==list(range(2*n))
        assert ds.frame_mapping_field=='image_webcam_frame_index'
        assert rows[n]['label_valid_mask'].tolist()==[True]and rows[n]['vision_valid_mask'].tolist()==[False]
        assert all(r['rn18_fusion_repeat_index']==0 for r in rows)
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['foreign_memmap','foreign_crop_root','vision_mean','vision_std','patch_size','episode_mapping'])
def test_cached_native_reader_cannot_switch_input_provider_or_pixel_semantics(tmp_path,damage):
    m=api();data=tmp_path/'data';joint_store(data,n=3*W);crops=tmp_path/'crops';crop_store(crops)
    ds=dataset(data,crops,'val');view=ds.views[0]
    try:
        if damage=='foreign_memmap':
            replacement=tmp_path/'foreign-emg.dat';np.zeros((3*W,8),np.float32).tofile(replacement)
            view._frame_memmaps['emg_left_filtered_paper']=np.memmap(replacement,mode='r',dtype=np.float32,shape=(3*W,8))
        elif damage=='foreign_crop_root':
            other=tmp_path/'foreign-crops';crop_store(other);view.crops=other
        elif damage=='vision_mean':view.vision_mean+=1
        elif damage=='vision_std':view.vision_std*=2
        elif damage=='patch_size':view.vision_patch_size=224
        else:view._episode_id[0]='unadmitted_episode'
        with pytest.raises(ValueError):ds.check()
    finally:release_fixture_readers(tmp_path)


def test_changed_imported_reader_source_rejects_before_native_normalization(monkeypatch):
    m=api();original=m._file_hash
    def changed_hash(p):return'0'*64 if Path(p)==Path(m.__file__)else original(p)
    monkeypatch.setattr(m,'_file_hash',changed_hash)
    with pytest.raises(ValueError):m.rn18_fusion_normalization(UPSTREAM,config())
