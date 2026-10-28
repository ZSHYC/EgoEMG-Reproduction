"""Original joint EMG/RGB math over real small memmaps and JPEG/LMDB data."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egoemg_repro.data import LegacyStore
from test_classic_emg_data import W,make_store
from test_vision_literal_data import sample_store,crop_store,release_fixture_readers

UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()
POSITIONS=[10,12,0,1,2,4,5,6]


def api():
    try:return importlib.import_module('egoemg_repro.fusion_literal_data')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.fusion_literal_data':pytest.fail('original fusion joint reader missing')
        raise


def config():
    pytest.importorskip('hydra')
    from egoemg_repro.fusion_literal import compose_fusion
    return compose_fusion(UPSTREAM)


def joint_store(root,**kwargs):
    arrays=sample_store(root,**kwargs)
    manifest=json.loads((root/'manifest.json').read_text())
    # Explicit zero-copy schema alias lets the original reference consume the
    # same physical bytes using its requested filtered_paper field names.
    for hand in ('left','right'):
        manifest['fields'][f'emg_{hand}_filtered_paper']=copy.deepcopy(manifest['fields'][f'emg_{hand}_filtered'])
    (root/'manifest.json').write_text(json.dumps(manifest))
    return arrays


def dataset(data,crops,role,*,fraction=.12,cfg=None,norm=None,partitions=None):
    m=api()
    return m.FusionLiteralDataset(data,partitions or m.build_fusion_partitions(LegacyStore(data),fraction),crops,
        upstream=UPSTREAM,config=config() if cfg is None else cfg,
        normalization=m.fusion_normalization(UPSTREAM) if norm is None else norm,role=role)


def test_possible_fusion_jitter_footprints_never_touch_heldout_rows(tmp_path):
    m=api();splits=np.repeat([0,1,2,3],[8*W,W,W,W]);make_store(tmp_path,len(splits),splits=splits,signals=False)
    result=m.build_fusion_partitions(LegacyStore(tmp_path));roles={r:set() for r in ('train','val','test')}
    assert result['settings']['train_stride']==400 and result['settings']['eval_stride']==W
    assert result['settings']['repeat_by_role']=={'train':2,'val':2,'test':2}
    assert result['coverage']['val']['windows']==1 and result['coverage']['test']['windows']==3
    assert result['coverage']['train']['windows']==116 and result['discarded']['fusion_jitter_unsafe_anchor']==1
    for w in result['windows']:
        roles[w['role']].update(range(w['footprint_start'],w['footprint_end']))
        assert w['end']-w['start']==W
        if w['role']=='train':
            assert w['start']%400==0 and w['jitter_upper']==min(400,len(splits)-w['end'])
            assert w['footprint_end']==w['end']+max(0,w['jitter_upper']-1)
            assert np.all(splits[w['footprint_start']:w['footprint_end']]==0)
        else:assert w['jitter_upper']==0 and w['footprint_end']==w['end']
    assert all(not roles[a]&roles[b] for a,b in (('train','val'),('train','test'),('val','test')))
    assert result['coverage']['train']['consumed_hand_samples']==4*result['coverage']['train']['windows']
    assert 'legacy' in result['provenance']['available_corpus']


def test_original_normalization_namespace_failure_and_explicit_pinned_asset(tmp_path):
    pytest.importorskip('torch');m=api();norm=m.fusion_normalization(UPSTREAM)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    values=dict(norm_mode='per-dataset',dataset_name='egoemg_unified',emg_field_preference='filtered_paper',
        emg_layout='emg2pose_interpolate16',target_hand='left')
    missing=SimpleNamespace(**values,norm_stats_path=str(UPSTREAM/'assets/per_dataset_norm_stats.json'))
    with pytest.raises(KeyError):EgoEmgMemmapDataset._load_norm_stats(missing)
    original=SimpleNamespace(**values,norm_stats_path=str(UPSTREAM/'assets/per_dataset_norm_stats_unified.json'))
    EgoEmgMemmapDataset._load_norm_stats(original)
    assert norm['asset_sha256']=='6a0d029147fd3f4befef5c33f82c9044606de5a8b335b15ffa434ad3767701f5'
    assert norm['mean']==original._emg_mean==0.00011658444197544563
    assert norm['std']==original._emg_std==10.839283499203331
    assert norm['canonical_declared_namespace_available'] is False
    assert norm['estimated_from_project_train_rows'] is False
    assert '/home/' not in json.dumps(norm) and norm['scientific_eligible'] is False


@pytest.mark.parametrize('role',['train','val','test'])
@pytest.mark.parametrize('hand',['left','right'])
def test_original_joint_samples_transforms_masks_and_rng_are_bit_exact(tmp_path,role,hand):
    torch=pytest.importorskip('torch');m=api();cfg=config();data=tmp_path/'data';joint_store(data,test=role=='test')
    crops=tmp_path/'crops';crop_store(crops)
    ds=dataset(data,crops,role,cfg=cfg);count=len(ds.windows);index=0 if hand=='left' else count
    from hydra.utils import instantiate
    from egoemg.transforms import Compose
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset,_release_readonly_lmdb
    original=EgoEmgMemmapDataset(data,W,stride=400,target_hand=hand,modalities=['emg','joint_angles','labels'],
        fields=[f'emg_{hand}_filtered_paper'],emg_field_preference='filtered_paper',emg_layout='emg2pose_interpolate16',
        emg2pose_channel_indices=POSITIONS,channel_interpolate=False,norm_mode='per-dataset',
        norm_stats_path=str(UPSTREAM/'assets/per_dataset_norm_stats_unified.json'),dataset_name='egoemg_unified',
        vision_num_frames=1,center_target_only=True,per_episode_crops_dir=str(crops),jitter=role=='train',
        transform=Compose([instantiate(t) for t in cfg.transforms[role]]))
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir() if p.is_file()}
    w=ds.windows[0];np.random.seed(42);torch.manual_seed(42)
    expected=original[0] if role=='train' else original._getitem_center_supervised(0,w['start'],w['end'],w['center'])
    np_after=np.random.get_state();torch_after=torch.get_rng_state()
    for path,env,txn in list(original._pec_lmdb_cache.values()):txn.abort();_release_readonly_lmdb(path,env)
    original._pec_lmdb_cache.clear()
    try:
        np.random.seed(42);torch.manual_seed(42);actual=ds[index];got=np.random.get_state()
        for key in ('emg','emg_channel_mask','vision_img','joint_angles','label_valid_mask','vision_valid_mask','vision_frame_indices'):
            assert actual[key].shape==expected[key].shape and actual[key].dtype==expected[key].dtype
            assert actual[key].tobytes()==np.ascontiguousarray(expected[key]).tobytes(),key
        assert np_after[0]==got[0] and np.array_equal(np_after[1],got[1]) and np_after[2:]==got[2:]
        assert torch.equal(torch_after,torch.get_rng_state())
        assert actual['emg'].shape==(16,W) and actual['joint_angles'].shape==(22,1)
        assert actual['target_hand']==hand and actual['dataset_name']=='egoemg'
        assert actual['label_valid_mask'].tolist()==[True]  # false wrist-valid remains ignored by original
        assert actual['vision_valid_mask'].tolist()==[hand=='left']
        assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}==before
        assert not torch.cuda.is_initialized()
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('leftover',[0,1,17,400])
def test_episode_tail_keeps_original_jitter_range(tmp_path,leftover):
    torch=pytest.importorskip('torch');m=api();data=tmp_path/'data';joint_store(data,n=W+leftover)
    crops=tmp_path/'crops';crop_store(crops);ds=dataset(data,crops,'train',fraction=0)
    try:
        assert ds.windows[0]['jitter_upper']==min(400,leftover)
        for seed in (0,42):
            np.random.seed(seed);expected=int(np.random.randint(0,min(400,leftover))) if leftover>0 else 0
            np.random.seed(seed);torch.manual_seed(seed);actual=ds[0]
            assert actual['jitter_offset']==expected and actual['end']<=W+leftover
    finally:release_fixture_readers(crops)


def test_repeat_identity_frame_alias_and_zero_filled_normalized_channels(tmp_path):
    pytest.importorskip('torch');m=api();data=tmp_path/'data';joint_store(data,mapping='image_webcam_frame_index')
    crops=tmp_path/'crops';crop_store(crops,flat=True);ds=dataset(data,crops,'val');n=len(ds.windows)
    try:
        records=[ds[i] for i in range(len(ds))]
        assert [r['target_hand'] for r in records]==(['left']*n+['right']*n)*2
        assert [r['fusion_unique_sample_index'] for r in records]==list(range(2*n))*2
        assert [r['fusion_repeat_index'] for r in records]==[0]*(2*n)+[1]*(2*n)
        assert [r['fusion_sample_index'] for r in records]==list(range(4*n))
        assert ds.frame_mapping_field=='image_webcam_frame_index'
        inactive=[i for i in range(16) if i not in POSITIONS]
        expected=np.float32(-1.0755732e-5)  # native float32 subtraction and division, not one final cast
        assert np.all(records[0]['emg'][inactive]==expected)
        assert records[0]['emg_channel_mask'].nonzero()[0].tolist()==[0,1,2,4,5,6,10,12]
        assert records[n]['vision_valid_mask'].tolist()==[False] and records[n]['label_valid_mask'].tolist()==[True]
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['partition','normalization','config','missing_emg','short_emg','shape_emg','dtype_emg','merged','missing_database','corrupt_jpeg','negative_frame','frame_priority'])
def test_joint_reader_rejects_bad_inputs_without_dropping_requests(tmp_path,damage):
    pytest.importorskip('torch');m=api();data=tmp_path/'data';joint_store(data)
    crops=tmp_path/'crops';path=crop_store(crops,corrupt=damage=='corrupt_jpeg')
    partitions=m.build_fusion_partitions(LegacyStore(data));norm=m.fusion_normalization(UPSTREAM);cfg=config()
    manifest=json.loads((data/'manifest.json').read_text())
    if damage=='partition':partitions['windows'][0]['jitter_upper']=0
    elif damage=='normalization':norm['mean']=0
    elif damage=='config':cfg.egoemg_channel_interpolate=True
    elif damage=='missing_emg':del manifest['fields']['emg_left_filtered']
    elif damage=='short_emg':(data/'emg_left_filtered.dat').write_bytes(b'1234')
    elif damage=='shape_emg':manifest['fields']['emg_left_filtered']['shape']=[3*W,7]
    elif damage=='dtype_emg':manifest['fields']['emg_left_filtered']['dtype']='float64'
    elif damage=='merged':manifest['fields']['dataset_source_id']={'shape':[3*W],'dtype':'int32'}
    elif damage=='missing_database':path.rename(crops/'unavailable.lmdb')
    elif damage=='negative_frame':np.full(3*W,-1,np.int64).tofile(data/'video.dat')
    elif damage=='frame_priority':
        np.full(3*W,29,np.int64).tofile(data/'legacy-video.dat')
        manifest['fields']['image_webcam_frame_index']={'filename':'legacy-video.dat','shape':[3*W],'dtype':'int64'}
    (data/'manifest.json').write_text(json.dumps(manifest))
    if damage=='frame_priority':partitions=m.build_fusion_partitions(LegacyStore(data))
    try:
        if damage in ('corrupt_jpeg','negative_frame','frame_priority'):
            ds=dataset(data,crops,'val',cfg=cfg,norm=norm,partitions=partitions)
            assert len(ds)==4
            if damage=='corrupt_jpeg':
                with pytest.raises(OSError):ds[0]
            else:
                actual=ds[0];assert actual['vision_valid_mask'].tolist()==[damage=='frame_priority']
                assert actual['label_valid_mask'].tolist()==[True]
                if damage=='frame_priority':assert actual['vision_frame_indices'].tolist()==[17]
        else:
            with pytest.raises((ValueError,KeyError,FileNotFoundError)):
                dataset(data,crops,'val',cfg=cfg,norm=norm,partitions=partitions)
    finally:release_fixture_readers(crops)
