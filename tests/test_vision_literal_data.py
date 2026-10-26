"""Real small memmaps/JPEGs: source algorithm and data boundaries, never GPU qualification."""
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.data import LegacyStore
from test_classic_emg_data import make_store,W


UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.vision_literal_data')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.vision_literal_data':pytest.fail('original vision point-data adapter missing')
        raise


def config():
    pytest.importorskip('hydra')
    from egoemg_repro.vision_literal import compose_vision
    return compose_vision(UPSTREAM)


def sample_store(root,*,mapping='image_head_frame_index',test=False,n=3*W):
    arrays=make_store(root,n,splits=np.full(n,1 if test else 0,dtype=np.int8))
    manifest=json.loads((root/'manifest.json').read_text())
    for h,hand in enumerate(('left','right')):
        arrays[f'generated_joint_angles_{hand}'][:,0]=np.arange(n,dtype=np.float32)*.0001+h
        arrays[f'mocap_{hand}_wrist_pitch'][:]=73.12345+h
        arrays[f'mocap_{hand}_wrist_angles_valid'][:]=False
        for name in (f'generated_joint_angles_{hand}',f'mocap_{hand}_wrist_pitch',f'mocap_{hand}_wrist_angles_valid'):
            arrays[name].tofile(root/(name+'.dat'))
    frames=np.full(n,17,np.int64);frames.tofile(root/'video.dat')
    manifest['fields'][mapping]={'filename':'video.dat','shape':[n],'dtype':'int64'}
    (root/'manifest.json').write_text(json.dumps(manifest))
    with np.load(root/'metadata.npz',allow_pickle=False) as z:metadata={k:z[k].copy() for k in z.files}
    metadata['episode_id']=np.array(['episode_000000'],dtype='S32');np.savez(root/'metadata.npz',**metadata)
    return arrays


def crop_store(root,*,flat=False,corrupt=False):
    lmdb=pytest.importorskip('lmdb');Image=pytest.importorskip('PIL.Image');root.mkdir()
    rgb=np.zeros((256,256,3),np.uint8);rgb[:,:128,0]=255;rgb[:,128:,2]=255
    buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=100,subsampling=0)
    path=root/('episode_000000.data.mdb' if flat else 'episode_000000.lmdb')
    env=lmdb.open(str(path),subdir=not flat,map_size=2**20)
    with env.begin(write=True) as txn:txn.put(b'00000017_L',b'corrupt-jpeg' if corrupt else buffer.getvalue())
    env.close();return path


def release_fixture_readers(root):
    """Close only this test's readonly LMDB environments, across the original/bridge registries."""
    from egoemg_repro.crops import _READERS
    for key in list(_READERS):
        if Path(key[1]).is_relative_to(root):_READERS.pop(key).close()


def test_every_possible_train_jitter_span_is_disjoint_and_repeat_is_explicit(tmp_path):
    m=api();splits=np.repeat([0,1,2,3],[8*W,W,W,W]);make_store(tmp_path,len(splits),splits=splits,signals=False)
    value=m.build_vision_partitions(LegacyStore(tmp_path))
    assert value['coverage']['train']['windows']==59
    assert value['coverage']['train']['hand_samples']==118
    assert value['coverage']['train']['consumed_hand_samples']==236
    assert value['coverage']['val']['consumed_hand_samples']==4
    assert value['coverage']['test']['consumed_hand_samples']==12
    assert value['discarded']['vision_jitter_unsafe_anchor']==1
    assert value['settings']['repeat_by_role']=={'train':2,'val':2,'test':2}
    roles={role:set() for role in ('train','val','test')}
    for w in value['windows']:
        roles[w['role']].update(range(w['footprint_start'],w['footprint_end']))
        assert w['jitter_max_offset']==max(0,w['jitter_upper']-1)
        if w['role']=='train':
            assert w['jitter_upper']==min(780,len(splits)-w['end'])
            assert np.all(splits[w['footprint_start']:w['footprint_end']]==0)
        else:assert w['jitter_upper']==0 and w['footprint_end']==w['end']
    assert not roles['train']&roles['val'] and not roles['train']&roles['test'] and not roles['val']&roles['test']


@pytest.mark.parametrize('test',[False,True])
def test_flat_legacy_crops_keep_literal_masks_mapping_hand_order_and_repeated_identity(tmp_path,test):
    pytest.importorskip('torch');m=api();data=tmp_path/'data';sample_store(data,mapping='image_webcam_frame_index',test=test)
    crops=tmp_path/'crops';crop_store(crops,flat=True);partitions=m.build_vision_partitions(LegacyStore(data));role='test' if test else 'val'
    ds=m.VisionLiteralDataset(data,partitions,crops,upstream=UPSTREAM,config=config(),role=role)
    n=len([w for w in partitions['windows'] if w['role']==role]);assert len(ds)==4*n
    try:
        records=[ds[i] for i in range(len(ds))]
        assert [x['target_hand'] for x in records]==(['left']*n+['right']*n)*2
        assert [x['vision_unique_sample_index'] for x in records]==list(range(2*n))*2
        assert [x['vision_repeat_index'] for x in records]==[0]*(2*n)+[1]*(2*n)
        assert [x['vision_sample_index'] for x in records]==list(range(4*n))
        assert records[0]['vision_img'].shape==(3,256,256) and records[0]['joint_angles'].shape==(22,1)
        assert records[0]['vision_frame_indices'].tolist()==[17]
        assert records[0]['vision_valid_mask'].tolist()==[True]
        missing=records[n];assert missing['vision_valid_mask'].tolist()==[False]
        assert missing['label_valid_mask'].tolist()==[True] and 'angle_valid_mask' not in missing
        assert records[0]['label_valid_mask'].tolist()==[True]  # native ignores false wrist-valid field
        assert records[0]['emg'].shape==(16,0)
        assert records[0]['vision_img'][0,30,30]>2 and records[0]['vision_img'][2,30,200]>2
        np.testing.assert_allclose(missing['vision_img'][:,0,0],[-.485/.229,-.456/.224,-.406/.225],atol=1e-6)
        assert ds.frame_mapping_field=='image_webcam_frame_index'
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('role,leftover',[('train',None),('val',None),('train',0),('train',1),('train',17),('train',780)])
def test_point_values_and_rng_match_real_original_reader_including_wrist_rounding(tmp_path,role,leftover):
    pytest.importorskip('torch');m=api();cfg=config();data=tmp_path/'data'
    sample_store(data,n=3*W if leftover is None else W+leftover)
    crops=tmp_path/'crops';crop_store(crops);partitions=m.build_vision_partitions(LegacyStore(data))
    ds=m.VisionLiteralDataset(data,partitions,crops,upstream=UPSTREAM,config=cfg,role=role)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset,_release_readonly_lmdb
    original=EgoEmgMemmapDataset(data,W,stride=780,target_hand='left',modalities=['joint_angles','labels'],
        vision_num_frames=1,skip_emg_loading=True,per_episode_crops_dir=str(crops),emg_layout='emg2pose_interpolate16',
        emg2pose_channel_indices=[10,12,0,1,2,4,5,6],channel_interpolate=False,jitter=role=='train')
    paths=[p for p in data.iterdir() if p.is_file()];before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    window=next(w for w in partitions['windows'] if w['role']==role)
    np.random.seed(42)
    if role=='train':expected=original[0]
    else:expected=original._getitem_vision_only_crops(0,window['center'])
    after=np.random.get_state()
    for path,env,txn in list(original._pec_lmdb_cache.values()):txn.abort();_release_readonly_lmdb(path,env)
    original._pec_lmdb_cache.clear()
    try:
        np.random.seed(42);actual=ds[0];actual_after=np.random.get_state()
        for key in ('vision_img','joint_angles','label_valid_mask','vision_valid_mask','vision_frame_indices','emg','emg_channel_mask'):
            assert actual[key].dtype==expected[key].dtype and actual[key].shape==expected[key].shape
            assert actual[key].tobytes()==np.ascontiguousarray(expected[key]).tobytes(),key
        assert after[0]==actual_after[0] and np.array_equal(after[1],actual_after[1]) and after[2:]==actual_after[2:]
        assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}==before
    finally:release_fixture_readers(crops)


@pytest.mark.parametrize('damage',['partition','missing_database','corrupt_jpeg','negative_frame','conflicting_frame_alias'])
def test_data_failures_do_not_drop_population_or_rewrite_generated_validity(tmp_path,damage):
    pytest.importorskip('torch');m=api();data=tmp_path/'data';sample_store(data)
    crops=tmp_path/'crops';path=crop_store(crops,flat=True,corrupt=damage=='corrupt_jpeg')
    partitions=m.build_vision_partitions(LegacyStore(data))
    if damage=='partition':partitions['windows'][0]['jitter_max_offset']=0
    elif damage=='missing_database':path.unlink()
    elif damage=='negative_frame':np.full(3*W,-1,np.int64).tofile(data/'video.dat')
    elif damage=='conflicting_frame_alias':
        np.full(3*W,29,np.int64).tofile(data/'legacy-video.dat')
        manifest=json.loads((data/'manifest.json').read_text())
        manifest['fields']['image_webcam_frame_index']={'filename':'legacy-video.dat','shape':[3*W],'dtype':'int64'}
        (data/'manifest.json').write_text(json.dumps(manifest))
        partitions=m.build_vision_partitions(LegacyStore(data))
    try:
        if damage in ('partition','missing_database'):
            with pytest.raises((ValueError,FileNotFoundError)):
                m.VisionLiteralDataset(data,partitions,crops,upstream=UPSTREAM,config=config(),role='val')
        else:
            ds=m.VisionLiteralDataset(data,partitions,crops,upstream=UPSTREAM,config=config(),role='val')
            if damage=='corrupt_jpeg':
                with pytest.raises(OSError):ds[0]
            elif damage=='negative_frame':
                value=ds[0];assert value['vision_valid_mask'].tolist()==[False]
                assert value['label_valid_mask'].tolist()==[True] and len(ds)==4
            else:
                assert ds.frame_mapping_field=='image_head_frame_index'
                assert ds[0]['vision_valid_mask'].tolist()==[True]
    finally:release_fixture_readers(crops)
