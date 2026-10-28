"""Real native full trajectory reads and complete fixed-fusion holdout separation."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.data import LegacyStore
from test_classic_emg_data import make_store
from test_emg_initializer_literal import UPSTREAM,api as literal_api,config

W=12000


def api():
    try:return importlib.import_module('egoemg_repro.emg_initializer_data')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emg_initializer_data':pytest.fail('compatible native full trajectory reader missing')
        raise


def source(root,*,n=24*W,signals=True,optional_labels=False,webcam=False):
    splits=np.repeat([0,1,2,3],[20*W,W,W,2*W]) if n==24*W else np.zeros(n,np.int8)
    arrays=make_store(root,n,splits=splits,signals=signals)
    manifest=json.loads((root/'manifest.json').read_text())
    if signals:
        for h,hand in enumerate(('left','right')):
            arrays[f'generated_joint_angles_{hand}'][:,0]=np.arange(n,dtype=np.float32)*.0001+h
            arrays[f'mocap_{hand}_wrist_pitch'][:]=73.12345+h
            for name in (f'generated_joint_angles_{hand}',f'mocap_{hand}_wrist_pitch'):arrays[name].tofile(root/(name+'.dat'))
            manifest['fields'][f'emg_{hand}_filtered_paper']=copy.deepcopy(manifest['fields'][f'emg_{hand}_filtered'])
    if optional_labels:
        for name,arr in {'label_gesture_class':np.arange(n,dtype=np.int32)%59,'label_gesture_active':np.arange(n)%2==0}.items():
            arrays[name]=arr;arr.tofile(root/(name+'.dat'));manifest['fields'][name]=dict(filename=name+'.dat',dtype=str(arr.dtype),shape=list(arr.shape))
    (root/'manifest.json').write_text(json.dumps(manifest))
    if webcam:
        with np.load(root/'metadata.npz',allow_pickle=False) as z:md={k:z[k] for k in z.files}
        md['episode_webcam_video_path']=md.pop('episode_head_video_path');np.savez(root/'metadata.npz',**md)
    return arrays


def dataset(root,role='train',*,partitions=None,cfg=None,norm=None):
    m=api();return m.EmgInitializerDataset(root,partitions or m.build_initializer_partitions(LegacyStore(root)),
        upstream=UPSTREAM,config=config() if cfg is None else cfg,
        normalization=literal_api().emg_initializer_normalization(UPSTREAM) if norm is None else norm,role=role)


def test_all_possible_full_windows_exclude_current_fusion_holdouts_and_keep_native_grid(tmp_path):
    m=api();splits=source(tmp_path,signals=False)['frame_split_id'];store=LegacyStore(tmp_path)
    from egoemg_repro.fusion_literal_data import build_fusion_partitions
    fusion=build_fusion_partitions(store);p=m.build_initializer_partitions(store);again=m.build_initializer_partitions(store)
    assert p==again and p['coverage']['train']['windows']>0 and p['coverage']['val']['windows']>0
    assert p['settings']['repeat_by_role']==dict(train=1,val=1) and 'test' not in p['coverage']
    train=np.zeros(len(splits),bool);val=train.copy();forbidden=train.copy()
    for interval in fusion['validation_intervals']:forbidden[interval['start']:interval['end']]=True
    for w in fusion['windows']:
        if w['role']=='test':forbidden[w['footprint_start']:w['footprint_end']]=True
    for w in p['windows']:
        assert w['end']-w['start']==W and w['center']==w['start']+W//2
        if w['role']=='train':
            assert w['start']%1200==0 and w['jitter_upper']==min(1200,len(splits)-w['end'])
            assert w['footprint_end']==w['end']+max(0,w['jitter_upper']-1)
            assert np.all(splits[w['start']:w['footprint_end']]==0)
            train[w['start']:w['footprint_end']]=True
        else:
            assert w['jitter_upper']==0 and any(v['start']<=w['start'] and w['end']<=v['end'] for v in fusion['validation_intervals'])
            val[w['start']:w['end']]=True
    assert not np.any(train&forbidden) and not np.any(train&val)
    assert p['discarded']['overlaps_fusion_holdout']>0
    assert p['coverage']['train']['hand_samples']==2*p['coverage']['train']['windows']


@pytest.mark.parametrize('fraction',[True,-.1,1,float('nan')])
def test_invalid_holdout_parameters_reject(tmp_path,fraction):
    source(tmp_path,signals=False)
    with pytest.raises(ValueError,match='fraction|validation'):api().build_initializer_partitions(LegacyStore(tmp_path),fraction)


@pytest.mark.parametrize('role',['train','val'])
@pytest.mark.parametrize('hand',['left','right'])
@pytest.mark.parametrize('optional_labels',[False,True])
def test_full_samples_match_genuine_native_reader_rng_values_masks_and_optional_labels(tmp_path,role,hand,optional_labels):
    torch=pytest.importorskip('torch');m=api();source(tmp_path,optional_labels=optional_labels);cfg=config();ds=dataset(tmp_path,role,cfg=cfg)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from egoemg.transforms import Compose
    from hydra.utils import instantiate
    reference=EgoEmgMemmapDataset(tmp_path,W,stride=1200 if role=='train' else 1,
        allowed_splits=['train'],target_hand=hand,modalities=['emg','joint_angles','labels'],
        emg_field_preference='filtered_paper',emg_layout='target_hand',norm_mode='per-dataset',
        norm_stats_path=str(UPSTREAM/literal_api().NORM_ASSET),dataset_name='egoemg',jitter=role=='train',
        transform=Compose([instantiate(t) for t in cfg.transforms[role]]))
    w=ds.windows[0];native_index=int(np.flatnonzero(reference._block_start==w['start'])[0])
    np.random.seed(42);torch.manual_seed(42);want=reference[native_index];np_after=np.random.get_state();torch_after=torch.get_rng_state()
    np.random.seed(42);torch.manual_seed(42);got=ds[0 if hand=='left' else len(ds.windows)];after=np.random.get_state()
    for key in ('emg','emg_channel_mask','joint_angles','label_valid_mask','generated_label_valid','frame_split_id','label_gesture_class','label_gesture_active'):
        assert got[key].shape==want[key].shape and got[key].dtype==want[key].dtype
        assert got[key].tobytes()==np.ascontiguousarray(want[key]).tobytes(),key
    assert got['emg'].shape==(8,W) and got['joint_angles'].shape==(22,W) and got['label_valid_mask'].shape==(W,)
    assert got['dataset_name']==want['dataset_name']=='egoemg' and got['target_hand']==hand
    assert np.array_equal(np_after[1],after[1]) and np_after[2:]==after[2:] and torch.equal(torch_after,torch.get_rng_state())
    assert len(ds)==2*len(ds.windows) and got['initializer_repeat_index']==0
    assert got['jitter_offset']==got['window_start_idx']-w['start']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('leftover',[0,1,17,1200])
def test_original_tail_jitter_range_and_generated_false_mask_are_preserved(tmp_path,leftover):
    torch=pytest.importorskip('torch');source(tmp_path,n=W+leftover);m=api()
    p=m.build_initializer_partitions(LegacyStore(tmp_path),val_fraction=0);ds=dataset(tmp_path,partitions=p)
    for seed in (0,42):
        np.random.seed(seed);offset=int(np.random.randint(0,min(1200,leftover))) if leftover else 0
        np.random.seed(seed);torch.manual_seed(seed);got=ds[0]
        assert got['jitter_offset']==offset and got['window_end_idx']<=W+leftover
        assert got['joint_angles'].shape==(22,W)
    if leftover==0:assert ds[0]['label_valid_mask'][1]==False and np.all(ds[0]['emg_channel_mask'])


@pytest.mark.parametrize('damage',['partition','normalization','missing_emg','short_target','bad_target_dtype','merged','test_role'])
def test_bad_or_changed_inputs_are_rejected_without_repair(tmp_path,damage):
    pytest.importorskip('torch');source(tmp_path);m=api();p=m.build_initializer_partitions(LegacyStore(tmp_path));norm=literal_api().emg_initializer_normalization(UPSTREAM)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    if damage=='partition':p['windows'][0]['jitter_upper']=0
    elif damage=='normalization':norm['hands']['right']['std'][0]=1.
    elif damage=='missing_emg':del manifest['fields']['emg_right_filtered']
    elif damage=='short_target':(tmp_path/'generated_joint_angles_left.dat').write_bytes(b'bad')
    elif damage=='bad_target_dtype':manifest['fields']['generated_joint_angles_left']['dtype']='int32'
    elif damage=='merged':manifest['fields']['dataset_source_id']={'dtype':'int32','shape':[24*W]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError,KeyError),match='partition|normalization|EMG|field|byte|merged|role|dtype'):
        dataset(tmp_path,'test' if damage=='test_role' else 'train',partitions=p,norm=norm)


def test_webcam_metadata_alias_original_defaults_and_native_final_partial_collation(tmp_path):
    torch=pytest.importorskip('torch');source(tmp_path,webcam=True);ds=dataset(tmp_path,'val');cfg=config()
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cfg);dm.num_workers=dm.eval_num_workers=0
    subset=torch.utils.data.Subset(ds,[0,len(ds.windows),1]);loader=dm._make_loader(subset,shuffle=False)
    batch=next(iter(loader));assert batch['emg'].shape==(3,8,W) and batch['joint_angles'].shape==(3,22,W)
    assert loader.batch_size==260 and loader.drop_last is False and loader.generator is None
    assert batch['dataset_name']==['egoemg']*3 and batch['keystroke_labels']==[]
    assert ds[0]['episode_head_video_path']==ds[0]['episode_wrist_left_video_path']==''
    assert ds[0]['initializer_repeat_index']==0 and batch['initializer_repeat_index'].tolist()==[0,0,0]


@pytest.mark.parametrize('role',['train','val'])
def test_empty_required_role_rejects_without_reassigning_windows(tmp_path,role):
    pytest.importorskip('torch');source(tmp_path,n=W)
    if role=='train':np.ones(W,np.int8).tofile(tmp_path/'frame_split_id.dat')
    with pytest.raises(ValueError,match='role|nonempty'):dataset(tmp_path,role)


def test_actual_last_possible_jitter_row_is_consumed_without_clipping(tmp_path,monkeypatch):
    pytest.importorskip('torch');source(tmp_path,n=W+1200);m=api()
    p=m.build_initializer_partitions(LegacyStore(tmp_path),val_fraction=0);ds=dataset(tmp_path,partitions=p)
    monkeypatch.setattr(np.random,'randint',lambda low,high:high-1)
    got=ds[0]
    assert got['jitter_offset']==1199 and got['window_end_idx']==p['windows'][0]['footprint_end']==13199
    assert got['joint_angles'].shape==(22,W)


def test_masked_nonfinite_trajectory_values_are_preserved_for_later_explicit_admission(tmp_path):
    pytest.importorskip('torch');source(tmp_path,n=W)
    a=np.memmap(tmp_path/'generated_joint_angles_left.dat',dtype='float32',mode='r+',shape=(W,20))
    a[1,5]=np.nan;a.flush();del a
    ds=dataset(tmp_path);got=ds[0]
    assert not got['label_valid_mask'][1] and np.isnan(got['joint_angles'][5,1])
    assert np.isnan(np.fromfile(tmp_path/'generated_joint_angles_left.dat',dtype='float32').reshape(W,20)[1,5])


@pytest.mark.parametrize('damage',['missing_declared_gesture','missing_declared_wrist_valid','conflicting_alias'])
def test_declared_optional_files_and_physical_alias_cannot_be_silently_dropped(tmp_path,damage):
    pytest.importorskip('torch');source(tmp_path,optional_labels=True)
    if damage=='missing_declared_gesture':(tmp_path/'label_gesture_class.dat').unlink()
    elif damage=='missing_declared_wrist_valid':(tmp_path/'mocap_right_wrist_angles_valid.dat').unlink()
    else:
        manifest=json.loads((tmp_path/'manifest.json').read_text());manifest['fields']['emg_left_filtered_paper']['filename']='other.dat'
        (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError,FileNotFoundError),match='field|alias|label|wrist'):dataset(tmp_path)
