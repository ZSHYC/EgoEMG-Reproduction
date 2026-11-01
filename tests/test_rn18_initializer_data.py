"""New RN18-compatible fold with unchanged native cotrain full trajectories."""
import copy
import importlib
import pickle

import numpy as np
import pytest

from egoemg_repro.classic_emg import _hash
from egoemg_repro.data import LegacyStore,union_intervals
from test_emg_initializer_data import source,W
from test_emg_initializer_literal import UPSTREAM,config


def api():return importlib.import_module('egoemg_repro.rn18_initializer_data')


def require_runtime():
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')


def dataset(root,role='train',fraction=.12,**overrides):
    require_runtime()
    m=api();cfg=config()
    from egoemg_repro.emg_initializer_literal import emg_initializer_normalization
    args=dict(data=root,partitions=m.build_rn18_initializer_partitions(LegacyStore(root),fraction),upstream=UPSTREAM,
        config=cfg,normalization=emg_initializer_normalization(UPSTREAM),role=role)
    args.update(overrides);return m.Rn18InitializerDataset(**args)


def test_original_cotrain_recipe_and_actual12worker_loader_contract():
    require_runtime()
    m=api();p=m.rn18_initializer_contract(UPSTREAM,config());native=p['producer_protocol']
    assert native['upstream_experiment']=='emgformer/regression_egoemg_incre_cotrain_wl12000_small'
    assert native['trainer']['max_epochs']==150 and native['effective_loss_weights']['landmark/fingertip']==.01
    assert native['optimizer']['lr']==.0005 and native['window']==dict(train_length=W,eval_length=W,train_stride=1200,eval_stride=W)
    assert p['batch_size_by_role']==dict(train=260,val=260)and p['workers_by_role']==dict(train=12,val=12)
    assert p['loader']==dict(pin_memory=False,persistent_workers=True,prefetch_factor=2,train_context=None,val_context='forkserver')
    assert p['center_target_only_by_role']==dict(train=False,val=False)and p['repeat_by_role']==dict(train=1,val=1)
    assert not p['native_model_constructed']and not p['full_paper_complete']


@pytest.mark.parametrize('damage',['epochs','lr','loss','workers'])
def test_modified_cotrain_recipe_cannot_be_constructed_as_exact_initializer(damage):
    require_runtime()
    m=api();cfg=config()
    if damage=='epochs':cfg.trainer.max_epochs=200
    elif damage=='lr':cfg.optimizer.lr=.00005
    elif damage=='loss':cfg.loss_weights.fingertip_distance=.02
    else:cfg.num_workers=0
    with pytest.raises(ValueError):m.rn18_initializer_contract(UPSTREAM,cfg)


def test_exact_rn18_roles_full_trajectory_footprints_and_no_initializer_test(tmp_path):
    m=api();source(tmp_path,signals=False);s=LegacyStore(tmp_path)
    from egoemg_repro.rn18_fusion_data import build_rn18_fusion_partitions
    future=build_rn18_fusion_partitions(s);p=m.build_rn18_initializer_partitions(s)
    assert p['windows']==[w for w in future['windows']if w['role']in('train','val')]
    assert p['settings']['center_target_only_by_role']==dict(train=False,val=False)
    assert p['settings']['repeat_by_role']==dict(train=1,val=1)and set(p['coverage'])=={'train','val'}
    assert p['provenance']['future_fusion_partition_content_sha256']==_hash(future)
    assert p['provenance']['mask_policy']=='original generated time label mask broadcast to22; no wrist-valid cleanup'
    train=union_intervals((w['footprint_start'],w['footprint_end'])for w in p['windows']if w['role']=='train')
    held=union_intervals([(w['footprint_start'],w['footprint_end'])for w in future['windows']if w['role']=='test']+
        [(v['start'],v['end'])for v in future['validation_intervals']])
    mask=np.zeros(s.total_rows,bool)
    for a,b in held:mask[a:b]=True
    assert all(not mask[a:b].any()for a,b in train)
    assert p['future_holdout_overlap_rows']==dict(val=0,test=0)
    assert p['coverage']['train']['hand_samples']==future['coverage']['train']['hand_samples']


@pytest.mark.parametrize('fraction',[True,-.1,1,float('nan')])
def test_invalid_holdout_fraction_rejects(tmp_path,fraction):
    source(tmp_path,signals=False)
    with pytest.raises(ValueError):api().build_rn18_initializer_partitions(LegacyStore(tmp_path),fraction)


@pytest.mark.parametrize('role',['train','val'])
@pytest.mark.parametrize('hand',['left','right'])
@pytest.mark.parametrize('optional_labels',[False,True])
def test_native_full_reader_values_masks_and_rng_exact_both_roles(tmp_path,role,hand,optional_labels):
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    m=api();source(tmp_path,optional_labels=optional_labels);ds=dataset(tmp_path,role);cfg=config()
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from egoemg.transforms import Compose
    from hydra.utils import instantiate
    from egoemg_repro.emg_initializer_literal import NORM_ASSET
    native=EgoEmgMemmapDataset(tmp_path,W,stride=1200 if role=='train' else 1,allowed_splits=['train'],target_hand=hand,
        modalities=['emg','joint_angles','labels'],emg_field_preference='filtered_paper',emg_layout='target_hand',
        norm_mode='per-dataset',norm_stats_path=str(UPSTREAM/NORM_ASSET),dataset_name='egoemg',jitter=role=='train',
        transform=Compose([instantiate(t)for t in cfg.transforms[role]]))
    w=ds.windows[0];idx=int(np.flatnonzero(native._block_start==w['start'])[0])
    np.random.seed(42);torch.manual_seed(42);want=native[idx];nr=np.random.get_state();tr=torch.get_rng_state()
    np.random.seed(42);torch.manual_seed(42);got=ds[0 if hand=='left'else len(ds.windows)];actual=np.random.get_state()
    for k in('emg','emg_channel_mask','joint_angles','label_valid_mask','generated_label_valid','frame_split_id','label_gesture_class','label_gesture_active'):
        assert got[k].dtype==want[k].dtype and got[k].shape==want[k].shape
        assert np.ascontiguousarray(got[k]).tobytes()==np.ascontiguousarray(want[k]).tobytes(),k
    assert got['emg'].shape==(8,W)and got['joint_angles'].shape==(22,W)and got['label_valid_mask'].shape==(W,)
    assert np.array_equal(actual[1],nr[1])and actual[2:]==nr[2:]and torch.equal(torch.get_rng_state(),tr)
    assert got['initializer_repeat_index']==0 and got['jitter_offset']==got['window_start_idx']-w['start']
    assert ds.views[0]._center_target_only is False and ds.views[1]._center_target_only is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('leftover',[0,1,17,1200])
def test_native_last_jitter_offset_is_not_clipped_and_false_mask_is_retained(tmp_path,leftover):
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    m=api();source(tmp_path,n=W+leftover);store=LegacyStore(tmp_path)
    store.field('generated_label_valid')
    mask=np.memmap(tmp_path/'generated_label_valid.dat',mode='r+',dtype=bool,shape=(W+leftover,2));mask[:,0]=False;mask.flush()
    ds=dataset(tmp_path,fraction=0);p=m.build_rn18_initializer_partitions(store,0)
    first=p['windows'][0];upper=min(1200,leftover)
    assert first['jitter_upper']==upper and first['footprint_end']==W+max(0,upper-1)
    np.random.seed(9);torch.manual_seed(9);sample=ds[0]
    assert 0<=sample['jitter_offset']<=max(0,upper-1)
    assert sample['label_valid_mask'].shape==(W,)and sample['joint_angles'].shape==(22,W)
    assert not sample['label_valid_mask'].any()


def test_pickle_reconstructs_readonly_views_instead_of_native_memmap_payloads(tmp_path):
    m=api();source(tmp_path);ds=dataset(tmp_path,'val');payload=pickle.dumps(ds)
    assert len(payload)<1_000_000
    again=pickle.loads(payload)
    assert again is not ds and again.views[0]is not ds.views[0]
    assert again[0]['joint_angles'].tobytes()==ds[0]['joint_angles'].tobytes()
    assert all(isinstance(a,np.memmap)and a.mode=='r'and not a.flags.writeable for v in again.views for a in v._frame_memmaps.values())


@pytest.mark.parametrize('damage',['norm_dtype','view_mapping','partitions','center_view'])
def test_mutated_reader_cannot_become_native_loader(tmp_path,damage):
    m=api();source(tmp_path);ds=dataset(tmp_path);m.native_rn18_initializer_loader(ds,scope='cpu_engineering')
    if damage=='norm_dtype':ds.views[0]._emg_mean=ds.views[0]._emg_mean.astype(np.float64)
    elif damage=='view_mapping':
        name=next(k for k in ds.views[0]._frame_memmaps if k.startswith('emg_left_filtered'))
        ds.views[0]._frame_memmaps[name]=np.asarray(ds.views[0]._frame_memmaps[name])
    elif damage=='partitions':ds.partitions['windows'][0]['start']+=1
    else:ds.views[1]._center_target_only=True
    with pytest.raises(ValueError):m.native_rn18_initializer_loader(ds,scope='full')


def test_exact12worker_forkserver_validation_reads_full_native_values_and_closes(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    m=api();source(tmp_path);ds=dataset(tmp_path,'val');loader=m.native_rn18_initializer_loader(ds,scope='full')
    assert loader.batch_size==260 and loader.num_workers==12 and loader.persistent_workers and loader.prefetch_factor==2
    assert loader.pin_memory is False and loader.generator is None and loader.multiprocessing_context.get_start_method()=='forkserver'
    iterator=iter(loader)
    try:
        batch=next(iterator);want=np.stack([ds[j]['joint_angles']for j in range(len(ds))])
        assert torch.equal(batch['joint_angles'],torch.from_numpy(want))and tuple(batch['joint_angles'].shape)==(len(ds),22,W)
        assert all(w.is_alive()for w in iterator._workers)
    finally:iterator._shutdown_workers()
    assert all(not w.is_alive()for w in iterator._workers)


def test_test_role_and_unknown_loader_scope_are_rejected(tmp_path):
    m=api();source(tmp_path)
    with pytest.raises(ValueError):dataset(tmp_path,'test')
    with pytest.raises(ValueError):m.native_rn18_initializer_loader(dataset(tmp_path),scope='partial')
