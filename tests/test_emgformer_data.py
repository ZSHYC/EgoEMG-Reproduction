"""Native trajectory training and center-only evaluation over real memmaps."""
import copy
import importlib
import json

import numpy as np
import pytest

from egoemg_repro.data import LegacyStore
from test_emgformer_literal import UPSTREAM
from test_emg_initializer_data import source

W=12000


def api():
    try:return importlib.import_module('egoemg_repro.emgformer_data')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emgformer_data':pytest.fail('native EMGFormer role reader missing')
        raise


def cfg(size='small'):
    from egoemg_repro.emgformer_literal import compose_emgformer
    return compose_emgformer(UPSTREAM,size)


def dataset(root,role,*,size='small',partitions=None,normalization=None):
    m=api();c=cfg(size);contract=m.emgformer_data_contract(UPSTREAM,c)
    return m.EmgformerDataset(root,partitions or m.build_emgformer_partitions(LegacyStore(root)),
        upstream=UPSTREAM,config=c,normalization=contract['normalization'] if normalization is None else normalization,role=role)


@pytest.mark.parametrize('size',['small','middle','large'])
def test_exact_native_roles_and_missing_unified_namespace_are_explicit(size):
    m=api();c=cfg(size);p=m.emgformer_data_contract(UPSTREAM,c)
    assert p['center_target_only_by_role']==dict(train=False,val=True,test=True)
    assert p['declared_normalization_namespace']=='egoemg_unified'
    assert p['canonical_normalization_error_type']=='KeyError' and p['canonical_namespace_available']is False
    assert p['normalization_namespace']=='egoemg' and p['normalization']['size']==size
    assert p['batch_size']==500 and p['declared_workers']==6 and p['repeat_by_role']==dict(train=1,val=1,test=1)
    assert p['physical_legacy_emg_fields']==dict(left='emg_left_filtered',right='emg_right_filtered')
    assert p['training_target_shape']==[22,W] and p['evaluation_target_shape']==[22,1]
    assert p['data_admitted']is p['scientific_eligible']is p['production_ready']is p['full_paper_complete']is False


def test_original_eval_grid_and_whole_jitter_footprints_are_disjoint(tmp_path):
    m=api();splits=source(tmp_path,signals=False)['frame_split_id'];store=LegacyStore(tmp_path)
    p=m.build_emgformer_partitions(store);q=m.build_emgformer_partitions(store)
    assert p==q and p['settings']['repeat_by_role']==dict(train=1,val=1,test=1)
    occupied={r:np.zeros(len(splits),bool)for r in ('train','val','test')}
    want=[s for s in range(0,len(splits)-W+1,W)if splits[s+W//2]!=0]
    assert [w['start']for w in p['windows']if w['role']=='test']==want
    for w in p['windows']:
        assert w['center']==w['start']+W//2 and w['end']-w['start']==W
        if w['role']=='train':
            assert w['start']%1200==0 and w['jitter_upper']==min(1200,len(splits)-w['end'])
            assert w['footprint_end']==w['end']+max(0,w['jitter_upper']-1)
            assert np.all(splits[w['start']:w['footprint_end']]==0)
        else:assert w['jitter_upper']==0
        occupied[w['role']][w['footprint_start']:w['footprint_end']]=True
    assert all(np.any(a)for a in occupied.values())
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not np.any(occupied[a]&occupied[b])


@pytest.mark.parametrize('fraction',[True,-.1,1,float('nan')])
def test_invalid_validation_fraction_rejects_before_arrays(tmp_path,fraction):
    m=api();source(tmp_path,signals=False)
    with pytest.raises(ValueError):m.build_emgformer_partitions(LegacyStore(tmp_path),fraction)


@pytest.mark.parametrize('size',['small','middle','large'])
@pytest.mark.parametrize('role',['train','val','test'])
@pytest.mark.parametrize('hand',['left','right'])
def test_actual_native_reader_values_shapes_masks_and_rng(tmp_path,size,role,hand):
    m=api();torch=pytest.importorskip('torch');source(tmp_path,optional_labels=True)
    c=cfg(size);ds=dataset(tmp_path,role,size=size)
    from hydra.utils import instantiate
    from egoemg.transforms import Compose
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from egoemg_repro.emgformer_literal import NORM_ASSET
    reference=EgoEmgMemmapDataset(tmp_path,W,stride=1200 if role=='train' else W if role=='test' else 1,
        allowed_splits=['user','gesture','both']if role=='test'else ['train'],
        target_hand=hand,modalities=['emg','joint_angles','labels'],emg_field_preference='filtered_paper',
        emg_layout='target_hand',norm_mode='per-dataset',norm_stats_path=str(UPSTREAM/NORM_ASSET),
        dataset_name='egoemg',jitter=role=='train',center_target_only=role!='train',
        transform=Compose([instantiate(t)for t in c.transforms[role]]))
    w=ds.windows[0];i=int(np.flatnonzero(reference._block_start==w['start'])[0])
    np.random.seed(42);torch.manual_seed(42);want=reference[i];np_after=np.random.get_state();torch_after=torch.get_rng_state()
    np.random.seed(42);torch.manual_seed(42);got=ds[0 if hand=='left'else len(ds.windows)];after=np.random.get_state()
    for k in ('emg','emg_channel_mask','joint_angles','label_valid_mask'):
        assert got[k].dtype==want[k].dtype and got[k].shape==want[k].shape
        assert got[k].tobytes()==np.ascontiguousarray(want[k]).tobytes(),k
    target_length=W if role=='train'else 1
    assert got['emg'].shape==(8,W) and got['joint_angles'].shape==(22,target_length)
    assert got['label_valid_mask'].shape==(target_length,)
    assert got['dataset_name']==want['dataset_name']=='egoemg' and got['target_hand']==hand
    assert np.array_equal(after[1],np_after[1]) and after[2:]==np_after[2:] and torch.equal(torch_after,torch.get_rng_state())
    assert got['role']==role and got['split']==w['split'] and got['emgformer_repeat_index']==0
    assert got['jitter_offset']==got['start']-w['start'] and len(ds)==2*len(ds.windows)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('leftover',[0,1,17,1200])
def test_tail_jitter_uses_exact_exclusive_native_range(tmp_path,leftover,monkeypatch):
    m=api();pytest.importorskip('torch');source(tmp_path,n=W+leftover)
    p=m.build_emgformer_partitions(LegacyStore(tmp_path),0);ds=dataset(tmp_path,'train',partitions=p)
    monkeypatch.setattr(np.random,'randint',lambda low,high:high-1)
    got=ds[0]
    assert got['jitter_offset']==max(0,min(1200,leftover)-1)
    assert got['end']==p['windows'][0]['footprint_end']


@pytest.mark.parametrize('role',['train','val','test'])
def test_native_full_loader_policy_and_cpu_partial_collation(tmp_path,role):
    m=api();pytest.importorskip('torch');source(tmp_path,webcam=True);ds=dataset(tmp_path,role)
    loader=m.native_emgformer_loader(ds)
    assert loader.batch_size==500 and loader.num_workers==6 and loader.prefetch_factor==1
    assert loader.persistent_workers and not loader.pin_memory and not loader.drop_last and loader.generator is None
    expected='RandomSampler'if role=='train'else 'SequentialSampler'
    assert type(loader.sampler).__name__==expected
    diagnostic=m.native_emgformer_loader(ds,cpu_engineering=True)
    batch=next(iter(diagnostic))
    assert diagnostic.num_workers==0 and batch['emg'].shape[1:]==(8,W)
    assert batch['joint_angles'].shape[1:]==(22,W if role=='train'else 1)
    assert batch['emgformer_sample_index'].numel()==len(ds)<500


@pytest.mark.parametrize('damage',['partition','normalization','alias','missing_emg','missing_optional','bad_dtype','merged','role'])
def test_invalid_inputs_are_rejected_without_repair(tmp_path,damage):
    m=api();pytest.importorskip('torch');source(tmp_path,optional_labels=True)
    p=m.build_emgformer_partitions(LegacyStore(tmp_path));norm=copy.deepcopy(m.emgformer_data_contract(UPSTREAM,cfg())['normalization'])
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    if damage=='partition':p['windows'][0]['center']+=1
    elif damage=='normalization':norm['hands']['right']['std'][0]=1.
    elif damage=='alias':manifest['fields']['emg_left_filtered_paper']['filename']='other.dat'
    elif damage=='missing_emg':(tmp_path/'emg_left_filtered.dat').unlink()
    elif damage=='missing_optional':(tmp_path/'mocap_right_wrist_angles_valid.dat').unlink()
    elif damage=='bad_dtype':manifest['fields']['generated_joint_angles_right']['dtype']='int32'
    elif damage=='merged':manifest['fields']['dataset_source_id']={'dtype':'int32','shape':[24*W]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        dataset(tmp_path,'invented'if damage=='role'else 'train',partitions=p,normalization=norm)


def test_masked_nonfinite_values_stay_visible_for_full_admission(tmp_path):
    m=api();pytest.importorskip('torch');source(tmp_path,n=W)
    a=np.memmap(tmp_path/'generated_joint_angles_left.dat',dtype='float32',mode='r+',shape=(W,20));a[1,5]=np.nan;a.flush();del a
    p=m.build_emgformer_partitions(LegacyStore(tmp_path),0);got=dataset(tmp_path,'train',partitions=p)[0]
    assert not got['label_valid_mask'][1] and np.isnan(got['joint_angles'][5,1])


@pytest.mark.parametrize('role',['train','val','test'])
def test_empty_requested_role_is_rejected(tmp_path,role):
    m=api();pytest.importorskip('torch');source(tmp_path,n=W)
    if role=='train':np.ones(W,np.int8).tofile(tmp_path/'frame_split_id.dat')
    with pytest.raises(ValueError,match='role|nonempty'):dataset(tmp_path,role)


@pytest.mark.parametrize('role',['train','val','test'])
def test_original_optional_label_defaults_and_center_key_omissions(tmp_path,role):
    m=api();pytest.importorskip('torch');source(tmp_path);got=dataset(tmp_path,role)[0]
    if role=='train':
        assert np.all(got['label_gesture_class']==-1) and not np.any(got['label_gesture_active'])
    else:assert 'label_gesture_class'not in got and 'label_gesture_active'not in got


@pytest.mark.parametrize('role',['train','val','test'])
def test_native_full_and_center_wrist_rounding_ignores_stored_validity(tmp_path,role):
    m=api();pytest.importorskip('torch');source(tmp_path);ds=dataset(tmp_path,role);w=ds.windows[0]
    np.random.seed(42);offset=int(np.random.randint(0,w['jitter_upper']))if w['jitter_upper']else 0
    center=w['center']+offset
    valid=np.memmap(tmp_path/'mocap_left_wrist_angles_valid.dat',dtype=bool,mode='r+',shape=(24*W,))
    valid[center]=False;valid.flush();del valid
    np.random.seed(42);got=ds[0]
    pitch=np.memmap(tmp_path/'mocap_left_wrist_pitch.dat',dtype='float32',mode='r',shape=(24*W,))
    if role=='train':
        assert got['joint_angles'][20,W//2]==np.deg2rad(np.asarray(pitch[center],dtype=np.float32))
        assert got['joint_angles'][20,W//2]!=0 and got['label_valid_mask'][W//2]
    else:
        assert got['joint_angles'][20,0]==np.float32(np.deg2rad(float(pitch[center])))
        assert got['joint_angles'][20,0]!=0 and got['label_valid_mask'][0]
