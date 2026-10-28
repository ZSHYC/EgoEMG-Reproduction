"""Original full-window EMG inputs excluding all fixed future fusion holdouts."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import copy
import math
import os

import numpy as np

from .classic_emg import _activate,_hash
from .classic_emg_data import _safe_path
from .data import LegacyStore,union_intervals
from .emg_initializer_literal import _validated,emg_initializer_protocol,emg_initializer_normalization,NORM_ASSET
from .fusion_literal_data import build_fusion_partitions

W=12000
STRIDE=1200
ROLES=('train','val')


def build_initializer_partitions(store,val_fraction=.12):
    if type(val_fraction) not in (int,float) or not math.isfinite(val_fraction) or not 0<=val_fraction<1:
        raise ValueError('finite validation fraction in [0,1) required')
    fusion=build_fusion_partitions(store,val_fraction)
    forbidden=union_intervals([(v['start'],v['end']) for v in fusion['validation_intervals']]+
        [(w['footprint_start'],w['footprint_end']) for w in fusion['windows'] if w['role']=='test'])
    ends=[b for _,b in forbidden];splits=store.field('frame_split_id')
    train_id=next(k for k,v in store.split_labels.items() if v=='train');windows=[];discarded=Counter()
    def record(episode,start,role,upper):
        return dict(episode=episode,start=int(start),end=int(start+W),center=int(start+W//2),role=role,split='train',
            jitter_upper=int(upper),jitter_max_offset=max(0,int(upper)-1),footprint_start=int(start),footprint_end=int(start+W+max(0,upper-1)))
    for ep in store.episodes:
        for start in range(ep['start'],ep['end']-W+1,STRIDE):
            end=start+W;upper=min(STRIDE,ep['end']-end);hi=end+max(0,upper-1)
            if int(splits[start+W//2])!=train_id:discarded['native_center_not_train']+=1;continue
            i=bisect_right(ends,start)
            if i<len(forbidden) and forbidden[i][0]<hi:discarded['overlaps_fusion_holdout']+=1;continue
            if not np.all(splits[start:hi]==train_id):discarded['full_jitter_footprint_not_all_train']+=1;continue
            windows.append(record(ep['index'],start,'train',upper))
    for interval in fusion['validation_intervals']:
        for start in range(interval['start'],interval['end']-W+1,W):
            if not np.all(splits[start:start+W]==train_id):raise ValueError('validation interval is not wholly official train')
            windows.append(record(interval['episode'],start,'val',0))
    windows.sort(key=lambda w:(w['episode'],w['start'],w['role']))
    coverage={}
    for role in ROLES:
        chosen=[w for w in windows if w['role']==role]
        coverage[role]=dict(windows=len(chosen),hand_samples=2*len(chosen),consumed_hand_samples=2*len(chosen),
            unique_possible_rows=sum(b-a for a,b in union_intervals((w['footprint_start'],w['footprint_end']) for w in chosen)),
            episodes=len({w['episode'] for w in chosen}))
    result=dict(format='fixed_fusion_holdout_emg_initializer_partitions_v1',windows=windows,episodes=copy.deepcopy(store.episodes),
        settings=dict(window_length=W,train_stride=STRIDE,eval_stride=W,val_fraction=val_fraction,jitter=True,repeat_by_role={r:1 for r in ROLES}),
        validation_intervals=copy.deepcopy(fusion['validation_intervals']),forbidden_fusion_footprints=forbidden,
        coverage=coverage,discarded=dict(discarded),provenance=dict(fusion_partition_content_sha256=_hash(fusion),
            manifest_sha256=fusion['provenance']['manifest_sha256'],metadata_sha256=fusion['provenance']['metadata_sha256'],
            frame_split_sha256=fusion['provenance']['frame_split_sha256'],available_corpus='released legacy EgoEMG only; Incre source absent',
            normalization='original published per-hand/per-channel vectors; not project train-only statistics',
            limitations=copy.deepcopy(fusion['provenance']['limitations']),initializer_test_role=False))
    result['provenance']['partition_sha256']=_hash(result);return result


def _schema(store):
    if 'dataset_source_id' in store.fields:raise ValueError('merged source requires its own adapter')
    n=store.total_rows;expected={'frame_split_id':((n,),None),'generated_label_valid':((n,2),np.dtype(bool))}
    for hand in ('left','right'):
        expected[f'emg_{hand}_filtered']=((n,8),np.dtype('float32'))
        expected[f'generated_joint_angles_{hand}']=((n,20),np.dtype('float32'))
        for axis in ('pitch','yaw'):expected[f'mocap_{hand}_wrist_{axis}']=((n,),np.dtype('float32'))
        name=f'mocap_{hand}_wrist_angles_valid'
        if name in store.fields:expected[name]=((n,),np.dtype(bool))
        alias=f'emg_{hand}_filtered_paper'
        if alias in store.fields and store.fields[alias]!=store.fields.get(f'emg_{hand}_filtered'):
            raise ValueError('conflicting EMG physical field alias')
    for name,dtype in (('label_gesture_class','int32'),('label_gesture_active','bool')):
        if name in store.fields:expected[name]=((n,),np.dtype(dtype))
    paths={}
    for name,(shape,dtype) in expected.items():
        spec=store.fields.get(name)
        if not spec or tuple(spec['shape'])!=shape or (dtype is not None and np.dtype(spec['dtype'])!=dtype):
            raise ValueError('initializer field shape/dtype differs: '+name)
        if name=='frame_split_id' and np.dtype(spec['dtype']).kind not in 'iu':raise ValueError('integer split field required')
        path=_safe_path(store.root,spec.get('filename',name+'.dat'))
        if path.stat().st_size!=int(np.prod(shape))*np.dtype(spec['dtype']).itemsize:raise ValueError('initializer field byte size differs: '+name)
        paths[name]=path
    return paths


def _view(store,windows,hand,role,root,cfg):
    # Supply IO and approved indexes to an actual native class instance. Its
    # full __getitem__, transforms, norm helper and cache methods are unedited.
    from hydra.utils import instantiate
    from egoemg.transforms import Compose
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from egoemg.video_io import DecordVideoReaderCache
    view=object.__new__(EgoEmgMemmapDataset);view.memmap_dir=store.root
    view.modalities=list(cfg.dataset[role][0].modalities);view.fields=None;view.target_hand=hand
    view.emg_field_preference='filtered_paper';view.emg_layout='target_hand';view.channel_interpolate=False
    view.skip_emg_loading=False;view._vision_enabled=False;view._cached_vit_enabled=False;view._center_target_only=False
    view._preloaded_features=None;view.video_streams=None;view._episode_memmaps={};view._frame_split_available=True
    view._manifest=copy.deepcopy(store.manifest)
    for h in ('left','right'):view._manifest['fields'][f'emg_{h}_filtered_paper']=copy.deepcopy(store.fields[f'emg_{h}_filtered'])
    wanted,episode_wanted=view._resolve_requested_fields()
    if episode_wanted:raise ValueError('unexpected initializer episode fields')
    wanted=view._prune_hand_fields(wanted,target_hand=hand,emg_field_preference='filtered_paper',emg_layout='target_hand')
    wanted.add('frame_split_id')
    wanted.update(name for name in (f'mocap_{hand}_wrist_pitch',f'mocap_{hand}_wrist_yaw',f'mocap_{hand}_wrist_angles_valid') if name in store.fields)
    view._frame_memmaps={name:store.field(name.removesuffix('_paper') if name.endswith('_filtered_paper') else name)
        for name in view._manifest['fields'] if name in wanted}
    md=store.metadata
    for name in ('episode_id','episode_subject','episode_subject_id','episode_source_parquet','episode_zed_video_path',
        'episode_head_video_path','episode_start_idx','episode_end_idx','episode_length','episode_beta_idx'):
        if name not in md:raise ValueError('required initializer metadata field missing: '+name)
        setattr(view,'_'+name,copy.deepcopy(md[name]))
    for name in ('episode_wrist_left_video_path','episode_wrist_right_video_path'):
        setattr(view,'_'+name,copy.deepcopy(md.get(name,['']*len(store.episodes))))
    view._splits=list(md['splits_split']);view._source_id_to_name=None;view.dataset_name='egoemg'
    if view._splits!=[store.split_labels[i] for i in range(len(store.split_labels))]:raise ValueError('split metadata binding differs')
    view._runtime_pid=os.getpid();view._intrinsics_cache={};view._video_reader_cache=DecordVideoReaderCache()
    view._mano_npy_pose={};view._mano_npy_beta={};view._mano_npy_trans={};view._mano_npy_world_R={};view._mano_npy_world_t={}
    view.window_length=W;view.stride=STRIDE if role=='train' else W;view.jitter=role=='train'
    view.norm_mode='per-dataset';view.norm_stats_path=str(root/NORM_ASSET);view._load_norm_stats()
    view.transform=Compose([instantiate(t) for t in cfg.transforms[role]])
    view._block_episode_idx=np.array([w['episode'] for w in windows],np.int32)
    view._block_start=np.array([w['start'] for w in windows],np.int64);view._block_end=np.array([w['end'] for w in windows],np.int64)
    view._block_cumsum=np.arange(len(windows)+1,dtype=np.int64)
    train_id=next(i for i,name in store.split_labels.items() if name=='train')
    view._window_split_id=np.full(len(windows),train_id,np.int32)
    return view


class EmgInitializerDataset:
    """Left windows followed by right windows, repeat1; native full trajectories."""
    def __init__(self,data,partitions,*,upstream,config,normalization,role):
        if role not in ROLES:raise ValueError('initializer role must be train or val; no test role')
        root,_=_validated(upstream,config);self.protocol=emg_initializer_protocol(root,config)
        expected_norm=emg_initializer_normalization(root)
        if _hash(normalization)!=_hash(expected_norm):raise ValueError('initializer normalization binding differs')
        self.normalization=copy.deepcopy(expected_norm);self.store=LegacyStore(data);_schema(self.store)
        expected=build_initializer_partitions(self.store,partitions['settings']['val_fraction'])
        if _hash(partitions)!=_hash(expected):raise ValueError('initializer partition binding differs')
        self.partitions=copy.deepcopy(expected);self.role=role;self.repeat=1
        self.windows=[w for w in expected['windows'] if w['role']==role]
        if not self.windows:raise ValueError('initializer role must be nonempty')
        _activate(root);self.views=[_view(self.store,self.windows,h,role,root,config) for h in ('left','right')]

    def __len__(self):return 2*len(self.windows)

    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError(index)
        hand,window_index=divmod(index,len(self.windows));w=self.windows[window_index]
        result=self.views[hand][window_index];start=int(result['window_start_idx']);end=int(result['window_end_idx'])
        result.update(role=self.role,split='train',episode=w['episode'],start=start,end=end,center=start+W//2,
            hand=('left','right')[hand],anchor_start=w['start'],anchor_end=w['end'],anchor_center=w['center'],
            jitter_offset=start-w['start'],initializer_window_index=window_index,initializer_sample_index=index,
            initializer_unique_sample_index=index,initializer_repeat_index=0)
        return result
