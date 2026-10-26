"""Original vision point reads on explicit jitter-safe, independent legacy roles.

Only IO/schema/index adaptation lives here. Actual RGB normalization, wrist
rounding, label validity and missing-key behavior execute the pinned original
point-reader methods. This is not the original physically merged corpus.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import copy
import math
import os
from pathlib import Path

import numpy as np

from . import crops as crop_io
from .classic_emg_data import _safe_path
from .data import LegacyStore,union_intervals
from .partitions import build_partitions
from .vision_literal import _activate,_hash,_validated,vision_protocol


W=7790
STRIDE=780


def build_vision_partitions(store,val_fraction=.12):
    """Keep native jitter's full range; exclude unsafe anchors rather than clipping it."""
    if (type(val_fraction) not in (int,float) or not math.isfinite(val_fraction) or not 0<=val_fraction<1):
        raise ValueError('finite vision validation fraction in [0,1) required')
    result=build_partitions(store,window_length=W,train_stride=STRIDE,eval_stride=W,val_fraction=val_fraction)
    forbidden=union_intervals([(v['start'],v['end']) for v in result['validation_intervals']]+
        [(w['start'],w['end']) for w in result['windows'] if w['role']=='test'])
    ends=[end for _,end in forbidden];windows=[];discarded=0
    train_id=next(i for i,name in store.split_labels.items() if name=='train');splits=store.field('frame_split_id')
    for w in result['windows']:
        leftover=store.episodes[w['episode']]['end']-w['end']
        upper=min(STRIDE,leftover) if w['role']=='train' else 0
        maximum=max(0,upper-1);end=w['end']+maximum
        i=bisect_right(ends,w['start'])
        if w['role']=='train' and ((i<len(forbidden) and forbidden[i][0]<end) or
            not np.all(splits[w['end']:end]==train_id)):
            discarded+=1;continue
        windows.append(dict(w,jitter_upper=upper,jitter_max_offset=maximum,footprint_start=w['start'],footprint_end=end))
    result['windows']=windows;result['discarded']['vision_jitter_unsafe_anchor']=discarded
    result['settings'].update(jitter=True,repeat_by_role={role:2 for role in ('train','val','test')},
        index_order='repeat(left-role-windows then right-role-windows)')
    for role in ('train','val','test'):
        chosen=[w for w in windows if w['role']==role]
        intervals=union_intervals((w['footprint_start'],w['footprint_end']) for w in chosen)
        result['coverage'][role]=dict(windows=len(chosen),hand_samples=2*len(chosen),consumed_hand_samples=4*len(chosen),
            unique_rows=sum(b-a for a,b in intervals),episodes=len({w['episode'] for w in chosen}),
            by_split=dict(Counter(w['split'] for w in chosen)))
    provenance=result['provenance'];del provenance['partition_sha256']
    provenance.update(protocol='original-vision-on-jitter-safe-legacy-v1',normalization='original RGB ImageNet255 mean/std; no EMG read',
        mask_policy='original generated center label mask broadcast to22 angles',
        available_corpus='released legacy EgoEMG only; not declared EgoEMG/ShowEE/Incre merged source',
        jitter='original randint(0,min(stride,episode_end-anchor_end)); unsafe anchors excluded prospectively')
    provenance['partition_sha256']=_hash(result)
    return result


def _crop_path(root,episode):
    if type(episode) is not str or Path(episode).name!=episode or episode in ('.','..'):
        raise ValueError('single episode component required')
    path=root/(episode+'.lmdb')
    if not path.exists():path=root/(episode+'.data.mdb')
    if not path.exists():raise FileNotFoundError('requested complete episode crop database unavailable')
    if path.is_symlink():raise ValueError('crop database cannot be a symlink')
    return path


class _PointView:
    """The small IO context consumed by the unchanged original point methods."""
    def __init__(self,store,crops,hand,frame_field,native):
        self.store=store;self.crops=crops;self.target_hand=hand;self.native=native
        self._frame_memmaps={name:store.field(name) for name in store.fields if name in {
            'generated_joint_angles_left','generated_joint_angles_right','generated_label_valid',
            'mocap_left_wrist_pitch','mocap_left_wrist_yaw','mocap_right_wrist_pitch','mocap_right_wrist_yaw'}}
        self._frame_memmaps['image_head_frame_index']=store.field(frame_field)
        self._episode_id=[episode['id'] for episode in store.episodes]
        self.vision_patch_size=256;self.emg_layout='emg2pose_interpolate16'
        self.vision_mean=255.*np.array([.485,.456,.406],np.float32)
        self.vision_std=255.*np.array([.229,.224,.225],np.float32)

    def _resolve_dataset_name(self,center):return 'egoemg'

    def _get_pec_txn(self,episode):
        # Share the established process-local readonly cache with old consumers;
        # py-lmdb forbids duplicate Environment objects for the same path.
        import lmdb
        path=_crop_path(self.crops,episode);key=(os.getpid(),str(path))
        if key not in crop_io._READERS:
            while len(crop_io._READERS)>=8:
                _,environment=crop_io._READERS.popitem(last=False);environment.close()
            crop_io._READERS[key]=lmdb.open(str(path),subdir=path.is_dir(),readonly=True,
                lock=False,readahead=False,create=False)
        crop_io._READERS.move_to_end(key)
        return crop_io._READERS[key].begin()

    def _read_episode_crops(self,episode,indices,hand_code):
        return self.native._read_episode_crops(self,episode,indices,hand_code)


def _schema(store):
    n=store.total_rows
    if 'dataset_source_id' in store.fields:raise ValueError('merged source needs its own provenance-aware adapter')
    expected={'generated_label_valid':((n,2),np.dtype(bool))}
    for hand in ('left','right'):
        expected[f'generated_joint_angles_{hand}']=((n,20),np.dtype('float32'))
        for axis in ('pitch','yaw'):expected[f'mocap_{hand}_wrist_{axis}']=((n,),np.dtype('float32'))
    frame=next((name for name in ('image_head_frame_index','image_webcam_frame_index') if name in store.fields),None)
    if frame is None:raise ValueError('original head frame mapping or explicit legacy webcam alias required')
    expected[frame]=((n,),None)
    for name,(shape,dtype) in expected.items():
        spec=store.fields[name]
        if tuple(spec['shape'])!=shape or (dtype is not None and np.dtype(spec['dtype'])!=dtype):
            raise ValueError('vision point field shape/dtype differs: '+name)
        path=_safe_path(store.root,spec.get('filename',name+'.dat'))
        if path.stat().st_size!=int(np.prod(shape))*np.dtype(spec['dtype']).itemsize:
            raise ValueError('vision point field byte size differs')
    if np.dtype(store.fields[frame]['dtype']).kind not in 'iu':raise ValueError('integer source frame indices required')
    return frame


class VisionLiteralDataset:
    """All role samples, ordered left/right then repeat2, with the original jitter draw."""
    def __init__(self,data,partitions,crops,*,upstream,config,role):
        if role not in ('train','val','test'):raise ValueError('vision role must be train/val/test')
        root,_,_=_validated(upstream,config);self.protocol=vision_protocol(upstream,config)
        self.store=LegacyStore(data)
        expected=build_vision_partitions(self.store,val_fraction=partitions['settings']['val_fraction'])
        if _hash(partitions)!=_hash(expected):raise ValueError('vision partition or jitter footprint differs')
        self.partitions=copy.deepcopy(expected);self.windows=[w for w in self.partitions['windows'] if w['role']==role]
        self.role=role;self.repeat=self.protocol['repeat_by_role'][role];self.frame_mapping_field=_schema(self.store)
        crops=Path(crops).resolve()
        if not crops.is_dir():raise FileNotFoundError('episode crop root unavailable')
        for episode in {w['episode'] for w in self.windows}:_crop_path(crops,self.store.episodes[episode]['id'])
        _activate(root)
        from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
        self.native=EgoEmgMemmapDataset
        self.views=[_PointView(self.store,crops,hand,self.frame_mapping_field,self.native) for hand in ('left','right')]

    def __len__(self):return 2*len(self.windows)*self.repeat

    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError(index)
        unique=index%(2*len(self.windows));repeat=index//(2*len(self.windows))
        hand=unique//len(self.windows);window_index=unique%len(self.windows);w=self.windows[window_index]
        offset=int(np.random.randint(0,w['jitter_upper'])) if w['jitter_upper']>0 else 0
        start=w['start']+offset;center=start+W//2
        value=self.native._getitem_vision_only_crops(self.views[hand],w['episode'],center)
        value.update(episode=w['episode'],role=self.role,split=w['split'],hand=('left','right')[hand],
            subject=self.store.episodes[w['episode']]['subject'],start=start,end=start+W,center=center,
            anchor_start=w['start'],anchor_end=w['end'],anchor_center=w['center'],jitter_offset=offset,
            vision_window_index=window_index,vision_sample_index=index,vision_unique_sample_index=unique,
            vision_repeat_index=repeat,frame_mapping_field=self.frame_mapping_field)
        return value
