"""Original RN18 fusion center inputs, with explicit legacy namespace/field IO."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .classic_emg import _activate,_file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard
from .data import LegacyStore
from .emgformer_data import build_emgformer_partitions
from .fusion_literal_data import _emg_schema
from .rn18_fusion_literal import _validated,rn18_fusion_protocol,NATIVE,NORM_ASSET
from .vision_literal_data import _PointView,_crop_path,_schema

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
NORM_SHA='d4b73febf386ec7b2028639eed8bbd7363d13cbe3fe489149915bcef071ee96a'
W=12000
STRIDE=1200
ROLES=('train','val','test')
_CROP_GENERATIONS={}


def _check_source():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 fusion reader source changed')


def rn18_fusion_normalization(upstream,config):
    _check_source()
    root,value=_validated(upstream,config)
    guard=_StatGuard([*(root/n for n in NATIVE),root/NORM_ASSET,Path(__file__)])
    if _file_hash(root/NORM_ASSET)!=NORM_SHA:raise ValueError('original RN18 fusion normalization asset differs')
    _activate(root)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    errors={};hands={}
    for hand in('left','right'):
        base=dict(norm_mode='per-dataset',norm_stats_path=str(root/NORM_ASSET),emg_field_preference='filtered_paper',
            emg_layout='target_hand',target_hand=hand)
        try:EgoEmgMemmapDataset._load_norm_stats(SimpleNamespace(**base,dataset_name='egoemg_unified'))
        except KeyError:errors[hand]='KeyError'
        else:raise ValueError('original declared RN18 fusion namespace behavior changed')
        view=SimpleNamespace(**base,dataset_name='egoemg');EgoEmgMemmapDataset._load_norm_stats(view)
        mean=np.asarray(view._emg_mean);std=np.asarray(view._emg_std)
        if (mean.shape!=(8,)or std.shape!=(8,)or mean.dtype!=np.float32 or std.dtype!=np.float32
            or not np.isfinite(mean).all()or not np.isfinite(std).all()or np.any(std<=0)):
            raise ValueError('actual per-hand RN18 fusion eight-channel vectors required')
        hands[hand]=dict(mean=mean.tolist(),std=std.tolist(),dtype='float32')
    guard.check()
    result=dict(format='original_rn18_fusion_legacy_normalization_v1',asset=NORM_ASSET,asset_sha256=NORM_SHA,
        declared_namespace='egoemg_unified',canonical_normalization_errors_by_hand=errors,normalization_namespace='egoemg',
        substitution='explicit legacy namespace in the unchanged declared author alias asset',hands=hands,epsilon=1e-6,
        requested_field=value['egoemg_emg_field_preference'],physical_fields={h:f'emg_{h}_filtered'for h in('left','right')},
        estimated_from_project_train_rows=False,scientific_eligible=False,full_paper_complete=False)
    result['sha256']=_hash(result);return result


def build_rn18_fusion_partitions(store,val_fraction=.12):
    result=build_emgformer_partitions(store,val_fraction);original=_hash(result)
    del result['provenance']['partition_sha256']
    result['settings']['center_target_only_by_role']={r:True for r in ROLES}
    result['provenance'].update(protocol='original-rn18-fusion-on-jitter-safe-legacy-v1',
        source_window_provider='closed native EMGFormer12000/full-jitter; identical anchors/footprints',
        source_window_partition_content_sha256=original,
        mask_policy='original generated center label mask broadcast to22; no wrist-valid cleanup')
    result['provenance']['partition_sha256']=_hash(result);return result


def _layout(view):
    from .fusion_literal_cohort import _transform
    if any(not isinstance(a,np.memmap)for a in view._frame_memmaps.values()):raise ValueError('native readonly memmaps required')
    return (type(view),view.native,view.store,view.target_hand,view.emg_layout,view.emg_field_preference,view.channel_interpolate,
        str(view.crops),view.vision_patch_size,tuple(view._episode_id),
        view.norm_mode,tuple((name,str(a.filename),a.mode,a.dtype.str,tuple(a.shape),tuple(a.strides),int(a.offset),bool(a.flags.writeable))
            for name,a in sorted(view._frame_memmaps.items())),
        tuple((a.dtype.str,tuple(a.shape),a.tobytes(),bool(a.flags.writeable))
            for a in(view._emg_mean,view._emg_std,view.vision_mean,view.vision_std)),
        _hash(_transform(view.transform)))


class _Rn18PointView(_PointView):
    def __init__(self,store,crops,hand,frame,native,norm,transform):
        super().__init__(store,crops,hand,frame,native)
        self.emg_layout='target_hand';self.emg_field_preference='filtered_paper';self.channel_interpolate=False
        self.norm_mode='per-dataset';self.transform=transform
        self._emg_mean=np.array(norm['hands'][hand]['mean'],np.float32);self._emg_std=np.array(norm['hands'][hand]['std'],np.float32)
        self._emg_mean.flags.writeable=False;self._emg_std.flags.writeable=False
        self._frame_memmaps[f'emg_{hand}_filtered_paper']=store.field(f'emg_{hand}_filtered')


class Rn18FusionDataset:
    """Native coupled8×12000 EMG/center RGB/22×1 target, left then right once."""
    def __init__(self,data,partitions,crops,*,upstream,config,normalization,role):
        if type(role)is not str or role not in ROLES:raise ValueError('explicit RN18 fusion role required')
        _check_source()
        root=_safe(upstream);data=_safe(data);crops=_safe(crops)
        self._guard=_StatGuard([data/'manifest.json',data/'metadata.npz',Path(__file__),ROOT/'egoemg_repro/rn18_fusion_literal.py',
            *(root/n for n in NATIVE),root/NORM_ASSET])
        root,_=_validated(root,config);self.protocol=rn18_fusion_protocol(root,config)
        self.store=LegacyStore(data);self.frame_mapping_field=_schema(self.store);_emg_schema(self.store)
        names={'frame_split_id',self.frame_mapping_field,'generated_label_valid'}
        for hand in('left','right'):
            names.update((f'emg_{hand}_filtered',f'generated_joint_angles_{hand}',f'mocap_{hand}_wrist_pitch',f'mocap_{hand}_wrist_yaw'))
        for name in names:self._guard.add(data/self.store.fields[name].get('filename',name+'.dat'))
        expected=build_rn18_fusion_partitions(self.store,partitions['settings']['val_fraction'])
        if _hash(partitions)!=_hash(expected):raise ValueError('complete RN18 fusion partition/full jitter differs')
        norm=rn18_fusion_normalization(root,config)
        if type(normalization)is not dict or _hash(normalization)!=_hash(norm):raise ValueError('actual RN18 fusion normalization differs')
        self.normalization=copy.deepcopy(norm);self.partitions=copy.deepcopy(expected)
        self.windows=[w for w in self.partitions['windows']if w['role']==role];self.role=role;self.repeat=1
        self._window_contract=tuple(tuple(sorted(w.items()))for w in self.windows)
        from . import crops as crop_io
        for episode in {w['episode']for w in self.windows}:
            database=_safe(_crop_path(crops,self.store.episodes[episode]['id']))
            payload=_safe(database/'data.mdb'if database.is_dir()else database);self._guard.add(payload)
            key=(os.getpid(),str(database));generation=self._guard.paths[payload]
            if key in crop_io._READERS and _CROP_GENERATIONS.get(key)!=generation:
                raise ValueError('RN18 fusion crop cache generation is unknown or changed')
            _CROP_GENERATIONS[key]=generation
        from hydra.utils import instantiate
        from egoemg.transforms import Compose
        from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
        self.native=EgoEmgMemmapDataset
        self.views=[_Rn18PointView(self.store,crops,h,self.frame_mapping_field,self.native,norm,
            Compose([instantiate(t)for t in config.transforms[role]]))for h in('left','right')]
        self._layouts=tuple(_layout(v)for v in self.views);self._registered_role=role
        self._registered_frame_field=self.frame_mapping_field;self.check()

    def check(self):
        _check_source()
        self._guard.check()
        if (self.role!=self._registered_role or self.repeat!=1 or len(self.windows)!=len(self._window_contract)
            or self.frame_mapping_field!=self._registered_frame_field or len(self.views)!=2
            or any(v.native is not self.native for v in self.views)or tuple(_layout(v)for v in self.views)!=self._layouts):
            raise ValueError('RN18 fusion admitted typed reader/layout/normalization changed')

    def __len__(self):return 2*len(self.windows)

    def __getitem__(self,index):
        if type(index)is not int or not 0<=index<len(self):raise IndexError(index)
        self.check();n=len(self.windows);hand,wi=divmod(index,n);w=self.windows[wi]
        if tuple(sorted(w.items()))!=self._window_contract[wi]:raise ValueError('RN18 fusion admitted native anchor changed')
        offset=int(np.random.randint(0,w['jitter_upper']))if w['jitter_upper']>0 else 0
        start=w['start']+offset;center=start+W//2
        sample=self.native._getitem_center_supervised(self.views[hand],w['episode'],start,start+W,center)
        self.check()
        sample.update(episode=w['episode'],role=self.role,split=w['split'],hand=('left','right')[hand],
            subject=self.store.episodes[w['episode']]['subject'],start=start,end=start+W,center=center,
            anchor_start=w['start'],anchor_end=w['end'],anchor_center=w['center'],jitter_offset=offset,
            rn18_fusion_sample_index=index,rn18_fusion_unique_sample_index=index,rn18_fusion_window_index=wi,
            rn18_fusion_repeat_index=0,frame_mapping_field=self.frame_mapping_field)
        return sample
