"""Original coupled fusion reads over explicitly adapted legacy input files.

The public merged-source recipe is preserved. Its unavailable normalization
namespace in the declared asset is an explicit, source-bound data substitution;
this reader grants no full-content, model, GPU or scientific admission.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .classic_emg import _activate,_approved_upstream,_file_hash,_hash
from .classic_emg_data import _safe_path
from .data import LegacyStore,union_intervals
from .fusion_literal import _validated,fusion_protocol
from .partitions import build_partitions
from .vision_literal_data import _PointView,_crop_path,_schema as _point_schema

W=7790
STRIDE=400
POSITIONS=[10,12,0,1,2,4,5,6]
NORM_ASSET='assets/per_dataset_norm_stats_unified.json'
NORM_SHA='6a0d029147fd3f4befef5c33f82c9044606de5a8b335b15ffa434ad3767701f5'


def fusion_normalization(upstream):
    """Execute original scalar selection and disclose the canonical missing key."""
    root=_approved_upstream(upstream);_activate(root)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    if _file_hash(root/NORM_ASSET)!=NORM_SHA:raise ValueError('pinned fusion normalization asset differs')
    kwargs=dict(norm_mode='per-dataset',dataset_name='egoemg_unified',emg_field_preference='filtered_paper',
        emg_layout='emg2pose_interpolate16',target_hand='left')
    declared=SimpleNamespace(**kwargs,norm_stats_path=str(root/'assets/per_dataset_norm_stats.json'))
    try:EgoEmgMemmapDataset._load_norm_stats(declared)
    except KeyError:pass
    else:raise ValueError('canonical normalization namespace behavior changed')
    actual=SimpleNamespace(**kwargs,norm_stats_path=str(root/NORM_ASSET))
    EgoEmgMemmapDataset._load_norm_stats(actual)
    result=dict(format='fusion_fixed_normalization_v1',scientific_eligible=False,production_ready=False,
        full_paper_complete=False,asset='upstream://'+NORM_ASSET,asset_sha256=NORM_SHA,
        namespace='egoemg_unified__filtered_paper',mean=actual._emg_mean,std=actual._emg_std,epsilon=1e-6,
        canonical_declared_asset='upstream://assets/per_dataset_norm_stats.json',
        canonical_declared_asset_sha256=_file_hash(root/'assets/per_dataset_norm_stats.json'),
        canonical_declared_namespace_available=False,canonical_error_type='KeyError',
        substitution='explicit same-revision official unified-stat asset; canonical model config is unchanged',
        requested_field_preference='filtered_paper',physical_fields={hand:f'emg_{hand}_filtered' for hand in ('left','right')},
        sample_dataset_name='egoemg',estimated_from_project_train_rows=False,
        order='float32 T,8; original zero-fill16; original fixed scalar; original role transforms; transpose16,T',
        sources_sha256={name:_file_hash(root/name) for name in ('egoemg/datasets/egoemg_memmap_dataset.py',
            'egoemg/datasets/layout_utils.py','egoemg/transforms.py')},
        limitation='legacy-only rows; public pooled-stat sampling is not the independent retained training rows or proof of merged-source/PDF equivalence')
    result['sha256']=_hash(result);return result


def build_fusion_partitions(store,val_fraction=.12):
    """Exclude unsafe anchors, retaining the entire original random jitter range."""
    if type(val_fraction) not in (int,float) or not math.isfinite(val_fraction) or not 0<=val_fraction<1:
        raise ValueError('finite fusion validation fraction in [0,1) required')
    result=build_partitions(store,window_length=W,train_stride=STRIDE,eval_stride=W,val_fraction=val_fraction)
    forbidden=union_intervals([(v['start'],v['end']) for v in result['validation_intervals']]+
        [(w['start'],w['end']) for w in result['windows'] if w['role']=='test'])
    ends=[end for _,end in forbidden];splits=store.field('frame_split_id')
    train_id=next(i for i,name in store.split_labels.items() if name=='train');windows=[];discarded=0
    for w in result['windows']:
        leftover=store.episodes[w['episode']]['end']-w['end']
        upper=min(STRIDE,leftover) if w['role']=='train' else 0
        maximum=max(0,upper-1);end=w['end']+maximum;i=bisect_right(ends,w['start'])
        if w['role']=='train' and ((i<len(forbidden) and forbidden[i][0]<end) or
            not np.all(splits[w['end']:end]==train_id)):
            discarded+=1;continue
        windows.append(dict(w,jitter_upper=upper,jitter_max_offset=maximum,footprint_start=w['start'],footprint_end=end))
    result['windows']=windows;result['discarded']['fusion_jitter_unsafe_anchor']=discarded
    result['settings'].update(jitter=True,repeat_by_role={r:2 for r in ('train','val','test')},
        index_order='repeat(left-role-windows then right-role-windows)')
    for role in ('train','val','test'):
        chosen=[w for w in windows if w['role']==role];spans=union_intervals((w['footprint_start'],w['footprint_end']) for w in chosen)
        result['coverage'][role]=dict(windows=len(chosen),hand_samples=2*len(chosen),consumed_hand_samples=4*len(chosen),
            unique_rows=sum(b-a for a,b in spans),episodes=len({w['episode'] for w in chosen}),
            by_split=dict(Counter(w['split'] for w in chosen)))
    provenance=result['provenance'];del provenance['partition_sha256']
    provenance.update(protocol='original-fusion-on-jitter-safe-legacy-v1',
        available_corpus='released legacy EgoEMG only; not physically merged EgoEMG/ShowEE/Incre',
        normalization='separately bound explicit official unified-stat substitution; not estimated on retained train rows',
        mask_policy='original generated center label mask broadcast to22 angles',
        jitter='original randint(0,min(400,episode_end-anchor_end)); unsafe anchors excluded prospectively')
    provenance['partition_sha256']=_hash(result);return result


class _FusionPointView(_PointView):
    def __init__(self,store,crops,hand,frame_field,native,normalization,transform):
        super().__init__(store,crops,hand,frame_field,native)
        self.emg_field_preference='filtered_paper';self.channel_interpolate=False
        self._emg2pose_channel_indices=list(POSITIONS);self.norm_mode='per-dataset';self.transform=transform
        self._emg_mean=normalization['mean'];self._emg_std=normalization['std']
        self._frame_memmaps[f'emg_{hand}_filtered_paper']=store.field(f'emg_{hand}_filtered')

    def _convert_to_emg2pose_layout(self,emg,*,interpolate_missing):
        return self.native._convert_to_emg2pose_layout(self,emg,interpolate_missing=interpolate_missing)


def _emg_schema(store):
    for hand in ('left','right'):
        name=f'emg_{hand}_filtered';spec=store.fields.get(name)
        if not spec or spec['shape']!=[store.total_rows,8] or np.dtype(spec['dtype'])!=np.dtype('float32'):
            raise ValueError('fusion requires complete float32 eight-channel legacy EMG: '+name)
        path=_safe_path(store.root,spec.get('filename',name+'.dat'))
        if path.stat().st_size!=store.total_rows*8*4:raise ValueError('fusion EMG payload byte size differs')


class FusionLiteralDataset:
    """Original joint arrays, explicit native repeat ordering and safe jitter."""
    def __init__(self,data,partitions,crops,*,upstream,config,normalization,role):
        if role not in ('train','val','test'):raise ValueError('fusion role must be train/val/test')
        root,_=_validated(upstream,config);self.protocol=fusion_protocol(root,config)
        expected_norm=fusion_normalization(root)
        if not isinstance(normalization,dict) or _hash(normalization)!=_hash(expected_norm):
            raise ValueError('fusion normalization must match the explicit pinned source binding')
        self.normalization=copy.deepcopy(expected_norm);self.store=LegacyStore(data)
        self.frame_mapping_field=_point_schema(self.store);_emg_schema(self.store)
        expected=build_fusion_partitions(self.store,partitions['settings']['val_fraction'])
        if _hash(partitions)!=_hash(expected):raise ValueError('fusion partition/jitter footprint differs')
        self.partitions=copy.deepcopy(expected);self.windows=[w for w in expected['windows'] if w['role']==role]
        self.role=role;self.repeat=self.protocol['repeat_by_role'][role]
        crops=Path(crops).resolve()
        if not crops.is_dir():raise FileNotFoundError('episode crop root unavailable')
        for episode in {w['episode'] for w in self.windows}:_crop_path(crops,self.store.episodes[episode]['id'])
        from hydra.utils import instantiate
        from egoemg.transforms import Compose
        from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
        self.native=EgoEmgMemmapDataset
        self.views=[_FusionPointView(self.store,crops,hand,self.frame_mapping_field,self.native,
            self.normalization,Compose([instantiate(t) for t in config.transforms[role]])) for hand in ('left','right')]

    def __len__(self):return 2*len(self.windows)*self.repeat

    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError(index)
        unique=index%(2*len(self.windows));repeat=index//(2*len(self.windows))
        hand=unique//len(self.windows);window_index=unique%len(self.windows);w=self.windows[window_index]
        offset=int(np.random.randint(0,w['jitter_upper'])) if w['jitter_upper']>0 else 0
        start=w['start']+offset;center=start+W//2
        value=self.native._getitem_center_supervised(self.views[hand],w['episode'],start,start+W,center)
        value.update(episode=w['episode'],role=self.role,split=w['split'],hand=('left','right')[hand],
            subject=self.store.episodes[w['episode']]['subject'],start=start,end=start+W,center=center,
            anchor_start=w['start'],anchor_end=w['end'],anchor_center=w['center'],jitter_offset=offset,
            fusion_window_index=window_index,fusion_sample_index=index,fusion_unique_sample_index=unique,
            fusion_repeat_index=repeat,frame_mapping_field=self.frame_mapping_field)
        return value
