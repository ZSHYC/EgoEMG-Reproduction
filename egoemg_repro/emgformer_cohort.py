"""Complete native EMGFormer numeric admission; no GPU or fit permit."""
from __future__ import annotations

import copy
from dataclasses import dataclass,field
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard,_source_paths
from .data import LegacyStore,union_intervals
from .emg_initializer_cohort import _physical,_kind,_value,_transform
from .emg_initializer_data import _schema
from .emgformer_literal import _validated,emgformer_protocol,emgformer_normalization,NORM_ASSET
from .emgformer_data import (EmgformerDataset,build_emgformer_partitions,
    emgformer_data_contract,native_emgformer_loader,LOCAL,ROLES)

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_egoemg_emgformer_full_content_admission_v1'
MAX_CHUNK_ROWS=262144
FULL_COUNTS=dict(train=62470,val=1052,test=2792)
MANIFEST_SHA='ebcef7877444253869845b9337b7c0df2b16b0b6f55d83c58d6c3c03bae8ac09'
METADATA_SHA='cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57'


def _nine(path):
    s=path.lstat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink)


class _Guard(_StatGuard):
    def __init__(self,paths):
        self.nine={};super().__init__(paths)

    def add(self,path):
        path=_safe(path);before=_nine(path);super().add(path)
        if _nine(path)!=before:raise ValueError('source generation changed during guard registration')
        if path in self.nine and self.nine[path]!=before:raise ValueError('source stamp changed')
        self.nine[path]=before

    def check(self):
        super().check()
        if any(_nine(p)!=s for p,s in self.nine.items()):raise ValueError('nine-attribute source stamp changed')


def _sources(upstream):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported EMGFormer cohort source changed')
    names=set(LOCAL)|{'egoemg_repro/emgformer_cohort.py','egoemg_repro/emg_initializer_cohort.py'}
    paths=set(_source_paths(upstream))|{ROOT/n for n in names}|{upstream/NORM_ASSET}
    return {('upstream/'+str(p.relative_to(upstream))if p.is_relative_to(upstream)else str(p.relative_to(ROOT))):p for p in sorted(paths)}


def _stream_sha(path,guard):
    guard.check();h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    guard.check();return h.hexdigest()


def _paths(store):
    return {'metadata:manifest':_safe(store.manifest_path),'metadata:episodes':_safe(store.metadata_path),
        **{'field:'+n:_safe(p)for n,p in _schema(store).items()}}


def _reader_binding(ds,paths):
    if _paths(ds.store)!=paths:raise ValueError('native EMGFormer reader path binding changed')
    scalars=('target_hand','dataset_name','window_length','stride','jitter','norm_mode','emg_layout',
        'emg_field_preference','channel_interpolate','skip_emg_loading','center_target_only','_center_target_only',
        '_vision_enabled','_cached_vit_enabled','_frame_split_available','_preloaded_features','modalities','fields','video_streams')
    metadata=('_episode_id','_episode_subject','_episode_subject_id','_episode_source_parquet','_episode_zed_video_path',
        '_episode_head_video_path','_episode_wrist_left_video_path','_episode_wrist_right_video_path','_episode_start_idx',
        '_episode_end_idx','_episode_length','_episode_beta_idx','_splits','_source_id_to_name',
        '_block_episode_idx','_block_start','_block_end','_block_cumsum','_window_split_id')
    views=[]
    for view in ds.views:
        arrays={}
        for name,a in view._frame_memmaps.items():
            physical=_physical(name)
            if (not isinstance(a,np.memmap)or a.flags.writeable or a.mode!='r'or a.offset!=0 or not a.flags.c_contiguous or
                _safe(a.filename)!=paths['field:'+physical]or a is not ds.store.field(physical)):
                raise ValueError('native EMGFormer mapping layout binding changed')
            arrays[name]=dict(field=physical,shape=list(a.shape),dtype=a.dtype.str,strides=list(a.strides),offset=int(a.offset),mode=a.mode)
        for name in ('_episode_memmaps','_mano_npy_pose','_mano_npy_beta','_mano_npy_trans','_mano_npy_world_R','_mano_npy_world_t'):
            if getattr(view,name):raise ValueError('native EMGFormer unused mapping binding changed')
        views.append(dict(kind=_kind(view),settings={n:_value(getattr(view,n))for n in scalars},
            metadata={n:_value(getattr(view,n))for n in metadata},arrays=arrays,manifest=view._manifest,
            norm_stats_path=str(_safe(view.norm_stats_path)),mean=_value(view._emg_mean),std=_value(view._emg_std),
            transform=_transform(view.transform)))
    return _hash(dict(kind=_kind(ds),role=ds.role,repeat=ds.repeat,size=ds.size,
        upstream=str(ds.upstream),normalization=ds.normalization,contract=ds.contract,
        manifest=ds.store.manifest,metadata={k:_value(v)for k,v in ds.store.metadata.items()},
        episodes=ds.store.episodes,split_labels=ds.store.split_labels,views=views))


def _finite(datasets,chunk_rows,guard):
    result={}
    for role,ds in datasets.items():
        store=ds.store;spans=union_intervals((w['footprint_start'],w['footprint_end'])for w in ds.windows)
        input_rows=sum(b-a for a,b in spans);raw_bad=normalized_bad=0
        for lo,hi in spans:
            for start in range(lo,hi,chunk_rows):
                guard.check();end=min(hi,start+chunk_rows)
                for hand,view in zip(('left','right'),ds.views):
                    raw=np.asarray(store.field(f'emg_{hand}_filtered')[start:end],dtype=np.float32)
                    raw_bad+=int(np.count_nonzero(~np.isfinite(raw)))
                    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                        normalized=(raw-view._emg_mean)/(view._emg_std+1e-6)
                    normalized_bad+=int(np.count_nonzero(~np.isfinite(normalized)))
                    if raw_bad or normalized_bad:raise ValueError('nonfinite native EMGFormer raw/normalized consumed EMG')
        if role=='train':
            batches=(slice(start,min(hi,start+chunk_rows))for lo,hi in spans for start in range(lo,hi,chunk_rows))
            target_rows=input_rows
        else:
            centers=np.asarray(sorted({w['center']for w in ds.windows}),dtype=np.int64)
            batches=(centers[i:i+chunk_rows]for i in range(0,len(centers),chunk_rows));target_rows=len(centers)
        target_bad=converted_bad=supervision=0
        for selected in batches:
            guard.check();supervision+=int(np.count_nonzero(store.field('generated_label_valid')[selected]))
            for hand in ('left','right'):
                angles=np.asarray(store.field(f'generated_joint_angles_{hand}')[selected],dtype=np.float32)
                target_bad+=int(np.count_nonzero(~np.isfinite(angles)))
                for axis in ('pitch','yaw'):
                    raw=np.asarray(store.field(f'mocap_{hand}_wrist_{axis}')[selected],dtype=np.float32)
                    target_bad+=int(np.count_nonzero(~np.isfinite(raw)))
                    converted=np.deg2rad(raw)if role=='train'else np.deg2rad(raw.astype(np.float64)).astype(np.float32)
                    converted_bad+=int(np.count_nonzero(~np.isfinite(converted)))
                if target_bad or converted_bad:raise ValueError('nonfinite native EMGFormer consumed full/center target; no mask repair')
        if supervision==0:raise ValueError('every native EMGFormer role needs actual generated supervision')
        result[role]=dict(unique_possible_input_rows=input_rows,consumed_emg_elements=input_rows*16,
            consumed_emg_nonfinite_entries=raw_bad,normalized_emg_nonfinite_entries=normalized_bad,
            unique_target_rows=target_rows,consumed_target_elements=target_rows*44,
            consumed_target_nonfinite_entries=target_bad,converted_wrist_nonfinite_entries=converted_bad,
            possible_supervised_hand_rows=supervision,target_mode='full_trajectory'if role=='train'else 'center_only')
    guard.check();return result


@dataclass(frozen=True)
class EmgformerCohort:
    datasets: object
    partitions: dict
    identity: dict
    guard: object
    upstream: Path
    config: object
    normalization: dict
    input_paths: object
    _seal: str=field(repr=False)

    def check(self):
        self.guard.check()
        if (self.identity.get('sha256')!=self._seal or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=self._seal or
            emgformer_protocol(self.upstream,self.config)['sha256']!=self.identity['protocol_sha256']or
            emgformer_data_contract(self.upstream,self.config)['sha256']!=self.identity['data_contract_sha256']or
            _hash(self.normalization)!=self.identity['normalization_sha256']or _file_hash(__file__)!=IMPORTED_SHA):
            raise ValueError('native EMGFormer cohort identity/source/config changed')
        if _hash(self.partitions)!=self.identity['partition_content_sha256']:raise ValueError('native EMGFormer partition changed')
        try:
            if set(self.datasets)!=set(ROLES):raise ValueError('native EMGFormer role membership binding changed')
            for role,ds in self.datasets.items():
                if (type(ds)is not EmgformerDataset or ds.role!=role or len(ds)!=self.identity['counts'][role]or
                    ds.windows!=[w for w in self.partitions['windows']if w['role']==role]or
                    _hash(ds.partitions)!=self.identity['partition_content_sha256']or
                    _hash(ds.contract)!=_hash(emgformer_data_contract(self.upstream,ds.config))or
                    _reader_binding(ds,dict(self.input_paths))!=self.identity['reader_bindings_sha256'][role]):
                    raise ValueError('native EMGFormer reader computation/index binding changed')
        except (KeyError,AttributeError,TypeError,OSError)as exc:raise ValueError('native EMGFormer reader binding changed')from exc


def admit_emgformer_cohort(data,*,upstream,config,normalization,expected=None,chunk_rows=MAX_CHUNK_ROWS):
    if type(chunk_rows)is not int or not 1<=chunk_rows<=MAX_CHUNK_ROWS:raise ValueError('chunk_rows must be integer1..262144')
    root,size,_=_validated(_safe(upstream),config);data=_safe(data)
    protocol=emgformer_protocol(root,config);contract=emgformer_data_contract(root,config)
    if _hash(normalization)!=_hash(emgformer_normalization(root,config)):raise ValueError('native EMGFormer normalization binding differs')
    if expected is not None and(not isinstance(expected,dict)or expected.get('format')!=FORMAT or expected.get('sha256')!=_hash({k:v for k,v in expected.items()if k!='sha256'})):
        raise ValueError('invalid expected native EMGFormer receipt')
    sources=_sources(root);guard=_Guard([data/'manifest.json',data/'metadata.npz',*sources.values()])
    store=LegacyStore(data);guard.check();paths=_paths(store)
    for path in paths.values():guard.add(path)
    guard.check();partitions=build_emgformer_partitions(store);guard.check()
    datasets={r:EmgformerDataset(data,partitions,upstream=root,config=config,normalization=normalization,role=r)for r in ROLES}
    guard.check();source_sha={n:_stream_sha(p,guard)for n,p in sources.items()}
    payload={n:_stream_sha(p,guard)for n,p in paths.items()};sizes={n:p.stat().st_size for n,p in paths.items()}
    finite=_finite(datasets,chunk_rows,guard);counts={r:len(ds)for r,ds in datasets.items()}
    identity=dict(format=FORMAT,scope='complete physical input bytes and exact possible native full/point consumption; no GPU/fit permit',
        size=size,available_corpus='released legacy EgoEMG only; not unified EgoEMG/ShowEE/Incre or PDF equivalent',
        scientific_eligible=False,production_ready=False,full_paper_complete=False,
        canonical_unified_namespace_available=contract['canonical_namespace_available'],explicit_normalization_namespace='egoemg',
        protocol_sha256=protocol['sha256'],data_contract_sha256=contract['sha256'],normalization=copy.deepcopy(normalization),normalization_sha256=_hash(normalization),
        partition_sha256=partitions['provenance']['partition_sha256'],partition_content_sha256=_hash(partitions),
        sources_sha256=source_sha,payload_files_sha256=payload,payload_file_bytes=sizes,payload_file_count=len(paths),full_payload_content_hashed=True,
        counts=counts,unique_counts=counts.copy(),repeat_by_role={r:ds.repeat for r,ds in datasets.items()},episode_records=len(store.episodes),
        metadata_matches_registered_legacy_release=(payload['metadata:manifest']==MANIFEST_SHA and payload['metadata:episodes']==METADATA_SHA and counts==FULL_COUNTS and len(store.episodes)==41),
        finite_footprints=finite,consumed_emg_nonfinite_entries=sum(v['consumed_emg_nonfinite_entries']for v in finite.values()),
        consumed_target_nonfinite_entries=sum(v['consumed_target_nonfinite_entries']for v in finite.values()),
        reader_bindings_sha256={r:_reader_binding(ds,paths)for r,ds in datasets.items()},
        finite_scope='EMG all possible jitter input rows; train targets full possible trajectory rows; val/test unique centers; masked values unmodified',
        preserved_filter_and_normalization_limitations=partitions['provenance']['limitations'])
    identity['sha256']=_hash(identity)
    if expected is not None and _hash(expected)!=_hash(identity):raise ValueError('actual native EMGFormer input content/source identity differs from receipt')
    cohort=EmgformerCohort(MappingProxyType(datasets),copy.deepcopy(partitions),identity,guard,root,copy.deepcopy(config),copy.deepcopy(normalization),MappingProxyType(paths),identity['sha256'])
    cohort.check();return cohort


def native_emgformer_loaders(cohort,*,cpu_engineering=False):
    if type(cohort)is not EmgformerCohort:raise ValueError('native EMGFormer admitted cohort required')
    cohort.check()
    loaders={r:native_emgformer_loader(ds,cpu_engineering=cpu_engineering)for r,ds in cohort.datasets.items()}
    cohort.check();return loaders
