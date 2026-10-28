"""Complete initializer numeric inputs and full possible target trajectories."""
from __future__ import annotations

import copy
from dataclasses import dataclass,field
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .classic_emg import _activate,_file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard,_source_paths
from .data import LegacyStore,union_intervals
from .emg_initializer_literal import _validated,emg_initializer_protocol,emg_initializer_normalization,NORM_ASSET
from .emg_initializer_data import EmgInitializerDataset,build_initializer_partitions,_schema,ROLES

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_emg_initializer_full_content_admission_v1'
MAX_CHUNK_ROWS=262144
FULL_COUNTS=dict(train=63026,val=582)
MANIFEST_SHA='ebcef7877444253869845b9337b7c0df2b16b0b6f55d83c58d6c3c03bae8ac09'
METADATA_SHA='cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57'
LOCAL_SOURCES=('egoemg_repro/__init__.py','egoemg_repro/emg_initializer_literal.py',
    'egoemg_repro/emg_initializer_data.py','egoemg_repro/emg_initializer_cohort.py',
    'egoemg_repro/fusion_literal.py','egoemg_repro/fusion_literal_data.py',
    'egoemg_repro/vision_literal.py','egoemg_repro/vision_literal_data.py','egoemg_repro/partitions.py')


def _sources(upstream):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported initializer cohort source changed')
    paths=set(_source_paths(upstream))|{ROOT/name for name in LOCAL_SOURCES}|{upstream/NORM_ASSET}
    return {('upstream/'+str(p.relative_to(upstream)) if p.is_relative_to(upstream) else str(p.relative_to(ROOT))):p for p in sorted(paths)}


def _stream_sha(path,guard):
    guard.check();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    guard.check();return digest.hexdigest()


def _physical(name):return name.removesuffix('_paper') if name.endswith('_filtered_paper') else name


def _input_paths(datasets):
    store=datasets['train'].store
    return {'metadata:manifest':_safe(store.manifest_path),'metadata:episodes':_safe(store.metadata_path),
        **{'field:'+name:_safe(path) for name,path in _schema(store).items()}}


def _kind(value):
    cls=type(value);return cls.__module__+'.'+cls.__qualname__


def _transform(value):
    if hasattr(value,'transforms'):
        return dict(kind=_kind(value),attributes={k:v for k,v in vars(value).items() if k!='transforms'},transforms=[_transform(t) for t in value.transforms])
    return dict(kind=_kind(value),attributes=vars(value))


def _value(value):return value.tolist() if isinstance(value,np.ndarray) else value


def _reader_binding(ds,paths):
    if ({'field:'+k:_safe(p) for k,p in _schema(ds.store).items()}!={k:v for k,v in paths.items() if k.startswith('field:')} or
        _safe(ds.store.manifest_path)!=paths['metadata:manifest'] or _safe(ds.store.metadata_path)!=paths['metadata:episodes']):
        raise ValueError('initializer reader resolved path binding changed')
    views=[]
    scalar=('target_hand','dataset_name','window_length','stride','jitter','norm_mode','emg_layout','emg_field_preference',
        'channel_interpolate','skip_emg_loading','_center_target_only','_vision_enabled','_cached_vit_enabled','modalities','fields','video_streams')
    metadata=('_episode_id','_episode_subject','_episode_subject_id','_episode_source_parquet','_episode_zed_video_path',
        '_episode_head_video_path','_episode_wrist_left_video_path','_episode_wrist_right_video_path','_episode_start_idx',
        '_episode_end_idx','_episode_length','_episode_beta_idx','_splits','_source_id_to_name',
        '_block_episode_idx','_block_start','_block_end','_block_cumsum','_window_split_id')
    for view in ds.views:
        arrays={}
        for name,array in view._frame_memmaps.items():
            physical=_physical(name)
            if (not isinstance(array,np.memmap) or array.flags.writeable or array.mode!='r' or array.offset!=0 or not array.flags.c_contiguous or
                _safe(array.filename)!=paths['field:'+physical] or array is not ds.store.field(physical)):
                raise ValueError('initializer reader mapping layout binding changed')
            arrays[name]=dict(field=physical,shape=list(array.shape),dtype=str(array.dtype),strides=list(array.strides),offset=int(array.offset),mode=array.mode)
        views.append(dict(kind=_kind(view),settings={name:_value(getattr(view,name)) for name in scalar},
            metadata_sha256=_hash({name:_value(getattr(view,name)) for name in metadata}),arrays=arrays,
            mean=_value(view._emg_mean),std=_value(view._emg_std),transform=_transform(view.transform)))
    return _hash(dict(kind=_kind(ds),role=ds.role,repeat=ds.repeat,normalization=ds.normalization,protocol=ds.protocol,
        manifest_sha256=_hash(ds.store.manifest),episodes_sha256=_hash(ds.store.episodes),views=views))


def _finite(datasets,chunk_rows,guard):
    result={}
    for role,ds in datasets.items():
        rows=emg_bad=target_bad=label_true=0;store=ds.store
        for lo,hi in union_intervals((w['footprint_start'],w['footprint_end']) for w in ds.windows):
            rows+=hi-lo
            for start in range(lo,hi,chunk_rows):
                guard.check();end=min(hi,start+chunk_rows)
                label_true+=int(np.count_nonzero(store.field('generated_label_valid')[start:end]))
                for hand in ('left','right'):
                    emg_bad+=int(np.count_nonzero(~np.isfinite(store.field(f'emg_{hand}_filtered')[start:end])))
                    for name in (f'generated_joint_angles_{hand}',f'mocap_{hand}_wrist_pitch',f'mocap_{hand}_wrist_yaw'):
                        target_bad+=int(np.count_nonzero(~np.isfinite(store.field(name)[start:end])))
                if emg_bad or target_bad:raise ValueError('nonfinite initializer full-trajectory input; no mask-based repair')
        result[role]=dict(unique_possible_rows=rows,consumed_emg_elements=rows*16,consumed_emg_nonfinite_entries=emg_bad,
            full_trajectory_target_elements=rows*44,full_trajectory_target_nonfinite_entries=target_bad,possible_label_true_hand_rows=label_true)
    guard.check();return result


@dataclass(frozen=True)
class EmgInitializerCohort:
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
        if (self.identity.get('sha256')!=self._seal or _hash({k:v for k,v in self.identity.items() if k!='sha256'})!=self._seal or
            emg_initializer_protocol(self.upstream,self.config)['sha256']!=self.identity['protocol_sha256'] or
            _hash(self.normalization)!=self.identity['normalization_sha256'] or _file_hash(__file__)!=IMPORTED_SHA):
            raise ValueError('initializer cohort source/config/normalization identity changed')
        if _hash(self.partitions)!=self.identity['partition_content_sha256']:raise ValueError('initializer cohort partition changed')
        try:
            if _input_paths(self.datasets)!=dict(self.input_paths):raise ValueError('initializer reader input binding changed')
            for role,ds in self.datasets.items():
                if (type(ds) is not EmgInitializerDataset or len(ds)!=self.identity['counts'][role] or
                    ds.windows!=[w for w in self.partitions['windows'] if w['role']==role] or
                    _hash(ds.partitions)!=self.identity['partition_content_sha256'] or
                    _reader_binding(ds,self.input_paths)!=self.identity['reader_bindings_sha256'][role]):
                    raise ValueError('initializer reader role/index binding changed')
        except (KeyError,AttributeError,TypeError,OSError) as exc:raise ValueError('initializer reader binding changed') from exc


def admit_emg_initializer_cohort(data,*,upstream,config,normalization,expected=None,chunk_rows=MAX_CHUNK_ROWS):
    if type(chunk_rows) is not int or not 1<=chunk_rows<=MAX_CHUNK_ROWS:raise ValueError('chunk_rows must be integer1..262144')
    root,_=_validated(_safe(upstream),config);data=_safe(data);protocol=emg_initializer_protocol(root,config)
    if _hash(normalization)!=_hash(emg_initializer_normalization(root)):raise ValueError('initializer normalization source binding differs')
    if expected is not None and (not isinstance(expected,dict) or expected.get('format')!=FORMAT or expected.get('sha256')!=_hash({k:v for k,v in expected.items() if k!='sha256'})):
        raise ValueError('invalid expected initializer receipt')
    sources=_sources(root);guard=_StatGuard([data/'manifest.json',data/'metadata.npz',*sources.values()])
    store=LegacyStore(data);guard.check()
    paths={'metadata:manifest':_safe(store.manifest_path),'metadata:episodes':_safe(store.metadata_path),**{'field:'+k:_safe(p) for k,p in _schema(store).items()}}
    for path in paths.values():guard.add(path)
    guard.check();partitions=build_initializer_partitions(store);guard.check()
    datasets={role:EmgInitializerDataset(data,partitions,upstream=root,config=config,normalization=normalization,role=role) for role in ROLES}
    if _input_paths(datasets)!=paths:raise ValueError('initializer preopened input binding differs')
    guard.check();source_hashes={k:_stream_sha(p,guard) for k,p in sources.items()}
    payload={k:_stream_sha(p,guard) for k,p in paths.items()};sizes={k:p.stat().st_size for k,p in paths.items()}
    finite=_finite(datasets,chunk_rows,guard);counts={r:len(ds) for r,ds in datasets.items()}
    identity=dict(format=FORMAT,scope='complete actual numeric input bytes and full possible trajectories; no fit permit',
        available_corpus='released legacy EgoEMG only; missing original Incre training source',
        scientific_eligible=False,production_ready=False,full_paper_complete=False,protocol_sha256=protocol['sha256'],
        normalization=copy.deepcopy(normalization),normalization_sha256=_hash(normalization),
        partition_sha256=partitions['provenance']['partition_sha256'],partition_content_sha256=_hash(partitions),
        future_fusion_partition_content_sha256=partitions['provenance']['fusion_partition_content_sha256'],
        sources_sha256=source_hashes,payload_files_sha256=payload,payload_file_bytes=sizes,payload_file_count=len(paths),
        full_payload_content_hashed=True,counts=counts,unique_counts=counts.copy(),repeat_by_role={r:ds.repeat for r,ds in datasets.items()},
        metadata_matches_registered_legacy_release=(payload['metadata:manifest']==MANIFEST_SHA and payload['metadata:episodes']==METADATA_SHA and counts==FULL_COUNTS and len(store.episodes)==41),
        episode_records=len(store.episodes),finite_footprints=finite,consumed_emg_nonfinite_entries=sum(v['consumed_emg_nonfinite_entries'] for v in finite.values()),
        full_trajectory_target_nonfinite_entries=sum(v['full_trajectory_target_nonfinite_entries'] for v in finite.values()),
        reader_bindings_sha256={r:_reader_binding(ds,paths) for r,ds in datasets.items()},
        finite_scope='unique union of all possible full-jitter input AND target trajectory rows; both hands; not an epoch/error denominator')
    identity['sha256']=_hash(identity)
    if expected is not None and _hash(expected)!=_hash(identity):raise ValueError('actual initializer input content/source identity differs from receipt')
    result=EmgInitializerCohort(MappingProxyType(datasets),copy.deepcopy(partitions),identity,guard,root,copy.deepcopy(config),copy.deepcopy(normalization),MappingProxyType(paths),identity['sha256'])
    result.check();return result


def native_emg_initializer_loaders(cohort):
    cohort.check();_activate(cohort.upstream)
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0
    loaders={r:dm._make_loader(ds,shuffle=r=='train') for r,ds in cohort.datasets.items()}
    if any(l.batch_size!=260 or l.num_workers!=0 or l.generator is not None or l.drop_last for l in loaders.values()):
        raise ValueError('native initializer batch260 loader contract differs')
    cohort.check();return loaders
