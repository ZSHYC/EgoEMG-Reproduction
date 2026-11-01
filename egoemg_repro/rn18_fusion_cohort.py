"""Complete RN18 fusion input bytes and possible numeric footprints, no fit permit."""
from __future__ import annotations

import copy
from dataclasses import dataclass,field
from pathlib import Path
from types import MappingProxyType
import weakref

import numpy as np

from .classic_emg import _activate,_file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard,_source_paths
from .data import LegacyStore
from .fusion_literal_cohort import (_numeric_paths,_crop_paths,_input_paths,_finite,_stream_sha,
    _physical,_kind,_transform,MANIFEST_SHA,METADATA_SHA)
from .rn18_fusion_literal import _validated,rn18_fusion_protocol,NORM_ASSET
from .rn18_fusion_data import Rn18FusionDataset,build_rn18_fusion_partitions,rn18_fusion_normalization
from .vision_literal_data import _crop_path

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
ROLES=('train','val','test')
FORMAT='original_rn18_fusion_full_content_admission_v1'
FULL_COUNTS=dict(train=62470,val=1052,test=2792)
MAX_CHUNK_ROWS=262144
_COHORTS=weakref.WeakSet()
LOCAL=('rn18_fusion_literal','rn18_fusion_data','rn18_fusion_cohort','emgformer_data','emgformer_literal',
    'emg_initializer_data','emg_initializer_literal','fusion_literal_data','fusion_literal_cohort',
    'vision_literal_data','vision_literal','visual','crops')


def _sources(upstream):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 fusion cohort source changed')
    paths={*_source_paths(upstream),*(ROOT/'egoemg_repro'/(n+'.py')for n in LOCAL),upstream/NORM_ASSET}
    return {('upstream/'+str(p.relative_to(upstream))if p.is_relative_to(upstream)else str(p.relative_to(ROOT))):p for p in sorted(paths)}


def _config_token(config):
    from omegaconf import OmegaConf
    return _hash(OmegaConf.to_container(config,resolve=True))


def _reader_binding(ds,paths):
    ds.check();views=[]
    for view in ds.views:
        if view.store is not ds.store:raise ValueError('RN18 fusion reader store binding changed')
        arrays={}
        for name,a in view._frame_memmaps.items():
            physical=_physical(ds,name)
            if (not isinstance(a,np.memmap)or a.flags.writeable or a.mode!='r'or a.offset!=0 or not a.flags.c_contiguous
                or _safe(a.filename)!=paths['field:'+physical]or a is not ds.store.field(physical)):
                raise ValueError('RN18 fusion actual readonly map binding changed')
            arrays[name]=dict(field=physical,dtype=a.dtype.str,shape=list(a.shape),strides=list(a.strides),offset=int(a.offset))
        for episode in {w['episode']for w in ds.windows}:
            crop=_crop_path(view.crops,ds.store.episodes[episode]['id'])
            if _safe(crop/'data.mdb'if crop.is_dir()else crop)!=paths[f'crop:{episode:06d}']:
                raise ValueError('RN18 fusion crop provider binding changed')
        views.append(dict(kind=_kind(view),native=_kind(view.native),hand=view.target_hand,layout=view.emg_layout,
            preference=view.emg_field_preference,interpolate=view.channel_interpolate,norm_mode=view.norm_mode,
            mean=view._emg_mean.tolist(),std=view._emg_std.tolist(),patch_size=view.vision_patch_size,
            vision_mean=view.vision_mean.tolist(),vision_std=view.vision_std.tolist(),episode_ids_sha256=_hash(view._episode_id),
            transform=_transform(view.transform),arrays=arrays))
    return _hash(dict(kind=_kind(ds),native=_kind(ds.native),role=ds.role,repeat=ds.repeat,
        frame_field=ds.frame_mapping_field,protocol=ds.protocol,normalization=ds.normalization,views=views,
        manifest_sha256=_hash(ds.store.manifest),episodes_sha256=_hash(ds.store.episodes)))


@dataclass(frozen=True,eq=False)
class Rn18FusionCohort:
    datasets: object
    partitions: dict
    identity: dict
    guard: object
    upstream: Path
    config: object
    input_paths: object
    normalization: dict
    _seal: str=field(repr=False)
    _config_sha: str=field(repr=False)

    def check(self,*,require_full=True):
        if type(require_full)is not bool:raise ValueError('RN18 fusion full requirement must be boolean')
        if type(self)is not Rn18FusionCohort or self not in _COHORTS:raise ValueError('actual process-registered RN18 fusion cohort required')
        self.guard.check()
        if (self.identity.get('sha256')!=self._seal or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=self._seal
            or _config_token(self.config)!=self._config_sha or _file_hash(__file__)!=IMPORTED_SHA
            or _hash(self.normalization)!=self.identity['normalization_sha256']):
            raise ValueError('RN18 fusion admitted source/config/content identity changed')
        if require_full and not self.identity['registered_full_legacy']:raise ValueError('registered complete legacy RN18 fusion cohort required')
        current_sources=_sources(self.upstream)
        if (set(current_sources)!=set(self.identity['sources_sha256'])
            or any(_file_hash(p)!=self.identity['sources_sha256'][name]for name,p in current_sources.items())):
            raise ValueError('RN18 fusion actual source/provider content changed')
        try:
            if _hash(self.partitions)!=self.identity['partition_content_sha256']or _input_paths(self.datasets)!=dict(self.input_paths):
                raise ValueError('RN18 fusion partition/input provider changed')
            if set(self.datasets)!=set(ROLES):raise ValueError('RN18 fusion all roles required')
            for role,ds in self.datasets.items():
                if (type(ds)is not Rn18FusionDataset or ds.role!=role or len(ds)!=self.identity['counts'][role]
                    or ds.windows!=[w for w in self.partitions['windows']if w['role']==role]
                    or _hash(ds.partitions)!=self.identity['partition_content_sha256']
                    or _reader_binding(ds,self.input_paths)!=self.identity['reader_bindings_sha256'][role]):
                    raise ValueError('RN18 fusion native role/reader binding changed')
        except (KeyError,AttributeError,TypeError,OSError)as exc:raise ValueError('RN18 fusion native reader binding changed')from exc
        self.guard.check()


def admit_rn18_fusion_cohort(data,crops,*,upstream,config,expected=None,require_full=True,chunk_rows=MAX_CHUNK_ROWS):
    if type(chunk_rows)is not int or not 1<=chunk_rows<=MAX_CHUNK_ROWS:raise ValueError('integer RN18 fusion scan chunk1..262144 required')
    if type(require_full)is not bool:raise ValueError('RN18 fusion full requirement must be boolean')
    root,_=_validated(_safe(upstream),config);data=_safe(data);crops=_safe(crops);sources=_sources(root)
    guard=_StatGuard([data/'manifest.json',data/'metadata.npz',*sources.values()])
    if expected is not None and(type(expected)is not dict or expected.get('format')!=FORMAT
        or expected.get('sha256')!=_hash({k:v for k,v in expected.items()if k!='sha256'})):
        raise ValueError('sealed expected RN18 fusion receipt required')
    store=LegacyStore(data)
    paths={'metadata:manifest':store.manifest_path,'metadata:episodes':store.metadata_path,**_numeric_paths(store)}
    for p in paths.values():guard.add(p)
    guard.check();partitions=build_rn18_fusion_partitions(store);guard.check()
    crops_paths,_=_crop_paths(store,partitions,crops);paths.update(crops_paths)
    for p in crops_paths.values():guard.add(p)
    norm=rn18_fusion_normalization(root,config);protocol=rn18_fusion_protocol(root,config)
    datasets={r:Rn18FusionDataset(data,partitions,crops,upstream=root,config=config,normalization=norm,role=r)for r in ROLES}
    if any(not len(ds)for ds in datasets.values()):raise ValueError('all RN18 fusion roles must be nonempty')
    if _input_paths(datasets)!=paths:raise ValueError('RN18 fusion preopened input binding differs')
    for ds in datasets.values():
        for p,stamp in ds._guard.paths.items():
            if p in guard.paths and guard.paths[p]!=stamp:raise ValueError('RN18 fusion source/input generation changed while opening readers')
    source_hashes={k:_stream_sha(p,guard)for k,p in sources.items()}
    payload={k:_stream_sha(p,guard)for k,p in paths.items()};sizes={k:p.stat().st_size for k,p in paths.items()}
    finite=_finite(datasets,chunk_rows,guard);counts={r:len(ds)for r,ds in datasets.items()}
    registered=(payload['metadata:manifest']==MANIFEST_SHA and payload['metadata:episodes']==METADATA_SHA
        and counts==FULL_COUNTS and len(store.episodes)==41)
    identity=dict(format=FORMAT,scope='complete actual bytes and every possible native jitter/center footprint; no trained model permit',
        scientific_eligible=False,production_ready=False,full_paper_complete=False,
        available_corpus='released legacy EgoEMG only; not physically merged EgoEMG/ShowEE/Incre',
        protocol_sha256=protocol['sha256'],normalization=copy.deepcopy(norm),normalization_sha256=_hash(norm),
        partition_sha256=partitions['provenance']['partition_sha256'],partition_content_sha256=_hash(partitions),
        sources_sha256=source_hashes,payload_files_sha256=payload,payload_file_bytes=sizes,payload_file_count=len(paths),
        full_payload_content_hashed=True,counts=counts,unique_counts=dict(counts),repeat_by_role={r:1 for r in ROLES},
        registered_full_legacy=registered,metadata_matches_registered_legacy_release=registered,episode_records=len(store.episodes),
        finite_footprints=finite,consumed_emg_nonfinite_entries=sum(v['consumed_emg_nonfinite_entries']for v in finite.values()),
        possible_center_target_nonfinite_entries=sum(v['possible_center_target_nonfinite_entries']for v in finite.values()),
        reader_bindings_sha256={r:_reader_binding(ds,paths)for r,ds in datasets.items()},
        finite_scope='union of all possible jitter input rows/center labels, both hands; not epoch/error denominators')
    identity['sha256']=_hash(identity)
    if expected is not None and _hash(expected)!=_hash(identity):raise ValueError('recomputed RN18 fusion source/content receipt differs')
    result=Rn18FusionCohort(MappingProxyType(datasets),copy.deepcopy(partitions),identity,guard,root,copy.deepcopy(config),
        MappingProxyType(paths),copy.deepcopy(norm),identity['sha256'],_config_token(config))
    _COHORTS.add(result);result.check(require_full=require_full);return result


def native_rn18_fusion_loaders(cohort,*,scope='full'):
    if type(scope)is not str or scope not in('full','cuda_engineering','cpu_engineering'):raise ValueError('explicit RN18 fusion loader scope required')
    cpu=scope=='cpu_engineering';cohort.check(require_full=not cpu);_activate(cohort.upstream)
    from egoemg.datamodule import make_data_module
    from torch.utils.data import Subset
    dm=make_data_module(cohort.config)
    if cpu:dm.batch_size=2;dm.num_workers=dm.eval_num_workers=0;dm.pin_memory=False
    loaders={}
    for role,ds in cohort.datasets.items():
        if cpu:ds=Subset(ds,range(min(len(ds),4)))
        elif scope=='cuda_engineering'and role=='train':ds=Subset(ds,range(400))
        workers=dm.num_workers if role=='train'else dm.eval_num_workers
        loader=dm._make_loader(ds,shuffle=role=='train',num_workers=workers)
        if (loader.batch_size!=(2 if cpu else 200)or loader.num_workers!=(0 if cpu else 10)
            or loader.generator is not None or loader.drop_last or loader.pin_memory is not(not cpu)
            or(not cpu and(loader.prefetch_factor!=2 or not loader.persistent_workers))):
            raise ValueError('original RN18 fusion native loader policy differs')
        loaders[role]=loader
    cohort.check(require_full=not cpu);return loaders
