"""Full actual numeric admission for the RN18-compatible cotrain-S producer.

Every possible full-jitter input AND trajectory-target row is checked. Actual
raw files are byte-hashed before and after extraction; live process registry,
independent typed reader snapshots and provider hashes reject resealed state.
No model, optimizer, CUDA or trained-checkpoint permission is produced here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import weakref

from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard,_source_paths
from .data import LegacyStore
from .emg_initializer_literal import _validated,emg_initializer_protocol,emg_initializer_normalization,NORM_ASSET
from .emg_initializer_cohort import _stream_sha,_input_paths,_reader_binding,_finite,MANIFEST_SHA,METADATA_SHA,MAX_CHUNK_ROWS
from .rn18_initializer_data import (Rn18InitializerDataset,build_rn18_initializer_partitions,
    rn18_initializer_contract,_config_token,ROLES)
from .emg_initializer_data import _schema

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='rn18_fixed_holdout_cotrain_full_content_admission_v1'
FULL_COUNTS=dict(train=62470,val=1052)
FULL_FUTURE_PARTITION_SHA='44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0'
FROZEN_READER_SHA='0287bb5aa8031556222f374f733a7301db02373fcf80101550f6cdd0cb1d6136'
FROZEN_READER_BINDING_SHA='7c3624c1d22640144ea2e2d79c369fad917c5a9a4e48f1e60ef36048bf43a5b2'
LOCAL=('__init__','rn18_initializer_data','rn18_initializer_cohort','emg_initializer_literal','emg_initializer_data',
    'emg_initializer_cohort','rn18_fusion_literal','rn18_fusion_data','emgformer_literal','emgformer_data',
    'fusion_literal','fusion_literal_data','fusion_initializer_handoff','fusion_literal_runtime','vision_literal',
    'vision_literal_data','vision_literal_runtime','classic_emg','classic_emg_cohort','classic_emg_data','data','partitions')
_LIVE=weakref.WeakKeyDictionary()


def _sources(root):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 initializer cohort source changed')
    if(_file_hash(ROOT/'egoemg_repro/emg_initializer_data.py')!=FROZEN_READER_SHA
        or _file_hash(ROOT/'egoemg_repro/emg_initializer_cohort.py')!=FROZEN_READER_BINDING_SHA):
        raise ValueError('closed original full reader/binding source changed')
    paths={*_source_paths(root),*(ROOT/'egoemg_repro'/(n+'.py')for n in LOCAL),root/NORM_ASSET}
    return {('upstream/'+str(p.relative_to(root))if p.is_relative_to(root)else str(p.relative_to(ROOT))):_safe(p)for p in sorted(paths)}


@dataclass(eq=False)
class Rn18InitializerCohort:
    datasets:object
    partitions:dict
    identity:dict
    guard:object
    upstream:Path
    config:object
    normalization:dict
    input_paths:object

    def check(self):
        r=_LIVE.get(self)
        if r is None:raise ValueError('registered RN18 initializer numeric cohort required')
        self.guard.check();paths=_sources(self.upstream)
        if (paths!=r['source_paths']or any(_file_hash(p)!=r['sources'][n]for n,p in paths.items())
            or self.identity.get('sha256')!=r['identity']or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=r['identity']
            or _config_token(self.config)!=r['config']or _hash(self.normalization)!=self.identity['normalization_sha256']
            or _hash(self.partitions)!=self.identity['partition_content_sha256']):
            raise ValueError('RN18 initializer source/config/normalization/identity changed')
        try:
            if set(self.datasets)!=set(ROLES)or _input_paths(self.datasets)!=dict(self.input_paths):
                raise ValueError('RN18 initializer role/input paths changed')
            for role,ds in self.datasets.items():
                if(type(ds)is not Rn18InitializerDataset or ds.role!=role or len(ds)!=self.identity['counts'][role]
                    or ds.windows!=[w for w in self.partitions['windows']if w['role']==role]
                    or _hash(ds.partitions)!=self.identity['partition_content_sha256']
                    or _reader_binding(ds,self.input_paths)!=self.identity['reader_bindings_sha256'][role]):
                    raise ValueError('RN18 initializer full-trajectory reader binding changed')
                ds.check()
        except(KeyError,TypeError,AttributeError,OSError)as exc:raise ValueError('RN18 initializer full reader state changed')from exc


def admit_rn18_initializer_cohort(data,*,upstream,config,normalization,expected=None,require_full=True,chunk_rows=MAX_CHUNK_ROWS):
    if type(require_full)is not bool or type(chunk_rows)is not int or not 1<=chunk_rows<=MAX_CHUNK_ROWS:
        raise ValueError('explicit full boolean and integer chunk_rows1..262144 required')
    root=_safe(upstream);sources=_sources(root);data=_safe(data)
    guard=_StatGuard([data/'manifest.json',data/'metadata.npz',*sources.values()])
    source_hashes={n:_file_hash(p)for n,p in sources.items()};guard.check();root,_=_validated(root,config)
    contract=rn18_initializer_contract(root,config);protocol=emg_initializer_protocol(root,config)
    norm=emg_initializer_normalization(root)
    if _hash(normalization)!=_hash(norm):raise ValueError('original RN18 cotrain normalization differs')
    if expected is not None and(not isinstance(expected,dict)or expected.get('format')!=FORMAT
        or expected.get('sha256')!=_hash({k:v for k,v in expected.items()if k!='sha256'})):
        raise ValueError('invalid expected RN18 cotrain numeric receipt')
    store=LegacyStore(data);guard.check();fields=_schema(store)
    paths={'metadata:manifest':_safe(store.manifest_path),'metadata:episodes':_safe(store.metadata_path),
        **{'field:'+n:_safe(p)for n,p in fields.items()}}
    for p in paths.values():guard.add(p)
    partitions=build_rn18_initializer_partitions(store);guard.check()
    datasets={role:Rn18InitializerDataset(data,partitions,upstream=root,config=config,normalization=norm,role=role)for role in ROLES}
    if _input_paths(datasets)!=paths:raise ValueError('actual RN18 initializer reader paths differ')
    payload={n:_stream_sha(p,guard)for n,p in paths.items()};sizes={n:p.stat().st_size for n,p in paths.items()}
    finite=_finite(datasets,chunk_rows,guard);counts={r:len(ds)for r,ds in datasets.items()}
    # A fixed source hash authenticates the full possible target scan even when
    # the underlying filesystem stamps have coarse timestamp granularity.
    if any(_stream_sha(p,guard)!=payload[n]for n,p in paths.items()):raise ValueError('raw RN18 initializer input bytes changed during admission')
    now=_sources(root)
    if now!=sources or any(_file_hash(p)!=source_hashes[n]for n,p in sources.items()):raise ValueError('RN18 cotrain providers changed during admission')
    future=partitions['provenance']['future_fusion_partition_content_sha256']
    registered=(payload['metadata:manifest']==MANIFEST_SHA and payload['metadata:episodes']==METADATA_SHA
        and counts==FULL_COUNTS and len(store.episodes)==41 and future==FULL_FUTURE_PARTITION_SHA)
    if require_full and not registered:raise ValueError('full initializer requires exact registered RN18 holdout corpus/population')
    identity=dict(format=FORMAT,scope='complete actual bytes and all possible full trajectories; data only',
        available_corpus='released legacy schema only; original Incre right-hand corpus absent',
        registered_full_legacy=registered,metadata_matches_registered_legacy_release=registered,
        counts=counts,unique_counts=counts.copy(),repeat_by_role={r:1 for r in ROLES},episode_records=len(store.episodes),
        full_payload_content_hashed=True,all_payload_bytes_rehashed_after_full_target_scan=True,
        payload_files_sha256=payload,payload_file_bytes=sizes,payload_file_count=len(paths),
        source_files_sha256=source_hashes,protocol_sha256=protocol['sha256'],data_contract_sha256=contract['sha256'],
        normalization=copy.deepcopy(norm),normalization_sha256=_hash(norm),
        partition_sha256=partitions['provenance']['partition_sha256'],partition_content_sha256=_hash(partitions),
        future_fusion_partition_content_sha256=future,future_holdout_overlap_rows=partitions['future_holdout_overlap_rows'],
        finite_footprints=finite,consumed_emg_nonfinite_entries=sum(v['consumed_emg_nonfinite_entries']for v in finite.values()),
        full_trajectory_target_nonfinite_entries=sum(v['full_trajectory_target_nonfinite_entries']for v in finite.values()),
        reader_bindings_sha256={r:_reader_binding(ds,paths)for r,ds in datasets.items()},
        finite_scope='unique union of every possible full-jitter INPUT and full TRAJECTORY TARGET row, both hands; masked-false values not repaired; not an epoch denominator',
        model_or_training_permit=False,scientific_eligible=False,full_paper_complete=False)
    identity['sha256']=_hash(identity)
    if expected is not None and _hash(expected)!=_hash(identity):raise ValueError('fresh RN18 initializer complete numeric identity differs from expected receipt')
    v=Rn18InitializerCohort(MappingProxyType(datasets),copy.deepcopy(partitions),identity,guard,root,
        copy.deepcopy(config),copy.deepcopy(norm),MappingProxyType(paths))
    _LIVE[v]=dict(identity=identity['sha256'],sources=source_hashes,source_paths=sources,config=_config_token(config))
    v.check();return v
