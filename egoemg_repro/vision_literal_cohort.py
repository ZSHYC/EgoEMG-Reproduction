"""Complete available vision-input admission and original loader construction.

Hashes actual point-reader files and all consumed episode crop databases. This
does not authenticate a publisher, admit the unavailable merged corpus, or
qualify a model/training run. No model or RNG stream is constructed here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass,field
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .classic_emg import _activate,_file_hash,_hash
from .classic_emg_cohort import _StatGuard,_safe,_source_paths
from .data import LegacyStore,union_intervals
from .vision_literal import _validated,vision_protocol
from .vision_literal_data import VisionLiteralDataset,build_vision_partitions,_crop_path


ROLES=('train','val','test')
FORMAT='original_vision_full_content_admission_v1'
ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
MAX_CHUNK_ROWS=262144
FULL_COUNTS=dict(train=202020,val=3152,test=8612)
REGISTERED_MANIFEST_SHA='ebcef7877444253869845b9337b7c0df2b16b0b6f55d83c58d6c3c03bae8ac09'
REGISTERED_METADATA_SHA='cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57'
LOCAL_SOURCES=('egoemg_repro/__init__.py','egoemg_repro/vision_literal_cohort.py',
    'egoemg_repro/vision_literal.py','egoemg_repro/vision_literal_data.py',
    'egoemg_repro/vision_literal_runtime.py','egoemg_repro/visual.py','egoemg_repro/crops.py')


def _stream_sha(path,guard):
    guard.check();digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    guard.check()
    return digest.hexdigest()


def _input_paths(datasets):
    train=datasets['train'];store=train.store
    paths={'metadata:manifest':store.manifest_path,'metadata:episodes':store.metadata_path,
        'field:frame_split_id':Path(store.field('frame_split_id').filename)}
    for name,array in train.views[0]._frame_memmaps.items():
        name=train.frame_mapping_field if name=='image_head_frame_index' else name
        paths['field:'+name]=Path(array.filename)
    episodes=sorted({w['episode'] for ds in datasets.values() for w in ds.windows})
    for episode in episodes:
        path=_crop_path(train.views[0].crops,store.episodes[episode]['id'])
        paths[f'crop:{episode:06d}']=path/'data.mdb' if path.is_dir() else path
    return {name:_safe(path) for name,path in paths.items()}


def _sources(upstream):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported vision cohort source changed')
    paths=set(_source_paths(upstream))|{ROOT/name for name in LOCAL_SOURCES}
    result={}
    for path in sorted(paths):
        if path.is_relative_to(upstream):name='upstream/'+str(path.relative_to(upstream))
        else:name=str(path.relative_to(ROOT))
        result[name]=path
    return result


def _finite_centers(datasets,chunk_rows,guard):
    result={}
    for role,ds in datasets.items():
        store=ds.store;rows=nonfinite=label_true=0
        spans=union_intervals((w['center'],w['center']+max(1,w['jitter_upper'])) for w in ds.windows)
        for lo,hi in spans:
            rows+=hi-lo
            for start in range(lo,hi,chunk_rows):
                guard.check();end=min(hi,start+chunk_rows)
                label_true+=int(np.count_nonzero(store.field('generated_label_valid')[start:end]))
                for hand in ('left','right'):
                    for name in ('generated_joint_angles_'+hand,'mocap_'+hand+'_wrist_pitch','mocap_'+hand+'_wrist_yaw'):
                        nonfinite+=int(np.count_nonzero(~np.isfinite(store.field(name)[start:end])))
                if nonfinite:raise ValueError('nonfinite target at a possible native vision center; no label cleanup')
        result[role]=dict(possible_center_rows=rows,possible_center_label_true_hand_rows=label_true,
            possible_center_target_nonfinite_entries=nonfinite)
    guard.check();return result


@dataclass(frozen=True)
class VisionCohort:
    datasets: object
    partitions: dict
    identity: dict
    guard: object
    upstream: Path
    config: object
    input_paths: object
    _seal: str=field(repr=False)

    def check(self):
        self.guard.check()
        if (self.identity.get('sha256')!=self._seal or
            _hash({k:v for k,v in self.identity.items() if k!='sha256'})!=self._seal or
            vision_protocol(self.upstream,self.config)['sha256']!=self.identity['protocol_sha256'] or
            _file_hash(__file__)!=IMPORTED_SHA):
            raise ValueError('vision cohort source/config/content identity changed')
        if _hash(self.partitions)!=self.identity['partition_content_sha256']:
            raise ValueError('vision cohort partition changed')
        for role,ds in self.datasets.items():
            if (type(ds) is not VisionLiteralDataset or len(ds)!=self.identity['counts'][role] or
                ds.windows!=[w for w in self.partitions['windows'] if w['role']==role] or
                _hash(ds.partitions)!=self.identity['partition_content_sha256']):
                raise ValueError('original vision dataset role/index binding changed')


def admit_vision_cohort(data,crops,*,upstream,config,expected=None,chunk_rows=MAX_CHUNK_ROWS):
    """Read/hash all actual payloads; scan every possible admitted jitter center.

    ``expected`` is re-opened by recomputation, never accepted as a certificate.
    Metadata equality only identifies the registered release layout, not signed
    publisher authenticity. Full runners must bind a previously frozen content
    receipt and independently prove budget, selection, and evaluation completion.
    """
    if type(chunk_rows) is not int or not 1<=chunk_rows<=MAX_CHUNK_ROWS:
        raise ValueError('finite scan chunk_rows must be an integer in [1,262144]')
    root,_,_=_validated(_safe(upstream),config);data=_safe(data);crops=_safe(crops)
    protocol=vision_protocol(root,config)
    if expected is not None and (not isinstance(expected,dict) or expected.get('format')!=FORMAT or
        expected.get('sha256')!=_hash({k:v for k,v in expected.items() if k!='sha256'})):
        raise ValueError('invalid expected vision content receipt')
    sources=_sources(root)
    # Bind metadata and code before creating the readers that consume them.
    guard=_StatGuard([data/'manifest.json',data/'metadata.npz',*sources.values()])
    store=LegacyStore(data);partitions=build_vision_partitions(store)
    datasets={role:VisionLiteralDataset(data,partitions,crops,upstream=root,config=config,role=role) for role in ROLES}
    if any(len(ds)==0 for ds in datasets.values()):raise ValueError('all complete vision roles must be nonempty')
    paths=_input_paths(datasets)
    for path in paths.values():guard.add(path)
    guard.check()
    source_hashes={name:_stream_sha(path,guard) for name,path in sources.items()}
    payload={name:_stream_sha(path,guard) for name,path in paths.items()}
    finite=_finite_centers(datasets,chunk_rows,guard)
    counts={role:len(ds) for role,ds in datasets.items()}
    registered=(payload['metadata:manifest']==REGISTERED_MANIFEST_SHA and payload['metadata:episodes']==REGISTERED_METADATA_SHA
        and counts==FULL_COUNTS and len(store.episodes)==41)
    identity=dict(format=FORMAT,scope='complete actual input content and loader population only; no model/training/test completion',
        scientific_eligible=False,production_ready=False,full_paper_complete=False,
        available_corpus='released legacy EgoEMG only; not declared physically merged EgoEMG/ShowEE/Incre',
        protocol_sha256=protocol['sha256'],partition_sha256=partitions['provenance']['partition_sha256'],
        partition_content_sha256=_hash(partitions),sources_sha256=source_hashes,
        payload_files_sha256=payload,payload_file_count=len(paths),full_payload_content_hashed=True,
        counts=counts,unique_counts={role:len(ds)//ds.repeat for role,ds in datasets.items()},
        repeat_by_role={role:ds.repeat for role,ds in datasets.items()},
        metadata_matches_registered_legacy_release=registered,episode_records=len(store.episodes),
        finite_centers=finite,possible_center_target_nonfinite_entries=sum(v['possible_center_target_nonfinite_entries'] for v in finite.values()),
        finite_scope='unweighted union of every possible center row; not an observed training epoch or error denominator')
    identity['sha256']=_hash(identity)
    if expected is not None and _hash(expected)!=_hash(identity):raise ValueError('actual vision input content/source identity differs from receipt')
    result=VisionCohort(MappingProxyType(datasets),copy.deepcopy(partitions),identity,guard,root,
        copy.deepcopy(config),MappingProxyType(paths),identity['sha256'])
    result.check();return result


def native_vision_loaders(cohort):
    """Original factory/_make_loader/_collate_fn; one rank/workers0 adaptation.

    The bridge already repeats all roles twice. Native _make_loader neither
    repeats again nor supplies a separate RNG generator. Plain vision datasets
    have no session_index_groups, matching the original repeated ConcatDataset.
    """
    cohort.check();_activate(cohort.upstream)
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cohort.config)
    dm.num_workers=dm.eval_num_workers=0
    loaders={role:dm._make_loader(ds,shuffle=role=='train') for role,ds in cohort.datasets.items()}
    if any(loader.batch_size!=360 or loader.generator is not None or loader.num_workers!=0 or loader.drop_last
        for loader in loaders.values()):raise ValueError('original vision loader execution contract differs')
    cohort.check();return loaders
