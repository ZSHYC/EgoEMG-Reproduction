"""Complete coupled input bytes and finite footprints, never a training permit."""
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
from .fusion_literal import _validated,fusion_protocol
from .fusion_literal_data import FusionLiteralDataset,build_fusion_partitions,fusion_normalization,NORM_ASSET
from .vision_literal_data import _crop_path

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
ROLES=('train','val','test')
FORMAT='original_fusion_full_content_admission_v1'
MAX_CHUNK_ROWS=262144
FULL_COUNTS=dict(train=395276,val=3152,test=8612)
MANIFEST_SHA='ebcef7877444253869845b9337b7c0df2b16b0b6f55d83c58d6c3c03bae8ac09'
METADATA_SHA='cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57'
LOCAL_SOURCES=('egoemg_repro/__init__.py','egoemg_repro/fusion_literal_cohort.py',
    'egoemg_repro/fusion_literal.py','egoemg_repro/fusion_literal_data.py',
    'egoemg_repro/vision_literal_data.py','egoemg_repro/vision_literal.py',
    'egoemg_repro/visual.py','egoemg_repro/crops.py','egoemg_repro/partitions.py')


def _stream_sha(path,guard):
    guard.check();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    guard.check();return digest.hexdigest()


def _sources(upstream):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported fusion cohort source changed')
    paths=set(_source_paths(upstream))|{ROOT/p for p in LOCAL_SOURCES}|{upstream/NORM_ASSET}
    return {('upstream/'+str(p.relative_to(upstream)) if p.is_relative_to(upstream) else str(p.relative_to(ROOT))):p for p in sorted(paths)}


def _physical(ds,name):
    if name=='image_head_frame_index':return ds.frame_mapping_field
    if name.startswith('emg_') and name.endswith('_filtered_paper'):return name.removesuffix('_paper')
    return name


def _input_paths(datasets):
    ds=datasets['train'];store=ds.store
    paths={'metadata:manifest':store.manifest_path,'metadata:episodes':store.metadata_path,
        'field:frame_split_id':Path(store.field('frame_split_id').filename)}
    for view in ds.views:
        for name,array in view._frame_memmaps.items():
            key='field:'+_physical(ds,name);path=_safe(array.filename)
            if key in paths and paths[key]!=path:raise ValueError('coupled input path binding differs')
            paths[key]=path
    for episode in sorted({w['episode'] for d in datasets.values() for w in d.windows}):
        path=_crop_path(ds.views[0].crops,store.episodes[episode]['id'])
        paths[f'crop:{episode:06d}']=path/'data.mdb' if path.is_dir() else path
    return {k:_safe(p) for k,p in paths.items()}


def _kind(value):
    cls=value if isinstance(value,type) else type(value)
    return cls.__module__+'.'+cls.__qualname__


def _transform(value):
    # The original recipe's dataclass transforms contain only scalar settings;
    # retain the class identity as well as every setting and composition order.
    if hasattr(value,'transforms'):
        return dict(kind=_kind(value),attributes={k:v for k,v in vars(value).items() if k!='transforms'},
            transforms=[_transform(t) for t in value.transforms])
    return dict(kind=_kind(value),attributes=vars(value))


def _reader_binding(ds,paths):
    views=[]
    for view in ds.views:
        arrays={}
        if view.store is not ds.store:raise ValueError('fusion reader store binding changed')
        episode=ds.windows[0]['episode']
        crop=_crop_path(view.crops,ds.store.episodes[episode]['id'])
        if _safe(crop/'data.mdb' if crop.is_dir() else crop)!=paths[f'crop:{episode:06d}']:
            raise ValueError('fusion reader crop root binding changed')
        for name,array in view._frame_memmaps.items():
            physical=_physical(ds,name)
            if (not isinstance(array,np.memmap) or array.flags.writeable or
                _safe(array.filename)!=paths['field:'+physical] or array is not ds.store.field(physical)):
                raise ValueError('fusion reader array mapping binding changed')
            arrays[name]=dict(field=physical,shape=list(array.shape),dtype=str(array.dtype))
        views.append(dict(kind=_kind(view),native=_kind(view.native),hand=view.target_hand,
            layout=view.emg_layout,positions=view._emg2pose_channel_indices,interpolate=view.channel_interpolate,
            preference=view.emg_field_preference,norm_mode=view.norm_mode,mean=view._emg_mean,std=view._emg_std,
            patch_size=view.vision_patch_size,vision_mean=view.vision_mean.tolist(),vision_std=view.vision_std.tolist(),
            episode_ids_sha256=_hash(view._episode_id),transform=_transform(view.transform),arrays=arrays))
    return _hash(dict(kind=_kind(ds),role=ds.role,repeat=ds.repeat,frame_field=ds.frame_mapping_field,
        native=_kind(ds.native),normalization=ds.normalization,protocol=ds.protocol,views=views,
        manifest_sha256=_hash(ds.store.manifest),episodes_sha256=_hash(ds.store.episodes)))


def _finite(datasets,chunk_rows,guard):
    result={}
    for role,ds in datasets.items():
        store=ds.store;emg_rows=emg_bad=center_rows=target_bad=label_true=0
        for lo,hi in union_intervals((w['footprint_start'],w['footprint_end']) for w in ds.windows):
            emg_rows+=hi-lo
            for start in range(lo,hi,chunk_rows):
                guard.check();end=min(hi,start+chunk_rows)
                for hand in ('left','right'):
                    emg_bad+=int(np.count_nonzero(~np.isfinite(store.field(f'emg_{hand}_filtered')[start:end])))
                if emg_bad:raise ValueError('nonfinite EMG in a possible native fusion window; no input repair')
        for lo,hi in union_intervals((w['center'],w['center']+max(1,w['jitter_upper'])) for w in ds.windows):
            center_rows+=hi-lo
            for start in range(lo,hi,chunk_rows):
                guard.check();end=min(hi,start+chunk_rows)
                label_true+=int(np.count_nonzero(store.field('generated_label_valid')[start:end]))
                for hand in ('left','right'):
                    for name in ('generated_joint_angles_'+hand,'mocap_'+hand+'_wrist_pitch','mocap_'+hand+'_wrist_yaw'):
                        target_bad+=int(np.count_nonzero(~np.isfinite(store.field(name)[start:end])))
                if target_bad:raise ValueError('nonfinite target at a possible native fusion center; no label cleanup')
        result[role]=dict(unique_possible_emg_rows=emg_rows,consumed_emg_elements=emg_rows*16,
            consumed_emg_nonfinite_entries=emg_bad,possible_center_rows=center_rows,
            possible_center_label_true_hand_rows=label_true,possible_center_target_elements=center_rows*44,
            possible_center_target_nonfinite_entries=target_bad)
    guard.check();return result


@dataclass(frozen=True)
class FusionCohort:
    datasets: object
    partitions: dict
    identity: dict
    guard: object
    upstream: Path
    config: object
    input_paths: object
    normalization: dict
    _seal: str=field(repr=False)

    def check(self):
        self.guard.check()
        if (self.identity.get('sha256')!=self._seal or
            _hash({k:v for k,v in self.identity.items() if k!='sha256'})!=self._seal or
            fusion_protocol(self.upstream,self.config)['sha256']!=self.identity['protocol_sha256'] or
            _hash(self.normalization)!=self.identity['normalization_sha256'] or _file_hash(__file__)!=IMPORTED_SHA):
            raise ValueError('fusion cohort source/config/content identity changed')
        if _hash(self.partitions)!=self.identity['partition_content_sha256']:raise ValueError('fusion cohort partition changed')
        try:
            if _input_paths(self.datasets)!=dict(self.input_paths):raise ValueError('fusion reader resolved input binding changed')
            for role,ds in self.datasets.items():
                if (type(ds) is not FusionLiteralDataset or len(ds)!=self.identity['counts'][role] or
                    ds.windows!=[w for w in self.partitions['windows'] if w['role']==role] or
                    _hash(ds.partitions)!=self.identity['partition_content_sha256'] or
                    _reader_binding(ds,self.input_paths)!=self.identity['reader_bindings_sha256'][role]):
                    raise ValueError('original fusion reader role/transform binding changed')
        except (KeyError,AttributeError,TypeError,OSError) as exc:raise ValueError('original fusion reader binding changed') from exc


def admit_fusion_cohort(data,crops,*,upstream,config,normalization,expected=None,chunk_rows=MAX_CHUNK_ROWS):
    """Hash complete actual inputs and scan every possible consumed numeric row.

    Expected receipts are recomputed against actual bytes; no model or RNG draw
    occurs. This does not decode every image, authenticate a publisher or admit
    the unavailable physically merged corpus. Original JPEG failures stay fatal.
    """
    if type(chunk_rows) is not int or not 1<=chunk_rows<=MAX_CHUNK_ROWS:raise ValueError('chunk_rows must be integer1..262144')
    root,_=_validated(_safe(upstream),config);data=_safe(data);crops=_safe(crops);protocol=fusion_protocol(root,config)
    if _hash(normalization)!=_hash(fusion_normalization(root)):raise ValueError('fusion normalization source binding differs')
    if expected is not None and (not isinstance(expected,dict) or expected.get('format')!=FORMAT or
        expected.get('sha256')!=_hash({k:v for k,v in expected.items() if k!='sha256'})):
        raise ValueError('invalid expected fusion input receipt')
    sources=_sources(root);guard=_StatGuard([data/'manifest.json',data/'metadata.npz',*sources.values()])
    partitions=build_fusion_partitions(LegacyStore(data))
    datasets={r:FusionLiteralDataset(data,partitions,crops,upstream=root,config=config,normalization=normalization,role=r) for r in ROLES}
    if any(not len(ds) for ds in datasets.values()):raise ValueError('all full fusion roles must be nonempty')
    paths=_input_paths(datasets)
    for path in paths.values():guard.add(path)
    guard.check();source_hashes={k:_stream_sha(p,guard) for k,p in sources.items()}
    payload={k:_stream_sha(p,guard) for k,p in paths.items()};sizes={k:p.stat().st_size for k,p in paths.items()}
    finite=_finite(datasets,chunk_rows,guard);counts={r:len(ds) for r,ds in datasets.items()}
    registered=(payload['metadata:manifest']==MANIFEST_SHA and payload['metadata:episodes']==METADATA_SHA and
        counts==FULL_COUNTS and len(datasets['train'].store.episodes)==41)
    identity=dict(format=FORMAT,scope='complete actual input bytes and numeric footprint/population only; no training/test permit',
        scientific_eligible=False,production_ready=False,full_paper_complete=False,
        available_corpus='released legacy EgoEMG only; not physically merged EgoEMG/ShowEE/Incre',
        protocol_sha256=protocol['sha256'],normalization=copy.deepcopy(normalization),normalization_sha256=_hash(normalization),
        partition_sha256=partitions['provenance']['partition_sha256'],partition_content_sha256=_hash(partitions),
        sources_sha256=source_hashes,payload_files_sha256=payload,payload_file_bytes=sizes,payload_file_count=len(paths),
        full_payload_content_hashed=True,counts=counts,unique_counts={r:len(ds)//ds.repeat for r,ds in datasets.items()},
        repeat_by_role={r:ds.repeat for r,ds in datasets.items()},metadata_matches_registered_legacy_release=registered,
        episode_records=len(datasets['train'].store.episodes),finite_footprints=finite,
        consumed_emg_nonfinite_entries=sum(v['consumed_emg_nonfinite_entries'] for v in finite.values()),
        possible_center_target_nonfinite_entries=sum(v['possible_center_target_nonfinite_entries'] for v in finite.values()),
        reader_bindings_sha256={r:_reader_binding(ds,paths) for r,ds in datasets.items()},
        finite_scope='unweighted union of every possible jitter footprint/center, both hands; not an observed epoch or error denominator')
    identity['sha256']=_hash(identity)
    if expected is not None and _hash(expected)!=_hash(identity):raise ValueError('actual fusion input content/source identity differs from receipt')
    result=FusionCohort(MappingProxyType(datasets),copy.deepcopy(partitions),identity,guard,root,copy.deepcopy(config),
        MappingProxyType(paths),copy.deepcopy(normalization),identity['sha256']);result.check();return result


def native_fusion_loaders(cohort):
    """Original native loaders/collation; explicit single rank and workers0."""
    cohort.check();_activate(cohort.upstream)
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0
    loaders={r:dm._make_loader(ds,shuffle=r=='train') for r,ds in cohort.datasets.items()}
    if any(l.batch_size!=200 or l.generator is not None or l.num_workers!=0 or l.drop_last for l in loaders.values()):
        raise ValueError('native fusion batch200 loader contract differs')
    cohort.check();return loaders
