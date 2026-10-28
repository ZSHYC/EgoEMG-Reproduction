"""Source-bound WiLoR data and fresh native fit boundaries.

The vision-S companion admits only unchanged point-reader data semantics. It
does not initialize a DINO model or grant WiLoR training/CUDA qualification.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import weakref

from .classic_emg import _activate,_file_hash,_hash
from .wilor_literal import _validated,wilor_protocol

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FULL_COUNTS=dict(train=202020,val=3152,test=8612)
ROLES=('train','val','test')
SCOPES=('full','cuda_engineering','cpu_engineering')
_COHORTS=weakref.WeakSet()
_DATA_FIELDS=('dataset','datamodule','transforms','egoemg_unified_memmap_dir',
    'egoemg_emg_layout','egoemg_emg2pose_channel_indices','egoemg_channel_interpolate',
    'skip_emg_loading','per_episode_crops_dir','vision_num_frames',
    'vision_frame_selection','vision_patch_size','center_frame_eval')


def _sealed(value):
    value=copy.deepcopy(value);value['sha256']=_hash(value);return value


def wilor_policy(*,scope='full'):
    if type(scope) is not str or scope not in SCOPES:raise ValueError('explicit supported WiLoR scope required')
    full=scope=='full';cpu=scope=='cpu_engineering'
    return _sealed(dict(format='native_wilor_fit_policy_v1',scope=scope,max_epochs=30 if full else 2,
        precision='32-true' if cpu else 'bf16-mixed',batch_size_by_role={r:1 if cpu else 18 for r in ROLES},
        engineering_train_samples=None if full else 2 if cpu else 36,engineering_eval_samples=2 if cpu else None,
        full_counts=FULL_COUNTS.copy(),monitor='val_mae',monitor_mode='min',patience=30,
        save_top_k=3,save_last=True,gradient_clip_val=1,scheduler_T_max=30,scheduler_eta_min=1e-5,
        scheduler_behavior='constant_initial_lr_equals_eta_min',num_sanity_val_steps=2 if full else 0,
        native_global_loader_rng=True,num_workers=0,single_rank=True,checkpoint_continuation_supported=False,
        bitwise_training_trajectory_guaranteed=False,scientific_eligible=False,full_paper_complete=False))


def _data_projection(upstream,config):
    from .vision_literal import compose_vision,_validated as validate_vision,vision_protocol
    root,actual=_validated(upstream,config)
    companion=compose_vision(root,'vit_small');_,_,expected=validate_vision(root,companion)
    left={name:actual[name] for name in _DATA_FIELDS};right={name:expected[name] for name in _DATA_FIELDS}
    if _hash(left)!=_hash(right):raise ValueError('WiLoR and original S companion data projection differs')
    return companion,_sealed(dict(format='native_wilor_data_projection_v1',
        companion_role='original vision-S data only; no DINO model/training/CUDA permit',
        compared_fields=list(_DATA_FIELDS),data_fields_sha256=_hash(left),
        wilor_protocol_sha256=wilor_protocol(root,config)['sha256'],
        companion_protocol_sha256=vision_protocol(root,companion)['sha256'],
        scientific_eligible=False,full_paper_complete=False))


def _data_sources(root):
    from .vision_literal_cohort import _sources
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported WiLoR training source changed')
    paths=_sources(root)
    for name in ('egoemg_repro/wilor_native_train.py','egoemg_repro/wilor_literal.py'):
        paths[name]=ROOT/name
    return paths


def _data_identity(common,projection,sources):
    v=common.identity
    registered=(v['metadata_matches_registered_legacy_release'] is True and v['counts']==FULL_COUNTS and
        v['episode_records']==41 and v['repeat_by_role']=={r:2 for r in ROLES})
    return _sealed(dict(format='native_wilor_data_admission_v1',
        scope='complete actual native point data only; no model/GPU/training or paper equivalence',
        projection=projection,common_data=copy.deepcopy(v),sources_sha256=sources,
        counts=v['counts'].copy(),unique_counts=v['unique_counts'].copy(),
        full_payload_content_hashed=v['full_payload_content_hashed'],registered_full_legacy=registered,
        scientific_eligible=False,production_ready=False,full_paper_complete=False))


def _check_readers(common):
    import numpy as np
    from .vision_literal_data import VisionLiteralDataset,_PointView
    from .vision_literal import vision_protocol
    protocol=vision_protocol(common.upstream,common.config)
    for role,ds in common.datasets.items():
        if (type(ds) is not VisionLiteralDataset or ds.role!=role or ds.repeat!=2 or ds.protocol!=protocol or
            len(ds.views)!=2 or ds.frame_mapping_field not in ('image_head_frame_index','image_webcam_frame_index')):
            raise ValueError('actual original WiLoR companion dataset layout changed')
        for hand,view in zip(('left','right'),ds.views):
            if (type(view) is not _PointView or view.target_hand!=hand or view.store is not ds.store or
                view.native is not ds.native or view.vision_patch_size!=256 or view.emg_layout!='emg2pose_interpolate16' or
                not np.array_equal(view.vision_mean,255.*np.array([.485,.456,.406],np.float32)) or
                not np.array_equal(view.vision_std,255.*np.array([.229,.224,.225],np.float32))):
                raise ValueError('actual original WiLoR point-reader normalization/layout changed')


@dataclass(frozen=True,eq=False)
class WilorCohort:
    identity:dict
    common:object
    upstream:Path
    config:object
    _seal:str

    @property
    def datasets(self):return self.common.datasets
    @property
    def partitions(self):return self.common.partitions
    @property
    def guard(self):return self.common.guard

    def check(self,*,require_full=False):
        if type(require_full) is not bool:raise ValueError('full requirement must be boolean')
        if self not in _COHORTS:raise ValueError('actual admitted WiLoR data object required')
        self.common.check();_check_readers(self.common)
        companion,projection=_data_projection(self.upstream,self.config)
        from omegaconf import OmegaConf
        if OmegaConf.to_container(companion,resolve=True)!=OmegaConf.to_container(self.common.config,resolve=True):
            raise ValueError('WiLoR common data companion configuration changed')
        sources={name:_file_hash(path) for name,path in _data_sources(self.upstream).items()}
        expected=_data_identity(self.common,projection,sources)
        if self.identity!=expected or expected['sha256']!=self._seal:raise ValueError('WiLoR outer data identity changed')
        if require_full and not expected['registered_full_legacy']:
            raise ValueError('complete registered full legacy WiLoR data required')
        self.guard.check()


def admit_wilor_cohort(data,crops,*,upstream,config,expected=None,require_full=True,chunk_rows=262144):
    if type(require_full) is not bool:raise ValueError('full requirement must be boolean')
    root,_=_validated(upstream,config);companion,projection=_data_projection(root,config)
    from .classic_emg_cohort import _StatGuard
    from .vision_literal_cohort import admit_vision_cohort
    paths=_data_sources(root);outer=_StatGuard(paths.values())
    hashes={name:_file_hash(path) for name,path in paths.items()}
    # expected is always compared with freshly recomputed real content below.
    common=admit_vision_cohort(data,crops,upstream=root,config=companion,chunk_rows=chunk_rows)
    outer.check()
    for path,stamp in outer.paths.items():
        if path in common.guard.paths and common.guard.paths[path]!=stamp:raise ValueError('data source generation changed')
        common.guard.paths[path]=stamp
    common.guard.parents.update(outer.parents)
    identity=_data_identity(common,projection,hashes)
    if expected is not None and expected!=identity:raise ValueError('actual WiLoR input content identity differs from receipt')
    result=WilorCohort(identity,common,root,copy.deepcopy(config),identity['sha256']);_COHORTS.add(result)
    result.check(require_full=require_full);return result


def native_wilor_loaders(cohort,*,scope='full'):
    policy=wilor_policy(scope=scope)
    if type(cohort) is not WilorCohort or cohort not in _COHORTS:raise ValueError('actual admitted WiLoR cohort required')
    cohort.check(require_full=scope!='cpu_engineering');_activate(cohort.upstream)
    from egoemg.datamodule import make_data_module
    from torch.utils.data import Subset
    dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0
    dm.batch_size=policy['batch_size_by_role']['train'];loaders={}
    for role,ds in cohort.datasets.items():
        limit=policy['engineering_train_samples'] if role=='train' else policy['engineering_eval_samples']
        if limit is not None:ds=Subset(ds,range(min(len(ds),limit)))
        loaders[role]=dm._make_loader(ds,shuffle=role=='train')
    if any(loader.batch_size!=policy['batch_size_by_role'][role] or loader.generator is not None or
        loader.num_workers!=0 or loader.drop_last for role,loader in loaders.items()):
        raise ValueError('original WiLoR native loader execution differs')
    cohort.check(require_full=scope!='cpu_engineering');return loaders
