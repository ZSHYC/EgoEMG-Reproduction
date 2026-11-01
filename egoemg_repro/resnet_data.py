"""Source-bound native ResNet data, reusing the original vision point methods."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import weakref

from .classic_emg import _activate, _file_hash, _hash
from .resnet_literal import _validated, resnet_protocol


ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
ROLES=('train','val','test')
FULL_COUNTS=dict(train=202020,val=3152,test=8612)
SCOPES=('full','cuda_engineering','cpu_engineering')
_COHORTS=weakref.WeakSet()
_DATA_FIELDS=('dataset','datamodule','transforms','egoemg_unified_memmap_dir',
    'egoemg_emg_layout','egoemg_emg2pose_channel_indices','egoemg_channel_interpolate',
    'skip_emg_loading','per_episode_crops_dir','vision_num_frames',
    'vision_frame_selection','vision_patch_size','center_frame_eval')


def _sealed(value):
    result=copy.deepcopy(value);result['sha256']=_hash(result);return result


def _loader_fields(value):
    workers=value['datamodule'].get('eval_num_workers')
    return dict(batch_size=value['batch_size'],num_workers=value['num_workers'],
        eval_num_workers=value['num_workers'] if workers is None else workers)


def _data_projection(upstream,config):
    from .vision_literal import compose_vision, _validated as validate_vision, vision_protocol
    root,variant,actual=_validated(upstream,config)
    companion=compose_vision(root,'vit_small');_,_,expected=validate_vision(root,companion)
    left={k:actual[k]for k in _DATA_FIELDS};right={k:expected[k]for k in _DATA_FIELDS}
    if _hash(left)!=_hash(right):raise ValueError('ResNet and vision companion numerical data projection differs')
    return companion,_sealed(dict(format='original_resnet_data_projection_v1',variant=variant,
        companion_role='original vision-S data only; no ViT construction or training',
        compared_fields=list(_DATA_FIELDS),data_fields_sha256=_hash(left),
        loader_only=dict(resnet=_loader_fields(actual),companion=_loader_fields(expected)),
        resnet_protocol_sha256=resnet_protocol(root,config)['sha256'],
        companion_protocol_sha256=vision_protocol(root,companion)['sha256'],scientific_eligible=False))


def _sources(upstream):
    from .vision_literal_cohort import _sources as common_sources
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported ResNet data source changed')
    paths=common_sources(upstream)
    for name in ('egoemg_repro/resnet_data.py','egoemg_repro/resnet_literal.py'):
        paths[name]=ROOT/name
    return paths


def _identity(common,projection,sources):
    value=common.identity
    registered=(value['metadata_matches_registered_legacy_release'] is True and value['counts']==FULL_COUNTS
        and value['episode_records']==41 and value['repeat_by_role']=={r:2 for r in ROLES}
        and value['full_payload_content_hashed'] is True)
    return _sealed(dict(format='original_resnet_data_admission_v1',
        scope='actual point data only; no model, GPU, training or paper-equivalence permit',
        projection=projection,common_data=value,sources_sha256=sources,counts=value['counts'],
        unique_counts=value['unique_counts'],full_payload_content_hashed=value['full_payload_content_hashed'],
        registered_full_legacy=registered,scientific_eligible=False,production_ready=False,full_paper_complete=False))


def _typed_layout(array):
    return (array.dtype.str,tuple(array.shape),tuple(array.strides),int(array.offset),bool(array.flags.writeable))


def _reader_layouts(common):
    return tuple((role,hand,name,_typed_layout(array))
        for role,ds in sorted(common.datasets.items())
        for hand,view in zip(('left','right'),ds.views)
        for name,array in sorted(view._frame_memmaps.items()))


def _check_readers(common,crops,layouts):
    import numpy as np
    from .vision_literal_data import VisionLiteralDataset, _PointView
    from .vision_literal import vision_protocol
    _activate(common.upstream)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    protocol=vision_protocol(common.upstream,common.config)
    fields={'generated_joint_angles_left','generated_joint_angles_right','generated_label_valid',
        'mocap_left_wrist_pitch','mocap_left_wrist_yaw','mocap_right_wrist_pitch','mocap_right_wrist_yaw',
        'image_head_frame_index'}
    expected={(role,hand,name):layout for role,hand,name,layout in layouts}
    for role,ds in common.datasets.items():
        if (type(ds) is not VisionLiteralDataset or ds.role!=role or ds.repeat!=2 or ds.protocol!=protocol
            or len(ds.views)!=2 or ds.native is not EgoEmgMemmapDataset
            or ds.frame_mapping_field not in ('image_head_frame_index','image_webcam_frame_index')):
            raise ValueError('actual ResNet companion dataset layout changed')
        for hand,view in zip(('left','right'),ds.views):
            if (type(view) is not _PointView or view.target_hand!=hand or view.store is not ds.store
                or view.native is not ds.native or view.crops!=crops or view.vision_patch_size!=256
                or view.emg_layout!='emg2pose_interpolate16'
                or view._episode_id!=[e['id']for e in ds.store.episodes]
                or not np.array_equal(view.vision_mean,255.*np.array([.485,.456,.406],np.float32))
                or not np.array_equal(view.vision_std,255.*np.array([.229,.224,.225],np.float32))
                or set(view._frame_memmaps)!=fields):
                raise ValueError('actual ResNet point-reader normalization/index/layout changed')
            for name,array in view._frame_memmaps.items():
                field=ds.frame_mapping_field if name=='image_head_frame_index' else name
                if (not isinstance(array,np.memmap) or array.mode!='r' or array is not ds.store.field(field)
                    or Path(array.filename).resolve()!=common.input_paths['field:'+field]
                    or _typed_layout(array)!=expected[(role,hand,name)]):
                    raise ValueError('actual ResNet point target/mask/mapping changed')


@dataclass(frozen=True,eq=False)
class ResnetCohort:
    identity:dict
    common:object
    upstream:Path
    config:object
    crops:Path
    _seal:str
    _layouts:tuple

    @property
    def datasets(self):return self.common.datasets

    @property
    def partitions(self):return self.common.partitions

    @property
    def guard(self):return self.common.guard

    def check(self,*,require_full=False):
        from omegaconf import OmegaConf
        if type(require_full) is not bool:raise ValueError('full requirement must be boolean')
        if self not in _COHORTS:raise ValueError('actual registered ResNet admission required')
        self.common.check();_check_readers(self.common,self.crops,self._layouts)
        companion,projection=_data_projection(self.upstream,self.config)
        if OmegaConf.to_container(companion,resolve=True)!=OmegaConf.to_container(self.common.config,resolve=True):
            raise ValueError('ResNet common companion configuration changed')
        sources={name:_file_hash(path)for name,path in _sources(self.upstream).items()}
        expected=_identity(self.common,projection,sources)
        if self.identity!=expected or expected['sha256']!=self._seal:raise ValueError('ResNet data identity changed')
        if require_full and not expected['registered_full_legacy']:raise ValueError('complete registered full ResNet cohort required')
        self.guard.check()


def admit_resnet_cohort(data,crops,*,upstream,config,expected=None,require_full=True,chunk_rows=262144):
    if type(require_full) is not bool:raise ValueError('full requirement must be boolean')
    from .classic_emg_cohort import _StatGuard
    from .vision_literal_cohort import admit_vision_cohort
    root,_,_=_validated(upstream,config);companion,projection=_data_projection(root,config)
    paths=_sources(root);outer=_StatGuard(paths.values())
    sources={name:_file_hash(path)for name,path in paths.items()}
    common=admit_vision_cohort(data,crops,upstream=root,config=companion,chunk_rows=chunk_rows)
    outer.check()
    for path,stamp in outer.paths.items():
        if path in common.guard.paths and common.guard.paths[path]!=stamp:raise ValueError('source/input generation changed')
        common.guard.paths[path]=stamp
    common.guard.parents.update(outer.parents)
    value=_identity(common,projection,sources)
    if expected is not None and expected!=value:raise ValueError('recomputed ResNet content identity differs from receipt')
    result=ResnetCohort(value,common,root,copy.deepcopy(config),Path(crops).resolve(),value['sha256'],_reader_layouts(common))
    _COHORTS.add(result);result.check(require_full=require_full);return result


def native_resnet_loaders(cohort,*,scope='full'):
    if type(scope) is not str or scope not in SCOPES:raise ValueError('explicit supported ResNet scope required')
    if type(cohort) is not ResnetCohort or cohort not in _COHORTS:raise ValueError('actual admitted ResNet cohort required')
    cpu=scope=='cpu_engineering';cohort.check(require_full=not cpu);_activate(cohort.upstream)
    from egoemg.datamodule import make_data_module
    from torch.utils.data import Subset
    protocol=resnet_protocol(cohort.upstream,cohort.config);dm=make_data_module(cohort.config)
    if cpu:dm.batch_size=2;dm.num_workers=dm.eval_num_workers=0
    loaders={}
    for role,ds in cohort.datasets.items():
        if cpu:ds=Subset(ds,range(min(len(ds),4)))
        elif scope=='cuda_engineering' and role=='train':
            ds=Subset(ds,range(2*protocol['batch_size_by_role'][role]))
        workers=dm.num_workers if role=='train' else dm.eval_num_workers
        loader=dm._make_loader(ds,shuffle=role=='train',num_workers=workers)
        if (loader.batch_size!=(2 if cpu else protocol['batch_size_by_role'][role])
            or loader.num_workers!=(0 if cpu else protocol['workers_by_role'][role])
            or loader.generator is not None or loader.drop_last or loader.pin_memory
            or (not cpu and (loader.prefetch_factor!=2 or not loader.persistent_workers))):
            raise ValueError('original ResNet native loader execution differs')
        loaders[role]=loader
    cohort.check(require_full=not cpu);return loaders
