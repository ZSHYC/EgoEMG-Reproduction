"""Exact ordinary Figure7 selections with original, producer-specific inputs.

This is read-only IO, not permission to construct models or claim a figure.
The registered selection, complete payload bytes and native reader semantics
remain bound for every sample. Dense datasets and their grids are unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np

from .classic_emg import _activate, _hash
from .classic_emg_cohort import _safe, _StatGuard
from .data import LegacyStore, file_sha256
from .emg_initializer_literal import compose_emg_initializer, emg_initializer_normalization
from .fusion_literal_cohort import _numeric_paths, _bind_crop_generations
from .fusion_literal_data import _FusionPointView, fusion_normalization
from .native_ik_inputs import _Guard
from .occlusion_dense_data import _recipe, _typed_sha
from .occlusion_dense_inputs import DensePredictionDataset
from .occlusion_dense_models import _code
from .qualitative_data import QualitativeSelection, _ADMITTED as _SELECTED, _paths
from .rn18_fusion_data import _Rn18PointView, rn18_fusion_normalization
from .vision_literal_data import _crop_path, _schema

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=file_sha256(__file__)
REFERENCE=ROOT/'results/native-dense-occlusion/paired-full-io-complete-20261101.json'
REFERENCE_SHA='387735569301d6714b205c381c82ca1b77d3007b1e745dc0bc5dc91cfe4f1456'
LOCAL=('qualitative_inputs','occlusion_dense_inputs','occlusion_dense_models','native_ik_inputs',
    'fusion_literal_data','fusion_literal_cohort','vision_literal_data','rn18_fusion_data',
    'emg_initializer_literal','crops','classic_emg','classic_emg_cohort','data')
_ADMITTED=weakref.WeakKeyDictionary()


def _reader_binding(io,upstream):
    for name in ('_getitem_center_supervised','_getitem_center_supervised_emg'):
        method=inspect.getattr_static(io.native,name)
        if (not inspect.isfunction(method) or method.__code__!=
            _code(upstream/'egoemg/datasets/egoemg_memmap_dataset.py','EgoEmgMemmapDataset',name)):
            raise ValueError('unmodified native qualitative reader required')
    return DensePredictionDataset._reader_binding(io)


def _state(value):
    if type(value) is not QualitativeInputs or value not in _ADMITTED:
        raise ValueError('registered ordinary qualitative input object required')
    return _ADMITTED[value]


@dataclass(frozen=True,eq=False)
class QualitativeInputs:
    @property
    def identity(self):
        self.check();return copy.deepcopy(_state(self)['identity'])

    @property
    def selection(self):
        self.check();return _state(self)['selection']

    @property
    def paths(self):
        self.check();return dict(_state(self)['paths'])

    def check(self):
        s=_state(self);s['selection'].check();s['guard'].check()
        if (file_sha256(__file__)!=IMPORTED_SHA or _hash(s['identity'])!=s['identity_hash']
            or _reader_binding(s['io'],s['upstream'])!=s['reader_binding']
            or any(a.flags.writeable or _typed_sha(a)!=s['point_hashes'][n] for n,a in s['points'].items())):
            raise ValueError('ordinary qualitative identity, keys or native reader binding changed')

    def __len__(self):
        self.check();return len(_state(self)['points']['sequence'])

    def hash_payloads(self):
        self.check();s=_state(self);actual={}
        for n,p in s['paths'].items():
            s['guard'].check();actual[n]=file_sha256(p);s['guard'].check()
        if actual!=s['identity']['complete_payload_sha256']:
            raise ValueError('complete qualitative input payload bytes changed')
        self.check();return actual

    def __getitem__(self,index):
        self.check();s=_state(self)
        if type(index) is not int or not 0<=index<len(s['points']['sequence']):raise IndexError(index)
        p=s['points'];io=s['io']
        ep,start,end,center,hi=(int(p[n][index])for n in ('episode','start','end','center','hand_index'))
        fusion=io.native._getitem_center_supervised(io.fusion_views[hi],ep,start,end,center)
        emg=io.native._getitem_center_supervised_emg(io.standalone_views[hi],ep,start,end)
        for packet,channels in ((fusion,s['identity']['fusion_input_channels']),(emg,8)):
            if (packet['emg'].dtype!=np.float32 or packet['emg'].shape!=(channels,end-start)
                or packet['joint_angles'].dtype!=np.float32 or packet['joint_angles'].shape!=(22,1)
                or not np.isfinite(packet['emg']).all() or not np.isfinite(packet['joint_angles']).all()
                or packet['label_valid_mask'].dtype!=np.bool_ or packet['label_valid_mask'].shape!=(1,)):
                raise ValueError('complete finite native qualitative EMG/center targets and boolean masks required')
        if (not np.array_equal(fusion['joint_angles'],emg['joint_angles'])
            or not np.array_equal(fusion['label_valid_mask'],emg['label_valid_mask'])
            or fusion['vision_img'].dtype!=np.float32 or fusion['vision_img'].shape!=(3,256,256)
            or not np.isfinite(fusion['vision_img']).all()
            or fusion['vision_valid_mask'].dtype!=np.bool_ or fusion['vision_valid_mask'].shape!=(1,)):
            raise ValueError('same native center labels/masks and finite canonical RGB crop required')
        key=dict(sequence=index,split=('user','gesture','both')[int(p['group_index'][index])],
            hand=('left','right')[hi],ep_idx=ep,center_idx=center,window_start=start,window_end=end,
            native_index=int(p['native_index'][index]),group_index=int(p['group_index'][index]))
        typed={'fusion:'+n:_typed_sha(fusion[n])for n in ('emg','emg_channel_mask','vision_img',
            'vision_valid_mask','vision_frame_indices','joint_angles','label_valid_mask')}
        typed.update({'standalone:'+n:_typed_sha(emg[n])for n in
            ('emg','emg_channel_mask','joint_angles','label_valid_mask')})
        self.check()
        return dict(key=key,fusion=fusion,standalone_emg=emg,typed_input_sha256=typed,
            verified_input_sha256=_hash(typed))


def admit_qualitative_inputs(selection,*,data,crops,upstream,config,recipe):
    if type(selection) is not QualitativeSelection or selection not in _SELECTED:
        raise ValueError('registered original qualitative selection required')
    selection.check();chosen=selection.identity
    upstream,value,_=_recipe(upstream,config,recipe)
    if (chosen['protocol']['recipe']!=recipe or chosen['protocol']['config_sha256']!=_hash(value)):
        raise ValueError('same selected qualitative recipe/config required')
    data,crops=_safe(data),_safe(crops)
    selected_paths=_SELECTED[selection]['guard'].stamps
    if any(_safe(data/n) not in selected_paths for n in ('manifest.json','metadata.npz')):
        raise ValueError('same registered qualitative source root required')
    sources=_paths(upstream)
    sources.update({'egoemg_repro/'+n+'.py':_safe(ROOT/'egoemg_repro'/(n+'.py'))for n in LOCAL})
    if not chosen['engineering']:sources['complete-raw-input-reference']=_safe(REFERENCE)
    initial=_Guard([*sources.values(),data/'manifest.json',data/'metadata.npz'],[])
    store=LegacyStore(data);frame=_schema(store);paths=_numeric_paths(store)
    paths.update({'metadata:manifest':_safe(data/'manifest.json'),'metadata:episodes':_safe(data/'metadata.npz')})
    databases={}
    # All episode payloads authenticate against the already complete released
    # source reference. Only explicitly selected windows are consumed below.
    for ep,record in enumerate(store.episodes):
        database=_crop_path(crops,record['id']);payload=_safe(database/'data.mdb' if database.is_dir()else database)
        paths['crop:'+str(ep).zfill(6)]=payload;databases[database]=payload
    guard=_Guard([*sources.values(),*paths.values()],[])
    cache_guard=_StatGuard(paths.values());_bind_crop_generations(databases,cache_guard)
    hashes={}
    for n,p in paths.items():guard.check();hashes[n]=file_sha256(p);guard.check()
    if not chosen['engineering']:
        if file_sha256(REFERENCE)!=REFERENCE_SHA:raise ValueError('fixed complete raw qualitative reference required')
        if hashes!=json.loads(REFERENCE.read_text())['complete_raw_input_files_sha256']:
            raise ValueError('same complete released raw input bytes required')
    emg_config=compose_emg_initializer(upstream)
    standalone_norm=emg_initializer_normalization(upstream)
    fusion_norm=fusion_normalization(upstream)if recipe=='vits_s'else rn18_fusion_normalization(upstream,config)
    for asset in {standalone_norm['asset'],fusion_norm['asset']}:
        path=_safe(upstream/asset.removeprefix('upstream://'));sources['upstream/'+asset]=path
    guard=_Guard([*guard.stamps,*sources.values()],[])
    _activate(upstream)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from egoemg.transforms import Compose
    from hydra.utils import instantiate
    io=SimpleNamespace(store=store,frame_mapping_field=frame,native=EgoEmgMemmapDataset,
        paths=paths,fusion_views=[],standalone_views=[])
    length=chosen['protocol']['window_length']
    for hand in ('left','right'):
        cls=_FusionPointView if recipe=='vits_s'else _Rn18PointView
        fusion=cls(store,crops,hand,frame,io.native,fusion_norm,Compose([instantiate(t)for t in config.transforms.test]))
        emg=_Rn18PointView(store,crops,hand,frame,io.native,standalone_norm,Compose([instantiate(t)for t in emg_config.transforms.val]))
        fusion.window_length=emg.window_length=length
        io.fusion_views.append(fusion);io.standalone_views.append(emg)
    reader=_reader_binding(io,upstream)
    source_hashes={n:file_sha256(p)for n,p in sources.items()}
    if source_hashes['egoemg_repro/qualitative_inputs.py']!=IMPORTED_SHA:
        raise ValueError('imported ordinary qualitative input source differs')
    guard.check();initial.check();cache_guard.check();selection.check()
    identity=dict(format='complete_source_native_ordinary_qualitative_inputs_v1',recipe=recipe,
        engineering=chosen['engineering'],selection_identity_sha256=_hash(chosen),
        selected_hand_points=chosen['selected_hand_points'],window_length=length,
        ordinary_stride=chosen['protocol']['stride'],windows_crossing_center_split=chosen['windows_crossing_center_split'],
        fusion_input_channels=value['module']['featurizer']['conv_blocks'][0]['in_channels'],
        standalone_input_channels=8,fusion_normalization=fusion_norm,standalone_normalization=standalone_norm,
        complete_payload_sha256=hashes,complete_payload_files=len(paths),
        complete_payload_bytes=sum(p.stat().st_size for p in paths.values()),
        complete_raw_input_reference_sha256=REFERENCE_SHA if not chosen['engineering']else None,
        source_files_sha256=source_hashes,reader_binding_sha256=reader,
        crop_and_label_masks_retained=True,left_crop_already_canonical=True,
        native_point_wrist_conversion=True,standalone_window_policy='exact selected fusion window',
        repeat=1,jitter=False,full_window_split_filter=False,trained_model_or_inference_permit=False,
        figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False)
    result=QualitativeInputs();points=selection.points
    _ADMITTED[result]=dict(selection=selection,guard=guard,identity=identity,identity_hash=_hash(identity),
        points=points,point_hashes=chosen['arrays_sha256'],io=io,paths=dict(paths),
        reader_binding=reader,upstream=upstream)
    result.check();return result
