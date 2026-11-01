"""Byte-bound original quarter-stride index selection, with no model permit.

The author filters on the center frame's split. Full windows crossing split
boundaries are intentionally retained and counted. These are distinct from
the standard independent, full-footprint evaluation roles.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from . import UPSTREAM_COMMIT
from .classic_emg import _activate,_file_hash,_hash
from .classic_emg_cohort import _safe,_source_paths,_StatGuard
from .data import LegacyStore

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
SPLITS=('user','gesture','both')
HANDS=('left','right')
COLUMNS=dict(sequence=np.int64,episode=np.int32,start=np.int64,end=np.int64,center=np.int64,
    split_id=np.int32,hand_index=np.int8,native_index=np.int64,group_index=np.int8)
LOCAL=('occlusion_dense_data','classic_emg','classic_emg_cohort','data','fusion_literal','rn18_fusion_literal')


def _source_check():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported dense selection source changed')


def _providers(root):
    paths={*_source_paths(root),root/'scripts/eval/evaluate_occlusion_fusion.py',
        *(ROOT/'egoemg_repro'/(name+'.py')for name in LOCAL)}
    return {('upstream/'+str(p.relative_to(root))if p.is_relative_to(root)else str(p.relative_to(ROOT))):_safe(p)
        for p in sorted(paths)}


def _recipe(upstream,config,recipe):
    if recipe=='vits_s':
        from .fusion_literal import _validated,EXPERIMENT
    elif recipe=='rn18_s':
        from .rn18_fusion_literal import _validated,EXPERIMENT
    else:raise ValueError('explicit vits_s or rn18_s dense recipe required')
    root,value=_validated(_safe(upstream),config)
    return root,value,EXPERIMENT


def dense_occlusion_protocol(upstream,config,*,recipe):
    """Validate the exact released fusion config, without instantiating models."""
    _source_check()
    if recipe not in('vits_s','rn18_s'):raise ValueError('unknown dense recipe')
    root=_safe(upstream);paths=_providers(root);guard=_StatGuard(paths.values())
    hashes={name:_file_hash(p)for name,p in paths.items()};guard.check()
    root,value,experiment=_recipe(root,config,recipe)
    dm=value['datamodule'];window=dm['val_test_window_length'];stride=dm['val_test_stride']//4
    expected=(7790,1947)if recipe=='vits_s'else(12000,3000)
    if (type(window)is not int or type(stride)is not int or(window,stride)!=expected):
        raise ValueError('exact native evaluation window/quarter stride required')
    if any(_file_hash(p)!=hashes[name]for name,p in paths.items()):raise ValueError('dense provider bytes changed')
    guard.check();_source_check()
    result=dict(format='original_dense_occlusion_index_contract_v1',recipe=recipe,upstream_revision=UPSTREAM_COMMIT,
        upstream_experiment=experiment,window_length=window,dense_stride=stride,
        original_eval_stride=dm['val_test_stride'],eval_center_stride=None,jitter=False,
        splits=list(SPLITS),hands=list(HANDS),repeat=1,max_samples_per_split=0,
        selection='native episode-start grid, center split only',
        ordering='split user/gesture/both, hand left/right, original native dataset index',
        full_window_split_filter=False,config_sha256=_hash(value),source_files_sha256=hashes,
        model_or_geometry_permit=False,scientific_eligible=False,full_paper_complete=False)
    result['sha256']=_hash(result);return result


def _native_view(store,upstream,window,stride,split,hand):
    """IO-only original index instance; an internal helper grants no permit."""
    if (type(window)is not int or type(stride)is not int or window<=0 or stride<=0
        or split not in('train',*SPLITS)or hand not in HANDS):raise ValueError('native index arguments differ')
    _activate(upstream)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    provider=Path(sys.modules[EgoEmgMemmapDataset.__module__].__file__).resolve()
    if provider!=upstream/'egoemg/datasets/egoemg_memmap_dataset.py':raise ValueError('native index provider differs')
    view=object.__new__(EgoEmgMemmapDataset)
    view.window_length=window;view.stride=stride;view.eval_center_stride=None;view.jitter=False
    view.allowed_splits=[split];view.allowed_subjects=None;view.allowed_episode_ids=None;view.excluded_episode_ids=None
    view.target_hand=hand;view._frame_split_available=True;view._splits=list(store.metadata['splits_split'])
    view._episode_id=list(store.metadata['episode_id'])
    view._episode_start_idx=np.asarray(store.metadata['episode_start_idx'])
    view._episode_end_idx=np.asarray(store.metadata['episode_end_idx'])
    view._frame_memmaps={'frame_split_id':store.field('frame_split_id')}
    view._build_window_index()
    return view


def _typed_sha(a):
    import hashlib,json
    h=hashlib.sha256(json.dumps([a.dtype.str,list(a.shape)],separators=(',',':')).encode())
    h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()


def native_dense_points(data,*,upstream,config,recipe,expected=None):
    """Rebuild every native heldout-center key and the actual input byte chain.

    Expected identities are checked against a fresh full reconstruction. This
    does not read EMG/labels/meshes, instantiate a learned model, or authorize
    downstream geometry/inference on a caller-supplied/resealed point packet.
    """
    protocol=dense_occlusion_protocol(upstream,config,recipe=recipe)
    root=_safe(upstream);data=_safe(data);providers=_providers(root)
    inputs={'manifest':_safe(data/'manifest.json'),'metadata':_safe(data/'metadata.npz')}
    guard=_StatGuard([*providers.values(),*inputs.values()]);guard.check()
    before={name:_file_hash(p)for name,p in providers.items()}
    if before!=protocol['source_files_sha256']:raise ValueError('dense source chain changed before selection')
    payload={name:_file_hash(p)for name,p in inputs.items()}
    store=LegacyStore(data);spec=store.fields.get('frame_split_id',{})
    if (spec.get('shape')!=[store.total_rows]or np.dtype(spec.get('dtype','float64')).kind not in'iu'):
        raise ValueError('integer one-dimensional split field required')
    inputs['frame_split_id']=_safe(data/spec.get('filename','frame_split_id.dat'));guard.add(inputs['frame_split_id'])
    payload['frame_split_id']=_file_hash(inputs['frame_split_id']);guard.check()
    labels=list(store.metadata.get('splits_split',[]))
    if (list(store.split_labels.values())!=['train',*SPLITS]or labels!=['train',*SPLITS]
        or list(store.metadata.get('splits_split_id',[]))!=[0,1,2,3]):raise ValueError('exact unique native split labels required')
    splits=store.field('frame_split_id')
    if (splits.flags.writeable or splits.mode!='r'or splits.offset!=0 or not splits.flags.c_contiguous
        or Path(splits.filename).resolve()!=inputs['frame_split_id']or np.any((splits<0)|(splits>3))):
        raise ValueError('native split values/readonly typed mapping differs')
    columns={name:[]for name in COLUMNS};groups=[];crossings=0;seen=set()
    for group,split in enumerate(SPLITS):
        sid=labels.index(split)
        for hi,hand in enumerate(HANDS):
            guard.check();view=_native_view(store,root,protocol['window_length'],protocol['dense_stride'],split,hand)
            groups.append(dict(split=split,hand=hand,points=len(view)))
            for idx in range(len(view)):
                ep,center=view._resolve_index_to_center(idx);start=center-protocol['window_length']//2;end=start+protocol['window_length']
                if (int(splits[center])!=sid or(ep,center,hi)in seen
                    or not store.episodes[ep]['start']<=start<end<=store.episodes[ep]['end']):
                    raise ValueError('unique native dense key/split/window differs')
                seen.add((ep,center,hi));crossings+=int(np.any(splits[start:end]!=sid))
                values=(len(columns['sequence']),ep,start,end,center,sid,hi,idx,group)
                for name,value in zip(columns,values):columns[name].append(value)
            guard.check()
    arrays={name:np.asarray(values,dtype=COLUMNS[name])for name,values in columns.items()}
    for a in arrays.values():a.setflags(write=False)
    if any(_file_hash(p)!=payload[name]for name,p in inputs.items()):raise ValueError('dense index input bytes changed')
    if any(_file_hash(p)!=before[name]for name,p in providers.items()):raise ValueError('dense provider bytes changed during selection')
    guard.check();_source_check()
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():raise ValueError('CPU-only dense selection required')
    identity=dict(format='original_dense_occlusion_complete_index_v1',protocol=protocol,
        input_files_sha256=payload,source_files_sha256=before,total_rows=store.total_rows,episode_records=len(store.episodes),
        unique_hand_points=len(seen),groups=groups,arrays_sha256={n:_typed_sha(a)for n,a in arrays.items()},
        windows_crossing_center_split=crossings,independent_unseen_input_footprints_proven=False,
        full_producer_footprint_intersections_measured=False,full_native_index_population_selected=True,
        available_corpus='supplied legacy-schema corpus; current108participant and missing modalities not established',
        learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,
        native_geometry_executed=False,model_or_geometry_permit=False,scientific_eligible=False,full_paper_complete=False)
    identity['sha256']=_hash(identity)
    if expected is not None and(not isinstance(expected,dict)or _hash(expected)!=_hash(identity)):
        raise ValueError('fresh complete native dense identity differs from expected receipt')
    return dict(identity=identity,points=arrays)
