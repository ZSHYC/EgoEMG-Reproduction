"""Original evaluation data selection with explicit population and FD admission.

This component constructs no model, runs no inference and grants no calibration
or scientific completion. It borrows the verified source; callers must supervise
their consumers before closing these independently owned dataset descriptors.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields,is_dataclass
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType,MethodType

import numpy as np

from . import emg2pose_memory_runtime as runtime
from .emg2pose_memory_index import VerifiedSource


SCOPE='metadata and dataset admission only; no model, inference or scientific qualification'
_IMPORTED_SHA=runtime.storage._sha256(__file__)
_METADATA=('_session_start','_session_length','_session_end','_session_user_id','_session_stage_id',
    '_session_side_id','_session_split_id','_session_held_out_user','_session_held_out_stage',
    '_session_id','_session_filename','_users','_stages','_sides','_splits')
_BLOCKS=('_block_session_idx','_block_start','_block_end','_block_cumsum')


def evaluation_data_policy(scope='full',selection='official_literal'):
    if type(scope) is not str or scope not in ('full','cpu_engineering','cuda_engineering'):
        raise ValueError('explicit full/cpu_engineering/cuda_engineering scope required')
    if type(selection) is not str or selection not in ('official_literal','metadata_indices'):
        raise ValueError('explicit official_literal or metadata_indices selection required')
    cpu=scope=='cpu_engineering'
    return dict(scope=scope,selection=selection,batch_size=1 if cpu else 320,workers=0 if cpu else 12,
        precision='32-true',window_length=7790,stride=7790,padding=[0,0],jitter=False,skip_ik_failures=True,
        factory_pin_memory=True,native_evaluation_pin_memory='device_is_cuda',multiprocessing_context=None,
        scope_description=SCOPE,inference_performed=False,scientific_complete=False,full_paper_complete=False)


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _value(value):
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float and math.isfinite(value):return value
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise ValueError('object metadata arrays are not admissible')
        array=np.ascontiguousarray(value)
        return dict(dtype=array.dtype.str,shape=list(array.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())
    if isinstance(value,Path):return str(value)
    if isinstance(value,(tuple,list)):return [_value(v) for v in value]
    if isinstance(value,dict):return {str(k):_value(v) for k,v in value.items()}
    if is_dataclass(value) and not isinstance(value,type):
        return dict(type=type(value).__module__+'.'+type(value).__qualname__,
                    fields={f.name:_value(getattr(value,f.name)) for f in fields(value)})
    raise ValueError('unsupported or nonfinite dataset/transform metadata')


def _full_source(value):
    if (value.get('cohort')!='full' or value.get('full_corpus_complete') is not True
            or value.get('engineering_only') is not False or value.get('file_count')!=25253
            or value.get('counts',{}).get('split')!={'train':17136,'val':1950,'test':6167}
            or value.get('archive_bytes')!=462824048640):
        raise ValueError('full evaluation data requires complete25253 original source and roles')


def _groups(context):
    result=[]
    for name,frame in context.groupby:
        name=name[0] if isinstance(name,tuple) and len(name)==1 else name
        if not isinstance(name,str) or not name:raise ValueError('native generalization group required')
        result.append(dict(name=name,rows=list(map(int,frame.index)),sessions=frame['session_id'].tolist()))
    if not result or len({g['name'] for g in result})!=len(result):raise ValueError('nonempty native condition groups required')
    return result


def _text(values):
    return [v.decode('utf-8') if isinstance(v,bytes) else str(v) for v in values]


def _npz_call(function,instance,*args):
    """Run original metadata code; close only the NPZ handles it opens here.

    Original metadata readers omit a close. A private numpy binding retains the
    exact function code and values while avoiding reliance on cycle collection.
    No original module globals or persistent instance methods are changed.
    """
    opened=[]
    class Numpy:
        def __getattr__(self,name):return getattr(np,name)
        def load(self,*a,**kwargs):
            value=np.load(*a,**kwargs)
            if not isinstance(value,np.lib.npyio.NpzFile):raise ValueError('verified metadata must be an NPZ file')
            opened.append(value);return value
    bound=FunctionType(function.__code__,dict(function.__globals__,np=Numpy()),
                       name=function.__name__,argdefs=function.__defaults__,closure=function.__closure__)
    try:return bound(instance,*args)
    finally:
        for value in opened:value.close()


def _preflight(metadata,groups,role,selection,native_dataset):
    recording=_text(metadata['session_session_id']);filenames=_text(metadata['session_filename'])
    roles=_text(metadata['splits_split']);row_roles=[roles[int(i)] for i in metadata['session_split_id']]
    generalization=_text(metadata['session_generalization']);population={}
    for group in groups:
        requested=group['rows']
        if (not requested or len(set(requested))!=len(requested) or any(i<0 or i>=len(recording) for i in requested)
                or any(row_roles[i]!=role or generalization[i]!=group['name'] for i in requested)
                or group['sessions']!=[recording[i] for i in requested]):
            raise ValueError('native requested groups differ from verified metadata rows')
        probe=object.__new__(native_dataset)
        probe._session_id=recording;probe._session_filename=filenames
        probe._session_split_id=metadata['session_split_id'];probe._splits=roles
        for name in ('allowed_session_indices','allowed_users','allowed_stages','allowed_sides','allowed_splits'):
            setattr(probe,name,None)
        probe.allowed_sessions=group['sessions']
        if selection=='metadata_indices':probe.allowed_session_indices=tuple(requested);probe.allowed_splits=(role,)
        selected=list(map(int,probe._filter_session_indices()))
        if not selected:raise ValueError('native selected metadata population is empty')
        counts=dict(sorted(Counter(row_roles[i] for i in selected).items()))
        if role=='val' and set(counts)!={'val'}:raise ValueError('validation selection contains non-val metadata roles')
        exact=set(selected)==set(requested)
        if selection=='metadata_indices' and not exact:raise ValueError('explicit metadata row selection differs')
        population[group['name']]=dict(requested_rows=len(requested),selected_rows=len(selected),
            requested_row_indices=requested,selected_row_indices=selected,
            extra_rows=len(set(selected)-set(requested)),omitted_rows=len(set(requested)-set(selected)),
            selected_by_role=counts,selected_by_condition=dict(sorted(Counter(generalization[i] for i in selected).items())),
            requested_selection_exact=exact)
    return population


def _sources(upstream):
    if runtime.storage._sha256(__file__)!=_IMPORTED_SHA:raise ValueError('imported evaluation data implementation changed')
    return dict(evaluation_data=_IMPORTED_SHA,
        native_evaluation=runtime.storage._sha256(upstream/'egoemg/test_analysis.py'),
        native_dataset=runtime.storage._sha256(upstream/'egoemg/datasets/emg2pose_dataset.py'))


def _dataset_signature(dataset,source,native_dataset):
    from .emg2pose_memory_dataset import SealedTarDataset,_fd_guard,_binding_guard
    if type(dataset) is not SealedTarDataset or Path(dataset.memmap_dir).resolve()!=source.index:
        raise ValueError('native evaluation dataset storage/index differs')
    _fd_guard(dataset._archive_descriptor,dataset.archive_handle_identity);_binding_guard(dataset.archive_handle_identity)
    options={f.name:_value(getattr(dataset,f.name)) for f in fields(native_dataset) if f.name!='memmap_dir'}
    manifest={k:dataset._manifest[k] for k in ('fields','sessions','archive_bytes','archive_sha256')}
    return dict(options=options,metadata={k:_value(getattr(dataset,k)) for k in _METADATA},
        blocks={k:_value(getattr(dataset,k)) for k in _BLOCKS},manifest_sha256=_hash(manifest),
        normalization={k:_value(getattr(dataset,k)) for k in ('_emg_mean','_emg_std','_norm_ready')})


class EvaluationData:
    def __init__(self,upstream,source,context,policy,groups,population,datasets,native_dataset):
        self._upstream=upstream;self._source=source;self.context=context;self._policy=policy
        self._groups=groups;self._population=population;self._datasets=tuple(datasets)
        self._native_dataset=native_dataset;self._closed=False
        signature=self._snapshot()
        self._identity=dict(format='emg2pose_evaluation_data_identity_v1',role=context.split,
            policy=deepcopy(policy),runtime=runtime.runtime_identity(upstream,context.config,source=source),
            population=deepcopy(population),snapshot=signature,inference_performed=False,scientific_complete=False)
        self._identity['sha256']=_hash(self._identity)

    @property
    def config(self):return self.context.config
    @property
    def identity(self):return deepcopy(self._identity)
    @property
    def population(self):return deepcopy(self._population)

    def _snapshot(self):
        context=self.context;policy=self._policy
        if (context.split not in ('val','test') or context.dataset_type!='emg2pose' or context.per_user is not True or
                context.conditions!=['generalization'] or
                context.batch_size!=policy['batch_size'] or context.num_workers!=policy['workers'] or
                context.window_length!=7790 or context.skip_ik_failures is not True or
                context.data_dir!=str(self._source.index) or context.memmap_dir!=str(self._source.index) or
                _groups(context)!=self._groups or len(context._emg2pose_datasets)!=len(self._datasets)
                or any(a is not b for a,b in zip(context._emg2pose_datasets,self._datasets))):
            raise ValueError('native context/profile/group binding differs')
        import torch
        if len(context.dataloaders)!=len(self._datasets):raise ValueError('native factory loader coverage differs')
        for loader,dataset in zip(context.dataloaders,self._datasets):
            if (loader.dataset is not dataset or loader.batch_size!=policy['batch_size'] or loader.num_workers!=policy['workers']
                    or loader.drop_last or not loader.pin_memory or loader.multiprocessing_context is not None
                    or loader.persistent_workers is not bool(policy['workers'])
                    or type(loader.sampler) is not torch.utils.data.SequentialSampler):
                raise ValueError('native metadata loader profile/order differs')
        return dict(sources=_sources(self._upstream),role=context.split,frame_sha256=_hash(context.df.to_dict(orient='split')),
                    datasets=[_dataset_signature(d,self._source,self._native_dataset) for d in self._datasets])

    def check(self):
        if self._closed:raise ValueError('evaluation data context is closed')
        self._source.assert_current()
        if (self._snapshot()!=self._identity['snapshot'] or
                runtime.runtime_identity(self._upstream,self.config,source=self._source)!=self._identity['runtime']):
            raise ValueError('evaluation data/config/native population changed after admission')

    def close(self):
        self._closed=True
        first=None
        for dataset in self._datasets:
            try:dataset.close()
            except Exception as error:
                if first is None:first=error
        if first is not None:raise first

    def __enter__(self):self.check();return self
    def __exit__(self,*_args):self.close()


def build_evaluation_data(upstream,source,metadata_csv,size,*,role,scope='full',selection='official_literal'):
    policy=evaluation_data_policy(scope,selection)
    if type(role) is not str or role not in ('val','test'):raise ValueError('val or test metadata role required')
    if type(source) is not VerifiedSource:raise ValueError('an actual live VerifiedSource is required')
    source.assert_current()
    if scope=='full':_full_source(source.data_identity)
    upstream=runtime.storage._safe_path(upstream)
    config=runtime.compose_config(upstream,size,source.index,source=source,metadata_csv=metadata_csv)
    runtime.native._activate(upstream)
    from egoemg.test_analysis import EMG2PoseEvaluation
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    from omegaconf import OmegaConf
    from .emg2pose_memory_dataset import SealedTarDataset
    context=object.__new__(EMG2PoseEvaluation)
    context.config=config;context.split=role;context.conditions=['generalization'];context.dataset_type='emg2pose'
    context.data_dir=str(source.index);context.memmap_dir=str(source.index)
    context.batch_size=policy['batch_size'];context.num_workers=policy['workers']
    context.window_length=7790;context.skip_ik_failures=True;context.per_user=True
    def read_frame(instance,path):return _npz_call(EMG2PoseEvaluation._get_corpus_df_from_npz,instance,path)
    context._get_corpus_df_from_npz=MethodType(read_frame,context)
    try:context.df=context.get_corpus_df()
    finally:context.__dict__.pop('_get_corpus_df_from_npz',None)
    context.groupby=context.df.groupby(context.conditions)
    groups=_groups(context)
    if len(context.df)!=source.data_identity['counts']['split'].get(role,0):raise ValueError('native role metadata population differs')
    with np.load(source.index/'metadata.npz',allow_pickle=False) as opened:metadata={k:opened[k] for k in opened.files}
    population=_preflight(metadata,groups,role,selection,Emg2PoseDataset)
    original=EMG2PoseEvaluation.get_dataloaders
    if original.__globals__['Emg2PoseDataset'] is not Emg2PoseDataset:raise ValueError('original dataset constructor global changed')
    proof=OmegaConf.to_container(config.dataset[role][0].archive_handle_identity,resolve=True)
    created=[]
    def construct(**kwargs):
        i=len(created)
        if i>=len(groups) or kwargs.get('allowed_sessions')!=groups[i]['sessions']:
            raise ValueError('native storage factory group order differs')
        if selection=='metadata_indices':
            kwargs.update(allowed_session_indices=tuple(groups[i]['rows']),allowed_splits=(role,))
        dataset=object.__new__(SealedTarDataset);created.append(dataset)
        def read_metadata(instance):return _npz_call(Emg2PoseDataset._load_metadata,instance)
        dataset._load_metadata=MethodType(read_metadata,dataset)
        try:SealedTarDataset.__init__(dataset,**kwargs,archive_fd=source.fileno(),archive_handle_identity=proof)
        finally:dataset.__dict__.pop('_load_metadata',None)
        return dataset
    bound=FunctionType(original.__code__,dict(original.__globals__,Emg2PoseDataset=construct),
                       name=original.__name__,argdefs=original.__defaults__,closure=original.__closure__)
    try:
        context.dataloaders=bound(context)
        if len(created)!=len(groups):raise ValueError('native condition factory coverage differs')
        for group,dataset in zip(groups,created):
            entry=population[group['name']]
            if list(map(int,dataset._filter_session_indices()))!=entry['selected_row_indices'] or len(dataset)<1:
                raise ValueError('native constructed population/window coverage differs')
            entry.update(windows=len(dataset),window_plan_sha256=_hash({k:_value(getattr(dataset,k)) for k in _BLOCKS}))
        names=[g['name'] for g in groups]
        result=dict(by_condition=population,pairwise_row_overlap={a+'|'+b:len(set(population[a]['selected_row_indices'])&set(population[b]['selected_row_indices']))
                    for i,a in enumerate(names) for b in names[i+1:]})
        handle=EvaluationData(upstream,source,context,policy,groups,result,created,Emg2PoseDataset)
        handle.check();return handle
    except BaseException:
        for dataset in created:dataset.close()
        raise
