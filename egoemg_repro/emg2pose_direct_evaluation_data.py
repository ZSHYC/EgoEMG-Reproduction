"""Original evaluation populations over admitted ordinary original files.

No model or signal window is evaluated during metadata admission. The original
dataset class and native evaluation factory remain unchanged. Callers own data
worker lifetimes; close this context only after its consumers have exited.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace

import numpy as np

from . import emg2pose_memory_evaluation_data as _kernel
from . import emg2pose_direct_runtime as runtime
from .emg2pose_direct_source import VerifiedDirectSource

FORMAT = 'emg2pose_direct_evaluation_data_identity_v1'
_ROOT = Path(__file__).parent
_FILES = ('emg2pose_direct_evaluation_data.py', 'emg2pose_memory_evaluation_data.py')
_IMPORTED = {name: runtime.storage._sha256(_ROOT/name) for name in _FILES}


def _sources(upstream):
    actual = {name: runtime.storage._sha256(_ROOT/name) for name in _FILES}
    if actual != _IMPORTED or actual['emg2pose_memory_evaluation_data.py'] != _kernel._IMPORTED_SHA:
        raise ValueError('imported direct evaluation data implementation changed')
    return dict(evaluation_data=actual,
        native_evaluation=runtime.storage._sha256(upstream/'egoemg/test_analysis.py'),
        native_dataset=runtime.storage._sha256(upstream/'egoemg/datasets/emg2pose_dataset.py'))


def _dataset_signature(dataset, source, native_dataset):
    if type(dataset) is not native_dataset or Path(dataset.memmap_dir).resolve() != source.index:
        raise ValueError('actual original evaluation dataset/index binding differs')
    source.assert_current()
    expected = {name: dict(path=str(source.index/info['filename']), dtype=info['dtype'], shape=list(info['shape']))
        for name, info in source.data_identity['fields'].items()}
    actual = {name: dict(path=spec['path'], dtype=np.dtype(spec['dtype']).str, shape=list(spec['shape']))
        for name, spec in dataset._memmap_specs.items()}
    if actual != expected:
        raise ValueError('original ordinary field path/dtype/shape binding differs')
    if dataset._memmaps and set(dataset._memmaps) != set(expected):
        raise ValueError('original lazy field view coverage differs')
    for name, value in dataset._memmaps.items():
        spec = expected[name]
        if (not isinstance(value, np.memmap) or value.mode != 'r' or value.flags.writeable
                or value._mmap.closed or value.offset != 0 or str(Path(value.filename).resolve()) != spec['path']
                or value.dtype.str != spec['dtype'] or list(value.shape) != spec['shape']):
            raise ValueError('readonly original lazy field view binding differs')
    return dict(options={f.name: _value(getattr(dataset, f.name)) for f in fields(native_dataset) if f.name != 'memmap_dir'},
        metadata={name: _value(getattr(dataset, name)) for name in _kernel._METADATA},
        blocks={name: _value(getattr(dataset, name)) for name in _kernel._BLOCKS},
        manifest_sha256=_hash(dataset._manifest), ordinary_field_specs=actual,
        normalization={name: _value(getattr(dataset, name)) for name in ('_emg_mean', '_emg_std', '_norm_ready')})


def _close(dataset):
    if hasattr(dataset, '_memmaps'):
        runtime.close_data_module(SimpleNamespace(val_dataset=dataset))


_namespace = dict(_kernel.__dict__, __name__=__name__, __file__=__file__, runtime=runtime,
    VerifiedSource=VerifiedDirectSource, _sources=_sources, _dataset_signature=_dataset_signature)


def _bind(function):
    bound = FunctionType(function.__code__, _namespace, name=function.__name__,
        argdefs=function.__defaults__, closure=function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = dict(function.__annotations__)
    return bound


for _name in ('evaluation_data_policy', '_hash', '_value', '_full_source', '_groups', '_text', '_npz_call', '_preflight'):
    _namespace[_name] = globals()[_name] = _bind(getattr(_kernel, _name))


class EvaluationData:
    def __init__(self, upstream, source, context, policy, groups, population, datasets, native_dataset):
        self._upstream = upstream; self._source = source; self.context = context; self._policy = policy
        self._groups = groups; self._population = population; self._datasets = tuple(datasets)
        self._native_dataset = native_dataset; self._closed = False
        self._identity = dict(format=FORMAT, role=context.split, policy=deepcopy(policy),
            runtime=runtime.runtime_identity(upstream, context.config, source=source),
            population=deepcopy(population), snapshot=self._snapshot(),
            inference_performed=False, scientific_complete=False, full_paper_complete=False)
        self._identity['sha256'] = _hash(self._identity)

    def close(self):
        self._closed = True
        first = None
        for dataset in self._datasets:
            try: _close(dataset)
            except Exception as error:
                if first is None: first = error
        if first is not None: raise first


for _name in ('_snapshot', 'check', '__enter__', '__exit__'):
    setattr(EvaluationData, _name, _bind(getattr(_kernel.EvaluationData, _name)))
for _name in ('config', 'identity', 'population'):
    setattr(EvaluationData, _name, property(_bind(getattr(_kernel.EvaluationData, _name).fget)))


def build_evaluation_data(upstream, source, metadata_csv, size, *, role, scope='full', selection='official_literal'):
    policy = evaluation_data_policy(scope, selection)
    if type(role) is not str or role not in ('val', 'test'):
        raise ValueError('val or test metadata role required')
    if type(source) is not VerifiedDirectSource:
        raise ValueError('an actual live VerifiedDirectSource is required')
    source.assert_current()
    if scope == 'full': _full_source(source.data_identity)
    upstream = runtime.storage._safe_path(upstream)
    config = runtime.compose_config(upstream, size, source.index, source=source, metadata_csv=metadata_csv)
    runtime.native._activate(upstream)
    from egoemg.test_analysis import EMG2PoseEvaluation
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    context = object.__new__(EMG2PoseEvaluation)
    context.config = config; context.split = role; context.conditions = ['generalization']; context.dataset_type = 'emg2pose'
    context.data_dir = str(source.index); context.memmap_dir = str(source.index)
    context.batch_size = policy['batch_size']; context.num_workers = policy['workers']
    context.window_length = 7790; context.skip_ik_failures = True; context.per_user = True
    def read_frame(instance, path): return _npz_call(EMG2PoseEvaluation._get_corpus_df_from_npz, instance, path)
    context._get_corpus_df_from_npz = MethodType(read_frame, context)
    try: context.df = context.get_corpus_df()
    finally: context.__dict__.pop('_get_corpus_df_from_npz', None)
    context.groupby = context.df.groupby(context.conditions)
    groups = _groups(context)
    if len(context.df) != source.data_identity['counts']['split'].get(role, 0):
        raise ValueError('native role metadata population differs')
    with np.load(source.index/'metadata.npz', allow_pickle=False) as opened:
        metadata = {name: opened[name] for name in opened.files}
    population = _preflight(metadata, groups, role, selection, Emg2PoseDataset)
    original = EMG2PoseEvaluation.get_dataloaders
    if original.__globals__['Emg2PoseDataset'] is not Emg2PoseDataset:
        raise ValueError('original dataset constructor global changed')
    created = []
    def construct(**kwargs):
        i = len(created)
        if i >= len(groups) or kwargs.get('allowed_sessions') != groups[i]['sessions']:
            raise ValueError('native ordinary factory group order differs')
        if selection == 'metadata_indices':
            kwargs.update(allowed_session_indices=tuple(groups[i]['rows']), allowed_splits=(role,))
        dataset = object.__new__(Emg2PoseDataset); created.append(dataset)
        def read_metadata(instance): return _npz_call(Emg2PoseDataset._load_metadata, instance)
        dataset._load_metadata = MethodType(read_metadata, dataset)
        try: Emg2PoseDataset.__init__(dataset, **kwargs)
        finally: dataset.__dict__.pop('_load_metadata', None)
        return dataset
    bound = FunctionType(original.__code__, dict(original.__globals__, Emg2PoseDataset=construct),
        name=original.__name__, argdefs=original.__defaults__, closure=original.__closure__)
    try:
        context.dataloaders = bound(context)
        if len(created) != len(groups): raise ValueError('native condition factory coverage differs')
        for group, dataset in zip(groups, created):
            entry = population[group['name']]
            if list(map(int, dataset._filter_session_indices())) != entry['selected_row_indices'] or len(dataset) < 1:
                raise ValueError('native constructed population/window coverage differs')
            entry.update(windows=len(dataset), window_plan_sha256=_hash({k: _value(getattr(dataset, k)) for k in _kernel._BLOCKS}))
        names = [g['name'] for g in groups]
        result = dict(by_condition=population, pairwise_row_overlap={a+'|'+b:
            len(set(population[a]['selected_row_indices']) & set(population[b]['selected_row_indices']))
            for i, a in enumerate(names) for b in names[i+1:]})
        handle = EvaluationData(upstream, source, context, policy, groups, result, created, Emg2PoseDataset)
        handle.check(); return handle
    except BaseException:
        for dataset in created: _close(dataset)
        raise
