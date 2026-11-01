"""Exact cotrain-S full trajectories on the fixed RN18 fusion holdouts.

Only IO/partition/worker transport differs from the closed ViT-compatible
producer. Native reader, transforms, normalization and returned trajectory
math remain unchanged. This module grants no trained-initializer eligibility.
"""
from __future__ import annotations

import copy
import importlib
import inspect
from pathlib import Path
import types

import numpy as np

from .classic_emg import _activate,_file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard
from .data import LegacyStore,union_intervals
from .emg_initializer_literal import _validated,emg_initializer_protocol,emg_initializer_normalization,NORM_ASSET
from .emg_initializer_data import EmgInitializerDataset,_schema,_view,W,STRIDE
from .emg_initializer_cohort import _reader_binding
from .rn18_fusion_data import build_rn18_fusion_partitions
from .fusion_initializer_handoff import _intersection

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
ROLES=('train','val')
SCOPES=('full','cuda_engineering','cpu_engineering')
_COMPILED_PROVIDERS={}


def _loaded_provider_exact(name,path):
    """Compare loaded declared callables with approved bytes, without executing them."""
    module=importlib.import_module(name);path=Path(path)
    if Path(module.__file__).resolve()!=path.resolve():raise ValueError('imported reader provider origin differs')
    digest=_file_hash(path);key=(str(path),digest)
    if key not in _COMPILED_PROVIDERS:
        _COMPILED_PROVIDERS[key]=compile(path.read_bytes(),module.__file__,'exec',dont_inherit=True)
    def check(code,namespace):
        for part in code.co_consts:
            if not isinstance(part,types.CodeType)or part.co_name.startswith('<'):continue
            value=inspect.getattr_static(namespace,part.co_name,None)
            if not part.co_flags&inspect.CO_NEWLOCALS:
                if not inspect.isclass(value):raise ValueError('imported reader provider class differs')
                check(part,value);continue
            if isinstance(value,(staticmethod,classmethod)):value=value.__func__
            if isinstance(value,property):value=value.fget
            if not inspect.isfunction(value)or value.__code__!=part:
                raise ValueError('stale imported reader provider implementation: '+name+'.'+part.co_qualname)
    check(_COMPILED_PROVIDERS[key],module)
    return module


def _source_check():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 cotrain data source changed')
    data=_loaded_provider_exact('egoemg_repro.emg_initializer_data',ROOT/'egoemg_repro/emg_initializer_data.py')
    binding=_loaded_provider_exact('egoemg_repro.emg_initializer_cohort',ROOT/'egoemg_repro/emg_initializer_cohort.py')
    if (_view is not data._view or _schema is not data._schema or EmgInitializerDataset is not data.EmgInitializerDataset
        or _reader_binding is not binding._reader_binding):raise ValueError('imported reader provider alias differs')


def _config_token(config):
    from omegaconf import OmegaConf
    return _hash(OmegaConf.to_container(config,resolve=True))


def rn18_initializer_contract(upstream,config):
    _source_check();root,value=_validated(_safe(upstream),config);protocol=emg_initializer_protocol(root,config)
    _activate(root)
    _loaded_provider_exact('egoemg.datasets.egoemg_memmap_dataset',root/'egoemg/datasets/egoemg_memmap_dataset.py')
    _loaded_provider_exact('egoemg.transforms',root/'egoemg/transforms.py')
    from egoemg.datamodule import make_data_module
    dm=make_data_module(config)
    if(dm.batch_size!=260 or dm.num_workers!=12 or dm.eval_num_workers!=12 or dm.dataset_repeat!=1
        or dm.window_length!=W or dm.val_test_window_length!=W or dm.stride!=STRIDE or dm.val_test_stride!=W
        or protocol['trainer']['max_epochs']!=150 or protocol['effective_loss_weights']['landmark/fingertip']!=.01):
        raise ValueError('exact original cotrain150/0.01/12worker contract required')
    v=dict(format='rn18_fixed_holdout_exact_cotrain_data_contract_v1',producer_protocol=protocol,
        batch_size_by_role={r:dm.batch_size for r in ROLES},workers_by_role=dict(train=dm.num_workers,val=dm.eval_num_workers),
        loader=dict(pin_memory=dm.pin_memory,persistent_workers=dm.persistent_workers,prefetch_factor=dm.prefetch_factor,
            train_context=dm.multiprocessing_context,val_context=dm.eval_multiprocessing_context),
        center_target_only_by_role={r:False for r in ROLES},repeat_by_role={r:1 for r in ROLES},
        sampler='native global RandomSampler across concatenated hands; no private generator/session group interface',
        declared_incre_right_corpus_available=False,available_corpus='supplied legacy-schema corpus only; Incre not fabricated',
        future_holdout_family='fusion/fusion_rn18_s_center_8ch',
        full_targets='unchanged native22x12000 trajectory in train and validation; generated mask, no wrist cleanup',
        local_data_source_sha256=IMPORTED_SHA,native_model_constructed=False,data_admitted=False,
        scientific_eligible=False,full_paper_complete=False)
    v['sha256']=_hash(v);return v


def build_rn18_initializer_partitions(store,val_fraction=.12):
    _source_check();future=build_rn18_fusion_partitions(store,val_fraction);future_hash=_hash(future)
    result=copy.deepcopy(future);result['format']='rn18_fixed_holdout_cotrain_full_trajectory_partitions_v1'
    result['windows']=[w for w in result['windows']if w['role']in ROLES]
    result['coverage']={r:result['coverage'][r]for r in ROLES}
    result['settings']['center_target_only_by_role']={r:False for r in ROLES}
    result['settings']['repeat_by_role']={r:1 for r in ROLES}
    train=union_intervals((w['footprint_start'],w['footprint_end'])for w in result['windows']if w['role']=='train')
    holds=dict(val=union_intervals((v['start'],v['end'])for v in future['validation_intervals']),
        test=union_intervals((w['footprint_start'],w['footprint_end'])for w in future['windows']if w['role']=='test'))
    overlaps={r:_intersection(train,spans)for r,spans in holds.items()}
    if any(overlaps.values()):raise ValueError('initializer full trajectory training overlaps future RN18 holdouts')
    result['future_holdout_overlap_rows']=overlaps;result['future_fusion_holdout_intervals']=holds
    result['future_fusion_counts']={r:future['coverage'][r]['hand_samples']for r in('train','val','test')}
    result['provenance'].pop('partition_sha256')
    result['provenance'].update(protocol='exact-cotrain-s-on-fixed-rn18-holdouts-v1',
        future_fusion_partition_content_sha256=future_hash,
        source_window_provider='unchanged native RN18 fusion full-jitter anchors; train/val only',
        target_policy='full original trajectory in both roles, never center-only',
        mask_policy='original generated time label mask broadcast to22; no wrist-valid cleanup',
        evaluation='full native trajectory on fixed future RN18 validation intervals; no initializer test role',
        normalization='original published per-hand/per-channel vectors; not project train-only statistics',
        available_corpus='released legacy only; declared Incre right-hand data absent',initializer_test_role=False)
    result['provenance']['partition_sha256']=_hash(result);return result


class Rn18InitializerDataset(EmgInitializerDataset):
    """Original full __getitem__; distinct validated roles and worker reentry."""
    def __init__(self,data,partitions,*,upstream,config,normalization,role):
        if role not in ROLES:raise ValueError('RN18 initializer has train/val only; no test role')
        _source_check();root,_=_validated(_safe(upstream),config);self.contract=rn18_initializer_contract(root,config)
        expected_norm=emg_initializer_normalization(root)
        if _hash(normalization)!=_hash(expected_norm):raise ValueError('original cotrain per-hand normalization binding differs')
        data=_safe(data);self.store=LegacyStore(data);paths=_schema(self.store)
        self._paths={'metadata:manifest':_safe(self.store.manifest_path),'metadata:episodes':_safe(self.store.metadata_path),
            **{'field:'+k:_safe(p)for k,p in paths.items()}}
        providers=[Path(__file__),ROOT/'egoemg_repro/emg_initializer_data.py',ROOT/'egoemg_repro/emg_initializer_cohort.py',
            ROOT/'egoemg_repro/emg_initializer_literal.py',root/'egoemg/datasets/egoemg_memmap_dataset.py',root/'egoemg/transforms.py',root/NORM_ASSET]
        self._guard=_StatGuard([*self._paths.values(),*providers]);self._provider_digests={str(p):_file_hash(p)for p in providers}
        expected=build_rn18_initializer_partitions(self.store,partitions['settings']['val_fraction'])
        if _hash(partitions)!=_hash(expected):raise ValueError('RN18 initializer partition binding differs')
        self.partitions=copy.deepcopy(expected);self.role=role;self.repeat=1;self.upstream=root;self.config=copy.deepcopy(config)
        self.normalization=copy.deepcopy(expected_norm);self.protocol=emg_initializer_protocol(root,config)
        self.windows=[w for w in expected['windows']if w['role']==role]
        if not self.windows:raise ValueError('RN18 initializer role must be nonempty')
        _activate(root);self.views=[_view(self.store,self.windows,h,role,root,config)for h in('left','right')]
        self._partition_seal=_hash(self.partitions);self._contract_seal=_hash(self.contract)
        self._config_seal=_config_token(config);self._normalization_seal=_hash(self.normalization)
        self._reader_seal=_reader_binding(self,self._paths);self.check()

    def check(self):
        _source_check();self._guard.check()
        if (_config_token(self.config)!=self._config_seal or _hash(self.contract)!=self._contract_seal
            or _hash(self.normalization)!=self._normalization_seal or _hash(self.partitions)!=self._partition_seal
            or self.windows!=[w for w in self.partitions['windows']if w['role']==self.role]
            or _reader_binding(self,self._paths)!=self._reader_seal
            or any(_file_hash(p)!=h for p,h in self._provider_digests.items())):
            raise ValueError('RN18 initializer source/config/reader role binding changed')

    def __getstate__(self):
        self.check()
        return dict(data=self.store.root,partitions=self.partitions,upstream=self.upstream,
            config=self.config,normalization=self.normalization,role=self.role)

    def __setstate__(self,state):self.__init__(**state)


def native_rn18_initializer_loader(dataset,*,scope):
    if type(dataset)is not Rn18InitializerDataset or scope not in SCOPES:
        raise ValueError('native RN18 initializer dataset and explicit fit scope required')
    dataset.check();_activate(dataset.upstream)
    from egoemg.datamodule import make_data_module
    dm=make_data_module(dataset.config);cpu=scope=='cpu_engineering'
    if cpu:dm.batch_size=2;dm.num_workers=dm.eval_num_workers=0
    workers=dm.num_workers if dataset.role=='train'else dm.eval_num_workers
    context=dm.multiprocessing_context if dataset.role=='train'else dm.eval_multiprocessing_context
    loader=dm._make_loader(dataset,shuffle=dataset.role=='train',num_workers=workers,multiprocessing_context=context)
    if(loader.batch_size!=(2 if cpu else 260)or loader.num_workers!=(0 if cpu else 12)
        or loader.generator is not None or loader.drop_last or loader.pin_memory):
        raise ValueError('original RN18 cotrain loader B260/12worker contract differs')
    loader.egoemg_repro_scope='cpu_engineering'if cpu else'native_loader_construction_only'
    dataset.check();return loader
