"""Exact native ordinary-data factories with an actual live source binding.

Only private globals copies use the already deep-verified field identity.
Original validators, model/data factories, recipes and module globals stay intact.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.metadata
from pathlib import Path
import platform
from types import FunctionType

from . import UPSTREAM_COMMIT
from . import emg2pose_data as storage,emg2pose_runtime as native
from . import emg2pose_direct_source as sources
from .emg2pose_direct_source import VerifiedDirectSource

CONTEXT_KEY='_emg2pose_direct_runtime'
CONTEXT_FORMAT='emg2pose_direct_runtime_binding_v1'
FORMAT='emg2pose_direct_runtime_identity_v1'
POLICY='unchanged original Emg2PoseDataset and model factories; one deep admission with live ordinary-file guards'
_CODE=(*sources._CODE,'emg2pose_direct_runtime.py')
_ROOT=Path(__file__).parent
_IMPORTED={n:storage._sha256(_ROOT/n) for n in _CODE}
_STAMPS={_ROOT/n:sources._stamp(_ROOT/n) for n in _CODE}


def _binding(upstream,index,source,csv):
    if type(source) is not VerifiedDirectSource:raise ValueError('an exact live VerifiedDirectSource is required')
    source.assert_current()
    if any(sources._stamp(p)!=v for p,v in _STAMPS.items()):raise ValueError('imported direct runtime changed')
    upstream,index,csv=storage._safe_path(upstream),storage._safe_path(index),storage._safe_path(csv,file=True)
    if upstream!=source.upstream or index!=source.index or csv!=source.metadata_csv:
        raise ValueError('concrete original source/upstream/metadata binding differs')
    return upstream,index,csv,source.data_identity


def compose_config(upstream,size,index,*,source:VerifiedDirectSource,metadata_csv):
    if size not in native.SIZES:raise ValueError('size must be small, middle or large')
    upstream,index,csv,identity=_binding(upstream,index,source,metadata_csv)
    cfg=native._compose(upstream,size,index,csv)
    from omegaconf import OmegaConf
    OmegaConf.update(cfg,CONTEXT_KEY,dict(format=CONTEXT_FORMAT,size=size,index=str(index),metadata_csv=str(csv),
        source_completion_sha256=source.completion_sha256,data_identity=identity,seed_policy=native.SEED_POLICY,adapter_policy=POLICY),force_add=True)
    source.assert_current();return cfg


def _validated_config(upstream,config,source):
    from omegaconf import OmegaConf
    if type(source) is not VerifiedDirectSource:raise ValueError('an exact live VerifiedDirectSource is required')
    source.assert_current()
    try:
        if native.CONTEXT_KEY in config:raise ValueError('foreign original runtime binding is not a direct source binding')
        context=OmegaConf.to_container(config[CONTEXT_KEY],resolve=True)
        upstream,index,csv,identity=_binding(upstream,context['index'],source,context['metadata_csv'])
        if (context.get('format')!=CONTEXT_FORMAT or context.get('size') not in native.SIZES or
                context.get('source_completion_sha256')!=source.completion_sha256 or context.get('data_identity')!=identity or
                context.get('seed_policy')!=native.SEED_POLICY or context.get('adapter_policy')!=POLICY):
            raise ValueError('direct original source runtime identity differs')
        value=OmegaConf.to_container(config,resolve=True);value.pop(CONTEXT_KEY)
        canonical=native._canonical_config(OmegaConf.create(value),upstream,index,csv)
        expected=native._canonical_config(native._compose(upstream,context['size'],index,csv),upstream,index,csv)
        if canonical!=expected:raise ValueError('configuration differs from exact original scientific recipe')
        return upstream,context,identity,canonical
    except (KeyError,TypeError,AttributeError) as e:raise ValueError('invalid direct original runtime binding') from e


def build_runtime(upstream,config,*,source:VerifiedDirectSource):
    """Run the exact original seeded factory code, after concrete source checks."""
    _validated_config(upstream,config,source)
    def validate(root,cfg,*_):return _validated_config(root,cfg,source)
    original=native.build_runtime
    bound=FunctionType(original.__code__,dict(original.__globals__,_validated_config=validate),
        name=original.__name__,argdefs=original.__defaults__,closure=original.__closure__)
    result=bound(upstream,config);source.assert_current();return result


def runtime_identity(upstream,config,*,source:VerifiedDirectSource):
    upstream,context,identity,canonical=_validated_config(upstream,config,source)
    bundle=hashlib.sha256()
    for path in sources._official_files(upstream):
        bundle.update(str(path.relative_to(upstream)).encode());bundle.update(path.read_bytes())
    deps={}
    for name in ('torch','torchvision','numpy','hydra-core','omegaconf','pytorch-lightning','torchmetrics','lightning-utilities','pandas','joblib','h5py','scipy'):
        try:deps[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:deps[name]=None
    source.assert_current()
    return dict(format=FORMAT,size=context['size'],data=identity,config_sha256=native._hash(canonical),
        upstream_revision=UPSTREAM_COMMIT,upstream_bundle_sha256=bundle.hexdigest(),
        runtime_files_sha256={'egoemg_repro/'+n:storage._sha256(_ROOT/n) for n in _CODE},dependencies=deps,
        python=platform.python_version(),seed_policy=native.SEED_POLICY,adapter_policy=POLICY,
        initialization='seeded random official module; no pretrained or resume path; no downloads',
        identity_path_policy='validated concrete ordinary files; content identity survives exact relocation')


def reload_runtime(upstream,config,state_dict,*,identity,source:VerifiedDirectSource):
    """Strict trusted local weight restore; optimizer/RNG recovery is separate."""
    if identity!=runtime_identity(upstream,config,source=source):raise ValueError('strict direct runtime/config/source identity differs')
    if not isinstance(state_dict,Mapping):raise ValueError('strict reload requires a state dictionary')
    module,dm=build_runtime(upstream,config,source=source)
    import torch
    expected=module.state_dict()
    for name,value in state_dict.items():
        if name in expected and (not isinstance(value,torch.Tensor) or value.dtype!=expected[name].dtype or value.layout!=expected[name].layout):
            raise ValueError('strict reload tensor dtype/layout differs: '+name)
    module.load_state_dict(state_dict,strict=True);return module,dm


def close_data_module(data_module):
    """Close only owned original lazy memmap views after all consumers exit."""
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    seen=set();maps=set()
    def close(dataset):
        if dataset is None or id(dataset) in seen:return
        seen.add(id(dataset))
        if type(dataset) is Emg2PoseDataset:
            for value in dataset._memmaps.values():
                mapping=value._mmap
                if id(mapping) not in maps:maps.add(id(mapping));mapping.close()
            dataset._memmaps.clear()
        else:
            for child in getattr(dataset,'datasets',()):close(child)
            close(getattr(dataset,'dataset',None))
    for name in ('train_dataset','val_dataset','test_dataset'):close(getattr(data_module,name,None))
