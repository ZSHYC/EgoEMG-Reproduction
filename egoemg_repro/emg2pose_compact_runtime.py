"""Exact original factories/recipes with an explicitly verified compact source.

Private copies of the qualified raw-runtime binding/factory code retain their
algorithms while using separate source, config and content-identity boundaries.
No original module globals or raw/direct-memmap validators are changed.
"""
from __future__ import annotations

from pathlib import Path
from types import FunctionType

from . import emg2pose_data as storage,emg2pose_runtime as native
from . import emg2pose_memory_runtime as original
from .emg2pose_compact_index import VerifiedCompactSource


CONTEXT_KEY='_emg2pose_compact_runtime'
TARGET='egoemg_repro.emg2pose_compact_dataset.CompactTarDataset'
POLICY='storage-only official Emg2PoseDataset subclass over verified compact pair; exact native factories and scientific recipe'
_CODE=tuple(dict.fromkeys((*original._CODE,'emg2pose_memory_runtime.py','emg2pose_compact_codec.py',
    'emg2pose_compact_archive.py','emg2pose_compact_data.py','emg2pose_compact_index.py',
    'emg2pose_compact_dataset.py','emg2pose_compact_runtime.py')))
_GLOBALS={**vars(original),'__name__':__name__,'__file__':__file__,
    'VerifiedSource':VerifiedCompactSource,'CONTEXT_KEY':CONTEXT_KEY,'TARGET':TARGET,'POLICY':POLICY,'_CODE':_CODE}


def _core(name):return FunctionType(getattr(original,name).__code__,_GLOBALS,name=name)
_binding_core=_core('_binding')
_compose_core=_core('compose_config')
_build_core=_core('build_runtime')
_identity_core=_core('runtime_identity')
_reload_core=_core('reload_runtime')


def _binding(upstream,index,source,metadata_csv):
    if type(source) is not VerifiedCompactSource:raise ValueError('an actual live VerifiedCompactSource is required')
    return _binding_core(upstream,index,source,metadata_csv)


def _compose(upstream,size,index,source,csv,identity,proof):
    from omegaconf import OmegaConf
    cfg=native._compose(upstream,size,index,csv)
    for collection in (cfg.dataset,cfg.datamodule.dataset_conf):
        for role in ('train','val','test'):
            for entry in collection[role]:
                entry._target_=TARGET
                for name,value in (('compact_payload_fd',source.payload_fileno()),
                                   ('compact_index_fd',source.block_index_fileno()),
                                   ('compact_handle_identity',proof)):
                    OmegaConf.update(entry,name,value,force_add=True)
    OmegaConf.update(cfg,CONTEXT_KEY,dict(format='emg2pose_compact_runtime_binding_v1',size=size,
        index=str(index),metadata_csv=str(csv),data_identity=identity,handle_identity=proof,
        seed_policy=native.SEED_POLICY,adapter_policy=POLICY),force_add=True)
    source.assert_current();return cfg


def compose_config(upstream:Path,size:str,index:Path,*,source:VerifiedCompactSource,metadata_csv:Path):
    """Require the exact admitted source and preserve the native scientific recipe."""
    return _compose_core(upstream,size,index,source=source,metadata_csv=metadata_csv)


def _validated_config(upstream,config,source):
    from omegaconf import OmegaConf
    upstream,_,_=storage._official(upstream)
    try:
        context=OmegaConf.to_container(config[CONTEXT_KEY],resolve=True)
        if (context.get('format')!='emg2pose_compact_runtime_binding_v1' or
                context.get('size') not in native.SIZES or context.get('seed_policy')!=native.SEED_POLICY or
                context.get('adapter_policy')!=POLICY):raise ValueError('invalid compact-source runtime binding')
        index,csv,identity,proof=_binding(upstream,context['index'],source,context['metadata_csv'])
        expected=_compose(upstream,context['size'],index,source,csv,identity,proof)
        if OmegaConf.to_container(config,resolve=True)!=OmegaConf.to_container(expected,resolve=True):
            raise ValueError('config differs from exact native recipe or actual compact source')
        clean=OmegaConf.to_container(config,resolve=True);clean.pop(CONTEXT_KEY)
        canonical=native._canonical_config(OmegaConf.create(clean),upstream,index,csv)
        for collection in (canonical['dataset'],canonical['datamodule']['dataset_conf']):
            for role in ('train','val','test'):
                for entry in collection[role]:
                    entry['compact_payload_fd']='verified-compact://payload'
                    entry['compact_index_fd']='verified-compact://block-index'
                    entry['compact_handle_identity']='verified-compact://content-identity'
        return upstream,context,identity,canonical
    except (KeyError,TypeError,AttributeError) as error:raise ValueError('invalid compact runtime config') from error


def build_runtime(upstream:Path,config,*,source:VerifiedCompactSource):
    """Run the exact original seeded setup and model/datamodule factory code."""
    return _build_core(upstream,config,source=source)


def runtime_identity(upstream:Path,config,*,source:VerifiedCompactSource):
    """Validate private kernel bindings before deriving relocatable identity."""
    value=_identity_core(upstream,config,source=source)
    value['format']='emg2pose_compact_runtime_identity_v1';return value


def reload_runtime(upstream:Path,config,state_dict,*,identity:dict,source:VerifiedCompactSource):
    """Strict original weights restore, after exact compact runtime validation."""
    return _reload_core(upstream,config,state_dict,identity=identity,source=source)


def close_data_module(data_module):
    """Close each owned compact leaf once; worker termination belongs to caller."""
    from .emg2pose_compact_dataset import CompactTarDataset
    seen=set()
    def close(dataset):
        if dataset is None or id(dataset) in seen:return
        seen.add(id(dataset))
        if isinstance(dataset,CompactTarDataset):dataset.close()
        else:
            for child in getattr(dataset,'datasets',()):close(child)
            close(getattr(dataset,'dataset',None))
    for role in ('train','val','test','predict'):close(getattr(data_module,role+'_dataset',None))


_GLOBALS.update(_binding=_binding,_compose=_compose,_validated_config=_validated_config,
    build_runtime=build_runtime,runtime_identity=runtime_identity)
