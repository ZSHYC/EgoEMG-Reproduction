"""Original vision module construction and strict, self-contained weight reload.

Fresh construction requires an explicitly prefetched genuine asset in an offline
HF cache. Construction evidence is not evidence of training or paper accuracy.
No original forward, loss, optimizer or scheduler implementation is replaced.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
from collections.abc import Mapping
import hashlib
import json

from .classic_emg import _activate, _file_hash, _hash
from .vision_literal import _validated, vision_protocol


ROOT=Path(__file__).resolve().parents[1]
_IMPORTED_SHA=_file_hash(__file__)
ASSETS={
    'vit_small': ('4610ca143709d58a633b6397a74412c2c3842454',88240510,
        '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'),
    'vit_base': ('4685c99dabffe5affac90bd99dbffd25801ae58d',346334872,
        '55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b'),
    'vit_large': ('4741e1cafbf45415e77074bb0cb42dba76c8684a',1217502758,
        '0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355'),
}
_LOCAL_SOURCES=('egoemg_repro/vision_literal_runtime.py','egoemg_repro/vision_literal.py',
    'egoemg_repro/vision_literal_data.py','egoemg_repro/visual.py','egoemg_repro/classic_emg.py')
_NATIVE_SOURCES=('egoemg/_train_utils.py','egoemg/utils.py','egoemg/metrics.py',
    'egoemg/models/modules/_vision_constants.py')
_PACKAGES=('torch','torchvision','timm','pytorch-lightning','huggingface-hub','safetensors','hydra-core')
_RUNTIME_MODULES=('timm.models._builder','timm.models._hub','timm.models.vision_transformer',
    'huggingface_hub.file_download','torch.optim.adamw','torch.optim.lr_scheduler')


def _asset_descriptor(variant):
    revision,size,digest=ASSETS[variant]
    return dict(repo='timm/'+variant+'_patch14_dinov2.lvd142m',revision=revision,
        filename='model.safetensors',bytes=size,sha256=digest)


def _stamp(path):
    s=Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _sources(root,protocol):
    if _file_hash(__file__)!=_IMPORTED_SHA:raise ValueError('imported vision runtime source changed')
    result={name:_file_hash(ROOT/name) for name in _LOCAL_SOURCES}
    result.update({'upstream/'+name:digest for name,digest in protocol['sources_sha256'].items()})
    result.update({'upstream/'+name:_file_hash(root/name) for name in _NATIVE_SOURCES})
    for name in _RUNTIME_MODULES:
        result['runtime/'+name]=_file_hash(importlib.import_module(name).__file__)
    return result


def _versions():
    return dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in _PACKAGES})


def _schema(state):
    import torch
    if not isinstance(state,Mapping) or not state:raise ValueError('complete vision tensor state is required')
    result={}
    for name,tensor in state.items():
        if type(name) is not str or not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided:
            raise ValueError('vision state must contain named strided tensors')
        if not bool(torch.isfinite(tensor).all()):raise ValueError('vision state contains nonfinite values')
        result[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype))
    return result


def vision_state_sha256(state):
    """Hash names, shapes, dtypes and complete tensor bytes, including buffers."""
    import torch
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        header=json.dumps(dict(name=name,shape=list(value.shape),dtype=str(value.dtype)),
            sort_keys=True,separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'big'));digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _offline_asset(variant):
    from huggingface_hub import constants,try_to_load_from_cache
    import timm.models._builder as builder
    from timm.models import get_pretrained_cfg
    asset=_asset_descriptor(variant)
    cache=os.environ.get('HF_HUB_CACHE',constants.HF_HUB_CACHE)
    if not constants.HF_HUB_OFFLINE or Path(cache).resolve()!=Path(constants.HF_HUB_CACHE).resolve():
        raise ValueError('fresh vision requires HF offline/cache settings before importing the runtime')
    route,location=builder._resolve_pretrained_source(get_pretrained_cfg(variant+'_patch14_dinov2').to_dict())
    if route!='hf-hub' or location!=asset['repo']:
        raise ValueError('native timm route differs from the admitted HF asset')
    pinned=try_to_load_from_cache(asset['repo'],asset['filename'],revision=asset['revision'],cache_dir=cache)
    default=try_to_load_from_cache(asset['repo'],asset['filename'],revision='main',cache_dir=cache)
    if not isinstance(pinned,str) or not isinstance(default,str) or Path(pinned).resolve()!=Path(default).resolve():
        raise ValueError('prefetched pinned asset and native main cache must resolve to the same file')
    path=Path(pinned).resolve();stamp=_stamp(path)
    if stamp[2]!=asset['bytes'] or _file_hash(path)!=asset['sha256'] or _stamp(path)!=stamp:
        raise ValueError('prefetched vision asset size/hash differs or changed during admission')
    return asset,path,stamp


def _restore_identity(identity,variant,protocol,sources,versions,state):
    if not isinstance(identity,dict):raise ValueError('fresh initialization identity is required for strict restore')
    identity=copy.deepcopy(identity)
    if (identity.get('format')!='original_vision_module_construction_v1' or
        identity.get('sha256')!=_hash({k:v for k,v in identity.items() if k!='sha256'}) or
        identity.get('variant')!=variant or identity.get('protocol_sha256')!=protocol['sha256'] or
        identity.get('sources_sha256')!=sources or identity.get('runtime_versions')!=versions or
        any(identity.get(k) is not False for k in ('scientific_eligible','production_ready','full_paper_complete'))):
        raise ValueError('vision initialization source/config/runtime identity differs')
    init=identity.get('initialization',{})
    if (init.get('fresh_pretrained_verified') is not True or init.get('asset')!=_asset_descriptor(variant) or
        init.get('pretrained_identity',{}).get('source_type')!='hf-hub' or
        init.get('pretrained_identity',{}).get('source_location')!=_asset_descriptor(variant)['repo']):
        raise ValueError('genuine native pretrained initialization identity differs')
    if _schema(state)!=identity.get('model_schema'):raise ValueError('complete vision state keys/shapes/dtypes differ')
    return identity


@dataclass
class PreparedVision:
    module: object
    identity: dict
    mode: str
    factory_pretrained: bool
    loaded_state_sha256: str
    _root: Path
    _config: object
    _asset_path: Path | None=None
    _asset_stamp: tuple | None=None

    def check(self):
        """Check source/config stability; learned weights are allowed to change."""
        protocol=vision_protocol(self._root,self._config)
        if (protocol['sha256']!=self.identity['protocol_sha256'] or
            _sources(self._root,protocol)!=self.identity['sources_sha256'] or
            _versions()!=self.identity['runtime_versions'] or
            self.identity['sha256']!=_hash({k:v for k,v in self.identity.items() if k!='sha256'})):
            raise ValueError('prepared vision source/config identity changed')
        if self._asset_path is not None and _stamp(self._asset_path)!=self._asset_stamp:
            raise ValueError('admitted pretrained asset changed')


def build_vision_runtime(upstream,config,*,state=None,expected_initialization=None):
    """Call original setup and factory; reload all native module tensors strictly.

    ``expected_initialization`` is the complete fresh construction identity. It
    records lineage only; a subsequent training runner must verify its own saved
    checkpoints, optimizer, loops, callbacks, RNG and termination evidence.
    """
    from omegaconf import OmegaConf
    root,variant,_=_validated(upstream,config);protocol=vision_protocol(root,config)
    if (state is None)!=(expected_initialization is None):
        raise ValueError('strict restore requires both complete state and initialization identity')
    sources=_sources(root,protocol);versions=_versions();fresh=state is None
    asset_path=asset_stamp=None
    if fresh:
        asset,asset_path,asset_stamp=_offline_asset(variant)
    else:
        identity=_restore_identity(expected_initialization,variant,protocol,sources,versions,state)
    _activate(root)
    from egoemg._train_utils import setup_runtime
    import egoemg.train as native
    effective=OmegaConf.create(OmegaConf.to_container(config,resolve=True))
    if not fresh:effective.module.pretrained=False
    setup_runtime(effective)
    if fresh:
        from .visual import _capture_timm_pretrained_source,pretrained_weight_identity
        with _capture_timm_pretrained_source() as observed:module=native.make_lightning_module(effective)
        if observed!=[dict(source_type='hf-hub',source_location=asset['repo'])]:
            raise ValueError('native factory did not load the admitted pretrained transport')
        module.model.reproduction_pretrained_source=observed[0]
        from safetensors.torch import load_file
        from timm.models.vision_transformer import checkpoint_filter_fn
        expected=checkpoint_filter_fn(load_file(str(asset_path)),module.model.backbone,
            interpolation='bicubic',antialias=True)
        actual=module.model.backbone.state_dict()
        import torch
        if set(expected)!=set(actual) or any(expected[k].dtype!=actual[k].dtype or
            expected[k].shape!=actual[k].shape or not torch.equal(expected[k],actual[k]) for k in actual):
            raise ValueError('native module did not retain every interpolated pretrained backbone tensor')
        if _stamp(asset_path)!=asset_stamp:raise ValueError('pretrained asset changed during construction')
        identity=dict(format='original_vision_module_construction_v1',variant=variant,
            scope='genuine native construction only; no data admission, optimization or performance claim',
            scientific_eligible=False,production_ready=False,full_paper_complete=False,
            protocol_sha256=protocol['sha256'],sources_sha256=sources,runtime_versions=versions,
            model_schema=_schema(module.state_dict()),parameter_names=[n for n,_ in module.named_parameters()],
            initial_state_sha256=vision_state_sha256(module.state_dict()),
            initialization=dict(fresh_pretrained_verified=True,retained_tensor_count=len(actual),asset=asset,
                pretrained_identity=pretrained_weight_identity(module.model)))
        identity['sha256']=_hash(identity)
    else:
        module=native.make_lightning_module(effective)
        if _schema(module.state_dict())!=identity['model_schema']:
            raise ValueError('native reload factory schema differs from saved module')
        module.load_state_dict(state,strict=True)
        if vision_state_sha256(module.state_dict())!=vision_state_sha256(state):
            raise ValueError('strict restored native tensor bytes differ')
    if type(module).__module__!='egoemg.lightning' or type(module.model).__name__!='VisionViTPose':
        raise ValueError('original vision module class differs')
    result=PreparedVision(module,identity,'fresh_native_pretrained' if fresh else 'strict_self_contained_restore',
        fresh,vision_state_sha256(module.state_dict()),root,copy.deepcopy(config),asset_path,asset_stamp)
    result.check()
    return result


def vision_optimizer_identity(prepared,optimizer,scheduler):
    """Validate, without replacing, original component AdamW/cosine objects."""
    import torch
    prepared.check()
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError('original AdamW and CosineAnnealingLR objects required')
    if scheduler.optimizer is not optimizer or scheduler.T_max!=150 or scheduler.eta_min!=1e-5:
        raise ValueError('native scheduler recipe/optimizer binding differs')
    if type(scheduler.last_epoch) is not int or not 0<=scheduler.last_epoch<=150:
        raise ValueError('native cosine epoch is outside original budget')
    named=dict(prepared.module.named_parameters());by_id={id(p):n for n,p in named.items() if p.requires_grad}
    expected={group:['model.'+n for n,p in prepared.module.model.named_parameters()
        if n.startswith(group+'.') and p.requires_grad] for group in ('head','backbone')}
    if len(optimizer.param_groups)!=2 or scheduler.base_lrs!=[.0001,.0001]:
        raise ValueError('original component group count/base learning rates differ')
    lr=1e-5+(.0001-1e-5)*(1+math.cos(math.pi*scheduler.last_epoch/150))/2
    groups=[];seen=[]
    for group_name,group in zip(('head','backbone'),optimizer.param_groups):
        names=[by_id.get(id(p)) for p in group['params']]
        if (group.get('name')!=group_name or names!=expected[group_name] or
            not math.isclose(group['lr'],lr,rel_tol=1e-10,abs_tol=1e-14) or
            group.get('initial_lr')!=.0001 or group['weight_decay']!=.01 or
            tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group['amsgrad'] is not False or
            any(group.get(k,False) is not False for k in ('maximize','capturable','differentiable')) or
            any(group.get(k) is not None for k in ('foreach','fused'))):
            raise ValueError('native component membership or AdamW hyperparameters differ')
        seen.extend(names);groups.append(dict(name=group_name,parameter_names=names,lr=group['lr'],
            initial_lr=group['initial_lr'],weight_decay=group['weight_decay'],betas=list(group['betas']),
            eps=group['eps'],amsgrad=group['amsgrad']))
    if len(seen)!=len(set(seen)) or set(seen)!=set(by_id.values()):
        raise ValueError('native optimizer does not cover each trainable parameter exactly once')
    if len(scheduler._last_lr)!=2 or any(not math.isclose(x,lr,rel_tol=1e-10,abs_tol=1e-14) for x in scheduler._last_lr):
        raise ValueError('scheduler stored learning rates differ from native groups')
    return dict(format='original_vision_optimizer_observation_v1',groups=groups,
        scheduler=dict(type='CosineAnnealingLR',T_max=150,eta_min=1e-5,last_epoch=scheduler.last_epoch,
            interval='epoch',frequency=1),scientific_eligible=False)
