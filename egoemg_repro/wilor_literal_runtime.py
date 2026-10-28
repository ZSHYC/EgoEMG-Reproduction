"""Verified original WiLoR22-angle assets, native module and strict tensor reload.

Source/config/mean and real pretrained bytes are admitted before pickle/factory.
These objects prove construction only; they are never CUDA/full-training permits.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import weakref

from .classic_emg import _activate,_file_hash,_hash
from .classic_emg_cohort import _StatGuard,_safe
from .vision_literal_runtime import _schema,vision_state_sha256
from .wilor_literal import _validated,wilor_protocol

ROOT=Path(__file__).resolve().parents[1]
CATALOG_PATH=ROOT/'configs/wilor-assets.json'
CATALOG_SHA256='0d5834c66b69a8c245bb340ce93869bc153cb6cfdc330efe8fc0ba494f821250'
_IMPORTED_SHA=_file_hash(__file__)
_ASSETS=weakref.WeakSet()
_PREPARED=weakref.WeakSet()
_CONSTRUCTION_SCOPE='genuine native construction only; not a training or CUDA permit'
_LOCAL_SOURCES=('egoemg_repro/wilor_literal_runtime.py','egoemg_repro/wilor_literal.py',
    'egoemg_repro/classic_emg.py','egoemg_repro/classic_emg_cohort.py',
    'egoemg_repro/vision_literal_runtime.py','configs/wilor-assets.json')
_PACKAGES=('torch','torchvision','pytorch-lightning','hydra-core','numpy','timm','yacs','smplx','pyrender')
_RUNTIME_MODULES=('torch.optim.adamw','torch.optim.lr_scheduler',
    'pytorch_lightning.plugins.precision.amp','pytorch_lightning.strategies.strategy')


def _stream_sha(path,guard):
    import hashlib
    guard.check();h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    guard.check();return h.hexdigest()


def _catalog():
    import hashlib
    guard=_StatGuard([CATALOG_PATH]);guard.check();raw=CATALOG_PATH.read_bytes();guard.check()
    if hashlib.sha256(raw).hexdigest()!=CATALOG_SHA256:raise ValueError('fixed WiLoR asset catalog changed')
    return json.loads(raw)


def _read_asset(path,expected,guard):
    if (type(expected.get('bytes')) is not int or expected['bytes']<=0 or
        type(expected.get('sha256')) is not str or re.fullmatch('[0-9a-f]{64}',expected['sha256']) is None):
        raise ValueError('explicit positive asset bytes and complete SHA256 required')
    path=_safe(path);guard.add(path);guard.check()
    if path.stat().st_size!=expected['bytes']:raise ValueError('actual WiLoR asset byte size differs')
    digest=_stream_sha(path,guard);guard.check()
    if digest!=expected['sha256']:raise ValueError('actual WiLoR asset SHA256 differs')
    return dict(bytes=expected['bytes'],sha256=digest)


@dataclass(frozen=True,eq=False)
class WilorAssets:
    runtime: Path
    checkpoint: Path | None
    restore: bool
    identity: dict
    _guard: object
    _seal: str

    def check(self):
        if self not in _ASSETS:raise ValueError('producer-owned WiLoR admission required')
        self._guard.check()
        if (_file_hash(__file__)!=_IMPORTED_SHA or _catalog()['source_revision']!=self.identity['source_revision'] or
            self.identity.get('sha256')!=self._seal or
            _hash({k:v for k,v in self.identity.items() if k!='sha256'})!=self._seal):
            raise ValueError('WiLoR asset/source identity changed')


def admit_wilor_assets(runtime,checkpoint=None,*,restore=False):
    """Hash actual fixed files; restore reads source/config/mean, never old weights."""
    if (not isinstance(runtime,(str,os.PathLike)) or type(restore) is not bool or
        (restore and checkpoint is not None) or
        (not restore and not isinstance(checkpoint,(str,os.PathLike)))):
        raise ValueError('explicit runtime/checkpoint paths and boolean restore mode required')
    root=_safe(runtime)
    if not root.is_dir():raise ValueError('actual WiLoR runtime directory required')
    catalog=_catalog();guard=_StatGuard([__file__,CATALOG_PATH]);files={}
    for name,digest in catalog['source_files_sha256'].items():
        files[name]=_read_asset(root/name,dict(bytes=catalog['source_files_bytes'][name],sha256=digest),guard)
    mean=catalog['mean'];_read_asset(root/mean['filename'],mean,guard)
    path=None if restore else _safe(checkpoint)
    if path is not None:_read_asset(path,catalog['checkpoint'],guard)
    identity=dict(format='native_wilor_asset_admission_v1',source_revision=catalog['source_revision'],
        catalog_sha256=CATALOG_SHA256,source_files=files,mean=mean,
        checkpoint=None if restore else catalog['checkpoint'],mode='restore_source_only' if restore else 'fresh',
        scientific_eligible=False,full_paper_complete=False)
    identity['sha256']=_hash(identity);guard.check()
    result=WilorAssets(root,path,restore,identity,guard,identity['sha256']);_ASSETS.add(result);result.check();return result


def _extract_backbone(checkpoint):
    if not isinstance(checkpoint,Mapping):raise ValueError('actual pretrained checkpoint mapping required')
    state=checkpoint.get('state_dict',checkpoint)
    if not isinstance(state,Mapping) or not state:raise ValueError('actual pretrained state mapping required')
    result={k.removeprefix('backbone.'):v for k,v in state.items() if type(k) is str and k.startswith('backbone.')}
    if not result:raise ValueError('nonempty pretrained backbone prefix required; no random fallback')
    _schema(result);return result


def _require_exact_state(actual,expected):
    if _schema(actual)!=_schema(expected):raise ValueError('complete tensor keys/shapes/dtypes differ')
    import torch
    if any(not torch.equal(actual[k],expected[k]) for k in actual):raise ValueError('actual tensor bytes differ')
    return vision_state_sha256(actual)


def _check_import_roots(runtime,upstream):
    roots=(('wilor',Path(runtime)/'wilor'),('egoemg',Path(upstream)/'egoemg'))
    for prefix,root in roots:
        root=_safe(root)
        for name,module in list(sys.modules.items()):
            if name!=prefix and not name.startswith(prefix+'.'):continue
            try:
                filename=getattr(module,'__file__',None)
                origins=[filename] if filename else list(getattr(module,'__path__',[]))
                belongs=bool(origins) and all(_safe(p).is_relative_to(root) for p in origins)
            except (OSError,TypeError,ValueError):belongs=False
            if not belongs:raise ValueError('already imported '+name+' belongs to another source root')
    prior=sys.modules.get('egoemg.models.modules.wilor_vit')
    if prior is not None:
        try:matches=_safe(getattr(prior,'WILOR_PATH','/missing'))==_safe(runtime)
        except (OSError,TypeError,ValueError):matches=False
        if not matches:raise ValueError('WiLoR path was captured from another runtime at import')


def _versions():
    return dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in _PACKAGES})


def _source_paths(root,protocol,assets):
    paths={n:ROOT/n for n in _LOCAL_SOURCES}
    paths.update({'upstream/'+n:root/n for n in protocol['sources_sha256']})
    paths.update({'upstream/'+n:root/n for n in ('egoemg/_train_utils.py','egoemg/utils.py','egoemg/metrics.py')})
    paths.update({'wilor/'+n:assets.runtime/n for n in assets.identity['source_files']})
    paths.update({'runtime/'+n:Path(importlib.import_module(n).__file__) for n in _RUNTIME_MODULES})
    return paths


def _source_guard(root,protocol,assets):
    return _StatGuard(_source_paths(root,protocol,assets).values())


def _sources(root,protocol,assets):
    if _file_hash(__file__)!=_IMPORTED_SHA:raise ValueError('imported WiLoR runtime source changed')
    return {n:_file_hash(p) for n,p in _source_paths(root,protocol,assets).items()}


def _head_contract(model):
    import torch
    nn=torch.nn
    if (type(model.out_channels) is not int or model.out_channels!=22 or
        type(model.left_context) is not int or model.left_context!=0 or
        type(model.right_context) is not int or model.right_context!=0 or
        type(model.avgpool) is not nn.AdaptiveAvgPool2d or model.avgpool.output_size!=(1,1) or
        type(model.head) is not nn.Sequential or len(model.head)!=4):
        raise ValueError('original WiLoR22angle head/pool/context required')
    first,activation,drop,last=model.head
    if (type(first) is not nn.Linear or first.in_features!=1280 or first.out_features!=512 or first.bias is None or
        type(activation) is not nn.GELU or activation.approximate!='none' or
        type(drop) is not nn.Dropout or drop.p!=.1 or drop.inplace is not False or
        type(last) is not nn.Linear or last.in_features!=512 or last.out_features!=22 or last.bias is None or
        any(p.dtype!=torch.float32 for p in model.head.parameters())):
        raise ValueError('original FP32parameter1280/GELU/dropout.1/22angle head required')


def _model_contract(module,protocol):
    if type(module).__module__!='egoemg.lightning' or type(module.model).__name__!='WiLoRViTPose':
        raise ValueError('original native WiLoR lightning/model classes required')
    expected=protocol['model_expectations'];model=module.model
    _head_contract(model)
    import torch
    prefixes=tuple('backbone.'+n+'.' for n in expected['frozen_backbone_components'])
    for n,p in model.named_parameters():
        if p.requires_grad is not (not n.startswith(prefixes)) or p.dtype!=torch.float32:
            raise ValueError('original WiLoR frozen component membership or parameter dtype differs')
    if (sum(p.numel() for p in model.parameters())!=expected['parameters'] or
        sum(p.numel() for p in model.parameters() if p.requires_grad)!=expected['trainable_parameters']):
        raise ValueError('actual original WiLoR parameter counts differ')
    return copy.deepcopy(expected)


def _restore_identity(identity,protocol,sources,versions,state):
    keys={'format','scope','scientific_eligible','full_paper_complete','protocol_sha256','sources_sha256',
        'runtime_versions','model_schema','parameter_names','model_contract','initial_state_sha256','initialization','sha256'}
    if (not isinstance(identity,dict) or set(identity)!=keys or identity.get('format')!='native_wilor_module_construction_v1' or
        identity.get('scientific_eligible') is not False or identity.get('full_paper_complete') is not False or
        identity.get('scope')!=_CONSTRUCTION_SCOPE or
        identity.get('sha256')!=_hash({k:v for k,v in identity.items() if k!='sha256'}) or
        identity.get('protocol_sha256')!=protocol['sha256'] or _hash(identity.get('sources_sha256'))!=_hash(sources) or
        _hash(identity.get('runtime_versions'))!=_hash(versions) or _hash(identity.get('model_contract'))!=_hash(protocol['model_expectations']) or
        _hash(identity.get('model_schema'))!=_hash(_schema(state))):
        raise ValueError('complete native WiLoR initial source/config/runtime and typed state required')
    init=identity.get('initialization',{});catalog=_catalog()
    if (not isinstance(init,dict) or set(init)!={'fresh_pretrained_verified','retained_tensor_count','backbone_sha256','asset','source_revision','factory_seed'} or
        init.get('fresh_pretrained_verified') is not True or type(init.get('retained_tensor_count')) is not int or
        init['retained_tensor_count']!=404 or _hash(init.get('asset'))!=_hash(catalog['checkpoint']) or
        init.get('source_revision')!=catalog['source_revision'] or type(init.get('factory_seed')) is not int or init['factory_seed']!=42 or
        re.fullmatch('[0-9a-f]{64}',str(init.get('backbone_sha256',''))) is None or
        re.fullmatch('[0-9a-f]{64}',str(identity.get('initial_state_sha256',''))) is None):
        raise ValueError('actual genuine WiLoR initialization lineage required')
    names=identity.get('parameter_names')
    if not isinstance(names,list) or not names or any(type(n) is not str for n in names) or len(names)!=len(set(names)):
        raise ValueError('complete actual native parameter names required')
    return copy.deepcopy(identity)


@dataclass(eq=False)
class PreparedWilor:
    module: object
    identity: dict
    mode: str
    factory_pretrained: bool
    loaded_state_sha256: str
    _root: Path
    _config: object
    _assets: WilorAssets
    _source_guard: object
    _seal: str

    def check(self):
        if self not in _PREPARED:raise ValueError('producer-owned native WiLoR module required')
        self._source_guard.check();self._assets.check();_check_import_roots(self._assets.runtime,self._root)
        protocol=wilor_protocol(self._root,self._config)
        if (self.identity.get('sha256')!=self._seal or _hash({k:v for k,v in self.identity.items() if k!='sha256'})!=self._seal or
            self.identity['protocol_sha256']!=protocol['sha256'] or self.identity['sources_sha256']!=_sources(self._root,protocol,self._assets) or
            self.identity['runtime_versions']!=_versions() or self.identity['parameter_names']!=[n for n,_ in self.module.named_parameters()] or
            self.identity['model_contract']!=_model_contract(self.module,protocol)):
            raise ValueError('native WiLoR source/config/runtime/model identity changed')


def build_wilor_runtime(upstream,config,*,assets,state=None,expected_initialization=None):
    """Original factory with complete actual pretrained load, or strict full-state reload."""
    if type(assets) is not WilorAssets:raise ValueError('producer-owned actual WiLoR assets required')
    assets.check();fresh=state is None
    if fresh!=(expected_initialization is None) or assets.restore is fresh:
        raise ValueError('fresh versus strict restore asset/state/initialization mode differs')
    root,_=_validated(upstream,config);protocol=wilor_protocol(root,config)
    _check_import_roots(assets.runtime,root)
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    if available<256*1024**3:raise ValueError('actual native WiLoR construction requires256GiB host reserve')
    os.environ['WILOR_PATH']=str(assets.runtime);_activate(root)
    # The original wrapper captures WILOR_PATH at import; validate both sides.
    _check_import_roots(assets.runtime,root)
    from omegaconf import OmegaConf
    from egoemg._train_utils import setup_runtime
    import egoemg.train as native
    import torch
    effective=OmegaConf.create(OmegaConf.to_container(config,resolve=True))
    del effective['_native_wilor'];effective.module.pretrained_path=str(assets.checkpoint) if fresh else None
    source_guard=_source_guard(root,protocol,assets);source_guard.check()
    sources=_sources(root,protocol,assets);versions=_versions();source_guard.check()
    if not fresh:identity=_restore_identity(expected_initialization,protocol,sources,versions,state)
    assets.check();setup_runtime(effective);seed=int(torch.initial_seed())
    if seed!=42:raise ValueError('original seeded native factory required')
    checkpoint=torch.load(str(assets.checkpoint),map_location='cpu',weights_only=False,mmap=True) if fresh else None
    expected=_extract_backbone(checkpoint) if fresh else None
    module=native.make_lightning_module(effective);_check_import_roots(assets.runtime,root);assets.check()
    contract=_model_contract(module,protocol)
    if fresh:
        actual=module.model.backbone.state_dict();backbone_sha=_require_exact_state(actual,expected)
        if len(actual)!=404:raise ValueError('all404original backbone tensors required')
        identity=dict(format='native_wilor_module_construction_v1',scope=_CONSTRUCTION_SCOPE,
            scientific_eligible=False,full_paper_complete=False,protocol_sha256=protocol['sha256'],sources_sha256=sources,
            runtime_versions=versions,model_schema=_schema(module.state_dict()),parameter_names=[n for n,_ in module.named_parameters()],
            model_contract=contract,initial_state_sha256=vision_state_sha256(module.state_dict()),
            initialization=dict(fresh_pretrained_verified=True,retained_tensor_count=len(actual),backbone_sha256=backbone_sha,
                asset=_catalog()['checkpoint'],source_revision=_catalog()['source_revision'],factory_seed=seed))
        identity['sha256']=_hash(identity)
    else:
        if (_schema(module.state_dict())!=identity['model_schema'] or
            [n for n,_ in module.named_parameters()]!=identity['parameter_names']):raise ValueError('actual native reload schema differs')
        module.load_state_dict(state,strict=True);_require_exact_state(module.state_dict(),state)
    del checkpoint,expected
    result=PreparedWilor(module,identity,'fresh_native_pretrained' if fresh else 'strict_self_contained_restore',fresh,
        vision_state_sha256(module.state_dict()),root,copy.deepcopy(config),assets,source_guard,identity['sha256'])
    _PREPARED.add(result);result.check();return result


def _optimizer_contract(module,optimizer,scheduler):
    import torch
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError('original AdamW/CosineAnnealingLR required')
    if (scheduler.optimizer is not optimizer or scheduler.T_max!=30 or scheduler.eta_min!=1e-5 or
        type(scheduler.last_epoch) is not int or not 0<=scheduler.last_epoch<=30 or
        scheduler.base_lrs!=[1e-5,1e-5] or scheduler._last_lr!=[1e-5,1e-5] or len(optimizer.param_groups)!=2):
        raise ValueError('native constant-LR cosine30 identity differs')
    named={id(p):n for n,p in module.named_parameters() if p.requires_grad};seen=[];groups=[]
    for name,group in zip(('head','backbone'),optimizer.param_groups):
        expected=['model.'+n for n,p in module.model.named_parameters() if n.startswith(name+'.') and p.requires_grad]
        actual=[named.get(id(p)) for p in group['params']]
        if (group.get('name')!=name or actual!=expected or group.get('lr')!=1e-5 or group.get('initial_lr')!=1e-5 or
            group['weight_decay']!=.01 or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group['amsgrad'] is not False or
            any(group.get(k,False) is not False for k in ('maximize','capturable','differentiable')) or
            any(group.get(k) is not None for k in ('foreach','fused'))):
            raise ValueError('original WiLoR component membership/AdamW hyperparameters differ')
        seen.extend(actual);groups.append(dict(name=name,parameter_names=actual,lr=group['lr'],initial_lr=group['initial_lr'],
            weight_decay=group['weight_decay'],betas=list(group['betas']),eps=group['eps'],amsgrad=group['amsgrad']))
    if len(seen)!=len(set(seen)) or set(seen)!=set(named.values()):raise ValueError('optimizer must cover each trainable parameter exactly once')
    return dict(format='native_wilor_optimizer_observation_v1',groups=groups,
        scheduler=dict(type='CosineAnnealingLR',T_max=30,eta_min=1e-5,last_epoch=scheduler.last_epoch,
            interval='epoch',frequency=1,behavior='constant_initial_lr_equals_eta_min'),scientific_eligible=False)


def wilor_optimizer_identity(prepared,optimizer,scheduler):
    if type(prepared) is not PreparedWilor:raise ValueError('actual native WiLoR prepared module required')
    prepared.check();return _optimizer_contract(prepared.module,optimizer,scheduler)
