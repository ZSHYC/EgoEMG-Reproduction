"""Unchanged original cotrain-S factory and strict complete learned-state reload.

Construction/optimizer evidence alone grants no data or scientific fit permit.
All checkpoint tensors must originate in trusted local task artifacts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import importlib.metadata
import math
from pathlib import Path
import platform

from .classic_emg import _activate,_file_hash,_hash
from .emg_initializer_literal import _validated,emg_initializer_protocol
from .vision_literal_runtime import _schema,vision_state_sha256 as emg_initializer_state_sha256

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_emg_initializer_module_construction_v1'
LOCAL=('egoemg_repro/emg_initializer_runtime.py','egoemg_repro/emg_initializer_literal.py',
    'egoemg_repro/vision_literal_runtime.py','egoemg_repro/classic_emg.py')
PACKAGES=('torch','torchvision','numpy','timm','pytorch-lightning','hydra-core','omegaconf')
RUNTIME=('torch.optim.adamw','torch.optim.lr_scheduler','torch.nn.functional','pytorch_lightning.core.module')


def _sources(root,protocol):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported initializer runtime source changed')
    sources={name:_file_hash(ROOT/name) for name in LOCAL}
    sources.update({'upstream/'+name:digest for name,digest in protocol['sources_sha256'].items()})
    sources.update({'upstream/'+str(path.relative_to(root)):_file_hash(path) for path in root.glob('egoemg/**/*.py')})
    sources.update({'runtime/'+name:_file_hash(importlib.import_module(name).__file__) for name in RUNTIME})
    return sources


def _versions():
    return dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in PACKAGES})


def _restore_identity(identity,protocol,sources,versions,state):
    if (not isinstance(identity,dict) or identity.get('format')!=FORMAT or
        identity.get('sha256')!=_hash({k:v for k,v in identity.items() if k!='sha256'}) or
        identity.get('protocol_sha256')!=protocol['sha256'] or identity.get('sources_sha256')!=sources or
        identity.get('runtime_versions')!=versions or
        identity.get('initialization')!=dict(source='original factory random initialization',seed=42,pretrained=False) or
        any(identity.get(k) is not False for k in ('scientific_eligible','production_ready','full_paper_complete'))):
        raise ValueError('original initializer construction lineage/config/runtime differs')
    if _schema(state)!=identity.get('model_schema'):raise ValueError('complete initializer state keys/shapes/dtypes differ')
    return copy.deepcopy(identity)


@dataclass
class PreparedEmgInitializer:
    module: object
    identity: dict
    mode: str
    loaded_state_sha256: str
    _root: Path
    _config: object

    def check(self):
        protocol=emg_initializer_protocol(self._root,self._config)
        if (self.identity.get('sha256')!=_hash({k:v for k,v in self.identity.items() if k!='sha256'}) or
            protocol['sha256']!=self.identity['protocol_sha256'] or
            _sources(self._root,protocol)!=self.identity['sources_sha256'] or _versions()!=self.identity['runtime_versions'] or
            _schema(self.module.state_dict())!=self.identity['model_schema'] or
            [name for name,_ in self.module.named_parameters()]!=self.identity['parameter_names']):
            raise ValueError('prepared initializer source/config/runtime/schema changed')


def build_emg_initializer_runtime(upstream,config,*,state=None,expected_initialization=None):
    root,_=_validated(upstream,config);protocol=emg_initializer_protocol(root,config)
    if (state is None)!=(expected_initialization is None):
        raise ValueError('strict initializer restore requires complete state and original initialization identity')
    fresh=state is None;sources=_sources(root,protocol);versions=_versions()
    if not fresh:identity=_restore_identity(expected_initialization,protocol,sources,versions,state)
    _activate(root)
    from egoemg._train_utils import setup_runtime
    from egoemg import train as native
    # Seed before constructing trainable random parameters, not after it.
    setup_runtime(config);module=native.make_lightning_module(config)
    if type(module).__module__!='egoemg.lightning' or type(module.model).__name__!='Emg2PoseFormer':
        raise ValueError('original EMG initializer module class differs')
    if fresh:
        identity=dict(format=FORMAT,scope='original random module construction only; no fit/data permit',
            scientific_eligible=False,production_ready=False,full_paper_complete=False,
            protocol_sha256=protocol['sha256'],sources_sha256=sources,runtime_versions=versions,
            model_schema=_schema(module.state_dict()),parameter_names=[n for n,_ in module.named_parameters()],
            parameter_count=sum(p.numel() for p in module.parameters()),
            initial_state_sha256=emg_initializer_state_sha256(module.state_dict()),
            effective_loss_weights=protocol['effective_loss_weights'],batch_augmentation=protocol['batch_augmentation'],
            initialization=dict(source='original factory random initialization',seed=42,pretrained=False))
        identity['sha256']=_hash(identity)
    else:
        if _schema(module.state_dict())!=identity['model_schema']:raise ValueError('original reload factory schema differs')
        module.load_state_dict(state,strict=True)
        if emg_initializer_state_sha256(module.state_dict())!=emg_initializer_state_sha256(state):
            raise ValueError('strict restored initializer tensor bytes differ')
    result=PreparedEmgInitializer(module,identity,'fresh_native_random' if fresh else 'strict_self_contained_restore',
        emg_initializer_state_sha256(module.state_dict()),root,copy.deepcopy(config))
    result.check();return result


def emg_initializer_optimizer_identity(prepared,optimizer,scheduler):
    """Validate native all-parameter AdamW and original150-epoch cosine objects."""
    import torch
    prepared.check()
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError('original initializer AdamW/cosine objects required')
    if (scheduler.optimizer is not optimizer or scheduler.T_max!=150 or scheduler.eta_min!=5e-6 or
        type(scheduler.last_epoch) is not int or not 0<=scheduler.last_epoch<=150 or scheduler.base_lrs!=[.0005] or
        len(optimizer.param_groups)!=1):raise ValueError('original initializer scheduler/group recipe differs')
    named={id(p):n for n,p in prepared.module.named_parameters() if p.requires_grad};group=optimizer.param_groups[0]
    names=[named.get(id(p)) for p in group['params']]
    expected_lr=5e-6+(.0005-5e-6)*(1+math.cos(math.pi*scheduler.last_epoch/150))/2
    if (names!=list(named.values()) or any(n is None for n in names) or
        not math.isclose(group['lr'],expected_lr,rel_tol=1e-10,abs_tol=1e-14) or
        group['weight_decay']!=.01 or group['eps']!=1e-8 or group['betas']!=(.9,.999) or group['amsgrad'] is not False):
        raise ValueError('initializer native all-parameter optimizer coverage/options differ')
    return dict(format='original_emg_initializer_optimizer_v1',
        groups=[dict(parameter_names=names,lr=group['lr'],weight_decay=group['weight_decay'],betas=list(group['betas']),eps=group['eps'])],
        scheduler=dict(T_max=scheduler.T_max,eta_min=scheduler.eta_min,last_epoch=scheduler.last_epoch,base_lrs=scheduler.base_lrs))
