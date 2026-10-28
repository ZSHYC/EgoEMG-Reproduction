"""Original EgoEMG S/M/L factory, complete tensor reload and native optimizer.

Runtime construction is not a data admission or a full-training permit.
All learned tensors must come from trusted local task artifacts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import importlib.metadata
import math
from pathlib import Path
import platform

from .classic_emg import _activate, _file_hash, _hash
from .emgformer_literal import _validated, emgformer_protocol
from .vision_literal_runtime import _schema, vision_state_sha256 as emgformer_state_sha256

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = _file_hash(__file__)
FORMAT = 'original_egoemg_emgformer_construction_v1'
PARAMETERS = dict(small=3460134, middle=6617382, large=16320806)
LOCAL = ('egoemg_repro/emgformer_runtime.py', 'egoemg_repro/emgformer_literal.py',
         'egoemg_repro/vision_literal_runtime.py', 'egoemg_repro/classic_emg.py')
PACKAGES = ('torch', 'torchvision', 'numpy', 'timm', 'pytorch-lightning', 'hydra-core', 'omegaconf')
RUNTIME = ('torch.optim.adamw', 'torch.optim.lr_scheduler', 'torch.nn.functional',
           'pytorch_lightning.core.module')
EXECUTION_DEFAULTS = dict(maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)


def _sources(root, protocol):
    if _file_hash(__file__) != IMPORTED_SHA:
        raise ValueError('imported EMGFormer runtime source changed')
    result = {name: _file_hash(ROOT/name) for name in LOCAL}
    result.update({'upstream/'+n: h for n, h in protocol['sources_sha256'].items()})
    result.update({'upstream/'+str(p.relative_to(root)): _file_hash(p) for p in root.glob('egoemg/**/*.py')})
    result.update({'runtime/'+n: _file_hash(importlib.import_module(n).__file__) for n in RUNTIME})
    return result


def _versions():
    return dict(python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in PACKAGES})


def _restore_identity(identity, size, protocol, sources, versions, state):
    if (not isinstance(identity, dict) or identity.get('format') != FORMAT or identity.get('size') != size
            or identity.get('sha256') != _hash({k: v for k, v in identity.items() if k != 'sha256'})
            or identity.get('protocol_sha256') != protocol['sha256'] or identity.get('sources_sha256') != sources
            or identity.get('runtime_versions') != versions or identity.get('parameter_count') != PARAMETERS[size]
            or identity.get('effective_loss_weights') != protocol['effective_loss_weights']
            or identity.get('batch_augmentation') != protocol['batch_augmentation']
            or identity.get('initialization') != dict(source='original factory random initialization', seed=42, pretrained=False)
            or any(identity.get(k) is not False for k in ('scientific_eligible', 'production_ready', 'full_paper_complete'))):
        raise ValueError('EMGFormer complete state lineage/config/runtime differs')
    if _schema(state) != identity.get('model_schema'):
        raise ValueError('EMGFormer complete named tensor schema differs')
    return copy.deepcopy(identity)


@dataclass
class PreparedEmgformer:
    module: object
    identity: dict
    mode: str
    loaded_state_sha256: str
    _root: Path
    _config: object

    def check(self):
        protocol = emgformer_protocol(self._root, self._config)
        if (self.identity.get('sha256') != _hash({k: v for k, v in self.identity.items() if k != 'sha256'})
                or protocol['sha256'] != self.identity['protocol_sha256']
                or _sources(self._root, protocol) != self.identity['sources_sha256']
                or _versions() != self.identity['runtime_versions']
                or self.identity.get('effective_loss_weights') != protocol['effective_loss_weights']
                or self.identity.get('batch_augmentation') != protocol['batch_augmentation']
                or dict(self.module.loss_weights) != protocol['raw_loss_weights']
                or any(self.identity.get(k) is not False for k in ('scientific_eligible', 'production_ready', 'full_paper_complete'))
                or _schema(self.module.state_dict()) != self.identity['model_schema']
                or [n for n, _ in self.module.named_parameters()] != self.identity['parameter_names']
                or any(not p.requires_grad for p in self.module.parameters())
                or sum(p.numel() for p in self.module.parameters()) != PARAMETERS[self.identity['size']]):
            raise ValueError('prepared original EMGFormer source/config/runtime/schema changed')


def build_emgformer_runtime(upstream, config, *, state=None, expected_initialization=None):
    root, size, _ = _validated(upstream, config)
    protocol = emgformer_protocol(root, config)
    if (state is None) != (expected_initialization is None):
        raise ValueError('strict reload requires both complete state and construction identity')
    fresh = state is None
    sources, versions = _sources(root, protocol), _versions()
    if not fresh:
        identity = _restore_identity(expected_initialization, size, protocol, sources, versions, state)
    _activate(root)
    from egoemg._train_utils import setup_runtime
    from egoemg import train as native
    setup_runtime(config)
    module = native.make_lightning_module(config)
    if type(module).__module__ != 'egoemg.lightning' or type(module.model).__name__ != 'Emg2PoseFormer':
        raise ValueError('original EMGFormer native class differs')
    if sum(p.numel() for p in module.parameters()) != PARAMETERS[size]:
        raise ValueError('original EMGFormer size/parameter count differs')
    if fresh:
        identity = dict(format=FORMAT, size=size, protocol_sha256=protocol['sha256'], sources_sha256=sources,
            runtime_versions=versions, model_schema=_schema(module.state_dict()),
            parameter_names=[n for n, _ in module.named_parameters()], parameter_count=PARAMETERS[size],
            initial_state_sha256=emgformer_state_sha256(module.state_dict()),
            effective_loss_weights=protocol['effective_loss_weights'], batch_augmentation=protocol['batch_augmentation'],
            initialization=dict(source='original factory random initialization', seed=42, pretrained=False),
            seed_policy='original setup_runtime before original factory; explicit launch ordering',
            scientific_eligible=False, production_ready=False, full_paper_complete=False)
        identity['sha256'] = _hash(identity)
    else:
        module.load_state_dict(state, strict=True)
        if emgformer_state_sha256(module.state_dict()) != emgformer_state_sha256(state):
            raise ValueError('strict EMGFormer reload changed tensor bytes')
    prepared = PreparedEmgformer(module, identity, 'fresh_native_random' if fresh else 'strict_self_contained_restore',
        emgformer_state_sha256(module.state_dict()), root, copy.deepcopy(config))
    prepared.check()
    return prepared


def emgformer_optimizer_identity(prepared, optimizer, scheduler):
    import torch
    prepared.check()
    protocol = emgformer_protocol(prepared._root, prepared._config)
    recipe, policy = protocol['optimizer'], protocol['lr_scheduler']['scheduler']
    horizon, eta, initial = policy['T_max'], policy['eta_min'], recipe['lr']
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError('original EMGFormer AdamW/cosine types required')
    if (scheduler.optimizer is not optimizer or scheduler.T_max != horizon or scheduler.eta_min != eta
            or type(scheduler.last_epoch) is not int or not 0 <= scheduler.last_epoch <= horizon
            or scheduler.base_lrs != [initial] or len(optimizer.param_groups) != 1):
        raise ValueError('original EMGFormer cosine horizon/eta/group recipe differs')
    named = {id(p): n for n, p in prepared.module.named_parameters()}
    group = optimizer.param_groups[0]
    names = [named.get(id(p)) for p in group['params']]
    expected_lr = eta + (initial-eta)*(1+math.cos(math.pi*scheduler.last_epoch/horizon))/2
    if (names != list(named.values()) or any(n is None for n in names)
            or not math.isclose(group['lr'], expected_lr, rel_tol=1e-10, abs_tol=1e-14)
            or group['weight_decay'] != recipe['weight_decay'] or group['eps'] != recipe['eps']
            or group['betas'] != tuple(recipe['betas']) or group['amsgrad'] is not recipe['amsgrad']
            or any(k not in group or group[k] is not v for k, v in EXECUTION_DEFAULTS.items())):
        raise ValueError('original EMGFormer optimizer parameter coverage/options differ')
    return dict(format='original_egoemg_emgformer_optimizer_v1',
        groups=[dict(parameter_names=names, lr=group['lr'], weight_decay=group['weight_decay'],
                     betas=list(group['betas']), eps=group['eps'],
                     execution_defaults={k: group[k] for k in EXECUTION_DEFAULTS})],
        scheduler=dict(T_max=scheduler.T_max, eta_min=scheduler.eta_min,
                       last_epoch=scheduler.last_epoch, base_lrs=scheduler.base_lrs))
