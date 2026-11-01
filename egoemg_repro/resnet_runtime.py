"""Original ResNet factories with genuine cached ImageNet initialization.

The original forward, loss and optimizer implementations remain unchanged.
Construction and strict restore are engineering evidence, never a fit permit.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import threading
import weakref

from .classic_emg import _activate, _file_hash, _hash
from .resnet_literal import _validated, resnet_protocol
from .vision_literal_runtime import _schema, _stamp, _versions, vision_state_sha256


ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
ASSETS={
    'resnet18':('resnet18-f37072fd.pth',46830571,
        'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',100,20),
    'resnet50':('resnet50-0676ba61.pth',102530333,
        '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a',265,53),
    'resnet152':('resnet152-394f9c45.pth',241627721,
        '394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58',775,155),
}
_LOCAL=('egoemg_repro/resnet_runtime.py','egoemg_repro/resnet_literal.py',
    'egoemg_repro/vision_literal_runtime.py','egoemg_repro/visual.py','egoemg_repro/classic_emg.py')
_NATIVE=('egoemg/_train_utils.py','egoemg/utils.py','egoemg/metrics.py',
    'egoemg/models/modules/_vision_constants.py')
_RUNTIME=('torch.hub','torchvision.models.resnet','torchvision.models._api',
    'torchvision.models._utils','torch.optim.adamw','torch.optim.lr_scheduler')
_PREPARED=weakref.WeakSet()
_TRANSPORT_LOCK=threading.RLock()


def _asset_descriptor(variant):
    filename,size,digest,_,_=ASSETS[variant]
    return dict(filename=filename,bytes=size,sha256=digest,
        source_url='https://download.pytorch.org/models/'+filename,
        enum='torchvision.models.ResNet'+variant[len('resnet'):]+'_Weights.IMAGENET1K_V1',
        verification='official torchvision URL and filename hash prefix; full SHA from admitted local bytes')


def _sources(root,protocol):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported ResNet runtime source changed')
    result={name:_file_hash(ROOT/name)for name in _LOCAL}
    result.update({'upstream/'+name:digest for name,digest in protocol['sources_sha256'].items()})
    result.update({'upstream/'+name:_file_hash(root/name)for name in _NATIVE})
    result.update({'runtime/'+name:_file_hash(importlib.import_module(name).__file__)for name in _RUNTIME})
    return result


def _offline_asset(variant):
    import torch
    import torchvision.models as models
    if os.environ.get('EGOEMG_NO_PRETRAINED_DOWNLOAD','') not in ('','0'):
        raise ValueError('native pretrained initialization cannot silently fall back to random weights')
    descriptor=_asset_descriptor(variant)
    weights=getattr(models,'ResNet'+variant[len('resnet'):]+'_Weights').IMAGENET1K_V1
    if weights.url!=descriptor['source_url']:raise ValueError('native torchvision weight enumeration differs')
    path=Path(torch.hub.get_dir())/'checkpoints'/descriptor['filename']
    if not path.is_file():raise ValueError('genuine prefetched ResNet ImageNet asset required before factory')
    stamp=_stamp(path)
    if stamp[2]!=descriptor['bytes'] or _file_hash(path)!=descriptor['sha256'] or _stamp(path)!=stamp:
        raise ValueError('prefetched ResNet ImageNet asset content changed or differs')
    return descriptor,path.resolve(),stamp


@contextmanager
def _cached_transport_only():
    """A vanished cached file must fail, rather than trigger a hidden download."""
    import torch
    with _TRANSPORT_LOCK:
        original=torch.hub.download_url_to_file
        def forbidden(*args,**kwargs):raise ValueError('native ResNet construction requires validated cached weights')
        try:
            torch.hub.download_url_to_file=forbidden
            yield
        finally:torch.hub.download_url_to_file=original


def _backbone_state(asset):
    import torch
    raw=torch.load(str(asset),map_location='cpu',weights_only=True)
    _schema(raw)
    replacements={'conv1':'0','bn1':'1','layer1':'4','layer2':'5','layer3':'6','layer4':'7'}
    result={}
    for name,tensor in raw.items():
        prefix,_,suffix=name.partition('.')
        if prefix=='fc':continue
        if prefix not in replacements:raise ValueError('unexpected original ImageNet tensor name')
        result[replacements[prefix]+'.'+suffix]=tensor
    if set(raw)-{k for k in raw if not k.startswith('fc.')}!={'fc.weight','fc.bias'}:
        raise ValueError('original ImageNet classification tensors differ')
    return result


def _restore_identity(identity,variant,protocol,sources,versions,state):
    if not isinstance(identity,dict):raise ValueError('fresh ResNet initialization identity required for restore')
    identity=copy.deepcopy(identity)
    if (identity.get('format')!='original_resnet_module_construction_v1'
        or identity.get('sha256')!=_hash({k:v for k,v in identity.items()if k!='sha256'})
        or identity.get('variant')!=variant or identity.get('protocol_sha256')!=protocol['sha256']
        or identity.get('sources_sha256')!=sources or identity.get('runtime_versions')!=versions
        or any(identity.get(k)is not False for k in ('scientific_eligible','production_ready','full_paper_complete'))):
        raise ValueError('ResNet initialization source/config/runtime identity differs')
    init=identity.get('initialization',{});asset=_asset_descriptor(variant)
    if (init.get('fresh_pretrained_verified') is not True or init.get('asset')!=asset
        or init.get('retained_tensor_count')!=ASSETS[variant][3]
        or init.get('native_backbone_tensor_count')!=sum(ASSETS[variant][3:])
        or len(init.get('native_default_batch_counter_names',[]))!=ASSETS[variant][4]
        or init.get('pretrained_identity',{}).get('enum')!=asset['enum']
        or init.get('pretrained_identity',{}).get('source_url')!=asset['source_url']):
        raise ValueError('genuine ResNet pretrained initialization identity differs')
    if _schema(state)!=identity.get('model_schema'):raise ValueError('complete ResNet state keys/shapes/dtypes differ')
    return identity


@dataclass(eq=False)
class PreparedResnet:
    module:object
    identity:dict
    mode:str
    factory_pretrained:bool
    loaded_state_sha256:str
    _root:Path
    _config:object
    _asset_path:Path|None=None
    _asset_stamp:tuple|None=None

    def check(self):
        if self not in _PREPARED:raise ValueError('actual prepared native ResNet runtime required')
        protocol=resnet_protocol(self._root,self._config)
        schema={n:dict(shape=list(t.shape),dtype=str(t.dtype))for n,t in self.module.state_dict().items()}
        if (protocol['sha256']!=self.identity['protocol_sha256']
            or _sources(self._root,protocol)!=self.identity['sources_sha256']
            or _versions()!=self.identity['runtime_versions']
            or self.identity['sha256']!=_hash({k:v for k,v in self.identity.items()if k!='sha256'})
            or type(self.module).__module__!='egoemg.lightning'
            or type(self.module.model).__name__!='ResNetVisionPose'
            or self.module.model.backbone_type!=self.identity['variant']
            or [n for n,_ in self.module.named_parameters()]!=self.identity['parameter_names']
            or schema!=self.identity['model_schema']):
            raise ValueError('prepared ResNet source/config/native model identity changed')
        if self._asset_path is not None and _stamp(self._asset_path)!=self._asset_stamp:
            raise ValueError('admitted ResNet pretrained asset changed')


def build_resnet_runtime(upstream,config,*,state=None,expected_initialization=None):
    import torch
    from omegaconf import OmegaConf
    root,variant,_=_validated(upstream,config);protocol=resnet_protocol(root,config)
    if (state is None)!=(expected_initialization is None):
        raise ValueError('strict restore requires both complete state and initialization identity')
    sources=_sources(root,protocol);versions=_versions();fresh=state is None
    path=stamp=None
    if fresh:asset,path,stamp=_offline_asset(variant)
    else:identity=_restore_identity(expected_initialization,variant,protocol,sources,versions,state)
    _activate(root)
    from egoemg._train_utils import setup_runtime
    import egoemg.train as native
    effective=OmegaConf.create(OmegaConf.to_container(config,resolve=True))
    if not fresh:effective.module.pretrained=False
    setup_runtime(effective)
    with _cached_transport_only():module=native.make_lightning_module(effective)
    if type(module).__module__!='egoemg.lightning' or type(module.model).__name__!='ResNetVisionPose':
        raise ValueError('original ResNet module class differs')
    if fresh:
        from .visual import pretrained_weight_identity
        expected=_backbone_state(path);actual=module.model.backbone.state_dict()
        # The publisher's old ImageNet states predate num_batches_tracked.
        # Original torchvision BatchNorm loading supplies scalar int64 zeros;
        # observe those defaults, without adding/replacing any model tensor.
        defaults={n+'.num_batches_tracked'for n,m in module.model.backbone.named_modules()
                  if isinstance(m,torch.nn.BatchNorm2d)}
        if (len(expected)!=ASSETS[variant][3] or len(defaults)!=ASSETS[variant][4]
            or set(actual)-set(expected)!=defaults or set(expected)-set(actual)
            or any(actual[k].shape!=torch.Size([]) or actual[k].dtype!=torch.int64 or actual[k].item()!=0 for k in defaults)
            or any(expected[k].dtype!=actual[k].dtype or expected[k].shape!=actual[k].shape
                   or not torch.equal(expected[k],actual[k])for k in expected)):
            raise ValueError('native ResNet did not retain every genuine pretrained backbone tensor')
        if _stamp(path)!=stamp:raise ValueError('ResNet pretrained asset changed during construction')
        identity=dict(format='original_resnet_module_construction_v1',variant=variant,
            scope='genuine original construction only; no data, GPU, optimization or accuracy claim',
            scientific_eligible=False,production_ready=False,full_paper_complete=False,
            protocol_sha256=protocol['sha256'],sources_sha256=sources,runtime_versions=versions,
            model_schema=_schema(module.state_dict()),parameter_names=[n for n,_ in module.named_parameters()],
            parameter_count=sum(p.numel()for p in module.parameters()),
            initial_state_sha256=vision_state_sha256(module.state_dict()),
            initialization=dict(fresh_pretrained_verified=True,retained_tensor_count=len(expected),asset=asset,
                native_backbone_tensor_count=len(actual),native_default_batch_counter_names=sorted(defaults),
                pretrained_identity=pretrained_weight_identity(module.model)))
        identity['sha256']=_hash(identity)
    else:
        if _schema(module.state_dict())!=identity['model_schema']:raise ValueError('original ResNet reload factory schema differs')
        module.load_state_dict(state,strict=True)
        if vision_state_sha256(module.state_dict())!=vision_state_sha256(state):
            raise ValueError('strict restored ResNet named tensor bytes differ')
    result=PreparedResnet(module,identity,'fresh_native_pretrained' if fresh else 'strict_self_contained_restore',
        fresh,vision_state_sha256(module.state_dict()),root,copy.deepcopy(config),path,stamp)
    _PREPARED.add(result);result.check();return result


def resnet_optimizer_identity(prepared,optimizer,scheduler):
    if type(prepared) is not PreparedResnet or prepared not in _PREPARED:raise ValueError('actual prepared ResNet required')
    from .vision_literal_runtime import vision_optimizer_identity
    result=vision_optimizer_identity(prepared,optimizer,scheduler)
    result['format']='original_resnet_optimizer_observation_v1'
    return result
