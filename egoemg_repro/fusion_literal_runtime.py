"""Original fusion construction with checked local branch transfer and full reload.

This proves construction/transfer only. File identity is not evidence that an
initializer was trained, selected correctly, or trained on the paper's corpus.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import stat

from .classic_emg import _activate, _file_hash, _hash
from .data import file_sha256
from .fusion_literal import _validated, fusion_protocol
from .vision_literal_runtime import _schema, _versions, vision_state_sha256 as fusion_state_sha256


ROOT=Path(__file__).resolve().parents[1]
_IMPORTED_SHA=_file_hash(__file__)
_LOCAL=('egoemg_repro/fusion_literal.py','egoemg_repro/fusion_literal_runtime.py',
    'egoemg_repro/classic_emg.py','egoemg_repro/data.py','egoemg_repro/vision_literal_runtime.py')
_NATIVE=('egoemg/models/modules/base.py','egoemg/models/modules/_pooling.py',
    'egoemg/models/modules/_vision_constants.py','egoemg/models/featurizers/tds.py',
    'egoemg/models/decoders/transformer.py','egoemg/metrics.py','egoemg/utils.py')
_RUNTIME=('timm.models.vision_transformer','torch.nn.modules.module','torch.optim.adamw','torch.optim.lr_scheduler')
_PREFIXES=('featurizer.','decoder.','backbone.','avgpool.','vision_backbone.','vision_proj.',
    'fusion_proj.','head_vision.','temporal_attn.','token_fusion.','early_fusion.')


def _sources(root,protocol):
    if _file_hash(__file__)!=_IMPORTED_SHA:raise ValueError('imported fusion runtime source changed')
    result={name:_file_hash(ROOT/name) for name in _LOCAL}
    result.update({'upstream/'+name:digest for name,digest in protocol['sources_sha256'].items()})
    result.update({'upstream/'+name:_file_hash(root/name) for name in _NATIVE})
    result.update({'runtime/'+name:_file_hash(importlib.import_module(name).__file__) for name in _RUNTIME})
    return result


def _stamp(path):
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('initializer must remain a regular nonsymlink file')
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _check_assets(assets):
    for path,before in assets:
        try:after=_stamp(path)
        except OSError as exc:raise ValueError('admitted fusion initializer disappeared') from exc
        if before!=after:raise ValueError('admitted fusion initializer changed')


def _admit(path,digest):
    if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('explicit lowercase initializer SHA256 required')
    try:
        path=Path(path).expanduser().absolute()
        if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('initializer symlink is not admitted')
        before=_stamp(path)
        if file_sha256(path)!=digest or _stamp(path)!=before:raise ValueError('initializer hash differs or file changed')
    except (OSError,TypeError) as exc:raise ValueError('initializer file unavailable') from exc
    return path,before


def _read_state(path):
    import torch
    packet=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
    if not isinstance(packet,Mapping):raise ValueError('native initializer checkpoint mapping required')
    state=packet.get('state_dict',packet);_schema(state)
    names={}
    for name in state:
        stripped=name[6:] if name.startswith('model.') else name
        if stripped in names:raise ValueError('ambiguous duplicate initializer tensor name')
        names[stripped]=name
    return state,names


def _input_branches(vision,emg):
    vs,vnames=_read_state(vision);es,enames=_read_state(emg)
    backbone={k[len('backbone.'):]:vs[name] for k,name in vnames.items() if k.startswith('backbone.')}
    head={k[len('head.'):]:vs[name] for k,name in vnames.items() if k.startswith('head.')}
    shapes={'0.weight':[512,384],'0.bias':[512],'3.weight':[22,512],'3.bias':[22]}
    if not backbone or set(head)!=set(shapes) or any(list(head[k].shape)!=s for k,s in shapes.items()):
        raise ValueError('complete canonical native ViT-S vision backbone/head required')
    if not all(any(k.startswith(prefix) for k in enames) for prefix in ('featurizer.','decoder.')):
        raise ValueError('EMG initializer requires both native featurizer and decoder tensors')
    return vs,es,enames,backbone,head


def _equal_native(actual,expected):
    import torch
    # Native load_state_dict copies into the model's dtype. Preserve that rule
    # rather than changing source tensors or claiming a different byte dtype.
    return actual.shape==expected.shape and torch.equal(actual,expected.to(dtype=actual.dtype))


def _verify_vision(model,backbone,head):
    for actual,expected in ((model.vision_backbone.state_dict(),backbone),(model.head_vision.state_dict(),head)):
        if set(actual)!=set(expected) or any(not _equal_native(actual[k],expected[k]) for k in actual):
            raise ValueError('native vision branch transfer is incomplete or differs')
    return dict(backbone_tensors=len(backbone),head_tensors=len(head),
        source_backbone_state_sha256=fusion_state_sha256(backbone),source_head_state_sha256=fusion_state_sha256(head),
        loaded_backbone_state_sha256=fusion_state_sha256(model.vision_backbone.state_dict()),
        loaded_head_state_sha256=fusion_state_sha256(model.head_vision.state_dict()),
        comparison='complete key/shape match and exact values after native target-dtype conversion')


def _verify_emg(model,state,names):
    target=model.state_dict();matched=[];skipped=[]
    for stripped,source in names.items():
        tensor=state[source];item=dict(source_key=source,target_key=stripped,source_shape=list(tensor.shape),source_dtype=str(tensor.dtype))
        if not stripped.startswith(_PREFIXES):reason='not_backbone_prefix'
        elif stripped not in target:reason='missing_target'
        elif target[stripped].shape!=tensor.shape:reason='shape_mismatch'
        else:
            actual=target[stripped]
            if not _equal_native(actual,tensor):raise ValueError('native EMG matched tensor transfer differs')
            matched.append(dict(item,target_dtype=str(actual.dtype),loaded_sha256=fusion_state_sha256({stripped:actual})))
            continue
        skipped.append(dict(item,reason=reason,target_shape=list(target[stripped].shape) if stripped in target else None))
    if not all(any(x['target_key'].startswith(prefix) for x in matched) for prefix in ('featurizer.','decoder.')):
        raise ValueError('native EMG transfer did not populate both featurizer and decoder')
    return dict(matched=matched,skipped=skipped,strict=False,weight_expansion_performed=False,
        comparison='original allowed prefixes and exact names/shapes; native target-dtype conversion retained')


def _saved_initialization(identity):
    """Require the complete transfer receipt, beyond its content checksum."""
    import torch
    def digest(value):
        return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    try:
        init=identity['initialization'];schema=identity['model_schema']
        if (type(identity['parameters']) is not int or identity['parameters']<=0 or
            init['initializer_training_provenance_verified'] is not False):raise ValueError('invalid fusion initialization scope')
        for role in ('vision_file','emg_file'):
            entry=init[role]
            if (type(entry['bytes']) is not int or entry['bytes']<=0 or
                not all(digest(entry[k]) for k in ('sha256','state_sha256'))):raise ValueError('invalid branch file receipt')
        vision=init['vision_transfer']
        for label,prefix in (('backbone','model.vision_backbone.'),('head','model.head_vision.')):
            count=sum(k.startswith(prefix) for k in schema)
            if count<=0 or type(vision[label+'_tensors']) is not int or vision[label+'_tensors']!=count:
                raise ValueError('vision transfer receipt differs from complete schema')
            if not all(digest(vision[k]) for k in ('source_'+label+'_state_sha256','loaded_'+label+'_state_sha256')):
                raise ValueError('invalid vision tensor receipt')
        emg=init['emg_transfer'];matched=emg['matched'];skipped=emg['skipped']
        if (emg['strict'] is not False or emg['weight_expansion_performed'] is not False or
            not isinstance(matched,list) or not isinstance(skipped,list) or
            not all(any(x['target_key'].startswith(prefix) for x in matched) for prefix in ('featurizer.','decoder.'))):
            raise ValueError('incomplete native EMG transfer receipt')
        inventory=init['emg_source_schema'];normalized={}
        if not isinstance(inventory,dict) or not inventory:raise ValueError('complete EMG source inventory required')
        for source,descriptor in inventory.items():
            if type(source) is not str or not source or not isinstance(descriptor,dict):
                raise ValueError('invalid EMG source descriptor')
            shape=descriptor['shape'];dtype=descriptor['dtype']
            if (type(shape) is not list or any(type(n) is not int or n<0 for n in shape) or
                type(dtype) is not str or not dtype.startswith('torch.') or
                not isinstance(getattr(torch,dtype[6:],None),torch.dtype)):
                raise ValueError('invalid EMG source shape/dtype')
            target=source[6:] if source.startswith('model.') else source
            if not target or target in normalized:raise ValueError('ambiguous normalized EMG source inventory')
            normalized[target]=source
        seen=set()
        for loaded,items in ((True,matched),(False,skipped)):
            for item in items:
                source=item['source_key'];target=item['target_key']
                if source not in inventory or source in seen or normalized.get(target)!=source:
                    raise ValueError('duplicate or inconsistent EMG source transfer receipt')
                seen.add(source);descriptor=inventory[source];destination=schema.get('model.'+target)
                if item['source_shape']!=descriptor['shape'] or item['source_dtype']!=descriptor['dtype']:
                    raise ValueError('EMG transfer source differs from inventory')
                if not target.startswith(_PREFIXES):reason='not_backbone_prefix'
                elif destination is None:reason='missing_target'
                elif descriptor['shape']!=destination['shape']:reason='shape_mismatch'
                else:reason=None
                if loaded:
                    if (reason is not None or item['target_dtype']!=destination['dtype'] or
                        not digest(item['loaded_sha256'])):
                        raise ValueError('matched EMG receipt differs from native loading rule')
                elif (reason is None or item['reason']!=reason or
                    item['target_shape']!=(destination['shape'] if destination is not None else None)):
                    raise ValueError('skipped EMG receipt differs from native loading rule')
        if seen!=set(inventory):raise ValueError('EMG transfer receipt does not cover every source tensor')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('incomplete fusion initialization receipt') from exc


@contextmanager
def _restore_offline(enabled):
    key='EGOEMG_NO_PRETRAINED_DOWNLOAD';prior=os.environ.get(key)
    if enabled:os.environ[key]='1'
    try:yield
    finally:
        if enabled:
            if prior is None:os.environ.pop(key,None)
            else:os.environ[key]=prior


@dataclass
class PreparedFusion:
    module: object
    identity: dict
    mode: str
    loaded_state_sha256: str
    _root: Path
    _config: object
    _assets: tuple=()

    def check(self):
        _saved_initialization(self.identity)
        protocol=fusion_protocol(self._root,self._config)
        if (self.identity.get('sha256')!=_hash({k:v for k,v in self.identity.items() if k!='sha256'}) or
            protocol['sha256']!=self.identity['protocol_sha256'] or
            _sources(self._root,protocol)!=self.identity['sources_sha256'] or
            _versions()!=self.identity['runtime_versions']):
            raise ValueError('prepared fusion source/config/runtime identity changed')
        _check_assets(self._assets)


def build_fusion_runtime(upstream,config,*,vision_checkpoint=None,vision_sha256=None,
        emg_checkpoint=None,emg_sha256=None,state=None,expected_initialization=None):
    """Use the original factory/loaders, or restore all native tensors strictly.

    Local inputs must be trusted checkpoints owned by this task. Their hashes
    bind bytes only; producer training/selection/data provenance is a later gate.
    """
    root,value=_validated(upstream,config);protocol=fusion_protocol(root,config)
    init_args=(vision_checkpoint,vision_sha256,emg_checkpoint,emg_sha256)
    fresh=state is None and expected_initialization is None
    if fresh and any(x is None for x in init_args):raise ValueError('both local branch files and hashes required')
    if not fresh and (state is None or expected_initialization is None or any(x is not None for x in init_args)):
        raise ValueError('complete-state restore must be separate from branch initialization')
    _activate(root);sources=_sources(root,protocol);versions=_versions();assets=()
    if fresh:
        vision,vstamp=_admit(vision_checkpoint,vision_sha256);emg,estamp=_admit(emg_checkpoint,emg_sha256)
        vs,es,enames,backbone,head=_input_branches(vision,emg);assets=((vision,vstamp),(emg,estamp))
    else:
        if not isinstance(expected_initialization,dict):raise ValueError('fresh fusion initialization identity required')
        identity=copy.deepcopy(expected_initialization)
        if (identity.get('format')!='original_fusion_module_construction_v1' or
            identity.get('sha256')!=_hash({k:v for k,v in identity.items() if k!='sha256'}) or
            identity.get('protocol_sha256')!=protocol['sha256'] or identity.get('sources_sha256')!=sources or
            identity.get('runtime_versions')!=versions or
            any(identity.get(k) is not False for k in ('scientific_eligible','production_ready','full_paper_complete')) or
            identity.get('model_schema')!=_schema(state)):
            raise ValueError('fusion initialization identity or complete typed state differs')
        _saved_initialization(identity)
    from omegaconf import OmegaConf
    from egoemg._train_utils import setup_runtime
    import egoemg.train as native
    effective=OmegaConf.create(value)
    effective.vision_vit_checkpoint=str(vision) if fresh else None
    effective.module.vision_pretrained_checkpoint=str(vision) if fresh else None
    effective.pretrained_emg_checkpoint=str(emg) if fresh else None
    setup_runtime(effective)
    _check_assets(assets)
    with _restore_offline(True):module=native.make_lightning_module(effective)
    if type(module).__module__!='egoemg.lightning' or type(module.model).__name__!='MidFusionPoseFormer':
        raise ValueError('original fusion module class differs')
    if fresh:
        vision_transfer=_verify_vision(module.model,backbone,head)
        emg_transfer=_verify_emg(module.model,es,enames)
        _check_assets(assets)
        identity=dict(format='original_fusion_module_construction_v1',scientific_eligible=False,
            production_ready=False,full_paper_complete=False,scope='native construction/transfer only; no data or training admission',
            protocol_sha256=protocol['sha256'],sources_sha256=sources,runtime_versions=versions,
            model_schema=_schema(module.state_dict()),parameters=sum(p.numel() for p in module.parameters()),
            initial_state_sha256=fusion_state_sha256(module.state_dict()),
            initialization=dict(vision_file=dict(bytes=vstamp[2],sha256=vision_sha256,state_sha256=fusion_state_sha256(vs)),
                emg_file=dict(bytes=estamp[2],sha256=emg_sha256,state_sha256=fusion_state_sha256(es)),
                vision_transfer=vision_transfer,emg_source_schema=_schema(es),emg_transfer=emg_transfer,
                initializer_training_provenance_verified=False,
                residual_rule='original normal(std=1e-5) last weight and zero bias; not overwritten'))
        identity['sha256']=_hash(identity)
    else:
        if _schema(module.state_dict())!=identity['model_schema']:raise ValueError('native fusion restore factory schema differs')
        module.load_state_dict(state,strict=True)
        if fusion_state_sha256(module.state_dict())!=fusion_state_sha256(state):raise ValueError('strict fusion tensor bytes differ')
    result=PreparedFusion(module,identity,'fresh_local_branch_initialization' if fresh else 'strict_self_contained_restore',
        fusion_state_sha256(module.state_dict()),root,copy.deepcopy(config),assets)
    result.check();return result


def fusion_optimizer_identity(prepared,optimizer,scheduler):
    """Observe original component AdamW/cosine, including temporal attention."""
    import torch
    prepared.check()
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError('native fusion AdamW and cosine scheduler required')
    if (scheduler.optimizer is not optimizer or scheduler.T_max!=150 or scheduler.eta_min!=1e-5 or
        type(scheduler.last_epoch) is not int or not 0<=scheduler.last_epoch<=150):
        raise ValueError('native fusion scheduler ownership/budget differs')
    labels=('featurizer','decoder','vision_proj','fusion_proj','head','head_vision','vision_backbone','_unassigned')
    named=dict(prepared.module.named_parameters());by_id={id(p):n for n,p in named.items() if p.requires_grad}
    if len(by_id)!=len(named):raise ValueError('native fusion recipe does not freeze optimizer parameters')
    expected={label:[] for label in labels}
    for name in named:
        label=next((key for key in labels[:-1] if name.startswith('model.'+key+'.')),'_unassigned')
        expected[label].append(name)
    if (any(not expected[label] for label in labels) or len(optimizer.param_groups)!=len(labels) or
        scheduler.base_lrs!=[.0001]*len(labels)):
        raise ValueError('native fusion component count/base learning rates differ')
    rate=1e-5+(.0001-1e-5)*(1+math.cos(math.pi*scheduler.last_epoch/150))/2
    groups=[];seen=[]
    for label,group in zip(labels,optimizer.param_groups):
        names=[by_id.get(id(p)) for p in group['params']]
        if (group.get('name')!=label or names!=expected[label] or group.get('initial_lr')!=.0001 or
            not math.isclose(group['lr'],rate,rel_tol=1e-10,abs_tol=1e-14) or group['weight_decay']!=.01 or
            tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group['amsgrad'] is not False or
            any(group.get(k,False) is not False for k in ('maximize','capturable','differentiable')) or
            any(group.get(k) is not None for k in ('foreach','fused'))):
            raise ValueError('native fusion component membership or AdamW hyperparameters differ')
        seen.extend(names);groups.append(dict(name=label,parameter_names=names,lr=group['lr'],initial_lr=group['initial_lr'],
            weight_decay=group['weight_decay'],betas=list(group['betas']),eps=group['eps'],amsgrad=group['amsgrad']))
    if len(seen)!=len(set(seen)) or set(seen)!=set(by_id.values()):
        raise ValueError('fusion optimizer must cover each trainable parameter exactly once')
    if len(scheduler._last_lr)!=len(labels) or any(not math.isclose(x,rate,rel_tol=1e-10,abs_tol=1e-14) for x in scheduler._last_lr):
        raise ValueError('fusion scheduler stored rates differ from native component groups')
    result=dict(format='original_fusion_optimizer_observation_v1',groups=groups,
        scheduler=dict(type='CosineAnnealingLR',T_max=150,eta_min=1e-5,last_epoch=scheduler.last_epoch,interval='epoch',frequency=1),
        scientific_eligible=False)
    result['sha256']=_hash(result);return result
