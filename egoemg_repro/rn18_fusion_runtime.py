"""Original RN18+S construction and self-contained reload, without a fit permit.

Engineering branch registration proves local bytes and complete native transfer.
Completed producer provenance and future-holdout admission remain separate gates.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
from pathlib import Path
import tempfile
import weakref

from .classic_emg import _activate,_file_hash,_hash
from .data import file_sha256
from .fusion_literal_runtime import (_admit,_check_assets,_read_state,_restore_offline,
    _saved_initialization,_verify_emg,_verify_vision)
from .rn18_fusion_literal import _validated,rn18_fusion_protocol
from .vision_literal_runtime import _schema,_versions,vision_state_sha256

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_rn18_small_fusion_construction_v1'
LOCAL=('rn18_fusion_literal','rn18_fusion_runtime','fusion_literal_runtime','vision_literal_runtime','classic_emg','data')
RUNTIME=('torchvision.models.resnet','torch.nn.modules.module','torch.optim.adam','torch.optim.lr_scheduler')
EMG_HEAD={'head.net.0.weight','head.net.0.bias','head.net.3.weight','head.net.3.bias'}
_BRANCHES=weakref.WeakKeyDictionary()
_PREPARED=weakref.WeakSet()


def _sources(root,protocol):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 fusion runtime source changed')
    result={'egoemg_repro/'+n+'.py':_file_hash(ROOT/'egoemg_repro'/(n+'.py'))for n in LOCAL}
    result.update({'upstream/'+n:h for n,h in protocol['sources_sha256'].items()})
    result.update({'runtime/'+n:_file_hash(importlib.import_module(n).__file__)for n in RUNTIME})
    return result


def _input_branches(vision,emg):
    vs,vnames=_read_state(vision);es,enames=_read_state(emg)
    backbone={k[len('backbone.'):]:vs[n]for k,n in vnames.items()if k.startswith('backbone.')}
    head={k[len('head.'):]:vs[n]for k,n in vnames.items()if k.startswith('head.')}
    shapes={'0.weight':[512,512],'0.bias':[512],'3.weight':[22,512],'3.bias':[22]}
    if not backbone or set(head)!=set(shapes)or any(list(head[n].shape)!=s for n,s in shapes.items()):
        raise ValueError('complete original RN18 backbone and512-dimensional head required')
    if not all(any(n.startswith(p)for n in enames)for p in('featurizer.','decoder.')):
        raise ValueError('complete original8channel EMG branches required')
    return vs,es,enames,backbone,head


@dataclass(eq=False)
class EngineeringRn18FusionInitializers:
    identity:dict
    upstream:Path
    config:object
    _assets:tuple

    def check(self):
        record=_BRANCHES.get(self)
        if record is None:raise ValueError('registered RN18 engineering branches required')
        protocol=rn18_fusion_protocol(self.upstream,self.config)
        if (self.identity.get('sha256')!=record['identity']or
            _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=record['identity']or
            protocol['sha256']!=record['protocol']or _sources(self.upstream,protocol)!=record['sources']or
            self._assets!=record['assets']):
            raise ValueError('registered RN18 initializer identity/source/config changed')
        _check_assets(self._assets)
        for role,(path,_)in zip(('vision','emg'),self._assets):
            if file_sha256(path)!=self.identity['files'][role]['sha256']:
                raise ValueError('registered RN18 initializer bytes changed')
        _check_assets(self._assets)


def engineering_rn18_fusion_initializers(upstream,config,*,vision_checkpoint,vision_sha256,emg_checkpoint,emg_sha256):
    """Register untrained/local diagnostic branches; never grant full eligibility."""
    root,_=_validated(upstream,config);protocol=rn18_fusion_protocol(root,config);sources=_sources(root,protocol)
    vp,vs=_admit(vision_checkpoint,vision_sha256);ep,es=_admit(emg_checkpoint,emg_sha256)
    _input_branches(vp,ep)
    identity=dict(format='rn18_fusion_engineering_branch_registration_v1',scope='engineering',
        trained_branches_admitted=False,scientific_eligible=False,full_paper_complete=False,
        protocol_sha256=protocol['sha256'],sources_sha256=sources,
        files=dict(vision=dict(sha256=vision_sha256,bytes=vs[2]),emg=dict(sha256=emg_sha256,bytes=es[2])))
    identity['sha256']=_hash(identity);assets=((vp,vs),(ep,es))
    result=EngineeringRn18FusionInitializers(identity,root,copy.deepcopy(config),assets)
    _BRANCHES[result]=dict(identity=identity['sha256'],protocol=protocol['sha256'],sources=sources,assets=assets)
    result.check();return result


def _saved(identity):
    try:
        _saved_initialization(identity)
        init=identity['initialization'];transfer=init['vision_transfer'];emg=init['emg_transfer']
        if (identity['format']!=FORMAT or identity['upstream_experiment']!='fusion/fusion_rn18_s_center_8ch'or
            identity['initializers_scope']!='engineering'or identity['trained_branches_admitted']is not False or
            any(identity[k]is not False for k in('scientific_eligible','production_ready','full_paper_complete'))or
            transfer['backbone_tensors']!=120 or transfer['head_tensors']!=4 or
            len(emg['matched'])!=106 or
            {x['target_key']for x in emg['skipped']}!=EMG_HEAD or
            len(emg['skipped'])!=4 or
            any(not x['target_key'].startswith(('featurizer.','decoder.'))for x in emg['matched'])):
            raise ValueError('incomplete RN18 native transfer or incorrect initializer scope')
        schema=identity['model_schema']
        if (schema['model.head_vision.0.weight']['shape']!=[512,512]or
            schema['model.featurizer.layers.0.conv.0.weight']['shape'][1]!=8):
            raise ValueError('RN18 fusion saved native architecture differs')
    except (KeyError,TypeError,AttributeError)as exc:raise ValueError('incomplete RN18 construction receipt')from exc


@dataclass(eq=False)
class PreparedRn18Fusion:
    module:object
    identity:dict
    mode:str
    loaded_state_sha256:str
    _root:Path
    _config:object
    _assets:tuple=()

    def check(self):
        if self not in _PREPARED:raise ValueError('registered prepared RN18 fusion required')
        _saved(self.identity);protocol=rn18_fusion_protocol(self._root,self._config)
        if (self.identity.get('sha256')!=_hash({k:v for k,v in self.identity.items()if k!='sha256'})or
            self.identity['protocol_sha256']!=protocol['sha256']or
            self.identity['sources_sha256']!=_sources(self._root,protocol)or
            self.identity['runtime_versions']!=_versions()or
            self.identity['model_schema']!=_schema(self.module.state_dict())or
            type(self.module).__module__!='egoemg.lightning'or
            type(self.module.model).__name__!='MidFusionPoseFormer'or
            self.module.model.vision_backbone_type!='resnet18'or
            not all(p.requires_grad for p in self.module.parameters())or
            self.module.model.freeze_vision_branch or self.module.model.freeze_vision_head):
            raise ValueError('prepared RN18 fusion source/config/native structure changed')
        _check_assets(self._assets)


def build_rn18_fusion_runtime(upstream,config,*,initializers=None,state=None,expected_initialization=None):
    """Run unchanged native factories, or strictly restore all saved native tensors.

    The current public branch path accepts engineering registration only. A full
    fit must separately admit completed trained producers before using a runtime.
    """
    root,value=_validated(upstream,config);protocol=rn18_fusion_protocol(root,config)
    fresh=state is None and expected_initialization is None
    if fresh:
        if type(initializers)is not EngineeringRn18FusionInitializers:
            raise ValueError('registered explicit RN18 initializers required')
        initializers.check()
        if initializers.upstream!=root or initializers.identity['protocol_sha256']!=protocol['sha256']:
            raise ValueError('RN18 initializer source/family differs')
        assets=initializers._assets;vision=assets[0][0];emg=assets[1][0]
        vs,es,enames,backbone,head=_input_branches(vision,emg)
    elif state is None or expected_initialization is None or initializers is not None:
        raise ValueError('complete RN18 state restore and branch initialization must be separate')
    sources=_sources(root,protocol);versions=_versions()
    if not fresh:
        identity=copy.deepcopy(expected_initialization)
        if (not isinstance(identity,dict)or identity.get('sha256')!=_hash({k:v for k,v in identity.items()if k!='sha256'})or
            identity.get('protocol_sha256')!=protocol['sha256']or identity.get('sources_sha256')!=sources or
            identity.get('runtime_versions')!=versions or identity.get('model_schema')!=_schema(state)):
            raise ValueError('RN18 complete-state identity/source/schema differs')
        _saved(identity);assets=()
    _activate(root)
    import torch
    from omegaconf import OmegaConf
    from egoemg._train_utils import setup_runtime
    import egoemg.train as native
    effective=OmegaConf.create(value)
    with tempfile.TemporaryDirectory(prefix='egoemg-rn18-fusion-restore-')as scratch:
        if not fresh:
            # Original RN fusion ignores the offline flag when no branch path
            # exists. Derive this local packet solely from the complete saved
            # model, so no original initializer/cache/download is required.
            vision=Path(scratch)/'saved-vision.pt'
            branch={}
            for prefix,dest in (('model.vision_backbone.','model.backbone.'),('model.head_vision.','model.head.')):
                branch.update({dest+k[len(prefix):]:v for k,v in state.items()if k.startswith(prefix)})
            torch.save({'state_dict':branch},vision)
        effective.vision_resnet_checkpoint=str(vision)
        effective.module.vision_pretrained_checkpoint=str(vision)
        effective.pretrained_emg_checkpoint=str(emg)if fresh else None
        setup_runtime(effective);_check_assets(assets)
        with _restore_offline(True):module=native.make_lightning_module(effective)
    if type(module).__module__!='egoemg.lightning'or type(module.model).__name__!='MidFusionPoseFormer':
        raise ValueError('original RN18 fusion factory differs')
    if fresh:
        vtransfer=_verify_vision(module.model,backbone,head)
        target={n for n in module.model.state_dict()if n.startswith(('featurizer.','decoder.'))}
        if set(enames)!=target|EMG_HEAD:raise ValueError('all original RN18-compatible EMG tensors required')
        etransfer=_verify_emg(module.model,es,enames)
        initializers.check()
        identity=dict(format=FORMAT,upstream_experiment='fusion/fusion_rn18_s_center_8ch',
            initializers_scope='engineering',trained_branches_admitted=False,
            scientific_eligible=False,production_ready=False,full_paper_complete=False,
            protocol_sha256=protocol['sha256'],sources_sha256=sources,runtime_versions=versions,
            model_schema=_schema(module.state_dict()),parameters=sum(p.numel()for p in module.parameters()),
            initial_state_sha256=vision_state_sha256(module.state_dict()),
            initialization=dict(vision_file=dict(bytes=assets[0][1][2],sha256=initializers.identity['files']['vision']['sha256'],state_sha256=vision_state_sha256(vs)),
                emg_file=dict(bytes=assets[1][1][2],sha256=initializers.identity['files']['emg']['sha256'],state_sha256=vision_state_sha256(es)),
                vision_transfer=vtransfer,emg_source_schema=_schema(es),emg_transfer=etransfer,
                initializer_training_provenance_verified=False,
                residual_rule='original normal(std=1e-5) last weight and zero bias; not overwritten'))
        identity['sha256']=_hash(identity)
    else:
        if _schema(module.state_dict())!=identity['model_schema']:raise ValueError('RN18 restore native factory schema differs')
        module.load_state_dict(state,strict=True)
        if vision_state_sha256(module.state_dict())!=vision_state_sha256(state):raise ValueError('RN18 strict restored tensor bytes differ')
    result=PreparedRn18Fusion(module,identity,'fresh_engineering_branch_initialization'if fresh else'strict_self_contained_restore',
        vision_state_sha256(module.state_dict()),root,copy.deepcopy(config),assets)
    _PREPARED.add(result);result.check();return result
