"""Strict native RN18/EMG/fusion reloads from the completed source audit.

This admits the frozen trained triplet only. The separate dense input, real
batch qualification and complete new-process replay are still required.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import weakref
from .classic_emg import _activate,_approved_upstream,_hash
from .classic_emg_cohort import _safe,_StatGuard,_json
from .occlusion_dense_models import _code
from .occlusion_dense_analysis import source_analysis_contract

ROOT=Path(__file__).resolve().parents[1]
ACCEPTED_PROOF=ROOT/'results/native-dense-occlusion/rn18-source-audit-20261102/completion.json'
PROOF_SHA='4e0d4d84e1d5919640eafb4cdad078a601b5a1dcd1ea5d07be4eb616f41f7d6e'
LOCAL=('rn18_occlusion_models','rn18_fusion_runtime','rn18_fusion_literal',
 'rn18_fusion_transfer','vision_literal_runtime','fusion_literal_runtime',
 'emg_initializer_literal','occlusion_dense_models','occlusion_dense_analysis','occlusion_dense_inference')
CLASSES=dict(fusion=('egoemg.lightning','EmgPredictionModule'),
 vision=('egoemg.models.modules.resnet_vision','ResNetVisionPose'),
 emg=('egoemg.models.modules.emgformer','Emg2PoseFormer'))
_LIVE=weakref.WeakKeyDictionary()


def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


IMPORTED_SHA=_sha(__file__)


def _sealed(value):
    if not isinstance(value,dict)or value.get('sha256')!=_hash({k:v for k,v in value.items()if k!='sha256'}):
        raise ValueError('complete source document seal differs')
    return value


def _methods(models,root):
    items=[(models['fusion'],'forward','egoemg/lightning.py'),
        (models['fusion'].model,'forward','egoemg/models/modules/mid_fusion.py'),
        (models['fusion'].model,'_forward_center_supervised','egoemg/models/modules/mid_fusion.py'),
        (models['fusion'].model.temporal_attn,'forward','egoemg/models/modules/_pooling.py'),
        (models['vision'],'forward','egoemg/models/modules/resnet_vision.py'),
        (models['emg'],'forward','egoemg/models/modules/emgformer.py')]
    for obj,name,source in items:
        function=inspect.getattr_static(type(obj),name);bound=getattr(obj,name)
        if (not inspect.ismethod(bound)or bound.__self__ is not obj or bound.__func__ is not function
            or not inspect.isfunction(function)or function.__code__!=_code(root/source,type(obj).__name__,name)):
            raise ValueError('unmodified native RN18 triplet forward required: '+source+':'+name)
    return items


def _admit_sources(fusion_fit,upstream):
    """Reopen the accepted exact producers; do not repeat the closed data audit."""
    if _sha(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 dense loader source changed')
    try:
        root=_approved_upstream(upstream);fit=_safe(fusion_fit)
        proof_path=_safe(ACCEPTED_PROOF)
        if _sha(proof_path)!=PROOF_SHA:raise ValueError('fixed complete RN18 source audit bytes required')
        proof=_json(proof_path);a=proof['audit']
        if (not proof['all6_original_actors_physically_closed']or not proof['collector_physically_closed_after_exit0']
            or not proof['full_bound_metadata_and5weight_bytes_fresh_postexit_exact']
            or not proof['full_index_inputs_and_points_fresh_postexit_exact']
            or proof['actual_complete_fresh_audits']!=2 or proof['actual_independent_full_boolean_mask_checks']!=18
            or a['hand_requests']!=11114 or not a['checkpoint_bytes_verified']or a['dense_input_or_inference_permit']):
            raise ValueError('actual closed complete RN18 producer audit required')
        local=[ROOT/'egoemg_repro'/(n+'.py')for n in LOCAL]
        g=_StatGuard([proof_path,*local,fit/'identity.json',fit/'fit-result.json'])
        fi=_sealed(_json(fit/'identity.json'));fr=_sealed(_json(fit/'fit-result.json'))
        if (fi['sha256']!=a['actual_completed_fit_identity_sha256']or fr['sha256']!=a['actual_completed_fit_result_sha256']
            or fr['identity_sha256']!=fi['sha256']or fi['policy']['scope']!='full'
            or not fr['full_training_terminated']or not fr['stage_complete']
            or fr['completed_epochs']!=150 or fr['optimizer_executions']!=46950
            or fi['data_locations']['upstream']!=str(root)):
            raise ValueError('actual completed RN18 full fit/source required')
        d=fi['initializer_descriptor'];lineage=_sealed(d['identity']);vision=_sealed(lineage['vision']);emg=_sealed(lineage['emg']);parents=d['producers']
        if (lineage['scope']!='full'or not lineage['producer_training_verified']or not lineage['trained_branches_admitted']
            or lineage['fusion_partition_content_sha256']!=a['source_partition_content_sha256']['fusion']
            or vision['checkpoint_selection']!='LAST'or not vision['producer_training_verified']
            or vision['producer']['source_partition_content_sha256']!=a['source_partition_content_sha256']['vision_initializer']
            or emg['checkpoint_selection']!='native validation BEST'or not emg['full_initializer']
            or emg['completed_epochs']!=150 or emg['selected_epoch']!=74
            or emg['input_binding']['partition_content_sha256']!=a['source_partition_content_sha256']['emg_initializer']):
            raise ValueError('exact actual trained LAST/BEST parent selections required')
        emgroot=_safe(parents['emg_fit']);visionroot=_safe(parents['vision_fit'])
        metadata=[fit/'identity.json',fit/'fit-result.json',emgroot/'identity.json',emgroot/'fit-result.json',visionroot/'partitions.json']
        for p in metadata:g.add(_safe(p))
        if sorted(_sha(p)for p in metadata)!=a['actual_bound_metadata_file_sha256']:
            raise ValueError('actual complete linked producer metadata bytes differ')
        checkpoints=dict(fusion=_safe(fit/'checkpoints'/fr['selected_best']['file']),
            vision=_safe(d['files']['vision_checkpoint']),emg=_safe(d['files']['emg_checkpoint']))
        er=_sealed(_json(emgroot/'fit-result.json'))
        weights=dict(fusion=checkpoints['fusion'],vision_export=checkpoints['vision'],emg_export=checkpoints['emg'],
            vision_original_LAST=_safe(visionroot/'last.pt'),emg_original_BEST=_safe(emgroot/'checkpoints'/er['selected_best']['file']))
        for role,p in weights.items():
            g.add(p);expected=a['actual_selected_weights_sha256'][role]
            if p.stat().st_size!=expected['bytes']or _sha(p)!=expected['sha256']:
                raise ValueError('exact complete trained RN18 selected weight bytes differ')
        g.check()
        return root,fit,fi,fr,proof,g,checkpoints,local
    except (OSError,KeyError,TypeError,AttributeError)as exc:
        raise ValueError('accepted complete RN18 fit and producer sources unavailable')from exc


@dataclass(eq=False)
class RN18DenseTrainedModels:
    models:dict
    identity:dict
    checkpoint_paths:dict
    data_locations:dict
    _guard:object
    _fusion:object
    _root:Path

    def check(self):
        import torch
        from .vision_literal_runtime import vision_state_sha256
        saved=_LIVE.get(self)
        if saved is None:raise ValueError('registered actual RN18 trained triplet required')
        if self._guard is not saved['guard']or self._fusion is not saved['prepared']:
            raise ValueError('registered RN18 source guard or prepared runtime replaced')
        self._guard.check()
        if (self.identity.get('sha256')!=saved['identity']or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=saved['identity']
            or {k:id(v)for k,v in self.models.items()}!=saved['models']
            or {k:str(v)for k,v in self.checkpoint_paths.items()}!=saved['paths']
            or _hash(self.data_locations)!=saved['locations']or _sha(__file__)!=IMPORTED_SHA):
            raise ValueError('RN18 triplet source/role/identity changed')
        self._fusion.check()
        for role,model in self.models.items():
            if ((type(model).__module__,type(model).__name__)!=CLASSES[role]
                or any(m.training for m in model.modules())or any(p.dtype!=torch.float32 for p in model.parameters())
                or vision_state_sha256(model.state_dict())!=self.identity['model_state_sha256'][role]):
                raise ValueError('native RN18 model class/eval/FP32/complete state changed')
        _methods(self.models,self._root);self._guard.check()


def load_rn18_dense_trained_models(fusion_fit,*,upstream,recipe='rn18_s'):
    if recipe!='rn18_s':raise ValueError('RN18 loader requires the native rn18_s recipe')
    root,fit,meta,result,proof,guard,paths,local=_admit_sources(fusion_fit,upstream)
    import torch
    if torch.get_default_dtype()!=torch.float32:raise ValueError('native FP32 inference construction required')
    from .fusion_literal_runtime import _read_state
    from .rn18_fusion_runtime import build_rn18_fusion_runtime
    from .vision_literal_runtime import vision_state_sha256
    from .emg_initializer_literal import compose_emg_initializer
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    _activate(root)
    from egoemg.models.modules.emgformer import Emg2PoseFormer
    from egoemg.models.modules.resnet_vision import ResNetVisionPose
    states={};normalized={};source_hashes={}
    for role,p in paths.items():
        state,names=_read_state(p);states[role]=state
        normalized[role]={n:state[k]for n,k in names.items()}
        source_hashes[role]=vision_state_sha256(state)
    for role in ('vision','emg'):
        if source_hashes[role]!=meta['initializer_descriptor']['identity'][role]['model_state_sha256']:
            raise ValueError('complete standalone native state differs from selected parent')
    cfg=OmegaConf.create(meta['saved_config'])
    fusion=build_rn18_fusion_runtime(root,cfg,state=states['fusion'],expected_initialization=meta['construction'])
    # These are the exact author evaluation constructors, with the admitted
    # actually trained cotrain recipe replacing the unavailable archived alias.
    ecfg=compose_emg_initializer(root)
    emg=Emg2PoseFormer(featurizer=instantiate(ecfg.module.featurizer),decoder=instantiate(ecfg.module.decoder),
        head=instantiate(ecfg.module.head),out_channels=ecfg.module.get('out_channels',22),
        provide_initial_pos=ecfg.module.get('provide_initial_pos',False))
    vision=ResNetVisionPose(out_channels=22,backbone_type='resnet18',pretrained=False,head_hidden=512,head_dropout=.1)
    emg.load_state_dict(normalized['emg'],strict=True);vision.load_state_dict(normalized['vision'],strict=True)
    models=dict(fusion=fusion.module,vision=vision,emg=emg)
    for model in models.values():model.eval()
    actual={n:vision_state_sha256(m.state_dict())for n,m in models.items()}
    expected=dict(fusion=source_hashes['fusion'],vision=vision_state_sha256(normalized['vision']),emg=vision_state_sha256(normalized['emg']))
    if actual!=expected:raise ValueError('strict complete trained native RN18 tensor reload differs')
    for _,_,name in _methods(models,root):guard.add(root/name)
    contract=source_analysis_contract(root)
    identity=dict(format='actual_native_rn18_dense_trained_triplet_v1',recipe=recipe,
        scope='complete trained triplet only; separate dense input/qualification/inference permit required',
        trained_full_producers_verified=True,dense_input_or_inference_permit=False,
        source_audit_file_sha256=PROOF_SHA,source_contract_sha256=contract['sha256'],
        fusion_fit_identity_sha256=meta['sha256'],fusion_fit_result_sha256=result['sha256'],
        initializer_handoff_identity_sha256=meta['initializer_descriptor']['identity']['sha256'],
        selected_completed_epochs=dict(fusion=145,vision=150,emg=74),
        selection_policy=dict(fusion='completed native full fusion validation BEST',vision='completed historical vision LAST',emg='completed RN18 cotrain validation BEST export'),
        model_state_sha256=actual,complete_source_state_sha256=source_hashes,
        complete_tensor_counts={n:len(s)for n,s in states.items()},
        checkpoint_file_sha256={n:_sha(p)for n,p in paths.items()},
        input_normalization=copy.deepcopy(meta['cohort']['normalization']),
        producer_partition_content_sha256=copy.deepcopy(proof['audit']['source_partition_content_sha256']),
        source_files_sha256={str(p.relative_to(ROOT)):_sha(p)for p in local},
        scientific_eligible=False,figure13_reproduced=False,full_paper_complete=False)
    identity['sha256']=_hash(identity);guard.check()
    value=RN18DenseTrainedModels(models,identity,paths,copy.deepcopy(meta['data_locations']),guard,fusion,root)
    _LIVE[value]=dict(identity=identity['sha256'],models={k:id(v)for k,v in models.items()},paths={k:str(v)for k,v in paths.items()},locations=_hash(value.data_locations),guard=guard,prepared=fusion)
    value.check();return value


def predict_rn18_dense_batch(trained,fusion_batch,standalone_batch):
    """Forward a registered actual RN18 triplet; no complete input permit."""
    if type(trained)is not RN18DenseTrainedModels:
        raise ValueError('registered actual RN18 trained triplet required')
    trained.check()
    from .occlusion_dense_inference import native_forward_arrays
    result=native_forward_arrays(trained.models,fusion_batch,standalone_batch)
    trained.check()
    return result
