"""Completed ViT-S/EMG/fusion source admission and native evaluation reloads.

The public full fusion reader freshly verifies its complete input cohort and
both trained producers. Standalone constructors follow the author's evaluation
script; every source tensor is retained under a bijective prefix removal.
This admits trained model sources only, never dense inputs or full predictions.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
from pathlib import Path
import types
import weakref

from .classic_emg import _activate, _approved_upstream, _file_hash, _hash
from .classic_emg_cohort import _safe, _StatGuard, _json
from .occlusion_dense_analysis import source_analysis_contract

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = _file_hash(__file__)
_LIVE = weakref.WeakKeyDictionary()
_COMPILED = {}
LOCAL = ('occlusion_dense_models', 'fusion_literal_train', 'fusion_initializer_handoff',
    'fusion_literal_runtime', 'vision_literal_runtime', 'emg_initializer_literal')
CLASSES = dict(fusion=('egoemg.lightning', 'EmgPredictionModule'),
    vision=('egoemg.models.modules.vit_vision', 'VisionViTPose'),
    emg=('egoemg.models.modules.emgformer', 'Emg2PoseFormer'))
POLICY = dict(fusion='completed full fusion validation best',
    vision='completed full vision LAST', emg='completed full EMG validation-best export')


def _code(path, class_name, method):
    path = Path(path)
    key = (str(path), _file_hash(path))
    if key not in _COMPILED:
        _COMPILED[key] = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    cls = next(c for c in _COMPILED[key].co_consts if isinstance(c, types.CodeType) and c.co_name == class_name)
    return next(c for c in cls.co_consts if isinstance(c, types.CodeType) and c.co_name == method)


def _methods(models, root):
    entries = [(models['fusion'], 'forward', 'egoemg/lightning.py'),
        (models['fusion'].model, 'forward', 'egoemg/models/modules/mid_fusion.py'),
        (models['fusion'].model, '_forward_center_supervised', 'egoemg/models/modules/mid_fusion.py'),
        (models['fusion'].model.temporal_attn, 'forward', 'egoemg/models/modules/_pooling.py'),
        (models['vision'], 'forward', 'egoemg/models/modules/vit_vision.py'),
        (models['emg'], 'forward', 'egoemg/models/modules/emgformer.py')]
    for obj, method, name in entries:
        function = inspect.getattr_static(type(obj), method)
        bound = getattr(obj, method)
        if (not inspect.ismethod(bound) or bound.__self__ is not obj
            or bound.__func__ is not function or not inspect.isfunction(function)
            or function.__code__ != _code(root/name, type(obj).__name__, method)):
            raise ValueError('unmodified native model forward required: ' + name + ':' + method)
    return entries


@dataclass(eq=False)
class DenseTrainedModels:
    models: dict
    identity: dict
    checkpoint_paths: dict
    data_locations: dict
    _guard: object
    _fusion: object
    _root: Path

    def check(self):
        import torch
        from .vision_literal_runtime import vision_state_sha256
        saved = _LIVE.get(self)
        if saved is None:
            raise ValueError('registered actual trained dense triplet required')
        self._guard.check()
        if (self.identity.get('sha256') != saved['identity']
            or _hash({k: v for k, v in self.identity.items() if k != 'sha256'}) != saved['identity']
            or {k: id(v) for k, v in self.models.items()} != saved['models']
            or {k: str(v) for k, v in self.checkpoint_paths.items()} != saved['paths']
            or _hash(self.data_locations) != saved['locations']
            or _file_hash(__file__) != IMPORTED_SHA):
            raise ValueError('trained dense model identity, roles or source changed')
        self._fusion.check()
        for role, module in self.models.items():
            if ((type(module).__module__, type(module).__name__) != CLASSES[role]
                or any(x.training for x in module.modules())
                or any(p.dtype != torch.float32 for p in module.parameters())
                or vision_state_sha256(module.state_dict()) != self.identity['model_state_sha256'][role]):
                raise ValueError('trained native model class/eval/dtype/tensor state changed')
        _methods(self.models, self._root)
        self._guard.check()


def load_dense_trained_models(fusion_fit, *, upstream, recipe):
    """Reopen actual full sources, selecting native LAST/BEST without fallback."""
    if recipe != 'vits_s':
        raise ValueError('this completed-producer loader requires vits_s; RN18 native producer is separate')
    if _file_hash(__file__) != IMPORTED_SHA:
        raise ValueError('imported trained dense model source changed')
    root = _approved_upstream(upstream)
    try:
        fit = _safe(fusion_fit)
        source_paths = [ROOT/'egoemg_repro'/(n+'.py') for n in LOCAL]
        guard = _StatGuard([fit/'identity.json', fit/'fit-result.json', fit/'epochs.jsonl', *source_paths])
        meta = _json(fit/'identity.json')
        descriptor = meta['initializer_descriptor']
        parents = descriptor['producers']
        if (meta['policy']['scope'] != 'full' or descriptor['identity']['scope'] != 'full'
            or not descriptor['identity']['producer_training_verified']
            or not isinstance(parents, dict)):
            raise ValueError('actual complete full fusion and trained parent producers required')
        for name in ('vision_fit', 'emg_fit'):
            p = _safe(parents[name])
            for filename in ('identity.json', 'fit-result.json', 'epochs.jsonl', 'checkpoints/last.ckpt'):
                guard.add(p/filename)
        for n in ('vision_checkpoint', 'emg_checkpoint'):
            guard.add(_safe(descriptor['files'][n]))
        from .fusion_initializer_handoff import HISTORICAL_VISION_SOURCES
        for name in HISTORICAL_VISION_SOURCES:
            guard.add(_safe(Path(parents['vision_source_root'])/name))
        guard.add(_safe(meta['cohort_receipt']['path']))
        guard.check()
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('complete trusted fusion/producer fit documents unavailable') from exc
    # This public reader performs its unchanged, fresh full-content cohort and
    # completed historical-vision/full-EMG producer validation. Do not substitute
    # a copied result JSON or bypass it with caller-supplied live objects.
    from .fusion_literal_train import read_fusion_fit
    result = read_fusion_fit(fit, require_full=True)
    guard.check()
    if not result['full_training_terminated'] or not result['stage_complete']:
        raise ValueError('full fusion training must actually have terminated')
    contract = source_analysis_contract(root)
    locations = copy.deepcopy(meta['data_locations'])
    if Path(locations['upstream']).resolve() != root:
        raise ValueError('native trained source upstream location differs')
    checkpoints = dict(fusion=_safe(fit/'checkpoints'/result['selected_best']['file']),
        vision=_safe(descriptor['files']['vision_checkpoint']), emg=_safe(descriptor['files']['emg_checkpoint']))
    for p in (fit/'checkpoints/last.ckpt', *checkpoints.values()):
        guard.add(p)
    expected_sha = dict(fusion=result['selected_best']['checkpoint_sha256'],
        vision=descriptor['identity']['vision']['checkpoint_sha256'],
        emg=descriptor['identity']['emg']['checkpoint_sha256'])
    if any(_file_hash(checkpoints[n]) != h for n, h in expected_sha.items()):
        raise ValueError('actual native selected checkpoint bytes differ')
    vision_meta = _json(Path(parents['vision_fit'])/'identity.json')
    emg_meta = _json(Path(parents['emg_fit'])/'identity.json')
    if (vision_meta['sha256'] != descriptor['identity']['vision']['source_fit_identity_sha256']
        or emg_meta['sha256'] != descriptor['identity']['emg']['source_fit_identity_sha256']):
        raise ValueError('actual linked standalone producer identities differ')
    import torch
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('original FP32 inference construction required')
    from .fusion_literal_runtime import build_fusion_runtime, _read_state
    from .vision_literal_runtime import vision_state_sha256
    from .emg_initializer_literal import compose_emg_initializer
    from hydra.utils import instantiate
    _activate(root)
    from egoemg.models.modules.emgformer import Emg2PoseFormer
    from egoemg.models.modules.vit_vision import VisionViTPose
    from omegaconf import OmegaConf
    sources, states, normalized = {}, {}, {}
    for role, path in checkpoints.items():
        state, names = _read_state(path)
        sources[role] = vision_state_sha256(state)
        states[role] = state
        normalized[role] = {n: state[k] for n, k in names.items()}
    for role in ('vision', 'emg'):
        if sources[role] != descriptor['identity'][role]['model_state_sha256']:
            raise ValueError('complete standalone state differs from completed producer selection')
    cfg = OmegaConf.create(meta['saved_config'])
    prepared = build_fusion_runtime(root, cfg, state=states['fusion'], expected_initialization=meta['construction'])
    emg_cfg = compose_emg_initializer(root)
    # Exact standalone class/arguments used by the author evaluation script;
    # the unavailable archived recipe is replaced by the admitted producer's
    # actual cotrain architecture. No model math or tensor expansion is added.
    emg = Emg2PoseFormer(featurizer=instantiate(emg_cfg.module.featurizer),
        decoder=instantiate(emg_cfg.module.decoder), head=instantiate(emg_cfg.module.head),
        out_channels=emg_cfg.module.get('out_channels', 22),
        provide_initial_pos=emg_cfg.module.get('provide_initial_pos', False))
    vision = VisionViTPose(out_channels=22, backbone_type=cfg.module.vision_backbone_type,
        pretrained=False, head_hidden=512, head_dropout=.1)
    emg.load_state_dict(normalized['emg'], strict=True)
    vision.load_state_dict(normalized['vision'], strict=True)
    models = dict(fusion=prepared.module, emg=emg, vision=vision)
    for model in models.values():
        model.eval()
    actual = {n: vision_state_sha256(model.state_dict()) for n, model in models.items()}
    expected_states = dict(fusion=sources['fusion'],
        vision=vision_state_sha256(normalized['vision']), emg=vision_state_sha256(normalized['emg']))
    if actual != expected_states:
        raise ValueError('strict native complete trained tensor reload differs')
    for obj, method, name in _methods(models, root):
        guard.add(root/name)
    for name in ('scripts/eval/evaluate_occlusion_fusion.py', 'scripts/paper/plot_occlusion_fusion.py'):
        guard.add(root/name)
    identity = dict(format='actual_native_vits_dense_trained_triplet_v1', recipe=recipe,
        scope='completed trained model sources and strict native reload only; no dense input/full prediction permit',
        trained_full_producers_verified=True, dense_input_or_inference_permit=False,
        source_contract_sha256=contract['sha256'], fusion_fit_identity_sha256=meta['sha256'],
        fusion_fit_result_sha256=result['sha256'], fusion_cohort_identity_sha256=meta['cohort']['sha256'],
        initializer_handoff_identity_sha256=descriptor['identity']['sha256'],
        selection_policy=copy.deepcopy(POLICY), fusion_selected_best=copy.deepcopy(result['selected_best']),
        standalone_producers={n: copy.deepcopy(descriptor['identity'][n]) for n in ('vision', 'emg')},
        checkpoint_file_sha256=expected_sha, complete_source_state_sha256=sources,
        model_state_sha256=actual, complete_tensor_counts={n: len(v) for n, v in states.items()},
        source_files_sha256={str(p.relative_to(ROOT)): _file_hash(p) for p in source_paths},
        standalone_architecture='pinned author evaluation native classes; complete bijective model-prefix removal',
        scientific_eligible=False, figure13_reproduced=False, full_paper_complete=False)
    identity['sha256'] = _hash(identity)
    guard.check()
    value = DenseTrainedModels(models, identity, checkpoints, locations, guard, prepared, root)
    _LIVE[value] = dict(identity=identity['sha256'], models={k: id(v) for k, v in models.items()},
        paths={k: str(v) for k, v in checkpoints.items()}, locations=_hash(locations))
    value.check()
    return value
