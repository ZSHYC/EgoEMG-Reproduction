"""Fixed-center visual data and unchanged official vision/fusion architectures."""
from __future__ import annotations

import io
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from contextlib import contextmanager

import numpy as np

from .crops import CropStore, video_frame_at_center
from .data import LegacyDataset, file_sha256

SPARSE16_ANCHORS = (10, 12, 0, 1, 2, 4, 5, 6)
_TIMM_TRANSPORT_LOCK = threading.RLock()


def sparse16(emg):
    """Place a normalized target wrist in official slots; never interpolate."""
    if emg.shape[0] != 8:
        raise ValueError('sparse16 requires 8 target-hand channels')
    output = np.zeros((16,) + emg.shape[1:], dtype=emg.dtype)
    output[list(SPARSE16_ANCHORS)] = emg
    return output


class VisualDataset:
    """Both hands reuse fixed W=12000 footprints and center-only labels.

    Official pre-crops already flip the left hand. A second flip is incorrect.
    Missing keys use normalized black images and retain an explicit vision mask;
    an unavailable whole episode LMDB fails instead of dropping its population.
    """
    def __init__(self, root, windows, crops_root, norm=None, kind='vision',
                 emg_window_length=7790, emg_layout='target_hand'):
        if kind not in ('vision', 'fusion'):
            raise ValueError('kind must be vision or fusion')
        if emg_layout not in ('target_hand', 'emg2pose_sparse16'):
            raise ValueError('Only target_hand and emg2pose_sparse16 layouts are supported')
        if int(emg_window_length) <= 0:
            raise ValueError('emg_window_length must be positive')
        self.base = LegacyDataset(root, windows, norm=norm)
        self.windows, self.store = self.base.windows, self.base.store
        self.reader = CropStore(crops_root)
        self.kind, self.emg_layout = kind, emg_layout
        label_windows, emg_windows = [], []
        for window in self.windows:
            center = int(window['center'])
            label_windows.append(dict(window, start=center, end=center+1))
            start = center-int(emg_window_length)//2
            end = start+int(emg_window_length)
            if kind == 'fusion' and (start < window['start'] or end > window['end']):
                raise ValueError('Centered fusion EMG must fit inside the guarded manifest footprint')
            emg_windows.append(dict(window, start=start, end=end))
        self.labels = LegacyDataset(root, label_windows, norm=norm)
        self.emg_data = LegacyDataset(root, emg_windows, norm=norm) if kind == 'fusion' else None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        labels = self.labels[index]
        if index < 0:
            index += len(self)
        window = self.windows[index//2]
        episode_id = self.store.episodes[int(window['episode'])]['id']
        frame = video_frame_at_center(self.store, window)
        payload = self.reader.read(episode_id, frame, labels['hand'])
        if payload is None:
            rgb = np.zeros((256,256,3), dtype=np.uint8)
        else:
            with Image.open(io.BytesIO(payload)) as image:
                if image.size != (256,256) or image.mode != 'RGB':
                    raise ValueError('Official crop must be a 256x256 RGB JPEG')
                if image.format != 'JPEG':
                    raise ValueError('Official crop payload must be JPEG')
                rgb = np.asarray(image).copy()
        patch = rgb.transpose(2,0,1).astype(np.float32)
        mean = 255*np.array([.485,.456,.406],np.float32)[:,None,None]
        std = 255*np.array([.229,.224,.225],np.float32)[:,None,None]
        result = dict(labels)
        result.update(window)
        if self.emg_data is not None:
            emg = self.emg_data[index]['emg']
            if self.emg_layout == 'emg2pose_sparse16':
                emg = torch.from_numpy(sparse16(emg.numpy()))
        else:
            emg = torch.empty((8 if self.emg_layout == 'target_hand' else 16, 0), dtype=torch.float32)
        result.update(emg=emg, vision_img=torch.from_numpy(np.ascontiguousarray((patch-mean)/std)),
                      vision_valid_mask=torch.tensor([payload is not None]),
                      vision_frame_indices=torch.tensor([frame], dtype=torch.int64),
                      target_hand=labels['hand'], target_hand_index=index%2)
        return result


def _clean_state(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
        raise ValueError('Clean initializer must use the reproduction checkpoint schema {model: state_dict}')
    return checkpoint['model']


@contextmanager
def _capture_timm_pretrained_source():
    """Observe successful timm transport without selecting or retrying its route.

    timm 1.0.15 does not expose the selected source on the returned model.
    Its HF/URL transport entry points are observed only during construction;
    all calls and exceptions retain timm's behavior and are restored on exit.
    """
    import timm.models._builder as builder
    with _TIMM_TRANSPORT_LOCK:
        owner = threading.get_ident()
        observed = []
        originals = {name: getattr(builder, name) for name in
                     ('load_state_dict_from_hf', 'load_state_dict_from_url')}
        def observer(original, source_type):
            def load(location, *args, **kwargs):
                result = original(location, *args, **kwargs)
                if threading.get_ident() == owner:
                    observed.append({'source_type': source_type, 'source_location': location})
                return result
            return load
        try:
            for name, original in originals.items():
                source_type = 'hf-hub' if name.endswith('_hf') else 'url'
                setattr(builder, name, observer(original, source_type))
            yield observed
        finally:
            for name, original in originals.items():
                setattr(builder, name, original)


def pretrained_weight_identity(model):
    """Describe observed pretrained source and hash exact loaded backbone tensors.

    Digest order is sorted state_dict keys. Each UTF-8 JSON name/dtype/shape
    header is prefixed with its 8-byte big-endian length, then followed by the
    contiguous CPU tensor's raw bytes. Parameters and persistent buffers are
    included; the newly initialized pose head and unused ImageNet FC are not.
    """
    import torch
    backbone = model.backbone_type
    if backbone.startswith('vit_'):
        import timm
        from egoemg.models.modules._vision_constants import DINOV2_VARIANTS
        source = getattr(model, 'reproduction_pretrained_source', None)
        if source is None:
            raise ValueError('DINOv2 pretrained source must be observed during weight loading')
        identity = {'family': 'DINOv2', 'timm_model': DINOV2_VARIANTS[backbone][0],
                    'timm_version': timm.__version__, **source,
                    'resolved_pretrained_cfg': dict(model.backbone.pretrained_cfg)}
    else:
        import torchvision
        enum_name = 'ResNet' + backbone[len('resnet'):].upper() + '_Weights'
        weights = getattr(torchvision.models, enum_name).IMAGENET1K_V1
        identity = {'enum': 'torchvision.models.' + enum_name + '.IMAGENET1K_V1',
                    'source_url': weights.url, 'torchvision_version': torchvision.__version__}
    digest = hashlib.sha256()
    for name, tensor in sorted(model.backbone.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps({'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape)},
                            sort_keys=True, separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {**identity, 'backbone_tensor_sha256': digest.hexdigest(),
            'digest_scope': 'sorted backbone state_dict names/dtypes/shapes and raw contiguous CPU bytes; excludes pose head and unused ImageNet FC',
            'digest_format': 'length-prefixed canonical JSON header + raw tensor bytes, v1'}


def build_visual_model(upstream, config, vision_checkpoint=None, emg_checkpoint=None, resume_state=None):
    """Instantiate the fixed official class; clean branch loads are strict.

    Fusion uses the upstream 8-channel S architecture, so clean EMG weights need
    no speculative input-convolution expansion. This is explicitly distinct
    from published 16-channel fusion-checkpoint verification.
    """
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from .train import seed_everything
    upstream = Path(upstream).resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    seed_everything(config['seed'])
    kind = config['model_kind']
    if kind not in ('vision', 'fusion'):
        raise ValueError('model_kind must be vision or fusion')
    if 'vision_backbone_type' in config:
        if kind != 'vision':
            raise ValueError('vision_backbone_type is supported only for vision models')
        backbone = config['vision_backbone_type']
        family = {'fusion/vision_resnet18': ('resnet18', 'resnet50', 'resnet152'),
                  'fusion/vision_vit_small': ('vit_small', 'vit_base', 'vit_large')}
        choices = family.get(config['upstream_experiment'], ())
        if not isinstance(backbone, str) or backbone not in choices:
            raise ValueError('vision_backbone_type must match the supported official ResNet or ViT vision experiment')
    if kind == 'fusion' and resume_state is None and (vision_checkpoint is None or emg_checkpoint is None):
        raise ValueError('Clean fusion requires independent-validation vision and EMG initializers')
    overrides = ['experiment=' + config['upstream_experiment']]
    if kind == 'vision':
        if 'vision_backbone_type' in config:
            overrides += ['module.backbone_type=' + config['vision_backbone_type']]
        needs_pretrained = config.get('vision_pretrained', True) and resume_state is None and vision_checkpoint is None
        pretrained_family = 'DINOv2' if config['upstream_experiment'] == 'fusion/vision_vit_small' else 'ImageNet1K_V1'
        if needs_pretrained and os.environ.get('EGOEMG_NO_PRETRAINED_DOWNLOAD', '') not in ('', '0'):
            raise ValueError(f'Requested {pretrained_family} initialization conflicts with EGOEMG_NO_PRETRAINED_DOWNLOAD; set vision_pretrained=false explicitly for random initialization')
        overrides += ['module.pretrained=' + str(bool(needs_pretrained)).lower()]
    else:
        overrides += ['vision_resnet_checkpoint=null', 'pretrained_emg_checkpoint=null',
                      'module.vision_pretrained_checkpoint=null', 'module.mano_model_path=null',
                      'module.freeze_vision_branch=' + str(config.get('freeze_vision_branch', False)).lower()]
    with initialize_config_dir(config_dir=str(upstream/'config'), version_base=None):
        composed = compose(config_name='base', overrides=overrides)
    initialization = dict(kind=kind, upstream_experiment=config['upstream_experiment'])
    if kind == 'vision':
        if needs_pretrained and pretrained_family == 'DINOv2':
            with _capture_timm_pretrained_source() as sources:
                model = instantiate(composed.module)
            if len(sources) != 1:
                raise RuntimeError('DINOv2 pretrained source was not uniquely observed during loading')
            model.reproduction_pretrained_source = sources[0]
        else:
            model = instantiate(composed.module)
        initialization['vision_backbone'] = (pretrained_family if needs_pretrained else 'random')
        if needs_pretrained:
            initialization['pretrained_weights'] = pretrained_weight_identity(model)
        if resume_state is not None:
            model.load_state_dict(resume_state, strict=True)
            initialization.update(vision_backbone='self_contained_checkpoint', restored_self_contained=True)
        elif vision_checkpoint is not None:
            model.load_state_dict(_clean_state(vision_checkpoint), strict=True)
            initialization.update(vision_backbone='clean_checkpoint', vision_checkpoint_sha256=file_sha256(vision_checkpoint))
    else:
        if resume_state is None:
            vision_state, emg_state = _clean_state(vision_checkpoint), _clean_state(emg_checkpoint)
        else:
            vision_state = {}
            for key, value in resume_state.items():
                if key.startswith('vision_backbone.'):
                    vision_state['backbone.' + key[len('vision_backbone.'):]] = value
                elif key.startswith('head_vision.'):
                    vision_state['head.' + key[len('head_vision.'):]] = value
        # Validate the complete source vision schema before upstream's permissive
        # constructor loader sees it. Its local checkpoint also prevents a hidden
        # ImageNet network download when the clean branch is already available.
        from egoemg.models.modules.resnet_vision import ResNetVisionPose
        validation_model = ResNetVisionPose(pretrained=False)
        validation_model.load_state_dict(vision_state, strict=True)
        del validation_model
        with tempfile.TemporaryDirectory(prefix='egoemg-clean-vision-') as temporary:
            translated = Path(temporary)/'vision.pt'
            torch.save({'state_dict': {'model.'+key: value for key,value in vision_state.items()}}, translated)
            composed.module.vision_pretrained_checkpoint = str(translated)
            model = instantiate(composed.module)
            composed.module.vision_pretrained_checkpoint = None
        model.vision_backbone.load_state_dict({key[len('backbone.'):]: value for key,value in vision_state.items()
                                              if key.startswith('backbone.')}, strict=True)
        model.head_vision.load_state_dict({key[len('head.'):]: value for key,value in vision_state.items()
                                          if key.startswith('head.')}, strict=True)
        if resume_state is not None:
            model.load_state_dict(resume_state, strict=True)
            initialization.update(vision_backbone='self_contained_checkpoint', restored_self_contained=True)
        else:
            allowed = ('featurizer.', 'decoder.', 'head.')
            if any(not key.startswith(allowed) for key in emg_state):
                raise ValueError('Unexpected clean EMG checkpoint keys')
            for name in ('featurizer', 'decoder', 'head'):
                state = {key[len(name)+1:]: value for key,value in emg_state.items() if key.startswith(name+'.')}
                getattr(model,name).load_state_dict(state, strict=True)
            final = model.head.net[-1]
            if not isinstance(final, torch.nn.Linear):
                raise RuntimeError('Expected official MLP residual final Linear')
            torch.nn.init.zeros_(final.weight)
            if final.bias is not None:
                torch.nn.init.zeros_(final.bias)
            initialization.update(vision_checkpoint_sha256=file_sha256(vision_checkpoint),
                                  emg_checkpoint_sha256=file_sha256(emg_checkpoint),
                                  emg_layout='target_hand_8ch_no_expansion', emg_window_length=config.get('emg_window_length',12000),
                                  emg_head='loaded_then_final_layer_reset',
                                  residual_final='exact_zero_after_official_initialization',
                                  difference_from_published_fusion=f'8ch official S branch; shared 12000 footprint; {config.get("emg_window_length",12000)} centered input')
    if model.out_channels != 22:
        raise RuntimeError('Visual model must produce 22 angles')
    model.reproduction_hydra_config = OmegaConf.to_container(composed.module, resolve=True)
    model.reproduction_initialization = initialization
    return model


def align_visual_batch(model, batch):
    """Return only the true center target and authoritative per-angle validity."""
    prediction = model(batch)
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    target, valid = batch['joint_angles'], batch['angle_valid_mask'].bool()
    if target.ndim != 3 or target.shape[1:] != (22,1):
        raise ValueError('Visual targets must have shape (B,22,1) at fixed centers')
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Prediction, target and per-angle mask must agree at fixed centers')
    return prediction, target, valid
