"""Exact native EMG2Pose recipes with a verified sealed-TAR storage adapter.

Concrete descriptors and file bindings are checked before canonicalization.
Only content identities cross cache rebuilds; no validator for the original
completed memmap format is broadened or bypassed.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.metadata
from pathlib import Path
import platform

from . import UPSTREAM_COMMIT
from . import emg2pose_data as storage, emg2pose_runtime as native
from .emg2pose_memory_index import VerifiedSource


CONTEXT_KEY = '_emg2pose_memory_runtime'
TARGET = 'egoemg_repro.emg2pose_memory_dataset.SealedTarDataset'
POLICY = 'storage-only official Emg2PoseDataset subclass; exact native factories and scientific recipe'
_CODE = ('__init__.py','download.py','emg2pose_archive.py','emg2pose_data.py',
         'emg2pose_runtime.py','emg2pose_memory_archive.py','emg2pose_memory_data.py',
         'emg2pose_memory_index.py','emg2pose_memory_dataset.py','emg2pose_memory_runtime.py')


def _official_files(upstream):
    return sorted({*upstream.glob('egoemg/**/*.py'),*upstream.glob('config/**/*.yaml'),
                   *upstream.glob('egoemg/UmeTrack/**/*.json'),upstream/'assets/per_dataset_norm_stats.json'})


def _binding(upstream,index,source,metadata_csv):
    if type(source) is not VerifiedSource:
        raise ValueError('an actual live VerifiedSource is required')
    source.assert_current()
    index = storage._safe_path(index)
    if index != source.index:
        raise ValueError('index differs from the admitted source')
    proof = source.handle_identity
    if any(str(index/name) not in proof['binding_stamps'] for name in
           ('completion.json','manifest.json','metadata.npz','source-inventory.json')):
        raise ValueError('index is not bound by the verified source')
    csv = storage._safe_path(metadata_csv,file=True)
    identity = source.data_identity
    if storage._sha256(csv) != identity['metadata_csv_sha256']:
        raise ValueError('source metadata CSV differs')
    if storage._sha256(upstream/'assets/per_dataset_norm_stats.json') != native.NORM_SHA256:
        raise ValueError('official normalization asset differs')
    # These are private concrete instance guards, not content identity. The
    # original datamodule may instantiate its datasets well after construction.
    files = [csv,*_official_files(upstream),*(Path(__file__).parent/name for name in _CODE)]
    for path in files:
        proof['binding_stamps'][str(path)] = list(native._field_stamp(path))
    return index,csv,identity,proof


def _compose(upstream,size,index,source,csv,identity,proof):
    from omegaconf import OmegaConf
    cfg = native._compose(upstream,size,index,csv)
    for collection in (cfg.dataset,cfg.datamodule.dataset_conf):
        for role in ('train','val','test'):
            for entry in collection[role]:
                entry._target_ = TARGET
                OmegaConf.update(entry,'archive_fd',source.fileno(),force_add=True)
                OmegaConf.update(entry,'archive_handle_identity',proof,force_add=True)
    OmegaConf.update(cfg,CONTEXT_KEY,{'format':'emg2pose_memory_runtime_binding_v1','size':size,
                                   'index':str(index),'metadata_csv':str(csv),'data_identity':identity,
                                   'handle_identity':proof,'seed_policy':native.SEED_POLICY,
                                   'adapter_policy':POLICY},force_add=True)
    source.assert_current()
    return cfg


def compose_config(upstream: Path,size: str,index: Path,*,source: VerifiedSource,metadata_csv: Path):
    """Compose an unchanged native recipe, replacing only source storage."""
    if size not in native.SIZES:
        raise ValueError('size must be small, middle or large')
    upstream,_,_ = storage._official(upstream)
    index,csv,identity,proof = _binding(upstream,index,source,metadata_csv)
    return _compose(upstream,size,index,source,csv,identity,proof)


def _validated_config(upstream,config,source):
    from omegaconf import OmegaConf
    upstream,_,_ = storage._official(upstream)
    try:
        context = OmegaConf.to_container(config[CONTEXT_KEY],resolve=True)
        if (context.get('format') != 'emg2pose_memory_runtime_binding_v1' or
                context.get('size') not in native.SIZES or context.get('seed_policy') != native.SEED_POLICY or
                context.get('adapter_policy') != POLICY):
            raise ValueError('invalid sealed-source runtime binding')
        index,csv,identity,proof = _binding(upstream,context['index'],source,context['metadata_csv'])
        expected = _compose(upstream,context['size'],index,source,csv,identity,proof)
        if OmegaConf.to_container(config,resolve=True) != OmegaConf.to_container(expected,resolve=True):
            raise ValueError('config differs from exact native recipe or concrete verified source')
        clean = OmegaConf.to_container(config,resolve=True)
        clean.pop(CONTEXT_KEY)
        canonical = native._canonical_config(OmegaConf.create(clean),upstream,index,csv)
        for collection in (canonical['dataset'],canonical['datamodule']['dataset_conf']):
            for role in ('train','val','test'):
                for entry in collection[role]:
                    entry['archive_fd'] = 'verified-source://descriptor'
                    entry['archive_handle_identity'] = 'verified-source://content-identity'
        return upstream,context,identity,canonical
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('invalid sealed-source runtime config') from error


def build_runtime(upstream: Path,config,*,source: VerifiedSource):
    """Use native seeded setup and the original model/datamodule factories."""
    upstream,context,_,_ = _validated_config(upstream,config,source)
    native._activate(upstream)
    from egoemg._train_utils import setup_runtime
    from egoemg.train import make_lightning_module
    from egoemg.datamodule import make_data_module
    setup_runtime(config)
    module = make_lightning_module(config)
    dm = make_data_module(config)
    import torch
    model = module.model
    conv = next(layer for layer in model.featurizer.modules() if isinstance(layer,torch.nn.Conv1d))
    if (model.out_channels != 20 or (model.left_context,model.right_context) != (510,0) or
            conv.in_channels != 16 or sum(p.numel() for p in model.parameters()) != native.SIZES[context['size']]):
        raise RuntimeError('actual native module differs from approved16ch/20D architecture')
    return module,dm


def runtime_identity(upstream: Path,config,*,source: VerifiedSource) -> dict:
    """Bind data, exact scientific recipe and runtime content across rebuilds."""
    upstream,context,identity,canonical = _validated_config(upstream,config,source)
    bundle = hashlib.sha256()
    for path in _official_files(upstream):
        bundle.update(str(path.relative_to(upstream)).encode())
        bundle.update(path.read_bytes())
    code = {'egoemg_repro/'+name:storage._sha256(Path(__file__).parent/name) for name in _CODE}
    dependencies = {}
    for name in ('torch','torchvision','numpy','hydra-core','omegaconf','pytorch-lightning',
                 'torchmetrics','lightning-utilities','pandas','joblib','h5py','scipy'):
        try:dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:dependencies[name] = None
    return {'format':'emg2pose_memory_runtime_identity_v1','size':context['size'],'data':identity,
            'config_sha256':native._hash(canonical),'upstream_revision':UPSTREAM_COMMIT,
            'upstream_bundle_sha256':bundle.hexdigest(),'runtime_files_sha256':code,
            'dependencies':dependencies,'python':platform.python_version(),
            'seed_policy':native.SEED_POLICY,'adapter_policy':POLICY,
            'initialization':'seeded random official module; no pretrained or resume path; no downloads',
            'identity_path_policy':'validated private instance bindings; content identity survives exact cache rebuild'}


def reload_runtime(upstream: Path,config,state_dict,*,identity: dict,source: VerifiedSource):
    """Strict weight restore; optimizer/RNG/loop restoration belongs to callers."""
    if identity != runtime_identity(upstream,config,source=source):
        raise ValueError('strict reload runtime/config/data identity differs')
    if not isinstance(state_dict,Mapping):
        raise ValueError('strict reload requires a state dictionary')
    module,dm = build_runtime(upstream,config,source=source)
    import torch
    expected = module.state_dict()
    for key,value in state_dict.items():
        if key in expected and (not isinstance(value,torch.Tensor) or value.dtype != expected[key].dtype or
                                value.layout != expected[key].layout):
            raise ValueError('strict reload tensor dtype/layout differs: '+key)
    module.load_state_dict(state_dict,strict=True)
    return module,dm


def close_data_module(data_module):
    """Release owned dataset FDs after consumers finish, including shared leaves.

    This does not terminate workers or release descriptor rights held by another
    process. The owning controller must wait for those consumers separately.
    """
    from .emg2pose_memory_dataset import SealedTarDataset
    seen = set()
    def close(dataset):
        if dataset is None or id(dataset) in seen:return
        seen.add(id(dataset))
        if isinstance(dataset,SealedTarDataset):dataset.close()
        else:
            for child in getattr(dataset,'datasets',()):close(child)
            close(getattr(dataset,'dataset',None))
    for role in ('train','val','test','predict'):
        close(getattr(data_module,role+'_dataset',None))
