"""Exact classic EgoEMG model contracts around unchanged official factories.

This module does not load data, adapt masks, train, or restore checkpoints.
VEMG's declared 16-slot data layout is explicitly resolved to its real 8-channel
network; the official network, tuple output and two-dimensional mask stay intact.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from . import UPSTREAM_COMMIT


VARIANTS = ('vemg2pose', 'vemg2pose_with_aug', 'neuropose', 'neuropose_with_aug')
CONTEXT_KEY = '_classic_emg'
SEED_POLICY = 'official setup_runtime before model factory; explicit correction of upstream train() construction-before-seeding order'
NORM_SHA256 = 'b787a1ef98d48f434cb808e0e61e49c1ee0c09fae5b7ca9fc6a05ba4b14ee122'


@dataclass(frozen=True)
class ClassicContract:
    variant: str
    experiment: str
    input_channels: int
    output_channels: int
    metadata_out_channels: int
    left_context: int
    right_context: int
    parameters: int
    declared_layout: str
    effective_layout: str
    layout_reason: str
    channel_indices: tuple[int, ...]
    channel_interpolate: bool
    mask_contract: str


def classic_contract(variant: str) -> ClassicContract:
    if variant not in VARIANTS:
        raise ValueError('variant must be one of ' + ', '.join(VARIANTS))
    vemg = variant.startswith('vemg')
    experiment = 'emg2pose/regression_' + variant.removesuffix('_with_aug') + '_egoemg'
    if variant.endswith('_with_aug'):
        experiment += '_with_aug'
    return ClassicContract(variant, experiment, 8 if vemg else 16, 22, 20,
                           1790 if vemg else 0, 0, 5963948 if vemg else 6354969,
                           'emg2pose_interpolate16', 'target_hand' if vemg else 'emg2pose_interpolate16',
                           'official model Conv1d input=8; explicit data-layout correction, network unchanged'
                           if vemg else 'official 16-slot layout retained',
                           (10, 12, 0, 1, 2, 4, 5, 6), False,
                           'original official tuple with two-dimensional label mask; loader protocol not selected')


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _approved_upstream(upstream):
    root = Path(upstream).resolve()
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError('upstream revision must match the approved pin ' + UPSTREAM_COMMIT)
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'],
                            capture_output=True, text=True)
    if status.returncode or status.stdout.strip():
        raise ValueError('upstream tree must be clean before classic composition or construction')
    if _file_hash(root / 'assets/per_dataset_norm_stats.json') != NORM_SHA256:
        raise ValueError('pinned official normalization source differs')
    return root


def _compose_pair(root, contract):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root / 'config'), version_base='1.1'):
        original = compose(config_name='base', overrides=['experiment=' + contract.experiment])
    raw = OmegaConf.to_container(original, resolve=True)
    # Keep interpolations until applying the sole layout decision and explicit
    # pinned normalization-asset location, including dataset_conf references.
    effective = OmegaConf.create(OmegaConf.to_container(original, resolve=False))
    effective.egoemg_emg_layout = contract.effective_layout
    effective.datamodule.per_dataset_norm_stats_path = str(root / 'assets/per_dataset_norm_stats.json')
    OmegaConf.resolve(effective)
    return raw, effective


def _binding(root, contract):
    return {'format': 'classic_emg_model_binding_v1', 'upstream': str(root),
            'variant': contract.variant, 'seed_policy': SEED_POLICY}


def compose_classic(upstream: Path, variant: str):
    """Compose one of four recipes; arbitrary Hydra overrides are not accepted."""
    contract = classic_contract(variant)  # Reject unknown names before source access.
    root = _approved_upstream(upstream)
    _, config = _compose_pair(root, contract)
    from omegaconf import OmegaConf
    OmegaConf.update(config, CONTEXT_KEY, _binding(root, contract), force_add=True)
    return config


def _validated(upstream, config):
    from omegaconf import OmegaConf
    root = _approved_upstream(upstream)
    try:
        context = OmegaConf.to_container(config[CONTEXT_KEY], resolve=True)
        contract = classic_contract(context['variant'])
        if context != _binding(root, contract):
            raise ValueError('config private source binding differs')
        actual = OmegaConf.to_container(config, resolve=True)
        del actual[CONTEXT_KEY]
        raw, expected = _compose_pair(root, contract)
        effective = OmegaConf.to_container(expected, resolve=True)
        # Exact JSON comparison also distinguishes bool/int and int/float edits.
        if _hash(actual) != _hash(effective):
            raise ValueError('config differs from the exact approved classic recipe')
        return root, contract, raw, effective
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid classic config binding') from exc


def _activate(root):
    loaded = sys.modules.get('egoemg')
    if loaded is not None and Path(loaded.__file__).resolve() != root / 'egoemg/__init__.py':
        raise ValueError('another official package path is already active')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def build_classic_model(upstream: Path, config):
    """Return the actual official model, with its original tuple/mask behavior."""
    root, contract, _, _ = _validated(upstream, config)
    _activate(root)
    from egoemg._train_utils import setup_runtime
    from egoemg.train import make_lightning_module
    setup_runtime(config)
    module = make_lightning_module(config)  # Its actual factory uses _convert_='all'.
    model = module.model
    expected_class = 'VEMG2PoseWithInitialState' if contract.input_channels == 8 else 'PoseModule'
    valid = (type(model).__module__ == 'egoemg.models.modules.pose' and type(model).__name__ == expected_class
             and model.out_channels == contract.metadata_out_channels
             and (model.left_context, model.right_context) == (contract.left_context, contract.right_context)
             and sum(p.numel() for p in model.parameters()) == contract.parameters
             and model.provide_initial_pos is False)
    if contract.input_channels == 8:
        import torch
        conv = next(layer for layer in model.featurizer.modules() if isinstance(layer, torch.nn.Conv1d))
        valid = (valid and conv.in_channels == 8 and model.decoder.lstm.input_size == 86
                 and model.decoder.lstm.hidden_size == 512 and model.decoder.lstm.num_layers == 2
                 and model.decoder.mlp_out[-1].out_features == 44 and model.decoder.scale == .01
                 and model.rollout_freq == 50 and model.num_position_steps == 500)
    else:
        valid = (valid and type(model.featurizer).__name__ == 'NeuroPose'
                 and model.featurizer.linear.in_features == 32 and model.featurizer.linear.out_features == 22
                 and model.predict_vel is False)
    if not valid:
        raise RuntimeError('actual official classic model does not match its approved contract')
    return model


def _canonical_config(value):
    # Only the approved source normalization location is path-independent.
    # Other resolved recipe values remain exact; no arbitrary key is dropped.
    value = json.loads(json.dumps(value, allow_nan=False))
    value['datamodule']['per_dataset_norm_stats_path'] = 'upstream://assets/per_dataset_norm_stats.json'
    for collection in (value['dataset'], value['datamodule']['dataset_conf']):
        for entries in collection.values():
            for entry in entries:
                entry['norm_stats_path'] = 'upstream://assets/per_dataset_norm_stats.json'
    return value


def _loss_evidence(root, weights):
    """Execute the pinned original alias reduction, without importing Lightning."""
    source = (root / 'egoemg/lightning.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EmgPredictionModule')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_step')
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == '_LOSS_KEY_COMPAT' for t in n.targets))
    nodes = method.body[start:start + 3]
    aliases = ast.literal_eval(nodes[0].value)
    names = sorted({aliases.get(name, name) for name in weights})
    code = compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-classic-loss-reducer>', 'exec')
    effective = {}
    for name in names:
        scope = dict(self=SimpleNamespace(loss_weights=weights), stage='audit',
                     metrics={'audit_' + key: float(key == name) for key in names})
        exec(code, scope)
        effective[name] = scope['loss']
    return effective, hashlib.sha256(''.join(ast.get_source_segment(source, n) for n in nodes).encode()).hexdigest()


def resolved_classic_identity(config, contract: ClassicContract) -> dict:
    """Bind exact raw/effective recipes and source; make no data/run claims."""
    try:
        upstream = config[CONTEXT_KEY]['upstream']
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid classic config source binding') from exc
    root, expected, raw, effective = _validated(upstream, config)
    if contract != expected:
        raise ValueError('contract differs from the configured classic variant')
    files = sorted({*root.glob('egoemg/**/*.py'), *root.glob('config/**/*.yaml'),
                    *root.glob('egoemg/UmeTrack/**/*.json'), root / 'assets/per_dataset_norm_stats.json'})
    records = {str(path.relative_to(root)): _file_hash(path) for path in files}
    critical = ('egoemg/train.py', 'egoemg/_train_utils.py', 'egoemg/lightning.py',
                'egoemg/models/modules/base.py', 'egoemg/models/modules/pose.py',
                'egoemg/models/featurizers/tds.py', 'egoemg/models/featurizers/neuropose.py',
                'egoemg/models/decoders/lstm.py', 'config/base.yaml',
                'config/experiment/' + contract.experiment + '.yaml', 'assets/per_dataset_norm_stats.json')
    weights, loop_hash = _loss_evidence(root, raw['loss_weights'])
    project = Path(__file__).resolve().parents[1]
    runtime = {name: _file_hash(project / name) for name in ('egoemg_repro/__init__.py', 'egoemg_repro/classic_emg.py')}
    dependencies = {}
    for name in ('torch', 'torchvision', 'numpy', 'hydra-core', 'omegaconf', 'pytorch-lightning', 'torchmetrics'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {'format': 'classic_emg_model_contract_identity_v1', 'contract': asdict(contract),
            'raw_config_sha256': _hash(_canonical_config(raw)),
            'effective_config_sha256': _hash(_canonical_config(effective)),
            'raw_loss_weights': raw['loss_weights'], 'effective_loss_weights': weights,
            'official_loss_reducer_sha256': loop_hash,
            'official_source': {'revision': UPSTREAM_COMMIT, 'sha256': _hash(records), 'file_count': len(records),
                                'critical_files_sha256': {name: records[name] for name in critical}},
            'runtime_files_sha256': runtime, 'dependencies': dependencies, 'seed_policy': SEED_POLICY,
            'initialization': 'seeded random official model; no checkpoint, resume or pretrained path accepted',
            'scope': 'model/config contract only; no loader protocol, data identity, training or benchmark result',
            'identity_path_policy': 'approved normalization-asset locations tokenized; other resolved recipe values hashed exactly'}
