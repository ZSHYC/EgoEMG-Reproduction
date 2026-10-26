"""Strict, independent access to the pinned official 16ch/20D runtime.

This module builds models and datamodules; it never fits, selects checkpoints or
evaluates a benchmark. Actual paths live only in the private resolved config.
Identity records compare content, so relocating identical artifacts is allowed.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from . import UPSTREAM_COMMIT
from . import emg2pose_data as storage


CONVERTER_SHA256 = '226f6420a89c27916434f4f2b46d7297beea2c1c030bc2149e294a66374431e6'
NORM_SHA256 = 'b787a1ef98d48f434cb808e0e61e49c1ee0c09fae5b7ca9fc6a05ba4b14ee122'
SIZES = {'small': 3481636, 'middle': 6638884, 'large': 16342308}
CONTEXT_KEY = '_emg2pose_runtime'
SEED_POLICY = 'official setup_runtime before model factory; explicit correction of upstream train() launch ordering'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _decode(values):
    if values.dtype.kind != 'S' or values.ndim != 1:
        raise ValueError('metadata strings must be one-dimensional fixed UTF-8 bytes')
    return [bytes(v).decode('utf-8') for v in values]


def _field_stamp(path):
    status = storage._safe_path(path, file=True).stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _source_records(source, rows):
    sessions = source.get('sessions')
    if not isinstance(sessions, list) or not sessions:
        raise ValueError('source receipt needs session records')
    seen, offset = set(), 0
    for session in sessions:
        name = storage._safe_name(session['filename'])
        if name in seen or name not in rows:
            raise ValueError('duplicate or unknown source filename')
        seen.add(name)
        for key, expected in rows[name].items():
            actual = session.get('session_id') if key == 'session' else session.get(key)
            if actual != expected:
                raise ValueError('source metadata differs from the bound CSV')
        count = _integer(session.get('rows'), 'session rows')
        if session.get('start_idx') != offset or session.get('end_idx') != offset + count:
            raise ValueError('source session offsets must cover contiguous half-open rows')
        offset += count
    return sessions, offset, seen


def validate_completed_data(upstream: Path, data: Path, source_receipt: Path, *,
                            metadata_csv: Path | None = None) -> dict:
    """Hash actual completed artifacts and verify their source/metadata binding.

    The original receipt's exact byte hash is mandatory. Its embedded semantic
    copy may have a different serialization; it is never substituted for that
    hash. Original HDF5 need not be reloaded by a relocated memmap consumer.
    """
    _, official, _ = storage._official(upstream)
    data = storage._safe_path(data)
    if data.name.endswith('.incomplete'):
        raise ValueError('incomplete staging cannot be a runtime dataset')
    source_receipt = storage._safe_path(source_receipt, file=True)
    try:
        completion_path = storage._safe_path(data / 'completion.json', file=True)
        manifest_path = storage._safe_path(data / 'manifest.json', file=True)
        metadata_path = storage._safe_path(data / 'metadata.npz', file=True)
        snapshot_path = storage._safe_path(data / 'source-receipt.json', file=True)
        completion_hash, snapshot_hash = storage._sha256(completion_path), storage._sha256(snapshot_path)
        completion, manifest = storage._json(completion_path), storage._json(manifest_path)
        source, snapshot = storage._json(source_receipt), storage._json(snapshot_path)
        receipt_hash = storage._sha256(source_receipt)
        if (completion.get('format') != 'emg2pose_conversion_v1' or completion.get('complete') is not True or
                manifest.get('format_version') != 'emg2pose_direct_memmap_v1' or
                source.get('format') != 'emg2pose_source_inventory_v1'):
            raise ValueError('dataset is not a completed direct EMG2Pose conversion')
        if (source != snapshot or completion.get('source_receipt_sha256') != receipt_hash or
                manifest.get('source_receipt_sha256') != receipt_hash):
            raise ValueError('original source receipt identity or embedded snapshot differs')
        if (completion.get('manifest_sha256') != storage._sha256(manifest_path) or
                completion.get('metadata_sha256') != storage._sha256(metadata_path)):
            raise ValueError('manifest or metadata hash differs from completion receipt')
        for record in (completion, source):
            if record.get('converter_sha256') != CONVERTER_SHA256:
                raise ValueError('conversion source identity is not the approved corrected source')
            upstream_record = record.get('upstream', {})
            if (upstream_record.get('revision') != official['revision'] or
                    upstream_record.get('files_sha256') != official['files_sha256']):
                raise ValueError('conversion upstream identity differs')
        cohort = source.get('cohort')
        if cohort not in {'engineering', 'mini', 'full'}:
            raise ValueError('unknown source cohort')
        for key, expected in [('cohort', cohort), ('full_corpus_complete', cohort == 'full'),
                              ('engineering_only', cohort != 'full'),
                              ('inspected_file_set_complete', cohort != 'engineering')]:
            if (source.get(key) != expected or completion.get(key) != expected or
                    (type(expected) is bool and (type(source.get(key)) is not bool or type(completion.get(key)) is not bool))):
                raise ValueError('source and completion cohort claims disagree')
        csv_path = storage._safe_path(metadata_csv if metadata_csv is not None else source['metadata_csv']['path'], file=True)
        csv_hash = storage._sha256(csv_path)
        rows = storage._csv_rows(csv_path)
        if (csv_hash != source['metadata_csv']['sha256'] or
                len(rows) != source['metadata_csv']['filename_count']):
            raise ValueError('actual source metadata CSV identity differs')
        if cohort != 'engineering' and (csv_hash != storage.CSV_SHA256 or len(rows) != storage.FULL_FILE_COUNT):
            raise ValueError('official cohort requires pinned complete metadata')
        sessions, total_rows, names = _source_records(source, rows)
        if cohort == 'full' and names != set(rows):
            raise ValueError('full cohort lacks the exact official filename set')
        if cohort == 'mini' and (len(sessions) != 30 or source.get('acquisition', {}).get('archive_sha256') != storage.MINI_SHA256):
            raise ValueError('mini identity does not match the complete official archive')
        if (completion.get('file_count') != len(sessions) or source.get('total_rows') != total_rows or
                completion.get('total_rows') != total_rows or manifest.get('total_rows') != total_rows):
            raise ValueError('source, manifest and completion row/file counts differ')
        expected_sessions = [{k: s[k] for k in ('session_id', 'filename', 'start_idx', 'end_idx', 'rows', 'split', 'generalization')} for s in sessions]
        if manifest.get('sessions') != expected_sessions:
            raise ValueError('manifest session records differ from the original receipt')
        counts = {key: dict(Counter(s[key] for s in sessions)) for key in ('split', 'generalization', 'side')}
        if source.get('counts') != counts or completion.get('counts') != counts:
            raise ValueError('cohort counts differ from source sessions')
        shapes = {'emg': [total_rows, 16], 'joint_angles': [total_rows, 20],
                  'time': [total_rows], 'valid_mask': [total_rows]}
        fields = manifest.get('fields', {})
        if set(fields) != set(shapes) or fields != completion.get('fields'):
            raise ValueError('runtime requires exactly the four bound EMG2Pose fields')
        field_bytes, field_stamps = 0, {}
        for key, shape in shapes.items():
            spec = fields[key]
            dtype = storage.FIELD_DTYPES[key]
            size = int(np.prod(shape)) * dtype.itemsize
            if (spec.get('filename') != key + '.dat' or spec.get('shape') != shape or
                    not isinstance(spec.get('dtype'), str) or np.dtype(spec['dtype']) != dtype or spec.get('bytes') != size):
                raise ValueError('field shape/dtype/path is not the approved 16ch/20D schema')
            path = storage._safe_path(data / spec['filename'], file=True)
            before = _field_stamp(path)
            actual_hash = storage._sha256(path)
            if _field_stamp(path) != before:
                raise ValueError('field identity changed during runtime admission')
            if before[2] != size or actual_hash != spec.get('sha256'):
                raise ValueError('actual field bytes or hash differ')
            field_stamps[path] = before
            field_bytes += size
        if field_bytes != completion.get('field_bytes'):
            raise ValueError('completed field byte count differs')
        with np.load(metadata_path, allow_pickle=False) as archive:
            md = {key: archive[key].copy() for key in archive.files}
        if any(value.dtype.kind == 'O' for value in md.values()):
            raise ValueError('object sidecars are forbidden')
        for category, key in [('users', 'user'), ('stages', 'stage'), ('sides', 'side'), ('splits', 'split')]:
            values = _decode(md[category + '_' + key])
            ids = md['session_' + key + '_id']
            if (len(values) != len(set(values)) or ids.shape != (len(sessions),) or ids.dtype.kind not in 'iu' or
                    any(i < 0 or i >= len(values) or values[i] != s[key] for i, s in zip(ids, sessions))):
                raise ValueError('sidecar lookup or original partition membership differs')
        for key, source_key in [('start_idx', 'start_idx'), ('end_idx', 'end_idx'), ('length', 'rows'),
                                ('sample_rate', 'sample_rate'), ('held_out_user', 'held_out_user'), ('held_out_stage', 'held_out_stage')]:
            value = md['session_' + key]
            if value.shape != (len(sessions),) or value.tolist() != [s[source_key] for s in sessions]:
                raise ValueError('sidecar session metadata differs from source')
            if key.startswith('held_out') and value.dtype.kind != 'b':
                raise ValueError('held-out flags must be boolean')
            expected_dtype = np.dtype('bool' if key.startswith('held_out') else ('<f4' if key == 'sample_rate' else '<i8'))
            if value.dtype != expected_dtype:
                raise ValueError('sidecar session index/sample-rate dtype differs from the converter schema')
        for key in ('session_id', 'filename', 'generalization', 'moving_hand'):
            if _decode(md['session_' + key]) != [s[key] for s in sessions]:
                raise ValueError('sidecar session identity/generalization differs')
        block_keys = ['blocks_' + key for key in ('session_idx', 'start', 'end', 'length')]
        block_arrays = [md[k] for k in block_keys]
        if any(v.ndim != 1 or v.dtype.kind not in 'iu' for v in block_arrays) or len({len(v) for v in block_arrays}) != 1:
            raise ValueError('invalid block sidecar shapes')
        if len(block_arrays[0]) != completion.get('valid_blocks'):
            raise ValueError('block count differs from completion')
        prior = 0
        for si, a, b, length in zip(*block_arrays):
            if (si < 0 or si >= len(sessions) or a < prior or b <= a or b - a != length or
                    a < sessions[si]['start_idx'] or b > sessions[si]['end_idx']):
                raise ValueError('valid block crosses session bounds or overlaps')
            prior = b
        # Recheck the small transaction files after validating the large fields.
        if (storage._sha256(source_receipt) != receipt_hash or storage._sha256(csv_path) != csv_hash or
                storage._sha256(completion_path) != completion_hash or storage._sha256(snapshot_path) != snapshot_hash or
                storage._sha256(manifest_path) != completion['manifest_sha256'] or
                storage._sha256(metadata_path) != completion['metadata_sha256']):
            raise ValueError('data identity changed during runtime admission')
        # These transient filesystem stamps guard this admission transaction;
        # they are excluded from the relocatable content identity below.
        if any(_field_stamp(path) != before for path, before in field_stamps.items()):
            raise ValueError('field identity changed during runtime admission')
        return {'format': 'emg2pose_runtime_data_identity_v1', 'cohort': cohort,
                'full_corpus_complete': cohort == 'full', 'engineering_only': cohort != 'full',
                'rows': total_rows, 'file_count': len(sessions), 'input_channels': 16, 'angle_dimensions': 20,
                'counts': counts, 'present_splits': sorted(counts['split']),
                'source_receipt_sha256': receipt_hash, 'embedded_snapshot_sha256': snapshot_hash,
                'completion_sha256': completion_hash, 'manifest_sha256': completion['manifest_sha256'],
                'metadata_sha256': completion['metadata_sha256'], 'metadata_csv_sha256': csv_hash,
                'fields': fields, 'converter_sha256': CONVERTER_SHA256}
    except (OSError, KeyError, TypeError, UnicodeError, EOFError) as exc:
        raise ValueError('invalid completed EMG2Pose artifacts') from exc


def _compose(upstream, size, data, metadata_csv):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(upstream / 'config'), version_base=None):
        cfg = compose(config_name='base', overrides=['experiment=emgformer/emg2pose_emgformer_' + size])
    cfg.data_location = str(data)
    cfg.data_split.metadata_file = str(metadata_csv)
    cfg.datamodule.per_dataset_norm_stats_path = str(upstream / 'assets/per_dataset_norm_stats.json')
    for role in ('train', 'val', 'test'):
        cfg.dataset[role][0].memmap_dir = str(data)
        cfg.dataset[role][0].root_dir = str(data)
    OmegaConf.resolve(cfg)
    return cfg


def compose_config(upstream: Path, size: str, data: Path, *, source_receipt: Path,
                   metadata_csv: Path | None = None):
    """Compose one exact official recipe, bound to verified completed data."""
    if size not in SIZES:
        raise ValueError('size must be small, middle or large')
    upstream, _, _ = storage._official(upstream)
    data, source_receipt = storage._safe_path(data), storage._safe_path(source_receipt, file=True)
    identity = validate_completed_data(upstream, data, source_receipt, metadata_csv=metadata_csv)
    if storage._sha256(upstream / 'assets/per_dataset_norm_stats.json') != NORM_SHA256:
        raise ValueError('official normalization asset differs')
    source = storage._json(source_receipt)
    csv_path = storage._safe_path(metadata_csv if metadata_csv is not None else source['metadata_csv']['path'], file=True)
    cfg = _compose(upstream, size, data, csv_path)
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, CONTEXT_KEY, {'format': 'emg2pose_runtime_binding_v1', 'size': size,
                                      'data': str(data), 'source_receipt': str(source_receipt),
                                      'metadata_csv': str(csv_path), 'data_identity': identity,
                                      'seed_policy': SEED_POLICY}, force_add=True)
    return cfg


def _canonical_config(config, upstream, data, metadata_csv):
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(config, resolve=True)
    value.pop(CONTEXT_KEY, None)
    if value['data_location'] != str(data):
        raise ValueError('config dataset location does not match its bound path')
    value['data_location'] = 'artifact://emg2pose'
    if value['data_split']['metadata_file'] != str(metadata_csv):
        raise ValueError('config metadata path differs from bound CSV')
    value['data_split']['metadata_file'] = 'source://metadata.csv'
    for collection in (value['dataset'], value['datamodule']['dataset_conf']):
        for role in ('train', 'val', 'test'):
            for entry in collection[role]:
                for key in ('memmap_dir', 'root_dir'):
                    if entry[key] != str(data):
                        raise ValueError('config partition path differs from bound dataset')
                    entry[key] = 'artifact://emg2pose'
    norm = value['datamodule']['per_dataset_norm_stats_path']
    if norm != str(upstream / 'assets/per_dataset_norm_stats.json'):
        raise ValueError('normalization path differs from approved source')
    value['datamodule']['per_dataset_norm_stats_path'] = 'upstream://assets/per_dataset_norm_stats.json'
    return value


def _validated_config(upstream, config, source_receipt=None):
    from omegaconf import OmegaConf
    upstream, _, _ = storage._official(upstream)
    try:
        context = OmegaConf.to_container(config[CONTEXT_KEY], resolve=True)
        if (context.get('format') != 'emg2pose_runtime_binding_v1' or context.get('size') not in SIZES or
                context.get('seed_policy') != SEED_POLICY):
            raise ValueError('invalid private runtime binding')
        data = storage._safe_path(context['data'])
        source_receipt = storage._safe_path(source_receipt if source_receipt is not None else context['source_receipt'], file=True)
        identity = validate_completed_data(upstream, data, source_receipt, metadata_csv=Path(context['metadata_csv']))
        if identity != context['data_identity']:
            raise ValueError('config completed-data identity differs')
        csv_path = storage._safe_path(context['metadata_csv'], file=True)
        canonical = _canonical_config(config, upstream, data, csv_path)
        if canonical != _canonical_config(_compose(upstream, context['size'], data, csv_path), upstream, data, csv_path):
            raise ValueError('config differs from the exact approved EMG2Pose recipe')
        if storage._sha256(upstream / 'assets/per_dataset_norm_stats.json') != NORM_SHA256:
            raise ValueError('official normalization asset differs')
        return upstream, context, identity, canonical
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid EMG2Pose config binding') from exc


def _activate(upstream):
    loaded = sys.modules.get('egoemg')
    if loaded is not None and Path(loaded.__file__).resolve() != upstream / 'egoemg/__init__.py':
        raise ValueError('another official package path is already active in this process')
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))


def build_runtime(upstream: Path, config):
    """Build unchanged official factories after deterministic seeded setup."""
    upstream, context, _, _ = _validated_config(upstream, config)
    _activate(upstream)
    from egoemg._train_utils import setup_runtime
    from egoemg.train import make_lightning_module
    from egoemg.datamodule import make_data_module
    setup_runtime(config)  # Deliberately before construction; see SEED_POLICY.
    module = make_lightning_module(config)
    data_module = make_data_module(config)
    model = module.model
    import torch
    convolution = next(layer for layer in model.featurizer.modules() if isinstance(layer, torch.nn.Conv1d))
    if (model.out_channels != 20 or model.left_context != 510 or model.right_context != 0 or
            convolution.in_channels != 16 or sum(p.numel() for p in model.parameters()) != SIZES[context['size']]):
        raise RuntimeError('actual official module does not match the approved 16ch/20D architecture')
    return module, data_module


def runtime_identity(upstream: Path, config, source_receipt: Path) -> dict:
    """Return content identities, without source paths or participant strings."""
    upstream, context, data_identity, canonical = _validated_config(upstream, config, source_receipt)
    bundle = hashlib.sha256()
    files = sorted({*upstream.glob('egoemg/**/*.py'), *upstream.glob('config/**/*.yaml'),
                    *upstream.glob('egoemg/UmeTrack/**/*.json'), upstream / 'assets/per_dataset_norm_stats.json'})
    for path in files:
        bundle.update(str(path.relative_to(upstream)).encode())
        bundle.update(path.read_bytes())
    root = Path(__file__).resolve().parents[1]
    runtime_files = {name: storage._sha256(root / name) for name in
                     ('egoemg_repro/__init__.py', 'egoemg_repro/emg2pose_runtime.py', 'egoemg_repro/emg2pose_data.py')}
    dependencies = {}
    for name in ('torch', 'torchvision', 'numpy', 'hydra-core', 'omegaconf', 'pytorch-lightning',
                 'torchmetrics', 'lightning-utilities', 'pandas', 'joblib', 'h5py', 'scipy'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {'format': 'emg2pose_runtime_identity_v1', 'size': context['size'], 'data': data_identity,
            'config_sha256': _hash(canonical), 'upstream_revision': UPSTREAM_COMMIT,
            'upstream_bundle_sha256': bundle.hexdigest(), 'runtime_files_sha256': runtime_files,
            'dependencies': dependencies, 'seed_policy': SEED_POLICY,
            'initialization': 'seeded random official module; no pretrained or resume path; no downloads',
            'identity_path_policy': 'explicit validated locations in private config; content hashes compare relocated artifacts'}


def reload_runtime(upstream: Path, config, state_dict, *, identity: dict):
    """Strictly restore an already loaded trusted local weight dictionary.

    This is a weight/runtime check, not an optimizer/RNG resume API or a claim
    of a selected trained checkpoint. Task4 owns training-state checkpoints.
    """
    context = config[CONTEXT_KEY]
    current = runtime_identity(upstream, config, Path(context['source_receipt']))
    if identity != current:
        raise ValueError('strict reload runtime/config/data identity differs')
    module, data_module = build_runtime(upstream, config)
    if not isinstance(state_dict, Mapping):
        raise ValueError('strict reload requires a state dictionary')
    import torch
    expected = module.state_dict()
    for key, value in state_dict.items():
        if key in expected and (not isinstance(value, torch.Tensor) or value.dtype != expected[key].dtype or value.layout != expected[key].layout):
            raise ValueError('strict reload tensor dtype/layout differs: ' + key)
    module.load_state_dict(state_dict, strict=True)
    return module, data_module
