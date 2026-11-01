"""Full held-out labels on unchanged native EMGFormer windows.

The original point cohort is preserved. This adapter only admits input bytes
and numeric full-window targets; it grants no trained-model or result permit.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import weakref

import numpy as np
from omegaconf import OmegaConf

from .classic_emg import _file_hash, _hash
from .classic_emg_cohort import _safe
from .data import union_intervals
from . import emgformer_cohort as point
from .emgformer_data import EmgformerDataset, W

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = _file_hash(__file__)
ROLES = ('val', 'test')
_INPUTS = weakref.WeakKeyDictionary()


def _sources(upstream):
    if _file_hash(__file__) != IMPORTED_SHA:
        raise ValueError('imported trajectory data source changed')
    return dict(point._sources(upstream), **{'egoemg_repro/emgformer_trajectory_data.py': Path(__file__).resolve()})


def _config_sha(config):
    return _hash(OmegaConf.to_container(config, resolve=True))


def _point_live_binding(cohort):
    """Bind the already-validated parent without reconstructing its datamodules."""
    cohort.guard.check()
    if type(cohort) is not point.EmgformerCohort or set(cohort.datasets) != {'train', 'val', 'test'}:
        raise ValueError('actual canonical point parent binding changed')
    readers = {}
    for role, reader in cohort.datasets.items():
        if type(reader) is not EmgformerDataset or reader.role != role:
            raise ValueError('canonical point reader role binding changed')
        readers[role] = dict(binding=point._reader_binding(reader, dict(cohort.input_paths)),
            config=_config_sha(reader.config), windows=reader.windows, partitions=reader.partitions,
            samples=len(reader))
    return _hash(dict(identity=cohort.identity, config=_config_sha(cohort.config),
        normalization=cohort.normalization, partitions=cohort.partitions,
        upstream=str(cohort.upstream), readers=readers))


class TrajectoryDataset:
    """Separate original reader instances with their supported full-target flag."""
    def __init__(self, data, partitions, *, upstream, config, normalization, role,
                 expected_stamps=None, expected_binding=None, expected_sources=None):
        if role not in ROLES:
            raise ValueError('full trajectory role must be val or test')
        self.reader = EmgformerDataset(data, partitions, upstream=upstream, config=config,
            normalization=normalization, role=role)
        for view in self.reader.views:
            view.center_target_only = view._center_target_only = False
        self.paths = point._paths(self.reader.store)
        paths = _sources(self.reader.upstream)
        self._guard = point._Guard([*self.paths.values(), *paths.values()])
        if expected_stamps is not None and self._guard.nine != expected_stamps:
            raise ValueError('trajectory worker input/source generation changed')
        self._sources_sha256 = {n: _file_hash(p) for n, p in paths.items()}
        self._binding = point._reader_binding(self.reader, self.paths)
        self._partition_sha256 = _hash(self.reader.partitions)
        self._config_sha256 = _config_sha(self.reader.config)
        if ((expected_binding is not None and self._binding != expected_binding)
            or (expected_sources is not None and self._sources_sha256 != expected_sources)):
            raise ValueError('trajectory worker reader/source binding changed')
        self.check()

    def check(self):
        self._guard.check()
        reader = self.reader
        sources = _sources(reader.upstream)
        if (type(reader) is not EmgformerDataset or reader.role not in ROLES
            or _hash(reader.partitions) != self._partition_sha256
            or reader.windows != [w for w in reader.partitions['windows'] if w['role'] == reader.role]
            or _config_sha(reader.config) != self._config_sha256
            or any(view.center_target_only or view._center_target_only for view in reader.views)
            or {n: _file_hash(p) for n, p in sources.items()} != self._sources_sha256
            or point._reader_binding(reader, self.paths) != self._binding):
            raise ValueError('trajectory full reader computation binding changed')
        self._guard.check()

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        self.check()
        sample = self.reader[index]
        if (sample['joint_angles'].shape != (22, W) or sample['joint_angles'].dtype != np.float32
            or sample['label_valid_mask'].shape != (W,) or sample['label_valid_mask'].dtype != np.bool_
            or not np.isfinite(sample['joint_angles']).all()):
            raise ValueError('finite original full trajectory target/mask required')
        self._guard.check()
        return sample

    def __getstate__(self):
        self.check(); reader = self.reader
        return dict(data=reader.store.root, partitions=reader.partitions, upstream=reader.upstream,
            config=reader.config, normalization=reader.normalization, role=reader.role,
            expected_stamps=self._guard.nine, expected_binding=self._binding,
            expected_sources=self._sources_sha256)

    def __setstate__(self, state):
        self.__init__(**state)


def _full_target_scan(datasets, *, chunk_rows, check):
    result = {}
    for role, dataset in datasets.items():
        reader = dataset.reader; store = reader.store
        spans = union_intervals((w['start'], w['end']) for w in reader.windows)
        rows = sum(hi-lo for lo, hi in spans); bad = converted_bad = supervised = 0
        for lo, hi in spans:
            for start in range(lo, hi, chunk_rows):
                check(); stop = min(hi, start+chunk_rows)
                supervised += int(np.count_nonzero(store.field('generated_label_valid')[start:stop]))
                for hand in ('left', 'right'):
                    target = np.asarray(store.field('generated_joint_angles_'+hand)[start:stop], np.float32)
                    bad += int(np.count_nonzero(~np.isfinite(target)))
                    for axis in ('pitch', 'yaw'):
                        raw = np.asarray(store.field('mocap_'+hand+'_wrist_'+axis)[start:stop], np.float32)
                        bad += int(np.count_nonzero(~np.isfinite(raw)))
                        # Literal dtype path in the original full-window reader.
                        converted_bad += int(np.count_nonzero(~np.isfinite(np.deg2rad(raw))))
                if bad or converted_bad:
                    raise ValueError('nonfinite full held-out trajectory label; masked rows are not repaired')
                check()
        if not supervised:
            raise ValueError('full held-out trajectory needs actual generated supervision')
        result[role] = dict(unique_target_rows=rows, consumed_target_elements=rows*44,
            nonfinite_target_entries=bad, nonfinite_converted_wrist_entries=converted_bad,
            supervised_hand_rows=supervised, target_mode='full_trajectory')
    return result


@dataclass(eq=False, frozen=True)
class TrajectoryInputs:
    point_cohort: object
    datasets: object
    identity: dict
    _guard: object

    def check(self):
        bound = _INPUTS.get(self)
        if (bound is None or bound['point'] is not self.point_cohort
            or bound['guard'] is not self._guard or self.datasets is not bound['datasets']):
            raise ValueError('actual registered trajectory input object required')
        self._guard.check()
        if (self.identity.get('sha256') != bound['seal']
            or _hash({k: v for k, v in self.identity.items() if k != 'sha256'}) != bound['seal']
            or _point_live_binding(self.point_cohort) != bound['point_live_binding']
            or self.identity['point_cohort_sha256'] != self.point_cohort.identity['sha256']
            or set(self.datasets) != set(ROLES)
            or self.identity['sources_sha256'] != {n: _file_hash(p) for n, p in _sources(self.point_cohort.upstream).items()}):
            raise ValueError('trajectory input identity/source binding changed')
        for role, dataset in self.datasets.items():
            if (dataset is not bound['members'][role] or type(dataset) is not TrajectoryDataset
                or dataset.reader.role != role or len(dataset) != self.identity['counts'][role]
                or dataset._binding != self.identity['reader_bindings_sha256'][role]):
                raise ValueError('trajectory dataset role/window binding changed')
            dataset.check()
        self._guard.check()


def admit_trajectory_inputs(data, *, upstream, config, normalization,
                            expected_point_identity=None, expected=None, chunk_rows=point.MAX_CHUNK_ROWS):
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= point.MAX_CHUNK_ROWS:
        raise ValueError('trajectory chunk_rows must be integer1..262144')
    upstream = _safe(upstream)
    source_paths = _sources(upstream); guard = point._Guard(source_paths.values())
    cohort = point.admit_emgformer_cohort(data, upstream=upstream, config=config,
        normalization=normalization, expected=expected_point_identity, chunk_rows=chunk_rows)
    guard.check(); parent_binding = _point_live_binding(cohort)
    for path in cohort.input_paths.values(): guard.add(path)
    datasets = {role: TrajectoryDataset(data, cohort.partitions, upstream=upstream,
        config=config, normalization=normalization, role=role) for role in ROLES}
    def check():
        guard.check(); cohort.guard.check()
        for dataset in datasets.values(): dataset._guard.check()
    def validate():
        check()
        if _point_live_binding(cohort) != parent_binding:
            raise ValueError('validated point parent computation binding changed')
        for dataset in datasets.values(): dataset.check()
    validate()
    finite = _full_target_scan(datasets, chunk_rows=chunk_rows, check=check)
    validate()
    identity = dict(format='original_emgformer_full_heldout_trajectory_inputs_v1',
        size=cohort.identity['size'], point_cohort_sha256=cohort.identity['sha256'],
        counts={role: len(ds) for role, ds in datasets.items()}, repeat_by_role={role: 1 for role in ROLES},
        payload_file_count=cohort.identity['payload_file_count'],
        payload_files_sha256=copy.deepcopy(cohort.identity['payload_files_sha256']),
        payload_file_bytes=copy.deepcopy(cohort.identity['payload_file_bytes']),
        full_payload_content_hashed=True, finite_heldout_targets=finite,
        reader_bindings_sha256={role: ds._binding for role, ds in datasets.items()},
        sources_sha256={n: _file_hash(p) for n, p in source_paths.items()},
        metadata_matches_registered_legacy_release=cohort.identity['metadata_matches_registered_legacy_release'],
        target_override='supported original center_target_only=False on separate held-out native views',
        wrist_conversion='original full reader float32 np.deg2rad; not scalar point-reader rounding',
        point_evaluation_readers_unchanged=True, stored_wrist_valid_flag_applied=False,
        scientific_eligible=False, full_fit_or_evaluation_permit=False, full_paper_complete=False)
    identity['sha256'] = _hash(identity)
    if expected is not None and _hash(expected) != _hash(identity):
        raise ValueError('actual complete trajectory input identity differs from receipt')
    mapping = MappingProxyType(datasets)
    value = TrajectoryInputs(cohort, mapping, identity, guard)
    _INPUTS[value] = dict(point=cohort, datasets=mapping, members=dict(datasets), guard=guard,
        seal=identity['sha256'], point_live_binding=parent_binding)
    value.check()
    return value


def trajectory_loader(inputs, role, *, cpu_engineering=False):
    if type(inputs) is not TrajectoryInputs or role not in ROLES or type(cpu_engineering) is not bool:
        raise ValueError('registered trajectory inputs, held-out role and explicit engineering flag required')
    inputs.check()
    from egoemg.datamodule import make_data_module
    dm = make_data_module(inputs.point_cohort.config)
    if cpu_engineering:
        dm.batch_size = 2; dm.num_workers = dm.eval_num_workers = 0
    loader = dm._make_loader(inputs.datasets[role], shuffle=False, num_workers=dm.eval_num_workers)
    loader.egoemg_repro_scope = 'cpu_engineering' if cpu_engineering else 'native_trajectory_loader_only'
    inputs.check()
    return loader
