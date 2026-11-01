"""Selected native EMGFormer full trajectories and closed-process acceptance.

The owning workflow stays Torch-free. Each learned pass restores the actual
selected complete checkpoint; report verification only reads saved tensors.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'original_emgformer_full_trajectory_pass_v1'
PROFILE = 'original_emgformer_full_trajectory_pretest_profile_v1'
REPORT = 'original_emgformer_full_trajectory_report_v1'
POLICY = dict(prediction='exact typed prediction/target/mask/input replay',
    native_mae_atol_rad=2e-6, native_mae_rtol=2e-6,
    native_aggregation='sample-count weighted original per-batch masked L1, float32 accumulation',
    pooled_aggregation='float64 absolute-error sum / exact valid-angle count',
    scope='fixed before validation; no test-driven tolerance or checkpoint changes')
LOCAL = ('emgformer_trajectory_eval', 'emgformer_trajectory_data', 'emgformer_trajectory_results',
    'emgformer_eval', 'emgformer_train', 'emgformer_runtime', 'emgformer_cohort',
    'emgformer_literal', 'emgformer_data', 'classic_emg', 'classic_emg_cohort',
    'classic_emg_train', 'classic_emg_report', 'vision_literal_eval',
    'vision_literal_runtime', 'vision_literal_train', 'vision_literal_check',
    'wilor_native_train', 'emg_initializer_cohort', 'emg_initializer_data',
    'data', 'partitions', 'metrics', 'paper_analysis', 'emg2pose_memory_process')
ACTOR_KEYS = ('pid', 'start_ticks', 'uid', 'sid', 'pgrp')
OWNED_ENV = 'EGOEMG_TRAJECTORY_OWNED_STAGE'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


IMPORTED_SHA = _sha(__file__)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _seal(value):
    return dict(value, sha256=_hash(value))


def _sealed(value, format=None):
    if (not isinstance(value, dict) or value.get('sha256') != _hash({k: v for k, v in value.items() if k != 'sha256'})
        or (format is not None and value.get('format') != format)):
        raise ValueError('exact trajectory document format/seal required')
    return value


def _safe(path, *, exists=True):
    value = Path(path).expanduser()
    if '..' in value.parts:
        raise ValueError('trajectory parent traversal forbidden')
    value = value if value.is_absolute() else Path.cwd()/value
    if any(p.is_symlink() for p in (value, *value.parents)):
        raise ValueError('trajectory source/output symlink forbidden')
    try:
        return value.resolve(strict=exists)
    except OSError as error:
        raise ValueError('trajectory source/artifact unavailable') from error


def _nine(path):
    value = _safe(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('regular trajectory source/artifact required')
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
        value.st_mode, value.st_uid, value.st_gid, value.st_nlink]


def _sources():
    if _sha(__file__) != IMPORTED_SHA:
        raise ValueError('imported trajectory evaluator changed')
    names = ['egoemg_repro/'+n+'.py' for n in LOCAL]+['scripts/paper/run_emgformer_trajectory.py']
    return {n: _sha(_safe(ROOT/n)) for n in names}


def _write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


class _Read:
    """One fresh source/raw transaction; no persistent completed-result cache."""
    def __init__(self):
        self.files, self.cache, self.active = {}, {}, set()
        self.sources = _sources()
        for name in self.sources: self.add(ROOT/name)

    def add(self, path):
        path = _safe(path); before = _nine(path)
        if path in self.files and self.files[path] != before:
            raise ValueError('trajectory source/artifact generation changed')
        self.files[path] = before
        return path

    def check(self):
        if any(_nine(p) != s for p, s in self.files.items()):
            raise ValueError('trajectory source/artifact generation changed')

    def json(self, path):
        path = self.add(path)
        value = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        self.check()
        return value

    def once(self, key, function):
        self.check()
        if key in self.active or len(self.active) >= 16:
            raise ValueError('cyclic trajectory reference/profile graph')
        if key not in self.cache:
            self.active.add(key)
            try: self.cache[key] = function()
            finally: self.active.remove(key)
        self.check()
        return self.cache[key]


def _actor_key(actor):
    if (not isinstance(actor, dict) or any(k not in actor for k in ACTOR_KEYS)
        or any(type(actor[k]) is not int or actor[k] < 0 for k in ('pid', 'uid', 'sid', 'pgrp'))
        or actor['pid'] <= 0 or type(actor['start_ticks']) is not str or not actor['start_ticks'].isdigit()):
        raise ValueError('actual trajectory process identity required')
    return tuple(actor[k] for k in ACTOR_KEYS)


def _actor_closed(actor):
    from .emg2pose_memory_process import process_identity
    key = _actor_key(actor); current = process_identity(actor['pid'])
    if current is not None and _actor_key(current) == key:
        raise ValueError('trajectory actor has not exited')


def _owned_binding(tx):
    from .emg2pose_memory_process import process_identity, same_process
    path = os.environ.get(OWNED_ENV)
    if not path: raise ValueError('actual owned trajectory worker launch required')
    path = _safe(path); start = tx.json(path/'started.json'); actor = process_identity()
    if (_actor_key(start['worker']) != _actor_key(actor)
        or _actor_key(start['guardian']) != _actor_key(process_identity(os.getppid()))
        or not same_process(start['owner']) or not same_process(start['guardian'])):
        raise ValueError('trajectory worker/guardian/owner launch differs')
    return dict(actor=actor, owned_stage_path=str(path), owned_started_file_sha256=_sha(path/'started.json'))


def _closed_binding(value, tx):
    root = _safe(value['owned_stage_path']); started = tx.json(root/'started.json')
    result = tx.json(root/'stage-result.json')
    if (_sha(root/'started.json') != value['owned_started_file_sha256']
        or _actor_key(started['worker']) != _actor_key(value['actor'])
        or result['status'] != 'success' or result['worker_exit_code'] != 0
        or result['all_owned_consumers_exited'] is not True
        or any(_actor_key(result[k]) != _actor_key(started[k]) for k in ('owner', 'worker', 'guardian'))):
        raise ValueError('actual closed trajectory owned stage differs')
    for name in ('worker', 'guardian'): _actor_closed(result[name])
    return result


def _fit(path, tx, *, require_full=False):
    path = _safe(path)
    def read():
        from .emgformer_train import read_emgformer_fit, _loop_paths
        from .emgformer_runtime import _versions
        import torch
        identity = tx.json(path/'identity.json')
        if identity['cohort_receipt'] is not None:
            tx.add(identity['cohort_receipt']['path'])
        for runtime_path in _loop_paths().values(): tx.add(runtime_path)
        for name, digest in identity['construction']['sources_sha256'].items():
            if name.startswith('runtime/'):
                runtime_path = tx.add(importlib.import_module(name[len('runtime/'):]).__file__)
                if _sha(runtime_path) != digest: raise ValueError('selected trajectory construction runtime source differs')
        if _versions() != identity['construction']['runtime_versions']:
            raise ValueError('selected trajectory construction runtime versions differ')
        tx.add(path/'fit-result.json'); tx.add(path/'epochs.jsonl'); tx.add(path/'checkpoints/last.ckpt')
        for candidate in (path/'checkpoints').glob('*.ckpt'): tx.add(candidate)
        result = read_emgformer_fit(path, require_full=False)
        if not result['stage_complete']: raise ValueError('complete selected trajectory fit required')
        checkpoint = path/'checkpoints'/result['selected_best']['file']; tx.add(checkpoint)
        selected = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = selected['emgformer_fit']['records'][-1]['model_sha256']
        return result, identity, model
    result, identity, model = tx.once(('fit', str(path)), read)
    if require_full:
        from .emgformer_cohort import FULL_COUNTS
        if (identity['policy']['scope'] != 'full' or not result['full_training_terminated']
            or identity['cohort']['metadata_matches_registered_legacy_release'] is not True
            or identity['cohort']['counts'] != FULL_COUNTS or identity['train_samples'] != FULL_COUNTS['train']):
            raise ValueError('engineering or incomplete population is not full trajectory evaluation')
    return result, identity, model


def _inputs(request, identity, tx, *, expected=None):
    def read():
        from .emgformer_literal import compose_emgformer
        from .emgformer_trajectory_data import admit_trajectory_inputs
        value = admit_trajectory_inputs(request['data'], upstream=request['upstream'],
            config=compose_emgformer(request['upstream'], identity['policy']['size']),
            normalization=identity['cohort']['normalization'], expected_point_identity=identity['cohort'], expected=expected)
        for path in value._guard.nine: tx.add(path)
        return value
    key = ('inputs', _hash(request), identity['cohort']['sha256'])
    value = tx.once(key, read); value.check()
    if expected is not None and value.identity != expected:
        raise ValueError('actual trajectory input identity differs')
    return value


def _metadata(batch, reader, position):
    ids = batch['emgformer_sample_index'].tolist(); n = len(ids)
    def integers(name): return [int(v) for v in batch[name].tolist()]
    value = dict(index=ids, unique=integers('emgformer_unique_sample_index'), repeat=integers('emgformer_repeat_index'),
        start=integers('start'), end=integers('end'), center=integers('center'), episode=integers('episode'),
        hand=list(batch['hand']), split=list(batch['split']), user=[])
    if ids != list(range(position, position+n)) or value['unique'] != ids or any(value['repeat']):
        raise ValueError('actual trajectory batch order/repeat differs')
    for j, index in enumerate(ids):
        window = reader.windows[index % len(reader.windows)]
        subject = str(reader.store.episodes[window['episode']]['subject']); value['user'].append(subject)
        if (any(value[k][j] != window[k] for k in ('start', 'end', 'center', 'episode', 'split'))
            or value['hand'][j] != ('left' if index < len(reader.windows) else 'right')
            or int(batch['jitter_offset'][j]) != 0
            or ('episode_subject' in batch and batch['episode_subject'][j] != subject)):
            raise ValueError('actual trajectory source window/subject differs')
    return value


def _sample_digests(batch):
    from .classic_emg_train import state_digest
    return [state_digest(dict(emg=batch['emg'][j], target=batch['joint_angles'][j],
        mask=batch['label_valid_mask'][j], index=batch['emgformer_sample_index'][j])) for j in range(len(batch['emg']))]


def _content(record):
    return {k: v for k, v in record.items() if k not in ('sha256', 'file_sha256')}


def _packets(directory, value, inputs, tx, *, validate_inputs):
    import torch
    from .classic_emg_train import state_digest
    from .emgformer_trajectory_data import trajectory_loader
    from .emgformer_trajectory_results import read_trajectory_packet
    directory = _safe(directory); journal = tx.add(directory/'packets.jsonl')
    if _sha(journal) != value['packet_journal_sha256']:
        raise ValueError('trajectory packet journal changed')
    dataset = inputs.datasets[value['role']]; position = 0; digests = []
    loader = trajectory_loader(inputs, value['role'], cpu_engineering=value['training_scope'] == 'cpu_engineering') if validate_inputs else None
    with journal.open() as stream:
        for ordinal, line in enumerate(stream):
            record = json.loads(line)
            if record['ordinal'] != ordinal: raise ValueError('ordered trajectory packet journal required')
            tx.add(directory/'raw'/record['file'])
            packet = read_trajectory_packet(directory/'raw', record, upstream=inputs.point_cohort.upstream)
            n = len(packet['metadata']['index'])
            if validate_inputs:
                inputs.check()
                # Same original reader and collator, without repeating admission
                # for each sample. The registered wrapper is checked on both sides.
                batch = loader.collate_fn([dataset.reader[i] for i in range(position, position+n)])
                if (packet['metadata'] != _metadata(batch, dataset.reader, position)
                    or packet['input_sha256'] != state_digest(batch)
                    or packet['sample_input_sha256'] != _sample_digests(batch)
                    or not torch.equal(packet['arrays']['source_target'], batch['joint_angles'])
                    or not torch.equal(packet['arrays']['source_mask'], batch['label_valid_mask'])):
                    raise ValueError('raw trajectory differs from actual source inputs/targets/masks')
                inputs.check()
            position += n; digests.append(_hash(_content(record))); tx.check()
            yield packet
    if position != value['samples'] or _hash(digests) != value['packet_content_sha256']:
        raise ValueError('complete typed trajectory raw content differs')


def _ensure_fresh(first, second):
    if (_safe(first['directory']) == _safe(second['directory'])
        or _actor_key(first['actor']) == _actor_key(second['actor'])):
        raise ValueError('fresh replay requires distinct directory and actor')
    for value in (first, second): _actor_closed(value['actor'])
    if (first['kind'] != 'native' or second['kind'] != 'replay' or first['role'] != second['role']
        or second['reference_path'] != first['directory'] or second['reference_sha256'] != first['sha256']
        or second['reference_file_sha256'] != _sha(Path(first['directory'])/'pass.json')
        or first['finished_ns'] > second['started_ns']):
        raise ValueError('fresh replay reference or chronology differs')
    for name in ('fit_result_sha256', 'trajectory_input_identity', 'selected_best', 'training_scope', 'require_full',
                 'profile_sha256', 'profile_file_sha256', 'packet_content_sha256', 'statistics', 'hardware',
                 'batch_size', 'workers', 'precision', 'model_before_sha256', 'model_after_sha256'):
        if first[name] != second[name]: raise ValueError('fresh replay reference '+name+' differs')


def _pass(directory, tx):
    directory = _safe(directory)
    def read():
        from .emgformer_trajectory_results import trajectory_statistics
        from .vision_literal_eval import _close_native, POLICY as shared_policy
        value = _sealed(tx.json(directory/'pass.json'), FORMAT)
        if value['directory'] != str(directory) or value['sources_sha256'] != tx.sources or value['fixed_agreement_policy'] != POLICY or POLICY != shared_policy:
            raise ValueError('trajectory source/policy/directory differs')
        if value['scientific_eligible'] is not False or value['full_paper_complete'] is not False:
            raise ValueError('private trajectory pass cannot claim scientific completion')
        if (value['role'] not in ('val', 'test') or value['kind'] not in ('native', 'replay')
            or type(value['require_full']) is not bool or type(value['started_ns']) is not int
            or type(value['finished_ns']) is not int or not 0 < value['started_ns'] <= value['finished_ns']):
            raise ValueError('explicit trajectory role/kind/scope/chronology required')
        _closed_binding(value, tx)
        fit, identity, model = _fit(value['fit_path'], tx, require_full=value['require_full'])
        inputs = _inputs(value['request'], identity, tx, expected=value['trajectory_input_identity'])
        policy = identity['policy']; formal = value['require_full'] and fit['full_training_terminated']
        if (value['fit_result_sha256'] != fit['sha256'] or value['fit_identity_sha256'] != identity['sha256']
            or value['selected_best'] != fit['selected_best'] or value['training_scope'] != fit['scope']
            or value['samples'] != len(inputs.datasets[value['role']])
            or value['batch_size'] != policy['batch_size_by_role']['val'] or value['workers'] != policy['workers']
            or value['precision'] != policy['precision'] or value['formal_evaluation'] is not formal
            or value['model_before_sha256'] != model or value['model_after_sha256'] != model
            or value['actual_trained_model_constructions'] != 1 or value['actual_optimizer_updates'] != 0
            or value['actual_model_forwards'] != math.ceil(value['samples']/value['batch_size'])
            or value['all_loader_workers_exited'] is not True or len(value['loader_workers']) != value['workers']):
            raise ValueError('trajectory fit/model state/formal scope or execution differs')
        for actor in value['loader_workers']: _actor_closed(actor)
        statistics = trajectory_statistics(_packets(directory, value, inputs, tx, validate_inputs=True),
            samples=value['samples'], batch_size=value['batch_size'], role=value['role'], upstream=inputs.point_cohort.upstream)
        if statistics != value['statistics']: raise ValueError('actual raw trajectory statistics differ')
        if value['role'] == 'test':
            if value['profile_path'] is None: raise ValueError('trajectory validation profile required before test')
            frozen = _profile(value['profile_path'], tx)
            if (frozen['sha256'] != value['profile_sha256'] or frozen['fit_result_sha256'] != fit['sha256']
                or frozen['trajectory_input_sha256'] != inputs.identity['sha256']
                or _sha(Path(value['profile_path'])/'profile.json') != value['profile_file_sha256']
                or frozen['finished_ns'] > value['started_ns']):
                raise ValueError('trajectory pretest profile/fit/source chronology differs')
        elif any(value[n] is not None for n in ('profile_path', 'profile_sha256', 'profile_file_sha256')):
            raise ValueError('trajectory validation cannot consume test profile')
        if value['kind'] == 'native':
            if any(value[n] is not None for n in ('reference_path', 'reference_sha256', 'reference_file_sha256')):
                raise ValueError('native trajectory cannot consume replay reference')
            metrics = tx.json(directory/'native-metrics.json')
            if metrics != value['native_metrics'] or set(metrics) != {value['role']+'_mae'}:
                raise ValueError('actual original trajectory metrics differ')
            _close_native(statistics, metrics, value['role'])
        else:
            if value['native_metrics'] != {}: raise ValueError('manual trajectory replay cannot invent native metrics')
            _ensure_fresh(_pass(value['reference_path'], tx), value)
        return value
    return tx.once(('pass', str(directory)), read)


def read_trajectory_pass(directory):
    tx = _Read(); value = _pass(directory, tx); tx.check(); return copy.deepcopy(value)


def _profile(directory, tx):
    directory = _safe(directory)
    def read():
        value = _sealed(tx.json(directory/'profile.json'), PROFILE)
        if (value['sources_sha256'] != tx.sources or value['fixed_agreement_policy'] != POLICY
            or value['scientific_eligible'] is not False or value['full_paper_complete'] is not False
            or value['prior_point_test_exposure_disclosed'] is not True):
            raise ValueError('private trajectory validation profile/source differs')
        _closed_binding(value, tx)
        first, second = _pass(value['reference_path'], tx), _pass(value['replay_path'], tx)
        _ensure_fresh(first, second)
        if (first['role'] != 'val' or value['reference_sha256'] != first['sha256'] or value['replay_sha256'] != second['sha256']
            or value['fit_result_sha256'] != first['fit_result_sha256']
            or value['trajectory_input_sha256'] != first['trajectory_input_identity']['sha256']
            or value['finished_ns'] < max(first['finished_ns'], second['finished_ns'])
            or value['reference_file_sha256'] != _sha(Path(first['directory'])/'pass.json')
            or value['replay_file_sha256'] != _sha(Path(second['directory'])/'pass.json')):
            raise ValueError('trajectory validation profile reference differs')
        return value
    return tx.once(('profile', str(directory)), read)


def read_trajectory_profile(directory):
    tx = _Read(); value = _profile(directory, tx); tx.check(); return copy.deepcopy(value)


def freeze_trajectory_validation(first, second, output):
    tx = _Read(); a, b = _pass(first, tx), _pass(second, tx); _ensure_fresh(a, b)
    if a['role'] != 'val': raise ValueError('full trajectory validation pair required')
    binding = _owned_binding(tx); output = _safe(output, exists=False); output.mkdir(mode=0o700)
    value = _seal(dict(format=PROFILE, reference_path=a['directory'], replay_path=b['directory'],
        reference_sha256=a['sha256'], replay_sha256=b['sha256'],
        reference_file_sha256=_sha(Path(first)/'pass.json'), replay_file_sha256=_sha(Path(second)/'pass.json'),
        fit_result_sha256=a['fit_result_sha256'], trajectory_input_sha256=a['trajectory_input_identity']['sha256'],
        sources_sha256=tx.sources, fixed_agreement_policy=copy.deepcopy(POLICY), finished_ns=time.time_ns(),
        prior_point_test_exposure_disclosed=True,
        selection='existing original center-val_mae best checkpoint; trajectory validation does not reselect weights',
        scientific_eligible=False, full_paper_complete=False, **binding))
    tx.check(); _write(output/'profile.json', value); return value


def _close_loader(loader, *, snapshot=None):
    from .emg2pose_memory_process import process_identity
    if snapshot is None:
        iterator = getattr(loader, '_iterator', None)
        workers = [] if iterator is None else list(getattr(iterator, '_workers', []))
        owned = [(process, process_identity(process.pid)) for process in workers]
    else:
        iterator, owned = snapshot
        workers = [process for process, _ in owned]
    handles = [handle for _, handle in owned]
    if any(h is None for h in handles): raise ValueError('owned trajectory loader worker disappeared before closure')
    if iterator is not None: iterator._shutdown_workers()
    for process in workers:
        process.join(timeout=15)
        if process.is_alive(): raise ValueError('owned trajectory loader worker failed to exit')
    for handle in handles: _actor_closed(handle)
    return handles


def run_trajectory_pass(inputs, fit, output, *, role, kind, reference=None, profile=None, require_full=True):
    if role not in ('val', 'test') or kind not in ('native', 'replay') or type(require_full) is not bool:
        raise ValueError('explicit trajectory role/kind/scope required')
    from .emgformer_trajectory_data import TrajectoryInputs, trajectory_loader
    if type(inputs) is not TrajectoryInputs: raise ValueError('actual registered trajectory inputs required')
    inputs.check(); output = _safe(output, exists=False)
    if output.exists(): raise ValueError('fresh trajectory output directory required')
    started = time.time_ns(); tx = _Read(); fit_path = _safe(fit)
    result, identity, selected_model = _fit(fit_path, tx, require_full=require_full)
    if identity['cohort'] != inputs.point_cohort.identity: raise ValueError('trajectory source differs from actual selected fit')
    request = dict(data=str(_safe(inputs.point_cohort.datasets['train'].store.root)), upstream=str(inputs.point_cohort.upstream))
    tx.cache[('inputs', _hash(request), identity['cohort']['sha256'])] = inputs
    for path in inputs._guard.nine: tx.add(path)
    frozen = ref = None
    if role == 'test':
        if profile is None: raise ValueError('freeze trajectory validation profile before any test')
        frozen = _profile(profile, tx)
        if (frozen['fit_result_sha256'] != result['sha256'] or frozen['trajectory_input_sha256'] != inputs.identity['sha256']
            or frozen['finished_ns'] > started): raise ValueError('trajectory validation profile/fit differs')
    elif profile is not None: raise ValueError('trajectory validation cannot consume test profile')
    if kind == 'replay':
        if reference is None: raise ValueError('fresh trajectory replay requires native reference')
        ref = _pass(reference, tx)
        if (ref['role'] != role or ref['kind'] != 'native' or ref['fit_result_sha256'] != result['sha256']
            or ref['trajectory_input_identity'] != inputs.identity or ref['require_full'] is not require_full
            or ref['profile_sha256'] != (None if frozen is None else frozen['sha256']) or ref['finished_ns'] > started):
            raise ValueError('fresh trajectory replay reference/fit/scope/profile differs')
    elif reference is not None: raise ValueError('native trajectory cannot consume replay reference')
    binding = _owned_binding(tx)
    import torch
    from . import emgformer_train as train
    from .emgformer_runtime import build_emgformer_runtime, emgformer_state_sha256
    from .emgformer_trajectory_results import capture_native_trajectory, write_trajectory_packet, trajectory_statistics
    from .vision_literal_eval import _close_native
    policy = identity['policy']; device = train._runtime_device(policy)
    hardware = {k: v for k, v in device.items() if k != 'free_memory_before'}
    checkpoint = fit_path/'checkpoints'/result['selected_best']['file']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    prepared = build_emgformer_runtime(inputs.point_cohort.upstream, inputs.point_cohort.config,
        state=state, expected_initialization=identity['construction'])
    before = emgformer_state_sha256(prepared.module.state_dict())
    if before != selected_model: raise ValueError('actual selected trajectory model state differs')
    output = _safe(output, exists=False); train.storage._space(output.parent); output.mkdir(mode=0o700)
    (output/'raw').mkdir(mode=0o700)
    loader = trajectory_loader(inputs, role, cpu_engineering=policy['scope'] == 'cpu_engineering')
    if (loader.batch_size != policy['batch_size_by_role']['val'] or loader.num_workers != policy['workers']
        or loader.drop_last or loader.pin_memory or loader.persistent_workers is not policy['persistent_workers']
        or loader.prefetch_factor != policy['prefetch_factor']):
        raise ValueError('original full trajectory loader settings differ')
    original = prepared.module.forward; position = 0; digests = []; native = {}; loader_workers = []; loader_snapshot = None
    with (output/'packets.jsonl').open('x') as journal:
        def observe(batch):
            nonlocal position, loader_snapshot
            inputs.check(); tx.check()
            if loader_snapshot is None:
                from .emg2pose_memory_process import process_identity
                iterator = getattr(loader, '_iterator', None)
                workers = [] if iterator is None else list(getattr(iterator, '_workers', []))
                loader_snapshot = (iterator, [(process, process_identity(process.pid)) for process in workers])
                if len(workers) != loader.num_workers or any(h is None for _, h in loader_snapshot[1]):
                    raise ValueError('actual original trajectory loader worker identities required')
            metadata = _metadata(batch, inputs.datasets[role].reader, position)
            if len(batch['emg']) != min(loader.batch_size, len(inputs.datasets[role])-position):
                raise ValueError('complete original trajectory batch required')
            observed = []
            def call(arguments):
                value = original(arguments); observed.append(value); return value
            prepared.module.forward = call
            try: capture = capture_native_trajectory(prepared.module, batch, upstream=inputs.point_cohort.upstream)
            finally: prepared.module.forward = observe
            if len(observed) != 1: raise ValueError('exact one original learned trajectory forward required')
            record = write_trajectory_packet(output/'raw', len(digests), arrays=capture['arrays'], alignment=capture['alignment'],
                metadata=metadata, input_sha256=capture['input_sha256'], sample_input_sha256=_sample_digests(batch),
                upstream=inputs.point_cohort.upstream)
            journal.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n'); journal.flush(); os.fsync(journal.fileno())
            digests.append(_hash(_content(record))); position += len(batch['emg'])
            _write_progress = dict(consumed_samples=position, expected_samples=len(inputs.datasets[role]), actual_model_forwards=len(digests), role=role, kind=kind)
            pending = output/'progress.pending.json'; _write(pending, _write_progress); os.replace(pending, output/'progress.json')
            tx.check(); return observed[0]
        prepared.module.forward = observe
        try:
            if kind == 'native':
                trainer = train.storage.pl.Trainer(accelerator=device['accelerator'], devices=1, precision=policy['precision'],
                    logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
                method = trainer.validate if role == 'val' else trainer.test
                values = method(prepared.module, dataloaders=loader, verbose=False)
                native = {role+'_mae': float(values[0][role+'_mae'])}
            else:
                from lightning_fabric.utilities.apply_func import move_data_to_device
                target_device = 'cpu' if device['accelerator'] == 'cpu' else 'cuda:0'
                prepared.module.to(target_device).eval()
                with torch.inference_mode():
                    for batch in loader:
                        batch = move_data_to_device(batch, target_device)
                        with (nullcontext() if target_device == 'cpu' else torch.autocast('cuda', dtype=torch.bfloat16)):
                            prepared.module.forward(batch)
            if position != len(inputs.datasets[role]): raise ValueError('incomplete full trajectory population')
            inputs.check(); tx.check(); prepared.check()
            after = emgformer_state_sha256(prepared.module.state_dict())
            if after != before: raise ValueError('trajectory model state changed during inference')
        finally:
            prepared.module.forward = original
            loader_workers = _close_loader(loader, snapshot=loader_snapshot)
    value = dict(format=FORMAT, directory=str(output), role=role, kind=kind, request=request,
        fit_path=str(fit_path), fit_result_sha256=result['sha256'], fit_identity_sha256=identity['sha256'],
        selected_best=copy.deepcopy(result['selected_best']), trajectory_input_identity=copy.deepcopy(inputs.identity),
        training_scope=result['scope'], require_full=require_full, formal_evaluation=require_full and result['full_training_terminated'],
        samples=position, batch_size=loader.batch_size, workers=loader.num_workers, precision=policy['precision'], hardware=hardware,
        sources_sha256=tx.sources, fixed_agreement_policy=copy.deepcopy(POLICY),
        packet_journal_sha256=_sha(output/'packets.jsonl'), packet_content_sha256=_hash(digests),
        model_before_sha256=before, model_after_sha256=after, actual_trained_model_constructions=1,
        actual_model_forwards=len(digests), actual_optimizer_updates=0, loader_workers=loader_workers, all_loader_workers_exited=True,
        native_metrics=native, started_ns=started, finished_ns=time.time_ns(),
        profile_path=None if frozen is None else str(_safe(profile)), profile_sha256=None if frozen is None else frozen['sha256'],
        profile_file_sha256=None if frozen is None else _sha(Path(profile)/'profile.json'),
        reference_path=None if ref is None else ref['directory'], reference_sha256=None if ref is None else ref['sha256'],
        reference_file_sha256=None if ref is None else _sha(Path(reference)/'pass.json'),
        scientific_eligible=False, full_paper_complete=False, **binding)
    value['statistics'] = trajectory_statistics(_packets(output, value, inputs, tx, validate_inputs=False),
        samples=position, batch_size=loader.batch_size, role=role, upstream=inputs.point_cohort.upstream)
    if kind == 'native': _close_native(value['statistics'], native, role); _write(output/'native-metrics.json', native)
    elif value['packet_content_sha256'] != ref['packet_content_sha256'] or value['statistics'] != ref['statistics'] or hardware != ref['hardware']:
        raise ValueError('actual fresh trained trajectory replay differs')
    tx.check(); inputs.check(); value = _seal(value); _write(output/'pass.json', value)
    return value


def _report_core(test_replay, tx):
    replay = _pass(test_replay, tx)
    if replay['role'] != 'test' or replay['kind'] != 'replay': raise ValueError('complete fresh trajectory test replay required')
    native = _pass(replay['reference_path'], tx); fit, identity, _ = _fit(replay['fit_path'], tx)
    _ensure_fresh(native, replay)
    from .emgformer_eval import _admitted_population
    inputs = _inputs(replay['request'], identity, tx, expected=replay['trajectory_input_identity'])
    population = _admitted_population(replay['request']['data'], inputs.point_cohort.partitions, identity['cohort'])
    statistics = copy.deepcopy(replay['statistics'])
    users = sorted(statistics['trajectory']['by_user']); aliases = {u: 'user_'+str(i).zfill(3) for i, u in enumerate(users)}
    for name in ('trajectory', 'aligned_center'):
        statistics[name]['by_user'] = {aliases[u]: v for u, v in statistics[name]['by_user'].items()}
    eligible = all(v['formal_evaluation'] for v in (native, replay))
    frozen = _profile(replay['profile_path'], tx)
    eligible = eligible and all(_pass(frozen[n], tx)['formal_evaluation'] for n in ('reference_path', 'replay_path'))
    core = dict(format=REPORT, size=identity['policy']['size'], test=statistics, population=population['public'],
        native_test_metrics=native['native_metrics'], training=dict(scope=fit['scope'], full_training_terminated=fit['full_training_terminated'],
            completed_epochs=fit['completed_epochs'], optimizer_executions=fit['optimizer_executions'], selected_best=fit['selected_best'], policy=identity['policy']),
        source_identity_sha256=inputs.identity['sha256'], fit_result_sha256=fit['sha256'], validation_profile_sha256=frozen['sha256'],
        native_test_sha256=native['sha256'], independent_test_sha256=replay['sha256'], packet_content_sha256=replay['packet_content_sha256'],
        sources_sha256=tx.sources, fixed_agreement_policy=copy.deepcopy(POLICY),
        prior_point_test_exposure_disclosed=True, paper_numerical_equivalence_established=False,
        available_corpus='released legacy EgoEMG; original full-window held-out grid; independent train-only validation',
        center_definition='center of context-aligned full trajectory; not prior native point-reader center',
        limitations=identity['cohort']['preserved_filter_and_normalization_limitations'],
        full_paper_complete=False)
    return core, bool(eligible)


def _render(core):
    lines = ['# Native EMGFormer full trajectories', '',
        'Acceptance is recorded separately in completion.json.', '',
        '| Reduction | MAE (degrees) | Valid angles |', '|---|---:|---:|']
    for name in ('trajectory', 'aligned_center'):
        item = core['test'][name]['overall']; value = item['mae_deg']
        score = '—' if value is None else f'{value:.9f}'
        lines.append(f"| {name} | {score} | {item['valid_angle_count']} |")
    lines.extend(['', f"Original test batch MAE (radians): {core['native_test_metrics']['test_mae']:.12g}.", '',
        core['center_definition']+'.', '', 'Per-joint, hand, split and anonymized user statistics are in metrics.json.', ''])
    return '\n'.join(lines)


def publish_trajectory_report(test_replay, output):
    tx = _Read(); core, _ = _report_core(test_replay, tx); binding = _owned_binding(tx)
    output = _safe(output, exists=False); output.mkdir(mode=0o700)
    value = _seal(dict(**core, core_sha256=_hash(core), trajectory_pipeline_complete=False,
        scientific_eligible=False, published_ns=time.time_ns(), **binding))
    tx.check(); _write(output/'metrics.pending.json', value); _write(output/'metrics.json', value)
    with (output/'metrics.md').open('x') as stream: stream.write(_render(core))
    return value


def _completion(directory, pending, core, eligible, tx):
    completion = _sealed(tx.json(directory/'completion.json'))
    receipt_path = _safe(completion['verification_path']); receipt = _sealed(tx.json(receipt_path))
    actual = _closed_binding(receipt, tx)
    if (completion['format'] != 'closed_owned_native_trajectory_completion_v1'
        or completion['actual_owned_verification_stage'] != actual
        or completion['verification_file_sha256'] != _sha(receipt_path)
        or receipt['pending_file_sha256'] != _sha(directory/'metrics.pending.json')
        or receipt['report_core_sha256'] != _hash(core) or receipt['markdown_sha256'] != _sha(directory/'metrics.md')
        or receipt['eligible_after_closed_verification'] is not eligible or receipt['all_raw_graph_verified'] is not True
        or _actor_key(receipt['actor']) == _actor_key(pending['actor'])
        or completion['trajectory_pipeline_complete'] is not True or completion['scientific_eligible'] is not eligible
        or completion['full_paper_complete'] is not False):
        raise ValueError('actual closed trajectory verification completion differs')
    return completion


def verify_trajectory_report(test_replay, output, *, receipt=None):
    tx = _Read(); core, eligible = _report_core(test_replay, tx); output = _safe(output)
    pending = _sealed(tx.json(output/'metrics.pending.json'), REPORT); _closed_binding(pending, tx)
    current = _sealed(tx.json(output/'metrics.json'), REPORT); tx.add(output/'metrics.md')
    reserved = {'sha256', 'core_sha256', 'trajectory_pipeline_complete', 'scientific_eligible',
        'published_ns', 'actor', 'owned_stage_path', 'owned_started_file_sha256', 'completion_file_sha256'}
    if ({k: v for k, v in pending.items() if k not in reserved} != core or pending['core_sha256'] != _hash(core)
        or pending['trajectory_pipeline_complete'] is not False or pending['scientific_eligible'] is not False
        or (output/'metrics.md').read_text() != _render(core)):
        raise ValueError('pending trajectory report/core/render differs from actual raw graph')
    if (output/'completion.json').exists():
        _completion(output, pending, core, eligible, tx)
        promoted = {k: v for k, v in pending.items() if k != 'sha256'}
        promoted.update(trajectory_pipeline_complete=True, scientific_eligible=eligible,
            completion_file_sha256=_sha(output/'completion.json'))
        if current != _seal(promoted): raise ValueError('promoted trajectory report differs')
    elif current != pending:
        raise ValueError('trajectory report marked complete before closed independent verification')
    if receipt is not None:
        if (output/'completion.json').exists(): raise ValueError('independent receipt requires pending trajectory report')
        binding = _owned_binding(tx)
        if _actor_key(binding['actor']) == _actor_key(pending['actor']): raise ValueError('fresh independent trajectory report actor required')
        destination = _safe(receipt, exists=False); destination.parent.mkdir(mode=0o700, exist_ok=True)
        tx.check()
        value = _seal(dict(format='independent_raw_trajectory_verification_v1',
            report_directory=str(output), pending_file_sha256=_sha(output/'metrics.pending.json'),
            report_core_sha256=_hash(core), markdown_sha256=_sha(output/'metrics.md'),
            all_raw_graph_verified=True, eligible_after_closed_verification=eligible,
            graph_generations={str(p): s for p, s in tx.files.items() if p != output/'metrics.json'},
            verified_ns=time.time_ns(), scientific_eligible=False, full_paper_complete=False, **binding))
        _write(destination, value)
    tx.check(); return current


def promote_trajectory_report(report, verification_file, *, owned_stage):
    from .emg2pose_memory_process import process_identity
    if (not isinstance(owned_stage, dict) or owned_stage.get('status') != 'success'
        or owned_stage.get('worker_exit_code') != 0 or owned_stage.get('all_owned_consumers_exited') is not True):
        raise ValueError('successful closed trajectory verification stage required')
    for name in ('worker', 'guardian'): _actor_closed(owned_stage[name])
    if _actor_key(owned_stage['owner']) != _actor_key(process_identity()):
        raise ValueError('actual trajectory controller alone may promote report')
    tx = _Read(); report = _safe(report); verification_file = _safe(verification_file)
    receipt = _sealed(tx.json(verification_file)); actual = _closed_binding(receipt, tx)
    pending = _sealed(tx.json(report/'metrics.pending.json'), REPORT); _closed_binding(pending, tx)
    if (actual != owned_stage or receipt['report_directory'] != str(report)
        or receipt['format'] != 'independent_raw_trajectory_verification_v1'
        or _actor_key(receipt['actor']) == _actor_key(pending['actor'])
        or receipt['pending_file_sha256'] != _sha(report/'metrics.pending.json')
        or receipt['report_core_sha256'] != pending['core_sha256']
        or receipt['markdown_sha256'] != _sha(report/'metrics.md') or receipt['all_raw_graph_verified'] is not True
        or receipt['scientific_eligible'] is not False or receipt['full_paper_complete'] is not False
        or pending['trajectory_pipeline_complete'] is not False or pending['scientific_eligible'] is not False
        or tx.json(report/'metrics.json') != pending):
        raise ValueError('actual pending report and fresh closed verification differ')
    for name, stamp in receipt['graph_generations'].items():
        if _nine(name) != stamp: raise ValueError('verified trajectory graph generation changed before promotion')
        tx.add(name)
    eligible = receipt['eligible_after_closed_verification']
    if type(eligible) is not bool: raise ValueError('explicit trajectory eligibility required')
    value = _seal(dict(format='closed_owned_native_trajectory_completion_v1', verification_path=str(verification_file),
        verification_file_sha256=_sha(verification_file), actual_owned_verification_stage=owned_stage,
        trajectory_pipeline_complete=True, scientific_eligible=eligible, full_paper_complete=False))
    tx.check(); _write(report/'completion.json', value)
    promoted = {k: v for k, v in pending.items() if k != 'sha256'}
    promoted.update(trajectory_pipeline_complete=True, scientific_eligible=eligible, completion_file_sha256=_sha(report/'completion.json'))
    promoted = _seal(promoted); _write(report/'metrics.verified.pending.json', promoted)
    os.replace(report/'metrics.verified.pending.json', report/'metrics.json'); return promoted


def run_trajectory_workflow(request, output):
    from .emg2pose_memory_process import process_identity, run_owned_stage, same_process
    if (not isinstance(request, dict) or set(request) != {'fit', 'data', 'upstream', 'engineering'}
        or type(request['engineering']) is not bool): raise ValueError('explicit trajectory workflow request required')
    tx = _Read(); parent = process_identity(os.getppid())
    root = _safe(output, exists=False); root.mkdir(mode=0o700); _write(root/'request.json', request); tx.add(root/'request.json')
    def check():
        tx.check()
        if 'torch' in sys.modules or not same_process(parent):
            raise ValueError('trajectory controller must stay Torch-free with actual parent alive')
    cli = str(ROOT/'scripts/paper/run_emgformer_trajectory.py')
    common = [a for n in ('fit', 'data', 'upstream') for a in ('--'+n, str(request[n]))]
    if request['engineering']: common.append('--engineering')
    stages = []
    for role in ('val', 'test'):
        for kind in ('native', 'replay'):
            args = ['pass', *common, '--role', role, '--kind', kind, '--output', str(root/(role+'-'+kind))]
            if kind == 'replay': args.extend(['--reference', str(root/(role+'-native'))])
            if role == 'test': args.extend(['--profile', str(root/'profile')])
            stages.append((role+'-'+kind, args))
        if role == 'val': stages.append(('freeze', ['freeze', '--first', str(root/'val-native'), '--second', str(root/'val-replay'), '--output', str(root/'profile')]))
    stages.extend([('report', ['report', '--test-replay', str(root/'test-replay'), '--output', str(root/'report')]),
        ('verify', ['verify', '--test-replay', str(root/'test-replay'), '--report', str(root/'report'), '--output', str(root/'verified')])])
    owned = []
    for name, args in stages:
        stage = root/('owned-'+name); environment = dict(os.environ, **{OWNED_ENV: str(stage)})
        value = run_owned_stage([sys.executable, '-B', cli, *args], stage, descriptors={}, guard=check, environment=environment)
        for key in ('worker', 'guardian'): _actor_closed(value[key])
        owned.append(value)
    promoted = promote_trajectory_report(root/'report', root/'verified/verification.json', owned_stage=owned[-1]); check()
    value = _seal(dict(format='closed_native_trajectory_workflow_v1', actual_owned_stages=owned,
        all_owned_consumers_exited=True, parent_torch_unimported='torch' not in sys.modules,
        trajectory_pipeline_complete=True, scientific_eligible=promoted['scientific_eligible'], full_paper_complete=False,
        report_file_sha256=_sha(root/'report/metrics.json'), sources_sha256=tx.sources, finished_ns=time.time_ns()))
    _write(root/'workflow-result.json', value); return value
