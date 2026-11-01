"""Complete legacy ViT-S dense prediction, fresh replay and source analysis.

Pinned completed journals bind the complete input selection and existing GT.
Only the actual registered trained triplet can predict. Raw packet utilities
also serve explicit engineering tests and never certify trained provenance.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
import weakref

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe, _StatGuard, _json
from .occlusion_dense_data import _typed_sha
from .occlusion_dense_inference import native_batch_ranges, source_collate_fusion, predict_dense_batch
from .occlusion_dense_predictions import KEY_FIELDS, _key, join_dense_predictions
from .occlusion_dense_analysis import source_analysis_contract, analyze_source_csv

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PROOFS = {
    'full-index-admission-20261101.json': 'b57de34dcf8c18fa94cb6390c1a92cf90c619f7b1e95dfc77ce3f78ca0417318',
    'paired-full-io-complete-20261101.json': '387735569301d6714b205c381c82ca1b77d3007b1e745dc0bc5dc91cfe4f1456',
    'vits-geometry-complete-20261101.json': 'acc901cf9d23b1b4b9e03ceb0d1b799d1af4b282717df95ad09db8abf7608228',
    'producer-footprint-audit-20261101.json': '646eb10961eae8a6178c5860c3cdc6b828e1122c23112913995fe49cb0cd5baa',
}
MODEL_ID = '7543495ccd161d11bc2db25f277a7dfb1fddd4fd0051b9f1d0531e5faeea41fd'
LOCAL = ('occlusion_dense_run', 'occlusion_dense_models', 'occlusion_dense_inputs',
    'occlusion_dense_inference', 'occlusion_dense_predictions', 'occlusion_dense_analysis')
_EVIDENCE, _INPUTS = weakref.WeakKeyDictionary(), weakref.WeakKeyDictionary()


def _sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


IMPORTED_SHA = _sha(__file__)


def _guard():
    paths = [ROOT/'egoemg_repro'/(n+'.py') for n in LOCAL]
    guard = _StatGuard(paths)
    if _sha(__file__) != IMPORTED_SHA:
        raise ValueError('imported dense run source changed')
    return guard, {str(p.relative_to(ROOT)): _sha(p) for p in paths}


def _write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def _seal(value):
    return dict(value, sha256=_hash(value))


def _sealed(value):
    if not isinstance(value, dict) or value.get('sha256') != _hash({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('exact sealed dense artifact required')
    return value


def _closed_workflow(root, expected, guard):
    from .emg2pose_memory_process import process_identity
    root = _safe(root)
    path, state_path = root/'private/workflow-result.json', root/'private/execution-state.json'
    guard.add(path); guard.add(state_path)
    if _sha(path) != expected:
        raise ValueError('accepted completed source workflow bytes differ')
    value, state = _json(path), _json(state_path)
    if state.get('status') != 'complete' or state.get('actual_exit_code') != 0:
        raise ValueError('source workflow must have actually terminated successfully')
    handles = [state[n] for n in ('wrapper', 'child')]
    for stage in value['actual_owned_stages']:
        if stage['status'] != 'success' or not stage['all_owned_consumers_exited']:
            raise ValueError('complete source stage and consumer closure required')
        handles.extend(stage[n] for n in ('worker', 'guardian', 'owner'))
    # Recorder handles have only pid/start_ticks. Do not pass those to
    # same_process(), whose stricter schema would incorrectly report them dead.
    for handle in handles:
        if type(handle.get('pid')) is not int or not str(handle.get('start_ticks', '')).isdigit():
            raise ValueError('exact completed source process handle required')
        now = process_identity(handle['pid'])
        if now is not None and now['start_ticks'] == str(handle['start_ticks']):
            raise ValueError('original source owner is still present')
    guard.check()
    return value


@dataclass(eq=False)
class DenseEvidence:
    identity: dict
    records: tuple
    geometry_records: tuple
    upstream: Path
    _guard: object

    def check(self):
        saved = _EVIDENCE.get(self)
        if saved is None:
            raise ValueError('registered completed dense evidence required')
        self._guard.check()
        if (self._guard is not saved['guard'] or self.records is not saved['records']
            or self.geometry_records is not saved['geometry'] or self.upstream != saved['upstream']
            or _sealed(self.identity)['sha256'] != saved['identity']):
            raise ValueError('registered dense evidence identity or immutable records changed')

    def keys(self):
        self.check()
        return [json.loads(s)['key'] for s in self.records]

    def geometry(self):
        self.check()
        return [json.loads(s) for s in self.geometry_records]


def load_dense_evidence(*, evidence, upstream, index_root, paired_root, geometry_root):
    """Authenticate small completed source artifacts; no bulk/model permit."""
    guard, sources = _guard()
    directory = _safe(evidence)
    proof = {}
    for name, digest in PUBLIC_PROOFS.items():
        path = _safe(directory/name); guard.add(path)
        if _sha(path) != digest:
            raise ValueError('immutable accepted dense evidence differs: ' + name)
        proof[name] = _json(path)
    contract = source_analysis_contract(upstream)
    upstream = Path(upstream).resolve()
    for name in contract['source_files_sha256']:
        guard.add(upstream/name)
    ix = _closed_workflow(index_root, proof['full-index-admission-20261101.json']['raw_workflow_file_sha256'], guard)
    io = proof['paired-full-io-complete-20261101.json']
    _closed_workflow(paired_root, io['raw_workflow_file_sha256'], guard)
    gt = _closed_workflow(geometry_root, proof['vits-geometry-complete-20261101.json']['raw_workflow_file_sha256'], guard)
    journal = _safe(Path(paired_root)/'private/paired-a/vits_s-inputs.jsonl')
    geometry = _safe(Path(geometry_root)/'private/native-a/points.jsonl')
    guard.add(journal); guard.add(geometry)
    expected_gt = next(row for row in gt['outputs'] if row['label'] == 'native-a')
    if (_sha(journal) != io['files_sha256']['private/paired-a/vits_s-inputs.jsonl']
        or _sha(geometry) != expected_gt['journal_file_sha256']):
        raise ValueError('complete paired inputs or original GT journal changed')
    raw = [_json_line(s) for s in journal.read_text().splitlines()]
    rows = [_json_line(s) for s in geometry.read_text().splitlines()]
    keys = [row['key'] for row in raw]
    batches = native_batch_ranges(keys)
    if len(keys) != 17168 or len(rows) != len(keys) or len(batches) != 272:
        raise ValueError('complete original ViTS dense population required')
    for i, (k, g, r) in enumerate(zip(keys, rows, raw)):
        if (_key(k, ('sequence', 'hand', 'ep_idx', 'center_idx'))
            != _key(g, ('sequence', 'hand', 'ep_idx', 'center_idx')) or k['sequence'] != i
            or len(r['typed_input_sha256']) != 11
            or g['status'] not in ('native_geometry', 'invalid_zero_pose')):
            raise ValueError('original full input and GT ordered keys differ')
    false = _typed_sha(np.array([False], bool))
    counts = dict(points=len(keys), zero_geometry=sum(g['status'] == 'invalid_zero_pose' for g in rows),
        crop_missing=sum(r['typed_input_sha256']['fusion:vision_valid_mask'] == false for r in raw),
        label_invalid=sum(r['typed_input_sha256']['fusion:label_valid_mask'] == false for r in raw))
    if counts != dict(points=17168, zero_geometry=518, crop_missing=695, label_invalid=518):
        raise ValueError('all original missing/zero/invalid requests must be retained')
    footprint = proof['producer-footprint-audit-20261101.json']['audit']['recipes']['vits_s']
    identity = _seal(dict(format='authenticated_completed_vits_dense_evidence_v1', recipe='vits_s',
        counts=counts, public_proofs_sha256=copy.deepcopy(PUBLIC_PROOFS),
        index_identity=ix['outputs']['vits_s']['identity'],
        paired_input_identity=io['recipes']['vits_s']['input_identity'],
        complete_payload_sha256=io['complete_raw_input_files_sha256'],
        complete_io_journal_file_sha256=_sha(journal), geometry_journal_file_sha256=_sha(geometry),
        source_keys_sha256=_hash(keys), batch_sizes=[stop-start for start, stop in batches],
        source_files_sha256=sources, source_contract_sha256=contract['sha256'],
        actual_completed_producer_footprints_bound=True, completed_producer_footprints=footprint,
        expected_trained_model_identity_sha256=MODEL_ID, trained_model_or_inference_permit=False,
        figure13_reproduced=False, full_paper_complete=False))
    value = DenseEvidence(identity, tuple(_canonical(r) for r in raw),
        tuple(_canonical(r) for r in rows), upstream, guard)
    _EVIDENCE[value] = dict(identity=identity['sha256'], records=value.records,
        geometry=value.geometry_records, upstream=upstream, guard=guard)
    value.check()
    return value


def _json_line(line):
    return json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(eq=False)
class DenseRunInputs:
    evidence: DenseEvidence
    dataset: object
    paths: dict
    identity: dict
    _guard: object
    verified_samples: int = 0

    def check(self):
        saved = _INPUTS.get(self)
        if saved is None:
            raise ValueError('registered actual complete dense inputs required')
        if (self.evidence is not saved['evidence'] or self.dataset is not saved['dataset']
            or self._guard is not saved['guard'] or self.paths != saved['paths']
            or _sealed(self.identity)['sha256'] != saved['identity']):
            raise ValueError('complete dense input identity or providers changed')
        self.evidence.check(); self.dataset.check(); self._guard.check()

    def __len__(self):
        return len(self.evidence.records)

    def __getitem__(self, index):
        self.check()
        sample = self.dataset[index]
        expected = json.loads(self.evidence.records[index])
        typed = {'fusion:'+n: _typed_sha(sample['fusion'][n]) for n in
            ('emg', 'emg_channel_mask', 'vision_img', 'vision_valid_mask', 'vision_frame_indices', 'joint_angles', 'label_valid_mask')}
        typed.update({'standalone:'+n: _typed_sha(sample['standalone_emg'][n]) for n in
            ('emg', 'emg_channel_mask', 'joint_angles', 'label_valid_mask')})
        if sample['key'] != expected['key'] or typed != expected['typed_input_sha256']:
            raise ValueError('fresh original sample differs from authenticated complete input journal')
        self.check(); self.verified_samples += 1
        return dict(sample, verified_input_sha256=_hash(typed))

    def hash_payloads(self):
        self.check()
        actual = {}
        for name, path in self.paths.items():
            self._guard.check(); actual[name] = _sha(path); self._guard.check()
        if actual != self.evidence.identity['complete_payload_sha256']:
            raise ValueError('complete original payload content changed')
        self.check()
        return actual


def admit_dense_run_inputs(evidence, *, data, crops):
    if not isinstance(evidence, DenseEvidence):
        raise ValueError('registered actual completed evidence required')
    evidence.check()
    from .occlusion_dense_inputs import DensePredictionDataset
    from .fusion_literal import compose_fusion
    ds = DensePredictionDataset(data, crops, upstream=evidence.upstream,
        config=compose_fusion(evidence.upstream), recipe='vits_s', expected_index=evidence.identity['index_identity'])
    if ds.identity != evidence.identity['paired_input_identity']:
        raise ValueError('same original paired normalization/readers/index required')
    paths = {('crop:'+name.split(':')[1].zfill(6) if name.startswith('crop:') else name): p
        for name, p in ds.paths.items()}
    paths.update({'metadata:manifest': _safe(Path(data)/'manifest.json'), 'metadata:episodes': _safe(Path(data)/'metadata.npz')})
    if set(paths) != set(evidence.identity['complete_payload_sha256']) or len(paths) != 54:
        raise ValueError('complete known54-file input footprint required')
    guard, _ = _guard()
    for p in paths.values(): guard.add(p)
    before = {name: _sha(path) for name, path in paths.items()}
    guard.check(); evidence.check(); ds.check()
    if before != evidence.identity['complete_payload_sha256']:
        raise ValueError('full source payload bytes differ from actual complete IO')
    identity = _seal(dict(format='actual_vits_dense_complete_input_admission_v1',
        evidence_identity_sha256=evidence.identity['sha256'], paired_input_identity_sha256=ds.identity['sha256'],
        actual_complete_payload_files=len(paths), actual_complete_payload_bytes=sum(p.stat().st_size for p in paths.values()),
        complete_pre_forward_payload_hashes_verified=True, complete_payload_sha256=before,
        trained_model_or_inference_permit=False, figure13_reproduced=False, full_paper_complete=False))
    value = DenseRunInputs(evidence, ds, paths, identity, guard)
    _INPUTS[value] = dict(identity=identity['sha256'], evidence=evidence, dataset=ds,
        paths=dict(paths), guard=guard)
    value.check()
    return value


ARRAYS = ('fusion_prediction', 'vision_prediction', 'emg_prediction', 'emg_full_prediction',
    'targets', 'label_valid_mask', 'crop_valid_mask', 'mae_fusion_rad', 'mae_vision_rad', 'mae_emg_rad')


def _validate_arrays(arrays, keys):
    import torch
    if not isinstance(arrays, dict) or set(arrays) != set(ARRAYS) or not 1 <= len(keys) <= 64:
        raise ValueError('complete original ten-array raw batch required')
    seen = set()
    for i, k in enumerate(keys):
        _key(k, KEY_FIELDS)
        point = _key(k, ('hand', 'ep_idx', 'center_idx'))
        if (k['sequence'] != keys[0]['sequence']+i or point in seen
            or (k['split'], k['hand']) != (keys[0]['split'], keys[0]['hand'])
            or k['window_end']-k['window_start'] != 7790 or k['center_idx'] != k['window_start']+3895):
            raise ValueError('one contiguous native source batch with exact unique keys required')
        seen.add(point)
    for name in ARRAYS:
        a = arrays[name]
        shape = ((len(keys), 22, 146) if name == 'emg_full_prediction' else
            (len(keys), 1) if name.endswith('_mask') else
            (len(keys),) if name.startswith('mae_') else (len(keys), 22))
        dtype = np.dtype(bool if name.endswith('_mask') else np.float64 if name.startswith('mae_') else np.float32)
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
            raise ValueError('finite native raw batch shape/dtype required: ' + name)
    if not np.array_equal(arrays['emg_prediction'], arrays['emg_full_prediction'][:, :, 73]):
        raise ValueError('raw fullEMG midpoint must exactly equal exported center')
    for role in ('fusion', 'vision', 'emg'):
        expected = np.array([float((torch.from_numpy(arrays[role+'_prediction'][i])
            - torch.from_numpy(arrays['targets'][i])).abs().mean()) for i in range(len(keys))], np.float64)
        if not np.array_equal(expected, arrays['mae_'+role+'_rad']):
            raise ValueError('source unmasked per-row raw error differs')


def write_dense_batch(directory, ordinal, *, keys, arrays, inputs_sha256):
    _guard()
    if (type(ordinal) is not int or ordinal < 0 or not isinstance(inputs_sha256, dict) or not inputs_sha256
        or any(not isinstance(n, str) or not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h)
            for n, h in inputs_sha256.items())):
        raise ValueError('exact ordinal and complete input fingerprints required')
    _validate_arrays(arrays, keys)
    directory = _safe(directory)
    filename = f'{ordinal:06d}.npz'
    with (directory/filename).open('xb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    return _seal(dict(format='native_dense_prediction_raw_batch_v1', ordinal=ordinal, file=filename,
        file_sha256=_sha(directory/filename), keys=copy.deepcopy(keys), inputs_sha256=copy.deepcopy(inputs_sha256),
        arrays_sha256={n: _typed_sha(a) for n, a in arrays.items()}))


def read_dense_batch(directory, record, *, expected_keys):
    _guard(); record = _sealed(record)
    try:
        ordinal = record['ordinal']
        if (type(ordinal) is not int or ordinal < 0 or record['file'] != f'{ordinal:06d}.npz'
            or record['keys'] != expected_keys):
            raise ValueError('exact ordered raw batch path/keys required')
        path = _safe(Path(directory)/record['file']); guard = _StatGuard([path])
        if _sha(path) != record['file_sha256']:
            raise ValueError('raw dense packet bytes changed')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {n: archive[n] for n in archive.files}
        _validate_arrays(arrays, expected_keys)
        if {n: _typed_sha(a) for n, a in arrays.items()} != record['arrays_sha256']:
            raise ValueError('raw dense typed arrays changed')
        guard.check()
        return arrays
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('complete original raw dense packet unavailable') from exc


def compare_dense_batches(first, left, second, right, *, expected_keys):
    a = read_dense_batch(first, left, expected_keys=expected_keys)
    b = read_dense_batch(second, right, expected_keys=expected_keys)
    if (left['inputs_sha256'] != right['inputs_sha256']
        or any(not np.array_equal(a[n], b[n]) or a[n].dtype != b[n].dtype for n in ARRAYS)):
        raise ValueError('fresh native dense replay differs')
    return True


def _scope_batches(evidence, scope):
    batches = native_batch_ranges(evidence.keys())
    if scope == 'complete':
        return batches
    if scope != 'qualification':
        raise ValueError('explicit complete or qualification scope required')
    keys = evidence.keys()
    chosen = []
    for split in ('user', 'gesture', 'both'):
        for hand in ('left', 'right'):
            group = [r for r in batches if (keys[r[0]]['split'], keys[r[0]]['hand']) == (split, hand)]
            chosen.extend([group[0], group[-1]])
    return chosen


def _actor_closed(handle):
    from .emg2pose_memory_process import process_identity
    if (not isinstance(handle, dict) or any(type(handle.get(n)) is not int or handle[n] <= 0 for n in ('pid', 'sid', 'pgrp'))
        or type(handle.get('uid')) is not int or handle['uid'] < 0
        or not isinstance(handle.get('start_ticks'), str) or not handle['start_ticks'].isdigit()):
        raise ValueError('actual dense worker identity required')
    current = process_identity(handle['pid'])
    if current is not None and current['start_ticks'] == handle['start_ticks']:
        raise ValueError('actual dense worker has not exited')


def _actor_key(handle):
    return tuple(handle[n] for n in ('pid', 'start_ticks', 'uid', 'sid', 'pgrp'))


def _cpu_hardware():
    import importlib.metadata
    import platform
    model = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
        if line.startswith('model name'))
    return dict(device='cpu', dtype='float32', autocast=False, torch_threads=2,
        batch_size=64, num_workers=4, pin_memory=True, shuffle=False, drop_last=False,
        hostname=platform.node(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cpu_model=model, architecture=platform.machine(), logical_cpus=os.cpu_count(),
        torch_version=importlib.metadata.version('torch'))


def read_dense_pass(directory, *, evidence):
    """Reopen every real raw batch after worker exit, including all row errors."""
    if not isinstance(evidence, DenseEvidence):
        raise ValueError('registered completed evidence required for a full pass')
    evidence.check(); directory = _safe(directory)
    try:
        journal = directory/'batches.jsonl'
        guard = _StatGuard([directory/'identity.json', directory/'result.json', journal])
        identity = _sealed(_json(directory/'identity.json'))
        result = _sealed(_json(directory/'result.json'))
        result_file_sha256 = _sha(directory/'result.json')
        ranges = _scope_batches(evidence, identity['scope'])
        admission = _sealed(identity['input_admission'])
        hardware = _cpu_hardware()
        if (identity['format'] != 'actual_native_dense_frozen_pass_v1'
            or identity['evidence_identity_sha256'] != evidence.identity['sha256']
            or identity['source_files_sha256'] != evidence.identity['source_files_sha256']
            or identity['model_identity']['sha256'] != MODEL_ID
            or _sealed(identity['model_identity'])['sha256'] != MODEL_ID
            or identity['batch_ranges'] != [list(r) for r in ranges]
            or identity['hardware'] != hardware
            or admission['evidence_identity_sha256'] != evidence.identity['sha256']
            or admission['paired_input_identity_sha256'] != evidence.identity['paired_input_identity']['sha256']
            or admission['complete_payload_sha256'] != evidence.identity['complete_payload_sha256']
            or not admission['complete_pre_forward_payload_hashes_verified']
            or (admission['actual_complete_payload_files'], admission['actual_complete_payload_bytes']) != (54, 61379659366)
            or result['identity_sha256'] != identity['sha256']
            or result['journal_file_sha256'] != _sha(journal)
            or result['points'] != sum(b-a for a, b in ranges)
            or result['complete_payload_sha256_after_forward'] != evidence.identity['complete_payload_sha256']
            or not result['all_model_states_before_after_exact']
            or result['model_state_sha256_before'] != identity['model_identity']['model_state_sha256']
            or result['model_state_sha256_after'] != result['model_state_sha256_before']
            or result['actual_optimizer_updates'] != 0 or result['cuda_initialized']):
            raise ValueError('complete source-bound native prediction result required')
        _actor_closed(result['actor'])
        if len(result['loader_groups']) != 6:
            raise ValueError('all six original loader groups required')
        for group in result['loader_groups']:
            if len(group['workers']) != 4 or not group['all_workers_exited']:
                raise ValueError('four actual source loader workers and closure required')
            for handle in group['workers']: _actor_closed(handle)
        records = [_json_line(line) for line in journal.read_text().splitlines()]
        if len(records) != len(ranges):
            raise ValueError('missing or extra full dense batches')
        keys = evidence.keys(); predictions = []; missing = invalid = 0
        for ordinal, ((start, stop), record) in enumerate(zip(ranges, records)):
            expected = keys[start:stop]
            if record['ordinal'] != ordinal:
                raise ValueError('raw dense batch order changed')
            if record['file'] != f'{ordinal:06d}.npz':
                raise ValueError('exact raw packet filenames required')
            guard.add(_safe(directory/'batches'/record['file']))
            arrays = read_dense_batch(directory/'batches', record, expected_keys=expected)
            for i, key in enumerate(expected):
                original = json.loads(evidence.records[key['sequence']])
                typed = original['typed_input_sha256']
                if (record['inputs_sha256']['row:'+str(key['sequence'])] != _hash(typed)
                    or _typed_sha(arrays['targets'][i, :, None]) != typed['fusion:joint_angles']
                    or _typed_sha(arrays['label_valid_mask'][i]) != typed['fusion:label_valid_mask']
                    or _typed_sha(arrays['crop_valid_mask'][i]) != typed['fusion:vision_valid_mask']):
                    raise ValueError('raw targets/flags/inputs no longer match actual full source samples')
                predictions.append(dict({n: key[n] for n in KEY_FIELDS},
                    **{'mae_'+role+'_rad': float(arrays['mae_'+role+'_rad'][i]) for role in ('vision', 'emg', 'fusion')}))
            missing += int((~arrays['crop_valid_mask']).sum())
            invalid += int((~arrays['label_valid_mask']).sum())
        if result['crop_missing'] != missing or result['label_invalid'] != invalid:
            raise ValueError('retained missing-crop or invalid-label counts changed')
        if identity['scope'] == 'complete' and (missing, invalid) != (695, 518):
            raise ValueError('full original missing/invalid samples must remain in predictions')
        guard.check(); evidence.check()
        return dict(directory=str(directory), identity=identity, result=result,
            records=records, predictions=predictions, _guard=guard,
            _result_file_sha256=result_file_sha256)
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('complete native dense pass unavailable or malformed') from exc


def _compare_passes(first, second, evidence):
    a, b = first['identity'], second['identity']
    left, right = _safe(first['directory']), _safe(second['directory'])
    if left == right:
        raise ValueError('fresh replay requires a distinct canonical result directory')
    for item in (first, second):
        _actor_closed(item['result']['actor'])
        if item.get('_guard') is not None: item['_guard'].check()
    if ((first['result']['actor']['pid'], first['result']['actor']['start_ticks'])
        == (second['result']['actor']['pid'], second['result']['actor']['start_ticks'])):
        raise ValueError('fresh replay requires distinct producing actors')
    preceding_sha = first.get('_result_file_sha256') or _sha(left/'result.json')
    if (not second['result'].get('fresh_model_forward_replay_verified')
        or second['result'].get('replay_of_result_file_sha256') != preceding_sha
        or second['result'].get('replay_of_identity_sha256') != a['sha256']):
        raise ValueError('fresh replay is not bound to this exact preceding result')
    if a != b or first['result']['model_state_sha256_after'] != second['result']['model_state_sha256_after']:
        raise ValueError('fresh native replay identity/model source differs')
    keys = evidence.keys()
    for (start, stop), left, right in zip(a['batch_ranges'], first['records'], second['records']):
        compare_dense_batches(Path(first['directory'])/'batches', left,
            Path(second['directory'])/'batches', right, expected_keys=keys[start:stop])
    for item in (first, second):
        if item.get('_guard') is not None: item['_guard'].check()


def accept_dense_qualification(first, second, *, evidence, output):
    """Only completed real B64/group-tail passes can release a full worker."""
    a = read_dense_pass(first, evidence=evidence); b = read_dense_pass(second, evidence=evidence)
    if a['identity']['scope'] != 'qualification' or b['identity']['scope'] != 'qualification':
        raise ValueError('two actual engineering qualification passes required')
    _compare_passes(a, b, evidence)
    value = _seal(dict(format='actual_dense_cpu_qualification_acceptance_v1',
        evidence_identity_sha256=evidence.identity['sha256'], model_identity_sha256=MODEL_ID,
        source_files_sha256=evidence.identity['source_files_sha256'],
        first=str(_safe(first)), second=str(_safe(second)),
        first_result_file_sha256=_sha(Path(first)/'result.json'),
        second_result_file_sha256=_sha(Path(second)/'result.json'),
        actual_batch_size=64, actual_workers=4, pin_memory_requested=True,
        actual_points_each=a['result']['points'], actual_batches_each=len(a['records']),
        separate_fresh_replay_exact=True, all_workers_exited=True,
        full_population_predicted=False, scientific_eligible=False, full_paper_complete=False))
    _write_json(output, value)
    return value


def _qualification(path, evidence):
    if path is None:
        raise ValueError('actual reviewed B64/4worker qualification required before complete prediction')
    path = _safe(path); guard = _StatGuard([path]); value = _sealed(_json(path))
    if (value['format'] != 'actual_dense_cpu_qualification_acceptance_v1'
        or value['evidence_identity_sha256'] != evidence.identity['sha256']
        or value['model_identity_sha256'] != MODEL_ID
        or value['source_files_sha256'] != evidence.identity['source_files_sha256']
        or (value['actual_batch_size'], value['actual_workers'], value['actual_points_each'], value['actual_batches_each']) != (64, 4, 528, 12)
        or not value['separate_fresh_replay_exact'] or not value['all_workers_exited']):
        raise ValueError('same actual reviewed source/input/model qualification required')
    for label in ('first', 'second'):
        if _sha(Path(value[label])/'result.json') != value[label+'_result_file_sha256']:
            raise ValueError('actual qualification result bytes changed')
    first = read_dense_pass(value['first'], evidence=evidence)
    second = read_dense_pass(value['second'], evidence=evidence)
    if first['identity']['scope'] != 'qualification' or second['identity']['scope'] != 'qualification':
        raise ValueError('qualification must reopen actual engineering passes')
    _compare_passes(first, second, evidence)
    guard.check()
    return value


def _collate_pairs(samples, *, upstream):
    return dict(keys=[s['key'] for s in samples],
        verified_input_sha256=[s['verified_input_sha256'] for s in samples],
        fusion=source_collate_fusion([s['fusion'] for s in samples], upstream=upstream),
        standalone=source_collate_fusion([s['standalone_emg'] for s in samples], upstream=upstream))


@contextmanager
def _managed_loader(dataset, indices, *, upstream):
    """Keep literal loader settings and close its own children even on failure."""
    from functools import partial
    from torch.utils.data import DataLoader, Subset
    from .emg2pose_memory_process import process_identity
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False,
        num_workers=4, collate_fn=partial(_collate_pairs, upstream=upstream),
        pin_memory=True, drop_last=False)
    iterator = iter(loader)
    workers = [process_identity(w.pid) for w in iterator._workers]
    try:
        if len(workers) != 4 or any(w is None for w in workers):
            raise ValueError('four genuine source DataLoader workers required')
        yield iterator, workers, bool(iterator._pin_memory)
    finally:
        # DataLoader normally does this at StopIteration; it is idempotent.
        # Explicitly cover exceptions before exhaustion without signaling any
        # process outside this iterator. The outer owned guardian is separate.
        iterator._shutdown_workers()
        for worker in workers:
            if worker is not None: _actor_closed(worker)


def _bind_models(inputs, trained):
    trained.check(); inputs.check()
    p = trained.identity
    footprints = inputs.evidence.identity['completed_producer_footprints']['producers']
    if (p['sha256'] != MODEL_ID
        or p['fusion_fit_identity_sha256'] != footprints['fusion']['source_fit_identity_sha256']
        or p['fusion_fit_result_sha256'] != footprints['fusion']['source_fit_result_content_sha256']
        or any(p['standalone_producers'][role]['source_fit_identity_sha256']
            != footprints[name]['source_fit_identity_sha256'] for role, name in
            (('vision', 'vision_initializer'), ('emg', 'emg_initializer')))
        or Path(trained.data_locations['data']).resolve() != inputs.dataset.store.root.resolve()
        or Path(trained.data_locations['crops']).resolve() != inputs.dataset.fusion_views[0].crops.resolve()):
        raise ValueError('actual admitted trained producers/data/footprints differ')


def run_dense_pass(inputs, fusion_fit, output, *, scope, qualification=None, replay_of=None):
    """Execute all selected native batches, with no truncation or silent retry."""
    if not isinstance(inputs, DenseRunInputs):
        raise ValueError('registered actual complete dense inputs required')
    inputs.check(); evidence = inputs.evidence
    if scope == 'complete': _qualification(qualification, evidence)
    ranges = _scope_batches(evidence, scope); keys = evidence.keys()
    previous = read_dense_pass(replay_of, evidence=evidence) if replay_of is not None else None
    import torch
    from .occlusion_dense_models import load_dense_trained_models
    from .vision_literal_runtime import vision_state_sha256
    from .emg2pose_memory_process import process_identity
    torch.set_num_threads(2)
    trained = load_dense_trained_models(fusion_fit, upstream=evidence.upstream, recipe='vits_s')
    _bind_models(inputs, trained)
    output = _safe(output, exists=False); output.mkdir(mode=0o700)
    (output/'batches').mkdir(mode=0o700)
    identity = _seal(dict(format='actual_native_dense_frozen_pass_v1', scope=scope,
        evidence_identity_sha256=evidence.identity['sha256'], input_admission=copy.deepcopy(inputs.identity),
        source_files_sha256=evidence.identity['source_files_sha256'], model_identity=copy.deepcopy(trained.identity),
        batch_ranges=[list(r) for r in ranges],
        hardware=_cpu_hardware(),
        scientific_eligible=False, figure13_reproduced=False, full_paper_complete=False))
    if previous is not None and previous['identity'] != identity:
        raise ValueError('fresh replay must consume identical complete source/model/batch identity')
    _write_json(output/'identity.json', identity)
    before = {n: vision_state_sha256(m.state_dict()) for n, m in trained.models.items()}
    groups = []; ordinal = points = missing = invalid = 0; started = time.monotonic()
    with (output/'batches.jsonl').open('x') as journal:
        for split in ('user', 'gesture', 'both'):
            for hand in ('left', 'right'):
                current = [(a, b) for a, b in ranges if (keys[a]['split'], keys[a]['hand']) == (split, hand)]
                indices = [i for a, b in current for i in range(a, b)]
                count = 0
                with _managed_loader(inputs, indices, upstream=evidence.upstream) as (iterator, workers, actual_pin):
                    for batch in iterator:
                        if count >= len(current): raise ValueError('extra native dense batch')
                        a, b = current[count]; expected_keys = keys[a:b]
                        expected_fingerprints = [_hash(json.loads(evidence.records[k['sequence']])['typed_input_sha256']) for k in expected_keys]
                        if batch['keys'] != expected_keys or batch['verified_input_sha256'] != expected_fingerprints:
                            raise ValueError('source DataLoader order/input verification changed')
                        arrays_out = predict_dense_batch(trained, batch['fusion'], batch['standalone'])
                        arrays = dict(arrays_out['arrays'], **arrays_out['errors'],
                            crop_valid_mask=batch['fusion']['vision_valid_mask'].numpy().copy())
                        fingerprints = {'row:'+str(k['sequence']): h for k, h in zip(expected_keys, expected_fingerprints)}
                        for label, payload in (('fusion', batch['fusion']), ('standalone', batch['standalone'])):
                            fingerprints.update({label+':'+n: _typed_sha(v.numpy()) for n, v in payload.items() if isinstance(v, torch.Tensor)})
                        record = write_dense_batch(output/'batches', ordinal, keys=expected_keys,
                            arrays=arrays, inputs_sha256=fingerprints)
                        read_dense_batch(output/'batches', record, expected_keys=expected_keys)
                        if previous is not None:
                            previous['_guard'].check()
                            compare_dense_batches(Path(previous['directory'])/'batches', previous['records'][ordinal],
                                output/'batches', record, expected_keys=expected_keys)
                        journal.write(_canonical(record)+'\n'); journal.flush(); os.fsync(journal.fileno())
                        ordinal += 1; count += 1; points += b-a
                        missing += int((~arrays['crop_valid_mask']).sum()); invalid += int((~arrays['label_valid_mask']).sum())
                        progress = dict(scope=scope, batches=ordinal, total_batches=len(ranges), points=points,
                            total_points=sum(b-a for a, b in ranges), split=split, hand=hand, seconds=time.monotonic()-started)
                        (output/'progress.json').write_text(json.dumps(progress, sort_keys=True)+'\n')
                        print(json.dumps(progress), flush=True)
                        inputs.check()
                if count != len(current): raise ValueError('missing native dense batch')
                for worker in workers: _actor_closed(worker)
                groups.append(dict(split=split, hand=hand, workers=workers, all_workers_exited=True,
                    pin_memory_requested=True, actual_pin_memory=actual_pin, points=len(indices), batches=count))
    if ordinal != len(ranges) or points != sum(b-a for a, b in ranges):
        raise ValueError('full native source population was not consumed exactly')
    inputs.check(); trained.check()
    after = {n: vision_state_sha256(m.state_dict()) for n, m in trained.models.items()}
    if before != after or torch.cuda.is_initialized():
        raise ValueError('frozen complete CPU model state changed or CUDA initialized')
    payloads = inputs.hash_payloads()
    if previous is not None: previous['_guard'].check()
    result = _seal(dict(format='actual_native_dense_frozen_pass_result_v1', identity_sha256=identity['sha256'],
        actor=process_identity(), points=points, batches=ordinal, crop_missing=missing, label_invalid=invalid,
        journal_file_sha256=_sha(output/'batches.jsonl'), model_state_sha256_before=before,
        model_state_sha256_after=after, all_model_states_before_after_exact=True,
        complete_payload_sha256_after_forward=payloads, loader_groups=groups,
        fresh_model_forward_replay_verified=previous is not None,
        replay_of_result_file_sha256=previous['_result_file_sha256'] if previous is not None else None,
        replay_of_identity_sha256=previous['identity']['sha256'] if previous is not None else None,
        actual_optimizer_updates=0, cuda_initialized=False, scope=scope,
        scientific_eligible=False, figure13_reproduced=False, full_paper_complete=False,
        seconds=time.monotonic()-started))
    _write_json(output/'result.json', result)
    return result


def build_dense_report(first, second, *, evidence, output):
    """Rebuild full real CSV/statistics and run the entire original plot main."""
    import csv
    import subprocess
    import sys
    import importlib.metadata
    from .emg2pose_memory_process import process_identity
    a = read_dense_pass(first, evidence=evidence); b = read_dense_pass(second, evidence=evidence)
    if a['identity']['scope'] != 'complete' or not b['result']['fresh_model_forward_replay_verified']:
        raise ValueError('two genuine complete prediction passes and fresh replay required')
    _compare_passes(a, b, evidence)
    joined = join_dense_predictions(evidence.keys(), a['predictions'], evidence.geometry(),
        recipe='vits_s', upstream=evidence.upstream)
    output = _safe(output, exists=False); output.mkdir(mode=0o700)
    import pandas as pd
    csv_path = output/'occlusion_fusion_results.csv'
    pd.DataFrame(joined['csv_rows']).to_csv(csv_path, index=False)
    stats = analyze_source_csv(csv_path, upstream=evidence.upstream)
    _write_json(output/'statistics.json', stats)
    fields = ['count', 'occlusion_mean', 'vision_mean_deg', 'vision_sem_deg',
        'fusion_mean_deg', 'fusion_sem_deg', 'delta_mean_deg', 'delta_sem_deg']
    with (output/'bins.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in stats['bins'])
    pd.DataFrame([dict(p, **{n: g[n] for n in ('occlusion_score', 'visible_ratio', 'status')})
        for p, g in zip(a['predictions'], evidence.geometry())]).to_csv(output/'raw_paired_errors.csv', index=False)
    plot = output/'occlusion_fusion_gain.png'
    cmd = [sys.executable, '-B', str(evidence.upstream/'scripts/paper/plot_occlusion_fusion.py'),
        '--csv', str(csv_path), '--output', str(plot)]
    process = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    (output/'original_plot.log').write_text(process.stdout+process.stderr)
    if process.returncode or not plot.is_file():
        raise ValueError('original complete plotting script failed: '+process.stderr[-1200:])
    evidence.check()
    value = _seal(dict(format='actual_complete_legacy_vits_dense_paired_report_v1',
        evidence_identity_sha256=evidence.identity['sha256'], model_identity_sha256=MODEL_ID,
        actual_points=17168, actual_complete_prediction_passes=2, actual_complete_batches_each=272,
        actual_input_crop_missing_retained=695, actual_invalid_label_retained=518,
        actual_zero_geometry_retained=518, exact_all_raw_prediction_replay=True,
        source_contract=source_analysis_contract(evidence.upstream),
        source_filter_population=stats['filtered_rows'], source_binned_population=stats['binned_rows'],
        source_spearman=stats['spearman'], native_csv_statistics_sha256=stats['sha256'],
        completed_producer_footprints=evidence.identity['completed_producer_footprints'],
        original_plot_exit_code=process.returncode,
        artifacts_sha256={p.name: _sha(p) for p in output.iterdir() if p.is_file()},
        hardware=a['identity']['hardware'], report_actor=process_identity(),
        matplotlib_version=importlib.metadata.version('matplotlib'),
        verification_status='pending independent reconstruction',
        available_legacy_figure13_pipeline_complete=False,
        limitations=['Available41episode legacy release, not established as the PDF complete corpus.',
            'Official1920x1080 video fallback, original X mirror/no distortion; physical equivalence unproven.',
            'Dense center-heldout windows include original disclosed training-input overlap; all keys retained.',
            'Original CSV unmasked22-angle sample errors differ from masked standard Table3 results.',
            'Stale archived eval config/attention patch replaced by current pinned native forwards and actual cotrain source.'],
        scientific_eligible=False, figure13_reproduced=False,
        paper_numerical_equivalence_established=False, full_paper_complete=False))
    _write_json(output/'report.json', value)
    return value


def verify_dense_report(first, second, report, *, evidence, output):
    from .emg2pose_memory_process import process_identity
    report_path = _safe(Path(report)/'report.json')
    guard = _StatGuard([report_path])
    original = _sealed(_json(report_path))
    _actor_closed(original['report_actor'])
    if _actor_key(original['report_actor']) == _actor_key(process_identity()):
        raise ValueError('independent report verification requires a distinct producing actor')
    original_sha = _sha(report_path)
    for name, digest in original['artifacts_sha256'].items():
        path = _safe(Path(report)/name); guard.add(path)
        if _sha(path) != digest:
            raise ValueError('original dense report artifact changed')
    rebuilt = build_dense_report(first, second, evidence=evidence, output=output)
    for name in ('occlusion_fusion_results.csv', 'statistics.json', 'bins.csv', 'raw_paired_errors.csv', 'occlusion_fusion_gain.png'):
        if original['artifacts_sha256'][name] != rebuilt['artifacts_sha256'][name]:
            raise ValueError('fresh complete dense report reconstruction differs: '+name)
    guard.check()
    value = _seal(dict(original_report_sha256=original_sha,
        rebuilt_report_sha256=_sha(Path(output)/'report.json'), all_numeric_csv_statistics_and_png_bytes_exact=True,
        actor=process_identity(), available_legacy_figure13_pipeline_complete=False,
        full_paper_complete=False))
    _write_json(Path(output)/'verification.json', value)
    return value


def _promote_dense_report(report, rebuilt, *, evidence, owned_stage):
    """Publish completion only after the separate verifier's owned stage exits."""
    from .emg2pose_memory_process import process_identity
    if (not isinstance(evidence, DenseEvidence) or owned_stage.get('status') != 'success'
        or owned_stage.get('worker_exit_code') != 0 or not owned_stage.get('all_owned_consumers_exited')):
        raise ValueError('successful closed independent verification stage required')
    evidence.check()
    for name in ('worker', 'guardian'): _actor_closed(owned_stage[name])
    if _actor_key(owned_stage['owner']) != _actor_key(process_identity()):
        raise ValueError('only the actual owning controller may publish verification completion')
    directory, rebuilt = _safe(report), _safe(rebuilt)
    original_path, rebuilt_path, verification_path = directory/'report.json', rebuilt/'report.json', rebuilt/'verification.json'
    guard = _StatGuard([original_path, rebuilt_path, verification_path])
    original = _sealed(_json(original_path)); other = _sealed(_json(rebuilt_path)); verified = _sealed(_json(verification_path))
    _actor_closed(original['report_actor']); _actor_closed(verified['actor'])
    if (_actor_key(verified['actor']) != _actor_key(owned_stage['worker'])
        or _actor_key(other['report_actor']) != _actor_key(verified['actor'])
        or (original['report_actor']['pid'], original['report_actor']['start_ticks'])
            == (verified['actor']['pid'], verified['actor']['start_ticks'])
        or verified['original_report_sha256'] != _sha(original_path)
        or verified['rebuilt_report_sha256'] != _sha(rebuilt_path)
        or not verified['all_numeric_csv_statistics_and_png_bytes_exact']):
        raise ValueError('actual fresh owned verification must bind both exact pending reports')
    if (original['format'] != 'actual_complete_legacy_vits_dense_paired_report_v1'
        or original['evidence_identity_sha256'] != evidence.identity['sha256']
        or original['model_identity_sha256'] != MODEL_ID
        or original['actual_points'] != 17168 or original['actual_complete_prediction_passes'] != 2
        or not original['exact_all_raw_prediction_replay']
        or original['scientific_eligible'] or original['available_legacy_figure13_pipeline_complete']
        or original['verification_status'] != 'pending independent reconstruction'
        or original['figure13_reproduced'] or original['paper_numerical_equivalence_established'] or original['full_paper_complete']):
        raise ValueError('canonical pending legacy report required before publication')
    exclude = {'sha256', 'artifacts_sha256', 'report_actor'}
    if {k: v for k, v in original.items() if k not in exclude} != {k: v for k, v in other.items() if k not in exclude}:
        raise ValueError('fresh report scientific metadata differs')
    for name in ('occlusion_fusion_results.csv', 'statistics.json', 'bins.csv', 'raw_paired_errors.csv', 'occlusion_fusion_gain.png'):
        for parent, document in ((directory, original), (rebuilt, other)):
            path = _safe(parent/name); guard.add(path)
            if _sha(path) != document['artifacts_sha256'][name]:
                raise ValueError('verified report artifact changed before completion')
        if original['artifacts_sha256'][name] != other['artifacts_sha256'][name]:
            raise ValueError('independent report artifact digest differs')
    guard.check(); evidence.check()
    completion = _seal(dict(format='actual_owned_verified_legacy_dense_report_completion_v1',
        evidence_identity_sha256=evidence.identity['sha256'], model_identity_sha256=MODEL_ID,
        original_pending_report_file_sha256=_sha(original_path), rebuilt_report_file_sha256=_sha(rebuilt_path),
        verification_file_sha256=_sha(verification_path), actual_owned_verification_stage=owned_stage,
        all_owned_verification_consumers_exited=True, available_legacy_figure13_pipeline_complete=True,
        scientific_eligible=True, figure13_reproduced=False, full_paper_complete=False))
    _write_json(directory/'completion.json', completion)
    promoted = {k: v for k, v in original.items() if k != 'sha256'}
    promoted.update(verification_status='verified by separate process and closed owned stage',
        available_legacy_figure13_pipeline_complete=True, scientific_eligible=True,
        completion_receipt_file_sha256=_sha(directory/'completion.json'))
    promoted = _seal(promoted)
    temporary = original_path.with_name('report.json.verified-pending')
    _write_json(temporary, promoted); os.replace(temporary, original_path)
    return promoted


def run_dense_workflow(request, output):
    """Server-local seven-stage owner; no Torch/model imports in the parent."""
    import sys
    from .emg2pose_memory_process import run_owned_stage, process_identity, same_process
    required = ('evidence', 'upstream', 'index_root', 'paired_root', 'geometry_root', 'data', 'crops', 'fit')
    if not isinstance(request, dict) or set(request) != set(required):
        raise ValueError('explicit complete native dense workflow paths required')
    guard, sources = _guard()
    cli = _safe(ROOT/'scripts/paper/run_dense_occlusion.py'); guard.add(cli)
    sources[str(cli.relative_to(ROOT))] = _sha(cli)
    root = _safe(output, exists=False); root.mkdir(mode=0o700)
    _write_json(root/'request.json', request); guard.add(root/'request.json')
    evidence = load_dense_evidence(**{n: request[n] for n in required[:5]})
    parent = process_identity(os.getppid())
    def check():
        guard.check(); evidence.check()
        if 'torch' in sys.modules or not same_process(parent):
            raise ValueError('dense controller must retain its actual parent and stay model-free')
    common = [sys.executable, '-B', str(cli)]
    args = [item for n in required[:5] for item in ('--'+n.replace('_', '-'), str(request[n]))]
    predict = [item for n in ('data', 'crops', 'fit') for item in ('--'+n, str(request[n]))]
    a, b = root/'qual-a', root/'qual-b'
    full_a, full_b = root/'full-a', root/'full-b'
    qualification = root/'qualification.json'
    stages = [
        ('qual-a', ['predict', *args, *predict, '--scope', 'qualification', '--output', str(a)]),
        ('qual-b', ['predict', *args, *predict, '--scope', 'qualification', '--replay-of', str(a), '--output', str(b)]),
        ('accept-qualification', ['accept-qualification', *args, '--first', str(a), '--second', str(b), '--output', str(qualification)]),
        ('full-a', ['predict', *args, *predict, '--scope', 'complete', '--qualification', str(qualification), '--output', str(full_a)]),
        ('full-b', ['predict', *args, *predict, '--scope', 'complete', '--qualification', str(qualification), '--replay-of', str(full_a), '--output', str(full_b)]),
        ('report', ['report', *args, '--first', str(full_a), '--second', str(full_b), '--output', str(root/'report')]),
        ('verify-report', ['verify-report', *args, '--first', str(full_a), '--second', str(full_b), '--report', str(root/'report'), '--output', str(root/'report-rebuilt')]),
    ]
    owned = []; started = time.monotonic()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', MPLCONFIGDIR=str(root/'mpl-cache'))
    for name, command in stages:
        check(); print(json.dumps(dict(phase=name, elapsed_seconds=time.monotonic()-started)), flush=True)
        result = run_owned_stage([*common, *command], root/('owned-'+name), descriptors={},
            guard=check, environment=environment, poll_seconds=2., terminate_grace=15.)
        if (result['status'] != 'success' or result['worker_exit_code'] != 0
            or not result['all_owned_consumers_exited']
            or any(same_process(result[n]) for n in ('worker', 'guardian'))):
            raise ValueError('native dense owned stage failed: '+name)
        owned.append(result); check()
    report = _promote_dense_report(root/'report', root/'report-rebuilt', evidence=evidence, owned_stage=owned[-1])
    verified = _json(root/'report-rebuilt/verification.json')
    if not verified['all_numeric_csv_statistics_and_png_bytes_exact']:
        raise ValueError('fresh complete report reconstruction not verified')
    if any(_sha(ROOT/name) != digest for name, digest in sources.items()):
        raise ValueError('frozen dense execution source changed')
    check()
    value = _seal(dict(format='actual_legacy_vits_dense_complete_owned_workflow_v1',
        actual_owned_stages=owned, all_owned_consumers_exited=True,
        qualification_file_sha256=_sha(qualification), report=report,
        report_file_sha256=_sha(root/'report/report.json'), verification=verified,
        source_files_sha256=sources, parent_torch_unimported='torch' not in sys.modules,
        actual_complete_prediction_passes=2, actual_complete_points_each=17168,
        available_legacy_figure13_pipeline_complete=True, figure13_reproduced=False,
        full_paper_complete=False, seconds=time.monotonic()-started))
    _write_json(root/'workflow-result.json', value)
    return value
