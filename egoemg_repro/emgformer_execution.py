"""Standard-library admission of an immutable native EMGFormer execution."""
from pathlib import Path
import hashlib
import json
import os
import re
import stat

from .emg2pose_memory_process import process_identity, same_process


OPERATIONAL_SOURCES = frozenset((
    'private/runtime-worker.py', 'private/runtime-probe.py', 'private/recorder.py',
    'egoemg_repro/emgformer_resource.py', 'egoemg_repro/emgformer_execution.py',
    'egoemg_repro/emg2pose_memory_process.py'))
REVIEWED_CANDIDATE = '24290047e6eaab7e845f009f645c58c9b1d18e38'
REVIEWED_BASE = '9f22cfa8d973af8b4ee20a75bd8ad5e53ace0a29'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def file_stamp(path):
    s = Path(path).lstat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,
            s.st_mode,s.st_uid,s.st_gid,s.st_nlink]


def read_json(path):
    return json.loads(Path(path).read_text())


def pin_stage(root):
    root = Path(root).resolve()
    path = root/'private/stage-receipt.json'
    before = file_stamp(path); raw = path.read_bytes(); value = json.loads(raw)
    if (set(value) != {'source_commit','files_sha256','full_paper_complete','bundle_sha256','root'}
            or value['root'] != str(root) or value['full_paper_complete'] is not False
            or type(value['source_commit']) is not str
            or re.fullmatch('[0-9a-f]{40}',value['source_commit']) is None
            or type(value['files_sha256']) is not dict or not value['files_sha256']):
        raise ValueError('stage root or manifest schema differs')
    body = {k: value[k] for k in ('source_commit','files_sha256','full_paper_complete')}
    bundle = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if value['bundle_sha256'] != bundle:
        raise ValueError('stage canonical bundle differs')
    pins = {}
    for name, expected in value['files_sha256'].items():
        p = root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or name == 'private/stage-receipt.json'
                or p.resolve() != p or not stat.S_ISREG(p.lstat().st_mode)
                or type(expected) is not str or re.fullmatch('[0-9a-f]{64}',expected) is None):
            raise ValueError('stage member path or digest differs')
        stamp = file_stamp(p)
        if file_sha(p) != expected or file_stamp(p) != stamp:
            raise ValueError('stage member bytes changed: '+name)
        pins[str(p)] = (expected,stamp)
    if file_stamp(path) != before or not stat.S_ISREG(before[5]):
        raise ValueError('stage receipt changed while admitted')
    pins[str(path)] = (hashlib.sha256(raw).hexdigest(),before)
    return value,pins


def validate_review(root, stage):
    """Bind the one review and its regression resolution to the exact scripts."""
    root = Path(root); review = read_json(root/'private/final-review.json')
    initial = read_json(root/'private/review-initial.json')
    resolution = read_json(root/'private/review-resolution.json')
    sources = {n: stage['files_sha256'].get(n) for n in OPERATIONAL_SOURCES}
    if (review.get('format') != 'native_emgformer_operational_final_review_v1'
            or any(type(review.get(k)) is not int or review[k] != 0 for k in ('critical','important'))
            or review.get('second_review_requested') is not False
            or None in sources.values() or review.get('sources_sha256') != sources
            or review.get('reviewed_candidate_commit') != REVIEWED_CANDIDATE
            or review.get('resolved_source_commit') != stage['source_commit']
            or review.get('initial_review_sha256') != file_sha(root/'private/review-initial.json')
            or review.get('resolution_sha256') != file_sha(root/'private/review-resolution.json')
            or initial.get('format') != 'native_emgformer_operational_initial_review_v1'
            or initial.get('candidate_commit') != REVIEWED_CANDIDATE
            or initial.get('base_commit') != REVIEWED_BASE
            or initial.get('reviewer') != '/root/native_emgformer_gpu_workflow_review'
            or initial.get('finding_ids') != ['I1','I2','I3','I4']
            or type(initial.get('critical')) is not int or initial['critical'] != 0
            or type(initial.get('important')) is not int or initial['important'] != 4
            or resolution.get('format') != 'native_emgformer_operational_review_resolution_v1'
            or resolution.get('reviewed_candidate_commit') != REVIEWED_CANDIDATE
            or resolution.get('resolved_source_commit') != stage['source_commit']
            or resolution.get('resolved_important_ids') != ['I1','I2','I3','I4']
            or any(type(resolution.get(k)) is not int or resolution[k] != 0 for k in ('critical','important'))
            or resolution.get('second_review_requested') is not False
            or type(resolution.get('regression_tests_passed')) is not int or resolution['regression_tests_passed'] <= 0
            or re.fullmatch('[0-9a-f]{64}',str(resolution.get('regression_log_sha256'))) is None
            or resolution.get('sources_sha256') != sources):
        raise ValueError('review source coverage, candidate, or regression resolution differs')
    for name in ('private/final-review.json','private/review-initial.json','private/review-resolution.json'):
        if stage['files_sha256'].get(name) != file_sha(root/name):
            raise ValueError('review metadata is outside the sealed stage')
    return file_sha(root/'private/final-review.json')


def preflight_binding(root, stage, stage_sha, controller):
    root = Path(root); path = root/'private/phase-preflight.json'
    completion = root/'private/owned-preflight/stage-result.json'
    if not path.exists() or not completion.exists():
        raise ValueError('current completed preflight prerequisite is missing')
    phase = read_json(path); result = read_json(completion)
    registration = read_json(root/'private/owned-preflight/started.json')
    identity = read_json(root/'private/registered-cohort.json')
    if (phase.get('phase') != 'preflight' or phase.get('source_commit') != stage['source_commit']
            or phase.get('bundle_sha256') != stage['bundle_sha256'] or phase.get('stage_receipt_sha256') != stage_sha
            or phase.get('controller') != controller or result.get('owner') != controller
            or registration.get('owner') != controller
            or any(result.get(k) != registration.get(k) for k in ('worker','guardian'))
            or result.get('status') != 'success' or type(result.get('worker_exit_code')) is not int
            or result['worker_exit_code'] != 0 or result.get('all_owned_consumers_exited') is not True
            or any(same_process(result[k]) for k in ('worker','guardian'))
            or any(type(phase.get(k)) is not int or phase[k] != 0 for k in
                   ('actual_model_constructions','actual_optimizer_constructions','actual_optimizer_updates'))
            or phase.get('actual_cuda_initialized') is not False or phase.get('actual_gpu_forward_observations') != {}
            or phase.get('result',{}).get('identity') != identity
            or phase['result'].get('physical_input_bytes') != 16540457757
            or len(phase['result'].get('input_paths',{})) != 16
            or phase.get('actual_owner') != {k:registration['worker'][k] for k in ('pid','start_ticks')}):
        raise ValueError('current preflight source, input, owner, or completion differs')
    external = phase['result']['immutable_external_file_stamps']
    if any(file_stamp(n) != s for n,s in external.items()):
        raise ValueError('preflight external source/input generation changed')
    return dict(phase_sha256=file_sha(path), owned_completion_sha256=file_sha(completion))


def authorize_worker(root, stage, stage_sha, mode, review_sha):
    """Run before Torch: only this live controller's registered phase may enter."""
    root = Path(root); path = root/('private/authorization-'+mode+'.json')
    if not path.exists():
        raise ValueError('current controller phase authorization is missing')
    auth = read_json(path)
    if (os.environ.get('EGOEMG_PHASE_AUTHORIZATION_SHA256') != file_sha(path)
            or os.environ.get('EGOEMG_STAGE_RECEIPT_SHA256') != stage_sha
            or auth.get('phase') != mode or auth.get('stage_receipt_sha256') != stage_sha
            or auth.get('review_sha256') != review_sha):
        raise ValueError('current controller phase authorization binding differs')
    registration = read_json(root/('private/owned-'+mode+'/started.json'))
    controller = auth.get('controller')
    if (controller != registration.get('owner') or not same_process(controller)
            or not same_process(registration.get('guardian'))
            or registration['guardian']['pid'] != os.getppid()
            or any(registration['worker'].get(k) != process_identity().get(k)
                   for k in ('pid','start_ticks','uid','sid','pgrp'))):
        raise ValueError('phase authorization does not own this actual worker/controller')
    expected = None if mode == 'preflight' else preflight_binding(root,stage,stage_sha,controller)
    if auth.get('preflight') != expected:
        raise ValueError('phase authorization preflight differs')
    return controller
