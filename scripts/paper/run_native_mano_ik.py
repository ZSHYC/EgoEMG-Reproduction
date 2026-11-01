"""Owned native IK: qualify original batches, then stream all twelve shards.

CPU/small-source runs are explicitly engineering. Full released-source runs
require a closed same-source/default-batch qualification on the named idle GPU.
No queue or GPU is acquired merely by importing or validating this command.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from egoemg_repro.classic_emg_cohort import _safe
from egoemg_repro.data import file_sha256
from egoemg_repro.emg2pose_memory_process import process_identity, same_process, run_owned_stage
from egoemg_repro.native_ik_inputs import admit_native_ik_inputs, _Guard, _hash, HANDS
from egoemg_repro.native_ik_runtime import admit_ik_sources
from egoemg_repro import native_ik_results as results

IMPORTED_SHA = file_sha256(__file__)
SOURCE_NAMES = ['egoemg_repro/' + n + '.py' for n in (
    '__init__', 'data', 'partitions', 'classic_emg', 'classic_emg_data', 'classic_emg_cohort',
    'native_ik_inputs', 'native_ik_runtime', 'native_ik_results', 'emg2pose_memory_process')]
SOURCE_NAMES += ['scripts/paper/run_native_mano_ik.py', 'assets/native_ik_dependency_sources.json',
                 'results/legacy-geometry-complete-numeric-admission-20261101.json']
REQUEST_KEYS = {'data', 'geometry', 'upstream', 'manotorch', 'chumpy', 'mano_assets',
    'engineering', 'batch_size', 'device', 'gpu_uuid', 'qualification', 'minimum_free_bytes'}


def workflow_sources():
    if file_sha256(__file__) != IMPORTED_SHA:
        raise ValueError('imported native IK workflow source changed')
    return {n: file_sha256(_safe(ROOT/n)) for n in SOURCE_NAMES}


def validate_request(action, config):
    if action not in ('workflow', 'qualify'):
        raise ValueError('explicit native IK workflow or qualify action required')
    if type(config) is not dict or set(config) != REQUEST_KEYS:
        raise ValueError('exact native IK workflow configuration required')
    if type(config['engineering']) is not bool:
        raise ValueError('explicit engineering boolean required')
    if type(config['batch_size']) is not int or not 1 <= config['batch_size'] <= 10096:
        raise ValueError('positive original IK batch size up to10096 required')
    if config['device'] not in ('cpu', 'cuda:0'):
        raise ValueError('explicit cpu or one admitted CUDA device required')
    if type(config['minimum_free_bytes']) is not int or config['minimum_free_bytes'] < 15*1024**3:
        raise ValueError('positive storage reserve of at least15GiB required')
    if any(type(config[n]) is not str or not config[n] for n in (
        'data', 'geometry', 'upstream', 'manotorch', 'chumpy', 'mano_assets')):
        raise ValueError('explicit original IK source paths required')
    if config['device'] == 'cpu' and config['gpu_uuid'] is not None:
        raise ValueError('CPU engineering cannot claim a GPU identity')
    if config['device'] == 'cuda:0' and (type(config['gpu_uuid']) is not str or not config['gpu_uuid'].startswith('GPU-')):
        raise ValueError('explicit physical CUDA GPU UUID required')
    if action == 'qualify' or not config['engineering']:
        if config['engineering'] or config['batch_size'] != 10096 or config['device'] != 'cuda:0':
            raise ValueError('full-source qualification requires original batch10096 on CUDA')
    if action == 'workflow' and not config['engineering']:
        if type(config['qualification']) is not str or not config['qualification']:
            raise ValueError('closed original-batch qualification required before full IK')
        if config['minimum_free_bytes'] < 200*1024**3:
            raise ValueError('initial full-IK raw-state storage reserve must be at least200GiB')
    elif config['qualification'] is not None:
        raise ValueError('engineering and qualification creation cannot reuse a qualification')
    return config


def _gpu_record(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pci.bus_id,name,driver_version,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = [[v.strip() for v in line.split(',')] for line in raw.splitlines() if line.strip()]
    matches = [r for r in rows if r[0] == uuid]
    if len(matches) != 1: raise ValueError('admitted physical GPU UUID is unavailable')
    row = matches[0]
    return dict(uuid=row[0], pci_bus_id=row[1], name=row[2], driver_version=row[3], total_memory_mib=int(row[4]))


def _gpu_applications(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    return [int(row[1].strip()) for line in raw.splitlines() if line.strip()
            for row in [line.split(',')] if row[0].strip() == uuid]


def _read(path):
    path = _safe(path); frozen = _Guard([path], [])
    value = results._read_json(path, frozen)
    sha = file_sha256(path); frozen.check()
    return value, frozen, sha


def _same_actor(a, b):
    return all(a.get(k) == b.get(k) for k in ('pid', 'start_ticks', 'uid', 'sid', 'pgrp'))


def _admit(config):
    inputs = admit_native_ik_inputs(config['data'], config['geometry'], config['upstream'],
        engineering=config['engineering'], batch_size=config['batch_size'])
    sources = admit_ik_sources(*(config[n] for n in ('upstream', 'manotorch', 'chumpy', 'mano_assets')))
    return inputs, sources


def _runtime(config, *, verification=False):
    import torch
    torch.set_num_threads(2)
    if verification:
        def denied(*a, **kw): raise RuntimeError('native IK verification cannot construct or step an optimizer')
        torch.optim.Optimizer.__init__ = denied; torch.optim.LBFGS.step = denied
    if config['device'] == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise ValueError('CPU engineering requires CUDA hidden and uninitialized')
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuid']:
        raise ValueError('worker CUDA_VISIBLE_DEVICES must name exactly the admitted physical GPU UUID')
    return torch


def _worker(action, config_path, output, *, hand=None, shard=None, original=None, flow=None, batch_index=None):
    document, frozen, config_sha = _read(config_path)
    config = document['request']; validate_request(document['action'], config)
    source_guard = _Guard([ROOT/n for n in SOURCE_NAMES], [])
    if document['sources_sha256'] != workflow_sources():
        raise ValueError('native IK worker code snapshot differs')
    output = _safe(output, exists=False)
    if output.exists(): raise ValueError('new native IK worker output required')
    inputs, sources = _admit(config)
    if (_hash(inputs.identity) != document['input_identity_sha256']
        or _hash(sources.identity) != document['source_identity_sha256']):
        raise ValueError('fresh native IK source admission differs from controller')
    def guard():
        inputs.check(); sources.check(); frozen.check(); source_guard.check()
    runtime = None
    if action in ('produce', 'verify', 'produce-one', 'verify-one'):
        torch = _runtime(config, verification=action.startswith('verify'))
        runtime = sources.build(device=config['device'])
        if config['device'] == 'cuda:0':
            if _gpu_record(config['gpu_uuid']) != document['physical_device']:
                raise ValueError('native IK CUDA physical hardware/backend changed')
            properties = torch.cuda.get_device_properties(0)
            if properties.total_memory // 1024**2 < document['physical_device']['total_memory_mib'] - 1024:
                raise ValueError('visible Torch GPU memory differs from admitted physical card')
    if action in ('produce-one', 'verify-one'):
        if document['action'] != 'qualify' or batch_index != 32 or shard != 0:
            raise ValueError('qualification uses fixed original shard0 batch32')
        batch = results.select_ik_batch(inputs, hand, shard, batch_index)
        if batch.identity['rows'] != 10096 or not batch.identity['full_available_release']:
            raise ValueError('actual complete default batch10096 required for qualification')
        if action == 'produce-one': return results.produce_ik_batch(batch, runtime, output, guard=guard)
        return results.verify_ik_batch(inputs, runtime, original, output, guard=guard)
    if action == 'produce':
        results.select_ik_batch(inputs, hand, shard, 0)
        output.mkdir(parents=True, mode=0o700)
        records = []
        for index in range(inputs.identity['batches_per_hand'][shard]):
            batch = results.select_ik_batch(inputs, hand, shard, index)
            target = output/f'batch-{index:05d}'
            result = results.produce_ik_batch(batch, runtime, target, guard=guard)
            records.append(dict(batch_index=index, result_sha256=file_sha256(target/'result.json'),
                                state_sha256=result['state_file_sha256']))
            print(json.dumps(dict(action='produced-batch', hand=hand, shard=shard, batch_index=index,
                rows=batch.identity['rows'], outer_steps=result['record']['actual_ik_optimizer_steps'])), flush=True)
        summary = dict(format='original_native_ik_pending_shard_v1', hand=hand, shard=shard, batches=records)
    elif action == 'verify':
        pending, pending_guard, _ = _read(Path(original)/'pending.json')
        results.require_closed(pending['actor'])
        if (pending.get('format') != 'original_native_ik_pending_shard_v1' or pending['hand'] != hand
            or pending['shard'] != shard or pending['configuration_sha256'] != config_sha
            or len(pending['batches']) != inputs.identity['batches_per_hand'][shard]):
            raise ValueError('complete exited native IK producer shard required')
        output.mkdir(parents=True, mode=0o700)
        records = []
        for index, entry in enumerate(pending['batches']):
            directory = Path(original)/f'batch-{index:05d}'
            packet = results.inspect_ik_batch(inputs, directory)
            if (entry['batch_index'] != index or file_sha256(directory/'result.json') != entry['result_sha256']
                or packet.metadata['state_file_sha256'] != entry['state_sha256']
                or not _same_actor(packet.metadata['actor'], pending['actor'])):
                raise ValueError('native IK shard producer raw packet/actor differs')
            result = results.verify_ik_batch(inputs, runtime, directory, output/f'batch-{index:05d}', guard=guard)
            pending_guard.check(); packet.check(rehash=True)
            records.append(dict(batch_index=index, verification_sha256=file_sha256(output/f'batch-{index:05d}'/'verification.json')))
            print(json.dumps(dict(action='verified-batch', hand=hand, shard=shard, batch_index=index)), flush=True)
        summary = dict(format='original_native_ik_verified_shard_v1', hand=hand, shard=shard, batches=records)
    elif action == 'finalize':
        verified, verify_guard, _ = _read(Path(original)/'verified.json')
        results.require_closed(verified['actor'])
        if (verified.get('format') != 'original_native_ik_verified_shard_v1'
            or verified['hand'] != hand or verified['shard'] != shard
            or verified['configuration_sha256'] != config_sha
            or len(verified['batches']) != inputs.identity['batches_per_hand'][shard]):
            raise ValueError('complete exited native IK geometric verifier shard required')
        prefix = Path(flow)/f'{hand}-shard-{shard}'
        pairs = []
        for index, entry in enumerate(verified['batches']):
            receipt, receipt_guard, sha = _read(prefix/'verify'/f'batch-{index:05d}'/'verification.json')
            if entry['batch_index'] != index or entry['verification_sha256'] != sha or not _same_actor(receipt['actor'], verified['actor']):
                raise ValueError('native IK verification actor or raw receipt differs')
            pairs.append((prefix/'produce'/f'batch-{index:05d}', prefix/'verify'/f'batch-{index:05d}'))
            receipt_guard.check()
        result = results.finalize_ik_shard(inputs, hand, shard, pairs, output, guard=guard)
        verify_guard.check(); return result
    elif action == 'assemble':
        directories = [Path(flow)/f'{h}-shard-{s}'/'labels' for h in HANDS for s in range(6)]
        return results.assemble_ik_labels(inputs, directories, output, guard=guard)
    else:
        raise ValueError('unknown native IK worker action')
    guard()
    result = results.seal(dict(**summary, actor=process_identity(), configuration_sha256=config_sha,
        input_identity_sha256=_hash(inputs.identity), source_identity_sha256=_hash(sources.identity),
        runtime_identity=runtime.identity, full_frame_labels_regenerated=False,
        scientific_eligible=False, full_paper_complete=False))
    results.write_json(output/('pending.json' if action == 'produce' else 'verified.json'), result)
    guard(); return result


def _check_qualification(config, inputs, sources, physical, source_snapshot):
    path = _safe(config['qualification'])/'qualification.json'
    permit, frozen, sha = _read(path)
    results.require_closed(permit.get('actor'))
    if (permit.get('format') != 'original_native_ik_default_batch_qualification_v1'
        or permit.get('original_default_batch10096_qualified') is not True
        or permit.get('input_identity_sha256') != _hash(inputs.identity)
        or permit.get('source_identity_sha256') != _hash(sources.identity)
        or permit.get('physical_device') != physical or permit.get('sources_sha256') != source_snapshot
        or len(permit.get('batches', [])) != 2):
        raise ValueError('same-source, same-physical-GPU closed native qualification required')
    root = path.parent
    for hand, record in zip(HANDS, permit['batches']):
        packet, receipt_guard, _, receipt_sha = results._accepted(inputs, root/f'qual-{hand}-produce', root/f'qual-{hand}-verify')
        if (packet.metadata['batch'] != results.select_ik_batch(inputs, hand, 0, 32).identity
            or packet.metadata['batch']['rows'] != 10096
            or packet.metadata['state_file_sha256'] != record.get('raw_state_sha256')
            or receipt_sha != record.get('verification_sha256')):
            raise ValueError('actual default-batch native qualification payload differs')
        receipt_guard.check()
    for stage in permit.get('owned_stages', []):
        if stage.get('status') != 'success' or stage.get('worker_exit_code') != 0 or stage.get('all_owned_consumers_exited') is not True:
            raise ValueError('qualification owning stage did not complete')
        results.require_closed(stage.get('worker')); results.require_closed(stage.get('guardian'))
    if len(permit.get('owned_stages', [])) != 4:
        raise ValueError('all four qualification stages required')
    frozen.check(); return sha, frozen


def run_workflow(action, config, output):
    validate_request(action, config)
    output = _safe(output, exists=False)
    if output.exists(): raise ValueError('new workflow output required; existing output is preserved')
    start = time.monotonic()
    paths = [_safe(ROOT/n) for n in SOURCE_NAMES]
    frozen_sources = _Guard(paths, [])
    snapshot = workflow_sources(); frozen_sources.check()
    inputs, sources = _admit(config)
    results._new_output(inputs, output)
    if shutil.disk_usage(output.parent).free < config['minimum_free_bytes']:
        raise ValueError('native IK initial raw-state storage admission failed')
    physical = _gpu_record(config['gpu_uuid']) if config['device'] == 'cuda:0' else None
    permit_sha, permit_guard = (None, None)
    if action == 'workflow' and not config['engineering']:
        permit_sha, permit_guard = _check_qualification(config, inputs, sources, physical, snapshot)
    output.mkdir(parents=True, mode=0o700)
    configuration = results.seal(dict(action=action, request=config, sources_sha256=snapshot,
        input_identity_sha256=_hash(inputs.identity), source_identity_sha256=_hash(sources.identity),
        physical_device=physical, qualification_file_sha256=permit_sha))
    config_path = output/'configuration.json'; results.write_json(config_path, configuration)
    config_guard = _Guard([config_path], [])
    owned = []; stage_paths = []; lock = None
    def guard():
        frozen_sources.check(); config_guard.check(); inputs.check(); sources.check()
        if permit_guard is not None: permit_guard.check()
        if shutil.disk_usage(output).free < 15*1024**3:
            raise ValueError('native IK remaining storage reserve fell below15GiB')
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available < 2*1024**3: raise ValueError('native IK remaining host memory below2GiB')
        if physical is not None:
            if _gpu_record(config['gpu_uuid']) != physical:
                raise ValueError('native IK physical GPU admission changed')
            permitted = []
            for stage_path in stage_paths:
                launched = stage_path/'started.json'
                if launched.is_file():
                    value = json.loads(launched.read_text())
                    if not _same_actor(value['owner'], process_identity()):
                        raise ValueError('native IK GPU stage owner changed')
                    if same_process(value['worker']): permitted.append(value['worker']['sid'])
            for pid in _gpu_applications(config['gpu_uuid']):
                actor = process_identity(pid)
                if actor is not None and actor['sid'] not in permitted:
                    raise ValueError('foreign CUDA consumer appeared; native IK stops only its own stage')
    def stage(role, worker_action, target, **kwargs):
        owner_output = output/('owned-'+role); stage_paths.append(owner_output)
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--worker-action', worker_action,
                '--config', str(config_path), '--output', str(target)]
        for key, value in kwargs.items(): argv.extend(['--'+key.replace('_','-'), str(value)])
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   CUDA_VISIBLE_DEVICES=config['gpu_uuid'] if physical and worker_action in ('produce', 'verify', 'produce-one', 'verify-one') else '')
        receipt = run_owned_stage(argv, owner_output, descriptors={}, guard=guard, environment=env,
                                  poll_seconds=2., terminate_grace=10.)
        results.require_closed(receipt['worker']); results.require_closed(receipt['guardian'])
        if receipt['worker_exit_code'] != 0 or receipt['all_owned_consumers_exited'] is not True:
            raise ValueError('actual native IK owning stage did not close successfully')
        name = {'produce':'pending.json', 'verify':'verified.json', 'finalize':'shard.json',
                'assemble':'labels.json', 'produce-one':'result.json', 'verify-one':'verification.json'}[worker_action]
        metadata, metadata_guard, _ = _read(target/name)
        if not _same_actor(metadata['actor'], receipt['worker']):
            raise ValueError('native IK output actor differs from actual owned worker')
        owned.append(dict(role=role, **receipt)); metadata_guard.check(); guard()
        return metadata
    try:
        if physical is not None:
            lock_path = output.parent/('.native-ik-'+config['gpu_uuid']+'.lock')
            if lock_path.is_symlink(): raise ValueError('native IK GPU lock cannot be a symlink')
            lock = lock_path.open('a+b')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _gpu_applications(config['gpu_uuid']): raise ValueError('native IK qualification/full execution requires an actually idle GPU')
            free = subprocess.check_output(['nvidia-smi', '-i', config['gpu_uuid'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            if int(free.strip()) < 64*1024: raise ValueError('native IK GPU admission requires64GiB free')
        guard()
        if action == 'qualify':
            batches = []
            for hand in HANDS:
                producer = stage('qual-'+hand+'-produce', 'produce-one', output/f'qual-{hand}-produce', hand=hand, shard=0, batch_index=32)
                verified = stage('qual-'+hand+'-verify', 'verify-one', output/f'qual-{hand}-verify', hand=hand, shard=0, batch_index=32,
                                 original=output/f'qual-{hand}-produce')
                batches.append(dict(hand=hand, rows=producer['batch']['rows'], raw_state_sha256=producer['state_file_sha256'],
                                    verification_sha256=file_sha256(output/f'qual-{hand}-verify'/'verification.json')))
            value = dict(format='original_native_ik_default_batch_qualification_v1', batches=batches,
                         original_default_batch10096_qualified=True, full_frame_labels_regenerated=False)
            terminal_name = 'qualification.json'
        else:
            flow = output/'flow'; flow.mkdir(mode=0o700)
            for hand in HANDS:
                for shard in range(6):
                    prefix = flow/f'{hand}-shard-{shard}'; prefix.mkdir(mode=0o700)
                    stage(f'{hand}-{shard}-produce', 'produce', prefix/'produce', hand=hand, shard=shard)
                    stage(f'{hand}-{shard}-verify', 'verify', prefix/'verify', hand=hand, shard=shard, original=prefix/'produce')
                    stage(f'{hand}-{shard}-finalize', 'finalize', prefix/'labels', hand=hand, shard=shard,
                          original=prefix/'verify', flow=flow)
            labels = stage('assemble', 'assemble', output/'labels', flow=flow)
            value = dict(format='original_native_ik_complete_owned_workflow_v1', labels_result_sha256=file_sha256(output/'labels'/'labels.json'),
                rows_per_hand=labels['rows_per_hand'], total_rows=labels['total_rows'],
                actual_ik_optimizer_steps=labels['actual_ik_optimizer_steps'],
                actual_ik_closure_evaluations=labels['actual_ik_closure_evaluations'],
                original_default_batch10096_qualified=not config['engineering'],
                full_frame_labels_regenerated=not config['engineering'], engineering_all_rows_complete=config['engineering'])
            terminal_name = 'completion.json'
        if 'torch' in sys.modules: raise ValueError('native IK controller must remain Torch-free')
        if workflow_sources() != snapshot: raise ValueError('native IK source bytes changed after owned stages')
        guard()
        final = results.seal(dict(**value, actor=process_identity(), engineering=config['engineering'],
            configuration_sha256=file_sha256(config_path), sources_sha256=snapshot, physical_device=physical,
            input_identity_sha256=_hash(inputs.identity), source_identity_sha256=_hash(sources.identity),
            owned_stages=owned, all_owned_consumers_exited=True, actual_learned_pose_models_constructed=0,
            actual_scientific_model_training_updates=0, source_labels_replaced=False,
            regenerated_wrist_angles=False, parent_torch_unimported=True, seconds=time.monotonic()-start,
            scientific_eligible=False, full_paper_complete=False))
        results.write_json(output/terminal_name, final)
        guard(); return final
    finally:
        if lock is not None: lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('workflow', 'qualify', 'worker'))
    parser.add_argument('--output', required=True)
    for name in ('data', 'geometry', 'upstream', 'manotorch', 'chumpy', 'mano-assets', 'gpu-uuid', 'qualification',
                 'worker-action', 'config', 'hand', 'original', 'flow'):
        parser.add_argument('--'+name)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--batch-size', type=int, default=10096)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--minimum-free-bytes', type=int, default=200*1024**3)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--batch-index', type=int)
    args = parser.parse_args()
    if args.action == 'worker':
        value = _worker(args.worker_action, args.config, args.output, hand=args.hand, shard=args.shard,
                        original=args.original, flow=args.flow, batch_index=args.batch_index)
    else:
        config = {key: getattr(args, key) for key in REQUEST_KEYS}
        value = run_workflow(args.action, config, args.output)
    print(json.dumps(dict(action=args.action, sha256=value['sha256'])), flush=True)


if __name__ == '__main__': main()
