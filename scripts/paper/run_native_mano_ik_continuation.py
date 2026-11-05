"""Owned continuation of original IK batches with explicit mixed lineage.

Old packets remain read-only. New optimization, mixed-map persistence, fresh
geometry, shard finalization and bilateral assembly have separate owned actors.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))

from egoemg_repro.classic_emg_cohort import _safe
from egoemg_repro.data import file_sha256
from egoemg_repro.emg2pose_memory_process import process_identity,same_process,run_owned_stage
from egoemg_repro.native_ik_inputs import _hash,_Guard,HANDS
from egoemg_repro import native_ik_results as results
from egoemg_repro.native_ik_continuation import admit_ik_prefix,write_continued_shard,read_continued_shard
from scripts.paper import run_native_mano_ik as original

IMPORTED_SHA=file_sha256(__file__)
ALLOWED_GPUS={
    'GPU-3b5a6250-aa89-83c4-f375-844893f374f5':2,
    'GPU-20575323-8e23-b0b8-74b4-b987d870b0e3':3,
    'GPU-07ce4138-60af-137b-b134-7f267cece52d':4,
    'GPU-9dcbac35-9a85-6e75-e412-bf4dd590578a':5,
    'GPU-6b41c390-53fc-6ccc-ea07-0648926ae5cb':6,
    'GPU-83b7beb5-2729-4e3b-ed78-b949192152fb':7}
SOURCE_NAMES=original.SOURCE_NAMES+['egoemg_repro/native_ik_continuation.py',
    'scripts/paper/run_native_mano_ik_continuation.py',
    'results/native-mano-ik/interrupted-prefix-raw-verified-20261104.json','AGENTS.md']


def validate_request(configuration):
    original.validate_request('workflow',configuration)
    if configuration['device']=='cuda:0'and configuration['gpu_uuid']not in ALLOWED_GPUS:
        raise ValueError('user allows only physical GPU2-7; GPU0/1 are prohibited')
    if not configuration['engineering']:
        if configuration['minimum_free_bytes']<225*1024**3:raise ValueError('full continuation requires225GiB storage reserve')
    return configuration


def workflow_sources():
    if file_sha256(__file__)!=IMPORTED_SHA:raise ValueError('imported continuation runner changed')
    return {name:file_sha256(_safe(ROOT/name))for name in SOURCE_NAMES}


def require_allowed_physical_gpu(uuid,physical_index):
    if (uuid not in ALLOWED_GPUS or type(physical_index)is not int
        or ALLOWED_GPUS[uuid]!=physical_index or physical_index not in range(2,8)):
        raise ValueError('same actually allowed physical GPU2-7 UUID/index required')


def _physical_index(uuid):
    raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    rows=[[v.strip()for v in line.split(',')]for line in raw.splitlines()if line.strip()]
    if len(rows)!=1 or len(rows[0])!=2 or rows[0][1]!=uuid:
        raise ValueError('actual physical GPU index/UUID required')
    index=int(rows[0][0]);require_allowed_physical_gpu(uuid,index);return index


def require_scientific_request(configuration,old_request):
    operational={'minimum_free_bytes'}
    if not configuration['engineering']:operational|={'gpu_uuid','qualification'}
    if (set(configuration)!=set(old_request)
        or any(old_request[k]!=v for k,v in configuration.items()if k not in operational)):
        raise ValueError('original continued scientific request/source paths must remain identical')


def require_qualified_physical_backend(configuration,qualification,actual_physical):
    if not configuration['engineering']and actual_physical!=qualification.get('physical_device'):
        raise ValueError('same newly qualified physical GPU and driver/backend record required')


def bind_current_qualification_configuration(configuration,permit,physical,bindings):
    """Bind permit provenance; this check alone grants no qualification."""
    path=Path(configuration['qualification'])/'configuration.json'
    qualified,frozen,sha=original._read(path);bindings.guards.append(frozen)
    wanted=dict(configuration,qualification=None);actual=qualified.get('request')
    if (configuration['engineering']or qualified.get('action')!='qualify'
        or type(actual)is not dict or set(actual)!=set(wanted)
        or any(actual[k]!=v for k,v in wanted.items()if k!='minimum_free_bytes')
        or permit.get('configuration_sha256')!=sha
        or any(qualified.get(k)!=permit.get(k)for k in ('input_identity_sha256','source_identity_sha256','sources_sha256'))):
        raise ValueError('current qualification must bind its actual same-source/GPU creation configuration')
    original.validate_request('qualify',actual)
    require_qualified_physical_backend(configuration,qualified,physical)
    bindings.add_file(path,sha);bindings.check();return sha


def admit_current_backend(configuration,inputs,sources,prefix,physical,physical_index):
    """Authenticate a new permit without replacing historic prefix provenance."""
    validate_request(configuration);inputs.check();prefix.check()
    from egoemg_repro.native_ik_continuation import _state
    old=_state(prefix);history=old['original_configuration'];runtime=prefix.identity['runtime_identity']
    if configuration['engineering']is not prefix.identity['engineering']:
        raise ValueError('engineering prefix cannot authorize a formal backend')
    if sources is not None:
        sources.check()
        if _hash(sources.identity)!=old['audit']['original_native_source_identity_sha256']:
            raise ValueError('same original scientific source/assets required on new card')
    if configuration['device']=='cuda:0':
        require_allowed_physical_gpu(configuration['gpu_uuid'],physical_index)
        if type(physical)is not dict or physical.get('uuid')!=configuration['gpu_uuid']:
            raise ValueError('current physical backend UUID differs from request')
    elif physical is not None or physical_index is not None:
        raise ValueError('CPU engineering cannot claim physical CUDA hardware')
    bindings=results.IKArtifactBindings();qualification_sha=qualification_configuration_sha=None
    if configuration['engineering']:
        if physical!=history.get('physical_device'):
            raise ValueError('engineering retains its actual original physical backend')
    else:
        if sources is None or inputs.identity.get('engineering')is not False:
            raise ValueError('actual full source providers required for current qualification')
        qualification_sha,bindings,actual_runtime=original._check_qualification(
            configuration,inputs,sources,physical,original.workflow_sources())
        permit,frozen,sha=original._read(Path(configuration['qualification'])/'qualification.json')
        bindings.guards.append(frozen)
        if sha!=qualification_sha:raise ValueError('current qualification changed after authentication')
        require_qualified_physical_backend(configuration,permit,physical)
        qualification_configuration_sha=bind_current_qualification_configuration(configuration,permit,physical,bindings)
        original.check_qualified_runtime(actual_runtime,dict(action='workflow',request=configuration,qualified_runtime_identity=runtime))
    record=dict(format='original_IK_separate_current_qualification_and_historical_prefix_backend_v1',
        engineering=configuration['engineering'],physical_device=copy.deepcopy(physical),physical_index=physical_index,
        original_prefix_physical_device=copy.deepcopy(history.get('physical_device')),
        original_prefix_qualification_file_sha256=history.get('qualification_file_sha256'),
        qualification_file_sha256=qualification_sha,qualification_configuration_file_sha256=qualification_configuration_sha,
        qualified_runtime_identity=copy.deepcopy(runtime),
        original_default_batch10096_qualified=not configuration['engineering'],scientific_eligible=False,full_paper_complete=False)
    bindings.check(rehash=True);prefix.check();return record,bindings


def read_closed_stage(directory,owner):
    """Authenticate real started/terminal role bindings before reading outputs."""
    directory=_safe(directory);bindings=results.IKArtifactBindings();values=[]
    for name in ('started.json','stage-result.json'):
        path=directory/name;bindings.add_file(path);values.append(json.loads(path.read_text()));bindings.check()
    started,result=values
    if (result.get('status')!='success'or type(result.get('worker_exit_code'))is not int
        or result['worker_exit_code']!=0 or result.get('all_owned_consumers_exited')is not True
        or any(not original._same_actor(x.get('owner',{}),owner)for x in values)
        or any(not original._same_actor(started.get(k,{}),result.get(k,{}))for k in ('worker','guardian'))
        or original._same_actor(result.get('worker',{}),result.get('guardian',{}))):
        raise ValueError('actual successful closed IK stage role bindings required')
    for key in ('worker','guardian'):results.require_closed(result[key])
    bindings.check();return result,bindings


def _admit_document(document):
    cfg=document['request'];validate_request(cfg);inputs,sources=original._admit(cfg)
    prefix=admit_ik_prefix(inputs,document['original_root'],proof=document['prefix_proof'],engineering=cfg['engineering'])
    from egoemg_repro.native_ik_continuation import _state
    require_scientific_request(cfg,_state(prefix)['original_configuration']['request'])
    physical=original._gpu_record(cfg['gpu_uuid'])if cfg['device']=='cuda:0'else None
    index=_physical_index(cfg['gpu_uuid'])if physical is not None else None
    backend,backend_bindings=admit_current_backend(cfg,inputs,sources,prefix,physical,index)
    if (_hash(inputs.identity)!=document['input_identity_sha256']
        or _hash(sources.identity)!=document['source_identity_sha256']
        or _hash(prefix.identity)!=document['prefix_identity_sha256']
        or backend!=document['backend_binding']or physical!=document['physical_device']
        or backend['qualified_runtime_identity']!=document['qualified_runtime_identity']):
        raise ValueError('fresh continuation source/input/prefix admission differs')
    return inputs,sources,prefix,backend_bindings


def _worker(action,config_path,output,*,hand=None,shard=None):
    document,frozen,config_sha=original._read(config_path);cfg=document['request'];validate_request(cfg)
    source_guard=_Guard([ROOT/name for name in SOURCE_NAMES],[])
    if document['continuation_sources_sha256']!=workflow_sources():raise ValueError('continued worker source snapshot differs')
    inputs,sources,prefix,backend_bindings=_admit_document(document);output=_safe(output,exists=False)
    if output.exists():raise FileExistsError(output)
    flow=Path(config_path).parent/'flow';branch=flow/f'{hand}-shard-{shard}';stage_bindings=results.IKArtifactBindings()
    def guard():
        frozen.check();source_guard.check();inputs.check();sources.check();prefix.check();stage_bindings.check();backend_bindings.check()
        if not same_process(document['controller_actor']):raise ValueError('continuation controller owner exited')
    def stage_receipt(role):
        receipt,binding=read_closed_stage(Path(config_path).parent/f'owned-{hand}-{shard}-{role}',document['controller_actor'])
        stage_bindings.absorb(binding);return receipt
    def mapped():
        stage=stage_receipt('map')
        return read_continued_shard(inputs,branch/'map',configuration_sha256=config_sha,
            receipt_actor=stage['worker'],prefix=prefix)
    runtime=None
    if action in ('produce','verify'):
        torch=original._runtime(cfg,verification=action=='verify');runtime=sources.build(device=cfg['device'])
        if runtime.identity!=document['qualified_runtime_identity']:
            raise ValueError('same actual original qualified geometry/runtime backend required')
        if cfg['device']=='cuda:0':
            if original._gpu_record(cfg['gpu_uuid'])!=document['physical_device']:
                raise ValueError('actual continued physical GPU differs')
            if torch.cuda.get_device_properties(0).total_memory//1024**2<document['physical_device']['total_memory_mib']-1024:
                raise ValueError('visible Torch GPU differs from admitted physical card')
    if action=='produce':
        first=len(prefix.batch_prefix(hand,shard));count=inputs.identity['batches_per_hand'][shard]
        output.mkdir(parents=True,mode=0o700);records=[]
        for index in range(first,count):
            result=results.produce_ik_batch(results.select_ik_batch(inputs,hand,shard,index),runtime,
                output/f'batch-{index:05d}',guard=guard)
            records.append(dict(batch_index=index,result_sha256=file_sha256(output/f'batch-{index:05d}'/'result.json'),
                state_sha256=result['state_file_sha256'],optimizer_steps=result['record']['actual_ik_optimizer_steps'],
                closure_evaluations=result['record']['actual_ik_closure_evaluations']))
            print(json.dumps(dict(phase='produced-missing-original-batch',hand=hand,shard=shard,batch_index=index)),flush=True)
        value=results.seal(dict(format='continued_IK_new_packets_only_v1',actor=process_identity(),
            configuration_sha256=config_sha,hand=hand,shard=shard,first_batch_index=first,entries=records,
            full_frame_labels_regenerated=False,scientific_eligible=False,full_paper_complete=False))
        results.write_json(output/'new-packets.json',value)
    elif action=='map':
        stage=stage_receipt('produce');pending,pending_guard,_=original._read(branch/'produce/new-packets.json')
        stage_bindings.guards.append(pending_guard)
        first=len(prefix.batch_prefix(hand,shard));count=inputs.identity['batches_per_hand'][shard]
        if (pending.get('format')!='continued_IK_new_packets_only_v1'or pending['configuration_sha256']!=config_sha
            or pending['hand']!=hand or pending['shard']!=shard or pending['first_batch_index']!=first
            or not original._same_actor(pending['actor'],stage['worker'])
            or [e['batch_index']for e in pending['entries']]!=list(range(first,count))):
            raise ValueError('same actually closed new producer packets required')
        for record in pending['entries']:
            p=branch/'produce'/f"batch-{record['batch_index']:05d}"
            stage_bindings.add_file(p/'result.json',record['result_sha256']);stage_bindings.add_file(p/'state.npz',record['state_sha256'])
        value=write_continued_shard(inputs,prefix,hand=hand,shard=shard,new_root=branch/'produce',
            new_producer_actor=stage['worker'],configuration_sha256=config_sha,output=output)
    elif action=='verify':
        mapping=mapped();output.mkdir(parents=True,mode=0o700);records=[]
        for entry in mapping.entries:
            index=entry['batch_index'];destination=output/f'batch-{index:05d}'
            result=results.verify_ik_batch(inputs,runtime,entry['directory'],destination,guard=guard)
            mapping.check()
            records.append(dict(batch_index=index,verification_sha256=file_sha256(destination/'verification.json'),
                producer_actor=result['producer_actor']))
            print(json.dumps(dict(phase='verified-original-or-new-batch',hand=hand,shard=shard,batch_index=index,reused=entry['reused'])),flush=True)
        value=results.seal(dict(format='continued_IK_all_mixed_packets_geometry_verified_v1',actor=process_identity(),
            configuration_sha256=config_sha,map_file_sha256=mapping.identity['map_file_sha256'],
            hand=hand,shard=shard,entries=records,actual_ik_optimizer_steps=0,
            full_frame_labels_regenerated=False,scientific_eligible=False,full_paper_complete=False))
        mapping.check(rehash=True)
        results.write_json(output/'verified.json',value)
    elif action=='finalize':
        mapping=mapped();stage=stage_receipt('verify');verified,verified_guard,_=original._read(branch/'verify/verified.json')
        stage_bindings.guards.append(verified_guard)
        if (verified.get('format')!='continued_IK_all_mixed_packets_geometry_verified_v1'
            or verified['configuration_sha256']!=config_sha or verified['hand']!=hand or verified['shard']!=shard
            or verified['map_file_sha256']!=mapping.identity['map_file_sha256']
            or not original._same_actor(verified['actor'],stage['worker'])
            or len(verified['entries'])!=len(mapping.entries)):
            raise ValueError('complete actually closed mixed geometry verifier required')
        pairs=[]
        for entry,record in zip(mapping.entries,verified['entries']):
            index=entry['batch_index'];p=branch/'verify'/f'batch-{index:05d}'
            receipt,receipt_guard,sha=original._read(p/'verification.json');stage_bindings.guards.append(receipt_guard)
            if (record['batch_index']!=index or record['verification_sha256']!=sha
                or record['producer_actor']!=entry['producer_actor']
                or not original._same_actor(receipt['actor'],stage['worker'])):
                raise ValueError('per-packet fresh verifier actor/hash differs')
            pairs.append((entry['directory'],p))
        value=results.finalize_ik_shard(inputs,hand,shard,pairs,output,guard=guard);mapping.check(rehash=True)
    elif action=='assemble':
        value=results.assemble_ik_labels(inputs,[flow/f'{h}-shard-{s}'/'labels'for h in HANDS for s in range(6)],output,guard=guard)
    else:raise ValueError('explicit continued IK worker action required')
    guard();return value


def run_continued_workflow(configuration,original_root,prefix_proof,output):
    validate_request(configuration);cfg=dict(configuration);output=_safe(output,exists=False)
    if output.exists():raise FileExistsError(output)
    source_guard=_Guard([ROOT/name for name in SOURCE_NAMES],[]);snapshot=workflow_sources();source_guard.check()
    inputs,sources=original._admit(cfg);prefix=admit_ik_prefix(inputs,original_root,proof=prefix_proof,engineering=cfg['engineering'])
    from egoemg_repro.native_ik_continuation import _state
    require_scientific_request(cfg,_state(prefix)['original_configuration']['request'])
    if _hash(sources.identity)!=_state(prefix)['audit']['original_native_source_identity_sha256']:
        raise ValueError('original actual geometry source/assets must remain identical')
    results._new_output(inputs,output)
    if output.is_relative_to(_safe(original_root)):raise ValueError('old workflow must remain read-only')
    if shutil.disk_usage(output.parent).free<cfg['minimum_free_bytes']:raise ValueError('continued IK initial storage reserve unavailable')
    physical=original._gpu_record(cfg['gpu_uuid'])if cfg['device']=='cuda:0'else None
    index=_physical_index(cfg['gpu_uuid'])if physical is not None else None
    backend,backend_bindings=admit_current_backend(cfg,inputs,sources,prefix,physical,index)
    output.mkdir(parents=True,mode=0o700);owner=process_identity();started=time.monotonic()
    document=results.seal(dict(format='original_native_IK_continued_configuration_v1',action='workflow',request=cfg,
        controller_actor=owner,original_root=str(_safe(original_root)),prefix_proof=str(_safe(prefix_proof)),
        input_identity_sha256=_hash(inputs.identity),source_identity_sha256=_hash(sources.identity),
        prefix_identity_sha256=_hash(prefix.identity),scientific_sources_sha256=original.workflow_sources(),
        continuation_sources_sha256=snapshot,physical_device=physical,qualified_runtime_identity=backend['qualified_runtime_identity'],backend_binding=backend))
    config_path=output/'configuration.json';results.write_json(config_path,document);config_guard=_Guard([config_path],[])
    stage_paths=[];owned=[];artifacts=results.IKArtifactBindings();artifacts.absorb(backend_bindings);lock=None
    def guard(*,check_gpu=True):
        source_guard.check();config_guard.check();inputs.check();sources.check();prefix.check();artifacts.check()
        reserve=cfg['minimum_free_bytes']if not cfg['engineering']else 15*1024**3
        if shutil.disk_usage(output).free<reserve:raise ValueError('continued IK storage headroom fell below required reserve')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'))
        if available<(128 if not cfg['engineering']else 2)*1024**3:raise ValueError('continued IK host-memory headroom unavailable')
        if physical is not None:
            require_allowed_physical_gpu(cfg['gpu_uuid'],_physical_index(cfg['gpu_uuid']))
            if original._gpu_record(cfg['gpu_uuid'])!=physical:raise ValueError('qualified continued GPU hardware changed')
            if check_gpu:
                sessions=[]
                for p in stage_paths:
                    if (p/'started.json').is_file():
                        launched=json.loads((p/'started.json').read_text())
                        if not original._same_actor(launched['owner'],owner):raise ValueError('continued GPU stage owner changed')
                        if same_process(launched['worker']):sessions.append(launched['worker']['sid'])
                for pid in original._gpu_applications(cfg['gpu_uuid']):
                    actor=process_identity(pid)
                    if actor is not None and actor['sid']not in sessions:
                        raise ValueError('foreign GPU consumer appeared; stop only owned continuation stage')
    def stage(hand,shard,role,target):
        role_name='assemble'if role=='assemble'else f'{hand}-{shard}-{role}';stage_root=output/('owned-'+role_name);stage_paths.append(stage_root)
        argv=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--worker-action',role,'--config',str(config_path),'--output',str(target)]
        if role!='assemble':argv+=['--hand',hand,'--shard',str(shard)]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUDA_VISIBLE_DEVICES=cfg['gpu_uuid']if physical is not None and role in ('produce','verify')else '')
        receipt=run_owned_stage(argv,stage_root,descriptors={},guard=guard,environment=env,poll_seconds=2.,terminate_grace=10.)
        checked,binding=read_closed_stage(stage_root,owner);artifacts.absorb(binding)
        if checked!=receipt:raise ValueError('actual continued stage receipt changed')
        name=dict(produce='new-packets.json',map='mixed.json',verify='verified.json',finalize='shard.json',assemble='labels.json')[role]
        result,frozen,_=original._read(target/name);artifacts.guards.append(frozen)
        actor=result.get('receipt_actor')if role=='map'else result.get('actor')
        if not original._same_actor(actor,receipt['worker']):raise ValueError('actual continued output actor differs from owned worker')
        if role=='finalize':
            _,bound=results.bind_ik_shard(inputs,target);artifacts.absorb(bound)
        if role=='assemble':
            _,bound=results.inspect_ik_labels(inputs,target);artifacts.absorb(bound)
        owned.append(dict(role=role_name,**receipt));guard();return result
    try:
        if physical is not None:
            lock_path=output.parent/('.native-ik-'+cfg['gpu_uuid']+'.lock')
            if lock_path.is_symlink():raise ValueError('continued GPU lock cannot be a symlink')
            lock=lock_path.open('a+b')
            while True:
                guard(check_gpu=False)
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
                except BlockingIOError:locked=False
                if locked:
                    free=subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                    if not original._gpu_applications(cfg['gpu_uuid'])and int(free.strip())>=64*1024:break
                    fcntl.flock(lock,fcntl.LOCK_UN)
                print(json.dumps(dict(phase='waiting-original-qualified-idle-GPU',gpu_uuid=cfg['gpu_uuid'],reused_batches=prefix.identity['reused_batches'])),flush=True)
                time.sleep(30.)
        guard();flow=output/'flow';flow.mkdir(mode=0o700);new_steps=new_closures=new_batches=0
        for hand in HANDS:
            for shard in range(6):
                branch=flow/f'{hand}-shard-{shard}';branch.mkdir(mode=0o700)
                produced=stage(hand,shard,'produce',branch/'produce')
                new_batches+=len(produced['entries']);new_steps+=sum(e['optimizer_steps']for e in produced['entries']);new_closures+=sum(e['closure_evaluations']for e in produced['entries'])
                stage(hand,shard,'map',branch/'map');stage(hand,shard,'verify',branch/'verify');stage(hand,shard,'finalize',branch/'labels')
        labels=stage(None,None,'assemble',output/'labels');guard();artifacts.check(rehash=True);prefix.check(rehash=True)
        if 'torch'in sys.modules:raise ValueError('continued IK parent must remain Torch-free')
        value=results.seal(dict(format='original_native_IK_complete_mixed_owned_workflow_v1',actor=owner,backend_binding=backend,
            configuration_sha256=file_sha256(config_path),continuation_sources_sha256=snapshot,
            scientific_sources_sha256=original.workflow_sources(),prefix_identity=prefix.identity,
            labels_result_sha256=file_sha256(output/'labels/labels.json'),rows_per_hand=labels['rows_per_hand'],total_rows=labels['total_rows'],
            reused_batches=prefix.identity['reused_batches'],reused_rows=prefix.identity['reused_rows'],new_batches=new_batches,
            new_actual_ik_optimizer_steps=new_steps,new_actual_ik_closure_evaluations=new_closures,
            total_actual_ik_optimizer_steps=labels['actual_ik_optimizer_steps'],total_actual_ik_closure_evaluations=labels['actual_ik_closure_evaluations'],
            owned_stages=owned,all_owned_consumers_exited=True,source_labels_replaced=False,regenerated_wrist_angles=False,
            original_default_batch10096_qualified=not cfg['engineering'],full_frame_labels_regenerated=not cfg['engineering'],
            engineering_all_rows_complete=cfg['engineering'],parent_torch_unimported=True,seconds=time.monotonic()-started,
            scientific_eligible=False,full_paper_complete=False))
        results.write_json(output/'completion.json',value);guard();return value
    finally:
        if lock is not None:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('workflow','worker'))
    parser.add_argument('--output',required=True)
    for name in ('original-root','prefix-proof','data','geometry','upstream','manotorch','chumpy','mano-assets','gpu-uuid','qualification','config','worker-action','hand'):
        parser.add_argument('--'+name)
    parser.add_argument('--shard',type=int);parser.add_argument('--engineering',action='store_true')
    parser.add_argument('--batch-size',type=int,default=10096);parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--minimum-free-bytes',type=int,default=225*1024**3);args=parser.parse_args()
    if args.action=='worker':value=_worker(args.worker_action,args.config,args.output,hand=args.hand,shard=args.shard)
    else:value=run_continued_workflow({key:getattr(args,key)for key in original.REQUEST_KEYS},args.original_root,args.prefix_proof,args.output)
    print(json.dumps(dict(action=args.action,sha256=value['sha256'])),flush=True)


if __name__=='__main__':main()
