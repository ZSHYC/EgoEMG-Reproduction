"""Selected original EMGFormer weights on the complete shared-center grid.

Neither a short qualification nor one pass certifies a completed model result.
"""
from __future__ import annotations

from contextvars import ContextVar
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import numpy as np

from .classic_emg import _file_hash, _hash
from .classic_emg_cohort import _safe, _StatGuard

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_SHA = _file_hash(__file__)
FORMAT = 'selected_native_emgformer_shared_center_pass_v1'
INPUT_PROOF_SHA = '9ce29e7b51444cf9c280f0b8ba6e422931ebca27d71eb510a1e6d76fd6640810'
CATALOG = {
    'small': ('0d2d2d213557456f2b1d6a137765616d733a1f2db3b4be842fde74947a84cbf8',
        '24d8825b2b757e47822836142185d424cd6daaa0d1872a2aead1a6ca22b5bd0b', 169),
    'middle': ('d5e8622f145d38348d796ce2b9bb970c61228a767f57e51b894c21ba91bee3ca',
        'ad59d65a0b32d791b85b25979e0ac2d901b95eab28c97cbc8556cc3b7d894c40', 124),
    'large': ('cc813f3054fc53d9cd6c0a7def81cd16b20e7c81334b6a18dc7d176a33af01b9',
        'c621682640ad1e6a162df706e743ca1dfc24d4b06470f0cee8a28267096ca029', 142),
}
_READING = ContextVar('shared_center_pass_reading', default=frozenset())
_META = ('episode', 'start', 'end', 'center', 'hand', 'split', 'subject')


def _json(path, expected=None):
    path = _safe(path)
    guard = _StatGuard([path])
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('accepted source checksum differs: ' + path.name)
    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    guard.check()
    return value


def _sealed(value):
    if type(value) is not dict or value.get('sha256') != _hash({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('exact shared-center document seal required')
    return value


def _seal(value):
    return dict(value, sha256=_hash(value))


def guarded_source_files(expected):
    """Take generations before hashing, so a moving source is never adopted."""
    guard=_StatGuard([Path(p)for p in expected])
    if any(_file_hash(p)!=digest for p,digest in expected.items()):
        raise ValueError('accepted source checksum changed')
    guard.check()
    return guard


def _write(path, value, *, replace=False):
    path = Path(path)
    temporary = path.parent / ('.' + path.name + '.' + uuid.uuid4().hex)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _process():
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=fields[19], uid=os.getuid())


def _live(actor):
    if (type(actor) is not dict or set(actor) != {'pid','start_ticks','uid'}
            or type(actor['pid']) is not int or actor['pid'] <= 0
            or type(actor['uid']) is not int or actor['uid'] < 0
            or type(actor['start_ticks']) is not str or not actor['start_ticks'].isdigit()):
        raise ValueError('complete shared-center process identity required')
    path = Path('/proc', str(actor['pid']))
    try:
        fields = (path/'stat').read_text().rsplit(')',1)[1].split()
        return (fields[19] == actor['start_ticks'] and path.stat().st_uid == actor['uid']
            and fields[0] not in ('Z','X'))
    except (FileNotFoundError, ProcessLookupError):
        return False


def reference_source_indices(samples, scope):
    if type(samples) is not int or samples != 4282 or scope not in ('engineering','full'):
        raise ValueError('complete admitted4282 source and explicit reference scope required')
    return list(range(samples)) if scope == 'full' else [0,2140,2141,4281]


def source_fit_metadata(source_fit):
    """Bind actual accepted200-epoch metadata; do not deserialize weights."""
    fit_path = _safe(source_fit)
    identity = _json(fit_path/'identity.json')
    size = identity.get('policy', {}).get('size')
    if size not in CATALOG:
        raise ValueError('selected original native S/M/L fit required')
    report_sha, proof_sha, epoch = CATALOG[size]
    report_path = ROOT/'results/original-emgformer'/size/'metrics.json'
    proof_path = ROOT/('results/native-emgformer-'+size+'-complete-20261101.json')
    report = _sealed(_json(report_path, report_sha))
    proof = _json(proof_path, proof_sha)
    pins = proof['artifact_files_after_exit_sha256']
    paths = {'identity.json': fit_path/'identity.json', 'fit-result.json': fit_path/'fit-result.json',
        'epochs.jsonl': fit_path/'epochs.jsonl',
        'profile.json': fit_path.parent/'full-evaluation/profile/profile.json'}
    physical = {name: pins['private/full-fit/'+name] for name in ('identity.json','fit-result.json','epochs.jsonl')}
    physical['profile.json'] = pins['private/full-evaluation/profile/profile.json']
    guard = _StatGuard([*paths.values(), report_path, proof_path])
    if any(_file_hash(path) != physical[name] for name,path in paths.items()):
        raise ValueError('accepted complete source fit metadata checksum differs')
    result = _sealed(_json(paths['fit-result.json']))
    profile = _sealed(_json(paths['profile.json']))
    _sealed(identity)
    if (identity['policy']['scope'] != 'full' or result['scope'] != 'full'
            or result['completed_epochs'] != 200 or result['optimizer_executions'] != 25000
            or result['stage_complete'] is not True or result['full_training_terminated'] is not True
            or result['identity_sha256'] != identity['sha256']
            or result['sha256'] != report['fit_result_sha256']
            or result['selected_best'] != report['training']['selected_best']
            or profile['sha256'] != report['validation_profile_sha256']
            or profile['fit_result_sha256'] != result['sha256']
            or proof['full_training_epochs'] != 200 or proof['full_training_optimizer_updates'] != 25000):
        raise ValueError('engineering, selection or incomplete source cannot become full reference')
    records=[json.loads(line)for line in paths['epochs.jsonl'].read_text().splitlines()]
    if len(records)!=200:
        raise ValueError('complete selected source journal required')
    selected_state=records[epoch-1]['record']['model_sha256']
    if re.fullmatch('[0-9a-f]{64}',selected_state)is None:
        raise ValueError('selected journal model state required')
    guard.check()
    return dict(size=size, fit_path=str(fit_path), selected_best=copy.deepcopy(result['selected_best']),
        fit_result_sha256=result['sha256'], completed_epochs=200, optimizer_updates=25000,
        selected_epoch=epoch, learned_weights_loaded=False, report_file_sha256=report_sha,
        selected_model_state_sha256=selected_state,
        closure_proof_file_sha256=proof_sha, metadata_files_sha256=physical,
        _identity=identity, _paths={**{str(p):physical[n]for n,p in paths.items()},
            str(report_path):report_sha, str(proof_path):proof_sha})


def load_reference_runtime(meta, *, device, upstream=None):
    """Strict old trained factory/state restore, after physical weight admission."""
    current = source_fit_metadata(meta['fit_path'])
    if current != meta:
        raise ValueError('selected source fit identity changed')
    checkpoint = _safe(Path(meta['fit_path'])/'checkpoints'/meta['selected_best']['file'])
    guard = _StatGuard([checkpoint, *map(Path, meta['_paths'])])
    if _file_hash(checkpoint) != meta['selected_best']['checkpoint_sha256']:
        raise ValueError('selected trained checkpoint checksum differs')
    import torch
    from .emgformer_literal import compose_emgformer
    from .emgformer_runtime import build_emgformer_runtime, emgformer_state_sha256
    root = _safe(upstream or os.environ.get('EGOEMG_UPSTREAM', ROOT/'upstream'))
    packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (packet['emgformer_fit']['identity'] != meta['_identity']
            or packet['epoch'] != meta['selected_epoch']-1):
        raise ValueError('selected source checkpoint/epoch differs')
    state = packet['state_dict']
    expected = packet['emgformer_fit']['records'][-1]['model_sha256']
    if emgformer_state_sha256(state) != expected or expected!=meta['selected_model_state_sha256']:
        raise ValueError('selected named trained tensor state differs')
    prepared = build_emgformer_runtime(root, compose_emgformer(root,meta['size']),
        state=state, expected_initialization=meta['_identity']['construction'])
    if prepared.loaded_state_sha256 != expected or prepared.mode != 'strict_self_contained_restore':
        raise ValueError('strict selected tensor restore differs')
    prepared.module.to(device).eval()
    if any(x.dtype != torch.float32 for x in prepared.module.state_dict().values() if x.is_floating_point()):
        raise ValueError('original selected FP32 state required')
    guard.check()
    return prepared, expected


def validate_reference_sample(sample, record, source_index):
    """Match one actual request to the closed full-input byte commitments."""
    try:
        if (type(source_index) is not int or record['index'] != source_index
                or any(sample[k] != source_index for k in ('reference_sample_index','reference_unique_sample_index'))
                or sample['reference_repeat_index'] != 0 or sample['jitter_offset'] != 0
                or sample['role'] != 'test'):
            raise ValueError('same-center source request index differs')
        for name,shape,dtype in (('emg',(8,12000),np.float32),('joint_angles',(22,1),np.float32),
                ('label_valid_mask',(1,),np.bool_)):
            a=sample[name]
            if not isinstance(a,np.ndarray) or a.shape!=shape or a.dtype!=dtype or not np.isfinite(a).all():
                raise ValueError('finite typed same-center input required')
        meta={k:sample[k]for k in _META}
        if (sample['end']-sample['start'] != 12000 or sample['center'] != sample['start']+6000
                or sample['hand'] != ('left'if source_index<2141 else'right')):
            raise ValueError('same-center source hand/window differs')
        actual=dict(index=source_index,emg_sha256=hashlib.sha256(sample['emg'].tobytes()).hexdigest(),
            target_mask_sha256=hashlib.sha256(sample['joint_angles'].tobytes()+sample['label_valid_mask'].tobytes()).hexdigest(),
            identity_sha256=_hash(meta))
        if actual != record:
            raise ValueError('actual input/target/mask/private identity differs from admission')
        return _hash(record)
    except (KeyError,TypeError,IndexError,AttributeError) as error:
        raise ValueError('malformed same-center source request') from error


def reference_prediction(raw):
    import torch
    if (not torch.is_tensor(raw) or raw.dtype != torch.float32 or raw.ndim != 3
            or raw.shape[:2] != (1,22) or raw.shape[-1] <= 0 or not torch.isfinite(raw).all()):
        raise ValueError('finite FP32 single-request raw native22-angle prediction required')
    center=raw.shape[-1]//2
    return raw[...,center:center+1]


def predict_reference_dataset(model, dataset, records, indices, *, device, check=None, progress=None):
    """Run every requested native single-sample forward; sampling errors propagate."""
    import torch
    parts=[];identities=[];lengths=set()
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for local,source in enumerate(indices):
            if check is not None:check(local)
            sample=dataset[source];token=validate_reference_sample(sample,records[source],source)
            # This is the upstream center helper's ndarray conversion policy.
            batch={k:torch.from_numpy(v.copy()).float().to(device).unsqueeze(0)
                if isinstance(v,np.ndarray)else v for k,v in sample.items()}
            raw=model(batch);prediction=reference_prediction(raw);lengths.add(raw.shape[-1])
            part=dict(pred=prediction.detach().cpu().clone(),
                target=torch.from_numpy(sample['joint_angles'].copy()).unsqueeze(0),
                mask=torch.from_numpy(sample['label_valid_mask'].copy()).unsqueeze(0),
                index=torch.tensor([local],dtype=torch.int64),unique=torch.tensor([local],dtype=torch.int64),
                repeat=torch.tensor([0],dtype=torch.int64),center=torch.tensor([sample['center']],dtype=torch.int64),
                episode=torch.tensor([sample['episode']],dtype=torch.int64),hand=[sample['hand']],
                split=[sample['split']],user=[sample['subject']],sample_input_sha256=[token],
                input_sha256=_hash([records[source]]))
            parts.append(part);identities.append({k:sample[k]for k in _META})
            if progress is not None:progress(local+1)
    return dict(parts=parts,source_indices=list(indices),request_identities=identities),sorted(lengths)


def validate_reference_payload(payload, records, *, scope):
    import torch
    from .emgformer_eval import emgformer_statistics
    try:
        indices=reference_source_indices(len(records),scope)
        if (type(payload)is not dict or set(payload)!={'parts','source_indices','request_identities'}
                or payload['source_indices']!=indices or len(payload['parts'])!=len(indices)
                or len(payload['request_identities'])!=len(indices)):
            raise ValueError('complete declared reference payload required')
        for local,(source,part,meta)in enumerate(zip(indices,payload['parts'],payload['request_identities'])):
            if (type(meta)is not dict or set(meta)!=set(_META)
                    or part['pred'].dtype!=torch.float32 or part['target'].dtype!=torch.float32
                    or part['pred'].shape!=(1,22,1)or part['target'].shape!=(1,22,1)
                    or part['mask'].dtype!=torch.bool or part['mask'].shape!=(1,1)
                    or any(part[k].device.type!='cpu'for k in('pred','target','mask'))):
                raise ValueError('typed FP32 point prediction/target/mask required')
            if (meta['center']!=meta['start']+6000 or meta['end']-meta['start']!=12000
                    or int(part['center'][0])!=meta['center']or int(part['episode'][0])!=meta['episode']
                    or part['hand']!=[meta['hand']]or part['split']!=[meta['split']]or part['user']!=[meta['subject']]
                    or part['sample_input_sha256']!=[_hash(records[source])]
                    or part['input_sha256']!=_hash([records[source]])
                    or records[source]['index']!=source or _hash(meta)!=records[source]['identity_sha256']
                    or hashlib.sha256(part['target'][0].numpy().tobytes()+part['mask'][0].numpy().tobytes()).hexdigest()!=records[source]['target_mask_sha256']):
                raise ValueError('typed same-center source identity/target/mask differs')
        return emgformer_statistics(payload['parts'],samples=len(indices),batch_size=1,role='test')
    except (KeyError,TypeError,IndexError,AttributeError,RuntimeError)as error:
        raise ValueError('malformed same-center typed payload')from error


def validate_replay_identity(reference, candidate):
    keys=('scope','fit_result_sha256','input_admission_sha256','source_indices','model_before_sha256',
        'model_after_sha256','prediction_sha256','runtime_policy','sources_sha256')
    if (reference['kind']!='native' or candidate['kind']!='replay'
            or reference['process']==candidate['process'] or _live(reference['process'])
            or reference['finished_ns']>candidate['started_ns']
            or any(reference[k]!=candidate[k]for k in keys)):
        raise ValueError('closed distinct exact same-context native reference required')


def validate_reference_execution(value, source):
    policy=value['runtime_policy']
    numeric=('samples','batch_size','neural_forwards','optimizer_updates','optimizer_construction_attempts',
        'native_runtime_construction_calls','started_ns','finished_ns')
    if (any(type(value[k])is not int for k in numeric)
            or value['samples']<=0 or policy['precision']!='32-true'or policy['batch_size']!=1
            or policy['autocast_enabled']is not False or policy['matmul_precision']!='high'
            or policy['point_selector']!='raw[...,T//2:T//2+1]'
            or policy['device']not in('cpu','cuda:0')
            or any(type(policy[k])is not bool for k in('cuda_matmul_allow_tf32','cudnn_allow_tf32'))
            or (policy['device']=='cpu'and policy['gpu_uuid']is not None)
            or (policy['device']=='cuda:0'and(type(policy['gpu_uuid'])is not str
                or re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',policy['gpu_uuid'])is None))
            or type(value['cuda_initialized'])is not bool
            or value['cuda_initialized']is not (policy['device']=='cuda:0')
            or value['batch_size']!=1 or value['neural_forwards']!=value['samples']
            or value['optimizer_updates']!=0 or value['optimizer_construction_attempts']!=0
            or value['native_runtime_construction_calls']!=1
            or value['model_before_sha256']!=source['selected_model_state_sha256']
            or value['model_after_sha256']!=source['selected_model_state_sha256']
            or not 0<value['started_ns']<=value['finished_ns']
            or value['scientific_eligible']is not False or value['full_paper_complete']is not False):
        raise ValueError('selected state, actual counts or reference numerical policy differs')


def _admission_metadata(directory):
    directory=_safe(directory)
    proof_path=ROOT/'results/native-center-reference-input-admission/admission.json'
    proof=_json(proof_path,INPUT_PROOF_SHA)
    if not proof['complete_reference_input_validation']or not proof['independent_reader_physically_closed_after_exit0']:
        raise ValueError('complete independently closed4282 input admission required')
    paths={str(proof_path):INPUT_PROOF_SHA}
    for name,record in proof['actual_artifact_files'].items():paths[str(_safe(directory/name))]=record['sha256']
    guard=guarded_source_files(paths)
    complete=_json(directory/'private/completed.json');config=_json(directory/'config.json')
    for name,h in config['selected_new_source_sha256'].items():
        if _file_hash(ROOT/name)!=h:raise ValueError('admitted input implementation changed')
    records=_json(directory/'private/second/input-digests.json')
    if (not complete['complete_reference_input_validation']or len(records)!=4282
            or any(r['index']!=i for i,r in enumerate(records))
            or complete['passes']!=proof['passes']):
        raise ValueError('complete actual input admission identity differs')
    guard.check()
    return dict(path=str(directory),sha256=paths[str(directory/'private/completed.json')],
        config=config,records=records,grid_sha256=proof['passes'][0]['grid_sha256'],
        counts=proof['passes'][0]['counts'],source_files_sha256=proof['passes'][0]['source_files_sha256'],
        _paths=paths)


def _sources(upstream):
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('loaded reference evaluator changed')
    names=('center_reference_eval','center_reference_data','emgformer_eval','emgformer_runtime',
        'emgformer_literal','emg_initializer_data','classic_emg_train','classic_emg_cohort','metrics',
        'classic_emg','emgformer_resource')
    result={'egoemg_repro/'+n+'.py':_file_hash(ROOT/'egoemg_repro'/(n+'.py'))for n in names}
    from .center_reference_data import OFFICIAL
    result.update({'upstream/'+n:_file_hash(Path(upstream)/n)for n in OFFICIAL})
    return result


def _resources():
    if shutil.disk_usage('/data/shihao').free<225*1024**3:
        raise OSError('reference evaluation requires225GiB disk reserve')
    available=next(int(v.split()[1])for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:'))*1024
    if available<128*1024**3:raise OSError('reference evaluation requires128GiB available RAM')


def _gpu_snapshot():
    def query(flag):
        p=subprocess.run(['nvidia-smi',flag,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if p.returncode:raise RuntimeError('complete actual NVML observation unavailable')
        return [list(map(str.strip,line.split(',')))for line in p.stdout.splitlines()if line.strip()]
    devices=[dict(index=int(a),uuid=b,free_mib=int(c),total_mib=int(d))for a,b,c,d in query('--query-gpu=index,uuid,memory.free,memory.total')]
    apps=[dict(pid=int(a),uuid=b)for a,b in query('--query-compute-apps=pid,gpu_uuid')]
    return devices,apps


def _gpu_guard(gpu_uuid, *, idle=False):
    from .emgformer_resource import select_idle_emgformer_gpu
    devices,apps=_gpu_snapshot()
    if idle:
        if select_idle_emgformer_gpu(devices,apps,candidates=[gpu_uuid])is None:
            raise RuntimeError('selected GPU is not actually idle with64GiB available')
    else:
        target=next((d for d in devices if d['uuid']==gpu_uuid),None)
        if (target is None or target['free_mib']<8192
                or any(a['uuid']==gpu_uuid and a['pid']!=os.getpid()for a in apps)):
            raise RuntimeError('selected reference GPU resource/foreign-process guard failed')


def _literal_statistics(parts):
    result={}
    for hand in('left','right'):
        pairs=[(p['pred'][0,:,0].numpy(),p['target'][0,:,0].numpy())for p in parts
            if p['hand']==[hand]and bool(p['mask'][0,0])]
        if not pairs:
            result[hand]=dict(n_valid=0,per_joint=None,overall_mae_rad=None)
        else:
            pred=np.array([p for p,t in pairs]);target=np.array([t for p,t in pairs])
            joints=np.mean(np.abs(pred-target),axis=0)
            result[hand]=dict(n_valid=len(pairs),per_joint=joints.tolist(),overall_mae_rad=float(np.mean(joints)))
    return result


def _public_source(meta):
    return {k:v for k,v in meta.items()if not k.startswith('_')and k!='learned_weights_loaded'}


def run_reference_pass(source_fit, admitted_inputs, output, *, kind, reference=None,
        scope='full', device='cpu', gpu_uuid=None):
    """One actual complete selected-weight pass, or four explicit real probes."""
    import torch
    from .classic_emg_train import state_digest
    from .center_reference_data import reference_windows,CenterReferenceDataset
    from .emgformer_literal import compose_emgformer
    from .emgformer_runtime import emgformer_state_sha256
    if kind not in('native','replay')or (kind=='native')!=(reference is None):
        raise ValueError('native or matching fresh replay reference required')
    if device not in('cpu','cuda:0'):raise ValueError('explicit CPU or single visible CUDA device required')
    if device=='cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU reference requires explicitly hidden CUDA')
    else:
        if not isinstance(gpu_uuid,str)or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu_uuid:
            raise ValueError('explicit selected GPU UUID visibility binding required')
        _gpu_guard(gpu_uuid,idle=True);time.sleep(1);_gpu_guard(gpu_uuid,idle=True)
    _resources();started=time.time_ns();meta=source_fit_metadata(source_fit);inputs=_admission_metadata(admitted_inputs)
    indices=reference_source_indices(len(inputs['records']),scope);upstream=_safe(inputs['config']['upstream'])
    sources=_sources(upstream);ref=read_reference_pass(reference)if reference is not None else None
    if ref is not None and(ref['kind']!='native'or ref['scope']!=scope
            or ref['fit_result_sha256']!=meta['fit_result_sha256']or ref['input_admission_sha256']!=inputs['sha256']
            or ref['source_indices']!=indices or ref['finished_ns']>started):
        raise ValueError('replay source/scope/frozen population differs before construction')
    data=_safe(inputs['config']['data']);manifest=_json(data/'manifest.json',inputs['source_files_sha256']['metadata:manifest'])
    from .classic_emg_data import _safe_path
    data_paths={}
    for name,h in inputs['source_files_sha256'].items():
        if name=='metadata:manifest':p=data/'manifest.json'
        elif name=='metadata:episodes':p=data/'metadata.npz'
        else:
            key=name.removeprefix('field:');p=_safe_path(data,manifest['fields'][key].get('filename',key+'.dat'))
        data_paths[str(p)]=h
    checkpoint=Path(meta['fit_path'])/'checkpoints'/meta['selected_best']['file']
    expected={**meta['_paths'],**inputs['_paths'],**data_paths,
        str(checkpoint):meta['selected_best']['checkpoint_sha256']}
    expected.update({str(upstream/n.removeprefix('upstream/')if n.startswith('upstream/')else ROOT/n):h
        for n,h in sources.items()})
    guard=guarded_source_files(expected)
    output=_safe(output,exists=False);output.parent.mkdir(parents=True,exist_ok=True);output.mkdir(mode=0o700)
    _write(output/'started.json',dict(process=_process(),scope=scope,kind=kind,source_indices=indices,
        fit_result_sha256=meta['fit_result_sha256'],input_admission_sha256=inputs['sha256'],full_paper_complete=False))
    optimizer_init=torch.optim.Optimizer.__init__;attempts={'optimizer':0}
    def denied_optimizer(*args,**kwargs):
        attempts['optimizer']+=1;raise RuntimeError('reference prediction cannot construct an optimizer')
    torch.optim.Optimizer.__init__=denied_optimizer
    try:
        prepared,selected_hash=load_reference_runtime(meta,device=device,upstream=upstream)
        before=emgformer_state_sha256(prepared.module.state_dict())
        if before!=selected_hash:raise ValueError('actual selected model state differs before prediction')
        cfg=compose_emgformer(upstream,meta['size']);grid=reference_windows(data,upstream)
        if grid['sha256']!=inputs['grid_sha256']:raise ValueError('complete reference grid changed')
        dataset=CenterReferenceDataset(data,upstream,cfg,grid)
        policy=dict(device=device,gpu_uuid=gpu_uuid,precision='32-true',batch_size=1,
            autocast_enabled=False,point_selector='raw[...,T//2:T//2+1]',
            matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
        if policy['matmul_precision']!='high':raise ValueError('official native matmul policy changed')
        if ref is not None and policy!=ref['runtime_policy']:
            raise ValueError('replay actual device/numerical policy differs')
        def check(local):
            guard.check()
            if local%32==0:
                _resources()
                if device!='cpu':_gpu_guard(gpu_uuid)
        def progress(count):
            if count%32==0:_write(output/'progress.json',dict(forwards=count,target_forwards=len(indices),
                process=_process(),full_paper_complete=False),replace=True)
        payload,lengths=predict_reference_dataset(prepared.module.model,dataset,inputs['records'],indices,
            device=device,check=check,progress=progress)
        statistics=validate_reference_payload(payload,inputs['records'],scope=scope)
        literal=_literal_statistics(payload['parts']);after=emgformer_state_sha256(prepared.module.state_dict())
        if after!=before or attempts['optimizer']!=0:raise ValueError('reference prediction mutated selected state')
        prepared.check()
        guard.check()
        temporary=output/('.predictions.'+uuid.uuid4().hex)
        with temporary.open('xb')as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,output/'predictions.pt');temporary.unlink()
        value=_seal(dict(format=FORMAT,scope=scope,kind=kind,size=meta['size'],source_fit_path=meta['fit_path'],
            source_fit=_public_source(meta),fit_result_sha256=meta['fit_result_sha256'],
            input_admission_path=inputs['path'],input_admission_sha256=inputs['sha256'],grid_sha256=grid['sha256'],
            source_indices=indices,samples=len(indices),batch_size=1,raw_prediction_steps=lengths,
            runtime_policy=policy,sources_sha256=sources,statistics=statistics,
            official_float32_by_hand=literal,payload_sha256=_file_hash(output/'predictions.pt'),
            prediction_sha256=state_digest(payload),model_before_sha256=before,model_after_sha256=after,
            native_runtime_construction_calls=1,optimizer_construction_attempts=attempts['optimizer'],
            neural_forwards=len(indices),optimizer_updates=0,process=_process(),started_ns=started,finished_ns=time.time_ns(),
            cuda_initialized=torch.cuda.is_initialized(),reference_path=None if ref is None else str(_safe(reference)),
            reference_sha256=None if ref is None else ref['sha256'],scientific_eligible=False,full_paper_complete=False))
        validate_reference_execution(value,meta)
        if ref is not None:validate_replay_identity(ref,value)
        if sources!=_sources(upstream):raise ValueError('reference execution sources changed')
        guard.check();_write(output/'pass.json',value);return value
    except BaseException as error:
        _write(output/'failed.json',dict(error=type(error).__name__+': '+str(error),process=_process(),
            partial_output_not_accepted=True,full_paper_complete=False))
        raise
    finally:
        torch.optim.Optimizer.__init__=optimizer_init


def read_reference_pass(output, *, require_closed=True):
    """Read typed original artifacts and source selection without a model."""
    import torch
    from .classic_emg_train import state_digest
    output=_safe(output);seen=_READING.get()
    if str(output)in seen or len(seen)>=4:raise ValueError('cyclic reference artifact graph')
    token=_READING.set(seen|{str(output)})
    try:
        pass_guard=_StatGuard([output/'pass.json'])
        value=_sealed(_json(output/'pass.json'))
        pass_bytes_sha256=_file_hash(output/'pass.json')
        pass_guard.check()
        if value['format']!=FORMAT or value['kind']not in('native','replay'):
            raise ValueError('explicit completed reference pass required')
        live=_live(value['process'])
        if require_closed and live:raise ValueError('reference producer still live')
        meta=source_fit_metadata(value['source_fit_path']);inputs=_admission_metadata(value['input_admission_path'])
        upstream=_safe(inputs['config']['upstream'])
        if (value['source_fit']!=_public_source(meta)or value['fit_result_sha256']!=meta['fit_result_sha256']
                or value['input_admission_sha256']!=inputs['sha256']or value['grid_sha256']!=inputs['grid_sha256']
                or value['source_indices']!=reference_source_indices(len(inputs['records']),value['scope'])
                or value['samples']!=len(value['source_indices'])or value['sources_sha256']!=_sources(upstream)):
            raise ValueError('reference complete source/selection/population differs')
        checkpoint=_safe(Path(meta['fit_path'])/'checkpoints'/meta['selected_best']['file'])
        validate_reference_execution(value,meta)
        path=output/'predictions.pt'
        guard=guarded_source_files({**meta['_paths'],**inputs['_paths'],
            str(checkpoint):meta['selected_best']['checkpoint_sha256'],str(path):value['payload_sha256'],
            str(output/'pass.json'):pass_bytes_sha256})
        payload=torch.load(path,map_location='cpu',weights_only=True)
        statistics=validate_reference_payload(payload,inputs['records'],scope=value['scope'])
        if (state_digest(payload)!=value['prediction_sha256']or statistics!=value['statistics']
                or _literal_statistics(payload['parts'])!=value['official_float32_by_hand']):
            raise ValueError('reference actual typed prediction/metrics differ')
        if value['kind']=='replay':
            if value['reference_path']is None:raise ValueError('completed native reference required')
            reference=read_reference_pass(value['reference_path'])
            if reference['sha256']!=value['reference_sha256']:raise ValueError('reference source pass changed')
            validate_replay_identity(reference,value)
        elif value['reference_path']is not None or value['reference_sha256']is not None:
            raise ValueError('native reference cannot consume a prior replay')
        guard.check()
        pass_guard.check()
        return value
    finally:
        _READING.reset(token)
