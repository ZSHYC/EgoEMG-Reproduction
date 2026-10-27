"""Owned ordinary-file EMG2Pose experiments; parent never imports Torch.

The original fit and inference CLIs execute in separate owned processes.
Full experiments retain the complete source,250epochs and both declared
evaluation populations. Engineering completion never becomes paper accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from . import emg2pose_data as storage,emg2pose_memory_archive as memory
from .emg2pose_direct_source import open_source
from .emg2pose_memory_process import run_owned_stage,_atomic,process_identity,same_process

FORMAT='emg2pose_direct_experiment_driver_v1'
MODES=('official_literal','metadata_indices')
SIZES=('small','middle','large')
_ROOT=Path(__file__).parent
_CORE=('emg2pose_direct_experiment.py','emg2pose_direct_source.py','emg2pose_direct_runtime.py',
    'emg2pose_direct_train.py','emg2pose_direct_recovery.py','emg2pose_direct_evaluate.py',
    'emg2pose_direct_evaluation_data.py','emg2pose_direct_report.py','emg2pose_memory_train.py',
    'emg2pose_memory_recovery.py','emg2pose_memory_evaluate.py','emg2pose_memory_evaluation_data.py',
    'emg2pose_memory_statistics.py','emg2pose_memory_report.py','classic_emg_train.py','emg2pose_check.py',
    'emg2pose_memory_process.py')
_IMPORTED={n:storage._sha256(_ROOT/n) for n in _CORE}
_HASH=re.compile('[0-9a-f]{64}')

def _hash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _seal(v):return dict(v,sha256=_hash(v))
def _read(p):return storage._json(storage._safe_path(p,file=True))
def _sources():
    actual={n:storage._sha256(_ROOT/n) for n in _CORE}
    if actual!=_IMPORTED:raise ValueError('imported experiment implementation changed')
    return actual
def _check(v):
    if v.get('format')!=FORMAT or v.get('sha256')!=_hash({k:x for k,x in v.items() if k!='sha256'}):
        raise ValueError('experiment document format or seal differs')
def _expected(path,expected):
    if type(expected) is not str or not _HASH.fullmatch(expected) or storage._sha256(path)!=expected:
        raise ValueError('frozen input SHA256 differs: '+Path(path).name)
def _write(path,v,*,replace=False):
    path=Path(path)
    if not replace and path.exists():raise FileExistsError('experiment artifact already exists')
    storage._space(path.parent,2*1024**2);_atomic(path,v)
def _alive(p):return same_process(p) and process_identity(p['pid'])['state']!='Z'

def stage_plan(size,scope='full'):
    if type(size) is not str or size not in SIZES:raise ValueError('known native model size required')
    if type(scope) is not str or scope not in ('full','cpu_engineering','cuda_engineering'):
        raise ValueError('explicit full or engineering scope required')
    names=['admission','fit',*[mode+'-'+name for mode in MODES
        for name in ('val-native','val-replay','profile','test-native','test-replay')],'export','verify']
    return [dict(name=n,cuda=scope!='cpu_engineering' and (n=='fit' or n.endswith(('-native','-replay')))) for n in names]

def _gpu(uuid):
    p=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=name,uuid,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    fields=[s.strip() for s in p.stdout.strip().split(',')]
    if len(fields)!=3 or fields[1]!=uuid or not fields[2].isdigit():raise ValueError('actual selected GPU identity unavailable')
    return dict(name=fields[0],uuid=fields[1],free_bytes=int(fields[2])*1024**2)

def _full_source(d):
    if (d.get('cohort')!='full' or d.get('full_corpus_complete') is not True or d.get('engineering_only') is not False
            or d.get('file_count')!=25253 or d.get('archive_bytes')!=462824048640
            or d.get('counts',{}).get('split')!={'train':17136,'val':1950,'test':6167}):
        raise ValueError('full source requires complete25253original sessions and original roles')

def _input_files(q):
    return [(Path(q[key])/name,q['expected_'+key+'_sha256']) for key,name in
        (('qualification','fit-result.json'),('evaluation_qualification','completion.json'),
         ('controller_qualification','driver-completion.json')) if q[key] is not None]

def _request(path,expected=None):
    if expected is not None:_expected(path,expected)
    q=_read(path);_check(q);stage_plan(q['size'],q['scope'])
    if q['sources_sha256']!=_sources():raise ValueError('frozen experiment producers differ')
    for p,h in _input_files(q):_expected(p,h)
    return q

def _source(q):
    return open_source(q['source_root'],expected_completion_sha256=q['expected_source_completion_sha256'],
        upstream=q['upstream'],metadata_csv=q['metadata_csv'])

def _admission(q,*,seen=()):
    # This runs in an owned CPU worker; native checkpoint readers may import
    # Torch there. The NoTorch parent only handles JSON and resource observation.
    from . import emg2pose_direct_runtime as runtime
    with _source(q) as source:
        d=source.data_identity
        if q['scope']=='full':_full_source(d)
        cfg=runtime.compose_config(q['upstream'],q['size'],source.index,source=source,metadata_csv=q['metadata_csv'])
        identity=runtime.runtime_identity(q['upstream'],cfg,source=source)
    qualifications={}
    if q['scope']!='cpu_engineering':
        from . import emg2pose_direct_train as fit,emg2pose_direct_report as report
        qualified=fit.read_qualification(q['qualification'],q['expected_qualification_sha256'])
        if ({k:v for k,v in qualified['runtime'].items() if k!='data'}!={k:v for k,v in identity.items() if k!='data'}
                or qualified['profile']['device_name']!=q['gpu_name']):
            raise ValueError('same-model native training qualification/runtime/hardware differs')
        _expected(Path(q['evaluation_qualification'])/'completion.json',q['expected_evaluation_qualification_sha256'])
        value=report.read_report(q['evaluation_qualification'],require_full=False)
        if value['scope']!='cuda_engineering' or value['size']!=q['size']:
            raise ValueError('same-model actual CUDA evaluation qualification required')
        evidence=_read(Path(q['evaluation_qualification'])/'evidence.json')
        for mode in MODES:
            directory=Path(evidence['evaluations'][mode]['directory'])
            for name in ('val-native','val-replay','test-native','test-replay'):
                plan=_read(directory/name/'plan.json');actual=_read(directory/name/'identity.json')
                if ([p['samples'] for p in plan['batches']]!=[320,1] or actual['policy']['batch_size']!=320
                        or actual['policy']['workers']!=12 or actual['policy']['precision']!='32-true'
                        or actual['inference_runtime']['hardware']['name']!=q['gpu_name']
                        or actual['inference_runtime']['hardware']['visible_device']!=q['gpu_uuid']):
                    raise ValueError('actual321window320+1FP32/12worker/GPU qualification differs')
        qualifications=dict(training_identity_sha256=qualified['sha256'],
            evaluation_completion_sha256=q['expected_evaluation_qualification_sha256'])
        if q['scope']=='full':
            prior=read_experiment(q['controller_qualification'],expected_completion_sha256=q['expected_controller_qualification_sha256'],
                require_full=False,_seen=seen)
            prior_request=_request(Path(q['controller_qualification'])/'request.json')
            if prior['scope']!='cuda_engineering' or prior_request['gpu_uuid']!=q['gpu_uuid']:
                raise ValueError('actual CUDA shared controller qualification required')
            qualifications['controller_completion_sha256']=q['expected_controller_qualification_sha256']
    return _seal(dict(format=FORMAT,document='admission',scope=q['scope'],size=q['size'],
        sources_sha256=_sources(),runtime_identity_sha256=_hash(identity),source_completion_sha256=q['expected_source_completion_sha256'],
        qualifications=qualifications,scientific_complete=False,full_paper_complete=False))

def _evaluations(q):
    output=Path(q['output'])
    return {mode:dict(directory=output/mode,owned={name:output/'stages'/(mode+'-'+name)
        for name in ('val-native','val-replay','test-native','test-replay')}) for mode in MODES}

def _command(q,name):
    output=Path(q['output']);common=['--upstream',q['upstream'],'--source-root',q['source_root'],
        '--expected-source-completion-sha256',q['expected_source_completion_sha256'],'--metadata-csv',q['metadata_csv'],
        '--size',q['size'],'--scope',q['scope']]
    if name=='fit':
        argv=[sys.executable,'-m','egoemg_repro.emg2pose_direct_train',*common,'--output',str(output/'fit')]
        if q['scope']=='full':argv+=['--qualification',q['qualification'],'--expected-qualification-sha256',q['expected_qualification_sha256']]
        return argv
    for mode in MODES:
        if name.startswith(mode+'-') and not name.endswith('-profile'):
            role,kind=name[len(mode)+1:].split('-');directory=output/mode
            argv=[sys.executable,'-m','egoemg_repro.emg2pose_direct_evaluate','pass',*common,'--fit',str(output/'fit'),
                '--selection',mode,'--role',role,'--kind',kind,'--output',str(directory/(role+'-'+kind))]
            if kind=='replay':argv+=['--reference',str(directory/(role+'-native')),
                '--expected-reference-sha256',storage._sha256(directory/(role+'-native/pass.json'))]
            if role=='test':argv+=['--profile',str(directory/'profile'),
                '--expected-profile-sha256',storage._sha256(directory/'profile/profile.json')]
            return argv
    return [sys.executable,'-m','egoemg_repro.emg2pose_direct_experiment','stage','--request',str(output/'request.json'),
        '--expected-request-sha256',storage._sha256(output/'request.json'),'--name',name]

def _stage(request,expected,name):
    q=_request(request,expected);output=Path(q['output']);full=q['scope']=='full'
    if name=='admission':_write(output/'admission.json',_admission(q,seen=(output,)));return
    from . import emg2pose_direct_evaluate as evaluation,emg2pose_direct_report as report
    if name.endswith('-profile'):
        mode=name[:-len('-profile')]
        if mode not in MODES:raise ValueError('known profile phase required')
        directory=output/mode
        evaluation.freeze_evaluation_profile(directory/'val-native',directory/'val-replay',directory/'profile',
            expected_native_sha256=storage._sha256(directory/'val-native/pass.json'),
            expected_replay_sha256=storage._sha256(directory/'val-replay/pass.json'),require_full=full,
            native_owned=output/'stages'/(mode+'-val-native'),replay_owned=output/'stages'/(mode+'-val-replay'))
    elif name=='export':
        selected=_read(output/'selected-fit.json');_check(selected)
        with _source(q) as source:
            report.export_report(q['upstream'],source,q['metadata_csv'],q['size'],output/'fit',output/'stages/fit',
                _evaluations(q),output/'report',scope=q['scope'],expected_fit_sha256=selected['fit_result_sha256'])
    elif name=='verify':
        report.read_report(output/'report',require_full=full)
        if _admission(q,seen=(output,))!=_read(output/'admission.json'):raise ValueError('frozen qualification admission changed')
    else:raise ValueError('known CPU experiment phase required')
    _request(request,expected)

def _owned(path,controller):
    v=_read(Path(path)/'stage-result.json')
    if (v.get('format')!='egoemg_owned_stage_v1' or v.get('status')!='success'
            or type(v.get('worker_exit_code')) is not int or v['worker_exit_code']!=0
            or v.get('all_owned_consumers_exited') is not True or v.get('descriptor_bindings')!={}
            or v.get('scientific_complete') is not False
            or any(v.get('owner',{}).get(k)!=controller[k] for k in ('pid','start_ticks','uid','sid','pgrp'))):
        raise ValueError('actual successful experiment-owned exit required')
    if any(_alive(v[k]) for k in ('worker','guardian')):raise ValueError('owned experiment consumers still live')
    return v

def run_experiment(upstream,source_root,metadata_csv,size,output,*,expected_source_completion_sha256,scope='full',
        gpu_uuid=None,qualification=None,expected_qualification_sha256=None,evaluation_qualification=None,
        expected_evaluation_qualification_sha256=None,controller_qualification=None,expected_controller_qualification_sha256=None,
        deadline_seconds=None,on_progress=None):
    plan=stage_plan(size,scope)
    if 'torch' in sys.modules:raise ValueError('experiment parent must not import Torch')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('experiment parent must hide CUDA')
    if 'EGOEMG_STAGE_FD_ARCHIVE' in os.environ:raise ValueError('experiment does not accept an inherited archive descriptor')
    if on_progress is not None and not callable(on_progress):raise ValueError('progress callback must be callable')
    if deadline_seconds is not None and (type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds) or deadline_seconds<=0):
        raise ValueError('deadline must be finite and positive')
    pairs=((qualification,expected_qualification_sha256),(evaluation_qualification,expected_evaluation_qualification_sha256),
        (controller_qualification,expected_controller_qualification_sha256))
    if scope=='cpu_engineering':
        if gpu_uuid is not None or any(p is not None or h is not None for p,h in pairs):
            raise ValueError('CPU engineering has no GPU or full qualifications')
    else:
        if type(gpu_uuid) is not str or not re.fullmatch(r'GPU-[0-9a-fA-F-]{16,}',gpu_uuid):raise ValueError('one explicit GPU UUID required')
        for i,(p,h) in enumerate(pairs):
            required=i<2 or scope=='full'
            if required and (p is None or type(h) is not str or not _HASH.fullmatch(h)):raise ValueError('separately frozen qualifications required')
            if not required and (p is not None or h is not None):raise ValueError('CUDA engineering does not consume a controller qualification')
    output=storage._safe_path(output)
    if output.exists():raise FileExistsError('experiment output already exists')
    q=dict(format=FORMAT,document='request',upstream=str(storage._safe_path(upstream)),source_root=str(storage._safe_path(source_root)),
        metadata_csv=str(storage._safe_path(metadata_csv,file=True)),size=size,output=str(output),scope=scope,
        expected_source_completion_sha256=expected_source_completion_sha256,gpu_uuid=gpu_uuid,gpu_name=None,
        qualification=str(storage._safe_path(qualification)) if qualification is not None else None,expected_qualification_sha256=expected_qualification_sha256,
        evaluation_qualification=str(storage._safe_path(evaluation_qualification)) if evaluation_qualification is not None else None,
        expected_evaluation_qualification_sha256=expected_evaluation_qualification_sha256,
        controller_qualification=str(storage._safe_path(controller_qualification)) if controller_qualification is not None else None,
        expected_controller_qualification_sha256=expected_controller_qualification_sha256,
        deadline_seconds=deadline_seconds if deadline_seconds is not None else (None if scope=='full' else 7200),
        sources_sha256=_sources(),controller=process_identity())
    for other in (q['upstream'],q['source_root'],q['qualification'],q['evaluation_qualification'],q['controller_qualification']):
        if other is not None:
            other=Path(other)
            if output==other or output in other.parents or other in output.parents:raise ValueError('output overlaps immutable source/qualification')
    started=time.monotonic();stages=[];frozen_fit=None
    with _source(q) as source:
        d=source.data_identity
        if scope=='full':_full_source(d)
        if any(d['counts']['split'].get(k,0)<=0 for k in ('train','val','test')):
            raise ValueError('three independent original source roles required')
        def guard():
            source.assert_current();_sources()
            if memory.available_memory_bytes()<memory.RESERVE_BYTES:raise RuntimeError('observed256GiB host reserve lost')
            storage._space(output if output.exists() else output.parent)
            if q['deadline_seconds'] is not None and time.monotonic()-started>=q['deadline_seconds']:raise RuntimeError('experiment deadline reached')
            if gpu_uuid is not None and _gpu(gpu_uuid)['free_bytes']<8*1024**3:raise RuntimeError('observed8GiB GPU reserve lost')
            for p,h in _input_files(q):_expected(p,h)
            if output.exists():
                _expected(output/'request.json',request_sha)
                if frozen_fit is not None:_expected(output/'fit/fit-result.json',frozen_fit)
        if gpu_uuid is not None:q['gpu_name']=_gpu(gpu_uuid)['name']
        guard();q=_seal(q);output.mkdir(mode=0o700);(output/'stages').mkdir()
        for mode in MODES:(output/mode).mkdir(mode=0o700)
        _write(output/'request.json',q);request_sha=storage._sha256(output/'request.json')
        try:
            for p in plan:
                name=p['name'];guard()
                if p['cuda'] and _gpu(gpu_uuid)['free_bytes']<64*1024**3:raise RuntimeError('prelaunch64GiB GPU margin required')
                _write(output/'driver-status.json',dict(status='running',name=name,completed_stages=len(stages),scientific_complete=False),replace=True)
                value=run_owned_stage(_command(q,name),output/'stages'/name,descriptors={},guard=guard,
                    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid if p['cuda'] else ''),poll_seconds=1.,terminate_grace=5.)
                _owned(output/'stages'/name,q['controller'])
                stages.append(dict(name=name,owned_sha256=storage._sha256(output/'stages'/name/'stage-result.json')))
                if name=='fit':
                    frozen_fit=storage._sha256(output/'fit/fit-result.json')
                    _write(output/'selected-fit.json',_seal(dict(format=FORMAT,fit_result_sha256=frozen_fit,fit_owned_sha256=stages[-1]['owned_sha256'])))
                if on_progress is not None:on_progress(dict(name=name,completed_stages=len(stages),scope=scope))
            guard()
            if 'torch' in sys.modules:raise ValueError('experiment parent imported Torch')
            for p in stages:_owned(output/'stages'/p['name'],q['controller'])
            result=_seal(dict(format=FORMAT,document='driver-completion',scope=scope,size=size,request_sha256=request_sha,
                admission_sha256=storage._sha256(output/'admission.json'),selected_fit_sha256=storage._sha256(output/'selected-fit.json'),
                report_completion_sha256=storage._sha256(output/'report/completion.json'),stages=stages,seconds=time.monotonic()-started,
                parent_torch_unimported=True,all_owned_consumers_exited=True,requires_controller_exit=True,
                sources_sha256=_sources(),scientific_complete=scope=='full',full_paper_complete=False))
            _write(output/'driver-status.json',dict(status='complete',completed_stages=14,scientific_complete=scope=='full'),replace=True)
            _write(output/'driver-completion.json',result);return result
        except BaseException as error:
            (output/'driver-completion.json').unlink(missing_ok=True)
            _write(output/'driver-status.json',dict(status='failed',completed_stages=len(stages),error_type=type(error).__name__,
                error=str(error),scientific_complete=False),replace=True)
            raise

def read_experiment(output,*,expected_completion_sha256=None,require_full=True,_seen=()):
    if type(require_full) is not bool:raise ValueError('explicit full admission policy required')
    output=storage._safe_path(output)
    if output in _seen or len(_seen)>4:raise ValueError('cyclic experiment qualification')
    seen=(*_seen,output)
    if expected_completion_sha256 is not None:_expected(output/'driver-completion.json',expected_completion_sha256)
    v=_read(output/'driver-completion.json');_check(v);q=_request(output/'request.json',v['request_sha256'])
    if require_full and q['scope']!='full':raise ValueError('engineering driver is not a full experiment')
    expected_names=[p['name'] for p in stage_plan(q['size'],q['scope'])]
    if (v['scope']!=q['scope'] or v['size']!=q['size'] or v['sources_sha256']!=_sources()
            or v['parent_torch_unimported'] is not True or v['requires_controller_exit'] is not True
            or v['all_owned_consumers_exited'] is not True or v['scientific_complete'] is not (q['scope']=='full')
            or v['full_paper_complete'] is not False or [p['name'] for p in v['stages']]!=expected_names):
        raise ValueError('complete native driver policy/evidence differs')
    if _alive(q['controller']):raise ValueError('experiment controller is still live')
    for p in v['stages']:
        _expected(output/'stages'/p['name']/'stage-result.json',p['owned_sha256']);_owned(output/'stages'/p['name'],q['controller'])
    for filename,key in (('admission.json','admission_sha256'),('selected-fit.json','selected_fit_sha256'),('report/completion.json','report_completion_sha256')):
        _expected(output/filename,v[key])
    selected=_read(output/'selected-fit.json');_check(selected)
    _expected(output/'fit/fit-result.json',selected['fit_result_sha256'])
    if _admission(q,seen=seen)!=_read(output/'admission.json'):raise ValueError('current source/qualification admission differs')
    from .emg2pose_direct_report import read_report
    report=read_report(output/'report',require_full=q['scope']=='full');evidence=_read(output/'report/evidence.json')
    if (report['scope']!=q['scope'] or report['size']!=q['size'] or report['scientific_complete'] is not v['scientific_complete']
            or evidence['fit_path']!=str(output/'fit') or evidence['fit_owned']!=str(output/'stages/fit')
            or evidence['source_root']!=q['source_root'] or evidence['source_completion_sha256']!=q['expected_source_completion_sha256']
            or report['training']['fit_receipt_sha256']!=selected['fit_result_sha256']):
        raise ValueError('driver and actual final report differ')
    return v

def main(argv=None):
    if 'EGOEMG_STAGE_FD_ARCHIVE' in os.environ:raise ValueError('experiment does not accept an inherited archive descriptor')
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run')
    for n in ('upstream','source-root','metadata-csv','size','output','expected-source-completion-sha256'):run.add_argument('--'+n,required=True)
    run.add_argument('--scope',default='full');run.add_argument('--gpu-uuid');run.add_argument('--deadline-seconds',type=float)
    for n in ('qualification','evaluation-qualification','controller-qualification'):
        run.add_argument('--'+n);run.add_argument('--expected-'+n+'-sha256')
    stage=sub.add_parser('stage')
    for n in ('request','expected-request-sha256','name'):stage.add_argument('--'+n,required=True)
    verify=sub.add_parser('verify');verify.add_argument('--output',required=True);verify.add_argument('--expected-completion-sha256')
    verify.add_argument('--allow-engineering',action='store_true')
    a=vars(p.parse_args(argv));command=a.pop('command')
    if command=='run':
        v=run_experiment(**a,on_progress=lambda r:print(json.dumps(r,sort_keys=True),flush=True))
        print(json.dumps(dict(scope=v['scope'],stages=len(v['stages']),scientific_complete=v['scientific_complete'])))
    elif command=='stage':_stage(a['request'],a['expected_request_sha256'],a['name'])
    else:
        v=read_experiment(a['output'],expected_completion_sha256=a['expected_completion_sha256'],require_full=not a['allow_engineering'])
        print(json.dumps(dict(scope=v['scope'],stages=len(v['stages']),scientific_complete=v['scientific_complete'])))

if __name__=='__main__':main()
