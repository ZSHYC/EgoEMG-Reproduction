"""Owned original-batch qualification, then a fresh native150 RN18 fusion fit.

This controller imports no ML. Each trained-source read, fit, recovery, native
evaluation, replay and report verification runs in its own owned process. It is
launched only after the full trained CPU handoff and whole-family review close;
preparing this entry point does not grant either prerequisite or a GPU permit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .classic_emg_cohort import _safe,_json,_StatGuard
from .emg2pose_memory_process import process_identity,same_process,run_owned_stage

REQUEST_KEYS={'data','crops','upstream','cohort_receipt','branches','handoff_request'}
FORMAT='original_rn18_fusion_qualified_full_experiment_v1'
IMPORTED_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _locators(request,output):
    if type(request)is not dict or set(request)!=REQUEST_KEYS:raise ValueError('exact RN18 experiment locators required')
    q=copy.deepcopy(request)
    for value in [*q.values(),output]:
        if not isinstance(value,(str,Path))or not Path(value).is_absolute()or '..'in Path(value).parts:
            raise ValueError('absolute nontraversing experiment paths required')
    return {k:str(v)for k,v in q.items()},Path(output)


def rn18_fusion_experiment_stages(request,output):
    """CLI graph only: it never constructs a model or admits trained weights."""
    q,out=_locators(request,output);stages=[]
    def add(name,argv,cuda=False):stages.append(dict(name=name,argv=argv,cuda=cuda))
    add('trained-reopen',['-m','egoemg_repro.rn18_fusion_workflow','reopen','--request',q['handoff_request'],
        '--output',str(out/'trained-reopen.json')])
    fitbase=['-m','egoemg_repro.rn18_fusion_train','fit',*[item for k in ('data','crops','upstream','cohort_receipt','branches')for item in ('--'+k.replace('_','-'),q[k])]]
    for family in ('qual','full'):
        fit=out/(family+'-fit');evaluation=out/(family+'-evaluation');engineering=family=='qual'
        if engineering:
            add('qual-prefix',[*fitbase,'--output',str(fit),'--scope','cuda_engineering','--stop-after-epoch','1'],True)
            add('qual-resume',[*fitbase,'--output',str(fit),'--scope','cuda_engineering','--resume'],True)
        else:add('full-fit',[*fitbase,'--output',str(fit),'--scope','full'],True)
        add(family+'-accept',['-m','egoemg_repro.rn18_fusion_experiment','accept','--fit',str(fit),
            '--scope','cuda_engineering'if engineering else'full'])
        for role in ('val','test'):
            for kind in ('native','replay'):
                args=['-m','egoemg_repro.rn18_fusion_eval','pass',*[item for k in ('data','crops','upstream')for item in ('--'+k,q[k])],
                    '--fit',str(fit),'--output',str(evaluation/(role+'-'+kind)),'--role',role,'--kind',kind]
                if engineering:args+=['--engineering']
                if kind=='replay':args+=['--reference',str(evaluation/(role+'-native'))]
                if role=='test':args+=['--profile',str(evaluation/'profile')]
                add(family+'-'+role+'-'+kind,args,True)
            if role=='val':add(family+'-freeze',['-m','egoemg_repro.rn18_fusion_eval','freeze',
                '--reference',str(evaluation/'val-native'),'--replay',str(evaluation/'val-replay'),'--output',str(evaluation/'profile')])
        suffix=['--engineering']if engineering else[]
        add(family+'-report',['-m','egoemg_repro.rn18_fusion_eval','report','--fit',str(fit),
            '--evaluation',str(evaluation/'test-native'),'--replay',str(evaluation/'test-replay'),'--output',str(evaluation/'report'),*suffix])
        add(family+'-verify',['-m','egoemg_repro.rn18_fusion_eval','verify','--recompute',str(evaluation/'test-replay'),
            '--output',str(evaluation/'report'),*suffix])
    return stages


def _check_fit_result(value,*,full):
    """Additional orchestration assertions; only the native reader admits fits."""
    epochs,updates,samples=(150,46950,62470)if full else(2,4,400)
    try:
        if(value['scope']!=('full'if full else'cuda_engineering')or value['stage_complete']is not True
            or value['full_training_terminated']is not full or value['scientific_eligible']is not False
            or type(value['completed_epochs'])is not int or value['completed_epochs']!=epochs
            or type(value['optimizer_executions'])is not int or value['optimizer_executions']!=updates
            or value['train_samples_per_epoch']!=[samples]*epochs or value['val_samples_per_epoch']!=[1052]*epochs
            or not full and(value['resume_validation_exact']is not True
                or value['resume_start_exact']!={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')})):
            raise ValueError('actual native fusion budget, complete population or six-state recovery differs')
    except(KeyError,TypeError)as exc:raise ValueError('complete actual native fusion fit result required')from exc


def accept_rn18_fusion_fit(fit,*,full):
    from .rn18_fusion_train import read_rn18_fusion_fit
    value=read_rn18_fusion_fit(fit,require_full=full);_check_fit_result(value,full=full)
    return dict(fit_sha256=value['sha256'],completed_epochs=value['completed_epochs'],
        optimizer_executions=value['optimizer_executions'],full_training_terminated=full,scientific_eligible=False,full_paper_complete=False)


def _run_stages(stages,output,*,guard,environment,after_stage,before_stage=None):
    output=Path(output);output.mkdir(mode=0o700);owned=[]
    for stage in stages:
        if before_stage is not None:before_stage(stage)
        guard()
        if '--output'in stage['argv']:
            target=_safe(stage['argv'][stage['argv'].index('--output')+1],exists=False)
            target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        env=dict(environment)
        env['CUDA_VISIBLE_DEVICES']=environment['CUDA_VISIBLE_DEVICES']if stage['cuda']else''
        result=run_owned_stage([sys.executable,'-u','-B',*stage['argv']],output/('owned-'+stage['name']),
            descriptors={},guard=guard,environment=env,poll_seconds=.5,terminate_grace=10.)
        if(result['status']!='success'or result['worker_exit_code']!=0 or result['all_owned_consumers_exited']is not True
            or any(same_process(result[k])for k in ('worker','guardian'))):
            raise RuntimeError('owned RN18 experiment stage failed: '+stage['name'])
        owned.append(result);guard();after_stage(stage)
    return owned


def _gpu_snapshot(gpu_uuid):
    row=subprocess.check_output(['nvidia-smi','-i',gpu_uuid,'--query-gpu=uuid,name,memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
    if len(row)!=3 or row[0].strip()!=gpu_uuid or row[1].strip()!='NVIDIA A800-SXM4-80GB':
        raise ValueError('the assigned actual A800 GPU identity differs')
    apps=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines():
        pid,uuid=(x.strip()for x in line.split(','))
        if uuid==gpu_uuid:apps.append(int(pid))
    return int(row[2])*1024**2,apps


def _write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def run_rn18_fusion_experiment(request,output,*,gpu_uuid):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or 'torch'in sys.modules:
        raise ValueError('ML-free experiment controller requires CUDA hidden before startup')
    if type(gpu_uuid)is not str or re.fullmatch(r'GPU-[0-9a-f-]{36}',gpu_uuid)is None:
        raise ValueError('one explicit assigned GPU UUID required')
    q,out=_locators(request,output);out=_safe(out,exists=False)
    if out.exists():raise ValueError('fresh experiment output required; no automatic retry or overwrite')
    for name in ('data','crops','upstream','cohort_receipt','branches','handoff_request'):
        p=_safe(q[name])
        if p==out or out in p.parents or p in out.parents:raise ValueError('experiment output overlaps its immutable inputs')
    code=Path(__file__).resolve().parent
    source_files=sorted(code.glob('rn18_fusion_*.py'))
    pins=_StatGuard([*source_files,*[Path(q[k])for k in ('cohort_receipt','branches','handoff_request')]])
    hashes={str(p.relative_to(code.parent)):hashlib.sha256(p.read_bytes()).hexdigest()for p in source_files}
    if hashes.get(str(Path(__file__).resolve().relative_to(code.parent)))!=IMPORTED_SHA:
        raise ValueError('imported experiment source changed before dispatch')
    handoff=_json(q['handoff_request']);branches=_json(q['branches'])
    if(any(handoff[k]!=q[k]for k in ('data','crops','upstream','cohort_receipt'))or branches!=dict(
        vision_fit=handoff['vision_fit'],emg_fit=handoff['emg_fit'],output=handoff['admission'],
        vision_source_root=handoff['vision_source_root'],emg_source_root=handoff['emg_source_root'])):
        raise ValueError('experiment must use the same full trained handoff and complete cohort')
    pins.check();out.mkdir(mode=0o700);stages=rn18_fusion_experiment_stages(q,out)
    _write(out/'experiment-plan.json',dict(stages=stages,source_files_sha256=hashes,scientific_eligible=False,full_paper_complete=False))
    parent=process_identity(os.getppid());phase=None;observed={};qualification_accepted=False;started=time.monotonic()
    def guard():
        pins.check()
        if 'torch'in sys.modules or not same_process(parent):raise RuntimeError('ML-free controller or original parent identity lost')
        if shutil.disk_usage(out).free<25*1024**3:raise RuntimeError('experiment disk reserve below25GiB')
        available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
        if available<128*1024**3:raise RuntimeError('experiment available host memory below128GiB')
        if phase is not None and phase['cuda']:
            free,apps=_gpu_snapshot(gpu_uuid)
            if free<8*1024**3:raise RuntimeError('assigned GPU reserve below8GiB')
            path=out/'stages'/('owned-'+phase['name'])/'started.json';worker=_json(path)['worker']if path.exists()else None
            for pid in apps:
                if worker is None or pid!=worker['pid']or not same_process(worker):
                    raise RuntimeError('foreign GPU arrival; abort only this owned stage')
                observed[phase['name']]=dict(worker=worker,actual_nvml_pid=pid,gpu_uuid=gpu_uuid)
    def before(stage):
        nonlocal phase
        phase=None;guard()
        if stage['name']=='full-fit'and not qualification_accepted:raise ValueError('full fit requires completed independent qualification report')
        if stage['cuda']:
            stable=None
            while True:
                guard();free,apps=_gpu_snapshot(gpu_uuid);now=time.monotonic()
                stable=(now if stable is None else stable)if not apps and free>=64*1024**3 else None
                if stable is not None and now-stable>=20:break
                time.sleep(2)
        phase=stage
    def after(stage):
        nonlocal qualification_accepted
        if stage['cuda']and stage['name']not in observed:raise RuntimeError('no actual owned NVML GPU observation')
        if stage['name']=='qual-prefix':
            v=_json(out/'qual-fit/fit-result.json')
            if(v['scope']!='cuda_engineering'or v['completed_epochs']!=1 or v['optimizer_executions']!=2
                or v['stage_complete']is not False or v['full_training_terminated']is not False):raise ValueError('actual native one-epoch qualification prefix required')
            _write(out/'qualification-prefix-result.json',v)
        if stage['name']in ('qual-verify','full-verify'):
            full=stage['name']=='full-verify';directory=out/('full-evaluation'if full else'qual-evaluation')/'report'
            pins.add(directory/'metrics.json');pins.add(directory/'completion.json');report=_json(directory/'metrics.json')
            if(report['scientific_eligible']is not full or report['full_paper_complete']is not False
                or any(report['test'][k]!=v for k,v in dict(consumed_samples=2792,unique_samples=2792,duplicate_consumptions=0).items())):
                raise ValueError('complete original fusion test population or final report eligibility differs')
            if not full:qualification_accepted=True
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid,PYTHONDONTWRITEBYTECODE='1')
    owned=_run_stages(stages,out/'stages',guard=guard,environment=env,before_stage=before,after_stage=after)
    phase=None;guard();report=_json(out/'full-evaluation/report/metrics.json')
    result=dict(format=FORMAT,actual_owned_stages=owned,actual_nvml_observations=observed,
        qualified_native_batch=200,qualified_native_workers=10,actual_qualification_completed=True,
        formal_fusion_epochs=150,formal_optimizer_updates=46950,actual_full_report=report,
        source_files_sha256=hashes,all_owned_consumers_exited=True,parent_torch_unimported=True,
        scientific_eligible=True,full_paper_complete=False,seconds=time.monotonic()-started)
    pins.check();_write(out/'workflow-result.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run');run.add_argument('--request',required=True,type=Path);run.add_argument('--output',required=True,type=Path)
    run.add_argument('--gpu-uuid',required=True)
    accept=sub.add_parser('accept');accept.add_argument('--fit',required=True,type=Path);accept.add_argument('--scope',choices=('full','cuda_engineering'),required=True)
    args=parser.parse_args()
    value=accept_rn18_fusion_fit(args.fit,full=args.scope=='full')if args.action=='accept'else run_rn18_fusion_experiment(_json(args.request),args.output,gpu_uuid=args.gpu_uuid)
    print(json.dumps({k:v for k,v in value.items()if k in ('format','scientific_eligible','full_paper_complete','completed_epochs','optimizer_executions')}))


if __name__=='__main__':main()
