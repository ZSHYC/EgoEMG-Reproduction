"""Selected-best original coupled fusion evaluation and pretest exact replay calibration.

Private passes retain typed predictions, original generated masks and input
fingerprints. A fresh validation replay freezes the fixed agreement policy
before any test pass. Native sample-weighted MAE is kept separate from pooled
valid-angle MAE. Engineering fits cannot become full scientific results.
Only trusted local fit checkpoints and evaluation artifacts may be opened.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import uuid


from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _StatGuard,_safe
from .classic_emg_report import PUBLIC_NATIVE_METRICS,JOINT_NAMES,FINGERS,PD_GROUPS
from .metrics import PairedCenterAccumulator
import numpy as np
from .rn18_fusion_literal import compose_rn18_fusion
from .rn18_fusion_data import build_rn18_fusion_partitions
from .data import LegacyStore
from .rn18_fusion_cohort import admit_rn18_fusion_cohort
from .rn18_fusion_runtime import build_rn18_fusion_runtime
from .vision_literal_runtime import vision_state_sha256 as fusion_state_sha256
from .emgformer_trajectory_eval import _close_loader
from .emg2pose_memory_process import process_identity
from .rn18_fusion_train import (ROOT,_json,_write,_sealed,_check_seal,_runtime_device,storage,_batch,read_rn18_fusion_fit,
    _read_fit_context,_fit_loaders)
from .vision_literal_train import _space,_read_epochs
from .vision_literal_eval import _record,_population_public


FORMAT='original_rn18_fusion_selected_prediction_v1'
PROFILE='original_rn18_fusion_pretest_validation_profile_v1'
REPORT='original_rn18_fusion_center_completed_report_v1'
IMPORTED_SHA=_file_hash(__file__)
POLICY=dict(prediction='exact typed prediction/target/mask/input replay',
    native_mae_atol_rad=2e-6,native_mae_rtol=2e-6,
    native_aggregation='sample-count weighted original per-batch masked L1, float32 accumulation',
    pooled_aggregation='float64 absolute-error sum / exact valid-angle count',
    scope='fixed before validation; no test-driven tolerance or checkpoint changes')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported fusion evaluation source changed')
    return {p:_file_hash(ROOT/p) for p in ('egoemg_repro/rn18_fusion_eval.py',
        'egoemg_repro/rn18_fusion_train.py','egoemg_repro/vision_literal_eval.py',
        'egoemg_repro/metrics.py','egoemg_repro/classic_emg_report.py',
        'egoemg_repro/emgformer_trajectory_eval.py','egoemg_repro/emg2pose_memory_process.py')}


def _process():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=fields[19],uid=os.getuid())


def rn18_fusion_population(cohort):
    cohort.check(require_full=False);users={};splits={n:set()for n in ('user','gesture','both')}
    for role,ds in cohort.datasets.items():
        users[role]=sorted({str(ds.store.episodes[w['episode']]['subject'])for w in ds.windows})
        if role=='test':
            for w in ds.windows:splits[w['split']].add(str(ds.store.episodes[w['episode']]['subject']))
    splits={n:sorted(v)for n,v in splits.items()}
    return dict(users_by_role=users,test_users_by_split=splits,public=_population_public(users,splits,len(cohort.datasets['train'].store.episodes)))


def _recorded_population(identity):
    store=LegacyStore(identity['data_locations']['data']);partitions=build_rn18_fusion_partitions(store)
    if _hash(partitions)!=identity['cohort']['partition_content_sha256']:
        raise ValueError('actual population partition differs from admitted fit')
    users={r:sorted({str(store.episodes[w['episode']]['subject']) for w in partitions['windows'] if w['role']==r})
        for r in ('train','val','test')}
    splits={r:sorted({str(store.episodes[w['episode']]['subject']) for w in partitions['windows'] if w['role']=='test' and w['split']==r})
        for r in ('user','gesture','both')}
    return dict(users_by_role=users,test_users_by_split=splits,public=_population_public(users,splits,len(store.episodes)))


def rn18_fusion_statistics(parts,*,samples,batch_size,role):
    """Recompute raw 22D sums, counts and both aggregation definitions."""
    import torch
    if role not in ('val','test') or type(samples) is not int or samples<=0 or samples%2 or type(batch_size) is not int or batch_size<=0:
        raise ValueError('complete repeat-one RN18 evaluation population required')
    acc={name:PairedCenterAccumulator() for name in ('consumed','unique_first_repeat')}
    sums={name:np.zeros(22,np.float64) for name in acc};counts={name:np.zeros(22,np.int64) for name in acc}
    split_users={name:{} for name in acc};position=0;weighted=torch.tensor(0.,dtype=torch.float32)
    expected_sizes=[min(batch_size,samples-i) for i in range(0,samples,batch_size)]
    if len(parts)!=len(expected_sizes):raise ValueError('complete native batch coverage required')
    for part,n in zip(parts,expected_sizes):
        p,t,m=part['pred'],part['target'],part['mask']
        if (p.shape!=(n,22,1) or t.shape!=p.shape or m.shape!=(n,1) or
            p.dtype not in (torch.float32,torch.bfloat16) or t.dtype!=torch.float32 or m.dtype!=torch.bool or
            not bool(torch.isfinite(p).all() and torch.isfinite(t).all()) or
            part['vision'].shape!=(n,1) or part['vision'].dtype!=torch.bool or
            part['index'].dtype!=torch.int64 or not torch.equal(part['index'],torch.arange(position,position+n)) or
            not torch.equal(part['unique'],part['index']) or
            not torch.equal(part['repeat'],torch.zeros(n,dtype=torch.int64)) or
            any(len(part[key])!=n for key in ('hand','split','user','center')) or
            not isinstance(part['input_sha256'],str) or len(part['input_sha256'])!=64):
            raise ValueError('typed original tuple, metadata or ordered repeat coverage differs')
        if any(part[k].dtype!=torch.int64 or part[k].shape!=(n,)for k in ('unique','repeat','center','episode','window')):
            raise ValueError('typed RN18 request/center/episode/window identity required')
        fingerprints=part['sample_input_sha256']
        if(len(fingerprints)!=n or any(type(h)is not str or len(h)!=64 or any(c not in '0123456789abcdef'for c in h)for h in fingerprints)
            or part['input_sha256']!=_hash(fingerprints)):
            raise ValueError('actual RN18 per-sample inputs and batch digest must agree')
        for j,index in enumerate(part['index']):
            if part['hand'][j]!=('left'if int(index)<samples//2 else'right'):
                raise ValueError('complete left-then-right native RN18 request order differs')
        mask=m[:,None,:].expand_as(t)
        batch_mae=(p.float()-t).abs()[mask].mean() if bool(mask.any()) else p.float().sum()*0.
        weighted+=batch_mae*n
        for i in range(n):
            hand,split,user=part['hand'][i],part['split'][i],part['user'][i]
            allowed_splits=('train',) if role=='val' else ('user','gesture','both')
            if hand not in ('left','right') or split not in allowed_splits or not isinstance(user,str) or not user:
                raise ValueError('actual private hand/split/user provenance required')
            pred=p[i].float().numpy();target=t[i].numpy();valid=m[i].numpy();visible=bool(part['vision'][i,0])
            for name in acc:
                if name=='unique_first_repeat' and int(part['repeat'][i])!=0:continue
                acc[name].update(pred,target,valid,vision_valid=visible,hand=hand,split=split,user=user)
                error=np.abs(pred.astype(np.float64)-target.astype(np.float64))[:,valid]
                sums[name]+=error.sum(axis=1);counts[name]+=int(valid.sum())
                cell=split_users[name].setdefault(split,{}).setdefault(user,[0.,0])
                cell[0]+=float(error.sum());cell[1]+=int(error.size)
        position+=n
    result=dict(format='original_rn18_fusion_center_statistics_v1',role=role,consumed_samples=samples,
        unique_samples=samples,duplicate_consumptions=0,batch_sizes=expected_sizes,
        native_batch_weighted_mae_rad=float(weighted/samples),
        unique_rule='source repeat1; every original hand request is consumed exactly once; both views have the same actual population')
    for name,value in acc.items():
        report=value.report();all_=report['all'];s,c=sums[name],counts[name]
        all_['by_joint']=[dict(index=i,name=JOINT_NAMES[i],**_record(s[i],c[i])) for i in range(22)]
        for key,groups in (('by_finger',FINGERS),('by_pd_group',PD_GROUPS)):
            all_[key]={label:_record(s[list(ids)].sum(),c[list(ids)].sum()) for label,ids in groups.items()}
        all_['by_split_user_summary']={}
        for split,users in sorted(split_users[name].items()):
            values=[e/n*180/math.pi for e,n in users.values() if n]
            all_['by_split_user_summary'][split]=dict(user_count=len(values),mean_mae_deg=float(np.mean(values)) if values else None,
                std_mae_deg=float(np.std(values)) if values else None,aggregation='unweighted user means; ddof=0')
        if int(c.sum())!=all_['overall']['valid_angle_count'] or not math.isclose(float(s.sum()),all_['overall']['absolute_error_sum_rad'],rel_tol=2e-12,abs_tol=1e-9):
            raise ValueError('joint and pooled sufficient statistics do not conserve')
        result[name]=report
    return result


def _close_native(statistics,metrics,role):
    value=metrics.get(role+'_mae')
    if type(value) is not float or not math.isfinite(value) or not math.isclose(value,
            statistics['native_batch_weighted_mae_rad'],abs_tol=POLICY['native_mae_atol_rad'],rel_tol=POLICY['native_mae_rtol']):
        raise ValueError('native batch aggregation differs under fixed pretest policy')


def _payload_sha(parts):return storage.state_digest(parts)


def _load_pass(directory):
    import torch
    directory=_safe(directory);value=_json(directory/'pass.json');_check_seal(value,FORMAT)
    if value['sources_sha256']!=_sources() or value['fixed_agreement_policy']!=POLICY:
        raise ValueError('fusion evaluation source or fixed agreement policy differs')
    file=directory/'predictions.pt'
    if _file_hash(file)!=value['payload_sha256']:raise ValueError('actual prediction file bytes differ')
    payload=torch.load(str(file),map_location='cpu',weights_only=True)
    if type(payload) is not dict or set(payload)!={'parts','population'}:raise ValueError('complete prediction/population payload required')
    parts=payload['parts'];population=payload['population']
    public=_population_public(population['users_by_role'],population['test_users_by_split'],population['public']['episode_records'])
    if public!=population['public'] or public!=value['population'] or _hash(population)!=value['population_sha256']:
        raise ValueError('actual private population and public intersections differ')
    if _payload_sha(parts)!=value['prediction_sha256']:raise ValueError('typed input/prediction replay digest differs')
    statistics=rn18_fusion_statistics(parts,samples=value['samples'],batch_size=value['batch_size'],role=value['role'])
    if statistics!=value['statistics']:raise ValueError('raw prediction sufficient statistics differ from report')
    if value['kind']=='native':
        metrics=_json(directory/'native-metrics.json')
        if metrics!=value['native_metrics']:raise ValueError('actual native metric sidecar differs')
        _close_native(statistics,metrics,value['role'])
    elif value['kind']!='replay' or value['native_metrics']!={}:raise ValueError('independent replay cannot invent native metrics')
    if value['model_before_sha256']!=value['model_after_sha256'] or value['scientific_eligible'] is not False or value['full_paper_complete'] is not False:
        raise ValueError('prediction-only run mutated model or claimed completion')
    return value


class _Evidence:
    """A single API transaction retaining every admitted raw/source generation."""
    def __init__(self):
        self.fits={};self.passes={};self.profiles={};self.active=set();self.selected={};self.inputs={}
        self.guard=_StatGuard([ROOT/p for p in _sources()])

    def check(self):
        self.guard.check()
        for cohort,initializers,guard in self.inputs.values():
            guard.check();cohort.guard.check()
            if cohort.identity['sha256']!=_hash({k:v for k,v in cohort.identity.items()if k!='sha256'}):
                raise ValueError('RN18 transaction admitted identity changed')
            if initializers.identity['sha256']!=_hash({k:v for k,v in initializers.identity.items()if k!='sha256'}):
                raise ValueError('RN18 transaction producer identity changed')
        self.guard.check()

    def fit(self,path):
        path=_safe(path);self.check()
        if path not in self.fits:
            fit,identity,cohort,initializers,guard=_read_fit_context(path,require_full=False)
            for p,stamp in guard.paths.items():
                self.guard.add(p)
                if self.guard.paths[p]!=stamp:raise ValueError('RN18 source/input generation changed while binding transaction')
            selected=_safe(path/'checkpoints'/fit['selected_best']['file'])
            import torch
            checkpoint=torch.load(str(selected),map_location='cpu',weights_only=False)
            selected_model=fusion_state_sha256(checkpoint['state_dict'])
            self.selected[path]=copy.deepcopy(checkpoint['rn18_fusion_fit']['records'][-1]);del checkpoint
            population=_recorded_population(identity)
            self.inputs[path]=(cohort,initializers,guard);self.fits[path]=(fit,identity,selected_model,population)
        self.check();return self.fits[path]


def _pose_sha(parts):
    import torch
    return storage.state_digest({k:torch.cat([p[k]for p in parts])for k in ('pred','target','mask')})


def _saved_validation(role,parts,statistics,record):
    if role!='val':return None
    reference=record['validation']
    if reference['samples']!=statistics['consumed_samples']:raise ValueError('saved RN18 validation population differs')
    pose=_pose_sha(parts)
    return dict(completed_epoch=record['epoch']+1,fit_pose_sha256=reference['pose_sha256'],evaluation_pose_sha256=pose,
        pose_exact_match=pose==reference['pose_sha256'],fit_native_mae_rad=reference['val_mae'],
        native_mae_difference_rad=statistics['native_batch_weighted_mae_rad']-reference['val_mae'],
        view_policy='original saved-fit and standalone native validation views recorded separately; same selected checkpoint; standalone independent replay exact')


def _read_pass(directory,context):
    directory=_safe(directory)
    if directory in context.passes:context.check();return context.passes[directory]
    key=('pass',directory)
    if key in context.active:raise ValueError('cyclic fusion evidence references')
    context.active.add(key)
    try:
        for name in ('pass.json','predictions.pt'):context.guard.add(directory/name)
        if (directory/'native-metrics.json').exists():context.guard.add(directory/'native-metrics.json')
        value=_load_pass(directory)
        fit,identity,selected_model,population=context.fit(value['fit_path'])
        if (not fit['stage_complete'] or value['fit_result_sha256']!=fit['sha256'] or
            value['selected_best']!=fit['selected_best'] or value['cohort_sha256']!=identity['cohort']['sha256'] or
            value['samples']!=identity['cohort']['counts'][value['role']] or
            value['batch_size']!=identity['policy']['batch_size_by_role'][value['role']] or
            value['training_scope']!=fit['scope'] or value['training_full_terminated'] is not fit['full_training_terminated'] or
            value['population_sha256']!=_hash(population) or value['population']!=population['public']):
            raise ValueError('pass differs from actual completed fit, cohort population or selected best')
        import torch
        parts=torch.load(directory/'predictions.pt',map_location='cpu',weights_only=True)['parts']
        if value['saved_fit_validation']!=_saved_validation(value['role'],parts,value['statistics'],context.selected[_safe(value['fit_path'])]):
            raise ValueError('actual saved-fit versus independent validation comparison differs')
        workers=value['loader_workers'];expected_workers=identity['policy']['workers_by_role'][value['role']]
        if(type(workers)is not dict or workers['count']!=expected_workers or len(workers['handles'])!=expected_workers
            or workers['all_closed']is not True):raise ValueError('complete actual RN18 worker closure evidence required')
        if value['model_before_sha256']!=selected_model or value['model_after_sha256']!=selected_model:
            raise ValueError('pass model digest differs from actual raw selected model checkpoint')
        if value['role']=='test':
            profile=_read_profile(value['profile_path'],context)
            if (value['profile_sha256']!=profile['sha256'] or value['fit_result_sha256']!=profile['fit_result_sha256'] or
                value['started_ns']<=profile['finished_ns']):raise ValueError('test is not bound to an earlier frozen validation profile')
        elif value['profile_path'] is not None or value['profile_sha256'] is not None:
            raise ValueError('validation may not consume a test profile')
        if value['kind']=='replay':
            ref=_read_pass(value['reference_path'],context)
            if (ref['kind']!='native' or ref['role']!=value['role'] or ref['fit_result_sha256']!=value['fit_result_sha256'] or
                ref['sha256']!=value['reference_sha256'] or ref['prediction_sha256']!=value['prediction_sha256'] or
                ref['population_sha256']!=value['population_sha256'] or ref['model_before_sha256']!=selected_model or
                ref['process']==value['process'] or value['started_ns']<=ref['finished_ns']):
                raise ValueError('fresh-process exact native prediction replay differs')
        elif value['reference_path'] is not None or value['reference_sha256'] is not None:
            raise ValueError('native pass cannot claim an independent replay')
        context.check();context.passes[directory]=value
        return value
    finally:context.active.remove(key)


def _read_profile(directory,context):
    directory=_safe(directory)
    if directory in context.profiles:context.check();return context.profiles[directory]
    key=('profile',directory)
    if key in context.active:raise ValueError('cyclic fusion validation profile')
    context.active.add(key)
    try:
        context.guard.add(directory/'profile.json')
        profile=_json(directory/'profile.json');_check_seal(profile,PROFILE)
        reference=_read_pass(profile['reference_path'],context);replay=_read_pass(profile['replay_path'],context)
        if (profile['saved_fit_validation']!=reference['saved_fit_validation']or profile['fixed_agreement_policy']!=POLICY or reference['role']!='val' or reference['kind']!='native' or
            replay['role']!='val' or replay['kind']!='replay' or replay['reference_sha256']!=reference['sha256'] or
            profile['reference_sha256']!=reference['sha256'] or profile['replay_sha256']!=replay['sha256'] or
            profile['fit_result_sha256']!=reference['fit_result_sha256'] or
            profile['finished_ns']<=max(reference['finished_ns'],replay['finished_ns']) or
            profile['scientific_eligible'] is not False or profile['full_paper_complete'] is not False):
            raise ValueError('frozen independent validation replay profile differs')
        context.check();context.profiles[directory]=profile
        return profile
    finally:context.active.remove(key)


def read_rn18_fusion_pass(directory):
    context=_Evidence();value=_read_pass(directory,context);context.check();return value


def read_rn18_fusion_profile(directory):
    context=_Evidence();value=_read_profile(directory,context);context.check();return value


def freeze_rn18_fusion_validation(reference,replay,output):
    ref=read_rn18_fusion_pass(reference);other=read_rn18_fusion_pass(replay)
    if ref['role']!='val' or ref['kind']!='native' or other['role']!='val' or other['kind']!='replay' or other['reference_sha256']!=ref['sha256']:
        raise ValueError('complete native validation and fresh exact replay required')
    output=_safe(output,exists=False);_space(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    value=_sealed(dict(format=PROFILE,reference_path=str(_safe(reference)),replay_path=str(_safe(replay)),
        reference_sha256=ref['sha256'],replay_sha256=other['sha256'],fit_result_sha256=ref['fit_result_sha256'],
        saved_fit_validation=copy.deepcopy(ref['saved_fit_validation']),fixed_agreement_policy=copy.deepcopy(POLICY),finished_ns=time.time_ns(),scientific_eligible=False,full_paper_complete=False))
    _write(output/'profile.json',value);return read_rn18_fusion_profile(output)


def _evaluation_loader(cohort,policy,role):
    # The fit adapter bounds only engineering training; val/test always complete.
    return _fit_loaders(cohort,policy)[0][role]


def _capture(batch,out,cohort,policy,role,parts,seen):
    import torch
    pred,target,mask=out;n=len(pred);position=sum(len(p['index']) for p in parts)
    _batch(cohort,policy,batch,role,seen,len(cohort.datasets[role]))
    if (not torch.equal(target,batch['joint_angles']) or not torch.equal(mask,batch['label_valid_mask']) or
        batch['rn18_fusion_sample_index'].tolist()!=list(range(position,position+n))):
        raise ValueError('original fusion tuple target, generated mask or sequential input differs')
    ds=cohort.datasets[role]
    for j in range(n):
        episode=int(batch['episode'][j])
        if (int(batch['jitter_offset'][j])!=0 or batch['subject'][j]!=ds.store.episodes[episode]['subject'] or
            any(int(batch['anchor_'+k][j])!=int(batch[k][j]) for k in ('start','end','center'))):
            raise ValueError('original fusion evaluation provenance differs from admitted cohort')
    def cpu(value):return value.detach().cpu().clone()
    # Transfer once; hashing is per original hand sample so repetition pairing
    # also works when the repeat boundary crosses a partial native batch.
    emg=cpu(batch['emg']);rgb=cpu(batch['vision_img'])
    target_cpu=cpu(target);mask_cpu=cpu(mask);visible=cpu(batch['vision_valid_mask'])
    hashes=[storage.state_digest(dict(emg=emg[j],vision_img=rgb[j],target=target_cpu[j],mask=mask_cpu[j],vision=visible[j])) for j in range(n)]
    parts.append(dict(pred=cpu(pred),target=target_cpu,mask=mask_cpu,index=cpu(batch['rn18_fusion_sample_index']),
        unique=cpu(batch['rn18_fusion_unique_sample_index']),repeat=cpu(batch['rn18_fusion_repeat_index']),
        vision=visible,center=cpu(batch['center']),episode=cpu(batch['episode']),window=cpu(batch['rn18_fusion_window_index']),
        hand=list(batch['hand']),split=list(batch['split']),user=list(batch['subject']),
        input_sha256=_hash(hashes),sample_input_sha256=hashes))


def run_rn18_fusion_pass(cohort,fit_path,output,*,role,kind,reference=None,profile=None,require_full=True):
    import torch
    if role not in ('val','test') or kind not in ('native','replay'):raise ValueError('explicit native/replay and val/test required')
    cohort.check(require_full=require_full);fit_path=_safe(fit_path);context=_Evidence();fit,identity,_,_=context.fit(fit_path)
    if require_full and not fit['full_training_terminated']:raise ValueError('engineering fusion is not full terminated training')
    if not fit['stage_complete']:raise ValueError('partial engineering or full fit cannot be evaluated as completed')
    if identity['cohort']!=cohort.identity:raise ValueError('actual rehashed evaluation cohort differs from training')
    frozen=None;ref=None
    if role=='test':
        if profile is None:raise ValueError('freeze independent validation profile before test')
        frozen=_read_profile(profile,context)
        if frozen['fit_result_sha256']!=fit['sha256']:raise ValueError('pretest profile belongs to another fit')
    elif profile is not None:raise ValueError('validation cannot consume a test profile')
    if kind=='replay':
        if reference is None:raise ValueError('native pass required before fresh replay')
        ref=_read_pass(reference,context)
        if ref['kind']!='native' or ref['role']!=role or ref['fit_result_sha256']!=fit['sha256'] or ref['process']==_process():
            raise ValueError('fresh process and matching native reference required')
    elif reference is not None:raise ValueError('native pass has no reference')
    policy=identity['policy'];device=_runtime_device(policy);output=_safe(output,exists=False);_space(output.parent)
    checkpoint=fit_path/'checkpoints'/fit['selected_best']['file']
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)['state_dict']
    prepared=build_rn18_fusion_runtime(cohort.upstream,cohort.config,state=state,expected_initialization=identity['construction'])
    population=rn18_fusion_population(cohort)
    before=fusion_state_sha256(prepared.module.state_dict());parts=[];seen=set();start=time.time_ns()
    guard=_StatGuard([fit_path/'identity.json',fit_path/'fit-result.json',checkpoint,*[ROOT/p for p in _sources()]])
    loader=_evaluation_loader(cohort,policy,role);original=prepared.module.forward;worker_snapshot=None
    def observed(batch):
        nonlocal worker_snapshot
        cohort.guard.check();guard.check();context.check()
        if worker_snapshot is None:
            iterator=getattr(loader,'_iterator',None);workers=[]if iterator is None else list(getattr(iterator,'_workers',[]))
            worker_snapshot=(iterator,[(p,process_identity(p.pid))for p in workers])
            if len(workers)!=policy['workers_by_role'][role]:raise ValueError('actual native RN18 loader worker count differs')
        out=original(batch);_capture(batch,out,cohort,policy,role,parts,seen);return out
    prepared.module.forward=observed;native={}
    output.mkdir(mode=0o700,exist_ok=False)
    try:
        if kind=='native':
            trainer=storage.pl.Trainer(accelerator=device['accelerator'],devices=1,precision=policy['precision'],
                logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
            method=trainer.validate if role=='val' else trainer.test
            result=method(prepared.module,dataloaders=loader,verbose=False)
            native={k:float(v) for k,v in result[0].items()}
            if not all(math.isfinite(v) for v in native.values()):raise ValueError('nonfinite original native metrics')
        else:
            from contextlib import nullcontext
            from lightning_fabric.utilities.apply_func import move_data_to_device
            target_device='cpu' if device['accelerator']=='cpu' else 'cuda:0'
            prepared.module.to(target_device).eval()
            with torch.inference_mode():
                for batch in loader:
                    batch=move_data_to_device(batch,target_device)
                    with (nullcontext() if target_device=='cpu' else torch.autocast('cuda',dtype=torch.bfloat16)):
                        prepared.module.forward(batch)
        statistics=rn18_fusion_statistics(parts,samples=len(cohort.datasets[role]),batch_size=policy['batch_size_by_role'][role],role=role)
        if kind=='native':_close_native(statistics,native,role)
        if ref is not None and _payload_sha(parts)!=ref['prediction_sha256']:raise ValueError('fixed exact typed prediction/input replay failed')
        cohort.check(require_full=require_full);guard.check();context.check();prepared.check();after=fusion_state_sha256(prepared.module.state_dict())
        if before!=after:raise ValueError('selected model changed during evaluation')
        temporary=output/('.predictions-'+uuid.uuid4().hex)
        try:
            with temporary.open('xb') as f:torch.save(dict(parts=parts,population=population),f);f.flush();os.fsync(f.fileno())
            os.link(temporary,output/'predictions.pt')
        finally:
            if temporary.exists():temporary.unlink()
        if kind=='native':_write(output/'native-metrics.json',native)
        handles=_close_loader(loader,snapshot=worker_snapshot);worker_snapshot=None
        worker_evidence=dict(count=len(handles),handles=handles,all_closed=True)
        value=_sealed(dict(format=FORMAT,role=role,kind=kind,fit_path=str(fit_path),fit_result_sha256=fit['sha256'],
            training_scope=fit['scope'],training_full_terminated=fit['full_training_terminated'],cohort_sha256=cohort.identity['sha256'],
            selected_best=copy.deepcopy(fit['selected_best']),sources_sha256=_sources(),fixed_agreement_policy=copy.deepcopy(POLICY),
            samples=len(cohort.datasets[role]),batch_size=policy['batch_size_by_role'][role],statistics=statistics,
            population=population['public'],population_sha256=_hash(population),
            payload_sha256=_file_hash(output/'predictions.pt'),prediction_sha256=_payload_sha(parts),native_metrics=native,
            native_aggregation_agrees=kind=='native',loader_workers=worker_evidence,
            saved_fit_validation=_saved_validation(role,parts,statistics,context.selected[fit_path]),model_before_sha256=before,model_after_sha256=after,
            process=_process(),started_ns=start,finished_ns=time.time_ns(),
            profile_path=str(_safe(profile)) if profile is not None else None,profile_sha256=frozen['sha256'] if frozen is not None else None,
            reference_path=str(_safe(reference)) if reference is not None else None,reference_sha256=ref['sha256'] if ref is not None else None,
            scientific_eligible=False,full_paper_complete=False))
        _write(output/'pass.json',value);return _read_pass(output,context)
    finally:
        prepared.module.forward=original
        if worker_snapshot is not None:_close_loader(loader,snapshot=worker_snapshot)


def build_rn18_fusion_report(recompute,*,require_full=True):
    context=_Evidence();replay=_read_pass(recompute,context)
    if replay['role']!='test' or replay['kind']!='replay':raise ValueError('completed native test and independent test replay required')
    native=_read_pass(replay['reference_path'],context);fit,identity,_,_=context.fit(replay['fit_path'])
    if require_full and not fit['full_training_terminated']:raise ValueError('engineering report is not full scientific fusion')
    statistics=copy.deepcopy(replay['statistics'])
    users=sorted(statistics['consumed']['all']['by_user']);names={user:'user_'+str(i).zfill(3) for i,user in enumerate(users)}
    def sanitize(value):
        if isinstance(value,dict):
            return {key:({names[u]:sanitize(v) for u,v in item.items()} if key=='by_user' else sanitize(item)) for key,item in value.items()}
        if isinstance(value,list):return [sanitize(v) for v in value]
        return value
    statistics=sanitize(statistics)
    result=_sealed(dict(format=REPORT,scientific_eligible=fit['full_training_terminated'],full_paper_complete=False,
        scope='available legacy cohort; single rank; original full200/10worker loaders; original fusion semantics; independent train-only validation',
        paper_numerical_equivalence_established=False,variant='fusion_rn18_s_center_8ch',
        training=dict(scope=fit['scope'],completed_epochs=fit['completed_epochs'],optimizer_executions=fit['optimizer_executions'],
            full_training_terminated=fit['full_training_terminated'],selected_best=fit['selected_best'],fit_result_sha256=fit['sha256'],
            execution_policy=copy.deepcopy(identity['policy'])),
        initializers=copy.deepcopy(identity['initializer_descriptor']['identity']),
        curve_data=_curve_data(replay['fit_path'],fit,identity),plot_runtime=_plot_runtime(),
        test=statistics,population=copy.deepcopy(replay['population']),native_metrics={k:v for k,v in native['native_metrics'].items() if k in PUBLIC_NATIVE_METRICS},
        saved_fit_validation=copy.deepcopy(_read_profile(replay['profile_path'],context)['saved_fit_validation']),
        fixed_agreement_policy=copy.deepcopy(POLICY),cohort_sha256=replay['cohort_sha256'],
        validation_profile_sha256=replay['profile_sha256'],native_test_sha256=native['sha256'],independent_test_sha256=replay['sha256'],
        sources_sha256=_sources(),prediction_sha256=replay['prediction_sha256']))
    context.check()
    return result


def render_rn18_fusion_report(report):
    training=report['training'];test=report['test'];scope='完整训练实验' if report['scientific_eligible'] else '工程验证；未完成正式完整训练'
    lines=['# ResNet-18 + EMGFormer-S 融合中心评价','',scope,'',
        f"模型：{report['variant']}；训练完成 {training['completed_epochs']} 轮，实际更新 {training['optimizer_executions']} 次。",'',
        '训练预算：完整实验150轮、不早停；工程实验显式限于两轮。','',
        f"验证选择：{training['selected_best']['file']}；不以最后一轮替代验证最优。",'',
        f"测试消费 {test['consumed_samples']} 手样本，唯一 {test['unique_samples']}，重复消费 {test['duplicate_consumptions']}。",'',
        '| 统计人口 | MAE (°) | 有效角度数 | 弧度绝对误差和 |','| --- | ---: | ---: | ---: |']
    for label,record in [('全部请求中心',test['consumed']['all']['overall']),
        ('唯一中心',test['unique_first_repeat']['all']['overall']),('全部请求中的有图子集',test['consumed']['vision_valid']['overall'])]:
        error='—' if record['mae_deg'] is None else f"{record['mae_deg']:.6f}"
        lines.append(f"| {label} | {error} | {record['valid_angle_count']} | {record['absolute_error_sum_rad']:.10g} |")
    lines+=['',f"原生按样本数加权的批量 MAE：{test['native_batch_weighted_mae_rad']:.10g} rad。",'',
        '该指标与按有效角度数汇总的 pooled MAE 分开报告。缺图仍保留原生成标签 mask；不增加腕角过滤。','',
        '采用可用旧版 EgoEMG 队列、独立 train 留出验证、单 rank，正式实验每批200/10worker；未证明与论文原表或缺失的合并训练来源数值等价。',
        '当前报告只验收所列实验，整篇论文复现仍未完成。','',
        '## 完整公开证据','', '以下与 metrics.json 一致，包含逐手、逐条件、匿名逐用户、逐角、指组、用户交集与来源摘要。','',
        '```json',json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),'```','']
    return '\n'.join(lines)


def evaluate_rn18_fusion(cohort,fit,output,*,role,profile=None,require_full=True):
    return run_rn18_fusion_pass(cohort,fit,output,role=role,kind='native',profile=profile,require_full=require_full)


def verify_rn18_fusion_evaluation(cohort,fit,reference,output,*,profile=None,require_full=True):
    ref=read_rn18_fusion_pass(reference)
    return run_rn18_fusion_pass(cohort,fit,output,role=ref['role'],kind='replay',reference=reference,
        profile=profile,require_full=require_full)


def _curve_data(fit_path,fit,identity):
    import torch
    path=_safe(Path(fit_path)/'epochs.jsonl');rows,_=_read_epochs(path,identity['sha256'])
    if len(rows)!=fit['completed_epochs']:raise ValueError('actual complete fusion curve population differs')
    selected=[r['epoch'] for r in rows if r['validation']['val_mae']==fit['selected_best']['score']]
    if not selected:raise ValueError('selected best has no actual validation curve epoch')
    checkpoint=torch.load(str(_safe(Path(fit_path)/'checkpoints'/fit['selected_best']['file'])),map_location='cpu',weights_only=False)
    selected_epoch=checkpoint['rn18_fusion_fit']['completed_epochs']-1
    if selected_epoch not in selected:raise ValueError('actual selected checkpoint epoch differs from validation curve')
    return dict(format='original_rn18_fusion_native_curve_data_v1',fit_result_sha256=fit['sha256'],
        journal_sha256=_file_hash(path),selected_best=copy.deepcopy(fit['selected_best']),
        selected_epoch_zero_based=selected_epoch,
        selected_epoch_candidates_zero_based=selected,training_unit='native sample-weighted composite loss',
        validation_unit='radian; original sample-weighted val_mae',epochs=copy.deepcopy(rows))


def _plot_runtime():
    import matplotlib
    from matplotlib import font_manager,ft2font
    font=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
    return dict(matplotlib_version=matplotlib.__version__,freetype_version=ft2font.__freetype_version__,
        font_sha256=_file_hash(font),backend='Agg',svg_hashsalt='egoemg-original-rn18-fusion-v1')


def _curve_images(curves):
    import io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'svg.hashsalt':'egoemg-original-rn18-fusion-v1','font.family':'DejaVu Sans'}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        try:
            rows=curves['epochs'];x=[r['epoch']+1 for r in rows]
            axes[0].plot(x,[r['train_loss'] for r in rows],color='#1d4ed8')
            axes[0].set(ylabel='Native composite training loss',xlabel='Completed epoch',title='Training')
            axes[1].plot(x,[r['validation']['val_mae'] for r in rows],color='#047857')
            axes[1].axhline(curves['selected_best']['score'],color='#b45309',linestyle='--',label='Validation best')
            axes[1].scatter([curves['selected_epoch_zero_based']+1],[curves['selected_best']['score']],color='#b45309',zorder=3)
            axes[1].set(ylabel='Native sample-weighted MAE (rad)',xlabel='Completed epoch',title='Held-out validation')
            axes[1].legend()
            for ax in axes:ax.grid(alpha=.2)
            result={}
            for ext,metadata in [('png',{'Software':'EgoEMG-Reproduction'}),('svg',{'Date':None,'Creator':'EgoEMG-Reproduction'})]:
                stream=io.BytesIO();fig.savefig(stream,format=ext,dpi=150,metadata=metadata);result['training-curves.'+ext]=stream.getvalue()
            return result
        finally:plt.close(fig)


PUBLIC_FILES=('metrics.json','metrics.md','curve-data.json','training-curves.png','training-curves.svg')


def _completion(output,report):
    return _sealed(dict(format='original_rn18_fusion_public_report_completion_v1',report_sha256=report['sha256'],
        files_sha256={name:_file_hash(_safe(output/name)) for name in PUBLIC_FILES},
        scientific_eligible=report['scientific_eligible'],full_paper_complete=False))


def verify_rn18_fusion_report(recompute,output,*,require_full=True):
    output=_safe(output);report=build_rn18_fusion_report(recompute,require_full=require_full)
    if (_json(output/'metrics.json')!=report or _safe(output/'metrics.md').read_text()!=render_rn18_fusion_report(report) or
        _json(output/'curve-data.json')!=report['curve_data'] or
        any(_safe(output/name).read_bytes()!=raw for name,raw in _curve_images(report['curve_data']).items()) or
        _json(output/'completion.json')!=_completion(output,report)):
        raise ValueError('public fusion report, actual epoch curves or last completion marker differs from reopened raw chain')
    return report


def publish_rn18_fusion_report(fit,evaluation,replay,output,*,require_full=True):
    report=build_rn18_fusion_report(replay,require_full=require_full)
    ref=read_rn18_fusion_pass(replay)
    if _safe(fit)!=_safe(ref['fit_path']) or _safe(evaluation)!=_safe(ref['reference_path']):
        raise ValueError('public report must pair the exact completed fit, native test and independent replay')
    output=_safe(output,exists=False);_space(output.parent)
    images=_curve_images(report['curve_data']);output.mkdir(mode=0o700,exist_ok=False)
    _write(output/'metrics.json',report);_write(output/'curve-data.json',report['curve_data'])
    with (output/'metrics.md').open('x') as f:f.write(render_rn18_fusion_report(report));f.flush();os.fsync(f.fileno())
    for name,raw in images.items():
        with (output/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if report!=build_rn18_fusion_report(replay,require_full=require_full):raise ValueError('raw chain changed during public report preparation')
    _write(output/'completion.json',_completion(output,report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('pass')
    for name in ('data','crops','upstream','fit','output'):run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--role',choices=('val','test'),required=True);run.add_argument('--kind',choices=('native','replay'),required=True)
    run.add_argument('--profile',type=Path);run.add_argument('--reference',type=Path);run.add_argument('--engineering',action='store_true')
    freeze=sub.add_parser('freeze')
    for name in ('reference','replay','output'):freeze.add_argument('--'+name,type=Path,required=True)
    report=sub.add_parser('report')
    for name in ('fit','evaluation','replay','output'):report.add_argument('--'+name,type=Path,required=True)
    report.add_argument('--engineering',action='store_true')
    verify=sub.add_parser('verify')
    for name in ('recompute','output'):verify.add_argument('--'+name,type=Path,required=True)
    verify.add_argument('--engineering',action='store_true')
    args=vars(parser.parse_args());action=args.pop('action')
    engineering=args.pop('engineering',False)
    if action=='freeze':result=freeze_rn18_fusion_validation(**args)
    elif action=='report':result=publish_rn18_fusion_report(require_full=not engineering,**args)
    elif action=='verify':result=verify_rn18_fusion_report(require_full=not engineering,**args)
    else:
        data=args.pop('data');crops=args.pop('crops');upstream=args.pop('upstream');fit=args.pop('fit')
        expected=_json(fit/'identity.json')['cohort']
        cohort=admit_rn18_fusion_cohort(data,crops,upstream=upstream,config=compose_rn18_fusion(upstream),
            expected=expected,require_full=not engineering)
        result=run_rn18_fusion_pass(cohort,fit,require_full=not engineering,**args)
    print(json.dumps(dict(format=result['format'],scientific_eligible=result['scientific_eligible'],full_paper_complete=False)))


if __name__=='__main__':main()
