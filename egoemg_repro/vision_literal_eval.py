"""Selected-best original vision evaluation and pretest exact replay calibration.

Private passes retain typed predictions, original generated masks and input
fingerprints. A fresh validation replay freezes the fixed agreement policy
before any test pass. Native sample-weighted MAE is kept separate from pooled
valid-angle MAE. Engineering fits cannot become full scientific results.
Only trusted local fit checkpoints and evaluation artifacts may be opened.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _StatGuard,_safe
from .classic_emg_report import JOINT_NAMES,FINGERS,PD_GROUPS,PUBLIC_NATIVE_METRICS
from .metrics import PairedCenterAccumulator
from .vision_literal import compose_vision
from .vision_literal_cohort import admit_vision_cohort,native_vision_loaders
from .vision_literal_runtime import build_vision_runtime,vision_state_sha256
from .vision_literal_train import (ROOT,_json,_write,_sealed,_check_seal,_space,
    read_vision_fit,_runtime_device,storage)


FORMAT='original_vision_selected_prediction_v1'
PROFILE='original_vision_pretest_validation_profile_v1'
REPORT='original_vision_center_completed_report_v1'
IMPORTED_SHA=_file_hash(__file__)
POLICY=dict(prediction='exact typed prediction/target/mask/input replay',
    native_mae_atol_rad=2e-6,native_mae_rtol=2e-6,
    native_aggregation='sample-count weighted original per-batch masked L1, float32 accumulation',
    pooled_aggregation='float64 absolute-error sum / exact valid-angle count',
    scope='fixed before validation; no test-driven tolerance or checkpoint changes')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported vision evaluation source changed')
    return {p:_file_hash(ROOT/p) for p in ('egoemg_repro/vision_literal_eval.py',
        'egoemg_repro/vision_literal_train.py','egoemg_repro/metrics.py','egoemg_repro/classic_emg_report.py')}


def _process():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=fields[19],uid=os.getuid())


def _record(error,count):
    return dict(absolute_error_sum_rad=float(error),valid_angle_count=int(count),
        mae_deg=float(error)/int(count)*180/math.pi if count else None)


def _population_public(users,test_splits,episodes):
    if (set(users)!={'train','val','test'} or set(test_splits)!={'user','gesture','both'} or
        type(episodes) is not int or episodes<=0 or any(not isinstance(u,str) or not u for values in [*users.values(),*test_splits.values()] for u in values)):
        raise ValueError('actual episode/user population metadata required')
    sets={role:set(values) for role,values in users.items()}
    return dict(episode_records=episodes,unique_subjects_by_role={role:len(values) for role,values in sets.items()},
        cross_role_user_intersection_counts={a+'_'+b:len(sets[a]&sets[b]) for a,b in (('train','val'),('train','test'),('val','test'))},
        train_test_user_intersection_by_split={split:len(sets['train']&set(values)) for split,values in test_splits.items()},
        counts_describe='released metadata subject keys, not independently verified natural-person identities')


def vision_population(cohort):
    cohort.check();users={};splits={name:set() for name in ('user','gesture','both')}
    for role,ds in cohort.datasets.items():
        users[role]=sorted({str(ds.store.episodes[w['episode']]['subject']) for w in ds.windows})
        if role=='test':
            for w in ds.windows:splits[w['split']].add(str(ds.store.episodes[w['episode']]['subject']))
    splits={name:sorted(values) for name,values in splits.items()}
    result=dict(users_by_role=users,test_users_by_split=splits,
        public=_population_public(users,splits,len(cohort.datasets['train'].store.episodes)))
    cohort.check();return result


def vision_statistics(parts,*,samples,batch_size,role):
    """Recompute raw 22D sums, counts and both aggregation definitions."""
    import torch
    if role not in ('val','test') or type(samples) is not int or samples<=0 or samples%4 or type(batch_size) is not int or batch_size<=0:
        raise ValueError('complete repeated vision evaluation population required')
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
            not torch.equal(part['unique'],part['index']%(samples//2)) or
            not torch.equal(part['repeat'],part['index']//(samples//2)) or
            any(len(part[key])!=n for key in ('hand','split','user','center')) or
            not isinstance(part['input_sha256'],str) or len(part['input_sha256'])!=64):
            raise ValueError('typed original tuple, metadata or ordered repeat coverage differs')
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
    result=dict(format='original_vision_center_statistics_v1',role=role,consumed_samples=samples,
        unique_samples=samples//2,duplicate_consumptions=samples//2,batch_sizes=expected_sizes,
        native_batch_weighted_mae_rad=float(weighted/samples),
        unique_rule='first original repetition only; duplicated consumptions remain in native and consumed metrics')
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
        raise ValueError('vision evaluation source or fixed agreement policy differs')
    file=directory/'predictions.pt'
    if _file_hash(file)!=value['payload_sha256']:raise ValueError('actual prediction file bytes differ')
    payload=torch.load(str(file),map_location='cpu',weights_only=True)
    if type(payload) is not dict or set(payload)!={'parts','population'}:raise ValueError('complete prediction/population payload required')
    parts=payload['parts'];population=payload['population']
    public=_population_public(population['users_by_role'],population['test_users_by_split'],population['public']['episode_records'])
    if public!=population['public'] or public!=value['population'] or _hash(population)!=value['population_sha256']:
        raise ValueError('actual private population and public intersections differ')
    if _payload_sha(parts)!=value['prediction_sha256']:raise ValueError('typed input/prediction replay digest differs')
    statistics=vision_statistics(parts,samples=value['samples'],batch_size=value['batch_size'],role=value['role'])
    if statistics!=value['statistics']:raise ValueError('raw prediction sufficient statistics differ from report')
    if value['kind']=='native':
        metrics=_json(directory/'native-metrics.json')
        if metrics!=value['native_metrics']:raise ValueError('actual native metric sidecar differs')
        _close_native(statistics,metrics,value['role'])
    elif value['kind']!='replay' or value['native_metrics']!={}:raise ValueError('independent replay cannot invent native metrics')
    if value['model_before_sha256']!=value['model_after_sha256'] or value['scientific_eligible'] is not False or value['full_paper_complete'] is not False:
        raise ValueError('prediction-only run mutated model or claimed completion')
    return value


def read_vision_pass(directory):
    import torch
    value=_load_pass(directory);fit=read_vision_fit(value['fit_path'],require_full=value['training_scope']=='full')
    identity=_json(Path(value['fit_path'])/'identity.json')
    if (not fit['stage_complete'] or value['fit_result_sha256']!=fit['sha256'] or
        value['selected_best']!=fit['selected_best'] or value['cohort_sha256']!=identity['cohort']['sha256'] or
        value['samples']!=identity['cohort']['counts'][value['role']] or
        value['batch_size']!=identity['policy']['batch_size_by_role'][value['role']] or
        value['training_scope']!=fit['scope'] or value['training_full_terminated'] is not fit['full_training_terminated']):
        raise ValueError('pass differs from actual completed fit, cohort or selected best')
    selected=_safe(Path(value['fit_path'])/'checkpoints'/fit['selected_best']['file'])
    guard=_StatGuard([selected])
    if _file_hash(selected)!=fit['selected_best']['checkpoint_sha256']:raise ValueError('raw selected model checkpoint bytes differ')
    checkpoint=torch.load(str(selected),map_location='cpu',weights_only=False)
    selected_model=vision_state_sha256(checkpoint['state_dict']);del checkpoint
    guard.check()
    if value['model_before_sha256']!=selected_model or value['model_after_sha256']!=selected_model:
        raise ValueError('pass model digest differs from actual raw selected model checkpoint')
    if value['role']=='test':
        profile=read_vision_profile(value['profile_path'])
        if (value['profile_sha256']!=profile['sha256'] or value['fit_result_sha256']!=profile['fit_result_sha256'] or
            value['started_ns']<=profile['finished_ns']):raise ValueError('test is not bound to an earlier frozen validation profile')
    elif value['profile_path'] is not None or value['profile_sha256'] is not None:
        raise ValueError('validation may not consume a test profile')
    if value['kind']=='replay':
        ref=read_vision_pass(value['reference_path'])
        if (ref['kind']!='native' or ref['role']!=value['role'] or ref['fit_result_sha256']!=value['fit_result_sha256'] or
            ref['sha256']!=value['reference_sha256'] or ref['prediction_sha256']!=value['prediction_sha256'] or
            ref['population_sha256']!=value['population_sha256'] or ref['model_before_sha256']!=selected_model or
            ref['process']==value['process'] or value['started_ns']<=ref['finished_ns']):
            raise ValueError('fresh-process exact native prediction replay differs')
    elif value['reference_path'] is not None or value['reference_sha256'] is not None:
        raise ValueError('native pass cannot claim an independent replay')
    return value


def read_vision_profile(directory):
    directory=_safe(directory);profile=_json(directory/'profile.json');_check_seal(profile,PROFILE)
    reference=read_vision_pass(profile['reference_path']);replay=read_vision_pass(profile['replay_path'])
    if (profile['fixed_agreement_policy']!=POLICY or reference['role']!='val' or reference['kind']!='native' or
        replay['role']!='val' or replay['kind']!='replay' or replay['reference_sha256']!=reference['sha256'] or
        profile['reference_sha256']!=reference['sha256'] or profile['replay_sha256']!=replay['sha256'] or
        profile['fit_result_sha256']!=reference['fit_result_sha256'] or
        profile['finished_ns']<=max(reference['finished_ns'],replay['finished_ns'])):
        raise ValueError('frozen independent validation replay profile differs')
    return profile


def freeze_vision_validation(reference,replay,output):
    ref=read_vision_pass(reference);other=read_vision_pass(replay)
    if ref['role']!='val' or ref['kind']!='native' or other['role']!='val' or other['kind']!='replay' or other['reference_sha256']!=ref['sha256']:
        raise ValueError('complete native validation and fresh exact replay required')
    output=_safe(output,exists=False);_space(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    value=_sealed(dict(format=PROFILE,reference_path=str(_safe(reference)),replay_path=str(_safe(replay)),
        reference_sha256=ref['sha256'],replay_sha256=other['sha256'],fit_result_sha256=ref['fit_result_sha256'],
        fixed_agreement_policy=copy.deepcopy(POLICY),finished_ns=time.time_ns(),scientific_eligible=False,full_paper_complete=False))
    _write(output/'profile.json',value);return read_vision_profile(output)


def _evaluation_loader(cohort,policy,role):
    if policy['batch_size_by_role'][role]==360:return native_vision_loaders(cohort)[role]
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0;dm.batch_size=policy['batch_size_by_role'][role]
    return dm._make_loader(cohort.datasets[role],shuffle=False)


def _capture(batch,out,ds,parts):
    import torch
    pred,target,mask=out;n=len(pred);position=sum(len(p['index']) for p in parts)
    if (not torch.equal(target,batch['joint_angles']) or not torch.equal(mask,batch['label_valid_mask']) or
        batch['vision_sample_index'].tolist()!=list(range(position,position+n))):
        raise ValueError('original vision tuple target, generated mask or sequential input differs')
    for j,index in enumerate(range(position,position+n)):
        a=len(ds.windows);w=ds.windows[index%a];hand=('left','right')[(index%(2*a))//a]
        if (batch['role'][j]!=ds.role or batch['hand'][j]!=hand or batch['split'][j]!=w['split'] or
            int(batch['episode'][j])!=w['episode'] or int(batch['center'][j])!=w['center'] or
            int(batch['jitter_offset'][j])!=0 or batch['subject'][j]!=ds.store.episodes[w['episode']]['subject']):
            raise ValueError('original evaluation provenance differs from admitted cohort')
    def cpu(value):return value.detach().cpu().clone()
    parts.append(dict(pred=cpu(pred),target=cpu(target),mask=cpu(mask),index=cpu(batch['vision_sample_index']),
        unique=cpu(batch['vision_unique_sample_index']),repeat=cpu(batch['vision_repeat_index']),
        vision=cpu(batch['vision_valid_mask']),center=cpu(batch['center']),hand=list(batch['hand']),split=list(batch['split']),
        user=list(batch['subject']),input_sha256=storage.state_digest(batch)))


def run_vision_pass(cohort,fit_path,output,*,role,kind,reference=None,profile=None,require_full=True):
    import torch
    if role not in ('val','test') or kind not in ('native','replay'):raise ValueError('explicit native/replay and val/test required')
    cohort.check();fit_path=_safe(fit_path);fit=read_vision_fit(fit_path,require_full=require_full)
    if not fit['stage_complete']:raise ValueError('partial engineering or full fit cannot be evaluated as completed')
    identity=_json(fit_path/'identity.json')
    if identity['cohort']!=cohort.identity:raise ValueError('actual rehashed evaluation cohort differs from training')
    frozen=None;ref=None
    if role=='test':
        if profile is None:raise ValueError('freeze independent validation profile before test')
        frozen=read_vision_profile(profile)
        if frozen['fit_result_sha256']!=fit['sha256']:raise ValueError('pretest profile belongs to another fit')
    elif profile is not None:raise ValueError('validation cannot consume a test profile')
    if kind=='replay':
        if reference is None:raise ValueError('native pass required before fresh replay')
        ref=read_vision_pass(reference)
        if ref['kind']!='native' or ref['role']!=role or ref['fit_result_sha256']!=fit['sha256'] or ref['process']==_process():
            raise ValueError('fresh process and matching native reference required')
    elif reference is not None:raise ValueError('native pass has no reference')
    policy=identity['policy'];device=_runtime_device(policy);output=_safe(output,exists=False);_space(output.parent)
    checkpoint=fit_path/'checkpoints'/fit['selected_best']['file']
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)['state_dict']
    prepared=build_vision_runtime(cohort.upstream,cohort.config,state=state,expected_initialization=identity['construction'])
    population=vision_population(cohort)
    before=vision_state_sha256(prepared.module.state_dict());parts=[];start=time.time_ns()
    guard=_StatGuard([fit_path/'identity.json',fit_path/'fit-result.json',checkpoint,*[ROOT/p for p in _sources()]])
    loader=_evaluation_loader(cohort,policy,role);original=prepared.module.forward
    def observed(batch):
        cohort.guard.check();guard.check();out=original(batch);_capture(batch,out,cohort.datasets[role],parts);return out
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
        statistics=vision_statistics(parts,samples=len(cohort.datasets[role]),batch_size=policy['batch_size_by_role'][role],role=role)
        if kind=='native':_close_native(statistics,native,role)
        if ref is not None and _payload_sha(parts)!=ref['prediction_sha256']:raise ValueError('fixed exact typed prediction/input replay failed')
        cohort.check();guard.check();prepared.check();after=vision_state_sha256(prepared.module.state_dict())
        if before!=after:raise ValueError('selected model changed during evaluation')
        temporary=output/('.predictions-'+uuid.uuid4().hex)
        try:
            with temporary.open('xb') as f:torch.save(dict(parts=parts,population=population),f);f.flush();os.fsync(f.fileno())
            os.link(temporary,output/'predictions.pt')
        finally:
            if temporary.exists():temporary.unlink()
        if kind=='native':_write(output/'native-metrics.json',native)
        value=_sealed(dict(format=FORMAT,role=role,kind=kind,fit_path=str(fit_path),fit_result_sha256=fit['sha256'],
            training_scope=fit['scope'],training_full_terminated=fit['full_training_terminated'],cohort_sha256=cohort.identity['sha256'],
            selected_best=copy.deepcopy(fit['selected_best']),sources_sha256=_sources(),fixed_agreement_policy=copy.deepcopy(POLICY),
            samples=len(cohort.datasets[role]),batch_size=policy['batch_size_by_role'][role],statistics=statistics,
            population=population['public'],population_sha256=_hash(population),
            payload_sha256=_file_hash(output/'predictions.pt'),prediction_sha256=_payload_sha(parts),native_metrics=native,
            native_aggregation_agrees=kind=='native',model_before_sha256=before,model_after_sha256=after,
            process=_process(),started_ns=start,finished_ns=time.time_ns(),
            profile_path=str(_safe(profile)) if profile is not None else None,profile_sha256=frozen['sha256'] if frozen is not None else None,
            reference_path=str(_safe(reference)) if reference is not None else None,reference_sha256=ref['sha256'] if ref is not None else None,
            scientific_eligible=False,full_paper_complete=False))
        _write(output/'pass.json',value);return read_vision_pass(output)
    finally:prepared.module.forward=original


def build_vision_report(recompute):
    replay=read_vision_pass(recompute)
    if replay['role']!='test' or replay['kind']!='replay':raise ValueError('completed native test and independent test replay required')
    native=read_vision_pass(replay['reference_path']);fit=read_vision_fit(replay['fit_path'],require_full=replay['training_scope']=='full')
    identity=_json(Path(replay['fit_path'])/'identity.json');statistics=copy.deepcopy(replay['statistics'])
    users=sorted(statistics['consumed']['all']['by_user']);names={user:'user_'+str(i).zfill(3) for i,user in enumerate(users)}
    def sanitize(value):
        if isinstance(value,dict):
            return {key:({names[u]:sanitize(v) for u,v in item.items()} if key=='by_user' else sanitize(item)) for key,item in value.items()}
        if isinstance(value,list):return [sanitize(v) for v in value]
        return value
    statistics=sanitize(statistics)
    return _sealed(dict(format=REPORT,scientific_eligible=fit['full_training_terminated'],full_paper_complete=False,
        scope='available legacy cohort; single rank/workers0; original vision semantics; independent train-only validation',
        paper_numerical_equivalence_established=False,variant=identity['policy']['variant'],
        training=dict(scope=fit['scope'],completed_epochs=fit['completed_epochs'],optimizer_executions=fit['optimizer_executions'],
            full_training_terminated=fit['full_training_terminated'],selected_best=fit['selected_best'],fit_result_sha256=fit['sha256'],
            execution_policy=copy.deepcopy(identity['policy'])),
        test=statistics,population=copy.deepcopy(replay['population']),native_metrics={k:v for k,v in native['native_metrics'].items() if k in PUBLIC_NATIVE_METRICS},
        fixed_agreement_policy=copy.deepcopy(POLICY),cohort_sha256=replay['cohort_sha256'],
        validation_profile_sha256=replay['profile_sha256'],native_test_sha256=native['sha256'],independent_test_sha256=replay['sha256'],
        sources_sha256=_sources(),prediction_sha256=replay['prediction_sha256']))


def render_vision_report(report):
    training=report['training'];test=report['test'];scope='完整训练实验' if report['scientific_eligible'] else '工程验证；未完成正式完整训练'
    lines=['# DINOv2 视觉中心评价','',scope,'',
        f"模型：{report['variant']}；训练完成 {training['completed_epochs']} 轮，实际更新 {training['optimizer_executions']} 次。",'',
        '激活重计算：'+('开启（显式适配）' if training['execution_policy']['activation_checkpointing'] else '关闭')+'。','',
        f"验证选择：{training['selected_best']['file']}；不以最后一轮替代验证最优。",'',
        f"测试消费 {test['consumed_samples']} 手样本，唯一 {test['unique_samples']}，重复消费 {test['duplicate_consumptions']}。",'',
        '| 统计人口 | MAE (°) | 有效角度数 | 弧度绝对误差和 |','| --- | ---: | ---: | ---: |']
    for label,record in [('全部重复消费',test['consumed']['all']['overall']),
        ('唯一首遍',test['unique_first_repeat']['all']['overall']),('重复消费中的有图子集',test['consumed']['vision_valid']['overall'])]:
        error='—' if record['mae_deg'] is None else f"{record['mae_deg']:.6f}"
        lines.append(f"| {label} | {error} | {record['valid_angle_count']} | {record['absolute_error_sum_rad']:.10g} |")
    lines+=['',f"原生按样本数加权的批量 MAE：{test['native_batch_weighted_mae_rad']:.10g} rad。",'',
        '该指标与按有效角度数汇总的 pooled MAE 分开报告。缺图仍保留原生成标签 mask；不增加腕角过滤。','',
        '采用可用旧版 EgoEMG 队列、独立 train 留出验证、单 rank/workers0；未证明与论文原表或缺失的合并训练来源数值等价。',
        '当前报告只验收所列实验，整篇论文复现仍未完成。','',
        '## 完整公开证据','', '以下与 metrics.json 一致，包含逐手、逐条件、匿名逐用户、逐角、指组、用户交集与来源摘要。','',
        '```json',json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),'```','']
    return '\n'.join(lines)


def _completion(output,report):
    return _sealed(dict(format='original_vision_public_report_completion_v1',report_sha256=report['sha256'],
        files_sha256={name:_file_hash(_safe(output/name)) for name in ('metrics.json','metrics.md')},
        scientific_eligible=report['scientific_eligible'],full_paper_complete=False))


def verify_vision_report(recompute,output):
    output=_safe(output);report=build_vision_report(recompute)
    if (_json(output/'metrics.json')!=report or _safe(output/'metrics.md').read_text()!=render_vision_report(report) or
        _json(output/'completion.json')!=_completion(output,report)):
        raise ValueError('public vision report or last completion marker differs from reopened raw chain')
    return report


def publish_vision_report(recompute,output):
    report=build_vision_report(recompute);output=_safe(output,exists=False);_space(output.parent)
    output.mkdir(mode=0o700,exist_ok=False);_write(output/'metrics.json',report)
    with (output/'metrics.md').open('x') as f:f.write(render_vision_report(report));f.flush();os.fsync(f.fileno())
    if report!=build_vision_report(recompute):raise ValueError('raw chain changed during public report preparation')
    _write(output/'completion.json',_completion(output,report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('pass')
    for name in ('data','crops','upstream','fit','output'):run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--variant',choices=('vit_small','vit_base','vit_large'),default='vit_small')
    run.add_argument('--role',choices=('val','test'),required=True);run.add_argument('--kind',choices=('native','replay'),required=True)
    run.add_argument('--profile',type=Path);run.add_argument('--reference',type=Path);run.add_argument('--engineering',action='store_true')
    freeze=sub.add_parser('freeze')
    for name in ('reference','replay','output'):freeze.add_argument('--'+name,type=Path,required=True)
    for name in ('report','verify'):
        public=sub.add_parser(name);public.add_argument('--recompute',type=Path,required=True);public.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());action=args.pop('action')
    if action=='freeze':result=freeze_vision_validation(**args)
    elif action=='report':result=publish_vision_report(**args)
    elif action=='verify':result=verify_vision_report(**args)
    else:
        data=args.pop('data');crops=args.pop('crops');upstream=args.pop('upstream');variant=args.pop('variant');engineering=args.pop('engineering')
        fit=args.pop('fit');expected=_json(fit/'identity.json')['cohort']
        cohort=admit_vision_cohort(data,crops,upstream=upstream,config=compose_vision(upstream,variant),expected=expected)
        result=run_vision_pass(cohort,fit,require_full=not engineering,**args)
    print(json.dumps(dict(format=result['format'],scientific_eligible=result['scientific_eligible'],full_paper_complete=False)))


if __name__=='__main__':main()
