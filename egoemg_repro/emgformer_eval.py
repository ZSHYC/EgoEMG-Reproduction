"""Original EMGFormer selected point evaluation, without visual repetitions."""
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np

from .classic_emg_report import JOINT_NAMES,FINGERS,PD_GROUPS
from .metrics import AngleAccumulator


_FIELDS={'pred','target','mask','index','unique','repeat','center','episode','hand','split','user',
    'input_sha256','sample_input_sha256'}


def _digest_string(value):
    return type(value)is str and re.fullmatch('[0-9a-f]{64}',value)is not None


def emgformer_statistics(parts,*,samples,batch_size,role):
    """One native hand/window consumption; retain both native and pooled MAE."""
    import torch
    if (role not in ('val','test')or type(samples)is not int or samples<=0 or samples%2 or
        type(batch_size)is not int or batch_size<=0 or type(parts)is not list):
        raise ValueError('complete single-consumption EMGFormer point population required')
    expected_sizes=[min(batch_size,samples-i)for i in range(0,samples,batch_size)]
    if len(parts)!=len(expected_sizes):raise ValueError('complete native point batch coverage required')
    acc=AngleAccumulator('center');sums=np.zeros(22,np.float64);counts=np.zeros(22,np.int64)
    weighted=torch.tensor(0.,dtype=torch.float32);position=0;split_users={};paired={}
    for part,n in zip(parts,expected_sizes):
        try:
            if type(part)is not dict or set(part)!=_FIELDS:raise ValueError('complete native point tuple required')
            p,t,m=part['pred'],part['target'],part['mask'];index=torch.arange(position,position+n,dtype=torch.int64)
            if (any(not torch.is_tensor(x)or x.device.type!='cpu'for x in (p,t,m))or
                p.shape!=(n,22,1)or t.shape!=p.shape or m.shape!=(n,1)or
                p.dtype not in (torch.float32,torch.bfloat16)or t.dtype!=torch.float32 or m.dtype!=torch.bool or
                not bool(torch.isfinite(p).all())or not bool(torch.isfinite(t).all())):
                raise ValueError('typed finite original point prediction/target/mask required')
            for name in ('index','unique','repeat','center','episode'):
                x=part[name]
                if not torch.is_tensor(x)or x.device.type!='cpu'or x.dtype!=torch.int64 or x.shape!=(n,):
                    raise ValueError('typed native point source metadata required')
            if (not torch.equal(part['index'],index)or not torch.equal(part['unique'],index)or
                bool(part['repeat'].any())or bool((part['center']<0).any())or bool((part['episode']<0).any())or
                any(type(part[k])is not list or len(part[k])!=n for k in ('hand','split','user','sample_input_sha256'))or
                not _digest_string(part['input_sha256'])or not all(_digest_string(x)for x in part['sample_input_sha256'])):
                raise ValueError('native point order/unique source/repetition/input digest differs')
            mask=m[:,None,:].expand_as(t);prediction=p.float()
            batch_mae=(prediction-t).abs()[mask].mean()if bool(mask.any())else prediction.sum()*0.
            weighted+=batch_mae*n
            for j,source in enumerate(index.tolist()):
                hand,split,user=(part[k][j]for k in ('hand','split','user'))
                if (hand!=('left'if source<samples//2 else'right')or
                    split not in (('train',)if role=='val'else('user','gesture','both'))or type(user)is not str or not user):
                    raise ValueError('native hand/split/subject metadata differs')
                center=int(part['center'][j]);episode=int(part['episode'][j]);window=source%(samples//2)
                metadata=(center,episode,split,user)
                if source<samples//2:
                    if any(v[0]==center for v in paired.values()):raise ValueError('duplicate native window center')
                    paired[window]=metadata
                elif paired.get(window)!=metadata:raise ValueError('paired native hand/window metadata differs')
                pred=prediction[j].numpy();target=t[j].numpy();valid=m[j].numpy()
                acc.update(pred,target,valid,hand=hand,split=split,user=user,center_index=0)
                error=np.abs(pred.astype(np.float64)-target.astype(np.float64))[:,valid]
                sums+=error.sum(axis=1);counts+=int(valid.sum())
                cell=split_users.setdefault(split,{}).setdefault(user,[0.,0]);cell[0]+=float(error.sum());cell[1]+=int(error.size)
        except (KeyError,TypeError,IndexError,AttributeError,RuntimeError)as exc:
            raise ValueError('malformed original EMGFormer point payload')from exc
        position+=n
    report=acc.report()
    def record(values):return AngleAccumulator._record([float(values[0]),int(values[1])])
    report['by_joint']=[dict(index=i,name=JOINT_NAMES[i],**record([sums[i],counts[i]]))for i in range(22)]
    report['by_finger']={k:record([sums[list(ids)].sum(),counts[list(ids)].sum()])for k,ids in FINGERS.items()}
    report['by_pd_group']={k:record([sums[list(ids)].sum(),counts[list(ids)].sum()])for k,ids in PD_GROUPS.items()}
    report['by_split_user_summary']={}
    for split,users in sorted(split_users.items()):
        means=[e/c*180/math.pi for e,c in users.values()if c]
        report['by_split_user_summary'][split]=dict(user_count=len(means),mean_mae_deg=float(np.mean(means))if means else None,
            std_mae_deg=float(np.std(means))if means else None,aggregation='unweighted user means; ddof=0')
    if int(counts.sum())!=report['overall']['valid_angle_count']or not math.isclose(float(sums.sum()),report['overall']['absolute_error_sum_rad'],rel_tol=2e-12,abs_tol=1e-9):
        raise ValueError('native point joint and pooled sufficient statistics do not conserve')
    return dict(format='original_emgformer_single_consumption_point_statistics_v1',role=role,consumed_samples=samples,
        unique_samples=samples,duplicate_consumptions=0,batch_sizes=expected_sizes,
        native_batch_weighted_mae_rad=float(weighted/samples),all=report)


# These helpers concern generic metadata/typed metrics, not visual populations.
import argparse
import copy
import os
import time
import uuid
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from . import emgformer_train as train
from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard
from .emgformer_runtime import build_emgformer_runtime,emgformer_state_sha256
from .vision_literal_train import _json,_write,_sealed,_check_seal
from .vision_literal_eval import POLICY,_process,_population_public,vision_population,_close_native

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_emgformer_selected_point_pass_v1'
PROFILE='original_emgformer_pretest_point_profile_v1'
REPORT='original_emgformer_completed_point_report_v1'
_READING=ContextVar('emgformer_evaluation_artifact_graph',default=frozenset())


def _acyclic_read(function):
    @wraps(function)
    def checked(directory):
        directory=_safe(directory);key=(function.__name__,str(directory));seen=_READING.get()
        if key in seen or len(seen)>=8:raise ValueError('cyclic or excessive native point artifact references')
        token=_READING.set(seen|{key})
        try:return function(directory)
        finally:_READING.reset(token)
    return checked


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported EMGFormer evaluation source changed')
    return dict(train._sources(),**{n:_file_hash(ROOT/n)for n in ('egoemg_repro/emgformer_eval.py',
        'egoemg_repro/vision_literal_eval.py','egoemg_repro/classic_emg_report.py','egoemg_repro/metrics.py')})


def _check(value,format):
    _check_seal(value,format)
    if value['scientific_eligible']is not False or value['full_paper_complete']is not False:
        raise ValueError('private native point pass/profile cannot claim scientific completion')


def _identity(fit_path):
    fit_path=_safe(fit_path);fit=train.read_emgformer_fit(fit_path,require_full=False)
    if not fit['stage_complete']:raise ValueError('complete native EMGFormer fit required before selected evaluation')
    return fit,_json(fit_path/'identity.json')


def _load_pass(directory):
    import torch
    directory=_safe(directory);v=_json(directory/'pass.json');_check(v,FORMAT)
    if v['sources_sha256']!=_sources()or v['fixed_agreement_policy']!=POLICY:
        raise ValueError('native EMGFormer evaluation source/fixed agreement differs')
    if v['role']not in ('val','test')or v['kind']not in ('native','replay'):
        raise ValueError('explicit native point role and pass kind required')
    if v['kind']=='native'and(v['reference_path']is not None or v['reference_sha256']is not None or v['native_aggregation_agrees']is not True):
        raise ValueError('native point pass cannot be relabeled from independent replay')
    if (type(v['started_ns'])is not int or type(v['finished_ns'])is not int or not 0<v['started_ns']<=v['finished_ns']or
        set(v['process'])!={'pid','uid','start_ticks'}or type(v['process']['pid'])is not int or v['process']['pid']<=0 or
        type(v['process']['uid'])is not int or v['process']['uid']<0 or type(v['process']['start_ticks'])is not str or
        not v['process']['start_ticks'].isdigit()or v['native_point_center_selection_exact']is not True or
        type(v['optimizer_updates'])is not int or v['optimizer_updates']!=0):
        raise ValueError('native point execution/chronology/witness differs')
    fit,identity=_identity(v['fit_path'])
    selected=torch.load(str(_safe(Path(v['fit_path'])/'checkpoints'/fit['selected_best']['file'])),map_location='cpu',weights_only=False)
    if (v['fit_result_sha256']!=fit['sha256']or v['selected_best']!=fit['selected_best']or
        v['cohort_sha256']!=identity['cohort']['sha256']or v['training_scope']!=fit['scope']or
        v['training_full_terminated']is not fit['full_training_terminated']or
        v['samples']!=identity['cohort']['counts'][v['role']]or
        v['batch_size']!=identity['policy']['batch_size_by_role']['val']or
        v['model_before_sha256']!=v['model_after_sha256']or
        v['model_before_sha256']!=selected['emgformer_fit']['records'][-1]['model_sha256']):
        raise ValueError('native point fit/selected state/role/population differs')
    payload_path=directory/'predictions.pt'
    if _file_hash(payload_path)!=v['payload_sha256']:raise ValueError('actual native point payload bytes changed')
    payload=torch.load(str(payload_path),map_location='cpu',weights_only=True)
    if type(payload)is not dict or set(payload)!={'parts','population'}:raise ValueError('complete typed point/population payload required')
    parts,population=payload['parts'],payload['population']
    public=_population_public(population['users_by_role'],population['test_users_by_split'],population['public']['episode_records'])
    if (public!=population['public']or public!=v['population']or _hash(population)!=v['population_sha256']or
        train.storage.state_digest(parts)!=v['prediction_sha256']):raise ValueError('actual native point metadata/input digest differs')
    statistics=emgformer_statistics(parts,samples=v['samples'],batch_size=v['batch_size'],role=v['role'])
    if statistics!=v['statistics']:raise ValueError('raw native point statistics differ')
    if sorted({u for p in parts for u in p['user']})!=population['users_by_role'][v['role']]:
        raise ValueError('actual point subject population differs')
    if v['kind']=='native':
        metrics=_json(directory/'native-metrics.json')
        if metrics!=v['native_metrics']or v['native_aggregation_agrees']is not True:raise ValueError('native point metric sidecar differs')
        _close_native(statistics,metrics,v['role'])
    elif v['native_metrics']!={}or v['native_aggregation_agrees']is not False:
        raise ValueError('independent point replay cannot invent native metrics')
    return v


@_acyclic_read
def read_emgformer_pass(directory):
    v=_load_pass(directory)
    if v['role']=='val':
        if v['profile_path']is not None or v['profile_sha256']is not None:raise ValueError('validation cannot consume a pretest profile')
    else:
        if v['profile_path']is None:raise ValueError('freeze validation before native point test')
        frozen=read_emgformer_profile(v['profile_path'])
        if (frozen['sha256']!=v['profile_sha256']or frozen['fit_result_sha256']!=v['fit_result_sha256']or
            frozen['finished_ns']>v['started_ns']):raise ValueError('native point profile/fit/pretest chronology differs')
    if v['kind']=='native':
        if v['reference_path']is not None or v['reference_sha256']is not None:raise ValueError('native point pass cannot consume a replay reference')
    else:
        reference=read_emgformer_pass(v['reference_path'])
        if (reference['kind']!='native'or reference['role']!=v['role']or reference['sha256']!=v['reference_sha256']or
            reference['fit_result_sha256']!=v['fit_result_sha256']or reference['process']==v['process']or
            reference['finished_ns']>v['started_ns']or reference['prediction_sha256']!=v['prediction_sha256']or
            reference['population_sha256']!=v['population_sha256']or reference['profile_sha256']!=v['profile_sha256']):
            raise ValueError('fresh matching exact native point reference required')
    return v


@_acyclic_read
def read_emgformer_profile(directory):
    v=_json(_safe(directory)/'profile.json');_check(v,PROFILE)
    reference=read_emgformer_pass(v['reference_path']);replay=read_emgformer_pass(v['replay_path'])
    if (v['sources_sha256']!=_sources()or v['fixed_agreement_policy']!=POLICY or
        reference['role']!='val'or reference['kind']!='native'or replay['role']!='val'or replay['kind']!='replay'or
        replay['reference_sha256']!=reference['sha256']or reference['sha256']!=v['reference_sha256']or
        replay['sha256']!=v['replay_sha256']or v['fit_result_sha256']!=reference['fit_result_sha256']or
        type(v['finished_ns'])is not int or v['finished_ns']<max(reference['finished_ns'],replay['finished_ns'])):
        raise ValueError('complete frozen original point validation/profile required')
    return v


def freeze_emgformer_validation(reference,replay,output):
    a=read_emgformer_pass(reference);b=read_emgformer_pass(replay)
    if a['role']!='val'or a['kind']!='native'or b['role']!='val'or b['kind']!='replay'or b['reference_sha256']!=a['sha256']:
        raise ValueError('matching independently replayed native point validation required')
    output=_safe(output,exists=False);train.storage._space(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    v=_sealed(dict(format=PROFILE,reference_path=str(_safe(reference)),replay_path=str(_safe(replay)),
        reference_sha256=a['sha256'],replay_sha256=b['sha256'],fit_result_sha256=a['fit_result_sha256'],
        sources_sha256=_sources(),fixed_agreement_policy=copy.deepcopy(POLICY),finished_ns=time.time_ns(),
        scientific_eligible=False,full_paper_complete=False))
    _write(output/'profile.json',v);return read_emgformer_profile(output)


def _loader(cohort,policy,role):
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cohort.config);dm.batch_size=policy['batch_size_by_role']['val'];dm.num_workers=dm.eval_num_workers=policy['workers']
    return dm._make_loader(cohort.datasets[role],shuffle=False,num_workers=dm.eval_num_workers)


def _capture(batch,out,ds,parts):
    import torch
    pred,target,mask=out;n=len(pred);position=sum(len(p['index'])for p in parts);index=batch['emgformer_sample_index']
    if (index.tolist()!=list(range(position,position+n))or not torch.equal(batch['emgformer_unique_sample_index'],index)or
        not torch.equal(target,batch['joint_angles'])or not torch.equal(mask,batch['label_valid_mask'])):
        raise ValueError('actual point source order/unique index/target/mask differs')
    users=[]
    for j,i in enumerate(index.tolist()):
        w=ds.windows[i%len(ds.windows)];subject=str(ds.store.episodes[w['episode']]['subject']);users.append(subject)
        if int(batch['episode'][j])!=w['episode']or int(batch['center'][j])!=w['center']or int(batch['jitter_offset'][j])!=0:
            raise ValueError('actual point window/episode/center binding differs')
        if 'episode_subject'in batch and batch['episode_subject'][j]!=subject:raise ValueError('actual point subject differs')
    def cpu(x):return x.detach().cpu().clone()
    sample_sha=[train.storage.state_digest(dict(emg=batch['emg'][j],target=target[j],mask=mask[j],index=index[j]))for j in range(n)]
    parts.append(dict(pred=cpu(pred),target=cpu(target),mask=cpu(mask),index=cpu(index),
        unique=cpu(batch['emgformer_unique_sample_index']),repeat=cpu(batch['emgformer_repeat_index']),
        center=cpu(batch['center']),episode=cpu(batch['episode']),hand=list(batch['hand']),split=list(batch['split']),
        user=users,input_sha256=train.storage.state_digest(batch),sample_input_sha256=sample_sha))


def run_emgformer_pass(cohort,fit_path,output,*,role,kind,reference=None,profile=None,require_full=True):
    import torch
    if role not in ('val','test')or kind not in ('native','replay')or type(require_full)is not bool:
        raise ValueError('explicit native point evaluation role/kind/scope required')
    cohort.check();fit_path=_safe(fit_path);fit=train.read_emgformer_fit(fit_path,require_full=require_full)
    if not fit['stage_complete']:raise ValueError('completed selected EMGFormer fit required')
    identity=_json(fit_path/'identity.json')
    if identity['cohort']!=cohort.identity:raise ValueError('actual point evaluation cohort differs from training')
    frozen=None;ref=None
    if role=='test':
        if profile is None:raise ValueError('freeze native point validation before any test')
        frozen=read_emgformer_profile(profile)
        if frozen['fit_result_sha256']!=fit['sha256']:raise ValueError('point profile belongs to another fit')
    elif profile is not None:raise ValueError('native point validation cannot consume test profile')
    if kind=='replay':
        if reference is None:raise ValueError('native point pass required before fresh replay')
        ref=read_emgformer_pass(reference)
        if (ref['kind']!='native'or ref['role']!=role or ref['fit_result_sha256']!=fit['sha256']or ref['process']==_process()or
            ref['profile_sha256']!=(None if frozen is None else frozen['sha256'])):
            raise ValueError('fresh matching native point reference required')
    elif reference is not None:raise ValueError('native point pass cannot consume reference')
    policy=identity['policy'];device=train._runtime_device(policy);output=_safe(output,exists=False);train.storage._space(output.parent)
    checkpoint=fit_path/'checkpoints'/fit['selected_best']['file'];sources=_sources()
    guarded=[fit_path/'identity.json',fit_path/'fit-result.json',fit_path/'epochs.jsonl',fit_path/'checkpoints/last.ckpt',checkpoint,
        *[ROOT/n for n in sources],*([]if profile is None else[_safe(profile)/'profile.json'])]
    guard=_StatGuard(guarded);started=time.time_ns()
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)['state_dict']
    prepared=build_emgformer_runtime(cohort.upstream,cohort.config,state=state,expected_initialization=identity['construction'])
    before=emgformer_state_sha256(prepared.module.state_dict());population=vision_population(cohort);loader=_loader(cohort,policy,role)
    point=train.ValidationWitness(cohort,policy);point.parts=[];point.phase='val';point._observe_forward(prepared.module)
    original_point=prepared.module.forward;parts=[];seen=set();allowed=set(range(len(cohort.datasets[role])))
    evaluation_policy=copy.deepcopy(policy);evaluation_policy['batch_size_by_role']['test']=policy['batch_size_by_role']['val']
    def observe(batch):
        cohort.guard.check();guard.check();train._batch(cohort,evaluation_policy,batch,role,seen,allowed)
        out=original_point(batch);_capture(batch,out,cohort.datasets[role],parts);return out
    prepared.module.forward=observe;native={};output.mkdir(mode=0o700,exist_ok=False)
    try:
        if kind=='native':
            trainer=train.storage.pl.Trainer(accelerator=device['accelerator'],devices=1,precision=policy['precision'],
                logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
            method=trainer.validate if role=='val'else trainer.test
            result=method(prepared.module,dataloaders=loader,verbose=False);native={k:float(v)for k,v in result[0].items()}
            if not all(math.isfinite(v)for v in native.values()):raise ValueError('nonfinite original point metrics')
        else:
            from lightning_fabric.utilities.apply_func import move_data_to_device
            target_device='cpu'if device['accelerator']=='cpu'else'cuda:0';prepared.module.to(target_device).eval()
            with torch.inference_mode():
                for batch in loader:
                    batch=move_data_to_device(batch,target_device)
                    with(nullcontext()if target_device=='cpu'else torch.autocast('cuda',dtype=torch.bfloat16)):
                        prepared.module.forward(batch)
        if seen!=allowed:raise ValueError('partial selected point population')
        statistics=emgformer_statistics(parts,samples=len(seen),batch_size=policy['batch_size_by_role']['val'],role=role)
        if kind=='native':_close_native(statistics,native,role)
        digest=train.storage.state_digest(parts)
        if ref is not None and digest!=ref['prediction_sha256']:raise ValueError('exact typed point/input replay failed')
        cohort.check();guard.check();prepared.check();after=emgformer_state_sha256(prepared.module.state_dict())
        if before!=after:raise ValueError('selected EMGFormer changed during inference')
        temporary=output/('.predictions-'+uuid.uuid4().hex)
        try:
            with temporary.open('xb')as f:torch.save(dict(parts=parts,population=population),f);f.flush();os.fsync(f.fileno())
            os.link(temporary,output/'predictions.pt')
        finally:
            if temporary.exists():temporary.unlink()
        if kind=='native':_write(output/'native-metrics.json',native)
        v=_sealed(dict(format=FORMAT,role=role,kind=kind,fit_path=str(fit_path),fit_result_sha256=fit['sha256'],
            training_scope=fit['scope'],training_full_terminated=fit['full_training_terminated'],cohort_sha256=cohort.identity['sha256'],
            selected_best=copy.deepcopy(fit['selected_best']),sources_sha256=sources,fixed_agreement_policy=copy.deepcopy(POLICY),
            samples=len(seen),batch_size=policy['batch_size_by_role']['val'],statistics=statistics,
            population=population['public'],population_sha256=_hash(population),payload_sha256=_file_hash(output/'predictions.pt'),
            prediction_sha256=digest,native_metrics=native,native_aggregation_agrees=kind=='native',
            native_point_center_selection_exact=True,optimizer_updates=0,model_before_sha256=before,model_after_sha256=after,
            process=_process(),started_ns=started,finished_ns=time.time_ns(),
            profile_path=None if profile is None else str(_safe(profile)),profile_sha256=None if frozen is None else frozen['sha256'],
            reference_path=None if reference is None else str(_safe(reference)),reference_sha256=None if ref is None else ref['sha256'],
            scientific_eligible=False,full_paper_complete=False))
        _write(output/'pass.json',v);return read_emgformer_pass(output)
    finally:
        prepared.module.forward=point.original_forward;train._close_loaders({role:loader})


def build_emgformer_report(test_replay):
    replay=read_emgformer_pass(test_replay)
    if replay['role']!='test'or replay['kind']!='replay':raise ValueError('native point test and fresh replay required')
    native=read_emgformer_pass(replay['reference_path']);fit,identity=_identity(replay['fit_path'])
    users=sorted(replay['statistics']['all']['by_user']);aliases={u:'user_'+str(i).zfill(3)for i,u in enumerate(users)}
    statistics=copy.deepcopy(replay['statistics']);statistics['all']['by_user']={aliases[u]:v for u,v in statistics['all']['by_user'].items()}
    return _sealed(dict(format=REPORT,scientific_eligible=fit['full_training_terminated'],full_paper_complete=False,
        size=identity['policy']['size'],paper_numerical_equivalence_established=False,
        scope='released legacy EgoEMG; native one-consumption point grid; independent train-only validation',
        training=dict(scope=fit['scope'],full_training_terminated=fit['full_training_terminated'],completed_epochs=fit['completed_epochs'],
            optimizer_executions=fit['optimizer_executions'],selected_best=copy.deepcopy(fit['selected_best']),policy=identity['policy']),
        test=statistics,native_test_metrics=native['native_metrics'],population=replay['population'],
        validation_profile_sha256=replay['profile_sha256'],fit_result_sha256=fit['sha256'],
        native_test_sha256=native['sha256'],independent_test_sha256=replay['sha256'],sources_sha256=_sources(),
        actual_test_pass_seconds=(native['finished_ns']-native['started_ns'])/1e9,
        actual_replay_seconds=(replay['finished_ns']-replay['started_ns'])/1e9,
        limitations=identity['cohort']['preserved_filter_and_normalization_limitations'],
        vision_subset_reported=False,statement='This single setting is not full-paper completion or proof of PDF population equivalence.'))


def _render(report):
    test=report['test'];record=test['all']['overall'];score='—'if record['mae_deg']is None else f"{record['mae_deg']:.6f}"
    import json
    return '\n'.join(['# 原生 EMGFormer 中心测试报告','',
        f"规模：{report['size']}；验证最优检查点：{report['training']['selected_best']['file']}。",'',
        '| 唯一手样本 | 重复消费 | pooled MAE (°) | 有效角度数 |','| ---: | ---: | ---: | ---: |',
        f"| {test['unique_samples']} | {test['duplicate_consumptions']} | {score} | {record['valid_angle_count']} |",'',
        f"原生按样本数加权批量 MAE：{test['native_batch_weighted_mae_rad']:.10g} rad。",'',
        '保留原生中心标签与生成掩码；总体指标按有效角度数汇总，分别列出左右手、条件、匿名用户、22关节和指组。',
        '采用可用旧版数据与独立训练留出验证；未证明与PDF原表人口等价，未取得全论文完成资格。','',
        '```json',json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),'```',''])


def _completion(output,report):
    return _sealed(dict(format='original_emgformer_public_point_completion_v1',report_sha256=report['sha256'],
        files_sha256={name:_file_hash(_safe(output/name))for name in ('metrics.json','metrics.md')},
        scientific_eligible=report['scientific_eligible'],full_paper_complete=False))


def verify_emgformer_report(test_replay,output):
    output=_safe(output);report=build_emgformer_report(test_replay)
    if (_json(output/'metrics.json')!=report or (output/'metrics.md').read_text()!=_render(report)or
        _json(output/'completion.json')!=_completion(output,report)):raise ValueError('public native point report differs from raw recomputation')
    return report


def publish_emgformer_report(test_replay,output):
    report=build_emgformer_report(test_replay);output=_safe(output,exists=False);train.storage._space(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    _write(output/'metrics.json',report)
    with(output/'metrics.md').open('x')as f:f.write(_render(report));f.flush();os.fsync(f.fileno())
    _write(output/'completion.json',_completion(output,report));return verify_emgformer_report(test_replay,output)


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run')
    for name in ('fit','data','upstream','output'):run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--role',choices=('val','test'),required=True);run.add_argument('--kind',choices=('native','replay'),required=True)
    run.add_argument('--reference',type=Path);run.add_argument('--profile',type=Path);run.add_argument('--engineering',action='store_true')
    freeze=sub.add_parser('freeze')
    for name in ('native','replay','output'):freeze.add_argument('--'+name,type=Path,required=True)
    for name in ('report','verify'):
        p=sub.add_parser(name);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());action=args.pop('action')
    if action=='run':
        from .emgformer_literal import compose_emgformer,emgformer_normalization
        from .emgformer_cohort import admit_emgformer_cohort
        identity=_json(args['fit']/'identity.json');cfg=compose_emgformer(args['upstream'],identity['policy']['size']);upstream=args.pop('upstream');data=args.pop('data')
        cohort=admit_emgformer_cohort(data,upstream=upstream,config=cfg,normalization=emgformer_normalization(upstream,cfg),expected=identity['cohort'])
        args['fit_path']=args.pop('fit');args['require_full']=not args.pop('engineering');value=run_emgformer_pass(cohort,**args)
    elif action=='freeze':value=freeze_emgformer_validation(args['native'],args['replay'],args['output'])
    else:value=(publish_emgformer_report if action=='report'else verify_emgformer_report)(args['replay'],args['output'])
    print(json.dumps(dict(action=action,sha256=value['sha256'],scientific_eligible=value['scientific_eligible'],full_paper_complete=False)))


if __name__=='__main__':main()
