"""Selected original ResNet evaluation, frozen validation and exact replay.

Numeric point aggregation and point capture reuse the closed original vision
reader helpers. Model construction, fit identity and all result formats bind
the actual ResNet family. A prediction pass never grants scientific eligibility.
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
from .classic_emg_cohort import _safe,_StatGuard,_source_paths
from .classic_emg_report import PUBLIC_NATIVE_METRICS
from .resnet_literal import compose_resnet,VARIANTS
from .resnet_data import ResnetCohort,admit_resnet_cohort
from .resnet_runtime import build_resnet_runtime
from .resnet_train import (read_resnet_fit,_trusted_checkpoint,_json,_write,_sealed,_check_seal,
    _space,_runtime_device,storage,vision_state_sha256,_validate_identity,_loop_paths)
from .vision_literal_eval import (POLICY as _POINT_POLICY,_process,_population_public,
    vision_population,vision_statistics,_close_native,_payload_sha,_capture)


ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_resnet_selected_prediction_v1'
PROFILE='original_resnet_pretest_validation_profile_v1'
REPORT='original_resnet_center_completed_report_v1'
MODEL_CLASS='egoemg.models.modules.resnet_vision.ResNetVisionPose'
POLICY=copy.deepcopy(_POINT_POLICY)


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported ResNet evaluation source changed')
    names=('egoemg_repro/resnet_eval.py','egoemg_repro/resnet_train.py',
        'egoemg_repro/resnet_data.py','egoemg_repro/resnet_runtime.py','egoemg_repro/resnet_literal.py',
        'egoemg_repro/vision_literal_eval.py','egoemg_repro/metrics.py','egoemg_repro/classic_emg_report.py')
    return {n:_file_hash(ROOT/n)for n in names}


def _pose_sha(parts):
    import torch
    return storage.state_digest({n:torch.cat([p[n]for p in parts])for n in ('pred','target','mask')})


def _saved_validation(role,parts,statistics,checkpoint):
    if role!='val':return None
    record=checkpoint['resnet_fit']['records'][-1];reference=record['validation']
    if reference['samples']!=statistics['consumed_samples']:
        raise ValueError('complete saved validation population must match the evaluation role')
    pose=_pose_sha(parts)
    return dict(completed_epoch=record['epoch']+1,fit_pose_sha256=reference['prediction_sha256'],
        evaluation_pose_sha256=pose,pose_exact_match=pose==reference['prediction_sha256'],
        fit_native_mae_rad=reference['val_mae'],
        native_mae_difference_rad=statistics['native_batch_weighted_mae_rad']-reference['val_mae'],
        view_policy='fit validation and standalone native evaluation are recorded separately; selected checkpoint is unchanged; independent standalone replay remains exact')


def _load_pass(directory):
    import torch
    directory=_safe(directory);value=_json(directory/'pass.json');_check_seal(value,FORMAT)
    if value['sources_sha256']!=_sources()or value['fixed_agreement_policy']!=POLICY:
        raise ValueError('ResNet point evaluation source/fixed policy differs')
    file=directory/'predictions.pt'
    if _file_hash(file)!=value['payload_sha256']:raise ValueError('actual ResNet prediction payload bytes differ')
    payload=torch.load(str(file),map_location='cpu',weights_only=True)
    if type(payload)is not dict or set(payload)!={'parts','population'}:
        raise ValueError('complete original point/population payload required')
    parts=payload['parts'];population=payload['population']
    public=_population_public(population['users_by_role'],population['test_users_by_split'],population['public']['episode_records'])
    if public!=population['public']or public!=value['population']or _hash(population)!=value['population_sha256']:
        raise ValueError('actual ResNet private population/public summary differs')
    if _payload_sha(parts)!=value['prediction_sha256']:raise ValueError('typed ResNet input/prediction digest differs')
    statistics=vision_statistics(parts,samples=value['samples'],batch_size=value['batch_size'],role=value['role'])
    if statistics!=value['statistics']:raise ValueError('raw original point statistics differ from ResNet report')
    if value['kind']=='native':
        metrics=_json(directory/'native-metrics.json')
        if metrics!=value['native_metrics']:raise ValueError('actual native ResNet metrics differ')
        _close_native(statistics,metrics,value['role'])
    elif value['kind']!='replay'or value['native_metrics']!={}:
        raise ValueError('fresh ResNet replay cannot invent native metrics')
    if (value['model_before_sha256']!=value['model_after_sha256']or value['scientific_eligible']is not False
        or value['full_paper_complete']is not False):
        raise ValueError('prediction-only ResNet pass mutated state or claimed completion')
    return value,parts


class _Audit:
    """One fresh verification transaction; memoization never outlives its guard."""
    def __init__(self):
        self.guard=_StatGuard([]);self.cache={};self.active=set();self.identities={}
        self.bind([ROOT/n for n in _sources()])

    def bind(self,paths):
        self.guard.check()
        for path in paths:
            try:self.guard.add(path)
            except OSError as exc:raise ValueError('required raw audit input is missing')from exc

    def check(self):
        self.guard.check()
        for identity in self.identities.values():_validate_identity(identity)
        self.guard.check()

    def _get(self,kind,path,read):
        path=_safe(path);self.guard.check();key=(kind,str(path))
        if key in self.active:raise ValueError('cyclic ResNet artifact reference')
        if key in self.cache:return copy.deepcopy(self.cache[key])
        self.active.add(key)
        try:value=read(path)
        finally:self.active.remove(key)
        self.guard.check();self.cache[key]=copy.deepcopy(value);return value

    def fit(self,path):
        def read(root):
            self.bind([root/'identity.json',root/'fit-result.json',root/'epochs.jsonl',root/'checkpoints/last.ckpt'])
            identity=_json(root/'identity.json');result=_json(root/'fit-result.json')
            self.bind([root/'checkpoints'/result['selected_best']['file']])
            evidence=result['resume_evidence']
            if evidence is not None:self.bind([root/evidence['parent_file'],root/evidence['observation_file']])
            if identity['cohort_receipt']is not None:self.bind([identity['cohort_receipt']['path']])
            self.bind([*_source_paths(_safe(identity['upstream'])),*_loop_paths().values()])
            value=read_resnet_fit(root,require_full=result['scope']=='full')
            self.identities[str(root)]=identity;return value
        return self._get('fit',path,read)

    def pass_(self,path):
        def read(root):
            self.bind([root/'pass.json',root/'predictions.pt'])
            if _json(root/'pass.json')['kind']=='native':self.bind([root/'native-metrics.json'])
            return _read_resnet_pass(root,self)
        return self._get('pass',path,read)

    def profile(self,path):
        def read(root):
            self.bind([root/'profile.json']);return _read_resnet_profile(root,self)
        return self._get('profile',path,read)

    def report(self,path):
        value=_build_resnet_report(path,self);self.check();return value


def _read_resnet_pass(directory,audit):
    value,parts=_load_pass(directory)
    fit_path=_safe(value['fit_path']);fit=audit.fit(fit_path)
    identity=_json(fit_path/'identity.json');policy=identity['policy']
    if (not fit['stage_complete']or value['fit_result_sha256']!=fit['sha256']
        or value['fit_result_file_sha256']!=_file_hash(fit_path/'fit-result.json')
        or value['selected_best']!=fit['selected_best']or value['cohort_sha256']!=identity['cohort']['sha256']
        or value['samples']!=identity['cohort']['counts'][value['role']]
        or identity['val_samples']!=identity['cohort']['counts']['val']
        or value['batch_size']!=policy['batch_size_by_role'][value['role']]
        or value['training_scope']!=fit['scope']or value['training_full_terminated']is not fit['full_training_terminated']
        or value['model_class']!=MODEL_CLASS or value['variant']!=policy['variant']):
        raise ValueError('actual ResNet fit, complete role, model or selected state binding differs')
    path=fit_path/'checkpoints'/fit['selected_best']['file'];guard=_StatGuard([path])
    if _file_hash(path)!=fit['selected_best']['checkpoint_sha256']:raise ValueError('raw selected ResNet checkpoint bytes differ')
    checkpoint=_trusted_checkpoint(path);state_sha=vision_state_sha256(checkpoint['state_dict']);guard.check()
    if value['model_before_sha256']!=state_sha or value['model_after_sha256']!=state_sha:
        raise ValueError('ResNet pass did not retain complete actual selected state')
    if value['saved_fit_validation']!=_saved_validation(value['role'],parts,value['statistics'],checkpoint):
        raise ValueError('actual saved/native ResNet validation comparison differs')
    if value['role']=='test':
        profile=audit.profile(value['profile_path'])
        if (value['profile_sha256']!=profile['sha256']or value['fit_result_sha256']!=profile['fit_result_sha256']
            or value['started_ns']<=profile['finished_ns']):
            raise ValueError('ResNet test lacks an earlier frozen original validation profile')
    elif value['profile_path']is not None or value['profile_sha256']is not None:
        raise ValueError('ResNet validation cannot consume a test profile')
    if value['kind']=='replay':
        ref=audit.pass_(value['reference_path'])
        if (ref['kind']!='native'or ref['role']!=value['role']or ref['fit_result_sha256']!=value['fit_result_sha256']
            or ref['sha256']!=value['reference_sha256']or ref['prediction_sha256']!=value['prediction_sha256']
            or ref['population_sha256']!=value['population_sha256']or ref['model_before_sha256']!=state_sha
            or ref['process']==value['process']or value['started_ns']<=ref['finished_ns']):
            raise ValueError('fresh-process exact native ResNet replay differs')
    elif value['reference_path']is not None or value['reference_sha256']is not None:
        raise ValueError('native ResNet pass cannot claim a replay reference')
    return value


def _read_resnet_profile(directory,audit):
    directory=_safe(directory);value=_json(directory/'profile.json');_check_seal(value,PROFILE)
    native=audit.pass_(value['reference_path']);replay=audit.pass_(value['replay_path'])
    if (value['fixed_agreement_policy']!=POLICY or native['role']!='val'or native['kind']!='native'
        or replay['role']!='val'or replay['kind']!='replay'or replay['reference_sha256']!=native['sha256']
        or value['reference_sha256']!=native['sha256']or value['replay_sha256']!=replay['sha256']
        or value['fit_result_sha256']!=native['fit_result_sha256']
        or value['saved_fit_validation']!=native['saved_fit_validation']
        or value['finished_ns']<=max(native['finished_ns'],replay['finished_ns'])):
        raise ValueError('frozen independent ResNet validation profile differs')
    return value


def read_resnet_pass(directory):
    audit=_Audit();value=audit.pass_(directory);audit.check();return value


def read_resnet_profile(directory):
    audit=_Audit();value=audit.profile(directory);audit.check();return value


def freeze_resnet_validation(reference,replay,output):
    audit=_Audit();native=audit.pass_(reference);other=audit.pass_(replay);audit.check()
    if (native['role']!='val'or native['kind']!='native'or other['role']!='val'or other['kind']!='replay'
        or other['reference_sha256']!=native['sha256']):
        raise ValueError('complete native ResNet validation and fresh exact replay required')
    output=_safe(output,exists=False);_space(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    value=_sealed(dict(format=PROFILE,reference_path=str(_safe(reference)),replay_path=str(_safe(replay)),
        reference_sha256=native['sha256'],replay_sha256=other['sha256'],fit_result_sha256=native['fit_result_sha256'],
        saved_fit_validation=copy.deepcopy(native['saved_fit_validation']),fixed_agreement_policy=copy.deepcopy(POLICY),
        finished_ns=time.time_ns(),scientific_eligible=False,full_paper_complete=False))
    _write(output/'profile.json',value);return read_resnet_profile(output)


def _loader(cohort,policy,role):
    from egoemg.datamodule import make_data_module
    dm=make_data_module(cohort.config)
    dm.batch_size=policy['batch_size_by_role'][role]
    workers=policy['workers_by_role'][role]
    loader=dm._make_loader(cohort.datasets[role],shuffle=False,num_workers=workers)
    if (loader.batch_size!=policy['batch_size_by_role'][role]or loader.num_workers!=workers
        or loader.generator is not None or loader.drop_last or loader.pin_memory
        or(workers and(loader.prefetch_factor!=2 or not loader.persistent_workers))):
        raise ValueError('original full-role ResNet evaluation loader differs')
    return loader


def run_resnet_pass(cohort,fit_path,output,*,role,kind,reference=None,profile=None,require_full=True):
    if role not in ('val','test')or kind not in ('native','replay'):
        raise ValueError('explicit original val/test and native/replay pass required')
    if type(require_full)is not bool or type(cohort)is not ResnetCohort:
        raise ValueError('actual admitted ResNet cohort and boolean full requirement required')
    import torch
    cohort.check(require_full=require_full);fit_path=_safe(fit_path);fit=read_resnet_fit(fit_path,require_full=require_full)
    identity=_json(fit_path/'identity.json')
    if (not fit['stage_complete']or identity['cohort']!=cohort.identity
        or identity['val_samples']!=cohort.identity['counts']['val']):
        raise ValueError('completed original ResNet fit and complete saved-validation cohort required')
    frozen=ref=None
    if role=='test':
        if profile is None:raise ValueError('freeze independent native validation before ResNet test')
        frozen=read_resnet_profile(profile)
        if frozen['fit_result_sha256']!=fit['sha256']:raise ValueError('ResNet pretest profile belongs to another fit')
    elif profile is not None:raise ValueError('validation cannot consume a ResNet test profile')
    if kind=='replay':
        if reference is None:raise ValueError('native ResNet pass required before replay')
        ref=read_resnet_pass(reference)
        if(ref['kind']!='native'or ref['role']!=role or ref['fit_result_sha256']!=fit['sha256']or ref['process']==_process()):
            raise ValueError('fresh process and matching native ResNet reference required')
    elif reference is not None:raise ValueError('native ResNet pass has no reference')
    policy=identity['policy'];device=_runtime_device(policy);output=_safe(output,exists=False);_space(output.parent)
    selected=fit_path/'checkpoints'/fit['selected_best']['file'];checkpoint=_trusted_checkpoint(selected)
    prepared=build_resnet_runtime(cohort.upstream,cohort.config,state=checkpoint['state_dict'],expected_initialization=identity['construction'])
    population=vision_population(cohort);before=vision_state_sha256(prepared.module.state_dict());parts=[];started=time.time_ns()
    guard=_StatGuard([fit_path/'identity.json',fit_path/'fit-result.json',selected,*[ROOT/n for n in _sources()]])
    loader=_loader(cohort,policy,role);original=prepared.module.forward
    def observed(batch):
        cohort.guard.check();guard.check();out=original(batch);_capture(batch,out,cohort.datasets[role],parts);return out
    prepared.module.forward=observed;native={};output.mkdir(mode=0o700,exist_ok=False)
    try:
        if kind=='native':
            trainer=storage.pl.Trainer(accelerator=device['accelerator'],devices=1,precision=policy['precision'],
                logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
            values=(trainer.validate if role=='val'else trainer.test)(prepared.module,dataloaders=loader,verbose=False)
            native={k:float(v)for k,v in values[0].items()}
            if not all(math.isfinite(v)for v in native.values()):raise ValueError('nonfinite original ResNet native metrics')
        else:
            from contextlib import nullcontext
            from lightning_fabric.utilities.apply_func import move_data_to_device
            target='cpu'if device['accelerator']=='cpu'else'cuda:0';prepared.module.to(target).eval()
            with torch.inference_mode():
                for batch in loader:
                    batch=move_data_to_device(batch,target)
                    with(nullcontext()if target=='cpu'else torch.autocast('cuda',dtype=torch.bfloat16)):
                        prepared.module.forward(batch)
        statistics=vision_statistics(parts,samples=len(cohort.datasets[role]),batch_size=policy['batch_size_by_role'][role],role=role)
        if kind=='native':_close_native(statistics,native,role)
        if ref is not None and _payload_sha(parts)!=ref['prediction_sha256']:
            raise ValueError('exact typed native ResNet prediction/input replay failed')
        cohort.check(require_full=require_full);guard.check();prepared.check();after=vision_state_sha256(prepared.module.state_dict())
        if before!=after:raise ValueError('selected original ResNet changed during inference')
        temp=output/('.predictions-'+uuid.uuid4().hex)
        try:
            with temp.open('xb')as f:torch.save(dict(parts=parts,population=population),f);f.flush();os.fsync(f.fileno())
            os.link(temp,output/'predictions.pt')
        finally:
            if temp.exists():temp.unlink()
        if kind=='native':_write(output/'native-metrics.json',native)
        value=_sealed(dict(format=FORMAT,role=role,kind=kind,fit_path=str(fit_path),fit_result_sha256=fit['sha256'],
            fit_result_file_sha256=_file_hash(fit_path/'fit-result.json'),training_scope=fit['scope'],
            training_full_terminated=fit['full_training_terminated'],cohort_sha256=cohort.identity['sha256'],
            variant=policy['variant'],model_class=MODEL_CLASS,selected_best=copy.deepcopy(fit['selected_best']),
            sources_sha256=_sources(),fixed_agreement_policy=copy.deepcopy(POLICY),samples=len(cohort.datasets[role]),
            batch_size=policy['batch_size_by_role'][role],statistics=statistics,population=population['public'],
            population_sha256=_hash(population),payload_sha256=_file_hash(output/'predictions.pt'),prediction_sha256=_payload_sha(parts),
            saved_fit_validation=_saved_validation(role,parts,statistics,checkpoint),native_metrics=native,
            native_aggregation_agrees=kind=='native',model_before_sha256=before,model_after_sha256=after,
            process=_process(),started_ns=started,finished_ns=time.time_ns(),
            profile_path=str(_safe(profile))if profile is not None else None,profile_sha256=frozen['sha256']if frozen is not None else None,
            reference_path=str(_safe(reference))if reference is not None else None,reference_sha256=ref['sha256']if ref is not None else None,
            scientific_eligible=False,full_paper_complete=False))
        _write(output/'pass.json',value);return read_resnet_pass(output)
    finally:prepared.module.forward=original


def _build_resnet_report(recompute,audit):
    replay=audit.pass_(recompute)
    if replay['role']!='test'or replay['kind']!='replay':raise ValueError('completed original ResNet test/replay chain required')
    native=audit.pass_(replay['reference_path']);profile=audit.profile(replay['profile_path'])
    fit=audit.fit(replay['fit_path'])
    identity=_json(Path(replay['fit_path'])/'identity.json');statistics=copy.deepcopy(replay['statistics'])
    names={u:'user_'+str(i).zfill(3)for i,u in enumerate(sorted(statistics['consumed']['all']['by_user']))}
    def sanitize(value):
        if isinstance(value,dict):return{k:({names[u]:sanitize(v)for u,v in item.items()}if k=='by_user'else sanitize(item))for k,item in value.items()}
        if isinstance(value,list):return[sanitize(v)for v in value]
        return value
    return _sealed(dict(format=REPORT,variant=identity['policy']['variant'],model_class=MODEL_CLASS,
        scientific_eligible=fit['full_training_terminated'],full_paper_complete=False,paper_numerical_equivalence_established=False,
        scope='complete admitted role points; original generated masks/RGB point readers; explicit native ResNet recipe or backbone/paper-batch derivative; single rank',
        model_parameter_count=identity['construction']['parameter_count'],initialization=identity['construction']['initialization'],
        training=dict(scope=fit['scope'],completed_epochs=fit['completed_epochs'],optimizer_executions=fit['optimizer_executions'],
            full_training_terminated=fit['full_training_terminated'],selected_best=fit['selected_best'],
            fit_result_sha256=fit['sha256'],execution_policy=copy.deepcopy(identity['policy'])),
        test=sanitize(statistics),population=copy.deepcopy(replay['population']),
        native_metrics={k:v for k,v in native['native_metrics'].items()if k in PUBLIC_NATIVE_METRICS},
        saved_fit_validation=copy.deepcopy(profile['saved_fit_validation']),
        fixed_agreement_policy=copy.deepcopy(POLICY),cohort_sha256=replay['cohort_sha256'],
        validation_profile_sha256=replay['profile_sha256'],native_test_sha256=native['sha256'],
        independent_test_sha256=replay['sha256'],sources_sha256=_sources(),prediction_sha256=replay['prediction_sha256']))


def build_resnet_report(recompute):return _Audit().report(recompute)


def render_resnet_report(report):
    t=report['training'];s=report['test']
    lines=['# ResNet 原生中心评价','',
        '完整训练实验'if report['scientific_eligible']else'工程验证；没有完成正式完整训练','',
        f"模型：{report['variant']}；训练 {t['completed_epochs']} 轮、实际优化更新 {t['optimizer_executions']} 次。",'',
        f"原生每2轮验证；验证选优：{t['selected_best']['file']}。",'',
        f"测试消费 {s['consumed_samples']}，唯一手样本 {s['unique_samples']}，重复消费 {s['duplicate_consumptions']}。",'',
        '| 统计口径 | MAE (°) | 有效角度数 |','| --- | ---: | ---: |']
    for label,v in [('全部消费',s['consumed']['all']['overall']),('唯一首遍',s['unique_first_repeat']['all']['overall']),
                    ('有图子集',s['consumed']['vision_valid']['overall'])]:
        error='—'if v['mae_deg']is None else f"{v['mae_deg']:.6f}"
        lines.append(f"| {label} | {error} | {v['valid_angle_count']} |")
    lines+=['','原生按样本数加权的批量MAE与按有效角度汇总的pooled MAE分别保留。缺图仍使用原生成标签mask。',
        '训练时验证与独立评价的比较在测试前冻结；原始验证最优检查点不重新选择，独立评价重放要求逐字节一致。',
        '旧版队列、独立train留出验证和单rank适配均明确记录；未证明与论文原表数值等价。整篇论文尚未完成。','',
        '```json',json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),'```','']
    return'\n'.join(lines)


def _completion(output,report):
    return _sealed(dict(format='original_resnet_public_report_completion_v1',report_sha256=report['sha256'],
        files_sha256={n:_file_hash(_safe(output/n))for n in ('metrics.json','metrics.md')},
        scientific_eligible=report['scientific_eligible'],full_paper_complete=False))


def verify_resnet_report(recompute,output):
    output=_safe(output);report=build_resnet_report(recompute)
    if(_json(output/'metrics.json')!=report or(output/'metrics.md').read_text()!=render_resnet_report(report)
       or _json(output/'completion.json')!=_completion(output,report)):
        raise ValueError('public ResNet report differs from reopened original raw chain')
    return report


def publish_resnet_report(recompute,output):
    report=build_resnet_report(recompute);output=_safe(output,exists=False);_space(output.parent)
    output.mkdir(mode=0o700,exist_ok=False);_write(output/'metrics.json',report)
    with(output/'metrics.md').open('x')as f:f.write(render_resnet_report(report));f.flush();os.fsync(f.fileno())
    if report!=build_resnet_report(recompute):raise ValueError('raw ResNet chain changed during report preparation')
    _write(output/'completion.json',_completion(output,report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    run=sub.add_parser('pass')
    for name in ('data','crops','upstream','fit','output'):run.add_argument('--'+name,required=True,type=Path)
    run.add_argument('--role',required=True,choices=('val','test'));run.add_argument('--kind',required=True,choices=('native','replay'))
    run.add_argument('--variant',required=True,choices=VARIANTS);run.add_argument('--engineering',action='store_true')
    run.add_argument('--reference',type=Path);run.add_argument('--profile',type=Path)
    freeze=sub.add_parser('freeze')
    for n in ('reference','replay','output'):freeze.add_argument('--'+n,required=True,type=Path)
    for action in ('report','verify'):
        q=sub.add_parser(action);q.add_argument('--replay',required=True,type=Path);q.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.action=='pass':
        identity=_json(a.fit/'identity.json')
        cohort=admit_resnet_cohort(a.data,a.crops,upstream=a.upstream,config=compose_resnet(a.upstream,a.variant),
            expected=identity['cohort'],require_full=not a.engineering)
        value=run_resnet_pass(cohort,a.fit,a.output,role=a.role,kind=a.kind,reference=a.reference,profile=a.profile,require_full=not a.engineering)
    elif a.action=='freeze':value=freeze_resnet_validation(a.reference,a.replay,a.output)
    elif a.action=='report':value=publish_resnet_report(a.replay,a.output)
    else:value=verify_resnet_report(a.replay,a.output)
    print(json.dumps(dict(format=value['format'],sha256=value['sha256'],scientific_eligible=value['scientific_eligible'])))


if __name__=='__main__':main()
