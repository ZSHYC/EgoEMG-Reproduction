"""Selected-best FP32 streamed native evaluation on verified ordinary files.

Profiles and passes retain complete original predictions/statistics digests;
they do not declare a scientific experiment or full-paper completion.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import FunctionType

from . import emg2pose_memory_evaluate as _kernel
from . import emg2pose_data as storage, emg2pose_direct_runtime as runtime
from . import emg2pose_direct_train as fit
from .emg2pose_direct_source import VerifiedDirectSource, open_source, FORMAT as SOURCE_FORMAT
from .emg2pose_direct_evaluation_data import build_evaluation_data, evaluation_data_policy
from .emg2pose_memory_statistics import Emg2PoseStatistics, validate_original_output, comparison_policy

FORMAT = 'emg2pose_direct_selected_streamed_pass_v1'
PROFILE = 'emg2pose_direct_validation_streamed_profile_v1'
_ROOT = Path(__file__).parent
_FILES = ('emg2pose_direct_evaluate.py', 'emg2pose_memory_evaluate.py',
    'emg2pose_direct_evaluation_data.py', 'emg2pose_memory_evaluation_data.py', 'emg2pose_memory_statistics.py')
_IMPORTED = {name: storage._sha256(_ROOT/name) for name in _FILES}


def _sources():
    actual = {name: storage._sha256(_ROOT/name) for name in _FILES}
    if actual != _IMPORTED or actual['emg2pose_memory_evaluate.py'] != _kernel.IMPORTED_SHA:
        raise ValueError('imported direct evaluation implementation changed')
    return dict(fit=fit._sources(), evaluation=actual)


def _fit_binding(directory, scope):
    directory = storage._safe_path(directory)
    result = fit.read_fit(directory, require_full=scope == 'full')
    if result['scope'] != scope or result['stage_complete'] is not True:
        raise ValueError('completed matching native direct fit scope required')
    identity = _json(directory/'identity.json'); selected = result['selected_best']
    path = directory/'checkpoints'/selected['file']; checkpoint = fit._load(path)
    return dict(fit_path=str(directory), fit_result_sha256=storage._sha256(directory/'fit-result.json'),
        best_file=selected['file'], best_checkpoint_sha256=storage._sha256(path),
        model_sha256=fit._state_sha(checkpoint['state_dict']), runtime=identity['runtime']), checkpoint


def _source_binding(data):
    if (type(data) is not dict or data.get('format') != SOURCE_FORMAT
            or type(data.get('source_pipeline_completion_sha256')) is not str
            or not _kernel.HASH.fullmatch(data['source_pipeline_completion_sha256'])):
        raise ValueError('verified ordinary source identity required')
    return dict(format='emg2pose_ordinary_evaluation_source_binding_v1',
        source_pipeline_completion_sha256=data['source_pipeline_completion_sha256'], data_identity_sha256=_hash(data))


def _owned(path, passed):
    _check(passed, FORMAT)
    value = _json(Path(path)/'stage-result.json')
    if (value.get('format') != 'egoemg_owned_stage_v1' or value.get('status') != 'success'
            or type(value.get('worker_exit_code')) is not int or value['worker_exit_code'] != 0
            or value.get('all_owned_consumers_exited') is not True or value.get('descriptor_bindings') != {}
            or value.get('scientific_complete') is not False
            or any(value['worker'].get(k) != v for k, v in passed['process'].items())
            or passed.get('source_file_binding', {}).get('format') != 'emg2pose_ordinary_evaluation_source_binding_v1'
            or 'source_kernel_binding' in passed):
        raise ValueError('actual empty-right owned exit/ordinary source binding required')
    for name in ('worker', 'guardian'):
        process = value[name]; path_to_process = Path('/proc', str(process['pid']), 'stat')
        if path_to_process.exists():
            fields = path_to_process.read_text().rsplit(')', 1)[1].split()
            if fields[19] == str(process['start_ticks']) and fields[0] != 'Z':
                raise ValueError('validation consumer or guardian is still live')
    return dict(path=str(storage._safe_path(path)), sha256=storage._sha256(Path(path)/'stage-result.json'))


_namespace = dict(_kernel.__dict__, __name__=__name__, __file__=__file__, FORMAT=FORMAT, PROFILE=PROFILE,
    runtime=runtime, VerifiedSource=VerifiedDirectSource, open_source=open_source,
    build_evaluation_data=build_evaluation_data, evaluation_data_policy=evaluation_data_policy,
    _sources=_sources, _fit_binding=_fit_binding, _owned=_owned)
for _name in ('main', 'run_evaluation_pass'):
    _namespace.pop(_name, None)


def _bind(function):
    bound = FunctionType(function.__code__, _namespace, name=function.__name__,
        argdefs=function.__defaults__, closure=function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = dict(function.__annotations__)
    return bound


for _name, _value in vars(_kernel).items():
    if (isinstance(_value, FunctionType) and _value.__module__ == _kernel.__name__
            and _name not in ('main', 'run_evaluation_pass', '_sources', '_fit_binding', '_owned', 'read_evaluation_pass')):
        _namespace[_name] = globals()[_name] = _bind(_value)
_read_pass_core = _bind(_kernel.read_evaluation_pass)


def read_evaluation_pass(output, *, expected_sha256=None, require_full=True, _seen=()):
    result = _read_pass_core(output, expected_sha256=expected_sha256, require_full=require_full, _seen=_seen)
    identity = _json(storage._safe_path(output)/'identity.json')
    expected = _source_binding(identity['data_identity']['runtime']['data'])
    if result.get('source_file_binding') != expected or 'source_kernel_binding' in result:
        raise ValueError('ordinary evaluation source-file binding differs')
    return result


_namespace['read_evaluation_pass'] = read_evaluation_pass


def run_evaluation_pass(upstream,source,metadata_csv,size,fit_path,output,*,role,kind,scope='full',
        selection='official_literal',reference=None,expected_reference_sha256=None,profile=None,expected_profile_sha256=None):
    policy=evaluation_data_policy(scope,selection)
    if type(role) is not str or role not in ('val','test') or type(kind) is not str or kind not in ('native','replay'):
        raise ValueError('explicit val/test role and native/replay kind required')
    if type(source) is not VerifiedDirectSource:raise ValueError('actual verified source required')
    source.assert_current()
    if scope=='full':
        from .emg2pose_direct_evaluation_data import _full_source
        _full_source(source.data_identity)
    if role=='test' and (profile is None or expected_profile_sha256 is None):raise ValueError('frozen pretest calibration profile required')
    if role=='val' and profile is not None:raise ValueError('validation cannot consume a profile')
    output=storage._safe_path(output)
    if output.exists():raise FileExistsError('evaluation output already exists')
    _separate(output,[fit_path,source.index,reference,profile])
    binding,checkpoint=_fit_binding(fit_path,scope)
    reference_value=None;profile_value=None
    if kind=='replay':
        if reference is None:raise ValueError('frozen native reference required')
        _expected_sha(Path(reference)/'pass.json',expected_reference_sha256,'reference')
        reference_value=read_evaluation_pass(reference,require_full=scope=='full')
        if reference_value['process']==_process():raise ValueError('replay requires a fresh actual process')
        if (reference_value['role']!=role or reference_value['selection']!=selection or reference_value['fit_binding']!=binding):
            raise ValueError('native reference role/mode/selected best differs')
    elif reference is not None:raise ValueError('native pass cannot consume a reference')
    if role=='test':
        profile_value=read_evaluation_profile(profile,expected_profile_sha256,require_full=scope=='full')
        _profile_matches(profile_value,scope,selection,binding)
    from . import emg2pose_direct_train as fit,classic_emg_train as tensors
    import torch
    data=module=dm=None;result_written=False
    try:
        data=build_evaluation_data(upstream,source,metadata_csv,size,role=role,scope=scope,selection=selection)
        if runtime.runtime_identity(upstream,data.config,source=source)!=binding['runtime']:
            raise ValueError('actual source/scientific recipe differs from selected fit')
        if scope=='cpu_engineering' and torch.cuda.is_available():raise ValueError('CPU inference must hide CUDA')
        if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():raise ValueError('original inference requires FP32 without autocast')
        module,dm=runtime.reload_runtime(upstream,data.config,checkpoint['state_dict'],identity=binding['runtime'],source=source)
        device,inference=_environment(scope)
        if profile_value is not None:_profile_matches(profile_value,scope,selection,binding,inference)
        plan=_plan(data);before=tensors.state_digest(module.state_dict())
        if before!=binding['model_sha256']:raise ValueError('restored selected best model state differs')
        identity=_seal(dict(format=FORMAT,document='identity',scope=scope,selection=selection,role=role,policy=policy,
            fit_binding=binding,data_identity=data.identity,inference_runtime=inference,sources_sha256=_sources(),
            fixed_arithmetic_policy=comparison_policy()))
        if reference_value is not None and reference_value['identity_sha256']!=identity['sha256']:
            raise ValueError('fresh reference data/runtime/coverage identity differs')
        budget=sum(p['budget_bytes'] for p in plan)+16*1024**2
        storage._space(output.parent,budget);output.mkdir(mode=0o700)
        _write(output/'identity.json',identity)
        _write(output/'plan.json',_seal(dict(format=FORMAT,identity_sha256=identity['sha256'],batches=plan)))
        count=0;previous=identity['sha256']
        class Observer:
            def to(self,where):module.to(where);return self
            def eval(self):module.eval();return self
            def forward(self,batch):
                nonlocal count,previous
                if count>=len(plan):raise ValueError('extra native batch')
                p=plan[count];ds=data.context._emg2pose_datasets[list(data.population['by_condition']).index(p['condition'])]
                expected=_metadata(ds,p['start'],p['samples'])
                actual={k:(batch[k].detach().cpu().tolist() if isinstance(batch[k],torch.Tensor) else list(batch[k])) for k in expected}
                if actual!=expected or _hash(actual)!=p['metadata_sha256']:raise ValueError('original complete ordered batch metadata differs')
                if tuple(batch['emg'].shape)!=(p['samples'],16,7790) or batch['emg'].dtype!=torch.float32:
                    raise ValueError('original16ch FP32 EMG input differs')
                source.assert_current();storage._space(output)
                if _sources()!=identity['sources_sha256']:raise ValueError('inference source changed')
                if device!='cpu' and torch.cuda.mem_get_info(0)[0]<8*1024**3:raise RuntimeError('observed8GiB GPU margin lost')
                value=module.forward(batch)
                pred,target,mask=[v.detach().cpu().numpy() for v in value]
                validate_original_output((pred,target,mask),original_target=batch['joint_angles'].detach().cpu().numpy(),
                                         original_mask=batch['label_valid_mask'].detach().cpu().numpy())
                statistics=Emg2PoseStatistics();statistics.update(pred,target,mask,condition=p['condition'],users=list(batch['user']))
                row=_seal(dict(format=FORMAT,sequence=count,previous_sha256=previous,plan=p,
                    input_sha256=tensors.state_digest(batch),output_sha256=tensors.state_digest(value),
                    users=_contribution(statistics,p['condition'])))
                raw=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                if len(raw)>p['budget_bytes']:raise ValueError('batch journal exceeds predeclared bound')
                storage._space(output,len(raw))
                if journal.write(raw)!=len(raw):raise OSError('short batch journal write')
                previous=row['sha256'];count+=1
                return value
        data.context.module=Observer()
        from egoemg.test_analysis import EMG2PoseEvaluation
        with (output/'batches.jsonl').open('xb',buffering=0) as journal:
            rows,users=EMG2PoseEvaluation._evaluate_emg2pose_single_pass(data.context,[{} for _ in data.population['by_condition']])
            os.fsync(journal.fileno())
        if count!=len(plan):raise ValueError('partial original native pass')
        statistics,summary=_journal_statistics(output/'batches.jsonl',identity,plan)
        names=list(data.population['by_condition']);native=dict(rows=dict(zip(names,rows)),users=dict(zip(names,users)))
        statistics.check_native(native['rows'],native['users'])
        after=tensors.state_digest(module.state_dict())
        if after!=before:raise ValueError('model changed during original evaluation')
        data.check();source.assert_current()
        if _environment(scope,prelaunch=False)[1]!=inference:raise ValueError('inference runtime/backend changed')
        _write(output/'statistics.json',statistics.report());_write(output/'native-metrics.json',native)
        result=_seal(dict(format=FORMAT,scope=scope,selection=selection,role=role,kind=kind,
            process=_process(),source_file_binding=_source_binding(source.data_identity),identity_sha256=identity['sha256'],fit_binding=binding,
            model_before_sha256=before,model_after_sha256=after,**summary,
            plan_sha256=storage._sha256(output/'plan.json'),journal_sha256=storage._sha256(output/'batches.jsonl'),
            statistics_sha256=storage._sha256(output/'statistics.json'),native_metrics_sha256=storage._sha256(output/'native-metrics.json'),
            reference_path=str(storage._safe_path(reference)) if reference is not None else None,
            reference_sha256=expected_reference_sha256 if reference is not None else None,
            profile_path=str(storage._safe_path(profile)) if profile is not None else None,profile_sha256=expected_profile_sha256,
            stage_complete=True,test_performed=role=='test',scientific_complete=False,full_paper_complete=False))
        runtime.close_data_module(dm);dm=None;data.close();data=None
        _write(output/'pass.json',result);result_written=True
        return read_evaluation_pass(output,require_full=scope=='full')
    except BaseException:
        if result_written:(output/'pass.json').unlink(missing_ok=True)
        raise
    finally:
        if data is not None:data.close()
        if dm is not None:runtime.close_data_module(dm)


def main(argv=None):
    if 'EGOEMG_STAGE_FD_ARCHIVE' in os.environ:
        raise ValueError('direct evaluation does not accept an inherited archive descriptor')
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pass')
    for name in ('upstream','source-root','metadata-csv','fit','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-source-completion-sha256','size','role','kind'):p.add_argument('--'+name,required=True)
    p.add_argument('--scope',default='full');p.add_argument('--selection',default='official_literal')
    for name in ('reference','profile'):p.add_argument('--'+name,type=Path)
    for name in ('expected-reference-sha256','expected-profile-sha256'):p.add_argument('--'+name)
    args=parser.parse_args(argv)
    with open_source(args.source_root,expected_completion_sha256=args.expected_source_completion_sha256,
            upstream=args.upstream,metadata_csv=args.metadata_csv) as source:
        value=run_evaluation_pass(args.upstream,source,args.metadata_csv,args.size,args.fit,args.output,
            role=args.role,kind=args.kind,scope=args.scope,selection=args.selection,reference=args.reference,
            expected_reference_sha256=args.expected_reference_sha256,profile=args.profile,expected_profile_sha256=args.expected_profile_sha256)
        print(json.dumps(dict(role=value['role'],kind=value['kind'],stage_complete=True,scientific_complete=False)))
        return value


if __name__=='__main__':main()
