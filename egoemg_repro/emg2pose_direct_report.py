"""Final ordinary-file EMG2Pose experiment evidence and anonymous tables.

Full admission requires the complete original source, 250 real native epochs,
both declared evaluation populations, selected-best fresh-process replay and
actual exited process owners. Engineering reports remain explicitly incomplete.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import emg2pose_data as storage,emg2pose_direct_runtime as runtime
from . import emg2pose_direct_train as fit,emg2pose_direct_evaluate as evaluation
from .emg2pose_direct_source import VerifiedDirectSource,open_source
from .emg2pose_direct_evaluation_data import build_evaluation_data,_full_source
from .emg2pose_memory_process import process_identity,same_process
from .emg2pose_memory_report import summarize_tables
from .emg2pose_check import _atomic_write

FORMAT='emg2pose_direct_experiment_report_v1'
MODES=('official_literal','metadata_indices')
PASSES=('val-native','val-replay','test-native','test-replay')
_ROOT=Path(__file__).parent
_FILES=('emg2pose_direct_report.py','emg2pose_memory_report.py')
_IMPORTED={n:storage._sha256(_ROOT/n) for n in _FILES}


def _sources():
    if {n:storage._sha256(_ROOT/n) for n in _FILES}!=_IMPORTED:
        raise ValueError('imported final report implementation changed')
    return dict(report=_IMPORTED,evaluation=evaluation._sources())


def _owned_receipt(path):
    path=storage._safe_path(path);v=storage._json(path/'stage-result.json')
    if (v.get('format')!='egoemg_owned_stage_v1' or v.get('status')!='success'
            or type(v.get('worker_exit_code')) is not int or v['worker_exit_code']!=0
            or v.get('all_owned_consumers_exited') is not True or v.get('descriptor_bindings')!={}
            or v.get('scientific_complete') is not False):
        raise ValueError('actual successful empty-right owned exit required')
    for name in ('worker','guardian'):
        p=v.get(name)
        if (not isinstance(p,dict) or type(p.get('pid')) is not int or p['pid']<=0
                or not isinstance(p.get('start_ticks'),str) or not p['start_ticks'].isdigit()
                or type(p.get('uid')) is not int or p['uid']<0
                or any(type(p.get(k)) is not int or p[k]<=0 for k in ('sid','pgrp'))):
            raise ValueError('actual owned process identity required')
        current=process_identity(p['pid'])
        if same_process(p) and current['state']!='Z':raise ValueError('owned consumer or guardian is still live')
    return v


def _fit_owned(path,directory,result):
    owner=_owned_receipt(path);prefix=result['resume_checkpoint']
    if prefix is None:
        attempts=list(directory.glob('attempt-*.json'))
        if len(attempts)!=1:raise ValueError('one original fit invocation required without a resume prefix')
        attempt=attempts[0]
    else:
        name=Path(prefix['file']).name
        if not name.startswith('resume-') or not name.endswith('.ckpt'):raise ValueError('native resume attempt differs')
        attempt=directory/('attempt-'+name[len('resume-'):-len('.ckpt')]+'.json')
    record=storage._json(attempt)
    if (record['scope']!=result['scope'] or not isinstance(record['process'],dict)
            or any(owner['worker'].get(k)!=v for k,v in record['process'].items())):
        raise ValueError('actual terminal fit invocation and owned worker differ')
    return dict(receipt_sha256=storage._sha256(Path(path)/'stage-result.json'),
        attempt_sha256=storage._sha256(attempt))


def _public_population(population):
    keys=('requested_rows','selected_rows','extra_rows','omitted_rows','selected_by_role',
        'selected_by_condition','requested_selection_exact','windows','window_plan_sha256')
    return dict(by_condition={c:{k:r[k] for k in keys} for c,r in population['by_condition'].items()},
        pairwise_row_overlap=population['pairwise_row_overlap'])


def _derive(upstream,source,metadata_csv,size,fit_path,fit_owned,evaluations,*,scope,expected_fit_sha256):
    if scope not in ('full','cpu_engineering','cuda_engineering'):raise ValueError('explicit full or engineering report scope required')
    if type(source) is not VerifiedDirectSource:raise ValueError('actual verified ordinary source required')
    source.assert_current();data=source.data_identity
    if scope=='full':_full_source(data)
    if size not in runtime.native.SIZES:raise ValueError('known native model size required')
    if type(evaluations) is not dict or set(evaluations)!=set(MODES):raise ValueError('both declared evaluation modes required')
    fit_path=storage._safe_path(fit_path)
    evaluation._expected_sha(fit_path/'fit-result.json',expected_fit_sha256,'fit')
    trained=fit.read_fit(fit_path,require_full=scope=='full')
    if trained['scope']!=scope or not trained['stage_complete']:raise ValueError('matching complete native fit scope required')
    identity=storage._json(fit_path/'identity.json')
    config=runtime.compose_config(upstream,size,source.index,source=source,metadata_csv=metadata_csv)
    if identity['runtime']!=runtime.runtime_identity(upstream,config,source=source):
        raise ValueError('actual source, model size or original runtime differs from fit')
    owned_fit=_fit_owned(fit_owned,fit_path,trained);modes={}
    for mode in MODES:
        entry=evaluations[mode]
        if type(entry) is not dict or set(entry)!={'directory','owned'} or set(entry['owned'])!=set(PASSES):
            raise ValueError('all four actual owned val/test native/replay phases required')
        directory=storage._safe_path(entry['directory']);passes={};population={}
        profile=evaluation.read_evaluation_profile(directory/'profile',storage._sha256(directory/'profile/profile.json'),require_full=scope=='full')
        for role in ('val','test'):
            with build_evaluation_data(upstream,source,metadata_csv,size,role=role,scope=scope,selection=mode) as actual:
                expected_data=actual.identity
            for kind in ('native','replay'):
                name=role+'-'+kind;path=directory/name
                value=evaluation.read_evaluation_pass(path,require_full=scope=='full')
                observed=storage._json(path/'identity.json')
                if (value['scope']!=scope or value['selection']!=mode or value['role']!=role or value['kind']!=kind
                        or value['fit_binding']['fit_path']!=str(fit_path)
                        or value['fit_binding']['fit_result_sha256']!=expected_fit_sha256
                        or observed['data_identity']!=expected_data
                        or value['source_file_binding']!=evaluation._source_binding(data)):
                    raise ValueError('actual original population/source/selected fit or pass role differs')
                _owned_receipt(entry['owned'][name]);owned=evaluation._owned(entry['owned'][name],value)
                if role=='test' and (value['profile_path']!=str(directory/'profile') or
                        value['profile_sha256']!=storage._sha256(directory/'profile/profile.json')):
                    raise ValueError('test did not bind this frozen pretest profile')
                passes[name]={k:value[k] for k in ('typed_input_output_digest','statistics_sha256','native_metrics_sha256',
                    'samples_by_condition','batch_count','model_before_sha256','model_after_sha256','source_file_binding')}
                passes[name].update(pass_sha256=storage._sha256(path/'pass.json'),owned_receipt_sha256=owned['sha256'])
                population[role]=_public_population(expected_data['population'])
            native=evaluation.read_evaluation_pass(directory/(role+'-native'),require_full=scope=='full')
            replay=evaluation.read_evaluation_pass(directory/(role+'-replay'),require_full=scope=='full')
            evaluation._pair(native,replay,role)
            if role=='val' and (profile['native_path']!=str(directory/'val-native') or profile['replay_path']!=str(directory/'val-replay')):
                raise ValueError('profile validation populations differ from declared phases')
        stats=storage._json(directory/'test-native/statistics.json');native=storage._json(directory/'test-native/native-metrics.json')
        tables=summarize_tables(stats,native['rows'],native['users'],require_all_conditions=scope=='full')
        modes[mode]=dict(passes=passes,population=population,tables=tables,
            profile_sha256=storage._sha256(directory/'profile/profile.json'),
            population_basis='unchanged official recording-session selection' if mode=='official_literal' else 'explicit original metadata-file-row isolation control')
    source.assert_current()
    return dict(format=FORMAT,official_commit=runtime.native.UPSTREAM_COMMIT,scope=scope,size=size,
        source=dict(pipeline_completion_sha256=source.completion_sha256,archive_sha256=data['archive_sha256'],
            archive_bytes=data['archive_bytes'],cohort=data['cohort'],files=data['file_count'],rows=data['rows'],
            role_file_counts=data['counts']['split'],full_corpus_complete=data['full_corpus_complete']),
        training=dict(completed_epochs=trained['completed_epochs'],optimizer_executions=trained['optimizer_executions'],
            selected_validation_score_rad=trained['selected_best']['score'],
            selected_checkpoint_sha256=trained['selected_best']['checkpoint_sha256'],
            fit_receipt_sha256=expected_fit_sha256,epoch_journal_sha256=storage._sha256(fit_path/'epochs.jsonl'),
            last_checkpoint_sha256=trained['last_checkpoint_sha256'],parameters=runtime.native.SIZES[size],
            actual_train_samples_per_epoch=trained['train_samples_per_epoch'],actual_val_samples_per_epoch=trained['val_samples_per_epoch'],
            train_unique_windows=identity['train_unique_windows'],policy=identity['policy'],owned=owned_fit),
        evaluations=modes,producer_files_sha256=_sources(),scientific_complete=scope=='full',full_paper_complete=False,
        limitations=['Official literal and metadata-file-row control populations are reported separately; overlapping condition consumptions are not independent samples.',
            'Table11 is unweighted users with population SD; Table13 is condition-window weighted. Valid-angle pooled MAE is a separate view.',
            'A complete EMG2Pose model experiment does not complete the entire EgoEMG paper.']+
            ([] if scope=='full' else ['Engineering source/training/evaluation; not benchmark accuracy or a250epoch scientific experiment.']))


def _markdown(value):
    lines=['# EMG2Pose '+value['size']+' — '+value['scope'],'',
        'Scientific experiment complete: '+str(value['scientific_complete'])+'. Full paper complete: False.','',
        'Native training epochs: '+str(value['training']['completed_epochs'])+'.','',
        '| Population | Condition | Native user mean MAE (°) | Native user SD (°) | Users |','| --- | --- | ---: | ---: | ---: |']
    for mode,row in value['evaluations'].items():
        for condition,v in row['tables']['table11'].items():
            lines.append(f"| {mode} | {condition} | {v['native_mean_mae_deg']:.6f} | {v['native_std_mae_deg']:.6f} | {v['native_user_count']} |")
    lines+=['','Table13 — condition-window weighted; separate native FP32 and independent FP64 arithmetic.','',
        '| Population | Group | Native MAE (°) | Independent MAE (°) |','| --- | --- | ---: | ---: |']
    for mode,row in value['evaluations'].items():
        table=row['tables']['table13']
        for group,mae in table['native_group_mae_deg'].items():
            lines.append(f"| {mode} | {group} | {mae:.6f} | {table['independent_group_mae_deg'][group]:.6f} |")
    lines+=['',*('- '+text for text in value['limitations']),'']
    return '\n'.join(lines)


def export_report(upstream,source,metadata_csv,size,fit_path,fit_owned,evaluations,output,*,scope='full',expected_fit_sha256):
    output=storage._safe_path(output)
    if output.exists():raise FileExistsError('final report output already exists')
    evidence=dict(format=FORMAT,upstream=str(storage._safe_path(upstream)),source_root=str(source.source_root),
        metadata_csv=str(storage._safe_path(metadata_csv)),size=size,fit_path=str(storage._safe_path(fit_path)),
        fit_owned=str(storage._safe_path(fit_owned)),evaluations={m:dict(directory=str(storage._safe_path(v['directory'])),
            owned={n:str(storage._safe_path(p)) for n,p in v['owned'].items()}) for m,v in evaluations.items()},
        scope=scope,expected_fit_sha256=expected_fit_sha256,source_completion_sha256=source.completion_sha256)
    evaluation._separate(output,[source.source_root,fit_path,fit_owned,*[v['directory'] for v in evaluations.values()]])
    value=_derive(upstream,source,metadata_csv,size,fit_path,fit_owned,evaluations,scope=scope,expected_fit_sha256=expected_fit_sha256)
    storage._space(output.parent,8*1024**2);output.mkdir(mode=0o700)
    try:
        evaluation._write(output/'evidence.json',evidence)
        evaluation._write(output/'metrics.json',value)
        raw=_markdown(value).encode()
        _atomic_write(output/'metrics.md',lambda h:h.write(raw),len(raw)+4096)
        source.assert_current();_sources()
        evaluation._write(output/'completion.json',dict(format=FORMAT,scope=scope,scientific_complete=value['scientific_complete'],
            full_paper_complete=False,files_sha256={n:storage._sha256(output/n) for n in ('evidence.json','metrics.json','metrics.md')}))
    except BaseException:
        (output/'completion.json').unlink(missing_ok=True);raise
    return value


def read_report(output,*,require_full=True):
    if type(require_full) is not bool:raise ValueError('explicit full admission policy required')
    output=storage._safe_path(output);completion=storage._json(output/'completion.json');evidence=storage._json(output/'evidence.json')
    if (completion.get('format')!=FORMAT or evidence.get('format')!=FORMAT or completion.get('full_paper_complete') is not False
            or set(completion.get('files_sha256',{}))!={'evidence.json','metrics.json','metrics.md'}
            or any(storage._sha256(output/n)!=h for n,h in completion['files_sha256'].items())):
        raise ValueError('final report completion/actual evidence bytes differ')
    if require_full and evidence['scope']!='full':raise ValueError('engineering report is not a full scientific experiment')
    with open_source(evidence['source_root'],expected_completion_sha256=evidence['source_completion_sha256'],
            upstream=evidence['upstream'],metadata_csv=evidence['metadata_csv']) as source:
        value=_derive(evidence['upstream'],source,evidence['metadata_csv'],evidence['size'],evidence['fit_path'],evidence['fit_owned'],
            evidence['evaluations'],scope=evidence['scope'],expected_fit_sha256=evidence['expected_fit_sha256'])
    if (storage._json(output/'metrics.json')!=value or (output/'metrics.md').read_text()!=_markdown(value)
            or completion['scope']!=value['scope'] or completion['scientific_complete'] is not value['scientific_complete']):
        raise ValueError('reopened scientific evidence disagrees with public report')
    return value
