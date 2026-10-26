"""Publication boundaries exercise real evidence, arithmetic and sanitization."""
import copy
import hashlib
import json
import math
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

from egoemg_repro.report import ReportError, build_status, build_export, write_report


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def record(error, count):
    return {'absolute_error_sum_rad': error, 'valid_angle_count': count,
            'mae_deg': error / count * 180 / math.pi if count else None}


def metrics(mode='center', scale=1, validation=False, trajectory_points=11490):
    # Four paired-hand centers, two users: means 1 and 3 rad; pooled 2 rad.
    factor = trajectory_points if mode == 'trajectory' else 1
    r = lambda error, count: record(error * factor * scale, count * factor)
    overall = r(176, 88)
    by_user = {'/private/participant-Alice': r(44, 44), 'participant-Bob': r(132, 44)}
    by_split = {'user': r(44, 44), 'both': r(132, 44)}
    hand_split = {hand: {'user': r(22, 22), 'both': r(66, 22)} for hand in ('left', 'right')}
    if validation:
        by_split = {'train': overall}
        hand_split = {hand: {'train': r(88, 44)} for hand in ('left', 'right')}
    return {'mode': mode, 'overall': overall, 'invalid_angle_count': 0,
            'by_hand': {hand: r(88,44) for hand in ('left','right')},
            'by_split': by_split, 'by_hand_split': hand_split, 'by_user': by_user,
            'by_component': {'fingers': r(160,80), 'wrist': r(16,8)},
            'user_summary': {'mean_mae_deg': 2*scale*180/math.pi, 'std_mae_deg': scale*180/math.pi,
                             'user_count': 2, 'aggregation': 'population'}}


def evaluation(kind='emg', scale=1, validation=False):
    report = {'bounded': False, 'evaluated_batches': 2,
              'sampling': {'center': metrics('center', scale, validation)}}
    if kind == 'emg':
        report['sampling']['trajectory'] = metrics('trajectory', scale, validation)
    else:
        report['paired_population'] = {'all': copy.deepcopy(report['sampling']['center']),
            'vision_valid': copy.deepcopy(report['sampling']['center']),
            'samples': {'all': 4, 'vision_valid': 4, 'missing_crop': 0},
            'missing_by_hand': {}, 'missing_by_split': {},
            'definition': '/private/raw/cohort/path'}
    return report


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def save_checkpoint(path, value):
    path.write_bytes(pickle.dumps(value))


def read_checkpoint(path):
    return pickle.loads(Path(path).read_bytes())


def complete_run(root, kind='emg', recipe_config=None):
    root.mkdir()
    config = {'model_kind': kind, 'seed': 42, 'epochs': 2, 'batch_size': 2,
              'upstream_experiment': 'emgformer/egoemg_emgformer_small',
              'window_length': 12000, 'lr': 5e-5, 'weight_decay': .01,
              'private_path': '/home/alice/secret', 'participant': 'participant-Bob'}
    if recipe_config is not None:
        config = dict(recipe_config, **config)
        config['upstream_experiment'] = recipe_config['upstream_experiment']
    recipe = json.dumps(dict(config, batch_size=500)) + '\n'
    norm = {'left': {'mean':[0]*8,'std':[1]*8,'count':12000},
            'right': {'mean':[0]*8,'std':[1]*8,'count':12000}}
    windows = []
    for role, start, split in [('train',0,'train'),('val',12000,'train'),
                               ('test',24000,'user'),('test',36000,'both')]:
        windows.append({'episode': 0,'start':start,'end':start+12000,'center':start+6000,
                        'role':role,'split':split})
    # Four val centers simplify arithmetic; all evidence counts remain explicit.
    windows.insert(2, {'episode':0,'start':48000,'end':60000,'center':54000,'role':'val','split':'train'})
    manifest = {'windows': windows, 'episodes':[{'id':'participant-Alice','subject':'participant-Bob'}],
                'coverage': {role:{'windows':sum(w['role']==role for w in windows),
                                  'hand_samples':2*sum(w['role']==role for w in windows)}
                             for role in ('train','val','test')}}
    provenance = {'source': {'revision':'f7cbd048b7092139852270491d3859cd472e0311','sha256':'a'*64},
        'reproduction': {'sha256':'b'*64,'revision':'d'*40,
                         'selected_recipe_sha256':hashlib.sha256(recipe.encode()).hexdigest()},
        'model_kind':kind,'data':{'sha256':'e'*64,'private_path':'/private/data'},
        'config_sha256':canonical(config), 'partition_sha256':canonical(manifest),
        'normalization_sha256':canonical(norm),
        'initialization':{'policy':'random','seed_before_model':42},
        'recipe':{'source_utf8':recipe,'effective_config':config}}
    test = evaluation(kind)
    summary = {'status':'training_complete','test_evaluated':True,'global_step':2,
               'selected_checkpoint_step':1,'best_val_mae_deg':180/math.pi,
               'test':test,'provenance':provenance,'private_path':'/private/alice'}
    best = {'model':{'weight':[1.,2.]},'optimizer':{},'scheduler':{},'rng':{},
            'epoch':1,'next_batch':0,'global_step':1,'best_val_mae_deg':180/math.pi,
            'validation_bounded':False,'provenance':copy.deepcopy(provenance),'run_id':'private-run-id'}
    last = dict(copy.deepcopy(best), model={'weight':[9.,9.]}, epoch=2, global_step=2,
                checkpoint_schema=2,best=copy.deepcopy(best))
    best_file = dict(copy.deepcopy(best),checkpoint_schema=2,best=copy.deepcopy(best))
    progress = [{'event':'start','bounded_integration':False,'config':config},
                {'event':'train_step','step':1,'epoch':0,'loss':1.,'angle_l1_rad':.9,'lr':5e-5},
                {'event':'validation','step':1,'epoch':0,'improved':True,'best_val_mae_deg':180/math.pi,
                 'metrics':evaluation(kind, .5, validation=True)},
                {'event':'train_step','step':2,'epoch':1,'loss':.5,'angle_l1_rad':.4,'lr':3e-5},
                {'event':'validation','step':2,'epoch':1,'improved':False,'best_val_mae_deg':180/math.pi,
                 'metrics':evaluation(kind,.75,validation=True)},
                {'event':'final_test',**summary}]
    for name,value in [('config',config),('partitions',manifest),('normalization',norm),
                       ('provenance',provenance),('summary',summary),('run',{'run_id':'private-run-id'})]:
        save_json(root/(name+'.json'),value)
    (root/'recipe_source.json').write_text(recipe)
    (root/'progress.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in progress))
    save_checkpoint(root/'last.pt',last);save_checkpoint(root/'best.pt',best_file)
    return summary


def test_status_never_reads_test_summary_or_checkpoints_and_ignores_partial_tail(tmp_path):
    run = tmp_path/'run'
    complete_run(run)
    (run/'summary.json').write_text('this file must not be parsed by status')
    (run/'last.pt').write_bytes(b'not checkpoint data')
    with (run/'progress.jsonl').open('a') as stream:
        stream.write('{"event":"train_step","step":')
    report = build_status(run)
    assert report['benchmark_eligible'] is False
    assert report['report_type'] == 'training_status'
    assert report['progress']['global_step'] == 2
    assert report['incomplete_trailing_line_ignored'] is True
    public = json.dumps(report)
    assert 'participant' not in public and '/private/' not in public and '/home/' not in public
    assert 'test' not in report


@pytest.mark.parametrize('change', ['bounded','unfinished','no_test','bounded_test'])
def test_export_rejects_nonfinal_results(tmp_path, change):
    run = tmp_path/'run'; summary = complete_run(run)
    if change=='bounded': summary['status']='bounded_integration_complete'
    if change=='unfinished': summary['status']='running'
    if change=='no_test': summary['test_evaluated']=False
    if change=='bounded_test': summary['test']['bounded']=True
    save_json(run/'summary.json',summary)
    with pytest.raises(ReportError):
        build_export(run,'EMGFormer-S',checkpoint_loader=read_checkpoint)


@pytest.mark.parametrize('artifact', ['config','partitions','normalization','recipe_source','best_score','best_weights'])
def test_export_rejects_mismatched_saved_evidence(tmp_path, artifact):
    run=tmp_path/'run';complete_run(run)
    if artifact=='recipe_source': (run/'recipe_source.json').write_text('{}')
    elif artifact.startswith('best_'):
        state=read_checkpoint(run/'best.pt')
        if artifact=='best_score':state['best_val_mae_deg']=.1
        else:state['model']['weight']=[123.,123.]
        save_checkpoint(run/'best.pt',state)
    else:
        value=json.loads((run/(artifact+'.json')).read_text());value['tampered']=True
        save_json(run/(artifact+'.json'),value)
    with pytest.raises(ReportError):
        build_export(run,'EMGFormer-S',checkpoint_loader=read_checkpoint)


def test_export_recomputes_degrees_anonymizes_users_and_preserves_sampling(tmp_path):
    run=tmp_path/'run';complete_run(run)
    report=build_export(run,'EMGFormer-S',checkpoint_loader=read_checkpoint)
    assert report['benchmark_eligible'] is True
    assert set(report['test']['sampling']) == {'center','trajectory'}
    center=report['test']['sampling']['center']
    assert center['overall']['valid_angle_count']==88
    assert center['overall']['absolute_error_sum_rad']==176
    assert center['overall']['mae_deg']==pytest.approx(2*180/math.pi)
    assert set(center['by_user'])=={'user_001','user_002'}
    assert center['user_summary']['std_mae_deg']==pytest.approx(180/math.pi)
    assert set(center['by_component'])=={'fingers','wrist'}
    public=json.dumps(report)
    assert 'Alice' not in public and 'Bob' not in public and '/private/' not in public and '/home/' not in public


@pytest.mark.parametrize('change', ['degrees','sum','count'])
def test_export_rejects_inconsistent_error_denominators(tmp_path,change):
    run=tmp_path/'run';summary=complete_run(run)
    metric=summary['test']['sampling']['center']
    if change=='degrees':metric['overall']['mae_deg']*=180/math.pi
    if change=='sum':metric['by_hand']['left']['absolute_error_sum_rad']+=1
    if change=='count':metric['by_component']['wrist']['valid_angle_count']+=1
    save_json(run/'summary.json',summary)
    with pytest.raises(ReportError):build_export(run,'EMGFormer-S',checkpoint_loader=read_checkpoint)


def test_visual_export_retains_all_and_matched_populations(tmp_path):
    run=tmp_path/'run';complete_run(run,'vision')
    report=build_export(run,'ResNet-18',checkpoint_loader=read_checkpoint)
    assert set(report['test']['sampling'])=={'center'}
    assert report['test']['paired_population']['samples']=={'all':4,'vision_valid':4,'missing_crop':0}
    assert 'participant' not in json.dumps(report)


def test_status_cli_works_without_site_packages(tmp_path):
    run=tmp_path/'run';complete_run(run)
    output=tmp_path/'public'
    result=subprocess.run([sys.executable,'-S','-m','egoemg_repro.report','status','--run',str(run),
                           '--output',str(output),'--no-plots'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads((output/'status.json').read_text())['benchmark_eligible'] is False
    assert '未完成' in (output/'status.md').read_text()


def test_report_renders_actual_curves_and_clear_status_label(tmp_path):
    pytest.importorskip('matplotlib')
    run=tmp_path/'run';complete_run(run)
    output=tmp_path/'public'
    write_report(build_status(run),output)
    assert (output/'training-curves.png').stat().st_size>1000
    assert 'INCOMPLETE' in (output/'training-curves.svg').read_text()
    assert 'participant' not in (output/'status.md').read_text()


def test_matched_population_can_have_only_one_hand_and_keeps_anonymous_scope(tmp_path):
    run=tmp_path/'run';summary=complete_run(run,'vision')
    matched=copy.deepcopy(summary['test']['paired_population']['vision_valid'])
    def halve(value):
        if isinstance(value,dict):
            if 'valid_angle_count' in value:
                value['valid_angle_count']//=2;value['absolute_error_sum_rad']/=2
            else:
                for child in value.values():halve(child)
    halve(matched)
    matched['by_hand']={'left':copy.deepcopy(matched['overall'])}
    matched['by_hand_split']={'left':copy.deepcopy(matched['by_split'])}
    paired=summary['test']['paired_population']
    paired.update(vision_valid=matched,samples={'all':4,'vision_valid':2,'missing_crop':2},
                  missing_by_hand={'right':2},missing_by_split={'user':1,'both':1})
    save_json(run/'summary.json',summary)
    events=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
    events[-1]['test']=summary['test']
    (run/'progress.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    report=build_export(run,'ResNet-18',checkpoint_loader=read_checkpoint)
    population=report['test']['paired_population']['vision_valid']
    assert set(population['by_hand'])=={'left'}
    assert population['overall']['valid_angle_count']==44
    assert set(population['by_user'])=={'user_001','user_002'}
    assert 'all reported hands and test splits' in population['user_summary']['scope']
    assert 'Alice' not in json.dumps(report)


def test_status_names_only_known_public_models_and_never_raw_experiment_paths(tmp_path):
    run=tmp_path/'run';complete_run(run)
    config=json.loads((run/'config.json').read_text())
    config['upstream_experiment']='emgformer/egoemg_emgformer_small'
    save_json(run/'config.json',config)
    assert build_status(run)['model_label']=='EMGFormer-S'
    config['upstream_experiment']='/private/participant-Alice/model'
    save_json(run/'config.json',config)
    report=build_status(run)
    assert report['model_label']=='EMG model'
    assert 'Alice' not in json.dumps(report)


def test_status_output_refuses_stale_completed_metrics(tmp_path):
    run=tmp_path/'run';complete_run(run)
    output=tmp_path/'public';output.mkdir()
    (output/'metrics.json').write_text('{"benchmark_eligible":true}')
    with pytest.raises(ReportError,match='separate'):
        write_report(build_status(run),output,plots=False)
    assert not (output/'status.json').exists()


def test_export_rejects_selected_checkpoint_epoch_disagreement(tmp_path):
    run=tmp_path/'run';complete_run(run)
    state=read_checkpoint(run/'best.pt')
    state['epoch']=state['best']['epoch']=0
    save_checkpoint(run/'best.pt',state)
    with pytest.raises(ReportError):
        build_export(run,'EMGFormer-S',checkpoint_loader=read_checkpoint)


def test_real_torch_checkpoint_reader_and_cli_export_are_sanitized(tmp_path):
    torch=pytest.importorskip('torch')
    run=tmp_path/'run';complete_run(run)
    for name in ('last.pt','best.pt'):
        state=read_checkpoint(run/name)
        torch.save(state,run/name)
    output=tmp_path/'public'
    result=subprocess.run([sys.executable,'-m','egoemg_repro.report','export','--run',str(run),
        '--output',str(output),'--model-label','EMGFormer-S','--no-plots'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((output/'metrics.json').read_text())
    assert report['benchmark_eligible'] is True
    assert len(report['identities']['selected_checkpoint_file_sha256'])==64
    for path in output.iterdir():
        assert 'participant-Alice' not in path.read_text()
        assert '/home/alice' not in path.read_text()


def test_export_rejects_partial_full_validation_evidence(tmp_path):
    run=tmp_path/'run';complete_run(run)
    events=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
    events[2]['metrics']['evaluated_batches']=1
    (run/'progress.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    with pytest.raises(ReportError,match='validation'):
        build_export(run,'EMGFormer-S',checkpoint_loader=read_checkpoint)


def test_empty_crop_valid_population_has_null_metrics_and_explicit_missing_counts(tmp_path):
    run=tmp_path/'run';summary=complete_run(run,'vision')
    empty={'mode':'center','overall':record(0,0),'invalid_angle_count':0,
           'by_hand':{},'by_split':{},'by_hand_split':{},'by_component':{},'by_user':{},
           'user_summary':{'user_count':0,'mean_mae_deg':None,'std_mae_deg':None}}
    summary['test']['paired_population'].update(vision_valid=empty,
        samples={'all':4,'vision_valid':0,'missing_crop':4},
        missing_by_hand={'left':2,'right':2},missing_by_split={'user':2,'both':2})
    save_json(run/'summary.json',summary)
    events=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
    events[-1]['test']=summary['test']
    (run/'progress.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    report=build_export(run,'ResNet-18',checkpoint_loader=read_checkpoint)
    assert report['test']['paired_population']['vision_valid']['overall']['mae_deg'] is None
    assert report['test']['paired_population']['samples']['missing_crop']==4


def test_complete_corrupt_progress_line_is_not_silently_ignored(tmp_path):
    run=tmp_path/'run';complete_run(run)
    with (run/'progress.jsonl').open('a') as stream:stream.write('{broken}\n')
    with pytest.raises(ReportError,match='complete progress'):
        build_status(run)


@pytest.mark.parametrize('stage', ['test', 'validation'])
@pytest.mark.parametrize('points', [1, 11489, 11491, 12001])
def test_export_rejects_trajectory_coverage_different_from_pinned_context(tmp_path, stage, points):
    run = tmp_path/'run'; summary = complete_run(run)
    events = [json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
    if stage == 'test':
        summary['test']['sampling']['trajectory'] = metrics('trajectory', trajectory_points=points)
        save_json(run/'summary.json', summary)
        events[-1]['test'] = summary['test']
    else:
        events[2]['metrics']['sampling']['trajectory'] = metrics('trajectory', .5, True, points)
    (run/'progress.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    with pytest.raises(ReportError, match='trajectory'):
        build_export(run, 'EMGFormer-S', checkpoint_loader=read_checkpoint)


@pytest.mark.parametrize('name', ['emgformer_s', 'emgformer_m'])
def test_actual_supported_s_and_m_recipes_require_full_11490_point_trajectory(tmp_path, name):
    config = json.loads(Path('configs', name+'.json').read_text())
    run = tmp_path/'run'; complete_run(run, recipe_config=config)
    report = build_export(run, name, checkpoint_loader=read_checkpoint)
    trajectory = report['test']['sampling']['trajectory']
    assert trajectory['time_points_per_window'] == 11490
    assert trajectory['overall']['valid_angle_count'] == 1011120


@pytest.mark.parametrize('experiment', ['emgformer/unknown', None])
def test_export_rejects_unknown_emg_context_contract(tmp_path, experiment):
    run = tmp_path/'run'
    complete_run(run, recipe_config={'upstream_experiment': experiment})
    with pytest.raises(ReportError, match='unsupported|context'):
        build_export(run, 'Unknown', checkpoint_loader=read_checkpoint)


def test_large_official_experiment_exports_complete_aligned_trajectory(tmp_path):
    run = tmp_path/'run'
    complete_run(run, recipe_config={
        'upstream_experiment': 'emgformer/egoemg_emgformer_large'})
    report = build_export(run, 'EMGFormer-L', checkpoint_loader=read_checkpoint)
    trajectory = report['test']['sampling']['trajectory']
    assert trajectory['time_points_per_window'] == 11490
    assert trajectory['overall']['valid_angle_count'] == 1011120
    assert build_status(run)['model_label'] == 'EMGFormer-L'


def test_large_export_rejects_full_input_instead_of_aligned_trajectory(tmp_path):
    run = tmp_path/'run'
    summary = complete_run(run, recipe_config={
        'upstream_experiment': 'emgformer/egoemg_emgformer_large'})
    summary['test']['sampling']['trajectory'] = metrics('trajectory', trajectory_points=12000)
    save_json(run/'summary.json', summary)
    events = [json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
    events[-1]['test'] = summary['test']
    (run/'progress.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    with pytest.raises(ReportError, match='trajectory observations'):
        build_export(run, 'EMGFormer-L', checkpoint_loader=read_checkpoint)


@pytest.mark.parametrize('stage', ['test', 'validation'])
def test_full_trajectory_accepts_masked_angles_without_shortening_coverage(tmp_path, stage):
    run = tmp_path/'run'; summary = complete_run(run)
    events = [json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
    # One invalid point for all 22 angles in all four hand windows; remaining
    # points still supply coherent per-group sums, counts and user statistics.
    masked = metrics('trajectory', 1 if stage == 'test' else .5,
                     stage == 'validation', trajectory_points=11489)
    masked['invalid_angle_count'] = 88
    if stage == 'test':
        summary['test']['sampling']['trajectory'] = masked
        save_json(run/'summary.json', summary)
        events[-1]['test'] = summary['test']
    else:
        events[2]['metrics']['sampling']['trajectory'] = masked
    (run/'progress.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    report = build_export(run, 'EMGFormer-S', checkpoint_loader=read_checkpoint)
    assert report['benchmark_eligible'] is True
    if stage == 'test':
        assert report['test']['sampling']['trajectory']['invalid_angle_count'] == 88
        assert report['test']['sampling']['trajectory']['time_points_per_window'] == 11490
