"""New model-free same-center graph and explicit anonymous table projection."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]
NAMES=[size+'-'+kind for size in('small','middle','large')for kind in('native','replay')]


def api():
    try:return importlib.import_module('scripts.paper.collect_native_center_reference')
    except ModuleNotFoundError as exc:
        if exc.name=='scripts.paper.collect_native_center_reference':pytest.fail('closed same-center table collector missing')
        raise


def public_old_statistics():
    # Formatting is tested with already accepted statistics. It cannot certify
    # a new complete4282 run; that requires the separately closed source graph.
    value=json.loads((ROOT/'results/original-emgformer/small/metrics.json').read_text())['test']
    keys=sorted(value['all']['by_user']);old=value['all']['by_user']
    value['all']['by_user']={('private_subject_A'if i==0 else'private_subject_B'if i==1 else k):old[k]
        for i,k in enumerate(keys)}
    return value


def test_public_projection_anonymizes_all_users_and_preserves_actual_sufficient_statistics():
    raw=public_old_statistics();before=copy.deepcopy(raw);v=api().public_reference_statistics(raw)
    assert raw==before
    assert 'private_subject'not in json.dumps(v)
    assert set(v['by_user'])=={'user_'+str(i).zfill(3)for i in range(40)}
    assert v['overall']==raw['all']['overall']and v['by_split_user_summary']==raw['all']['by_split_user_summary']
    assert sum(x['valid_angle_count']for x in v['by_user'].values())==59576


@pytest.mark.parametrize('damage',['extra_private','user_extra','nan_sd','wrong_sd_rule'])
def test_unknown_private_schema_or_invalid_summary_is_rejected(damage):
    raw=public_old_statistics()
    if damage=='extra_private':raw['all']['sample_metadata']={'subject':'private_subject_A'}
    elif damage=='user_extra':next(iter(raw['all']['by_user'].values()))['subject']='private_subject_A'
    elif damage=='nan_sd':raw['all']['by_split_user_summary']['gesture']['std_mae_deg']=float('nan')
    else:raw['all']['by_split_user_summary']['gesture']['aggregation']='ddof=1'
    with pytest.raises(ValueError):api().public_reference_statistics(raw)


def workflow(root):
    root.mkdir();(root/'private').mkdir()
    config=dict(scope='full',source_commit='a'*40,device='cuda:0',gpu_uuid='GPU-00000000-0000-0000-0000-000000000001')
    (root/'config.json').write_text(json.dumps(config));(root/'controller.py').write_text('# source fixture, no model\n')
    pins={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in('config.json','controller.py')}
    (root/'stage-receipt.json').write_text(json.dumps(dict(source_commit='a'*40,files_sha256=pins)))
    actor=dict(pid=2147483647,start_ticks='0')
    stages=[dict(name=name,worker=dict(pid=2147483600+i,start_ticks='0'),status='complete',actual_exit_code=0)
        for i,name in enumerate(NAMES)]
    state=dict(status='complete',actual_exit_code=0,wrapper=actor,scope='full',stages=stages)
    (root/'private/execution-state.json').write_text(json.dumps(state))
    registration=dict(root=str(root),source_commit='a'*40,actual_controller=actor,frozen_members_sha256=pins,scope='full')
    return state,registration


def test_source_graph_checks_all_six_roles_and_seven_closed_actors(tmp_path):
    root=tmp_path/'workflow';state,registration=workflow(root)
    v=api().read_closed_reference_workflow(root,registration,scope='full',stage_names=NAMES)
    assert len(v['actors'])==7 and v['state']==state


@pytest.mark.parametrize('damage',['alive','partial','duplicate_actor','reordered','scope','source_bytes'])
def test_live_partial_or_changed_source_cannot_become_a_completed_model(damage,tmp_path):
    root=tmp_path/'workflow';state,registration=workflow(root)
    if damage=='alive':
        fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        live=dict(pid=os.getpid(),start_ticks=fields[19]);state['wrapper']=live;registration['actual_controller']=live
    elif damage=='partial':state['stages'].pop()
    elif damage=='duplicate_actor':state['stages'][1]['worker']=state['stages'][0]['worker']
    elif damage=='reordered':state['stages'].reverse()
    elif damage=='scope':state['scope']='engineering'
    else:(root/'controller.py').write_text('# changed source\n')
    (root/'private/execution-state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError):api().read_closed_reference_workflow(root,registration,scope='full',stage_names=NAMES)


def test_qualifications_cannot_be_promoted_by_relabeling_scope():
    # Actual completed qualified four-request summaries are not the4282 job.
    record=dict(kind='native',scope='engineering',samples=4,source_indices=[0,2140,2141,4281])
    with pytest.raises(ValueError):api().require_full_reference_pair(record,dict(record,kind='replay'))


@pytest.mark.parametrize('damage',['old_point','old_trajectory','different_target_packet','missing_invalid'])
def test_full_counts_alone_do_not_accept_another_target_protocol(damage):
    common=dict(scope='full',samples=4282,source_indices=list(range(4282)),size='small',
        fit_result_sha256='a'*64,input_admission_sha256='b'*64,grid_sha256='c'*64,
        model_before_sha256='d'*64,model_after_sha256='d'*64,prediction_sha256='e'*64,
        statistics=dict(all=dict(overall=dict(valid_angle_count=91388),invalid_angle_count=2816)))
    native=dict(common,kind='native',sha256='f'*64)
    replay=copy.deepcopy(dict(common,kind='replay',reference_sha256='f'*64))
    api().require_full_reference_pair(native,replay)
    if damage=='old_point':native['statistics']['all']['overall']['valid_angle_count']=59576
    elif damage=='old_trajectory':native['statistics']['all']['overall']['valid_angle_count']=684769448
    elif damage=='different_target_packet':replay['prediction_sha256']='9'*64
    else:native['samples']=4154
    with pytest.raises(ValueError):api().require_full_reference_pair(native,replay)


def qualification_workflow(root,monkeypatch):
    """Gate fixture uses actual accepted CPU summaries; raw reader is a unit boundary.

    It cannot certify inference: independent real packet checks are separately
    exercised by the evaluator and the closed server qualification.
    """
    state,registration=workflow(root)
    config=json.loads((root/'config.json').read_text());config['scope']='engineering'
    (root/'config.json').write_text(json.dumps(config))
    pins={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in('config.json','controller.py')}
    (root/'stage-receipt.json').write_text(json.dumps(dict(source_commit='a'*40,files_sha256=pins)))
    registration.update(scope='engineering',frozen_members_sha256=pins)
    source=json.loads((ROOT/'results/native-center-reference-CPU-qualification-20261102.json').read_text())
    models=copy.deepcopy(source['models']);passes={};stages=[];targets={}
    for size in ('small','middle','large'):
        v=models[size];v['source_commit']='a'*40
        for kind in ('native','replay'):
            p=root/'private'/size/kind;p.mkdir(parents=True)
            a=copy.deepcopy(v[kind]);a['sources_sha256']={'closed_call_boundary':'1'*64}
            (p/'pass.json').write_text(json.dumps(a));(p/'predictions.pt').write_bytes(b'gate-only reader boundary')
            passes[str(p)]=a
            stages.append(dict(name=size+'-'+kind,worker={k:a['process'][k]for k in('pid','start_ticks')},status='complete',actual_exit_code=0))
        stages.append(dict(name=size+'-verify',worker=v['reader'],status='complete',actual_exit_code=0))
        (root/'private'/size/'qualification.json').write_text(json.dumps(v))
        targets[size]={'native':passes[str(root/'private'/size/'native')]}
    state.update(scope='engineering',stages=stages)
    (root/'private/execution-state.json').write_text(json.dumps(state))
    complete=dict(format='actual_native_center_reference_three_selected_models_CPU_qualification_v1',source_commit='a'*40,
        all_three_CPU_model_qualifications_complete=True,scope='engineering',total_actual_new_neural_forwards=24,
        total_optimizer_updates=0,models=models)
    (root/'private/completed.json').write_text(json.dumps(complete))
    artifacts=['private/completed.json']
    for size in ('small','middle','large'):
        artifacts.append('private/'+size+'/qualification.json')
        for kind in ('native','replay'):
            artifacts.extend('private/'+size+'/'+kind+'/'+n for n in ('pass.json','predictions.pt'))
    registration['accepted_artifact_files_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in artifacts}
    calls=[]
    def validated_reader(path):calls.append(str(path));return copy.deepcopy(passes[str(path)])
    monkeypatch.setattr(importlib.import_module('egoemg_repro.center_reference_eval'),'read_reference_pass',validated_reader)
    return state,registration,targets,calls


def test_qualification_reads_all_six_bound_passes_before_accepting_closed_summaries(tmp_path,monkeypatch):
    root=tmp_path/'qualified';state,registration,targets,calls=qualification_workflow(root,monkeypatch)
    result=api()._qualification(root,registration,'CPU',targets)
    assert len(calls)==6 and set(calls)=={str(root/'private'/s/k)for s in('small','middle','large')for k in('native','replay')}
    assert result['actual_neural_forwards']==24


@pytest.mark.parametrize('damage',['missing_raw','changed_raw','producer','reader','completion_digest','summary_only'])
def test_qualification_cannot_adopt_missing_replaced_or_unrelated_producer_artifacts(damage,tmp_path,monkeypatch):
    root=tmp_path/'qualified';state,registration,targets,calls=qualification_workflow(root,monkeypatch)
    if damage=='missing_raw':(root/'private/small/native/predictions.pt').unlink()
    elif damage=='changed_raw':(root/'private/small/native/predictions.pt').write_bytes(b'replaced packet')
    elif damage in('producer','reader'):
        stage=state['stages'][0 if damage=='producer'else 2];stage['worker']=dict(pid=2147483555,start_ticks='0')
        (root/'private/execution-state.json').write_text(json.dumps(state))
    elif damage=='completion_digest':
        p=root/'private/completed.json';v=json.loads(p.read_text());v['unrelated_observation']='replaced summary';p.write_text(json.dumps(v))
    else:registration.pop('accepted_artifact_files_sha256')
    with pytest.raises((ValueError,FileNotFoundError)):api()._qualification(root,registration,'CPU',targets)
