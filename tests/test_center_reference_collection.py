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
