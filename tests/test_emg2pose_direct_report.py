"""Final scientific admission uses real owned fits and both original populations."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys

import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning');pytest.importorskip('hydra');pytest.importorskip('tensorboard')
from test_emg2pose_direct_evaluate import memory,corpus,source,sha,UPSTREAM
from egoemg_repro.emg2pose_memory_process import run_owned_stage

MODES=('official_literal','metadata_indices')
def api():return importlib.import_module('egoemg_repro.emg2pose_direct_report')

@pytest.fixture(scope='module')
def evidence(memory,tmp_path_factory):
    cached=[]
    def load():
        if cached:return cached[0]
        from egoemg_repro.emg2pose_direct_evaluate import freeze_evaluation_profile
        root=tmp_path_factory.mktemp('report-owned-chain');fit=root/'fit';fit_owned=root/'fit-owned'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        def owned(command,path):
            return run_owned_stage(command,path,descriptors={},guard=lambda:None,environment=env,poll_seconds=.05)
        common=['--upstream',str(UPSTREAM),'--source-root',str(memory.source_root),
            '--expected-source-completion-sha256',memory.receipt['completion_sha256'],'--metadata-csv',str(memory.metadata),
            '--size','small','--scope','cpu_engineering']
        owned([sys.executable,'-m','egoemg_repro.emg2pose_direct_train',*common,'--output',str(fit)],fit_owned)
        evaluations={}
        for mode in MODES:
            directory=root/mode;directory.mkdir();paths={}
            for role in ('val','test'):
                for kind in ('native','replay'):
                    name=role+'-'+kind;paths[name]=root/(mode+'-'+name+'-owned')
                    command=[sys.executable,'-m','egoemg_repro.emg2pose_direct_evaluate','pass',*common,
                        '--fit',str(fit),'--selection',mode,'--role',role,'--kind',kind,'--output',str(directory/name)]
                    if kind=='replay':command+=['--reference',str(directory/(role+'-native')),
                        '--expected-reference-sha256',sha(directory/(role+'-native/pass.json'))]
                    if role=='test':command+=['--profile',str(directory/'profile'),
                        '--expected-profile-sha256',sha(directory/'profile/profile.json')]
                    owned(command,paths[name])
                if role=='val':
                    freeze_evaluation_profile(directory/'val-native',directory/'val-replay',directory/'profile',
                        expected_native_sha256=sha(directory/'val-native/pass.json'),
                        expected_replay_sha256=sha(directory/'val-replay/pass.json'),require_full=False)
            evaluations[mode]=dict(directory=directory,owned=paths)
        value=dict(fit_path=fit,fit_owned=fit_owned,evaluations=evaluations,expected_fit_sha256=sha(fit/'fit-result.json'))
        cached.append(value);return value
    return load

def export(memory,evidence,output,**changes):
    kwargs=dict(evidence(),scope='cpu_engineering');kwargs.update(changes)
    with source(memory) as handle:
        return api().export_report(UPSTREAM,handle,memory.metadata,'small',output=output,**kwargs)

def test_real_owned_both_mode_report_and_independent_reopen(memory,evidence,tmp_path):
    m=api();value=export(memory,evidence,tmp_path/'report')
    assert value['format']=='emg2pose_direct_experiment_report_v1'
    assert value['scientific_complete'] is False and value['full_paper_complete'] is False
    assert value['training']['completed_epochs']==2 and value['training']['optimizer_executions']==32
    assert set(value['evaluations'])==set(MODES)
    for mode,row in value['evaluations'].items():
        assert row['tables']['condition_labels_complete'] is False
        assert row['tables']['table13']['condition_sample_weights']=={'user_stage':1}
        assert row['passes']['test-native']['samples_by_condition']=={'user_stage':1}
        assert row['passes']['test-native']['typed_input_output_digest']==row['passes']['test-replay']['typed_input_output_digest']
    assert m.read_report(tmp_path/'report',require_full=False)==value
    with pytest.raises(ValueError,match='engineering|full'):m.read_report(tmp_path/'report')

@pytest.mark.parametrize('damage',['full','scope','size','fit_sha','missing_mode','wrong_fit_owner','wrong_pass_owner'])
def test_incomplete_or_rebound_evidence_cannot_publish(memory,evidence,tmp_path,damage):
    m=api();args=dict(evidence(),scope='cpu_engineering');size='small'
    if damage=='full':args['scope']='full'
    elif damage=='scope':args['scope']='unknown'
    elif damage=='size':size='large'
    elif damage=='fit_sha':args['expected_fit_sha256']='0'*64
    elif damage=='missing_mode':args['evaluations']={k:v for k,v in args['evaluations'].items() if k==MODES[0]}
    elif damage=='wrong_fit_owner':args['fit_owned']=args['evaluations'][MODES[0]]['owned']['val-native']
    elif damage=='wrong_pass_owner':
        args['evaluations']=copy.deepcopy(args['evaluations'])
        args['evaluations'][MODES[0]]['owned']['test-native']=args['fit_owned']
    output=tmp_path/'absent'
    with source(memory) as handle:
        with pytest.raises((ValueError,FileNotFoundError)):
            m.export_report(UPSTREAM,handle,memory.metadata,size,output=output,**args)
    assert not output.exists()

@pytest.mark.parametrize('damage',['statistics','profile','failed_owner','live_fit_owner'])
def test_real_evidence_corruption_is_refused_and_never_changes_original_evidence(memory,evidence,tmp_path,damage):
    m=api();e=evidence();row=e['evaluations'][MODES[0]]
    if damage=='statistics':path=row['directory']/'test-native/statistics.json'
    elif damage=='profile':path=row['directory']/'profile/profile.json'
    elif damage=='failed_owner':path=row['owned']['test-native']/'stage-result.json'
    else:path=e['fit_owned']/'stage-result.json'
    raw=path.read_bytes();value=json.loads(raw)
    if damage in ('statistics','profile'):value['changed']=True
    elif damage=='failed_owner':value['worker_exit_code']=1
    else:
        from egoemg_repro.emg2pose_memory_process import process_identity
        value['guardian']=process_identity()
    try:
        path.write_text(json.dumps(value))
        with pytest.raises((ValueError,FileNotFoundError)):export(memory,evidence,tmp_path/'absent')
        assert not (tmp_path/'absent').exists()
    finally:path.write_bytes(raw)

def test_public_output_contains_no_private_source_paths_or_users_and_no_overwrite(memory,evidence,tmp_path):
    api();export(memory,evidence,tmp_path/'public')
    for name in ('metrics.json','metrics.md','completion.json'):
        body=(tmp_path/'public'/name).read_text()
        assert str(memory.source_root) not in body and str(UPSTREAM) not in body
        assert 'private-runtime-user' not in body and 'separate-recording' not in body
    with pytest.raises(FileExistsError):export(memory,evidence,tmp_path/'public')

def test_resealed_public_metric_or_private_input_mutation_is_detected(memory,evidence,tmp_path):
    m=api();export(memory,evidence,tmp_path/'report')
    marker=tmp_path/'report/completion.json';original_marker=marker.read_bytes()
    for filename in ('metrics.json','evidence.json'):
        path=tmp_path/'report'/filename;raw=path.read_bytes();v=json.loads(raw);v['scope']='full'
        try:
            path.write_text(json.dumps(v))
            completion=json.loads(original_marker);completion['files_sha256'][filename]=sha(path)
            marker.write_text(json.dumps(completion))
            with pytest.raises(ValueError):m.read_report(tmp_path/'report',require_full=False)
        finally:path.write_bytes(raw);marker.write_bytes(original_marker)

def test_report_checks_live_source_and_never_closes_borrowed_source(memory,evidence,tmp_path):
    m=api()
    with source(memory) as handle:
        m.export_report(UPSTREAM,handle,memory.metadata,'small',output=tmp_path/'report',scope='cpu_engineering',**evidence())
        handle.assert_current()
        handle.close()
        with pytest.raises(ValueError,match='closed'):
            m.export_report(UPSTREAM,handle,memory.metadata,'small',output=tmp_path/'absent',scope='cpu_engineering',**evidence())
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('field,value',[('uid',-1),('uid',True),('uid','0'),('sid',0),('pgrp','1')])
def test_owned_receipt_requires_actual_typed_process_coordinates(tmp_path,field,value):
    m=api();path=tmp_path/'owned'
    run_owned_stage([sys.executable,'-c','pass'],path,descriptors={},guard=lambda:None,poll_seconds=.05)
    packet=json.loads((path/'stage-result.json').read_text());packet['guardian'][field]=value
    (path/'stage-result.json').write_text(json.dumps(packet))
    with pytest.raises(ValueError,match='identity'):m._owned_receipt(path)


def test_markdown_exposes_table13_groups_and_weighting():
    m=api();groups=('thumb','index','middle','ring','pinky','proximal','mid','distal')
    value=dict(size='small',scope='cpu_engineering',scientific_complete=False,training={'completed_epochs':2},limitations=[],
        evaluations={'official_literal':{'tables':{'table11':{},'table13':{
            'native_group_mae_deg':dict.fromkeys(groups,1.),'independent_group_mae_deg':dict.fromkeys(groups,1.)}}}})
    body=m._markdown(value)
    assert 'Table13' in body and 'condition-window weighted' in body
    for group in groups:assert '| '+group+' |' in body
