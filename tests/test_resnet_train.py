"""Native ResNet validation cadence and complete epoch recovery."""
import importlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_vision_literal_data import UPSTREAM


def api():return importlib.import_module('egoemg_repro.resnet_train')


@pytest.mark.parametrize('helper',['vision_literal_data.py','vision_literal_runtime.py','visual.py'])
def test_fit_identity_rechecks_recorded_cohort_and_construction_helper_hashes(request,monkeypatch,helper):
    m=api();fit=request.getfixturevalue('native_fit');identity=m._json(fit['output']/'identity.json')
    original=m._file_hash;queried=[]
    def changed(path):
        if Path(path).name==helper:
            queried.append(str(path));return '0'*64
        return original(path)
    monkeypatch.setattr(m,'_file_hash',changed)
    with pytest.raises(ValueError):m._validate_identity(identity)
    assert queried,'recorded helper source must be read during checkpoint identity verification'


def test_standalone_checkpoint_reader_rejects_helper_generation_change_during_raw_read(request,monkeypatch):
    import egoemg_repro.classic_emg_cohort as guards
    m=api();fit=request.getfixturevalue('native_fit');path=Path(m.__file__).parent/'vision_literal_data.py'
    original_read=m._trusted_checkpoint;original_stamp=guards._stamp;reading=[False]
    def read(candidate):
        value=original_read(candidate);reading[0]=True;return value
    def stamp(candidate):
        value=original_stamp(candidate)
        return (*value[:-1],value[-1]+1)if reading[0]and Path(candidate)==path else value
    monkeypatch.setattr(m,'_trusted_checkpoint',read);monkeypatch.setattr(guards,'_stamp',stamp)
    with pytest.raises(ValueError,match='stamp'):m.read_resnet_fit(fit['output'],require_full=False)


@pytest.mark.parametrize('variant,batch',[('resnet18',900),('resnet50',400),('resnet152',180)])
@pytest.mark.parametrize('scope',['full','cpu_engineering','cuda_engineering'])
def test_policy_keeps_every_two_epoch_validation_and_original_per_variant_loaders(variant,batch,scope):
    m=api()
    from egoemg_repro.resnet_literal import compose_resnet
    p=m.resnet_fit_policy(UPSTREAM,compose_resnet(UPSTREAM,variant),scope=scope)
    cpu=scope=='cpu_engineering';full=scope=='full'
    assert p['max_epochs']==(150 if full else 4) and p['check_val_every_n_epoch']==2
    assert p['precision']==('32-true'if cpu else'bf16-mixed')
    assert p['batch_size_by_role']=={r:2 if cpu else batch for r in ('train','val','test')}
    assert p['workers_by_role']=={r:0 if cpu else 8 for r in ('train','val','test')}
    assert p['monitor']=='val_mae' and p['monitor_mode']=='min' and p['patience']==30
    assert p['patience_counts']=='actual validation checks only'
    assert p['save_top_k']==3 and p['save_last'] is True
    assert p['scheduler_T_max']==150 and p['scheduler_eta_min']==1e-5
    assert p['gradient_clip_val']==1 and p['single_rank'] is True
    assert p['num_sanity_val_steps']==(2 if full else 0)
    assert p['scientific_eligible'] is False and p['full_paper_complete'] is False


@pytest.mark.parametrize('scope',[None,True,'bounded',''])
def test_unknown_scope_rejected_before_cohort_or_runtime(scope):
    with pytest.raises(ValueError):api().resnet_fit_policy(None,None,scope=scope)


def records(n=4,train=4,val=4,improve=True):
    result=[]
    for i in range(n):
        validation=None
        if i%2:
            score=.5-(.1*(i//2)if improve else 0.)
            validation=dict(samples=val,val_mae=score,prediction_sha256='2'*64)
        seen=(i+1)//2
        best=None if not seen else .5-(.1*(seen-1)if improve else 0.)
        wait=0 if improve or not seen else seen-1
        result.append(dict(epoch=i,optimizer_executions=2*(i+1),global_step=2*(i+1),
            train_samples=train,train_batches=2,train_loss=1.,validation=validation,
            model_sha256='1'*64,early=dict(wait_count=wait,stopped_epoch=i if wait==30 and i%2 else 0,best_score=best)))
    return result


def policy(scope='cpu_engineering'):
    from egoemg_repro.resnet_literal import compose_resnet
    return api().resnet_fit_policy(UPSTREAM,compose_resnet(UPSTREAM),scope=scope)


def test_complete_history_counts_only_actual_even_epoch_validations():
    v=api()._validate_records(policy(),records(),train_samples=4,val_samples=4)
    assert v==dict(completed_epochs=4,optimizer_executions=8,validation_checks=2,minimum_val_mae=.4)


@pytest.mark.parametrize('damage',['odd_validation','missing_even_validation','duplicate_epoch','missing_epoch',
    'partial_train','partial_val','optimizer_steps','global_step','batches','loss_nan','bool_epoch',
    'stale_patience','stale_best','model_digest'])
def test_partial_or_relabelled_history_cannot_be_a_complete_native_fit(damage):
    rows=records()
    if damage=='odd_validation':rows[2]['validation']=copy.deepcopy(rows[1]['validation'])
    elif damage=='missing_even_validation':rows[1]['validation']=None
    elif damage=='duplicate_epoch':rows[2]['epoch']=1
    elif damage=='missing_epoch':rows.pop(1)
    elif damage=='partial_train':rows[1]['train_samples']=3
    elif damage=='partial_val':rows[1]['validation']['samples']=3
    elif damage=='optimizer_steps':rows[2]['optimizer_executions']=5
    elif damage=='global_step':rows[2]['global_step']=5
    elif damage=='batches':rows[0]['train_batches']=1
    elif damage=='loss_nan':rows[0]['train_loss']=float('nan')
    elif damage=='bool_epoch':rows[0]['epoch']=False
    elif damage=='stale_patience':rows[2]['early']['wait_count']=1
    elif damage=='stale_best':rows[3]['early']['best_score']=.5
    else:rows[2]['model_sha256']='not-a-sha'
    with pytest.raises(ValueError):api()._validate_records(policy(),rows,train_samples=4,val_samples=4)


def test_native_patience_means_thirty_validation_checks_not_thirty_epochs():
    p=policy('full');r=records(62,train=1800,val=900,improve=False)
    assert api().resnet_stage_complete(p,r,r[-1]['early'],train_samples=1800,val_samples=900) is True
    for n in (30,31,60):
        early=r[n-1]['early']
        assert api().resnet_stage_complete(p,r[:n],early,train_samples=1800,val_samples=900) is False


@pytest.fixture(scope='session')
def native_fit(tmp_path_factory):
    tmp_path=tmp_path_factory.mktemp('genuine-resnet-fit')
    m=api()
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    if not list((Path(torch.hub.get_dir())/'checkpoints').glob('resnet18-*.pth')):
        if os.environ.get('EGOEMG_REQUIRE_RESNET_PRETRAINED_TEST')=='1':pytest.fail('genuine RN18 cache required')
        pytest.skip('actual ImageNet RN18 asset required; no synthetic model positive')
    from test_vision_literal_data import sample_store,crop_store
    import numpy as np
    from egoemg_repro.resnet_data import admit_resnet_cohort
    from egoemg_repro.resnet_literal import compose_resnet
    data=tmp_path/'data';crops=tmp_path/'crops';sample_store(data,n=9*7790);crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*7790,7790,7790,7790]).astype('int8').tofile(data/'frame_split_id.dat')
    c=admit_resnet_cohort(data,crops,upstream=UPSTREAM,config=compose_resnet(UPSTREAM),require_full=False)
    output=tmp_path/'fit';prefix=m.fit_resnet(c,output,scope='cpu_engineering',stop_after_epoch=2)
    assert prefix['completed_epochs']==2 and prefix['optimizer_executions']==4 and prefix['stage_complete'] is False
    run=subprocess.run([sys.executable,'-B','-m','egoemg_repro.resnet_train','fit',
        '--data',str(data),'--crops',str(crops),'--upstream',str(UPSTREAM),'--output',str(output),
        '--variant','resnet18','--scope','cpu_engineering','--resume'],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=180)
    assert run.returncode==0,run.stderr[-4000:]
    yield dict(root=tmp_path,output=output,prefix=prefix)
    from test_vision_literal_data import release_fixture_readers
    release_fixture_readers(tmp_path)


def test_real_native_cpu_prefix_two_then_fresh_resume_four_preserves_validation_cadence(native_fit):
    m=api();output=native_fit['output']
    result=m.read_resnet_fit(output,require_full=False)
    assert result['completed_epochs']==4 and result['optimizer_executions']==8 and result['stage_complete'] is True
    assert result['resume_start_exact']=={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')}
    assert result['train_samples_per_epoch']==[4,4,4,4]
    assert result['val_samples_per_epoch']==[None,4,None,4]
    assert result['selected_best']['completed_epochs'] in (2,4)
    assert result['selected_best']['validated'] is True and result['selected_best']['file']!='last.ckpt'
    assert result['scientific_eligible'] is False and result['full_training_terminated'] is False
    assert result['bitwise_training_trajectory_guaranteed'] is False
    with pytest.raises(ValueError):m.read_resnet_fit(output,require_full=True)


@pytest.mark.parametrize('damage',['epoch_count','optimizer_count','odd_validation_coverage','train_coverage',
    'best_last_fallback','best_epoch','best_score','full_claim','scientific_claim','resume_rng','last_digest','scope'])
def test_resealed_result_metadata_does_not_override_real_checkpoint_evidence(native_fit,monkeypatch,damage):
    from egoemg_repro.classic_emg import _hash
    m=api();original=m._json
    def read(path):
        v=original(path)
        if Path(path).name!='fit-result.json':return v
        v=copy.deepcopy(v)
        if damage=='epoch_count':v['completed_epochs']=3
        elif damage=='optimizer_count':v['optimizer_executions']=7
        elif damage=='odd_validation_coverage':v['val_samples_per_epoch']=[4,4,4,4]
        elif damage=='train_coverage':v['train_samples_per_epoch']=[3,4,4,4]
        elif damage=='best_last_fallback':v['selected_best']['file']='last.ckpt'
        elif damage=='best_epoch':v['selected_best']['completed_epochs']=3
        elif damage=='best_score':v['selected_best']['score']=100.
        elif damage=='full_claim':v['full_training_terminated']=True
        elif damage=='scientific_claim':v['scientific_eligible']=True
        elif damage=='resume_rng':v['resume_start_exact']['rng']=False
        elif damage=='last_digest':v['last_checkpoint_sha256']='0'*64
        else:v['scope']='full'
        v['sha256']=_hash({k:x for k,x in v.items()if k!='sha256'});return v
    monkeypatch.setattr(m,'_json',read)
    with pytest.raises(ValueError):m.read_resnet_fit(native_fit['output'],require_full=False)


@pytest.mark.parametrize('damage',['missing_bn','moment_shape','moment_nonfinite','missing_step','optimizer_step',
    'maximize','empty_scheduler_lr','scheduler_period','rng_domain','callback_missing','patience',
    'loop_boundary','bool_checkpoint_epoch','bool_pending_progress','loss_objective','model_recipe'])
def test_incomplete_or_changed_native_saved_domains_are_rejected(native_fit,damage):
    import torch
    m=api();output=native_fit['output'];identity=m._json(output/'identity.json')
    result=m._json(output/'fit-result.json')
    path=output/(result['resume_evidence']['parent_file']if damage.startswith('bool_')else'checkpoints/last.ckpt')
    c=m._trusted_checkpoint(path);rows=c['resnet_fit']['records']
    opt=c['optimizer_states'][0];index=opt['param_groups'][0]['params'][0]
    if damage=='missing_bn':c['state_dict'].pop('model.backbone.1.running_mean')
    elif damage=='moment_shape':opt['state'][index]['exp_avg']=opt['state'][index]['exp_avg'].reshape(-1)
    elif damage=='moment_nonfinite':opt['state'][index]['exp_avg'].fill_(float('nan'))
    elif damage=='missing_step':opt['state'][index].pop('step')
    elif damage=='optimizer_step':opt['state'][index]['step']-=1
    elif damage=='maximize':opt['param_groups'][0]['maximize']=True
    elif damage=='empty_scheduler_lr':c['lr_schedulers'][0]['_last_lr']=[]
    elif damage=='scheduler_period':c['lr_schedulers'][0]['T_max']=149
    elif damage=='rng_domain':c['resnet_fit']['rng'].pop('numpy')
    elif damage=='callback_missing':
        key=next(k for k in c['callbacks']if k.startswith('ResnetFitWitness'));c['callbacks'].pop(key)
    elif damage=='patience':m._native_callback(c,'EarlyStopping{')['patience']=1
    elif damage=='loop_boundary':c['loops']['fit_loop']['epoch_progress']['current']['completed']+=1
    elif damage=='bool_checkpoint_epoch':c['epoch']=True
    elif damage=='bool_pending_progress':c['loops']['fit_loop']['epoch_progress']['current']['completed']=True
    elif damage=='loss_objective':
        c['hyper_parameters']['loss_weights']=dict(c['hyper_parameters']['loss_weights'])
        c['hyper_parameters']['loss_weights']['fingertip_distance']=0.
    else:
        c['hyper_parameters']['module_conf']=dict(c['hyper_parameters']['module_conf'])
        c['hyper_parameters']['module_conf']['backbone_type']='resnet50'
    with pytest.raises(ValueError):m._validate_checkpoint(c,identity,rows)
