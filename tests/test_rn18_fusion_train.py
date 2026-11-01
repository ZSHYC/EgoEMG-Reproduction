"""Actual RN18+S native Adam epochs, full role coverage and durable restore."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_rn18_fusion_runtime import branches
from test_fusion_literal_data import joint_store,UPSTREAM
from test_vision_literal_data import crop_store,release_fixture_readers
from test_rn18_fusion_data import config

ROOT=Path(__file__).resolve().parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.rn18_fusion_train')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_fusion_train':pytest.fail('RN18 native fusion fit missing')
        raise


def test_exact_full_native_policy_preserves_all200_ten_workers_and150_budget():
    pytest.importorskip('pytorch_lightning');m=api();p=m.rn18_fusion_fit_policy(UPSTREAM,config())
    assert p['max_epochs']==p['scheduler_T_max']==150 and p['scheduler_eta_min']==1e-5
    assert p['batch_size_by_role']==dict(train=200,val=200,test=200)
    assert p['workers_by_role']==dict(train=10,val=10,test=10)
    assert p['loader']==dict(pin_memory=True,persistent_workers=True,prefetch_factor=2)
    assert p['optimizer_class']=='torch.optim.adam.Adam'and p['optimizer_lr']==.0006 and p['optimizer_weight_decay']==.0001
    assert p['precision']=='bf16-mixed'and p['gradient_clip_val']==1 and not p['early_stopping']
    assert p['check_val_every_n_epoch']==1 and p['save_top_k']==3 and p['save_last']
    assert p['engineering_train_samples']is None and p['num_sanity_val_steps']==2
    rows=[dict(train_loss=1.,validation=dict(val_mae=.2))for _ in range(150)]
    assert m.rn18_fusion_stage_complete(p,rows)and not m.rn18_fusion_stage_complete(p,rows[:-1])
    rows[-1]['validation']['val_mae']=float('nan');assert not m.rn18_fusion_stage_complete(p,rows)


@pytest.mark.parametrize('scope',[None,True,'smoke','full_cpu'])
def test_unrecognized_scope_is_not_a_budget_override(scope):
    pytest.importorskip('pytorch_lightning')
    with pytest.raises(ValueError):api().rn18_fusion_fit_policy(UPSTREAM,config(),scope=scope)


@pytest.fixture(scope='module')
def native_runs(tmp_path_factory,request):
    pytest.importorskip('pytorch_lightning');api()
    existing=os.environ.get('EGOEMG_RN18_FIT_FIXTURE')
    if existing:
        root=Path(existing);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        base=[sys.executable,'-u','-B','-m','egoemg_repro.rn18_fusion_train','fit','--data',str(root/'data'),'--crops',str(root/'crops'),'--upstream',str(UPSTREAM),'--output',str(root/'fit'),'--scope','cpu_engineering','--branches',str(root/'branches.json')]
        prefix=(root/'prefix.json').read_bytes();logs={str(x.relative_to(root/'fit')):hashlib.sha256(x.read_bytes()).hexdigest()for x in(root/'fit/logs').rglob('*')if x.is_file()}
        yield root,env,base,prefix,logs
        return
    branches=request.getfixturevalue('branches')
    root=tmp_path_factory.mktemp('rn18-native-fusion-fit');data=root/'data';joint_store(data,n=9*12000)
    np.repeat([0,1,2,3],[6*12000,12000,12000,12000]).astype(np.int8).tofile(data/'frame_split_id.dat')
    crops=root/'crops';crop_store(crops,flat=True)
    (root/'branches.json').write_text(json.dumps({k:str(v)for k,v in branches.items()}))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    base=[sys.executable,'-u','-B','-m','egoemg_repro.rn18_fusion_train','fit','--data',str(data),'--crops',str(crops),'--upstream',str(UPSTREAM),'--output',str(root/'fit'),'--scope','cpu_engineering','--branches',str(root/'branches.json')]
    for phase,extra in [('prefix',['--stop-after-epoch','1']),('resume',['--resume'])]:
        child=subprocess.run(base+extra,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
        (root/(phase+'.log')).write_text(child.stdout+child.stderr);assert child.returncode==0,child.stderr[-5500:]
        if phase=='prefix':
            prefix=(root/'fit/fit-result.json').read_bytes();(root/'prefix.json').write_bytes(prefix)
            logs={str(x.relative_to(root/'fit')):hashlib.sha256(x.read_bytes()).hexdigest()for x in(root/'fit/logs').rglob('*')if x.is_file()}
    try:yield root,env,base,prefix,logs
    finally:release_fixture_readers(root)


def test_actual_native_prefix_resume_restores_six_domains_and_complete_validation(native_runs):
    import torch
    root,_,_,prefix,logs=native_runs;m=api();fit=root/'fit';early=json.loads(prefix);value=m.read_rn18_fusion_fit(fit,require_full=False)
    assert early['completed_epochs']==1 and early['optimizer_executions']==2 and not early['stage_complete']
    assert value['completed_epochs']==2 and value['optimizer_executions']==4 and value['stage_complete']
    assert value['resume_start_exact']=={k:True for k in('model','optimizer','scheduler','callbacks','loops','rng')}
    assert value['resume_validation_exact']and value['train_samples_per_epoch']==[4,4]and value['val_samples_per_epoch']==[2,2]
    assert all(len(r['validation']['pose_sha256'])==64 for r in packet_records(fit))
    assert not value['full_training_terminated']and not value['scientific_eligible']and not value['cuda_initialized']
    assert logs and all(hashlib.sha256((fit/n).read_bytes()).hexdigest()==h for n,h in logs.items())
    packet=torch.load(fit/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    assert packet['rn18_fusion_fit']['optimizer_class']=='torch.optim.adam.Adam'
    assert packet['lr_schedulers'][0]['T_max']==150 and packet['lr_schedulers'][0]['last_epoch']==2
    assert not any(k.startswith('EarlyStopping')for k in packet['callbacks'])
    groups=packet['optimizer_states'][0]['param_groups'];assert[group['name']for group in groups]==['featurizer','decoder','vision_proj','fusion_proj','head','head_vision','vision_backbone','_unassigned']
    assert all(group['initial_lr']==.0006 and group['weight_decay']==.0001 for group in groups)
    assert value['resume_evidence']and(fit/value['resume_evidence']['parent_file']).exists()
    assert(fit/value['resume_evidence']['observation_file']).exists()
    with pytest.raises(ValueError):m.read_rn18_fusion_fit(fit)


@pytest.mark.parametrize('damage',['optimizer_step','optimizer_moment','scheduler','loop','rng','hyperparameters','missing_temporal','packet_class','callback','epoch_number'])
def test_actual_saved_native_state_damage_is_not_accepted_under_resealed_summaries(native_runs,damage):
    import torch
    from egoemg_repro.classic_emg import _hash
    root,*_=native_runs;fit=root/'fit';p=fit/'checkpoints/last.ckpt';r=fit/'fit-result.json';before=p.read_bytes();old=r.read_bytes()
    try:
        value=torch.load(p,map_location='cpu',weights_only=False);state=value['optimizer_states'][0]
        if damage=='optimizer_step':state['state'][0]['step']-=1
        elif damage=='optimizer_moment':state['state'][0]['exp_avg']=torch.zeros(1)
        elif damage=='scheduler':value['lr_schedulers'][0]['T_max']=149
        elif damage=='loop':value['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped']-=1
        elif damage=='rng':value['rn18_fusion_fit']['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
        elif damage=='hyperparameters':value['hyper_parameters']['optimizer_conf']['_target_']='torch.optim.AdamW'
        elif damage=='missing_temporal':state['param_groups'][-1]['params'].pop()
        elif damage=='callback':value['callbacks']['FusionFitWitness']['steps']-=1
        elif damage=='epoch_number':value['rn18_fusion_fit']['records'][-1]['epoch']=0
        else:value['rn18_fusion_fit']['optimizer_class']='torch.optim.adamw.AdamW'
        torch.save(value,p);result=json.loads(old);result['last_checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();result['sha256']=_hash({k:v for k,v in result.items()if k!='sha256'});r.write_text(json.dumps(result))
        with pytest.raises(ValueError):api().read_rn18_fusion_fit(fit,require_full=False)
    finally:p.write_bytes(before);r.write_bytes(old)


def test_completed_fit_cannot_be_retrained_or_smaller_full_budget(native_runs,tmp_path):
    root,env,base,_,_=native_runs
    child=subprocess.run([*base,'--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    assert child.returncode!=0 and'complete'in child.stderr.lower()
    identity=json.loads((root/'fit/identity.json').read_text());policy=copy.deepcopy(identity['policy']);policy['max_epochs']=2
    from egoemg_repro.classic_emg import _hash
    policy['scope']='full';policy['sha256']=_hash({k:v for k,v in policy.items()if k!='sha256'})
    with pytest.raises(ValueError):api()._check_policy(policy,identity['cohort'])


def test_engineering_branches_never_start_formal_training(native_runs,tmp_path,monkeypatch):
    import torch
    from egoemg_repro.rn18_fusion_cohort import admit_rn18_fusion_cohort
    from egoemg_repro.rn18_fusion_runtime import engineering_rn18_fusion_initializers
    root,*_=native_runs;c=admit_rn18_fusion_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config(),require_full=False)
    b=engineering_rn18_fusion_initializers(UPSTREAM,c.config,**json.loads((root/'branches.json').read_text()))
    def deny(*a,**k):raise AssertionError('invalid full source reached a constructor')
    monkeypatch.setattr(torch.nn.Module,'__init__',deny)
    output=tmp_path/'must-not-exist'
    with pytest.raises(ValueError):api().fit_rn18_fusion(c,b,output,scope='full')
    assert not output.exists()


def packet_records(fit):
    import torch
    return torch.load(fit/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)['rn18_fusion_fit']['records']
