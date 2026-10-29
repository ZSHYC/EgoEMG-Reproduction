"""Native200/B500/six-worker policy and actual point-target fit recovery."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_emgformer_data import UPSTREAM,cfg
from test_emg_initializer_data import source

ROOT=Path(__file__).resolve().parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.emgformer_train')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emgformer_train':pytest.fail('native EMGFormer training and point validation missing')
        raise


@pytest.mark.parametrize('size',['small','middle','large'])
def test_full_policy_preserves_native200_500_six_callbacks_point_targets(size):
    pytest.importorskip('pytorch_lightning')
    p=api().emgformer_fit_policy(UPSTREAM,cfg(size))
    assert p['max_epochs']==p['scheduler_T_max']==200 and p['scheduler_eta_min']==1e-5
    assert p['precision']=='bf16-mixed'and p['batch_size_by_role']==dict(train=500,val=500)
    assert p['workers']==6 and p['prefetch_factor']==1 and p['persistent_workers']and not p['pin_memory']
    assert p['early_stopping']['patience']==200 and p['early_stopping']['monitor']=='val_mae'
    assert p['monitor']=='val_mae'and p['monitor_mode']=='min'and p['save_top_k']==3 and p['save_last']
    assert p['num_sanity_val_steps']==2 and p['native_global_loader_rng']and p['engineering_train_samples']is None
    assert p['effective_loss_weights']['landmark/fingertip']==.02
    assert p['validation_target_shape']==[22,1]and p['training_target_shape']==[22,12000]
    assert p['gradient_clip_val']==1 and not p['bitwise_training_trajectory_guaranteed']


@pytest.mark.parametrize('scope',[None,True,'smoke','full_cpu'])
def test_unknown_scope_rejects(scope):
    with pytest.raises(ValueError):api().emgformer_fit_policy(UPSTREAM,cfg(),scope=scope)


def test_shortened_or_nonfinite_history_cannot_claim_full_completion():
    pytest.importorskip('pytorch_lightning')
    m=api();p=m.emgformer_fit_policy(UPSTREAM,cfg());r=[dict(train_loss=1.,validation=dict(val_mae=.2))for _ in range(200)]
    assert m.emgformer_stage_complete(p,r)and not m.emgformer_stage_complete(p,r[:-1])
    r[-1]['validation']['val_mae']=float('nan');assert not m.emgformer_stage_complete(p,r)
    cpu=m.emgformer_fit_policy(UPSTREAM,cfg(),scope='cpu_engineering');gpu=m.emgformer_fit_policy(UPSTREAM,cfg(),scope='cuda_engineering')
    assert cpu['max_epochs']==gpu['max_epochs']==2 and cpu['workers']==0 and gpu['workers']==6
    assert cpu['engineering_train_samples']==4 and gpu['engineering_train_samples']==1000
    assert cpu['batch_size_by_role']==dict(train=2,val=2)and gpu['batch_size_by_role']==dict(train=500,val=500)


def test_owned_output_lock_rejects_before_model_or_checkpoint_creation(tmp_path):
    pytest.importorskip('torch')
    import fcntl
    from egoemg_repro.emgformer_cohort import admit_emgformer_cohort
    from egoemg_repro.emgformer_literal import emgformer_normalization
    data=tmp_path/'data';data.mkdir();source(data);config=cfg()
    cohort=admit_emgformer_cohort(data,upstream=UPSTREAM,config=config,normalization=emgformer_normalization(UPSTREAM,config))
    out=tmp_path/'locked';out.mkdir();fd=os.open(out/'fit.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):api().fit_emgformer(cohort,out,scope='cpu_engineering',resume=True)
        assert not(out/'identity.json').exists()and not(out/'checkpoints').exists()
    finally:os.close(fd)


@pytest.fixture(scope='module')
def native_runs(tmp_path_factory):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');pytest.importorskip('timm');api()
    root=tmp_path_factory.mktemp('native-emgformer-point-fit');data=root/'data';data.mkdir();source(data,optional_labels=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    output=root/'resumed';base=[sys.executable,'-B','-m','egoemg_repro.emgformer_train','fit','--size','small',
        '--data',str(data),'--upstream',str(UPSTREAM),'--output',str(output),'--scope','cpu_engineering']
    for mode,extra in [('prefix',['--stop-after-epoch','1']),('resume',['--resume'])]:
        p=subprocess.run(base+extra,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
        (root/(mode+'.stdout')).write_text(p.stdout);(root/(mode+'.stderr')).write_text(p.stderr)
        assert p.returncode==0,p.stderr[-7000:]
        if mode=='prefix':(root/'prefix-result.json').write_bytes((output/'fit-result.json').read_bytes())
    return root,output,data,env,base


def test_actual_native_point_forward_balanced_hands_and_six_state_fresh_resume(native_runs):
    import torch
    root,out,_,_,_=native_runs;first=json.loads((root/'prefix-result.json').read_text());final=api().read_emgformer_fit(out,require_full=False)
    assert first['completed_epochs']==1 and first['optimizer_executions']==2 and not first['stage_complete']
    assert final['completed_epochs']==2 and final['optimizer_executions']==4 and final['stage_complete']
    assert final['resume_start_exact']=={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')}
    assert final['resume_validation_exact']and final['train_samples_per_epoch']==[4,4]and final['val_samples_per_epoch']==[4,4]
    assert final['train_hand_samples_per_epoch']==[dict(left=2,right=2)]*2
    assert final['selected_best']['file']!='last.ckpt'and final['selected_best']['validated']
    assert not final['cuda_initialized']and not final['full_training_terminated']and not final['bitwise_training_trajectory_guaranteed']
    last=torch.load(str(out/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False);identity=last['emgformer_fit']['identity']
    assert identity['train_indices']==[0,1,170,171]and identity['construction']['effective_loss_weights']['landmark/fingertip']==.02
    assert last['lr_schedulers'][0]['T_max']==200 and last['lr_schedulers'][0]['eta_min']==1e-5
    early=next(v for k,v in last['callbacks'].items()if k.startswith('EarlyStopping'))
    assert early['patience']==200
    for r in last['emgformer_fit']['records']:
        assert r['validation']['target_shape']==[22,1]and r['validation']['native_center_selection_exact']
        assert r['validation']['hand_samples']==dict(left=2,right=2)


def test_engineering_not_full_and_finished_run_refuses_more_updates(native_runs):
    _,out,_,env,base=native_runs
    with pytest.raises(ValueError):api().read_emgformer_fit(out)
    before=(out/'fit-result.json').read_bytes();p=subprocess.run(base+['--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    assert p.returncode!=0 and 'completed EMGFormer cannot be retrained'in p.stderr
    assert (out/'fit-result.json').read_bytes()==before


@pytest.mark.parametrize('damage',['history','last','best','coverage','selected_score','hand_coverage'])
def test_durable_reader_rejects_tampered_selected_state_and_coverage(native_runs,damage):
    from egoemg_repro.classic_emg import _hash
    _,out,_,_,_=native_runs;result=json.loads((out/'fit-result.json').read_text())
    if damage in ('history','last','best'):
        path=out/('epochs.jsonl'if damage=='history'else 'checkpoints/last.ckpt'if damage=='last'else 'checkpoints/'+result['selected_best']['file'])
        old=path.read_bytes()
        try:
            path.write_bytes(old+b'bad')
            with pytest.raises(ValueError):api().read_emgformer_fit(out,require_full=False)
        finally:path.write_bytes(old)
    else:
        path=out/'fit-result.json';old=path.read_bytes();v=json.loads(old)
        if damage=='coverage':v['train_samples_per_epoch'][-1]-=1
        elif damage=='selected_score':v['selected_best']['score']+=.2
        else:v['train_hand_samples_per_epoch'][-1]['right']=0
        v['sha256']=_hash({k:x for k,x in v.items()if k!='sha256'})
        try:
            path.write_text(json.dumps(v))
            with pytest.raises(ValueError):api().read_emgformer_fit(out,require_full=False)
        finally:path.write_bytes(old)


def test_complete_checkpoint_recovers_stale_result_without_more_updates(native_runs):
    root,out,_,env,base=native_runs;old=(out/'fit-result.json').read_bytes()
    checkpoint=out/'checkpoints/last.ckpt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    try:
        prefix=(root/'prefix-result.json').read_bytes();(out/'fit-result.json').write_bytes(prefix)
        p=subprocess.run(base+['--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        (root/'recover-complete.stderr').write_text(p.stderr)
        assert p.returncode==0,p.stderr[-6000:]
        result=api().read_emgformer_fit(out,require_full=False)
        assert result['completion_recovered_from_native_checkpoint']and result['additional_optimizer_updates_during_recovery']==0
        assert result['optimizer_executions']==4 and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
        assert any(p.read_bytes()==prefix for p in out.glob('historical-fit-result-*.json'))
    finally:(out/'fit-result.json').write_bytes(old)


def _raw_copy(native_runs,tmp_path):
    """Private checkpoint copies; original fit evidence is never modified."""
    import copy
    import torch
    from egoemg_repro.vision_literal_train import _epoch_bytes,_sealed
    _,original,_,_,_=native_runs;out=tmp_path/'fit';(out/'checkpoints').mkdir(parents=True)
    identity=json.loads((original/'identity.json').read_text());result=json.loads((original/'fit-result.json').read_text())
    identity['fit_sources_sha256']=api()._sources()
    identity['policy']=api().emgformer_fit_policy(UPSTREAM,cfg(),scope='cpu_engineering')
    identity=_sealed({k:v for k,v in identity.items()if k!='sha256'})
    def rebase(v):
        if isinstance(v,str):return str(out)+v[len(str(original)):]if v.startswith(str(original))else v
        if isinstance(v,dict):return {rebase(k):rebase(x)for k,x in v.items()}
        if isinstance(v,list):return [rebase(x)for x in v]
        return v
    for name in {'last.ckpt',result['selected_best']['file']}:
        c=torch.load(str(original/'checkpoints'/name),map_location='cpu',weights_only=False)
        c['emgformer_fit']['identity']=copy.deepcopy(identity);c['callbacks']=rebase(c['callbacks'])
        if 'native_state_sha256'in c['emgformer_fit']:c['emgformer_fit']['native_state_sha256']=api()._checkpoint_snap(c)
        torch.save(c,str(out/'checkpoints'/name))
        if name=='last.ckpt':last=c
    (out/'identity.json').write_text(json.dumps(identity))
    (out/'epochs.jsonl').write_bytes(_epoch_bytes(last['emgformer_fit']['records'],identity['sha256']))
    result['identity_sha256']=identity['sha256'];_raw_result(out,result)
    return out,identity,result


def _raw_result(out,result):
    from egoemg_repro.vision_literal_train import _sealed
    result['last_checkpoint_sha256']=hashlib.sha256((out/'checkpoints/last.ckpt').read_bytes()).hexdigest()
    result['selected_best']['checkpoint_sha256']=hashlib.sha256((out/'checkpoints'/result['selected_best']['file']).read_bytes()).hexdigest()
    (out/'fit-result.json').write_text(json.dumps(_sealed({k:v for k,v in result.items()if k!='sha256'})))


def test_raw_native_checkpoint_copy_accepts_real_save_boundary_without_models(native_runs,tmp_path,monkeypatch):
    import torch,time
    out,identity,_=_raw_copy(native_runs,tmp_path)
    monkeypatch.setattr(api(),'build_emgformer_runtime',lambda *a,**kw:pytest.fail('raw acceptance must not construct a model'))
    before=torch.get_rng_state().clone();c=torch.load(str(out/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False)
    assert c['loops']['fit_loop']['epoch_progress']['total']==dict(ready=2,completed=1,started=2,processed=2)
    assert api().read_emgformer_fit(out,require_full=False)['stage_complete']
    assert api()._completed_checkpoint_result(out,identity,c,dict(accelerator='cpu',cuda_initialized=False),True,time.monotonic())['stage_complete']
    assert torch.equal(torch.get_rng_state(),before)and not torch.cuda.is_initialized()


@pytest.mark.parametrize('which',['last','selected'])
@pytest.mark.parametrize('damage',[
    'missing_all','missing_optimizer','missing_scheduler','missing_loops','missing_moment','extra_moment',
    'moment_shape','moment_dtype','moment_step','moment_negative','moment_nonfinite','finite_moment_corruption',
    'parameter_order','optimizer_option','optimizer_lr','scheduler_horizon','scheduler_progress','scheduler_lr',
    'epoch','loop_progress','loop_boundary','witness','rng_missing','rng_shape','rng_cuda','scaler','missing_snapshot'])
def test_raw_native_states_reject_before_recovery_or_independent_acceptance(native_runs,tmp_path,monkeypatch,which,damage):
    import torch,time
    out,identity,result=_raw_copy(native_runs,tmp_path)
    monkeypatch.setattr(api(),'build_emgformer_runtime',lambda *a,**kw:pytest.fail('corrupt raw state must fail before model construction'))
    path=out/'checkpoints'/('last.ckpt'if which=='last'else result['selected_best']['file'])
    c=torch.load(str(path),map_location='cpu',weights_only=False)
    if damage=='missing_all':
        for k in ('optimizer_states','lr_schedulers','loops'):c.pop(k)
    elif damage.startswith('missing_')and damage!='missing_moment'and damage!='missing_snapshot':
        c.pop({'missing_optimizer':'optimizer_states','missing_scheduler':'lr_schedulers','missing_loops':'loops'}[damage])
    elif damage=='missing_moment':c['optimizer_states'][0]['state'].pop(0)
    elif damage=='extra_moment':c['optimizer_states'][0]['state'][9999]=c['optimizer_states'][0]['state'][0]
    elif damage=='moment_shape':c['optimizer_states'][0]['state'][0]['exp_avg']=torch.zeros(1)
    elif damage=='moment_dtype':c['optimizer_states'][0]['state'][0]['exp_avg']=c['optimizer_states'][0]['state'][0]['exp_avg'].double()
    elif damage=='moment_step':c['optimizer_states'][0]['state'][0]['step']-=1
    elif damage=='moment_negative':c['optimizer_states'][0]['state'][0]['exp_avg_sq'].flatten()[0]=-1
    elif damage=='moment_nonfinite':c['optimizer_states'][0]['state'][0]['exp_avg'].flatten()[0]=float('nan')
    elif damage=='finite_moment_corruption':c['optimizer_states'][0]['state'][0]['exp_avg'].flatten()[0]+=.1
    elif damage=='parameter_order':c['optimizer_states'][0]['param_groups'][0]['params'].reverse()
    elif damage=='optimizer_option':c['optimizer_states'][0]['param_groups'][0]['maximize']=True
    elif damage=='optimizer_lr':c['optimizer_states'][0]['param_groups'][0]['lr']*=2
    elif damage=='scheduler_horizon':c['lr_schedulers'][0]['T_max']=2
    elif damage=='scheduler_progress':c['lr_schedulers'][0]['_step_count']-=1
    elif damage=='scheduler_lr':c['lr_schedulers'][0]['_last_lr'][0]*=2
    elif damage=='epoch':c['epoch']+=1
    elif damage=='loop_progress':c['loops']['fit_loop']['epoch_loop.batch_progress']['total']['processed']-=1
    elif damage=='loop_boundary':c['loops']['fit_loop']['epoch_progress']['total']['completed']+=1
    elif damage=='witness':c['callbacks']['EmgformerFitWitness']['steps']-=1
    elif damage=='rng_missing':c['emgformer_fit']['rng'].pop('python')
    elif damage=='rng_shape':c['emgformer_fit']['rng']['torch']=torch.zeros(3,dtype=torch.uint8)
    elif damage=='rng_cuda':c['emgformer_fit']['rng']['cuda']=[torch.zeros(16,dtype=torch.uint8)]
    elif damage=='scaler':c['MixedPrecision']={}
    elif damage=='missing_snapshot':c['emgformer_fit'].pop('native_state_sha256',None)
    # Re-seal semantic mutations to exercise raw structure validation too. The
    # finite moment mutation intentionally retains its original content hash.
    if damage not in ('finite_moment_corruption','missing_snapshot')and 'native_state_sha256'in c['emgformer_fit']:
        try:c['emgformer_fit']['native_state_sha256']=api()._checkpoint_snap(c)
        except KeyError:pass
    torch.save(c,str(path));_raw_result(out,result);before=(out/'fit-result.json').read_bytes()
    with pytest.raises(ValueError):api().read_emgformer_fit(out,require_full=False)
    last=torch.load(str(out/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False)
    with pytest.raises(ValueError):api()._completed_checkpoint_result(out,identity,last,dict(accelerator='cpu',cuda_initialized=False),True,time.monotonic())
    assert (out/'fit-result.json').read_bytes()==before and not torch.cuda.is_initialized()
