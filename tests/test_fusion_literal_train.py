"""Real native center-fusion epochs, full coverage and fresh-process recovery."""
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

from test_fusion_literal_data import joint_store,config,UPSTREAM,W
from test_vision_literal_data import crop_store
from test_fusion_literal_runtime import branch_files,native_packet
from test_fusion_literal_cohort import files

ROOT=Path(__file__).parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.fusion_literal_train')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.fusion_literal_train':pytest.fail('complete native fusion training missing')
        raise


def test_full_policy_is_original_150_no_early_stop_all200_bf16_cosine():
    pytest.importorskip('pytorch_lightning');p=api().fusion_fit_policy(UPSTREAM,config())
    assert p['max_epochs']==p['scheduler_T_max']==150 and p['scheduler_eta_min']==1e-5
    assert p['batch_size_by_role']==dict(train=200,val=200,test=200)
    assert p['precision']=='bf16-mixed' and p['gradient_clip_val']==1
    assert p['monitor']=='val_mae' and p['monitor_mode']=='min' and p['save_top_k']==3 and p['save_last']
    assert p['early_stopping'] is False and p['engineering_train_samples'] is None
    assert p['num_sanity_val_steps']==2 and p['native_global_loader_rng'] and p['workers']==0
    assert not p['scientific_eligible'] and not p['full_paper_complete']


@pytest.mark.parametrize('scope',[None,True,'smoke','full_cpu'])
def test_implicit_scope_or_smaller_full_budget_is_refused(scope):
    pytest.importorskip('pytorch_lightning')
    with pytest.raises(ValueError):api().fusion_fit_policy(UPSTREAM,config(),scope=scope)


def test_full_budget_requires_all150_complete_finite_epochs():
    pytest.importorskip('pytorch_lightning');m=api();p=m.fusion_fit_policy(UPSTREAM,config())
    rows=[dict(train_loss=1.,validation=dict(val_mae=.2)) for _ in range(150)]
    assert m.fusion_stage_complete(p,rows) and not m.fusion_stage_complete(p,rows[:-1])
    rows[-1]['validation']['val_mae']=float('nan');assert not m.fusion_stage_complete(p,rows)


@pytest.fixture(scope='module')
def native_runs(branch_files,tmp_path_factory):
    pytest.importorskip('pytorch_lightning');api()
    root=tmp_path_factory.mktemp('native-fusion-full-fit');data=root/'data';crops=root/'crops'
    joint_store(data,n=9*W);crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*W,W,W,W]).astype('int8').tofile(data/'frame_split_id.dat')
    args={k:str(v) for k,v in branch_files.items()};(root/'branches.json').write_text(json.dumps(args))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    code='''from pathlib import Path
import json,sys,torch
from egoemg_repro.fusion_literal import compose_fusion
from egoemg_repro.fusion_literal_data import fusion_normalization
from egoemg_repro.fusion_literal_cohort import admit_fusion_cohort
from egoemg_repro.fusion_initializer_handoff import engineering_fusion_initializers
from egoemg_repro.fusion_literal_train import fit_fusion
root=Path(sys.argv[1]);up=Path(sys.argv[2]);name=sys.argv[3];mode=sys.argv[4]
c=admit_fusion_cohort(root/'data',root/'crops',upstream=up,config=compose_fusion(up),normalization=fusion_normalization(up))
b=engineering_fusion_initializers(c,**json.loads((root/'branches.json').read_text()))
r=fit_fusion(c,b,root/name,scope='cpu_engineering',resume=mode=='resume',stop_after_epoch=1 if mode=='prefix' else None)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(epochs=r['completed_epochs'],updates=r['optimizer_executions'],complete=r['stage_complete'])))
'''
    def run(name,mode):
        p=subprocess.run([sys.executable,'-B','-c',code,str(root),str(UPSTREAM),name,mode],cwd=ROOT,env=env,
            capture_output=True,text=True,timeout=240)
        (root/(name+'-'+mode+'.stdout')).write_text(p.stdout);(root/(name+'-'+mode+'.stderr')).write_text(p.stderr)
        assert p.returncode==0,p.stderr[-8000:]
    run('resumed','prefix');prefix=(root/'resumed/fit-result.json').read_bytes()
    logs={str(p.relative_to(root/'resumed')):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/'resumed/logs').rglob('*') if p.is_file()}
    assert logs,'actual native TensorBoard prefix logs must exist'
    run('resumed','resume')
    run('crash','prefix');crash_prefix=(root/'crash/fit-result.json').read_bytes()
    with (root/'crash/epochs.jsonl').open('ab') as f:f.write(b'{"uncommitted":true}\n')
    (root/'crash/fit-result.json').unlink();run('crash','resume')
    final=(root/'crash/fit-result.json').read_bytes()
    (root/'crash/fit-result.json').write_bytes(crash_prefix);run('crash','resume')
    return root,env,code,prefix,logs,crash_prefix,final


def test_real_native_epochs_restore_six_states_and_entire_frozen_validation(native_runs):
    import torch
    root,_,_,prefix,logs,_,_=native_runs;m=api();out=root/'resumed'
    early=json.loads(prefix);done=m.read_fusion_fit(out,require_full=False)
    assert early['completed_epochs']==1 and early['optimizer_executions']==2 and not early['stage_complete']
    assert done['completed_epochs']==2 and done['optimizer_executions']==4 and done['stage_complete']
    assert done['resume_start_exact']=={key:True for key in ('model','optimizer','scheduler','callbacks','loops','rng')}
    assert done['resume_validation_exact'] is True
    assert done['train_samples_per_epoch']==[4,4] and done['val_samples_per_epoch']==[4,4]
    assert done['selected_best']['file']!='last.ckpt' and done['selected_best']['validated']
    assert not done['full_training_terminated'] and not done['scientific_eligible'] and not done['cuda_initialized']
    assert done['bitwise_training_trajectory_guaranteed'] is False
    assert logs
    for name,digest in logs.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
    last=torch.load(str(out/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False)
    assert not any(k.startswith('EarlyStopping') for k in last['callbacks'])
    assert last['lr_schedulers'][0]['T_max']==150 and last['lr_schedulers'][0]['eta_min']==1e-5
    groups=last['optimizer_states'][0]['param_groups'];assert groups[-1]['name']=='_unassigned' and groups[-1]['params']


def test_engineering_fit_never_becomes_a_full_scientific_model(native_runs):
    root,*_=native_runs
    with pytest.raises(ValueError,match='engineering|full'):api().read_fusion_fit(root/'resumed')


def test_uncommitted_tail_and_stale_result_recover_without_extra_budget_updates(native_runs):
    root,_,_,_,_,old,previous=native_runs;out=root/'crash';value=api().read_fusion_fit(out,require_full=False)
    assert value['completed_epochs']==2 and value['optimizer_executions']==4 and value['stage_complete']
    assert value['completion_recovered_from_native_checkpoint'] and value['additional_optimizer_updates_during_recovery']==0
    assert value['resume_validation_exact'] is True
    assert list(out.glob('uncommitted-epochs-*.jsonl'))[0].read_bytes()==b'{"uncommitted":true}\n'
    assert (out/('historical-fit-result-'+hashlib.sha256(old).hexdigest()+'.json')).read_bytes()==old
    assert value['last_checkpoint_sha256']==json.loads(previous)['last_checkpoint_sha256']


@pytest.mark.parametrize('damage',['result_counter','wrong_best','coverage','last_state','handoff','receipt'])
def test_completed_native_files_and_lineage_are_checked_not_self_declared(native_runs,damage):
    import torch
    from egoemg_repro.classic_emg import _hash
    root,*_=native_runs;out=root/'resumed';path=out/'fit-result.json'
    if damage=='last_state':path=out/'checkpoints/last.ckpt'
    elif damage in ('handoff','receipt'):path=out/'identity.json'
    original=path.read_bytes()
    try:
        if damage=='last_state':
            v=torch.load(str(path),map_location='cpu',weights_only=False)
            first=next(iter(v['state_dict']));v['state_dict'][first]=v['state_dict'][first]+.01;torch.save(v,path)
        else:
            v=json.loads(original)
            if damage=='result_counter':v['optimizer_executions']+=1
            elif damage=='wrong_best':v['selected_best']['file']='last.ckpt'
            elif damage=='coverage':v['val_samples_per_epoch'][-1]-=1
            elif damage=='handoff':
                h=v['initializer_descriptor']['identity'];h['vision']['checkpoint_sha256']='0'*64
                h['sha256']=_hash({k:x for k,x in h.items() if k!='sha256'})
            elif damage=='receipt':v['cohort']['payload_files_sha256']['field:emg_left_filtered']='0'*64
            v['sha256']=_hash({k:x for k,x in v.items() if k!='sha256'});path.write_text(json.dumps(v))
        with pytest.raises(ValueError):api().read_fusion_fit(out,require_full=False)
    finally:path.write_bytes(original)


def test_full_on_engineering_handoff_refuses_before_gpu_model_and_output(files,branch_files,tmp_path,monkeypatch):
    from test_fusion_literal_cohort import admit
    from egoemg_repro.fusion_initializer_handoff import engineering_fusion_initializers
    import egoemg_repro.fusion_literal_runtime as runtime
    c=admit(files);b=engineering_fusion_initializers(c,**branch_files)
    def forbidden(*args,**kwargs):raise AssertionError('no CUDA/model before full admission')
    monkeypatch.setattr(runtime,'build_fusion_runtime',forbidden);output=tmp_path/'forbidden'
    with pytest.raises(ValueError,match='full|registered|receipt|engineering'):api().fit_fusion(c,b,output)
    assert not output.exists()


def test_completed_budget_with_valid_result_refuses_retraining(native_runs):
    root,env,code,*_=native_runs;out=root/'resumed'
    before={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    p=subprocess.run([sys.executable,'-B','-c',code,str(root),str(UPSTREAM),'resumed','resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    assert p.returncode!=0 and 'completed' in p.stderr
    assert {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('damage',['actual_input','actual_branch'])
def test_fit_reader_reopens_actual_data_and_initializer_bytes(native_runs,damage):
    root,*_=native_runs;out=root/'resumed';identity=json.loads((out/'identity.json').read_text())
    path=(root/'data/frame_split_id.dat') if damage=='actual_input' else Path(identity['initializer_descriptor']['files']['emg_checkpoint'])
    original=path.read_bytes()
    try:
        if damage=='actual_input':path.write_bytes(bytes([1])+original[1:])
        else:
            with path.open('ab') as f:f.write(b'changed actual initializer')
        with pytest.raises(ValueError):api().read_fusion_fit(out,require_full=False)
    finally:path.write_bytes(original)
