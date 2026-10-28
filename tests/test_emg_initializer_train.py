"""Native complete epochs, six-state fresh restore and validation-selected export."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_emg_initializer_literal import UPSTREAM,config
from test_emg_initializer_data import source

ROOT=Path(__file__).parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.emg_initializer_train')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emg_initializer_train':pytest.fail('original EMG initializer fit and recovery missing')
        raise


def test_full_policy_preserves_150_no_early_stop_original_260_bf16_cosine():
    pytest.importorskip('pytorch_lightning');p=api().emg_initializer_fit_policy(UPSTREAM,config(),scope='full')
    assert p['max_epochs']==p['scheduler_T_max']==150 and p['scheduler_eta_min']==5e-6
    assert p['precision']=='bf16-mixed' and p['batch_size_by_role']==dict(train=260,val=260)
    assert p['monitor']=='val_mae' and p['monitor_mode']=='min' and p['save_top_k']==3 and p['save_last']
    assert p['early_stopping'] is False and p['gradient_clip_val']==1 and p['engineering_train_samples'] is None
    assert p['native_global_loader_rng'] and p['workers']==0 and p['num_sanity_val_steps']==2
    assert not p['scientific_eligible'] and not p['full_paper_complete']


@pytest.mark.parametrize('scope',[True,None,'smoke','full_cpu'])
def test_implicit_or_unknown_scope_rejects(scope):
    pytest.importorskip('pytorch_lightning')
    with pytest.raises(ValueError):api().emg_initializer_fit_policy(UPSTREAM,config(),scope=scope)


def test_full_budget_cannot_early_stop_or_accept_nonfinite_history():
    pytest.importorskip('pytorch_lightning');m=api();p=m.emg_initializer_fit_policy(UPSTREAM,config())
    r=[dict(train_loss=1.,validation=dict(val_mae=.2)) for _ in range(150)]
    assert m.emg_initializer_stage_complete(p,r)
    assert not m.emg_initializer_stage_complete(p,r[:-1])
    r[-1]['validation']['val_mae']=float('nan');assert not m.emg_initializer_stage_complete(p,r)


@pytest.fixture(scope='module')
def native_runs(tmp_path_factory):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');pytest.importorskip('timm');api()
    root=tmp_path_factory.mktemp('native-emg-initializer-fit');data=root/'data';data.mkdir();source(data,optional_labels=True)
    output=root/'resumed';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    base=[sys.executable,'-B','-m','egoemg_repro.emg_initializer_train','fit','--data',str(data),
        '--upstream',str(UPSTREAM),'--output',str(output),'--scope','cpu_engineering']
    for label,extra in [('prefix',['--stop-after-epoch','1']),('resume',['--resume'])]:
        p=subprocess.run(base+extra,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
        (root/(label+'.stdout')).write_text(p.stdout);(root/(label+'.stderr')).write_text(p.stderr)
        assert p.returncode==0,p.stderr[-8000:]
        if label=='prefix':
            (root/'prefix-result.json').write_bytes((output/'fit-result.json').read_bytes())
            logs={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('metrics.csv')}
    return root,output,data,env,logs


def test_actual_native_epochs_six_state_restore_and_complete_frozen_validation(native_runs):
    root,out,_,_,logs=native_runs;m=api();prefix=json.loads((root/'prefix-result.json').read_text())
    final=m.read_emg_initializer_fit(out,require_full=False)
    assert prefix['completed_epochs']==1 and prefix['optimizer_executions']==2 and not prefix['stage_complete']
    assert final['completed_epochs']==2 and final['optimizer_executions']==4 and final['stage_complete']
    assert final['resume_start_exact']=={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')}
    assert final['resume_validation_exact'] is True
    assert final['train_samples_per_epoch']==[4,4] and final['val_samples_per_epoch']==[4,4]
    assert final['selected_best']['file']!='last.ckpt' and final['selected_best']['validated']
    assert not final['cuda_initialized'] and not final['full_training_terminated'] and not final['scientific_eligible']
    assert final['bitwise_training_trajectory_guaranteed'] is False
    for name,digest in logs.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
    import torch
    last=torch.load(str(out/'checkpoints/last.ckpt'),map_location='cpu',weights_only=False)
    assert not any(k.startswith('EarlyStopping') for k in last['callbacks'])
    assert last['lr_schedulers'][0]['T_max']==150 and last['lr_schedulers'][0]['eta_min']==5e-6


def test_engineering_is_not_complete_full_initializer(native_runs):
    _,out,_,_,_=native_runs
    with pytest.raises(ValueError):api().read_emg_initializer_fit(out)


def test_selected_best_export_has_every_native_tensor_and_exact_source_bytes(native_runs,tmp_path):
    import torch
    _,out,_,_,_=native_runs;m=api();dest=tmp_path/'emg-best.pt'
    with pytest.raises(ValueError):m.export_emg_initializer_best(out,dest)
    assert not dest.exists()
    receipt=m.export_emg_initializer_best(out,dest,require_full=False)
    result=m.read_emg_initializer_fit(out,require_full=False)
    native=torch.load(str(out/'checkpoints'/result['selected_best']['file']),map_location='cpu',weights_only=False)
    exported=torch.load(str(dest),map_location='cpu',weights_only=False)
    assert set(exported['state_dict'])==set(native['state_dict'])==set(receipt['tensor_inventory'])
    for name,value in exported['state_dict'].items():assert torch.equal(value,native['state_dict'][name]),name
    assert receipt['source_checkpoint_sha256']==result['selected_best']['checkpoint_sha256']
    assert receipt['selection']=='actual native val_mae/min best; no last fallback' and not receipt['full_initializer']
    assert m.read_emg_initializer_export(dest,source_fit=out,require_full=False)==receipt
    key=next(iter(exported['state_dict']));exported['state_dict'][key].reshape(-1)[0]+=.25
    torch.save(exported,dest)
    with pytest.raises(ValueError):m.read_emg_initializer_export(dest,source_fit=out,require_full=False)


def test_export_reader_requires_an_independent_selected_fit_reference(native_runs,tmp_path):
    _,out,_,_,_=native_runs;m=api();dest=tmp_path/'unanchored.pt'
    m.export_emg_initializer_best(out,dest,require_full=False)
    with pytest.raises(ValueError):m.read_emg_initializer_export(dest)


@pytest.mark.parametrize('damage',['missing_tensor','invented_tensor','missing_lineage','checkpoint_lineage','fit_lineage','relabel_full'])
def test_resealed_export_cannot_invent_schema_or_change_selected_fit_lineage(native_runs,tmp_path,damage):
    import torch
    from egoemg_repro.classic_emg import _hash
    _,out,_,_,_=native_runs;m=api();dest=tmp_path/'changed.pt';m.export_emg_initializer_best(out,dest,require_full=False)
    value=torch.load(str(dest),map_location='cpu',weights_only=False);receipt=value['emg_initializer_lineage']
    if damage=='missing_tensor':value['state_dict'].pop(next(iter(value['state_dict'])))
    elif damage=='invented_tensor':value['state_dict']={'invented':torch.ones(1)}
    elif damage=='missing_lineage':receipt.pop('source_checkpoint_sha256')
    elif damage=='checkpoint_lineage':receipt['source_checkpoint_sha256']='0'*64
    elif damage=='fit_lineage':receipt['source_fit_identity_sha256']='0'*64
    elif damage=='relabel_full':receipt['full_initializer']=True
    receipt['tensor_inventory']=m._inventory(value['state_dict'])
    from egoemg_repro.emg_initializer_runtime import emg_initializer_state_sha256
    receipt['state_sha256']=emg_initializer_state_sha256(value['state_dict'])
    receipt['sha256']=_hash({k:v for k,v in receipt.items() if k!='sha256'});torch.save(value,dest)
    with pytest.raises(ValueError):m.read_emg_initializer_export(dest,source_fit=out,require_full=False)


@pytest.mark.parametrize('when',['after_checkpoint','after_journal'])
def test_three_process_recovery_preserves_stale_result_and_uses_latest_complete_epoch(native_runs,when):
    root,_,data,env,_=native_runs;m=api();out=root/('interrupted-'+when)
    base=[sys.executable,'-B','-m','egoemg_repro.emg_initializer_train','fit','--data',str(data),'--upstream',str(UPSTREAM),
        '--output',str(out),'--scope','cpu_engineering']
    prefix=subprocess.run(base+['--stop-after-epoch','1'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert prefix.returncode==0,prefix.stderr[-5000:]
    old=(out/'fit-result.json').read_bytes()
    code='''import runpy,sys
from pathlib import Path
import egoemg_repro.emg_initializer_train as m
when=sys.argv[1]
if when=='after_checkpoint':
 original=m._write
 def failed(path,value,**kw):
  if Path(path).name=='fit-result.json' and kw.get('replace'):raise RuntimeError('intended result replacement interruption')
  return original(path,value,**kw)
 m._write=failed
else:
 original=m.InitializerFitWitness.on_train_epoch_end
 def failed(self,trainer,module):
  result=original(self,trainer,module)
  if self.completed_epochs==2:raise RuntimeError('intended journal-before-checkpoint interruption')
  return result
 m.InitializerFitWitness.on_train_epoch_end=failed
sys.argv=['emg_initializer_train',*sys.argv[2:]]
m.main()
'''
    crashed=subprocess.run([sys.executable,'-B','-c',code,when,*base[4:],'--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    (root/('interrupted-'+when+'.stderr')).write_text(crashed.stderr)
    assert crashed.returncode!=0 and 'intended ' in crashed.stderr
    assert (out/'fit-result.json').read_bytes()==old
    checkpoint_sha=hashlib.sha256((out/'checkpoints/last.ckpt').read_bytes()).hexdigest()
    recovered=subprocess.run(base+['--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    (root/('recovered-'+when+'.stderr')).write_text(recovered.stderr)
    assert recovered.returncode==0,recovered.stderr[-6000:]
    final=m.read_emg_initializer_fit(out,require_full=False)
    assert final['completed_epochs']==2 and final['optimizer_executions']==4 and final['stage_complete']
    assert any(p.read_bytes()==old for p in out.glob('historical-fit-result-*.json'))
    if when=='after_checkpoint':
        assert final['completion_recovered_from_native_checkpoint'] is True
        assert hashlib.sha256((out/'checkpoints/last.ckpt').read_bytes()).hexdigest()==checkpoint_sha
    else:
        assert final['resume_start_exact']=={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')}
        assert final['resume_validation_exact'] and list(out.glob('uncommitted-epochs-*.jsonl'))


def test_completed_native_budget_still_cannot_be_retrained(native_runs):
    _,out,data,env,_=native_runs;before=(out/'fit-result.json').read_bytes()
    p=subprocess.run([sys.executable,'-B','-m','egoemg_repro.emg_initializer_train','fit','--data',str(data),
        '--upstream',str(UPSTREAM),'--output',str(out),'--scope','cpu_engineering','--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert p.returncode!=0 and 'completed initializer cannot be retrained' in p.stderr
    assert (out/'fit-result.json').read_bytes()==before


@pytest.mark.parametrize('damage',['history','last','best','coverage','selected_score'])
def test_reader_rejects_modified_native_training_outputs(native_runs,tmp_path,damage):
    from egoemg_repro.classic_emg import _hash
    _,out,_,_,_=native_runs;m=api()
    # Native callback paths are absolute: corrupt the owned original temporarily
    # and restore exact bytes, rather than silently rebasing a checkpoint.
    if damage in ('history','last','best'):
        result=json.loads((out/'fit-result.json').read_text())
        path=out/('epochs.jsonl' if damage=='history' else 'checkpoints/last.ckpt' if damage=='last' else 'checkpoints/'+result['selected_best']['file'])
        old=path.read_bytes()
        try:
            path.write_bytes(old+b'bad')
            with pytest.raises(ValueError):m.read_emg_initializer_fit(out,require_full=False)
        finally:path.write_bytes(old)
    else:
        path=out/'fit-result.json';old=path.read_bytes();value=json.loads(old)
        if damage=='coverage':value['train_samples_per_epoch'][-1]-=1
        else:value['selected_best']['score']+=.1
        value['sha256']=_hash({k:v for k,v in value.items() if k!='sha256'})
        try:
            path.write_text(json.dumps(value))
            with pytest.raises(ValueError):m.read_emg_initializer_fit(out,require_full=False)
        finally:path.write_bytes(old)
