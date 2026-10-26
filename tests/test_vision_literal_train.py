"""Original-budget policy, actual native epoch checkpoints and global-RNG resume."""
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

from test_vision_literal_data import sample_store,crop_store,config,UPSTREAM

ROOT=Path(__file__).parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.vision_literal_train')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.vision_literal_train':pytest.fail('full original vision fit runner missing')
        raise


def test_full_policy_keeps_original_budget_callbacks_bf16_and_all_360_batches():
    pytest.importorskip('hydra');p=api().vision_fit_policy(UPSTREAM,config(),scope='full')
    assert p['max_epochs']==150 and p['precision']=='bf16-mixed'
    assert p['batch_size_by_role']==dict(train=360,val=360,test=360)
    assert p['gradient_clip_val']==1 and p['scheduler_T_max']==150 and p['scheduler_eta_min']==1e-5
    assert p['monitor']=='val_mae' and p['monitor_mode']=='min' and p['patience']==30
    assert p['save_top_k']==3 and p['save_last'] is True and p['num_sanity_val_steps']==2
    assert p['engineering_train_samples'] is None and p['native_global_loader_rng'] is True
    assert p['scientific_eligible'] is False


@pytest.mark.parametrize('scope',[True,None,'full_cpu','smoke'])
def test_unknown_or_implicit_training_scope_rejects_before_runtime(scope):
    pytest.importorskip('hydra')
    with pytest.raises(ValueError):api().vision_fit_policy(UPSTREAM,config(),scope=scope)


class PrivateRun:
    def __init__(self,*values):self.values=values
    def __iter__(self):return iter(self.values)
    def __repr__(self):return 'PrivateRun(paths and child environment omitted)'


@pytest.fixture(scope='module')
def native_runs(tmp_path_factory):
    pytest.importorskip('pytorch_lightning');pytest.importorskip('timm');api()
    cache=Path(os.environ.get('EGOEMG_VISION_TEST_HF_CACHE',ROOT/'tmp/vision-literal-hf/hub')).resolve()
    if not (cache/'models--timm--vit_small_patch14_dinov2.lvd142m/refs/main').is_file():
        if os.environ.get('EGOEMG_REQUIRE_VISION_PRETRAINED_TEST')=='1':pytest.fail('actual DINO-S asset required')
        pytest.skip('genuine pretrained native fit requires actual DINO-S cache')
    root=tmp_path_factory.mktemp('native-vision-full-fit');data=root/'data';crops=root/'crops'
    sample_store(data,n=9*7790);crop_store(crops,flat=True)
    np.repeat([0,1,2,3],[6*7790,7790,7790,7790]).astype('int8').tofile(data/'frame_split_id.dat')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_HUB_CACHE=str(cache),HF_HOME=str(cache.parent),
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('HF_TOKEN',None);env.pop('HUGGING_FACE_HUB_TOKEN',None)
    base=[sys.executable,'-B','-m','egoemg_repro.vision_literal_train','fit',
        '--data',str(data),'--crops',str(crops),'--upstream',str(UPSTREAM),'--scope','cpu_engineering']
    probe='''import hashlib,json,runpy,sys,torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
trace_path=sys.argv[1];original=_BaseDataLoaderIter.__init__
def rng():return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
def observe(self,loader):
 before=rng();out=original(self,loader);ds=loader.dataset
 while hasattr(ds,'dataset'):ds=ds.dataset
 with open(trace_path,'a') as f:f.write(json.dumps(dict(role=ds.role,before=before,after=rng(),seed=self._base_seed))+'\\n')
 return out
_BaseDataLoaderIter.__init__=observe
sys.argv=['egoemg_repro.vision_literal_train',*sys.argv[2:]]
runpy.run_module('egoemg_repro.vision_literal_train',run_name='__main__')
'''
    phases=[('control',root/'control',[]),('prefix',root/'resumed',['--stop-after-epoch','1']),
        ('resume',root/'resumed',['--resume'])]
    prefix_logs=None
    for label,output,extra in phases:
        command=[sys.executable,'-B','-c',probe,str(root/(label+'.iterators.jsonl')),*base[4:],'--output',str(output),*extra]
        result=subprocess.run(command,cwd=ROOT,env=env,
            capture_output=True,text=True,timeout=240)
        (root/(label+'.stdout')).write_text(result.stdout);(root/(label+'.stderr')).write_text(result.stderr)
        assert result.returncode==0,result.stderr[-7000:]
        if label=='prefix':
            prefix_logs={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('metrics.csv')}
            (root/'prefix-result.json').write_bytes((output/'fit-result.json').read_bytes())
    return PrivateRun(root,env,base,prefix_logs)


def test_real_native_epoch_checkpoint_resume_and_complete_validation(native_runs):
    root,_,_,logs=native_runs;m=api()
    control=m.read_vision_fit(root/'control',require_full=False)
    resumed=m.read_vision_fit(root/'resumed',require_full=False)
    prefix=json.loads((root/'prefix-result.json').read_text())
    assert prefix['completed_epochs']==1 and prefix['optimizer_executions']==2 and prefix['stage_complete'] is False
    assert resumed['completed_epochs']==2 and resumed['optimizer_executions']==4 and resumed['stage_complete'] is True
    assert resumed['resume_start_exact']=={k:True for k in ('model','optimizer','scheduler','callbacks','loops','rng')}
    assert resumed['bitwise_training_trajectory_guaranteed'] is False
    c_records,_=m._read_epochs(root/'control/epochs.jsonl',m._json(root/'control/identity.json')['sha256'])
    r_records,_=m._read_epochs(root/'resumed/epochs.jsonl',m._json(root/'resumed/identity.json')['sha256'])
    assert c_records[0]['model_sha256']==r_records[0]['model_sha256']
    traces={name:[r for r in map(json.loads,(root/(name+'.iterators.jsonl')).read_text().splitlines()) if r['role']=='train']
        for name in ('control','prefix','resume')}
    assert len(traces['control'])==2 and len(traces['prefix'])==1 and len(traces['resume'])==2
    assert traces['control'][1]['before']==traces['resume'][0]['before']
    assert traces['control'][1]['after']==traces['resume'][0]['after']
    assert traces['resume'][0]['seed']!=traces['resume'][1]['seed']
    assert 'torch.utils.data.dataloader' in control['native_loop_sources_sha256']
    assert resumed['val_samples_per_epoch']==[4,4] and resumed['train_samples_per_epoch']==[4,4]
    assert resumed['full_training_terminated'] is False and resumed['scientific_eligible'] is False
    assert resumed['selected_best']['file']!='last.ckpt' and resumed['selected_best']['validated'] is True
    assert resumed['cuda_initialized'] is False
    for name,digest in logs.items():assert hashlib.sha256((root/'resumed'/name).read_bytes()).hexdigest()==digest


def test_engineering_fit_cannot_be_read_as_complete_full_training(native_runs):
    root,_,_,_=native_runs
    with pytest.raises(ValueError,match='full|engineering'):api().read_vision_fit(root/'control',require_full=True)


@pytest.mark.parametrize('damage',['steps','best_file','best_score','scientific','scope','epoch_type'])
def test_completed_metadata_claims_are_checked_against_native_checkpoint_and_epoch_carry(native_runs,damage):
    from egoemg_repro.classic_emg import _hash
    root,_,_,_=native_runs;path=root/'control/fit-result.json';before=path.read_bytes();d=json.loads(before)
    if damage=='steps':d['optimizer_executions']-=1
    elif damage=='best_file':d['selected_best']['file']='last.ckpt'
    elif damage=='best_score':d['selected_best']['score']=0.
    elif damage=='scientific':d['scientific_eligible']=True
    elif damage=='scope':d['scope']='full'
    elif damage=='epoch_type':d['completed_epochs']=float(d['completed_epochs'])
    d['sha256']=_hash({k:v for k,v in d.items() if k!='sha256'})
    try:
        path.write_text(json.dumps(d))
        with pytest.raises(ValueError):api().read_vision_fit(root/'control',require_full=False)
    finally:path.write_bytes(before)


def test_small_fixture_rejects_full_fit_before_cuda_model_or_output_creation(native_runs,monkeypatch,tmp_path):
    from egoemg_repro.vision_literal_cohort import admit_vision_cohort
    root,_,_,_=native_runs;m=api()
    c=admit_vision_cohort(root/'data',root/'crops',upstream=UPSTREAM,config=config())
    def forbidden(*args,**kwargs):raise AssertionError('unregistered tiny cohort must reject before model')
    monkeypatch.setattr(m,'build_vision_runtime',forbidden);output=tmp_path/'should-not-exist'
    with pytest.raises(ValueError,match='registered|complete legacy'):m.fit_vision(c,output,scope='full')
    assert not output.exists()
