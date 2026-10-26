"""Actual native CPU optimization and new-process restoration on real tiny files."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_vision_literal_data import sample_store,crop_store


ROOT=Path(__file__).parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()


class PrivateCpuRun:
    def __init__(self,*values):self.values=values
    def __iter__(self):return iter(self.values)
    def __repr__(self):return 'PrivateCpuRun(paths and child environment omitted)'


def api():
    try:return importlib.import_module('egoemg_repro.vision_literal_check')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.vision_literal_check':pytest.fail('native vision optimization/reload entry missing')
        raise


@pytest.fixture(scope='module')
def cpu_run(tmp_path_factory):
    pytest.importorskip('pytorch_lightning');pytest.importorskip('timm')
    api()
    cache=Path(os.environ.get('EGOEMG_VISION_TEST_HF_CACHE',ROOT/'tmp/vision-literal-hf/hub')).resolve()
    if not (cache/'models--timm--vit_small_patch14_dinov2.lvd142m/refs/main').is_file():
        if os.environ.get('EGOEMG_REQUIRE_VISION_PRETRAINED_TEST')=='1':pytest.fail('genuine DINO-S cache is required')
        pytest.skip('genuine DINO-S asset required; no random pretrained fixture')
    root=tmp_path_factory.mktemp('literal-vision-native-fit');sample_store(root/'data');crop_store(root/'crops',flat=True)
    output=root/'run'
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_CACHE=str(cache),HF_HOME=str(cache.parent),
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('HF_TOKEN',None);env.pop('HUGGING_FACE_HUB_TOKEN',None)
    base=[sys.executable,'-B','-m','egoemg_repro.vision_literal_check']
    options=['--data',str(root/'data'),'--crops',str(root/'crops'),'--upstream',str(UPSTREAM),
        '--output',str(output),'--device','cpu']
    for phase in ('prefix','resume'):
        result=subprocess.run(base+[phase]+options,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
        (root/(phase+'.stdout')).write_text(result.stdout);(root/(phase+'.stderr')).write_text(result.stderr)
        assert result.returncode==0,result.stderr[-6500:]
        if phase=='prefix':
            logs={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('metrics.csv')}
            assert logs
            (root/'prefix-log-snapshot.json').write_text(json.dumps(logs))
    return PrivateCpuRun(root,env,base,options)


def test_actual_native_fit_and_new_process_restore_preserve_full_optimizer_loop_rng(cpu_run):
    root,_,_,_=cpu_run
    prefix=json.loads((root/'run/prefix.json').read_text());resume=json.loads((root/'run/resume.json').read_text())
    assert prefix['actual_optimizer_updates']==2 and resume['actual_optimizer_updates']==3
    assert prefix['full_validation_samples']==4 and resume['full_validation_samples']==4
    assert prefix['validation_unique_hand_samples']==2
    assert prefix['gradient_finite'] is True and prefix['trained_weights_changed'] is True
    assert prefix['native_gradient_clip_val']==1 and prefix['precision']=='32-true'
    assert prefix['optimizer']['scheduler']['last_epoch']==1
    assert resume['optimizer']['scheduler']['last_epoch']==2
    assert resume['fresh_process_restore'] is True and resume['full_validation_prediction_equal'] is True
    assert prefix['process']!=resume['process'] and resume['restored_from_process']==prefix['process']
    assert resume['exact_restore']=={name:True for name in ('model','optimizer','scheduler','callbacks','loops','rng')}
    assert 'ModelCheckpoint.current_score' in resume['callback_restore_policy']
    assert resume['new_actual_updates']==1 and resume['cuda_initialized'] is False
    assert prefix['scientific_eligible'] is False and resume['scientific_eligible'] is False


def test_changed_checkpoint_rejects_before_creating_any_resume_trainer(cpu_run,tmp_path):
    root,env,base,options=cpu_run
    # A damaged copy has the original ownership receipt but different bytes.
    import shutil
    copy=tmp_path/'damaged';copy.mkdir()
    for name in ('prefix.json','engineering-final.ckpt','validation-reference.pt'):
        shutil.copy2(root/'run'/name,copy/name)
    with (copy/'engineering-final.ckpt').open('ab') as f:f.write(b'changed')
    damaged=list(options);damaged[damaged.index('--output')+1]=str(copy)
    result=subprocess.run(base+['resume']+damaged,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode!=0 and 'checkpoint bytes differ' in result.stderr
    assert not (copy/'resume.json').exists()


def test_engineering_entry_does_not_overwrite_existing_output(cpu_run):
    root,env,base,options=cpu_run
    before=(root/'run/prefix.json').read_bytes()
    result=subprocess.run(base+['prefix']+options,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode!=0 and (root/'run/prefix.json').read_bytes()==before


def test_fresh_resume_preserves_every_original_prefix_csv_log(cpu_run):
    root,_,_,_=cpu_run
    for name,digest in json.loads((root/'prefix-log-snapshot.json').read_text()).items():
        assert hashlib.sha256((root/'run'/name).read_bytes()).hexdigest()==digest


def test_resume_requires_an_actual_different_process_before_building_model(cpu_run,monkeypatch,tmp_path):
    root,_,_,_=cpu_run;m=api();prefix=json.loads((root/'run/prefix.json').read_text())
    assert 'process' in prefix,'fresh-process claim must be backed by actual PID/start identity'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(m,'_process_identity',lambda:prefix['process'],raising=False)
    def forbidden(*args,**kwargs):raise AssertionError('same-process resume must reject before model construction')
    monkeypatch.setattr(m,'build_vision_runtime',forbidden)
    output=tmp_path/'same-process';output.mkdir();(output/'prefix.json').write_text(json.dumps(prefix))
    with pytest.raises(ValueError,match='different process'):
        m.run_check('resume',data=root/'data',crops=root/'crops',upstream=UPSTREAM,
            output=output,device='cpu')
