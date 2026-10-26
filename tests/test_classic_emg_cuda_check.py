"""CPU rejection/schema checks; real CUDA positives are individually gated."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def runtime():
    try:
        return importlib.import_module('egoemg_repro.classic_emg_cuda_check')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.classic_emg_cuda_check':
            pytest.fail('independent CUDA qualifier interface is not implemented')
        raise


@pytest.mark.parametrize('variant', ['vemg2pose', 'neuropose'])
def test_prefix_resume_share_one_bounded_identity_policy(variant):
    r = runtime()
    a = r.cuda_check_policy(variant=variant, device='cuda:0', phase='prefix')
    b = r.cuda_check_policy(variant=variant, device='cuda:0', phase='resume')
    assert a == b
    assert a['scope'] == 'cuda_qualification' and not a['production_ready'] and not a['scientific_eligible']
    assert (a['batch_size'], a['num_workers'], a['planned_epochs'], a['train_samples'], a['val_samples']) == (1,0,2,4,2)
    assert a['precision'] == '16-mixed' and a['max_attempts'] == 8


def test_canonical_backend_policy_limits_resume_claim_to_typed_full_state():
    policy=runtime().cuda_check_policy(variant='neuropose')
    assert policy['runtime_profile']=='canonical_backend_nondeterministic_v1'
    assert policy['deterministic'] is False and policy['cudnn_deterministic'] is False
    assert policy['cudnn_benchmark'] is False and policy['trainer_deterministic'] is False
    assert policy['matmul_precision']=='high' and policy['cublas_workspace_config']==':4096:8'
    assert policy['trajectory_agreement_qualified'] is False and policy['bitwise_trajectory_guaranteed'] is False
    assert policy['resume_scope']=='exact typed full checkpoint and RNG state restoration only; no bitwise trajectory guarantee'


def profile_identity_metadata(r):
    """Schema-only identity metadata, not CUDA execution or a pass receipt."""
    pl=pytest.importorskip('pytorch_lightning');project=Path(__file__).parents[1]
    value=dict(format=r.FORMAT,scope='cuda_qualification',recipe={'variant':'neuropose'},
        policy=r.cuda_check_policy(variant='neuropose'),
        runtime_files_sha256={name:r.file_sha256(project/name) for name in r.RUNTIME_FILES},
        framework_files_sha256={name:r.file_sha256(Path(pl.__file__).resolve().parent/name) for name in r.storage.FRAMEWORK_FILES},
        runtime=dict(runtime_profile='canonical_backend_nondeterministic_v1',deterministic=False,cudnn_benchmark=False,
            cudnn_deterministic=False,trainer_deterministic=False,matmul_precision='high',cublas_workspace_config=':4096:8',
            trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False,
            resume_scope='exact typed full checkpoint and RNG state restoration only; no bitwise trajectory guarantee',
            precision_plugin='MixedPrecision',scaler_enabled=True))
    value['sha256']=r.storage._json_hash(value)
    return value


@pytest.mark.parametrize('damage',['profile','algorithms','cudnn','benchmark','trainer','matmul','workspace',
                                 'trajectory','bitwise','restore_scope','flag_dtype','policy_flag_dtype'])
def test_identity_rejects_backend_policy_mismatch_even_when_resealed(damage):
    r=runtime();value=profile_identity_metadata(r);r._check_identity(value)
    key={'profile':'runtime_profile','algorithms':'deterministic','cudnn':'cudnn_deterministic',
         'benchmark':'cudnn_benchmark','trainer':'trainer_deterministic','matmul':'matmul_precision',
         'workspace':'cublas_workspace_config','trajectory':'trajectory_agreement_qualified',
         'bitwise':'bitwise_trajectory_guaranteed','restore_scope':'resume_scope',
         'flag_dtype':'deterministic','policy_flag_dtype':'deterministic'}[damage]
    if damage=='policy_flag_dtype': value['policy'][key]=0
    elif damage=='flag_dtype': value['runtime'][key]=0
    else:
        value['runtime'][key]={'profile':'strict_v1','matmul':'highest','workspace':':16:8',
                              'restore_scope':'bitwise trajectory'}[damage] if damage in ('profile','matmul','workspace','restore_scope') else True
    value['sha256']=r.storage._json_hash({k:v for k,v in value.items() if k!='sha256'})
    with pytest.raises(ValueError,match='profile|backend|runtime|policy|identity'):
        r._check_identity(value)


@pytest.mark.parametrize('variant,device,phase', [('bad','cuda:0','prefix'),('neuropose','cpu','prefix'),
    ('neuropose','bf16','prefix'),('neuropose','cuda:1','prefix'),('neuropose','cuda:0','formal')])
def test_unknown_or_cpu_policy_cannot_claim_cuda_qualification(variant, device, phase):
    with pytest.raises(ValueError):
        runtime().cuda_check_policy(variant=variant,device=device,phase=phase)


def test_no_cuda_fails_before_data_or_factory(tmp_path,monkeypatch):
    torch = pytest.importorskip('torch')
    if torch.cuda.is_available():
        pytest.skip('this negative case is for the CPU tier')
    r=runtime()
    # The hardware boundary is independent of CI host capacity; reserve has its own case.
    monkeypatch.setattr(r.storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    project=Path(__file__).parents[1]
    with pytest.raises(RuntimeError, match='CUDA'):
        r.run_cuda_check(tmp_path/'absent',upstream=project/'upstream',config=project/'configs/neuropose.json',
            partitions={},output=tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_phase_resume_contract_rejects_before_cuda(tmp_path):
    with pytest.raises(ValueError, match='resume'):
        runtime().run_cuda_check(tmp_path/'absent',upstream=tmp_path/'absent',config=tmp_path/'absent',
            partitions={},output=tmp_path/'run',phase='resume')
    with pytest.raises(ValueError, match='prefix'):
        runtime().run_cuda_check(tmp_path/'absent',upstream=tmp_path/'absent',config=tmp_path/'absent',
            partitions={},output=tmp_path/'run',resume=tmp_path/'old')


@pytest.mark.parametrize('damage',['variant','override'])
def test_invalid_config_is_rejected_before_cuda_profile(tmp_path,monkeypatch,damage):
    if damage=='override': pytest.importorskip('hydra')
    r=runtime();project=Path(__file__).parents[1]
    value=json.loads((project/'configs/neuropose.json').read_text())
    if damage=='variant': value['variant']='unknown-classic-variant'
    else: value['lr']*=2
    config=tmp_path/'rejected.json';config.write_text(json.dumps(value))
    def forbidden_profile(): pytest.fail('invalid recipe reached CUDA context/scaler allocation')
    monkeypatch.setattr(r,'_profile',forbidden_profile)
    monkeypatch.setattr(r.storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    with pytest.raises(ValueError):
        r.run_cuda_check(tmp_path/'absent',upstream=project/'upstream',config=config,
            partitions={},output=tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_prepare_rechecks_pre_cuda_recipe_before_data_or_factory(tmp_path):
    pytest.importorskip('hydra');r=runtime();project=Path(__file__).parents[1]
    expected=json.loads((project/'configs/neuropose.json').read_text());expected['lr']*=2
    # Structural metadata rejection only; no CUDA profile, context or positive receipt.
    with pytest.raises(ValueError,match='changed after pre-CUDA'):
        r._prepare(tmp_path/'absent',project/'upstream',project/'configs/neuropose.json',{},
            {},expected_recipe=expected)


def test_nonfinite_gradient_observation_is_json_native_without_repair():
    torch = pytest.importorskip('torch')
    gradient = torch.tensor([1., float('inf')])
    before = gradient.clone()
    result = runtime().gradient_observation([gradient])
    assert result == {'finite':False,'absolute_max':None}
    json.dumps(result, allow_nan=False)
    assert torch.equal(before,gradient)
    assert runtime().gradient_observation([torch.tensor([2.,-3.])]) == {'finite':True,'absolute_max':3.}


def schema_fixture():
    """CPU structural-negative fixture; never an actual GPU qualification."""
    torch = pytest.importorskip('torch')
    r = runtime()
    rng = dict(python=__import__('random').getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=[torch.get_rng_state()],generators={k:torch.get_rng_state() for k in ('train_loader','train_sampler','val_loader')})
    state = dict(epoch=0,global_step=4,state_dict={'model.weight':torch.ones(2,2)},
        optimizer_states=[dict(state={0:dict(step=torch.tensor(3.),exp_avg=torch.ones(2,2),exp_avg_sq=torch.ones(2,2))},
                               param_groups=[dict(params=[0])])],
        lr_schedulers=[dict(T_max=100,last_epoch=1)],
        callbacks={'CudaProtocol':dict(events=[{}]*4,executions=3,injection_count=1,completed_epochs=1,
                       validation_events=[dict(epoch=0,samples=2,full_fixed_validation=True,
                           full_cohort_validation=False,val_loss=.1,batch_weighted_loss=.1,scientific_eligible=False)]),
                   "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}":
                       dict(dirpath='/old/checkpoints',last_model_path='/old/checkpoints/last.ckpt')},
        loops={'fit_loop':dict(epoch_progress=dict(current=dict(ready=1,started=1,processed=1,completed=0)),
            **{'epoch_loop.batch_progress':dict(current=dict(ready=4,started=4,processed=4,completed=4),is_last_batch=True),
               'epoch_loop.val_loop.batch_progress':dict(current=dict(ready=2,started=2,processed=2,completed=2),is_last_batch=True)})},
        MixedPrecision=dict(scale=32768.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=0))
    state['cuda_check'] = dict(format=r.FORMAT,scope='cuda_qualification',identity={'negative_fixture':True},
        role='qualification_last',completed_epochs=1,epoch_boundary=True,rng=rng,
        model_schema={'model.weight':dict(dtype='torch.float32',shape=[2,2])},
        optimizer_schema=r.optimizer_schema(state['optimizer_states']),scaler_schema=r.scaler_schema(state['MixedPrecision']))
    loop = state['loops']['fit_loop']
    loop['epoch_progress']['total'] = dict(loop['epoch_progress']['current'])
    for key in r.LOOP_KEYS - set(loop):
        loop[key] = {}
    return state


def test_structural_fixture_is_not_a_cuda_qualification_receipt(tmp_path):
    pytest.importorskip('torch')
    r = runtime(); value = schema_fixture()
    # Structural validation is not hardware execution or a pass receipt.
    assert r.validate_checkpoint_state(value, value['cuda_check']['identity']) is None
    with pytest.raises((ValueError,FileNotFoundError)):
        r.verify_cuda_check(tmp_path/'missing-prefix',tmp_path/'missing-resume')


@pytest.mark.parametrize('damage', ['format','scope','identity','epoch','partial','model_key','dtype','shape','nonfinite',
    'optimizer','scheduler','scaler','scaler_key','callbacks','cuda_rng','rng_tuple','loop_key','optimizer_moment',
    'optimizer_step','scheduler_step','scaler_dtype','callback_path','callback_field','optimizer_counter',
    'scheduler_dtype','globalstep_dtype','numpy_array'])
def test_malformed_full_state_is_rejected_without_gpu_acceptance(damage):
    torch = pytest.importorskip('torch')
    r = runtime(); value = schema_fixture(); expected_identity = copy.deepcopy(value['cuda_check']['identity'])
    if damage == 'format': value['cuda_check']['format'] = 'classic_literal_lightning_epoch_v1'
    elif damage == 'scope': value['cuda_check']['scope'] = 'formal'
    elif damage == 'identity': value['cuda_check']['identity'] = {'other':True}
    elif damage == 'epoch': value['loops']['fit_loop']['epoch_progress']['current']['completed'] = 1
    elif damage == 'partial': value['cuda_check']['epoch_boundary'] = False
    elif damage == 'model_key': value['state_dict']['unexpected'] = value['state_dict'].pop('model.weight')
    elif damage == 'dtype': value['state_dict']['model.weight'] = value['state_dict']['model.weight'].half()
    elif damage == 'shape': value['state_dict']['model.weight'] = torch.ones(1)
    elif damage == 'nonfinite': value['state_dict']['model.weight'][0,0] = float('nan')
    elif damage == 'optimizer': value['optimizer_states'][0]['state'] = {}
    elif damage == 'scheduler': value['lr_schedulers'] = []
    elif damage == 'scaler': value['MixedPrecision'] = {}
    elif damage == 'scaler_key': del value['MixedPrecision']['_growth_tracker']
    elif damage == 'callbacks': del value['callbacks']['CudaProtocol']
    elif damage == 'cuda_rng': value['cuda_check']['rng']['cuda'] = []
    elif damage == 'rng_tuple': value['cuda_check']['rng']['numpy'] = list(value['cuda_check']['rng']['numpy'])
    elif damage == 'loop_key': del value['loops']['fit_loop']['epoch_loop.scheduler_progress']
    elif damage == 'optimizer_moment': value['optimizer_states'][0]['state'][0]['exp_avg_sq'] = torch.ones(1)
    elif damage == 'optimizer_step': value['optimizer_states'][0]['state'][0]['step'] = torch.tensor(float('inf'))
    elif damage == 'scheduler_step': value['lr_schedulers'][0]['last_epoch'] = 0
    elif damage == 'scaler_dtype': value['MixedPrecision']['_growth_tracker'] = 0.
    elif damage == 'callback_path': next(v for k,v in value['callbacks'].items() if k.startswith('ModelCheckpoint{'))['last_model_path']='/elsewhere/last.ckpt'
    elif damage == 'callback_field': del value['callbacks']['CudaProtocol']['validation_events']
    elif damage == 'optimizer_counter': value['callbacks']['CudaProtocol']['executions']=2
    elif damage == 'scheduler_dtype': value['lr_schedulers'][0]['last_epoch']=1.
    elif damage == 'globalstep_dtype': value['global_step']=4.
    elif damage == 'numpy_array':
        rng=value['cuda_check']['rng']['numpy'];value['cuda_check']['rng']['numpy']=(rng[0],rng[1].tolist(),*rng[2:])
    with pytest.raises(ValueError):
        r.validate_checkpoint_state(value, expected_identity)


def test_truncated_or_cpu_format_checkpoint_never_resumes(tmp_path):
    pytest.importorskip('torch')
    r = runtime()
    bad = tmp_path/'truncated.ckpt'; bad.write_bytes(b'not a complete checkpoint')
    with pytest.raises(ValueError, match='truncated|unreadable'):
        r.read_cuda_checkpoint(bad, {'sha256':'untrusted'})


def test_low_capacity_does_not_create_output(tmp_path,monkeypatch):
    r = runtime()
    monkeypatch.setattr(r.storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=1024**3))
    with pytest.raises(OSError,match='15GiB'):
        r.run_cuda_check(tmp_path/'absent',upstream=tmp_path/'absent',config=tmp_path/'absent',
            partitions={},output=tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_final_status_fsync_fault_invalidates_owned_success_without_low_space_write(tmp_path,monkeypatch):
    """Artifact-layer fault, not a simulated CUDA execution."""
    r=runtime(); output=tmp_path/'owned';output.mkdir()
    (output/'checkpoint').write_bytes(b'previous valid checkpoint bytes')
    neighbor=tmp_path/'neighbor';neighbor.write_bytes(b'untouched')
    original=r.storage._fsync_dir
    def fsync(path):
        status=output/'status.json'
        if status.exists() and json.loads(status.read_text()).get('state')=='qualification_phase_finished':
            monkeypatch.setattr(r.storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=1024**3))
            raise OSError('injected post-rename ENOSPC')
        original(path)
    monkeypatch.setattr(r.storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(r.storage,'_fsync_dir',fsync)
    result=dict(scope='cuda_qualification',scientific_eligible=False,production_ready=False)
    with r._capture_owned_output(output) as owned:
        with pytest.raises(OSError,match='ENOSPC'):
            try: r._publish_result(output,result)
            except OSError as exc:
                r._fail_owned_output(output,exc,owned=owned)
                raise
    assert not (output/'status.json').exists()
    assert (output/'result.json').exists()
    assert (output/'checkpoint').read_bytes()==b'previous valid checkpoint bytes'
    assert neighbor.read_bytes()==b'untouched'


def test_checkpoint_dynamic_capacity_keeps_previous_bytes(tmp_path,monkeypatch):
    pytest.importorskip('torch');r=runtime();value=schema_fixture()
    checkpoint=tmp_path/'last.ckpt';checkpoint.write_bytes(b'earlier checkpoint')
    count=[]
    def free(path):
        count.append(True)
        return SimpleNamespace(free=(100 if len(count)<3 else 1)*1024**3)
    monkeypatch.setattr(r.storage.shutil,'disk_usage',free)
    io=r.CudaCheckpointIO(tmp_path,value['cuda_check']['identity'])
    with pytest.raises(OSError,match='reserve'):
        io.save_checkpoint(value,checkpoint)
    assert checkpoint.read_bytes()==b'earlier checkpoint'
    assert not list(tmp_path.glob('*.incomplete'))


@pytest.mark.parametrize('replacement',['symlink','directory','ancestor_symlink'])
def test_failure_cleanup_refuses_rebound_owned_directory(tmp_path,monkeypatch,replacement):
    """Real directory replacement; no mocked CUDA or model work."""
    r=runtime();parent=tmp_path/'parent';parent.mkdir();output=parent/'owned';output.mkdir()
    (output/'status.json').write_bytes(b'original owned running')
    (output/'checkpoint').write_bytes(b'original owned checkpoint')
    with r._capture_owned_output(output) as owned:
        if replacement=='ancestor_symlink':
            parent.rename(tmp_path/'old-parent')
            neighbor=tmp_path/'neighbor-parent';neighbor.mkdir();target=neighbor/'owned';target.mkdir()
            parent.symlink_to(neighbor,target_is_directory=True);original=tmp_path/'old-parent/owned'
        else:
            original=parent/'old-owned';output.rename(original)
            if replacement=='symlink':
                target=tmp_path/'neighbor';target.mkdir();output.symlink_to(target,target_is_directory=True)
            else:
                output.mkdir();target=output
        (target/'status.json').write_bytes(b'neighbor valid success')
        (target/'checkpoint').write_bytes(b'neighbor valid checkpoint')
        r._fail_owned_output(output,OSError('failure after namespace replacement'),owned=owned)
        assert (target/'status.json').read_bytes()==b'neighbor valid success'
        assert (target/'checkpoint').read_bytes()==b'neighbor valid checkpoint'
        assert (original/'status.json').read_bytes()==b'original owned running'
        assert (original/'checkpoint').read_bytes()==b'original owned checkpoint'


def test_cleanup_after_guard_rebind_uses_captured_dirfd_not_neighbor_path(tmp_path,monkeypatch):
    r=runtime();output=tmp_path/'owned';output.mkdir();neighbor=tmp_path/'neighbor';neighbor.mkdir()
    (output/'status.json').write_bytes(b'owned running')
    (output/'checkpoint').write_bytes(b'owned checkpoint')
    (neighbor/'status.json').write_bytes(b'neighbor valid success')
    real_unlink=r.os.unlink;changed=[]
    with r._capture_owned_output(output) as owned:
        def rebind(name,*args,**kwargs):
            if name=='status.json' and kwargs.get('dir_fd')==owned.fd and not changed:
                changed.append(True);output.rename(tmp_path/'old-owned');output.symlink_to(neighbor,target_is_directory=True)
            return real_unlink(name,*args,**kwargs)
        monkeypatch.setattr(r.os,'unlink',rebind)
        r._fail_owned_output(output,OSError('injected after-guard replacement'),owned=owned)
    assert changed
    assert (neighbor/'status.json').read_bytes()==b'neighbor valid success'
    assert (tmp_path/'old-owned/checkpoint').read_bytes()==b'owned checkpoint'
    assert not (tmp_path/'old-owned/status.json').exists()


def test_failed_status_write_rechecks_owned_identity_after_unlink(tmp_path,monkeypatch):
    r=runtime();output=tmp_path/'owned';output.mkdir();(output/'status.json').write_bytes(b'owned running')
    real_unlink=r.os.unlink;changed=[]
    with r._capture_owned_output(output) as owned:
        def rebind(name,*args,**kwargs):
            result=real_unlink(name,*args,**kwargs)
            if name=='status.json' and kwargs.get('dir_fd')==owned.fd and not changed:
                changed.append(True);output.rename(tmp_path/'old-owned');output.mkdir()
                (output/'status.json').write_bytes(b'other run valid success')
            return result
        monkeypatch.setattr(r.os,'unlink',rebind)
        r._fail_owned_output(output,OSError('injected before failed-status write'),owned=owned)
    assert changed and (output/'status.json').read_bytes()==b'other run valid success'
    assert not (tmp_path/'old-owned/status.json').exists()


def test_owned_failure_writes_atomic_private_status_with_original_directory_capability(tmp_path,monkeypatch):
    r=runtime();output=tmp_path/'owned';output.mkdir()
    (output/'status.json').write_bytes(b'owned prior marker')
    (output/'checkpoint').write_bytes(b'checkpoint retained')
    monkeypatch.setattr(r.storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    with r._capture_owned_output(output) as owned:
        r._fail_owned_output(output,OSError('deliberate failure'),owned=owned)
    status=json.loads((output/'status.json').read_text())
    assert status['state']=='failed' and status['scientific_eligible'] is False and status['production_ready'] is False
    assert (output/'status.json').stat().st_mode&0o777==0o600
    assert (output/'checkpoint').read_bytes()==b'checkpoint retained'
    assert not list(output.glob('*.incomplete'))


@pytest.mark.parametrize('variant', ['vemg2pose','neuropose'])
def test_actual_cuda_prefix_resume_and_scaler_evidence(tmp_path,variant):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires actual root-selected CUDA device; no simulated positive')
    # GPU integration consumes explicit real, preflighted server paths only.
    names = ('EGOEMG_CLASSIC_DATA','EGOEMG_CLASSIC_PARTITIONS','EGOEMG_UPSTREAM')
    if not all(os.environ.get(name) for name in names):
        pytest.skip('root must explicitly provide real corpus/partition/upstream')
    r = runtime(); project=Path(__file__).parents[1]
    args=dict(upstream=Path(os.environ[names[2]]),config=project/'configs'/f'{variant}.json',
              partitions=json.loads(Path(os.environ[names[1]]).read_text()))
    data=Path(os.environ[names[0]])
    common = ['--data',str(data),'--upstream',str(args['upstream']),'--config',str(args['config']),
              '--partitions',os.environ[names[1]]]
    # Each actual phase has a fresh interpreter/CUDA context, never an in-memory reload.
    for phase in ('prefix','resume'):
        output=tmp_path/('prefix' if phase=='prefix' else 'resumed')
        command=[sys.executable,'-B','-m','egoemg_repro.classic_emg_cuda_check',phase,*common,'--output',str(output)]
        if phase=='resume': command += ['--checkpoint',str(tmp_path/'prefix/checkpoints/last.ckpt')]
        with (tmp_path/(phase+'.log')).open('wb') as log:
            subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assert json.loads((tmp_path/'prefix/result.json').read_text())['scientific_eligible'] is False
    receipt=r.verify_cuda_check(tmp_path/'prefix',tmp_path/'resumed')
    assert receipt['cuda_numeric_passed'] and not receipt['production_ready']
    assert receipt['attempts'] == 8 and receipt['optimizer_executions'] >= 2
    assert receipt['injected_skips'] == 1 and receipt['strict_resume']
    assert receipt['strict_state_restore'] is True and receipt['trajectory_agreement_qualified'] is False
    assert receipt['bitwise_trajectory_guaranteed'] is False
    assert receipt['runtime_profile']=='canonical_backend_nondeterministic_v1'
