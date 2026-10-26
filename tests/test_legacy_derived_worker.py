"""Bounded worker contracts; no real trained weights, released inference or GPU."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_legacy_derived import admitted_fixture


def api():
    try:
        return importlib.import_module('tools.legacy_derived.worker')
    except ModuleNotFoundError as exc:
        if exc.name=='tools.legacy_derived.worker':pytest.fail('isolated legacy worker is not implemented')
        raise


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def inputs(admitted_fixture,tmp_path,monkeypatch):
    w=api();p=w.preflight
    data=tmp_path/'data';data.mkdir();crops=tmp_path/'crops';crops.mkdir()
    (data/'manifest.json').write_text('{}');(crops/'receipt.json').write_text('{}')
    run=admitted_fixture['runs'][0]
    for name in ('last.pt','best.pt'):(Path(run['run'])/name).write_bytes(('synthetic-'+name).encode())
    (Path(run['run'])/'progress.jsonl').write_text('{}\n')
    monkeypatch.setattr(w,'CHECKPOINTS',{**w.CHECKPOINTS,run['model']:{name:sha(Path(run['run'])/name) for name in ('last.pt','best.pt')}})
    spec=dict(format=w.FORMAT,run_spec=copy.deepcopy(run),legacy_root=admitted_fixture['legacy_root'],
              upstream_root=admitted_fixture['upstream_root'],data=str(data),crops=str(crops),
              output=str(tmp_path/'new-output'),device='cuda:0')
    cohort=dict(format=w.COHORT_FORMAT,partition_sha256=p.REFERENCE['partition_sha256'],
        partition_raw_sha256=p.RAW_FILES[run['model']]['partitions.json'],window_ordinals=[0,1])
    return spec,cohort


@pytest.mark.parametrize('damage',['model','spec_extra','batch','precision','cpu','selection','cohort_extra','rawcanonical','formal','output_inside_input'])
def test_contract_rejects_before_tensor_data_or_child(inputs,monkeypatch,damage):
    w=api();spec,cohort=copy.deepcopy(inputs);bounded=True
    if damage=='model':spec['run_spec']['model']='unknown'
    elif damage=='spec_extra':spec['max_samples']=1
    elif damage=='batch':spec['batch_size']=1
    elif damage=='precision':spec['precision']='float32'
    elif damage=='cpu':spec['device']='cpu'
    elif damage=='selection':cohort['window_ordinals']=[1,2]
    elif damage=='cohort_extra':cohort['search_for_finite']=True
    elif damage=='rawcanonical':cohort['partition_raw_sha256']=cohort['partition_sha256']
    elif damage=='formal':bounded=False
    else:spec['output']=str(Path(spec['data'])/'must-not-write')
    monkeypatch.setattr(w,'_launch',lambda *_:pytest.fail('rejected contract launched a worker'))
    with pytest.raises(ValueError):w.run_model_pass(spec,cohort,bounded=bounded)
    assert not Path(spec['output']).exists()


@pytest.mark.parametrize('damage',['source','selected','initialization','rawrecipe','last','best'])
def test_admission_rejects_changed_inputs_before_tensor_or_child(inputs,monkeypatch,damage):
    w=api();spec,cohort=copy.deepcopy(inputs);run=Path(spec['run_spec']['run'])
    if damage=='source':(Path(spec['legacy_root'])/'egoemg_repro/train.py').write_text('changed source\n')
    else:
        name={'selected':'summary.json','initialization':'provenance.json','rawrecipe':'recipe_source.json','last':'last.pt','best':'best.pt'}[damage]
        (run/name).write_bytes((run/name).read_bytes()+b' ')
    monkeypatch.setattr(w,'_launch',lambda *_:pytest.fail('changed inputs reached child'))
    with pytest.raises(ValueError):w.run_model_pass(spec,cohort,bounded=True)
    assert not Path(spec['output']).exists()


def test_metadata_admission_preserves_raw_and_canonical_separate(inputs):
    w=api();spec,cohort=inputs;admitted=w._admit_inputs(spec,cohort,bounded=True)
    assert admitted['scope']=='bounded_engineering' and admitted['benchmark_eligible'] is False
    assert admitted['cohort']['window_ordinals']==[0,1]
    assert admitted['cohort']['partition_raw_sha256']!=admitted['cohort']['partition_sha256']
    assert admitted['checkpoint_sha256']==w.CHECKPOINTS[spec['run_spec']['model']]['last.pt']
    assert admitted['recipe_raw_sha256']==w.preflight.BASELINES[spec['run_spec']['model']]['recipe_sha256']


def test_snapshot_rejects_same_bytes_replacement_and_symlink(tmp_path):
    w=api();path=tmp_path/'source.py';path.write_text('original\n');snapshot=w.InputSnapshot([path])
    before=path.stat();raw=path.read_bytes();path.unlink();path.write_bytes(raw)
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError):snapshot.check()
    snapshot=w.InputSnapshot([path]);other=tmp_path/'other';other.write_bytes(raw);path.unlink();path.symlink_to(other)
    with pytest.raises(ValueError):snapshot.check()


def test_checkpoint_whole_hash_precedes_real_tensor_loader(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');w=api();path=tmp_path/'last.pt';torch.save({'model':{'x':torch.ones(2)}},path)
    called=[];original=torch.load
    def load(*a,**k):called.append(True);return original(*a,**k)
    monkeypatch.setattr(torch,'load',load)
    with pytest.raises(ValueError):w._load_checkpoint(path,'0'*64,torch)
    assert not called
    result=w._load_checkpoint(path,sha(path),torch)
    assert called and torch.equal(result['model']['x'],torch.ones(2))


@pytest.mark.parametrize('damage',['key','shape','dtype','nan'])
def test_typed_strict_load_refuses_implicit_cast_and_partial_state(damage):
    torch=pytest.importorskip('torch');w=api();model=torch.nn.Linear(2,2);state=copy.deepcopy(model.state_dict())
    if damage=='key':state['unexpected']=state.pop('weight')
    elif damage=='shape':state['weight']=state['weight'][:1]
    elif damage=='dtype':state['weight']=state['weight'].half()
    else:state['weight'][0,0]=float('nan')
    before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError):w.strict_load_state(model,state)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_typed_load_records_length_prefixed_dtype_shape_and_bytes():
    torch=pytest.importorskip('torch');w=api();model=torch.nn.Linear(2,2);state=copy.deepcopy(model.state_dict())
    identity=w.strict_load_state(model,state)
    assert identity==w.typed_state_sha256(state)
    assert identity!=w.typed_state_sha256({k:v.double() for k,v in state.items()})
    assert identity!=w.typed_state_sha256({k:v.reshape(-1) for k,v in state.items()})


def test_actual_cpu_cannot_qualify_original_bf16_precision():
    torch=pytest.importorskip('torch');w=api()
    with pytest.raises(ValueError,match='precision|CUDA'):w._precision_guard(torch,{'precision':'bfloat16'},'cpu')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='CUDA'):w._precision_guard(torch,{'precision':'bfloat16'},'cuda:0')


def test_offline_transport_attempt_is_refused_and_restored():
    w=api();original=socket.socket.connect
    with w.offline_transport():
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as stream:
            with pytest.raises(RuntimeError,match='offline'):stream.connect(('127.0.0.1',9))
        with pytest.raises(RuntimeError,match='offline'):socket.getaddrinfo('example.invalid',443)
    assert socket.socket.connect is original


def test_atomic_attempt_late_fsync_cannot_leave_finished_or_touch_neighbor(tmp_path,monkeypatch):
    w=api();output=tmp_path/'new';neighbor=tmp_path/'neighbor';neighbor.mkdir();(neighbor/'status.json').write_bytes(b'other valid')
    original=w.os.fsync
    with w.Attempt(output,{'synthetic':True}) as attempt:
        attempt.write('centers.npz',b'private partial centers')
        def sync(fd):
            status=output/'status.json'
            if status.exists() and json.loads(status.read_text()).get('state')=='engineering_finished':
                monkeypatch.setattr(w,'_free',lambda _:0)
                raise OSError('injected directory fsync ENOSPC')
            return original(fd)
        monkeypatch.setattr(w.os,'fsync',sync)
        with pytest.raises(OSError):attempt.finish({'benchmark_eligible':False})
        attempt.fail()
    assert not (output/'status.json').exists()
    assert (output/'centers.npz').read_bytes()==b'private partial centers'
    assert (neighbor/'status.json').read_bytes()==b'other valid'


def test_attempt_rebind_cannot_delete_or_publish_neighbor_status(tmp_path):
    w=api();output=tmp_path/'owned'
    with w.Attempt(output,{}) as attempt:
        output.rename(tmp_path/'moved');output.mkdir();(output/'status.json').write_bytes(b'other valid')
        attempt.fail()
    assert (output/'status.json').read_bytes()==b'other valid'


def test_worker_has_no_formal_cli_or_implicit_checkpoint_search():
    w=api()
    with pytest.raises(ValueError):w._private_main(['run','--formal'])
    assert set(w.CHECKPOINTS)=={'s','m','rn18','fusion'}
    assert all(set(value)=={'last.pt','best.pt'} for value in w.CHECKPOINTS.values())


def test_offline_transport_keeps_real_local_unix_ipc_available(tmp_path):
    w=api();address=str(tmp_path/'ipc.sock')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
        listener.bind(address);listener.listen(1)
        with w.offline_transport(),socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.connect(address)
            with listener.accept()[0] as accepted:
                client.sendall(b'local IPC');assert accepted.recv(9)==b'local IPC'


def test_parent_final_admission_rejects_late_payload_mutation(inputs,monkeypatch):
    """Fake child result only exercises transaction failure, never CUDA success."""
    w=api();spec,cohort=inputs;payload=Path(spec['data'])/'payload.dat';payload.write_bytes(b'old')
    admitted=w._admit_inputs(spec,cohort,bounded=True)
    def child(*_):
        payload.write_bytes(b'new')
        return dict(admission=admitted,admission_sha256=w.preflight._canonical(admitted),
                    benchmark_eligible=False,this_derived_full_passes_per_model=0)
    monkeypatch.setattr(w,'_launch',child)
    with pytest.raises(ValueError,match='input|namespace'):w.run_model_pass(spec,cohort,bounded=True)
    status=Path(spec['output'])/'status.json'
    assert not status.exists() or json.loads(status.read_text())['state']=='failed'


def test_private_result_identity_does_not_repeat_sample_ordinals_or_paths(inputs):
    w=api();spec,cohort=inputs;admitted=w._admit_inputs(spec,cohort,bounded=True)
    result=w.result_identity(admitted)
    text=json.dumps(result)
    assert all(token not in text for token in ('window_ordinals','user_ordinal','legacy_root','private-source','private-run'))
    assert result['admission_sha256']==w.preflight._canonical(admitted)


def test_subject_scalar_conversion_matches_original_audit_memory_grouping():
    torch=pytest.importorskip('torch');w=api()
    assert w._user_key(torch.tensor(7))==w._user_key(np.int64(7))==w._user_key(7)=='7'


@pytest.fixture(scope='module')
def actual_old_source(tmp_path_factory):
    """Read Git objects into a new private fixture; never alter an old checkout."""
    w=api();root=tmp_path_factory.mktemp('actual-old-namespace')
    project=Path(__file__).parents[1]
    for name,expected in w.preflight.LEGACY_FILES.items():
        result=subprocess.run(['git','show','8dc57cdd667cb05341cb3f79064c59f58d27f7f8:'+name],
                              cwd=project,capture_output=True,check=True)
        assert hashlib.sha256(result.stdout).hexdigest()==expected
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(result.stdout);path.chmod(0o444)
    return root


def test_real_old_namespace_factory_and_alignment_are_used_in_fresh_process(actual_old_source):
    pytest.importorskip('torch');pytest.importorskip('hydra');project=Path(__file__).parents[1]
    script='''
import json,sys,torch
from pathlib import Path
from tools.legacy_derived import worker as w
transport=w.offline_transport();transport.__enter__()
runtime=w._load_runtime(sys.argv[1],sys.argv[2])
assert Path(runtime.train.__file__).is_relative_to(Path(sys.argv[1]))
cfg=json.loads(Path(sys.argv[3]).read_text())
runtime.train.seed_everything(cfg['seed'])
model=runtime.train.build_training_model(sys.argv[2],cfg)
before=w.pre_builder_state_sha256(model.state_dict())
token=w.strict_load_state(model,model.state_dict())
assert before==token
assert len(token)==64 and model.left_context==510 and model.right_context==0
class SyntheticDense:
    left_context=510
    right_context=0
    def __call__(self,batch):return batch['synthetic_prediction']
    def align_predictions(self,value,n):
        return torch.nn.functional.interpolate(value,size=n,mode='linear',align_corners=False)
prediction=torch.zeros(1,22,11490);prediction[:,:,5490]=1;prediction[:,:,5745]=9
batch=dict(emg=torch.zeros(1,8,12000),joint_angles=torch.zeros(1,22,12000),
           angle_valid_mask=torch.ones(1,22,12000,dtype=torch.bool),synthetic_prediction=prediction)
p,t,v=w._forward_chunk(runtime,SyntheticDense(),batch,dict(model_kind='emg',precision='float32'),'cpu')
assert p.shape==(1,22,11490) and (p[:,:,5490]==1).all() and (p[:,:,5745]==9).all()
w._source_check(sys.argv[1],sys.argv[2])
transport.__exit__(None,None,None)
print(json.dumps(dict(old_namespace=True,actual_original_builder=True,synthetic_cpu_alignment=True,gpu_qualified=False)))
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(actual_old_source),str(project/'upstream'),
        str(project/'configs/emgformer_s.json')],capture_output=True,text=True,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},timeout=60)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])==dict(old_namespace=True,actual_original_builder=True,
                                                                 synthetic_cpu_alignment=True,gpu_qualified=False)


def test_real_same_named_preimport_rejects_before_loading_old_runtime(actual_old_source):
    project=Path(__file__).parents[1]
    script='''
import sys
from tools.legacy_derived import worker as w
import egoemg_repro
try:w._load_runtime(sys.argv[1],sys.argv[2])
except ValueError as exc:
    assert 'preimport' in str(exc)
else:raise AssertionError('same named new runtime was admitted')
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(actual_old_source),str(project/'upstream')],
                          capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=30)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('damage',['float64','half','integer','bn_shape','bn_dtype','nonfinite'])
def test_pre_builder_dtype_gate_rejects_before_original_internal_load(damage):
    torch=pytest.importorskip('torch');w=api();state={'weight':torch.ones(2,2),'bn.num_batches_tracked':torch.tensor(3,dtype=torch.int64)}
    if damage=='float64':state['weight']=state['weight'].double()
    elif damage=='half':state['weight']=state['weight'].half()
    elif damage=='integer':state['unknown_buffer']=torch.ones(1,dtype=torch.int64)
    elif damage=='bn_shape':state['bn.num_batches_tracked']=torch.ones(1,dtype=torch.int64)
    elif damage=='bn_dtype':state['bn.num_batches_tracked']=torch.tensor(3,dtype=torch.int32)
    else:state['weight'][0,0]=float('inf')
    with pytest.raises(ValueError):w.pre_builder_state_sha256(state)


def test_pre_builder_float32_and_bn_scalar_have_same_post_strict_identity():
    torch=pytest.importorskip('torch');w=api();model=torch.nn.BatchNorm1d(2);state=model.state_dict()
    assert w.pre_builder_state_sha256(state)==w.strict_load_state(model,state)


def test_late_data_change_during_final_source_read_is_not_published(inputs,monkeypatch):
    w=api();spec,cohort=inputs;payload=Path(spec['data'])/'payload.dat';payload.write_bytes(b'old')
    admitted=w._admit_inputs(spec,cohort,bounded=True);original=w._admit_inputs;calls=[]
    def child(request,attempt):
        cache=b'synthetic artifact transaction fixture';attempt.write('centers.npz',cache)
        return dict(**w.result_identity(admitted),benchmark_eligible=False,format=w.FORMAT,status='bounded_engineering_only',
            scientific_eligible=False,this_derived_full_passes_per_model=0,center_cache_sha256=hashlib.sha256(cache).hexdigest())
    monkeypatch.setattr(w,'_launch',child)
    def admission(*args,**kwargs):
        value=original(*args,**kwargs);calls.append(True)
        if len(calls)==2:payload.write_bytes(b'late')
        return value
    monkeypatch.setattr(w,'_admit_inputs',admission)
    with pytest.raises(ValueError,match='input|namespace'):w.run_model_pass(spec,cohort,bounded=True)
    status=Path(spec['output'])/'status.json'
    assert not status.exists() or json.loads(status.read_text())['state']=='failed'


@pytest.mark.parametrize('damage',['scientific','scope','format','model'])
def test_artifact_parent_rejects_wrong_child_scope(inputs,monkeypatch,damage):
    w=api();spec,cohort=inputs;admitted=w._admit_inputs(spec,cohort,bounded=True)
    result=dict(**w.result_identity(admitted),benchmark_eligible=False,format=w.FORMAT,status='bounded_engineering_only',
                scientific_eligible=False,this_derived_full_passes_per_model=0)
    if damage=='scientific':result['scientific_eligible']=True
    elif damage=='scope':result['status']='full_complete'
    elif damage=='format':result['format']='other'
    else:result['model']='other'
    monkeypatch.setattr(w,'_launch',lambda *_:result)
    with pytest.raises(ValueError):w.run_model_pass(spec,cohort,bounded=True)


def test_network_guard_allows_actual_torch_loader_local_ipc():
    torch=pytest.importorskip('torch');w=api()
    with w.offline_transport():
        result=list(torch.utils.data.DataLoader(torch.arange(3),batch_size=2,num_workers=1))
    assert [value.tolist() for value in result]==[[0,1],[2]]


def test_numeric_environment_checks_actual_versions_and_keeps_missing_recorded_fields_explicit():
    torch=pytest.importorskip('torch');w=api();import platform
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                     cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    profile=w.numeric_environment(torch,environment,visual=False)
    assert profile['observed']['torch']==str(torch.__version__)
    assert profile['torchvision_recorded'] is False
    environment['numpy']='unapproved-version'
    with pytest.raises(ValueError):w.numeric_environment(torch,environment,visual=False)


@pytest.mark.parametrize('damage',['missing','sha'])
def test_partial_child_cannot_publish_without_matching_cache(inputs,monkeypatch,damage):
    w=api();spec,cohort=inputs;admitted=w._admit_inputs(spec,cohort,bounded=True)
    def child(request,attempt):
        if damage=='sha':attempt.write('centers.npz',b'changed private cache')
        return dict(**w.result_identity(admitted),benchmark_eligible=False,format=w.FORMAT,
            status='bounded_engineering_only',scientific_eligible=False,this_derived_full_passes_per_model=0,
            center_cache_sha256='0'*64)
    monkeypatch.setattr(w,'_launch',child)
    with pytest.raises(ValueError):w.run_model_pass(spec,cohort,bounded=True)
    status=Path(spec['output'])/'status.json'
    assert not status.exists() or json.loads(status.read_text())['state']=='failed'


def test_launcher_is_fresh_isolated_and_never_inherits_account_credentials(inputs,monkeypatch):
    w=api();spec,cohort=inputs;monkeypatch.setenv('HF_TOKEN','SYNTHETIC_SECRET')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY','SYNTHETIC_SECRET');seen=[]
    def subprocess_run(args,**kwargs):
        seen.append(True)
        assert '-I' in args and '-B' in args and '--_worker_request' in args
        assert kwargs['cwd']=='/' and 'PYTHONPATH' not in kwargs['env']
        assert 'HF_TOKEN' not in kwargs['env'] and 'AWS_SECRET_ACCESS_KEY' not in kwargs['env']
        assert kwargs['env']['HF_HUB_OFFLINE']=='1'
        assert kwargs['env']['EGOEMG_NO_PRETRAINED_DOWNLOAD']=='1'
        assert kwargs['env']['TORCH_HOME'].startswith(spec['output'])
        assert kwargs['env']['TMPDIR'].startswith(spec['output']) and Path(kwargs['env']['TMPDIR']).is_dir()
        return SimpleNamespace(returncode=1,stdout='',stderr='synthetic private traceback')
    monkeypatch.setattr(w.subprocess,'run',subprocess_run)
    with w.Attempt(spec['output'],{}) as attempt:
        with pytest.raises(RuntimeError,match='exit=1'):w._launch({},attempt)
        assert (Path(spec['output'])/'request.json').stat().st_mode&0o777==0o600
    assert seen


def test_factory_temporary_budget_is_admitted_without_spending_reserve(tmp_path,monkeypatch):
    w=api();output=tmp_path/'never-created'
    monkeypatch.setattr(w,'_free',lambda _:w.RESERVE+100*1024**2)
    with pytest.raises(OSError,match='reserve'):w.Attempt(output,{})
    assert not output.exists()


def transaction_child(w,spec,cohort,monkeypatch):
    """Transaction-only cache fixture, never a positive inference credential."""
    monkeypatch.setattr(w,'_free',lambda _:100*1024**3)
    admitted=w._admit_inputs(spec,cohort,bounded=True)
    def child(request,attempt):
        cache=b'synthetic transaction fixture'
        attempt.write('centers.npz',cache)
        return dict(**w.result_identity(admitted),benchmark_eligible=False,format=w.FORMAT,
            status='bounded_engineering_only',scientific_eligible=False,
            this_derived_full_passes_per_model=0,
            center_cache_sha256=hashlib.sha256(cache).hexdigest())
    monkeypatch.setattr(w,'_launch',child)


def test_source_change_after_last_admission_cannot_finish(inputs,monkeypatch):
    w=api();spec,cohort=inputs;transaction_child(w,spec,cohort,monkeypatch)
    original=w._admit_inputs;calls=[]
    source=Path(spec['legacy_root'])/'egoemg_repro/train.py'
    def late_source(*a,**k):
        value=original(*a,**k);calls.append(True)
        if len(calls)==2:source.write_bytes(source.read_bytes()+b'# changed after admission\n')
        return value
    monkeypatch.setattr(w,'_admit_inputs',late_source)
    with pytest.raises(ValueError,match='input|source|namespace'):
        w.run_model_pass(spec,cohort,bounded=True)
    status=Path(spec['output'])/'status.json'
    assert not status.exists() or json.loads(status.read_text())['state']=='failed'


def test_pre_capture_public_directory_rebind_never_adopts_neighbor(tmp_path,monkeypatch):
    w=api();monkeypatch.setattr(w,'_free',lambda _:100*1024**3)
    output=tmp_path/'attempt';neighbor=tmp_path/'neighbor';neighbor.mkdir()
    (neighbor/'status.json').write_bytes(b'foreign status')
    (neighbor/'checkpoint.pt').write_bytes(b'foreign checkpoint')
    original=w.os.open;fired=[]
    def opening(path,flags,*a,**k):
        if Path(path)==output and flags&w.os.O_DIRECTORY and not fired:
            fired.append(True);output.rename(tmp_path/'original-moved');neighbor.rename(output)
        return original(path,flags,*a,**k)
    monkeypatch.setattr(w.os,'open',opening)
    with pytest.raises(ValueError,match='owned|identity|directory'):
        with w.Attempt(output,{}) as attempt:attempt.fail()
    assert fired
    assert (output/'status.json').read_bytes()==b'foreign status'
    assert (output/'checkpoint.pt').read_bytes()==b'foreign checkpoint'
    assert not (output/'attempt.json').exists()


@pytest.mark.parametrize('damage',['source','last','best','protocol','data','crop','cache'])
def test_final_status_commit_still_guards_all_used_inputs(inputs,monkeypatch,damage):
    w=api();spec,cohort=inputs;transaction_child(w,spec,cohort,monkeypatch)
    run=Path(spec['run_spec']['run'])
    path={'source':Path(spec['legacy_root'])/'egoemg_repro/train.py',
          'last':run/'last.pt','best':run/'best.pt','protocol':run/'config.json',
          'data':Path(spec['data'])/'manifest.json','crop':Path(spec['crops'])/'receipt.json',
          'cache':Path(spec['output'])/'centers.npz'}[damage]
    original=w.os.fsync;fired=[]
    def sync(fd):
        status=Path(spec['output'])/'status.json'
        if not fired and status.exists() and json.loads(status.read_text()).get('state')=='engineering_finished':
            fired.append(True);path.write_bytes(path.read_bytes()+b' ')
        return original(fd)
    monkeypatch.setattr(w.os,'fsync',sync)
    with pytest.raises(ValueError,match='input|source|namespace'):
        w.run_model_pass(spec,cohort,bounded=True)
    assert fired
    status=Path(spec['output'])/'status.json'
    assert not status.exists() or json.loads(status.read_text())['state']=='failed'


def test_failure_does_not_revoke_a_foreign_replacement_status(tmp_path,monkeypatch):
    w=api();monkeypatch.setattr(w,'_free',lambda _:100*1024**3)
    output=tmp_path/'attempt'
    with w.Attempt(output,{}) as attempt:
        attempt.finish({'benchmark_eligible':False})
        status=output/'status.json';status.unlink();status.write_bytes(b'foreign status')
        attempt.fail()
    assert status.read_bytes()==b'foreign status'


def test_attempt_refuses_unsupported_atomic_publication_platform(tmp_path,monkeypatch):
    w=api();monkeypatch.setattr(w,'_free',lambda _:100*1024**3)
    monkeypatch.setattr(w.sys,'platform','unsupported')
    output=tmp_path/'attempt'
    with pytest.raises(RuntimeError,match='Linux|unsupported|atomic'):
        with w.Attempt(output,{}):pass
    assert not output.exists()


def test_concurrent_final_name_cannot_replace_existing_directory(tmp_path,monkeypatch):
    w=api();monkeypatch.setattr(w,'_free',lambda _:100*1024**3)
    output=tmp_path/'attempt';original=w.os.mkdir;fired=[]
    def mkdir(path,*a,**k):
        value=original(path,*a,**k)
        if str(path).startswith('.legacy-derived-stage-') and not fired:
            fired.append(True);original(output);(output/'status.json').write_bytes(b'concurrent status')
        return value
    monkeypatch.setattr(w.os,'mkdir',mkdir)
    with pytest.raises(FileExistsError):
        with w.Attempt(output,{}):pass
    assert fired and (output/'status.json').read_bytes()==b'concurrent status'
    assert not (output/'attempt.json').exists()
