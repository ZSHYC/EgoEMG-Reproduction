"""One original full M/L controller and a fresh independent verifier.

No original queue/source/runtime mutation, automatic retry or model override.
The output is a real original per-model output, never a placeholder success.
"""
from pathlib import Path
import json,os,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from egoemg_repro import emg2pose_direct_experiment as driver
from egoemg_repro.emg2pose_memory_process import run_owned_stage,process_identity,same_process

load=driver.storage._json;sha=driver.storage._sha256
manifest=load(ROOT/'private/stage-receipt.json')
request=load(ROOT/'private/full-request.json')
assert request['format']=='actual_emg2pose_independent_full_request_v1'
assert request['size'] in ('middle','large')
assert request['scope']=='full' and request['deadline_seconds'] is None
assert Path(request['output']).name==request['size']
assert request['original_queue_source_changed'] is False
parent=process_identity(os.getppid());started=time.monotonic();private=ROOT/'private'

def guard():
 if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
  raise ValueError('independent full parent must remain NoTorch with CUDA hidden')
 if not same_process(parent):raise RuntimeError('independent outer recorder owner lost')
 for n,h in manifest['files_sha256'].items():driver._expected(ROOT/n,h)
 for n,h in request['immutable_external_files_sha256'].items():driver._expected(n,h)
 driver._sources();driver.storage._space(ROOT)
 if driver.memory.available_memory_bytes()<driver.memory.RESERVE_BYTES:
  raise RuntimeError('actual unchanged256GiB host reserve lost')
 if driver._gpu(request['gpu_uuid'])['free_bytes']<8*1024**3:
  raise RuntimeError('actual unchanged8GiB GPU reserve lost')

def state(status,**extra):
 driver._write(private/'independent-state.json',dict(status=status,size=request['size'],
  scope='full',full_paper_complete=False,**extra),replace=True)

def actual_closed(value):
 return (value['status']=='success' and value['worker_exit_code']==0 and
  value['all_owned_consumers_exited'] and
  all(not driver._alive(value[k]) for k in ('worker','guardian')))

guard()
for n in ('controller-owned','verify-owned','independent-result.json','independent-state.json'):
 if (private/n).exists():raise FileExistsError('independent full workflow already has durable output')
output=driver.storage._safe_path(request['output'])
common=[sys.executable,'-m','egoemg_repro.emg2pose_direct_experiment','run',
 '--upstream',request['upstream'],'--source-root',request['source_root'],
 '--metadata-csv',request['metadata_csv'],'--expected-source-completion-sha256',request['expected_source_completion_sha256'],
 '--size',request['size'],'--scope','full','--gpu-uuid',request['gpu_uuid'],'--output',str(output)]
for key in ('qualification','evaluation_qualification','controller_qualification'):
 reference=request[key]
 common+=['--'+key.replace('_','-'),reference['directory'],
  '--expected-'+key.replace('_','-')+'-sha256',reference['sha256']]

try:
 # An existing output is an already-owned candidate, not an invitation to
 # overwrite, retry elsewhere or claim its completion. The unchanged driver
 # additionally enforces this and atomic mkdir if two claimants race.
 if output.exists():raise FileExistsError('per-model output already claimed; no duplicate launch')
 state('full-controller-running',scientific_complete=False)
 controller=run_owned_stage(common,private/'controller-owned',descriptors={},guard=guard,
  environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''),poll_seconds=2.,terminate_grace=10.)
 if not actual_closed(controller):raise RuntimeError('original full controller failed; no automatic retry')
 completion=load(output/'driver-completion.json');driver._check(completion)
 if (completion['scope']!='full' or completion['size']!=request['size'] or
  completion['scientific_complete'] is not True or len(completion['stages'])!=14):
  raise ValueError('actual original full14-stage per-model completion required')
 frozen=sha(output/'driver-completion.json')
 state('independent-full-verification',scientific_complete=False)
 verification=run_owned_stage([sys.executable,'-m','egoemg_repro.emg2pose_direct_experiment','verify',
  '--output',str(output),'--expected-completion-sha256',frozen],private/'verify-owned',
  descriptors={},guard=guard,environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''),poll_seconds=2.,terminate_grace=10.)
 if not actual_closed(verification):raise RuntimeError('independent full per-model verification failed')
 guard();driver._expected(output/'driver-completion.json',frozen)
 result=dict(format='actual_emg2pose_independent_full_workflow_v1',scope='full',size=request['size'],
  gpu_uuid=request['gpu_uuid'],output=str(output),driver_completion_sha256=frozen,
  report_completion_sha256=sha(output/'report/completion.json'),
  controller_owned_sha256=sha(private/'controller-owned/stage-result.json'),
  verification_owned_sha256=sha(private/'verify-owned/stage-result.json'),
  actual_original_controller_exit_code=0,actual_independent_verifier_exit_code=0,
  all_owned_consumers_exited=True,parent_torch_unimported=True,automatic_retry=False,
  single_full_model_pipeline_verified=True,all_three_models_complete=False,full_paper_complete=False,
  request_sha256=sha(private/'full-request.json'),seconds=time.monotonic()-started)
 driver._write(private/'independent-result.json',result)
 state('complete',single_full_model_pipeline_verified=True)
 print(json.dumps(dict(status='complete',size=request['size'],full_paper_complete=False)),flush=True)
except BaseException as exc:
 # Only this workflow's state is written. Never delete or rewrite an existing
 # per-model output, the original S result, or the old serial queue's state.
 state('failed',error_type=type(exc).__name__,error=str(exc),scientific_complete=False)
 raise
