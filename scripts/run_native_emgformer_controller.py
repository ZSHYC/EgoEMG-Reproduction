"""Owned one-shot qualification/full execution; wait for an unreserved idle GPU."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
if sys.flags.optimize:raise RuntimeError('native workflow requires normal Python')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));started=time.monotonic()
from egoemg_repro.emg2pose_memory_process import run_owned_stage,same_process
from egoemg_repro.emgformer_resource import select_idle_emgformer_gpu
CANDIDATES=['GPU-8b2cacdf-4470-21f2-01c5-e15475dbdfae','GPU-20575323-8e23-b0b8-74b4-b987d870b0e3']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def stamp(p):
 s=Path(p).lstat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
def write(p,v,replace=False):
 p=Path(p);temporary=p.with_suffix(p.suffix+'.pending')
 if not replace:assert not p.exists()
 with temporary.open('x')as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,p)
stage=read(ROOT/'private/stage-receipt.json');request=read(ROOT/'private/request.json')
assert request['candidate_uuids']==CANDIDATES and request['size']=='small'
assert request['execution_mode']in('preflight_only','full_after_qualification')
if request['execution_mode']!='preflight_only':
 review=read(ROOT/'private/final-review.json')
 assert review['critical']==review['important']==0 and review['second_review_requested']is False
 for n,h in review['sources_sha256'].items():assert sha(ROOT/n)==h,n
files={str(ROOT/n):h for n,h in stage['files_sha256'].items()}
for n,h in files.items():assert sha(n)==h,n
stamps={n:stamp(n)for n in files};external={};selected=None;mode='preflight';phase_started=time.monotonic();observations=[];owned=[];gpu_owners={}
gpu_cache=[0.,None]
def inventory(force=False):
 if force or time.monotonic()-gpu_cache[0]>=5:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
  gpus=[]
  for line in raw.splitlines():
   i,u,f,t=[x.strip()for x in line.split(',')];gpus.append(dict(index=int(i),uuid=u,free_mib=int(f),total_mib=int(t)))
  raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
  apps=[]
  for line in raw.splitlines():
   p,u=[x.strip()for x in line.split(',')];apps.append(dict(pid=int(p),uuid=u))
  # Validate the entire typed snapshot even when the selected device is busy.
  select_idle_emgformer_gpu(gpus,apps,candidates=CANDIDATES)
  gpu_cache[:]=[time.monotonic(),dict(gpus=gpus,applications=apps,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())]
 return gpu_cache[1]
def guard():
 assert 'torch'not in sys.modules
 for n,s in stamps.items():assert stamp(n)==s,'immutable workflow source/input changed'
 mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 free=shutil.disk_usage(ROOT).free
 assert mem>256*1024**3 and free>15*1024**3,'workflow host/output reserve lost'
 if mode!='waiting':assert time.monotonic()-phase_started<(7*86400 if mode=='full-fit'else 7200),'bounded native phase exceeded'
 device_free=None
 if selected is not None:
  current=inventory();device=next(g for g in current['gpus']if g['uuid']==selected['uuid']);device_free=device['free_mib']*1024**2
  assert device_free>8*1024**3,'native GPU execution reserve lost'
  apps=[a for a in current['applications']if a['uuid']==selected['uuid']]
  path=ROOT/('private/owned-'+mode+'/started.json');owner=read(path)['worker']if path.exists()else None
  for app in apps:
   assert owner is not None and same_process(owner)and app['pid']==owner['pid'],'foreign compute application arrived; preserve external consumer'
   if mode not in gpu_owners:gpu_owners[mode]=dict(observed_utc=current['observed_utc'],application=app,actual_owned_worker=owner)
 for p in(ROOT/'private').glob('owned-*/worker.log'):assert p.stat().st_size<64*1024**2,'bounded worker log exceeded'
 if not observations or time.monotonic()-observations[-1]['monotonic']>=60:
  observations.append(dict(monotonic=time.monotonic(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=mode,host_available_bytes=mem,disk_free_bytes=free,gpu_free_bytes=device_free))
def phase(name):
 global mode,phase_started
 mode=name;phase_started=time.monotonic();cpu=name=='preflight'or name.endswith(('-accept','-freeze','-report','-verify'))
 if not cpu:
  snap=inventory(force=True);choice=select_idle_emgformer_gpu(snap['gpus'],snap['applications'],candidates=[selected['uuid']])
  assert choice is not None,'same admitted GPU is no longer idle with64GiB reserve'
 guard();env=dict(os.environ,CUDA_VISIBLE_DEVICES=''if cpu else selected['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 print(json.dumps(dict(phase=name+'-start',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
 result=run_owned_stage([sys.executable,'-u','-B',str(ROOT/'private/runtime-worker.py'),name],ROOT/('private/owned-'+name),descriptors={},guard=guard,environment=env,poll_seconds=1)
 assert result['all_owned_consumers_exited']and all(not same_process(result[k])for k in('worker','guardian'));owned.append(result)
 if not cpu:assert name in gpu_owners,'actual same-worker NVML GPU ownership was not observed'
 if selected is not None:inventory(force=True)
 guard();return read(ROOT/('private/phase-'+name+'.json'))
guard();preflight=phase('preflight');external=preflight['result']['immutable_external_files_sha256']
for n,s in preflight['result']['immutable_external_file_stamps'].items():
 assert stamp(n)==s,n;stamps[n]=s
admission_snapshot=None;admission_choice=None
if request['execution_mode']=='preflight_only':
 admission_snapshot=inventory(force=True)
 admission_choice=select_idle_emgformer_gpu(admission_snapshot['gpus'],admission_snapshot['applications'],candidates=CANDIDATES)
else:
 mode='waiting';wait_started=time.monotonic();last_notice=0.
 while selected is None:
  guard();snap=inventory(force=True);selected=select_idle_emgformer_gpu(snap['gpus'],snap['applications'],candidates=CANDIDATES)
  waiting=dict(status='waiting_for_idle_candidate_gpu'if selected is None else'gpu_admitted',observed_utc=snap['observed_utc'],
   waited_seconds=time.monotonic()-wait_started,candidate_uuids=CANDIDATES,snapshot=snap,chosen=selected,training_started=False,scientific_eligible=False)
  write(ROOT/'private/waiting-state.json',waiting,replace=(ROOT/'private/waiting-state.json').exists())
  if selected is not None:
   write(ROOT/'private/gpu-binding.json',dict(selected,admitted_snapshot=snap,candidate_uuids=CANDIDATES));break
  assert time.monotonic()-wait_started<86400,'bounded24hour idle-GPU wait expired; no retry'
  if time.monotonic()-last_notice>1800:print(json.dumps(dict(status='waiting-for-idle-permitted-gpu',waited_seconds=waiting['waited_seconds'])),flush=True);last_notice=time.monotonic()
  time.sleep(10)
 for name in ('qual-prefix','qual-resume','qual-accept','qual-val-native','qual-val-replay','qual-freeze','qual-test-native','qual-test-replay',
  'qual-report','qual-verify','full-fit','full-accept','full-val-native','full-val-replay','full-freeze','full-test-native','full-test-replay','full-report','full-verify'):
  phase(name)
guard()
for n,h in {**files,**external}.items():assert sha(n)==h,n
guard()
phases={p.name.removeprefix('phase-').removesuffix('.json'):read(p)for p in(ROOT/'private').glob('phase-*.json')}
full=request['execution_mode']!='preflight_only'
assert len(phases)==(20 if full else 1)
if full:
 assert phases['full-report']['result']==phases['full-verify']['result']and phases['full-verify']['result']['scientific_eligible']
 assert sum(p['actual_optimizer_updates']for p in phases.values())==25004
value=dict(format='actual_guarded_native_emgformer_gpu_workflow_v1',source_commit=stage['source_commit'],bundle_sha256=stage['bundle_sha256'],
 execution_mode=request['execution_mode'],seconds=time.monotonic()-started,phases=phases,owned_stages=owned,resource_observations=observations,
 all_source_and_input_bytes_and_nine_attributes_unchanged=True,all_owned_stage_consumers_exited=True,parent_torch_unimported=True,
 actual_nvml_owned_worker_observations=gpu_owners,preflight_only_resource_snapshot=admission_snapshot,preflight_only_idle_choice=admission_choice,
 actual_models_constructed=sum(p['actual_model_constructions']for p in phases.values()),actual_optimizer_updates=sum(p['actual_optimizer_updates']for p in phases.values()),
 actual_gpu_binding=selected,scientific_eligible=full,full_paper_complete=False)
write(ROOT/'private/workflow-result.json',value)
print(json.dumps(dict(status='full-native-small-complete'if full else'cpu-preflight-only-complete',seconds=value['seconds'],updates=value['actual_optimizer_updates'])),flush=True)
