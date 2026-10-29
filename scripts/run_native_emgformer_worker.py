"""One original native phase; passive counters never replace model mathematics."""
from pathlib import Path
import datetime,hashlib,importlib,json,os,sys,time
if sys.flags.optimize:raise RuntimeError('native execution evidence requires normal Python')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));started=time.monotonic()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def stamp(p):
 s=Path(p).lstat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
from egoemg_repro.emgformer_execution import pin_stage,validate_review,authorize_worker
stage,pinned=pin_stage(ROOT);stage_sha=pinned[str(ROOT/'private/stage-receipt.json')][0]
request=read(ROOT/'private/request.json');mode=sys.argv[1];review_sha=None
phase_modes=('preflight','qual-prefix','qual-resume','qual-accept','qual-val-native','qual-val-replay','qual-freeze',
 'qual-test-native','qual-test-replay','qual-report','qual-verify','full-fit','full-accept','full-val-native','full-val-replay',
 'full-freeze','full-test-native','full-test-replay','full-report','full-verify')
assert mode in phase_modes
cpu=mode=='preflight'or mode.endswith(('-accept','-freeze','-report','-verify'))
if request['execution_mode']=='preflight_only':assert mode=='preflight'
else:
 review_sha=validate_review(ROOT,stage)
controller=authorize_worker(ROOT,stage,stage_sha,mode,review_sha)
if not cpu:
 binding=read(ROOT/'private/gpu-binding.json');assert binding['uuid']in request['candidate_uuids']
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==binding['uuid']
else:assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from egoemg_repro import emgformer_train as train,emgformer_eval as evaluate
from egoemg_repro.emgformer_literal import compose_emgformer,emgformer_normalization
from egoemg_repro.emgformer_cohort import admit_emgformer_cohort,_sources as cohort_sources,FULL_COUNTS
from egoemg_repro.emgformer_runtime import _versions,RUNTIME
torch.set_num_threads(2)
assert train.build_emgformer_runtime is evaluate.build_emgformer_runtime
factory=train.build_emgformer_runtime;optimizer_init=torch.optim.Optimizer.__init__;adam_step=torch.optim.AdamW.step
constructs=optimizers=steps=0;parameters=[];forward_observations={}
def counted_factory(*a,**kw):
 global constructs
 constructs+=1;prepared=factory(*a,**kw);parameters.append(sum(p.numel()for p in prepared.module.parameters()))
 original=prepared.module.forward
 def observe(batch):
  assert not cpu and batch['emg'].device.type=='cuda'
  assert all(p.device==batch['emg'].device for p in prepared.module.parameters())
  assert batch['joint_angles'].device==batch['emg'].device==batch['label_valid_mask'].device
  out=original(batch);assert all(x.device==batch['emg'].device for x in out)
  kind='point'if batch['joint_angles'].shape[-1]==1 else'full'
  entry=forward_observations.setdefault(kind,dict(batches=0,samples=0,batch_sizes=[],input_device=str(batch['emg'].device),
   module_device=str(next(prepared.module.parameters()).device),prediction_device=str(out[0].device),prediction_dtypes=[]))
  entry['batches']+=1;entry['samples']+=len(batch['emg']);entry['batch_sizes'].append(len(batch['emg']))
  if str(out[0].dtype)not in entry['prediction_dtypes']:entry['prediction_dtypes'].append(str(out[0].dtype))
  return out
 prepared.module.forward=observe;return prepared
def counted_optimizer(*a,**kw):
 global optimizers
 optimizers+=1;return optimizer_init(*a,**kw)
def counted_step(*a,**kw):
 global steps
 steps+=1;return adam_step(*a,**kw)
train.build_emgformer_runtime=evaluate.build_emgformer_runtime=counted_factory
torch.optim.Optimizer.__init__=counted_optimizer;torch.optim.AdamW.step=counted_step
if not cpu:
 assert torch.cuda.is_available()and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
 assert 'A800'in torch.cuda.get_device_name(0)
 torch.cuda.reset_peak_memory_stats()
else:assert not torch.cuda.is_initialized()
up=Path(request['upstream']);data=Path(request['data']);receipt=ROOT/'private/registered-cohort.json'
expected=read(receipt);config=compose_emgformer(up,'small')
qual_fit=ROOT/'private/qualification-fit';full_fit=ROOT/'private/full-fit'
def evaluation_root(family):return ROOT/('private/'+family+'-evaluation')
def qualified_raw_gate():
 fit=train.read_emgformer_fit(qual_fit,require_full=False)
 assert fit['scope']=='cuda_engineering'and fit['completed_epochs']==2 and fit['optimizer_executions']==4
 assert fit['stage_complete']and not fit['full_training_terminated']and fit['resume_validation_exact']
 assert all(fit['resume_start_exact'].values())and fit['cuda_initialized']
 identity=read(qual_fit/'identity.json');policy=identity['policy']
 assert identity['cohort']==expected and identity['train_samples']==1000
 assert policy['workers']==6 and policy['precision']=='bf16-mixed'and policy['batch_size_by_role']==dict(train=500,val=500)
 report=evaluate.verify_emgformer_report(evaluation_root('qual')/'test-replay',evaluation_root('qual')/'report')
 assert not report['scientific_eligible']and not report['full_paper_complete']
 for name in ('qual-prefix','qual-resume','qual-val-native','qual-val-replay','qual-test-native','qual-test-replay'):
  phase=read(ROOT/('private/phase-'+name+'.json'))
  assert phase['actual_model_constructions']==1 and phase['actual_cuda_initialized']and phase['actual_gpu_forward_observations']
  assert phase['source_commit']==stage['source_commit']and phase['gpu_binding']['uuid']==binding['uuid']
 assert read(ROOT/'private/phase-qual-prefix.json')['actual_optimizer_updates']==2
 assert read(ROOT/'private/phase-qual-resume.json')['actual_optimizer_updates']==2
 return report['sha256']
if mode=='preflight':
 cohort=admit_emgformer_cohort(data,upstream=up,config=config,normalization=emgformer_normalization(up,config),expected=expected)
 assert cohort.identity['metadata_matches_registered_legacy_release']and cohort.identity['counts']==FULL_COUNTS
 sources=cohort_sources(up);external={str(p):h for n,p in sources.items()if not p.is_relative_to(ROOT)for h in[cohort.identity['sources_sha256'][n]]}
 for n,p in cohort.input_paths.items():external[str(p)]=cohort.identity['payload_files_sha256'][n]
 for p in [*train._loop_paths().values(),*[Path(importlib.import_module(n).__file__).resolve()for n in RUNTIME]]:
  if not Path(p).is_relative_to(ROOT):external[str(p)]=sha(p)
 for n,h in external.items():assert sha(n)==h,n
 result=dict(identity=cohort.identity,immutable_external_files_sha256=external,immutable_external_file_stamps={n:stamp(n)for n in external},
  input_paths={n:str(p)for n,p in cohort.input_paths.items()},physical_input_bytes=sum(cohort.identity['payload_file_bytes'].values()),
  actual_runtime_versions=_versions(),gpu_qualified=False,scientific_eligible=False)
 cohort.check()
elif mode.endswith('-accept'):
 full=mode.startswith('full-');result=train.read_emgformer_fit(full_fit if full else qual_fit,require_full=full)
 assert result['completed_epochs']==(200 if full else 2)and result['optimizer_executions']==(25000 if full else 4)
 assert result['full_training_terminated']is full
elif mode.endswith(('-freeze','-report','-verify')):
 family,action=mode.split('-');out=evaluation_root(family)
 if action=='freeze':result=evaluate.freeze_emgformer_validation(out/'val-native',out/'val-replay',out/'profile')
 elif action=='report':result=evaluate.publish_emgformer_report(out/'test-replay',out/'report')
 else:result=evaluate.verify_emgformer_report(out/'test-replay',out/'report')
 if action in ('report','verify'):assert result['scientific_eligible']is(family=='full')
else:
 if mode=='full-fit':qualified_raw_gate()
 cohort=admit_emgformer_cohort(data,upstream=up,config=config,normalization=emgformer_normalization(up,config),expected=expected)
 assert cohort.identity['counts']==FULL_COUNTS and cohort.identity['metadata_matches_registered_legacy_release']
 print(json.dumps(dict(phase=mode+'-complete-input-admission',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
 if mode in ('qual-prefix','qual-resume','full-fit'):
  full=mode=='full-fit';result=train.fit_emgformer(cohort,full_fit if full else qual_fit,
   scope='full'if full else'cuda_engineering',resume=mode=='qual-resume',stop_after_epoch=1 if mode=='qual-prefix'else None,cohort_receipt=receipt)
  assert result['completed_epochs']==(200 if full else 1 if mode=='qual-prefix'else 2)
  assert result['optimizer_executions']==(25000 if full else 2 if mode=='qual-prefix'else 4)
  assert result['full_training_terminated']is full
  assert constructs==optimizers==1 and steps==(25000 if full else 2)
 else:
  family,role,kind=mode.split('-');out=evaluation_root(family);out.mkdir(exist_ok=True)
  result=evaluate.run_emgformer_pass(cohort,full_fit if family=='full'else qual_fit,out/(role+'-'+kind),role=role,kind=kind,
   require_full=family=='full',reference=out/(role+'-native')if kind=='replay'else None,profile=out/'profile'if role=='test'else None)
  assert result['samples']==FULL_COUNTS[role]and result['batch_size']==500
  assert constructs==1 and optimizers==steps==0
 cohort.check()
if cpu:assert constructs==optimizers==steps==0 and not torch.cuda.is_initialized()
else:assert constructs==1 and torch.cuda.is_initialized()and forward_observations
for n,(h,s)in pinned.items():assert sha(n)==h and stamp(n)==s,n
value=dict(format='actual_native_emgformer_gpu_workflow_phase_v1',phase=mode,source_commit=stage['source_commit'],seconds=time.monotonic()-started,
 bundle_sha256=stage['bundle_sha256'],stage_receipt_sha256=stage_sha,controller=controller,
 actual_model_constructions=constructs,actual_model_parameter_counts=parameters,actual_optimizer_constructions=optimizers,actual_optimizer_updates=steps,
 actual_gpu_forward_observations=forward_observations,actual_cuda_initialized=torch.cuda.is_initialized(),
 actual_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()if not cpu else 0,actual_peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved()if not cpu else 0,
 actual_owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
 gpu_binding=None if cpu else binding,result=result,scientific_eligible=False,full_paper_complete=False)
write(ROOT/('private/phase-'+mode+'.json'),value)
print(json.dumps(dict(phase=mode,seconds=value['seconds'],models=constructs,optimizer_updates=steps)),flush=True)
