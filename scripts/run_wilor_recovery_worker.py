"""Frozen CPU engineering phases for explicit native WiLoR continuation."""
from pathlib import Path
import hashlib,json,os,socket,sys,time
if sys.flags.optimize:raise RuntimeError('normal Python required for recovery evidence')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));mode=sys.argv[1];started=time.monotonic()
request=json.loads((ROOT/'private/request.json').read_text());stage=json.loads((ROOT/'private/stage-receipt.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def check():
 for n,h in stage['files_sha256'].items():assert sha(ROOT/n)==h,n
check();assert mode in('prefix','resume','read')and request['scope']=='cpu_engineering'and os.environ['CUDA_VISIBLE_DEVICES']==''
assert request['output_kind']=='explicitly_volatile_engineering'
prefix=Path(request['prefix_output']);recovered=Path(request['recovered_output'])
assert prefix.is_relative_to('/dev/shm')and recovered.is_relative_to('/dev/shm')
def offline(*a,**k):raise AssertionError('native recovery engineering must stay offline')
socket.socket.connect=offline;socket.create_connection=offline
import torch
torch.set_num_threads(2);assert not torch.cuda.is_initialized()
from egoemg_repro import wilor_literal_runtime as runtime,wilor_native_train as native
from egoemg_repro.wilor_literal import compose_wilor
from egoemg_repro.wilor_recovery import admit_wilor_recovery,resume_wilor,read_wilor_recovery
from egoemg_repro.vision_literal_train import _read_epochs
counts=dict(models=0,optimizers=0,updates=0);parameters=[];forwards=[]
factory=runtime.build_wilor_runtime;optimizer_init=torch.optim.Optimizer.__init__;adam_step=torch.optim.AdamW.step

def counted_factory(*a,**kw):
 result=factory(*a,**kw);counts['models']+=1;parameters.append(sum(p.numel()for p in result.module.parameters()))
 original=result.module.forward
 def observed(batch):
  out=original(batch)
  assert batch['vision_img'].device.type=='cpu'and next(result.module.parameters()).device.type=='cpu'and out[0].device.type=='cpu'
  forwards.append(dict(samples=len(batch['vision_img']),prediction_dtype=str(out[0].dtype),prediction_shape=list(out[0].shape)))
  return out
 result.module.forward=observed;return result

def counted_optimizer(*a,**kw):
 counts['optimizers']+=1;return optimizer_init(*a,**kw)
def counted_step(*a,**kw):
 result=adam_step(*a,**kw);counts['updates']+=1;return result
runtime.build_wilor_runtime=counted_factory;torch.optim.Optimizer.__init__=counted_optimizer;torch.optim.AdamW.step=counted_step

if mode=='prefix':
 from pytorch_lightning.plugins.io import TorchCheckpointIO
 class ExpectedPrefixStop(RuntimeError):pass
 original_save=TorchCheckpointIO.save_checkpoint
 def stop_after_first_last(self,checkpoint,path,storage_options=None):
  result=original_save(self,checkpoint,path,storage_options=storage_options)
  if Path(path).name=='last.ckpt'and checkpoint.get('native_wilor',{}).get('completed_epochs')==1:
   raise ExpectedPrefixStop('intentional engineering interruption after complete first-epoch last save')
  return result
 TorchCheckpointIO.save_checkpoint=stop_after_first_last;stopped=False
 try:
  cfg=compose_wilor(request['upstream']);assets=runtime.admit_wilor_assets(request['runtime'],request['pretrained'])
  native.fit_wilor(request['data'],request['crops'],upstream=request['upstream'],config=cfg,assets=assets,output=prefix,scope='cpu_engineering')
 except ExpectedPrefixStop:stopped=True
 finally:TorchCheckpointIO.save_checkpoint=original_save
 assert stopped and counts==dict(models=1,optimizers=1,updates=2)and parameters==[631631657]
 identity=json.loads((prefix/'identity.json').read_text());records,_=_read_epochs(prefix/'epochs.jsonl',identity['sha256'])
 assert len(records)==1 and records[0]['optimizer_executions']==2 and not(prefix/'fit-result.json').exists()
 result=dict(format='actual_intentionally_interrupted_native_wilor_prefix_v1',historical_identity_sha256=identity['sha256'],
  actual_complete_epochs=1,actual_optimizer_updates=2,expected_stop_after_complete_native_save=True,full_training_terminated=False,
  native_last_sha256=sha(prefix/'checkpoints/last.ckpt'),last_epoch=records[0],scientific_eligible=False)
elif mode=='resume':
 admitted=admit_wilor_recovery(prefix);assert len(admitted.records)==1
 result=resume_wilor(admitted,recovered)
 assert result['completed_epochs']==2 and result['optimizer_executions']==4 and result['segment_optimizer_executions']==2
 assert result['checkpoint_continuation_supported']and not result['uninterrupted_execution']and not result['full_training_terminated']
 assert result['frozen_validation']['parent_prediction_exact']and all(result['native_restore']['equal'].values())
 assert counts==dict(models=1,optimizers=1,updates=2)and parameters==[631631657]
else:
 result=read_wilor_recovery(recovered,require_full=False)
 assert counts==dict(models=0,optimizers=0,updates=0)and parameters==[]and not forwards
 try:native.read_wilor_fit(recovered,require_full=False)
 except ValueError:pass
 else:raise AssertionError('old uninterrupted native reader accepted recovered format')
assert not torch.cuda.is_initialized();check()
value=dict(format='actual_native_wilor_explicit_recovery_cpu_phase_v1',mode=mode,source_commit=stage['source_commit'],bundle_sha256=stage['bundle_sha256'],
 seconds=time.monotonic()-started,actual_models_constructed=counts['models'],actual_model_parameter_counts=parameters,
 actual_optimizers_constructed=counts['optimizers'],actual_optimizer_updates=counts['updates'],actual_cpu_forwards=forwards,
 actual_cuda_initialized=False,output_storage_kind='volatile tmpfs; engineering only',result=result,scientific_eligible=False,full_paper_complete=False)
with(ROOT/('private/phase-'+mode+'.json')).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(mode=mode,seconds=value['seconds'],counts=counts,parameters=parameters)),flush=True)
