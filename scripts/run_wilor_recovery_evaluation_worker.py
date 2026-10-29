"""One fresh owned evaluation phase; request and all sources are frozen by caller."""
from pathlib import Path
import hashlib,json,os,socket,sys,time

if sys.flags.optimize:raise RuntimeError('normal Python required for evaluation evidence')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));mode=sys.argv[1];started=time.monotonic()
q=json.loads((ROOT/'private/request.json').read_text());stage=json.loads((ROOT/'private/stage-receipt.json').read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check():
    for n,h in stage['files_sha256'].items():assert sha(ROOT/n)==h,n
check();assert type(q['require_full'])is bool
assert mode in('val-native','val-replay','freeze','test-native','test-replay','report','verify')
assert os.environ['CUDA_VISIBLE_DEVICES']==q['gpu_uuid']
def offline(*a,**kw):raise AssertionError('recovered native evaluation must stay offline')
socket.socket.connect=offline;socket.create_connection=offline
import torch
torch.set_num_threads(2)
from egoemg_repro import wilor_literal_runtime as runtime,wilor_recovery_eval as api
counts=dict(models=0,optimizers=0,updates=0);parameters=[];forwards=[]
factory=runtime.build_wilor_runtime;optimizer_init=torch.optim.Optimizer.__init__;step=torch.optim.AdamW.step
def counted_factory(*a,**kw):
    result=factory(*a,**kw);counts['models']+=1;parameters.append(sum(p.numel()for p in result.module.parameters()));original=result.module.forward
    def observed(batch):
        out=original(batch);device=batch['vision_img'].device.type
        assert device==next(result.module.parameters()).device.type==out[0].device.type
        forwards.append(dict(samples=len(batch['vision_img']),device_type=device,prediction_dtype=str(out[0].dtype),prediction_shape=list(out[0].shape)))
        return out
    result.module.forward=observed;return result
def counted_optimizer(*a,**kw):counts['optimizers']+=1;return optimizer_init(*a,**kw)
def counted_step(*a,**kw):counts['updates']+=1;return step(*a,**kw)
runtime.build_wilor_runtime=counted_factory;torch.optim.Optimizer.__init__=counted_optimizer;torch.optim.AdamW.step=counted_step
out=Path(q['evaluation_root']);fit=Path(q['fit']);out.mkdir(mode=0o700,exist_ok=True)
if mode in('val-native','val-replay','test-native','test-replay'):
    role,kind=mode.split('-')
    result=api.run_wilor_recovery_pass(fit,out/mode,role=role,kind=kind,
        reference=out/(role+'-native')if kind=='replay'else None,profile=out/'profile'if role=='test'else None,require_full=q['require_full'])
    assert counts==dict(models=1,optimizers=0,updates=0)and parameters==[631631657]
    assert result['full_training_terminated']is q['require_full']
    assert len(forwards)==len(result['statistics']['batch_sizes'])
elif mode=='freeze':
    result=api.freeze_wilor_recovery_validation(out/'val-native',out/'val-replay',out/'profile')
elif mode=='report':
    result=api.publish_wilor_recovery_report(fit,out/'test-native',out/'test-replay',out/'report',require_full=q['require_full'])
else:
    result=api.verify_wilor_recovery_report(out/'test-replay',out/'report',require_full=q['require_full'])
    assert result['scientific_eligible']is q['require_full']
    assert result['recovery']['uninterrupted_execution']is False
    if not q['require_full']:
        from egoemg_repro import wilor_native_eval
        try:wilor_native_eval.read_wilor_pass(out/'val-native')
        except ValueError:pass
        else:raise AssertionError('closed uninterrupted reader accepted recovered evaluation format')
if mode in('freeze','report','verify'):assert counts==dict(models=0,optimizers=0,updates=0)and not forwards
assert all(x['device_type']==('cuda'if q['require_full']else'cpu')for x in forwards)
assert torch.cuda.is_initialized()is q['require_full'] if forwards else not torch.cuda.is_initialized()
check()
value=dict(format='actual_recovered_wilor_evaluation_phase_v1',mode=mode,source_commit=stage['source_commit'],bundle_sha256=stage['bundle_sha256'],
    seconds=time.monotonic()-started,actual_models_constructed=counts['models'],actual_optimizers_constructed=counts['optimizers'],actual_optimizer_updates=counts['updates'],
    actual_model_parameter_counts=parameters,actual_forwards=forwards,actual_cuda_initialized=torch.cuda.is_initialized(),result=result,
    scientific_eligible=False,full_paper_complete=False)
with(ROOT/('private/phase-'+mode+'.json')).open('x')as handle:json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(json.dumps(dict(mode=mode,seconds=value['seconds'],counts=counts,parameters=parameters)),flush=True)
