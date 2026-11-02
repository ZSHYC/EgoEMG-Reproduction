from pathlib import Path
import datetime,hashlib,json,os,sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from egoemg_repro.emg2pose_memory_process import run_owned_stage,same_process,process_identity
from egoemg_repro.classic_emg_cohort import _StatGuard
m=json.loads((R/'private/stage-receipt.json').read_text());parent=process_identity(os.getppid());began=time.monotonic()
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
pins=_StatGuard([*[R/n for n in m['files_sha256']],R/'private/stage-receipt.json'])
def guard():assert 'torch'not in sys.modules and same_process(parent);pins.check()
for n,h in m['files_sha256'].items():assert sha(R/n)==h
owned=[]
for label in('a','b'):
 guard();print(json.dumps(dict(phase=label,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
 stage=run_owned_stage([sys.executable,'-B',str(R/'private/audit-cli.py'),'audit-'+label+'.json'],R/('private/owned-'+label),descriptors={},guard=guard,environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),poll_seconds=2.,terminate_grace=10.)
 assert stage['status']=='success'and stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']and all(not same_process(stage[k])for k in('worker','guardian'))
 owned.append(stage);guard()
a=json.loads((R/'private/audit-a.json').read_text());b=json.loads((R/'private/audit-b.json').read_text());assert a==b
for n,h in m['files_sha256'].items():assert sha(R/n)==h
result=dict(format='actual_two_fresh_rn18_completed_producer_footprint_audits_v1',source_commit=m['source_commit'],bundle_sha256=m['bundle_sha256'],actual_owned_stages=owned,actual_fresh_complete_footprint_audits=2,actual_independent_boolean_mask_checks=2*a['actual_independent_full_boolean_mask_checks'],exact_audit_results_rebuilt=True,all_owned_consumers_exited=True,source_bytes_after_exit_exact=True,audit=a,audit_files_sha256=[sha(R/'private/audit-a.json'),sha(R/'private/audit-b.json')],learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,checkpoint_tensors_admitted=False,native_geometry_executed=False,figure13_reproduced=False,full_paper_complete=False,seconds=time.monotonic()-began)
with(R/'private/workflow-result.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='two-fresh-full-producer-footprint-audits-closed')),flush=True)
