from pathlib import Path
import datetime,hashlib,json,os,shutil,sys,time
R=Path(__file__).resolve().parents[1];q=json.loads((R/'private/request.json').read_text());P=Path(q['root']);size=sys.argv[1];assert size in('small','middle','large');F=P/('private/'+size+'-flow');sys.path.insert(0,str(P));started=time.monotonic()
import torch
torch.set_num_threads(2)
attempts=dict(learned_module=0,optimizer=0)
def no_model(*a,**kw):attempts['learned_module']+=1;raise RuntimeError('postexit trajectory reader cannot construct learned modules')
def no_optimizer(*a,**kw):attempts['optimizer']+=1;raise RuntimeError('postexit trajectory reader cannot construct optimizers')
torch.nn.Module.__init__=no_model;torch.optim.Optimizer.__init__=no_optimizer
from egoemg_repro.emgformer_trajectory_eval import verify_trajectory_report
from egoemg_repro.emgformer_cohort import _paths
from egoemg_repro.data import LegacyStore
from egoemg_repro.emg2pose_memory_process import process_identity,same_process

def sha(p):
 with Path(p).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def h(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sealed(v):return h({k:x for k,x in v.items()if k!='sha256'})
def nine(p):
 p=Path(p);assert p.is_file()and not any(x.is_symlink()for x in(p,*p.parents));s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
def closed(handle):
 now=process_identity(handle['pid']);return now is None or now['start_ticks']!=str(handle['start_ticks'])or not same_process(now)
def write(p,v):
 with p.open('x')as stream:json.dump(v,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
m=json.loads((P/'private/stage-receipt.json').read_text());assert h({k:v for k,v in m.items()if k!='bundle_sha256'})==m['bundle_sha256']==q['bundle_sha256']and m['source_commit']==q['source_commit']
assert all(sha(P/name)==digest for name,digest in m['files_sha256'].items())
request=json.loads((P/'private/request.json').read_text());done=json.loads((P/('private/'+size+'-actual-complete.json')).read_text());outer=json.loads((P/('private/owned-'+size+'/stage-result.json')).read_text());flow=json.loads((F/'workflow-result.json').read_text())
assert done['size']==size and done['owned_stage']==outer and done['scientific_eligible']and not done['full_paper_complete']
assert outer['status']=='success'and outer['worker_exit_code']==0 and outer['all_owned_consumers_exited']
actors=[]
for k in('worker','guardian'):assert closed(outer[k]);actors.append(outer[k])
assert flow['sha256']==sealed(flow)and flow['trajectory_pipeline_complete']and flow['scientific_eligible']and flow['all_owned_consumers_exited']and flow['parent_torch_unimported']and not flow['full_paper_complete']
names=('val-native','val-replay','freeze','test-native','test-replay','report','verify');assert len(flow['actual_owned_stages'])==len(names)==7
for name,stage in zip(names,flow['actual_owned_stages']):
 assert stage==json.loads((F/('owned-'+name+'/stage-result.json')).read_text())
 assert stage['status']=='success'and stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']
 assert all(stage['owner'][k]==outer['worker'][k]for k in('pid','start_ticks','uid','sid','pgrp'))
 for k in('worker','guardian'):assert closed(stage[k]);actors.append(stage[k])
assert len(actors)==16
assert done['report']['report_file_sha256']==flow['report_file_sha256']==sha(F/'report/metrics.json')and done['report']['workflow_file_sha256']==sha(F/'workflow-result.json')
files=sorted({*(P/name for name in m['files_sha256']),P/'private/stage-receipt.json',P/('private/'+size+'-actual-complete.json'),P/('private/owned-'+size+'/started.json'),P/('private/owned-'+size+'/stage-result.json'),*(p for p in F.rglob('*')if p.is_file())})
stamps={str(p):nine(p)for p in files}
passes={name:json.loads((F/name/'pass.json').read_text())for name in('val-native','val-replay','test-native','test-replay')}
identity=passes['val-native']['trajectory_input_identity'];assert identity['sha256']==sealed(identity)and identity['counts']==dict(val=1052,test=2792)and identity['repeat_by_role']==dict(val=1,test=1)
inputs=_paths(LegacyStore(request['data']));assert set(inputs)==set(identity['payload_files_sha256'])and len(inputs)==identity['payload_file_count']
input_stamps={key:nine(path)for key,path in inputs.items()}
assert {key:stamp[2]for key,stamp in input_stamps.items()}==identity['payload_file_bytes']
# The unchanged complete reader physically checks all input and raw packet SHA,
# selected full states, all4 source passes, pretest profile and promotion closure.
report=verify_trajectory_report(F/'test-replay',F/'report')
assert report['sha256']==sealed(report)and report['size']==size and report['scientific_eligible']and report['trajectory_pipeline_complete']and not report['full_paper_complete']and not report['paper_numerical_equivalence_established']
assert report['training']['scope']=='full'and report['training']['full_training_terminated']and report['training']['completed_epochs']==200 and report['training']['optimizer_executions']==25000
assert report['fit_result_sha256']==request['fits'][size]['fit_result_sha256']==done['report']['fit_result_sha256']
assert report['training']['selected_best']['checkpoint_sha256']==request['fits'][size]['selected_checkpoint_sha256']
assert report['source_identity_sha256']==identity['sha256']==done['report']['trajectory_input_sha256']
raw_hashes={};phase_summary={}
for name,value in passes.items():
 role=value['role'];samples=1052 if role=='val'else 2792
 assert value['formal_evaluation']and value['require_full']and value['samples']==samples and value['batch_size']==500 and value['workers']==6 and value['precision']=='bf16-mixed'
 assert value['all_loader_workers_exited']and len(value['loader_workers'])==6
 for handle in value['loader_workers']:assert closed(handle)
 assert value['statistics']['batch_sizes']==([500,500,52]if role=='val'else[500,500,500,500,500,292])
 phase_summary[name]=dict(actor=value['actor'],samples=samples,workers=6,batch_size=500,actual_learned_forwards=value['actual_model_forwards'],actual_optimizer_updates=value['actual_optimizer_updates'],precision=value['precision'],alignment=value['statistics']['alignment'],all_loader_workers_closed=True)
 for line in(F/name/'packets.jsonl').read_text().splitlines():
  record=json.loads(line);assert record['sha256']==sealed(record)
  payload=F/name/'raw'/record['file'];assert str(payload)in stamps
  raw_hashes[str(payload)]=record['file_sha256']
assert attempts==dict(learned_module=0,optimizer=0)and not torch.cuda.is_initialized()
assert all(nine(path)==input_stamps[key]for key,path in inputs.items())and all(nine(Path(path))==stamp for path,stamp in stamps.items())
small_artifacts={str(p):sha(p)for p in files if p.suffix!='.pt'}
assert all(nine(Path(path))==stamp for path,stamp in stamps.items())
export=R/('private/export/'+size);export.mkdir(mode=0o700,parents=True);exported={}
for name,src in [('metrics.json',F/'report/metrics.json'),('metrics.pending.json',F/'report/metrics.pending.json'),('metrics.md',F/'report/metrics.md'),('completion.json',F/'report/completion.json'),('verification.json',F/'verified/verification.json'),('workflow-result.json',F/'workflow-result.json')]:
 dst=export/name;shutil.copyfile(src,dst);assert sha(src)==sha(dst);exported[name]=sha(dst)
value=dict(format='actual_native_emgformer_complete_trajectory_model_free_postexit_v1',size=size,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actor=process_identity(),original_source_commit=m['source_commit'],original_bundle_sha256=m['bundle_sha256'],
 original_outer_size_stage=outer,all16_original_size_actors_closed=actors,all24_original_loader_workers_closed=True,
 actual_complete_original_role_passes=phase_summary,actual_report=report,exported_report_files_sha256=exported,
 full_input_payload_files=len(inputs),full_input_payload_bytes=sum(v[2]for v in input_stamps.values()),full_input_payload_hashes_verified_by_existing_native_reader=identity['payload_files_sha256'],full_input_payload_nine_attributes=input_stamps,
 raw_tensor_payload_sha256_physically_verified_by_existing_native_reader=raw_hashes,raw_and_other_artifact_nine_attributes=stamps,
 non_tensor_artifact_files_after_exit_sha256=small_artifacts,all_frozen_source_files_exact=True,
 source_parent_queue_may_still_run_other_sizes=True,source_parent_queue_not_required_closed_for_this_completed_size=True,
 construction_attempts=attempts,cuda_initialized=False,new_learned_forwards=0,new_optimizer_updates=0,seconds=time.monotonic()-started,
 scientific_eligible=True,paper_numerical_equivalence_established=False,full_paper_complete=False)
write(R/('private/'+size+'-postexit-readback.json'),value)
print(json.dumps(dict(status='complete-native-trajectory-size-reopened',size=size,seconds=value['seconds'])),flush=True)
