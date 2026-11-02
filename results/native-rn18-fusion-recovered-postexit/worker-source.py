from pathlib import Path
import datetime,hashlib,json,os,shutil,sys,time
R=Path(__file__).resolve().parents[1];q=json.loads((R/'private/request.json').read_text());Q=Path(q['root']);P=Path(q['source_root']);A=Path(q['admission_root']);sys.path.insert(0,str(P));started=time.monotonic()
import torch
torch.set_num_threads(2)
attempts=dict(learned_module=0,optimizer=0)
def no_model(*args,**kwargs):attempts['learned_module']+=1;raise RuntimeError('independent final reader cannot construct learned modules')
def no_optimizer(*args,**kwargs):attempts['optimizer']+=1;raise RuntimeError('independent final reader cannot construct optimizers')
torch.nn.Module.__init__=no_model;torch.optim.Optimizer.__init__=no_optimizer
from egoemg_repro.emg2pose_memory_process import process_identity,same_process
from egoemg_repro.rn18_fusion_eval import verify_rn18_fusion_report,PUBLIC_FILES
from egoemg_repro.rn18_fusion_experiment import rn18_fusion_experiment_stages
from egoemg_repro.data import LegacyStore
from egoemg_repro.vision_literal_data import _crop_path

def sha(p):
 with Path(p).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def nine(p):
 p=Path(p);assert p.is_file()and not any(x.is_symlink()for x in(p,*p.parents));s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
def h(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sealed(v):return h({k:x for k,x in v.items()if k!='sha256'})
def closed(handle):
 now=process_identity(handle['pid']);return now is None or now['start_ticks']!=str(handle['start_ticks'])or not same_process(now)
def write(p,v):
 with p.open('x')as stream:json.dump(v,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

state=json.loads((Q/'private/execution-state.json').read_text());assert state['status']=='complete'and state['actual_exit_code']==0
assert all(state[k]==q['actual_state'][k]and closed(state[k])for k in('wrapper','child'))
qm=json.loads((Q/'private/stage-receipt.json').read_text());assert h({k:v for k,v in qm.items()if k!='bundle_sha256'})==qm['bundle_sha256']==q['bundle_sha256']and qm['source_commit']==q['source_commit']
request_new=json.loads((Q/'private/request.json').read_text());assert request_new['source_root']==str(P)and request_new['admission_root']==str(A)
original_state=json.loads((P/'private/execution-state.json').read_text());assert original_state==request_new['source_terminal_state']and original_state['status']=='failed'and original_state['actual_exit_code']==1
m=json.loads((P/'private/stage-receipt.json').read_text());assert h({k:v for k,v in m.items()if k!='bundle_sha256'})==m['bundle_sha256']
am=json.loads((A/'private/stage-receipt.json').read_text());assert h({k:v for k,v in am.items()if k!='bundle_sha256'})==am['bundle_sha256']
for root,manifest in((P,m),(Q,qm),(A,am)):
 assert all(sha(root/name)==digest for name,digest in manifest['files_sha256'].items())
assert sha(P/'egoemg_repro/fusion_literal_runtime.py')==sha(Q/'egoemg_repro/fusion_literal_runtime.py')=='3b357bdb0e41f3536c2296324917465ad25b2d13c8cccb553815fb729310ca60'
request=json.loads((P/'private/request.json').read_text());assert request_new['gpu_uuid']==request['gpu_uuid']
E=P/'private/experiment';full=E/'full-fit';evaluation=E/'full-evaluation'
w=json.loads((Q/'private/workflow-result.json').read_text());e=w
assert w['format']=='actual_native_rn18_fusion_resource_recovered69_to150_and_original_full_evaluation_v1'
assert w['source_commit']==qm['source_commit']and w['source_bundle_sha256']==qm['bundle_sha256']and w['original_failed_source_state']==original_state
assert w['all_owned_consumers_exited']and w['parent_torch_unimported']and w['actual_qualification_completed']and w['scientific_eligible']and not w['paper_numerical_equivalence_established']and not w['full_paper_complete']
assert w['new_qualification_runs']==0 and w['formal_fusion_epochs']==150 and w['formal_optimizer_updates']==46950 and w['original_experiment_output']==str(E)
plan=json.loads((E/'experiment-plan.json').read_text());graph=rn18_fusion_experiment_stages(request['experiment'],E);assert plan['stages']==graph and len(graph)==20 and graph[11]['name']=='full-fit'
assert w['actual_owned_stages']==w['actual_prior11_qualification_stages']+w['actual_recovered9_full_stages']and len(w['actual_prior11_qualification_stages'])==11 and len(w['actual_recovered9_full_stages'])==9
assert set(w['actual_nvml_observations'])=={v['name']for v in graph[11:]if v['cuda']}
actors=[state[k]for k in('wrapper','child')]+[original_state[k]for k in('wrapper','child')]
failed=json.loads((E/'stages/owned-full-fit/stage-result.json').read_text());assert failed['status']=='failed'and failed['reason']=='owner_abort'and failed['all_owned_consumers_exited']and failed['worker_exit_code']!=0
actors.extend(failed[k]for k in('worker','guardian'))
for index,(spec,stage)in enumerate(zip(graph,w['actual_owned_stages'])):
 directory=(E/'stages')if index<11 else(Q/'private/stages');assert stage==json.loads((directory/('owned-'+spec['name'])/'stage-result.json').read_text())
 assert stage['status']=='success'and stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']
 expected_owner=original_state['child']if index<11 else state['child']
 assert all(str(stage['owner'][k])==str(expected_owner[k])for k in('pid','start_ticks'))
 for key in('worker','guardian'):assert closed(stage[key]);actors.append(stage[key])
 if index>=11 and spec['cuda']:
  observation=w['actual_nvml_observations'][spec['name']];assert observation['worker']==stage['worker']and observation['actual_nvml_pid']==stage['worker']['pid']and observation['gpu_uuid']==request['gpu_uuid']
a_state=json.loads((A/'private/execution-state.json').read_text());assert a_state['status']=='complete'and a_state['actual_exit_code']==0
aw=json.loads((A/'private/workflow-result.json').read_text());admission=json.loads((A/'private/checkpoint-admission.json').read_text());assert sha(A/'private/checkpoint-admission.json')==aw['checkpoint_admission_file_sha256']==w['preserved_checkpoint_admission_file_sha256']
assert admission['last_completed_epochs']==69 and admission['last_durable_updates']==21597 and admission['complete_saved_checkpoint_domains_validated']and admission['all_source_generations_unchanged']
assert aw['actual_owned_stage']['status']=='success'and aw['actual_owned_stage']['worker_exit_code']==0 and aw['actual_owned_stage']['all_owned_consumers_exited']
actors.extend(a_state[k]for k in('wrapper','child'));actors.extend(aw['actual_owned_stage'][k]for k in('worker','guardian'))
assert len(actors)==len({(a['pid'],str(a['start_ticks']))for a in actors})==50
for a in actors:assert closed(a)
assert all(sha(A/'private/preserved-source-fit'/name)==v['sha256']for name,v in admission['full_checkpoint_archives_sha256'].items())
proof=json.loads((P/'private/actual-full-trained-cpu-postexit.json').read_text());review=json.loads((P/'private/whole-family-review-accepted.json').read_text())
assert proof['ready']and proof['full_initializer_positive_admission_verified']and proof['postexit_reader_also_closed']
assert review['status']=='accepted'and review['critical_findings_remaining']==review['important_findings_remaining']==0
assert all(sha(P/name)==digest for name,digest in review['reviewed_sources_sha256'].items())
files=sorted({*(root/'private/stage-receipt.json'for root in(P,Q,A)),*(root/name for root,manifest in((P,m),(Q,qm),(A,am))for name in manifest['files_sha256']),*(p for root in(P,Q,A)for p in(root/'private').rglob('*')if p.is_file())})
stamps={str(p):nine(p)for p in files}
identity=json.loads((full/'identity.json').read_text());assert identity['sha256']==sealed(identity)
cohort=identity['cohort'];assert cohort['counts']==dict(train=62470,val=1052,test=2792)and cohort['registered_full_legacy']and cohort['repeat_by_role']==dict(train=1,val=1,test=1)
store=LegacyStore(identity['data_locations']['data']);inputs={}
for key in cohort['payload_files_sha256']:
 kind,name=key.split(':',1)
 if kind=='metadata':assert name in('manifest','episodes');path=store.manifest_path if name=='manifest'else store.metadata_path
 elif kind=='field':path=Path(store.field(name).filename)
 else:
  assert kind=='crop';path=_crop_path(Path(identity['data_locations']['crops']),store.episodes[int(name)]['id']);path=path/'data.mdb'if path.is_dir()else path
 inputs[key]=path
assert len(inputs)==cohort['payload_file_count']==53
input_stamps={key:nine(path)for key,path in inputs.items()};assert sum(stamp[2]for stamp in input_stamps.values())==61377230438
assert all(sha(path)==cohort['payload_files_sha256'][key]for key,path in inputs.items())
# This real existing reader reconstructs complete fit/checkpoint/producer/input,
# frozen validation, both typed test passes, report and every curve byte.
report=verify_rn18_fusion_report(evaluation/'test-replay',evaluation/'report',require_full=True)
fit=json.loads((full/'fit-result.json').read_text());qual=json.loads((E/'qual-fit/fit-result.json').read_text())
assert fit['sha256']==sealed(fit)==report['training']['fit_result_sha256']
assert fit['scope']=='full'and fit['completed_epochs']==150 and fit['optimizer_executions']==46950 and fit['full_training_terminated']and fit['stage_complete']
assert fit['train_samples_per_epoch']==[62470]*150 and fit['val_samples_per_epoch']==[1052]*150
assert fit['resume_start_exact']=={k:True for k in('model','optimizer','scheduler','callbacks','loops','rng')}and fit['resume_validation_exact']is True
assert fit['resume_evidence']['parent_checkpoint_sha256']==admission['full_checkpoint_archives_sha256']['checkpoints/last.ckpt']['sha256']
assert fit['completion_recovered_from_native_checkpoint']is False if 'completion_recovered_from_native_checkpoint'in fit else True
assert qual['sha256']==sealed(qual)and qual['scope']=='cuda_engineering'and qual['completed_epochs']==2 and qual['optimizer_executions']==4
assert qual['resume_start_exact']=={k:True for k in('model','optimizer','scheduler','callbacks','loops','rng')}and qual['resume_validation_exact']is True
assert report==e['actual_full_report']and report['sha256']==sealed(report)and report['scientific_eligible']and not report['paper_numerical_equivalence_established']and not report['full_paper_complete']
assert report['variant']=='fusion_rn18_s_center_8ch'and report['test']['consumed_samples']==report['test']['unique_samples']==2792 and report['test']['duplicate_consumptions']==0
assert report['saved_fit_validation']['pose_exact_match']is True
assert attempts==dict(learned_module=0,optimizer=0)and not torch.cuda.is_initialized()
assert all(nine(path)==input_stamps[key]for key,path in inputs.items())and all(nine(Path(path))==stamp for path,stamp in stamps.items())
artifacts={str(p):sha(p)for p in files};assert all(nine(Path(path))==stamp for path,stamp in stamps.items())
export=R/'private/export';export.mkdir(mode=0o700);exported={}
for name in(*PUBLIC_FILES,'completion.json'):
 src=evaluation/'report'/name;dst=export/name;shutil.copyfile(src,dst);assert sha(src)==sha(dst);exported[name]=sha(dst)
result=dict(format='actual_recovered69_native_rn18_fusion_full_model_free_postexit_readback_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actor=process_identity(),
 source_root=str(P),source_commit=m['source_commit'],bundle_sha256=m['bundle_sha256'],original_terminal_state=original_state,recovery_root=str(Q),recovery_source_commit=qm['source_commit'],recovery_bundle_sha256=qm['bundle_sha256'],recovery_terminal_state=state,preserved69_checkpoint_admission_sha256=sha(A/'private/checkpoint-admission.json'),
 all50_failed_original_recovered_and_admission_actors_closed=actors,actual_current_original_stages=20,all_current_frozen_source_files_exact=True,
 all_closed_artifact_sha256=artifacts,all_closed_artifact_nine_attributes=stamps,
 actual_full_source_payload_files=53,actual_full_source_payload_bytes=61377230438,full_source_payload_sha256=cohort['payload_files_sha256'],full_source_payload_nine_attributes=input_stamps,
 actual_full_fit_sha256=fit['sha256'],completed_epochs=150,optimizer_executions=46950,qualified_native_batch=200,qualified_native_workers=10,
 qualification_six_domain_resume_exact=qual['resume_start_exact'],formal_resume_start_exact=fit['resume_start_exact'],
 original_qualification_not_reexecuted=True,prior_qualification_nvml_map_not_reconstructed=True,actual_report=report,exported_report_files_sha256=exported,
 actual_whole_family_review_accepted=True,actual_full_trained_handoff_verified=True,
 construction_attempts=attempts,cuda_initialized=False,new_learned_forwards=0,new_optimizer_updates=0,
 scientific_eligible=True,paper_numerical_equivalence_established=False,full_paper_complete=False,seconds=time.monotonic()-started)
write(R/'private/postexit-readback.json',result)
print(json.dumps(dict(status='complete-native-rn18-fusion-report-checkpoints-producers-and-sources-reopened',epochs=150,updates=46950,seconds=result['seconds'])),flush=True)
