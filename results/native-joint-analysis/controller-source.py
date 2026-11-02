from pathlib import Path
import datetime,hashlib,json,os,shutil,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from egoemg_repro.emg2pose_memory_process import process_identity,same_process,run_owned_stage
q=json.loads((R/'private/request.json').read_text());m=json.loads((R/'private/stage-receipt.json').read_text());parent=process_identity(os.getppid())
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seal(v):return dict(v,sha256=digest(v))
def write(p,v):
 with p.open('x')as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def nine(p):
 assert p.is_file()and not any(v.is_symlink()for v in(p,*p.parents));s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
def closed(v):
 f=Path('/proc',str(v['pid']),'stat')
 return not f.exists()or f.read_text().rsplit(')',1)[1].split()[19]!=str(v['start_ticks'])
assert digest({k:v for k,v in m.items()if k!='bundle_sha256'})==m['bundle_sha256']
assert all(sha(R/name)==h for name,h in m['files_sha256'].items())
stamps={str(R/name):nine(R/name)for name in m['files_sha256']};stamps[str(R/'private/stage-receipt.json')]=nine(R/'private/stage-receipt.json')
def guard():
 assert 'torch'not in sys.modules and 'numpy'not in sys.modules and same_process(parent)
 assert shutil.disk_usage('/data').free>=225*1024**3
 available=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))*1024
 assert available>=128*1024**3
 assert all(nine(Path(p))==stamp for p,stamp in stamps.items())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
script=str(R/'scripts/paper/collect_native_joint_analysis.py');results={};stages=[];population=None
for size in('small','middle','large'):
 source=q['sources'][size];output=R/('private/analysis/'+size);guard()
 argv=[sys.executable,'-u','-B',script,'--postexit-root',q['postexit_root'],'--size',size,'--metrics',str(R/source['metrics']),'--proof',str(R/source['proof']),'--expected-proof-sha256',source['proof_sha256'],'--upstream',q['upstream'],'--output',str(output)]
 stage=run_owned_stage(argv,R/('private/owned-'+size+'-collect'),descriptors={},guard=guard,environment=env,poll_seconds=2.,terminate_grace=10.)
 assert stage['status']=='success'and stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']and all(closed(stage[k])for k in('worker','guardian'));stages.append(stage);guard()
 private=output/'private-records.json';pending=output/'analysis.pending.json';p=json.loads(pending.read_text());records=json.loads(private.read_text())
 assert p['sha256']==digest({k:v for k,v in p.items()if k!='sha256'})and p['source_run_scientific_eligible']and not p['analysis_reader_complete']
 assert all(records['producer_actor'][k]==stage['worker'][k]for k in('pid','start_ticks','uid','sid','pgrp'))
 private_hash=sha(private);pending_hash=sha(pending)
 verifier=output/'verification.json';argv=[sys.executable,'-u','-B',script,'verify','--private-records',str(private),'--pending',str(pending),'--expected-private-sha256',private_hash,'--expected-pending-sha256',pending_hash,'--output',str(verifier)]
 verification_stage=run_owned_stage(argv,R/('private/owned-'+size+'-verify'),descriptors={},guard=guard,environment=env,poll_seconds=2.,terminate_grace=10.)
 assert verification_stage['status']=='success'and verification_stage['worker_exit_code']==0 and verification_stage['all_owned_consumers_exited']and all(closed(verification_stage[k])for k in('worker','guardian'));stages.append(verification_stage);guard()
 v=json.loads(verifier.read_text());assert v['sha256']==digest({k:x for k,x in v.items()if k!='sha256'})and v['stdlib_only']and v['all_cells_independently_recomputed']and v['producer_actor_physically_closed']and v['source_generations_unchanged']and not v['analysis_reader_complete']
 assert all(v['actual_verifier_actor'][k]==verification_stage['worker'][k]for k in('pid','start_ticks','uid','sid','pgrp'))
 assert sha(private)==private_hash and sha(pending)==pending_hash and v['private_records_file_sha256']==private_hash and v['pending_file_sha256']==pending_hash
 fields=('key','hand','split','private_metadata','sample_input_sha256','trajectory_angle_valid_counts');current=[{k:r[k]for k in fields}for r in records['records']]
 assert digest(current)==p['request_population_sha256']
 if population is None:population=current
 else:assert current==population,'all3 requested windows, targets, masks and hand/split identities must match before a joint table'
 completion=seal(dict(format='owned_closed_native_joint_analysis_completion_v1',size=size,source_commit=m['source_commit'],source_bundle_sha256=m['bundle_sha256'],accepted_source_proof_sha256=source['proof_sha256'],pending_file_sha256=pending_hash,private_records_file_sha256=private_hash,verification_file_sha256=sha(verifier),actual_owned_collection_stage=stage,actual_owned_verification_stage=verification_stage,all4analysis_actors_physically_closed=True,source_generations_unchanged=True,independent_stdlib_cells_recomputed=True,analysis_reader_complete=True,source_run_scientific_eligible=True,new_learned_forwards=0,new_optimizer_updates=0,paper_numerical_equivalence_established=False,full_paper_complete=False))
 write(output/'completion.json',completion)
 final={k:v for k,v in p.items()if k!='sha256'};final.update(analysis_reader_complete=True,completion_file_sha256=sha(output/'completion.json'));write(output/'analysis.json',seal(final))
 public=R/('private/export/'+size);public.mkdir(mode=0o700,parents=True)
 for name in('analysis.json','analysis.pending.json','completion.json','verification.json'):shutil.copyfile(output/name,public/name)
 results[size]=dict(source_report_sha256=p['source_report_sha256'],request_population_sha256=p['request_population_sha256'],n_requested=p['views']['n_requested'],source_generation_count=p['source_generation_count'],files_sha256={name:sha(public/name)for name in('analysis.json','analysis.pending.json','completion.json','verification.json')})
 print(json.dumps(dict(size=size,status='actual-native-joint-analysis-closed',requests=p['views']['n_requested'])),flush=True)
assert set(results)=={'small','middle','large'}and len({r['request_population_sha256']for r in results.values()})==1
value=seal(dict(format='actual_all_three_native_joint_analysis_owned_workflow_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=m['source_commit'],source_bundle_sha256=m['bundle_sha256'],actual_owned_stages=stages,actual_results=results,all12analysis_actors_physically_closed=True,all3_request_target_mask_hand_split_populations_equal=True,parent_torch_and_numpy_unimported=True,all_frozen_source_files_exact=True,new_learned_forwards=0,new_optimizer_updates=0,paper_numerical_equivalence_established=False,full_paper_complete=False))
guard();write(R/'private/workflow-result.json',value)
print(json.dumps(dict(status='all-three-native-joint-analyses-closed',requests_per_model=len(population))),flush=True)
