from pathlib import Path
import copy,datetime,hashlib,json,shlex,subprocess
q=json.loads(Path('tmp/rn18-dense-completed-producer-footprints-launch-20261102.json').read_text())
CODE=r'''from pathlib import Path
import datetime,hashlib,json,os,sys
q=json.load(sys.stdin);r=Path(q['root'])
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def closed(a):
 p=Path('/proc',str(a['pid']),'stat');assert not p.exists()or p.read_text().rsplit(')',1)[1].split()[19]!=str(a['start_ticks'])
s=json.loads((r/'private/execution-state.json').read_text());m=json.loads((r/'private/stage-receipt.json').read_text());req=json.loads((r/'private/request.json').read_text())
assert s['status']=='complete'and s['actual_exit_code']==0
for key in ('wrapper','child'):assert s[key]==q['actual_state'][key];closed(s[key])
assert m['bundle_sha256']==q['bundle_sha256']==hashlib.sha256(json.dumps({k:v for k,v in m.items()if k!='bundle_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
for n,h in m['files_sha256'].items():assert sha(r/n)==h
w=json.loads((r/'private/workflow-result.json').read_text());assert w['source_commit']==q['source_commit']and w['bundle_sha256']==q['bundle_sha256']
assert len(w['actual_owned_stages'])==2 and w['all_owned_consumers_exited']and w['actual_independent_boolean_mask_checks']==18
for stage in w['actual_owned_stages']:
 assert stage['status']=='success'and stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']
 for key in ('worker','guardian'):closed(stage[key])
for j,label in enumerate(('a','b')):
 p=r/('private/audit-'+label+'.json');assert sha(p)==w['audit_files_sha256'][j]and json.loads(p.read_text())==w['audit']
 log=json.loads((r/('private/owned-'+label+'/worker.log')).read_text().splitlines()[-1]);assert not log['torch_imported']and log['full_boolean_mask_checks']==9 and log['hand_requests']==11114
v=w['audit'];assert v['checkpoint_bytes_verified']and not v['checkpoint_tensors_admitted']and not v['dense_input_or_inference_permit']
for name,h in v['actual_bound_source_files_sha256'].items():assert sha(name)==h
accepted=json.loads((r/'private/accepted-fusion.json').read_text());assert sha(req['postexit_readback'])==accepted['postexit_readback_file_sha256']
source=json.loads(Path(req['postexit_readback']).read_text());fit=Path(source['source_root'])/'private/experiment/full-fit';fi=json.loads((fit/'identity.json').read_text());fr=json.loads((fit/'fit-result.json').read_text());d=fi['initializer_descriptor'];emgroot=Path(d['producers']['emg_fit']);er=json.loads((emgroot/'fit-result.json').read_text())
paths=dict(fusion=fit/'checkpoints'/fr['selected_best']['file'],vision_export=Path(d['files']['vision_checkpoint']),emg_export=Path(d['files']['emg_checkpoint']),vision_original_LAST=Path(d['producers']['vision_fit'])/'last.pt',emg_original_BEST=emgroot/'checkpoints'/er['selected_best']['file'])
for name,p in paths.items():assert not p.is_symlink()and p.stat().st_size==v['actual_selected_weights_sha256'][name]['bytes']and sha(p)==v['actual_selected_weights_sha256'][name]['sha256']
ix=json.loads((r/'private/accepted-index.json').read_text());ip=ix['recipes']['rn18_s'];index=Path(req['index_root']);assert sha(index/'private/rn18_s-a/identity.json')==ip['identity_files_sha256'][0]and sha(index/'private/rn18_s-a/points.npz')==ip['point_files_sha256'][0]
for name,filename in [('manifest','manifest.json'),('metadata','metadata.npz'),('frame_split_id','frame_split_id.dat')]:assert sha(Path(req['data'])/filename)==ip['input_files_sha256'][name]
assert 'torch'not in sys.modules
p=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_terminal_state=s,raw_workflow=w,raw_workflow_file_sha256=sha(r/'private/workflow-result.json'),all6_original_actors_physically_closed=True,all_capsule_and_aggregate_bytes_after_exit_exact=True,full_bound_metadata_and5weight_bytes_fresh_postexit_exact=True,full_index_inputs_and_points_fresh_postexit_exact=True,collector_torch_unimported=True,collector_actor=dict(pid=os.getpid(),start_ticks=p[19]))))
'''
ssh=['ssh','-S','/home/zshyc/.ssh/egoemg-runtime-20261101.sock','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=15','eeg-server']
p=subprocess.run([*ssh,"env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/shihao/projects/EgoEMG-Reproduction/.runtime/env/bin/python -B -c "+shlex.quote(CODE)],input=json.dumps(q),capture_output=True,text=True,timeout=45)
Path('tmp/rn18-dense-completed-producer-footprints-collection-20261102.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-2000:]
v=json.loads(p.stdout);actor=v['collector_actor'];c='from pathlib import Path;import json;a='+repr(actor)+';p=Path("/proc",str(a["pid"]),"stat");print(json.dumps(dict(closed=not p.exists()or p.read_text().rsplit(")",1)[1].split()[19]!=a["start_ticks"])))'
z=subprocess.run([*ssh,'python3 -B -c '+shlex.quote(c)],capture_output=True,text=True,timeout=20);assert z.returncode==0 and json.loads(z.stdout)['closed'];v['collector_physically_closed_after_exit0']=True
Path('tmp/rn18-dense-completed-producer-footprints-verified-20261102.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
a=copy.deepcopy(v['raw_workflow']['audit']);a['actual_bound_metadata_file_sha256']=sorted(a.pop('actual_bound_source_files_sha256').values())
proof=dict(format='actual_closed_RN18_completed_producer_weights_and_dense_footprints_v1',source_commit=q['source_commit'],bundle_sha256=q['bundle_sha256'],actual_terminal_state=v['actual_terminal_state'],raw_workflow_file_sha256=v['raw_workflow_file_sha256'],actual_complete_fresh_audits=2,actual_independent_full_boolean_mask_checks=18,all6_original_actors_physically_closed=True,all_capsule_and_aggregate_bytes_after_exit_exact=True,full_bound_metadata_and5weight_bytes_fresh_postexit_exact=True,full_index_inputs_and_points_fresh_postexit_exact=True,collector_actor=actor,collector_physically_closed_after_exit0=True,collector_torch_unimported=True,audit=a,checkpoint_tensors_admitted=False,trained_model_dispatch_validated=False,dense_input_or_inference_permit=False,learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,figure13_reproduced=False,full_paper_complete=False)
text=json.dumps(proof,sort_keys=True,indent=2)+'\n';assert '/data/shihao'not in text and '/home/shihao'not in text
with Path('results/native-dense-occlusion/rn18-source-audit-20261102/completion.json').open('x')as f:f.write(text)
print(json.dumps(dict(status='actual-full-RN18-trained-producer-footprint-and-weight-bytes-closed',audits=2,boolean_masks=18,hand_requests=a['hand_requests'],postexit_closed=True,training_overlaps={n:p['roles']['train']for n,p in a['producers'].items()},full_paper_complete=False)))
