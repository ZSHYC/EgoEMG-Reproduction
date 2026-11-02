from pathlib import Path
import hashlib,json,os,shutil,sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));q=json.loads((R/'private/request.json').read_text());P=Path(q['root'])
from egoemg_repro.emg2pose_memory_process import process_identity,same_process,run_owned_stage
parent=process_identity(os.getppid());m=json.loads((R/'private/stage-receipt.json').read_text());pm=json.loads((P/'private/stage-receipt.json').read_text())
def live(h):
 now=process_identity(h['pid']);return now is not None and now['uid']==os.getuid()and now['start_ticks']==str(h['start_ticks'])and same_process(now)
def sha(p):
 with Path(p).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def nine(p):
 assert p.is_file()and not any(a.is_symlink()for a in(p,*p.parents));s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
for root,packet in((R,m),(P,pm)):
 assert hashlib.sha256(json.dumps({k:v for k,v in packet.items()if k!='bundle_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()==packet['bundle_sha256']
 assert all(sha(root/name)==digest for name,digest in packet['files_sha256'].items())
assert pm['source_commit']==q['source_commit']and pm['bundle_sha256']==q['bundle_sha256']
paths=[R/name for name in m['files_sha256']]+[R/'private/stage-receipt.json']+[P/name for name in pm['files_sha256']]+[P/'private/stage-receipt.json'];stamps={str(p):nine(p)for p in paths}
def guard():
 assert'torch'not in sys.modules and live(parent)and shutil.disk_usage('/data').free>225*1024**3
 assert int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024>128*1024**3
 assert all(nine(Path(p))==value for p,value in stamps.items())
print(json.dumps(dict(phase='waiting-exact-recovered69-rn18-fusion-full-workflow-closure',variant=q['variant'],target=str(P),deadline_seconds=None)),flush=True)
missing=None
while True:
 guard();state=json.loads((P/'private/execution-state.json').read_text());assert all(state[k]==q['actual_state'][k]for k in('wrapper','child'))
 if state['status']!='running':break
 if all(live(state[k])for k in('wrapper','child')):missing=None
 elif missing is None:missing=time.monotonic()
 else:assert time.monotonic()-missing<10,'target handle missing but no terminal record; do not restart'
 time.sleep(2)
assert state['status']=='complete'and state['actual_exit_code']==0,state
while any(live(state[k])for k in('wrapper','child')):guard();time.sleep(.1)
stage=run_owned_stage([sys.executable,'-u','-B',str(R/'private/worker.py')],R/'private/owned-postexit-readback',
 descriptors={},guard=guard,environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''),poll_seconds=2.,terminate_grace=10.)
assert stage['status']=='success'and stage['worker_exit_code']==0 and stage['all_owned_consumers_exited']
assert all(not same_process(stage[key])for key in('worker','guardian'));guard()
readback=json.loads((R/'private/postexit-readback.json').read_text());assert readback['construction_attempts']==dict(learned_module=0,optimizer=0)and not readback['cuda_initialized']
result=dict(format='owned_native_recovered69_rn18_fusion_independent_complete_collector_v1',variant=q['variant'],source_commit=m['source_commit'],bundle_sha256=m['bundle_sha256'],
 actual_owned_stages=[stage],all_owned_consumers_exited=True,parent_torch_unimported=True,
 postexit_readback_sha256=sha(R/'private/postexit-readback.json'),exported_report_files_sha256=readback['exported_report_files_sha256'],
 actual_full_fit_sha256=readback['actual_full_fit_sha256'],scientific_eligible=True,full_paper_complete=False)
with(R/'private/workflow-result.json').open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps(dict(status='independent-recovered69-rn18-fusion-complete-postexit-collector-closed',variant=q['variant'])),flush=True)
