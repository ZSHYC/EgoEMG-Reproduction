
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time,shutil
R=Path(__file__).resolve().parent;q=json.loads((R/'config.json').read_text());P=Path(q['package']);sys.path.insert(0,str(P));os.nice(10)
from egoemg_repro.classic_emg_cohort import _StatGuard
manifest=json.loads((R/'stage-receipt.json').read_text());package=json.loads((P/'stage-receipt.json').read_text())
def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert all(sha(R/n)==h for n,h in manifest['files_sha256'].items())and all(sha(P/n)==h for n,h in package['files_sha256'].items())
guard=_StatGuard([R/n for n in manifest['files_sha256']]+[P/n for n in package['files_sha256']])
def check():
 guard.check();assert shutil.disk_usage('/data/shihao').free>=225*1024**3
 assert next(int(x.split()[1])for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))*1024>=128*1024**3
def actor(pid):
 f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=f[19])
def closed(a):
 p=Path('/proc',str(a['pid']),'stat');return not p.exists()or p.read_text().rsplit(')',1)[1].split()[19]!=a['start_ticks']
def write(path,value,replace=False):
 raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
 if replace:
  t=path.with_suffix('.tmp');t.write_text(raw);os.replace(t,path)
 else:
  with path.open('x')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
state=dict(status='running',phase='waiting-original-GPU-qualification-exit0-and-physical-closure',wrapper=actor(os.getpid()),started_utc=utc(),stages=[],full_paper_complete=False)
def save():write(R/'private/execution-state.json',state,True)
def interrupt(signum,frame):raise RuntimeError('post-qualification chain interrupted')
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',EGOEMG_UPSTREAM='/home/shihao/projects/EgoEMG-Reproduction/upstream')
env['PYTHONPATH']=str(P)+':/home/shihao/projects/EgoEMG-Classic-Canonical-20261026/.deps:/home/shihao/projects/EgoEMG-WiLoR-20261026/.deps'
save()
def run(name,args,on_start=None):
 check();state['phase']=name;save()
 with(R/'private'/(name+'.log')).open('xb')as log:
  p=subprocess.Popen(args,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  phase=dict(name=name,worker=actor(p.pid),status='running');state['stages'].append(phase);save()
  if on_start is not None:on_start(phase['worker'])
  try:
   while p.poll()is None:check();time.sleep(2)
   code=p.wait();check()
  finally:
   if p.poll()is None:
    p.terminate()
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  phase.update(status='complete'if code==0 else'failed',actual_exit_code=code);save()
  assert code==0 and closed(phase['worker']),name+' failed; inspect preserved original child; no automatic retry'
  return phase['worker']
try:
 G=Path(q['GPU_qualification']['root'])
 while True:
  check();p=G/'private/execution-state.json';s=json.loads(p.read_text())if p.exists()else None
  if s is not None:
   assert s['wrapper']==q['GPU_qualification']['actual_controller']
   if s['status']=='failed':raise RuntimeError('original GPU qualification terminal failure, no continuation')
   if s['status']=='complete'and s['actual_exit_code']==0:
    actors=[s['wrapper'],*[x['worker']for x in s['stages']]]
    if all(closed(a)for a in actors):break
   elif closed(q['GPU_qualification']['actual_controller']):raise RuntimeError('original GPU controller disappeared without complete output')
  elif closed(q['GPU_qualification']['actual_controller']):raise RuntimeError('original GPU controller absent without state')
  time.sleep(2)
 gpu_reader=run('accept-exact-six-GPU-qualification-packets',[sys.executable,'-u','-B',str(R/'accept-gpu.py'),str(R)])
 a=json.loads((R/'private/GPU-acceptance.json').read_text());assert a['reader']==gpu_reader and closed(gpu_reader)
 full_config=json.loads((R/'full-config-template.json').read_text());full_config['gpu_qualification']=dict(a['accepted_registration'],independent_acceptance_reader=gpu_reader,independent_acceptance_reader_physically_closed=True)
 F=Path(q['full_root']);assert not F.exists();F.mkdir(mode=0o700);(F/'private').mkdir(mode=0o700)
 write(F/'config.json',full_config)
 with(F/'controller.py').open('xb')as stream:stream.write((R/'full-controller.py').read_bytes())
 fm=dict(format='actual_six_complete_native_center_reference_capsule_v1',source_commit=q['source_commit'],files_sha256={n:sha(F/n)for n in('config.json','controller.py')},creator_chain_root=str(R),creator_chain_files_sha256=manifest['files_sha256'],full_paper_complete=False)
 write(F/'stage-receipt.json',fm)
 def register(worker):
  value=dict(root=str(F),scope='full',source_commit=q['source_commit'],actual_controller=worker,frozen_members_sha256=fm['files_sha256'],creator_chain_root=str(R),full_paper_complete=False)
  write(F/'launch-registration.json',value);write(R/'private/full-registration.json',value)
 run('six-complete4282-neural-passes',[sys.executable,'-u','-B',str(F/'controller.py')],on_start=register)
 registration=F/'launch-registration.json'
 collector=run('closed-full-model-free-pending-collection',[sys.executable,'-u','-B',str(P/'scripts/paper/collect_native_center_reference.py'),'--workflow',str(F),'--registration',str(registration),'--expected-registration-sha256',sha(registration),'--output',str(F/'private/pending-table10')])
 result=dict(format='actual_native_center_reference_full_and_pending_collection_chain_v1',source_commit=q['source_commit'],full_root=str(F),full_registration_sha256=sha(registration),
  GPU_acceptance_sha256=sha(R/'private/GPU-acceptance.json'),GPU_acceptance_reader=gpu_reader,full_neural_forwards=25692,optimizer_updates=0,
  pending_table_root=str(F/'private/pending-table10'),pending_metrics_sha256=sha(F/'private/pending-table10/metrics.pending.json'),pending_collector=collector,
  independent_postexit_arithmetic_verification_complete=False,scientific_eligible=False,full_paper_complete=False)
 write(R/'private/completed.json',result);state.update(status='complete',actual_exit_code=0,phase='full-prediction-and-pending-table-closed-independent-arithmetic-still-required',ended_utc=utc());save()
except BaseException as error:
 state.update(status='failed',actual_exit_code=1,error=type(error).__name__+': '+str(error),ended_utc=utc());save();raise
