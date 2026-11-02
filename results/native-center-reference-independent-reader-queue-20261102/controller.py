
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time,shutil
R=Path(__file__).resolve().parent;q=json.loads((R/'config.json').read_text());P=Path(q['source_package']);sys.path.insert(0,str(P));os.nice(10)
from egoemg_repro.classic_emg_cohort import _StatGuard
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
m=json.loads((R/'stage-receipt.json').read_text());assert all(sha(R/n)==h for n,h in m['files_sha256'].items())
assert sha(P/'stage-receipt.json')==q['source_package_manifest_sha256']
pm=json.loads((P/'stage-receipt.json').read_text());assert all(sha(P/n)==h for n,h in pm['files_sha256'].items())
pins=_StatGuard([R/n for n in m['files_sha256']]+[P/'stage-receipt.json']+[P/n for n in pm['files_sha256']])
def check():
 assert 'torch'not in sys.modules;pins.check();assert shutil.disk_usage('/data/shihao').free>=225*1024**3
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
  with path.open('x')as f:f.write(raw);f.flush();os.fsync(f.fileno())
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
state=dict(status='running',phase='waiting-original-full-chain-exit0-and-all-owned-source-actors-absent',wrapper=actor(os.getpid()),started_utc=utc(),full_paper_complete=False)
def save():write(R/'private/execution-state.json',state,True)
def interrupt(signum,frame):raise RuntimeError('independent final reader owner interrupted')
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt);save()
try:
 C=Path(q['chain_registration']['root'])
 while True:
  check();p=C/'private/execution-state.json';s=json.loads(p.read_text())if p.exists()else None
  if s is not None:
   assert s['wrapper']==q['chain_registration']['actual_controller']
   if s['status']=='failed':raise RuntimeError('original full chain terminal failure; do not substitute partial data')
   if s['status']=='complete'and s['actual_exit_code']==0 and all(closed(a)for a in[s['wrapper'],*[x['worker']for x in s['stages']]]):break
   if closed(s['wrapper']):raise RuntimeError('original full owner disappeared without complete closed outputs')
  elif closed(q['chain_registration']['actual_controller']):raise RuntimeError('original full owner absent without state')
  time.sleep(2)
 state['phase']='independent-full-raw-packet-arithmetic';save()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',EGOEMG_UPSTREAM='/home/shihao/projects/EgoEMG-Reproduction/upstream')
 env['PYTHONPATH']=str(P)+':/home/shihao/projects/EgoEMG-Classic-Canonical-20261026/.deps:/home/shihao/projects/EgoEMG-WiLoR-20261026/.deps'
 output=R/'private/verified-artifacts'
 with(R/'private/reader.log').open('xb')as log:
  p=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/paper/verify_native_center_reference.py'),'--request',str(R/'config.json'),'--expected-request-sha256',sha(R/'config.json'),'--output',str(output)],cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state['reader']=actor(p.pid);save()
  try:
   while p.poll()is None:check();time.sleep(2)
   code=p.wait();check()
  finally:
   if p.poll()is None:
    p.terminate()
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
 assert code==0 and closed(state['reader']),'actual third reader failed; inspect preserved source and log, no retry'
 proof=json.loads((output/'verification.json').read_text());assert proof['independent_reader']==state['reader']and proof['independent_arithmetic_complete']
 assert all(closed(a)for a in proof['actual_all_known_source_actors_closed'])
 files={n:sha(output/n)for n in('verification.json','metrics.verified-pending-exit.json','cells.csv','users.csv','paper-comparison.csv')}
 value=dict(format='actual_closed_third_independent_native_center_reference_reader_v1',source_commit=q['source_commit'],
  chain_root=str(C),third_reader=state['reader'],actual_reader_exit_code=0,actual_reader_physically_closed_after_exit0=True,
  all_actual_source_actors_physically_closed=True,artifact_root=str(output),artifact_files_sha256=files,
  source_manifest_sha256=sha(R/'stage-receipt.json'),verifier_source_sha256=sha(R/'scripts/paper/verify_native_center_reference.py'),
  independent_full_packet_arithmetic_complete=True,pending_external_owner_closure_and_publication=True,scientific_eligible=False,full_paper_complete=False)
 write(R/'private/postexit-closure.json',value);state.update(status='complete',actual_exit_code=0,phase='third-reader-physically-closed-awaiting-external-publication',ended_utc=utc());save()
except BaseException as error:
 state.update(status='failed',actual_exit_code=1,error=type(error).__name__+': '+str(error),ended_utc=utc());save();raise
