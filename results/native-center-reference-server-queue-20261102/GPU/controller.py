
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent;q=json.loads((R/'config.json').read_text());P=Path(q['package']);os.nice(10)
def actor(pid):
 f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=f[19])
def write():
 p=R/'private/execution-state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');os.replace(t,p)
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
package=json.loads((P/'stage-receipt.json').read_text());manifest=json.loads((R/'stage-receipt.json').read_text())
def check():
 assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in package['files_sha256'].items())
 assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in q['new_sources_sha256'].items())
 assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in manifest['files_sha256'].items())

import signal,time
sys.path.insert(0,str(P))
from egoemg_repro.classic_emg_cohort import _StatGuard
pins=_StatGuard([R/n for n in manifest['files_sha256']]+[P/n for n in package['files_sha256']]+[R/'stage-receipt.json',P/'stage-receipt.json'])
active=None;active_actor=None;cuda_phase=False;nvml_observations={}
def strict_closed(a):
 p=Path('/proc',str(a['pid']),'stat')
 return not p.exists()or p.read_text().rsplit(')',1)[1].split()[19]!=str(a['start_ticks'])
def resource_guard():
 pins.check()
 import shutil
 assert shutil.disk_usage('/data/shihao').free>=225*1024**3
 assert next(int(x.split()[1])for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))*1024>=128*1024**3
def gpu():
 free=int(subprocess.check_output(['nvidia-smi','-i',q['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip())
 apps=[int(x.split(',')[0])for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()if x.strip()and x.split(',')[1].strip()==q['gpu_uuid']]
 return free,apps
def waiting_admission():
 stable=None
 while True:
  resource_guard();old=Path(q['previous_GPU0_workflow']['root']);s=json.loads((old/'private/execution-state.json').read_text())
  assert all(s[k]==q['previous_GPU0_workflow']['actual_state'][k]for k in('wrapper','child'))
  predecessor_closed=s['status']in('complete','failed')and all(strict_closed(s[k])for k in('wrapper','child'))
  free,apps=gpu();now=time.monotonic();ok=predecessor_closed and not apps and free>=65536
  stable=(now if stable is None else stable)if ok else None
  if stable is not None and now-stable>=20:return
  time.sleep(2)
def running_guard():
 resource_guard()
 if cuda_phase:
  free,apps=gpu();assert free>=8192,'selected GPU runtime reserve below8GiB'
  assert all(active_actor is not None and pid==active_actor['pid']and not strict_closed(active_actor)for pid in apps),'foreign GPU arrival; stop only this owned child'
  if apps:nvml_observations[name]=dict(worker=active_actor,actual_compute_pid=apps[0],gpu_uuid=q['gpu_uuid'],observed_utc=utc())
def interrupt(signum,frame):raise RuntimeError('owned qualification controller interrupted')
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)

env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',EGOEMG_UPSTREAM='/home/shihao/projects/EgoEMG-Reproduction/upstream')
env['PYTHONPATH']=str(P)+':/home/shihao/projects/EgoEMG-Classic-Canonical-20261026/.deps:/home/shihao/projects/EgoEMG-WiLoR-20261026/.deps'
state=dict(status='running',wrapper=actor(os.getpid()),started_utc=utc(),stages=[],scope='engineering',full_paper_complete=False);write()
try:
 for size in('small','middle','large'):
  base=R/'private'/size;base.mkdir(mode=0o700)
  for kind in('native','replay','verify'):
   check();name=size+'-'+kind
   cuda_phase=kind!='verify';active_actor=None
   if cuda_phase:waiting_admission()
   env['CUDA_VISIBLE_DEVICES']=q['gpu_uuid']if cuda_phase else''
   if kind=='verify':args=[sys.executable,'-u','-B',str(R/'verify.py'),str(R),size]
   else:
    args=[sys.executable,'-u','-B',str(P/'scripts/paper/run_native_center_reference.py'),'run','--source-fit',q['source_fits'][size],'--admitted-inputs',q['admitted_inputs'],'--output',str(base/kind),'--kind',kind,'--scope','engineering','--device','cuda:0','--gpu-uuid',q['gpu_uuid']]
    if kind=='replay':args.extend(['--reference',str(base/'native')])
   with(base/(kind+'.log')).open('xb')as log:
    p=subprocess.Popen(args,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active=p;active_actor=actor(p.pid)
    phase=dict(name=name,worker=active_actor,status='running');state['stages'].append(phase);write()
    try:
     while p.poll()is None:running_guard();time.sleep(1)
     code=p.wait();resource_guard()
    finally:
     if p.poll()is None:
      p.terminate()
      try:p.wait(timeout=15)
      except subprocess.TimeoutExpired:p.kill();p.wait()
    active=None;active_actor=None;cuda_phase=False
    phase.update(status='complete'if code==0 else'failed',actual_exit_code=code);write()
    assert code==0,'actual '+name+' failed; inspect owned log, no automatic retry'
 check();models={s:json.loads((R/'private'/s/'qualification.json').read_text())for s in('small','middle','large')}
 result=dict(format='actual_native_center_reference_three_selected_models_GPU_qualification_v1',source_commit=q['source_commit'],models=models,actual_nvml_observations=nvml_observations,scope='engineering',total_actual_new_neural_forwards=24,total_optimizer_updates=0,all_three_GPU_model_qualifications_complete=True,full_model_evaluation_complete=False,full_paper_complete=False)
 with(R/'private/completed.json').open('x')as f:f.write(json.dumps(result,sort_keys=True,indent=2)+'\n')
 state.update(status='complete',actual_exit_code=0,ended_utc=utc());write()
except BaseException as error:
 state.update(status='failed',actual_exit_code=1,ended_utc=utc(),error=type(error).__name__+': '+str(error));write();raise
