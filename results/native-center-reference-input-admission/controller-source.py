
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent
def actor(pid):
 f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=f[19])
def write(v):
 p=R/'private/execution-state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');os.replace(t,p)
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
m=json.loads((R/'stage-receipt.json').read_text());assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in m['files_sha256'].items())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env['PYTHONPATH']='/home/shihao/projects/EgoEMG-Classic-Canonical-20261026/.deps:/home/shihao/projects/EgoEMG-WiLoR-20261026/.deps'
state=dict(status='running',wrapper=actor(os.getpid()),started_utc=utc(),passes=[],scientific_eligible=False,full_paper_complete=False);write(state)
try:
 for label in('first','second'):
  with(R/'private'/(label+'.log')).open('xb')as log:
   p=subprocess.Popen([sys.executable,'-u','-B',str(R/'worker.py'),str(R),label],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   stage=dict(label=label,worker=actor(p.pid),status='running');state['passes'].append(stage);write(state);code=p.wait();stage.update(status='complete'if code==0 else'failed',actual_exit_code=code);write(state)
   assert code==0,'full input '+label+' failed; inspect actual log, no automatic restart'
 a=json.loads((R/'private/first/result.json').read_text());b=json.loads((R/'private/second/result.json').read_text())
 keys=('grid_sha256','counts','source_files_sha256','input_stream_sha256','target_mask_stream_sha256','private_identity_stream_sha256','input_digests_file_sha256','input_contract')
 assert all(a[k]==b[k]for k in keys)and a['actor']!=b['actor']
 assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in m['files_sha256'].items())
 result=dict(format='actual_two_process_complete_native_center_reference_input_admission_v1',source_commit=m['source_commit'],passes=[a,b],all4282_inputs_targets_masks_and_metadata_exact_across_two_processes=True,complete_reference_input_validation=True,model_evaluation_complete=False,scientific_eligible=False,full_paper_complete=False)
 with(R/'private/completed.json').open('x')as f:f.write(json.dumps(result,sort_keys=True,indent=2)+'\n')
 state.update(status='complete',actual_exit_code=0,ended_utc=utc());write(state)
except BaseException as exc:
 state.update(status='failed',actual_exit_code=1,ended_utc=utc(),error=type(exc).__name__+': '+str(exc));write(state);raise
