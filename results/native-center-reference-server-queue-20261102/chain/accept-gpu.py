
from pathlib import Path
import hashlib,importlib.util,json,os,sys
R=Path(sys.argv[1]);q=json.loads((R/'config.json').read_text());P=Path(q['package']);sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('qualified_collector',P/'scripts/paper/collect_native_center_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
registration=q['GPU_qualification'];G=Path(registration['root'])
graph=m.read_closed_reference_workflow(G,registration,scope='engineering',stage_names=m.QUAL_NAMES)
c=m._json(G/'private/completed.json');names=['private/completed.json']
for size in m.SIZES:
 names.append('private/'+size+'/qualification.json')
 for role in('native','replay'):names.extend('private/'+size+'/'+role+'/'+n for n in('pass.json','predictions.pt'))
from egoemg_repro.classic_emg_cohort import _StatGuard
guard=_StatGuard([G/n for n in names]+[G/'private/execution-state.json'])
pins={n:m._sha(G/n)for n in names};accepted=dict(registration,accepted_artifact_files_sha256=pins)
targets={s:dict(native=m._json(G/'private'/s/'native'/'pass.json'))for s in m.SIZES}
import torch
torch.set_num_threads(2);attempts=dict(learned_module=0,optimizer=0)
def no_model(*a,**k):attempts['learned_module']+=1;raise RuntimeError('GPU qualification acceptance is model-free')
def no_optimizer(*a,**k):attempts['optimizer']+=1;raise RuntimeError('GPU qualification acceptance is optimizer-free')
torch.nn.Module.__init__=no_model;torch.optim.Optimizer.__init__=no_optimizer
result=m._qualification(G,accepted,'GPU',targets)
assert attempts==dict(learned_module=0,optimizer=0)and not torch.cuda.is_initialized()
assert c['source_commit']==registration['source_commit']and c['total_actual_new_neural_forwards']==24
guard.check();f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
body=dict(format='actual_independent_closed_GPU_qualification_acceptance_v1',source_commit=q['source_commit'],accepted_registration=accepted,
 qualification=result,actual_models=c['models'],original_qualification_completion_sha256=pins['private/completed.json'],
 reader=dict(pid=os.getpid(),start_ticks=f[19]),construction_attempts=attempts,cuda_initialized=False,
 new_learned_forwards=0,new_optimizer_updates=0,full_model_evaluation_complete=False,full_paper_complete=False)
with(R/'private/GPU-acceptance.json').open('x')as stream:stream.write(json.dumps(body,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(actual_GPU_qualification_complete=True,actual_closed_source_actors=len(graph['actors']),actual_raw_passes=6,pending_full_model=True)))
