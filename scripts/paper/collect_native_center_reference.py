"""Model-free pending Table10 from an entirely closed full reference workflow.

An independent reader must verify these outputs after this collector exits.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from egoemg_repro.classic_emg_cohort import _safe,_StatGuard
from egoemg_repro.center_reference_eval import guarded_source_files,_hash

SIZES=('small','middle','large');SPLITS=('gesture','user','both');HANDS=('left','right')
PARAMETERS=dict(small=3460134,middle=6617382,large=16320806)
FULL_NAMES=[size+'-'+kind for size in SIZES for kind in('native','replay')]
QUAL_NAMES=[size+'-'+kind for size in SIZES for kind in('native','replay','verify')]
PAPER={
 'small':dict(gesture=(12.5,1.7),user=(15.5,2.5),both=(17.,1.4),average=14.5),
 'middle':dict(gesture=(12.9,1.7),user=(15.7,2.6),both=(17.7,1.6),average=14.9),
 'large':dict(gesture=(12.6,1.6),user=(16.1,1.6),both=(17.4,1.2),average=14.6),
}


def _sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _json(path):
 return json.loads(Path(path).read_text(),parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite source JSON')))


def _actor_key(actor):
 if (type(actor)is not dict or set(actor)!={'pid','start_ticks'}or type(actor['pid'])is not int
     or actor['pid']<=0 or type(actor['start_ticks'])is not str or not actor['start_ticks'].isdigit()):
  raise ValueError('complete declared PID/start_ticks actor required')
 return actor['pid'],actor['start_ticks']


def _closed(actor):
 pid,start=_actor_key(actor);p=Path('/proc',str(pid),'stat')
 try:return p.read_text().rsplit(')',1)[1].split()[19]!=start
 except(FileNotFoundError,ProcessLookupError):return True


def _metric(value):
 keys={'absolute_error_sum_rad','valid_angle_count','mae_deg'}
 if type(value)is not dict or set(value)!=keys:raise ValueError('public metric fields only')
 error,count,mae=(value[k]for k in('absolute_error_sum_rad','valid_angle_count','mae_deg'))
 if(type(error)not in(float,int)or not math.isfinite(error)or error<0 or type(count)is not int or count<0
     or count==0 and(error!=0 or mae is not None)
     or count>0 and(type(mae)not in(float,int)or not math.isfinite(mae)or mae<0
         or not math.isclose(mae,error/count*180/math.pi,abs_tol=1e-10,rel_tol=1e-12))):
  raise ValueError('finite consistent public sufficient statistics required')
 return copy.deepcopy(value)


def public_reference_statistics(statistics):
 """Explicit numeric projection; no raw input records or private schema additions."""
 top={'format','role','consumed_samples','unique_samples','duplicate_consumptions','batch_sizes',
     'native_batch_weighted_mae_rad','all'}
 all_keys={'by_component','by_finger','by_hand','by_hand_split','by_joint','by_pd_group','by_split',
     'by_split_user_summary','by_user','invalid_angle_count','mode','overall','user_summary'}
 if type(statistics)is not dict or set(statistics)!=top or set(statistics['all'])!=all_keys:
  raise ValueError('only the declared point statistics schema can become public')
 a=statistics['all'];users=a['by_user']
 if(type(users)is not dict or not users or any(type(u)is not str or not u for u in users)
     or set(a['by_hand'])!=set(HANDS)or set(a['by_split'])!=set(SPLITS)
     or set(a['by_hand_split'])!=set(HANDS)or set(a['by_split_user_summary'])!=set(SPLITS)):
  raise ValueError('complete actual public statistic dimensions required')
 summaries={}
 for split in SPLITS:
  v=a['by_split_user_summary'][split]
  if (type(v)is not dict or set(v)!={'user_count','mean_mae_deg','std_mae_deg','aggregation'}
      or v['aggregation']!='unweighted user means; ddof=0' or type(v['user_count'])is not int or v['user_count']<0):
   raise ValueError('actual population-SD user summary required')
  n=v['user_count']
  if any((x is not None if n==0 else type(x)not in(float,int)or not math.isfinite(x)or x<0)
      for x in(v['mean_mae_deg'],v['std_mae_deg'])):
   raise ValueError('finite or explicitly undefined user summaries required')
  summaries[split]=copy.deepcopy(v)
 aliases={u:'user_'+str(i).zfill(3)for i,u in enumerate(sorted(users))}
 result=dict(overall=_metric(a['overall']),by_user={aliases[u]:_metric(v)for u,v in sorted(users.items())},
     by_hand={h:_metric(a['by_hand'][h])for h in HANDS},
     by_split={s:_metric(a['by_split'][s])for s in SPLITS},
     by_hand_split={h:{s:_metric(a['by_hand_split'][h][s])for s in SPLITS}for h in HANDS},
     by_split_user_summary=summaries,invalid_angle_count=a['invalid_angle_count'],
     consumed_samples=statistics['consumed_samples'],unique_samples=statistics['unique_samples'],
     duplicate_consumptions=statistics['duplicate_consumptions'],
     native_batch_weighted_mae_rad=statistics['native_batch_weighted_mae_rad'])
 if(type(result['invalid_angle_count'])is not int or result['invalid_angle_count']<0
     or type(result['native_batch_weighted_mae_rad'])not in(float,int)
     or not math.isfinite(result['native_batch_weighted_mae_rad'])):
  raise ValueError('finite actual point reduction required')
 for group in(result['by_user'],result['by_hand'],result['by_split']):
  if(sum(v['valid_angle_count']for v in group.values())!=result['overall']['valid_angle_count']
      or not math.isclose(math.fsum(v['absolute_error_sum_rad']for v in group.values()),
          result['overall']['absolute_error_sum_rad'],rel_tol=2e-12,abs_tol=1e-8)):
   raise ValueError('public statistics must preserve actual margins')
 return result


def require_full_reference_pair(native,replay):
 try:
  for value,kind in((native,'native'),(replay,'replay')):
   if(value['kind']!=kind or value['scope']!='full' or value['samples']!=4282
       or value['source_indices']!=list(range(4282))
       or value['statistics']['all']['overall']['valid_angle_count']!=91388
       or value['statistics']['all']['invalid_angle_count']!=2816):
    raise ValueError('full4282 request pair with all128 invalid requests retained required')
  fields=('size','fit_result_sha256','input_admission_sha256','grid_sha256','source_indices',
      'model_before_sha256','model_after_sha256','prediction_sha256')
  if(any(native[k]!=replay[k]for k in fields)or replay['reference_sha256']!=native['sha256']):
   raise ValueError('same actual selected model/input/target packet pair required')
 except(KeyError,TypeError)as error:raise ValueError('full reference pair required')from error


def read_closed_reference_workflow(root,registration,*,scope,stage_names):
 """Verify registered source bytes and physical closure; no model certification."""
 root=_safe(root)
 if(registration.get('root')!=str(root)or registration.get('scope')!=scope
     or re.fullmatch('[0-9a-f]{40}',str(registration.get('source_commit')))is None):
  raise ValueError('registered workflow source and explicit scope required')
 manifest=_json(root/'stage-receipt.json');pins=registration['frozen_members_sha256']
 if manifest['source_commit']!=registration['source_commit']or manifest['files_sha256']!=pins:
  raise ValueError('registered original workflow manifest differs')
 expected={str(_safe(root/n)):h for n,h in pins.items()}
 guard=guarded_source_files(expected);state_guard=_StatGuard([root/'private/execution-state.json',root/'stage-receipt.json'])
 state=_json(root/'private/execution-state.json');config=_json(root/'config.json')
 if(state.get('status')!='complete'or state.get('actual_exit_code')!=0 or state.get('scope')!=scope
     or config.get('scope')!=scope or config.get('source_commit')!=registration['source_commit']
     or state['wrapper']!=registration['actual_controller']
     or [s['name']for s in state['stages']]!=stage_names):
  raise ValueError('complete original workflow and exact ordered stages required')
 actors=[state['wrapper']]
 for stage in state['stages']:
  if stage['status']!='complete'or stage['actual_exit_code']!=0:raise ValueError('every source stage must complete')
  actors.append(stage['worker'])
 keys=[_actor_key(a)for a in actors]
 if len(set(keys))!=len(keys)or not all(_closed(a)for a in actors):
  raise ValueError('all distinct source actors must physically exit')
 recovery=config.get('closed_verifier_failure')
 if recovery is not None:
  old=_safe(recovery['original_launch']['root']);old_state=_json(old/'private/execution-state.json')
  if(old_state!=recovery['terminal_state']or old_state['status']!='failed'or old_state['actual_exit_code']!=1
      or _sha(old/'private/execution-state.json')!=recovery['original_state_sha256']):
   raise ValueError('original verifier failure must remain preserved')
  old_actors=[old_state['wrapper'],*[s['worker']for s in old_state['stages']]]
  if not all(_closed(a)for a in old_actors):raise ValueError('original failed actors still exist')
  for n,h in recovery['successful_S_pass_files_sha256'].items():
   if _sha(_safe(old/n))!=h or _sha(_safe(root/n))!=h:raise ValueError('preserved successful prefix changed')
  known={_actor_key(a)for a in actors}
  actors.extend(a for a in old_actors if _actor_key(a)not in known)
 guard.check();state_guard.check()
 return dict(root=str(root),state=state,config=config,manifest=manifest,actors=actors)


def _qualification(root,registration,kind,full_models):
 graph=read_closed_reference_workflow(root,registration,scope='engineering',stage_names=QUAL_NAMES)
 value=_json(Path(root)/'private/completed.json')
 if(not value.get('all_three_'+kind+'_model_qualifications_complete')
     or value['scope']!='engineering'or value['total_actual_new_neural_forwards']!=24
     or value['total_optimizer_updates']!=0 or set(value['models'])!=set(SIZES)):
  raise ValueError('complete actual three-model qualification required')
 for size in SIZES:
  v=value['models'][size];a,b=v['native'],v['replay'];target=full_models[size]['native']
  if(a['samples']!=4 or b['samples']!=4 or a['scope']!='engineering'or b['scope']!='engineering'
      or a['source_indices']!=[0,2140,2141,4281]or a['prediction_sha256']!=b['prediction_sha256']
      or a['model_before_sha256']!=target['model_before_sha256']or b['model_after_sha256']!=target['model_after_sha256']
      or a['fit_result_sha256']!=target['fit_result_sha256']or a['input_admission_sha256']!=target['input_admission_sha256']
      or a['raw_prediction_steps']!=b['raw_prediction_steps']or a['raw_prediction_steps']!=target['raw_prediction_steps']
      or v['model_free_reader_construction_attempts']!=dict(learned_module=0,optimizer=0)):
   raise ValueError('qualification must use these exact original selected sources')
  if kind=='CPU'and a['runtime_policy']['device']!='cpu':raise ValueError('actual CPU qualification required')
  if kind=='GPU'and a['runtime_policy']!=target['runtime_policy']:raise ValueError('same actual full GPU policy qualification required')
 return dict(root=str(root),completion_sha256=_sha(Path(root)/'private/completed.json'),
     actual_closed_actors=graph['actors'],scope='engineering',models=list(SIZES),actual_neural_forwards=24)


def _csv(rows):
 stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
 writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def collect_reference_workflow(workflow,registration,output):
 """Write anonymous pending artifacts after all full/qualification actors exit."""
 import torch
 from egoemg_repro.center_reference_eval import read_reference_pass
 graph=read_closed_reference_workflow(workflow,registration,scope='full',stage_names=FULL_NAMES)
 root=Path(graph['root']);config=graph['config'];complete=_json(root/'private/completed.json')
 if(complete.get('format')!='actual_three_selected_native_center_reference_full_workflow_v1'
     or complete.get('scope')!='full'or complete.get('source_commit')!=registration['source_commit']
     or set(complete.get('models',{}))!=set(SIZES)):
  raise ValueError('original complete three-model full inference workflow required')
 models={};private_user_sets=[]
 for size in SIZES:
  a=read_reference_pass(root/'private'/size/'native');b=read_reference_pass(root/'private'/size/'replay')
  require_full_reference_pair(a,b)
  if(a['size']!=size or a['source_fit_path']!=config['source_fits'][size]
      or a['runtime_policy']['device']!='cuda:0'or a['runtime_policy']['gpu_uuid']!=config['gpu_uuid']
      or complete['models'][size]!={'native_sha256':a['sha256'],'replay_sha256':b['sha256']}):
   raise ValueError('registered full model/source/result differs')
  for kind,v in(('native',a),('replay',b)):
   owner=next(s['worker']for s in graph['state']['stages']if s['name']==size+'-'+kind)
   if(owner['pid']!=v['process']['pid']or owner['start_ticks']!=v['process']['start_ticks']):
    raise ValueError('raw prediction producer differs from original full stage')
  models[size]=dict(native=a,replay=b);private_user_sets.append(set(a['statistics']['all']['by_user']))
 if any(s!=private_user_sets[0]for s in private_user_sets):raise ValueError('cross-model private request population differs')
 if len({v['native']['input_admission_sha256']for v in models.values()})!=1:
  raise ValueError('cross-model complete input admission differs')
 qualifications={kind:_qualification(config[kind.lower()+'_qualification']['root'],
     config[kind.lower()+'_qualification'],kind,models)for kind in('CPU','GPU')}
 public_models=[];rows=[];users=[];comparisons=[]
 for size in SIZES:
  a,b=models[size]['native'],models[size]['replay'];statistics=public_reference_statistics(a['statistics'])
  means=statistics['by_split_user_summary'];label='EMGFormer-'+dict(small='S',middle='M',large='L')[size]
  if any(means[s]['mean_mae_deg']is None for s in SPLITS):raise ValueError('undefined Table10 user-summary cell')
  row=dict(method=label,parameter_count=PARAMETERS[size])
  for split in SPLITS:
   row.update({split+'_mean_mae_deg':means[split]['mean_mae_deg'],split+'_std_mae_deg':means[split]['std_mae_deg'],
       split+'_valid_users':means[split]['user_count'],split+'_valid_angles':statistics['by_split'][split]['valid_angle_count']})
   comparisons.append(dict(method=label,condition=split,actual_mean_deg=means[split]['mean_mae_deg'],
       paper_mean_deg=PAPER[size][split][0],gap_deg=means[split]['mean_mae_deg']-PAPER[size][split][0],
       actual_std_deg=means[split]['std_mae_deg'],paper_std_deg=PAPER[size][split][1],population_equivalence=False))
  row.update(average_valid_sample_mae_deg=statistics['overall']['mae_deg'],requested=4282,valid_samples=4154,
      invalid_samples_retained=128,valid_angles=91388)
  rows.append(row)
  comparisons.append(dict(method=label,condition='average',actual_mean_deg=statistics['overall']['mae_deg'],
      paper_mean_deg=PAPER[size]['average'],gap_deg=statistics['overall']['mae_deg']-PAPER[size]['average'],
      actual_std_deg=None,paper_std_deg=None,population_equivalence=False))
  users.extend(dict(method=label,user_alias=user,**metric)for user,metric in statistics['by_user'].items())
  public_models.append(dict(size=size,method=label,parameter_count=PARAMETERS[size],source_fit=a['source_fit'],
      native_pass_sha256=a['sha256'],independent_pass_sha256=b['sha256'],typed_packet_sha256=a['prediction_sha256'],
      grid_sha256=a['grid_sha256'],source_indices_sha256=_hash(a['source_indices']),statistics=statistics,
      official_float32_by_hand=a['official_float32_by_hand'],runtime_policy=a['runtime_policy']))
 fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 report=dict(format='native_shared_center_Table10_pending_analysis_v1',models=public_models,table_rows=rows,
     original_full_workflow=str(root),workflow_source_commit=registration['source_commit'],
     all_seven_full_neural_workflow_actors_physically_closed=graph['actors'],qualifications=qualifications,
     collector_process=dict(pid=os.getpid(),start_ticks=fields[19]),collector_source_sha256=_sha(__file__),
     input_admission_sha256=models['small']['native']['input_admission_sha256'],
     paper_reference=dict(physical_page=23,table=10,pdf_sha256='124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725',display_values=PAPER),
     aggregations=dict(splits='unweighted valid per-user means; population SD ddof=0; left/right pooled within user',
         average='valid-angle pooled, equal to valid-sample mean because each valid request has22angles',
         native_auxiliary='FP32 batch1 mean including zero loss for invalid requests; never substituted for Avg',
         literal_hand='original helper FP32 per-joint mean and hand mean, recorded separately'),
     full_source_model_evaluation_complete=True,independent_postexit_artifact_verification_complete=False,
     scientific_eligible=False,full_paper_complete=False,paper_population_or_label_version_equivalence=False)
 report['sha256']=_hash(report)
 output=_safe(output,exists=False);output.mkdir(mode=0o700,parents=True)
 files={'metrics.pending.json':(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode(),
     'cells.csv':_csv(rows),'users.csv':_csv(users),'paper-comparison.csv':_csv(comparisons)}
 for name,body in files.items():
  with(output/name).open('xb')as stream:stream.write(body)
 return report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workflow',required=True,type=Path)
 p.add_argument('--registration',required=True,type=Path);p.add_argument('--expected-registration-sha256',required=True)
 p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if _sha(_safe(a.registration))!=a.expected_registration_sha256:raise ValueError('original launch registration changed')
 import torch
 attempts={'learned_module':0,'optimizer':0}
 def deny_model(*args,**kwargs):attempts['learned_module']+=1;raise RuntimeError('model-free collector cannot construct a model')
 def deny_optimizer(*args,**kwargs):attempts['optimizer']+=1;raise RuntimeError('model-free collector cannot construct an optimizer')
 torch.nn.Module.__init__=deny_model;torch.optim.Optimizer.__init__=deny_optimizer
 v=collect_reference_workflow(a.workflow,_json(a.registration),a.output)
 if attempts!=dict(learned_module=0,optimizer=0)or torch.cuda.is_initialized():raise ValueError('collector initialized a learned runtime')
 print(json.dumps(dict(models=len(v['models']),table_rows=len(v['table_rows']),new_neural_forwards=0,new_optimizer_updates=0,
     pending_only=True,full_paper_complete=False)))

if __name__=='__main__':main()
