"""Third-reader arithmetic from raw FP32 packets, without the experiment reducer.

This reader writes pending-exit evidence. Its physical exit must be verified
separately before any public artifact is promoted.
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
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
SIZES=('small','middle','large');SPLITS=('gesture','user','both');HANDS=('left','right')
PART_KEYS={'pred','target','mask','index','unique','repeat','center','episode','hand','split','user',
    'sample_input_sha256','input_sha256'}
META_KEYS={'episode','start','end','center','hand','split','subject'}


def _digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _metric(errors,count):
 total=math.fsum(errors)
 return dict(absolute_error_sum_rad=total,valid_angle_count=count,mae_deg=total/count*180/math.pi if count else None)


def independent_point_statistics(payload,records):
 """Recompute typed identities, margins and user SD independently of reducers."""
 import torch
 try:
  n=len(records)
  if(n<=0 or n%2 or type(payload)is not dict or set(payload)!={'parts','source_indices','request_identities'}
      or payload['source_indices']!=list(range(n))or len(payload['parts'])!=n or len(payload['request_identities'])!=n):
   raise ValueError('independent complete ordered point population required')
  groups={k:{}for k in('hand','split','user','hand_split','split_user')};total=[];valid_samples=0
  paired={};centers=set();weighted=np.float32(0);hand_arrays={h:[]for h in HANDS}
  def add(kind,key,error,count):
   values=groups[kind].setdefault(key,[[],0]);values[0].append(error);values[1]+=count
  for i,(part,meta,record)in enumerate(zip(payload['parts'],payload['request_identities'],records)):
   if(type(part)is not dict or set(part)!=PART_KEYS or type(meta)is not dict or set(meta)!=META_KEYS
       or record['index']!=i or _digest(meta)!=record['identity_sha256']):
    raise ValueError('independent raw source identity differs')
   for name,dtype,shape in [('pred',torch.float32,(1,22,1)),('target',torch.float32,(1,22,1)),('mask',torch.bool,(1,1)),
       *[(k,torch.int64,(1,))for k in('index','unique','repeat','center','episode')]]:
    a=part[name]
    if(not torch.is_tensor(a)or a.dtype!=dtype or tuple(a.shape)!=shape or a.device.type!='cpu'
        or not bool(torch.isfinite(a).all())):
     raise ValueError('independent finite CPU FP32 typed point packet required')
   if(any(int(part[k][0])!=i for k in('index','unique'))or int(part['repeat'][0])!=0
       or int(part['center'][0])!=meta['center']or int(part['episode'][0])!=meta['episode']
       or meta['end']-meta['start']!=12000 or meta['center']!=meta['start']+6000
       or meta['hand']!=('left'if i<n//2 else'right')or meta['split']not in SPLITS
       or type(meta['subject'])is not str or not meta['subject']
       or any(part[k]!=[meta[field]]for k,field in [('hand','hand'),('split','split'),('user','subject')])
       or part['sample_input_sha256']!=[_digest(record)]or part['input_sha256']!=_digest([record])):
    raise ValueError('independent ordered hand/window/input commitment differs')
   target=part['target'][0].numpy();mask=part['mask'][0].numpy();pred=part['pred'][0].numpy()
   if hashlib.sha256(target.tobytes()+mask.tobytes()).hexdigest()!=record['target_mask_sha256']:
    raise ValueError('independent target/mask bytes differ from accepted input')
   joint_meta={k:x for k,x in meta.items()if k!='hand'}
   if i<n//2:
    if meta['center']in centers:raise ValueError('independent duplicate source center')
    centers.add(meta['center']);paired[i]=joint_meta
   elif paired[i-n//2]!=joint_meta:raise ValueError('independent left/right source pairing differs')
   valid=bool(mask[0]);error=float(np.abs(pred.astype(np.float64)-target.astype(np.float64)).sum())if valid else 0.
   count=22 if valid else 0;valid_samples+=int(valid);total.append(error)
   point32=np.mean(np.abs(pred-target),dtype=np.float32)if valid else np.float32(0)
   weighted=np.float32(weighted+point32)
   if valid:hand_arrays[meta['hand']].append((pred[:,0],target[:,0]))
   for kind,key in [('hand',meta['hand']),('split',meta['split']),('user',meta['subject']),
       ('hand_split',(meta['hand'],meta['split'])),('split_user',(meta['split'],meta['subject']))]:add(kind,key,error,count)
  aliases={u:'user_'+str(i).zfill(3)for i,u in enumerate(sorted(groups['user']))}
  result=dict(overall=_metric(total,22*valid_samples),
      by_user={aliases[u]:_metric(*v)for u,v in sorted(groups['user'].items())},
      by_hand={h:_metric(*groups['hand'][h])for h in HANDS},
      by_split={s:_metric(*groups['split'][s])for s in SPLITS},
      by_hand_split={h:{s:_metric(*groups['hand_split'][(h,s)])for s in SPLITS}for h in HANDS},
      by_split_user_summary={},consumed_samples=n,unique_samples=n,duplicate_consumptions=0,
      valid_samples=valid_samples,invalid_samples=n-valid_samples,invalid_angle_count=22*(n-valid_samples),
      native_batch_weighted_mae_rad=float(np.float32(weighted/np.float32(n))),official_float32_by_hand={})
  for split in SPLITS:
   means=[_metric(*v)['mae_deg']for(s,u),v in sorted(groups['split_user'].items())if s==split and v[1]]
   result['by_split_user_summary'][split]=dict(user_count=len(means),mean_mae_deg=float(np.mean(means))if means else None,
       std_mae_deg=float(np.std(means,ddof=0))if means else None,aggregation='unweighted user means; ddof=0')
  for hand in HANDS:
   pairs=hand_arrays[hand]
   if pairs:
    p=np.array([x for x,y in pairs]);t=np.array([y for x,y in pairs]);joints=np.mean(np.abs(p-t),axis=0)
    result['official_float32_by_hand'][hand]=dict(n_valid=len(pairs),per_joint=joints.tolist(),overall_mae_rad=float(np.mean(joints)))
   else:result['official_float32_by_hand'][hand]=dict(n_valid=0,per_joint=None,overall_mae_rad=None)
  return result
 except(KeyError,TypeError,IndexError,AttributeError,RuntimeError)as error:
  raise ValueError('malformed independent raw reference packet')from error


def compare_public_statistics(expected,actual):
 """Compare explicit public numeric trees; unknown/private fields also fail."""
 def check(a,b,path):
  if type(a)is dict:
   if type(b)is not dict or set(a)!=set(b):raise ValueError('independent public statistic dimensions differ')
   for k in a:check(a[k],b[k],path+(k,))
  elif type(a)is list:
   if type(b)is not list or len(a)!=len(b):raise ValueError('independent public statistic list differs')
   for i,(x,y)in enumerate(zip(a,b)):check(x,y,path+(str(i),))
  elif type(a)is float:
   tolerance=2e-6 if path==('native_batch_weighted_mae_rad',)else 1e-9
   if(type(b)not in(float,int)or not math.isfinite(b)or not math.isclose(a,b,rel_tol=2e-12,abs_tol=tolerance)):
    raise ValueError('independent public arithmetic differs at '+'.'.join(path))
  elif type(a)!=type(b)or a!=b:raise ValueError('independent public statistic value differs')
 check(expected,actual,())


def require_full_independent_population(value):
 if(value['consumed_samples']!=4282 or value['unique_samples']!=4282 or value['duplicate_consumptions']!=0
     or value['valid_samples']!=4154 or value['invalid_samples']!=128
     or value['overall']['valid_angle_count']!=91388 or value['invalid_angle_count']!=2816):
  raise ValueError('actual full4282/4154 valid/128 invalid population required')


def check_exact_csv(text,expected):
 reader=csv.DictReader(io.StringIO(text));rows=list(reader)
 if not expected or set(reader.fieldnames or[])!=set(expected[0])or len(rows)!=len(expected):
  raise ValueError('independent CSV population/columns differ')
 for row,wanted in zip(rows,expected):
  if set(row)!=set(wanted):raise ValueError('independent CSV row schema differs')
  for key,value in wanted.items():
   raw=row[key]
   if value is None:ok=raw==''
   elif type(value)is float:
    try:x=float(raw);ok=math.isfinite(x)and math.isclose(x,value,rel_tol=2e-12,abs_tol=1e-9)
    except(ValueError,TypeError):ok=False
   else:ok=raw==str(value)
   if not ok:raise ValueError('independent CSV value differs at '+key)


def _json(path):
 return json.loads(Path(path).read_text(),parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite verification JSON')))


def _sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _sealed(path):
 value=_json(path)
 if value.get('sha256')!=_digest({k:v for k,v in value.items()if k!='sha256'}):raise ValueError('independent document seal differs')
 return value


def _closed(actor):
 if(type(actor)is not dict or type(actor.get('pid'))is not int or actor['pid']<=0
     or type(actor.get('start_ticks'))is not str or not actor['start_ticks'].isdigit()):
  raise ValueError('independent complete actual process identity required')
 p=Path('/proc',str(actor['pid']),'stat')
 try:return p.read_text().rsplit(')',1)[1].split()[19]!=actor['start_ticks']
 except(FileNotFoundError,ProcessLookupError):return True


def _same_payload(a,b):
 import torch
 if torch.is_tensor(a):
  if not torch.is_tensor(b)or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
   raise ValueError('independent raw native/replay tensor differs')
 elif type(a)is dict:
  if type(b)is not dict or set(a)!=set(b):raise ValueError('independent native/replay payload schema differs')
  for key in a:_same_payload(a[key],b[key])
 elif type(a)is list:
  if type(b)is not list or len(a)!=len(b):raise ValueError('independent native/replay payload coverage differs')
  for x,y in zip(a,b):_same_payload(x,y)
 elif type(a)!=type(b)or a!=b:raise ValueError('independent native/replay identity differs')


def verify_closed_chain(request,output):
 """Read all closed sources and recompute three complete raw-packet tables."""
 import torch
 from egoemg_repro.classic_emg_cohort import _safe,_StatGuard
 from egoemg_repro.classic_emg_data import _safe_path
 from egoemg_repro.center_reference_eval import (_admission_metadata,source_fit_metadata,guarded_source_files,
     ROOT as SOURCE_ROOT)
 from scripts.paper.collect_native_center_reference import read_closed_reference_workflow,FULL_NAMES,QUAL_NAMES,PAPER,PARAMETERS
 registration=request['chain_registration'];C=_safe(registration['root']);manifest=_json(C/'stage-receipt.json')
 expected={}
 def pin(path,digest):
  path=str(_safe(path))
  if path in expected and expected[path]!=digest:raise ValueError('conflicting independent artifact digests')
  expected[path]=digest
 def observed(path):pin(path,_sha(path))
 if(manifest['source_commit']!=registration['source_commit']or manifest['files_sha256']!=registration['frozen_members_sha256']
     or request['expected_verifier_sha256']!=_sha(__file__)):
  raise ValueError('independent registered chain/source differs')
 for n,h in registration['frozen_members_sha256'].items():pin(C/n,h)
 state_guard=_StatGuard([C/'private/execution-state.json'])
 state=_json(C/'private/execution-state.json');chain_config=_json(C/'config.json');completed=_json(C/'private/completed.json')
 package=_safe(chain_config['package'])
 if(package!=SOURCE_ROOT or request['source_package']!=str(package)
     or request['source_package_manifest_sha256']!=_sha(package/'stage-receipt.json')):
  raise ValueError('independent reader must use the registered actual experiment source package')
 package_manifest=_json(package/'stage-receipt.json')
 pin(package/'stage-receipt.json',request['source_package_manifest_sha256'])
 for n,h in package_manifest['files_sha256'].items():pin(package/n,h)
 names=['accept-exact-six-GPU-qualification-packets','six-complete4282-neural-passes','closed-full-model-free-pending-collection']
 if(state['status']!='complete'or state['actual_exit_code']!=0 or state['wrapper']!=registration['actual_controller']
     or [s['name']for s in state['stages']]!=names
     or any(s['status']!='complete'or s['actual_exit_code']!=0 for s in state['stages'])):
  raise ValueError('complete original three-stage continuation required')
 actors=[state['wrapper'],*[s['worker']for s in state['stages']]]
 if not all(_closed(a)for a in actors):raise ValueError('independent reader requires closed chain owners')
 F=_safe(chain_config['full_root']);full_reg_path=C/'private/full-registration.json';full_reg=_json(full_reg_path)
 if(completed['format']!='actual_native_center_reference_full_and_pending_collection_chain_v1'
     or completed['source_commit']!=registration['source_commit']or completed['full_root']!=str(F)
     or completed['full_registration_sha256']!=_sha(full_reg_path)
     or full_reg!=_json(F/'launch-registration.json')or full_reg['source_commit']!=registration['source_commit']
     or full_reg['actual_controller']!=state['stages'][1]['worker']
     or completed['full_neural_forwards']!=25692 or completed['optimizer_updates']!=0):
  raise ValueError('actual complete registered six-pass workflow required')
 full=read_closed_reference_workflow(F,full_reg,scope='full',stage_names=FULL_NAMES);config=full['config']
 actors.extend(full['actors']);fc=_json(F/'private/completed.json')
 pending=_safe(F/'private/pending-table10');report=_sealed(pending/'metrics.pending.json')
 if(completed['pending_table_root']!=str(pending)or completed['pending_metrics_sha256']!=_sha(pending/'metrics.pending.json')
     or completed['pending_collector']!=state['stages'][2]['worker']
     or report['collector_process']!=state['stages'][2]['worker']
     or report['original_full_workflow']!=str(F)or report['workflow_source_commit']!=registration['source_commit']
     or report['format']!='native_shared_center_Table10_pending_analysis_v1'
     or report['full_source_model_evaluation_complete']is not True or report['scientific_eligible']is not False
     or report['independent_postexit_artifact_verification_complete']is not False
     or report['full_paper_complete']is not False or report['paper_population_or_label_version_equivalence']is not False
     or report['all_seven_full_neural_workflow_actors_physically_closed']!=full['actors']
     or [m['size']for m in report['models']]!=list(SIZES)):
  raise ValueError('exact original pending collector output required')
 if(fc['format']!='actual_three_selected_native_center_reference_full_workflow_v1'
     or fc['source_commit']!=registration['source_commit']or fc['scope']!='full'
     or fc['total_actual_new_neural_forwards']!=25692 or fc['total_optimizer_updates']!=0):
  raise ValueError('full original neural completion required')
 for root in(C,F):
  for name in('stage-receipt.json','private/execution-state.json','private/completed.json'):observed(root/name)
 for n,h in full_reg['frozen_members_sha256'].items():pin(F/n,h)
 observed(full_reg_path);observed(F/'launch-registration.json')
 gpu_acceptance_path=C/'private/GPU-acceptance.json';pin(gpu_acceptance_path,completed['GPU_acceptance_sha256'])
 gpu_acceptance=_json(gpu_acceptance_path)
 if(gpu_acceptance['reader']!=state['stages'][0]['worker']or gpu_acceptance['new_learned_forwards']!=0
     or gpu_acceptance['construction_attempts']!=dict(learned_module=0,optimizer=0)or gpu_acceptance['cuda_initialized']):
  raise ValueError('actual independent GPU acceptance reader differs')
 qualifications={}
 for kind in('CPU','GPU'):
  r=config[kind.lower()+'_qualification'];G=_safe(r['root'])
  g=read_closed_reference_workflow(G,r,scope='engineering',stage_names=QUAL_NAMES);actors.extend(g['actors'])
  recovery=g['config'].get('closed_verifier_failure')
  if recovery is not None:
   old=_safe(recovery['original_launch']['root'])
   pin(old/'private/execution-state.json',recovery['original_state_sha256'])
   for n,h in recovery['successful_S_pass_files_sha256'].items():pin(old/n,h)
   for n,h in recovery['original_launch']['frozen_members_sha256'].items():pin(old/n,h)
  pins=r['accepted_artifact_files_sha256']
  if len(pins)!=16 or report['qualifications'][kind]['accepted_artifact_files_sha256']!=pins:
   raise ValueError('complete independently accepted qualification artifacts required')
  for n,h in pins.items():pin(G/n,h)
  for n,h in r['frozen_members_sha256'].items():pin(G/n,h)
  observed(G/'stage-receipt.json');observed(G/'private/execution-state.json')
  q=_json(G/'private/completed.json');qualifications[kind]=q
  if(q['scope']!='engineering'or q['total_actual_new_neural_forwards']!=24 or q['total_optimizer_updates']!=0
      or q['source_commit']!=r['source_commit']or report['qualifications'][kind]['actual_closed_actors']!=g['actors']):
   raise ValueError('actual qualified source provenance differs')
 inputs=_admission_metadata(config['admitted_inputs']);records=inputs['records']
 if inputs['sha256']!=report['input_admission_sha256']or len(records)!=4282:raise ValueError('independent full input admission differs')
 for p,h in inputs['_paths'].items():pin(p,h)
 data=_safe(inputs['config']['data']);data_manifest=_json(data/'manifest.json');data_paths=[]
 for name,h in inputs['source_files_sha256'].items():
  if name=='metadata:manifest':p=data/'manifest.json'
  elif name=='metadata:episodes':p=data/'metadata.npz'
  else:
   key=name.removeprefix('field:');p=_safe_path(data,data_manifest['fields'][key].get('filename',key+'.dat'))
  pin(p,h);data_paths.append(p)
 headers={};metadata={}
 for size in SIZES:
  meta=source_fit_metadata(config['source_fits'][size]);metadata[size]=meta
  for p,h in meta['_paths'].items():pin(p,h)
  pin(Path(meta['fit_path'])/'checkpoints'/meta['selected_best']['file'],meta['selected_best']['checkpoint_sha256'])
  for role in('native','replay'):
   base=F/'private'/size/role;v=_sealed(base/'pass.json');headers[(size,role)]=v
   if(v['format']!='selected_native_emgformer_shared_center_pass_v1'or v['scope']!='full'or v['kind']!=role
       or v['size']!=size or v['samples']!=4282 or v['source_indices']!=list(range(4282))
       or v['input_admission_sha256']!=inputs['sha256']or v['grid_sha256']!=inputs['grid_sha256']
       or v['source_fit_path']!=meta['fit_path']or v['fit_result_sha256']!=meta['fit_result_sha256']
       or v['model_before_sha256']!=meta['selected_model_state_sha256']or v['model_after_sha256']!=meta['selected_model_state_sha256']
       or v['neural_forwards']!=4282 or v['optimizer_updates']!=0 or v['optimizer_construction_attempts']!=0
       or v['native_runtime_construction_calls']!=1 or v['raw_prediction_steps']!=[230]
       or v['runtime_policy']['device']!='cuda:0'or v['runtime_policy']['gpu_uuid']!=config['gpu_uuid']
       or v['runtime_policy']['precision']!='32-true'or v['runtime_policy']['autocast_enabled']is not False
       or v['runtime_policy']['matmul_precision']!='high'or v['runtime_policy']['point_selector']!='raw[...,T//2:T//2+1]'):
    raise ValueError('independent selected source/full FP32 execution differs')
   producer=next(s['worker']for s in full['state']['stages']if s['name']==size+'-'+role)
   if producer!={k:v['process'][k]for k in('pid','start_ticks')}:raise ValueError('independent full prediction producer differs')
   observed(base/'pass.json');pin(base/'predictions.pt',v['payload_sha256'])
   for n,h in v['sources_sha256'].items():
    p=Path(inputs['config']['upstream'])/n.removeprefix('upstream/')if n.startswith('upstream/')else SOURCE_ROOT/n
    pin(p,h)
  a,b=headers[(size,'native')],headers[(size,'replay')]
  if(a['reference_path']is not None or a['reference_sha256']is not None
      or b['reference_path']!=str(F/'private'/size/'native')or b['reference_sha256']!=a['sha256']
      or a['prediction_sha256']!=b['prediction_sha256']or a['sources_sha256']!=b['sources_sha256']
      or a['runtime_policy']!=b['runtime_policy']or a['finished_ns']>b['started_ns']
      or a['process']==b['process']or fc['models'][size]!={'native_sha256':a['sha256'],'replay_sha256':b['sha256']}):
   raise ValueError('independent fresh native/replay context differs')
  q=qualifications['GPU']['models'][size]['native']
  if(q['runtime_policy']!=a['runtime_policy']or q['fit_result_sha256']!=a['fit_result_sha256']
      or q['model_before_sha256']!=a['model_before_sha256']):raise ValueError('qualified GPU policy/source differs')
 for name in('metrics.pending.json','cells.csv','users.csv','paper-comparison.csv'):observed(pending/name)
 guard=guarded_source_files(expected);oracles={};expected_rows=[];expected_users=[];expected_comparisons=[]
 for model in report['models']:
  size=model['size'];a,b=headers[(size,'native')],headers[(size,'replay')]
  native=torch.load(F/'private'/size/'native/predictions.pt',map_location='cpu',weights_only=True)
  replay=torch.load(F/'private'/size/'replay/predictions.pt',map_location='cpu',weights_only=True);_same_payload(native,replay)
  oracle=independent_point_statistics(native,records);require_full_independent_population(oracle);oracles[size]=oracle
  public={k:v for k,v in oracle.items()if k not in('valid_samples','invalid_samples','official_float32_by_hand')}
  compare_public_statistics(public,model['statistics']);compare_public_statistics(oracle['official_float32_by_hand'],model['official_float32_by_hand'])
  if(model['native_pass_sha256']!=a['sha256']or model['independent_pass_sha256']!=b['sha256']
      or model['typed_packet_sha256']!=a['prediction_sha256']or model['runtime_policy']!=a['runtime_policy']
      or model['source_fit']!=a['source_fit']or model['grid_sha256']!=inputs['grid_sha256']
      or model['source_indices_sha256']!=_digest(list(range(4282)))or model['parameter_count']!=PARAMETERS[size]):
   raise ValueError('independent public model/source summary differs')
  label='EMGFormer-'+dict(small='S',middle='M',large='L')[size]
  if model['method']!=label:raise ValueError('independent public model label differs')
  row=dict(method=label,parameter_count=PARAMETERS[size])
  for split in SPLITS:
   v=oracle['by_split_user_summary'][split]
   if v['mean_mae_deg']is None:raise ValueError('undefined full Table10 split')
   row.update({split+'_mean_mae_deg':v['mean_mae_deg'],split+'_std_mae_deg':v['std_mae_deg'],
       split+'_valid_users':v['user_count'],split+'_valid_angles':oracle['by_split'][split]['valid_angle_count']})
   expected_comparisons.append(dict(method=label,condition=split,actual_mean_deg=v['mean_mae_deg'],paper_mean_deg=PAPER[size][split][0],
       gap_deg=v['mean_mae_deg']-PAPER[size][split][0],actual_std_deg=v['std_mae_deg'],paper_std_deg=PAPER[size][split][1],population_equivalence=False))
  row.update(average_valid_sample_mae_deg=oracle['overall']['mae_deg'],requested=4282,valid_samples=4154,invalid_samples_retained=128,valid_angles=91388)
  expected_rows.append(row)
  expected_comparisons.append(dict(method=label,condition='average',actual_mean_deg=oracle['overall']['mae_deg'],paper_mean_deg=PAPER[size]['average'],
      gap_deg=oracle['overall']['mae_deg']-PAPER[size]['average'],actual_std_deg=None,paper_std_deg=None,population_equivalence=False))
  expected_users.extend(dict(method=label,user_alias=u,**v)for u,v in oracle['by_user'].items())
  del native,replay
 compare_public_statistics(expected_rows,report['table_rows'])
 for name,rows in [('cells.csv',expected_rows),('users.csv',expected_users),('paper-comparison.csv',expected_comparisons)]:
  check_exact_csv((pending/name).read_text(),rows)
 if(_digest(report['paper_reference']['display_values'])!=_digest(PAPER)or report['paper_reference']['table']!=10
     or report['paper_reference']['pdf_sha256']!='124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725'):
  raise ValueError('original PDF displayed reference differs')
 guard.check();state_guard.check()
 unique={(a['pid'],a['start_ticks']):a for a in actors}
 if not all(_closed(a)for a in unique.values()):raise ValueError('source actor reappeared during independent read')
 fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split();reader=dict(pid=os.getpid(),start_ticks=fields[19])
 proof=dict(format='independent_raw_FP32_shared_center_arithmetic_pending_exit_v1',source_commit=registration['source_commit'],
     chain_root=str(C),full_root=str(F),source_artifacts_sha256=expected,source_files=len(expected),
     actual_legacy_data_files=len(data_paths),actual_legacy_data_bytes=sum(p.stat().st_size for p in data_paths),
     actual_all_known_source_actors_closed=list(unique.values()),models=SIZES,requested_per_model=4282,
     valid_per_model=4154,invalid_per_model=128,csv_rows=dict(cells=len(expected_rows),users=len(expected_users),comparison=len(expected_comparisons)),
     original_pending_metrics_sha256=_sha(pending/'metrics.pending.json'),independent_reader=reader,verifier_source_sha256=_sha(__file__),
     primary_absolute_tolerance_deg=1e-9,native_float32_auxiliary_absolute_tolerance_rad=2e-6,
     independent_arithmetic_complete=True,independent_reader_physically_closed=False,scientific_eligible=False,full_paper_complete=False)
 output=_safe(output,exists=False);output.mkdir(mode=0o700,parents=True)
 for name,body in [('verification.json',(json.dumps(proof,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()),
     ('metrics.verified-pending-exit.json',(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()),
     *[(n,(pending/n).read_bytes())for n in('cells.csv','users.csv','paper-comparison.csv')]]:
  with(output/name).open('xb')as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
 guard.check();state_guard.check()
 return proof


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True,type=Path)
 parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
 if _sha(args.request)!=args.expected_request_sha256:raise ValueError('independent frozen reader request changed')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('independent arithmetic reader requires hidden CUDA')
 import torch
 torch.set_num_threads(2);attempts=dict(learned_module=0,optimizer=0)
 def deny_model(*a,**k):attempts['learned_module']+=1;raise RuntimeError('independent reader cannot construct a model')
 def deny_optimizer(*a,**k):attempts['optimizer']+=1;raise RuntimeError('independent reader cannot construct an optimizer')
 torch.nn.Module.__init__=deny_model;torch.optim.Optimizer.__init__=deny_optimizer
 v=verify_closed_chain(_json(args.request),args.output)
 if attempts!=dict(learned_module=0,optimizer=0)or torch.cuda.is_initialized():raise ValueError('independent reader initialized learned runtime')
 print(json.dumps(dict(models=len(v['models']),requested_per_model=v['requested_per_model'],independent_arithmetic_complete=True,
     pending_actual_reader_exit=True,new_learned_forwards=0,new_optimizer_updates=0,full_paper_complete=False)))

if __name__=='__main__':main()
