"""Actual RN18 producer/data footprint experiment; no inference permit."""
from pathlib import Path
import hashlib,json,os,random,shutil,sys
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from egoemg_repro.classic_emg import _hash
from egoemg_repro.classic_emg_cohort import _safe,_StatGuard,_source_paths
from egoemg_repro.data import LegacyStore,union_intervals
from egoemg_repro.partitions import build_partitions
from egoemg_repro.rn18_fusion_data import build_rn18_fusion_partitions
from egoemg_repro.rn18_initializer_data import build_rn18_initializer_partitions
from egoemg_repro.occlusion_dense_data import _typed_sha
from egoemg_repro.occlusion_dense_footprints import summarize_dense_footprints
q=json.loads((R/'private/request.json').read_text());manifest=json.loads((R/'private/stage-receipt.json').read_text())
g=_StatGuard([R/'private/request.json',R/'private/stage-receipt.json',*[R/n for n in manifest['files_sha256']],*_source_paths(Path(q['upstream']))])
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p,expected=None,sealed=False):
 p=_safe(p);g.add(p)
 if expected is not None:assert sha(p)==expected,str(p)
 v=json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in()).throw(ValueError('nonfinite JSON')))
 if sealed:assert v['sha256']==_hash({k:x for k,x in v.items()if k!='sha256'}),str(p)
 g.check();return v
for n,h in manifest['files_sha256'].items():assert sha(R/n)==h
assert shutil.disk_usage(R).free>=225*1024**3
available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
assert available>=128*1024**3
os.nice(10)
accepted=load(R/'private/accepted-fusion.json',q['accepted_fusion_file_sha256'])
assert accepted['scientific_eligible']and accepted['actual_completed_epochs']==150 and accepted['actual_optimizer_updates']==46950
assert accepted['all50_original_failed_recovery_and_admission_actors_physically_closed']and accepted['all4_postexit_reader_actors_physically_closed']
readback=load(q['postexit_readback'],accepted['postexit_readback_file_sha256'])
for a in readback['all50_failed_original_recovered_and_admission_actors_closed']:
 p=Path('/proc',str(a['pid']),'stat');assert not p.exists()or p.read_text().rsplit(')',1)[1].split()[19]!=str(a['start_ticks'])
fit=_safe(Path(readback['source_root'])/'private/experiment/full-fit')
fi=load(fit/'identity.json',readback['all_closed_artifact_sha256'][str(fit/'identity.json')],True)
fr=load(fit/'fit-result.json',readback['all_closed_artifact_sha256'][str(fit/'fit-result.json')],True)
assert fr['sha256']==accepted['actual_fit_result_sha256']==readback['actual_full_fit_sha256']and fr['identity_sha256']==fi['sha256']
assert fr['full_training_terminated']and fr['stage_complete']and fr['completed_epochs']==150 and fr['optimizer_executions']==46950
assert fr['selected_best']==accepted['actual_selected_best']and fi['policy']['scope']=='full'
assert fi['data_locations']['data']==q['data']and fi['data_locations']['upstream']==q['upstream']
d=fi['initializer_descriptor'];lineage=d['identity'];vision=lineage['vision'];emg=lineage['emg'];parents=d['producers']
assert lineage['scope']=='full'and lineage['producer_training_verified']and lineage['trained_branches_admitted']
assert lineage['fusion_partition_content_sha256']==fi['cohort']['partition_content_sha256']
assert vision['checkpoint_selection']=='LAST'and vision['producer_training_verified']and vision['current_raw_input_bytes_revalidated']
assert vision['producer']['completed_epochs']==150 and vision['producer']['optimizer_executions']==10650
assert emg['scope']=='full'and emg['full_initializer']and emg['completed_epochs']==150 and emg['optimizer_executions']==36150
assert emg['checkpoint_selection']=='native validation BEST'and emg['selected_epoch']==74
emgroot=_safe(parents['emg_fit']);ei=load(emgroot/'identity.json',sealed=True)
er=load(emgroot/'fit-result.json',emg['source_fit_result_file_sha256'],True)
assert ei['sha256']==emg['source_fit_identity_sha256']and er['identity_sha256']==ei['sha256']and er['full_training_terminated']and er['stage_complete']
assert ei['cohort']['partition_content_sha256']==emg['input_binding']['partition_content_sha256']
assert ei['cohort']['future_fusion_partition_content_sha256']==emg['future_fusion_partition_content_sha256']==fi['cohort']['partition_content_sha256']
visionroot=_safe(parents['vision_fit']);vp=load(visionroot/'partitions.json',vision['producer']['source_run_files_sha256']['partitions.json'])
assert _hash(vp)==vision['producer']['source_partition_content_sha256']==vision['current_input_binding']['reconstructed_legacy_partition_sha256']
weights={'fusion':(fit/'checkpoints'/fr['selected_best']['file'],fr['selected_best']['checkpoint_sha256']),
 'vision_export':(Path(d['files']['vision_checkpoint']),d['files']['vision_sha256']),
 'emg_export':(Path(d['files']['emg_checkpoint']),d['files']['emg_sha256']),
 'vision_original_LAST':(visionroot/'last.pt',vision['producer']['source_checkpoint_sha256']),
 'emg_original_BEST':(emgroot/'checkpoints'/er['selected_best']['file'],emg['source_selected_checkpoint_sha256'])}
weightproof={}
for role,(p,h)in weights.items():
 p=_safe(p);g.add(p);assert sha(p)==h
 weightproof[role]=dict(sha256=h,bytes=p.stat().st_size)
assert weights['vision_export'][1]==vision['export_file_sha256']and weights['emg_export'][1]==emg['export_file_sha256']
index=_safe(q['index_root']);ix=load(R/'private/accepted-index.json',q['accepted_index_file_sha256']);i=ix['recipes']['rn18_s']
assert i['unique_hand_points']==11114 and i['dense_stride']==3000 and i['window_length']==12000
load(index/'private/workflow-result.json',ix['raw_workflow_file_sha256'])
identity=load(index/'private/rn18_s-a/identity.json',i['identity_files_sha256'][0],True)
pp=_safe(index/'private/rn18_s-a/points.npz');g.add(pp);assert sha(pp)==i['point_files_sha256'][0]
with np.load(pp,allow_pickle=False)as packet:points={k:packet[k]for k in packet.files}
assert identity['sha256']==i['identity_sha256']and all(_typed_sha(a)==identity['arrays_sha256'][n]for n,a in points.items())
data=_safe(q['data']);raw={n:data/f for n,f in [('manifest','manifest.json'),('metadata','metadata.npz'),('frame_split_id','frame_split_id.dat')]}
for name,p in raw.items():g.add(p);assert sha(p)==identity['input_files_sha256'][name]
g.check();store=LegacyStore(data);py=random.getstate();ns=np.random.get_state()
fp=build_rn18_fusion_partitions(store);ep=build_rn18_initializer_partitions(store);original_vp=build_partitions(store)
assert _hash(fp)==fi['cohort']['partition_content_sha256']and _hash(ep)==ei['cohort']['partition_content_sha256']and original_vp==vp
vision_projection={'windows':[dict(w,footprint_start=w['start'],footprint_end=w['end'])for w in original_vp['windows']]}
dense=union_intervals((int(a),int(b))for a,b,h in zip(points['start'],points['end'],points['hand_index'])if h==0)
dense_mask=np.zeros(store.total_rows,bool)
for a,b in dense:dense_mask[a:b]=True
out={};checks=0
for role,p,original in [('fusion',fp,fp),('vision_initializer',vision_projection,original_vp),('emg_initializer',ep,ep)]:
 stats=summarize_dense_footprints(points,p);assert stats['dense_input_union_rows']==int(np.count_nonzero(dense_mask))
 for kind,value in stats['roles'].items():
  mask=np.zeros(store.total_rows,bool)
  for w in p['windows']:
   if w['role']==kind:mask[w['footprint_start']:w['footprint_end']]=True
  assert value['producer_input_union_rows']==int(np.count_nonzero(mask))
  assert value['intersection_rows']==int(np.count_nonzero(mask&dense_mask))
  assert value['overlapping_hand_requests']==sum(bool(mask[int(a):int(b)].any())for a,b in zip(points['start'],points['end']))
  assert value['centers_in_producer_rows']==int(np.count_nonzero(mask[points['center']]))
  checks+=1
 stats.update(actual_producer_partition_content_sha256=_hash(original),actual_completed_producer_bound=True,independent_full_boolean_row_mask_verified=True)
 out[role]=stats
assert random.getstate()==py and ns[0]==np.random.get_state()[0]and np.array_equal(ns[1],np.random.get_state()[1])and ns[2:]==np.random.get_state()[2:]
for name,p in raw.items():assert sha(p)==identity['input_files_sha256'][name]
for role,(p,h)in weights.items():assert sha(p)==h
g.check();assert 'torch'not in sys.modules
result=dict(format='actual_complete_RN18_dense_trained_producer_footprints_and_weight_bytes_v1',recipe='rn18_s',
 actual_completed_fit_identity_sha256=fi['sha256'],actual_completed_fit_result_sha256=fr['sha256'],actual_accepted_completion_file_sha256=q['accepted_fusion_file_sha256'],
 actual_original_postexit_readback_file_sha256=accepted['postexit_readback_file_sha256'],actual_index_identity_sha256=identity['sha256'],
 actual_selected_weights_sha256=weightproof,source_selection=dict(fusion_best_completed_epoch=145,vision='legacy trained LAST150',emg='RN18 cotrain validation BEST74 of150'),
 source_partition_content_sha256=dict(fusion=_hash(fp),vision_initializer=_hash(original_vp),emg_initializer=_hash(ep)),
 actual_bound_source_files_sha256={str(p):sha(p)for p in [fit/'identity.json',fit/'fit-result.json',emgroot/'identity.json',emgroot/'fit-result.json',visionroot/'partitions.json']},
 hand_requests=len(points['sequence']),producers=out,all_native_dense_samples_retained=True,
 actual_independent_full_boolean_mask_checks=checks,python_numpy_rng_unchanged=True,torch_imported=False,learned_pose_models_constructed=0,
 optimizer_updates=0,cuda_initialized=False,checkpoint_bytes_verified=True,checkpoint_tensors_admitted=False,native_geometry_executed=False,
 trained_model_dispatch_validated=False,dense_input_or_inference_permit=False,independent_unseen_input_footprints_proven=False,figure13_reproduced=False,full_paper_complete=False)
with(R/'private'/sys.argv[1]).open('x')as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='actual-RN18-completed-producer-weights-footprints-audited',full_boolean_mask_checks=checks,hand_requests=len(points['sequence']),torch_imported=False)),flush=True)
