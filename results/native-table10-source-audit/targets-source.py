
from pathlib import Path
import copy, datetime, hashlib, json, os, shutil, sys
q=json.load(sys.stdin)
assert shutil.disk_usage('/data/shihao').free>225*1024**3
assert next(int(line.split()[1])for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:'))*1024>128*1024**3
import torch
torch.set_num_threads(2)
construction_attempts={'learned_module':0,'optimizer':0}
def deny_module(*args,**kwargs):
 construction_attempts['learned_module']+=1;raise AssertionError('target audit must not construct learned modules')
def deny_optimizer(*args,**kwargs):
 construction_attempts['optimizer']+=1;raise AssertionError('target audit must not construct optimizers')
torch.nn.Module.__init__=deny_module
torch.optim.Optimizer.__init__=deny_optimizer
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def seal(value):
 return hashlib.sha256(json.dumps({k:v for k,v in value.items()if k!='sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def stamp(path):
 assert path.is_file()and not path.is_symlink()
 s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
files={}
def frozen(path,expected=None):
 path=Path(path);before=stamp(path);digest=sha(path)
 assert stamp(path)==before and(expected is None or digest==expected),str(path)
 files[str(path)]=dict(sha256=digest,generation=before)
 return path
def doc(path,expected=None):return json.loads(frozen(path,expected).read_text())
def closed(handle):
 try:
  f=Path('/proc',str(handle['pid']),'stat').read_text().rsplit(')',1)[1].split()
  return f[19]!=str(handle['start_ticks'])or f[0]in('Z','X')
 except FileNotFoundError:return True
def sanitized(value):
 # Exact existing public report projection; never compare private user keys
 # directly with the accepted public aliases, or publish the private values.
 if 'consumed'in value:users=sorted(value['consumed']['all']['by_user'])
 else:users=sorted(value['all']['by_user'])
 aliases={u:'user_'+str(i).zfill(3)for i,u in enumerate(users)}
 def project(v):
  if isinstance(v,dict):return {k:({aliases[u]:project(x)for u,x in item.items()}if k=='by_user'else project(item))for k,item in v.items()}
  if isinstance(v,list):return [project(x)for x in v]
  return v
 return project(value)
def flat(parts,name):
 values=[part[name]for part in parts]
 return torch.cat(values)if torch.is_tensor(values[0])else sum(values,[])
sources={};loaded={};identities={}
for size,a in q['points'].items():
 root=Path(a['root']);base=root/'private/full-evaluation/test-native'
 state=doc(root/'private/execution-state.json')
 assert state==a['accepted_terminal_state']and state['status']=='complete'and state['actual_exit_code']==0
 assert all(closed(state[k])for k in('wrapper','child'))
 def accepted(relative):return a['accepted_files_sha256'][relative]
 report=doc(root/'private/full-evaluation/report/metrics.json',a['report_file_sha256'])
 assert report==a['report']and report['sha256']==seal(report)
 p=doc(base/'pass.json',accepted('private/full-evaluation/test-native/pass.json'))
 assert p['sha256']==seal(p)==report['native_test_sha256']
 assert sanitized(p['statistics'])==report['test']
 assert p['fit_result_sha256']==report['fit_result_sha256']and p['model_before_sha256']==p['model_after_sha256']and p['optimizer_updates']==0
 assert p['selected_best']==report['training']['selected_best']and closed(p['process'])
 payload=frozen(base/'predictions.pt',accepted('private/full-evaluation/test-native/predictions.pt'))
 assert files[str(payload)]['sha256']==p['payload_sha256']
 data=torch.load(payload,map_location='cpu',weights_only=True)
 assert set(data)=={'parts','population'}
 parts=data['parts'];n=report['test']['unique_samples']
 assert n==2792 and torch.equal(flat(parts,'index'),torch.arange(n))and torch.equal(flat(parts,'unique'),torch.arange(n))and not bool(flat(parts,'repeat').any())
 assert flat(parts,'target').shape==(n,22,1)and flat(parts,'target').dtype==torch.float32 and bool(torch.isfinite(flat(parts,'target')).all())
 assert flat(parts,'mask').shape==(n,1)and flat(parts,'mask').dtype==torch.bool
 identity=doc(Path(p['fit_path'])/'identity.json',accepted('private/full-fit/identity.json'))
 assert identity['sha256']==seal(identity)
 assert identity['cohort']['sha256']==p['cohort_sha256']
 assert report['training']['completed_epochs']==200 and report['training']['optimizer_executions']==25000
 windows=[w for w in data['population']['partitions']['windows']if w['role']=='test']
 assert len(windows)==1396
 for i,w in enumerate(windows):
  assert w['center']==w['start']+6000 and w['end']-w['start']==12000
  for j in(i,i+1396):
   assert int(flat(parts,'center')[j])==w['center']and int(flat(parts,'episode')[j])==w['episode']and flat(parts,'split')[j]==w['split']
   assert flat(parts,'hand')[j]==('left'if j<1396 else'right')
 loaded[size]=data;identities[size]=identity
 sources[size]=dict(report_file_sha256=a['report_file_sha256'],pass_file_sha256=files[str(base/'pass.json')]['sha256'],payload_sha256=p['payload_sha256'],fit_result_sha256=p['fit_result_sha256'],selected_best=p['selected_best'],cohort_sha256=p['cohort_sha256'],completed_epochs=200,optimizer_updates=25000,requested=n,public_alias_projection_exact=True,point_source_index=6000,source_pass_and_queue_owners_physically_closed=True)
a=q['vision'];root=Path(a['root']);base=root/'test-native'
report=doc(root/'public-report/metrics.json',a['report_file_sha256']);assert report==a['report']and report['sha256']==seal(report)
assert closed(a['accepted_controller'])
p=doc(base/'pass.json');assert p['sha256']==seal(p)==report['native_test_sha256']
assert sanitized(p['statistics'])==report['test']and closed(p['process'])and p['model_before_sha256']==p['model_after_sha256']
payload=frozen(base/'predictions.pt',p['payload_sha256']);vision=torch.load(payload,map_location='cpu',weights_only=True)
assert set(vision)=={'parts','population'}
identity=doc(root/'full-fit/identity.json',a['fit_identity_file_sha256']);assert identity['sha256']==seal(identity)and identity['cohort']['sha256']==p['cohort_sha256']
fit=doc(root/'full-fit/fit-result.json',a['fit_result_file_sha256']);assert fit['sha256']==seal(fit)==p['fit_result_sha256']and fit['selected_best']==p['selected_best']
vparts=vision['parts'];vn=report['test']['consumed_samples'];vu=report['test']['unique_samples'];assert vn==8612 and vu==4306
assert torch.equal(flat(vparts,'index'),torch.arange(vn))
assert torch.equal(flat(vparts,'unique'),torch.arange(vn)%vu)and torch.equal(flat(vparts,'repeat'),torch.arange(vn)//vu)
first=torch.nonzero(flat(vparts,'repeat')==0).flatten();assert len(first)==vu
second=torch.nonzero(flat(vparts,'repeat')==1).flatten();assert len(second)==vu
for key in('target','mask','center','vision'):
 assert torch.equal(flat(vparts,key)[first],flat(vparts,key)[second]),key
for key in('hand','split','user'):
 assert [flat(vparts,key)[i]for i in first.tolist()]==[flat(vparts,key)[i]for i in second.tolist()],key
vt=flat(vparts,'target')[first];vm=flat(vparts,'mask')[first];vc=flat(vparts,'center')[first];visible=flat(vparts,'vision')[first]
assert vt.shape==(vu,22,1)and vt.dtype==torch.float32 and vm.shape==(vu,1)and vm.dtype==torch.bool
sources['vision_vit_small']=dict(report_file_sha256=a['report_file_sha256'],pass_file_sha256=files[str(base/'pass.json')]['sha256'],payload_sha256=p['payload_sha256'],fit_result_sha256=p['fit_result_sha256'],selected_best=p['selected_best'],cohort_sha256=p['cohort_sha256'],consumed=vn,first_repeat_requested=vu,second_repeat_target_mask_and_metadata_exact=True,public_alias_projection_exact=True,source_pass_and_controller_physically_closed=True,original_window_length=7790,original_point_source_index=3895)
joins={}
vmeta={key:[flat(vparts,key)[i]for i in first.tolist()]for key in('hand','split','user')}
vkeys=[(int(vc[i]),vmeta['hand'][i],vmeta['split'][i],vmeta['user'][i])for i in range(vu)]
assert len(set(vkeys))==vu
vlookup={key:i for i,key in enumerate(vkeys)}
for size,data in loaded.items():
 parts=data['parts'];target=flat(parts,'target');mask=flat(parts,'mask');center=flat(parts,'center')
 checks={key:flat(parts,key)==vmeta[key]for key in('hand','split','user')}
 checks.update(center_exact=torch.equal(center,vc),target_float32_bytes_exact=target.contiguous().numpy().tobytes()==vt.contiguous().numpy().tobytes(),mask_bool_bytes_exact=mask.contiguous().numpy().tobytes()==vm.contiguous().numpy().tobytes())
 assert len(set((int(center[i]),flat(parts,'hand')[i])for i in range(2792)))==2792
 checks['all_requested_pairs_unique']=True
 pmeta={key:flat(parts,key)for key in('hand','split','user')}
 pkeys=[(int(center[i]),pmeta['hand'][i],pmeta['split'][i],pmeta['user'][i])for i in range(2792)]
 matched=[(i,vlookup[key])for i,key in enumerate(pkeys)if key in vlookup]
 common=set(identities[size]['cohort']['payload_files_sha256'])&set(identity['cohort']['payload_files_sha256'])
 required=('metadata:manifest','metadata:episodes')
 assert all(key in common for key in required)
 equal={name:identities[size]['cohort']['payload_files_sha256'][name]==identity['cohort']['payload_files_sha256'][name]for name in sorted(common)}
 # Shared frozen content identities must cover the actual labels and validity,
 # not just metadata or equal sample counts.
 labels=[name for name in equal if any(s in name for s in('joint_angles','label_valid'))]
 assert len(labels)>=3 and all(equal.values())
 checks['all_shared_frozen_label_and_metadata_content_sha256_exact']=True
 joins[size]=dict(checks=checks,point_requested=2792,vision_first_repeat_requested=vu,matched_actual_center_hand_split_user_requests=len(matched),point_requests_without_matching_visual_center=2792-len(matched),visual_requests_without_matching_point_center=vu-len(matched),valid_points=int(mask.sum()),invalid_points=int((~mask).sum()),point_valid_angles=int(mask.sum())*22,vision_valid_angles=int(vm.sum())*22,shared_content_files_sha256_equal=equal,visual_visible_requests=int(visible.sum()),visual_missing_image_requests=int((~visible).sum()),visual_visible_and_label_valid_requests=int((visible&vm).sum()),same_complete_target_population=all(checks.values()))
for path,value in files.items():assert stamp(Path(path))==value['generation']and sha(path)==value['sha256']
assert construction_attempts=={'learned_module':0,'optimizer':0}and not torch.cuda.is_initialized()
f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
result=dict(format='native_table10_point6000_visual_target_population_audit_v1',observed_utc=utc(),process=dict(pid=os.getpid(),start_ticks=f[19],uid=os.getuid()),sources=sources,joins=joins,actual_consumed_files=files,source_file_count=len(files),all_source_bytes_and_five_attributes_unchanged=True,all_three_point_sources_same_actual_visual_targets=all(j['same_complete_target_population']for j in joins.values()),trajectory_aligned_center6255_not_substituted=True,no_predictions_recomputed=True,new_neural_forwards=0,new_optimizer_updates=0,construction_attempts=construction_attempts,cuda_initialized=False,private_user_keys_not_exported=True,paper_population_or_data_version_equivalence=False,full_paper_complete=False)
print(json.dumps(result,sort_keys=True,allow_nan=False))
