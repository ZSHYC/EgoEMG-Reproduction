"""Independent column-wise bincount audit; never imports semantic_frames."""
from pathlib import Path
import datetime,hashlib,json,os,struct,sys,time
import numpy as np
if sys.flags.optimize:
 raise RuntimeError('optimized Python cannot verify frame validity; assertion checks are required')
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=load(ROOT/'private/request.json');producer=Path(q['producer_root'])
start=time.monotonic();state=load(producer/'private/execution-state.json')
def live(i):
 try:
  s=Path('/proc',str(i['pid']),'stat').read_text().rsplit(')',1)[1].split()
  return s[19]==str(i['start_ticks']) and s[0]!='Z'
 except FileNotFoundError:return False
assert state['actual_exit_code']==0 and state['status']=='complete'
assert all(not live(state[k]) for k in ('wrapper','child'))
m=load(producer/'private/stage-receipt.json')
for n,h in m['files_sha256'].items():assert sha(producer/n)==h
assert sha(producer/'private/frame-audit.json')==q['producer_audit_sha256']
audit=load(producer/'private/frame-audit.json');p=audit['public_projection'];data=Path(audit['private_receipt']['data_root'])
assert 'torch' not in sys.modules and os.environ['CUDA_VISIBLE_DEVICES']==''
N=p['timeline_frames'];slots=488;hands=('left','right');splits=['train','user','gesture','both']
scalar=['hand_frames','active_hand_frames','generated_valid_hand_frames','wrist_valid_hand_frames',
 'fingers_all20_valid_hand_frames','wrist_all2_valid_hand_frames','all22_valid_hand_frames',
 'any_angle_valid_hand_frames','possible_angle_elements','valid_angle_elements']
counts={k:np.zeros((2,slots),dtype='i8') for k in scalar}
finite_counts=np.zeros((2,slots,22),dtype='i8');valid_counts=finite_counts.copy();shared=np.zeros(slots,dtype='i8')
attrs=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_mode','st_uid','st_gid')
stamp=lambda s:[getattr(s,k) for k in attrs]
opened={};digests={};point_digests={};specs={x['name']:x for x in p['source_identity']['fields']}
try:
 for n,spec in specs.items():
  path=data/(n+'.dat');assert not path.is_symlink() and path.stat().st_size==spec['bytes']
  fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);before=stamp(os.fstat(fd))
  assert before==audit['private_receipt']['input_nine_attributes'][n]
  opened[n]=(fd,before);digests[n]=hashlib.sha256()
 for lo in range(0,N,131071):
  n=min(131071,N-lo);a={}
  for name,spec in specs.items():
   width=int(np.prod(spec['shape'][1:],dtype='i8'));size=np.dtype(spec['dtype']).itemsize*width
   raw=os.pread(opened[name][0],n*size,lo*size);assert len(raw)==n*size
   if np.dtype(spec['dtype']).kind=='b':assert max(raw)<=1
   digests[name].update(raw);a[name]=np.frombuffer(raw,dtype=spec['dtype']).reshape(n,*spec['shape'][1:])
  gid=a['label_gesture_class'];split=a['frame_split_id'];active=a['label_gesture_active']
  assert np.all((gid>=-1)&(gid<=59)) and np.all((split>=0)&(split<=3))
  code=(split.astype('i8')*61+gid+1)*2+active
  hist=np.bincount(code,minlength=slots);shared+=hist
  for h,hand in enumerate(hands):
   counts['hand_frames'][h]+=hist
   counts['active_hand_frames'][h]+=np.bincount(code[active],minlength=slots)
   generated=a['generated_label_valid'][:,h];wrist=a[f'mocap_{hand}_wrist_angles_valid']
   counts['generated_valid_hand_frames'][h]+=np.bincount(code[generated],minlength=slots)
   counts['wrist_valid_hand_frames'][h]+=np.bincount(code[wrist],minlength=slots)
   row_valid=np.zeros(n,dtype='u1');all_fingers=generated.copy();all_wrist=wrist.copy()
   for j in range(22):
    if j<20:value=a[f'generated_joint_angles_{hand}'][:,j];flag=generated
    else:value=np.deg2rad(a[f'mocap_{hand}_wrist_'+('pitch' if j==20 else 'yaw')]);flag=wrist
    good=np.isfinite(value);mask=good&flag
    finite_counts[h,:,j]+=np.bincount(code[good],minlength=slots)
    valid_counts[h,:,j]+=np.bincount(code[mask],minlength=slots)
    row_valid+=mask
    if j<20:all_fingers&=good
    else:all_wrist&=good
   for key,flag in [('fingers_all20_valid_hand_frames',all_fingers),('wrist_all2_valid_hand_frames',all_wrist),
     ('all22_valid_hand_frames',row_valid==22),('any_angle_valid_hand_frames',row_valid>0)]:
    counts[key][h]+=np.bincount(code[flag],minlength=slots)
 counts['possible_angle_elements']=counts['hand_frames']*22
 counts['valid_angle_elements']=valid_counts.sum(axis=2,dtype='i8')
 def bucket(indices,hand=None):
  hs=[hand] if hand is not None else [0,1]
  out={k:int(sum(v[h,indices].sum(dtype='i8') for h in hs)) for k,v in counts.items()}
  out['finite_by_angle']=sum((finite_counts[h,indices].sum(axis=0,dtype='i8') for h in hs)).tolist()
  out['valid_by_angle']=sum((valid_counts[h,indices].sum(axis=0,dtype='i8') for h in hs)).tolist()
  return out
 assert int(shared.sum())==N and bucket(list(range(slots)))==p['overall']
 for h,hand in enumerate(hands):assert bucket(list(range(slots)),h)==p['by_hand'][hand]
 nonzero=np.flatnonzero(shared).tolist();assert len(nonzero)==len(p['records'])
 for row in p['records']:
  c=(splits.index(row['split'])*61+row['class_id']+1)*2+int(row['active'])
  assert int(shared[c])==row['timeline_frames']
  for h,hand in enumerate(hands):assert bucket([c],h)==row[hand]
 for sid,split in enumerate(splits):
  idx=[c for c in range(slots) if c//122==sid]
  assert dict(timeline_frames=int(shared[idx].sum()),**bucket(idx))==p['by_split'][split]
 for row in p['by_gesture']+[dict(id=-1,**p['unassigned'])]:
  idx=[c for c in range(slots) if (c//2)%61==row['id']+1]
  assert int(shared[idx].sum())==row['timeline_frames']
  assert bucket(idx)=={k:row[k] for k in scalar+['finite_by_angle','valid_by_angle']}
 old={(r['class_id'],r['split'],r['active']):r['synchronized_rows'] for r in q['old_shared_audit']['counts']}
 new={(r['class_id'],r['split'],r['active']):r['timeline_frames'] for r in p['records']}
 assert old==new
 # Recheck the existing frozen-center byte digests, without models or repartitioning.
 part=Path(q['old_partition_path']);assert sha(part)==q['old_center_identity']['partition_file_sha256']
 centers=sorted(x['center'] for x in load(part)['windows']);assert len(set(centers))==len(centers)==33508
 for name,spec in specs.items():
  h=hashlib.sha256();width=np.dtype(spec['dtype']).itemsize*int(np.prod(spec['shape'][1:],dtype='i8'))
  for center in centers:
   raw=os.pread(opened[name][0],width,center*width);assert len(raw)==width
   h.update(struct.pack('<Q',center)+raw)
  original=next(x for x in q['old_center_identity']['fields'] if x['name']==name)
  assert h.hexdigest()==original['center_and_raw_bytes_sha256'];point_digests[name]=h.hexdigest()
 for name,spec in specs.items():
  assert digests[name].hexdigest()==spec['sha256']
  fd,before=opened[name];assert stamp(os.fstat(fd))==before==stamp((data/(name+'.dat')).stat())
 assert sha(part)==q['old_center_identity']['partition_file_sha256']
 for n,h in m['files_sha256'].items():assert sha(producer/n)==h
 assert sha(producer/'private/frame-audit.json')==q['producer_audit_sha256']
 assert 'torch' not in sys.modules
 result=dict(format='actual_independent_full_frame_validity_recount_v1',seconds=time.monotonic()-start,
  actual_complete_shared_rows=N,actual_hand_frames=2*N,actual_possible_angle_elements=44*N,
  compared_original_group_records=len(nonzero),all_integer_aggregates_exact=True,
  all_12_source_payload_sha256_exact=True,all_input_nine_attributes_unchanged=True,
  original_shared_157record_audit_exact=True,original_33508_frozen_center_bytes_exact=True,
  original_frozen_center_point_digests=point_digests,producer_audit_sha256=q['producer_audit_sha256'],
  actual_owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
  no_torch=True,model_training_performed=False,full_paper_complete=False)
 with (ROOT/'private/independent-replay.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['format','seconds','all_integer_aggregates_exact','all_12_source_payload_sha256_exact']}),flush=True)
finally:
 for fd,_ in opened.values():os.close(fd)
