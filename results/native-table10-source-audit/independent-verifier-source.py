
from pathlib import Path
import datetime,hashlib,json,os,sys
q=json.load(sys.stdin);D=Path(q['data'])
def closed(a):
 try:f=Path('/proc',str(a['pid']),'stat').read_text().rsplit(')',1)[1].split();return f[19]!=str(a['start_ticks'])or f[0]in('Z','X')
 except FileNotFoundError:return True
assert all(closed(q[k]['process'])for k in('audit','grid','probe'))
def digest(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stamp(p):
 s=Path(p).stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
files=dict(q['audit']['actual_consumed_files']);files.update(q['grid']['consumed_files'])
files[str(D/'metadata.npz')]=dict(sha256='cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57',generation=stamp(D/'metadata.npz'))
for p,v in files.items():assert stamp(p)==v['generation']and digest(p)==v['sha256']
import numpy as np,torch
torch.set_num_threads(2)
def deny(*a,**k):raise AssertionError('independent table10 verifier cannot construct a model or optimizer')
torch.nn.Module.__init__=deny;torch.optim.Optimizer.__init__=deny
points={};vision=None
for p,v in files.items():
 if not p.endswith('/test-native/predictions.pt'):continue
 a=torch.load(p,map_location='cpu',weights_only=True);parts=a['parts']
 def tensor(k):return torch.cat([part[k]for part in parts]).numpy()
 def seq(k):return [v for part in parts for v in part[k]]
 keys=[(int(c),h)for c,h,r in zip(tensor('center'),seq('hand'),tensor('repeat'))if r==0]
 assert len(set(keys))==len(keys)
 if '/private/full-evaluation/'in p:
  points[p]=dict(keys=set(keys),ordered=keys,target=tensor('target').tobytes(),mask=tensor('mask').tobytes())
 else:vision=set(keys)
assert len(points)==3 and vision is not None and len(vision)==4306
first=next(iter(points.values()))
assert len(first['keys'])==2792 and all(v==first for v in points.values())
assert not(first['keys']&vision)
manifest=json.loads((D/'manifest.json').read_text());md=np.load(D/'metadata.npz',allow_pickle=True)
fs=manifest['fields']['frame_split_id'];split=np.memmap(D/fs.get('filename','frame_split_id.dat'),dtype=fs['dtype'],mode='r',shape=tuple(fs['shape']))
names=manifest['frame_split_labels'];ei=manifest['fields']['episode_index'];episode=np.memmap(D/ei.get('filename','episode_index.dat'),dtype=ei['dtype'],mode='r',shape=tuple(ei['shape']))
# Independent oracle from frozen episode metadata, preserving the official
# inclusive last-row/end comparison, including its exact boundary convention.
ref=set();common=set();counts={k:{s:0 for s in('gesture','user','both')}for k in('reference','common')}
for e,(lo,exclusive_hi)in enumerate(zip(md['episode_start_idx'],md['episode_end_idx'])):
 lo=int(lo);last=int(exclusive_hi)-1
 assert int(episode[lo])==e and int(episode[last])==e
 for start in range(lo,last-7790+1,7790):
  center=start+3895;role=names[int(split[center])]
  if role not in counts['reference']:continue
  ref.add((e,center));counts['reference'][role]+=2
  if center-6000>=lo and center+6000<=last:
   common.add((e,center));counts['common'][role]+=2
assert len(ref)==2153 and len(common)==2141
assert counts['reference']==q['grid']['reference']['by_split']and counts['common']==q['grid']['common_max_required12000']['by_split']
ref_pairs={(c,h)for e,c in ref for h in('left','right')};common_pairs={(c,h)for e,c in common for h in('left','right')}
assert vision==ref_pairs and common_pairs<=vision and len(vision-common_pairs)==24
assert not torch.cuda.is_initialized()
for p,v in files.items():assert stamp(p)==v['generation']and digest(p)==v['sha256']
f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps(dict(format='independent_actual_native_table10_source_audit_verification_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),process=dict(pid=os.getpid(),start_ticks=f[19]),all_three_audit_grid_probe_owners_physically_closed=True,all_three_original_point_metadata_targets_masks_exact=True,independent_center_hand_join_count=0,point_requests=2792,visual_reference_requests=4306,official_common12000_requests=4282,boundary_excluded_requests=24,independent_episode_metadata_grid_matches_both_actual_official_calls=True,split_request_counts=counts,all_consumed_source_bytes_and_five_attributes_unchanged=True,consumed_file_count=len(files),cuda_initialized=False,new_neural_forwards=0,new_optimizer_updates=0,full_shared_grid_model_evaluation_complete=False,full_paper_complete=False),sort_keys=True))
