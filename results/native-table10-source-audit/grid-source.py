
from pathlib import Path
import contextlib,datetime,hashlib,io,json,os,shutil,sys,time
q=json.load(sys.stdin);started=time.monotonic();U=Path(q['upstream']);D=Path(q['data'])
assert shutil.disk_usage('/data/shihao').free>225*1024**3
assert next(int(line.split()[1])for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:'))*1024>128*1024**3
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stamp(p):
 assert p.is_file()and not p.is_symlink();s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
paths={str(U/n):h for n,h in q['sources'].items()};paths.update({str(D/n):h for n,h in q['inputs'].items()})
stamps={p:stamp(Path(p))for p in paths};assert all(sha(p)==h for p,h in paths.items())
sys.path.insert(0,str(U))
import torch
torch.set_num_threads(2)
attempts={'learned_module':0,'optimizer':0}
def denied_module(*a,**k):attempts['learned_module']+=1;raise AssertionError('grid audit cannot construct a model')
def denied_optimizer(*a,**k):attempts['optimizer']+=1;raise AssertionError('grid audit cannot construct an optimizer')
torch.nn.Module.__init__=denied_module;torch.optim.Optimizer.__init__=denied_optimizer
from egoemg import center_frame
assert sha(center_frame.__file__)==q['sources']['egoemg/center_frame.py']
assert center_frame.REF_WL==7790
log=io.StringIO()
with contextlib.redirect_stdout(log):
 reference=center_frame.collect_val_centers(D)
 common=center_frame.collect_val_centers(D,required_window_length=12000)
def centers(grid):return{(episode,center)for episode,rows in grid.items()for center,start in rows}
ref,common_set=centers(reference),centers(common);assert common_set<=ref
assert sum(len(v)for v in reference.values())==len(ref)and sum(len(v)for v in common.values())==len(common_set)
manifest=json.loads((D/'manifest.json').read_text());import numpy as np
fs=manifest['fields']['frame_split_id'];split=np.memmap(D/fs.get('filename','frame_split_id.dat'),dtype=fs['dtype'],mode='r',shape=tuple(fs['shape']))
names=manifest['frame_split_labels']
point_path=Path(q['point_root'])/'private/full-evaluation/test-native/predictions.pt';vision_path=Path(q['vision_root'])/'test-native/predictions.pt'
for p,h in((point_path,q['point_payload_sha256']),(vision_path,q['vision_payload_sha256'])):
 paths[str(p)]=h;stamps[str(p)]=stamp(p);assert sha(p)==h
point=torch.load(point_path,map_location='cpu',weights_only=True);vision=torch.load(vision_path,map_location='cpu',weights_only=True)
def flat(parts,k):
 v=[x[k]for x in parts];return torch.cat(v)if torch.is_tensor(v[0])else sum(v,[])
pp=point['parts'];vp=vision['parts'];pc=flat(pp,'center');pe=flat(pp,'episode');ph=flat(pp,'hand');vc=flat(vp,'center');vh=flat(vp,'hand');vr=flat(vp,'repeat')
point_centers={(int(pe[i]),int(pc[i]))for i in range(len(pc))if ph[i]=='left'}
# Global row ids are unique within this manifest, so vision raw records do not
# need an invented episode field to join to the official episode/row grid.
vision_centers={int(vc[i])for i in range(len(vc))if vh[i]=='left'and int(vr[i])==0}
ref_rows={row for episode,row in ref};common_rows={row for episode,row in common_set}
assert len(ref_rows)==len(ref)and len(common_rows)==len(common_set)
def summary(values):
 return dict(windows=len(values),hand_requests=2*len(values),episodes=len({e for e,c in values}),by_split={s:2*sum(names[int(split[c])]==s for e,c in values)for s in('gesture','user','both')})
for p,h in paths.items():assert stamp(Path(p))==stamps[p]and sha(p)==h
assert not torch.cuda.is_initialized()and attempts=={'learned_module':0,'optimizer':0}
f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps(dict(format='actual_pinned_official_center_frame_grid_source_audit_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),process=dict(pid=os.getpid(),start_ticks=f[19]),actual_seconds=time.monotonic()-started,reference=summary(ref),common_max_required12000=summary(common_set),official_required_window_boundary_excluded_hand_requests=2*len(ref-common_set),old_point_grid_hand_requests=2*len(point_centers),old_vision_grid_hand_requests=2*len(vision_centers),old_point_grid_shared_with_official_reference_hand_requests=2*len({c for e,c in point_centers}&ref_rows),old_vision_grid_shared_with_official_reference_hand_requests=2*len(vision_centers&ref_rows),old_vision_grid_shared_with_common12000_hand_requests=2*len(vision_centers&common_rows),common12000_without_old_vision_predictions_hand_requests=2*len(common_rows-vision_centers),old_vision_predictions_outside_common12000_hand_requests=2*len(vision_centers-common_rows),all_source_bytes_and_five_attributes_unchanged=True,consumed_files={p:dict(sha256=h,generation=stamps[p])for p,h in paths.items()},official_stdout=log.getvalue(),construction_attempts=attempts,cuda_initialized=False,new_neural_forwards=0,new_optimizer_updates=0,full_shared_grid_model_evaluation_complete=False,full_paper_complete=False),sort_keys=True))
