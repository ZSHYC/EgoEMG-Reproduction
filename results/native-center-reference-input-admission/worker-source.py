
from pathlib import Path
import copy,datetime,hashlib,json,os,sys,time
R=Path(sys.argv[1]);label=sys.argv[2];Q=R/'private'/label;Q.mkdir(mode=0o700)
q=json.loads((R/'config.json').read_text());P=Path(q['unit_package']);U=Path(q['upstream']);D=Path(q['data']);os.nice(10)
sys.path.insert(0,str(P))
from egoemg_repro.classic_emg_cohort import _StatGuard
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):
 with p.open('x')as f:f.write(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');os.replace(t,p)
def resources():
 import shutil
 assert shutil.disk_usage('/data/shihao').free>225*1024**3
 assert next(int(x.split()[1])for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))*1024>128*1024**3
def actor():
 f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split();return dict(pid=os.getpid(),start_ticks=f[19],uid=os.getuid())
started=time.monotonic();resources()
m=json.loads((P/'stage-receipt.json').read_text());code_paths={P/n:h for n,h in m['files_sha256'].items()};code_paths.update({U/n:h for n,h in q['official_sources_sha256'].items()})
assert all(sha(p)==h for p,h in code_paths.items())
assert all(m['files_sha256'][n]==h for n,h in q['selected_new_source_sha256'].items())
assert sha(q['original_fit_identity_path'])==q['original_fit_identity_sha256']
identity=json.loads(Path(q['original_fit_identity_path']).read_text())
from egoemg_repro.data import LegacyStore
from egoemg_repro.classic_emg_data import _safe_path
store=LegacyStore(D);paths={}
for name,expected in identity['cohort']['payload_files_sha256'].items():
 if name=='metadata:manifest':p=D/'manifest.json'
 elif name=='metadata:episodes':p=D/'metadata.npz'
 else:
  key=name.removeprefix('field:');p=_safe_path(D,store.fields[key].get('filename',key+'.dat'))
 paths[name]=(p,expected)
paths['field:episode_index']=(_safe_path(D,store.fields['episode_index'].get('filename','episode_index.dat')),'74d736c2ecc65dcf82defabeca606d1f854c6875ebbbfd1e5233aa530a914351')
assert len(paths)==17
guard=_StatGuard([*code_paths,*[p for p,h in paths.values()],Path(q['vision_payload']),Path(q['original_fit_identity_path'])])
actual={name:sha(p)for name,(p,h)in paths.items()};assert all(actual[name]==h for name,(p,h)in paths.items());guard.check()
import numpy as np,torch
torch.set_num_threads(2);attempts={'learned_module':0,'optimizer':0}
def deny_module(*a,**k):attempts['learned_module']+=1;raise AssertionError('input admission cannot construct a model')
def deny_optimizer(*a,**k):attempts['optimizer']+=1;raise AssertionError('input admission cannot construct an optimizer')
torch.nn.Module.__init__=deny_module;torch.optim.Optimizer.__init__=deny_optimizer
from egoemg_repro.emgformer_literal import compose_emgformer
from egoemg_repro.center_reference_data import CenterReferenceDataset,reference_windows
from omegaconf import OmegaConf
configs={s:compose_emgformer(U,s)for s in('small','middle','large')}
def input_contract(c):return dict(modalities=OmegaConf.to_container(c.dataset.test[0].modalities,resolve=True),transforms=OmegaConf.to_container(c.transforms.test,resolve=True))
contracts={s:input_contract(c)for s,c in configs.items()};assert contracts['small']==contracts['middle']==contracts['large']
grid=reference_windows(D,U);assert grid['counts']==dict(windows=2141,hand_requests=4282,by_split=dict(gesture=2314,user=1646,both=322))
ds=CenterReferenceDataset(D,U,configs['small'],grid)
assert len(ds)==4282
assert sha(q['vision_payload'])==q['vision_payload_sha256'];v=torch.load(q['vision_payload'],map_location='cpu',weights_only=True);lookup={}
for part in v['parts']:
 for j in range(len(part['index'])):
  if int(part['repeat'][j])==0:
   key=(int(part['center'][j]),part['hand'][j],part['split'][j],part['user'][j]);assert key not in lookup
   lookup[key]=(part['target'][j].numpy().tobytes(),part['mask'][j].numpy().tobytes(),bool(part['vision'][j]))
assert len(lookup)==4306
emg_hash=hashlib.sha256();target_hash=hashlib.sha256();metadata_hash=hashlib.sha256();records=[];valid=0;visible=0
for i in range(len(ds)):
 if i%64==0:resources();guard.check()
 sample=ds[i];key=(sample['center'],sample['hand'],sample['split'],sample['subject']);assert key in lookup
 target,mask,crop=lookup[key]
 assert sample['joint_angles'].tobytes()==target and sample['label_valid_mask'].tobytes()==mask
 assert sample['reference_sample_index']==sample['reference_unique_sample_index']==i and sample['reference_repeat_index']==0
 eb=np.ascontiguousarray(sample['emg']).tobytes();tb=target+mask
 meta={k:sample[k]for k in('episode','start','end','center','hand','split','subject')};mb=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
 emg_hash.update(eb);target_hash.update(tb);metadata_hash.update(len(mb).to_bytes(8,'little'));metadata_hash.update(mb)
 record=dict(index=i,emg_sha256=hashlib.sha256(eb).hexdigest(),target_mask_sha256=hashlib.sha256(tb).hexdigest(),identity_sha256=hashlib.sha256(mb).hexdigest());records.append(record)
 valid+=bool(sample['label_valid_mask'][0]);visible+=crop
 if (i+1)%128==0:atomic(Q/'progress.json',dict(requests_read=i+1,target_requests=4282,valid_label_requests=valid,actor=actor(),full_paper_complete=False))
guard.check();assert all(sha(p)==h for name,(p,h)in paths.items())and all(sha(p)==h for p,h in code_paths.items())
assert sha(q['vision_payload'])==q['vision_payload_sha256']and sha(q['original_fit_identity_path'])==q['original_fit_identity_sha256'];guard.check()
assert attempts=={'learned_module':0,'optimizer':0}and not torch.cuda.is_initialized()
dump(Q/'input-digests.json',records)
result=dict(format='complete_native_center_reference_input_pass_v1',label=label,actor=actor(),source_commit=q['source_commit'],grid_sha256=grid['sha256'],counts=dict(**grid['counts'],valid_label_requests=valid,invalid_label_requests=len(ds)-valid,valid_angles=valid*22,visible_requests=visible,missing_image_requests=len(ds)-visible),source_files_sha256=actual,input_stream_sha256=emg_hash.hexdigest(),target_mask_stream_sha256=target_hash.hexdigest(),private_identity_stream_sha256=metadata_hash.hexdigest(),input_digests_file_sha256=sha(Q/'input-digests.json'),input_contract_equal_across_nativeSML=True,input_contract=contracts['small'],same_actual_visual_target_and_mask_bytes_for_all4282=True,all_inputs_finite=True,all_source_bytes_and_guarded_generations_unchanged=True,construction_attempts=attempts,cuda_initialized=False,new_neural_forwards=0,new_optimizer_updates=0,actual_seconds=time.monotonic()-started,scientific_eligible=False,full_paper_complete=False)
dump(Q/'result.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('source_files_sha256','input_contract')},sort_keys=True))
