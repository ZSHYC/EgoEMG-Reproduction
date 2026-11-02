
from pathlib import Path
import contextlib,datetime,hashlib,io,json,os,sys
q=json.load(sys.stdin);U=Path(q['upstream']);R=Path(q['point_root'])
sys.path.insert(0,str(R))
from egoemg_repro.emgformer_literal import compose_emgformer
from egoemg_repro.emg_initializer_data import _view
from egoemg_repro.data import LegacyStore
from egoemg_repro.classic_emg import _activate
_activate(U)
from egoemg import center_frame
import numpy as np,torch
torch.set_num_threads(2)
attempts={'learned_module':0,'optimizer':0}
def denied_module(*a,**k):attempts['learned_module']+=1;raise AssertionError('input probe cannot construct a model')
def denied_optimizer(*a,**k):attempts['optimizer']+=1;raise AssertionError('input probe cannot construct an optimizer')
torch.nn.Module.__init__=denied_module;torch.optim.Optimizer.__init__=denied_optimizer
assert hashlib.sha256(Path(center_frame.__file__).read_bytes()).hexdigest()==q['sources']['egoemg/center_frame.py']
stdout=io.StringIO()
with contextlib.redirect_stdout(stdout):grid=center_frame.collect_val_centers(Path(q['data']),12000)
cfg=compose_emgformer(U,'small');store=LegacyStore(q['data']);windows=[]
for ep,values in grid.items():
 for c,s in values:windows.append(dict(episode=ep,start=c-6000,end=c+6000,center=c,role='test',split=store.split_labels[int(store.field('frame_split_id')[c])]))
assert len(windows)==2141
vp=Path(q['vision_root'])/'test-native/predictions.pt';assert hashlib.sha256(vp.read_bytes()).hexdigest()==q['vision_payload_sha256'];vision=torch.load(vp,map_location='cpu',weights_only=True)
lookup={}
for part in vision['parts']:
 for i in range(len(part['index'])):
  if int(part['repeat'][i])==0:lookup[(int(part['center'][i]),part['hand'][i])]=(part['target'][i],part['mask'][i])
rows=[]
for hand in('left','right'):
 view=_view(store,windows,hand,'test',U,cfg)
 for w in(windows[0],windows[-1]):
  expected,mask=lookup[(w['center'],hand)]
  original_fusion_reader_error=None
  try:view._getitem_center_supervised(w['episode'],w['start'],w['end'],w['center'])
  except Exception as e:original_fusion_reader_error=type(e).__name__+': '+str(e)
  sample=view._getitem_center_supervised_emg(w['episode'],w['start'],w['end'])
  assert sample['emg'].shape==(8,12000)and sample['emg'].dtype==np.float32 and bool(np.isfinite(sample['emg']).all())
  assert sample['joint_angles'].tobytes()==expected.numpy().tobytes()and sample['label_valid_mask'].tobytes()==mask.numpy().tobytes()
  assert sample['target_hand']==hand
  rows.append(dict(hand=hand,boundary='first'if w is windows[0]else'last',native_EMG_input_shape=list(sample['emg'].shape),native_EMG_finite=True,point_target_and_mask_bytes_equal_actual_visual=True,original_fusion_reader_error=original_fusion_reader_error))
assert attempts=={'learned_module':0,'optimizer':0}and not torch.cuda.is_initialized()
f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps(dict(format='native_table10_common_reference_real_input_engineering_probe_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),process=dict(pid=os.getpid(),start_ticks=f[19]),scope='four real bilateral first/last-boundary inputs only; not complete4282 input validation or model evaluation',probes=rows,source_files_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()for name,path in{'emg_initializer_data':str(R/'egoemg_repro/emg_initializer_data.py'),'emgformer_literal':str(R/'egoemg_repro/emgformer_literal.py'),'official_center_frame':center_frame.__file__}.items()},new_neural_forwards=0,new_optimizer_updates=0,construction_attempts=attempts,cuda_initialized=False,complete_reference_input_validation=False,full_paper_complete=False)))
